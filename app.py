import io
import requests
import os
import json
import tempfile
import re

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTImage, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal, LTAnno, LTFigure
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

# PyMuPDF for robust PDF handling and metadata
import fitz # PyMuPDF is imported as fitz

app = Flask(__name__)

# --- Configuration from Environment Variables ---
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')
RETURN_DIRECTLY_FOR_TESTING = os.environ.get('RETURN_DIRECTLY_FOR_TESTING', 'false').lower() == 'true'
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true'

# --- Constants for Structured Extraction ---
IMAGE_PAGE_PLACEHOLDER = "[IMAGE PAGE - NO OCR PERFORMED]"
INLINE_IMAGE_PLACEHOLDER = "[Image detected]"
TABLE_START_TAG = "<BEGINNING_OF_TABULAR_CONTENT>"
TABLE_END_TAG = "<END_OF_TABULAR_CONTENT>"

# Minimum number of text characters on a page to consider it "text-heavy"
MIN_CHARS_FOR_TEXT_PAGE = 50

# Table detection parameters
MIN_COLUMNS_FOR_TABLE = 2
MIN_ROWS_FOR_TABLE = 2
COLUMN_ALIGNMENT_TOLERANCE = 15  # pixels
ROW_HEIGHT_TOLERANCE = 5  # pixels

# --- Helper Functions for Structured Extraction ---

def clean_text(text):
    """Removes null bytes and other common problematic characters."""
    if not text:
        return ""
    # Remove null bytes, excessive whitespace, but preserve structure
    cleaned = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    # Remove excessive blank lines but keep paragraph structure
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()

def extract_text_from_container(container):
    """
    Properly extracts text from LTTextContainer, preserving word and line structure.
    """
    if not hasattr(container, '_objs'):
        return container.get_text() if hasattr(container, 'get_text') else ""
    
    text_parts = []
    for obj in container._objs:
        if hasattr(obj, 'get_text'):
            text_parts.append(obj.get_text())
        elif hasattr(obj, '_objs'):
            # Recursively extract from nested objects
            text_parts.append(extract_text_from_container(obj))
    
    return ''.join(text_parts)

def detect_table_structure(text_containers):
    """
    Improved table detection using container positioning and alignment.
    """
    if len(text_containers) < MIN_ROWS_FOR_TABLE:
        return []
    
    # Group containers by approximate Y position (rows)
    rows = {}
    for container in text_containers:
        # Use container center Y for more stable grouping
        y_center = (container.y0 + container.y1) / 2
        y_key = round(y_center / ROW_HEIGHT_TOLERANCE) * ROW_HEIGHT_TOLERANCE
        
        if y_key not in rows:
            rows[y_key] = []
        rows[y_key].append(container)
    
    # Sort rows by Y position (top to bottom in PDF coordinates)
    sorted_rows = sorted(rows.items(), key=lambda x: -x[0])  # Negative for top-to-bottom
    
    # Analyze potential table rows
    table_candidates = []
    current_table = []
    
    for y_pos, row_containers in sorted_rows:
        # Sort containers in each row by X position (left to right)
        row_containers.sort(key=lambda x: x.x0)
        
        # Check if row has enough containers to be a table row
        if len(row_containers) >= MIN_COLUMNS_FOR_TABLE:
            # Check column alignment with previous row if exists
            if current_table:
                prev_row = current_table[-1]
                if len(row_containers) == len(prev_row):
                    # Check horizontal alignment
                    alignment_matches = 0
                    for i, (curr, prev) in enumerate(zip(row_containers, prev_row)):
                        if abs(curr.x0 - prev.x0) <= COLUMN_ALIGNMENT_TOLERANCE:
                            alignment_matches += 1
                    
                    # If most columns align, it's part of the same table
                    if alignment_matches >= len(row_containers) * 0.7:  # 70% alignment threshold
                        current_table.append(row_containers)
                    else:
                        # Start new table
                        if len(current_table) >= MIN_ROWS_FOR_TABLE:
                            table_candidates.append(current_table)
                        current_table = [row_containers]
                else:
                    # Different column count, end current table
                    if len(current_table) >= MIN_ROWS_FOR_TABLE:
                        table_candidates.append(current_table)
                    current_table = [row_containers]
            else:
                # First potential table row
                current_table = [row_containers]
        else:
            # Not enough containers for table row, end current table
            if len(current_table) >= MIN_ROWS_FOR_TABLE:
                table_candidates.append(current_table)
            current_table = []
    
    # Don't forget the last table
    if len(current_table) >= MIN_ROWS_FOR_TABLE:
        table_candidates.append(current_table)
    
    return table_candidates

def format_table_text(table_rows):
    """
    Formats detected table into proper row-based text.
    """
    formatted_rows = []
    
    for row_containers in table_rows:
        # Sort containers by X position
        row_containers.sort(key=lambda x: x.x0)
        
        # Extract text from each cell
        cell_texts = []
        for container in row_containers:
            cell_text = extract_text_from_container(container)
            cell_text = clean_text(cell_text).replace('\n', ' ')  # Single line per cell
            cell_texts.append(cell_text if cell_text else "")
        
        # Join cells with appropriate spacing
        row_text = " ".join(cell_texts)
        if row_text.strip():
            formatted_rows.append(row_text)
    
    return "\n".join(formatted_rows)

def extract_structured_text_improved(pdf_path, page_number):
    """
    Improved hybrid extraction with better text quality and table detection.
    """
    has_images_on_page = False
    extracted_parts = []
    page_text_chars = 0
    
    doc = None
    fp_pdfminer = None

    try:
        # --- Step 1: Check for images using PyMuPDF ---
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        
        if page.get_images():
            has_images_on_page = True

        # --- Step 2: Extract structured content using pdfminer.six ---
        fp_pdfminer = open(pdf_path, 'rb')
        
        # Optimized LAParams for better text extraction
        laparams = LAParams(
            all_texts=True,
            detect_vertical=True,
            word_margin=0.1,    # Smaller margin for better word detection
            char_margin=2.0,    # Reasonable character grouping
            line_margin=0.5,    # Good line grouping
            boxes_flow=0.5      # Maintain reading order
        )
        
        pages = extract_pages(fp_pdfminer, page_numbers=[page_number], laparams=laparams)
        
        page_elements = []
        all_text_containers = []
        
        # First pass: collect all text containers and detect images
        for page_layout in pages:
            for element in page_layout._objs:
                if isinstance(element, LTImage) or isinstance(element, LTFigure):
                    has_images_on_page = True
                elif isinstance(element, LTTextContainer):
                    all_text_containers.append(element)
        
        # Detect tables from all text containers
        detected_tables = detect_table_structure(all_text_containers)
        
        # Create set of table container IDs for exclusion from regular text
        table_container_ids = set()
        for table_rows in detected_tables:
            for row_containers in table_rows:
                for container in row_containers:
                    table_container_ids.add(id(container))
        
        # Second pass: process elements in order
        for page_layout in pages:
            # Sort elements by position (top to bottom, left to right)
            sorted_elements = sorted(
                [elem for elem in page_layout._objs if isinstance(elem, LTTextContainer)],
                key=lambda x: (-x.y1, x.x0)  # Top to bottom, left to right
            )
            
            processed_tables = set()
            
            for element in sorted_elements:
                element_id = id(element)
                
                # Check if this element is part of a table
                if element_id in table_container_ids:
                    # Find which table this element belongs to and process only once
                    for table_idx, table_rows in enumerate(detected_tables):
                        if table_idx in processed_tables:
                            continue
                            
                        # Check if this element is the first element of this table
                        first_container_id = id(table_rows[0][0])
                        if element_id == first_container_id:
                            table_text = format_table_text(table_rows)
                            if table_text.strip():
                                page_elements.append(f"{TABLE_START_TAG}\n{table_text}\n{TABLE_END_TAG}")
                                page_text_chars += len(table_text)
                            processed_tables.add(table_idx)
                            break
                else:
                    # Regular text element
                    text_content = extract_text_from_container(element)
                    text_content = clean_text(text_content)
                    if text_content:
                        page_elements.append(text_content)
                        page_text_chars += len(text_content)
        
        # Join all elements with appropriate spacing
        if page_elements:
            extracted_parts.append("\n".join(page_elements))
            
    except Exception as e:
        print(f"pdfminer.six extraction failed for page {page_number}: {e}")
        
        # Fallback to PyMuPDF with better text extraction
        try:
            if not doc:
                doc = fitz.open(pdf_path)
            page = doc.load_page(page_number)
            
            # Use PyMuPDF's text extraction with flags for better quality
            fallback_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
            fallback_text = clean_text(fallback_text)
            
            if fallback_text:
                extracted_parts.append(fallback_text)
                page_text_chars += len(fallback_text)
                print(f"Using PyMuPDF fallback for page {page_number}")
        except Exception as fallback_error:
            print(f"Both extraction methods failed for page {page_number}: {fallback_error}")
            
    finally:
        if doc:
            doc.close()
        if fp_pdfminer:
            fp_pdfminer.close()

    # Combine all extracted parts
    final_text = "\n\n".join(extracted_parts).strip()
    has_significant_text = (page_text_chars >= MIN_CHARS_FOR_TEXT_PAGE)

    return final_text, has_significant_text, has_images_on_page

def process_document(file_content, filename, original_payload):
    total_extracted_text_pages = []
    num_pages = 0
    
    # Store the PDF temporarily for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        # Get page count using PyMuPDF
        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}. It might be corrupted or not a valid PDF.")

        # Process each page with improved extraction
        for page_num in range(num_pages):
            try:
                page_content_text, has_significant_text, has_images_on_page_actual = extract_structured_text_improved(temp_file_path, page_num)

                # Page type determination
                if not has_significant_text and has_images_on_page_actual:
                    print(f"Page {page_num + 1} detected as image-heavy. Inserting placeholder.")
                    total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
                elif not has_significant_text and not has_images_on_page_actual:
                    print(f"Page {page_num + 1} detected as empty.")
                    total_extracted_text_pages.append("")
                else:
                    print(f"Page {page_num + 1} has significant text. Extracting content.")
                    
                    # Add inline image placeholder if needed
                    final_page_output = page_content_text
                    if has_images_on_page_actual and has_significant_text:
                        if INLINE_IMAGE_PLACEHOLDER not in page_content_text:
                            final_page_output += f"\n\n{INLINE_IMAGE_PLACEHOLDER}"

                    total_extracted_text_pages.append(final_page_output)
                    
            except Exception as page_error:
                print(f"Error processing page {page_num + 1}: {page_error}")
                # Add empty placeholder for failed pages
                total_extracted_text_pages.append(f"[Error processing page {page_num + 1}]")

    # Combine all page texts with clear separators
    full_document_text = "\n\n--- PAGE BREAK ---\n\n".join(total_extracted_text_pages)

    # --- Return or send to n8n ---
    if RETURN_DIRECTLY_FOR_TESTING:
        print("RETURN_DIRECTLY_FOR_TESTING is active. Returning extracted text directly.")
        return {
            "success": True,
            "message": "Extracted text returned directly for testing with improved hybrid approach.",
            "extracted_text": full_document_text,
            "filename": filename,
            "page_count": num_pages,
            **original_payload
        }
    else:
        # Prepare payload for n8n
        n8n_payload = {
            "extracted_text": full_document_text,
            "filename": filename,
            "page_count": num_pages,
            **original_payload
        }

        print(f"Sending data to n8n workflow at: {N8N_WEBHOOK_URL}")

        if not N8N_WEBHOOK_URL:
            raise ValueError("N8N_WEBHOOK_URL environment variable is not set.")

        try:
            n8n_response = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=300)
            n8n_response.raise_for_status()
            return n8n_response.json()

        except requests.exceptions.Timeout:
            raise RuntimeError("Request to n8n workflow timed out.")
        except requests.exceptions.RequestException as e:
            error_details = getattr(n8n_response, 'text', 'No response body')
            raise RuntimeError(f"Failed to send data to n8n workflow: {e}. Details: {error_details}")

@app.route('/process_document', methods=['POST'])
def process_document_endpoint():
    request_json = request.get_json(silent=True)
    
    # Handle original_payload
    if request_json:
        original_payload = {
            "subject_id": request_json.get("subject_id"),
            "user_id": request_json.get("user_id"),
            "subject_type": request_json.get("subject_type")
        }
    else:
        original_payload = {
            "subject_id": request.form.get("subject_id"),
            "user_id": request.form.get("user_id"),
            "subject_type": request.form.get("subject_type")
        }

    file_content = None
    filename = "document.pdf"

    # Handle file input
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        file_content = file.read()
        filename = file.filename
        print(f"Received file upload: {filename}")
    
    elif request_json and 'file_url' in request_json:
        file_url = request_json['file_url']
        if not file_url:
            return jsonify({"error": "Missing 'file_url' in JSON body"}), 400
        
        try:
            response = requests.get(file_url, timeout=60)
            response.raise_for_status()
            file_content = response.content
            filename = file_url.split('/')[-1].split('?')[0]
            print(f"Received file via URL: {file_url}")
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error fetching URL: {e}"}), 400
    else:
        return jsonify({"error": "No file or file_url provided"}), 400

    if not file_content:
        return jsonify({"error": "Could not retrieve file content"}), 500

    try:
        response_from_processor = process_document(file_content, filename, original_payload)
        return jsonify(response_from_processor), 200

    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"General error in /process_document endpoint: {e}")
        return jsonify({"error": f"Document processing failed: {e}"}), 500

if __name__ == '__main__':
    if not N8N_WEBHOOK_URL and not RETURN_DIRECTLY_FOR_TESTING:
        print("WARNING: Neither N8N_WEBHOOK_URL nor RETURN_DIRECTLY_FOR_TESTING is set.")
    elif N8N_WEBHOOK_URL and RETURN_DIRECTLY_FOR_TESTING:
        print("WARNING: Both N8N_WEBHOOK_URL and RETURN_DIRECTLY_FOR_TESTING are set. "
              "RETURN_DIRECTLY_FOR_TESTING will take precedence.")

    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 8000))
