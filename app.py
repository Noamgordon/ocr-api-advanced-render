import io
import requests
import os
import json
import tempfile
import re

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal, LTAnno
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
MAX_COLUMN_WIDTH_VARIANCE = 0.3  # 30% variance in column alignment

# --- Helper Functions for Structured Extraction ---

def clean_text(text):
    """Removes null bytes and other common problematic characters."""
    return text.replace('\x00', '').strip()

def detect_table_structure(text_boxes):
    """
    Analyzes text boxes to detect tabular structure.
    Returns list of table regions with their bounding boxes.
    """
    if len(text_boxes) < MIN_ROWS_FOR_TABLE:
        return []
    
    # Group text boxes by approximate Y position (rows)
    rows = {}
    tolerance = 5  # pixels tolerance for row alignment
    
    for box in text_boxes:
        y_pos = round(box.y0 / tolerance) * tolerance
        if y_pos not in rows:
            rows[y_pos] = []
        rows[y_pos].append(box)
    
    # Sort rows by Y position (top to bottom)
    sorted_rows = sorted(rows.items(), key=lambda x: -x[0])  # Negative for top-to-bottom
    
    # Analyze column structure
    potential_tables = []
    current_table_rows = []
    
    for y_pos, row_boxes in sorted_rows:
        # Sort boxes in row by X position (left to right)
        row_boxes.sort(key=lambda x: x.x0)
        
        # Check if this row has similar column structure to previous rows
        if len(row_boxes) >= MIN_COLUMNS_FOR_TABLE:
            if not current_table_rows:
                current_table_rows = [row_boxes]
            else:
                # Check column alignment with previous row
                prev_row = current_table_rows[-1]
                if len(row_boxes) == len(prev_row):
                    # Check if columns are roughly aligned
                    aligned = True
                    for i, (curr_box, prev_box) in enumerate(zip(row_boxes, prev_row)):
                        x_diff = abs(curr_box.x0 - prev_box.x0)
                        if x_diff > (prev_box.width * MAX_COLUMN_WIDTH_VARIANCE):
                            aligned = False
                            break
                    
                    if aligned:
                        current_table_rows.append(row_boxes)
                    else:
                        # End current table, start new one
                        if len(current_table_rows) >= MIN_ROWS_FOR_TABLE:
                            potential_tables.append(current_table_rows)
                        current_table_rows = [row_boxes]
                else:
                    # Different number of columns, end current table
                    if len(current_table_rows) >= MIN_ROWS_FOR_TABLE:
                        potential_tables.append(current_table_rows)
                    current_table_rows = [row_boxes]
        else:
            # Row doesn't have enough columns, end current table
            if len(current_table_rows) >= MIN_ROWS_FOR_TABLE:
                potential_tables.append(current_table_rows)
            current_table_rows = []
    
    # Don't forget the last table
    if len(current_table_rows) >= MIN_ROWS_FOR_TABLE:
        potential_tables.append(current_table_rows)
    
    return potential_tables

def extract_table_text(table_rows):
    """
    Extracts text from detected table structure in proper row format.
    """
    table_lines = []
    
    for row_boxes in table_rows:
        # Sort boxes in row by X position
        row_boxes.sort(key=lambda x: x.x0)
        
        # Extract text from each cell
        row_cells = []
        for box in row_boxes:
            cell_text = ""
            for line in box:
                if hasattr(line, '_objs'):
                    for char in line._objs:
                        if hasattr(char, 'get_text'):
                            cell_text += char.get_text()
                elif hasattr(line, 'get_text'):
                    cell_text += line.get_text()
            row_cells.append(clean_text(cell_text))
        
        # Join cells with spaces to maintain row structure
        if row_cells:
            table_lines.append(" ".join(row_cells))
    
    return "\n".join(table_lines)

def extract_structured_text_hybrid(pdf_path, page_number):
    """
    Hybrid extraction combining PyMuPDF for general text and pdfminer.six for table detection.
    Returns (structured_text_content, has_significant_text, has_images).
    """
    extracted_content_parts = []
    page_text_chars = 0
    has_images_on_page = False
    
    doc = None
    fp_pdfminer = None

    try:
        # --- Pass 1: PyMuPDF for image detection and basic text ---
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        
        # Check for images
        if page.get_images():
            has_images_on_page = True

        # --- Pass 2: pdfminer.six for structured text and table detection ---
        fp_pdfminer = open(pdf_path, 'rb')
        laparams = LAParams(
            all_texts=True,
            detect_vertical=True,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
        
        pages = extract_pages(fp_pdfminer, page_numbers=[page_number], laparams=laparams)
        
        for page_layout in pages:
            # Collect all text containers for table detection
            text_boxes = []
            non_table_elements = []
            
            for element in page_layout._objs:
                if isinstance(element, LTImage):
                    has_images_on_page = True
                elif isinstance(element, LTTextContainer):
                    text_boxes.append(element)
            
            # Detect tables
            detected_tables = detect_table_structure(text_boxes)
            
            # Create set of table elements for exclusion from regular text
            table_elements = set()
            for table_rows in detected_tables:
                for row in table_rows:
                    for box in row:
                        table_elements.add(id(box))
            
            # Process elements in order
            page_elements = []
            
            for element in page_layout._objs:
                if isinstance(element, LTTextContainer):
                    element_id = id(element)
                    
                    # Check if this element is part of a table
                    is_table_element = element_id in table_elements
                    
                    if is_table_element:
                        # Find which table this element belongs to
                        for table_rows in detected_tables:
                            for row_idx, row in enumerate(table_rows):
                                if any(id(box) == element_id for box in row):
                                    # Only add table tags for the first element of the first row
                                    if row_idx == 0 and id(row[0]) == element_id:
                                        table_text = extract_table_text(table_rows)
                                        page_elements.append(f"{TABLE_START_TAG}\n{table_text}\n{TABLE_END_TAG}")
                                    break
                    else:
                        # Regular text element
                        text_content = clean_text(element.get_text())
                        if text_content:
                            page_elements.append(text_content)
                            page_text_chars += len(text_content)
            
            # Join all elements
            if page_elements:
                extracted_content_parts.append("\n".join(page_elements))
        
        # Fallback to PyMuPDF if pdfminer.six didn't extract much
        if page_text_chars < 10:
            fallback_text = clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE))
            if fallback_text:
                extracted_content_parts.append(fallback_text)
                page_text_chars += len(fallback_text)
                
    except Exception as e:
        print(f"Error in hybrid extraction for page {page_number}: {e}")
        # Fallback to PyMuPDF only
        if doc:
            page = doc.load_page(page_number)
            fallback_text = clean_text(page.get_text("text"))
            extracted_content_parts.append(fallback_text)
            page_text_chars += len(fallback_text)
        
    finally:
        if doc:
            doc.close()
        if fp_pdfminer:
            fp_pdfminer.close()

    final_text = "\n\n".join(extracted_content_parts).strip()
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

        # Get page count and metadata using PyMuPDF
        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            
            # Extract metadata (similar to n8n node)
            metadata = doc.metadata
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}. It might be corrupted or not a valid PDF.")

        # Process each page with hybrid approach
        for page_num in range(num_pages):
            page_content_text, has_significant_text, has_images_on_page_actual = extract_structured_text_hybrid(temp_file_path, page_num)

            # Page type determination logic (same as before)
            if not has_significant_text and has_images_on_page_actual:
                print(f"Page {page_num + 1} detected as image-heavy (no significant text, has raster images). Inserting placeholder.")
                total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
            elif not has_significant_text and not has_images_on_page_actual:
                print(f"Page {page_num + 1} detected as truly empty.")
                total_extracted_text_pages.append("")
            else:
                print(f"Page {page_num + 1} has significant text. Extracting structured content.")
                
                # Add inline image placeholder if needed
                final_page_output = page_content_text
                if has_images_on_page_actual and has_significant_text:
                    if INLINE_IMAGE_PLACEHOLDER not in page_content_text:
                        final_page_output += f"\n\n{INLINE_IMAGE_PLACEHOLDER}"

                total_extracted_text_pages.append(final_page_output)

    # Combine all page texts
    full_document_text = "\n\n--- PAGE BREAK ---\n\n".join(total_extracted_text_pages)

    # --- Conditional Return for Testing / Send to n8n ---
    if RETURN_DIRECTLY_FOR_TESTING:
        print("RETURN_DIRECTLY_FOR_TESTING is active. Returning extracted text directly.")
        return {
            "success": True,
            "message": "Extracted text returned directly for testing with hybrid approach.",
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
        print(f"Payload (excluding full text for brevity in logs): { {k: v for k, v in n8n_payload.items() if k != 'extracted_text'} }")

        if not N8N_WEBHOOK_URL:
            raise ValueError("N8N_WEBHOOK_URL environment variable is not set. Cannot send to n8n.")

        try:
            n8n_response = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=300)
            n8n_response.raise_for_status()

            n8n_result = n8n_response.json()
            print(f"n8n workflow response: {n8n_result}")
            return n8n_result

        except requests.exceptions.Timeout:
            raise RuntimeError("Request to n8n workflow timed out.")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with n8n workflow: {e}")
            try:
                error_details = n8n_response.json()
            except (json.JSONDecodeError, AttributeError):
                error_details = n8n_response.text if n8n_response else "No response body."
            raise RuntimeError(f"Failed to send data to n8n workflow: {e}. Details: {error_details}")
        except Exception as e:
            print(f"An unexpected error occurred during n8n communication: {e}")
            raise RuntimeError(f"An unexpected error occurred during n8n communication: {e}")

@app.route('/process_document', methods=['POST'])
def process_document_endpoint():
    request_json = request.get_json(silent=True)
    
    # Handle original_payload coming from either JSON body or form data
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
        return jsonify({"error": "No file or file_url provided, or 'file' is missing in multipart form data."}), 400

    if not file_content:
        return jsonify({"error": "Could not retrieve file content."}), 500

    try:
        response_from_processor = process_document(file_content, filename, original_payload)
        return jsonify(response_from_processor), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"General error in /process_document endpoint: {e}")
        return jsonify({"error": f"Document processing failed: {e}"}), 500

if __name__ == '__main__':
    if not N8N_WEBHOOK_URL and not RETURN_DIRECTLY_FOR_TESTING:
        print("WARNING: Neither N8N_WEBHOOK_URL nor RETURN_DIRECTLY_FOR_TESTING is set. "
              "The service will not send data anywhere or return useful results.")
    elif N8N_WEBHOOK_URL and RETURN_DIRECTLY_FOR_TESTING:
        print("WARNING: Both N8N_WEBHOOK_URL and RETURN_DIRECTLY_FOR_TESTING are set. "
              "RETURN_DIRECTLY_FOR_TESTING will take precedence.")

    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 8000))
