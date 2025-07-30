import io
import requests
import os
import json
import tempfile
import re

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams # For basic layout parameters for image detection

# PyMuPDF for robust PDF handling and text extraction
import fitz # PyMuPDF is imported as fitz

app = Flask(__name__)

# --- Configuration from Environment Variables ---
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')
RETURN_DIRECTLY_FOR_TESTING = os.environ.get('RETURN_DIRECTLY_FOR_TESTING', 'false').lower() == 'true'
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true' # For future module additions

# --- Constants for Structured Extraction ---
IMAGE_PAGE_PLACEHOLDER = "[IMAGE PAGE - NO OCR PERFORMED]"
INLINE_IMAGE_PLACEHOLDER = "[Image detected]"
TABLE_START_TAG = "<BEGINNING_OF_TABULAR_CONTENT>"
TABLE_END_TAG = "<END_OF_TABULAR_CONTENT>"

# Minimum number of text characters on a page to consider it "text-heavy"
# and not an image-only page (even if it has images). Very low to ensure
# even sparse text is extracted.
MIN_CHARS_FOR_TEXT_PAGE = 50 

# --- Table Detection Parameters (Conservative) ---
# These parameters are tuned for high precision (fewer false positives)
# even if it means lower recall (missing some complex tables).
TABLE_ALIGNMENT_TOLERANCE_X = 5  # Max pixel difference for X-alignment of columns
TABLE_ALIGNMENT_TOLERANCE_Y = 3  # Max pixel difference for Y-alignment of rows
MIN_TABLE_ROWS = 2               # Minimum number of rows to consider it a table
MIN_TABLE_COLUMNS = 2            # Minimum number of columns to consider it a table
MIN_COMMON_X_COORDS = 0.8        # Minimum proportion of common X-coordinates for column alignment

# --- Helper Functions for Structured Extraction ---

def clean_text(text):
    """Removes null bytes, excessive whitespace, and other common problematic characters."""
    # Remove null bytes
    cleaned_text = text.replace('\x00', '')
    # Replace multiple spaces/newlines with a single space/newline to normalize
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def get_text_from_element(element):
    """Recursively extracts text from an LTTextContainer or its children."""
    if isinstance(element, LTTextContainer):
        return clean_text(element.get_text())
    return ""

def detect_table_structure(text_boxes):
    """
    Analyzes text boxes to detect tabular structure based on strict horizontal and vertical alignment.
    Returns a list of detected tables, where each table is a list of its rows (list of LTTextBoxHorizontal).
    This function is designed to be highly conservative to avoid false positives.
    """
    if len(text_boxes) < MIN_TABLE_ROWS:
        return []

    # Sort text boxes primarily by Y-coordinate (top to bottom), then by X-coordinate (left to right)
    text_boxes.sort(key=lambda b: (-b.y1, b.x0))

    detected_tables = []
    
    # Step 1: Group text boxes into potential rows based on Y-overlap/proximity
    rows_by_y = []
    if text_boxes:
        current_row = [text_boxes[0]]
        for i in range(1, len(text_boxes)):
            # Check if current box's y-range overlaps significantly with the last box in current_row
            # Or if they are very close vertically
            if (text_boxes[i].y0 < current_row[-1].y1 and text_boxes[i].y1 > current_row[-1].y0) or \
               (abs(text_boxes[i].y1 - current_row[-1].y1) < TABLE_ALIGNMENT_TOLERANCE_Y):
                current_row.append(text_boxes[i])
            else:
                rows_by_y.append(sorted(current_row, key=lambda b: b.x0)) # Sort by X for columns
                current_row = [text_boxes[i]]
        rows_by_y.append(sorted(current_row, key=lambda b: b.x0)) # Add the last row

    # Step 2: Analyze rows for consistent column alignment to identify tables
    current_table_candidate = []
    for row in rows_by_y:
        if len(row) < MIN_TABLE_COLUMNS:
            # Row doesn't have enough columns for a table, reset candidate
            if len(current_table_candidate) >= MIN_TABLE_ROWS:
                detected_tables.append(current_table_candidate)
            current_table_candidate = []
            continue

        if not current_table_candidate:
            # Start a new table candidate with this row
            current_table_candidate.append(row)
        else:
            prev_row = current_table_candidate[-1]
            
            # Check if the number of columns is consistent
            if len(row) != len(prev_row):
                if len(current_table_candidate) >= MIN_TABLE_ROWS:
                    detected_tables.append(current_table_candidate)
                current_table_candidate = [row]
                continue

            # Check for strong X-alignment of columns across rows
            # We'll compare the x0 (left edge) of corresponding cells
            aligned_columns = 0
            for i in range(len(row)):
                if abs(row[i].x0 - prev_row[i].x0) < TABLE_ALIGNMENT_TOLERANCE_X:
                    aligned_columns += 1
            
            # If a significant proportion of columns are aligned, consider it part of the table
            if aligned_columns / len(row) >= MIN_COMMON_X_COORDS:
                current_table_candidate.append(row)
            else:
                # Not aligned, end current table and start a new one
                if len(current_table_candidate) >= MIN_TABLE_ROWS:
                    detected_tables.append(current_table_candidate)
                current_table_candidate = [row]
    
    # Add the last table candidate if it meets the criteria
    if len(current_table_candidate) >= MIN_TABLE_ROWS:
        detected_tables.append(current_table_candidate)
    
    return detected_tables

def extract_table_text(table_rows):
    """
    Extracts text from detected table structure in proper row format,
    using tabs to separate columns.
    """
    table_lines = []
    for row_boxes in table_rows:
        # Ensure boxes in row are sorted by X position for correct column order
        row_boxes.sort(key=lambda x: x.x0)
        
        row_cells = []
        for box in row_boxes:
            cell_text = get_text_from_element(box)
            row_cells.append(cell_text)
        
        # Join cells with a tab to indicate columns
        table_lines.append("\t".join(row_cells))
    
    return "\n".join(table_lines)


def extract_structured_text(pdf_path, page_number):
    """
    Extracts text content and inserts inline image placeholders from a specific PDF page.
    Prioritizes PyMuPDF for text layout and pdfminer.six for image detection and table detection.
    Returns (structured_text_content, has_significant_text, has_images, structured_tables_text).
    """
    extracted_content_parts = []
    page_text_chars = 0
    has_images_on_page = False
    structured_tables_text = [] # New list to store detected table text

    doc = None
    fp_pdfminer = None

    try:
        # --- Pass 1: PyMuPDF for main text content and image detection ---
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        
        # Extract text while preserving whitespace and ligatures for better layout
        text_from_pymu = clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES))
        extracted_content_parts.append(text_from_pymu)
        page_text_chars += len(text_from_pymu)

        # Update has_images_on_page based on PyMuPDF's direct image check
        if page.get_images():
            has_images_on_page = True

        # --- Pass 2: pdfminer.six for specific object detection (e.g., table structure) ---
        fp_pdfminer = open(pdf_path, 'rb')
        # Use default LAParams for pdfminer.six to get raw layout objects for table detection
        laparams = LAParams() 
        pages = extract_pages(fp_pdfminer, page_numbers=[page_number], laparams=laparams)
        
        for p in pages:
            text_boxes_for_table_detection = []
            for element in p._objs:
                if isinstance(element, LTTextBoxHorizontal):
                    text_boxes_for_table_detection.append(element)
            
            # Detect tables using the conservative strategy
            detected_tables = detect_table_structure(text_boxes_for_table_detection)
            
            for table_rows in detected_tables:
                table_content = extract_table_text(table_rows)
                if table_content.strip():
                    structured_tables_text.append(f"{TABLE_START_TAG}\n{table_content}\n{TABLE_END_TAG}")
        
    finally:
        if doc:
            doc.close()
        if fp_pdfminer:
            fp_pdfminer.close()

    final_text = "\n\n".join(extracted_content_parts).strip()

    # Determine if the page has significant text based on character count
    has_significant_text = (page_text_chars >= MIN_CHARS_FOR_TEXT_PAGE)

    return final_text, has_significant_text, has_images_on_page, structured_tables_text

# --- Main Document Processing Function ---
def process_document(file_content, filename, original_payload):
    total_extracted_text_pages = []
    num_pages = 0
    
    # Store the PDF temporarily for PyMuPDF and pdfminer.six
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        # PyMuPDF for basic page count
        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            doc.close() # Close immediately after getting page_count
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF with PyMuPDF for page count: {e}. It might be corrupted or not a valid PDF.")

        # Process each page
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1}...")
            # The returned `has_images_on_page_actual` and `has_significant_text` are derived directly from the page content analysis.
            page_content_text, has_significant_text, has_images_on_page_actual, structured_tables = extract_structured_text(temp_file_path, page_num)

            # Heuristic for determining page type (text vs. image-only)
            if not has_significant_text and has_images_on_page_actual:
                # If no significant text and it does have raster images, assume it's a scanned/image-only page.
                print(f"Page {page_num + 1} detected as image-heavy (no significant text, has raster images). Inserting placeholder.")
                total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
            elif not has_significant_text and not has_images_on_page_actual:
                # Page is truly empty (no text, no images)
                print(f"Page {page_num + 1} detected as truly empty.")
                total_extracted_text_pages.append("") 
            else:
                # Page has significant text. Add text directly.
                print(f"Page {page_num + 1} has significant text. Extracting structured content.")
                
                final_page_output = page_content_text
                
                # If images were detected on a page that also has text, we append the inline placeholder.
                if has_images_on_page_actual and has_significant_text:
                     # Only add if the text doesn't already implicitly contain it (from its original source)
                    if INLINE_IMAGE_PLACEHOLDER not in page_content_text:
                        final_page_output += f"\n\n{INLINE_IMAGE_PLACEHOLDER}"

                # Append structured tables after the main page content
                if structured_tables:
                    final_page_output += "\n\n" + "\n\n".join(structured_tables)

                total_extracted_text_pages.append(final_page_output)


    # Combine all page texts
    full_document_text = "\n\n--- PAGE BREAK ---\n\n".join(total_extracted_text_pages)

    # --- Conditional Return for Testing / Send to n8n ---
    if RETURN_DIRECTLY_FOR_TESTING:
        print("RETURN_DIRECTLY_FOR_TESTING is active. Returning extracted text directly.")
        return {
            "success": True,
            "message": "Extracted text returned directly for testing.",
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
            n8n_response = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=300) # Increased timeout
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
    else: # Assume multipart/form-data with fields
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
            # Extract filename from URL, handling query parameters
            filename = os.path.basename(file_url.split('?')[0])
            if not filename: # Fallback if URL doesn't have a clear filename
                filename = "downloaded_document.pdf"
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

