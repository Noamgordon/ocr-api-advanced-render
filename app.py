import io
import requests
import os
import json
import tempfile
import re

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal, LTAnno, LTCurve, LTRect
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

# PyMuPDF for robust PDF handling and metadata
import fitz # PyMuPDF is imported as fitz

app = Flask(__name__)

# --- Configuration from Environment Variables ---
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')
RETURN_DIRECTLY_FOR_TESTING = os.environ.get('RETURN_DIRECTLY_FOR_TESTING', 'false').lower() == 'true'
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true' # Not implemented in this version, but kept for future use

# --- Constants for Structured Extraction ---
IMAGE_PAGE_PLACEHOLDER = "[IMAGE PAGE - NO OCR PERFORMED]"
INLINE_IMAGE_PLACEHOLDER = "[Image detected]"
TABLE_START_TAG = "<BEGINNING_OF_TABULAR_CONTENT>"
TABLE_END_TAG = "<END_OF_TABULAR_CONTENT>"

# Minimum number of text characters on a page to consider it "text-heavy"
MIN_CHARS_FOR_TEXT_PAGE = 50

# Table detection parameters (adjusted for better flexibility)
# These parameters are heuristics and might need fine-tuning for specific document types.
# They aim to identify grid-like text structures.
MIN_COLUMNS_FOR_TABLE = 2
MIN_ROWS_FOR_TABLE = 2
# Max variance in X-coordinates for column alignment (percentage of page width)
MAX_COLUMN_X_VARIANCE_PERCENT = 0.02 # 2% of page width
# Max variance in Y-coordinates for row alignment (percentage of page height)
MAX_ROW_Y_VARIANCE_PERCENT = 0.01 # 1% of page height

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

def detect_table_structure(text_boxes, page_width, page_height):
    """
    Analyzes text boxes to detect tabular structure based on horizontal and vertical alignment.
    Returns a list of lists, where each inner list represents a detected table,
    and contains the LTTextBoxHorizontal objects forming that table.
    """
    if len(text_boxes) < MIN_ROWS_FOR_TABLE:
        return []

    # Sort text boxes primarily by Y-coordinate (top to bottom), then by X-coordinate (left to right)
    # This helps in processing rows sequentially
    text_boxes.sort(key=lambda b: (-b.y1, b.x0))

    potential_tables = []
    
    # Group text boxes into potential rows based on Y-overlap
    rows = []
    if text_boxes:
        current_row = [text_boxes[0]]
        for i in range(1, len(text_boxes)):
            # Check for significant Y-overlap or proximity to be considered part of the same row
            # Use a small tolerance based on page height
            y_tolerance = page_height * MAX_ROW_Y_VARIANCE_PERCENT
            if abs(text_boxes[i].y1 - current_row[-1].y1) < y_tolerance or \
               (text_boxes[i].y0 < current_row[-1].y1 and text_boxes[i].y1 > current_row[-1].y0):
                current_row.append(text_boxes[i])
            else:
                rows.append(sorted(current_row, key=lambda b: b.x0)) # Sort by X for columns
                current_row = [text_boxes[i]]
        rows.append(sorted(current_row, key=lambda b: b.x0))

    # Analyze rows for consistent column alignment
    current_table_candidate = []
    for row in rows:
        if len(row) >= MIN_COLUMNS_FOR_TABLE:
            if not current_table_candidate:
                current_table_candidate.append(row)
            else:
                prev_row = current_table_candidate[-1]
                # Check if current row aligns with the previous row's columns
                # This is a simplified check; more robust would involve clustering X-coordinates
                
                # If number of columns is different, it's likely a new table or not a table
                if len(row) != len(prev_row):
                    if len(current_table_candidate) >= MIN_ROWS_FOR_TABLE:
                        potential_tables.append(current_table_candidate)
                    current_table_candidate = [row]
                    continue

                # Check X-alignment of columns
                aligned = True
                x_tolerance = page_width * MAX_COLUMN_X_VARIANCE_PERCENT
                for i in range(len(row)):
                    # Check if the start of the current box is close to the start of the previous row's corresponding box
                    if abs(row[i].x0 - prev_row[i].x0) > x_tolerance:
                        aligned = False
                        break
                
                if aligned:
                    current_table_candidate.append(row)
                else:
                    # Not aligned, end current table and start a new one
                    if len(current_table_candidate) >= MIN_ROWS_FOR_TABLE:
                        potential_tables.append(current_table_candidate)
                    current_table_candidate = [row]
        else:
            # Row doesn't have enough columns, end current table candidate
            if len(current_table_candidate) >= MIN_ROWS_FOR_TABLE:
                potential_tables.append(current_table_candidate)
            current_table_candidate = []
    
    # Add the last table candidate if it meets the criteria
    if len(current_table_candidate) >= MIN_ROWS_FOR_TABLE:
        potential_tables.append(current_table_candidate)
    
    return potential_tables

def extract_table_text(table_rows):
    """
    Extracts text from detected table structure in proper row format.
    Assumes table_rows is a list of lists of LTTextBoxHorizontal objects,
    where each inner list is a row, sorted by X-coordinate.
    """
    table_lines = []
    for row_boxes in table_rows:
        # Sort boxes in row by X position to ensure correct column order
        row_boxes.sort(key=lambda x: x.x0)
        
        row_cells = []
        for box in row_boxes:
            cell_text = get_text_from_element(box)
            row_cells.append(cell_text)
        
        # Join cells with a tab or a consistent separator to indicate columns
        # Using a tab makes it easier to parse into columns later
        table_lines.append("\t".join(row_cells))
    
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
        # --- Pass 1: PyMuPDF for page dimensions, image detection, and fallback text ---
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Check for images using PyMuPDF (more reliable for raster images)
        if page.get_images():
            has_images_on_page = True

        # --- Pass 2: pdfminer.six for structured text and table detection ---
        fp_pdfminer = open(pdf_path, 'rb')
        # Adjusted LAParams for better text grouping and reading order
        # word_margin: Larger value allows more space between words before splitting
        # char_margin: Smaller value means characters must be closer to be part of the same word
        # line_margin: Larger value allows more space between lines before splitting into new text boxes
        # boxes_flow: Controls grouping of text boxes; 0.5 is a good default for reading order
        laparams = LAParams(
            all_texts=True,
            detect_vertical=True,
            word_margin=0.2,  # Increased from 0.1
            char_margin=1.0,  # Decreased from 2.0
            line_margin=0.6,  # Increased from 0.5
            boxes_flow=0.5
        )
        
        # Extract pages using pdfminer.six
        pages = extract_pages(fp_pdfminer, page_numbers=[page_number], laparams=laparams)
        
        for page_layout in pages:
            # Collect all text containers and sort them by their position (top-to-bottom, then left-to-right)
            # This is crucial for maintaining correct reading order.
            elements_on_page = []
            for element in page_layout._objs:
                # Filter out non-text and non-image elements that might interfere with layout analysis
                if isinstance(element, (LTTextContainer, LTImage, LTCurve, LTRect)):
                    elements_on_page.append(element)
            
            # Sort elements by Y-coordinate (top to bottom), then by X-coordinate (left to right)
            # This is the primary mechanism to ensure correct reading order.
            elements_on_page.sort(key=lambda e: (-e.y1, e.x0))

            text_boxes = [e for e in elements_on_page if isinstance(e, LTTextContainer)]
            
            # Detect tables using the sorted text boxes
            detected_tables = detect_table_structure(text_boxes, page_width, page_height)
            
            # Create a set of IDs for text boxes that are part of detected tables
            table_element_ids = set()
            for table_rows in detected_tables:
                for row in table_rows:
                    for box in row:
                        table_element_ids.add(id(box))
            
            # Process elements in the determined reading order
            for element in elements_on_page:
                if isinstance(element, LTImage):
                    # PyMuPDF is more reliable for general image detection, but pdfminer.six can also find them.
                    # We'll rely on PyMuPDF's has_images_on_page_actual for the overall page image check.
                    # This specific element processing is mainly for text and tables.
                    pass 
                elif isinstance(element, LTTextContainer):
                    element_id = id(element)
                    
                    # Check if this text element is part of a detected table
                    if element_id in table_element_ids:
                        # If it's the first element of a table row, add the table content
                        # This logic needs to be careful to avoid duplicating table content
                        # We'll add the table content only once when we encounter the first element of a table.
                        # To do this, we need to mark table elements as processed.
                        
                        # Find the table this element belongs to and its row index
                        found_table = None
                        for table_idx, table_rows in enumerate(detected_tables):
                            for row_idx, row in enumerate(table_rows):
                                if any(id(box) == element_id for box in row):
                                    found_table = (table_idx, row_idx, table_rows)
                                    break
                            if found_table:
                                break
                        
                        if found_table:
                            table_idx, row_idx, current_table_data = found_table
                            # Only add the table content if this is the very first box of the entire table
                            # To prevent re-adding for every cell in the table.
                            # We can achieve this by removing the table from detected_tables after processing
                            # or by using a processed_tables set.
                            
                            # A simpler approach: if this element is part of a table, skip it here
                            # and let the table extraction handle it separately.
                            # We need to ensure the table is added only once.
                            pass # Skip for now, will add tables after iterating all elements
                    else:
                        # Regular text element, not part of a table
                        text_content = get_text_from_element(element)
                        if text_content:
                            extracted_content_parts.append(text_content)
                            page_text_chars += len(text_content)
                
            # After processing all individual elements, add the detected tables.
            # This ensures tables are added as distinct blocks after regular text that precedes them.
            # To avoid duplicates, we need to ensure table text boxes are not added as regular text.
            # The previous loop already ensured this by skipping elements in table_element_ids.
            
            # Now, iterate through detected_tables and add their content.
            # We need to ensure each table is added only once.
            added_table_hashes = set() # To prevent adding the same table multiple times
            for table_rows in detected_tables:
                # Create a unique hash for the table based on its first few elements' IDs
                table_hash = hash(frozenset(id(box) for row in table_rows for box in row))
                if table_hash not in added_table_hashes:
                    table_text = extract_table_text(table_rows)
                    if table_text.strip(): # Only add if there's actual content
                        extracted_content_parts.append(f"{TABLE_START_TAG}\n{table_text}\n{TABLE_END_TAG}")
                        added_table_hashes.add(table_hash)
                        # Add characters from table text to page_text_chars count
                        page_text_chars += len(table_text)

        # Fallback to PyMuPDF if pdfminer.six didn't extract much or for empty pages
        # This fallback is now more of a safety net for truly unparseable pages by pdfminer.six
        if page_text_chars < MIN_CHARS_FOR_TEXT_PAGE and not extracted_content_parts:
            fallback_text = clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE))
            if fallback_text:
                extracted_content_parts.append(fallback_text)
                page_text_chars += len(fallback_text)
                print(f"Page {page_number + 1}: Falling back to PyMuPDF text extraction due to low pdfminer.six output.")
                
    except Exception as e:
        print(f"Error in hybrid extraction for page {page_number}: {e}")
        # Fallback to PyMuPDF only in case of an error during pdfminer.six processing
        if doc:
            page = doc.load_page(page_number)
            fallback_text = clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE))
            if fallback_text:
                extracted_content_parts.append(fallback_text)
                page_text_chars += len(fallback_text)
                print(f"Page {page_number + 1}: Error fallback to PyMuPDF text extraction.")
        
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
            print(f"Processing page {page_num + 1}...")
            page_content_text, has_significant_text, has_images_on_page_actual = extract_structured_text_hybrid(temp_file_path, page_num)

            # Page type determination logic
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
                    # Only add if there are images and significant text, and the placeholder isn't already there
                    if INLINE_IMAGE_PLACEHOLDER not in page_content_text:
                        final_page_output += f"\n\n{INLINE_IMAGE_PLACEHOLDER}"

                total_extracted_text_pages.append(final_page_output)

    # Combine all page texts with a clear page break marker
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
        # Log payload without the full extracted text for brevity
        log_payload = {k: v for k, v in n8n_payload.items() if k != 'extracted_text'}
        log_payload['extracted_text_length'] = len(n8n_payload.get('extracted_text', ''))
        print(f"Payload (excluding full text for brevity in logs): {log_payload}")

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

