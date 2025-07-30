import io
import requests
import os
import json
import tempfile

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextLineHorizontal, LTFigure, LTImage, LTCurve, LTChar, LTRect, LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

# PyMuPDF for robust PDF handling and image checks
import fitz # PyMuPDF is imported as fitz

app = Flask(__name__)

# --- Configuration from Environment Variables ---
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')
RETURN_DIRECTLY_FOR_TESTING = os.environ.get('RETURN_DIRECTLY_FOR_TESTING', 'false').lower() == 'true'
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true' # For future module additions

# --- Constants for Structured Extraction ---
IMAGE_PAGE_PLACEHOLDER = "[IMAGE PAGE - NO OCR PERFORMED]"
INLINE_IMAGE_PLACEHOLDER = "[Image detected]"

# Threshold for considering a page text-heavy vs potentially image-only
# If a page contains at least this many text characters, it's considered text-heavy.
MIN_CHARS_FOR_TEXT_PAGE = 50 # Adjust as needed. Very low to ensure even sparse text is processed.

# --- Helper Functions for Structured Extraction ---

def clean_text(text):
    """Removes null bytes and other common problematic characters."""
    return text.replace('\x00', '').strip()

def extract_structured_text(pdf_file_obj, page_number):
    """
    Extracts text and infers basic structure (headers, tables, inline images)
    from a specific PDF page using pdfminer.six.
    Returns (structured_text_content, has_significant_text, has_images).
    """
    # Rewind file_obj for each page extraction if it's the same stream
    pdf_file_obj.seek(0)
    
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams(
        char_margin=1.0, # Adjust for tighter character grouping
        line_margin=0.5, # Adjust for tighter line grouping
        word_margin=0.1, # Adjust for tighter word grouping
        boxes_flow=0.5,  # How text boxes are ordered (0.5 good for columns)
        detect_vertical=True # Detect vertical text
    )
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    extracted_content = []
    page_text_chars = 0
    has_images_on_page = False

    # Get specific page using PyMuPDF for image detection and to load only that page
    doc = fitz.open("pdf", pdf_file_obj.read()) # Re-read file content
    page = doc.load_page(page_number)
    pdf_file_obj.seek(0) # Reset for next iteration if needed
    
    # Check for raster images using PyMuPDF (more reliable than pdfminer's LTImage alone for all types)
    if page.get_images():
        has_images_on_page = True

    # Use pdfminer.six to get detailed layout objects
    pages = extract_pages(pdf_file_obj, page_numbers=[page_number], laparams=laparams)
    
    for p in pages:
        for element in p._objs:
            if isinstance(element, LTTextBoxHorizontal):
                text_block = clean_text(element.get_text())
                if not text_block:
                    continue
                page_text_chars += len(text_block)

                # Heuristic for Headers: Check font size, position, etc. (basic)
                # This is a very simple heuristic. For robust header detection,
                # you'd need to inspect element.layout.get_text_attributes() to check font sizes
                # and compare to average text size on page.
                # For now, just include the text with line breaks.
                extracted_content.append(text_block)

            elif isinstance(element, LTImage):
                # Basic inline image placeholder
                extracted_content.append(INLINE_IMAGE_PLACEHOLDER)
                has_images_on_page = True # Confirm image presence via pdfminer
            
            elif isinstance(element, LTCurve) or isinstance(element, LTRect):
                # These might indicate table lines or borders.
                # For advanced table extraction, you'd analyze their geometry.
                # For now, we rely on LTTextBoxHorizontal's spatial awareness for table text.
                pass # Not directly adding to text output, but useful for analysis

    device.close()
    retstr.close()
    doc.close()

    final_text = "\n\n".join(extracted_content).strip()

    # Heuristic for table-like structures:
    # A simple way to get better table representation is to extract text with whitespace preservation.
    # PyMuPDF's get_text("text") combined with its flags can often do a good job for aligned text.
    # We will prioritize pdfminer's detailed blocks, but if a page has significant whitespace structure,
    # we can try to improve it for RAG.
    # For now, we'll rely on LTTextBoxHorizontal's inherent grouping for better RAG.
    # A true Markdown table conversion is complex without dedicated parsers.
    # The output will be text with strong whitespace preservation, which is good for RAG.

    return final_text, (page_text_chars >= MIN_CHARS_FOR_TEXT_PAGE), has_images_on_page

# --- Main Document Processing Function ---
def process_document(file_content, filename, original_payload):
    total_extracted_text_pages = []
    num_pages = 0
    
    # Store the PDF temporarily for PyMuPDF and pdfminer.six
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        # PyMuPDF for basic page count and robust image detection
        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF with PyMuPDF: {e}. It might be corrupted or not a valid PDF.")

        # Re-open the file-like object for pdfminer to process page by page
        # It's better to pass file path/content once and let extract_structured_text handle seeking
        
        for page_num in range(num_pages):
            # Create a BytesIO object for the current page to be read by pdfminer
            # This is more efficient than reading the entire file repeatedly if pdfminer supports page_numbers
            with open(temp_file_path, 'rb') as fp:
                page_content_text, has_significant_text, has_images = extract_structured_text(fp, page_num)

            if not has_significant_text and has_images:
                # If no significant text, AND it has images, assume it's an image-only page (e.g., scan)
                print(f"Page {page_num + 1} detected as image-heavy (no significant text, has images). Inserting placeholder.")
                total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
            elif not has_significant_text and not has_images:
                # Page is empty (no text, no images)
                print(f"Page {page_num + 1} detected as empty.")
                total_extracted_text_pages.append("") # Or a specific placeholder for empty pages if preferred
            else:
                # Page has significant text, even if sparse, extract it.
                print(f"Page {page_num + 1} has significant text. Extracting structured content.")
                total_extracted_text_pages.append(page_content_text)

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
            n8n_response = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=120)
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
