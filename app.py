import io
import requests
import os
import json
import tempfile

from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LTChar, LTTextBoxHorizontal
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

# Minimum number of text characters on a page to consider it "text-heavy"
# and not an image-only page (even if it has images). Very low to ensure
# even sparse text is extracted.
MIN_CHARS_FOR_TEXT_PAGE = 50 

# --- Helper Functions for Structured Extraction ---

def clean_text(text):
    """Removes null bytes and other common problematic characters."""
    return text.replace('\x00', '').strip()

def extract_structured_text(pdf_path, page_number):
    """
    Extracts text content and inserts inline image placeholders from a specific PDF page.
    Prioritizes PyMuPDF for text layout and pdfminer.six for image detection.
    Returns (structured_text_content, has_significant_text, has_images).
    """
    extracted_content_parts = []
    page_text_chars = 0 # Initialize here
    has_images_on_page = False # Initialize here
    
    doc = None
    fp_pdfminer = None

    try:
        # --- Pass 1: PyMuPDF for main text content (good for tables and general layout) ---
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        
        # Extract text while preserving whitespace and ligatures for better layout (especially tables)
        text_from_pymu = clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES))
        extracted_content_parts.append(text_from_pymu)
        page_text_chars += len(text_from_pymu)

        # Update has_images_on_page based on PyMuPDF's direct image check
        if page.get_images():
            has_images_on_page = True

        # --- Pass 2: pdfminer.six for specific object detection (e.g., inline images) ---
        # Only if the page actually has images and some text content
        if has_images_on_page and page_text_chars > 0:
            fp_pdfminer = open(pdf_path, 'rb')
            laparams = LAParams() 
            pages = extract_pages(fp_pdfminer, page_numbers=[page_number], laparams=laparams)
            
            for p in pages:
                for element in p._objs:
                    if isinstance(element, LTImage):
                        # For simple inline detection, we'll just check if it's there
                        # More advanced would involve getting bbox and inserting at position
                        # For this version, we ensure the image flag is true if pdfminer sees it.
                        pass # has_images_on_page is already true from PyMuPDF check
                    elif isinstance(element, LTTextContainer):
                        # This loop primarily for image detection within a page that has text
                        pass
        
    finally:
        if doc:
            doc.close()
        if fp_pdfminer:
            fp_pdfminer.close()

    final_text = "\n\n".join(extracted_content_parts).strip()

    # Determine if the page has significant text based on character count
    has_significant_text = (page_text_chars >= MIN_CHARS_FOR_TEXT_PAGE)

    return final_text, has_significant_text, has_images_on_page # Use the flag updated by PyMuPDF check

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
            # The returned `has_images_on_page_actual` and `has_significant_text` are derived directly from the page content analysis.
            page_content_text, has_significant_text, has_images_on_page_actual = extract_structured_text(temp_file_path, page_num)

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
                
                # If images were detected on a page that also has text, we append the inline placeholder.
                # This simple appending means it will appear at the end of the page's text.
                # More precise inline insertion (e.g., in the middle of a paragraph) is very complex
                # and beyond the scope of current lightweight implementation.
                final_page_output = page_content_text
                if has_images_on_page_actual and has_significant_text:
                     # Only add if the text doesn't already implicitly contain it (from its original source)
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
