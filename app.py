import io
import requests
import os
import json
from flask import Flask, request, jsonify

# PDF parsing and layout analysis
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure, LTImage

# PyMuPDF for robust PDF handling and future image rendering
import fitz # PyMuPDF is imported as fitz

app = Flask(__name__)

# --- Configuration from Environment Variables ---
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')

# New environment variable for testing mode
RETURN_DIRECTLY_FOR_TESTING = os.environ.get('RETURN_DIRECTLY_FOR_TESTING', 'false').lower() == 'true'

# Placeholder for future optional modules (e.g., Google OCR)
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true'

# --- Heuristic for Image-Heavy Pages ---
TEXT_DENSITY_THRESHOLD = 0.05
IMAGE_PAGE_PLACEHOLDER = "[IMAGE PAGE - NO OCR PERFORMED]"

# --- Helper Function to Analyze Page Layout and Extract RAG-Optimized Text ---
def extract_rag_optimized_text(pdf_path, page_num):
    """
    Extracts text from a specific PDF page using pdfminer.six,
    attempting to preserve layout for RAG optimization.
    It also checks for the presence of images or very low text density.
    Returns (extracted_text, is_image_heavy_heuristic).
    """
    doc = fitz.open(pdf_path)
    page_layout = doc.load_page(page_num)

    total_text_area = 0
    total_page_area = page_layout.rect.width * page_layout.rect.height

    page_blocks = page_layout.get_text("blocks") # Get text blocks with coordinates

    # Sort blocks for reading order (left-to-right, top-to-bottom)
    page_blocks.sort(key=lambda block: (block[1], block[0])) # Sort by y0 then x0

    extracted_lines = []
    contains_raster_image = False
    for img_info in page_layout.get_images():
        xref = img_info[0] # xref of the image
        s = doc.extract_image(xref) # dict with image data
        if s and s['ext'] in ['png', 'jpeg', 'jpg', 'tiff', 'bmp', 'jp2']:
            contains_raster_image = True
            break

    for block in page_blocks:
        block_text = block[4]
        bbox = fitz.Rect(block[0], block[1], block[2], block[3])

        total_text_area += bbox.width * bbox.height

        extracted_lines.append(block_text.strip())

    if total_page_area > 0:
        text_density = total_text_area / total_page_area
        is_image_heavy_heuristic = (text_density < TEXT_DENSITY_THRESHOLD) or contains_raster_image
    else:
        is_image_heavy_heuristic = True

    combined_text = "\n\n".join(filter(None, extracted_lines))
    
    doc.close()
    return combined_text, is_image_heavy_heuristic

# --- Main Document Processing Function ---
def process_document(file_content, filename, original_payload):
    total_extracted_text_pages = []
    num_pages = 0 # Initialize num_pages

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            
            for page_num in range(num_pages):
                page_text, is_image_heavy = extract_rag_optimized_text(temp_file_path, page_num)

                if is_image_heavy:
                    print(f"Page {page_num + 1} detected as image-heavy. Inserting placeholder.")
                    total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
                else:
                    print(f"Page {page_num + 1} detected as text-heavy. Extracting text.")
                    total_extracted_text_pages.append(page_text)
            doc.close() # Close the document after processing all pages

        except Exception as e:
            print(f"Error processing PDF with native extractors: {e}")
            raise RuntimeError(f"Failed to process PDF: {e}. It might be corrupted or complex.")

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
            except (json.JSONDecodeError, AttributeError): # AttributeError if n8n_response is None due to network error
                error_details = n8n_response.text if n8n_response else "No response body."
            raise RuntimeError(f"Failed to send data to n8n workflow: {e}. Details: {error_details}")
        except Exception as e:
            print(f"An unexpected error occurred during n8n communication: {e}")
            raise RuntimeError(f"An unexpected error occurred during n8n communication: {e}")


@app.route('/process_document', methods=['POST'])
def process_document_endpoint():
    # Extract metadata first from the JSON body
    request_json = request.get_json(silent=True)
    if not request_json:
        # If not JSON, it might be a file upload with form fields
        original_payload = {
            "subject_id": request.form.get("subject_id"),
            "user_id": request.form.get("user_id"),
            "subject_type": request.form.get("subject_type")
        }
    else:
        original_payload = {
            "subject_id": request_json.get("subject_id"),
            "user_id": request_json.get("user_id"),
            "subject_type": request_json.get("subject_type")
        }


    file_content = None
    filename = "document.pdf"

    # Case 1: File provided as a multipart form data upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        file_content = file.read()
        filename = file.filename
        print(f"Received file upload: {filename}")
    
    # Case 2: File provided as a URL in the JSON body
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
