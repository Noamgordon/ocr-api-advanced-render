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
# The URL of your n8n workflow webhook
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL')

# Placeholder for future optional modules (e.g., Google OCR)
# Currently, its value doesn't change behavior for this base version,
# but it's here for future expansion.
ENABLE_GOOGLE_OCR = os.environ.get('ENABLE_GOOGLE_OCR', 'false').lower() == 'true'

# --- Heuristic for Image-Heavy Pages ---
# If text covers less than this percentage of the page area, consider it image-heavy.
# You might need to adjust this threshold based on your documents.
TEXT_DENSITY_THRESHOLD = 0.05

# Placeholder for pages identified as image-heavy when OCR is inactive
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
    # You might need more complex sorting for truly complex layouts
    page_blocks.sort(key=lambda block: (block[1], block[0])) # Sort by y0 then x0

    extracted_lines = []
    # Check for actual images in the page objects for a more robust image-heavy detection
    contains_raster_image = False
    for img_info in page_layout.get_images():
        xref = img_info[0] # xref of the image
        s = doc.extract_image(xref) # dict with image data
        if s and s['ext'] in ['png', 'jpeg', 'jpg', 'tiff', 'bmp', 'jp2']: # Check for common raster formats
            contains_raster_image = True
            break


    for block in page_blocks:
        block_text = block[4] # The text content of the block
        bbox = fitz.Rect(block[0], block[1], block[2], block[3]) # Bounding box of the block

        # Estimate text density from block content (rough estimate)
        # More precise density would involve char-level analysis from pdfminer.six's extract_pages
        # For RAG, we want the content as-is, but use density for the image check
        total_text_area += bbox.width * bbox.height # Accumulate area of text blocks

        # Simple RAG optimization: Add a newline between blocks to maintain paragraph separation
        # This can be further refined for tables, lists, etc.
        extracted_lines.append(block_text.strip())

    # Check for text density and actual image presence
    if total_page_area > 0:
        text_density = total_text_area / total_page_area
        is_image_heavy_heuristic = (text_density < TEXT_DENSITY_THRESHOLD) or contains_raster_image
    else:
        is_image_heavy_heuristic = True # If page has no area, assume image heavy or blank

    # Join lines to form page text, adding extra newlines for paragraph breaks
    # This is a basic step towards RAG optimization; advanced parsing would use markdown
    combined_text = "\n\n".join(filter(None, extracted_lines)) # Filter out empty strings before joining

    doc.close()
    return combined_text, is_image_heavy_heuristic

# --- Main Document Processing Function ---
def process_document(file_content, filename, original_payload):
    total_extracted_text_pages = []
    
    # Store the PDF temporarily for PyMuPDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)

        try:
            doc = fitz.open(temp_file_path)
            num_pages = doc.page_count
            doc.close() # Close after getting page count, re-open per-page if needed for other ops

            for page_num in range(num_pages):
                page_text, is_image_heavy = extract_rag_optimized_text(temp_file_path, page_num)

                if is_image_heavy:
                    print(f"Page {page_num + 1} detected as image-heavy. Inserting placeholder.")
                    total_extracted_text_pages.append(IMAGE_PAGE_PLACEHOLDER)
                else:
                    print(f"Page {page_num + 1} detected as text-heavy. Extracting text.")
                    total_extracted_text_pages.append(page_text)

        except Exception as e:
            # Handle cases where PyMuPDF or pdfminer.six fails to open/process the PDF
            print(f"Error processing PDF with native extractors: {e}")
            raise RuntimeError(f"Failed to process PDF: {e}. It might be corrupted or complex.")

    # Combine all page texts
    full_document_text = "\n\n--- PAGE BREAK ---\n\n".join(total_extracted_text_pages)

    # Prepare payload for n8n
    n8n_payload = {
        "extracted_text": full_document_text,
        "filename": filename,
        "page_count": num_pages,
        # Include the original payload data
        **original_payload
    }

    print(f"Sending data to n8n workflow at: {N8N_WEBHOOK_URL}")
    print(f"Payload (excluding full text for brevity in logs): { {k: v for k, v in n8n_payload.items() if k != 'extracted_text'} }")


    if not N8N_WEBHOOK_URL:
        raise ValueError("N8N_WEBHOOK_URL environment variable is not set.")

    # Send to n8n workflow
    try:
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=120) # Increased timeout
        n8n_response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Assuming n8n returns a JSON response indicating success/fail/error
        n8n_result = n8n_response.json()
        print(f"n8n workflow response: {n8n_result}")
        return n8n_result # Relay n8n's response directly

    except requests.exceptions.Timeout:
        raise RuntimeError("Request to n8n workflow timed out.")
    except requests.exceptions.RequestException as e:
        # Catch network errors, HTTP errors etc.
        print(f"Error communicating with n8n workflow: {e}")
        # Try to parse n8n's response even if it's an error, for more detail
        try:
            error_details = n8n_response.json()
        except json.JSONDecodeError:
            error_details = n8n_response.text
        raise RuntimeError(f"Failed to send data to n8n workflow: {e}. Details: {error_details}")
    except Exception as e:
        print(f"Unexpected error during n8n communication: {e}")
        raise RuntimeError(f"An unexpected error occurred during n8n communication: {e}")


@app.route('/process_document', methods=['POST'])
def process_document_endpoint():
    # Retrieve the original payload from the JSON body
    # This endpoint now expects a 'file_url' or a 'file' and the metadata
    
    # Extract metadata first
    original_payload = {
        "subject_id": request.json.get("subject_id"),
        "user_id": request.json.get("user_id"),
        "subject_type": request.json.get("subject_type")
    }

    file_content = None
    filename = "document.pdf" # Default name, will be updated if file is uploaded or from URL

    # Case 1: File provided as a multipart form data upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        file_content = file.read()
        filename = file.filename
        print(f"Received file upload: {filename}")
    
    # Case 2: File provided as a URL in the JSON body
    elif 'file_url' in request.json:
        file_url = request.json['file_url']
        if not file_url:
            return jsonify({"error": "Missing 'file_url' in JSON body"}), 400
        
        try:
            response = requests.get(file_url, timeout=60) # Increased timeout for download
            response.raise_for_status()
            file_content = response.content
            filename = file_url.split('/')[-1].split('?')[0] # Basic filename extraction
            print(f"Received file via URL: {file_url}")
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error fetching URL: {e}"}), 400
    else:
        return jsonify({"error": "No file or file_url provided, or 'file' is missing in multipart form data."}), 400

    if not file_content:
        return jsonify({"error": "Could not retrieve file content."}), 500

    try:
        # Process the document and send to n8n, then relay n8n's response
        n8n_result = process_document(file_content, filename, original_payload)
        return jsonify(n8n_result), 200 # Relay n8n's success/fail/error

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except RuntimeError as e:
        # Catch errors explicitly raised by process_document for n8n communication
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"General error in /process_document endpoint: {e}")
        return jsonify({"error": f"Document processing failed: {e}"}), 500

if __name__ == '__main__':
    # For local development, set a dummy n8n URL or ensure it's in your env
    if not N8N_WEBHOOK_URL:
        print("WARNING: N8N_WEBHOOK_URL environment variable is not set. n8n integration will fail.")
        print("For local testing without n8n, you might mock process_document.")

    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 8000))
