import io
import requests
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes, convert_from_path # <-- Ensure convert_from_path is here
from PIL import Image
import pytesseract
import tempfile
import os

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_from_url():
    # This endpoint still handles URLs as before
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in JSON body"}), 400

    image_url = data['url']
    lang = data.get('lang', 'eng') # Default to English, or 'auto' for auto-detection

    try:
        response = requests.get(image_url)
        response.raise_for_status() # Raise an exception for bad status codes
        file_content = response.content

        # Determine file type and process
        if image_url.lower().endswith(('.pdf')):
            # For PDFs, convert to images first
            with tempfile.TemporaryDirectory() as path:
                # pdf2image needs the PDF to be on disk, so we save it temporarily
                temp_pdf_path = os.path.join(path, 'temp.pdf')
                with open(temp_pdf_path, 'wb') as f:
                    f.write(file_content)

                images = convert_from_path(temp_pdf_path)
                text = ""
                for i, image in enumerate(images):
                    # --- START OF OPTIMIZATION CODE FOR PDFS (URL) ---
                    # Resize image if it's too large
                    if image.width > 1500 or image.height > 1500:
                        image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                    # Convert to grayscale
                    processed_image = image.convert('L')
                    text += pytesseract.image_to_string(processed_image, lang=lang) + "\n"
                    del image # Explicitly delete image to free memory after processing each
                    # --- END OF OPTIMIZATION CODE FOR PDFS (URL) ---
        else:
            # For images, process directly
            image = Image.open(io.BytesIO(file_content))
            # --- START OF OPTIMIZATION CODE FOR IMAGES (URL) ---
            # Resize image if it's too large
            if image.width > 1500 or image.height > 1500:
                image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
            # Convert to grayscale
            processed_image = image.convert('L')
            text = pytesseract.image_to_string(processed_image, lang=lang)
            del image # Explicitly delete image to free memory
            # --- END OF OPTIMIZATION CODE FOR IMAGES (URL) ---


        return jsonify({"success": True, "text": text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching URL: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"OCR processing failed: {e}"}), 500


@app.route('/upload_and_ocr', methods=['POST'])
def upload_and_ocr():
    # This new endpoint handles file uploads
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    lang = request.form.get('lang', 'auto') # Get language from form data

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read the file content directly from the FileStorage object
        file_content = file.read()
        filename = file.filename

        try:
            # Determine file type based on filename extension
            if filename.lower().endswith(('.pdf')):
                # For PDFs, convert to images first
                with tempfile.TemporaryDirectory() as path:
                    # pdf2image needs the PDF to be on disk, so we save it temporarily
                    temp_pdf_path = os.path.join(path, filename)
                    with open(temp_pdf_path, 'wb') as f:
                        f.write(file_content)

                    images = convert_from_path(temp_pdf_path)
                    text = ""
                    for i, image in enumerate(images):
                        # --- START OF OPTIMIZATION CODE FOR PDFS (UPLOAD) ---
                        # Resize image if it's too large
                        if image.width > 1500 or image.height > 1500:
                            image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                        # Convert to grayscale
                        processed_image = image.convert('L')
                        text += pytesseract.image_to_string(processed_image, lang=lang) + "\n"
                        del image # Explicitly delete image to free memory after processing each
                        # --- END OF OPTIMIZATION CODE FOR PDFS (UPLOAD) ---
            else:
                # For images, process directly
                image = Image.open(io.BytesIO(file_content))
                # --- START OF OPTIMIZATION CODE FOR IMAGES (UPLOAD) ---
                # Resize image if it's too large
                if image.width > 1500 or image.height > 1500:
                    image.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
                # Convert to grayscale
                processed_image = image.convert('L')
                text = pytesseract.image_to_string(processed_image, lang=lang)
                del image # Explicitly delete image to free memory
                # --- END OF OPTIMIZATION CODE FOR IMAGES (UPLOAD) ---

            return jsonify({"success": True, "text": text})

        except Exception as e:
            return jsonify({"error": f"OCR processing failed: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 8000))
