# Use an official Python base image with a Debian distribution (slim for smaller size)
FROM python:3.9-slim-bullseye

# Set environment variables for Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Tesseract OCR (if still needed for other purposes, though not used by this app.py for OCR)
# and essential image processing libs, PyMuPDF dependencies.
# We are no longer relying on Poppler, as PyMuPDF handles PDF rendering.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-heb \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PyMuPDF dependencies (often included in -dev packages or python wheels, but good to be explicit for common issues)
    # No specific apt-get for fitz/PyMuPDF itself is usually needed beyond build-essentials and zlib1g-dev if building from source
    # The pip install will handle the wheel.
    && rm -rf /var/lib/apt/lists/*

# Set the TESSDATA_PREFIX environment variable (if Tesseract is kept for other uses)
ENV TESSDATA_PREFIX /usr/share/tesseract-ocr/4.00/tessdata/

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port. Render will automatically map this.
EXPOSE 8000

# Command to run your application using Gunicorn.
CMD gunicorn app:app -w 4 --bind 0.0.0.0:$PORT --timeout 120
