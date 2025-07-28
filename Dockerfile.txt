# Use an official Python base image with a Debian distribution (slim for smaller size)
FROM python:3.9-slim-bullseye

# Set environment variables for Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Tesseract OCR and its language packs, Poppler utilities, and essential image processing libs
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-heb \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port. Render will automatically map this.
# Use $PORT to align with Render's dynamic port assignment.
EXPOSE 8000

# Command to run your application using Gunicorn.
# Use $PORT for dynamic binding.
# Command to run your application using Gunicorn.
CMD gunicorn app:app -w 4 --bind 0.0.0.0:$PORT --timeout 120
