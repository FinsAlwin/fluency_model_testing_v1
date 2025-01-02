# Use Python 3.11.10 slim image as base
FROM python:3.11.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libavcodec-extra \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libpostproc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads results models

# Expose port
EXPOSE 80

# Set environment variables
ENV CUDA_VISIBLE_DEVICES="-1" \
    TF_ENABLE_ONEDNN_OPTS="0" \
    TF_CPP_MIN_LOG_LEVEL="2"

# Command to run the application
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 

CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]

