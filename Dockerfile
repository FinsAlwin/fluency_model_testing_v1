# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for optimization
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONHASHSEED=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_KERAS_BACKEND=tensorflow

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Install CPU-only TensorFlow first
RUN pip install tensorflow-cpu

# Install any additional dependencies your app needs
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p models uploads results

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Command to run your application with memory-optimized settings
CMD ["gunicorn", \
     "--config", "gunicorn.conf.py", \
     "--timeout", "300", \
     "--workers", "1", \
     "--threads", "2", \
     "--preload", \
     "app:app"]
