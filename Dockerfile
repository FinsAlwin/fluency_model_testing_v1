# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Set environment variable for CPU-only operation
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install CPU-only TensorFlow first
RUN pip install tensorflow-cpu

# Install any additional dependencies your app needs
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
