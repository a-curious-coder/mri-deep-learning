# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port the app runs on
EXPOSE 5000

# Run the application - single worker (TF model is loaded per-process, so
# more workers means duplicating that memory) and a longer timeout since the
# first classify request has to cold-load the model.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "60", "app:app"]