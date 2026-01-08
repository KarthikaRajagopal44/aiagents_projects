# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some PDF/Math libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create folders for DB and Data
RUN mkdir -p db data

# Expose the FastAPI port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]