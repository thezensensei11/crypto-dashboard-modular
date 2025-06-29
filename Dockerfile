# Dockerfile for the crypto data platform
# Place in: crypto-dashboard/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /data

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "scripts.init_infrastructure"]