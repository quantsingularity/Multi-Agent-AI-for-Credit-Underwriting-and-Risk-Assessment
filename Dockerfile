FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    tesseract-ocr \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy entire project
COPY . .

# Create output directories
RUN mkdir -p results/logs results/models figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RANDOM_SEED=42

# Default command
CMD ["python", "-m", "pytest", "credit/tests/", "-v"]
