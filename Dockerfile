FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 60922

# Run the application with uvicorn
CMD ["uvicorn", "skreach:app", "--host", "0.0.0.0", "--port", "60922"]
