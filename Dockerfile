FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for cryptography, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port (Railway sets PORT env var)
ENV PORT=8000

CMD uvicorn bot.server:app --host 0.0.0.0 --port $PORT
