FROM python:3.9-slim

# System libs required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.8 GB vs default CUDA wheel).
# pip sees torch already satisfied when processing requirements.txt next.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 6000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6000"]
