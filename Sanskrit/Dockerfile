# ── Sanskrit MNIST — Dockerfile ─────────────────────────────────────────
# Builds a self-contained training environment.
#
# Build:
#   docker build -t sanskrit-mnist .
#
# Train (mount your data and output dirs):
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     sanskrit-mnist python train.py --epochs 25
#
# Evaluate:
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     sanskrit-mnist python evaluate.py
#
# GPU support: replace the base image with pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="Sanskrit MNIST — Devanagari character recognition with PyTorch CNN"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/       src/
COPY train.py   .
COPY evaluate.py .
COPY predict.py  .
COPY scripts/   scripts/

# Data and output dirs (mount at runtime)
RUN mkdir -p data/raw data/processed assets

# Default command
CMD ["python", "train.py"]
