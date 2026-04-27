FROM python:3.11-slim-bookworm

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specs and install
COPY pyproject.toml emotion_onnx.py producer.py face_db.py* ./
RUN pip install --no-cache-dir .

# Copy emotion model (FERPlus 8-class ONNX)
COPY models/ /app/models/

# Create data directories
RUN mkdir -p /app/data/captures /app/data/faces

ENV CAPTURE_DIR=/app/data/captures
ENV FACE_DB_DIR=/app/data/faces
ENV EMOTION_MODEL_PATH=/app/models/emotion-ferplus-8.onnx

EXPOSE 8630 8631
CMD ["python", "producer.py"]