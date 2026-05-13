FROM python:3.11-slim-bookworm

# No apt packages needed — opencv-python-headless + mediapipe work without
# X11/OpenGL libs (no libgl1, no mesa, no llvm).

WORKDIR /app

# Install mediapipe without heavy optional deps (no jax, scipy, sounddevice, etc.)
RUN pip install --no-cache-dir --no-deps mediapipe && \
    pip install --no-cache-dir \
    "protobuf<5" \
    absl-py \
    flatbuffers \
    "numpy<2" \
    opencv-python-headless \
    attrs \
    pyzmq \
    msgpack \
    fastapi \
    uvicorn \
    onnxruntime

# App code + model
COPY emotion_onnx.py producer.py face_db.py* ./
COPY models/ /app/models/

RUN mkdir -p /app/data/captures /app/data/faces

ENV CAPTURE_DIR=/app/data/captures
ENV FACE_DB_DIR=/app/data/faces
ENV EMOTION_MODEL_PATH=/app/models/emotion-ferplus-8.onnx

EXPOSE 8630 8631
CMD ["python", "producer.py"]