FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml producer.py ./
RUN pip install --no-cache-dir .

ENV CAPTURE_DIR=/app/data/captures
RUN mkdir -p /app/data/captures

EXPOSE 8630 8631
CMD ["python", "producer.py"]
