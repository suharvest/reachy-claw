# ── Builder: compile PyCUDA in full JetPack (has CUDA toolkit + gcc) ──
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pycuda

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Runtime: slim l4t-base ───────────────────────────────────────────
FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Runtime deps — use pip opencv (not apt) for numpy compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    libgl1 libglib2.0-0 v4l-utils psmisc \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder (includes pycuda + all requirements)
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install compatible numpy + opencv (from pip, not apt)
RUN pip3 install --no-cache-dir 'numpy>=1.24,<2' opencv-python-headless

COPY src/ /app/src/
COPY static/ /app/static/
COPY entrypoint.sh /app/entrypoint.sh
RUN mkdir -p /app/models /app/engines /app/data && chmod +x /app/entrypoint.sh

WORKDIR /app

EXPOSE 8630 8631 8632

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=60s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8630/health', timeout=2)"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8630"]
