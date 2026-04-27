# Self-contained image: HailoRT 4.21 C library + cp311 Python binding bundled in.
# Host only needs the kernel driver (hailo_pci) + /dev/hailo0; no userspace bind-mounts.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgcc-s1 \
    libstdc++6 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    v4l-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── HailoRT 4.21 C library (.deb) + Python 3.11 binding (wheel) ────────────
COPY debs/hailort_4.21.0_arm64.deb /tmp/
COPY debs/hailort-4.21.0-cp311-cp311-linux_aarch64.whl /tmp/
# Docker build does NOT create /.dockerenv (only runtime does).
# HailoRT postinst checks for /.dockerenv to skip systemctl calls.
# Create it before dpkg install to fool the postinst script.
RUN touch /.dockerenv \
    && dpkg -i /tmp/hailort_4.21.0_arm64.deb \
    && pip3 install --no-cache-dir --break-system-packages /tmp/hailort-4.21.0-cp311-cp311-linux_aarch64.whl \
    && rm /tmp/hailort_4.21.0_arm64.deb /tmp/hailort-4.21.0-cp311-cp311-linux_aarch64.whl \
    && rm /.dockerenv

RUN pip3 install --no-cache-dir --break-system-packages \
    'pyzmq>=25.0' \
    'msgpack>=1.0' \
    'fastapi>=0.100' \
    'uvicorn>=0.23' \
    'opencv-python-headless>=4.8' \
    'numpy>=1.24,<2'

COPY producer.py hailo_pipeline.py face_db.py /app/
COPY models/ /app/models/

WORKDIR /app

EXPOSE 8630 8631

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8630/', timeout=2)"

CMD ["python3", "producer.py"]
