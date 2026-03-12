"""Environment-based configuration for vision-trt service."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_camera(preferred: str) -> str:
    """Find camera device, auto-detecting if preferred doesn't exist."""
    if os.path.exists(preferred):
        return preferred

    # Scan /sys/class/video4linux for Reachy Mini Camera
    v4l_dir = Path("/sys/class/video4linux")
    if v4l_dir.exists():
        for entry in sorted(v4l_dir.iterdir()):
            name_file = entry / "name"
            if name_file.exists():
                name = name_file.read_text().strip()
                if "Reachy" in name or "reachy" in name:
                    dev = f"/dev/{entry.name}"
                    if os.path.exists(dev):
                        logger.info(f"Auto-detected camera: {dev} ({name})")
                        return dev

    # Fallback: first /dev/videoN that exists
    for i in range(8):
        dev = f"/dev/video{i}"
        if os.path.exists(dev):
            logger.info(f"Fallback camera device: {dev}")
            return dev

    logger.warning(f"No camera found, using configured: {preferred}")
    return preferred


class VisionConfig:
    """Configuration loaded from environment variables."""

    # Camera capture
    CAMERA_DEVICE: str = _find_camera(os.getenv("CAMERA_DEVICE", "/dev/video0"))
    # Camera MJPEG resolution (set via V4L2 ioctl before GStreamer)
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    STREAM_WIDTH: int = int(os.getenv("STREAM_WIDTH", "640"))

    # ZMQ publisher
    ZMQ_PUB_PORT: int = int(os.getenv("ZMQ_PUB_PORT", "8631"))

    # HTTP/WebSocket server
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8630"))

    # RTSP/WebRTC video stream
    STREAM_PORT: int = int(os.getenv("STREAM_PORT", "8632"))

    # Model paths
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/app/models")
    ENGINE_DIR: str = os.getenv("ENGINE_DIR", "/app/engines")

    # Face database
    DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")

    # Inference settings
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.5"))
    RECOGNITION_THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.4"))
    INPUT_WIDTH: int = int(os.getenv("INPUT_WIDTH", "640"))
    INPUT_HEIGHT: int = int(os.getenv("INPUT_HEIGHT", "640"))

    # Emotion smoothing
    EMOTION_WINDOW_SIZE: int = int(os.getenv("EMOTION_WINDOW_SIZE", "5"))

    # Performance
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "10"))


config = VisionConfig()
