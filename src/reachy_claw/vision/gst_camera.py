"""Subprocess-based GStreamer camera capture.

Spawns ``gst-launch-1.0`` as a child process and reads raw BGR frames via pipe.
This bypasses a known PyGObject bug where v4l2src gets stuck at PAUSED→PLAYING
inside Docker containers (affects GStreamer 1.22–1.26+).
"""

from __future__ import annotations

import logging
import subprocess
import shutil
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Camera capabilities for Reachy Mini Camera (UVC)
_DEFAULT_WIDTH = 1920
_DEFAULT_HEIGHT = 1080
_DEFAULT_FRAMERATE = 5  # YUY2 at 1920x1080 supports up to 5fps


class GstSubprocessCamera:
    """Read frames from a V4L2 camera via gst-launch-1.0 subprocess."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        framerate: int = _DEFAULT_FRAMERATE,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.framerate = framerate
        self._frame_size = width * height * 3  # BGR
        self._proc: Optional[subprocess.Popen] = None

    @staticmethod
    def available() -> bool:
        """Check if gst-launch-1.0 is installed."""
        return shutil.which("gst-launch-1.0") is not None

    @staticmethod
    def find_device() -> Optional[str]:
        """Find Reachy Mini camera device path via GStreamer device monitor."""
        try:
            result = subprocess.run(
                ["gst-device-monitor-1.0", "Video/Source"],
                capture_output=True, text=True, timeout=5,
            )
            current_path = None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                # GStreamer 1.22+ uses "api.v4l2.path", older uses "device.path"
                if stripped.startswith(("api.v4l2.path", "device.path")):
                    current_path = stripped.split("=", 1)[1].strip()
                if "Reachy" in stripped and current_path:
                    return current_path
            # Fallback: return first v4l2 device found
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith(("api.v4l2.path", "device.path")):
                    return stripped.split("=", 1)[1].strip()
        except Exception:
            pass
        return None

    def _build_pipelines(self) -> list[list[str]]:
        """Return pipeline commands to try, in order of preference."""
        base = ["gst-launch-1.0", "-q", "v4l2src", f"device={self.device}"]
        sink = ["fdsink", "fd=1"]
        bgr_caps = f"video/x-raw,format=BGR,width={self.width},height={self.height}"
        # 1. MJPEG → jpegdec (high fps, works without GST_V4L2_USE_LIBV4L2)
        mjpeg = [*base, "!", "jpegdec", "!", "videoconvert", "!", bgr_caps, "!", *sink]
        # 2. Raw YUY2 → videoconvert (works with GST_V4L2_USE_LIBV4L2=1)
        raw = [
            *base, "!",
            f"video/x-raw,width={self.width},height={self.height},"
            f"framerate={self.framerate}/1", "!",
            "videoconvert", "!", bgr_caps, "!", *sink,
        ]
        return [mjpeg, raw]

    def open(self) -> bool:
        """Start the gst-launch subprocess. Tries MJPEG first, then raw."""
        if self._proc is not None:
            return True

        for cmd in self._build_pipelines():
            if self._try_pipeline(cmd):
                return True

        return False

    def _try_pipeline(self, cmd: list[str]) -> bool:
        """Try to start a pipeline and read one frame within timeout."""
        import selectors

        desc = "mjpeg" if "jpegdec" in cmd else "raw"
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Read first frame with timeout to verify pipeline works
            sel = selectors.DefaultSelector()
            sel.register(self._proc.stdout, selectors.EVENT_READ)
            ready = sel.select(timeout=10.0)  # 10s for first frame
            sel.close()
            if not ready:
                stderr = self._proc.stderr.read(500) if self._proc.stderr else b""
                logger.info(
                    f"GstSubprocessCamera: {desc} pipeline timeout "
                    f"from {self.device}. stderr={stderr.decode(errors='replace')}"
                )
                self.close()
                return False
            data = self._proc.stdout.read(self._frame_size)
            if len(data) != self._frame_size:
                logger.info(
                    f"GstSubprocessCamera: {desc} first frame incomplete "
                    f"({len(data)}/{self._frame_size} bytes)"
                )
                self.close()
                return False
            logger.info(
                f"GstSubprocessCamera: opened {self.device} "
                f"({self.width}x{self.height} BGR, {desc})"
            )
            # Store first frame for immediate use
            self._first_frame = np.frombuffer(data, dtype=np.uint8).reshape(
                self.height, self.width, 3
            ).copy()
            return True
        except Exception as e:
            logger.error(f"GstSubprocessCamera: {desc} failed to start: {e}")
            self.close()
            return False

    def read(self) -> Optional[np.ndarray]:
        """Read one BGR frame. Returns None on error."""
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return frame

        if self._proc is None or self._proc.poll() is not None:
            return None

        try:
            data = self._proc.stdout.read(self._frame_size)
            if len(data) != self._frame_size:
                return None
            return np.frombuffer(data, dtype=np.uint8).reshape(
                self.height, self.width, 3
            ).copy()
        except Exception:
            return None

    def close(self) -> None:
        """Stop the subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._first_frame = None

    @property
    def is_opened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
