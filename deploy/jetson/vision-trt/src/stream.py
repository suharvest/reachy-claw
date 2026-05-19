"""Video streaming — holds pre-encoded JPEG frames for MJPEG output.

The GstCameraCapture pipeline handles all encoding in hardware.
This module simply buffers the latest JPEG for HTTP streaming.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class VideoStreamer:
    """JPEG buffer for MJPEG HTTP streaming.

    Tracks both the most recent frame and the most recent *non-empty* frame,
    so the HTTP MJPEG generator can fall back to a last-known frame during
    brief upstream stalls (e.g. ``appsink.try_pull_sample()`` returning None)
    instead of going silent and letting the client time out.
    """

    # How long a last-known frame stays usable as a fallback.
    # Bounded so the viewer doesn't see an indefinitely frozen image.
    STALE_FALLBACK_MAX_AGE_S = 5.0

    def __init__(self, port: int = 8632):
        self._port = port
        self._latest_jpeg: bytes | None = None
        self._last_known_jpeg: bytes | None = None
        self._last_known_ts: float = 0.0
        self._frame_lock = threading.Lock()
        self._has_clients = False

    def set_jpeg(self, data: bytes) -> None:
        """Store a pre-encoded JPEG frame.

        Non-empty frames also refresh the last-known fallback buffer.
        """
        with self._frame_lock:
            self._latest_jpeg = data
            if data:
                self._last_known_jpeg = data
                self._last_known_ts = time.monotonic()

    def get_jpeg(self) -> bytes | None:
        """Get latest JPEG frame for HTTP streaming.

        Returns the freshest frame, falling back to the last-known frame if
        it's still within ``STALE_FALLBACK_MAX_AGE_S``. Returns None only
        when there is no usable frame at all, in which case the HTTP layer
        is expected to emit a heartbeat instead of going silent.
        """
        self._has_clients = True
        with self._frame_lock:
            if self._latest_jpeg:
                return self._latest_jpeg
            if (
                self._last_known_jpeg
                and (time.monotonic() - self._last_known_ts)
                <= self.STALE_FALLBACK_MAX_AGE_S
            ):
                return self._last_known_jpeg
            return None

    def close(self):
        """No-op (capture pipeline owns lifecycle)."""
        pass
