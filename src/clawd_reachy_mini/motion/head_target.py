"""Shared head tracking bus for fusing multiple tracking sources.

Multiple plugins (face tracker, etc.) publish HeadTarget updates.
The MotionPlugin consumes the fused result to drive the robot head.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HeadTarget:
    """A head tracking target from any source."""

    yaw: float = 0.0  # degrees (positive = left)
    pitch: float = 0.0  # degrees (positive = up)
    confidence: float = 0.0  # 0 = no data, 1 = high confidence
    source: str = ""  # "face", "doa", "none"
    timestamp: float = field(default_factory=time.monotonic)


class HeadTargetBus:
    """Thread-safe bus for publishing and fusing head tracking targets.

    Fusion priority:
      1. Face tracking (when recent, < face_timeout)
      2. DOA tracking (when recent, < doa_timeout)
      3. Neutral (gradual decay to center)
    """

    def __init__(
        self,
        face_timeout: float = 0.5,
        doa_timeout: float = 3.0,
    ) -> None:
        self._lock = threading.Lock()
        self._face_target = HeadTarget(source="face")
        self._doa_target = HeadTarget(source="doa")
        self._face_timeout = face_timeout
        self._doa_timeout = doa_timeout

    def publish(self, target: HeadTarget) -> None:
        """Publish a new head target from a tracking source."""
        with self._lock:
            if target.source == "face":
                self._face_target = target
            elif target.source == "doa":
                self._doa_target = target

    def get_fused_target(self) -> HeadTarget:
        """Get the best available head target using priority fusion."""
        now = time.monotonic()
        with self._lock:
            face_age = now - self._face_target.timestamp
            doa_age = now - self._doa_target.timestamp

            # Face takes priority when recent and confident
            if self._face_target.confidence > 0 and face_age < self._face_timeout:
                return HeadTarget(
                    yaw=self._face_target.yaw,
                    pitch=self._face_target.pitch,
                    confidence=self._face_target.confidence,
                    source="face",
                    timestamp=self._face_target.timestamp,
                )

            # DOA fallback when face unavailable
            if self._doa_target.confidence > 0 and doa_age < self._doa_timeout:
                return HeadTarget(
                    yaw=self._doa_target.yaw,
                    pitch=0.0,
                    confidence=self._doa_target.confidence,
                    source="doa",
                    timestamp=self._doa_target.timestamp,
                )

            # No valid tracking data
            return HeadTarget(
                yaw=0.0,
                pitch=0.0,
                confidence=0.0,
                source="none",
                timestamp=now,
            )
