"""Abstract head tracker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class HeadTracker(ABC):
    """Abstract interface for head position tracking from camera frames."""

    @abstractmethod
    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[float]]:
        """Get head position from a camera frame.

        Args:
            img: Input image (BGR format).

        Returns:
            Tuple of (face_center in [-1,1] coords, roll_angle in radians).
            Returns (None, None) if no face detected.
        """
        ...

    def close(self) -> None:
        """Release resources."""


def create_head_tracker(tracker_type: str = "mediapipe", **kwargs) -> HeadTracker:
    """Factory function to create a head tracker.

    Args:
        tracker_type: "mediapipe" or "none".
    """
    if tracker_type == "mediapipe":
        from .mediapipe_tracker import MediaPipeTracker

        return MediaPipeTracker(**kwargs)
    elif tracker_type == "none":
        return _NoopTracker()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


class _NoopTracker(HeadTracker):
    def get_head_position(self, img):
        return None, None
