"""MediaPipe-based head tracker for face detection.

Uses MediaPipe Face Detection for lightweight face tracking on Mac/desktop.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .head_tracker import HeadTracker

logger = logging.getLogger(__name__)


class MediaPipeTracker(HeadTracker):
    """Head tracker using MediaPipe Face Detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        min_face_size: float = 0.05,
    ) -> None:
        """Initialize MediaPipe-based head tracker.

        Args:
            min_detection_confidence: Minimum confidence for detection.
            model_selection: 0 for short-range (2m), 1 for long-range (5m).
            min_face_size: Minimum face bbox size as fraction of frame (0-1).
                Faces smaller than this are ignored. Default 0.05 (5% of frame).
        """
        import mediapipe as mp

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )
        self._min_face_size = min_face_size
        logger.info("MediaPipe face detection initialized (min_face_size=%.0f%%)", min_face_size * 100)

    def get_head_position(
        self, img: NDArray[np.uint8]
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[float]]:
        try:
            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)
            rgb_img = np.ascontiguousarray(img[:, :, ::-1])
            results = self.face_detection.process(rgb_img)

            if not results.detections:
                return None, None

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Ignore faces that are too small (person too far away)
            if bbox.width < self._min_face_size or bbox.height < self._min_face_size:
                return None, None

            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2

            # Convert to [-1, 1] range
            norm_x = center_x * 2.0 - 1.0
            norm_y = center_y * 2.0 - 1.0

            face_center = np.array([norm_x, norm_y], dtype=np.float32)

            # Estimate roll from eye keypoints
            roll = 0.0
            keypoints = detection.location_data.relative_keypoints
            if len(keypoints) >= 2:
                left_eye = keypoints[0]
                right_eye = keypoints[1]
                dx = right_eye.x - left_eye.x
                dy = right_eye.y - left_eye.y
                roll = float(np.arctan2(dy, dx))

            return face_center, roll

        except Exception as e:
            logger.error(f"Error in head position detection: {e}")
            return None, None

    def close(self) -> None:
        if hasattr(self, "face_detection"):
            self.face_detection.close()

    def __del__(self):
        self.close()
