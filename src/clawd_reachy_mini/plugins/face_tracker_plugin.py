"""FaceTrackerPlugin -- camera-based face detection for head tracking.

Uses MediaPipe (Mac/desktop) for face detection and publishes
head tracking targets to the HeadTargetBus.
"""

import asyncio
import logging
import time

from ..motion.head_target import HeadTarget
from ..plugin import Plugin

logger = logging.getLogger(__name__)


class FaceTrackerPlugin(Plugin):
    """Camera-based face tracking plugin."""

    name = "face_tracker"

    def __init__(self, app):
        super().__init__(app)
        cfg = app.config
        self._tracker_type = cfg.vision_tracker_type
        self._camera_index = cfg.vision_camera_index
        self._max_yaw = cfg.vision_max_yaw
        self._max_pitch = cfg.vision_max_pitch
        self._smoothing_alpha = cfg.vision_smoothing_alpha
        self._deadzone = cfg.vision_deadzone
        self._face_lost_delay = cfg.vision_face_lost_delay

        self._tracker = None
        self._cap = None

        # Smoothing state
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._last_face_time = 0.0

    def setup(self) -> bool:
        """Check camera and tracker availability."""
        if self._tracker_type == "none":
            logger.info("Face tracker disabled by config")
            return False

        # Check mediapipe
        if self._tracker_type == "mediapipe":
            try:
                import mediapipe  # noqa: F401
            except ImportError:
                logger.info("mediapipe not installed, face tracker skipped")
                return False

        # Check camera
        try:
            import cv2

            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                cap.release()
                logger.info(
                    f"Camera {self._camera_index} not available, face tracker skipped"
                )
                return False
            cap.release()
        except ImportError:
            logger.info("opencv-python not installed, face tracker skipped")
            return False

        return True

    async def start(self):
        import cv2

        from ..vision.head_tracker import create_head_tracker

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            logger.error("Failed to open camera for face tracking")
            return

        try:
            self._tracker = create_head_tracker(self._tracker_type)
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            self._cap.release()
            return

        logger.info(
            f"Face tracker started (type={self._tracker_type}, camera={self._camera_index})"
        )

        self._face_lost_published = False

        try:
            while self._running:
                # Run blocking cv2.read() in a thread to avoid stalling the event loop
                ret, frame = await asyncio.to_thread(self._cap.read)
                if not ret or frame is None:
                    await asyncio.sleep(0.04)
                    continue

                # Run face detection in thread to avoid blocking
                eye_center, _roll = await asyncio.to_thread(
                    self._tracker.get_head_position, frame
                )

                now = time.monotonic()

                if eye_center is not None:
                    face_x, face_y = float(eye_center[0]), float(eye_center[1])
                    self._last_face_time = now
                    self._face_lost_published = False

                    if (
                        abs(face_x - self._smooth_x) >= self._deadzone
                        or abs(face_y - self._smooth_y) >= self._deadzone
                    ):
                        self._smooth_x += self._smoothing_alpha * (
                            face_x - self._smooth_x
                        )
                        self._smooth_y += self._smoothing_alpha * (
                            face_y - self._smooth_y
                        )

                    # Negative x = face is left = robot turns left (positive yaw)
                    yaw = -self._smooth_x * self._max_yaw
                    pitch = -self._smooth_y * self._max_pitch

                    self.app.head_targets.publish(
                        HeadTarget(
                            yaw=yaw,
                            pitch=pitch,
                            confidence=0.9,
                            source="face",
                            timestamp=now,
                        )
                    )

                elif (now - self._last_face_time) > self._face_lost_delay:
                    # Only publish face-lost once to avoid spamming the bus
                    if not self._face_lost_published:
                        self.app.head_targets.publish(
                            HeadTarget(
                                yaw=0.0,
                                pitch=0.0,
                                confidence=0.0,
                                source="face",
                                timestamp=now,
                            )
                        )
                        self._face_lost_published = True

                await asyncio.sleep(0.04)  # ~25Hz

        finally:
            if self._tracker:
                self._tracker.close()
            if self._cap:
                self._cap.release()
            logger.info("Face tracker stopped")

    async def stop(self):
        self._running = False
