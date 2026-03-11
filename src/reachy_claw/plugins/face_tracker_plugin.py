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
        self._camera_source = cfg.vision_camera_source
        self._camera_index = cfg.vision_camera_index
        self._max_yaw = cfg.vision_max_yaw
        self._max_pitch = cfg.vision_max_pitch
        self._smoothing_alpha = cfg.vision_smoothing_alpha
        self._deadzone = cfg.vision_deadzone
        self._face_lost_delay = cfg.vision_face_lost_delay

        self._tracker = None
        self._cap = None
        self._gst_cam = None  # GstSubprocessCamera
        self._use_sdk_camera = False

        # Smoothing state
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._last_face_time = 0.0

    def _has_sdk_camera(self) -> bool:
        """Check if the Reachy SDK camera is available."""
        reachy = self.app.reachy
        if reachy is None:
            return False
        media = getattr(reachy, "media_manager", None)
        if media is None:
            return False
        # Check if camera object was initialized (pipeline may not have frames yet)
        return getattr(media, "camera", None) is not None

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

        # Determine camera source (retry — SDK camera may not be ready immediately)
        if self._camera_source in ("sdk", "auto"):
            for attempt in range(5):
                if self._has_sdk_camera():
                    self._use_sdk_camera = True
                    logger.info("Using SDK camera (zenoh) for face tracking")
                    return True
                if attempt < 4:
                    import time as _time
                    _time.sleep(0.5)
            if self._camera_source == "sdk":
                logger.warning("SDK camera not available after retries")
                # Fall through to gst-subprocess / OpenCV

        # Try gst-launch subprocess (works in Docker where PyGObject v4l2src is broken)
        if self._try_gst_subprocess():
            return True

        # OpenCV camera check
        try:
            import cv2

            cap = self._open_cv_camera(cv2)
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

    def _try_gst_subprocess(self) -> bool:
        """Try to set up gst-launch subprocess camera."""
        from ..vision.gst_camera import GstSubprocessCamera

        if not GstSubprocessCamera.available():
            logger.debug("gst-launch-1.0 not found, skipping subprocess camera")
            return False

        device = f"/dev/video{self._camera_index}"
        # Try to find Reachy camera specifically
        found = GstSubprocessCamera.find_device()
        logger.info(f"Trying gst-subprocess camera (device={found or device})")
        if found:
            device = found

        cam = GstSubprocessCamera(device=device)
        if cam.open():
            logger.info(f"Using gst-subprocess camera: {device}")
            self._gst_cam = cam
            return True

        cam.close()
        return False

    def _open_cv_camera(self, cv2):
        """Open camera with best available backend (FFMPEG fallback for ARM64)."""
        cap = cv2.VideoCapture(self._camera_index)
        if cap.isOpened():
            return cap
        cap.release()
        # V4L2 backend may not be compiled in (e.g. ARM64 pip wheels) — try FFMPEG
        dev_path = f"/dev/video{self._camera_index}"
        cap = cv2.VideoCapture(dev_path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            logger.info(f"Opened camera via FFMPEG backend: {dev_path}")
        return cap

    async def start(self):
        from ..vision.head_tracker import create_head_tracker

        if not self._use_sdk_camera and self._gst_cam is None:
            import cv2

            self._cap = self._open_cv_camera(cv2)
            if not self._cap.isOpened():
                logger.error("Failed to open camera for face tracking")
                return

        try:
            self._tracker = create_head_tracker(self._tracker_type)
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            if self._cap:
                self._cap.release()
            if self._gst_cam:
                self._gst_cam.close()
            return

        if self._gst_cam:
            source_desc = f"gst-subprocess/{self._gst_cam.device}"
        elif self._use_sdk_camera:
            source_desc = "sdk/zenoh"
        else:
            source_desc = f"opencv/{self._camera_index}"
        logger.info(
            f"Face tracker started (type={self._tracker_type}, source={source_desc})"
        )

        self._face_lost_published = False

        consecutive_errors = 0
        max_consecutive_errors = 50  # ~2s at 25Hz — then back off

        try:
            while self._running:
                try:
                    if self._gst_cam:
                        frame = await asyncio.to_thread(self._gst_cam.read)
                    elif self._use_sdk_camera:
                        frame = await asyncio.to_thread(self.app.reachy.media.get_frame)
                    else:
                        ret, frame = await asyncio.to_thread(self._cap.read)
                        frame = frame if ret else None
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                        logger.warning(f"Frame capture error ({consecutive_errors}): {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        await asyncio.sleep(1.0)  # back off on persistent failure
                    else:
                        await asyncio.sleep(0.04)
                    continue

                if frame is None:
                    await asyncio.sleep(0.04)
                    continue

                consecutive_errors = 0

                # Run face detection in thread to avoid blocking
                try:
                    eye_center, _roll = await asyncio.to_thread(
                        self._tracker.get_head_position, frame
                    )
                except Exception as e:
                    logger.warning(f"Face detection error: {e}")
                    await asyncio.sleep(0.04)
                    continue

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
            if self._gst_cam:
                self._gst_cam.close()
            logger.info("Face tracker stopped")

    async def stop(self):
        self._running = False
