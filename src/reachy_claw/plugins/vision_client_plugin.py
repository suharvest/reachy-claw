"""VisionClientPlugin -- remote TensorRT vision service integration.

Receives inference results from the vision-trt container via ZMQ SUB
to drive head tracking and emotion mapping.  The vision-trt container
captures camera frames directly via GStreamer (no shared memory needed).
"""

import asyncio
import logging
import time

import numpy as np

from ..motion.head_target import HeadTarget
from ..plugin import Plugin

logger = logging.getLogger(__name__)

# HSEmotion output → EmotionMapper key
_EMOTION_REMAP = {
    "Anger": "angry",
    "Contempt": "neutral",
    "Disgust": "angry",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprised",
    # Lowercase variants (in case vision-trt sends lowercase)
    "anger": "angry",
    "contempt": "neutral",
    "disgust": "angry",
    "fear": "fear",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "angry": "angry",
}


class VisionClientPlugin(Plugin):
    """Remote vision service client: ZMQ result consumer."""

    name = "vision_client"

    def __init__(self, app):
        super().__init__(app)
        cfg = app.config
        self._zmq_url = cfg.vision_service_url
        self._max_yaw = cfg.vision_max_yaw
        self._max_pitch = cfg.vision_max_pitch
        self._pitch_offset = cfg.vision_pitch_offset
        self._max_roll = cfg.vision_max_roll
        self._smoothing_alpha = cfg.vision_smoothing_alpha
        self._deadzone = cfg.vision_deadzone
        self._face_lost_delay = cfg.vision_face_lost_delay
        self._emotion_threshold = cfg.vision_emotion_threshold
        self._emotion_cooldown = cfg.vision_emotion_cooldown

        # Smoothing state (same as FaceTrackerPlugin)
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._smooth_roll = 0.0
        self._last_face_time = 0.0
        self._face_lost_published = False

        # Body yaw: accumulated angle (closed-loop centering)
        self._body_yaw_acc = 0.0
        self._body_yaw_gain = cfg.vision_body_yaw_gain

        # Emotion state
        self._last_emotion = "neutral"
        self._last_emotion_conf = 0.0
        self._last_emotion_time = 0.0
        self._emotion_sustain = cfg.vision_emotion_sustain

        # Identity (shared with app for conversation context)
        self.current_identity = None

        # Multi-face state (for monologue mode)
        self._last_face_count: int = 0
        self._last_faces_summary: list[dict] = []

    def setup(self) -> bool:
        """Check ZMQ availability."""
        # Check ZMQ
        try:
            import zmq  # noqa: F401
            import msgpack  # noqa: F401
        except ImportError as e:
            logger.warning(f"ZMQ/msgpack not installed ({e}), vision client skipped")
            return False

        return True

    async def start(self):
        import zmq
        import msgpack

        self._running = True
        logger.info(f"Vision client started (zmq={self._zmq_url})")

        result_task = asyncio.create_task(
            asyncio.to_thread(self._result_loop_sync, zmq, msgpack)
        )

        try:
            await result_task
        finally:
            result_task.cancel()
            logger.info("Vision client stopped")

    def _result_loop_sync(self, zmq, msgpack):
        """Receive inference results from vision-trt via ZMQ (sync, runs in thread).

        Auto-reconnects on crash with exponential backoff.
        """
        while self._running:
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b"vision")
            # NOTE: CONFLATE=1 breaks multipart recv — use RCVHWM=1 instead
            sub.setsockopt(zmq.RCVHWM, 1)
            sub.setsockopt(zmq.RCVTIMEO, 500)
            sub.connect(self._zmq_url)
            logger.info(f"ZMQ subscriber connected to {self._zmq_url}")

            try:
                self._result_loop_inner(sub, zmq, msgpack)
            except Exception:
                logger.exception("ZMQ result loop crashed, reconnecting in 2s")
            finally:
                sub.close()
                ctx.term()

            if self._running:
                import time
                time.sleep(2)

        logger.info("ZMQ subscriber closed")

    def _result_loop_inner(self, sub, zmq, msgpack):
        """Inner loop for ZMQ result processing."""
        recv_count = 0
        while self._running:
            try:
                parts = sub.recv_multipart()
                msg = msgpack.unpackb(parts[1], raw=False)
            except zmq.Again:
                continue
            except Exception as e:
                logger.debug(f"ZMQ recv error: {e}")
                continue

            recv_count += 1
            if recv_count <= 3 or recv_count % 100 == 0:
                faces_n = len(msg.get("faces", []))
                logger.info(f"ZMQ recv #{recv_count}: {faces_n} faces")

            faces = msg.get("faces", [])
            now = time.monotonic()

            if not faces:
                self._last_face_count = 0
                self._last_faces_summary = []
                if (now - self._last_face_time) > self._face_lost_delay:
                    if not self._face_lost_published:
                        self.app.head_targets.publish(
                            HeadTarget(
                                yaw=0.0, pitch=0.0, confidence=0.0,
                                source="face", timestamp=now,
                            )
                        )
                        self._face_lost_published = True
                        self.current_identity = None
                continue

            # Update multi-face summary (sorted by bbox area descending)
            self._last_face_count = len(faces)
            sorted_faces = sorted(
                faces,
                key=lambda f: (
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                    if "bbox" in f else 0
                ),
                reverse=True,
            )
            self._last_faces_summary = [
                {
                    "identity": f.get("identity"),
                    "emotion": _EMOTION_REMAP.get(f.get("emotion", ""), "neutral"),
                }
                for f in sorted_faces
            ]

            # Select primary face (largest bbox area)
            primary = max(
                faces,
                key=lambda f: (
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                    if "bbox" in f else 0
                ),
            )

            self._last_face_time = now
            self._face_lost_published = False

            # Head tracking (same logic as FaceTrackerPlugin)
            center = primary.get("center")
            if center:
                face_x, face_y = float(center[0]), float(center[1])

                if (
                    abs(face_x - self._smooth_x) >= self._deadzone
                    or abs(face_y - self._smooth_y) >= self._deadzone
                ):
                    self._smooth_x += self._smoothing_alpha * (face_x - self._smooth_x)
                    self._smooth_y += self._smoothing_alpha * (face_y - self._smooth_y)

                # Body rotation: proportional centering
                # Camera is on robot head, so as body rotates, face naturally
                # moves toward center — no integral needed
                body_yaw = -self._smooth_x * self._max_yaw
                # Head pitch: face in upper frame (y<0) → look up (negative pitch in SDK)
                # SDK convention: positive pitch = look DOWN, negative = look UP
                pitch = self._smooth_y * self._max_pitch - self._pitch_offset

                # Roll estimation from eye landmarks (points 0=left_eye, 1=right_eye)
                roll = 0.0
                landmarks = primary.get("landmarks")
                if landmarks and len(landmarks) >= 2:
                    left_eye, right_eye = landmarks[0], landmarks[1]
                    eye_dx = right_eye[0] - left_eye[0]
                    eye_dy = right_eye[1] - left_eye[1]
                    raw_roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
                    # EMA smoothing
                    self._smooth_roll += self._smoothing_alpha * (raw_roll - self._smooth_roll)
                    roll = float(np.clip(self._smooth_roll, -self._max_roll, self._max_roll))

                # Head yaw follows body yaw for natural look
                head_yaw = -self._smooth_x * self._max_pitch  # gentle head turn toward face

                self.app.head_targets.publish(
                    HeadTarget(
                        yaw=head_yaw, pitch=pitch, roll=roll,
                        body_yaw=body_yaw, confidence=0.9,
                        source="face", timestamp=now,
                    )
                )

            # Emotion mapping
            emotion = primary.get("emotion")
            emotion_conf = primary.get("emotion_confidence", 0.0)
            if (
                emotion
                and emotion_conf >= self._emotion_threshold
                and (now - self._last_emotion_time) >= self._emotion_cooldown
            ):
                mapped = _EMOTION_REMAP.get(emotion)
                if mapped:
                    changed = mapped != self._last_emotion
                    # Resend same emotion if: sustained for N seconds,
                    # or confidence jumped significantly
                    sustained = (
                        not changed
                        and (now - self._last_emotion_time) >= self._emotion_sustain
                    )
                    conf_jump = (
                        not changed
                        and (emotion_conf - self._last_emotion_conf) >= 0.15
                    )
                    if changed or sustained or conf_jump:
                        self.app.emotions.queue_emotion(mapped)
                        self._last_emotion = mapped
                        self._last_emotion_conf = emotion_conf
                        self._last_emotion_time = now
                        reason = (
                            "change" if changed
                            else "sustain" if sustained
                            else "conf_jump"
                        )
                        logger.debug(
                            f"Vision emotion: {emotion} → {mapped} "
                            f"(conf={emotion_conf:.2f}, {reason})"
                        )

            # Identity
            identity = primary.get("identity")
            if identity != self.current_identity:
                self.current_identity = identity
                if identity:
                    logger.info(f"Face identified: {identity}")

    async def stop(self):
        self._running = False
