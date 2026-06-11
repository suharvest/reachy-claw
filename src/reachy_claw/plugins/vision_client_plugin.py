"""VisionClientPlugin -- remote TensorRT vision service integration.

Receives inference results from the vision-trt container via ZMQ SUB
to drive head tracking and emotion mapping.  The vision-trt container
captures camera frames directly via GStreamer (no shared memory needed).
"""

import asyncio
import logging
import time

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
        self._face_trigger_stable_s = getattr(
            cfg, "vision_interaction_face_stable_s", 1.0
        )
        self._face_trigger_min_area = getattr(
            cfg, "vision_interaction_face_min_area", 0.008
        )
        self._face_trigger_cooldown_s = getattr(
            cfg, "vision_interaction_face_cooldown_s", 12.0
        )
        self._emotion_threshold = cfg.vision_emotion_threshold
        self._emotion_cooldown = cfg.vision_emotion_cooldown

        # Smoothing state (same as FaceTrackerPlugin)
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._smooth_roll = 0.0
        self._last_face_time = 0.0
        self._face_lost_published = False
        self._face_trigger_seen_since = 0.0
        self._last_face_trigger_time = 0.0

        # Body yaw: accumulated angle (closed-loop centering)
        self._body_yaw_acc = 0.0
        self._body_yaw_gain = cfg.vision_body_yaw_gain

        # Emotion state
        self._last_emotion = "neutral"
        self._last_emotion_conf = 0.0
        self._last_emotion_time = 0.0
        self._emotion_sustain = cfg.vision_emotion_sustain
        self._emotion_log_time = 0.0  # throttled debug logging

        # Identity (shared with app for conversation context)
        self.current_identity = None

        # Multi-face state (for monologue mode)
        self._last_face_count: int = 0
        self._last_faces_summary: list[dict] = []

        # Rest window pause
        self._paused = False

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
        self._loop = asyncio.get_running_loop()

        self.app.events.subscribe("rest_start", self._on_rest_start_handler)
        self.app.events.subscribe("rest_end", self._on_rest_end_handler)

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
            if self._paused:
                import time as _time
                _time.sleep(0.5)
                continue
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

            self._process_vision_message(msg, now=time.monotonic())

    def _face_area(self, face: dict) -> float:
        bbox = face.get("bbox")
        if not bbox or len(bbox) < 4:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return width * height

    def _is_effective_tracking_face(self, face: dict) -> bool:
        return self._face_area(face) >= self._face_trigger_min_area

    def _process_vision_message(self, msg: dict, *, now: float | None = None) -> None:
        """Process one ZMQ vision result.

        Dashboard still sees all faces, but motion/conversation attention only
        follows faces that are large enough to represent a nearby visitor.
        """
        if now is None:
            now = time.monotonic()

        # Emit vision_faces for dashboard (capped at 5)
        faces_for_dash = [
            {
                "bbox": f.get("bbox", []),
                "emotion": _EMOTION_REMAP.get(f.get("emotion", ""), "neutral"),
                "emotion_confidence": float(f.get("emotion_confidence", 0)),
                "identity": f.get("identity"),
            }
            for f in msg.get("faces", [])[:5]
        ]
        self._emit_threadsafe("vision_faces", {"faces": faces_for_dash})

        # Check for smile capture event
        capture = msg.get("capture")
        if capture and capture.get("event"):
            self._emit_threadsafe("smile_capture", {
                "count": capture.get("count", 0),
                "file": capture.get("file"),
            })

        faces = [
            f for f in msg.get("faces", [])
            if self._is_effective_tracking_face(f)
        ]

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
            return

        # Update multi-face summary (sorted by bbox area descending)
        self._last_face_count = len(faces)
        sorted_faces = sorted(
            faces,
            key=self._face_area,
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
        primary = sorted_faces[0]

        self._last_face_time = now
        self._face_lost_published = False
        self._maybe_trigger_face_interaction(primary, now=now)

        # Head tracking (same horizontal split as FaceTrackerPlugin)
        center = primary.get("center")
        if center:
            face_x, face_y = float(center[0]), float(center[1])

            if (
                abs(face_x - self._smooth_x) >= self._deadzone
                or abs(face_y - self._smooth_y) >= self._deadzone
            ):
                self._smooth_x += self._smoothing_alpha * (face_x - self._smooth_x)
                self._smooth_y += self._smoothing_alpha * (face_y - self._smooth_y)

            # Body rotation: proportional centering. The remote vision stream
            # is reliable enough for coarse horizontal attention, but its
            # bbox/landmark coordinates are too noisy for head pitch/roll on
            # the exhibition camera, so keep the head neutral here.
            body_yaw = -self._smooth_x * self._max_yaw
            pitch = 0.0
            roll = 0.0

            self.app.head_targets.publish(
                HeadTarget(
                    yaw=0.0, pitch=pitch, roll=roll,
                    body_yaw=body_yaw, confidence=0.9,
                    source="face", timestamp=now,
                )
            )

        # Emotion mapping
        emotion = primary.get("emotion")
        emotion_conf = primary.get("emotion_confidence", 0.0)

        # Throttled diagnostic log — every 3s show raw detection
        if emotion and (now - self._emotion_log_time) >= 3.0:
            mapped_name = _EMOTION_REMAP.get(emotion, "?")
            below = emotion_conf < self._emotion_threshold
            cd = (now - self._last_emotion_time) < self._emotion_cooldown
            logger.info(
                f"Emotion raw: {emotion}→{mapped_name} "
                f"conf={emotion_conf:.2f} "
                f"{'BELOW_THRESH' if below else 'ok'} "
                f"{'COOLDOWN' if cd else 'ok'}"
            )
            self._emotion_log_time = now

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
                    self._last_emotion = mapped
                    self._last_emotion_conf = emotion_conf
                    self._last_emotion_time = now
                    reason = (
                        "change" if changed
                        else "sustain" if sustained
                        else "conf_jump"
                    )
                    logger.info(
                        "Vision emotion observed: %s -> %s "
                        "(conf=%.2f, %s; motion suppressed)",
                        emotion,
                        mapped,
                        emotion_conf,
                        reason,
                    )

        # Identity
        identity = primary.get("identity")
        if identity != self.current_identity:
            self.current_identity = identity
            if identity:
                logger.info(f"Face identified: {identity}")

    def _maybe_trigger_face_interaction(self, face: dict, *, now: float) -> None:
        bbox = face.get("bbox")
        if not bbox or len(bbox) < 4:
            self._face_trigger_seen_since = 0.0
            return

        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        area = width * height
        if area < self._face_trigger_min_area:
            self._face_trigger_seen_since = 0.0
            return

        if self._face_trigger_seen_since <= 0.0:
            self._face_trigger_seen_since = now
            return
        if (now - self._face_trigger_seen_since) < self._face_trigger_stable_s:
            return
        if (now - self._last_face_trigger_time) < self._face_trigger_cooldown_s:
            return

        motion = self.app.get_plugin("motion")
        if motion and hasattr(motion, "open_interaction_window"):
            motion.open_interaction_window()
            self.app.emotions.queue_emotion("attention")
            self._last_face_trigger_time = now

    def _emit_threadsafe(self, event: str, data: dict) -> None:
        """Emit EventBus event from a background thread."""
        loop = getattr(self, "_loop", None)
        if not loop:
            return
        for cb in self.app.events._subscribers.get(event, []):
            try:
                import inspect
                if inspect.iscoroutinefunction(cb):
                    asyncio.run_coroutine_threadsafe(cb(data), loop)
                else:
                    cb(data)
            except Exception:
                pass

    async def stop(self):
        self._running = False
        self.app.events.unsubscribe("rest_start", self._on_rest_start_handler)
        self.app.events.unsubscribe("rest_end", self._on_rest_end_handler)

    async def on_rest_start(self) -> None:
        self._paused = True

    async def on_rest_end(self) -> None:
        self._paused = False

    def _on_rest_start_handler(self, _data: dict) -> None:
        import asyncio
        asyncio.create_task(self.on_rest_start())

    def _on_rest_end_handler(self, _data: dict) -> None:
        import asyncio
        asyncio.create_task(self.on_rest_end())
