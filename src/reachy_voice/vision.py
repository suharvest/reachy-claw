"""Vision client — consumes the vision-trt face stream over ZMQ.

vision-trt (separate container) detects faces and publishes
msgpack messages on a ZMQ PUB socket (topic ``vision``). This client keeps the
latest snapshot and provides:

  * ``faces_context()`` — the ``[Faces: …]`` text injected into the LLM prompt.
  * a listener callback per message — the dashboard pushes face boxes live.

Message schema (from vision-trt):
  {"timestamp": float, "frame_id": int,
   "faces": [{"bbox":[x1,y1,x2,y2] normalized, "identity": str|None, ...}, ...]}
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("reachy_voice.vision")

FACE_FRESH_S = 2.0  # snapshot older than this counts as "no faces"


class VisionClient:
    def __init__(self, zmq_url: str) -> None:
        self._url = zmq_url
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._faces: list[dict] = []      # latest [{identity, bbox, conf}]
        self._last_time = 0.0
        self._listener: Callable[[dict], None] | None = None

    def set_listener(self, fn: Callable[[dict], None] | None) -> None:
        """Called from the ZMQ thread with {"faces": [...]} per message."""
        self._listener = fn

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> bool:
        try:
            import msgpack  # noqa: F401
            import zmq  # noqa: F401
        except ImportError as e:
            logger.warning("pyzmq/msgpack missing (%s) — vision disabled", e)
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vision-zmq")
        self._thread.start()
        logger.info("vision client started: %s", self._url)
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    # ── state for consumers ──────────────────────────────────────────
    def faces_fresh(self) -> bool:
        return (time.monotonic() - self._last_time) < FACE_FRESH_S

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._faces) if self.faces_fresh() else []

    def faces_context(self) -> str:
        """'[Faces: …]' body — '' when nobody is visible."""
        faces = self.snapshot()
        if not faces:
            return ""
        named: set[str] = set()
        strangers = 0
        for f in faces:
            name = f.get("identity")
            if name:
                named.add(str(name))
            else:
                strangers += 1
        descs = sorted(named)
        if strangers == 1:
            descs.append("a stranger")
        elif strangers > 1:
            descs.append(f"{strangers} strangers")
        return ", ".join(descs)

    # ── ZMQ thread ──────────────────────────────────────────────────
    def _loop(self) -> None:
        import msgpack
        import zmq

        while self._running:
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b"vision")
            sub.setsockopt(zmq.RCVHWM, 1)       # keep only the latest frame
            sub.setsockopt(zmq.RCVTIMEO, 500)
            sub.connect(self._url)
            try:
                while self._running:
                    try:
                        parts = sub.recv_multipart()
                        msg = msgpack.unpackb(parts[1], raw=False)
                    except zmq.Again:
                        continue
                    self._handle(msg)
            except Exception as e:  # noqa: BLE001 — reconnect loop
                if self._running:
                    logger.warning("vision stream error (%s); reconnecting", e)
                    time.sleep(1.0)
            finally:
                sub.close(linger=0)
                ctx.term()

    def _handle(self, msg: dict) -> None:
        faces_in = msg.get("faces") or []
        faces = [
            {
                "identity": f.get("identity"),
                "bbox": f.get("bbox"),
            }
            for f in faces_in[:5]
        ]
        with self._lock:
            self._faces = faces
            if faces:
                self._last_time = time.monotonic()
        listener = self._listener
        if listener is not None:
            try:
                listener({"faces": faces})
            except Exception:  # noqa: BLE001 — dashboard must not kill vision
                logger.debug("vision listener failed", exc_info=True)
