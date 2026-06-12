"""DailyLogPlugin — writes daily interaction events to SQLite.

Subscribes to the EventBus and persists timestamped rows into the shared
`Database` (app.db). Replaces the prior jsonl-based logging.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..plugin import Plugin
from ..storage.db import Database

logger = logging.getLogger(__name__)

EMOTION_SAMPLE_INTERVAL = 60
FACE_SAMPLE_INTERVAL = 60


class DailyLogPlugin(Plugin):
    name = "daily_log"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._db: Database = app.db
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._last_emotion: str | None = None
        self._last_emotion_ts: float = 0
        self._last_face_ts: float = 0
        self._pending_asr: dict | None = None

    def setup(self) -> bool:
        if self._db is None:
            logger.warning("Database not initialized, daily_log disabled")
            return False
        return True

    async def start(self) -> None:
        bus = self.app.events
        bus.subscribe("emotion", self._on_emotion)
        bus.subscribe("asr_final", self._on_asr_final)
        bus.subscribe("llm_end", self._on_llm_end)
        bus.subscribe("vision_faces", self._on_vision_faces)
        bus.subscribe("smile_capture", self._on_smile_capture)
        bus.subscribe("observation", self._on_observation)

        while self._running:
            try:
                kind, entry = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                self._write(kind, entry)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("DailyLog writer error: %s", e)

    async def stop(self) -> None:
        await super().stop()
        bus = self.app.events
        bus.unsubscribe("emotion", self._on_emotion)
        bus.unsubscribe("asr_final", self._on_asr_final)
        bus.unsubscribe("llm_end", self._on_llm_end)
        bus.unsubscribe("vision_faces", self._on_vision_faces)
        bus.unsubscribe("smile_capture", self._on_smile_capture)
        bus.unsubscribe("observation", self._on_observation)
        while not self._queue.empty():
            try:
                kind, entry = self._queue.get_nowait()
                self._write(kind, entry)
            except Exception:
                break

    # ── handlers ─────────────────────────────────────────────────────────
    def _on_emotion(self, data: Any) -> None:
        now = time.time()
        # Handle both {"emotion": "curious"} and plain string
        if isinstance(data, dict):
            emotion = data.get("emotion", str(data))
        else:
            emotion = str(data)
        if emotion == self._last_emotion and (now - self._last_emotion_ts) < EMOTION_SAMPLE_INTERVAL:
            return
        self._last_emotion = emotion
        self._last_emotion_ts = now
        self._queue.put_nowait(("emotion", {"label": emotion}))

    def _on_asr_final(self, data: Any) -> None:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        if not text.strip():
            return
        self._pending_asr = {"text": text}
        self._queue.put_nowait(("asr_user", {"text": text}))

    def _on_llm_end(self, data: Any) -> None:
        if isinstance(data, dict):
            reply = data.get("full_text", data.get("text", ""))
            emotion = data.get("emotion") or None
        else:
            reply = str(data)
            emotion = None
        if not reply.strip():
            return
        self._queue.put_nowait(
            ("asr_reachy", {"text": reply, "emotion": emotion})
        )
        self._pending_asr = None

    def _on_vision_faces(self, data: Any) -> None:
        now = time.time()
        if (now - self._last_face_ts) < FACE_SAMPLE_INTERVAL:
            return
        self._last_face_ts = now
        if isinstance(data, dict):
            faces = data.get("faces", [])
            count = len(faces)
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0
        self._queue.put_nowait(("face", {"count": count, "smile_count": 0, "capture_path": None}))

    def _on_smile_capture(self, data: Any) -> None:
        # Handle both "file" (from vision_client) and "path" (plan expectation)
        if isinstance(data, dict):
            path = data.get("path") or data.get("file")
            count = data.get("count", 1)
        else:
            path = None
            count = 1
        self._queue.put_nowait(
            ("face", {"count": 1, "smile_count": count, "capture_path": path})
        )

    def _on_observation(self, data: Any) -> None:
        if isinstance(data, dict):
            text = data.get("text", data.get("observation", ""))
            emotion = data.get("emotion") or None
        else:
            text = str(data)
            emotion = None
        if not text.strip():
            return
        self._queue.put_nowait(("thought", {"text": text, "emotion": emotion}))

    # ── writer ───────────────────────────────────────────────────────────
    def _write(self, kind: str, entry: dict) -> None:
        ts = int(time.time())
        try:
            if kind == "emotion":
                self._db.record_emotion(ts=ts, label=entry["label"])
            elif kind == "asr_user":
                self._db.record_asr(
                    ts=ts, role="user", text=entry["text"], emotion=None
                )
            elif kind == "asr_reachy":
                self._db.record_asr(
                    ts=ts,
                    role="reachy",
                    text=entry["text"],
                    emotion=entry.get("emotion"),
                )
            elif kind == "face":
                self._db.record_face(
                    ts=ts,
                    count=entry["count"],
                    smile_count=entry.get("smile_count", 0),
                    capture_path=entry.get("capture_path"),
                )
            elif kind == "thought":
                self._db.record_thought(
                    ts=ts, text=entry["text"], emotion=entry.get("emotion")
                )
        except Exception as e:
            logger.warning("DailyLog write failed for %s: %s", kind, e)