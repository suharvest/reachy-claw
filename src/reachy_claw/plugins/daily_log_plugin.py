"""DailyLogPlugin — silently logs daily interaction data to JSONL files.

Subscribes to EventBus events (emotion, ASR, LLM, vision) and writes
timestamped entries to per-day JSONL files under ~/.reachy-claw/daily-logs/.
These logs are consumed by the diary generation pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..plugin import Plugin

logger = logging.getLogger(__name__)

# Sampling intervals (seconds)
EMOTION_SAMPLE_INTERVAL = 60  # log emotion at most once per minute
FACE_SAMPLE_INTERVAL = 60  # log face count at most once per minute


def _data_base_dir() -> Path:
    """Return base directory for diary data.

    Uses DATA_DIR env var (set in Docker), falls back to ~/.reachy-claw/.
    """
    import os

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return Path(data_dir)
    return Path.home() / ".reachy-claw"


class DailyLogPlugin(Plugin):
    """Logs daily interaction data to JSONL files for diary generation."""

    name = "daily_log"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._log_dir = _data_base_dir() / "daily-logs"
        self._last_emotion: str | None = None
        self._last_emotion_ts: float = 0
        self._last_face_ts: float = 0
        # Buffer for pairing ASR final with LLM response
        self._pending_asr: dict | None = None

    def setup(self) -> bool:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        return True

    async def start(self) -> None:
        bus = self.app.events
        bus.subscribe("emotion", self._on_emotion)
        bus.subscribe("asr_final", self._on_asr_final)
        bus.subscribe("llm_end", self._on_llm_end)
        bus.subscribe("vision_faces", self._on_vision_faces)
        bus.subscribe("smile_capture", self._on_smile_capture)
        bus.subscribe("observation", self._on_observation)

        # Writer loop
        while self._running:
            try:
                log_type, entry = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
                self._write_entry(log_type, entry)
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

        # Drain remaining queue entries
        while not self._queue.empty():
            try:
                log_type, entry = self._queue.get_nowait()
                self._write_entry(log_type, entry)
            except Exception:
                break

    # ── Event handlers ──────────────────────────────────────────────────

    def _on_emotion(self, data: Any) -> None:
        """Log emotion changes, sampled at most once per minute."""
        now = time.time()
        emotion = data if isinstance(data, str) else str(data)

        # Only log on change or after interval
        if emotion == self._last_emotion and (now - self._last_emotion_ts) < EMOTION_SAMPLE_INTERVAL:
            return

        self._last_emotion = emotion
        self._last_emotion_ts = now
        self._enqueue("emotions", {"emotion": emotion})

    def _on_asr_final(self, data: Any) -> None:
        """Buffer ASR final text; will pair with next LLM response."""
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        if not text.strip():
            return
        self._pending_asr = {"ts": self._now_iso(), "user": text}

    def _on_llm_end(self, data: Any) -> None:
        """Log conversation turn (ASR + LLM response pair)."""
        if isinstance(data, dict):
            reply = data.get("full_text", data.get("text", ""))
            emotion = data.get("emotion", "")
        else:
            reply = str(data)
            emotion = ""

        if not reply.strip():
            return

        if self._pending_asr:
            entry = {
                **self._pending_asr,
                "reply": reply,
                "emotion": emotion,
            }
            self._pending_asr = None
        else:
            entry = {"reply": reply, "emotion": emotion}

        self._enqueue("conversations", entry)

    def _on_vision_faces(self, data: Any) -> None:
        """Log face detection count, sampled once per minute."""
        now = time.time()
        if (now - self._last_face_ts) < FACE_SAMPLE_INTERVAL:
            return
        self._last_face_ts = now

        if isinstance(data, dict):
            faces = data.get("faces", [])
            count = len(faces)
            names = [f.get("name", "") for f in faces if f.get("name")]
        elif isinstance(data, list):
            count = len(data)
            names = [f.get("name", "") for f in data if isinstance(f, dict) and f.get("name")]
        else:
            count = 0
            names = []

        self._enqueue("faces", {"count": count, "names": names})

    def _on_smile_capture(self, data: Any) -> None:
        """Log each smile capture event."""
        count = data.get("count", 0) if isinstance(data, dict) else data
        self._enqueue("faces", {"event": "smile_capture", "total": count})

    def _on_observation(self, data: Any) -> None:
        """Log robot observations/thoughts."""
        if isinstance(data, dict):
            text = data.get("text", data.get("observation", ""))
            emotion = data.get("emotion", "")
        else:
            text = str(data)
            emotion = ""

        if not text.strip():
            return

        self._enqueue("thoughts", {"text": text, "emotion": emotion})

    # ── Helpers ──────────────────────────────────────────────────────────

    def _enqueue(self, log_type: str, entry: dict) -> None:
        """Add a timestamped entry to the write queue."""
        if "ts" not in entry:
            entry["ts"] = self._now_iso()
        try:
            self._queue.put_nowait((log_type, entry))
        except asyncio.QueueFull:
            logger.warning("DailyLog queue full, dropping %s entry", log_type)

    def _write_entry(self, log_type: str, entry: dict) -> None:
        """Write a single JSONL entry to the appropriate file."""
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = self._log_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)

        filepath = day_dir / f"{log_type}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write %s log: %s", log_type, e)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")
