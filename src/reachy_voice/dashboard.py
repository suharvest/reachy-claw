"""Dashboard bridge — pipes live app events to the browser over WebSocket.

Two halves:

* ``DashboardHub`` — thread-safe pub/sub. The conversation engine (its own
  asyncio loop), the vision ZMQ thread, and the motion layer all ``publish()``
  plain dicts; each connected WebSocket (running on the settings server's
  uvicorn loop) gets them via per-connection asyncio queues.

* ``DashboardPlugin`` — an ovs_agent ``Plugin`` registered on the companion
  app; forwards the broadcast hooks (ASR partial/final, LLM tokens, state
  changes, TTS done) into the hub as typed messages the UI understands:
    {"type": "asr_partial"|"asr_final"|"llm_delta"|"llm_end"|"state"|"emotion"|"vision_faces", ...}
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any

from ovs_agent.plugin import Plugin

logger = logging.getLogger("reachy_voice.dashboard")

_TAG_RE = re.compile(r"\[([a-zA-Z_]+)\]")
# Vision-context tag (``[Faces: Alice]``) sometimes echoed by edge LLMs — drop
# it from the transcript too (not matched by _TAG_RE: it has a space/colon).
_FACES_RE = re.compile(r"\[Faces:[^\]]*\]", re.IGNORECASE)


class DashboardHub:
    def __init__(self) -> None:
        self._subs: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._lock = __import__("threading").Lock()

    def subscribe(self) -> asyncio.Queue:
        """Call from the WS handler's loop. Returns this connection's queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subs.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(qq, ll) for qq, ll in self._subs if qq is not q]

    def publish(self, msg: dict[str, Any]) -> None:
        """Safe from ANY thread/loop. Drops messages for slow consumers."""
        with self._lock:
            subs = list(self._subs)
        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(self._put, q, msg)
            except RuntimeError:
                pass  # loop closed; unsubscribe happens on WS teardown

    @staticmethod
    def _put(q: asyncio.Queue, msg: dict) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


class DashboardPlugin(Plugin):
    """Forwards ovs_agent broadcast hooks to the dashboard hub as the message
    types the (verbatim-copied) original UI consumes:

      asr_partial {text,is_stable} · asr_final {text} · state {state}
      llm_delta {text,run_id}      · llm_end {full_text,run_id} · emotion

    ``llm_delta``/``llm_end`` carry a per-turn ``run_id`` (the UI groups a
    streaming "thought card" by it) and **tag-free** text: the raw LLM token
    stream contains ``[emotion]`` tags, which are stripped here (streaming-safe,
    matching ``_TtsTagFilter``) so the transcript never shows them.
    """

    name = "dashboard_bridge"

    def __init__(self, app, hub: DashboardHub, on_state=None) -> None:  # noqa: ANN001
        super().__init__(app)
        self.hub = hub
        self._on_state = on_state  # extra sink for conv state (e.g. motion gating)
        self._run_id: str | None = None
        self._buf = ""           # holds back text inside an unclosed '[' tag
        self._full: list[str] = []  # accumulated clean text for llm_end

    # ── conversation feed ────────────────────────────────────────────
    async def on_user_partial(self, text: str) -> None:
        self.hub.publish({"type": "asr_partial", "text": text, "is_stable": False})

    async def on_user_utterance(self, text: str) -> None:
        self.hub.publish({"type": "asr_final", "text": text})

    async def on_assistant_token(self, token: str) -> None:
        # Lazily open a run on the first token of a turn (the UI starts a fresh
        # thought card whenever run_id changes).
        if self._run_id is None:
            self._run_id = uuid.uuid4().hex
            self._buf = ""
            self._full = []
        clean = self._strip(token)
        if clean:
            self._full.append(clean)
            self.hub.publish(
                {"type": "llm_delta", "text": clean, "run_id": self._run_id}
            )

    async def on_assistant_done(self) -> None:
        run_id = self._run_id
        if run_id is None:
            return
        # Flush any buffered '[...' tail (an unterminated tag is real text).
        tail, self._buf = self._buf, ""
        if tail:
            self._full.append(tail)
        self.hub.publish(
            {"type": "llm_end", "full_text": "".join(self._full), "run_id": run_id}
        )
        self._run_id = None
        self._full = []

    async def on_state_change(self, data: dict) -> None:
        state = data.get("state", "?")
        self.hub.publish({"type": "state", "state": state})
        if self._on_state is not None:
            try:
                self._on_state(state)  # e.g. motion: freeze while listening
            except Exception:  # noqa: BLE001 — never let a sink break the feed
                logger.debug("on_state sink failed", exc_info=True)

    def _strip(self, token: str) -> str:
        """Streaming-safe removal of ``[emotion]`` tags from the token stream.
        Text after an unclosed ``[`` is held back until the tag closes."""
        self._buf += token
        open_idx = self._buf.rfind("[")
        if open_idx != -1 and "]" not in self._buf[open_idx:]:
            head, self._buf = self._buf[:open_idx], self._buf[open_idx:]
        else:
            head, self._buf = self._buf, ""
        return _TAG_RE.sub("", _FACES_RE.sub("", head))
