"""InterpreterMode — real-time concurrent translation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..mode import Mode, ModeContext

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def _drain_queue(q: asyncio.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


class InterpreterMode(Mode):
    """Real-time translation: concurrent requests, ordered output."""

    name = "interpreter"
    barge_in = False
    temperature = 0.3
    max_history = 0
    skip_emotion_extraction = True
    enable_vlm = False
    play_emotions = False

    def __init__(self) -> None:
        self._sequencer: _InterpreterSequencer | None = None

    async def enter(self, ctx: ModeContext) -> None:
        self._sequencer = _InterpreterSequencer(
            config=ctx.app.config,
            sentence_queue=ctx.sentence_queue,
            events=ctx.events,
        )
        await self._sequencer.start()

    async def exit(self, ctx: ModeContext) -> None:
        if self._sequencer:
            await self._sequencer.stop()
            self._sequencer = None
        _drain_queue(ctx.sentence_queue)
        _drain_queue(ctx.audio_queue)

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        if self._sequencer:
            self._sequencer.submit(text)
        return None

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "ignore"


class _InterpreterSequencer:
    """Concurrent translator with ordered output for interpreter mode.

    Bypasses OllamaClient — calls Ollama /api/chat directly via httpx.
    Translates utterances concurrently but emits to sentence_queue in
    utterance order (FIFO by submission time, not completion time).
    """

    def __init__(self, config: Any, sentence_queue: asyncio.Queue, events: Any) -> None:
        self._config = config
        self._sentence_queue = sentence_queue
        self._events = events
        self._http: httpx.AsyncClient | None = None
        self._slots: list[asyncio.Task] = []
        self._seq = 0
        self._emitter_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for interpreter mode — pip install httpx"
            )
        self._http = httpx.AsyncClient(
            base_url=self._config.ollama_base_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )
        self._running = True
        self._emitter_task = asyncio.create_task(self._emitter_loop())

    async def stop(self) -> None:
        self._running = False
        for task in self._slots:
            if not task.done():
                task.cancel()
        self._slots.clear()
        if self._emitter_task:
            self._emitter_task.cancel()
        if self._http:
            await self._http.aclose()
            self._http = None

    def submit(self, text: str) -> None:
        """Submit an utterance for translation. Returns immediately."""
        self._seq += 1
        seq = self._seq
        task = asyncio.create_task(self._translate(seq, text))
        self._slots.append(task)

    async def _translate(self, seq: int, text: str) -> tuple[int, str]:
        """Call Ollama /api/chat (non-streaming) and return (seq, translated)."""
        from ..llm import INTERPRETER_SYSTEM_PROMPT

        system_prompt = (
            self._config.interpreter_prompt
            or INTERPRETER_SYSTEM_PROMPT.format(
                source_lang=self._config.interpreter_source_lang,
                target_lang=self._config.interpreter_target_lang,
            )
        )
        payload = {
            "model": self._config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Translate: "{text}"'},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }
        try:
            resp = await self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            translated = resp.json()["message"]["content"].strip()
            self._events.emit(
                "llm_end",
                {"full_text": translated, "run_id": f"interp-{seq}"},
            )
            return (seq, translated)
        except Exception as e:
            logger.error(f"Interpreter translation failed (seq={seq}): {e}")
            return (seq, "")

    async def _emitter_loop(self) -> None:
        """Consume completed translations in order and feed to sentence_queue."""
        from ..plugins.conversation_plugin import SentenceItem

        while self._running:
            if not self._slots:
                await asyncio.sleep(0.05)
                continue

            first = self._slots[0]
            try:
                seq, translated = await asyncio.shield(first)
            except (asyncio.CancelledError, Exception):
                self._slots.pop(0)
                continue

            self._slots.pop(0)

            if translated:
                logger.info(f'Interpreter emit seq={seq}: "{translated[:40]}"')
                run_id = f"interp-{seq}"
                self._events.emit(
                    "llm_delta", {"text": translated, "run_id": run_id}
                )
                self._events.emit(
                    "llm_end",
                    {"full_text": translated, "run_id": run_id, "emotion": ""},
                )
                await self._sentence_queue.put(
                    SentenceItem(text=translated, is_last=True)
                )
