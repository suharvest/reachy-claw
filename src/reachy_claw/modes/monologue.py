"""MonologueMode — autonomous self-talk with background listening."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from ..mode import Mode, ModeContext

logger = logging.getLogger(__name__)


class MonologueMode(Mode):
    """Robot autonomously generates speech based on vision and background audio."""

    name = "monologue"
    barge_in = False
    temperature = 0.9
    skip_emotion_extraction = False
    play_emotions = False

    def __init__(self) -> None:
        self._timer_task: asyncio.Task | None = None
        self.last_speech_time: float = 0.0
        self.pending_speech: str | None = None
        self._bg_speech_frames: list[np.ndarray] = []
        self._bg_silence_count: int = 0

    async def enter(self, ctx: ModeContext) -> None:
        self.last_speech_time = time.monotonic()
        self.pending_speech = None
        self._bg_speech_frames = []
        self._bg_silence_count = 0
        self._timer_task = ctx.spawn_task(
            self._timer_loop(ctx), name="monologue_timer"
        )

    async def exit(self, ctx: ModeContext) -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self.pending_speech = None
        self._bg_speech_frames.clear()

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        return self.compose_monologue_prompt(text, ctx)

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "bg_listen"

    def compose_monologue_prompt(self, transcript: str | None, ctx: ModeContext) -> str:
        """Build natural-language LLM input from speech + vision."""
        parts = []
        if transcript:
            parts.append(f'heard: "{transcript}"')

        vision = ctx.app.get_plugin("vision_client")
        if vision and getattr(vision, "_last_faces_summary", None):
            faces = vision._last_faces_summary
            named: dict[str, str] = {}
            stranger_count = 0
            for f in faces:
                name = f.get("identity")
                emo = f.get("emotion", "neutral")
                if name:
                    named[name] = emo
                else:
                    stranger_count += 1
            descs = [f"{n} looks {e}" for n, e in named.items()]
            real_strangers = max(0, stranger_count - len(named))
            if real_strangers == 1:
                descs.append("someone you don't know")
            elif real_strangers > 1:
                descs.append(f"{real_strangers} people you don't know")
            if descs:
                parts.append(f"you see: {', '.join(descs)}")
        elif vision:
            emo = getattr(vision, "_last_emotion", None)
            identity = getattr(vision, "current_identity", None)
            if identity:
                parts.append(f"you see: {identity} looks {emo or 'neutral'}")
            elif emo and emo != "neutral":
                parts.append(f"you see: someone looks {emo}")

        if not parts:
            parts.append("nobody around")
        return ". ".join(parts)

    async def bg_listen(
        self, chunk: np.ndarray, streaming_stt: bool, plugin: Any
    ) -> None:
        """Background listening during SPEAKING/THINKING in monologue mode."""
        max_silence_bg = int(
            plugin.app.config.silence_duration
            * plugin.app.config.sample_rate
            / 1024
        )
        has_speech = await asyncio.to_thread(plugin._audio._detect_speech, chunk)

        if has_speech:
            if not self._bg_speech_frames:
                logger.debug("BG listen: speech detected while speaking")
                if streaming_stt:
                    plugin._stt.cancel_stream()
                    await asyncio.to_thread(
                        plugin._stt.start_stream, plugin.app.config.sample_rate
                    )
            self._bg_speech_frames.append(chunk)
            self._bg_silence_count = 0

            if streaming_stt:
                partial = await asyncio.to_thread(plugin._stt.feed_chunk, chunk)
                if partial and partial.text:
                    plugin.app.events.emit("asr_partial", {"text": partial.text})
                    if partial.is_final:
                        text = await asyncio.to_thread(plugin._stt.finish_stream)
                        if text and text.strip():
                            logger.info(f'BG heard: "{text}"')
                            plugin.app.events.emit("asr_final", {"text": text})
                            self.pending_speech = text
                        self._bg_speech_frames = []
                        self._bg_silence_count = 0
                        if plugin._vad:
                            plugin._vad.reset()

        elif self._bg_speech_frames:
            self._bg_silence_count += 1
            self._bg_speech_frames.append(chunk)
            if streaming_stt:
                await asyncio.to_thread(plugin._stt.feed_chunk, chunk)

            if self._bg_silence_count >= max_silence_bg:
                if streaming_stt:
                    text = await asyncio.to_thread(plugin._stt.finish_stream)
                    if text and text.strip():
                        logger.info(f'BG heard: "{text}"')
                        plugin.app.events.emit("asr_final", {"text": text})
                        self.pending_speech = text
                self._bg_speech_frames = []
                self._bg_silence_count = 0
                if plugin._vad:
                    plugin._vad.reset()

    async def _timer_loop(self, ctx: ModeContext) -> None:
        """Periodically trigger monologue when idle."""
        interval = ctx.app.config.monologue_interval
        logger.info(f"Monologue timer started (interval={interval}s)")

        while True:
            await asyncio.sleep(1.0)
            interval = ctx.app.config.monologue_interval

            if time.monotonic() - self.last_speech_time < interval:
                continue

            transcript = self.pending_speech
            self.pending_speech = None
            prompt = self.compose_monologue_prompt(transcript, ctx)
            ctx.events.emit("monologue_trigger", {"prompt": prompt})
