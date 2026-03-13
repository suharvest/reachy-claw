"""Lightweight Ollama LLM client with streaming and emotion parsing.

Designed for tiny models (0.8b-2b) — no tool calling, just text + emotion tags.
Uses httpx (already a project dependency) to call Ollama's /api/chat endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from .gateway import StreamCallbacks

logger = logging.getLogger(__name__)

# Emotion tag pattern: [happy], [sad], etc. at the start of text or inline
_EMOTION_RE = re.compile(r"\[(\w+)\]")

# Supported emotions (subset that EmotionMapper knows about)
_KNOWN_EMOTIONS = frozenset({
    "happy", "laugh", "excited", "thinking", "confused", "curious",
    "sad", "angry", "surprised", "fear", "neutral", "listening",
    "agreeing", "disagreeing",
})

DEFAULT_SYSTEM_PROMPT = """\
You are Reachy, a playful robot at an exhibition. Be witty, curious, and never boring.
1-3 sentences. Vary your style. End with one tag: [happy] [sad] [thinking] [surprised] [curious] [excited] [neutral] [confused] [angry] [laugh]"""

MONOLOGUE_SYSTEM_PROMPT = """\
Shy robot mumbling to yourself. Guess, react, overthink. Reply with max 15 words then exactly ONE tag. No extra tags or emoji.
Examples: "Are you happy? [thinking]" "Oh wait she is angry???? [surprised]" "Did he smile at ME?! [excited]" "Why is everyone ignoring me... [sad]"
Tags: [happy] [sad] [thinking] [surprised] [curious] [excited] [neutral] [confused] [angry] [laugh]"""


@dataclass
class OllamaConfig:
    """Ollama-specific configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:0.8b"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    # Simple sliding window: keep last N messages (0 = no history)
    max_history: int = 0
    skip_emotion_extraction: bool = False  # monologue mode: no emotion tags


class OllamaClient:
    """Drop-in replacement for DesktopRobotClient, using local Ollama.

    Implements the same callback interface so ConversationPlugin can use it
    without changes to its pipeline logic.
    """

    def __init__(self, ollama_config: OllamaConfig):
        self._config = ollama_config
        self._http: httpx.AsyncClient | None = None
        self._history: list[dict[str, str]] = []
        self._connected = False
        self._current_task: asyncio.Task | None = None

        self.callbacks = StreamCallbacks()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        self._connected = True
        logger.info(
            f"OllamaClient ready: {self._config.model} @ {self._config.base_url}"
        )

    async def warmup_session(self) -> None:
        """Send a tiny request to preload the model into memory."""
        if not self._http:
            return
        try:
            resp = await self._http.post(
                "/api/chat",
                json={
                    "model": self._config.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 1},
                },
            )
            resp.raise_for_status()
            logger.info("Ollama model warmed up")
        except Exception as e:
            logger.warning(f"Ollama warmup failed: {e}")

    async def disconnect(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        if self._http:
            await self._http.aclose()
            self._http = None
        self._connected = False
        logger.info("OllamaClient disconnected")

    async def send_message_streaming(self, text: str) -> None:
        """Send user message, stream response via callbacks."""
        if not self._http:
            raise RuntimeError("Not connected")
        self._current_task = asyncio.create_task(self._stream_chat(text))

    async def send_interrupt(self) -> None:
        """Cancel the current generation."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Ollama generation interrupted")

    async def send_state_change(self, state: str) -> None:
        pass  # No server to notify

    async def send_robot_result(self, command_id: str, result: dict) -> None:
        pass  # No tool calling in this mode

    # ── Internal ──────────────────────────────────────────────────────

    async def _stream_chat(self, user_text: str) -> None:
        """Call Ollama /api/chat with streaming and fire callbacks."""
        run_id = f"ollama-{id(self)}"

        # Build messages
        messages = [{"role": "system", "content": self._config.system_prompt}]
        if self._config.max_history > 0:
            messages.extend(self._history[-(self._config.max_history * 2):])
        messages.append({"role": "user", "content": user_text})

        # Fire stream_start
        if self.callbacks.on_stream_start:
            await _maybe_await(self.callbacks.on_stream_start(run_id))

        full_text = ""

        try:
            async with self._http.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self._config.model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": {
                        "temperature": self._config.temperature,
                        "num_predict": 100 if self._config.skip_emotion_extraction else 200,
                    },
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if not token:
                        if chunk.get("done"):
                            break
                        continue

                    full_text += token

                    # Stream tokens immediately, stripping any emotion tags
                    clean_token = token if self._config.skip_emotion_extraction else _EMOTION_RE.sub("", token)
                    if clean_token and self.callbacks.on_stream_delta:
                        await _maybe_await(
                            self.callbacks.on_stream_delta(clean_token, run_id)
                        )

        except asyncio.CancelledError:
            # Barge-in interrupt
            if self.callbacks.on_stream_abort:
                await _maybe_await(
                    self.callbacks.on_stream_abort("interrupted", run_id)
                )
            return
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            if self.callbacks.on_stream_abort:
                await _maybe_await(
                    self.callbacks.on_stream_abort(str(e), run_id)
                )
            return

        # Extract emotion from the complete response (tag is at the end)
        if self._config.skip_emotion_extraction:
            clean_full = full_text.strip()
        else:
            clean_full, emotion = _extract_emotion(full_text)
            clean_full = clean_full.strip()
            if emotion and self.callbacks.on_emotion:
                await _maybe_await(self.callbacks.on_emotion(emotion))

        # Update history
        if self._config.max_history > 0:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": clean_full})
            # Trim to max_history turns
            max_msgs = self._config.max_history * 2
            if len(self._history) > max_msgs:
                self._history = self._history[-max_msgs:]

        # Fire stream_end
        if self.callbacks.on_stream_end:
            await _maybe_await(
                self.callbacks.on_stream_end(clean_full, run_id)
            )


def _extract_emotion(text: str) -> tuple[str, str | None]:
    """Extract the first emotion tag from text.

    Returns (cleaned_text, emotion_name) or (text, None) if no tag found.
    """
    m = _EMOTION_RE.search(text)
    if m and m.group(1).lower() in _KNOWN_EMOTIONS:
        emotion = m.group(1).lower()
        cleaned = text[:m.start()] + text[m.end():]
        return cleaned.strip(), emotion
    return text, None


async def _maybe_await(result) -> None:
    if asyncio.iscoroutine(result):
        await result
