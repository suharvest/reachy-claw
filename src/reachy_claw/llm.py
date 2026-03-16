"""Lightweight Ollama LLM client with streaming and emotion parsing.

Supports optional tool calling (describe_scene) for VLM vision queries.
Uses httpx (already a project dependency) to call Ollama's /api/chat endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

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
You are Reachy, a cute robot at an exhibition. Always reply in English. 1 short sentence, max 15 words, no emoji.
You MUST end with one of: [happy] [sad] [thinking] [surprised] [curious]
Example: "Hey there, welcome to the exhibition! [happy]\""""

MONOLOGUE_SYSTEM_PROMPT = """\
You are a shy cute robot at an exhibition, mumbling to yourself about what you see and hear. Keep it short, casual, and playful.
STRICT: reply with ONE short sentence (max 15 words), then exactly ONE emotion tag. Nothing else.
Talk like a real person — no "sensors", no "circuits", no robot clichés. Just react to what's happening around you.
Use "you" or the person's name when talking about someone. Never say he/she.
Examples: "Ooh are you smiling at me?? [excited]" "Hmm nobody's here... boring [sad]" "Wait who's that?? [curious]" "harvest looks grumpy today haha [laugh]"
Tags: [happy] [sad] [thinking] [surprised] [curious] [excited] [neutral] [confused] [angry] [laugh]"""


_DESCRIBE_SCENE_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_scene",
        "description": "Use your camera to look at the scene. Only call when the user asks you to LOOK or SEE something, or asks what is in front of you.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_VLM_SYSTEM_PROMPT = """\
You are Reachy, a cute robot at an exhibition. Always reply in English. 1 short sentence, max 15 words, no emoji.
You have a camera (describe_scene). Only use it when asked to look or see. Don't guess what you see.
You MUST end with one of: [happy] [sad] [thinking] [surprised] [curious]
Example: "Wow, a person sitting with a laptop, nice setup! [curious]\""""


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
    # VLM (Vision Language Model)
    enable_vlm: bool = False
    vlm_model: str = ""
    vlm_prompt: str = "Describe what you see in this image briefly."


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
        self._run_counter = 0
        self.capture_frame: Callable[[], str | None] | None = None  # returns base64 JPEG

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
        """Call Ollama /api/chat with streaming and optional tool-call loop."""
        self._run_counter += 1
        run_id = f"ollama-{self._run_counter}"

        # Build messages — inject current time into system prompt
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        vlm_active = self._config.enable_vlm and self.capture_frame
        if vlm_active:
            # Dedicated VLM prompt: tool rule first, reduced tags for reliability
            system = f"Current time: {now}\n{_VLM_SYSTEM_PROMPT}"
        else:
            system = f"Current time: {now}\n{self._config.system_prompt}"
        messages = [{"role": "system", "content": system}]
        if self._config.max_history > 0:
            messages.extend(self._history[-(self._config.max_history * 2):])
        messages.append({"role": "user", "content": user_text})

        # Fire stream_start
        if self.callbacks.on_stream_start:
            await _maybe_await(self.callbacks.on_stream_start(run_id))

        try:
            # First LLM call
            full_text, tool_calls = await self._stream_response(messages, run_id)

            # Tool-call loop (single round)
            if tool_calls and self._config.enable_vlm:
                # End first stream so TTS plays the preamble (e.g. "让我看看")
                if full_text.strip() and self.callbacks.on_stream_end:
                    await _maybe_await(
                        self.callbacks.on_stream_end(
                            _EMOTION_RE.sub("", full_text).strip(), run_id
                        )
                    )

                # Execute tools and append results
                assistant_msg = {"role": "assistant", "content": full_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                import time as _time
                for tc in tool_calls:
                    t0 = _time.monotonic()
                    result = await self._execute_tool(tc)
                    elapsed = _time.monotonic() - t0
                    logger.info(
                        "Tool %s executed in %.1fs (%d chars)",
                        tc.get("function", {}).get("name", "?"),
                        elapsed,
                        len(result),
                    )
                    messages.append({
                        "role": "tool",
                        "content": result,
                    })

                # Second LLM call (no tools, to produce final response)
                self._run_counter += 1
                run_id = f"ollama-{self._run_counter}"
                if self.callbacks.on_stream_start:
                    await _maybe_await(self.callbacks.on_stream_start(run_id))
                full_text, _ = await self._stream_response(
                    messages, run_id, include_tools=False
                )

        except asyncio.CancelledError:
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
            max_msgs = self._config.max_history * 2
            if len(self._history) > max_msgs:
                self._history = self._history[-max_msgs:]

        # Fire stream_end
        if self.callbacks.on_stream_end:
            await _maybe_await(
                self.callbacks.on_stream_end(clean_full, run_id)
            )

    async def _stream_response(
        self,
        messages: list[dict],
        run_id: str,
        include_tools: bool = True,
    ) -> tuple[str, list[dict]]:
        """Stream a single Ollama /api/chat call, return (full_text, tool_calls)."""
        payload: dict = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": 100 if self._config.skip_emotion_extraction else 80,
            },
        }
        use_tools = include_tools and self._config.enable_vlm and self.capture_frame
        if use_tools:
            # Tool call responses need more tokens for the tool JSON + preamble
            payload["options"]["num_predict"] = 200
            payload["tools"] = [_DESCRIBE_SCENE_TOOL]

        full_text = ""
        tool_calls: list[dict] = []

        async with self._http.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})

                # Collect tool calls
                chunk_tool_calls = msg.get("tool_calls")
                if chunk_tool_calls:
                    tool_calls.extend(chunk_tool_calls)

                token = msg.get("content", "")
                if not token:
                    if chunk.get("done"):
                        break
                    continue

                full_text += token

                # Stream tokens immediately, stripping any emotion tags
                clean_token = (
                    token
                    if self._config.skip_emotion_extraction
                    else _EMOTION_RE.sub("", token)
                )
                if clean_token and self.callbacks.on_stream_delta:
                    await _maybe_await(
                        self.callbacks.on_stream_delta(clean_token, run_id)
                    )

        return full_text, tool_calls

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result string."""
        func = tool_call.get("function", {})
        name = func.get("name", "")
        if name != "describe_scene" or not self.capture_frame:
            return "Tool not available"
        b64 = await asyncio.to_thread(self.capture_frame)
        if not b64:
            return "Camera not available"
        vlm_model = self._config.vlm_model or self._config.model
        resp = await self._http.post(
            "/api/chat",
            json={
                "model": vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": self._config.vlm_prompt,
                        "images": [b64],
                    }
                ],
                "stream": False,
                "think": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _extract_emotion(text: str) -> tuple[str, str | None]:
    """Extract emotion from text, strip all bracket tags.

    Scans all [tag] occurrences, uses the last known emotion,
    and removes every bracket tag from the text.
    """
    emotion = None
    for m in _EMOTION_RE.finditer(text):
        tag = m.group(1).lower()
        if tag in _KNOWN_EMOTIONS:
            emotion = tag
    cleaned = _EMOTION_RE.sub("", text).strip()
    return cleaned, emotion


async def _maybe_await(result) -> None:
    if asyncio.iscoroutine(result):
        await result
