"""Edge LLM client for the TensorRT-Edge-LLM OpenAI-compatible chat service.

Implements `EdgeLLMClient` with the same `StreamCallbacks` contract as
`DesktopRobotClient` / `OllamaClient`, so `ConversationPlugin` can swap
clients without changing pipeline logic.

Streaming is via SSE (`/v1/chat/completions` with `stream=true`). Cache
behavior is per the edge-llm README:
  - first request:  save_system_prompt_kv_cache=True
  - subsequent:     prefix_cache=True (when enabled in config)

When `config.model == ""`, `connect()` queries `GET /v1/models` and uses
the first returned id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .gateway import StreamCallbacks
from .llm import DEFAULT_SYSTEM_PROMPT, _extract_emotion

logger = logging.getLogger(__name__)


@dataclass
class EdgeLLMConfig:
    """Configuration for the OpenAI-compatible TensorRT-Edge-LLM chat service."""

    base_url: str = "http://localhost:8080"
    model: str = ""  # empty = auto-discover via GET /v1/models
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_history: int = 3
    max_tokens: int = 80
    prefix_cache: bool = True
    skip_emotion_extraction: bool = False


class EdgeLLMError(RuntimeError):
    """Raised when the edge-llm service returns a structured error."""

    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(f"[{code}] {message}" + (f" (request_id={request_id})" if request_id else ""))
        self.code = code
        self.message = message
        self.request_id = request_id


class EdgeLLMClient:
    """Streaming LLM client with the same callback contract as DesktopRobotClient.

    History format mirrors OllamaClient: a flat list of OpenAI-style messages
    {"role": "user"|"assistant", "content": str}. The system prompt is NOT
    stored; it is prepended in `_build_messages()` each turn.

    History is appended only on a successful stream_end so aborted turns do
    not poison context (matches `llm.py:317-322`).
    """

    def __init__(self, config: EdgeLLMConfig):
        self._config = config
        self._http: httpx.AsyncClient | None = None
        self._history: list[dict[str, str]] = []
        self._connected = False
        self._current_task: asyncio.Task | None = None
        self._system_prompt_cache_saved: bool = False
        self._discovered_model: str | None = None

        self.callbacks = StreamCallbacks()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._http is not None

    @property
    def is_streaming(self) -> bool:
        """True while a chat-completion stream task is in flight."""
        t = self._current_task
        return t is not None and not t.done()

    @property
    def model(self) -> str:
        return self._discovered_model or self._config.model

    async def connect(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        # Auto-discover model id if not configured.
        if not self._config.model:
            await self._discover_model()
        self._connected = True
        logger.info(
            "EdgeLLMClient ready: model=%s base_url=%s",
            self.model, self._config.base_url,
        )

    async def _discover_model(self) -> None:
        assert self._http is not None
        try:
            resp = await self._http.get("/v1/models")
            req_id = resp.headers.get("X-Request-Id", "")
            logger.info("EdgeLLM /v1/models X-Request-Id=%s status=%d", req_id, resp.status_code)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data") or []
            if not items:
                raise RuntimeError("edge-llm /v1/models returned no models")
            self._discovered_model = items[0].get("id") or ""
            if not self._discovered_model:
                raise RuntimeError("edge-llm /v1/models first item has no id")
            logger.info("EdgeLLM discovered model: %s", self._discovered_model)
        except httpx.HTTPError as e:
            logger.error("Failed to discover edge-llm model: %s", e)
            raise

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
        logger.info("EdgeLLMClient disconnected")

    async def warmup_session(self) -> None:
        """Send a tiny non-streaming request to warm the KV cache.

        Uses save_system_prompt_kv_cache=True so the system prompt is
        compiled into the cache on first hit.
        """
        if not self._http:
            return
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                "stream": False,
                "max_tokens": 1,
                "temperature": self._config.temperature,
                "save_system_prompt_kv_cache": True,
            }
            resp = await self._http.post("/v1/chat/completions", json=payload)
            req_id = resp.headers.get("X-Request-Id", "")
            logger.info("EdgeLLM warmup X-Request-Id=%s status=%d", req_id, resp.status_code)
            if resp.status_code == 200:
                self._system_prompt_cache_saved = True
                logger.info("EdgeLLM warmup ok; system prompt KV cache saved")
            else:
                _raise_for_error(resp)
        except Exception as e:
            logger.warning("EdgeLLM warmup failed: %s", e)

    async def send_message_streaming(self, text: str) -> None:
        """Schedule a streaming chat completion task."""
        if not self._http:
            raise RuntimeError("Not connected")
        self._current_task = asyncio.create_task(self._stream_chat(text))

    async def send_interrupt(self) -> None:
        """Cancel the in-flight generation, if any."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("EdgeLLM generation interrupted")

    async def send_state_change(self, state: str) -> None:  # noqa: ARG002
        """No-op: no server-side session in edge-llm.

        Kept for interface compatibility with DesktopRobotClient/OllamaClient.
        """
        logger.debug("EdgeLLMClient.send_state_change(%s) — no-op", state)

    async def send_robot_result(self, command_id: str, result: dict) -> None:  # noqa: ARG002
        """No-op: edge-llm has no robot tool channel here."""
        logger.debug(
            "EdgeLLMClient.send_robot_result(%s) — no-op", command_id,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _build_messages(
        self, text: str, image_b64: str | None = None,
    ) -> list[dict]:
        """Build the OpenAI chat messages array for one turn.

        `image_b64` is reserved for a future VLM extension; if non-empty,
        a multimodal user content part will be emitted instead of a plain
        string. Wave 1 sticks to plain text.
        """
        messages: list[dict] = [
            {"role": "system", "content": self._config.system_prompt},
        ]
        if self._config.max_history > 0:
            messages.extend(self._history[-(self._config.max_history * 2):])

        if image_b64:
            # Reserved: future multimodal support.
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            })
        else:
            messages.append({"role": "user", "content": text})
        return messages

    async def _stream_chat(self, user_text: str) -> None:
        assert self._http is not None
        run_id = uuid.uuid4().hex

        if self.callbacks.on_stream_start:
            await _maybe_await(self.callbacks.on_stream_start(run_id))

        messages = self._build_messages(user_text)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        # Cache flag selection: first-ever turn saves system prompt KV cache;
        # subsequent turns use prefix_cache (if enabled in config).
        if not self._system_prompt_cache_saved:
            payload["save_system_prompt_kv_cache"] = True
        elif self._config.prefix_cache:
            payload["prefix_cache"] = True

        full_text = ""
        try:
            async with self._http.stream(
                "POST", "/v1/chat/completions", json=payload,
            ) as resp:
                req_id = resp.headers.get("X-Request-Id", "")
                logger.info(
                    "EdgeLLM /v1/chat/completions X-Request-Id=%s status=%d",
                    req_id, resp.status_code,
                )
                if resp.status_code != 200:
                    # Drain body and surface structured error.
                    body = await resp.aread()
                    _raise_from_body(body, req_id)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        # SSE comment / keep-alive
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("EdgeLLM: skipping malformed SSE chunk: %r", data_str)
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        full_text += delta
                        if self.callbacks.on_stream_delta:
                            clean_token = (
                                delta if self._config.skip_emotion_extraction
                                else _EMOTION_RE.sub("", delta)
                            )
                            if clean_token:
                                await _maybe_await(
                                    self.callbacks.on_stream_delta(clean_token, run_id)
                                )
                    if choice.get("finish_reason"):
                        break
        except asyncio.CancelledError:
            if self.callbacks.on_stream_abort:
                await _maybe_await(
                    self.callbacks.on_stream_abort("interrupted", run_id)
                )
            return
        except EdgeLLMError as e:
            logger.error("EdgeLLM structured error: %s", e)
            if self.callbacks.on_stream_abort:
                await _maybe_await(
                    self.callbacks.on_stream_abort(str(e), run_id)
                )
            return
        except Exception as e:
            logger.error("EdgeLLM streaming error: %s", e)
            if self.callbacks.on_stream_abort:
                await _maybe_await(
                    self.callbacks.on_stream_abort(str(e), run_id)
                )
            return

        # Mark system-prompt-cache as saved after the first successful
        # request (so the next turn switches to prefix_cache).
        self._system_prompt_cache_saved = True

        if self._config.skip_emotion_extraction:
            clean_full = full_text.strip()
        else:
            clean_full, emotion = _extract_emotion(full_text)
            clean_full = clean_full.strip()
            if emotion and self.callbacks.on_emotion:
                await _maybe_await(self.callbacks.on_emotion(emotion))

        # Append to history ONLY on successful stream completion.
        if self._config.max_history > 0:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": clean_full})
            max_msgs = self._config.max_history * 2
            if len(self._history) > max_msgs:
                self._history = self._history[-max_msgs:]

        if self.callbacks.on_stream_end:
            await _maybe_await(
                self.callbacks.on_stream_end(clean_full, run_id)
            )


_EMOTION_RE = re.compile(r"\[(\w+)\]")


def _raise_from_body(body: bytes, request_id: str) -> None:
    """Parse an edge-llm error body and raise EdgeLLMError."""
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EdgeLLMError("invalid_response", body.decode("utf-8", errors="replace")[:200], request_id)
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        code = err.get("code", "unknown")
        msg = err.get("message", "")
        ctx = err.get("context") or {}
        rid = ctx.get("request_id") or request_id or None
        raise EdgeLLMError(code, msg, rid)
    raise EdgeLLMError("unknown", str(data)[:200], request_id or None)


def _raise_for_error(resp: httpx.Response) -> None:
    req_id = resp.headers.get("X-Request-Id", "")
    body = resp.content or b""
    _raise_from_body(body, req_id)


async def _maybe_await(result: Any) -> None:
    if asyncio.iscoroutine(result):
        await result
