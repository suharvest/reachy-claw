"""Desktop-robot WebSocket client for OpenClaw."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import websockets

from clawd_reachy_mini.config import Config

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A streaming event from the server."""

    type: str  # stream_start, stream_delta, stream_end, stream_abort, tool_start, ...
    run_id: str = ""
    text: str = ""
    full_text: str = ""
    reason: str = ""
    tool_name: str = ""
    task_label: str = ""
    task_run_id: str = ""
    summary: str = ""
    state: str = ""


@dataclass
class StreamCallbacks:
    """Callbacks for streaming events."""

    on_stream_start: Callable[[str], Any] | None = None  # run_id
    on_stream_delta: Callable[[str, str], Any] | None = None  # text, run_id
    on_stream_end: Callable[[str, str], Any] | None = None  # full_text, run_id
    on_stream_abort: Callable[[str, str], Any] | None = None  # reason, run_id
    on_tool_start: Callable[[str, str], Any] | None = None  # tool_name, run_id
    on_tool_end: Callable[[str, str], Any] | None = None  # tool_name, run_id
    on_task_spawned: Callable[[str, str], Any] | None = None  # label, task_run_id
    on_task_completed: Callable[[str, str], Any] | None = None  # summary, task_run_id
    on_state_change: Callable[[str], Any] | None = None  # state
    on_error: Callable[[str], Any] | None = None  # message


class DesktopRobotClient:
    """Client for the desktop-robot WebSocket channel."""

    def __init__(self, config: Config):
        self.config = config
        self._ws = None  # websockets client connection
        self._session_id: str = str(uuid.uuid4())
        self._connected = False
        self._welcome_event: asyncio.Event = asyncio.Event()
        self._listener_task: asyncio.Task | None = None

        # Per-run completion tracking
        self._run_futures: dict[str, asyncio.Future[str]] = {}
        self._run_buffers: dict[str, str] = {}

        # Global stream callbacks (set by the interface)
        self.callbacks = StreamCallbacks()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def connect(self) -> None:
        """Connect and perform the hello handshake."""
        if self.is_connected:
            return

        url = self.config.desktop_robot_url
        token = self.config.gateway_token

        # Append token as query param if set
        if token:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={token}"

        try:
            self._ws = await websockets.connect(url)
            self._connected = True
            self._welcome_event.clear()
            self._listener_task = asyncio.create_task(self._listen())

            # Send hello
            await self._send({"type": "hello", "sessionId": self._session_id})

            # Wait for welcome
            try:
                await asyncio.wait_for(self._welcome_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("No welcome received, proceeding anyway")

            logger.info(f"Connected to desktop-robot at {self.config.desktop_robot_url}")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        self._fail_pending_runs("desktop-robot disconnected")
        self._run_buffers.clear()

        logger.info("Disconnected from desktop-robot")

    async def send_message(self, text: str) -> str:
        """Send a user message and wait for the complete response.

        Returns the full response text after stream_end.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")

        # We'll get a stream_start event with a run_id — but we don't know it yet.
        # Use a single-slot future that the listener fills on stream_end.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._run_futures["_next"] = future
        await self._send({"type": "message", "text": text})

        try:
            return await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response")
            raise
        finally:
            self._run_futures.pop("_next", None)

    async def send_message_streaming(self, text: str) -> None:
        """Send a user message; responses arrive via callbacks.

        Use self.callbacks to handle stream_start / stream_delta / stream_end.
        Does NOT block for the full response.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected")
        await self._send({"type": "message", "text": text})

    async def send_interrupt(self) -> None:
        """Send barge-in interrupt."""
        if not self.is_connected:
            return
        await self._send({"type": "interrupt"})
        logger.info("Sent interrupt (barge-in)")

    async def send_state_change(self, state: str) -> None:
        """Notify server of local state change."""
        if not self.is_connected:
            return
        await self._send({"type": "state_change", "state": state})

    async def send_ping(self) -> None:
        if not self.is_connected:
            return
        import time
        await self._send({"type": "ping", "ts": int(time.time() * 1000)})

    # ── Internal ──────────────────────────────────────────────────────

    async def _send(self, data: dict) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(data))

    def _fail_pending_runs(self, reason: str) -> None:
        for future in self._run_futures.values():
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._run_futures.clear()

    async def _listen(self) -> None:
        if not self._ws:
            return
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._handle(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {raw}")
        except websockets.ConnectionClosed:
            logger.info("Connection closed")
            self._connected = False
        except Exception as e:
            logger.error(f"Listener error: {e}")
            self._connected = False
        finally:
            self._fail_pending_runs("desktop-robot listener stopped")
            self._run_buffers.clear()

    async def _handle(self, msg: dict) -> None:
        t = msg.get("type", "")
        run_id = msg.get("runId", "")

        if t == "welcome":
            self._session_id = msg.get("sessionId", self._session_id)
            self._welcome_event.set()
            logger.debug(f"Welcome: session={self._session_id}")

        elif t == "stream_start":
            # Associate this run_id with the pending _next future
            if "_next" in self._run_futures:
                f = self._run_futures.pop("_next")
                self._run_futures[run_id] = f
            self._run_buffers[run_id] = ""
            if self.callbacks.on_stream_start:
                await _maybe_await(self.callbacks.on_stream_start(run_id))

        elif t == "stream_delta":
            text = msg.get("text", "")
            if run_id in self._run_buffers:
                self._run_buffers[run_id] += text
            if self.callbacks.on_stream_delta:
                await _maybe_await(self.callbacks.on_stream_delta(text, run_id))

        elif t == "stream_end":
            full_text = msg.get("fullText", "")
            if not full_text and run_id in self._run_buffers:
                full_text = self._run_buffers[run_id]
            if self.callbacks.on_stream_end:
                await _maybe_await(self.callbacks.on_stream_end(full_text, run_id))
            # Resolve the future
            if run_id in self._run_futures:
                fut = self._run_futures.pop(run_id)
                if not fut.done():
                    fut.set_result(full_text)
            self._run_buffers.pop(run_id, None)

        elif t == "stream_abort":
            reason = msg.get("reason", "unknown")
            if self.callbacks.on_stream_abort:
                await _maybe_await(self.callbacks.on_stream_abort(reason, run_id))
            # Resolve the future — check both run_id and _next (abort may arrive
            # before stream_start, so _next was never reassigned).
            fut_key = run_id if run_id in self._run_futures else ("_next" if "_next" in self._run_futures else None)
            if fut_key:
                fut = self._run_futures.pop(fut_key)
                if not fut.done():
                    fut.set_result(self._run_buffers.get(run_id, ""))
            self._run_buffers.pop(run_id, None)

        elif t == "state":
            state = msg.get("state", "")
            if self.callbacks.on_state_change:
                await _maybe_await(self.callbacks.on_state_change(state))

        elif t == "tool_start":
            tool_name = msg.get("toolName", "")
            if self.callbacks.on_tool_start:
                await _maybe_await(self.callbacks.on_tool_start(tool_name, run_id))

        elif t == "tool_end":
            tool_name = msg.get("toolName", "")
            if self.callbacks.on_tool_end:
                await _maybe_await(self.callbacks.on_tool_end(tool_name, run_id))

        elif t == "task_spawned":
            label = msg.get("taskLabel", "")
            task_run_id = msg.get("taskRunId", "")
            if self.callbacks.on_task_spawned:
                await _maybe_await(self.callbacks.on_task_spawned(label, task_run_id))

        elif t == "task_completed":
            summary = msg.get("summary", "")
            task_run_id = msg.get("taskRunId", "")
            if self.callbacks.on_task_completed:
                await _maybe_await(self.callbacks.on_task_completed(summary, task_run_id))

        elif t == "error":
            message = msg.get("message", "unknown error")
            logger.error(f"Server error: {message}")
            if self.callbacks.on_error:
                await _maybe_await(self.callbacks.on_error(message))

        elif t == "pong":
            logger.debug(f"Pong: ts={msg.get('ts')}")

        else:
            logger.debug(f"Unknown message type: {t}")


async def _maybe_await(result: Any) -> None:
    """Await if the result is a coroutine."""
    if asyncio.iscoroutine(result):
        await result
