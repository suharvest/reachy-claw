"""Desktop-robot WebSocket client for OpenClaw."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import websockets

from reachy_claw.config import Config

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
    on_emotion: Callable[[str], Any] | None = None  # emotion name
    on_robot_command: Callable[[str, dict, str], Any] | None = None  # action, params, command_id
    on_error: Callable[[str], Any] | None = None  # message


class DesktopRobotClient:
    """Client for the desktop-robot WebSocket channel."""

    def __init__(self, config: Config):
        self.config = config
        self._ws = None  # websockets client connection
        self._session_id: str = _load_or_create_session_id(config)
        self._connected = False
        self._welcome_event: asyncio.Event = asyncio.Event()
        self._listener_task: asyncio.Task | None = None

        # Per-run completion tracking
        self._run_futures: dict[str, asyncio.Future[str]] = {}
        self._run_buffers: dict[str, str] = {}
        self._keepalive_task: asyncio.Task | None = None
        self._session_warmed: bool = False

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

            # Start keepalive pings
            interval = getattr(self.config, "gateway_keepalive_s", 60)
            if interval > 0:
                self._keepalive_task = asyncio.create_task(
                    self._keepalive_loop(interval)
                )

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def warmup_session(self) -> None:
        """Send a trivial message to pre-initialize the OpenClaw session.

        The first message in a new session incurs ~2-3s cold start (tool
        registration, prompt building, DashScope connection init). By sending
        a throwaway message during startup, we shift that cost to before the
        user starts speaking.
        """
        if not self.is_connected or self._session_warmed:
            return

        logger.info("Warming up gateway session...")
        t0 = time.perf_counter()
        try:
            # Send a minimal message — fast to process, minimal token usage
            response = await asyncio.wait_for(
                self.send_message("."), timeout=15.0
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self._session_warmed = True
            logger.info(f"Session warmup complete ({elapsed:.0f}ms)")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.warning(f"Session warmup failed ({elapsed:.0f}ms): {e}")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

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

    async def send_robot_result(
        self, command_id: str, result: dict
    ) -> None:
        """Send robot command execution result back to server."""
        if not self.is_connected:
            return
        await self._send({
            "type": "robot_result",
            "commandId": command_id,
            "result": result,
        })

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

    async def _notify_disconnect(self) -> None:
        """Fire a synthetic stream_abort so the plugin resets state on disconnect."""
        if self.callbacks.on_stream_abort:
            try:
                await _maybe_await(
                    self.callbacks.on_stream_abort("websocket_disconnected", "")
                )
            except Exception:
                pass

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
            logger.info("Connection closed, will attempt reconnect")
            self._connected = False
        except Exception as e:
            logger.error(f"Listener error: {e}")
            self._connected = False
        finally:
            self._fail_pending_runs("desktop-robot listener stopped")
            self._run_buffers.clear()
            # Notify plugin so it can reset from THINKING/SPEAKING → IDLE
            await self._notify_disconnect()

        # Auto-reconnect loop
        backoff = 1.0
        while not self._connected:
            try:
                logger.info(f"Reconnecting in {backoff:.0f}s...")
                await asyncio.sleep(backoff)
                await self.connect()
                logger.info("Reconnected to gateway")
                return  # connect() starts a new _listen task
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")
                backoff = min(backoff * 2, 30.0)

    async def _keepalive_loop(self, interval: int) -> None:
        """Send periodic pings to prevent gateway session timeout."""
        try:
            while self._connected:
                await asyncio.sleep(interval)
                if self._connected and self._ws:
                    try:
                        await self._send(
                            {"type": "ping", "ts": int(time.time() * 1000)}
                        )
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

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

        elif t == "emotion":
            emotion = msg.get("emotion", "")
            if emotion and self.callbacks.on_emotion:
                await _maybe_await(self.callbacks.on_emotion(emotion))

        elif t == "robot_command":
            action = msg.get("action", "")
            params = msg.get("params", {})
            command_id = msg.get("commandId", "")
            if action and self.callbacks.on_robot_command:
                await _maybe_await(
                    self.callbacks.on_robot_command(action, params, command_id)
                )

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


def _load_or_create_session_id(config: Config) -> str:
    """Load a persistent session ID from disk, or create one.

    This ensures the same session ID is reused across restarts, so the
    OpenClaw agent retains conversation history.
    """
    session_file = config.cache_dir / "session_id"
    try:
        sid = session_file.read_text().strip()
        if sid:
            logger.info(f"Resuming session: {sid}")
            return sid
    except FileNotFoundError:
        pass

    sid = str(uuid.uuid4())
    try:
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text(sid)
    except OSError as e:
        logger.warning(f"Could not persist session ID: {e}")
    logger.info(f"New session: {sid}")
    return sid
