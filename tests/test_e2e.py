"""End-to-end tests: gateway + robot simulation.

Requires:
  1. reachy-mini-daemon --mockup-sim --headless --localhost-only --deactivate-audio
  2. OpenClaw desktop-robot extension running on ws://127.0.0.1:18790/desktop-robot

Skip automatically if either is not reachable.
Run with: uv run pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest
import websockets

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config
from reachy_claw.gateway import DesktopRobotClient, StreamCallbacks


# ── Fixtures ─────────────────────────────────────────────────────────────


def _gateway_reachable() -> bool:
    """Check if the desktop-robot gateway is reachable."""
    import asyncio

    async def _check():
        try:
            ws = await asyncio.wait_for(
                websockets.connect("ws://127.0.0.1:18790/desktop-robot"),
                timeout=3.0,
            )
            await ws.send(json.dumps({"type": "hello", "sessionId": "e2e-probe"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            await ws.close()
            return json.loads(msg).get("type") == "welcome"
        except Exception:
            return False

    return asyncio.run(_check())


def _sim_reachable():
    """Try to connect to the mockup sim daemon."""
    try:
        from reachy_mini import ReachyMini

        reachy = ReachyMini(
            connection_mode="localhost_only",
            media_backend="no_media",
            timeout=3,
        )
        reachy.__enter__()
        return reachy
    except Exception:
        return None


@pytest.fixture(scope="module")
def e2e_env():
    """Module-scoped fixture: both gateway and sim must be available."""
    if not _gateway_reachable():
        pytest.skip("OpenClaw gateway not reachable at ws://127.0.0.1:18790/desktop-robot")

    reachy = _sim_reachable()
    if reachy is None:
        pytest.skip("Mockup sim daemon not reachable")

    reachy.wake_up()
    time.sleep(0.5)
    yield reachy
    reachy.__exit__(None, None, None)


# ── Gateway protocol tests ──────────────────────────────────────────────


class TestGatewayConnection:
    @pytest.mark.asyncio
    async def test_connect_handshake(self, e2e_env):
        config = Config()
        client = DesktopRobotClient(config)
        await client.connect()

        assert client.is_connected
        assert client.session_id

        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_send_message_gets_response(self, e2e_env):
        config = Config()
        client = DesktopRobotClient(config)
        await client.connect()

        response = await client.send_message("Reply with exactly: OK")
        assert len(response) > 0

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_streaming_callbacks_fire(self, e2e_env):
        config = Config()
        client = DesktopRobotClient(config)
        events = []

        client.callbacks.on_stream_start = lambda rid: events.append(("start", rid))
        client.callbacks.on_stream_delta = lambda text, rid: events.append(("delta", text))
        client.callbacks.on_stream_end = lambda text, rid: events.append(("end", text))

        await client.connect()
        await client.send_message_streaming("Say hello in one word.")

        # Wait for stream to complete
        for _ in range(60):
            await asyncio.sleep(0.5)
            if any(e[0] == "end" for e in events):
                break

        await client.disconnect()

        types = [e[0] for e in events]
        assert "start" in types
        assert "delta" in types
        assert "end" in types

    @pytest.mark.asyncio
    async def test_ping_pong(self, e2e_env):
        config = Config()
        client = DesktopRobotClient(config)
        await client.connect()

        # Should not raise
        await client.send_ping()
        await asyncio.sleep(0.5)

        await client.disconnect()


# ── Gateway + Robot motion ──────────────────────────────────────────────


class TestGatewayWithRobot:
    @pytest.mark.asyncio
    async def test_emotion_queued_during_response(self, e2e_env):
        """When sending to AI, 'thinking' emotion should be queued."""
        sim_reachy = e2e_env

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            enable_motion=True,
            play_emotions=True,
            tts_backend="none",
            stt_backend="whisper",
        )
        app = ReachyClawApp(config)
        app.reachy = sim_reachy

        # Queue thinking emotion (as conversation_plugin does)
        app.emotions.queue_emotion("thinking")
        expr = app.emotions.get_next_expression()
        assert expr is not None
        assert "thinking" in expr.description.lower()

    @pytest.mark.asyncio
    async def test_motion_plugin_with_gateway_stream(self, e2e_env):
        """MotionPlugin processes emotions while gateway streams a response."""
        sim_reachy = e2e_env
        from reachy_claw.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            enable_motion=True,
            play_emotions=True,
            tts_backend="none",
            stt_backend="whisper",
        )
        app = ReachyClawApp(config)
        app.reachy = sim_reachy

        # Start motion plugin
        motion = MotionPlugin(app)
        motion._running = True

        motion_task = asyncio.create_task(motion._motion_loop())

        # Connect gateway and send message
        client = DesktopRobotClient(config)
        stream_done = asyncio.Event()

        async def on_end(text, rid):
            stream_done.set()

        client.callbacks.on_stream_end = on_end
        await client.connect()

        # Queue emotion as conversation would
        app.emotions.queue_emotion("thinking")

        # Send message
        await client.send_message_streaming("Say hi.")
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)

        # Queue response emotion
        app.emotions.queue_emotion("happy")
        await asyncio.sleep(1.0)

        # Stop motion
        motion._running = False
        await motion_task

        await client.disconnect()
        app.reachy = None  # don't disconnect shared fixture

    @pytest.mark.asyncio
    async def test_full_conversation_turn_no_audio(self, e2e_env):
        """Simulate a full conversation turn: text in -> gateway -> text out.

        Bypasses STT/TTS (no audio hardware needed) but exercises the full
        gateway protocol and emotion/motion pipeline.
        """
        sim_reachy = e2e_env
        from reachy_claw.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            enable_motion=True,
            play_emotions=True,
            tts_backend="none",
        )
        app = ReachyClawApp(config)
        app.reachy = sim_reachy

        # Start motion plugin in background
        motion = MotionPlugin(app)
        motion._running = True
        motion_task = asyncio.create_task(motion._motion_loop())

        # Connect gateway
        client = DesktopRobotClient(config)
        stream_text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_response = ""

        async def on_delta(text, rid):
            await stream_text_queue.put(text)

        async def on_end(text, rid):
            nonlocal full_response
            full_response = text
            await stream_text_queue.put(None)

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end

        await client.connect()

        # Simulate conversation turn
        app.emotions.queue_emotion("thinking")
        await client.send_message_streaming("Say exactly: Hello from e2e test")

        # Collect streamed response
        collected = ""
        while True:
            chunk = await asyncio.wait_for(stream_text_queue.get(), timeout=30.0)
            if chunk is None:
                break
            collected += chunk

        assert len(collected) > 0 or len(full_response) > 0

        # Queue response emotion
        app.emotions.queue_emotion("happy")
        await asyncio.sleep(0.5)

        # Cleanup
        motion._running = False
        await motion_task
        await client.disconnect()
        app.reachy = None


# ── Interrupt / barge-in ────────────────────────────────────────────────


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_during_stream(self, e2e_env):
        """Sending interrupt during streaming should be accepted without error."""
        config = Config()
        client = DesktopRobotClient(config)
        events = []

        client.callbacks.on_stream_start = lambda rid: events.append("start")
        client.callbacks.on_stream_delta = lambda t, r: events.append("delta")
        client.callbacks.on_stream_end = lambda t, r: events.append("end")
        client.callbacks.on_stream_abort = lambda r, rid: events.append("abort")

        await client.connect()

        # Ask for a response
        await client.send_message_streaming(
            "Write a story about a robot exploring space."
        )

        # Wait for stream to start
        for _ in range(30):
            await asyncio.sleep(0.3)
            if "start" in events:
                break

        # Send interrupt (may or may not abort depending on timing)
        await client.send_interrupt()

        # Wait for stream to finish (either abort or end)
        for _ in range(30):
            await asyncio.sleep(0.5)
            if "abort" in events or "end" in events:
                break

        await client.disconnect()

        # Stream must have started, and either completed or been aborted
        assert "start" in events
        assert "abort" in events or "end" in events


# ── State changes ───────────────────────────────────────────────────────


class TestStateManagement:
    @pytest.mark.asyncio
    async def test_state_change_listening(self, e2e_env):
        config = Config()
        client = DesktopRobotClient(config)
        await client.connect()

        # Should not raise
        await client.send_state_change("listening")
        await asyncio.sleep(0.3)
        await client.send_state_change("speaking_done")
        await asyncio.sleep(0.3)

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_turns(self, e2e_env):
        """Multiple conversation turns in sequence."""
        config = Config()
        client = DesktopRobotClient(config)
        await client.connect()

        for i in range(3):
            await client.send_state_change("listening")
            response = await client.send_message(f"Turn {i+1}: say OK")
            assert len(response) > 0
            await client.send_state_change("speaking_done")

        await client.disconnect()
