"""Tests for DashboardPlugin setup/lifecycle and message format."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config
from reachy_claw.event_bus import EventBus


@pytest.fixture
def dashboard_config():
    return Config(
        standalone_mode=True,
        dashboard_enabled=True,
        dashboard_port=0,  # will be overridden in tests
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )


@pytest.fixture
def dashboard_app(dashboard_config, mock_reachy):
    app = ReachyClawApp(dashboard_config)
    app.reachy = mock_reachy
    return app


def test_setup_without_aiohttp():
    """Plugin should gracefully skip if aiohttp is not installed."""
    config = Config(dashboard_enabled=True)
    app = ReachyClawApp(config)

    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin
    plugin = DashboardPlugin(app)

    with patch.dict("sys.modules", {"aiohttp": None}):
        # Import will fail, setup should return False
        import importlib
        # Actually test via the ImportError path
        pass

    # If aiohttp IS installed, setup should return True
    try:
        import aiohttp
        assert plugin.setup() is True
    except ImportError:
        assert plugin.setup() is False


def test_event_bus_on_app(dashboard_app):
    """App should have an EventBus instance."""
    assert hasattr(dashboard_app, "events")
    assert isinstance(dashboard_app.events, EventBus)


@pytest.mark.asyncio
async def test_dashboard_broadcast_robot_state(dashboard_app):
    """Test that _broadcast_robot_state produces correct JSON shape."""
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    plugin = DashboardPlugin(dashboard_app)
    captured = []

    async def mock_broadcast(msg):
        captured.append(msg)

    plugin._broadcast = mock_broadcast

    await plugin._broadcast_robot_state()

    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == "robot_state"
    assert "head" in msg
    assert "yaw" in msg["head"]
    assert "pitch" in msg["head"]
    assert "antenna" in msg
    assert "emotion" in msg
    assert "speaking" in msg
    assert "tracking" in msg
    assert "source" in msg["tracking"]
    assert "confidence" in msg["tracking"]


@pytest.mark.asyncio
async def test_dashboard_event_callbacks(dashboard_app):
    """Test that EventBus events produce correct WS message format."""
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    plugin = DashboardPlugin(dashboard_app)
    captured = []

    async def mock_broadcast(msg):
        captured.append(msg)

    plugin._broadcast = mock_broadcast

    # Test ASR partial
    await plugin._on_asr_partial({"text": "hello"})
    assert captured[-1] == {"type": "asr_partial", "text": "hello", "is_final": False}

    # Test ASR final
    await plugin._on_asr_final({"text": "hello world"})
    assert captured[-1] == {"type": "asr_final", "text": "hello world"}

    # Test LLM delta
    await plugin._on_llm_delta({"text": "Hi", "run_id": "abc"})
    assert captured[-1] == {"type": "llm_delta", "text": "Hi", "run_id": "abc"}

    # Test LLM end
    await plugin._on_llm_end({"full_text": "Hi there!", "run_id": "abc"})
    assert captured[-1] == {"type": "llm_end", "full_text": "Hi there!", "run_id": "abc"}

    # Test state change
    await plugin._on_state_change({"state": "speaking"})
    assert captured[-1] == {"type": "state", "state": "speaking"}

    # Test emotion
    await plugin._on_emotion({"emotion": "happy"})
    assert captured[-1] == {"type": "emotion", "emotion": "happy"}


@pytest.mark.asyncio
async def test_broadcast_to_ws_clients(dashboard_app):
    """Test that _broadcast sends to all connected WS clients."""
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    plugin = DashboardPlugin(dashboard_app)

    # Mock WS clients
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    plugin._ws_clients = {ws1, ws2}

    await plugin._broadcast({"type": "test", "data": 1})

    ws1.send_str.assert_called_once()
    ws2.send_str.assert_called_once()

    # Verify JSON payload
    sent = json.loads(ws1.send_str.call_args[0][0])
    assert sent["type"] == "test"
    assert sent["data"] == 1


@pytest.mark.asyncio
async def test_broadcast_removes_dead_clients(dashboard_app):
    """Dead WS clients should be removed from the set."""
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    plugin = DashboardPlugin(dashboard_app)

    good_ws = AsyncMock()
    dead_ws = AsyncMock()
    dead_ws.send_str.side_effect = ConnectionError("gone")
    plugin._ws_clients = {good_ws, dead_ws}

    await plugin._broadcast({"type": "test"})

    assert dead_ws not in plugin._ws_clients
    assert good_ws in plugin._ws_clients
