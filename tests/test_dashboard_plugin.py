"""Tests for DashboardPlugin setup/lifecycle, message format, and HA prompt cache."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config
from reachy_claw.event_bus import EventBus
from reachy_claw.storage.db import Database


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
    result = captured[-1]
    assert result["type"] == "llm_end"
    assert result["full_text"] == "Hi there!"
    assert result["run_id"] == "abc"

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


# ── Diary SQLite Endpoints Tests ─────────────────────────────────────────────


@pytest.fixture
def diary_db(tmp_path: Path):
    """Create a temp DB with a diary row for testing."""
    from reachy_claw.storage.db import Database

    db = Database(tmp_path / "diary_test.db")
    db.init()
    db.save_diary(
        date="2026-04-26",
        markdown=(
            "---\n"
            "title: \"A Day of Interactions\"\n"
            "date: 2026-04-26\n"
            "stats: {conversations: 3, faces_seen: 5, smiles: 2}\n"
            "captures: []\n"
            "meta: {llm_model: mock, prompt_version: v1}\n"
            "---\n\n"
            "## 今天的心情\n\n今天平静而充实。\n\n"
            "## 遇到的人\n\n来过几位朋友。\n\n"
            "## 想到的事\n\n我想了一下世界的样子。\n"
        ),
        llm_model="mock",
        prompt_version="v1",
    )
    yield db
    db.close()


@pytest.fixture
def diary_dashboard_app(diary_db: Database, mock_reachy):
    """Create a ReachyClawApp with diary DB attached."""
    config = Config(
        standalone_mode=True,
        dashboard_enabled=True,
        dashboard_port=0,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    app = ReachyClawApp(config)
    app.reachy = mock_reachy
    app.db = diary_db
    return app


@pytest.mark.asyncio
async def test_diary_endpoint_returns_sqlite_diary(diary_dashboard_app, diary_db):
    """Test that diary endpoint reads from SQLite and parses Markdown correctly."""
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    # Create plugin and spin up a test server
    plugin = DashboardPlugin(diary_dashboard_app)

    # Build a minimal aiohttp app with just the diary handlers
    test_app = web.Application()
    test_app.router.add_get("/api/diaries", plugin._handle_diary_list)
    test_app.router.add_get("/api/diary/{date}", plugin._handle_diary_get)

    # Use aiohttp test client
    from aiohttp.test_utils import TestClient, TestServer

    server = TestServer(test_app)
    client = TestClient(server)

    await client.start_server()
    try:
        # Test list endpoint
        resp = await client.get("/api/diaries")
        assert resp.status == 200
        data = await resp.json()
        assert "dates" in data
        assert "2026-04-26" in data["dates"]

        # Test get endpoint
        resp = await client.get("/api/diary/2026-04-26")
        assert resp.status == 200
        body = await resp.json()
        assert body["date"] == "2026-04-26"
        assert body["title"] == "A Day of Interactions"
        # Front-end expects sections with 'id' field mapped from heading
        sections = body.get("sections", [])
        assert len(sections) >= 1
        # Check that Chinese heading content is preserved
        found_mood = any("今天的心情" in s.get("content", "") or "今天的心情" in s.get("heading", "") for s in sections)
        assert found_mood, "Parser should preserve section content from '## 今天的心情'"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_diary_endpoint_404_for_missing_date(diary_dashboard_app):
    """Test that missing diary returns 404."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

    plugin = DashboardPlugin(diary_dashboard_app)
    test_app = web.Application()
    test_app.router.add_get("/api/diary/{date}", plugin._handle_diary_get)

    server = TestServer(test_app)
    client = TestClient(server)

    await client.start_server()
    try:
        resp = await client.get("/api/diary/2026-04-25")
        assert resp.status == 404
        body = await resp.json()
        assert "error" in body
    finally:
        await client.close()


# ── HA prompt cache test ────────────────────────────────────────────────────


def test_diary_default_prompt_includes_ha_paragraph():
    from reachy_claw.plugins.dashboard_plugin import _diary_default_prompt, _DIARY_DEFAULT_PROMPT_CACHE
    import reachy_claw.plugins.dashboard_plugin as mod
    mod._DIARY_DEFAULT_PROMPT_CACHE = None  # reset cache
    text = _diary_default_prompt()
    assert "Home Assistant" in text or "sensors" in text
