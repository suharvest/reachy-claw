"""Tests for /api/settings/<namespace> endpoints."""

from __future__ import annotations


import pytest
from aiohttp import web

from reachy_claw.config import Config


@pytest.fixture
def app_with_dashboard(tmp_path, monkeypatch):
    """Construct a minimal aiohttp app exposing settings endpoints
    bound to a temp config + temp DATA_DIR."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from reachy_claw.plugins.dashboard_plugin import (
        _build_settings_handlers,
    )

    config = Config()

    class _StubApp:
        pass

    stub = _StubApp()
    stub.config = config

    aio = web.Application()
    handlers = _build_settings_handlers(stub)
    aio.router.add_get("/api/settings/{namespace}", handlers["get"])
    aio.router.add_put("/api/settings/{namespace}", handlers["put"])
    return aio, stub, tmp_path


@pytest.mark.asyncio
async def test_get_rest_namespace(app_with_dashboard, aiohttp_client):
    aio, stub, _ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.get("/api/settings/rest")
    assert r.status == 200
    body = await r.json()
    assert body["enabled"] is True
    assert body["window_start"] == "23:00"


@pytest.mark.asyncio
async def test_get_unknown_namespace_404(app_with_dashboard, aiohttp_client):
    aio, *_ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.get("/api/settings/nope")
    assert r.status == 404


@pytest.mark.asyncio
async def test_put_updates_config_and_persists(app_with_dashboard, aiohttp_client, tmp_path):
    aio, stub, dd = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put(
        "/api/settings/rest",
        json={"window_start": "22:30", "enabled": False},
    )
    assert r.status == 200
    body = await r.json()
    assert set(body["updated"]) == {"window_start", "enabled"}
    # In-process config mutated:
    assert stub.config.rest_window_start == "22:30"
    assert stub.config.rest_enabled is False
    # runtime-overrides.yaml on disk has the values:
    overrides_path = dd / "runtime-overrides.yaml"
    assert overrides_path.exists()
    content = overrides_path.read_text()
    assert "window_start" in content or "rest_window_start" in content


@pytest.mark.asyncio
async def test_put_rejects_invalid_value(app_with_dashboard, aiohttp_client):
    aio, stub, _ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put("/api/settings/rest", json={"window_start": "25:99"})
    assert r.status == 400


@pytest.mark.asyncio
async def test_put_rejects_unknown_key(app_with_dashboard, aiohttp_client):
    aio, *_ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put("/api/settings/rest", json={"made_up": 1})
    assert r.status == 400


@pytest.mark.asyncio
async def test_put_rejects_degenerate_window(app_with_dashboard, aiohttp_client):
    aio, *_ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put(
        "/api/settings/rest",
        json={"window_start": "23:00", "window_end": "23:00"},
    )
    assert r.status == 400


@pytest.mark.asyncio
async def test_overrides_reload_on_next_startup(app_with_dashboard, aiohttp_client, tmp_path, monkeypatch):
    """After PUT, a fresh Config load picks up the runtime-overrides.yaml values."""
    aio, _stub, _ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put("/api/settings/rest", json={"window_start": "21:15"})
    assert r.status == 200

    # Simulate a fresh process: reload Config from disk in the same DATA_DIR.
    from reachy_claw.config import load_config
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    fresh = load_config()
    assert fresh.rest_window_start == "21:15"
