"""Tests for dashboard HA endpoints: /api/ha/test, /api/ha/entities,
PUT /api/settings/ha."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from reachy_claw.config import Config
from reachy_claw.plugins.dashboard_plugin import _build_ha_handlers


class _FakeApp:
    def __init__(self, **cfg):
        self.config = Config(**cfg)


@pytest.fixture
def aio_app():
    app = web.Application()
    fake = _FakeApp(ha_url="http://ha.local:8123", ha_token="tok",
                    ha_entities=["weather.home"])
    handlers = _build_ha_handlers(fake)
    app.router.add_post("/api/ha/test", handlers["test"])
    app.router.add_get("/api/ha/entities", handlers["entities"])
    return app, fake


async def test_ha_test_uses_body_overrides(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.probe",
               new=AsyncMock(return_value={"ok": True, "status": 200, "message": "ok"})) as mp:
        resp = await client.post("/api/ha/test",
                                 json={"url": "http://other:8123", "token": "x"})
    assert resp.status == 200
    body = await resp.json()
    assert body == {"ok": True, "status": 200, "message": "ok"}
    mp.assert_awaited_once_with("http://other:8123", "x")


async def test_ha_test_falls_back_to_config(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.probe",
               new=AsyncMock(return_value={"ok": False, "status": 401, "message": "bad"})) as mp:
        resp = await client.post("/api/ha/test", json={})
    assert resp.status == 200
    mp.assert_awaited_once_with("http://ha.local:8123", "tok")


async def test_entities_unconfigured_returns_400(aiohttp_client):
    app = web.Application()
    fake = _FakeApp(ha_url="", ha_token="")
    handlers = _build_ha_handlers(fake)
    app.router.add_get("/api/ha/entities", handlers["entities"])
    client = await aiohttp_client(app)
    resp = await client.get("/api/ha/entities")
    assert resp.status == 400
    body = await resp.json()
    assert "ha_url" in body["error"]


async def test_entities_groups_by_domain(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    states = [
        {"entity_id": "weather.home", "state": "sunny",
         "attributes": {"friendly_name": "Home"}},
        {"entity_id": "sensor.temp", "state": "21", "attributes": {}},
        {"entity_id": "sensor.cpu", "state": "45",
         "attributes": {"friendly_name": "CPU"}},
        {"entity_id": "no_dot_here", "state": "x", "attributes": {}},  # filtered
    ]
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.list_states",
               new=AsyncMock(return_value=states)):
        resp = await client.get("/api/ha/entities")
    assert resp.status == 200
    body = await resp.json()
    domains = [g["domain"] for g in body["groups"]]
    assert domains == ["sensor", "weather"]  # sorted
    sensor_group = next(g for g in body["groups"] if g["domain"] == "sensor")
    assert sensor_group["count"] == 2
    assert [e["entity_id"] for e in sensor_group["entities"]] == ["sensor.cpu", "sensor.temp"]


async def test_entities_unauthorized_returns_502(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    from reachy_claw import ha_client as hc
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.list_states",
               new=AsyncMock(side_effect=hc.HAUnauthorized("bad"))):
        resp = await client.get("/api/ha/entities")
    assert resp.status == 502
    body = await resp.json()
    assert "unauthorized" in body["error"]


async def test_settings_ha_put_persists(tmp_path, monkeypatch, aiohttp_client):
    """PUT /api/settings/ha goes through the generic settings handler — verify
    list-type validation and Config update."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".reachy-claw"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("")

    from reachy_claw.config import load_config
    from reachy_claw.plugins.dashboard_plugin import _build_settings_handlers

    fake = _FakeApp()
    fake.config = load_config()
    handlers = _build_settings_handlers(fake)
    app = web.Application()
    app.router.add_put("/api/settings/{namespace}", handlers["put"])
    client = await aiohttp_client(app)

    resp = await client.put("/api/settings/ha", json={
        "url": "http://ha.local:8123",
        "token": "tok",
        "entities": ["weather.home", "sensor.bedroom_temp"],
    })
    assert resp.status == 200
    assert fake.config.ha_url == "http://ha.local:8123"
    assert fake.config.ha_entities == ["weather.home", "sensor.bedroom_temp"]


async def test_settings_ha_put_rejects_bad_entity(aiohttp_client):
    from reachy_claw.plugins.dashboard_plugin import _build_settings_handlers
    fake = _FakeApp()
    handlers = _build_settings_handlers(fake)
    app = web.Application()
    app.router.add_put("/api/settings/{namespace}", handlers["put"])
    client = await aiohttp_client(app)

    resp = await client.put("/api/settings/ha", json={"entities": ["BadFormat"]})
    assert resp.status == 400
    body = await resp.json()
    assert "entity_id" in body["error"]
