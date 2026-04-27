"""Tests for the stateless Home Assistant REST client."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from reachy_claw import ha_client


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_probe_ok(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/"
        assert req.headers["Authorization"] == "Bearer abc"
        return httpx.Response(200, json={"message": "API running."})

    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "abc")
    assert out["ok"] is True
    assert out["status"] == 200
    assert "running" in out["message"].lower()


@pytest.mark.asyncio
async def test_probe_unauthorized(monkeypatch):
    def handler(req): return httpx.Response(401, text="invalid")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "wrong")
    assert out == {"ok": False, "status": 401, "message": "Unauthorized — token rejected"}


@pytest.mark.asyncio
async def test_probe_connection_error(monkeypatch):
    def handler(req): raise httpx.ConnectError("refused")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "x")
    assert out["ok"] is False
    assert out["status"] == 0
    assert "ConnectError" in out["message"]


@pytest.mark.asyncio
async def test_probe_bad_url():
    out = await ha_client.probe("ha.local", "x")
    assert out["ok"] is False
    assert out["status"] == 0
    assert "http://" in out["message"]


@pytest.mark.asyncio
async def test_list_states_returns_list(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/states"
        assert req.headers["Authorization"] == "Bearer t"
        return httpx.Response(200, json=[
            {"entity_id": "weather.home", "state": "sunny",
             "attributes": {"friendly_name": "Home", "temperature": 22.5}},
            {"entity_id": "sensor.bedroom_temp", "state": "21.3",
             "attributes": {"unit_of_measurement": "°C"}},
        ])
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.list_states("http://ha.local:8123/", "t")
    assert len(out) == 2
    assert out[0]["entity_id"] == "weather.home"


@pytest.mark.asyncio
async def test_list_states_unauthorized_raises(monkeypatch):
    def handler(req): return httpx.Response(401, text="bad")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnauthorized):
        await ha_client.list_states("http://ha.local:8123", "x")


@pytest.mark.asyncio
async def test_list_states_timeout_raises(monkeypatch):
    def handler(req): raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnreachable):
        await ha_client.list_states("http://ha.local:8123", "x")


@pytest.mark.asyncio
async def test_get_history_builds_query(monkeypatch):
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json=[
            [
                {"entity_id": "weather.home", "state": "sunny",
                 "attributes": {"temperature": 22.0},
                 "last_updated": "2026-04-27T10:00:00+00:00"},
                {"entity_id": "weather.home", "state": "cloudy",
                 "attributes": {"temperature": 21.0},
                 "last_updated": "2026-04-27T14:00:00+00:00"},
            ],
            [
                {"entity_id": "sensor.temp", "state": "20",
                 "attributes": {},
                 "last_updated": "2026-04-27T08:00:00+00:00"},
            ],
        ])

    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    start = datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    out = await ha_client.get_history(
        "http://ha.local:8123", "t", ["weather.home", "sensor.temp"], start, end
    )
    assert captured["path"].startswith("/api/history/period/")
    assert "filter_entity_id" in captured["params"]
    assert "weather.home" in captured["params"]["filter_entity_id"]
    assert "sensor.temp" in captured["params"]["filter_entity_id"]
    # We deliberately do NOT pass minimal_response (see ha_client.get_history docstring).
    assert "minimal_response" not in captured["params"]
    assert out["weather.home"][0]["state"] == "sunny"
    assert out["weather.home"][1]["state"] == "cloudy"
    assert out["sensor.temp"][0]["state"] == "20"
    assert "ts" in out["weather.home"][0]
    assert "attributes" in out["weather.home"][0]


@pytest.mark.asyncio
async def test_get_history_handles_minimal_rows(monkeypatch):
    """Even without minimal_response, defensive parsing should handle rows
    that omit entity_id (carry-over from first row in same series) and rows
    with empty/missing attributes."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            [
                {"entity_id": "weather.home", "state": "sunny",
                 "attributes": {"temp": 22},
                 "last_updated": "2026-04-27T10:00:00+00:00"},
                # Subsequent rows in same series with missing entity_id /
                # missing attributes — must not crash and must associate
                # with weather.home.
                {"state": "cloudy",
                 "last_updated": "2026-04-27T14:00:00+00:00"},
            ],
            [],  # empty series
            [
                {"entity_id": "sensor.temp", "state": "20",
                 "last_updated": "2026-04-27T08:00:00+00:00"},
            ],
        ])
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.get_history(
        "http://ha.local:8123", "t", ["weather.home", "sensor.temp"],
        datetime(2026, 4, 27, tzinfo=timezone.utc),
        datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    assert len(out["weather.home"]) == 2
    assert out["weather.home"][1]["state"] == "cloudy"
    assert out["weather.home"][1]["attributes"] == {}
    assert len(out["sensor.temp"]) == 1


@pytest.mark.asyncio
async def test_get_history_empty_entity_list_returns_empty(monkeypatch):
    def handler(req): raise AssertionError("should not be called")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.get_history(
        "http://ha.local:8123", "t", [],
        datetime(2026, 4, 27, tzinfo=timezone.utc),
        datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    assert out == {}


@pytest.mark.asyncio
async def test_get_history_unauthorized(monkeypatch):
    def handler(req): return httpx.Response(401)
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnauthorized):
        await ha_client.get_history(
            "http://ha.local:8123", "x", ["sensor.a"],
            datetime(2026, 4, 27, tzinfo=timezone.utc),
            datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
