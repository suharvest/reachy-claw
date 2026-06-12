"""Route-wiring tests for the Tier-A HTTP endpoints.

The reachy-mini SDK can't be imported in CI (it needs gstreamer/gi), so we
can't instantiate ``ReachyVoiceApp`` directly. Instead we register the SAME
route bodies (``tier_a.register_http_routes`` — the exact function main.py
calls) onto a bare FastAPI app and drive them through ``TestClient``, with the
upstream proxy calls monkeypatched. This verifies the route plumbing:

  * /api/ollama/models passes the ?url= query through (and falls back to
    config.edge_llm_url when absent),
  * /api/captures/list forwards the proxied JSON,
  * /api/captures/image/{filename} maps (status, body, headers) onto the HTTP
    response (200 jpeg, upstream 404, 502 error).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_voice import tier_a


@dataclass
class _Cfg:
    edge_llm_url: str = "http://config-host:11434"
    vision_mjpeg: str = "http://192.168.1.50:8630/stream"


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    tier_a.register_http_routes(app, lambda: _Cfg())
    return TestClient(app)


# ── /api/ollama/models ───────────────────────────────────────────────────
def test_ollama_route_passes_url_query(client, monkeypatch):
    captured: dict = {}

    async def fake(base):
        captured["base"] = base
        return {"models": ["m1"], "source": "ollama"}

    monkeypatch.setattr(tier_a, "fetch_ollama_models", fake)

    r = client.get("/api/ollama/models", params={"url": "http://ui-host:11434"})
    assert r.status_code == 200
    assert r.json() == {"models": ["m1"], "source": "ollama"}
    assert captured["base"] == "http://ui-host:11434"  # ?url= wins


def test_ollama_route_falls_back_to_config_url(client, monkeypatch):
    captured: dict = {}

    async def fake(base):
        captured["base"] = base
        return {"models": [], "source": "fallback"}

    monkeypatch.setattr(tier_a, "fetch_ollama_models", fake)

    r = client.get("/api/ollama/models")  # no ?url=
    assert r.status_code == 200
    assert captured["base"] == "http://config-host:11434"


# ── /api/captures/list ───────────────────────────────────────────────────
def test_captures_list_route_forwards_json(client, monkeypatch):
    seen: dict = {}

    async def fake(mjpeg):
        seen["mjpeg"] = mjpeg
        return {"files": ["a.jpg"], "total": 1}

    monkeypatch.setattr(tier_a, "captures_list", fake)

    r = client.get("/api/captures/list")
    assert r.status_code == 200
    assert r.json() == {"files": ["a.jpg"], "total": 1}
    assert seen["mjpeg"] == "http://192.168.1.50:8630/stream"


# ── /api/captures/image/{filename} ───────────────────────────────────────
def test_captures_image_route_200_jpeg(client, monkeypatch):
    async def fake(mjpeg, filename):
        assert filename == "smile.jpg"
        return 200, b"\xff\xd8jpeg", {
            "Content-Type": "image/jpeg",
            "Cache-Control": "public, max-age=86400",
        }

    monkeypatch.setattr(tier_a, "captures_image", fake)

    r = client.get("/api/captures/image/smile.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8jpeg"
    assert r.headers["content-type"] == "image/jpeg"
    assert "max-age" in r.headers["cache-control"]


def test_captures_image_route_forwards_upstream_404(client, monkeypatch):
    async def fake(mjpeg, filename):
        return 404, b"", {}

    monkeypatch.setattr(tier_a, "captures_image", fake)

    r = client.get("/api/captures/image/missing.jpg")
    assert r.status_code == 404


def test_captures_image_route_502_on_error(client, monkeypatch):
    async def fake(mjpeg, filename):
        return 502, b"down", {"Content-Type": "text/plain"}

    monkeypatch.setattr(tier_a, "captures_image", fake)

    r = client.get("/api/captures/image/x.jpg")
    assert r.status_code == 502
    assert r.content == b"down"
