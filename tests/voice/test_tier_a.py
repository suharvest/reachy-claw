"""Unit tests for the Tier-A dashboard APIs (``reachy_voice.tier_a``).

These lock in the exact wire shapes the shared dashboard UI (``static/app.js``)
consumes, and exercise every branch (success / upstream-error / connection-
failure) against a mocked ``httpx`` transport — NO hardware, NO docker, NO
vision-trt, NO ollama.
"""

from __future__ import annotations

import httpx
import pytest

from reachy_voice import tier_a

VISION_MJPEG = "http://192.168.1.50:8630/stream"


def _client(handler) -> httpx.AsyncClient:
    """An AsyncClient whose every request is served by ``handler`` (a
    ``request -> httpx.Response`` callable). No network is touched."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── host helpers ─────────────────────────────────────────────────────────
def test_vision_host_parsed_from_mjpeg_url():
    assert tier_a.vision_host("http://192.168.1.50:8630/stream") == "192.168.1.50"
    assert tier_a.vision_host("http://127.0.0.1:8630/stream") == "127.0.0.1"


def test_vision_http_base_uses_8630():
    assert (
        tier_a.vision_http_base("http://10.0.0.9:8630/stream")
        == "http://10.0.0.9:8630"
    )


# ── 1. Ollama models proxy ───────────────────────────────────────────────
@pytest.mark.asyncio
async def test_ollama_models_success_returns_names():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:4b"}, {"name": "llama3:8b"}, {"name": ""}]},
        )

    out = await tier_a.fetch_ollama_models(
        "http://host:11434", client=_client(handler)
    )
    # hits /api/tags, returns only non-empty names, source=ollama
    assert seen["url"] == "http://host:11434/api/tags"
    assert out == {"models": ["qwen3:4b", "llama3:8b"], "source": "ollama"}


@pytest.mark.asyncio
async def test_ollama_models_trailing_slash_stripped():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://host:11434/api/tags"
        return httpx.Response(200, json={"models": [{"name": "m"}]})

    await tier_a.fetch_ollama_models("http://host:11434/", client=_client(handler))


@pytest.mark.asyncio
async def test_ollama_models_empty_list_falls_back():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    out = await tier_a.fetch_ollama_models("http://host", client=_client(handler))
    assert out["models"] == tier_a.DEFAULT_OLLAMA_MODELS
    assert out["source"] == "ollama"


@pytest.mark.asyncio
async def test_ollama_models_non_200_falls_back():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    out = await tier_a.fetch_ollama_models("http://host", client=_client(handler))
    assert out == {"models": tier_a.DEFAULT_OLLAMA_MODELS, "source": "fallback"}


@pytest.mark.asyncio
async def test_ollama_models_connection_error_falls_back():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    out = await tier_a.fetch_ollama_models("http://host", client=_client(handler))
    assert out == {"models": tier_a.DEFAULT_OLLAMA_MODELS, "source": "fallback"}


# ── 2. Docker service restart ────────────────────────────────────────────
def test_default_restart_order_uses_reachy_voice_not_claw():
    names = [n for n, _ in tier_a.DEFAULT_RESTART_CONTAINERS]
    assert names == ["vision-trt", "reachy-daemon", "reachy-voice"]
    assert "reachy-claw" not in names
    # vision-trt is the only wait_healthy=True entry (camera ordering)
    assert dict(tier_a.DEFAULT_RESTART_CONTAINERS)["vision-trt"] is True


def test_resolve_containers_skips_remote_vision():
    local = tier_a._resolve_containers("http://127.0.0.1:8630/stream")
    assert [n for n, _ in local] == ["vision-trt", "reachy-daemon", "reachy-voice"]

    remote = tier_a._resolve_containers("http://192.168.1.50:8630/stream")
    assert [n for n, _ in remote] == ["reachy-daemon", "reachy-voice"]


@pytest.mark.asyncio
async def test_restart_services_local_order_and_status_messages():
    posts: list[str] = []
    health_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/restart"):
            posts.append(path.split("/")[2])  # /containers/<name>/restart
            return httpx.Response(204)
        if request.method == "GET" and path.endswith("/json"):
            name = path.split("/")[2]
            health_calls.append(name)
            return httpx.Response(200, json={"State": {"Health": {"Status": "healthy"}}})
        return httpx.Response(404)

    msgs: list[dict] = []

    async def broadcast(m: dict) -> None:
        msgs.append(m)

    async def fast_sleep(_):  # never actually wait
        return None

    await tier_a.restart_services(
        "http://127.0.0.1:8630/stream",
        broadcast,
        client=_client(handler),
        sleep=fast_sleep,
        now=lambda: 0.0,
    )

    # restarted all three, in order, vision-trt first
    assert posts == ["vision-trt", "reachy-daemon", "reachy-voice"]
    # only vision-trt is wait_healthy → only it gets a health poll
    assert health_calls == ["vision-trt"]

    # status stream: starting, restarting×3 (named), done — no error
    assert msgs[0] == {"type": "restart_status", "status": "starting"}
    assert msgs[-1] == {"type": "restart_status", "status": "done"}
    restarting = [m for m in msgs if m.get("status") == "restarting"]
    assert [m["container"] for m in restarting] == [
        "vision-trt",
        "reachy-daemon",
        "reachy-voice",
    ]
    assert all(m["status"] != "error" for m in msgs)


@pytest.mark.asyncio
async def test_restart_services_remote_vision_skips_it():
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(request.url.path.split("/")[2])
            return httpx.Response(204)
        return httpx.Response(404)

    msgs: list[dict] = []

    async def broadcast(m: dict) -> None:
        msgs.append(m)

    await tier_a.restart_services(
        "http://192.168.1.50:8630/stream",  # remote vision host
        broadcast,
        client=_client(handler),
        sleep=lambda _: _noop(),
        now=lambda: 0.0,
    )
    assert posts == ["reachy-daemon", "reachy-voice"]  # vision-trt skipped
    assert msgs[-1] == {"type": "restart_status", "status": "done"}


async def _noop():
    return None


@pytest.mark.asyncio
async def test_restart_services_health_poll_advances_after_unhealthy():
    # vision-trt reports starting, then healthy → loop must poll twice then move on
    states = iter([
        {"State": {"Health": {"Status": "starting"}}},
        {"State": {"Health": {"Status": "healthy"}}},
    ])
    sleeps = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(204)
        if request.url.path.endswith("/json"):
            return httpx.Response(200, json=next(states))
        return httpx.Response(404)

    async def broadcast(_):
        return None

    async def sleep(_):
        sleeps["n"] += 1

    t = {"v": 0.0}

    def now():
        t["v"] += 1.0
        return t["v"]

    await tier_a.restart_services(
        "http://127.0.0.1:8630/stream",
        broadcast,
        client=_client(handler),
        sleep=sleep,
        now=now,
        health_timeout=100.0,
    )
    # polled at least twice (starting then healthy) → at least one sleep between
    assert sleeps["n"] >= 1


@pytest.mark.asyncio
async def test_restart_services_socket_error_broadcasts_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no docker socket")

    msgs: list[dict] = []

    async def broadcast(m: dict) -> None:
        msgs.append(m)

    # A POST that raises is swallowed per-container (continue), so the failure
    # path that yields status=error is a broadcast()-level fault. Force it by
    # making broadcast raise once we're inside the loop.
    calls = {"n": 0}

    async def flaky_broadcast(m: dict) -> None:
        calls["n"] += 1
        msgs.append(m)
        if calls["n"] == 2:  # the first per-container "restarting"
            raise RuntimeError("ws gone")

    await tier_a.restart_services(
        "http://127.0.0.1:8630/stream",
        flaky_broadcast,
        client=_client(handler),
        sleep=lambda _: _noop(),
        now=lambda: 0.0,
    )
    assert any(m.get("status") == "error" for m in msgs)


# ── 3. Captures (vision-trt :8630) ───────────────────────────────────────
@pytest.mark.asyncio
async def test_capture_info_success(monkeypatch):
    monkeypatch.setenv("HOST_DATA_DIR", "/data/reachy")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://192.168.1.50:8630/api/captures/count"
        return httpx.Response(200, json={"count": 42})

    out = await tier_a.capture_info(VISION_MJPEG, client=_client(handler))
    assert out == {
        "type": "capture_info",
        "path": "/data/reachy/vision/captures",
        "count": 42,
    }


@pytest.mark.asyncio
async def test_capture_info_error_count_zero(monkeypatch):
    monkeypatch.setenv("HOST_DATA_DIR", "/data/reachy")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    out = await tier_a.capture_info(VISION_MJPEG, client=_client(handler))
    assert out["type"] == "capture_info"
    assert out["count"] == 0
    assert out["path"] == "/data/reachy/vision/captures"


@pytest.mark.asyncio
async def test_clear_captures_sends_delete_and_returns_reset():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    out = await tier_a.clear_captures(VISION_MJPEG, client=_client(handler))
    assert seen["method"] == "DELETE"
    assert seen["url"] == "http://192.168.1.50:8630/api/captures"
    assert out == {"type": "capture_reset", "count": 0}


@pytest.mark.asyncio
async def test_clear_captures_swallows_error_still_resets():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    out = await tier_a.clear_captures(VISION_MJPEG, client=_client(handler))
    assert out == {"type": "capture_reset", "count": 0}


@pytest.mark.asyncio
async def test_captures_list_proxies_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://192.168.1.50:8630/api/captures/list"
        return httpx.Response(200, json={"files": ["a.jpg", "b.jpg"], "total": 2})

    out = await tier_a.captures_list(VISION_MJPEG, client=_client(handler))
    assert out == {"files": ["a.jpg", "b.jpg"], "total": 2}


@pytest.mark.asyncio
async def test_captures_list_error_returns_empty_with_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    out = await tier_a.captures_list(VISION_MJPEG, client=_client(handler))
    assert out["files"] == []
    assert out["total"] == 0
    assert "error" in out


@pytest.mark.asyncio
async def test_captures_image_200_returns_jpeg_bytes_and_cache_header():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://192.168.1.50:8630/api/captures/image/x.jpg"
        return httpx.Response(200, content=b"\xff\xd8jpeg")

    status, body, headers = await tier_a.captures_image(
        VISION_MJPEG, "x.jpg", client=_client(handler)
    )
    assert status == 200
    assert body == b"\xff\xd8jpeg"
    assert headers["Content-Type"] == "image/jpeg"
    assert "max-age" in headers["Cache-Control"]


@pytest.mark.asyncio
async def test_captures_image_upstream_404_forwarded_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    status, body, _ = await tier_a.captures_image(
        VISION_MJPEG, "missing.jpg", client=_client(handler)
    )
    assert status == 404
    assert body == b""


@pytest.mark.asyncio
async def test_captures_image_connection_error_becomes_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    status, body, _ = await tier_a.captures_image(
        VISION_MJPEG, "x.jpg", client=_client(handler)
    )
    assert status == 502
    assert b"down" in body


# ── WS integration: restart_status streams through the DashboardHub ──────
@pytest.mark.asyncio
async def test_restart_status_streams_through_dashboard_hub():
    """main.py wraps the hub's sync publish() as the async broadcast that
    restart_services drives. Verify that adapter delivers every restart_status
    message to a subscribed dashboard queue, in order."""
    from reachy_voice.dashboard import DashboardHub

    hub = DashboardHub()
    q = hub.subscribe()  # called from this loop, as the WS handler does

    async def broadcast(msg: dict) -> None:  # the adapter main.py builds
        hub.publish(msg)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(204)
        if request.url.path.endswith("/json"):
            return httpx.Response(200, json={"State": {"Health": {"Status": "healthy"}}})
        return httpx.Response(404)

    await tier_a.restart_services(
        "http://127.0.0.1:8630/stream",
        broadcast,
        client=_client(handler),
        sleep=lambda _: _noop(),
        now=lambda: 0.0,
    )

    # publish() is call_soon_threadsafe — let the loop drain the queue puts
    import asyncio

    await asyncio.sleep(0)
    drained: list[dict] = []
    while not q.empty():
        drained.append(q.get_nowait())

    assert drained[0] == {"type": "restart_status", "status": "starting"}
    assert drained[-1] == {"type": "restart_status", "status": "done"}
    assert all(m["type"] == "restart_status" for m in drained)
