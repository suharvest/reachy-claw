"""Tests for POST /api/diary/generate, /api/diary/publish, GET /api/diary/status."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from reachy_claw.config import Config
from reachy_claw.storage.db import Database


@pytest.fixture
def app_with_diary_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from reachy_claw.plugins.dashboard_plugin import _build_diary_trigger_handlers

    db = Database(tmp_path / "reachy.db")
    db.init()

    class _StubApp:
        pass

    stub = _StubApp()
    stub.config = Config()
    stub.db = db
    # tracks emitted ws messages for assertion
    stub.ws_emissions = []

    async def fake_broadcast(msg):
        stub.ws_emissions.append(msg)

    aio = web.Application()
    handlers = _build_diary_trigger_handlers(stub, broadcast=fake_broadcast)
    aio.router.add_get("/api/diary/status", handlers["status"])
    aio.router.add_post("/api/diary/generate", handlers["generate"])
    aio.router.add_post("/api/diary/publish", handlers["publish"])
    return aio, stub, tmp_path


@pytest.mark.asyncio
async def test_status_returns_dates(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    today = datetime.now().strftime("%Y-%m-%d")
    stub.db.save_diary(date=today, markdown="m", llm_model="m", prompt_version="v1")
    client = await aiohttp_client(aio)
    r = await client.get("/api/diary/status")
    assert r.status == 200
    body = await r.json()
    found = next((d for d in body["dates"] if d["date"] == today), None)
    assert found is not None
    assert found["generated"] is True
    assert found["published"] is False


@pytest.mark.asyncio
async def test_generate_already_done_returns_200_skip(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    stub.db.save_diary(
        date="2026-04-26", markdown="m", llm_model="m", prompt_version="v1"
    )
    client = await aiohttp_client(aio)
    r = await client.post(
        "/api/diary/generate", json={"date": "2026-04-26", "force": False}
    )
    assert r.status == 200
    body = await r.json()
    assert body["status"] == "already-generated"


@pytest.mark.asyncio
async def test_generate_kicks_off_subprocess_and_emits_ws(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    client = await aiohttp_client(aio)
    with patch(
        "reachy_claw.plugins.dashboard_plugin.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ):
        r = await client.post(
            "/api/diary/generate", json={"date": "2026-04-26", "force": True}
        )
        assert r.status == 202
        body = await r.json()
        assert "job_id" in body
        # let the background task complete
        await asyncio.sleep(0.05)

    phases = [m["phase"] for m in stub.ws_emissions if m.get("type") == "diary_job"]
    assert "generating" in phases
    assert "done" in phases or "error" in phases


@pytest.mark.asyncio
async def test_concurrent_generate_returns_409(app_with_diary_api, aiohttp_client):
    """Second generate for the same date while one is already in flight → 409."""
    aio, stub, _ = app_with_diary_api
    client = await aiohttp_client(aio)

    # Make the subprocess hang so the first call holds the per-date lock.
    hang = asyncio.Event()
    fake_proc = MagicMock()
    fake_proc.returncode = 0

    async def slow_communicate():
        await hang.wait()
        return (b"ok", b"")

    fake_proc.communicate = slow_communicate

    with patch(
        "reachy_claw.plugins.dashboard_plugin.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ):
        r1 = await client.post(
            "/api/diary/generate", json={"date": "2026-04-26", "force": True}
        )
        assert r1.status == 202
        # Yield so the background task acquires the lock.
        await asyncio.sleep(0.01)

        r2 = await client.post(
            "/api/diary/generate", json={"date": "2026-04-26", "force": True}
        )
        assert r2.status == 409
        body2 = await r2.json()
        assert body2["status"] == "in-progress"

        hang.set()
        await asyncio.sleep(0.01)
