"""Integration tests for DailyLogPlugin SQLite writes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from reachy_claw.event_bus import EventBus
from reachy_claw.plugins.daily_log_plugin import DailyLogPlugin
from reachy_claw.storage.db import Database


class _StubApp:
    def __init__(self, db: Database):
        self.events = EventBus()
        self.db = db


@pytest.mark.asyncio
async def test_asr_event_writes_to_sqlite(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.init()
    app = _StubApp(db)
    plugin = DailyLogPlugin(app)
    assert plugin.setup() is True
    plugin._running = True  # Must set before starting
    task = asyncio.create_task(plugin.start())

    # Wait for start() to subscribe to events
    await asyncio.sleep(0.1)

    app.events.emit("asr_final", {"text": "hello there"})
    app.events.emit(
        "llm_end", {"full_text": "hi human", "emotion": "happy"}
    )

    # Allow the writer loop to process queue items.
    await asyncio.sleep(0.3)
    plugin._running = False
    await plugin.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    rows = list(
        db.conn.execute(
            "SELECT role, text FROM asr_events ORDER BY id"
        )
    )
    assert ("user", "hello there") in rows
    assert ("reachy", "hi human") in rows
    db.close()


@pytest.mark.asyncio
async def test_emotion_event_writes_to_sqlite(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.init()
    app = _StubApp(db)
    plugin = DailyLogPlugin(app)
    assert plugin.setup() is True
    plugin._running = True
    task = asyncio.create_task(plugin.start())

    # Wait for start() to subscribe to events
    await asyncio.sleep(0.1)

    app.events.emit("emotion", {"emotion": "curious"})

    # Allow the writer loop to process queue items.
    await asyncio.sleep(0.3)
    plugin._running = False
    await plugin.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    rows = list(
        db.conn.execute("SELECT label FROM emotions")
    )
    assert ("curious",) in rows
    db.close()


@pytest.mark.asyncio
async def test_observation_writes_thought(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.init()
    app = _StubApp(db)
    plugin = DailyLogPlugin(app)
    assert plugin.setup() is True
    plugin._running = True
    task = asyncio.create_task(plugin.start())

    # Wait for start() to subscribe to events
    await asyncio.sleep(0.1)

    app.events.emit("observation", {"text": "I wonder about the world", "emotion": "contemplative"})

    await asyncio.sleep(0.3)
    plugin._running = False
    await plugin.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    rows = list(
        db.conn.execute("SELECT text, emotion FROM thoughts")
    )
    assert ("I wonder about the world", "contemplative") in rows
    db.close()


@pytest.mark.asyncio
async def test_vision_faces_writes_face_count(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.init()
    app = _StubApp(db)
    plugin = DailyLogPlugin(app)
    assert plugin.setup() is True
    plugin._running = True
    task = asyncio.create_task(plugin.start())

    # Wait for start() to subscribe to events
    await asyncio.sleep(0.1)

    app.events.emit("vision_faces", {"faces": [{"bbox": [0, 0, 100, 100]}, {"bbox": [0, 0, 50, 50]}]})

    await asyncio.sleep(0.3)
    plugin._running = False
    await plugin.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    rows = list(
        db.conn.execute("SELECT count FROM faces")
    )
    # Face count should be 2
    assert any(r[0] == 2 for r in rows)
    db.close()


@pytest.mark.asyncio
async def test_smile_capture_writes_to_faces(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.init()
    app = _StubApp(db)
    plugin = DailyLogPlugin(app)
    assert plugin.setup() is True
    plugin._running = True
    task = asyncio.create_task(plugin.start())

    # Wait for start() to subscribe to events
    await asyncio.sleep(0.1)

    # Actual emission uses "file", not "path" - handle both
    app.events.emit("smile_capture", {"count": 1, "file": "captures/test.jpg"})
    app.events.emit("smile_capture", {"count": 2, "path": "captures/other.jpg"})

    await asyncio.sleep(0.3)
    plugin._running = False
    await plugin.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    rows = list(
        db.conn.execute("SELECT smile_count, capture_path FROM faces WHERE smile_count > 0")
    )
    assert len(rows) >= 2
    db.close()