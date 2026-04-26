"""Tests for SQLite storage layer."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from reachy_claw.storage import db as db_mod
from reachy_claw.storage.migrations import CURRENT_VERSION, get_version


def _epoch(dt_str: str) -> int:
    return int(datetime.fromisoformat(dt_str).timestamp())


@pytest.fixture
def tmp_db(tmp_path: Path):
    path = tmp_path / "test.db"
    database = db_mod.Database(path)
    database.init()
    yield database
    database.close()


def test_init_creates_schema(tmp_path):
    path = tmp_path / "fresh.db"
    database = db_mod.Database(path)
    database.init()
    with sqlite3.connect(path) as raw:
        assert get_version(raw) == CURRENT_VERSION
        tables = {
            row[0]
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    database.close()
    assert {
        "asr_events",
        "emotions",
        "faces",
        "thoughts",
        "sensors",
        "diaries",
    }.issubset(tables)


def test_record_asr_inserts_row(tmp_db):
    ts = int(time.time())
    tmp_db.record_asr(ts=ts, role="user", text="hello", emotion="happy")
    rows = list(tmp_db.conn.execute("SELECT ts, role, text, emotion FROM asr_events"))
    assert rows == [(ts, "user", "hello", "happy")]


def test_record_emotion_inserts_row(tmp_db):
    ts = int(time.time())
    tmp_db.record_emotion(ts=ts, value=0.7, label="curious")
    rows = list(tmp_db.conn.execute("SELECT ts, value, label FROM emotions"))
    assert rows == [(ts, 0.7, "curious")]


def test_record_face_inserts_row(tmp_db):
    ts = int(time.time())
    tmp_db.record_face(ts=ts, count=3, smile_count=1, capture_path="captures/x.jpg")
    rows = list(
        tmp_db.conn.execute(
            "SELECT ts, count, smile_count, capture_path FROM faces"
        )
    )
    assert rows == [(ts, 3, 1, "captures/x.jpg")]


def test_record_thought_inserts_row(tmp_db):
    ts = int(time.time())
    tmp_db.record_thought(ts=ts, text="I wonder...", emotion="contemplative")
    rows = list(tmp_db.conn.execute("SELECT ts, text, emotion FROM thoughts"))
    assert rows == [(ts, "I wonder...", "contemplative")]


def test_record_sensor_numeric(tmp_db):
    ts = int(time.time())
    tmp_db.record_sensor(ts=ts, source="ha", key="weather.temp_c", value_num=24.5)
    rows = list(
        tmp_db.conn.execute(
            "SELECT ts, source, key, value_num, value_text FROM sensors"
        )
    )
    assert rows == [(ts, "ha", "weather.temp_c", 24.5, None)]


def test_record_sensor_text(tmp_db):
    ts = int(time.time())
    tmp_db.record_sensor(
        ts=ts, source="ha", key="weather.condition", value_text="sunny"
    )
    rows = list(
        tmp_db.conn.execute(
            "SELECT source, key, value_num, value_text FROM sensors"
        )
    )
    assert rows == [("ha", "weather.condition", None, "sunny")]


def test_events_for_day_filters_by_local_window(tmp_db):
    in_day = _epoch("2026-04-26T10:00:00")
    out_before = _epoch("2026-04-25T23:59:59")
    out_after = _epoch("2026-04-27T00:00:01")

    for ts in (in_day, out_before, out_after):
        tmp_db.record_asr(ts=ts, role="user", text=f"t={ts}", emotion=None)

    bundle = tmp_db.events_for_day("2026-04-26")
    asr_texts = [r["text"] for r in bundle["asr_events"]]
    assert asr_texts == [f"t={in_day}"]