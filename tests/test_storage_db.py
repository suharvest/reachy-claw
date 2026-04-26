"""Tests for SQLite storage layer."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from reachy_claw.storage import db as db_mod
from reachy_claw.storage.migrations import CURRENT_VERSION, get_version


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