"""SQLite-backed persistence for Reachy Claw."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .migrations import migrate


def _default_path() -> Path:
    data_dir = os.environ.get("DATA_DIR")
    base = Path(data_dir) if data_dir else Path.home() / ".reachy-claw"
    return base / "reachy.db"


DEFAULT_DB_PATH = _default_path()


class Database:
    """Thin wrapper around a sqlite3 connection with WAL + migrations."""

    def __init__(self, path: Path | str = DEFAULT_DB_PATH):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        migrate(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized; call init() first")
        return self._conn

    def record_asr(
        self, *, ts: int, role: str, text: str, emotion: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO asr_events (ts, role, text, emotion) VALUES (?, ?, ?, ?)",
            (ts, role, text, emotion),
        )

    def record_emotion(
        self, *, ts: int, value: float | None = None, label: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO emotions (ts, value, label) VALUES (?, ?, ?)",
            (ts, value, label),
        )

    def record_face(
        self,
        *,
        ts: int,
        count: int,
        smile_count: int = 0,
        capture_path: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO faces (ts, count, smile_count, capture_path) "
            "VALUES (?, ?, ?, ?)",
            (ts, count, smile_count, capture_path),
        )

    def record_thought(
        self, *, ts: int, text: str, emotion: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO thoughts (ts, text, emotion) VALUES (?, ?, ?)",
            (ts, text, emotion),
        )

    def record_sensor(
        self,
        *,
        ts: int,
        source: str,
        key: str,
        value_num: float | None = None,
        value_text: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO sensors (ts, source, key, value_num, value_text) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, source, key, value_num, value_text),
        )


def open_default() -> Database:
    db = Database(DEFAULT_DB_PATH)
    db.init()
    return db