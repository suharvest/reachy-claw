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


def open_default() -> Database:
    db = Database(DEFAULT_DB_PATH)
    db.init()
    return db