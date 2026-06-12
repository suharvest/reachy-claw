"""SQLite-backed persistence for Reachy Claw."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .migrations import migrate


def day_window(date_str: str) -> tuple[int, int]:
    """Return [start, end) unix-epoch seconds for the given local-time YYYY-MM-DD."""
    start = datetime.fromisoformat(date_str)
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


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

    def events_for_day(self, date: str) -> dict[str, list[dict[str, Any]]]:
        start, end = day_window(date)
        out: dict[str, list[dict[str, Any]]] = {}
        for table, cols in (
            ("asr_events", "ts, role, text, emotion"),
            ("emotions", "ts, value, label"),
            ("faces", "ts, count, smile_count, capture_path"),
            ("thoughts", "ts, text, emotion"),
            ("sensors", "ts, source, key, value_num, value_text"),
        ):
            rows = self.conn.execute(
                f"SELECT {cols} FROM {table} WHERE ts >= ? AND ts < ? ORDER BY ts ASC",
                (start, end),
            ).fetchall()
            keys = [c.strip() for c in cols.split(",")]
            out[table] = [dict(zip(keys, row)) for row in rows]
        return out

    def save_diary(
        self,
        *,
        date: str,
        markdown: str,
        llm_model: str,
        prompt_version: str,
    ) -> None:
        now = int(time.time())
        self.conn.execute(
            """
            INSERT INTO diaries (date, markdown, generated_at, llm_model, prompt_version)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
              markdown=excluded.markdown,
              generated_at=excluded.generated_at,
              llm_model=excluded.llm_model,
              prompt_version=excluded.prompt_version,
              published_at=NULL
            """,
            (date, markdown, now, llm_model, prompt_version),
        )

    def get_diary(self, date: str) -> dict | None:
        row = self.conn.execute(
            "SELECT date, markdown, generated_at, llm_model, prompt_version, published_at "
            "FROM diaries WHERE date = ?",
            (date,),
        ).fetchone()
        if row is None:
            return None
        keys = [
            "date",
            "markdown",
            "generated_at",
            "llm_model",
            "prompt_version",
            "published_at",
        ]
        return dict(zip(keys, row))

    def mark_published(self, date: str) -> None:
        self.conn.execute(
            "UPDATE diaries SET published_at = ? WHERE date = ?",
            (int(time.time()), date),
        )

    def list_diary_dates(self) -> list[str]:
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT date FROM diaries ORDER BY date DESC"
            )
        ]


def open_default() -> Database:
    db = Database(DEFAULT_DB_PATH)
    db.init()
    return db