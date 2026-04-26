# Diary Archive & Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace jsonl daily logs with SQLite, generate Markdown diaries, and publish them to an existing Astro site repo via deploy key + Cloudflare Workers.

**Architecture:** A new `storage/db.py` provides SQLite read/write APIs. `DailyLogPlugin` and other event-producing plugins call `db.record_*` instead of writing jsonl. The diary OpenClaw skill queries SQLite, has the LLM emit Markdown with Astro front matter (docs collection schema), stores it in the `diaries` table, and `scripts/publish_diary.py` clones the site repo, drops the file in, and pushes via deploy key. Cloudflare Workers on the site side handles Astro build + deploy.

**Tech Stack:** Python 3.10+, sqlite3 (stdlib, WAL mode), pyyaml, pytest, git CLI, ssh deploy key, Astro (in site repo, out of tree).

**Spec:** `docs/superpowers/specs/2026-04-26-diary-archive-and-publish-design.md`

---

## File Structure

### New files (in this repo)

| Path | Responsibility |
|------|---------------|
| `src/reachy_claw/storage/__init__.py` | package marker, re-export public API |
| `src/reachy_claw/storage/db.py` | sqlite connection, schema init, read/write helpers |
| `src/reachy_claw/storage/migrations.py` | schema versioning via `PRAGMA user_version` |
| `scripts/migrate_jsonl_to_sqlite.py` | one-time import of existing jsonl/JSON |
| `scripts/publish_diary.py` | push generated Markdown to Astro site repo |
| `tests/test_storage_db.py` | unit tests for db.py |
| `tests/test_storage_migrations.py` | unit tests for schema versioning |
| `tests/test_migrate_jsonl.py` | unit tests for migration script |
| `tests/test_publish_diary.py` | integration test against local bare git repo |
| `docs/ops/diary-publish-setup.md` | deploy key + SSH config first-run instructions |

### Modified files

| Path | Change |
|------|--------|
| `src/reachy_claw/plugins/daily_log_plugin.py` | write to SQLite instead of jsonl |
| `src/reachy_claw/plugins/dashboard_plugin.py` | diary endpoints query SQLite |
| `src/reachy_claw/plugins/conversation_plugin.py` | optional: tighter ASR row recording (event remains primary path) |
| `src/reachy_claw/plugins/face_tracker_plugin.py` | record `smile_count` and `capture_path` to faces table |
| `scripts/collect_daily_data.py` | read SQLite (replacing jsonl) |
| `scripts/generate_diary.py` | emit Markdown w/ Astro docs schema front matter, save to `diaries` table |
| `pyproject.toml` | add `pyyaml` (already present, verify) |

### Out of tree (handled in the site repo, not this plan)

- Astro `docs` collection / new category Reachy 日记 — DONE in chaihuo-mcv-site branch feat/reachy-diary-section
- Cloudflare Workers deployment configuration (existing wrangler.jsonc)

---

## Task 1: Storage scaffolding + schema

**Files:**
- Create: `src/reachy_claw/storage/__init__.py`
- Create: `src/reachy_claw/storage/db.py`
- Create: `src/reachy_claw/storage/migrations.py`
- Create: `tests/test_storage_db.py`

- [ ] **Step 1.1: Create empty package**

```python
# src/reachy_claw/storage/__init__.py
"""SQLite-backed persistence for daily interaction data and diaries."""

from .db import (
    Database,
    open_default,
    DEFAULT_DB_PATH,
)

__all__ = ["Database", "open_default", "DEFAULT_DB_PATH"]
```

- [ ] **Step 1.2: Write schema migration module**

```python
# src/reachy_claw/storage/migrations.py
"""Schema versioning for the SQLite database.

Each migration is a SQL script that brings the DB from version N to N+1.
Schema version is stored via SQLite's PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3

CURRENT_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE asr_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        emotion TEXT
    );
    CREATE INDEX idx_asr_events_ts ON asr_events(ts);

    CREATE TABLE emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        value REAL,
        label TEXT
    );
    CREATE INDEX idx_emotions_ts ON emotions(ts);

    CREATE TABLE faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        count INTEGER NOT NULL,
        smile_count INTEGER DEFAULT 0,
        capture_path TEXT
    );
    CREATE INDEX idx_faces_ts ON faces(ts);

    CREATE TABLE thoughts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        text TEXT NOT NULL,
        emotion TEXT
    );
    CREATE INDEX idx_thoughts_ts ON thoughts(ts);

    CREATE TABLE sensors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        source TEXT NOT NULL,
        key TEXT NOT NULL,
        value_num REAL,
        value_text TEXT
    );
    CREATE INDEX idx_sensors_ts ON sensors(ts);
    CREATE INDEX idx_sensors_key_ts ON sensors(key, ts);

    CREATE TABLE diaries (
        date TEXT PRIMARY KEY,
        markdown TEXT NOT NULL,
        generated_at INTEGER NOT NULL,
        llm_model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        published_at INTEGER
    );
    """,
}


def get_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def set_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA cannot use parameterized values; version is int so safe to format.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def migrate(conn: sqlite3.Connection) -> None:
    """Run all pending migrations to bring DB to CURRENT_VERSION."""
    current = get_version(conn)
    for v in range(current + 1, CURRENT_VERSION + 1):
        sql = MIGRATIONS[v]
        conn.executescript(sql)
        set_version(conn, v)
    conn.commit()
```

- [ ] **Step 1.3: Write the failing test for schema init**

```python
# tests/test_storage_db.py
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
```

- [ ] **Step 1.4: Run the test, watch it fail**

```
uv run pytest tests/test_storage_db.py::test_init_creates_schema -v
```

Expected: FAIL — `Database` not implemented in `db.py` yet.

- [ ] **Step 1.5: Implement `db.py` minimally to pass init test**

```python
# src/reachy_claw/storage/db.py
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
```

- [ ] **Step 1.6: Run test again, expect PASS**

```
uv run pytest tests/test_storage_db.py::test_init_creates_schema -v
```

Expected: PASS.

- [ ] **Step 1.7: Commit**

```bash
git add src/reachy_claw/storage/ tests/test_storage_db.py
git commit -m "feat(storage): add SQLite database scaffolding + schema v1"
```

---

## Task 2: Write helpers for events

**Files:**
- Modify: `src/reachy_claw/storage/db.py`
- Modify: `tests/test_storage_db.py`

- [ ] **Step 2.1: Write failing tests for record_* helpers**

Append to `tests/test_storage_db.py`:

```python
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
```

- [ ] **Step 2.2: Run; expect failures (no methods)**

```
uv run pytest tests/test_storage_db.py -v
```

Expected: 5 failures with `AttributeError: 'Database' object has no attribute 'record_asr'`.

- [ ] **Step 2.3: Add record_* methods to Database**

In `src/reachy_claw/storage/db.py`, add inside class `Database`:

```python
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
```

- [ ] **Step 2.4: Run tests, expect PASS**

```
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 2.5: Commit**

```bash
git add src/reachy_claw/storage/db.py tests/test_storage_db.py
git commit -m "feat(storage): add record_* write helpers for event tables"
```

---

## Task 3: Day-window query helpers

**Files:**
- Modify: `src/reachy_claw/storage/db.py`
- Modify: `tests/test_storage_db.py`

- [ ] **Step 3.1: Write failing test for `events_for_day`**

Append to `tests/test_storage_db.py`:

```python
from datetime import datetime


def _epoch(dt_str: str) -> int:
    return int(datetime.fromisoformat(dt_str).timestamp())


def test_events_for_day_filters_by_local_window(tmp_db):
    in_day = _epoch("2026-04-26T10:00:00")
    out_before = _epoch("2026-04-25T23:59:59")
    out_after = _epoch("2026-04-27T00:00:01")

    for ts in (in_day, out_before, out_after):
        tmp_db.record_asr(ts=ts, role="user", text=f"t={ts}", emotion=None)

    bundle = tmp_db.events_for_day("2026-04-26")
    asr_texts = [r["text"] for r in bundle["asr_events"]]
    assert asr_texts == [f"t={in_day}"]
```

- [ ] **Step 3.2: Run; expect failure**

```
uv run pytest tests/test_storage_db.py::test_events_for_day_filters_by_local_window -v
```

Expected: `AttributeError: events_for_day`.

- [ ] **Step 3.3: Implement `events_for_day` and `day_window`**

Add to `src/reachy_claw/storage/db.py`:

```python
from datetime import datetime, timedelta
from typing import Any


def day_window(date_str: str) -> tuple[int, int]:
    """Return [start, end) unix-epoch seconds for the given local-time YYYY-MM-DD."""
    start = datetime.fromisoformat(date_str)
    end = start + timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())
```

Add to `class Database`:

```python
    def events_for_day(self, date: str) -> dict[str, list[dict[str, Any]]]:
        start, end = day_window(date)
        out: dict[str, list[dict[str, Any]]] = {}
        for table, cols in (
            ("asr_events", "ts, role, text, emotion"),
            ("emotions", "ts, value, label"),
            ("faces", "ts, count, smile_count, capture_path"),
            ("thoughts", "ts, text, emotion"),
            ("weather", "ts, temp_c, humidity, condition, location"),
        ):
            rows = self.conn.execute(
                f"SELECT {cols} FROM {table} WHERE ts >= ? AND ts < ? ORDER BY ts ASC",
                (start, end),
            ).fetchall()
            keys = [c.strip() for c in cols.split(",")]
            out[table] = [dict(zip(keys, row)) for row in rows]
        return out
```

- [ ] **Step 3.4: Run; expect PASS**

```
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add src/reachy_claw/storage/db.py tests/test_storage_db.py
git commit -m "feat(storage): add events_for_day local-window query helper"
```

---

## Task 4: Diary CRUD on `diaries` table

**Files:**
- Modify: `src/reachy_claw/storage/db.py`
- Modify: `tests/test_storage_db.py`

- [ ] **Step 4.1: Write failing tests for diary CRUD**

Append to `tests/test_storage_db.py`:

```python
def test_save_and_get_diary(tmp_db):
    md = "---\ntitle: hi\n---\n\n# hi"
    tmp_db.save_diary(
        date="2026-04-26",
        markdown=md,
        llm_model="dashscope/kimi-k2.5",
        prompt_version="v1",
    )
    got = tmp_db.get_diary("2026-04-26")
    assert got is not None
    assert got["markdown"] == md
    assert got["llm_model"] == "dashscope/kimi-k2.5"
    assert got["published_at"] is None


def test_save_diary_replaces_existing(tmp_db):
    tmp_db.save_diary(
        date="2026-04-26",
        markdown="v1",
        llm_model="m",
        prompt_version="p",
    )
    tmp_db.save_diary(
        date="2026-04-26",
        markdown="v2",
        llm_model="m",
        prompt_version="p",
    )
    got = tmp_db.get_diary("2026-04-26")
    assert got["markdown"] == "v2"


def test_mark_published_sets_timestamp(tmp_db):
    tmp_db.save_diary(
        date="2026-04-26",
        markdown="m",
        llm_model="m",
        prompt_version="p",
    )
    tmp_db.mark_published("2026-04-26")
    got = tmp_db.get_diary("2026-04-26")
    assert got["published_at"] is not None
    assert isinstance(got["published_at"], int)


def test_list_diary_dates_returns_descending(tmp_db):
    for d in ("2026-04-24", "2026-04-26", "2026-04-25"):
        tmp_db.save_diary(date=d, markdown="m", llm_model="m", prompt_version="p")
    assert tmp_db.list_diary_dates() == ["2026-04-26", "2026-04-25", "2026-04-24"]
```

- [ ] **Step 4.2: Run; expect failures**

```
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 4.3: Add diary CRUD methods**

Add to `class Database`:

```python
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
```

- [ ] **Step 4.4: Run; expect PASS**

```
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 4.5: Commit**

```bash
git add src/reachy_claw/storage/db.py tests/test_storage_db.py
git commit -m "feat(storage): add diary CRUD (save/get/mark_published/list)"
```

---

## Task 5: Migrate DailyLogPlugin to SQLite

**Files:**
- Modify: `src/reachy_claw/plugins/daily_log_plugin.py`
- Modify: `tests/conftest.py` (if a fixture is shared; otherwise local fixture)
- Create: existing `tests/` may need an additional integration test, but reuse the existing one if present.

- [ ] **Step 5.1: Read existing test for DailyLogPlugin**

```bash
ls tests/ | grep -i daily_log
```

If absent, write a new test file `tests/test_daily_log_plugin.py`. If present, extend it.

- [ ] **Step 5.2: Write failing test asserting events land in SQLite**

Create `tests/test_daily_log_plugin.py` (or replace existing):

```python
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
    plugin.setup()
    task = asyncio.create_task(plugin.start())

    app.events.publish("asr_final", {"text": "hello there"})
    app.events.publish(
        "llm_end", {"full_text": "hi human", "emotion": "happy"}
    )

    # Allow the writer loop one tick.
    await asyncio.sleep(0.2)
    plugin._running = False
    await plugin.stop()
    task.cancel()

    rows = list(
        db.conn.execute(
            "SELECT role, text FROM asr_events ORDER BY id"
        )
    )
    assert ("user", "hello there") in rows
    assert ("reachy", "hi human") in rows
    db.close()
```

- [ ] **Step 5.3: Run; expect failure**

```
uv run pytest tests/test_daily_log_plugin.py -v
```

Expected: error on `app.db` (not yet provided) or assertion fail (still writes jsonl).

- [ ] **Step 5.4: Add `db` field to ClawdApp / reachy_app**

Open `src/reachy_claw/app.py` (or `reachy_app.py`) and:
- Add `from .storage.db import open_default` near top.
- In `__init__` or `setup`, set `self.db = open_default()`.
- In shutdown, call `self.db.close()`.

If app construction differs, consult `src/reachy_claw/main.py` to find the right hook. Implementer note: place this where other shared resources (event bus) are constructed.

- [ ] **Step 5.5: Rewrite `DailyLogPlugin` to use SQLite**

Replace the body of `src/reachy_claw/plugins/daily_log_plugin.py` with:

```python
"""DailyLogPlugin — writes daily interaction events to SQLite.

Subscribes to the EventBus and persists timestamped rows into the shared
`Database` (app.db). Replaces the prior jsonl-based logging.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..plugin import Plugin
from ..storage.db import Database

logger = logging.getLogger(__name__)

EMOTION_SAMPLE_INTERVAL = 60
FACE_SAMPLE_INTERVAL = 60


class DailyLogPlugin(Plugin):
    name = "daily_log"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._db: Database = app.db
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._last_emotion: str | None = None
        self._last_emotion_ts: float = 0
        self._last_face_ts: float = 0
        self._pending_asr: dict | None = None

    def setup(self) -> bool:
        return True

    async def start(self) -> None:
        bus = self.app.events
        bus.subscribe("emotion", self._on_emotion)
        bus.subscribe("asr_final", self._on_asr_final)
        bus.subscribe("llm_end", self._on_llm_end)
        bus.subscribe("vision_faces", self._on_vision_faces)
        bus.subscribe("smile_capture", self._on_smile_capture)
        bus.subscribe("observation", self._on_observation)

        while self._running:
            try:
                kind, entry = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                self._write(kind, entry)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("DailyLog writer error: %s", e)

    async def stop(self) -> None:
        await super().stop()
        bus = self.app.events
        bus.unsubscribe("emotion", self._on_emotion)
        bus.unsubscribe("asr_final", self._on_asr_final)
        bus.unsubscribe("llm_end", self._on_llm_end)
        bus.unsubscribe("vision_faces", self._on_vision_faces)
        bus.unsubscribe("smile_capture", self._on_smile_capture)
        bus.unsubscribe("observation", self._on_observation)
        while not self._queue.empty():
            try:
                kind, entry = self._queue.get_nowait()
                self._write(kind, entry)
            except Exception:
                break

    # ── handlers ─────────────────────────────────────────────────────────
    def _on_emotion(self, data: Any) -> None:
        now = time.time()
        emotion = data if isinstance(data, str) else str(data)
        if emotion == self._last_emotion and (now - self._last_emotion_ts) < EMOTION_SAMPLE_INTERVAL:
            return
        self._last_emotion = emotion
        self._last_emotion_ts = now
        self._queue.put_nowait(("emotion", {"label": emotion}))

    def _on_asr_final(self, data: Any) -> None:
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        if not text.strip():
            return
        self._pending_asr = {"text": text}
        self._queue.put_nowait(("asr_user", {"text": text}))

    def _on_llm_end(self, data: Any) -> None:
        if isinstance(data, dict):
            reply = data.get("full_text", data.get("text", ""))
            emotion = data.get("emotion") or None
        else:
            reply = str(data)
            emotion = None
        if not reply.strip():
            return
        self._queue.put_nowait(
            ("asr_reachy", {"text": reply, "emotion": emotion})
        )
        self._pending_asr = None

    def _on_vision_faces(self, data: Any) -> None:
        now = time.time()
        if (now - self._last_face_ts) < FACE_SAMPLE_INTERVAL:
            return
        self._last_face_ts = now
        if isinstance(data, dict):
            faces = data.get("faces", [])
            count = len(faces)
        elif isinstance(data, list):
            count = len(data)
        else:
            count = 0
        self._queue.put_nowait(("face", {"count": count, "smile_count": 0, "capture_path": None}))

    def _on_smile_capture(self, data: Any) -> None:
        path = data.get("path") if isinstance(data, dict) else None
        self._queue.put_nowait(
            ("face", {"count": 1, "smile_count": 1, "capture_path": path})
        )

    def _on_observation(self, data: Any) -> None:
        if isinstance(data, dict):
            text = data.get("text", data.get("observation", ""))
            emotion = data.get("emotion") or None
        else:
            text = str(data)
            emotion = None
        if not text.strip():
            return
        self._queue.put_nowait(("thought", {"text": text, "emotion": emotion}))

    # ── writer ───────────────────────────────────────────────────────────
    def _write(self, kind: str, entry: dict) -> None:
        ts = int(time.time())
        try:
            if kind == "emotion":
                self._db.record_emotion(ts=ts, label=entry["label"])
            elif kind == "asr_user":
                self._db.record_asr(
                    ts=ts, role="user", text=entry["text"], emotion=None
                )
            elif kind == "asr_reachy":
                self._db.record_asr(
                    ts=ts,
                    role="reachy",
                    text=entry["text"],
                    emotion=entry.get("emotion"),
                )
            elif kind == "face":
                self._db.record_face(
                    ts=ts,
                    count=entry["count"],
                    smile_count=entry.get("smile_count", 0),
                    capture_path=entry.get("capture_path"),
                )
            elif kind == "thought":
                self._db.record_thought(
                    ts=ts, text=entry["text"], emotion=entry.get("emotion")
                )
        except Exception as e:
            logger.warning("DailyLog write failed for %s: %s", kind, e)
```

- [ ] **Step 5.6: Run the test, expect PASS**

```
uv run pytest tests/test_daily_log_plugin.py -v
```

- [ ] **Step 5.7: Run the full suite to catch regressions**

```
uv run pytest -x
```

Expected: green. If older tests assert jsonl files on disk, mark them as deleted in this same commit (the jsonl path is gone).

- [ ] **Step 5.8: Commit**

```bash
git add src/reachy_claw/plugins/daily_log_plugin.py src/reachy_claw/app.py tests/test_daily_log_plugin.py
git commit -m "feat(daily-log): write events to SQLite via storage.db (replaces jsonl)"
```

---

## Task 6: Migration script (jsonl + diary JSON → SQLite)

**Files:**
- Create: `scripts/migrate_jsonl_to_sqlite.py`
- Create: `tests/test_migrate_jsonl.py`

- [ ] **Step 6.1: Write failing test using fixture jsonl files**

```python
# tests/test_migrate_jsonl.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from reachy_claw.storage.db import Database

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "migrate_jsonl_to_sqlite.py"


def test_migrate_imports_jsonl_and_diary_json(tmp_path: Path):
    legacy = tmp_path / "data"
    day = legacy / "daily-logs" / "2026-04-25"
    day.mkdir(parents=True)
    (day / "conversations.jsonl").write_text(
        json.dumps({"ts": "2026-04-25T10:00:00", "user": "hi", "reply": "hello", "emotion": "happy"})
        + "\n",
        encoding="utf-8",
    )
    diaries = legacy / "diaries"
    diaries.mkdir(parents=True)
    (diaries / "2026-04-25.json").write_text(
        json.dumps({"date": "2026-04-25", "title": "Day", "sections": []}),
        encoding="utf-8",
    )

    db_path = tmp_path / "out.db"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(legacy),
            "--db",
            str(db_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    db = Database(db_path)
    db.init()
    asr_rows = list(db.conn.execute("SELECT role, text FROM asr_events"))
    assert ("user", "hi") in asr_rows
    assert ("reachy", "hello") in asr_rows
    diary = db.get_diary("2026-04-25")
    assert diary is not None
    assert "Day" in diary["markdown"]
    db.close()
```

- [ ] **Step 6.2: Run; expect failure (script does not exist)**

```
uv run pytest tests/test_migrate_jsonl.py -v
```

- [ ] **Step 6.3: Implement the migration script**

```python
# scripts/migrate_jsonl_to_sqlite.py
#!/usr/bin/env python3
"""Migrate legacy jsonl daily logs and JSON diaries into the SQLite DB.

Usage:
    uv run python scripts/migrate_jsonl_to_sqlite.py [--source DIR] [--db PATH]

Defaults: source = ~/.reachy-claw/, db = $DATA_DIR/reachy.db (or ~/.reachy-claw/reachy.db).
Idempotent: re-running on the same source is safe (uses INSERT OR IGNORE for events;
diaries upsert on date).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running without install: add repo src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reachy_claw.storage.db import Database  # noqa: E402


def _iso_to_epoch(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except Exception:
        return 0


def _migrate_conversations(db: Database, path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = _iso_to_epoch(row.get("ts", ""))
        if "user" in row:
            db.record_asr(ts=ts, role="user", text=row["user"], emotion=None)
        if "reply" in row:
            db.record_asr(
                ts=ts,
                role="reachy",
                text=row["reply"],
                emotion=row.get("emotion"),
            )


def _migrate_emotions(db: Database, path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        db.record_emotion(
            ts=_iso_to_epoch(row.get("ts", "")), label=row.get("emotion")
        )


def _migrate_faces(db: Database, path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        db.record_face(
            ts=_iso_to_epoch(row.get("ts", "")),
            count=int(row.get("count", 0)),
            smile_count=int(row.get("total", 0)) if row.get("event") == "smile_capture" else 0,
            capture_path=row.get("path"),
        )


def _migrate_thoughts(db: Database, path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        db.record_thought(
            ts=_iso_to_epoch(row.get("ts", "")),
            text=row.get("text", ""),
            emotion=row.get("emotion"),
        )


def _migrate_diary_jsons(db: Database, source: Path) -> None:
    diary_dir = source / "diaries"
    if not diary_dir.exists():
        return
    for f in sorted(diary_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = doc.get("date") or f.stem
        # Convert old JSON-section diary into a minimal Markdown shell so the
        # row exists; future rendering uses fresh Markdown only.
        title = doc.get("title", date)
        body_parts = [f"# {title}\n"]
        for sec in doc.get("sections", []):
            content = sec.get("content", "")
            if content:
                body_parts.append(f"\n## {sec.get('id', 'section')}\n\n{content}\n")
        markdown = (
            f"---\ntitle: \"{title}\"\ndate: {date}\nlegacy: true\n---\n\n"
            + "".join(body_parts)
        )
        db.save_diary(
            date=date,
            markdown=markdown,
            llm_model=doc.get("meta", {}).get("llm_model", "legacy"),
            prompt_version=doc.get("meta", {}).get("prompt_version", "legacy"),
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default=os.environ.get("DATA_DIR") or str(Path.home() / ".reachy-claw"),
    )
    p.add_argument("--db", default=None)
    args = p.parse_args()

    source = Path(args.source)
    db_path = Path(args.db) if args.db else source / "reachy.db"

    db = Database(db_path)
    db.init()

    logs = source / "daily-logs"
    if logs.exists():
        for day_dir in sorted(p for p in logs.iterdir() if p.is_dir()):
            for jsonl, fn in (
                ("conversations.jsonl", _migrate_conversations),
                ("emotions.jsonl", _migrate_emotions),
                ("faces.jsonl", _migrate_faces),
                ("thoughts.jsonl", _migrate_thoughts),
            ):
                f = day_dir / jsonl
                if f.exists():
                    fn(db, f)

    _migrate_diary_jsons(db, source)
    db.close()
    print(f"Migrated to {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.4: Run; expect PASS**

```
uv run pytest tests/test_migrate_jsonl.py -v
```

- [ ] **Step 6.5: Commit**

```bash
git add scripts/migrate_jsonl_to_sqlite.py tests/test_migrate_jsonl.py
git commit -m "feat(scripts): migrate legacy jsonl + diary JSON into SQLite"
```

---

## Task 7: Update `collect_daily_data.py` to read SQLite

**Files:**
- Modify: `scripts/collect_daily_data.py`
- Create: `tests/test_collect_daily_data.py`

- [ ] **Step 7.1: Read current script and existing test (if any)**

```bash
ls tests/ | grep collect
```

- [ ] **Step 7.2: Write failing test**

Create `tests/test_collect_daily_data.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from reachy_claw.storage.db import Database

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "collect_daily_data.py"


def test_collect_outputs_sqlite_data(tmp_path: Path):
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.init()
    ts = int(time.mktime((2026, 4, 26, 12, 0, 0, 0, 0, -1)))
    db.record_asr(ts=ts, role="user", text="hello", emotion=None)
    db.record_asr(ts=ts + 1, role="reachy", text="hi", emotion="happy")
    db.close()

    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--date", "2026-04-26", "--db", str(db_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["date"] == "2026-04-26"
    asr = out["events"]["asr_events"]
    assert {r["role"] for r in asr} == {"user", "reachy"}
```

- [ ] **Step 7.3: Run; expect failure**

```
uv run pytest tests/test_collect_daily_data.py -v
```

- [ ] **Step 7.4: Replace `scripts/collect_daily_data.py`**

```python
#!/usr/bin/env python3
"""Collect a day's events from SQLite and emit a structured JSON blob.

Usage:
    uv run python scripts/collect_daily_data.py --date 2026-04-26 [--db PATH]

Output (stdout): JSON with shape:
    {
      "date": "2026-04-26",
      "events": {
         "asr_events": [...], "emotions": [...], "faces": [...],
         "thoughts": [...], "weather": [...]
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reachy_claw.storage.db import Database  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--db",
        default=os.environ.get("DATA_DIR")
        and str(Path(os.environ["DATA_DIR"]) / "reachy.db")
        or str(Path.home() / ".reachy-claw" / "reachy.db"),
    )
    args = p.parse_args()

    db = Database(args.db)
    db.init()
    bundle = db.events_for_day(args.date)
    db.close()
    json.dump(
        {"date": args.date, "events": bundle},
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.5: Run; expect PASS**

```
uv run pytest tests/test_collect_daily_data.py -v
```

- [ ] **Step 7.6: Commit**

```bash
git add scripts/collect_daily_data.py tests/test_collect_daily_data.py
git commit -m "feat(scripts): collect_daily_data reads SQLite (replaces jsonl)"
```

---

## Task 8: Diary generation emits Markdown

**Files:**
- Modify: `scripts/generate_diary.py`
- Create: `tests/test_generate_diary.py`

The script wraps the LLM call. The LLM response is mocked in tests; the contract is that we get Markdown with Astro docs schema front matter and store it via `db.save_diary`.

- [ ] **Step 8.1: Read current generator**

```bash
wc -l scripts/generate_diary.py && head -80 scripts/generate_diary.py
```

- [ ] **Step 8.2: Write failing test with a mock LLM**

```python
# tests/test_generate_diary.py
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from reachy_claw.storage.db import Database

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "generate_diary.py"


def test_generate_writes_markdown_to_diaries(tmp_path: Path):
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.init()
    ts = int(time.mktime((2026, 4, 26, 10, 0, 0, 0, 0, -1)))
    db.record_asr(ts=ts, role="user", text="hi", emotion=None)
    db.close()

    env = {**os.environ, "DIARY_LLM_MOCK": "1"}
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--date",
            "2026-04-26",
            "--db",
            str(db_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, res.stderr

    db = Database(db_path)
    db.init()
    diary = db.get_diary("2026-04-26")
    db.close()
    assert diary is not None
    md = diary["markdown"]
    assert md.startswith("---\n")
    assert "title:" in md
    assert "title_en:" in md
    assert "date: \"2026.04.26\"" in md  # Astro format: dots, quoted
    assert "category: \"Reachy 日记\"" in md
    assert "description:" in md
    assert "description_en:" in md
    assert "## 今天的心情" in md
```

- [ ] **Step 8.3: Run; expect failure**

```
uv run pytest tests/test_generate_diary.py -v
```

- [ ] **Step 8.4: Implement `generate_diary.py`**

Replace its contents with:

```python
#!/usr/bin/env python3
"""Generate the daily diary as Markdown with Astro front matter and store it.

Reads events from SQLite for a given date, asks the LLM to compose a first-person
Markdown diary using fixed section headings, and saves the result to the
`diaries` table. A mock mode (DIARY_LLM_MOCK=1) returns a deterministic Markdown
shell — used in tests and dry runs.

Usage:
    uv run python scripts/generate_diary.py --date 2026-04-26 [--db PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reachy_claw.storage.db import Database  # noqa: E402

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "dashscope/kimi-k2.5"

SYSTEM_PROMPT = """You are Reachy Mini, a small humanoid robot. Write today's diary as Markdown with YAML front matter for the Astro docs collection, in a warm reflective first-person tone.

Rules:
- Never quote user speech verbatim. Paraphrase what was said and discussed.
- Never include personal identifiers (names, addresses, phone numbers) from ASR.
- Use exactly these section headings, in this order: "## 今天的心情", "## 遇到的人", "## 想到的事".
- Front matter must follow the Astro docs schema:
  - title: Chinese title, format "Reachy 的日记 · X 月 Y 日"
  - title_en: English title, format "Reachy's Diary · Month Day"
  - date: "YYYY.MM.DD" (dots, not dashes)
  - category: "Reachy 日记"
  - description: Chinese summary (1-2 sentences)
  - description_en: English summary (1-2 sentences)
  - author: "Reachy Mini"
  - author_en: "Reachy Mini"
  - readTime: "X 分钟"
  - readTime_en: "X min read"
  - coverImage: a URL (use first available smile capture URL OR a placeholder Unsplash robot URL)
  - tags: array like ["Reachy 日记", "Reachy", "AI"]
- Body is Chinese only (no English body; English version is via title_en/description_en).
- Output ONLY the Markdown document. No code fences, no commentary.
"""


def _build_user_prompt(date: str, events: dict) -> str:
    return (
        f"Date: {date}\n"
        f"Events as JSON (paraphrase only, never quote):\n{json.dumps(events, ensure_ascii=False)}\n"
    )


def _mock_markdown(date: str, events: dict) -> str:
    """Generate a mock diary Markdown with Astro docs schema front matter."""
    n_asr = len(events.get("asr_events", []))
    n_faces = sum(r.get("count", 0) for r in events.get("faces", []))
    smiles = sum(r.get("smile_count", 0) for r in events.get("faces", []))

    # Parse date for title formatting
    parts = date.split("-")
    year, month, day = parts[0], int(parts[1]), int(parts[2])
    astro_date = f"{year}.{parts[1]}.{parts[2]}"  # YYYY.MM.DD format

    # Month names for English title
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_en = month_names[int(parts[1]) - 1]

    return (
        "---\n"
        f"title: \"Reachy 的日记 · {month} 月 {day} 日\"\n"
        f"title_en: \"Reachy's Diary · {month_en} {day}\"\n"
        f"date: \"{astro_date}\"\n"
        "category: \"Reachy 日记\"\n"
        f"description: \"今天来了 {n_faces} 位朋友，其中 {smiles} 位对我露出了笑容。\"\n"
        f"description_en: \"Today {n_faces} people stopped by, and {smiles} of them smiled at me.\"\n"
        "author: \"Reachy Mini\"\n"
        "author_en: \"Reachy Mini\"\n"
        f"readTime: \"{max(1, n_asr)} 分钟\"\n"
        f"readTime_en: \"{max(1, n_asr)} min read\"\n"
        "coverImage: \"https://images.unsplash.com/photo-1485827404703-89b55fcc595e\"\n"
        "tags: [\"Reachy 日记\", \"Reachy\", \"AI\"]\n"
        "---\n\n"
        "## 今天的心情\n\n今天平静而充实。\n\n"
        "## 遇到的人\n\n来过几位朋友，我用微笑回应了他们。\n\n"
        "## 想到的事\n\n我想了一下世界的样子。\n"
    )


def _call_llm(date: str, events: dict, model: str) -> str:
    """Real LLM call. In production this dispatches to OpenClaw or dashscope.

    Implementation note: the OpenClaw CLI is invoked from the daily-diary skill
    in production. This script supports a direct CLI bridge via the
    DIARY_LLM_CMD env var (a shell command that reads JSON from stdin and writes
    Markdown to stdout). This keeps the script testable and allows different
    backends without code change.
    """
    cmd = os.environ.get("DIARY_LLM_CMD")
    if not cmd:
        raise RuntimeError(
            "No LLM available: set DIARY_LLM_CMD or DIARY_LLM_MOCK=1"
        )
    import subprocess

    payload = json.dumps(
        {"system": SYSTEM_PROMPT, "user": _build_user_prompt(date, events), "model": model}
    )
    res = subprocess.run(
        cmd, input=payload, shell=True, capture_output=True, text=True, check=True
    )
    return res.stdout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument(
        "--db",
        default=os.environ.get("DATA_DIR")
        and str(Path(os.environ["DATA_DIR"]) / "reachy.db")
        or str(Path.home() / ".reachy-claw" / "reachy.db"),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    db = Database(args.db)
    db.init()

    existing = db.get_diary(args.date)
    if existing and existing["published_at"] is not None and not args.force:
        print(f"Already published: {args.date} (use --force to regenerate)")
        db.close()
        return 0

    events = db.events_for_day(args.date)
    if os.environ.get("DIARY_LLM_MOCK") == "1":
        md = _mock_markdown(args.date, events)
        model = "mock"
    else:
        md = _call_llm(args.date, events, args.model)
        model = args.model

    db.save_diary(
        date=args.date,
        markdown=md,
        llm_model=model,
        prompt_version=PROMPT_VERSION,
    )
    db.close()
    print(f"Generated diary for {args.date} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.5: Run; expect PASS**

```
uv run pytest tests/test_generate_diary.py -v
```

- [ ] **Step 8.6: Commit**

```bash
git add scripts/generate_diary.py tests/test_generate_diary.py
git commit -m "feat(diary): generate Markdown w/ front matter into diaries table"
```

---

## Task 9: ASR-quote linter (privacy guard)

**Files:**
- Modify: `scripts/generate_diary.py`
- Modify: `tests/test_generate_diary.py`

A safeguard: after the LLM produces Markdown, scan the body for verbatim user-ASR substrings of length ≥ 20 chars. If found, refuse to save (or log a warning and quarantine). For first iteration, log + abort.

- [ ] **Step 9.1: Add failing test**

Append to `tests/test_generate_diary.py`:

```python
def test_diary_aborts_when_user_asr_quoted_verbatim(tmp_path: Path):
    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.init()
    ts = int(time.mktime((2026, 4, 26, 10, 0, 0, 0, 0, -1)))
    long_phrase = "我今天去了城市里那家很难找到的咖啡馆喝咖啡"  # >20 chars (22 chars)
    db.record_asr(ts=ts, role="user", text=long_phrase, emotion=None)
    db.close()

    # Mock LLM emits a Markdown that includes the user's verbatim phrase.
    # Use a temp shell script for cleaner multi-line handling.
    leak_script = tmp_path / "leak_llm.sh"
    leak_script.write_text(
        f'''cat <<'MARKDOWN'
---
title: "Reachy 的日记 · 4 月 26 日"
title_en: "Reachy's Diary · April 26"
date: "2026.04.26"
category: "Reachy 日记"
description: "泄露测试"
description_en: "Leak test"
author: "Reachy Mini"
author_en: "Reachy Mini"
readTime: "1 分钟"
readTime_en: "1 min read"
coverImage: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e"
tags: ["Reachy 日记", "Reachy", "AI"]
---

## 今天的心情

{long_phrase} 然后我就睡了。

## 遇到的人

人。
## 想到的事

事。
MARKDOWN
''',
        encoding="utf-8",
    )
    leak_script.chmod(0o755)
    env = {**os.environ, "DIARY_LLM_CMD": str(leak_script)}

    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--date", "2026-04-26", "--db", str(db_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "verbatim" in (res.stderr + res.stdout).lower()

    db = Database(db_path)
    db.init()
    assert db.get_diary("2026-04-26") is None
    db.close()
```

- [ ] **Step 9.2: Run; expect failure**

```
uv run pytest tests/test_generate_diary.py::test_diary_aborts_when_user_asr_quoted_verbatim -v
```

- [ ] **Step 9.3: Implement linter in `generate_diary.py`**

Add helper near the top:

```python
MIN_QUOTE_LEN = 20


def _verbatim_asr_quotes(markdown: str, asr_user_texts: list[str]) -> list[str]:
    """Return any user-ASR substrings of length >= MIN_QUOTE_LEN found in markdown."""
    leaks = []
    for text in asr_user_texts:
        text = text.strip()
        if len(text) < MIN_QUOTE_LEN:
            continue
        # Slide a window of MIN_QUOTE_LEN over the user text; if any window
        # appears verbatim in the markdown body, flag the full text.
        for i in range(0, len(text) - MIN_QUOTE_LEN + 1):
            window = text[i : i + MIN_QUOTE_LEN]
            if window in markdown:
                leaks.append(text)
                break
    return leaks
```

In `main()`, after building `md` and before `db.save_diary`, add:

```python
    user_texts = [r["text"] for r in events.get("asr_events", []) if r["role"] == "user"]
    leaks = _verbatim_asr_quotes(md, user_texts)
    if leaks:
        sys.stderr.write(
            "ABORT: diary contains verbatim user ASR quotes: "
            + json.dumps(leaks, ensure_ascii=False)
            + "\n"
        )
        db.close()
        return 2
```

- [ ] **Step 9.4: Run; expect PASS**

```
uv run pytest tests/test_generate_diary.py -v
```

- [ ] **Step 9.5: Commit**

```bash
git add scripts/generate_diary.py tests/test_generate_diary.py
git commit -m "feat(diary): privacy linter aborts on verbatim user-ASR quotes"
```

---

## Task 10: Dashboard endpoints read SQLite

**Files:**
- Modify: `src/reachy_claw/plugins/dashboard_plugin.py`
- Modify: `tests/test_dashboard_plugin.py`

The existing endpoints `GET /api/diaries`, `GET /api/diary/{date}`, `GET /api/diary/latest` continue to return JSON, but the data is derived from the Markdown front matter + body in the `diaries` table.

- [ ] **Step 10.1: Read current dashboard endpoints**

```bash
grep -n "diary\|diaries" src/reachy_claw/plugins/dashboard_plugin.py | head -30
```

Note the current return shape — the front-end (`diary.js`) consumes it and must keep working.

- [ ] **Step 10.2: Write failing tests**

Append to `tests/test_dashboard_plugin.py`:

```python
@pytest.mark.asyncio
async def test_diary_endpoint_returns_sqlite_diary(aiohttp_client, tmp_path):
    # Spin a dashboard with a temp DB containing a diary row.
    from reachy_claw.storage.db import Database

    db = Database(tmp_path / "t.db")
    db.init()
    db.save_diary(
        date="2026-04-26",
        markdown=(
            "---\ntitle: \"Hi\"\ndate: 2026-04-26\nstats: {}\ncaptures: []\n"
            "meta: {llm_model: m, prompt_version: v1}\n---\n\n## 今天的心情\n\nok\n"
        ),
        llm_model="m",
        prompt_version="v1",
    )

    # Use the project's existing test helper to construct a dashboard app
    # bound to this DB. (Implementer: follow the pattern in the file's other
    # async tests.)
    client = await _make_dashboard_client(aiohttp_client, db=db)
    resp = await client.get("/api/diary/2026-04-26")
    assert resp.status == 200
    body = await resp.json()
    assert body["date"] == "2026-04-26"
    assert body["title"] == "Hi"
    assert any("今天的心情" in s.get("heading", "") for s in body["sections"])
```

The existing `test_dashboard_plugin.py` already constructs a dashboard for tests; reuse that helper. If naming differs, replace `_make_dashboard_client` with the actual helper.

- [ ] **Step 10.3: Implement Markdown→JSON parser**

Add to `src/reachy_claw/plugins/dashboard_plugin.py` (near top, helper):

```python
import re
import yaml  # ensure pyyaml in deps

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_diary_markdown(md: str) -> dict:
    m = _FRONT_MATTER_RE.match(md)
    if not m:
        return {"title": "", "sections": []}
    front = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)

    # Split on '## ' headings.
    sections = []
    current = {"heading": "", "content": ""}
    for line in body.splitlines():
        if line.startswith("## "):
            if current["heading"] or current["content"].strip():
                sections.append(current)
            current = {"heading": line[3:].strip(), "content": ""}
        else:
            current["content"] += line + "\n"
    if current["heading"] or current["content"].strip():
        sections.append(current)

    return {
        "title": front.get("title", ""),
        "date": front.get("date", ""),
        "weather": front.get("weather"),
        "stats": front.get("stats"),
        "captures": front.get("captures", []),
        "sections": sections,
    }
```

- [ ] **Step 10.4: Update endpoint handlers**

Replace the existing diary handlers with:

```python
async def _api_diaries(self, request):
    dates = self.app.db.list_diary_dates()
    return web.json_response({"dates": dates})


async def _api_diary(self, request):
    date = request.match_info["date"]
    row = self.app.db.get_diary(date)
    if row is None:
        return web.json_response({"error": "not found"}, status=404)
    parsed = _parse_diary_markdown(row["markdown"])
    parsed["date"] = date
    parsed["generated_at"] = row["generated_at"]
    return web.json_response(parsed)


async def _api_diary_latest(self, request):
    dates = self.app.db.list_diary_dates()
    if not dates:
        return web.json_response({"error": "no diaries"}, status=404)
    raise web.HTTPFound(f"/api/diary/{dates[0]}")
```

Also add `pyyaml` to `pyproject.toml` if not already present (it should be — verify).

- [ ] **Step 10.5: Add pyyaml to deps if missing**

```bash
grep -i pyyaml pyproject.toml || true
```

If missing:

```bash
uv add pyyaml
```

- [ ] **Step 10.6: Run tests, expect PASS**

```
uv run pytest tests/test_dashboard_plugin.py -v
```

- [ ] **Step 10.7: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_plugin.py tests/test_dashboard_plugin.py pyproject.toml uv.lock
git commit -m "feat(dashboard): diary endpoints read SQLite + parse Markdown front matter"
```

---


## Task 11: `publish_diary.py` (push to site repo)

**Files:**
- Create: `scripts/publish_diary.py`
- Create: `tests/test_publish_diary.py`
- Create: `docs/ops/diary-publish-setup.md`

Test plan: spin up a local **bare** git repo (no network needed), point the publish script at it via `SITE_REPO_URL=file:///path`, run the script, then clone the bare repo elsewhere to verify the file appeared.

- [ ] **Step 11.1: Write failing integration test**

```python
# tests/test_publish_diary.py
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from reachy_claw.storage.db import Database

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "publish_diary.py"


def _git(*args: str, cwd: Path):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def test_publish_pushes_markdown_to_bare_repo(tmp_path: Path):
    # Set up a bare site repo and a worktree that initializes it with main branch.
    bare = tmp_path / "site.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "clone", str(bare), str(seed)], check=True
    )
    (seed / "src").mkdir()
    (seed / "src" / "content").mkdir()
    (seed / "src" / "content" / "docs").mkdir()
    (seed / "README.md").write_text("seed")
    _git("add", ".", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-m", "init", cwd=seed)
    _git("push", "origin", "main", cwd=seed)

    db_path = tmp_path / "t.db"
    db = Database(db_path)
    db.init()
    db.save_diary(
        date="2026-04-26",
        markdown="---\ntitle: t\ndate: \"2026.04.26\"\ncategory: \"Reachy 日记\"\n---\n\nbody",
        llm_model="m",
        prompt_version="v1",
    )
    db.close()

    work = tmp_path / "work"
    env = {
        **os.environ,
        "SITE_REPO_URL": str(bare),
        "SITE_REPO_DIR": str(work),
        "SITE_DIARY_PATH": "src/content/docs",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
    }
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--date", "2026-04-26", "--db", str(db_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, res.stderr

    # Clone bare elsewhere and assert file present.
    verify = tmp_path / "verify"
    subprocess.run(["git", "clone", str(bare), str(verify)], check=True)
    f = verify / "src" / "content" / "docs" / "2026-04-26-reachy-diary.md"
    assert f.exists()
    assert "body" in f.read_text(encoding="utf-8")

    # And published_at is set.
    db = Database(db_path)
    db.init()
    assert db.get_diary("2026-04-26")["published_at"] is not None
    db.close()
```

- [ ] **Step 11.2: Run; expect failure**

```
uv run pytest tests/test_publish_diary.py -v
```

- [ ] **Step 11.3: Implement `publish_diary.py`**

```python
#!/usr/bin/env python3
"""Push a generated diary to the site repo and mark it published.

Reads `diaries.markdown` for the requested date, writes it into a clone of
the site repo at the configured path, copies referenced capture images, and
performs commit + push. On success, sets `diaries.published_at`.

Configuration (environment):
    SITE_REPO_URL       git URL of the Astro site (e.g., git@github-diary-site:owner/repo.git)
    SITE_REPO_DIR       local clone dir (default: ~/.reachy-claw/site-repo)
    SITE_DIARY_PATH     relative path within the repo (default: src/content/docs)
    SITE_STATIC_PATH    relative path for image copies (default: public/captures)
    SITE_BRANCH         branch to push to (default: main)
    CAPTURE_BASE_DIR    where smile capture jpgs live (default: ~/.reachy-claw/captures)

Usage:
    uv run python scripts/publish_diary.py --date 2026-04-26 [--force] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reachy_claw.storage.db import Database  # noqa: E402

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


def _ensure_clone(url: str, repo_dir: Path, branch: str) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        _git(["fetch", "origin"], cwd=repo_dir)
        _git(["checkout", branch], cwd=repo_dir)
        _git(["pull", "--rebase", "origin", branch], cwd=repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", branch, url, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def _front_matter_captures(markdown: str) -> list[dict]:
    m = _FRONT_MATTER_RE.match(markdown)
    if not m:
        return []
    fm = yaml.safe_load(m.group(1)) or {}
    raw = fm.get("captures") or []
    return [c for c in raw if isinstance(c, dict) and c.get("path")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument(
        "--db",
        default=os.environ.get("DATA_DIR")
        and str(Path(os.environ["DATA_DIR"]) / "reachy.db")
        or str(Path.home() / ".reachy-claw" / "reachy.db"),
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    url = os.environ.get("SITE_REPO_URL")
    if not url:
        sys.stderr.write("SITE_REPO_URL not set\n")
        return 2

    repo_dir = Path(
        os.environ.get(
            "SITE_REPO_DIR", str(Path.home() / ".reachy-claw" / "site-repo")
        )
    )
    diary_path = os.environ.get("SITE_DIARY_PATH", "src/content/docs")
    static_path = os.environ.get("SITE_STATIC_PATH", "public/captures")
    branch = os.environ.get("SITE_BRANCH", "main")
    capture_base = Path(
        os.environ.get(
            "CAPTURE_BASE_DIR", str(Path.home() / ".reachy-claw" / "captures")
        )
    )

    db = Database(args.db)
    db.init()
    diary = db.get_diary(args.date)
    if diary is None:
        sys.stderr.write(f"No diary for {args.date}\n")
        db.close()
        return 1
    if diary["published_at"] is not None and not args.force:
        print(f"Already published: {args.date}")
        db.close()
        return 0

    _ensure_clone(url, repo_dir, branch)

    target_md = repo_dir / diary_path / f"{args.date}-reachy-diary.md"
    target_md.parent.mkdir(parents=True, exist_ok=True)
    target_md.write_text(diary["markdown"], encoding="utf-8")

    for cap in _front_matter_captures(diary["markdown"]):
        src = capture_base / args.date / Path(cap["path"]).name
        if not src.exists():
            continue
        dst = repo_dir / static_path / args.date / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    _git(["add", "."], cwd=repo_dir)
    # Skip commit if nothing changed.
    status = _git(["status", "--porcelain"], cwd=repo_dir).stdout.strip()
    if not status:
        print(f"No changes to push for {args.date}")
        db.mark_published(args.date)
        db.close()
        return 0

    _git(["commit", "-m", f"diary: {args.date}"], cwd=repo_dir)
    _git(["push", "origin", branch], cwd=repo_dir)

    db.mark_published(args.date)
    db.close()
    print(f"Published {args.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 11.4: Run; expect PASS**

```
uv run pytest tests/test_publish_diary.py -v
```

- [ ] **Step 11.5: Write ops doc**

```markdown
<!-- docs/ops/diary-publish-setup.md -->
# Diary Publish — Deploy Key Setup

## 1. Generate deploy key (on Jetson)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/diary_publish_ed25519 -N "" -C "reachy-diary-publish"
chmod 600 ~/.ssh/diary_publish_ed25519
```

Copy the public key (`~/.ssh/diary_publish_ed25519.pub`) and add it to the
site repo's GitHub Settings → Deploy keys, with **Write access enabled**.

## 2. SSH host alias

Add to `~/.ssh/config`:

```
Host github-diary-site
  HostName github.com
  User git
  IdentityFile ~/.ssh/diary_publish_ed25519
  IdentitiesOnly yes
```

## 3. Environment variables for the OpenClaw skill

```
SITE_REPO_URL=git@github-diary-site:<owner>/<site-repo>.git
SITE_DIARY_PATH=content/<diary section>      # supplied by user; e.g. content/journal
SITE_BRANCH=main
GIT_AUTHOR_NAME="Reachy Mini"
GIT_AUTHOR_EMAIL="reachy@local"
GIT_COMMITTER_NAME="Reachy Mini"
GIT_COMMITTER_EMAIL="reachy@local"
```

## 4. First run

```bash
uv run python scripts/publish_diary.py --date 2026-04-26
```

If the clone succeeds and the push lands, you're done.

## 5. Key rotation

To rotate: generate a new key, add it to GitHub deploy keys, swap the
`IdentityFile` path, then remove the old key from GitHub.
```

- [ ] **Step 11.6: Commit**

```bash
git add scripts/publish_diary.py tests/test_publish_diary.py docs/ops/diary-publish-setup.md
git commit -m "feat(publish): publish_diary.py pushes Markdown to site repo via deploy key"
```

---

## Task 12: OpenClaw skill chains generate + publish

**Files:**
- (Out of this repo's tree — lives in OpenClaw extensions.)

The OpenClaw `daily-diary` skill is updated to invoke the two scripts in sequence. This task documents the integration but the actual change is in the OpenClaw repo. Implementer should:

- [ ] **Step 12.1: Locate the existing skill**

```bash
ls ~/project/openclaw/extensions/desktop-robot/src/ 2>/dev/null | grep -i diary || true
```

If the skill exists, modify it. Otherwise note the path and create a tracking issue/note.

- [ ] **Step 12.2: Update the skill flow**

The skill, when triggered (cron 23:00 or manual), should:

1. Run `uv run python scripts/generate_diary.py --date $(date +%F)` — fail-fast on non-zero.
2. Run `uv run python scripts/publish_diary.py --date $(date +%F)` — fail-fast on non-zero.
3. Log result.

The exact OpenClaw skill DSL is out of this plan's scope; commit the change in the OpenClaw repo separately.

- [ ] **Step 12.3: Commit a brief note in this repo**

Append to `docs/ops/diary-publish-setup.md`:

```markdown
## OpenClaw skill

The `daily-diary` OpenClaw skill (in `~/project/openclaw/extensions/desktop-robot/src/`)
is responsible for triggering generation + publish at 23:00 daily. See that
repo for the skill definition.
```

```bash
git add docs/ops/diary-publish-setup.md
git commit -m "docs(ops): note OpenClaw skill responsibility for daily trigger"
```

---

## Task 13: Final integration + manual smoke test

- [ ] **Step 13.1: Run the full test suite**

```
uv run pytest -x
```

Expected: all green. Any pre-existing tests that asserted jsonl files on disk should have been removed in earlier tasks.

- [ ] **Step 13.2: Manual end-to-end on Jetson (dispatch via claude-rescue)**

Per the project's operations playbook, this step is dispatched to a remote agent rather than run from the main thread. Suggested prompt:

```
On the Jetson at recomputer@100.67.111.58:
1. Pull latest code, run uv sync.
2. Run scripts/migrate_jsonl_to_sqlite.py to import any historical data.
3. Restart the clawd-reachy-mini container.
4. Confirm reachy.db exists at the configured DATA_DIR and event tables receive new rows.
5. Trigger generate_diary.py --date today with DIARY_LLM_MOCK=1 and verify the diaries row.
6. With SITE_REPO_URL pointing at a test branch of the site repo, run publish_diary.py and verify the file appears on GitHub.

EVIDENCE required: paste md5 of reachy.db, sqlite row counts per table, generated Markdown content,
git log of the site repo showing the publish commit.

Forbidden: rm -rf, sudo rm, recreating containers (only restart), modifying Dockerfile/compose.
```

- [ ] **Step 13.3: When real (non-mock) LLM works in production, set `DIARY_LLM_CMD` and re-run**

This is an environment configuration step, performed by the OpenClaw skill in production.

---

## Self-Review

**Spec coverage:**
- SQLite schema (incl. empty `sensors` table) → Tasks 1, 2, 3, 4 ✓
- DailyLogPlugin migration → Task 5 ✓
- One-time migration script → Task 6 ✓
- Markdown diary generation → Task 8 ✓
- Privacy linter (substring guard) → Task 9 ✓
- Dashboard endpoints over SQLite → Task 10 ✓
- Publish script + deploy key + ops doc → Task 11 ✓
- OpenClaw skill integration → Task 12 ✓
- Manual E2E → Task 13 ✓
- Sensor ingestion (HA pull + config panel) → **deferred to follow-up branch**, called out in spec
- Site-repo Astro template + Cloudflare Workers deployment → out of tree, called out in spec and Task 12

**Placeholder scan:** No "TBD"/"TODO"/"add appropriate error handling" remains in plan steps. The two values explicitly marked as values-at-implementation (site repo URL, diary content path) are configured via environment variables (`SITE_REPO_URL`, `SITE_DIARY_PATH`), not embedded in code, so no code-level placeholder.

**Type consistency:**
- `Database` constructor takes `path` everywhere ✓
- `record_*` use keyword-only args consistently ✓
- `events_for_day` returns `dict[table, list[row]]` and is consumed by both `collect_daily_data.py` and `generate_diary.py` ✓
- `save_diary` signature consistent across `generate_diary.py` and migration script ✓
- `publish_diary.py` reads `diary["markdown"]` and `diary["published_at"]` matching `get_diary` shape ✓

No issues found.
