"""Tests for jsonl→SQLite migration script."""

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