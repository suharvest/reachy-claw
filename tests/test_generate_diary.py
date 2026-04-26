"""Tests for generate_diary.py Markdown-based diary generation."""

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
    assert "date: 2026-04-26" in md
    assert "## 今天的心情" in md