"""Tests for collect_daily_data.py SQLite-based collection."""

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