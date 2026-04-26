#!/usr/bin/env python3
"""Collect a day's events from SQLite and emit a structured JSON blob.

Usage:
    uv run python scripts/collect_daily_data.py --date 2026-04-26 [--db PATH]

Output (stdout): JSON with shape:
    {
      "date": "2026-04-26",
      "events": {
         "asr_events": [...], "emotions": [...], "faces": [...],
         "thoughts": [...], "sensors": [...]
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