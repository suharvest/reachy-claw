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