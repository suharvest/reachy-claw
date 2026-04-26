#!/usr/bin/env python3
"""Generate the daily diary as Markdown with Hugo front matter and store it.

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

SYSTEM_PROMPT = """You are Reachy Mini, a small humanoid robot. Write today's diary as Markdown with YAML front matter, in a warm reflective first-person tone.

Rules:
- Never quote user speech verbatim. Paraphrase what was said and discussed.
- Never include personal identifiers (names, addresses, phone numbers) from ASR.
- Use exactly these section headings, in this order: "## 今天的心情", "## 遇到的人", "## 想到的事".
- Front matter must include: title, date, weather (object), stats (object), captures (list), meta (object with llm_model and prompt_version).
- Output ONLY the Markdown document. No code fences, no commentary.
"""


def _build_user_prompt(date: str, events: dict) -> str:
    return (
        f"Date: {date}\n"
        f"Events as JSON (paraphrase only, never quote):\n{json.dumps(events, ensure_ascii=False)}\n"
    )


def _mock_markdown(date: str, events: dict) -> str:
    n_asr = len(events.get("asr_events", []))
    n_faces = sum(r.get("count", 0) for r in events.get("faces", []))
    smiles = sum(r.get("smile_count", 0) for r in events.get("faces", []))
    return (
        "---\n"
        f"title: \"A Day on {date}\"\n"
        f"date: {date}\n"
        "weather: {condition: \"unknown\"}\n"
        f"stats: {{conversations: {n_asr}, faces_seen: {n_faces}, smiles: {smiles}}}\n"
        "captures: []\n"
        f"meta: {{llm_model: \"mock\", prompt_version: \"{PROMPT_VERSION}\"}}\n"
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