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
  - title: Chinese title, format "机器人 Reachy 的日记 · X 月 Y 日"
  - title_en: English title, format "Reachy's Diary · Month Day"
  - date: "YYYY.MM.DD" (dots, not dashes)
  - category: "机器人日记"
  - description: Chinese summary (1-2 sentences)
  - description_en: English summary (1-2 sentences)
  - author: "Reachy Mini"
  - author_en: "Reachy Mini"
  - readTime: "X 分钟"
  - readTime_en: "X min read"
  - coverImage: a URL (use first available smile capture URL OR a placeholder Unsplash robot URL)
  - tags: array like ["机器人日记", "Reachy", "AI"]
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
        f"title: \"机器人 Reachy 的日记 · {month} 月 {day} 日\"\n"
        f"title_en: \"Reachy's Diary · {month_en} {day}\"\n"
        f"date: \"{astro_date}\"\n"
        "category: \"机器人日记\"\n"
        f"description: \"今天来了 {n_faces} 位朋友，其中 {smiles} 位对我露出了笑容。\"\n"
        f"description_en: \"Today {n_faces} people stopped by, and {smiles} of them smiled at me.\"\n"
        "author: \"Reachy Mini\"\n"
        "author_en: \"Reachy Mini\"\n"
        f"readTime: \"{max(1, n_asr)} 分钟\"\n"
        f"readTime_en: \"{max(1, n_asr)} min read\"\n"
        "coverImage: \"https://images.unsplash.com/photo-1485827404703-89b55fcc595e\"\n"
        "tags: [\"机器人日记\", \"Reachy\", \"AI\"]\n"
        "---\n\n"
        "## 今天的心情\n\n今天平静而充实。\n\n"
        "## 遇到的人\n\n来过几位朋友，我用微笑回应了他们。\n\n"
        "## 想到的事\n\n我想了一下世界的样子。\n"
    )


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