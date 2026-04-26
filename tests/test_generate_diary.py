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
    assert "title_en:" in md
    assert "date: \"2026.04.26\"" in md  # Astro format: dots, quoted
    assert "category: \"机器人日记\"" in md
    assert "description:" in md
    assert "description_en:" in md
    assert "## 今天的心情" in md


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
title: "机器人 Reachy 的日记 · 4 月 26 日"
title_en: "Reachy's Diary · April 26"
date: "2026.04.26"
category: "机器人日记"
description: "泄露测试"
description_en: "Leak test"
author: "Reachy Mini"
author_en: "Reachy Mini"
readTime: "1 分钟"
readTime_en: "1 min read"
coverImage: "https://images.unsplash.com/photo-1485827404703-89b55fcc595e"
tags: ["机器人日记", "Reachy", "AI"]
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