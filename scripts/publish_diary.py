#!/usr/bin/env python3
"""Push a generated diary to the site repo and mark it published.

Reads `diaries.markdown` for the requested date, writes it into a clone of
the site repo at the configured path, copies referenced capture images, and
performs commit + push. On success, sets `diaries.published_at`.

Configuration (environment):
    SITE_REPO_URL       git URL of the Hugo site (e.g., git@github-diary-site:owner/repo.git)
    SITE_REPO_DIR       local clone dir (default: ~/.reachy-claw/site-repo)
    SITE_DIARY_PATH     relative path within the repo (e.g., content/diary)
    SITE_STATIC_PATH    relative path for image copies (default: static/captures)
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
    diary_path = os.environ.get("SITE_DIARY_PATH", "content/diary")
    static_path = os.environ.get("SITE_STATIC_PATH", "static/captures")
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

    target_md = repo_dir / diary_path / f"{args.date}.md"
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