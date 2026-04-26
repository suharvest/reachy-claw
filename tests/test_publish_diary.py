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
        markdown="---\ntitle: t\ndate: \"2026.04.26\"\ncategory: \"机器人日记\"\n---\n\nbody",
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