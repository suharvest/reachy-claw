"""Housekeeping tasks run during the rest window.

A HousekeepingTask is anything with `.name: str` and `.run(app)` coroutine.
v1 ships DiaryGenerateAndPublishTask. New tasks (DBVacuumTask,
CoverImageGenerateTask, etc.) just register with RestPlugin.register_task().
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


class HousekeepingTask(Protocol):
    name: str

    async def run(self, app) -> None: ...


class DiaryGenerateAndPublishTask:
    """Runs generate_diary.py for today, then publish_diary.py if auto_publish."""

    name = "diary_generate_and_publish"

    async def run(self, app) -> None:
        date = datetime.now().strftime("%Y-%m-%d")
        gen_script = REPO_ROOT / "scripts" / "generate_diary.py"
        pub_script = REPO_ROOT / "scripts" / "publish_diary.py"

        await self._run_subprocess(
            sys.executable,
            str(gen_script),
            "--date",
            date,
            label="generate_diary",
        )

        if not getattr(app.config, "diary_auto_publish", True):
            logger.info("Skipping publish: diary_auto_publish is false")
            return
        if not getattr(app.config, "diary_site_repo_url", "").strip():
            logger.warning("Skipping publish: diary_site_repo_url is empty")
            return

        env = os.environ.copy()
        env["SITE_REPO_URL"] = app.config.diary_site_repo_url
        env["SITE_DIARY_PATH"] = app.config.diary_site_diary_path
        env["SITE_BRANCH"] = app.config.diary_site_branch

        await self._run_subprocess(
            sys.executable,
            str(pub_script),
            "--date",
            date,
            label="publish_diary",
            env=env,
        )

    @staticmethod
    async def _run_subprocess(*args: str, label: str, env: dict | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            # Hit when RestPlugin's wait_for hard cap fires. Make sure the child
            # is killed and reaped so it can't keep consuming LLM tokens / NPU.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} failed (rc={proc.returncode}): {stderr.decode(errors='replace')}"
            )
        logger.info("%s OK: %s", label, stdout.decode(errors="replace").strip()[:200])
