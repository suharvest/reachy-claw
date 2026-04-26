"""Tests for the diary housekeeping task."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.plugins.housekeeping_tasks import DiaryGenerateAndPublishTask


@dataclass
class _StubConfig:
    diary_auto_publish: bool = True
    diary_site_repo_url: str = "git@github.com:org/site.git"
    diary_site_diary_path: str = "src/content/docs"
    diary_site_branch: str = "main"


class _StubApp:
    def __init__(self):
        self.config = _StubConfig()


@pytest.mark.asyncio
async def test_runs_generate_then_publish_when_auto_publish_true():
    app = _StubApp()
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        # Expect 2 subprocess calls: generate then publish
        assert mock_create.call_count == 2
        gen_args = mock_create.call_args_list[0].args
        pub_args = mock_create.call_args_list[1].args
        assert "generate_diary.py" in " ".join(gen_args)
        assert "publish_diary.py" in " ".join(pub_args)


@pytest.mark.asyncio
async def test_skips_publish_when_auto_publish_false():
    app = _StubApp()
    app.config.diary_auto_publish = False
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        assert mock_create.call_count == 1  # only generate


@pytest.mark.asyncio
async def test_skips_publish_when_site_url_empty():
    app = _StubApp()
    app.config.diary_site_repo_url = ""
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        assert mock_create.call_count == 1  # only generate, publish skipped
