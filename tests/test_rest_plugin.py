"""Tests for RestPlugin schedule logic and event emission."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from reachy_claw.event_bus import EventBus
from reachy_claw.plugins.rest_plugin import RestPlugin, _is_in_window


@dataclass
class _StubConfig:
    rest_enabled: bool = True
    rest_window_start: str = "23:00"
    rest_window_end: str = "24:00"
    rest_timezone: str = "Asia/Shanghai"


class _StubApp:
    def __init__(self):
        self.events = EventBus()
        self.config = _StubConfig()


def test_is_in_window_simple():
    tz = ZoneInfo("Asia/Shanghai")
    inside = datetime(2026, 4, 27, 23, 30, tzinfo=tz)
    outside = datetime(2026, 4, 27, 12, 0, tzinfo=tz)
    assert _is_in_window(inside, "23:00", "24:00") is True
    assert _is_in_window(outside, "23:00", "24:00") is False


def test_is_in_window_overnight():
    """Window 23:00-01:00 spans midnight."""
    tz = ZoneInfo("Asia/Shanghai")
    in_pre_midnight = datetime(2026, 4, 27, 23, 30, tzinfo=tz)
    in_post_midnight = datetime(2026, 4, 28, 0, 30, tzinfo=tz)
    outside = datetime(2026, 4, 28, 2, 0, tzinfo=tz)
    assert _is_in_window(in_pre_midnight, "23:00", "01:00") is True
    assert _is_in_window(in_post_midnight, "23:00", "01:00") is True
    assert _is_in_window(outside, "23:00", "01:00") is False


def test_is_in_window_24_means_end_of_day():
    """The legacy '24:00' notation means up to (but not including) the next 00:00."""
    tz = ZoneInfo("Asia/Shanghai")
    inside = datetime(2026, 4, 27, 23, 59, tzinfo=tz)
    outside = datetime(2026, 4, 28, 0, 0, tzinfo=tz)
    assert _is_in_window(inside, "23:00", "24:00") is True
    assert _is_in_window(outside, "23:00", "24:00") is False


@pytest.mark.asyncio
async def test_emits_rest_start_and_rest_end(monkeypatch):
    app = _StubApp()
    received = []

    app.events.subscribe("rest_start", lambda d: received.append(("start", d)))
    app.events.subscribe("rest_end", lambda d: received.append(("end", d)))

    plugin = RestPlugin(app)  # type: ignore[arg-type]
    # Stub out the ZMQ control publisher so the test doesn't bind a port.
    sent: list[str] = []
    plugin._publish_ctrl = lambda cmd: sent.append(cmd)
    plugin._ensure_ctrl_pub = lambda: None

    await plugin._enter_rest()
    await plugin._exit_rest()

    kinds = [r[0] for r in received]
    assert kinds == ["start", "end"]
    assert sent == ["pause", "resume"]


@pytest.mark.asyncio
async def test_disabled_window_does_not_enter():
    app = _StubApp()
    app.config.rest_enabled = False
    plugin = RestPlugin(app)  # type: ignore[arg-type]
    # Even if "now" would normally be inside the window, disabled means no entry.
    assert plugin._should_rest_now(datetime(2026, 4, 27, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))) is False
