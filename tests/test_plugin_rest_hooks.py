"""Tests for Plugin base on_rest_start / on_rest_end hooks."""

from __future__ import annotations

import pytest

from reachy_claw.plugin import Plugin


class _NoopPlugin(Plugin):
    name = "noop"

    async def start(self) -> None:
        pass


class _TrackingPlugin(Plugin):
    name = "tracking"

    def __init__(self, app):
        super().__init__(app)
        self.entered = False
        self.exited = False

    async def start(self) -> None:
        pass

    async def on_rest_start(self) -> None:
        self.entered = True

    async def on_rest_end(self) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_default_hooks_are_noop():
    p = _NoopPlugin(app=None)  # type: ignore[arg-type]
    # Must be coroutines that return without error
    await p.on_rest_start()
    await p.on_rest_end()


@pytest.mark.asyncio
async def test_subclass_can_override_hooks():
    p = _TrackingPlugin(app=None)  # type: ignore[arg-type]
    await p.on_rest_start()
    await p.on_rest_end()
    assert p.entered is True
    assert p.exited is True
