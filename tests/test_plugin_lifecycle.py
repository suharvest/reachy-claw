"""Tests for the plugin framework and ReachyClawApp lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config
from reachy_claw.plugin import Plugin


# ── Helpers ────────────────────────────────────────────────────────────


class DummyPlugin(Plugin):
    """Simple plugin that records lifecycle calls."""

    name = "dummy"

    def __init__(self, app, *, fail_setup=False, fail_start=False):
        super().__init__(app)
        self._fail_setup = fail_setup
        self._fail_start = fail_start
        self.setup_called = False
        self.start_called = False
        self.stop_called = False

    def setup(self) -> bool:
        self.setup_called = True
        if self._fail_setup:
            return False
        return True

    async def start(self) -> None:
        self.start_called = True
        if self._fail_start:
            raise RuntimeError("start failed on purpose")
        # Run until stopped
        while self._running:
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        await super().stop()
        self.stop_called = True


class CounterPlugin(Plugin):
    """Plugin that increments a counter each tick until stopped."""

    name = "counter"

    def __init__(self, app):
        super().__init__(app)
        self.ticks = 0

    async def start(self) -> None:
        while self._running:
            self.ticks += 1
            await asyncio.sleep(0.01)


# ── Plugin registration ───────────────────────────────────────────────


def test_register_success(app):
    p = DummyPlugin(app)
    assert app.register(p) is True
    assert p.setup_called
    assert p in app._plugins


def test_register_skip_on_setup_false(app):
    p = DummyPlugin(app, fail_setup=True)
    assert app.register(p) is False
    assert p.setup_called
    assert p not in app._plugins


def test_register_skip_on_setup_exception(app):
    class Boom(Plugin):
        name = "boom"

        def setup(self):
            raise RuntimeError("kaboom")

        async def start(self):
            pass

    p = Boom(app)
    assert app.register(p) is False
    assert p not in app._plugins


def test_register_multiple(app):
    p1 = DummyPlugin(app)
    p1.name = "first"
    p2 = CounterPlugin(app)

    app.register(p1)
    app.register(p2)

    assert len(app._plugins) == 2
    names = [p.name for p in app._plugins]
    assert names == ["first", "counter"]


# ── App run / shutdown ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_app_run_starts_and_stops_plugins(app):
    p = DummyPlugin(app)
    app.register(p)

    # Run for a short time then cancel
    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.05)

    assert p.start_called
    assert app.running

    # Trigger shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await app.shutdown()
    assert p.stop_called
    assert not app.running


@pytest.mark.asyncio
async def test_app_run_multiple_plugins_concurrently(app):
    c1 = CounterPlugin(app)
    c1.name = "counter_a"
    c2 = CounterPlugin(app)
    c2.name = "counter_b"

    app.register(c1)
    app.register(c2)

    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.1)

    # Both should have ticked
    assert c1.ticks > 0
    assert c2.ticks > 0

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await app.shutdown()


@pytest.mark.asyncio
async def test_app_run_no_plugins_returns_immediately(config, mock_reachy):
    a = ReachyClawApp(config)
    a.reachy = mock_reachy
    await a.run()
    # Should return without error


@pytest.mark.asyncio
async def test_shutdown_reverse_order(app):
    stop_order = []

    class OrderPlugin(Plugin):
        async def start(self):
            while self._running:
                await asyncio.sleep(0.01)

        async def stop(self):
            await super().stop()
            stop_order.append(self.name)

    p1 = OrderPlugin(app)
    p1.name = "first"
    p2 = OrderPlugin(app)
    p2.name = "second"
    p3 = OrderPlugin(app)
    p3.name = "third"

    app.register(p1)
    app.register(p2)
    app.register(p3)

    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await app.shutdown()

    assert stop_order == ["third", "second", "first"]


@pytest.mark.asyncio
async def test_shared_state_between_plugins(app):
    """Plugins share app.is_speaking and app.emotions."""

    class WriterPlugin(Plugin):
        name = "writer"

        async def start(self):
            self.app.is_speaking = True
            self.app.emotions.queue_emotion("happy")
            while self._running:
                await asyncio.sleep(0.01)

    class ReaderPlugin(Plugin):
        name = "reader"
        saw_speaking = False
        saw_emotion = False

        async def start(self):
            await asyncio.sleep(0.02)
            self.saw_speaking = self.app.is_speaking
            expr = self.app.emotions.get_next_expression()
            self.saw_emotion = expr is not None
            while self._running:
                await asyncio.sleep(0.01)

    w = WriterPlugin(app)
    r = ReaderPlugin(app)
    app.register(w)
    app.register(r)

    task = asyncio.create_task(app.run())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await app.shutdown()

    assert r.saw_speaking is True
    assert r.saw_emotion is True
