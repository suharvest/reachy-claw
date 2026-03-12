"""Tests for EventBus pub/sub."""

from __future__ import annotations

import asyncio

import pytest

from reachy_claw.event_bus import EventBus


def test_subscribe_and_emit_sync():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda data: received.append(data))
    bus.emit("test", {"key": "value"})
    assert received == [{"key": "value"}]


def test_unsubscribe():
    bus = EventBus()
    received = []
    cb = lambda data: received.append(data)
    bus.subscribe("test", cb)
    bus.emit("test", "first")
    bus.unsubscribe("test", cb)
    bus.emit("test", "second")
    assert received == ["first"]


def test_unsubscribe_missing_callback():
    bus = EventBus()
    # Should not raise
    bus.unsubscribe("nonexistent", lambda d: None)


def test_emit_no_subscribers():
    bus = EventBus()
    # Should not raise
    bus.emit("nobody_listening", {"data": 1})


def test_multiple_subscribers():
    bus = EventBus()
    results_a, results_b = [], []
    bus.subscribe("evt", lambda d: results_a.append(d))
    bus.subscribe("evt", lambda d: results_b.append(d))
    bus.emit("evt", 42)
    assert results_a == [42]
    assert results_b == [42]


def test_different_events_isolated():
    bus = EventBus()
    a_data, b_data = [], []
    bus.subscribe("a", lambda d: a_data.append(d))
    bus.subscribe("b", lambda d: b_data.append(d))
    bus.emit("a", 1)
    bus.emit("b", 2)
    assert a_data == [1]
    assert b_data == [2]


def test_callback_error_does_not_break_others():
    bus = EventBus()
    results = []

    def bad_cb(data):
        raise ValueError("boom")

    bus.subscribe("evt", bad_cb)
    bus.subscribe("evt", lambda d: results.append(d))
    bus.emit("evt", "ok")
    assert results == ["ok"]


@pytest.mark.asyncio
async def test_async_callback():
    bus = EventBus()
    received = []

    async def async_cb(data):
        received.append(data)

    bus.subscribe("async_evt", async_cb)
    bus.emit("async_evt", {"hello": "world"})
    # Give the task a chance to run
    await asyncio.sleep(0.05)
    assert received == [{"hello": "world"}]


@pytest.mark.asyncio
async def test_mixed_sync_async_callbacks():
    bus = EventBus()
    sync_results = []
    async_results = []

    bus.subscribe("mixed", lambda d: sync_results.append(d))

    async def async_cb(data):
        async_results.append(data)

    bus.subscribe("mixed", async_cb)
    bus.emit("mixed", "test")
    await asyncio.sleep(0.05)
    assert sync_results == ["test"]
    assert async_results == ["test"]


def test_emit_with_none_data():
    bus = EventBus()
    received = []
    bus.subscribe("evt", lambda d: received.append(d))
    bus.emit("evt")
    assert received == [None]
