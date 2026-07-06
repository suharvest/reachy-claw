"""Self-heal watchdog tests for DuplexAudioIO.

Exercises recovery from the two USB-re-enumeration failure modes (mono
fallback + capture stall) with a fake clock, so no audio hardware is needed.
"""
from __future__ import annotations

import asyncio

import pytest

from reachy_voice.audio import DuplexAudioIO


def _io() -> DuplexAudioIO:
    return DuplexAudioIO(device="Test Device:", sr=16000, chunk_ms=100)


# ── _reopen_duplex ──────────────────────────────────────────────────
def test_reopen_closes_refreshes_portaudio_then_opens(monkeypatch):
    io = _io()
    order: list[str] = []
    monkeypatch.setattr(io, "_close_duplex", lambda: order.append("close"))

    def _open() -> None:
        order.append("open")
        io._in_ch = 2

    monkeypatch.setattr(io, "_open_duplex", _open)
    monkeypatch.setattr("reachy_voice.audio.sd._terminate", lambda: order.append("term"))
    monkeypatch.setattr("reachy_voice.audio.sd._initialize", lambda: order.append("init"))

    io._reopen_duplex()

    # close BEFORE the portaudio refresh BEFORE the reopen — order matters.
    assert order == ["close", "term", "init", "open"]
    assert io._reopening is False
    assert io._in_ch == 2


def test_reopen_is_reentrancy_guarded(monkeypatch):
    io = _io()
    io._reopening = True
    monkeypatch.setattr(io, "_close_duplex", lambda: pytest.fail("must not run while reopening"))
    io._reopen_duplex()  # guarded → no-op, no exception


def test_reopen_survives_open_failure(monkeypatch):
    io = _io()
    monkeypatch.setattr(io, "_close_duplex", lambda: None)
    monkeypatch.setattr("reachy_voice.audio.sd._terminate", lambda: None)
    monkeypatch.setattr("reachy_voice.audio.sd._initialize", lambda: None)

    def _boom() -> None:
        raise RuntimeError("device absent mid-reenum")

    monkeypatch.setattr(io, "_open_duplex", _boom)
    io._reopen_duplex()  # swallows the error
    assert io._reopening is False  # flag always released


# ── watchdog decisions (fake clock) ─────────────────────────────────
def _fake_clock(monkeypatch, io, *, stop_after: int):
    """Patch time.monotonic + asyncio.sleep to advance past the reopen
    interval each poll, and end the loop after ``stop_after`` cycles."""
    t = {"now": 1000.0}
    cycles = {"n": 0}
    monkeypatch.setattr("reachy_voice.audio.time.monotonic", lambda: t["now"])

    async def _sleep(_seconds):
        cycles["n"] += 1
        t["now"] += io._REOPEN_INTERVAL_S + 0.1
        if cycles["n"] > stop_after:
            raise asyncio.CancelledError

    monkeypatch.setattr("reachy_voice.audio.asyncio.sleep", _sleep)


def test_watchdog_reopens_on_mono_then_stops_when_recovered(monkeypatch):
    io = _io()
    io._duplex_stream = object()  # pretend a stream is open
    io._in_ch = 1                 # mono → degraded
    io._last_capture_ts = 0.0

    n = {"reopens": 0}

    def _fake_reopen() -> None:
        n["reopens"] += 1
        if n["reopens"] >= 2:
            io._in_ch = 2         # recovered on the 2nd attempt

    monkeypatch.setattr(io, "_reopen_duplex", _fake_reopen)
    _fake_clock(monkeypatch, io, stop_after=8)

    asyncio.run(io._audio_health_watchdog())

    # Reopened until 2 channels came back, then stopped hammering.
    assert n["reopens"] == 2


def test_watchdog_reopens_on_capture_stall(monkeypatch):
    io = _io()
    io._duplex_stream = object()
    io._in_ch = 2                              # channels fine …
    io._last_capture_ts = 1.0                  # … but data went stale (USB drop)

    n = {"reopens": 0}

    def _fake_reopen() -> None:
        n["reopens"] += 1
        io._last_capture_ts = 10_000.0         # fresh stream produces data again

    monkeypatch.setattr(io, "_reopen_duplex", _fake_reopen)
    _fake_clock(monkeypatch, io, stop_after=6)

    asyncio.run(io._audio_health_watchdog())
    assert n["reopens"] == 1                    # one reopen restored the flow


def test_start_capture_exits_after_persistent_open_failure(monkeypatch):
    """Device absent from container /dev: start_capture's open fails every
    mic-pump retry; after _MAX_OPEN_FAILURES it must exit for a restart."""
    io = _io()

    def _boom() -> None:
        raise RuntimeError("No input device matching 'Reachy Mini Audio:'")

    monkeypatch.setattr(io, "_open_duplex", _boom)
    exited = {"code": None}

    def _fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr("reachy_voice.audio.os._exit", _fake_exit)

    # Simulate the mic pump re-calling start_capture on each crash.
    for _ in range(io._MAX_OPEN_FAILURES - 1):
        gen = io.start_capture()
        with pytest.raises(RuntimeError):
            asyncio.run(gen.__anext__())      # open fails → propagates, no exit yet
    assert exited["code"] is None             # not yet at the threshold

    gen = io.start_capture()
    with pytest.raises(SystemExit):           # threshold reached → os._exit
        asyncio.run(gen.__anext__())
    assert exited["code"] == 1


def test_successful_open_resets_failure_counter(monkeypatch):
    io = _io()
    io._open_failures = 5
    monkeypatch.setattr(io, "_open_duplex", lambda: None)
    monkeypatch.setattr(io, "_close_duplex", lambda: None)

    async def _drive():
        gen = io.start_capture()
        # prime the generator to the first await, then stop it
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        await gen.aclose()

    asyncio.run(_drive())
    assert io._open_failures == 0             # a good open clears the counter


def test_watchdog_exits_when_recovery_impossible(monkeypatch):
    io = _io()
    io._duplex_stream = object()
    io._in_ch = 1                              # permanently mono
    io._last_capture_ts = 0.0

    monkeypatch.setattr(io, "_reopen_duplex", lambda: None)  # never recovers
    _fake_clock(monkeypatch, io, stop_after=100)

    exited = {"code": None}

    def _fake_exit(code):
        exited["code"] = code
        raise SystemExit(code)                 # stand-in for process death

    monkeypatch.setattr("reachy_voice.audio.os._exit", _fake_exit)

    with pytest.raises(SystemExit):
        asyncio.run(io._audio_health_watchdog())
    assert exited["code"] == 1
