"""Protocol-focused tests for DesktopRobotClient."""

from __future__ import annotations

import asyncio
import json

import pytest

from reachy_claw.config import Config
from reachy_claw.gateway import DesktopRobotClient


@pytest.mark.asyncio
async def test_handle_welcome_sets_session_id():
    client = DesktopRobotClient(Config())

    await client._handle({"type": "welcome", "sessionId": "abc-123"})

    assert client._session_id == "abc-123"
    assert client._welcome_event.is_set()


@pytest.mark.asyncio
async def test_stream_lifecycle_resolves_future():
    """stream_start → stream_delta → stream_end resolves the _next future."""
    client = DesktopRobotClient(Config())

    future: asyncio.Future[str] = asyncio.Future()
    client._run_futures["_next"] = future

    await client._handle({"type": "stream_start", "runId": "run-1"})
    await client._handle({"type": "stream_delta", "text": "Hello ", "runId": "run-1"})
    await client._handle({"type": "stream_delta", "text": "world", "runId": "run-1"})
    await client._handle({"type": "stream_end", "runId": "run-1", "fullText": "Hello world"})

    assert future.done()
    assert future.result() == "Hello world"


@pytest.mark.asyncio
async def test_stream_abort_resolves_with_partial_text():
    client = DesktopRobotClient(Config())

    future: asyncio.Future[str] = asyncio.Future()
    client._run_futures["_next"] = future

    await client._handle({"type": "stream_start", "runId": "run-2"})
    await client._handle({"type": "stream_delta", "text": "Partial", "runId": "run-2"})
    await client._handle({"type": "stream_abort", "runId": "run-2", "reason": "interrupt"})

    assert future.done()
    assert future.result() == "Partial"


@pytest.mark.asyncio
async def test_send_message_handles_immediate_stream_events_without_race():
    client = DesktopRobotClient(Config())
    client._connected = True
    client._ws = object()

    async def fake_send(_payload):
        # Simulate server responding immediately while send_message is in-flight.
        await client._handle({"type": "stream_start", "runId": "run-fast"})
        await client._handle(
            {"type": "stream_end", "runId": "run-fast", "fullText": "ok"}
        )

    client._send = fake_send  # type: ignore[assignment]

    text = await client.send_message("hello")
    assert text == "ok"


@pytest.mark.asyncio
async def test_callbacks_fire_on_stream_events():
    client = DesktopRobotClient(Config())
    events: list[tuple[str, ...]] = []

    client.callbacks.on_stream_start = lambda rid: events.append(("start", rid))
    client.callbacks.on_stream_delta = lambda t, rid: events.append(("delta", t, rid))
    client.callbacks.on_stream_end = lambda t, rid: events.append(("end", t, rid))

    await client._handle({"type": "stream_start", "runId": "r1"})
    await client._handle({"type": "stream_delta", "text": "hi", "runId": "r1"})
    await client._handle({"type": "stream_end", "runId": "r1", "fullText": "hi"})

    assert ("start", "r1") in events
    assert ("delta", "hi", "r1") in events
    assert ("end", "hi", "r1") in events


@pytest.mark.asyncio
async def test_task_spawned_and_completed_callbacks():
    client = DesktopRobotClient(Config())
    events: list[tuple] = []

    client.callbacks.on_task_spawned = lambda l, r: events.append(("spawned", l, r))
    client.callbacks.on_task_completed = lambda s, r: events.append(("completed", s, r))

    await client._handle(
        {"type": "task_spawned", "taskLabel": "search", "taskRunId": "t1"}
    )
    await client._handle(
        {"type": "task_completed", "taskRunId": "t1", "summary": "Found 3 results"}
    )

    assert events == [
        ("spawned", "search", "t1"),
        ("completed", "Found 3 results", "t1"),
    ]


@pytest.mark.asyncio
async def test_error_callback():
    client = DesktopRobotClient(Config())
    errors: list[str] = []

    client.callbacks.on_error = lambda msg: errors.append(msg)

    await client._handle({"type": "error", "message": "bad request"})

    assert errors == ["bad request"]
