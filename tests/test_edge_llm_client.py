"""Tests for EdgeLLMClient (edge_llm_v2v backend, Wave 1)."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from reachy_claw.edge_llm import (
    EdgeLLMClient,
    EdgeLLMConfig,
)
from reachy_claw.gateway import StreamCallbacks


# ── Helpers ────────────────────────────────────────────────────────────


def _sse_lines(*chunks: dict) -> bytes:
    out: list[str] = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n")
    out.append("data: [DONE]\n")
    return "\n".join(out).encode("utf-8")


def _make_delta(content: str, finish_reason: str | None = None) -> dict:
    return {
        "choices": [
            {
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ]
    }


class _MockClient:
    """Wraps EdgeLLMClient with a configurable MockTransport handler."""

    def __init__(self, cfg: EdgeLLMConfig, handler):
        self.cfg = cfg
        self.transport = httpx.MockTransport(handler)
        self.client = EdgeLLMClient(cfg)
        self.requests: list[httpx.Request] = []

    async def connect(self) -> None:
        # Mimic the body of `connect()`, but inject MockTransport
        # so we never open a real socket.
        self.client._http = httpx.AsyncClient(  # noqa: SLF001
            base_url=self.cfg.base_url, transport=self.transport,
        )
        if not self.cfg.model:
            await self.client._discover_model()  # noqa: SLF001
        self.client._connected = True  # noqa: SLF001


# ── EdgeLLMConfig ──────────────────────────────────────────────────────


class TestEdgeLLMConfig:
    def test_defaults(self):
        c = EdgeLLMConfig()
        assert c.base_url == "http://localhost:8080"
        assert c.model == ""
        assert c.max_history == 3
        assert c.max_tokens == 80
        assert c.prefix_cache is True


# ── Model discovery via /v1/models ─────────────────────────────────────


class TestModelDiscovery:
    @pytest.mark.asyncio
    async def test_auto_discover_first_model(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/v1/models"
            return httpx.Response(
                200,
                json={"data": [
                    {"id": "qwen3-edge-7b", "object": "model"},
                    {"id": "other", "object": "model"},
                ]},
                headers={"X-Request-Id": "req-models-1"},
            )

        cfg = EdgeLLMConfig(model="")
        m = _MockClient(cfg, handler)
        await m.connect()
        assert m.client.model == "qwen3-edge-7b"
        await m.client.disconnect()

    @pytest.mark.asyncio
    async def test_explicit_model_skips_discovery(self):
        calls: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.path)
            return httpx.Response(200, json={"data": []})

        cfg = EdgeLLMConfig(model="my-explicit-model")
        m = _MockClient(cfg, handler)
        await m.connect()
        assert m.client.model == "my-explicit-model"
        assert calls == []  # no discovery happened
        await m.client.disconnect()


# ── SSE delta parsing + stream_end ─────────────────────────────────────


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_sse_deltas_dispatched(self, caplog):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/v1/chat/completions"
            body = _sse_lines(
                _make_delta("Hello"),
                _make_delta(", "),
                _make_delta("world", finish_reason="stop"),
            )
            return httpx.Response(
                200, content=body,
                headers={
                    "content-type": "text/event-stream",
                    "X-Request-Id": "req-chat-1",
                },
            )

        cfg = EdgeLLMConfig(model="test", max_history=3, skip_emotion_extraction=True)
        m = _MockClient(cfg, handler)
        await m.connect()

        deltas: list[tuple[str, str]] = []
        ended: list[tuple[str, str]] = []

        async def on_delta(text, rid):
            deltas.append((text, rid))

        async def on_end(text, rid):
            ended.append((text, rid))

        m.client.callbacks = StreamCallbacks(
            on_stream_delta=on_delta, on_stream_end=on_end,
        )

        with caplog.at_level(logging.INFO):
            await m.client.send_message_streaming("hi")
            await m.client._current_task  # noqa: SLF001

        assert [d[0] for d in deltas] == ["Hello", ", ", "world"]
        assert ended and ended[0][0] == "Hello, world"
        # run_id consistent across callbacks
        run_ids = {d[1] for d in deltas} | {ended[0][1]}
        assert len(run_ids) == 1
        # X-Request-Id logged
        assert any("X-Request-Id=req-chat-1" in r.message for r in caplog.records)
        await m.client.disconnect()

    @pytest.mark.asyncio
    async def test_cache_flag_toggle_first_vs_subsequent(self):
        seen_payloads: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen_payloads.append(json.loads(req.content))
            body = _sse_lines(_make_delta("ok", finish_reason="stop"))
            return httpx.Response(
                200, content=body,
                headers={"X-Request-Id": "x"},
            )

        cfg = EdgeLLMConfig(model="m", prefix_cache=True, skip_emotion_extraction=True)
        m = _MockClient(cfg, handler)
        await m.connect()
        await m.client.send_message_streaming("turn1")
        await m.client._current_task  # noqa: SLF001
        await m.client.send_message_streaming("turn2")
        await m.client._current_task  # noqa: SLF001

        assert seen_payloads[0].get("save_system_prompt_kv_cache") is True
        assert "prefix_cache" not in seen_payloads[0]
        assert seen_payloads[1].get("prefix_cache") is True
        assert "save_system_prompt_kv_cache" not in seen_payloads[1]
        await m.client.disconnect()


# ── History sliding window ─────────────────────────────────────────────


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_window_matches_ollama_behavior(self):
        def handler(req: httpx.Request) -> httpx.Response:
            body = _sse_lines(_make_delta("reply", finish_reason="stop"))
            return httpx.Response(
                200, content=body,
                headers={"X-Request-Id": "x"},
            )

        cfg = EdgeLLMConfig(model="m", max_history=2, skip_emotion_extraction=True)
        m = _MockClient(cfg, handler)
        await m.connect()
        # 3 turns; window should be at most max_history*2 = 4 messages
        for i in range(3):
            await m.client.send_message_streaming(f"q{i}")
            await m.client._current_task  # noqa: SLF001

        assert len(m.client._history) == 4  # noqa: SLF001
        roles = [msg["role"] for msg in m.client._history]  # noqa: SLF001
        assert roles == ["user", "assistant", "user", "assistant"]
        # window kept the latest pairs
        assert m.client._history[-2]["content"] == "q2"  # noqa: SLF001
        await m.client.disconnect()

    @pytest.mark.asyncio
    async def test_history_not_appended_on_abort(self):
        def handler(req: httpx.Request) -> httpx.Response:
            # Returns structured error
            return httpx.Response(
                500,
                json={"error": {
                    "code": "internal",
                    "message": "boom",
                    "context": {"request_id": "req-err-1"},
                }},
                headers={"X-Request-Id": "req-err-1"},
            )

        cfg = EdgeLLMConfig(model="m", max_history=3, skip_emotion_extraction=True)
        m = _MockClient(cfg, handler)
        await m.connect()

        aborts: list[tuple[str, str]] = []

        async def on_abort(reason, rid):
            aborts.append((reason, rid))

        m.client.callbacks = StreamCallbacks(on_stream_abort=on_abort)
        await m.client.send_message_streaming("hi")
        await m.client._current_task  # noqa: SLF001

        assert aborts, "stream_abort should fire on error"
        assert "internal" in aborts[0][0]
        assert m.client._history == []  # noqa: SLF001 — aborted turn poisons nothing
        await m.client.disconnect()

    @pytest.mark.asyncio
    async def test_history_not_appended_on_cancel(self):
        async def handler(req: httpx.Request) -> httpx.Response:
            # never returns
            await asyncio.sleep(5)
            return httpx.Response(200, content=b"")

        cfg = EdgeLLMConfig(model="m", max_history=3, skip_emotion_extraction=True)
        m = _MockClient(cfg, handler)
        await m.connect()
        await m.client.send_message_streaming("hi")
        await asyncio.sleep(0.01)
        await m.client.send_interrupt()
        try:
            await m.client._current_task  # noqa: SLF001
        except asyncio.CancelledError:
            pass
        assert m.client._history == []  # noqa: SLF001
        await m.client.disconnect()


# ── build_messages shape ───────────────────────────────────────────────


class TestBuildMessages:
    def test_system_plus_user(self):
        cfg = EdgeLLMConfig(model="m", system_prompt="SYS", max_history=0)
        c = EdgeLLMClient(cfg)
        msgs = c._build_messages("hello")  # noqa: SLF001
        assert msgs[0] == {"role": "system", "content": "SYS"}
        assert msgs[-1] == {"role": "user", "content": "hello"}
        assert len(msgs) == 2

    def test_history_appended_when_present(self):
        cfg = EdgeLLMConfig(model="m", system_prompt="SYS", max_history=2)
        c = EdgeLLMClient(cfg)
        c._history.extend([  # noqa: SLF001
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ])
        msgs = c._build_messages("u2")  # noqa: SLF001
        assert msgs[1]["content"] == "u1"
        assert msgs[2]["content"] == "a1"
        assert msgs[3] == {"role": "user", "content": "u2"}
