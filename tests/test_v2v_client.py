"""Tests for V2VClient (edge_llm_v2v backend, Wave 1)."""

from __future__ import annotations

import asyncio
import json
import struct

import pytest
import websockets

from reachy_claw.v2v_client import V2VClient, V2VConfig


# ── Helpers ────────────────────────────────────────────────────────────


class FakeV2VServer:
    """Tiny in-memory WebSocket echo/scripted server for tests."""

    def __init__(self):
        self.received: list[object] = []  # ordered list of frames seen
        self.scripted: list[object] = []  # frames to send on connect
        self._server = None
        self.port: int | None = None
        self._client_ready = asyncio.Event()
        self._first_msg_received = asyncio.Event()

    async def __aenter__(self):
        async def handler(ws):
            self._client_ready.set()
            # Send scripted frames immediately
            for f in self.scripted:
                if isinstance(f, (bytes, bytearray)):
                    await ws.send(bytes(f))
                elif isinstance(f, dict):
                    await ws.send(json.dumps(f))
                else:
                    await ws.send(f)
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        self.received.append(("binary", msg))
                    else:
                        try:
                            self.received.append(("json", json.loads(msg)))
                        except Exception:
                            self.received.append(("text", msg))
                    self._first_msg_received.set()
            except websockets.ConnectionClosed:
                pass

        self._server = await websockets.serve(handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


# ── Config frame on connect ────────────────────────────────────────────


class TestConnect:
    @pytest.mark.asyncio
    async def test_config_frame_sent_on_connect(self):
        async with FakeV2VServer() as srv:
            cfg = V2VConfig(
                url=srv.url, sample_rate=16000, asr_language="en",
                tts_language="zh", vad="silero", vad_silence_ms=500,
                multi_utterance=True,
            )
            client = V2VClient(cfg)
            await client.connect()
            # Wait briefly for server to receive config frame
            for _ in range(50):
                if srv.received:
                    break
                await asyncio.sleep(0.01)
            await client.disconnect()

        assert srv.received, "server should receive config frame"
        kind, frame = srv.received[0]
        assert kind == "json"
        assert frame["type"] == "config"
        assert frame["sample_rate"] == 16000
        assert frame["asr_language"] == "en"
        assert frame["tts_language"] == "zh"
        assert frame["vad"] == "silero"
        assert frame["vad_silence_ms"] == 500
        assert frame["multi_utterance"] is True

    @pytest.mark.asyncio
    async def test_multi_utterance_default_true(self):
        cfg = V2VConfig()
        assert cfg.multi_utterance is True
        assert cfg.url == "ws://localhost:8621/v2v/stream"


# ── Outbound frames ────────────────────────────────────────────────────


class TestOutboundFrames:
    @pytest.mark.asyncio
    async def test_send_text_delta_and_flush_and_abort(self):
        async with FakeV2VServer() as srv:
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)
            await client.connect()
            await client.send_text_delta("hello")
            await client.flush_tts()
            await client.abort()
            await client.send_asr_eos()
            # let server drain
            for _ in range(100):
                if len(srv.received) >= 5:
                    break
                await asyncio.sleep(0.01)
            await client.disconnect()

        # First is config, then text/flush/abort/eos in order.
        types = [
            f["type"] for k, f in srv.received if k == "json"
        ]
        assert types[0] == "config"
        assert "text" in types
        assert "tts_flush" in types
        assert "abort" in types
        assert "asr_eos" in types

    @pytest.mark.asyncio
    async def test_send_audio_is_binary(self):
        async with FakeV2VServer() as srv:
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)
            await client.connect()
            pcm = b"\x00\x01" * 100
            await client.send_audio(pcm)
            for _ in range(100):
                if any(k == "binary" for k, _ in srv.received):
                    break
                await asyncio.sleep(0.01)
            await client.disconnect()

        bins = [v for k, v in srv.received if k == "binary"]
        assert bins == [pcm]


# ── Inbound frames: audio header + JSON events ─────────────────────────


class TestInboundDispatch:
    @pytest.mark.asyncio
    async def test_binary_audio_parser_strips_sr_header(self):
        sr = 24000
        pcm = (b"\xaa\xbb" * 50)
        binary_frame = struct.pack("<I", sr) + pcm

        async with FakeV2VServer() as srv:
            srv.scripted = [binary_frame]
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)

            got: list[tuple[int, bytes]] = []
            event = asyncio.Event()

            async def on_audio(sample_rate, data):
                got.append((sample_rate, data))
                event.set()

            client.on_tts_audio = on_audio
            await client.connect()
            try:
                await asyncio.wait_for(event.wait(), timeout=2.0)
            finally:
                await client.disconnect()

        assert got == [(sr, pcm)]

    @pytest.mark.asyncio
    async def test_vad_event_dispatch(self):
        async with FakeV2VServer() as srv:
            srv.scripted = [
                {"type": "vad_event", "event": "speech_start"},
                {"type": "vad_event", "event": "speech_end"},
            ]
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)

            events: list[str] = []
            done = asyncio.Event()

            async def on_vad(ev):
                events.append(ev)
                if len(events) == 2:
                    done.set()

            client.on_vad_event = on_vad
            await client.connect()
            try:
                await asyncio.wait_for(done.wait(), timeout=2.0)
            finally:
                await client.disconnect()

        assert events == ["speech_start", "speech_end"]

    @pytest.mark.asyncio
    async def test_binary_audio_sr_header_once_per_session(self):
        """BUG 4: upstream sends the SR header ONLY on the first binary
        frame of the WS session. Subsequent binary frames are raw PCM.
        Cross-ref: seeed-local-voice/app/main.py:1214,1246-1253 — the
        `sr_header_sent` latch is scoped to `tts_out_task` lifetime and
        never reset, so it persists across utterances and tts_done events.
        """
        sr = 24000
        pcm0 = b"\xaa\xbb" * 50
        pcm1 = b"\xcc\xdd" * 50
        pcm2 = b"\xee\xff" * 50
        first_frame = struct.pack("<I", sr) + pcm0

        async with FakeV2VServer() as srv:
            srv.scripted = [
                first_frame,
                pcm1,                                # raw PCM, no header
                {"type": "tts_done"},                # mid-session sentinel
                pcm2,                                # still raw PCM
            ]
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)

            audio: list[tuple[int, bytes]] = []
            done_count = {"n": 0}
            done = asyncio.Event()

            async def on_audio(sample_rate, data):
                audio.append((sample_rate, data))
                if len(audio) == 3:
                    done.set()

            async def on_tts_done():
                done_count["n"] += 1

            client.on_tts_audio = on_audio
            client.on_tts_done = on_tts_done
            await client.connect()
            try:
                await asyncio.wait_for(done.wait(), timeout=2.0)
            finally:
                await client.disconnect()

        # All three audio frames must yield SR=24000 and exact bytes.
        assert audio == [(sr, pcm0), (sr, pcm1), (sr, pcm2)]
        assert done_count["n"] == 1

    @pytest.mark.asyncio
    async def test_disconnect_resets_sr_latch(self):
        """A reconnect must re-read the SR header."""
        sr_a, sr_b = 16000, 22050
        pcm_a, pcm_b = b"\xaa" * 32, b"\xbb" * 32

        async with FakeV2VServer() as srv:
            srv.scripted = [struct.pack("<I", sr_a) + pcm_a]
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)
            seen: list[tuple[int, bytes]] = []
            ev = asyncio.Event()

            async def on_audio(sample_rate, data):
                seen.append((sample_rate, data))
                ev.set()

            client.on_tts_audio = on_audio
            await client.connect()
            await asyncio.wait_for(ev.wait(), timeout=2.0)
            await client.disconnect()
            # Latch reset on disconnect.
            assert client._tts_sample_rate is None

        # Second session, different SR — header must be parsed afresh.
        async with FakeV2VServer() as srv2:
            srv2.scripted = [struct.pack("<I", sr_b) + pcm_b]
            cfg2 = V2VConfig(url=srv2.url)
            client2 = V2VClient(cfg2)
            seen2: list[tuple[int, bytes]] = []
            ev2 = asyncio.Event()

            async def on_audio2(sample_rate, data):
                seen2.append((sample_rate, data))
                ev2.set()

            client2.on_tts_audio = on_audio2
            await client2.connect()
            try:
                await asyncio.wait_for(ev2.wait(), timeout=2.0)
            finally:
                await client2.disconnect()

        assert seen2 == [(sr_b, pcm_b)]

    @pytest.mark.asyncio
    async def test_asr_final_carries_session_complete(self):
        async with FakeV2VServer() as srv:
            srv.scripted = [{
                "type": "asr_final",
                "text": "hello",
                "session_complete": False,
                "duplicate_of_streamed": True,
            }]
            cfg = V2VConfig(url=srv.url)
            client = V2VClient(cfg)

            got: list[tuple[str, bool, bool]] = []
            done = asyncio.Event()

            async def on_final(text, session_complete, dup):
                got.append((text, session_complete, dup))
                done.set()

            client.on_asr_final = on_final
            await client.connect()
            try:
                await asyncio.wait_for(done.wait(), timeout=2.0)
            finally:
                await client.disconnect()

        assert got == [("hello", False, True)]
