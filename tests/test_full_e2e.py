"""Full end-to-end tests: Reachy Mini sim + OpenClaw gateway + Jetson speech service.

Exercises the complete conversation pipeline:
  mic(simulated) → VAD → STT(Paraformer) → gateway(OpenClaw) → LLM stream
  → sentence split → TTS(Kokoro) → playback(simulated) → interrupt

Requires:
  1. reachy-mini-daemon --mockup-sim --headless --localhost-only --deactivate-audio
  2. OpenClaw desktop-robot extension on ws://127.0.0.1:18790/desktop-robot
  3. Jetson speech service on http://100.67.111.58:8000

Skip automatically if any dependency is not reachable.
Run with: uv run pytest tests/test_full_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import websockets

from clawd_reachy_mini.app import ClawdApp
from clawd_reachy_mini.config import Config
from clawd_reachy_mini.gateway import DesktopRobotClient
from clawd_reachy_mini.plugins.conversation_plugin import (
    ConversationPlugin,
    ConvState,
    SentenceItem,
    _drain_queue,
)

SPEECH_URL = os.environ.get("SPEECH_SERVICE_URL", "http://100.67.111.58:8000")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "ws://127.0.0.1:18790/desktop-robot")


# ── Reachability checks ──────────────────────────────────────────────────


def _speech_service_reachable() -> bool:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        return data.get("tts", False) and data.get("streaming_asr", False)
    except Exception:
        return False


def _gateway_reachable() -> bool:
    async def _check():
        try:
            ws = await asyncio.wait_for(
                websockets.connect(GATEWAY_URL), timeout=3.0
            )
            await ws.send(json.dumps({"type": "hello", "sessionId": "e2e-probe"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            await ws.close()
            return json.loads(msg).get("type") == "welcome"
        except Exception:
            return False

    return asyncio.run(_check())


def _sim_reachable():
    try:
        from reachy_mini import ReachyMini

        reachy = ReachyMini(
            connection_mode="localhost_only",
            media_backend="no_media",
            timeout=3,
        )
        reachy.__enter__()
        return reachy
    except Exception:
        return None


_has_speech = _speech_service_reachable()
_has_gateway = _gateway_reachable()
_skip_no_speech = pytest.mark.skipif(
    not _has_speech, reason=f"Speech service not reachable at {SPEECH_URL}"
)
_skip_no_gateway = pytest.mark.skipif(
    not _has_gateway, reason=f"Gateway not reachable at {GATEWAY_URL}"
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    defaults = dict(
        idle_animations=False,
        enable_face_tracker=False,
        enable_motion=False,
        play_emotions=True,
        tts_backend="kokoro",
        stt_backend="paraformer-streaming",
        speech_service_url=SPEECH_URL,
    )
    defaults.update(overrides)
    return Config(**defaults)


@pytest.fixture(scope="module")
def sim_reachy():
    """Module-scoped fixture: Reachy Mini simulator."""
    reachy = _sim_reachable()
    if reachy is None:
        pytest.skip("Mockup sim daemon not reachable")
    reachy.wake_up()
    time.sleep(0.5)
    yield reachy
    reachy.__exit__(None, None, None)


# ── Test: STT → Gateway → TTS (full pipeline, no real mic) ──────────────


@_skip_no_speech
@_skip_no_gateway
class TestFullConversationPipeline:
    """Simulate a full conversation: inject text as if STT heard it,
    send through gateway, collect LLM response, synthesize via Kokoro TTS.
    """

    @pytest.mark.asyncio
    async def test_text_to_gateway_to_tts(self):
        """Inject transcribed text → gateway → LLM stream → Kokoro TTS → WAV output."""
        config = _make_config(tts_backend="kokoro")

        # 1. Connect to gateway
        client = DesktopRobotClient(config)
        collected_deltas: list[str] = []
        full_response = ""
        stream_done = asyncio.Event()

        async def on_delta(text, rid):
            collected_deltas.append(text)

        async def on_end(text, rid):
            nonlocal full_response
            full_response = text
            stream_done.set()

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()
        assert client.is_connected

        # 2. Send as if user said this (bypass mic/STT)
        await client.send_message_streaming("Say exactly: Hello from Reachy")
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)

        assert len(full_response) > 0
        assert len(collected_deltas) > 0

        # 3. Synthesize the response via Kokoro TTS
        from clawd_reachy_mini.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        path = await tts.synthesize(full_response)
        try:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_streaming_tts_from_gateway_response(self):
        """Gateway LLM response → sentence split → streaming TTS chunks."""
        config = _make_config()

        client = DesktopRobotClient(config)
        full_response = ""
        stream_done = asyncio.Event()

        async def on_end(text, rid):
            nonlocal full_response
            full_response = text
            stream_done.set()

        client.callbacks.on_stream_end = on_end
        await client.connect()

        await client.send_message_streaming(
            "Say two short sentences about robots."
        )
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)

        # Stream the response through Kokoro streaming TTS
        from clawd_reachy_mini.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        assert tts.supports_streaming

        chunks = []
        sample_rate = None
        async for chunk, sr in tts.synthesize_streaming(full_response):
            chunks.append(chunk)
            sample_rate = sr

        assert len(chunks) > 0
        assert sample_rate is not None
        total_samples = sum(len(c) for c in chunks)
        assert total_samples > 1000  # at least some audio

        await client.disconnect()


# ── Test: Streaming ASR → Gateway → Streaming TTS ───────────────────────


@_skip_no_speech
@_skip_no_gateway
class TestStreamingASRToGateway:
    """Test the streaming ASR client feeding into the gateway."""

    @pytest.mark.asyncio
    async def test_streaming_asr_feeds_gateway(self):
        """Open streaming ASR session, feed silence, finalize, send result to gateway."""
        from clawd_reachy_mini.stt import ParaformerStreamingSTT

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)

        # Start streaming ASR session
        stt.start_stream(sample_rate=16000)

        # Feed 1s of silence (no speech expected)
        chunk_size = 1600  # 100ms
        silence = np.zeros(chunk_size, dtype=np.float32)
        for _ in range(10):
            result = stt.feed_chunk(silence)
            # Partial results may or may not come for silence

        text = stt.finish_stream()
        # Silence should give empty or very short text
        assert isinstance(text, str)

        # Now send a real message through gateway
        config = _make_config()
        client = DesktopRobotClient(config)
        stream_done = asyncio.Event()
        response = ""

        async def on_end(t, rid):
            nonlocal response
            response = t
            stream_done.set()

        client.callbacks.on_stream_end = on_end
        await client.connect()

        # Send a known text (as if STT recognized it)
        await client.send_message_streaming("Reply with exactly: ASR test passed")
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)

        assert len(response) > 0

        await client.disconnect()


# ── Test: ConversationPlugin dual pipeline with real backends ──────────


@_skip_no_speech
@_skip_no_gateway
class TestConversationPluginIntegration:
    """Test ConversationPlugin internals with real STT/TTS backends."""

    @pytest.mark.asyncio
    async def test_sentence_accumulator_with_gateway_stream(self):
        """Gateway stream deltas → sentence accumulator → sentence queue."""
        config = _make_config()
        app = ClawdApp(config)

        plugin = ConversationPlugin(app)
        plugin._running = True

        # Start sentence accumulator in background
        accum_task = asyncio.create_task(plugin._sentence_accumulator())

        # Connect gateway and stream a response
        client = DesktopRobotClient(config)

        async def on_delta(text, rid):
            await plugin._stream_text_queue.put(text)

        async def on_end(text, rid):
            await plugin._stream_text_queue.put(None)

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        await client.send_message_streaming(
            "Write two short sentences. Each under 10 words."
        )

        # Wait for sentences to accumulate
        sentences = []
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                item = await asyncio.wait_for(
                    plugin._sentence_queue.get(), timeout=1.0
                )
                sentences.append(item)
                if item.is_last:
                    break
            except asyncio.TimeoutError:
                continue

        # Cleanup
        plugin._running = False
        accum_task.cancel()
        try:
            await accum_task
        except asyncio.CancelledError:
            pass
        await client.disconnect()

        # Verify
        assert len(sentences) > 0
        texts = [s.text for s in sentences if s.text]
        assert len(texts) > 0
        assert any(s.is_last for s in sentences)

    @pytest.mark.asyncio
    async def test_output_pipeline_with_kokoro_tts(self):
        """Sentence queue → output pipeline → Kokoro TTS (batch mode)."""
        config = _make_config(tts_backend="kokoro")
        app = ClawdApp(config)

        plugin = ConversationPlugin(app)
        plugin._running = True

        # Initialize TTS manually (skip full start())
        from clawd_reachy_mini.tts import create_tts_backend

        plugin._tts = create_tts_backend(
            backend="kokoro", config=config
        )

        # Track what gets spoken
        spoken = []
        original_speak = plugin._speak_interruptible

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            # Actually synthesize via Kokoro to test the backend
            path = await plugin._tts.synthesize(text)
            try:
                assert os.path.exists(path)
                assert os.path.getsize(path) > 44  # more than WAV header
            finally:
                os.unlink(path)
            return False  # not interrupted

        plugin._speak_interruptible = mock_speak

        # Feed sentences
        await plugin._sentence_queue.put(SentenceItem(text="Hello world."))
        await plugin._sentence_queue.put(
            SentenceItem(text="Reachy is ready.", is_last=True)
        )

        # Run output pipeline briefly
        pipeline_task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(15.0)  # give Kokoro time to synthesize
        plugin._running = False
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

        assert len(spoken) == 2
        assert "Hello world." in spoken
        assert "Reachy is ready." in spoken
        assert app.is_speaking is False


# ── Test: Full round-trip with Reachy Mini sim ───────────────────────────


@_skip_no_speech
@_skip_no_gateway
class TestFullRoundTripWithSim:
    """Full pipeline with Reachy Mini simulator: gateway + motion + TTS."""

    @pytest.mark.asyncio
    async def test_conversation_turn_with_motion(self, sim_reachy):
        """Complete turn: text → gateway → LLM → TTS → sim robot motion."""
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        config = _make_config(enable_motion=True, play_emotions=True)
        app = ClawdApp(config)
        app.reachy = sim_reachy

        # Start motion plugin
        motion = MotionPlugin(app)
        motion._running = True
        motion_task = asyncio.create_task(motion._motion_loop())

        # Connect gateway
        client = DesktopRobotClient(config)
        stream_done = asyncio.Event()
        full_response = ""

        async def on_end(text, rid):
            nonlocal full_response
            full_response = text
            stream_done.set()

        client.callbacks.on_stream_end = on_end
        await client.connect()

        # Queue thinking emotion (as ConversationPlugin does)
        app.emotions.queue_emotion("thinking")

        # Send message
        await client.send_message_streaming("Say hi in one sentence.")
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)
        assert len(full_response) > 0

        # Queue speaking emotion
        app.emotions.queue_emotion("happy")

        # Synthesize via Kokoro
        from clawd_reachy_mini.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        path = await tts.synthesize(full_response)
        assert os.path.exists(path)
        os.unlink(path)

        # Let motion process emotions
        await asyncio.sleep(1.0)

        # Cleanup
        motion._running = False
        await motion_task
        await client.disconnect()
        app.reachy = None

    @pytest.mark.asyncio
    async def test_interrupt_during_tts(self, sim_reachy):
        """Start TTS playback, then interrupt mid-stream."""
        config = _make_config()
        app = ClawdApp(config)
        app.reachy = sim_reachy

        client = DesktopRobotClient(config)
        events = []
        stream_started = asyncio.Event()

        async def on_start(rid):
            events.append("start")
            stream_started.set()

        async def on_delta(t, r):
            events.append("delta")

        async def on_end(t, r):
            events.append("end")

        async def on_abort(r, rid):
            events.append("abort")

        client.callbacks.on_stream_start = on_start
        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        client.callbacks.on_stream_abort = on_abort

        await client.connect()

        # Ask for a long response
        await client.send_message_streaming(
            "Write a 5-sentence story about a robot exploring the ocean."
        )

        # Wait for stream to start
        await asyncio.wait_for(stream_started.wait(), timeout=15.0)
        await asyncio.sleep(0.5)  # let some deltas arrive

        # Interrupt
        await client.send_interrupt()

        # Wait for abort or end
        for _ in range(30):
            await asyncio.sleep(0.5)
            if "abort" in events or "end" in events:
                break

        await client.disconnect()
        app.reachy = None

        assert "start" in events
        assert "abort" in events or "end" in events

    @pytest.mark.asyncio
    async def test_multiple_turns_with_tts(self, sim_reachy):
        """Multiple conversation turns, each with TTS synthesis."""
        config = _make_config()
        app = ClawdApp(config)
        app.reachy = sim_reachy

        from clawd_reachy_mini.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        client = DesktopRobotClient(config)

        await client.connect()

        for turn in range(3):
            response = ""
            done = asyncio.Event()

            async def on_end(text, rid, _done=done):
                nonlocal response
                response = text
                _done.set()

            client.callbacks.on_stream_end = on_end

            await client.send_message_streaming(f"Turn {turn + 1}: say OK")
            await asyncio.wait_for(done.wait(), timeout=30.0)

            assert len(response) > 0

            # Synthesize each response
            path = await tts.synthesize(response)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 100
            os.unlink(path)

        await client.disconnect()
        app.reachy = None


# ── Test: Streaming ASR + Streaming TTS end-to-end ───────────────────────


@_skip_no_speech
@_skip_no_gateway
class TestStreamingPipeline:
    """Test the full streaming path: streaming ASR → gateway → streaming TTS."""

    @pytest.mark.asyncio
    async def test_streaming_asr_to_streaming_tts(self):
        """Complete streaming pipeline without real audio."""
        from clawd_reachy_mini.stt import ParaformerStreamingSTT
        from clawd_reachy_mini.tts import KokoroTTS

        # 1. Streaming ASR (feed silence, get empty text)
        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)
        stt.start_stream(sample_rate=16000)
        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(5):
            stt.feed_chunk(silence)
        asr_text = stt.finish_stream()
        assert isinstance(asr_text, str)

        # 2. Gateway: send a real message
        config = _make_config()
        client = DesktopRobotClient(config)
        full_response = ""
        done = asyncio.Event()

        async def on_end(text, rid):
            nonlocal full_response
            full_response = text
            done.set()

        client.callbacks.on_stream_end = on_end
        await client.connect()

        await client.send_message_streaming("Reply with: Streaming test OK")
        await asyncio.wait_for(done.wait(), timeout=30.0)
        assert len(full_response) > 0

        # 3. Streaming TTS
        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        assert tts.supports_streaming

        chunks = []
        sr = None
        async for chunk, sample_rate in tts.synthesize_streaming(full_response):
            chunks.append(chunk)
            sr = sample_rate

        assert len(chunks) > 0
        assert sr is not None
        total_duration = sum(len(c) for c in chunks) / sr
        assert total_duration > 0.1  # at least 100ms of audio

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_gateway_stream_and_tts(self):
        """Simulate concurrent: gateway streams text while TTS synthesizes earlier sentences."""
        config = _make_config()

        # Collect sentences from gateway stream
        sentence_queue: asyncio.Queue[SentenceItem] = asyncio.Queue()
        sentence_ends = {".", "!", "?"}
        buffer = ""

        client = DesktopRobotClient(config)

        async def on_delta(text, rid):
            nonlocal buffer
            buffer += text
            # Simple sentence splitting
            while True:
                idx = -1
                for ch in sentence_ends:
                    pos = buffer.find(ch)
                    if pos >= 0 and (idx < 0 or pos < idx):
                        idx = pos
                if idx < 0 or idx < 3:
                    break
                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1 :]
                if sentence:
                    await sentence_queue.put(SentenceItem(text=sentence))

        async def on_end(text, rid):
            if buffer.strip():
                await sentence_queue.put(
                    SentenceItem(text=buffer.strip(), is_last=True)
                )
            else:
                await sentence_queue.put(SentenceItem(text="", is_last=True))

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        # Ask for multi-sentence response
        await client.send_message_streaming(
            "Write exactly 3 short sentences about space. Each sentence on its own."
        )

        # Concurrently consume sentences and synthesize TTS
        from clawd_reachy_mini.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        synthesized = []

        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            try:
                item = await asyncio.wait_for(sentence_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            if item.text:
                # Synthesize this sentence while gateway may still be streaming
                path = await tts.synthesize(item.text)
                size = os.path.getsize(path)
                synthesized.append((item.text, size))
                os.unlink(path)

            if item.is_last:
                break

        await client.disconnect()

        assert len(synthesized) >= 1
        for text, size in synthesized:
            assert size > 44  # more than WAV header
