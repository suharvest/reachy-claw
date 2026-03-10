"""Latency benchmark: measure end-to-end delay from speech-end to first audio.

Pipeline stages measured:
  T0: User stops speaking (STT finalize starts)
  T1: STT text ready
  T2: Gateway receives text, first LLM delta arrives
  T3: First complete sentence accumulated
  T4: TTS synthesis starts producing audio (first chunk or WAV ready)

Run with: uv run pytest tests/test_latency.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.gateway import DesktopRobotClient

SPEECH_URL = os.environ.get("SPEECH_SERVICE_URL", "http://100.67.111.58:8000")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "ws://127.0.0.1:18790/desktop-robot")


def _check_services() -> tuple[bool, bool]:
    import urllib.request

    speech_ok = False
    try:
        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        speech_ok = data.get("tts", False)
    except Exception:
        pass

    gateway_ok = False
    try:
        import websockets.sync.client

        ws = websockets.sync.client.connect(GATEWAY_URL, open_timeout=3)
        ws.send(json.dumps({"type": "hello", "sessionId": "latency-probe"}))
        msg = json.loads(ws.recv(timeout=3))
        gateway_ok = msg.get("type") == "welcome"
        ws.close()
    except Exception:
        pass

    return speech_ok, gateway_ok


_has_speech, _has_gateway = _check_services()

_skip_no_speech = pytest.mark.skipif(not _has_speech, reason="Speech service not reachable")
_skip_no_gateway = pytest.mark.skipif(not _has_gateway, reason="Gateway not reachable")


def _make_config(**overrides) -> Config:
    defaults = dict(
        speech_service_url=SPEECH_URL,
        tts_backend="kokoro",
        stt_backend="paraformer-streaming",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _print_latency_table(title: str, stages: list[tuple[str, float]]):
    """Pretty-print a latency breakdown table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {'Stage':<40} {'Latency':>10}")
    print(f"  {'-' * 40} {'-' * 10}")
    total = 0.0
    for label, ms in stages:
        total += ms
        print(f"  {label:<40} {ms:>8.0f} ms")
    print(f"  {'-' * 40} {'-' * 10}")
    print(f"  {'TOTAL':<40} {total:>8.0f} ms")
    print(f"{'=' * 60}\n")


# ── Stage 1: STT finalization latency ────────────────────────────────────


@_skip_no_speech
class TestSTTLatency:
    def test_streaming_asr_finalize_latency(self):
        """Measure: feed audio → call finish_stream() → text ready."""
        from reachy_claw.stt import ParaformerStreamingSTT

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)

        # Feed 1s of silence (simulates end-of-speech moment)
        stt.start_stream(sample_rate=16000)
        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(10):
            stt.feed_chunk(silence)

        t0 = time.perf_counter()
        text = stt.finish_stream()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        _print_latency_table("STT Finalize Latency", [
            ("finish_stream() (silence → final text)", latency_ms),
        ])
        assert isinstance(text, str)

    def test_batch_asr_latency(self):
        """Measure: batch transcribe 1s of silence."""
        from reachy_claw.stt import ParaformerStreamingSTT

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)

        silence = np.zeros(16000, dtype=np.float32)

        t0 = time.perf_counter()
        text = stt.transcribe(silence, sample_rate=16000)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        _print_latency_table("Batch ASR Latency", [
            ("HTTP POST /asr (1s silence)", latency_ms),
        ])
        assert isinstance(text, str)


# ── Stage 2: Gateway + LLM first-token latency ──────────────────────────


@_skip_no_gateway
class TestGatewayLLMLatency:
    @pytest.mark.asyncio
    async def test_time_to_first_token(self):
        """Measure: send message → first stream delta arrives."""
        config = _make_config()
        client = DesktopRobotClient(config)

        first_delta = asyncio.Event()
        stream_done = asyncio.Event()
        timestamps = {}

        async def on_start(rid):
            timestamps["stream_start"] = time.perf_counter()

        async def on_delta(text, rid):
            if "first_delta" not in timestamps:
                timestamps["first_delta"] = time.perf_counter()
                first_delta.set()

        async def on_end(text, rid):
            timestamps["stream_end"] = time.perf_counter()
            stream_done.set()

        client.callbacks.on_stream_start = on_start
        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end

        await client.connect()

        timestamps["send"] = time.perf_counter()
        await client.send_message_streaming("Say exactly: Hello")

        await asyncio.wait_for(stream_done.wait(), timeout=30.0)

        send_to_start = (timestamps.get("stream_start", timestamps["send"]) - timestamps["send"]) * 1000
        start_to_first = (timestamps["first_delta"] - timestamps.get("stream_start", timestamps["send"])) * 1000
        first_to_end = (timestamps["stream_end"] - timestamps["first_delta"]) * 1000
        total = (timestamps["stream_end"] - timestamps["send"]) * 1000

        _print_latency_table("Gateway + LLM Latency", [
            ("send → stream_start", send_to_start),
            ("stream_start → first delta (TTFT)", start_to_first),
            ("first delta → stream_end", first_to_end),
            ("TOTAL (send → end)", total),
        ])

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_time_to_first_sentence(self):
        """Measure: send message → first complete sentence from accumulator."""
        from reachy_claw.plugins.conversation_plugin import (
            ConversationPlugin,
            SentenceItem,
        )

        config = _make_config()
        app = __import__("reachy_claw.app", fromlist=["ReachyClawApp"]).ReachyClawApp(config)
        plugin = ConversationPlugin(app)
        plugin._running = True

        accum_task = asyncio.create_task(plugin._sentence_accumulator())

        client = DesktopRobotClient(config)
        timestamps = {}

        async def on_delta(text, rid):
            if "first_delta" not in timestamps:
                timestamps["first_delta"] = time.perf_counter()
            await plugin._stream_text_queue.put(text)

        async def on_end(text, rid):
            timestamps["stream_end"] = time.perf_counter()
            await plugin._stream_text_queue.put(None)

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        timestamps["send"] = time.perf_counter()
        await client.send_message_streaming(
            "Say exactly two sentences: Hello world. I am Reachy."
        )

        # Wait for first sentence
        first_sentence = await asyncio.wait_for(
            plugin._sentence_queue.get(), timeout=30.0
        )
        timestamps["first_sentence"] = time.perf_counter()

        # Collect remaining
        deadline = time.monotonic() + 15.0
        sentences = [first_sentence]
        while time.monotonic() < deadline:
            try:
                item = await asyncio.wait_for(plugin._sentence_queue.get(), timeout=2.0)
                sentences.append(item)
                if item.is_last:
                    break
            except asyncio.TimeoutError:
                break

        plugin._running = False
        accum_task.cancel()
        try:
            await accum_task
        except asyncio.CancelledError:
            pass
        await client.disconnect()

        send_to_first_delta = (timestamps.get("first_delta", timestamps["send"]) - timestamps["send"]) * 1000
        first_delta_to_sentence = (timestamps["first_sentence"] - timestamps.get("first_delta", timestamps["send"])) * 1000
        send_to_sentence = (timestamps["first_sentence"] - timestamps["send"]) * 1000

        _print_latency_table("Time to First Sentence", [
            ("send → first LLM delta", send_to_first_delta),
            ("first delta → first sentence ready", first_delta_to_sentence),
            ("TOTAL (send → first sentence)", send_to_sentence),
        ])

        assert len(sentences) > 0


# ── Stage 3: TTS synthesis latency ───────────────────────────────────────


@_skip_no_speech
class TestTTSLatency:
    @pytest.mark.asyncio
    async def test_batch_tts_latency(self):
        """Measure: text → WAV file ready (batch mode)."""
        from reachy_claw.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)

        for text, label in [
            ("Hello.", "short (6 chars)"),
            ("Hello world, I am Reachy Mini.", "medium (30 chars)"),
            ("The quick brown fox jumps over the lazy dog. This is a longer sentence to test synthesis speed.", "long (95 chars)"),
        ]:
            t0 = time.perf_counter()
            path = await tts.synthesize(text)
            t1 = time.perf_counter()

            size = os.path.getsize(path)
            os.unlink(path)

            latency_ms = (t1 - t0) * 1000
            _print_latency_table(f"Batch TTS: {label}", [
                (f"synthesize({len(text)} chars) → WAV ({size} bytes)", latency_ms),
            ])

    @pytest.mark.asyncio
    async def test_streaming_tts_time_to_first_chunk(self):
        """Measure: text → first audio chunk arrives (streaming mode)."""
        from reachy_claw.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        if not tts.supports_streaming:
            pytest.skip("Streaming TTS not available")

        for text, label in [
            ("Hello.", "short"),
            ("Hello world, I am Reachy Mini.", "medium"),
            ("The quick brown fox jumps over the lazy dog. This is a longer test.", "long"),
        ]:
            t0 = time.perf_counter()
            t_first = None
            chunk_count = 0
            total_samples = 0
            sample_rate = 16000

            async for chunk, sr in tts.synthesize_streaming(text):
                if t_first is None:
                    t_first = time.perf_counter()
                chunk_count += 1
                total_samples += len(chunk)
                sample_rate = sr

            t_end = time.perf_counter()

            ttfc_ms = (t_first - t0) * 1000 if t_first else 0
            total_ms = (t_end - t0) * 1000
            audio_duration = total_samples / sample_rate

            _print_latency_table(f"Streaming TTS: {label} ({len(text)} chars)", [
                ("Time to first chunk (TTFC)", ttfc_ms),
                (f"Total ({chunk_count} chunks, {audio_duration:.2f}s audio)", total_ms),
            ])


# ── Full pipeline latency ────────────────────────────────────────────────


@_skip_no_speech
@_skip_no_gateway
class TestFullPipelineLatency:
    @pytest.mark.asyncio
    async def test_end_to_end_latency_batch_tts(self):
        """Full pipeline: STT done → gateway → LLM → sentence → batch TTS → audio ready.

        Simulates: user stops speaking, text already transcribed,
        measures from send to AI until first audio file is ready.
        """
        from reachy_claw.tts import KokoroTTS

        config = _make_config()
        client = DesktopRobotClient(config)
        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)

        timestamps = {}
        first_sentence_text = None
        sentence_ready = asyncio.Event()

        # Simple sentence splitter
        buffer_holder = {"buf": ""}
        sentence_ends = {".", "!", "?"}

        async def on_delta(text, rid):
            nonlocal first_sentence_text
            if "first_delta" not in timestamps:
                timestamps["first_delta"] = time.perf_counter()
            buffer_holder["buf"] += text

            # Try to extract first sentence
            if first_sentence_text is None:
                buf = buffer_holder["buf"]
                for ch in sentence_ends:
                    pos = buf.find(ch)
                    if pos >= 3:
                        first_sentence_text = buf[: pos + 1].strip()
                        timestamps["first_sentence"] = time.perf_counter()
                        sentence_ready.set()
                        break

        stream_done = asyncio.Event()

        async def on_end(text, rid):
            nonlocal first_sentence_text
            timestamps["stream_end"] = time.perf_counter()
            # If no sentence boundary was found, use full response
            if first_sentence_text is None:
                first_sentence_text = text.strip()
                timestamps["first_sentence"] = time.perf_counter()
                sentence_ready.set()
            stream_done.set()

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        # T0: "user stopped speaking, STT produced text, sending to AI"
        timestamps["send"] = time.perf_counter()
        await client.send_message_streaming("Say exactly: Hello from Reachy Mini.")

        # Wait for first sentence
        await asyncio.wait_for(sentence_ready.wait(), timeout=30.0)

        # TTS: synthesize first sentence
        timestamps["tts_start"] = time.perf_counter()
        path = await tts.synthesize(first_sentence_text)
        timestamps["tts_done"] = time.perf_counter()

        size = os.path.getsize(path)
        os.unlink(path)

        # Wait for stream to fully complete
        await asyncio.wait_for(stream_done.wait(), timeout=30.0)
        await client.disconnect()

        # Calculate
        send_to_delta = (timestamps["first_delta"] - timestamps["send"]) * 1000
        delta_to_sentence = (timestamps["first_sentence"] - timestamps["first_delta"]) * 1000
        tts_time = (timestamps["tts_done"] - timestamps["tts_start"]) * 1000
        total = (timestamps["tts_done"] - timestamps["send"]) * 1000

        _print_latency_table("Full Pipeline: Batch TTS", [
            ("send → first LLM token (TTFT)", send_to_delta),
            ("first token → first sentence", delta_to_sentence),
            (f"TTS batch synth ({len(first_sentence_text)} chars → {size}B)", tts_time),
            ("TOTAL (send → audio ready)", total),
        ])

    @pytest.mark.asyncio
    async def test_end_to_end_latency_streaming_tts(self):
        """Full pipeline with streaming TTS: send → LLM → sentence → first audio chunk."""
        from reachy_claw.tts import KokoroTTS

        config = _make_config()
        client = DesktopRobotClient(config)
        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)

        if not tts.supports_streaming:
            pytest.skip("Streaming TTS not available")

        timestamps = {}
        first_sentence_text = None
        sentence_ready = asyncio.Event()
        buffer_holder = {"buf": ""}
        sentence_ends = {".", "!", "?"}

        async def on_delta(text, rid):
            nonlocal first_sentence_text
            if "first_delta" not in timestamps:
                timestamps["first_delta"] = time.perf_counter()
            buffer_holder["buf"] += text

            if first_sentence_text is None:
                buf = buffer_holder["buf"]
                for ch in sentence_ends:
                    pos = buf.find(ch)
                    if pos >= 3:
                        first_sentence_text = buf[: pos + 1].strip()
                        timestamps["first_sentence"] = time.perf_counter()
                        sentence_ready.set()
                        break

        stream_done = asyncio.Event()

        async def on_end(text, rid):
            nonlocal first_sentence_text
            timestamps["stream_end"] = time.perf_counter()
            if first_sentence_text is None:
                first_sentence_text = text.strip()
                timestamps["first_sentence"] = time.perf_counter()
                sentence_ready.set()
            stream_done.set()

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        timestamps["send"] = time.perf_counter()
        await client.send_message_streaming("Say exactly: Hello from Reachy Mini.")

        await asyncio.wait_for(sentence_ready.wait(), timeout=30.0)

        # Streaming TTS: measure time to first audio chunk
        timestamps["tts_start"] = time.perf_counter()
        chunk_count = 0
        total_samples = 0
        sample_rate = 16000

        async for chunk, sr in tts.synthesize_streaming(first_sentence_text):
            if "tts_first_chunk" not in timestamps:
                timestamps["tts_first_chunk"] = time.perf_counter()
            chunk_count += 1
            total_samples += len(chunk)
            sample_rate = sr

        timestamps["tts_done"] = time.perf_counter()

        await asyncio.wait_for(stream_done.wait(), timeout=30.0)
        await client.disconnect()

        send_to_delta = (timestamps["first_delta"] - timestamps["send"]) * 1000
        delta_to_sentence = (timestamps["first_sentence"] - timestamps["first_delta"]) * 1000
        tts_ttfc = (timestamps["tts_first_chunk"] - timestamps["tts_start"]) * 1000
        tts_total = (timestamps["tts_done"] - timestamps["tts_start"]) * 1000
        total_to_first_audio = (timestamps["tts_first_chunk"] - timestamps["send"]) * 1000
        total = (timestamps["tts_done"] - timestamps["send"]) * 1000
        audio_duration = total_samples / sample_rate

        _print_latency_table("Full Pipeline: Streaming TTS", [
            ("send → first LLM token (TTFT)", send_to_delta),
            ("first token → first sentence", delta_to_sentence),
            (f"TTS streaming TTFC ({len(first_sentence_text)} chars)", tts_ttfc),
            (f"TTS total ({chunk_count} chunks, {audio_duration:.2f}s audio)", tts_total),
            ("─" * 38, 0),
            ("TOTAL send → first audio chunk", total_to_first_audio),
            ("TOTAL send → all audio ready", total),
        ])

    @pytest.mark.asyncio
    async def test_stt_to_audio_full_chain(self):
        """Measure the complete chain: STT finalize + LLM TTFT + sentence + TTS TTFC.

        This is the true end-to-end: from when the user stops speaking
        until they would hear the first audio output.
        """
        from reachy_claw.stt import ParaformerStreamingSTT
        from reachy_claw.tts import KokoroTTS

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)
        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)

        config = _make_config()
        client = DesktopRobotClient(config)

        timestamps = {}

        # ── Stage 1: STT finalize ──
        stt.start_stream(sample_rate=16000)
        silence = np.zeros(1600, dtype=np.float32)
        for _ in range(10):
            stt.feed_chunk(silence)

        timestamps["stt_finalize_start"] = time.perf_counter()
        stt_text = stt.finish_stream()
        timestamps["stt_finalize_end"] = time.perf_counter()

        # Since silence produces no text, use a known input for the rest
        user_text = "Say exactly: Hello from Reachy."

        # ── Stage 2: Gateway + LLM ──
        first_sentence_text = None
        sentence_ready = asyncio.Event()
        buffer_holder = {"buf": ""}
        sentence_ends = {".", "!", "?"}

        async def on_delta(text, rid):
            nonlocal first_sentence_text
            if "first_delta" not in timestamps:
                timestamps["first_delta"] = time.perf_counter()
            buffer_holder["buf"] += text

            if first_sentence_text is None:
                buf = buffer_holder["buf"]
                for ch in sentence_ends:
                    pos = buf.find(ch)
                    if pos >= 3:
                        first_sentence_text = buf[: pos + 1].strip()
                        timestamps["first_sentence"] = time.perf_counter()
                        sentence_ready.set()
                        break

        stream_done = asyncio.Event()

        async def on_end(text, rid):
            nonlocal first_sentence_text
            if first_sentence_text is None:
                first_sentence_text = text.strip()
                timestamps["first_sentence"] = time.perf_counter()
                sentence_ready.set()
            stream_done.set()

        client.callbacks.on_stream_delta = on_delta
        client.callbacks.on_stream_end = on_end
        await client.connect()

        timestamps["send"] = time.perf_counter()
        await client.send_message_streaming(user_text)

        await asyncio.wait_for(sentence_ready.wait(), timeout=30.0)

        # ── Stage 3: TTS ──
        timestamps["tts_start"] = time.perf_counter()

        if tts.supports_streaming:
            async for chunk, sr in tts.synthesize_streaming(first_sentence_text):
                if "tts_first_audio" not in timestamps:
                    timestamps["tts_first_audio"] = time.perf_counter()
                break  # only need first chunk timing
        else:
            path = await tts.synthesize(first_sentence_text)
            timestamps["tts_first_audio"] = time.perf_counter()
            os.unlink(path)

        await asyncio.wait_for(stream_done.wait(), timeout=30.0)
        await client.disconnect()

        # ── Results ──
        stt_ms = (timestamps["stt_finalize_end"] - timestamps["stt_finalize_start"]) * 1000
        send_to_delta = (timestamps["first_delta"] - timestamps["send"]) * 1000
        delta_to_sentence = (timestamps["first_sentence"] - timestamps["first_delta"]) * 1000
        tts_ttfc = (timestamps["tts_first_audio"] - timestamps["tts_start"]) * 1000

        # Total: from STT finalize start to first audio
        # In real pipeline, STT finalize + gateway happen sequentially
        total_estimated = stt_ms + send_to_delta + delta_to_sentence + tts_ttfc

        _print_latency_table("FULL CHAIN: Stop Speaking → First Audio", [
            ("① STT finalize (silence → text)", stt_ms),
            ("② Gateway send → first LLM token", send_to_delta),
            ("③ First token → first sentence", delta_to_sentence),
            ("④ TTS → first audio chunk/file", tts_ttfc),
            ("─" * 38, 0),
            ("ESTIMATED TOTAL (①+②+③+④)", total_estimated),
        ])
