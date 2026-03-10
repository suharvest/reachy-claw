"""End-to-end tests against a live Jetson speech service.

Requires the speech service running at SPEECH_SERVICE_URL (default: http://100.67.111.58:8000).
Skip automatically if not reachable.

Run with: uv run pytest tests/test_remote_e2e.py -v
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import wave

import numpy as np
import pytest

SPEECH_URL = os.environ.get("SPEECH_SERVICE_URL", "http://100.67.111.58:8000")
WS_URL = SPEECH_URL.replace("http://", "ws://").replace("https://", "wss://")


def _service_reachable() -> bool:
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        return data.get("tts", False)
    except Exception:
        return False


_skip_no_service = pytest.mark.skipif(
    not _service_reachable(),
    reason=f"Speech service not reachable at {SPEECH_URL}",
)


def _make_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 audio to WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (audio * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _make_sine_wav(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> bytes:
    """Generate a sine wave WAV for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return _make_wav_bytes(audio, sr)


# ── Health ──────────────────────────────────────────────────────────────


@_skip_no_service
class TestHealth:
    def test_health_endpoint(self):
        import urllib.request

        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        assert "tts" in data
        assert "asr" in data

    def test_health_streaming_asr(self):
        import urllib.request

        resp = urllib.request.urlopen(f"{SPEECH_URL}/health", timeout=5)
        data = json.loads(resp.read().decode())
        # streaming_asr may or may not be present
        if "streaming_asr" in data:
            assert isinstance(data["streaming_asr"], bool)


# ── Batch TTS (/tts) ───────────────────────────────────────────────────


@_skip_no_service
class TestBatchTTS:
    def test_synthesize_returns_wav(self):
        import urllib.request

        payload = json.dumps({"text": "Hello", "sid": 3, "speed": 1.0}).encode()
        req = urllib.request.Request(
            f"{SPEECH_URL}/tts",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        wav_bytes = resp.read()

        # Should be a valid WAV
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert len(wav_bytes) > 100

        # Check response headers
        duration = resp.headers.get("X-Audio-Duration")
        assert duration is not None
        assert float(duration) > 0

    def test_synthesize_chinese(self):
        import urllib.request

        payload = json.dumps({"text": "你好世界", "sid": 3, "speed": 1.0}).encode()
        req = urllib.request.Request(
            f"{SPEECH_URL}/tts",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        wav_bytes = resp.read()
        assert wav_bytes[:4] == b"RIFF"
        assert len(wav_bytes) > 100


# ── Streaming TTS (/tts/stream) ────────────────────────────────────────


@_skip_no_service
class TestStreamingTTS:
    def test_options_probe(self):
        import urllib.request

        req = urllib.request.Request(f"{SPEECH_URL}/tts/stream", method="OPTIONS")
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.status == 200

    def test_stream_returns_pcm(self):
        import urllib.request

        payload = json.dumps({"text": "Hello world", "sid": 3, "speed": 1.0}).encode()
        req = urllib.request.Request(
            f"{SPEECH_URL}/tts/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)

        # First 4 bytes: sample rate (uint32 LE)
        sr_bytes = resp.read(4)
        assert len(sr_bytes) == 4
        sample_rate = struct.unpack("<I", sr_bytes)[0]
        assert 8000 <= sample_rate <= 48000

        # Rest: int16 PCM chunks
        all_pcm = resp.read()
        assert len(all_pcm) > 0
        assert len(all_pcm) % 2 == 0  # int16 = 2 bytes per sample

        samples = np.frombuffer(all_pcm, dtype=np.int16)
        assert len(samples) > 100


# ── Batch ASR (/asr) ───────────────────────────────────────────────────


@_skip_no_service
class TestBatchASR:
    def test_transcribe_silence(self):
        """Silence should return empty or very short text."""
        import urllib.request

        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        wav_bytes = _make_wav_bytes(silence)

        boundary = "----TestBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{SPEECH_URL}/asr?language=auto",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        assert "text" in result

    def test_asr_endpoint_accepts_wav(self):
        """ASR endpoint should accept WAV and return JSON with text field."""
        import urllib.request

        # Generate a short sine wave (won't be meaningful speech, but tests the pipeline)
        wav_bytes = _make_sine_wav(duration=0.5)

        boundary = "----TestBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{SPEECH_URL}/asr?language=auto",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        assert "text" in result
        assert isinstance(result["text"], str)


# ── Streaming ASR (/asr/stream WebSocket) ──────────────────────────────


@_skip_no_service
class TestStreamingASR:
    def test_websocket_connect_and_finalize(self):
        """Connect, send silence, finalize — should get empty final result."""
        import websockets.sync.client

        ws = websockets.sync.client.connect(
            f"{WS_URL}/asr/stream?language=auto&sample_rate=16000",
            open_timeout=5,
        )

        # Send 0.5s of silence
        silence = np.zeros(8000, dtype=np.int16)
        ws.send(silence.tobytes())

        # Send empty bytes to finalize
        ws.send(b"")

        # Should get a final result
        raw = ws.recv(timeout=5)
        msg = json.loads(raw)
        assert msg["is_final"] is True
        assert isinstance(msg["text"], str)

        ws.close()

    def test_websocket_partial_results(self):
        """Send multiple chunks, check we get JSON responses."""
        import websockets.sync.client

        ws = websockets.sync.client.connect(
            f"{WS_URL}/asr/stream?language=auto&sample_rate=16000",
            open_timeout=5,
        )

        # Send several chunks of sine wave (simulates speech-like signal)
        t = np.linspace(0, 0.5, 8000, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)

        chunk_size = 1600  # 100ms chunks
        for i in range(0, len(audio), chunk_size):
            ws.send(audio[i:i + chunk_size].tobytes())

        # Finalize
        ws.send(b"")

        raw = ws.recv(timeout=5)
        msg = json.loads(raw)
        assert "text" in msg
        assert "is_final" in msg

        ws.close()


# ── Client-side backend integration ───────────────────────────────────


@_skip_no_service
class TestClientBackends:
    def test_kokoro_tts_batch(self):
        """KokoroTTS batch synthesize against live service."""
        from reachy_claw.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)

        async def _run():
            path = await tts.synthesize("Hello")
            try:
                assert os.path.exists(path)
                assert os.path.getsize(path) > 100
                with wave.open(path, "rb") as wf:
                    assert wf.getnframes() > 0
            finally:
                os.unlink(path)

        asyncio.run(_run())

    def test_kokoro_tts_streaming(self):
        """KokoroTTS streaming synthesize against live service."""
        from reachy_claw.tts import KokoroTTS

        tts = KokoroTTS(base_url=SPEECH_URL, speaker_id=3, speed=1.0)
        if not tts.supports_streaming:
            pytest.skip("Streaming TTS not available on server")

        async def _run():
            chunks = []
            sample_rate = None
            async for audio_chunk, sr in tts.synthesize_streaming("Hello world"):
                chunks.append(audio_chunk)
                sample_rate = sr

            assert len(chunks) > 0
            assert sample_rate is not None
            assert 8000 <= sample_rate <= 48000
            total_samples = sum(len(c) for c in chunks)
            assert total_samples > 100

        asyncio.run(_run())

    def test_paraformer_streaming_stt_batch_fallback(self):
        """ParaformerStreamingSTT batch transcribe against live service."""
        from reachy_claw.stt import ParaformerStreamingSTT

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)
        stt.preload()

        # Transcribe silence — should return empty/short text
        silence = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(silence, sample_rate=16000)
        assert isinstance(result, str)

    def test_paraformer_streaming_stt_websocket(self):
        """ParaformerStreamingSTT streaming interface against live service."""
        from reachy_claw.stt import ParaformerStreamingSTT

        stt = ParaformerStreamingSTT(base_url=SPEECH_URL)

        stt.start_stream(sample_rate=16000)

        # Feed 0.5s of silence in chunks
        chunk_size = 1600  # 100ms
        silence = np.zeros(chunk_size, dtype=np.float32)
        for _ in range(5):
            stt.feed_chunk(silence)

        result = stt.finish_stream()
        assert isinstance(result, str)



class TestFallbackBehavior:
    """These tests don't need the speech service — they test fallback when unreachable."""

    def test_fallback_stt_when_unreachable(self):
        """STT factory should fall back to whisper when remote is unreachable."""
        from reachy_claw.config import Config
        from reachy_claw.stt import WhisperSTT, create_stt_backend

        config = Config(
            stt_backend="paraformer-streaming",
            speech_service_url="http://192.0.2.1:9999",  # unreachable
            whisper_model="tiny",
        )
        backend = create_stt_backend(config)
        assert isinstance(backend, WhisperSTT)

    def test_fallback_tts_when_unreachable(self):
        """TTS factory should fall back to say/none when kokoro is unreachable."""
        import platform

        from reachy_claw.config import Config
        from reachy_claw.tts import MacOSSayTTS, NoopTTS, create_tts_backend

        config = Config(speech_service_url="http://192.0.2.1:9999")
        backend = create_tts_backend(
            backend="kokoro",
            config=config,
        )
        if platform.system() == "Darwin":
            assert isinstance(backend, MacOSSayTTS)
        else:
            assert isinstance(backend, NoopTTS)
