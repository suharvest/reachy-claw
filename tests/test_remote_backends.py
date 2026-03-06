"""Tests for remote ASR/TTS backends (mocked HTTP, no hardware needed)."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clawd_reachy_mini.config import Config
from clawd_reachy_mini.stt import SenseVoiceSTT, create_stt_backend
from clawd_reachy_mini.tts import KokoroTTS, create_tts_backend


# ── SenseVoiceSTT ───────────────────────────────────────────────────────


class TestSenseVoiceSTT:
    def test_init(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000", language="zh")
        assert stt._base_url == "http://jetson:8000"
        assert stt._language == "zh"

    def test_trailing_slash_stripped(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000/")
        assert stt._base_url == "http://jetson:8000"

    def test_preload_health_check(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"asr": true}'
        with patch("urllib.request.urlopen", return_value=mock_resp):
            stt.preload()  # should not raise

    def test_preload_unreachable_warns(self):
        stt = SenseVoiceSTT(base_url="http://unreachable:9999")
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            stt.preload()  # should not raise, just warn

    def test_transcribe_posts_wav(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000", language="auto")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"text": "hello world"}).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            audio = np.array([0.1, -0.2, 0.5], dtype=np.float32)
            result = stt.transcribe(audio, sample_rate=16000)

            assert result == "hello world"
            req = mock_open.call_args[0][0]
            assert "/asr" in req.full_url
            assert "language=auto" in req.full_url

    def test_transcribe_file(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"text": "from file"}).encode()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav")
            path = Path(f.name)

        try:
            with patch("urllib.request.urlopen", return_value=mock_resp):
                result = stt.transcribe_file(path)
                assert result == "from file"
        finally:
            path.unlink()

    def test_transcribe_strips_whitespace(self):
        stt = SenseVoiceSTT(base_url="http://jetson:8000")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"text": "  spaced  "}).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = stt.transcribe(np.zeros(100, dtype=np.float32))
            assert result == "spaced"


# ── Factory: SenseVoice ─────────────────────────────────────────────────


class TestSenseVoiceFactory:
    def test_creates_sensevoice(self):
        config = Config(stt_backend="sensevoice")
        backend = create_stt_backend(config)
        assert isinstance(backend, SenseVoiceSTT)
        assert backend._base_url == "http://localhost:8000"

    def test_sensevoice_custom_url(self):
        config = Config(
            stt_backend="sensevoice",
            speech_service_url="http://jetson:8000",
            sensevoice_language="zh",
        )
        backend = create_stt_backend(config)
        assert isinstance(backend, SenseVoiceSTT)
        assert backend._base_url == "http://jetson:8000"
        assert backend._language == "zh"


# ── KokoroTTS ───────────────────────────────────────────────────────────


class TestKokoroTTS:
    def test_init(self):
        tts = KokoroTTS(base_url="http://jetson:8000", speaker_id=45, speed=0.8)
        assert tts._base_url == "http://jetson:8000"
        assert tts._speaker_id == 45
        assert tts._speed == 0.8

    @pytest.mark.asyncio
    async def test_synthesize_returns_wav_path(self):
        tts = KokoroTTS(base_url="http://jetson:8000")
        mock_resp = MagicMock()
        # Minimal WAV bytes
        wav_bytes = b"RIFF" + struct.pack("<I", 36) + b"WAVE"
        wav_bytes += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        wav_bytes += b"data" + struct.pack("<I", 0)
        mock_resp.read.return_value = wav_bytes

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            path = await tts.synthesize("Hello world")

            try:
                assert os.path.exists(path)
                assert path.endswith(".wav")
                req = mock_open.call_args[0][0]
                assert "/tts" in req.full_url
                body = json.loads(req.data.decode())
                assert body["text"] == "Hello world"
                assert body["sid"] == 3
                assert body["speed"] == 1.0
            finally:
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_synthesize_custom_params(self):
        tts = KokoroTTS(base_url="http://jetson:8000", speaker_id=45, speed=1.5)
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"fake wav"

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            path = await tts.synthesize("Test")

            try:
                req = mock_open.call_args[0][0]
                body = json.loads(req.data.decode())
                assert body["sid"] == 45
                assert body["speed"] == 1.5
            finally:
                os.unlink(path)


# ── Factory: KokoroTTS ──────────────────────────────────────────────────


class TestKokoroTTSFactory:
    def test_creates_kokoro(self):
        config = Config(speech_service_url="http://jetson:8000", kokoro_speaker_id=45)
        backend = create_tts_backend(backend="kokoro", config=config)
        assert isinstance(backend, KokoroTTS)
        assert backend._base_url == "http://jetson:8000"
        assert backend._speaker_id == 45

    def test_creates_kokoro_defaults(self):
        backend = create_tts_backend(backend="kokoro")
        assert isinstance(backend, KokoroTTS)
        assert backend._base_url == "http://localhost:8000"
        assert backend._speaker_id == 3


# ── Config fields ───────────────────────────────────────────────────────


class TestConfigSpeechFields:
    def test_defaults(self):
        config = Config()
        assert config.speech_service_url == "http://localhost:8000"
        assert config.sensevoice_language == "auto"
        assert config.kokoro_speaker_id == 3
        assert config.kokoro_speed == 1.2

    def test_env_override(self):
        with patch.dict(os.environ, {"SPEECH_SERVICE_URL": "http://custom:9000"}):
            config = Config()
            assert config.speech_service_url == "http://custom:9000"
