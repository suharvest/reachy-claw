"""Tests for STT backends (no microphone needed)."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.stt import (
    WhisperSTT,
    FasterWhisperSTT,
    OpenAISTT,
    create_stt_backend,
)


# ── Factory ──────────────────────────────────────────────────────────────


class TestCreateSTTBackend:
    def test_creates_whisper(self):
        config = Config(stt_backend="whisper", whisper_model="tiny")
        backend = create_stt_backend(config)
        assert isinstance(backend, WhisperSTT)
        assert backend.model_name == "tiny"

    def test_creates_faster_whisper(self):
        config = Config(stt_backend="faster-whisper", whisper_model="small")
        backend = create_stt_backend(config)
        assert isinstance(backend, FasterWhisperSTT)
        assert backend.model_name == "small"

    def test_creates_openai(self):
        config = Config(stt_backend="openai", openai_api_key="test-key")
        backend = create_stt_backend(config)
        assert isinstance(backend, OpenAISTT)

    def test_openai_requires_key(self):
        config = Config(stt_backend="openai", openai_api_key=None)
        with pytest.raises(ValueError, match="API key"):
            create_stt_backend(config)

    def test_unknown_backend_raises(self):
        config = Config(stt_backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown STT backend"):
            create_stt_backend(config)


# ── Whisper audio normalization ──────────────────────────────────────────


class TestWhisperNormalization:
    def test_float32_passthrough(self):
        """float32 audio in [0,1] range should not be re-normalized."""
        stt = WhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello"}
        stt._model = mock_model

        audio = np.array([0.1, -0.2, 0.5], dtype=np.float32)
        stt.transcribe(audio)

        called_audio = mock_model.transcribe.call_args[0][0]
        assert called_audio.dtype == np.float32
        np.testing.assert_array_almost_equal(called_audio, audio)

    def test_int16_converted_to_float32(self):
        """int16 audio should be converted to float32 and normalized."""
        stt = WhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "world"}
        stt._model = mock_model

        audio = np.array([16384, -16384, 0], dtype=np.int16)
        stt.transcribe(audio)

        called_audio = mock_model.transcribe.call_args[0][0]
        assert called_audio.dtype == np.float32
        assert called_audio.max() <= 1.0

    def test_high_amplitude_float_normalized(self):
        """float32 audio with values > 1.0 should be divided by 32768."""
        stt = WhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        stt._model = mock_model

        audio = np.array([16384.0, -16384.0], dtype=np.float32)
        stt.transcribe(audio)

        called_audio = mock_model.transcribe.call_args[0][0]
        assert called_audio.max() <= 1.0
        assert called_audio[0] == pytest.approx(0.5, abs=0.001)

    def test_transcribe_strips_whitespace(self):
        stt = WhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  hello world  "}
        stt._model = mock_model

        result = stt.transcribe(np.zeros(100, dtype=np.float32))
        assert result == "hello world"

    def test_transcribe_file(self):
        stt = WhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "from file"}
        stt._model = mock_model

        result = stt.transcribe_file(Path("/fake/audio.wav"))
        mock_model.transcribe.assert_called_once_with("/fake/audio.wav", fp16=False)
        assert result == "from file"


# ── FasterWhisper ────────────────────────────────────────────────────────


class TestFasterWhisperNormalization:
    def test_segments_joined(self):
        stt = FasterWhisperSTT(model_name="base")
        seg1 = MagicMock()
        seg1.text = "hello"
        seg2 = MagicMock()
        seg2.text = "world"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        stt._model = mock_model

        result = stt.transcribe(np.zeros(100, dtype=np.float32))
        assert result == "hello world"

    def test_int16_normalized(self):
        stt = FasterWhisperSTT(model_name="base")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        stt._model = mock_model

        audio = np.array([32767, -32768], dtype=np.int16)
        stt.transcribe(audio)

        called_audio = mock_model.transcribe.call_args[0][0]
        assert called_audio.dtype == np.float32
        assert called_audio.max() <= 1.0


# ── OpenAI STT WAV encoding ─────────────────────────────────────────────


class TestOpenAISTTEncoding:
    def test_transcribe_creates_wav_and_calls_api(self):
        stt = OpenAISTT(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "  transcribed text  "
        mock_client.audio.transcriptions.create.return_value = mock_response
        stt._client = mock_client

        audio = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        result = stt.transcribe(audio, sample_rate=16000)

        assert result == "transcribed text"
        # Verify API was called
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"

    def test_transcribe_file_directly(self):
        stt = OpenAISTT(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "file result"
        mock_client.audio.transcriptions.create.return_value = mock_response
        stt._client = mock_client

        # Create a real temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav data")
            path = Path(f.name)

        try:
            result = stt.transcribe_file(path)
            assert result == "file result"
        finally:
            path.unlink()


# ── Preload ──────────────────────────────────────────────────────────────


class TestPreload:
    def test_whisper_preload_loads_model(self):
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = MagicMock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            stt = WhisperSTT(model_name="tiny")
            stt.preload()
            mock_whisper.load_model.assert_called_once_with("tiny")

    def test_faster_whisper_preload_loads_model(self):
        mock_fw = MagicMock()
        mock_fw.WhisperModel.return_value = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            stt = FasterWhisperSTT(model_name="base")
            stt.preload()
            mock_fw.WhisperModel.assert_called_once_with("base", compute_type="int8")
