"""Tests for ElevenLabs configuration loading and API (mocked)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.elevenlabs import (
    DEFAULT_ELEVENLABS_VOICE_ID,
    ElevenLabsConfig,
    _accept_header_for_output_format,
    _suffix_for_output_format,
    _validate_voice_id,
    elevenlabs_tts_bytes,
    elevenlabs_tts_to_temp_audio_file,
    load_elevenlabs_config,
)


def test_load_config_uses_default_voice_id_when_env_missing(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("REACHY_ELEVENLABS_VOICE_ID", raising=False)

    config = load_elevenlabs_config()

    assert config.voice_id == DEFAULT_ELEVENLABS_VOICE_ID


def test_load_config_uses_env_voice_id_when_present(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "env-voice-id")
    monkeypatch.delenv("REACHY_ELEVENLABS_VOICE_ID", raising=False)

    config = load_elevenlabs_config()

    assert config.voice_id == "env-voice-id"


def test_load_config_prefers_reachy_prefixed_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "env-voice-id")
    monkeypatch.setenv("REACHY_ELEVENLABS_VOICE_ID", "reachy-voice-id")

    config = load_elevenlabs_config()

    assert config.voice_id == "reachy-voice-id"


def test_load_config_rejects_invalid_voice_id(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "../evil")
    monkeypatch.delenv("REACHY_ELEVENLABS_VOICE_ID", raising=False)

    with pytest.raises(ValueError, match="Invalid ElevenLabs voice id"):
        load_elevenlabs_config()


def test_load_config_strips_voice_id(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-api-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "  env-voice-id  ")
    monkeypatch.delenv("REACHY_ELEVENLABS_VOICE_ID", raising=False)

    config = load_elevenlabs_config()

    assert config.voice_id == "env-voice-id"


def test_load_config_defaults(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("REACHY_ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL_ID", raising=False)
    monkeypatch.delenv("REACHY_ELEVENLABS_MODEL_ID", raising=False)
    monkeypatch.delenv("ELEVENLABS_OUTPUT_FORMAT", raising=False)
    monkeypatch.delenv("REACHY_ELEVENLABS_OUTPUT_FORMAT", raising=False)

    config = load_elevenlabs_config()
    assert config.model_id == "eleven_multilingual_v2"
    assert config.output_format == "mp3_44100_128"


# ── Voice ID validation ──────────────────────────────────────────────────


class TestVoiceIdValidation:
    def test_valid_alphanumeric(self):
        assert _validate_voice_id("JBFqnCBsd6RMkjVDRZzb") == "JBFqnCBsd6RMkjVDRZzb"

    def test_valid_with_underscore_dash(self):
        assert _validate_voice_id("abc_123-XYZ") == "abc_123-XYZ"

    def test_strips_whitespace(self):
        assert _validate_voice_id("  abc  ") == "abc"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_voice_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_voice_id("   ")

    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_voice_id("../secret")

    def test_url_characters_blocked(self):
        with pytest.raises(ValueError, match="Invalid"):
            _validate_voice_id("voice?param=1")


# ── Output format helpers ────────────────────────────────────────────────


class TestOutputFormatHelpers:
    def test_wav_accept_header(self):
        assert _accept_header_for_output_format("wav_44100") == "audio/wav"

    def test_mp3_accept_header(self):
        assert _accept_header_for_output_format("mp3_44100_128") == "audio/mpeg"

    def test_wav_suffix(self):
        assert _suffix_for_output_format("wav_44100") == ".wav"

    def test_mp3_suffix(self):
        assert _suffix_for_output_format("mp3_44100_128") == ".mp3"


# ── API calls (mocked) ──────────────────────────────────────────────────


class TestElevenLabsAPI:
    @pytest.mark.asyncio
    async def test_tts_bytes_sends_correct_request(self):
        config = ElevenLabsConfig(
            api_key="test-key",
            voice_id="voice123",
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
        )

        mock_response = MagicMock()
        mock_response.content = b"fake-audio-bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("reachy_claw.elevenlabs.httpx.AsyncClient", return_value=mock_client):
            result = await elevenlabs_tts_bytes(text="Hello", config=config)

        assert result == b"fake-audio-bytes"
        call_args = mock_client.post.call_args
        assert "voice123" in call_args[0][0]
        headers = call_args.kwargs["headers"]
        assert headers["xi-api-key"] == "test-key"

    @pytest.mark.asyncio
    async def test_tts_bytes_empty_text_raises(self):
        config = ElevenLabsConfig(api_key="k", voice_id="v")
        with pytest.raises(ValueError, match="non-empty"):
            await elevenlabs_tts_bytes(text="", config=config)

    @pytest.mark.asyncio
    async def test_tts_bytes_whitespace_only_raises(self):
        config = ElevenLabsConfig(api_key="k", voice_id="v")
        with pytest.raises(ValueError, match="non-empty"):
            await elevenlabs_tts_bytes(text="   ", config=config)

    @pytest.mark.asyncio
    async def test_tts_to_temp_file_writes_correct_data(self):
        config = ElevenLabsConfig(
            api_key="test-key",
            voice_id="voice123",
            output_format="mp3_44100_128",
        )

        with patch(
            "reachy_claw.elevenlabs.elevenlabs_tts_bytes",
            new_callable=AsyncMock,
            return_value=b"fake-mp3-data",
        ):
            path = await elevenlabs_tts_to_temp_audio_file(
                text="Hello", config=config
            )

        try:
            assert path.endswith(".mp3")
            with open(path, "rb") as f:
                assert f.read() == b"fake-mp3-data"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_tts_to_temp_file_wav_format(self):
        config = ElevenLabsConfig(
            api_key="test-key",
            voice_id="voice123",
            output_format="wav_44100",
        )

        with patch(
            "reachy_claw.elevenlabs.elevenlabs_tts_bytes",
            new_callable=AsyncMock,
            return_value=b"fake-wav-data",
        ):
            path = await elevenlabs_tts_to_temp_audio_file(
                text="Hello", config=config
            )

        try:
            assert path.endswith(".wav")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_voice_settings_passed_through(self):
        config = ElevenLabsConfig(api_key="k", voice_id="v")

        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("reachy_claw.elevenlabs.httpx.AsyncClient", return_value=mock_client):
            await elevenlabs_tts_bytes(
                text="Hi",
                config=config,
                voice_settings={"use_speaker_boost": True},
            )

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["voice_settings"] == {"use_speaker_boost": True}
