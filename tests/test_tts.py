"""Tests for TTS backends (no speaker needed)."""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.tts import (
    ElevenLabsTTS,
    MacOSSayTTS,
    NoopTTS,
    PiperTTS,
    create_tts_backend,
)


# ── Factory ──────────────────────────────────────────────────────────────


class TestCreateTTSBackend:
    def test_creates_elevenlabs(self):
        with patch("reachy_claw.elevenlabs.load_elevenlabs_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            backend = create_tts_backend(backend="elevenlabs", voice="test-voice")
            assert isinstance(backend, ElevenLabsTTS)
            mock_cfg.assert_called_once_with(
                api_key=None, voice_id="test-voice", model_id=None, output_format=None
            )

    def test_creates_macos_say(self):
        backend = create_tts_backend(backend="macos-say", voice="Samantha")
        assert isinstance(backend, MacOSSayTTS)
        assert backend._voice == "Samantha"

    def test_creates_macos_say_alias(self):
        backend = create_tts_backend(backend="say")
        assert isinstance(backend, MacOSSayTTS)

    def test_creates_piper(self):
        backend = create_tts_backend(backend="piper", model="/path/to/model.onnx")
        assert isinstance(backend, PiperTTS)
        assert backend._model == "/path/to/model.onnx"

    def test_creates_noop(self):
        backend = create_tts_backend(backend="none")
        assert isinstance(backend, NoopTTS)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS backend"):
            create_tts_backend(backend="nonexistent")

    def test_case_insensitive(self):
        backend = create_tts_backend(backend="  NONE  ")
        assert isinstance(backend, NoopTTS)


# ── NoopTTS ──────────────────────────────────────────────────────────────


class TestNoopTTS:
    @pytest.mark.asyncio
    async def test_synthesize_returns_valid_wav(self):
        tts = NoopTTS()
        path = await tts.synthesize("Hello world")

        try:
            assert os.path.exists(path)
            assert path.endswith(".wav")

            # Verify it's a valid WAV
            with wave.open(path, "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getframerate() == 16000
                assert wf.getsampwidth() == 2
                assert wf.getnframes() == 0  # silent
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_synthesize_creates_unique_files(self):
        tts = NoopTTS()
        paths = []
        try:
            for _ in range(3):
                p = await tts.synthesize("test")
                paths.append(p)
            # All paths should be different
            assert len(set(paths)) == 3
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def test_cleanup_is_noop(self):
        tts = NoopTTS()
        tts.cleanup()  # should not raise


# ── MacOSSayTTS ──────────────────────────────────────────────────────────


class TestMacOSSayTTS:
    @pytest.mark.asyncio
    async def test_builds_correct_command(self):
        tts = MacOSSayTTS(voice="Alex", rate=200)

        with patch("reachy_claw.tts.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            path = await tts.synthesize("Hello")

            cmd = mock_sub.run.call_args[0][0]
            assert cmd[0] == "say"
            assert "-v" in cmd
            assert "Alex" in cmd
            assert "-r" in cmd
            assert "200" in cmd
            assert "--" in cmd
            assert "Hello" in cmd

        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.mark.asyncio
    async def test_default_voice_no_flags(self):
        tts = MacOSSayTTS()

        with patch("reachy_claw.tts.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            path = await tts.synthesize("Hi")

            cmd = mock_sub.run.call_args[0][0]
            assert "-v" not in cmd
            assert "-r" not in cmd

        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.mark.asyncio
    async def test_failure_raises(self):
        tts = MacOSSayTTS()

        with patch("reachy_claw.tts.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                returncode=1, stderr=b"say error"
            )
            with pytest.raises(RuntimeError, match="say failed"):
                await tts.synthesize("fail")


# ── PiperTTS ─────────────────────────────────────────────────────────────


class TestPiperTTS:
    @pytest.mark.asyncio
    async def test_requires_model(self):
        tts = PiperTTS(model="")
        with pytest.raises(ValueError, match="Piper model not set"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_builds_correct_command(self):
        tts = PiperTTS(model="/models/en.onnx", speaker=2)

        with patch("reachy_claw.tts.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0)
            path = await tts.synthesize("Hello piper")

            cmd = mock_sub.run.call_args[0][0]
            assert "piper" in cmd
            assert "--model" in cmd
            assert "/models/en.onnx" in cmd
            assert "--speaker" in cmd
            assert "2" in cmd
            # Text passed via stdin
            assert mock_sub.run.call_args.kwargs.get("input") == b"Hello piper" or \
                   mock_sub.run.call_args[1].get("input") == b"Hello piper"

        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    @pytest.mark.asyncio
    async def test_failure_raises(self):
        tts = PiperTTS(model="/models/en.onnx")

        with patch("reachy_claw.tts.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                returncode=1, stderr=b"piper error"
            )
            with pytest.raises(RuntimeError, match="piper failed"):
                await tts.synthesize("fail")
