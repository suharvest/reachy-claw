"""Tests for AudioCapture and WakeWordDetector (no microphone needed)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from clawd_reachy_mini.audio import AudioCapture, WakeWordDetector
from clawd_reachy_mini.config import Config


# ── WakeWordDetector ─────────────────────────────────────────────────────


class TestWakeWordDetector:
    def test_detects_exact_match(self):
        d = WakeWordDetector("hey robot")
        assert d.detect("hey robot") is True

    def test_detects_case_insensitive(self):
        d = WakeWordDetector("Hey Robot")
        assert d.detect("HEY ROBOT how are you") is True

    def test_detects_substring(self):
        d = WakeWordDetector("robot")
        assert d.detect("hello robot world") is True

    def test_rejects_no_match(self):
        d = WakeWordDetector("alexa")
        assert d.detect("hey siri") is False

    def test_rejects_empty_text(self):
        d = WakeWordDetector("robot")
        assert d.detect("") is False

    def test_rejects_partial_match(self):
        d = WakeWordDetector("robot")
        assert d.detect("robo") is False


# ── AudioCapture lifecycle ───────────────────────────────────────────────


class TestAudioCaptureLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        config = Config()
        ac = AudioCapture(config)
        await ac.start()
        assert ac._running is True
        await ac.stop()
        assert ac._running is False

    @pytest.mark.asyncio
    async def test_capture_returns_none_when_stopped(self):
        config = Config()
        ac = AudioCapture(config)
        # Not started, _running is False
        result = await ac.capture_utterance()
        assert result is None

    def test_find_device_with_no_sounddevice(self):
        config = Config(audio_device="fake device")
        with patch.dict("sys.modules", {"sounddevice": None}):
            ac = AudioCapture(config)
            assert ac._device_id is None


# ── Speech/silence detection logic ───────────────────────────────────────


class TestSpeechDetection:
    @pytest.mark.asyncio
    async def test_detects_speech_then_silence(self):
        """Synthetic audio: loud frames then silent frames -> returns audio."""
        config = Config(
            silence_threshold=0.01,
            silence_duration=0.1,  # ~1.5 silence frames at 1024/16000
            sample_rate=16000,
            max_recording_duration=2.0,
        )
        ac = AudioCapture(config, reachy_mini=None)
        ac._running = True

        # Generate synthetic chunks: 10 loud + 10 silent
        loud_chunk = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        silent_chunk = np.zeros(1024, dtype=np.float32)

        chunks = [loud_chunk] * 10 + [silent_chunk] * 10
        chunk_iter = iter(chunks)

        async def mock_read_mic(frames):
            try:
                return next(chunk_iter)
            except StopIteration:
                return None

        ac._read_local_mic = mock_read_mic

        audio = await ac.capture_utterance()

        assert audio is not None
        assert len(audio) > 0
        # Should contain at least the loud frames
        assert len(audio) >= 1024 * 5

    @pytest.mark.asyncio
    async def test_no_speech_returns_none(self):
        """Only silent frames -> returns None."""
        config = Config(
            silence_threshold=0.01,
            silence_duration=0.05,
            sample_rate=16000,
            max_recording_duration=0.5,
        )
        ac = AudioCapture(config, reachy_mini=None)
        ac._running = True

        silent_chunk = np.zeros(1024, dtype=np.float32)
        count = [0]

        async def mock_read_mic(frames):
            count[0] += 1
            if count[0] > 50:
                ac._running = False
                return None
            return silent_chunk

        ac._read_local_mic = mock_read_mic

        audio = await ac.capture_utterance()
        assert audio is None

    @pytest.mark.asyncio
    async def test_max_recording_duration_limits_capture(self):
        """Should stop after max_recording_duration."""
        config = Config(
            silence_threshold=0.001,
            silence_duration=999,  # never trigger silence end
            sample_rate=16000,
            max_recording_duration=0.2,
        )
        ac = AudioCapture(config, reachy_mini=None)
        ac._running = True

        loud_chunk = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)

        async def mock_read_mic(frames):
            return loud_chunk

        ac._read_local_mic = mock_read_mic

        audio = await ac.capture_utterance()

        assert audio is not None
        max_samples = int(config.max_recording_duration * config.sample_rate)
        # Should not exceed max frames (max_recording_duration * sample_rate / 1024 chunks)
        max_chunks = int(max_samples / 1024)
        assert len(audio) <= (max_chunks + 1) * 1024

    @pytest.mark.asyncio
    async def test_reachy_media_used_when_available(self):
        """When reachy.media is available and no custom device, use it."""
        config = Config(
            silence_threshold=0.01,
            silence_duration=0.05,
            sample_rate=16000,
            max_recording_duration=0.5,
        )
        mock_reachy = MagicMock()
        mock_reachy.media = MagicMock()

        # Return loud then silent
        call_count = [0]
        loud = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        silent = np.zeros(1024, dtype=np.float32)

        def get_sample():
            call_count[0] += 1
            if call_count[0] <= 5:
                return loud
            if call_count[0] <= 10:
                return silent
            return None

        mock_reachy.media.get_audio_sample = get_sample

        ac = AudioCapture(config, reachy_mini=mock_reachy)
        ac._running = True

        audio = await ac.capture_utterance()

        mock_reachy.media.start_recording.assert_called_once()
        mock_reachy.media.stop_recording.assert_called_once()


# ── Energy calculation ───────────────────────────────────────────────────


class TestEnergyCalculation:
    def test_silence_below_threshold(self):
        config = Config(silence_threshold=0.01)
        audio = np.zeros(1024, dtype=np.float32)
        energy = np.abs(audio).mean()
        assert energy < config.silence_threshold

    def test_speech_above_threshold(self):
        config = Config(silence_threshold=0.01)
        audio = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        energy = np.abs(audio).mean()
        assert energy > config.silence_threshold

    def test_quiet_speech_near_threshold(self):
        config = Config(silence_threshold=0.01)
        audio = np.full(1024, 0.015, dtype=np.float32)
        energy = np.abs(audio).mean()
        assert energy > config.silence_threshold


# ── Continuous capture ────────────────────────────────────────────────────


class TestContinuousCapture:
    @pytest.mark.asyncio
    async def test_start_continuous_sets_running(self):
        config = Config()
        ac = AudioCapture(config)
        # Mock _read_local_mic to avoid actual mic access
        ac._read_local_mic = AsyncMock(return_value=np.zeros(1024, dtype=np.float32))
        await ac.start_continuous()
        assert ac._running is True
        assert ac._continuous is True
        await ac.stop()
        assert ac._running is False

    @pytest.mark.asyncio
    async def test_start_continuous_with_reachy(self):
        config = Config()
        mock_reachy = MagicMock()
        mock_reachy.media = MagicMock()
        ac = AudioCapture(config, reachy_mini=mock_reachy)
        await ac.start_continuous()
        mock_reachy.media.start_recording.assert_called_once()
        await ac.stop()
        mock_reachy.media.stop_recording.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_chunk_returns_data(self):
        config = Config()
        ac = AudioCapture(config)
        ac._running = True
        expected = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        ac._read_local_mic = AsyncMock(return_value=expected)
        chunk = await ac.read_chunk(1024)
        assert chunk is not None
        np.testing.assert_array_equal(chunk, expected)

    @pytest.mark.asyncio
    async def test_read_chunk_returns_none_when_stopped(self):
        config = Config()
        ac = AudioCapture(config)
        ac._running = False
        result = await ac.read_chunk()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_chunk_with_reachy_media(self):
        config = Config()
        mock_reachy = MagicMock()
        mock_reachy.media = MagicMock()
        expected = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        mock_reachy.media.get_audio_sample = MagicMock(return_value=expected)
        ac = AudioCapture(config, reachy_mini=mock_reachy)
        ac._running = True
        chunk = await ac.read_chunk()
        assert chunk is not None
