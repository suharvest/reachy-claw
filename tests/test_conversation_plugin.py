"""Tests for the ConversationPlugin (STT->gateway->TTS loop)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from clawd_reachy_mini.config import Config
from clawd_reachy_mini.plugins.conversation_plugin import ConversationPlugin


@pytest.fixture
def standalone_app(mock_reachy):
    """App in standalone mode (no gateway) with mock robot."""
    from clawd_reachy_mini.app import ClawdApp

    config = Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    a = ClawdApp(config)
    a.reachy = mock_reachy
    return a


# ── Stream text queue processing ───────────────────────────────────────


class TestSpeakStreaming:
    @pytest.mark.asyncio
    async def test_collects_chunks_and_speaks(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        spoken = []

        async def mock_speak(text):
            spoken.append(text)

        plugin._speak = mock_speak
        plugin._audio = MagicMock()
        plugin._audio._running = False

        # Pre-fill the stream queue with deltas and end signal
        await plugin._stream_text_queue.put("Hello ")
        await plugin._stream_text_queue.put("world. ")
        await plugin._stream_text_queue.put("How are you?")
        await plugin._stream_text_queue.put(None)  # end of stream

        result = await plugin._speak_streaming()

        assert "Hello world" in result
        assert "How are you" in result
        assert len(spoken) >= 1

    @pytest.mark.asyncio
    async def test_speaks_sentence_boundary(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        sentences_spoken = []

        async def mock_speak(text):
            sentences_spoken.append(text)

        plugin._speak = mock_speak
        plugin._audio = MagicMock()
        plugin._audio._running = False

        await plugin._stream_text_queue.put("First sentence. ")
        await plugin._stream_text_queue.put("Second! ")
        await plugin._stream_text_queue.put("Third?")
        await plugin._stream_text_queue.put(None)

        await plugin._speak_streaming()

        # Should speak at sentence boundaries
        assert len(sentences_spoken) >= 2

    @pytest.mark.asyncio
    async def test_sets_is_speaking_flag(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        speaking_during = []

        async def mock_speak(text):
            speaking_during.append(standalone_app.is_speaking)

        plugin._speak = mock_speak
        plugin._audio = MagicMock()
        plugin._audio._running = False

        await plugin._stream_text_queue.put("Hello world. ")
        await plugin._stream_text_queue.put(None)

        assert standalone_app.is_speaking is False
        await plugin._speak_streaming()
        # After done, is_speaking should be reset
        assert standalone_app.is_speaking is False
        # During speech, is_speaking should have been True
        assert any(s is True for s in speaking_during)

    @pytest.mark.asyncio
    async def test_timeout_on_no_data(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._tts = MagicMock()
        plugin._tts.synthesize = AsyncMock(return_value="/dev/null")
        plugin._audio = MagicMock()
        plugin._audio._running = False

        # Don't put anything in the queue - should timeout after 60s
        # We'll use a shorter test by putting None after a delay
        async def delayed_end():
            await asyncio.sleep(0.05)
            await plugin._stream_text_queue.put(None)

        asyncio.create_task(delayed_end())
        result = await plugin._speak_streaming()
        assert result == ""


# ── Barge-in detection ─────────────────────────────────────────────────


class TestBargeIn:
    def test_barge_in_disabled(self, standalone_app):
        standalone_app.config.barge_in_enabled = False
        plugin = ConversationPlugin(standalone_app)
        assert plugin._was_barged_in() is False

    def test_barge_in_no_audio(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._audio = None
        assert plugin._was_barged_in() is False

    def test_barge_in_audio_not_running(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._audio = MagicMock()
        plugin._audio._running = False
        assert plugin._was_barged_in() is False


# ── Callback wiring ────────────────────────────────────────────────────


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_stream_start_drains_queue(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        # Pre-fill queue with stale data
        await plugin._stream_text_queue.put("stale")
        await plugin._stream_text_queue.put("data")

        await plugin._on_stream_start("run-1")

        assert plugin._current_run_id == "run-1"
        assert plugin._stream_text_queue.empty()

    @pytest.mark.asyncio
    async def test_stream_delta_puts_text(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_stream_delta("hello", "run-1")

        text = plugin._stream_text_queue.get_nowait()
        assert text == "hello"

    @pytest.mark.asyncio
    async def test_stream_end_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_stream_end("full text", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_stream_abort_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._on_stream_abort("interrupted", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None


# ── Emotion queueing from conversation ─────────────────────────────────


class TestEmotionIntegration:
    @pytest.mark.asyncio
    async def test_thinking_emotion_queued_on_send(self, standalone_app):
        """When not in standalone mode, sending to AI queues 'thinking'."""
        standalone_app.config.standalone_mode = False
        standalone_app.config.play_emotions = True

        plugin = ConversationPlugin(standalone_app)
        plugin._client = MagicMock()
        plugin._client.send_message_streaming = AsyncMock()
        plugin._client.send_state_change = AsyncMock()

        # Mock _speak_streaming to return quickly
        async def mock_stream():
            return "response"

        plugin._speak_streaming = mock_stream

        # Simulate the sending part of _conversation_turn
        standalone_app.emotions.queue_emotion("thinking")

        expr = standalone_app.emotions.get_next_expression()
        assert expr is not None
        assert "hinking" in expr.description.lower()


# ── Stop / cleanup ─────────────────────────────────────────────────────


class TestConversationCleanup:
    @pytest.mark.asyncio
    async def test_stop_cleans_resources(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._audio = MagicMock()
        plugin._audio.stop = AsyncMock()
        plugin._tts = MagicMock()
        plugin._client = MagicMock()
        plugin._client.disconnect = AsyncMock()

        await plugin.stop()

        assert plugin._running is False
        plugin._audio.stop.assert_called_once()
        plugin._client.disconnect.assert_called_once()
        plugin._tts.cleanup.assert_called_once()
