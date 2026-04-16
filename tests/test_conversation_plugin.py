"""Tests for the ConversationPlugin (dual-pipeline architecture)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.plugins.conversation_plugin import (
    ConversationPlugin,
    ConvState,
    SentenceItem,
    _drain_queue,
    _RESET_BUFFER,
)


@pytest.fixture
def standalone_app(mock_reachy):
    """App in standalone mode (no gateway) with mock robot."""
    from reachy_claw.app import ReachyClawApp

    config = Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    a = ReachyClawApp(config)
    a.reachy = mock_reachy
    return a


# ── ConvState enum ────────────────────────────────────────────────────


class TestConvState:
    def test_all_states_exist(self):
        assert ConvState.IDLE.value == "idle"
        assert ConvState.LISTENING.value == "listening"
        assert ConvState.TRANSCRIBING.value == "transcribing"
        assert ConvState.THINKING.value == "thinking"
        assert ConvState.SPEAKING.value == "speaking"


# ── SentenceItem ──────────────────────────────────────────────────────


class TestSentenceItem:
    def test_defaults(self):
        item = SentenceItem(text="Hello.")
        assert item.text == "Hello."
        assert item.is_last is False

    def test_is_last(self):
        item = SentenceItem(text="Done.", is_last=True)
        assert item.is_last is True


# ── drain_queue helper ────────────────────────────────────────────────


class TestDrainQueue:
    @pytest.mark.asyncio
    async def test_drains_all_items(self):
        q: asyncio.Queue[str] = asyncio.Queue()
        await q.put("a")
        await q.put("b")
        await q.put("c")
        _drain_queue(q)
        assert q.empty()

    @pytest.mark.asyncio
    async def test_noop_on_empty_queue(self):
        q: asyncio.Queue[str] = asyncio.Queue()
        _drain_queue(q)
        assert q.empty()


# ── Sentence accumulator ─────────────────────────────────────────────


class TestSentenceAccumulator:
    @pytest.mark.asyncio
    async def test_splits_on_period(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        await plugin._stream_text_queue.put("Hello world. ")
        await plugin._stream_text_queue.put("How are you?")
        await plugin._stream_text_queue.put(None)

        # Run accumulator briefly
        task = asyncio.create_task(plugin._sentence_accumulator())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sentences = []
        while not plugin._sentence_queue.empty():
            sentences.append(plugin._sentence_queue.get_nowait())

        # Should get at least "Hello world." and "How are you?" (or combined final)
        texts = [s.text for s in sentences if s.text]
        assert any("Hello world" in t for t in texts)

    @pytest.mark.asyncio
    async def test_flushes_buffer_on_stream_end(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        await plugin._stream_text_queue.put("Short")
        await plugin._stream_text_queue.put(None)

        task = asyncio.create_task(plugin._sentence_accumulator())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sentences = []
        while not plugin._sentence_queue.empty():
            sentences.append(plugin._sentence_queue.get_nowait())

        # "Short" is too short for sentence split, should flush as is_last
        assert any(s.is_last for s in sentences)
        assert any("Short" in s.text for s in sentences)


# ── Output pipeline ──────────────────────────────────────────────────


class TestOutputPipeline:
    @pytest.mark.asyncio
    async def test_speaks_sentences(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Hello world."), None))
        await plugin._audio_queue.put((SentenceItem(text="Done.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        # Wait long enough for both sentences + inter-sentence pause (0.15s)
        await asyncio.sleep(0.5)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "Hello world." in spoken
        assert "Done." in spoken

    @pytest.mark.asyncio
    async def test_interrupt_drains_queue(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            # Simulate interrupt after first sentence
            plugin._interrupt_event.set()
            return True

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="First."), None))
        await plugin._audio_queue.put((SentenceItem(text="Second."), None))
        await plugin._audio_queue.put((SentenceItem(text="Third.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only first sentence spoken, rest drained
        assert len(spoken) == 1
        assert spoken[0] == "First."

    @pytest.mark.asyncio
    async def test_sets_is_speaking_flag(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        speaking_during = []

        async def mock_speak(text, prefetched_chunks=None):
            speaking_during.append(standalone_app.is_speaking)
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Hello.", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert any(s is True for s in speaking_during)
        # After done, is_speaking should be reset
        assert standalone_app.is_speaking is False

    @pytest.mark.asyncio
    async def test_empty_is_last_finishes_speaking(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True

        await plugin._audio_queue.put((SentenceItem(text="", is_last=True), None))

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.15)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert standalone_app.is_speaking is False
        assert plugin._state == ConvState.IDLE


# ── Callback wiring ──────────────────────────────────────────────────


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_stream_start_drains_queue(self, standalone_app):
        from reachy_claw.plugins.conversation_plugin import _RESET_BUFFER

        plugin = ConversationPlugin(standalone_app)
        await plugin._stream_text_queue.put("stale")
        await plugin._stream_text_queue.put("data")

        await plugin._on_stream_start("run-1")

        assert plugin._current_run_id == "run-1"
        # Stale data drained, only the reset sentinel remains
        assert plugin._stream_text_queue.qsize() == 1
        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is _RESET_BUFFER
        assert plugin._state == ConvState.THINKING

    @pytest.mark.asyncio
    async def test_stream_delta_puts_text(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.THINKING
        plugin._current_run_id = "run-1"
        await plugin._on_stream_delta("hello", "run-1")

        text = plugin._stream_text_queue.get_nowait()
        assert text == "hello"
        # State stays THINKING — transition to SPEAKING happens in _output_pipeline
        # when audio actually starts playing.
        assert plugin._state == ConvState.THINKING

    @pytest.mark.asyncio
    async def test_stream_end_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-1"
        await plugin._on_stream_end("full text", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_stream_abort_puts_none(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-1"
        await plugin._on_stream_abort("interrupted", "run-1")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_task_completed_queues_deferred_announcement(
        self, standalone_app
    ):
        plugin = ConversationPlugin(standalone_app)
        # Plugin has no client and is not IDLE, so announcement should be deferred
        plugin._state = ConvState.SPEAKING
        await plugin._on_task_completed("Background search finished", "task-1")

        assert len(plugin._pending_announcements) == 1


# ── Fire interrupt ────────────────────────────────────────────────────


class TestFireInterrupt:
    @pytest.mark.asyncio
    async def test_sets_event_and_drains_queues(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin.app.is_speaking = True

        await plugin._stream_text_queue.put("some text")
        await plugin._sentence_queue.put(SentenceItem(text="sentence"))
        await plugin._audio_queue.put((SentenceItem(text="audio"), None))

        await plugin._fire_interrupt()

        assert plugin._interrupt_event.is_set()
        # stream_text_queue has exactly 1 item: the _RESET_BUFFER sentinel
        assert plugin._stream_text_queue.qsize() == 1
        assert plugin._stream_text_queue.get_nowait() is _RESET_BUFFER
        assert plugin._sentence_queue.empty()
        assert plugin._audio_queue.empty()
        assert plugin.app.is_speaking is False

    @pytest.mark.asyncio
    async def test_sends_interrupt_to_gateway(self, standalone_app):
        standalone_app.config.standalone_mode = False
        plugin = ConversationPlugin(standalone_app)
        plugin._client = MagicMock()
        plugin._client.send_interrupt = AsyncMock()

        await plugin._fire_interrupt()

        plugin._client.send_interrupt.assert_called_once()


# ── State transitions ─────────────────────────────────────────────────


class TestStateTransitions:
    def test_initial_state_is_idle(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        assert plugin._state == ConvState.IDLE

    def test_set_state_logs_transition(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._set_state(ConvState.LISTENING)
        assert plugin._state == ConvState.LISTENING

    def test_set_state_noop_on_same(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.SPEAKING
        plugin._set_state(ConvState.SPEAKING)
        assert plugin._state == ConvState.SPEAKING


# ── Emotion queueing ─────────────────────────────────────────────────


class TestEmotionIntegration:
    @pytest.mark.asyncio
    async def test_thinking_emotion_queued_on_send(self, standalone_app):
        standalone_app.config.standalone_mode = False
        standalone_app.config.play_emotions = True

        plugin = ConversationPlugin(standalone_app)
        plugin._client = MagicMock()
        plugin._client.send_message_streaming = AsyncMock()
        plugin._client.send_state_change = AsyncMock()

        standalone_app.emotions.queue_emotion("thinking")

        expr = standalone_app.emotions.get_next_expression()
        assert expr is not None
        assert "hinking" in expr.description.lower()


# ── Stop / cleanup ────────────────────────────────────────────────────


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


# ── Config: barge_in_confirm_frames ───────────────────────────────────


class TestBargeInConfig:
    def test_default_confirm_frames(self):
        config = Config()
        assert config.barge_in_confirm_frames == 2

    def test_custom_confirm_frames(self):
        config = Config(barge_in_confirm_frames=5)
        assert config.barge_in_confirm_frames == 5

    def test_default_silero_threshold(self):
        config = Config()
        assert config.barge_in_silero_threshold == 0.5

    def test_custom_silero_threshold(self):
        config = Config(barge_in_silero_threshold=0.8)
        assert config.barge_in_silero_threshold == 0.8

    def test_default_cooldown_ms(self):
        config = Config()
        assert config.barge_in_cooldown_ms == 300

    def test_custom_cooldown_ms(self):
        config = Config(barge_in_cooldown_ms=1000)
        assert config.barge_in_cooldown_ms == 1000

    def test_default_energy_threshold(self):
        config = Config()
        assert config.barge_in_energy_threshold == 0.02


# ── Config: barge_in YAML mapping ─────────────────────────────────────


class TestBargeInYamlMapping:
    def test_yaml_maps_silero_threshold(self):
        from reachy_claw.config import _apply_yaml

        config = Config()
        _apply_yaml(config, {"barge_in": {"silero_threshold": 0.7}})
        assert config.barge_in_silero_threshold == 0.7

    def test_yaml_maps_cooldown_ms(self):
        from reachy_claw.config import _apply_yaml

        config = Config()
        _apply_yaml(config, {"barge_in": {"cooldown_ms": 1000}})
        assert config.barge_in_cooldown_ms == 1000


# ── VAD: speech_probability ───────────────────────────────────────────


class TestVADSpeechProbability:
    def test_energy_vad_returns_binary_probability(self):
        from reachy_claw.vad import EnergyVAD

        vad = EnergyVAD(threshold=0.01)
        silence = np.zeros(512, dtype=np.float32)
        loud = np.full(512, 0.5, dtype=np.float32)
        assert vad.speech_probability(silence) == 0.0
        assert vad.speech_probability(loud) == 1.0

    def test_base_class_default_delegates_to_is_speech(self):
        from reachy_claw.vad import VADBackend

        class DummyVAD(VADBackend):
            def is_speech(self, audio, sample_rate=16000):
                return True

        vad = DummyVAD()
        assert vad.speech_probability(np.zeros(512, dtype=np.float32)) == 1.0


# ── ConversationPlugin: _speaking_since tracking ─────────────────────


class TestSpeakingSinceTracking:
    def _make_plugin(self):
        app = MagicMock()
        app.config = Config()
        plugin = ConversationPlugin(app)
        return plugin

    def test_speaking_since_set_on_state_change(self):
        """_speaking_since is updated when state changes to SPEAKING."""
        import time as _time

        plugin = self._make_plugin()
        before = _time.monotonic()
        plugin._set_state(ConvState.SPEAKING)
        after = _time.monotonic()
        assert before <= plugin._speaking_since <= after

    def test_speaking_since_not_updated_for_other_states(self):
        """_speaking_since is NOT updated for non-SPEAKING states."""
        plugin = self._make_plugin()
        plugin._speaking_since = 0.0
        plugin._set_state(ConvState.LISTENING)
        assert plugin._speaking_since == 0.0


# ── Barge-in: _finish_speaking guards ────────────────────────────────


class TestFinishSpeakingGuard:
    """_finish_speaking should NOT set IDLE if barge-in already set LISTENING."""

    @pytest.mark.asyncio
    async def test_skips_idle_when_already_listening(self, standalone_app):
        """Barge-in sets LISTENING → _finish_speaking must NOT flicker to IDLE."""
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.LISTENING  # barge-in already moved here
        standalone_app.is_speaking = True

        await plugin._finish_speaking()

        assert plugin._state == ConvState.LISTENING  # NOT IDLE
        assert standalone_app.is_speaking is False

    @pytest.mark.asyncio
    async def test_transitions_to_idle_when_speaking(self, standalone_app):
        """Normal end of speech (no barge-in) → should go to IDLE."""
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.SPEAKING
        standalone_app.is_speaking = True

        await plugin._finish_speaking()

        assert plugin._state == ConvState.IDLE
        assert standalone_app.is_speaking is False

    @pytest.mark.asyncio
    async def test_transitions_to_idle_from_thinking(self, standalone_app):
        """Edge case: _finish_speaking from THINKING → IDLE."""
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.THINKING

        await plugin._finish_speaking()

        assert plugin._state == ConvState.IDLE


# ── Barge-in: _fire_interrupt sends RESET_BUFFER ─────────────────────


class TestFireInterruptResetBuffer:
    @pytest.mark.asyncio
    async def test_puts_reset_buffer_after_drain(self, standalone_app):
        """After draining, _fire_interrupt puts _RESET_BUFFER sentinel."""
        plugin = ConversationPlugin(standalone_app)
        await plugin._stream_text_queue.put("stale-1")
        await plugin._stream_text_queue.put("stale-2")

        await plugin._fire_interrupt()

        # Queue should have exactly 1 item: the RESET_BUFFER sentinel
        assert plugin._stream_text_queue.qsize() == 1
        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is _RESET_BUFFER

    @pytest.mark.asyncio
    async def test_fire_interrupt_sends_server_interrupt_async(self, standalone_app):
        """send_interrupt should be spawned as a background task (non-blocking)."""
        standalone_app.config.standalone_mode = False
        plugin = ConversationPlugin(standalone_app)
        plugin._event_loop = asyncio.get_running_loop()
        plugin._client = MagicMock()
        plugin._client.send_interrupt = AsyncMock()

        await plugin._fire_interrupt()

        # The task is spawned, not awaited directly. Give it a tick to run.
        await asyncio.sleep(0.01)
        plugin._client.send_interrupt.assert_called_once()


# ── TTS worker: interrupt guard after synthesis ──────────────────────


class TestTTSWorkerInterruptGuard:
    """TTS worker breaks out of streaming on interrupt but still enqueues —
    the output_pipeline is responsible for draining stale items."""

    @pytest.mark.asyncio
    async def test_breaks_streaming_on_interrupt(self, standalone_app):
        """Interrupt during synthesis stops collecting chunks early."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        mock_tts = MagicMock()
        mock_tts.supports_streaming = True
        chunks_yielded = 0

        async def mock_streaming(text):
            nonlocal chunks_yielded
            for _ in range(5):
                chunks_yielded += 1
                if chunks_yielded == 2:
                    plugin._interrupt_event.set()
                yield (np.zeros(1024, dtype=np.float32), 16000)

        mock_tts.synthesize_streaming = mock_streaming
        plugin._tts = mock_tts

        await plugin._sentence_queue.put(SentenceItem(text="Hello world."))

        task = asyncio.create_task(plugin._tts_worker())
        await asyncio.sleep(0.2)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # TTS worker still enqueues (output_pipeline handles draining),
        # but it should have broken out of streaming early
        assert chunks_yielded == 2  # stopped after interrupt, not all 5

    @pytest.mark.asyncio
    async def test_passes_through_when_not_interrupted(self, standalone_app):
        """Normal case: no interrupt → audio enqueued."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        mock_tts = MagicMock()
        mock_tts.supports_streaming = True

        async def mock_streaming(text):
            yield (np.zeros(1024, dtype=np.float32), 16000)

        mock_tts.synthesize_streaming = mock_streaming
        plugin._tts = mock_tts

        await plugin._sentence_queue.put(SentenceItem(text="Hello world."))

        task = asyncio.create_task(plugin._tts_worker())
        await asyncio.sleep(0.2)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Audio should be enqueued
        assert not plugin._audio_queue.empty()
        entry = plugin._audio_queue.get_nowait()
        assert entry[0].text == "Hello world."


# ── Output pipeline: interrupt guard after dequeue ───────────────────


class TestOutputPipelineInterruptDuringPlayback:
    """Output pipeline handles interrupt during _speak_interruptible."""

    @pytest.mark.asyncio
    async def test_interrupt_during_speak_drains_remaining(self, standalone_app):
        """When _speak_interruptible returns interrupted=True, remaining items drained."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            plugin._interrupt_event.set()
            return True  # interrupted

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="First."), None))
        await plugin._audio_queue.put((SentenceItem(text="Second."), None))
        await plugin._audio_queue.put(
            (SentenceItem(text="Third.", is_last=True), None)
        )

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.3)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert spoken == ["First."]
        assert plugin._audio_queue.empty()


# ── Stream callbacks: run_id guards ──────────────────────────────────


class TestRunIdGuards:
    """_on_stream_end and _on_stream_abort must ignore stale run IDs."""

    @pytest.mark.asyncio
    async def test_stream_end_ignores_stale_run_id(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-2"

        await plugin._on_stream_end("stale text", "run-1")

        # Queue should be empty — stale event was dropped
        assert plugin._stream_text_queue.empty()

    @pytest.mark.asyncio
    async def test_stream_end_accepts_current_run_id(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-2"

        await plugin._on_stream_end("good text", "run-2")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None  # end-of-stream sentinel

    @pytest.mark.asyncio
    async def test_stream_abort_ignores_stale_run_id(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-2"

        await plugin._on_stream_abort("old interrupt", "run-1")

        assert plugin._stream_text_queue.empty()

    @pytest.mark.asyncio
    async def test_stream_abort_accepts_current_run_id(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-2"

        await plugin._on_stream_abort("interrupt", "run-2")

        sentinel = plugin._stream_text_queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_stream_abort_emits_llm_end(self, standalone_app):
        """_on_stream_abort should emit llm_end so frontend finalizes thought card."""
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-1"
        emitted = []
        standalone_app.events.subscribe("llm_end", lambda data: emitted.append(data))

        await plugin._on_stream_abort("interrupted", "run-1")

        assert len(emitted) == 1
        assert emitted[0]["run_id"] == "run-1"


# ── switch_mode: barge-in auto-enable ────────────────────────────────


@pytest.fixture
def mode_plugin(standalone_app):
    """ConversationPlugin with ModeManager initialized for testing."""
    from reachy_claw.mode import ModeContext, ModeManager
    from reachy_claw.modes import ConversationMode, MonologueMode, InterpreterMode

    plugin = ConversationPlugin(standalone_app)
    ctx = ModeContext(
        app=standalone_app,
        sentence_queue=asyncio.Queue(),
        audio_queue=asyncio.Queue(),
        events=standalone_app.events,
        spawn_task=plugin._spawn_task,
        get_vision_context=lambda: None,
        capture_frame=lambda: None,
    )
    plugin._mode_manager = ModeManager(ctx)
    plugin._mode_manager.register(ConversationMode())
    plugin._mode_manager.register(MonologueMode())
    plugin._mode_manager.register(InterpreterMode())
    return plugin


class TestSwitchMode:
    """switch_mode delegates to ModeManager and updates config."""

    @pytest.mark.asyncio
    async def test_conversation_mode_enables_barge_in(self, mode_plugin, standalone_app):
        standalone_app.config.barge_in_enabled = False

        await mode_plugin._do_switch_mode("conversation")

        assert standalone_app.config.barge_in_enabled is True
        assert mode_plugin._mode_manager.current.name == "conversation"

    @pytest.mark.asyncio
    async def test_monologue_mode_does_not_force_barge_in(self, mode_plugin, standalone_app):
        standalone_app.config.barge_in_enabled = False

        await mode_plugin._do_switch_mode("monologue")

        # Monologue mode should NOT force barge_in on
        assert standalone_app.config.barge_in_enabled is False
        assert mode_plugin._mode_manager.current.name == "monologue"

    @pytest.mark.asyncio
    async def test_interpreter_mode_disables_barge_in(self, mode_plugin, standalone_app):
        standalone_app.config.barge_in_enabled = True

        await mode_plugin._do_switch_mode("interpreter")

        assert standalone_app.config.barge_in_enabled is False
        assert mode_plugin._mode_manager.current.name == "interpreter"

    @pytest.mark.asyncio
    async def test_interpreter_to_conversation_restores_barge_in(self, mode_plugin, standalone_app):
        await mode_plugin._do_switch_mode("interpreter")
        assert mode_plugin._mode_manager.current.name == "interpreter"

        await mode_plugin._do_switch_mode("conversation")
        assert mode_plugin._mode_manager.current.name == "conversation"
        assert standalone_app.config.barge_in_enabled is True

    @pytest.mark.asyncio
    async def test_interpreter_mode_ollama_config(self, mode_plugin, standalone_app):
        """Interpreter mode sets correct OllamaClient config."""
        from reachy_claw.llm import OllamaClient, OllamaConfig

        mode_plugin._client = OllamaClient(OllamaConfig())
        standalone_app.config.interpreter_source_lang = "Chinese"
        standalone_app.config.interpreter_target_lang = "English"

        await mode_plugin._do_switch_mode("interpreter")

        assert mode_plugin._client._config.skip_emotion_extraction is True
        assert mode_plugin._client._config.max_history == 0
        assert mode_plugin._client._config.temperature == 0.3
        assert mode_plugin._client._config.monologue_mode is False
        assert "Chinese" in mode_plugin._client._config.system_prompt
        assert "English" in mode_plugin._client._config.system_prompt

    @pytest.mark.asyncio
    async def test_switch_interpreter_to_conversation_restores_config(self, mode_plugin, standalone_app):
        """Switching from interpreter back to conversation restores OllamaClient config."""
        from reachy_claw.llm import OllamaClient, OllamaConfig

        standalone_app.config.ollama_max_history = 5
        standalone_app.config.ollama_temperature = 0.7
        standalone_app.config.enable_vlm = True
        mode_plugin._client = OllamaClient(OllamaConfig(
            max_history=5, temperature=0.7, enable_vlm=True,
        ))

        # Switch to interpreter — config gets overridden
        await mode_plugin._do_switch_mode("interpreter")
        assert mode_plugin._client._config.max_history == 0
        assert mode_plugin._client._config.temperature == 0.3
        assert mode_plugin._client._config.enable_vlm is False

        # Switch back to conversation — config restored
        await mode_plugin._do_switch_mode("conversation")
        assert mode_plugin._client._config.max_history == 5
        assert mode_plugin._client._config.temperature == 0.7
        assert mode_plugin._client._config.enable_vlm is True

    @pytest.mark.asyncio
    async def test_interpreter_mode_name(self, mode_plugin, standalone_app):
        """Interpreter mode is correctly identified by name."""
        await mode_plugin._do_switch_mode("interpreter")
        assert mode_plugin._mode_manager.current.name == "interpreter"


# ── Sentence accumulator: RESET_BUFFER clears buffer ────────────────


class TestSentenceAccumulatorReset:
    @pytest.mark.asyncio
    async def test_reset_buffer_discards_stale_text(self, standalone_app):
        """_RESET_BUFFER sentinel causes accumulator to discard buffered text."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True

        # Stale partial text, then reset, then new text, then end
        await plugin._stream_text_queue.put("stale partial ")
        await plugin._stream_text_queue.put(_RESET_BUFFER)
        await plugin._stream_text_queue.put("Fresh sentence.")
        await plugin._stream_text_queue.put(None)

        task = asyncio.create_task(plugin._sentence_accumulator())
        await asyncio.sleep(0.2)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        sentences = []
        while not plugin._sentence_queue.empty():
            sentences.append(plugin._sentence_queue.get_nowait())

        texts = [s.text for s in sentences if s.text]
        # "stale partial" should NOT appear in any sentence
        assert not any("stale" in t for t in texts)
        # "Fresh sentence" should appear
        assert any("Fresh" in t for t in texts)


# ── Full barge-in flow: SPEAKING → interrupt → LISTENING → no IDLE ───


class TestBargeInFullFlow:
    """Integration test: barge-in during speaking → interrupt → finish without IDLE flicker."""

    @pytest.mark.asyncio
    async def test_no_idle_flicker_on_barge_in(self, standalone_app):
        """Simulate the full barge-in sequence through the output pipeline."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        state_log = []

        # Monkey-patch _set_state to record transitions
        original_set_state = plugin._set_state

        def tracking_set_state(new_state):
            state_log.append(new_state)
            original_set_state(new_state)

        plugin._set_state = tracking_set_state

        async def mock_speak(text, prefetched_chunks=None):
            # Simulate barge-in during playback
            plugin._set_state(ConvState.LISTENING)  # audio_loop sets this
            plugin._interrupt_event.set()
            return True  # interrupted

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Hello."), None))
        await plugin._audio_queue.put(
            (SentenceItem(text="World.", is_last=True), None)
        )

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.3)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # State should NOT have passed through IDLE after LISTENING
        # (i.e., no LISTENING → IDLE → LISTENING sequence)
        for i in range(len(state_log) - 1):
            if state_log[i] == ConvState.LISTENING:
                assert state_log[i + 1] != ConvState.IDLE, (
                    f"IDLE flicker detected at index {i}: {[s.value for s in state_log]}"
                )

    @pytest.mark.asyncio
    async def test_interrupt_during_inter_sentence_pause(self, standalone_app):
        """Interrupt fires during the 0.15s pause between sentences."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        spoken = []

        call_count = 0

        async def mock_speak(text, prefetched_chunks=None):
            nonlocal call_count
            call_count += 1
            spoken.append(text)
            if call_count == 1:
                # First sentence completes normally, but interrupt fires
                # during the inter-sentence pause
                async def set_interrupt_soon():
                    await asyncio.sleep(0.05)
                    plugin._interrupt_event.set()

                asyncio.create_task(set_interrupt_soon())
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="First."), None))
        await plugin._audio_queue.put((SentenceItem(text="Second."), None))
        await plugin._audio_queue.put(
            (SentenceItem(text="Third.", is_last=True), None)
        )

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.5)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only first sentence spoken; second and third should be drained
        assert spoken == ["First."]

    @pytest.mark.asyncio
    async def test_multiple_rapid_interrupts(self, standalone_app):
        """Multiple rapid interrupts don't cause issues."""
        plugin = ConversationPlugin(standalone_app)

        # Fire interrupt multiple times rapidly
        await plugin._fire_interrupt()
        await plugin._fire_interrupt()
        await plugin._fire_interrupt()

        # Should still be in a consistent state
        assert plugin._interrupt_event.is_set()
        assert plugin.app.is_speaking is False
        # Each _fire_interrupt drains then puts RESET_BUFFER; subsequent calls
        # drain the previous RESET_BUFFER, so only the last one remains.
        assert plugin._stream_text_queue.qsize() == 1
        assert plugin._stream_text_queue.get_nowait() is _RESET_BUFFER


# ── Output pipeline: conversation_stopped skips TTS ──────────────────


class TestConversationStopped:
    @pytest.mark.asyncio
    async def test_stopped_mode_skips_tts_and_finishes(self, standalone_app):
        """When _conversation_stopped=True, pipeline drains without speaking."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        plugin._conversation_stopped = True
        spoken = []

        async def mock_speak(text, prefetched_chunks=None):
            spoken.append(text)
            return False

        plugin._speak_interruptible = mock_speak

        await plugin._audio_queue.put((SentenceItem(text="Skip this."), None))
        await plugin._audio_queue.put(
            (SentenceItem(text="Done.", is_last=True), None)
        )

        task = asyncio.create_task(plugin._output_pipeline())
        await asyncio.sleep(0.3)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(spoken) == 0
        assert standalone_app.is_speaking is False


# ── _on_stream_delta: run_id and state transition ────────────────────


class TestStreamDeltaRunId:
    @pytest.mark.asyncio
    async def test_first_delta_stays_thinking(self, standalone_app):
        """First delta stays in THINKING — transition to SPEAKING happens in output pipeline."""
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-1"
        plugin._state = ConvState.THINKING

        await plugin._on_stream_delta("Hi", "run-1")

        assert plugin._state == ConvState.THINKING

    @pytest.mark.asyncio
    async def test_delta_emits_llm_delta_event(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._current_run_id = "run-1"
        plugin._state = ConvState.THINKING
        emitted = []
        standalone_app.events.subscribe("llm_delta", lambda data: emitted.append(data))

        await plugin._on_stream_delta("chunk", "run-1")

        assert len(emitted) == 1
        assert emitted[0]["text"] == "chunk"
        assert emitted[0]["run_id"] == "run-1"


# ── _on_stream_start: state and queue reset ──────────────────────────


class TestStreamStartReset:
    @pytest.mark.asyncio
    async def test_sets_thinking_state(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        plugin._state = ConvState.LISTENING

        await plugin._on_stream_start("run-1")

        assert plugin._state == ConvState.THINKING
        assert plugin._current_run_id == "run-1"

    @pytest.mark.asyncio
    async def test_drains_stale_and_inserts_reset(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        await plugin._stream_text_queue.put("old-1")
        await plugin._stream_text_queue.put("old-2")

        await plugin._on_stream_start("run-new")

        assert plugin._stream_text_queue.qsize() == 1
        item = plugin._stream_text_queue.get_nowait()
        assert item is _RESET_BUFFER


# ── Barge-in pre-roll replay ──────────────────────────────────────────


class TestBargeInPreRoll:
    """Pre-roll buffer should replay ~500ms of audio, not just confirm_frames."""

    @pytest.mark.asyncio
    async def test_speaking_still_respects_cooldown(self, standalone_app):
        """SPEAKING state still applies cooldown (regression check)."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        standalone_app.config.barge_in_enabled = True
        standalone_app.config.barge_in_cooldown_ms = 50000  # huge cooldown

        interrupted = False

        async def tracking_fire():
            nonlocal interrupted
            interrupted = True

        plugin._fire_interrupt = tracking_fire

        plugin._vad = MagicMock()
        plugin._vad.speech_probability = MagicMock(return_value=0.9)

        loud_chunk = np.full(1024, 0.5, dtype=np.float32)
        plugin._stt = MagicMock()
        plugin._stt.supports_streaming = False

        call_idx = 0

        async def mock_read(size):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                plugin._state = ConvState.SPEAKING
                plugin._speaking_since = time.monotonic()  # just started
                return None
            if call_idx <= 8:
                return loud_chunk
            return None

        plugin._audio = MagicMock()
        plugin._audio.read_chunk = mock_read

        task = asyncio.create_task(plugin._audio_loop())
        await asyncio.sleep(0.5)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert not interrupted, "SPEAKING should respect cooldown"

    @pytest.mark.asyncio
    async def test_preroll_replays_more_than_confirm_frames(self, standalone_app):
        """Pre-roll buffer replays ~500ms of audio, not just confirm_frames."""
        plugin = ConversationPlugin(standalone_app)
        plugin._running = True
        standalone_app.config.barge_in_enabled = True
        standalone_app.config.barge_in_confirm_frames = 2
        standalone_app.config.barge_in_cooldown_ms = 0  # no cooldown

        replayed_chunks = []

        plugin._vad = MagicMock()
        plugin._vad.speech_probability = MagicMock(return_value=0.9)

        loud_chunk = np.full(1024, 0.5, dtype=np.float32)
        # Quiet chunk passes cooldown but fails energy gate → fills pre-roll
        quiet_chunk = np.full(1024, 0.005, dtype=np.float32)

        stt_mock = MagicMock()
        stt_mock.supports_streaming = True
        stt_mock.cancel_stream = MagicMock()
        stt_mock.start_stream = MagicMock()

        def tracking_feed(c):
            replayed_chunks.append(c)
            return None

        stt_mock.feed_chunk = tracking_feed
        plugin._stt = stt_mock

        # Sequence: init → set SPEAKING (past cooldown) → 4 quiet frames
        # (pre-roll only) → 2 loud frames (confirm barge-in) → done
        call_idx = 0

        async def mock_read(size):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                plugin._state = ConvState.SPEAKING
                plugin._speaking_since = 0  # far past cooldown
                return None
            if call_idx <= 5:
                return quiet_chunk  # fills pre-roll, fails energy gate
            if call_idx <= 7:
                return loud_chunk  # passes all layers → confirms barge-in
            return None

        plugin._audio = MagicMock()
        plugin._audio.read_chunk = mock_read

        task = asyncio.create_task(plugin._audio_loop())
        await asyncio.sleep(0.5)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Pre-roll should replay all 6 frames (4 quiet + 2 loud),
        # not just the 2 confirm frames
        assert len(replayed_chunks) == 6, (
            f"Expected 6 pre-roll frames, got {len(replayed_chunks)}"
        )
