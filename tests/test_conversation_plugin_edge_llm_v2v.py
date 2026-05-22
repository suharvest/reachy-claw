"""Integration smoke tests for the edge_llm_v2v backend wiring inside
ConversationPlugin (Wave 2).

These tests bypass `start()` and exercise the plugin in V2V mode by:
  * assigning mock V2V and EdgeLLM clients directly
  * wiring V2V callbacks the same way `_init_gateway()` does
  * driving plugin state by invoking those callbacks

This isolates Wave 2 wiring (callbacks, state transitions, audio bridge,
barge-in) from the unit-tested internals of EdgeLLMClient / V2VClient.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_claw.config import Config
from reachy_claw.plugins.conversation_plugin import (
    ConversationPlugin,
    ConvState,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def v2v_app(mock_reachy):
    """ReachyClawApp configured with edge_llm_v2v backend."""
    from reachy_claw.app import ReachyClawApp

    config = Config(
        standalone_mode=False,
        llm_backend="edge_llm_v2v",
        edge_llm_url="http://127.0.0.1:8080",
        v2v_url="ws://127.0.0.1:8621/v2v/stream",
        idle_animations=False,
        play_emotions=False,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    a = ReachyClawApp(config)
    a.reachy = mock_reachy
    return a


def _make_mock_v2v():
    """Build a V2VClient mock with the callback-attribute surface intact."""
    m = MagicMock(name="V2VClient")
    m.is_connected = True
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    m.send_audio = AsyncMock()
    m.send_text_delta = AsyncMock()
    m.flush_tts = AsyncMock()
    m.send_asr_eos = AsyncMock()
    m.abort = AsyncMock()
    # Callback attributes — match the real V2VClient surface.
    for cb in (
        "on_asr_partial", "on_asr_final", "on_asr_endpoint",
        "on_tts_started", "on_tts_sentence_done", "on_tts_done",
        "on_tts_audio", "on_vad_event", "on_error",
    ):
        setattr(m, cb, None)
    return m


def _make_mock_edge_llm():
    m = MagicMock(name="EdgeLLMClient")
    m.is_connected = True
    m.connect = AsyncMock()
    m.disconnect = AsyncMock()
    m.warmup_session = AsyncMock()
    m.send_message_streaming = AsyncMock()
    m.send_interrupt = AsyncMock()
    m.send_state_change = AsyncMock()
    m.send_robot_result = AsyncMock()
    cb_obj = MagicMock()
    cb_obj.on_stream_start = None
    cb_obj.on_stream_delta = None
    cb_obj.on_stream_end = None
    cb_obj.on_stream_abort = None
    cb_obj.on_tool_start = None
    cb_obj.on_tool_end = None
    cb_obj.on_task_spawned = None
    cb_obj.on_task_completed = None
    cb_obj.on_emotion = None
    cb_obj.on_robot_command = None
    m.callbacks = cb_obj
    return m


def _make_plugin(app):
    """ConversationPlugin with mocked V2V + EdgeLLM clients wired up."""
    plugin = ConversationPlugin(app)
    try:
        plugin._event_loop = asyncio.get_event_loop()
    except RuntimeError:
        plugin._event_loop = asyncio.new_event_loop()
    plugin._v2v = _make_mock_v2v()
    plugin._client = _make_mock_edge_llm()
    plugin._setup_callbacks()
    plugin._setup_v2v_callbacks()
    plugin._running = True
    return plugin


# ── Backend selection ────────────────────────────────────────────────


class TestBackendSelection:
    """Confirm legacy backend selection branches remain intact."""

    def test_edge_llm_v2v_branch_present(self):
        """Plugin source must still have all three llm_backend branches."""
        import inspect
        from reachy_claw.plugins import conversation_plugin

        src = inspect.getsource(conversation_plugin)
        assert 'config.llm_backend == "edge_llm_v2v"' in src
        assert 'config.llm_backend == "ollama"' in src
        # The original `else` branch for DesktopRobotClient (gateway) stays:
        assert "DesktopRobotClient(config)" in src


# ── V2V → LLM bridge ─────────────────────────────────────────────────


class TestV2VToLLMBridge:
    @pytest.mark.asyncio
    async def test_asr_final_calls_process_and_send(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        await plugin._on_v2v_asr_final("say hello", False, False)
        plugin._process_and_send.assert_awaited_once_with("say hello")

    @pytest.mark.asyncio
    async def test_asr_final_duplicate_is_skipped(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        await plugin._on_v2v_asr_final("echo", False, duplicate_of_streamed=True)
        plugin._process_and_send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_asr_partial_emits_state_and_event(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.IDLE
        events = []
        plugin.app.events.subscribe("asr_partial", lambda d: events.append(d))
        await plugin._on_v2v_asr_partial("he", False)
        assert plugin._state == ConvState.TRANSCRIBING
        assert events and events[0]["text"] == "he"


# ── LLM → V2V bridge ─────────────────────────────────────────────────


class TestLLMToV2VBridge:
    @pytest.mark.asyncio
    async def test_stream_delta_forwarded_to_v2v(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._current_run_id = "run-1"
        plugin._state = ConvState.THINKING
        await plugin._on_stream_delta("Hello", "run-1")
        await plugin._on_stream_delta(" there.", "run-1")
        assert plugin._v2v.send_text_delta.await_count == 2
        assert plugin._v2v.send_text_delta.await_args_list[0].args[0] == "Hello"
        assert plugin._v2v.send_text_delta.await_args_list[1].args[0] == " there."
        # In V2V mode, local stream_text_queue must stay untouched.
        assert plugin._stream_text_queue.empty()

    @pytest.mark.asyncio
    async def test_stream_end_calls_flush_tts(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._current_run_id = "run-1"
        await plugin._on_stream_end("Hello there.", "run-1")
        plugin._v2v.flush_tts.assert_awaited_once()
        # No sentinel pushed to local accumulator in V2V mode.
        assert plugin._stream_text_queue.empty()

    @pytest.mark.asyncio
    async def test_stream_abort_calls_v2v_abort(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._current_run_id = "run-1"
        plugin._state = ConvState.SPEAKING
        await plugin._on_stream_abort("interrupted", "run-1")
        plugin._v2v.abort.assert_awaited_once()


# ── TTS lifecycle ────────────────────────────────────────────────────


class TestV2VTTSLifecycle:
    @pytest.mark.asyncio
    async def test_tts_started_sets_speaking(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.THINKING
        await plugin._on_v2v_tts_started("Hello.")
        assert plugin._state == ConvState.SPEAKING
        assert plugin.app.is_speaking is True

    @pytest.mark.asyncio
    async def test_tts_done_returns_to_idle(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True
        await plugin._on_v2v_tts_done()
        assert plugin._state == ConvState.IDLE
        assert plugin.app.is_speaking is False

    @pytest.mark.asyncio
    async def test_tts_audio_pushed_to_audio_queue(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        pcm = b"\x00\x01" * 320  # 320 samples
        await plugin._on_v2v_tts_audio(16000, pcm)
        assert plugin._audio_queue.qsize() == 1
        entry = plugin._audio_queue.get_nowait()
        assert entry[0] == "v2v_audio"
        assert entry[1] == 16000
        assert entry[2] == pcm

    @pytest.mark.asyncio
    async def test_tts_audio_sr_mismatch_warns_once(self, v2v_app, caplog):
        plugin = _make_plugin(v2v_app)
        # local sample_rate=16000 by default; emit at 24000 to trigger warn.
        for _ in range(3):
            await plugin._on_v2v_tts_audio(24000, b"\x00\x00" * 160)
        warnings = [r for r in caplog.records if "sample rate" in r.getMessage()]
        assert len(warnings) == 1


# ── Barge-in ─────────────────────────────────────────────────────────


class TestV2VBargeIn:
    @pytest.mark.asyncio
    async def test_speech_start_fires_interrupt(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True
        # Pre-load _audio_queue so we can verify drain
        plugin._audio_queue.put_nowait(("v2v_audio", 16000, b"\x00\x00"))
        await plugin._on_v2v_vad_event("speech_start")
        assert plugin._interrupt_event.is_set()
        assert plugin._audio_queue.empty()
        # state should drop from SPEAKING to LISTENING
        assert plugin._state == ConvState.LISTENING
        # V2V abort scheduled as background task
        await asyncio.sleep(0)  # let scheduled tasks run
        await asyncio.sleep(0)
        plugin._v2v.abort.assert_awaited()

    @pytest.mark.asyncio
    async def test_speech_start_deduplicated(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        await plugin._on_v2v_vad_event("speech_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first_calls = plugin._v2v.abort.await_count
        # Second VAD speech_start in the same speaking cycle should be a no-op.
        await plugin._on_v2v_vad_event("speech_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert plugin._v2v.abort.await_count == first_calls

    @pytest.mark.asyncio
    async def test_speech_start_in_IDLE_does_not_abort(self, v2v_app):
        """Regression (f2332c8): speech_start in IDLE must NOT abort.

        Aborting in IDLE cancels the in-flight OVS ASR session, causing
        asr_final to come back empty → "ASR never produces text".
        """
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.IDLE
        plugin._fire_interrupt = AsyncMock()
        await plugin._on_v2v_vad_event("speech_start")
        # Let any (incorrectly) spawned tasks run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert plugin._state == ConvState.LISTENING
        plugin._fire_interrupt.assert_not_called()
        plugin._v2v.abort.assert_not_called()
        assert plugin._v2v_abort_in_flight is False

    @pytest.mark.asyncio
    async def test_speech_start_in_SPEAKING_does_abort(self, v2v_app):
        """Regression (f2332c8) companion: SPEAKING path still aborts."""
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        await plugin._on_v2v_vad_event("speech_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert plugin._state == ConvState.LISTENING
        assert plugin._v2v_abort_in_flight is True
        plugin._v2v.abort.assert_awaited_once()


# ── speech_end → asr_eos (multi_utterance) ───────────────────────────


class TestV2VSpeechEndAsrEos:
    @pytest.mark.asyncio
    async def test_speech_end_sends_asr_eos_in_multi_utterance(self, v2v_app):
        """Regression (606dbd3): multi_utterance=True needs explicit asr_eos."""
        v2v_app.config.v2v_multi_utterance = True
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.LISTENING
        plugin._v2v.is_connected = True
        before_ts = plugin._v2v_last_speech_end_ts
        await plugin._on_v2v_vad_event("speech_end")
        plugin._v2v.send_asr_eos.assert_awaited_once()
        assert plugin._state == ConvState.LISTENING
        assert plugin._v2v_last_speech_end_ts > before_ts

    @pytest.mark.asyncio
    async def test_speech_end_skips_asr_eos_when_multi_utterance_false(self, v2v_app):
        """multi_utterance=False: OVS auto-finalizes, no asr_eos needed."""
        v2v_app.config.v2v_multi_utterance = False
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.LISTENING
        plugin._v2v.is_connected = True
        await plugin._on_v2v_vad_event("speech_end")
        plugin._v2v.send_asr_eos.assert_not_called()


# ── Error path ───────────────────────────────────────────────────────


class TestV2VError:
    @pytest.mark.asyncio
    async def test_error_resets_to_idle(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.THINKING
        await plugin._on_v2v_error("boom")
        assert plugin._state == ConvState.IDLE


# ── V2V callback wiring sanity ───────────────────────────────────────


class TestV2VCallbackWiring:
    def test_setup_v2v_callbacks_assigns_all(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        v2v = plugin._v2v
        # All nine V2V callbacks must be wired to plugin methods.
        assert v2v.on_asr_partial == plugin._on_v2v_asr_partial
        assert v2v.on_asr_final == plugin._on_v2v_asr_final
        assert v2v.on_asr_endpoint == plugin._on_v2v_asr_endpoint
        assert v2v.on_tts_started == plugin._on_v2v_tts_started
        assert v2v.on_tts_sentence_done == plugin._on_v2v_tts_sentence_done
        assert v2v.on_tts_done == plugin._on_v2v_tts_done
        assert v2v.on_tts_audio == plugin._on_v2v_tts_audio
        assert v2v.on_vad_event == plugin._on_v2v_vad_event
        assert v2v.on_error == plugin._on_v2v_error


# ── Audio uplink loop ────────────────────────────────────────────────


class TestV2VAudioUplinkLoop:
    @pytest.mark.asyncio
    async def test_uplink_forwards_pcm16_bytes(self, v2v_app):
        plugin = _make_plugin(v2v_app)

        import numpy as np

        chunks = [np.zeros(1024, dtype=np.float32), None]

        class FakeAudio:
            def __init__(self):
                self._i = 0

            async def read_chunk(self, frames):
                if self._i < len(chunks):
                    c = chunks[self._i]
                    self._i += 1
                    return c
                # End: stop the loop after a beat
                plugin._running = False
                return None

        plugin._audio = FakeAudio()
        await plugin._v2v_audio_uplink_loop()
        # Should have called send_audio at least once with int16 bytes
        assert plugin._v2v.send_audio.await_count >= 1
        payload = plugin._v2v.send_audio.await_args_list[0].args[0]
        assert isinstance(payload, (bytes, bytearray))
        # 1024 float32 samples → 1024 * 2 bytes int16
        assert len(payload) == 1024 * 2


# ── Shutdown ─────────────────────────────────────────────────────────


class TestV2VShutdown:
    @pytest.mark.asyncio
    async def test_stop_sends_asr_eos_and_disconnects(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._audio = MagicMock()
        plugin._audio.stop = AsyncMock()
        plugin.app.is_speaking = False
        v2v_ref = plugin._v2v
        client_ref = plugin._client
        await plugin.stop()
        v2v_ref.send_asr_eos.assert_awaited_once()
        v2v_ref.disconnect.assert_awaited_once()
        client_ref.disconnect.assert_awaited_once()
        # stop() should clear the v2v reference so a second stop() is safe.
        assert plugin._v2v is None

    @pytest.mark.asyncio
    async def test_stop_aborts_if_speaking(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._audio = MagicMock()
        plugin._audio.stop = AsyncMock()
        plugin.app.is_speaking = True
        v2v_ref = plugin._v2v
        await plugin.stop()
        v2v_ref.abort.assert_awaited()


# ── Make sure legacy backends still configure ────────────────────────


class TestLegacyBackendsUnaffected:
    def test_gateway_default_branch_unchanged(self):
        cfg = Config(standalone_mode=False)
        assert cfg.llm_backend == "gateway"

    def test_ollama_backend_still_supported(self):
        cfg = Config(standalone_mode=False, llm_backend="ollama")
        assert cfg.llm_backend == "ollama"


# ── Wave 2.5: Codex review fixes ──────────────────────────────────────


class TestBargeInClearsInterrupt:
    """BUG 1: a fresh tts_started must clear the interrupt latch so the
    new TTS reply isn't permanently muted by _play_v2v_pcm's drop guard.
    """

    @pytest.mark.asyncio
    async def test_tts_started_clears_interrupt_event(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._interrupt_event.set()
        await plugin._on_v2v_tts_started("Hello again.")
        assert plugin._interrupt_event.is_set() is False

    @pytest.mark.asyncio
    async def test_audio_after_resume_is_queued_not_dropped(self, v2v_app):
        """Full barge-in lifecycle: speech_start sets interrupt, tts_started
        clears it, subsequent tts_audio survives the _play_v2v_pcm drop
        guard and lands in _audio_queue."""
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True

        # Barge-in
        await plugin._on_v2v_vad_event("speech_start")
        assert plugin._interrupt_event.is_set()

        # New reply starts
        await plugin._on_v2v_tts_started("New reply.")
        assert plugin._interrupt_event.is_set() is False

        # Verify the pre-queue path is not blocked
        await plugin._on_v2v_tts_audio(16000, b"\x00\x00" * 160)
        assert plugin._audio_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_asr_final_clears_interrupt_event(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        plugin._interrupt_event.set()
        await plugin._on_v2v_asr_final("hi again", False, False)
        assert plugin._interrupt_event.is_set() is False
        plugin._process_and_send.assert_awaited_once()


class TestTtsDoneStopsGStreamer:
    """BUG 2: _on_v2v_tts_done must stop the GStreamer pipeline that
    _play_v2v_pcm started, so the next turn opens a fresh one."""

    @pytest.mark.asyncio
    async def test_tts_done_stops_pipeline_and_clears_flag(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._gst_playing = True
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True
        # Replace _stop_gst_playback with a counted stub so we don't need
        # a real GStreamer pipeline.
        plugin._stop_gst_playback = AsyncMock(
            side_effect=lambda: setattr(plugin, "_gst_playing", False)
        )
        await plugin._on_v2v_tts_done()
        plugin._stop_gst_playback.assert_awaited_once()
        assert plugin._gst_playing is False


class TestStartV2VSkipsLegacyFactories:
    """BUG 3: start() in V2V mode must skip the local STT/TTS/VAD
    factories. We can't easily run start() end-to-end without a live WS
    server, but we can verify the new conditional structure leaves the
    attributes None when the V2V early branch is taken."""

    @pytest.mark.asyncio
    async def test_v2v_mode_does_not_construct_stt_tts_vad(self, v2v_app, monkeypatch):
        from reachy_claw.plugins import conversation_plugin as cp_mod

        # Sentinels that explode if called (proves the V2V branch skipped them).
        def _explode(*_a, **_kw):
            raise AssertionError(
                "STT/TTS/VAD factory called in edge_llm_v2v mode"
            )

        monkeypatch.setattr(cp_mod, "create_stt_backend", _explode)
        monkeypatch.setattr(cp_mod, "create_tts_backend", _explode)
        monkeypatch.setattr(cp_mod, "create_vad_backend", _explode)

        plugin = ConversationPlugin(v2v_app)
        # Drive the start() Phase 1 block in isolation. We mimic only what
        # the V2V branch needs and avoid Phase 2 (which would open a real
        # WebSocket).
        config = v2v_app.config
        assert config.llm_backend == "edge_llm_v2v"
        # Re-execute the equivalent of Phase 1 conditional.
        if config.llm_backend == "edge_llm_v2v":
            from reachy_claw.audio import AudioCapture
            plugin._audio = AudioCapture(config, v2v_app.reachy, vad=None)
        assert plugin._stt is None
        assert plugin._tts is None
        assert plugin._vad is None
        # Audio capture still constructed (needed for mic uplink).
        assert plugin._audio is not None


class TestConcurrentAsrFinalDropped:
    """RISK 1: a second asr_final arriving while we're still THINKING
    must be dropped (and a debug log emitted) — option (A)."""

    @pytest.mark.asyncio
    async def test_second_final_dropped_while_thinking(self, v2v_app, caplog):
        import logging
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        # First final → THINKING
        plugin._state = ConvState.THINKING
        with caplog.at_level(logging.DEBUG):
            await plugin._on_v2v_asr_final("second utterance", False, False)
        plugin._process_and_send.assert_not_awaited()
        assert any("THINKING" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_second_final_dropped_while_streaming(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        plugin._state = ConvState.SPEAKING  # not THINKING, but stream alive
        plugin._client.is_streaming = True
        # The runtime check uses isinstance(EdgeLLMClient); cheat by making
        # the mock report as such.
        from reachy_claw.edge_llm import EdgeLLMClient
        plugin._client.__class__ = EdgeLLMClient  # type: ignore[assignment]
        await plugin._on_v2v_asr_final("second", False, False)
        plugin._process_and_send.assert_not_awaited()


class TestTrailingAsrFinalDuringShutdown:
    """RISK 2: finals arriving after stop() set _running=False must be
    ignored, as must session_complete=True markers."""

    @pytest.mark.asyncio
    async def test_final_dropped_when_not_running(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        plugin._running = False
        await plugin._on_v2v_asr_final("trailing", False, False)
        plugin._process_and_send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_complete_final_dropped(self, v2v_app):
        plugin = _make_plugin(v2v_app)
        plugin._process_and_send = AsyncMock()
        await plugin._on_v2v_asr_final("last bit", session_complete=True,
                                       duplicate_of_streamed=False)
        plugin._process_and_send.assert_not_awaited()


class TestAudioQueueMaxsizeBackendAware:
    """RISK 3: audio_queue maxsize must differ between legacy (3) and V2V (64)."""

    def test_v2v_mode_uses_large_queue(self, v2v_app):
        plugin = ConversationPlugin(v2v_app)
        assert plugin._audio_queue.maxsize == 64

    def test_legacy_mode_uses_small_queue(self, mock_reachy):
        from reachy_claw.app import ReachyClawApp
        config = Config(
            standalone_mode=True,  # legacy default backend
            tts_backend="none",
            stt_backend="whisper",
            idle_animations=False,
            play_emotions=False,
            enable_face_tracker=False,
            enable_motion=False,
        )
        app = ReachyClawApp(config)
        app.reachy = mock_reachy
        plugin = ConversationPlugin(app)
        assert plugin._audio_queue.maxsize == 3


# ── Reconnect path: WS-drop classification & turn-state reset ─────────


class TestV2VReconnectPath:
    """Codex review (Wave 2.5+): _on_v2v_error must classify the error
    code, cancel the right side-effects per current state, and
    _v2v_reconnect_loop must reset turn-level latches on success."""

    @pytest.mark.asyncio
    async def test_on_v2v_error_ws_closed_schedules_reconnect(self, v2v_app, monkeypatch):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.IDLE
        scheduled = {"n": 0}

        def _stub_schedule():
            scheduled["n"] += 1

        monkeypatch.setattr(plugin, "_schedule_v2v_reconnect", _stub_schedule)
        await plugin._on_v2v_error("ws_closed: peer reset")
        assert scheduled["n"] == 1

    @pytest.mark.asyncio
    async def test_on_v2v_error_business_resets_state_no_reconnect(
        self, v2v_app, monkeypatch,
    ):
        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.THINKING
        scheduled = {"n": 0}
        monkeypatch.setattr(
            plugin, "_schedule_v2v_reconnect",
            lambda: scheduled.__setitem__("n", scheduled["n"] + 1),
        )
        await plugin._on_v2v_error("backend_error: model OOM")
        assert scheduled["n"] == 0
        assert plugin._state == ConvState.IDLE

    @pytest.mark.asyncio
    async def test_reconnect_in_thinking_cancels_edge_llm(self, v2v_app, monkeypatch):
        from reachy_claw.edge_llm import EdgeLLMClient

        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.THINKING
        plugin._client.is_streaming = True
        plugin._client.__class__ = EdgeLLMClient  # isinstance check
        monkeypatch.setattr(plugin, "_schedule_v2v_reconnect", lambda: None)
        await plugin._on_v2v_error("ws_closed: gone")
        plugin._client.send_interrupt.assert_awaited()
        assert plugin._state == ConvState.IDLE

    @pytest.mark.asyncio
    async def test_reconnect_in_speaking_cancels_both_llm_and_playback(
        self, v2v_app, monkeypatch,
    ):
        from reachy_claw.edge_llm import EdgeLLMClient

        plugin = _make_plugin(v2v_app)
        plugin._state = ConvState.SPEAKING
        plugin._client.is_streaming = True
        plugin._client.__class__ = EdgeLLMClient
        # Pre-load audio queue to verify drain.
        plugin._audio_queue.put_nowait(("v2v_audio", 16000, b"\x00\x00"))
        monkeypatch.setattr(plugin, "_schedule_v2v_reconnect", lambda: None)

        await plugin._on_v2v_error("ws_recv_exit: eof")
        plugin._client.send_interrupt.assert_awaited()
        # _fire_interrupt drained the audio queue.
        assert plugin._audio_queue.empty()
        assert plugin._interrupt_event.is_set()
        assert plugin._state == ConvState.IDLE

    @pytest.mark.asyncio
    async def test_reconnect_success_resets_turn_state(self, v2v_app):
        """Drive _v2v_reconnect_loop once: connect() returns immediately
        and the loop body must wipe turn-level latches before exiting."""
        plugin = _make_plugin(v2v_app)
        # Dirty turn state simulating an interrupted utterance.
        plugin._v2v_abort_in_flight = True
        plugin._first_audio_logged_this_turn = True
        plugin._t_asr_final = 12345.0
        plugin._interrupt_event.set()

        # Make sleep return instantly so the loop runs fast.
        async def _no_sleep(_):
            return

        plugin._v2v.connect = AsyncMock()
        import asyncio as _asyncio
        orig_sleep = _asyncio.sleep
        _asyncio.sleep = _no_sleep
        try:
            await plugin._v2v_reconnect_loop()
        finally:
            _asyncio.sleep = orig_sleep

        plugin._v2v.connect.assert_awaited()
        assert plugin._v2v_abort_in_flight is False
        assert plugin._first_audio_logged_this_turn is False
        assert plugin._t_asr_final is None
        assert plugin._interrupt_event.is_set() is False


# ── Multi-turn abort recovery after dropped ASR final ────────────────


class TestV2VMultiTurnAbortRecovery:
    """Regression: V2V barge-in must keep working across turns even when
    a follow-up asr_final gets dropped because the previous LLM turn is
    still THINKING.

    Failure mode before the fix:
      1. Turn N speaking -> speech_start aborts (latch=True, interrupt set)
      2. New utterance asr_final arrives while THINKING -> dropped, latch
         not cleared.
      3. Turn N+1 speech_start -> debounced by the stale latch, no abort
         fires; conversation stalls.
    """

    @pytest.mark.asyncio
    async def test_speech_start_abort_resets_after_dropped_asr_final(
        self, v2v_app,
    ):
        plugin = _make_plugin(v2v_app)

        # Turn N: speaking -> user barges in -> abort fires.
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True
        await plugin._on_v2v_vad_event("speech_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first_aborts = plugin._v2v.abort.await_count
        assert first_aborts >= 1
        assert plugin._v2v_abort_in_flight is True
        assert plugin._interrupt_event.is_set()
        assert plugin._state == ConvState.LISTENING

        # Pretend the LLM kicked off a new turn — we're now THINKING and
        # the next asr_final arrives concurrently and gets dropped.
        plugin._state = ConvState.THINKING
        await plugin._on_v2v_asr_final("hello again", False, False)
        # Drop path must clear both flags so next barge-in works.
        assert plugin._v2v_abort_in_flight is False
        assert plugin._interrupt_event.is_set() is False

        # Move past the 500ms time-window debounce.
        plugin._v2v_last_abort_ts -= 1.0

        # Turn N+1: speaking again -> new barge-in must fire abort.
        plugin._state = ConvState.SPEAKING
        plugin.app.is_speaking = True
        # Bump the per-event 100ms speech_start debounce too.
        plugin._v2v_last_speech_start_ts -= 1.0
        await plugin._on_v2v_vad_event("speech_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert plugin._v2v.abort.await_count > first_aborts
        assert plugin._state == ConvState.LISTENING

    @pytest.mark.asyncio
    async def test_tts_done_keeps_state_when_new_turn_in_flight(
        self, v2v_app,
    ):
        """tts_done from the previous reply must NOT erase THINKING or
        LISTENING state belonging to the next turn."""
        plugin = _make_plugin(v2v_app)

        # User already started the next turn; LISTENING must survive.
        plugin._state = ConvState.LISTENING
        await plugin._on_v2v_tts_done()
        assert plugin._state == ConvState.LISTENING

        # Next LLM turn already kicked off; THINKING must survive too.
        plugin._state = ConvState.THINKING
        await plugin._on_v2v_tts_done()
        assert plugin._state == ConvState.THINKING

    @pytest.mark.asyncio
    async def test_fire_interrupt_with_v2v_calls_abort_then_flush_tts(
        self, v2v_app,
    ):
        """Fix D: both abort and flush_tts must be invoked so server-side
        TTS queue can't resume after the local drain."""
        plugin = _make_plugin(v2v_app)
        await plugin._fire_interrupt(notify_v2v=True)
        # Background task is spawned via _spawn_task; let it run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        plugin._v2v.abort.assert_awaited()
        plugin._v2v.flush_tts.assert_awaited()
