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
