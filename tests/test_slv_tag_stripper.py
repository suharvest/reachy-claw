"""Unit tests for the SLV ConversationPlugin stream-safe TTS tag stripper.

Proves that emotion / [Faces: ...] tags split across LLM stream tokens are
DROPPED before reaching the SLV engine ``send_text`` (so they are never
spoken aloud), while still queuing the emotion (motion / dashboard keep
working). Reuses the fake-SLV harness style from
``tests/e2e_slv_plugin_local.py`` but runs fully offline (no ollama / no WS).
"""
from __future__ import annotations

import asyncio

import pytest

from reachy_claw.config import Config
from reachy_claw.app import ReachyClawApp
from reachy_claw.plugins import conversation_plugin_slv
from reachy_claw.plugins.conversation_plugin_slv import ConversationPlugin
from ovs_agent.slv_client import ASRFinal


class FakeSLV:
    """Stub SLVClient: records every send_text chunk; no real WS."""

    def __init__(self):
        self.text_chunks: list[str] = []
        self.flushed = False
        self.aborted = False

    async def send_text(self, text: str):
        if text:
            self.text_chunks.append(text)

    async def flush_tts(self):
        self.flushed = True

    async def abort(self):
        self.aborted = True


class FakeMotion:
    name = "motion"

    def __init__(self):
        self.opened: list[float | None] = []
        self.locked: list[float | None] = []

    def open_interaction_window(self, duration=None):
        self.opened.append(duration)

    def center_and_lock_tracking(self, duration=None):
        self.locked.append(duration)


@pytest.fixture
def slv_plugin():
    cfg = Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    cfg.conversation_backend = "slv"
    app = ReachyClawApp(cfg)
    app.reachy = None
    plugin = ConversationPlugin(app)
    plugin._slv = FakeSLV()
    return plugin


async def _feed(plugin, tokens):
    """Run a token stream through the stripper exactly as _on_tok does."""
    plugin._reset_tag_stripper()
    for tok in tokens:
        clean = plugin._strip_tags_for_tts(tok)
        if clean:
            await plugin._slv.send_text(clean)
    tail = plugin._flush_tag_stripper()
    if tail:
        await plugin._slv.send_text(tail)


class TestTagStripper:
    @pytest.mark.asyncio
    async def test_cross_token_emotion_tag_stripped_and_queued(self, slv_plugin):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)
        events: list[dict] = []
        slv_plugin.app.events.subscribe("emotion", lambda d: events.append(d))

        # "[", "happy", "]" split across tokens — a per-token regex misses it.
        await _feed(
            slv_plugin,
            ["Hello", " there", " [", "happy", "]"],
        )

        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "Hello there "  # tag dropped, trailing space eaten
        assert "[" not in spoken and "happy" not in spoken
        assert emotions == ["happy"]
        assert events and events[-1]["emotion"] == "happy"

    @pytest.mark.asyncio
    async def test_emotion_prefix_form(self, slv_plugin):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)

        await _feed(slv_plugin, ["Hi ", "[emotion:", "sad", "]", " bye"])

        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "Hi bye"
        assert emotions == ["sad"]

    @pytest.mark.asyncio
    async def test_cross_token_faces_tag_dropped_no_emotion(self, slv_plugin):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)

        await _feed(
            slv_plugin,
            ["Hi ", "[Faces:", " Alice", "]", " welcome"],
        )

        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "Hi welcome"
        assert "Faces" not in spoken and "Alice" not in spoken
        assert emotions == []  # vision context is NOT an emotion

    @pytest.mark.asyncio
    async def test_non_tag_bracket_preserved(self, slv_plugin):
        # A bracket that isn't an emotion/vision tag should survive.
        await _feed(slv_plugin, ["See ", "[note this]", " ok"])
        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "See [note this] ok"

    @pytest.mark.asyncio
    async def test_unterminated_tail_flushed(self, slv_plugin):
        # Stream ends mid-buffer: "[partial" was never a tag → must survive.
        await _feed(slv_plugin, ["Hello ", "[part", "ial"])
        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "Hello [partial"


class TestSlvModeSwitching:
    def test_switch_mode_queues_mode_entry_gesture(self, slv_plugin):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)

        slv_plugin.switch_mode("conversation")
        slv_plugin.switch_mode("interpreter")
        slv_plugin.switch_mode("monologue")

        assert emotions == ["happy", "listening", "curious"]

    @pytest.mark.asyncio
    async def test_switch_to_monologue_updates_config_and_starts_timer(self, slv_plugin):
        slv_plugin._running = True

        slv_plugin.switch_mode("monologue")

        assert slv_plugin.app.config.conversation_mode == "monologue"
        assert slv_plugin._monologue_task is not None
        assert not slv_plugin._monologue_task.done()

        task = slv_plugin._monologue_task
        slv_plugin.switch_mode("conversation")
        await asyncio.sleep(0)

        assert slv_plugin.app.config.conversation_mode == "conversation"
        assert slv_plugin._monologue_task is None
        assert task.cancelled()

    def test_interpreter_mode_uses_translation_prompt(self, slv_plugin):
        slv_plugin.app.config.interpreter_source_lang = "Chinese"
        slv_plugin.app.config.interpreter_target_lang = "English"

        slv_plugin.switch_mode("interpreter")

        prompt = slv_plugin._build_system_prompt()
        assert "translation machine" in prompt
        assert "Chinese to English" in prompt
        assert "cute robot at an exhibition" not in prompt

    def test_switch_mode_resets_session_when_mode_changes(self, slv_plugin):
        slv_plugin._session.add_user("previous request")
        slv_plugin._session.add_assistant("previous answer")
        slv_plugin._session.prefix_cache_warmed = True

        slv_plugin.switch_mode("interpreter")

        assert slv_plugin._session.history == []
        assert slv_plugin._session.prefix_cache_warmed is False

    def test_switch_mode_keeps_session_when_mode_is_unchanged(self, slv_plugin):
        slv_plugin.app.config.conversation_mode = "conversation"
        slv_plugin._session.add_user("previous request")

        slv_plugin.switch_mode("conversation")

        assert slv_plugin._session.history == [
            {"role": "user", "content": "previous request"}
        ]


class TestSlvInteractionWindow:
    @pytest.mark.asyncio
    async def test_asr_final_opens_motion_interaction_window(self, slv_plugin, monkeypatch):
        motion = FakeMotion()
        slv_plugin.app._plugins.append(motion)
        slv_plugin._running = True

        def fake_spawn(coro, *, name):
            coro.close()

            class DoneTask:
                def done(self):
                    return True

            return DoneTask()

        monkeypatch.setattr(slv_plugin, "_spawn_task", fake_spawn)

        await slv_plugin._dispatch_slv_event(
            ASRFinal("hello", session_complete=False)
        )

        assert motion.opened == [None]

    @pytest.mark.asyncio
    async def test_barge_in_centers_and_locks_motion(self, slv_plugin):
        motion = FakeMotion()
        slv_plugin.app._plugins.append(motion)
        slv_plugin._state = conversation_plugin_slv.ConvState.SPEAKING
        slv_plugin._gst_playing = True
        slv_plugin._speaking_since_ts = 0.0

        await slv_plugin._maybe_barge_in("hello")

        assert motion.locked == [None]


class TestSlvAudioRouting:
    @pytest.mark.asyncio
    async def test_v2v_pcm_prefers_duplex_audio_when_sdk_audio_exists(
        self, slv_plugin, mock_reachy
    ):
        class FakeDuplexAudio:
            def __init__(self):
                self.enqueued = []
                self._duplex_stream = object()

            async def enqueue_playback_async(self, samples):
                self.enqueued.append(samples)

        duplex_audio = FakeDuplexAudio()
        mock_reachy.media.audio = object()
        slv_plugin.app.reachy = mock_reachy
        slv_plugin._audio = duplex_audio

        pcm = b"\x00\x01" * 320

        await slv_plugin._play_v2v_pcm(16000, pcm)

        assert len(duplex_audio.enqueued) == 1
        mock_reachy.media.start_playing.assert_not_called()
        mock_reachy.media.push_audio_sample.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_playback_drains_duplex_route_when_sdk_audio_exists(
        self, slv_plugin, mock_reachy
    ):
        class FakeDuplexAudio:
            def __init__(self):
                self.drained = 0

            async def await_playback_drained(self):
                self.drained += 1

        duplex_audio = FakeDuplexAudio()
        mock_reachy.media.audio = object()
        slv_plugin.app.reachy = mock_reachy
        slv_plugin._audio = duplex_audio
        slv_plugin._gst_playing = True
        slv_plugin._playback_route = "duplex"

        await slv_plugin._stop_gst_playback()

        assert duplex_audio.drained == 1
        assert slv_plugin._gst_playing is False
        assert slv_plugin._playback_route is None
        mock_reachy.media.stop_playing.assert_not_called()

    @pytest.mark.asyncio
    async def test_immediate_stop_drops_duplex_playback_without_drain(
        self, slv_plugin
    ):
        class FakeDuplexAudio:
            def __init__(self):
                self.drained = 0
                self.dropped = 0

            async def await_playback_drained(self):
                self.drained += 1

            def drain_playback(self):
                self.dropped += 1

        duplex_audio = FakeDuplexAudio()
        slv_plugin._audio = duplex_audio
        slv_plugin._gst_playing = True
        slv_plugin._playback_route = "duplex"

        await slv_plugin._stop_gst_playback(immediate=True)

        assert duplex_audio.dropped == 1
        assert duplex_audio.drained == 0
        assert slv_plugin._gst_playing is False
        assert slv_plugin._playback_route is None


class TestSlvMonologue:
    @pytest.mark.asyncio
    async def test_monologue_queues_angle_specific_micro_gesture(
        self, slv_plugin, monkeypatch
    ):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)

        async def fake_stream_with_tools(*args, **kwargs):
            await kwargs["on_assistant_token"]("I notice something new.")
            return "I notice something new."

        monkeypatch.setattr(
            conversation_plugin_slv, "stream_with_tools", fake_stream_with_tools
        )
        slv_plugin._llm = object()

        await slv_plugin._run_monologue()

        assert emotions == ["thinking", "curious"]

    @pytest.mark.asyncio
    async def test_monologue_adds_variation_and_sampling(self, slv_plugin, monkeypatch):
        calls: list[dict] = []

        async def fake_stream_with_tools(*args, **kwargs):
            messages = args[1]
            calls.append({"messages": messages, "kwargs": kwargs})
            await kwargs["on_assistant_token"]("I notice something new. [curious]")
            return "I notice something new. [curious]"

        monkeypatch.setattr(
            conversation_plugin_slv, "stream_with_tools", fake_stream_with_tools
        )
        slv_plugin._llm = object()

        await slv_plugin._run_monologue()
        await slv_plugin._run_monologue()

        prompts = [call["messages"][-1]["content"] for call in calls]
        assert prompts[0] != prompts[1]
        assert "Avoid repeating recent lines" in prompts[1]
        assert calls[0]["kwargs"]["llm_kwargs"]["temperature"] == 0.8
        assert calls[0]["kwargs"]["llm_kwargs"]["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_monologue_does_not_speak_bare_emotion_word(self, slv_plugin, monkeypatch):
        emotions: list[str] = []
        slv_plugin.app.emotions.queue_emotion = lambda e: emotions.append(e)

        async def fake_stream_with_tools(*args, **kwargs):
            await kwargs["on_assistant_token"]("I'm glad to meet everyone! ")
            await kwargs["on_assistant_token"]("excited")
            return "I'm glad to meet everyone! excited"

        monkeypatch.setattr(
            conversation_plugin_slv, "stream_with_tools", fake_stream_with_tools
        )
        slv_plugin._llm = object()

        await slv_plugin._run_monologue()

        spoken = "".join(slv_plugin._slv.text_chunks)
        assert spoken == "I'm glad to meet everyone! "
        assert "excited" not in spoken
        assert emotions == ["thinking", "curious", "excited"]
