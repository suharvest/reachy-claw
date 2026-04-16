"""Tests for Mode base class, ModeContext, ModeManager, and ConversationMode."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_claw.mode import Mode, ModeContext, ModeManager


class DummyMode(Mode):
    """Concrete Mode for testing."""
    name = "dummy"
    barge_in = True
    temperature = 0.5
    max_history = 3
    skip_emotion_extraction = False
    enable_vlm = True
    play_emotions = True

    def __init__(self):
        self.entered = False
        self.exited = False

    async def enter(self, ctx: ModeContext) -> None:
        self.entered = True

    async def exit(self, ctx: ModeContext) -> None:
        self.exited = True


class DummyMode2(Mode):
    name = "dummy2"
    barge_in = False
    temperature = 0.3
    max_history = 0
    skip_emotion_extraction = True
    enable_vlm = False
    play_emotions = False

    def __init__(self):
        self.entered = False
        self.exited = False

    async def enter(self, ctx: ModeContext) -> None:
        self.entered = True

    async def exit(self, ctx: ModeContext) -> None:
        self.exited = True


def _make_ctx() -> ModeContext:
    app = MagicMock()
    app.config = MagicMock()
    app.config.barge_in_enabled = True
    app.config.ollama_temperature = 0.7
    app.config.ollama_max_history = 3
    app.config.enable_vlm = True
    app.config.ollama_system_prompt = "default prompt"
    events = MagicMock()
    return ModeContext(
        app=app,
        sentence_queue=asyncio.Queue(),
        audio_queue=asyncio.Queue(),
        events=events,
        spawn_task=MagicMock(),
        get_vision_context=MagicMock(return_value=None),
        capture_frame=MagicMock(return_value=None),
    )


class TestMode:
    def test_preprocess_utterance_default_returns_text(self):
        mode = DummyMode()
        ctx = _make_ctx()
        assert mode.preprocess_utterance("hello", ctx) == "hello"

    def test_on_speaking_audio_respects_barge_in(self):
        mode = DummyMode()
        mode.barge_in = True
        ctx = _make_ctx()
        assert mode.on_speaking_audio(b"chunk", ctx) == "barge_in"

    def test_on_speaking_audio_ignore_when_no_barge_in(self):
        mode = DummyMode()
        mode.barge_in = False
        ctx = _make_ctx()
        assert mode.on_speaking_audio(b"chunk", ctx) == "ignore"

    def test_get_ollama_config_returns_overrides(self):
        mode = DummyMode()
        overrides = mode.get_ollama_config(MagicMock())
        assert overrides["temperature"] == 0.5
        assert overrides["max_history"] == 3
        assert overrides["skip_emotion_extraction"] is False
        assert overrides["enable_vlm"] is True

    def test_get_ollama_config_omits_none_values(self):
        """Mode with None fields should not include them in overrides."""
        mode = DummyMode()
        mode.temperature = None
        mode.max_history = None
        overrides = mode.get_ollama_config(MagicMock())
        assert "temperature" not in overrides
        assert "max_history" not in overrides


class TestModeManager:
    @pytest.mark.asyncio
    async def test_register_and_switch(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        m1 = DummyMode()
        m2 = DummyMode2()
        mgr.register(m1)
        mgr.register(m2)

        await mgr.switch("dummy")
        assert mgr.current is m1
        assert m1.entered is True

    @pytest.mark.asyncio
    async def test_switch_calls_exit_then_enter(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        m1 = DummyMode()
        m2 = DummyMode2()
        mgr.register(m1)
        mgr.register(m2)

        await mgr.switch("dummy")
        assert m1.entered is True

        await mgr.switch("dummy2")
        assert m1.exited is True
        assert m2.entered is True

    @pytest.mark.asyncio
    async def test_switch_same_mode_is_noop(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        m1 = DummyMode()
        mgr.register(m1)

        await mgr.switch("dummy")
        m1.entered = False  # reset
        await mgr.switch("dummy")
        assert m1.entered is False  # not re-entered

    @pytest.mark.asyncio
    async def test_switch_emits_mode_change_event(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        mgr.register(DummyMode())
        mgr.register(DummyMode2())

        await mgr.switch("dummy")
        await mgr.switch("dummy2")

        calls = [c for c in ctx.events.emit.call_args_list if c[0][0] == "mode_change"]
        assert len(calls) == 2
        payload = calls[1][0][1]
        assert payload["mode"] == "dummy2"
        assert payload["prev"] == "dummy"

    @pytest.mark.asyncio
    async def test_switch_applies_barge_in(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        m1 = DummyMode()   # barge_in = True
        m2 = DummyMode2()  # barge_in = False
        mgr.register(m1)
        mgr.register(m2)

        await mgr.switch("dummy")
        assert ctx.app.config.barge_in_enabled is True

        await mgr.switch("dummy2")
        assert ctx.app.config.barge_in_enabled is False

    @pytest.mark.asyncio
    async def test_switch_unknown_mode_raises(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        with pytest.raises(KeyError):
            await mgr.switch("nonexistent")

    @pytest.mark.asyncio
    async def test_current_before_switch_raises(self):
        ctx = _make_ctx()
        mgr = ModeManager(ctx)
        with pytest.raises(RuntimeError):
            _ = mgr.current


from reachy_claw.modes import ConversationMode


class TestConversationMode:
    def test_name(self):
        m = ConversationMode()
        assert m.name == "conversation"

    def test_barge_in_enabled(self):
        m = ConversationMode()
        assert m.barge_in is True

    def test_play_emotions(self):
        m = ConversationMode()
        assert m.play_emotions is True

    def test_preprocess_injects_vision_context(self):
        m = ConversationMode()
        ctx = _make_ctx()
        ctx.get_vision_context.return_value = "Alice looks happy"
        result = m.preprocess_utterance("hello", ctx)
        assert result == "[Faces: Alice looks happy]\nhello"

    def test_preprocess_no_vision_returns_text(self):
        m = ConversationMode()
        ctx = _make_ctx()
        ctx.get_vision_context.return_value = None
        result = m.preprocess_utterance("hello", ctx)
        assert result == "hello"

    def test_on_speaking_audio_returns_barge_in(self):
        m = ConversationMode()
        ctx = _make_ctx()
        assert m.on_speaking_audio(b"chunk", ctx) == "barge_in"

    def test_get_ollama_config_minimal(self):
        m = ConversationMode()
        overrides = m.get_ollama_config(MagicMock())
        assert overrides["skip_emotion_extraction"] is False
        assert "temperature" not in overrides
        assert "max_history" not in overrides


from reachy_claw.modes.monologue import MonologueMode


class TestMonologueMode:
    def test_name(self):
        m = MonologueMode()
        assert m.name == "monologue"

    def test_config_overrides(self):
        m = MonologueMode()
        assert m.barge_in is False
        assert m.temperature == 0.9
        assert m.skip_emotion_extraction is False
        assert m.play_emotions is False

    def test_on_speaking_audio_returns_bg_listen(self):
        m = MonologueMode()
        ctx = _make_ctx()
        assert m.on_speaking_audio(b"chunk", ctx) == "bg_listen"

    @pytest.mark.asyncio
    async def test_enter_starts_timer(self):
        m = MonologueMode()
        ctx = _make_ctx()
        ctx.app.config.monologue_interval = 5.0
        mock_task = MagicMock()
        ctx.spawn_task.return_value = mock_task

        await m.enter(ctx)

        ctx.spawn_task.assert_called_once()
        assert m._timer_task is mock_task

    @pytest.mark.asyncio
    async def test_exit_cancels_timer(self):
        m = MonologueMode()
        ctx = _make_ctx()
        ctx.app.config.monologue_interval = 5.0
        mock_task = MagicMock()
        mock_task.done.return_value = False
        ctx.spawn_task.return_value = mock_task

        await m.enter(ctx)
        await m.exit(ctx)

        mock_task.cancel.assert_called_once()

    def test_compose_monologue_prompt_with_transcript(self):
        m = MonologueMode()
        ctx = _make_ctx()
        ctx.app.get_plugin = MagicMock(return_value=None)
        result = m.compose_monologue_prompt("hello world", ctx)
        assert 'heard: "hello world"' in result

    def test_compose_monologue_prompt_no_input(self):
        m = MonologueMode()
        ctx = _make_ctx()
        ctx.app.get_plugin = MagicMock(return_value=None)
        result = m.compose_monologue_prompt(None, ctx)
        assert "nobody around" in result

    def test_preprocess_utterance_calls_compose(self):
        m = MonologueMode()
        ctx = _make_ctx()
        ctx.app.get_plugin = MagicMock(return_value=None)
        result = m.preprocess_utterance("hi", ctx)
        assert 'heard: "hi"' in result


from reachy_claw.modes.interpreter import InterpreterMode


class TestInterpreterMode:
    def test_name(self):
        m = InterpreterMode()
        assert m.name == "interpreter"

    def test_config_overrides(self):
        m = InterpreterMode()
        assert m.barge_in is False
        assert m.temperature == 0.3
        assert m.max_history == 0
        assert m.skip_emotion_extraction is True
        assert m.enable_vlm is False
        assert m.play_emotions is False

    def test_on_speaking_audio_returns_ignore(self):
        m = InterpreterMode()
        ctx = _make_ctx()
        assert m.on_speaking_audio(b"chunk", ctx) == "ignore"

    def test_preprocess_utterance_returns_none(self):
        """Interpreter handles sending itself via sequencer."""
        m = InterpreterMode()
        ctx = _make_ctx()
        m._sequencer = MagicMock()
        result = m.preprocess_utterance("hello", ctx)
        assert result is None
        m._sequencer.submit.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_exit_stops_sequencer_and_drains(self):
        m = InterpreterMode()
        ctx = _make_ctx()
        await ctx.sentence_queue.put("stale")
        await ctx.audio_queue.put("stale")

        mock_seq = AsyncMock()
        m._sequencer = mock_seq

        await m.exit(ctx)

        mock_seq.stop.assert_awaited_once()
        assert ctx.sentence_queue.empty()
        assert ctx.audio_queue.empty()
