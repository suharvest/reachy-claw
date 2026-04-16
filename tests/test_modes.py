"""Tests for Mode base class, ModeContext, and ModeManager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_claw.mode import Mode, ModeContext


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
