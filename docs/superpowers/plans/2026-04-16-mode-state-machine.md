# Mode State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract conversation modes (conversation/monologue/interpreter) from ConversationPlugin's monolithic switch_mode() into pluggable Mode strategy objects managed by a ModeManager.

**Architecture:** Each mode is a Mode subclass declaring config overrides and implementing enter/exit/preprocess_utterance/on_speaking_audio. ModeManager handles lifecycle (exit → restore config → snapshot config → apply → enter) and broadcasts mode_change events. ConversationPlugin delegates to ModeManager instead of branching on boolean flags.

**Tech Stack:** Python 3.10+, asyncio, pytest, existing EventBus

**Spec:** `docs/superpowers/specs/2026-04-16-mode-state-machine-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| New | `src/reachy_claw/mode.py` | Mode ABC, ModeContext dataclass, ModeManager |
| New | `src/reachy_claw/modes/__init__.py` | Re-export three Mode subclasses |
| New | `src/reachy_claw/modes/conversation.py` | ConversationMode (default, barge-in, vision context injection) |
| New | `src/reachy_claw/modes/monologue.py` | MonologueMode (timer, bg_listen, compose_prompt) |
| New | `src/reachy_claw/modes/interpreter.py` | InterpreterMode + _InterpreterSequencer (moved from conversation_plugin.py) |
| Modify | `src/reachy_claw/plugins/conversation_plugin.py` | Remove boolean flags, switch_mode body, scattered if/elif; delegate to ModeManager |
| New | `tests/test_modes.py` | Mode + ModeManager unit tests |
| Modify | `tests/test_conversation_plugin.py` | Adapt TestSwitchMode and imports |

---

### Task 1: Mode base class and ModeContext

**Files:**
- Create: `src/reachy_claw/mode.py`
- Test: `tests/test_modes.py`

- [ ] **Step 1: Write the failing tests for Mode and ModeContext**

Create `tests/test_modes.py`:

```python
"""Tests for Mode base class, ModeContext, and ModeManager."""

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reachy_claw.mode'`

- [ ] **Step 3: Implement Mode base class, ModeContext, and get_ollama_config**

Create `src/reachy_claw/mode.py`:

```python
"""Mode abstraction for conversation modes.

Each conversation mode (conversation, monologue, interpreter) is a Mode
subclass that declares config overrides and implements enter/exit lifecycle.
ModeManager handles switching between modes.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .app import ReachyClawApp
    from .event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class ModeContext:
    """Facade over ConversationPlugin internals available to Mode objects."""

    app: ReachyClawApp
    sentence_queue: asyncio.Queue
    audio_queue: asyncio.Queue
    events: EventBus
    spawn_task: Callable  # plugin._spawn_task
    get_vision_context: Callable  # returns str | None
    capture_frame: Callable  # returns str | None (base64 JPEG)


class Mode(ABC):
    """Base class for a conversation mode.

    Subclasses declare config overrides as class attributes (None = use
    app.config default) and implement enter/exit for sub-task lifecycle.
    """

    name: str = "unnamed"

    # Declarative config overrides (None = use app.config default)
    barge_in: bool | None = None
    temperature: float | None = None
    max_history: int | None = None
    skip_emotion_extraction: bool = False
    enable_vlm: bool | None = None
    system_prompt: str | None = None
    play_emotions: bool = True

    def get_ollama_config(self, app_config: Any) -> dict[str, Any]:
        """Return dict of OllamaConfig field overrides for this mode.

        Only includes fields that are not None, so the caller can merge
        with defaults: ``{**defaults, **overrides}``.
        """
        overrides: dict[str, Any] = {}
        overrides["skip_emotion_extraction"] = self.skip_emotion_extraction
        if self.temperature is not None:
            overrides["temperature"] = self.temperature
        if self.max_history is not None:
            overrides["max_history"] = self.max_history
        if self.enable_vlm is not None:
            overrides["enable_vlm"] = self.enable_vlm
        if self.system_prompt is not None:
            overrides["system_prompt"] = self.system_prompt
        return overrides

    async def enter(self, ctx: ModeContext) -> None:
        """Called when this mode becomes active. Start sub-tasks here."""

    async def exit(self, ctx: ModeContext) -> None:
        """Called when this mode is deactivated. Cleanup sub-tasks here."""

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        """Mode-specific preprocessing before sending to LLM.

        Return the (possibly modified) text to send via the normal LLM path.
        Return None to indicate the mode handles sending itself (e.g. interpreter).
        """
        return text

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        """Behavior when audio arrives during SPEAKING state.

        Returns: "barge_in" | "bg_listen" | "ignore".
        """
        return "barge_in" if self.barge_in else "ignore"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py::TestMode -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reachy_claw/mode.py tests/test_modes.py
git commit -m "feat: add Mode base class and ModeContext"
```

---

### Task 2: ModeManager

**Files:**
- Modify: `src/reachy_claw/mode.py` (append ModeManager class)
- Test: `tests/test_modes.py` (append TestModeManager)

- [ ] **Step 1: Write the failing tests for ModeManager**

Append to `tests/test_modes.py`:

```python
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
        # Second call should have prev="dummy"
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py::TestModeManager -v`
Expected: FAIL — `ImportError: cannot import name 'ModeManager'`

- [ ] **Step 3: Implement ModeManager**

Append to `src/reachy_claw/mode.py`:

```python
class ModeManager:
    """Manages conversation mode lifecycle and switching."""

    def __init__(self, ctx: ModeContext) -> None:
        self._ctx = ctx
        self._modes: dict[str, Mode] = {}
        self._current: Mode | None = None

    def register(self, mode: Mode) -> None:
        """Register a mode. Can be called before the first switch."""
        self._modes[mode.name] = mode

    @property
    def current(self) -> Mode:
        """The currently active mode. Raises RuntimeError if no mode is active."""
        if self._current is None:
            raise RuntimeError("No mode is active — call switch() first")
        return self._current

    async def switch(self, name: str) -> None:
        """Switch to a different mode.

        Calls exit() on current mode, applies new mode's config, then
        calls enter(). No-op if already in the requested mode.
        """
        if self._current and self._current.name == name:
            return

        new_mode = self._modes[name]  # KeyError if unknown
        prev_name = self._current.name if self._current else None

        # Exit current mode
        if self._current:
            await self._current.exit(self._ctx)

        # Apply barge-in setting
        if new_mode.barge_in is not None:
            self._ctx.app.config.barge_in_enabled = new_mode.barge_in

        # Enter new mode
        self._current = new_mode
        await new_mode.enter(self._ctx)

        self._ctx.app.config.conversation_mode = name
        self._ctx.events.emit("mode_change", {"mode": name, "prev": prev_name})
        logger.info("Mode switched: %s → %s", prev_name, name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py -v`
Expected: All 12 tests PASS (5 TestMode + 7 TestModeManager)

- [ ] **Step 5: Commit**

```bash
git add src/reachy_claw/mode.py tests/test_modes.py
git commit -m "feat: add ModeManager with lifecycle and event broadcasting"
```

---

### Task 3: ConversationMode

**Files:**
- Create: `src/reachy_claw/modes/__init__.py`
- Create: `src/reachy_claw/modes/conversation.py`
- Test: `tests/test_modes.py` (append TestConversationMode)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
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
        # Conversation mode: skip_emotion_extraction=False, rest is None (defaults)
        assert overrides["skip_emotion_extraction"] is False
        assert "temperature" not in overrides
        assert "max_history" not in overrides
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py::TestConversationMode -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reachy_claw.modes'`

- [ ] **Step 3: Implement ConversationMode**

Create `src/reachy_claw/modes/__init__.py`:

```python
from .conversation import ConversationMode
from .monologue import MonologueMode
from .interpreter import InterpreterMode

__all__ = ["ConversationMode", "MonologueMode", "InterpreterMode"]
```

Create `src/reachy_claw/modes/conversation.py`:

```python
"""ConversationMode — standard interactive dialogue."""

from __future__ import annotations

from typing import Any

from ..mode import Mode, ModeContext


class ConversationMode(Mode):
    """Standard back-and-forth conversation with barge-in support."""

    name = "conversation"
    barge_in = True
    play_emotions = True

    # All other config overrides are None → use app.config defaults

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        vision_ctx = ctx.get_vision_context()
        if vision_ctx:
            return f"[Faces: {vision_ctx}]\n{text}"
        return text

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "barge_in"
```

Also create placeholder files so the `__init__.py` import doesn't fail (these will be fully implemented in Tasks 4 and 5):

Create `src/reachy_claw/modes/monologue.py`:

```python
"""MonologueMode — placeholder, implemented in Task 4."""

from ..mode import Mode


class MonologueMode(Mode):
    name = "monologue"
```

Create `src/reachy_claw/modes/interpreter.py`:

```python
"""InterpreterMode — placeholder, implemented in Task 5."""

from ..mode import Mode


class InterpreterMode(Mode):
    name = "interpreter"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py -v`
Expected: All 19 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reachy_claw/modes/ tests/test_modes.py
git commit -m "feat: add ConversationMode with vision context injection"
```

---

### Task 4: MonologueMode

**Files:**
- Rewrite: `src/reachy_claw/modes/monologue.py`
- Test: `tests/test_modes.py` (append TestMonologueMode)

The monologue logic is moved from `conversation_plugin.py:684-722` (`_compose_monologue_prompt`), `conversation_plugin.py:1578-1636` (`_bg_listen`), and `conversation_plugin.py:1771-1810` (`_monologue_timer`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py::TestMonologueMode -v`
Expected: FAIL — `MonologueMode` is still a placeholder

- [ ] **Step 3: Implement MonologueMode**

Rewrite `src/reachy_claw/modes/monologue.py`. Move `_compose_monologue_prompt` (conversation_plugin.py:684-722), `_bg_listen` (conversation_plugin.py:1578-1636), and `_monologue_timer` logic (conversation_plugin.py:1771-1810) into this class:

```python
"""MonologueMode — autonomous self-talk with background listening."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from ..mode import Mode, ModeContext

logger = logging.getLogger(__name__)


class MonologueMode(Mode):
    """Robot autonomously generates speech based on vision and background audio."""

    name = "monologue"
    barge_in = False
    temperature = 0.9  # minimum
    skip_emotion_extraction = False
    play_emotions = False

    def __init__(self) -> None:
        self._timer_task: asyncio.Task | None = None
        self.last_speech_time: float = 0.0
        self.pending_speech: str | None = None
        self._bg_speech_frames: list[np.ndarray] = []
        self._bg_silence_count: int = 0

    async def enter(self, ctx: ModeContext) -> None:
        self.last_speech_time = time.monotonic()
        self.pending_speech = None
        self._bg_speech_frames = []
        self._bg_silence_count = 0
        self._timer_task = ctx.spawn_task(
            self._timer_loop(ctx), name="monologue_timer"
        )

    async def exit(self, ctx: ModeContext) -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self.pending_speech = None
        self._bg_speech_frames.clear()

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        return self.compose_monologue_prompt(text, ctx)

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "bg_listen"

    def compose_monologue_prompt(self, transcript: str | None, ctx: ModeContext) -> str:
        """Build natural-language LLM input from speech + vision.

        Moved from ConversationPlugin._compose_monologue_prompt.
        """
        parts = []
        if transcript:
            parts.append(f'heard: "{transcript}"')

        vision = ctx.app.get_plugin("vision_client")
        if vision and getattr(vision, "_last_faces_summary", None):
            faces = vision._last_faces_summary
            named: dict[str, str] = {}
            stranger_count = 0
            for f in faces:
                name = f.get("identity")
                emo = f.get("emotion", "neutral")
                if name:
                    named[name] = emo
                else:
                    stranger_count += 1
            descs = [f"{n} looks {e}" for n, e in named.items()]
            real_strangers = max(0, stranger_count - len(named))
            if real_strangers == 1:
                descs.append("someone you don't know")
            elif real_strangers > 1:
                descs.append(f"{real_strangers} people you don't know")
            if descs:
                parts.append(f"you see: {', '.join(descs)}")
        elif vision:
            emo = getattr(vision, "_last_emotion", None)
            identity = getattr(vision, "current_identity", None)
            if identity:
                parts.append(f"you see: {identity} looks {emo or 'neutral'}")
            elif emo and emo != "neutral":
                parts.append(f"you see: someone looks {emo}")

        if not parts:
            parts.append("nobody around")
        return ". ".join(parts)

    async def bg_listen(
        self, chunk: np.ndarray, streaming_stt: bool, plugin: Any
    ) -> None:
        """Background listening during SPEAKING/THINKING in monologue mode.

        Moved from ConversationPlugin._bg_listen. Needs access to plugin's
        _stt, _audio, _vad, and app.config for STT streaming.
        """
        max_silence_bg = int(
            plugin.app.config.silence_duration
            * plugin.app.config.sample_rate
            / 1024
        )
        has_speech = await asyncio.to_thread(plugin._audio._detect_speech, chunk)

        if has_speech:
            if not self._bg_speech_frames:
                logger.debug("BG listen: speech detected while speaking")
                if streaming_stt:
                    plugin._stt.cancel_stream()
                    await asyncio.to_thread(
                        plugin._stt.start_stream, plugin.app.config.sample_rate
                    )
            self._bg_speech_frames.append(chunk)
            self._bg_silence_count = 0

            if streaming_stt:
                partial = await asyncio.to_thread(plugin._stt.feed_chunk, chunk)
                if partial and partial.text:
                    plugin.app.events.emit("asr_partial", {"text": partial.text})
                    if partial.is_final:
                        text = await asyncio.to_thread(plugin._stt.finish_stream)
                        if text and text.strip():
                            logger.info(f'BG heard: "{text}"')
                            plugin.app.events.emit("asr_final", {"text": text})
                            self.pending_speech = text
                        self._bg_speech_frames = []
                        self._bg_silence_count = 0
                        if plugin._vad:
                            plugin._vad.reset()

        elif self._bg_speech_frames:
            self._bg_silence_count += 1
            self._bg_speech_frames.append(chunk)
            if streaming_stt:
                await asyncio.to_thread(plugin._stt.feed_chunk, chunk)

            if self._bg_silence_count >= max_silence_bg:
                if streaming_stt:
                    text = await asyncio.to_thread(plugin._stt.finish_stream)
                    if text and text.strip():
                        logger.info(f'BG heard: "{text}"')
                        plugin.app.events.emit("asr_final", {"text": text})
                        self.pending_speech = text
                self._bg_speech_frames = []
                self._bg_silence_count = 0
                if plugin._vad:
                    plugin._vad.reset()

    async def _timer_loop(self, ctx: ModeContext) -> None:
        """Periodically trigger monologue when idle.

        Moved from ConversationPlugin._monologue_timer. This is a long-running
        task spawned via ctx.spawn_task in enter(). The actual LLM call is
        dispatched via the plugin — the timer just checks conditions and
        emits a "monologue_trigger" event with the composed prompt.
        """
        interval = ctx.app.config.monologue_interval
        logger.info(f"Monologue timer started (interval={interval}s)")

        while True:
            await asyncio.sleep(1.0)
            interval = ctx.app.config.monologue_interval

            if time.monotonic() - self.last_speech_time < interval:
                continue

            transcript = self.pending_speech
            self.pending_speech = None
            prompt = self.compose_monologue_prompt(transcript, ctx)
            ctx.events.emit("monologue_trigger", {"prompt": prompt})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py -v`
Expected: All 27 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reachy_claw/modes/monologue.py tests/test_modes.py
git commit -m "feat: add MonologueMode with timer and background listening"
```

---

### Task 5: InterpreterMode

**Files:**
- Rewrite: `src/reachy_claw/modes/interpreter.py`
- Test: `tests/test_modes.py` (append TestInterpreterMode)

Move `_InterpreterSequencer` (conversation_plugin.py:66-168) into this file.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modes.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py::TestInterpreterMode -v`
Expected: FAIL — InterpreterMode is still a placeholder

- [ ] **Step 3: Implement InterpreterMode**

Rewrite `src/reachy_claw/modes/interpreter.py`. Move `_InterpreterSequencer` (conversation_plugin.py:66-168) and `_drain_queue` (conversation_plugin.py:2578-2584) into this file:

```python
"""InterpreterMode — real-time concurrent translation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..mode import Mode, ModeContext

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def _drain_queue(q: asyncio.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


class InterpreterMode(Mode):
    """Real-time translation: concurrent requests, ordered output."""

    name = "interpreter"
    barge_in = False
    temperature = 0.3
    max_history = 0
    skip_emotion_extraction = True
    enable_vlm = False
    play_emotions = False

    def __init__(self) -> None:
        self._sequencer: _InterpreterSequencer | None = None

    async def enter(self, ctx: ModeContext) -> None:
        self._sequencer = _InterpreterSequencer(
            config=ctx.app.config,
            sentence_queue=ctx.sentence_queue,
            events=ctx.events,
        )
        await self._sequencer.start()

    async def exit(self, ctx: ModeContext) -> None:
        if self._sequencer:
            await self._sequencer.stop()
            self._sequencer = None
        _drain_queue(ctx.sentence_queue)
        _drain_queue(ctx.audio_queue)

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        if self._sequencer:
            self._sequencer.submit(text)
        return None  # mode handles sending itself

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "ignore"


class _InterpreterSequencer:
    """Concurrent translator with ordered output for interpreter mode.

    Bypasses OllamaClient — calls Ollama /api/chat directly via httpx.
    Translates utterances concurrently but emits to sentence_queue in
    utterance order (FIFO by submission time, not completion time).

    Moved from conversation_plugin.py (unchanged logic).
    """

    def __init__(self, config: Any, sentence_queue: asyncio.Queue, events: Any) -> None:
        self._config = config
        self._sentence_queue = sentence_queue
        self._events = events
        self._http: httpx.AsyncClient | None = None
        self._slots: list[asyncio.Task] = []
        self._seq = 0
        self._emitter_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for interpreter mode — pip install httpx"
            )
        self._http = httpx.AsyncClient(
            base_url=self._config.ollama_base_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )
        self._running = True
        self._emitter_task = asyncio.create_task(self._emitter_loop())

    async def stop(self) -> None:
        self._running = False
        for task in self._slots:
            if not task.done():
                task.cancel()
        self._slots.clear()
        if self._emitter_task:
            self._emitter_task.cancel()
        if self._http:
            await self._http.aclose()
            self._http = None

    def submit(self, text: str) -> None:
        """Submit an utterance for translation. Returns immediately."""
        self._seq += 1
        seq = self._seq
        task = asyncio.create_task(self._translate(seq, text))
        self._slots.append(task)

    async def _translate(self, seq: int, text: str) -> tuple[int, str]:
        """Call Ollama /api/chat (non-streaming) and return (seq, translated)."""
        from ..llm import INTERPRETER_SYSTEM_PROMPT

        system_prompt = (
            self._config.interpreter_prompt
            or INTERPRETER_SYSTEM_PROMPT.format(
                source_lang=self._config.interpreter_source_lang,
                target_lang=self._config.interpreter_target_lang,
            )
        )
        payload = {
            "model": self._config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Translate: "{text}"'},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        }
        try:
            resp = await self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
            translated = resp.json()["message"]["content"].strip()
            self._events.emit(
                "llm_end",
                {"full_text": translated, "run_id": f"interp-{seq}"},
            )
            return (seq, translated)
        except Exception as e:
            logger.error(f"Interpreter translation failed (seq={seq}): {e}")
            return (seq, "")

    async def _emitter_loop(self) -> None:
        """Consume completed translations in order and feed to sentence_queue."""
        from ..plugins.conversation_plugin import SentenceItem

        while self._running:
            if not self._slots:
                await asyncio.sleep(0.05)
                continue

            first = self._slots[0]
            try:
                seq, translated = await asyncio.shield(first)
            except (asyncio.CancelledError, Exception):
                self._slots.pop(0)
                continue

            self._slots.pop(0)

            if translated:
                logger.info(f'Interpreter emit seq={seq}: "{translated[:40]}"')
                run_id = f"interp-{seq}"
                self._events.emit(
                    "llm_delta", {"text": translated, "run_id": run_id}
                )
                self._events.emit(
                    "llm_end",
                    {"full_text": translated, "run_id": run_id, "emotion": ""},
                )
                await self._sentence_queue.put(
                    SentenceItem(text=translated, is_last=True)
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/test_modes.py -v`
Expected: All 32 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reachy_claw/modes/interpreter.py tests/test_modes.py
git commit -m "feat: add InterpreterMode with _InterpreterSequencer"
```

---

### Task 6: Integrate ModeManager into ConversationPlugin

**Files:**
- Modify: `src/reachy_claw/plugins/conversation_plugin.py`
- Modify: `tests/test_conversation_plugin.py`

This is the largest task — replace boolean flags and the switch_mode body with ModeManager delegation. The key changes are listed by location in conversation_plugin.py.

- [ ] **Step 1: Remove _InterpreterSequencer class (lines 66-168)**

Delete the entire `_InterpreterSequencer` class from conversation_plugin.py. It now lives in `src/reachy_claw/modes/interpreter.py`. Keep the `SentenceItem`, `ConvState`, and `_RESET_BUFFER` definitions.

Also update the imports at the top — remove `httpx` import (no longer needed here):

```python
# Remove these lines:
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]
```

- [ ] **Step 2: Replace boolean flags in __init__ with ModeManager**

In `__init__` (around line 218-226), replace:

```python
        # Mode state
        self._monologue_mode = False
        self._interpreter_mode = False
        self._narration_mode = False
        self._interp_sequencer: _InterpreterSequencer | None = None
        self._last_speech_time: float = 0.0
        self._pending_speech: str | None = None
        self._bg_speech_frames: list[np.ndarray] = []
        self._bg_silence_count: int = 0
```

With:

```python
        # Mode state
        self._mode_manager: ModeManager | None = None  # initialized in start()
        self._narration_mode = False  # separate from modes — suppresses all output
```

Add imports at top of file:

```python
from ..mode import ModeContext, ModeManager
from ..modes import ConversationMode, MonologueMode, InterpreterMode
```

- [ ] **Step 3: Initialize ModeManager in start() and set initial mode**

In `start()`, after phase 2 (backends connected) and before phase 3 (start listening), add ModeManager initialization:

```python
        # ── Phase 2.5: initialize mode manager ──────────────────────────
        ctx = ModeContext(
            app=self.app,
            sentence_queue=self._sentence_queue,
            audio_queue=self._audio_queue,
            events=self.app.events,
            spawn_task=self._spawn_task,
            get_vision_context=self._get_vision_context,
            capture_frame=self._capture_frame_b64,
        )
        self._mode_manager = ModeManager(ctx)
        self._mode_manager.register(ConversationMode())
        self._mode_manager.register(MonologueMode())
        self._mode_manager.register(InterpreterMode())
        await self._mode_manager.switch(config.conversation_mode)
```

Remove the monologue timer startup from the phase 3 tasks list (it's now handled by MonologueMode.enter):

```python
        # Remove this block:
        if self._monologue_mode:
            tasks.append(self._monologue_timer())
```

- [ ] **Step 4: Replace switch_mode body**

Replace the entire `switch_mode` method (lines 774-864) with:

```python
    def switch_mode(self, mode: str) -> None:
        """Hot-switch between conversation, monologue, and interpreter modes."""
        if self._mode_manager:
            asyncio.ensure_future(self._do_switch_mode(mode))

    async def _do_switch_mode(self, mode: str) -> None:
        """Async mode switch — updates ModeManager and OllamaClient config."""
        from ..llm import DEFAULT_SYSTEM_PROMPT, MONOLOGUE_SYSTEM_PROMPT

        await self._mode_manager.switch(mode)
        current = self._mode_manager.current

        # Update OllamaClient config if applicable
        if isinstance(self._client, OllamaClient):
            overrides = current.get_ollama_config(self.app.config)
            # Resolve system prompt
            if current.system_prompt:
                self._client._config.system_prompt = current.system_prompt
            elif current.name == "interpreter":
                from ..llm import INTERPRETER_SYSTEM_PROMPT
                self._client._config.system_prompt = (
                    self.app.config.interpreter_prompt
                    or INTERPRETER_SYSTEM_PROMPT.format(
                        source_lang=self.app.config.interpreter_source_lang,
                        target_lang=self.app.config.interpreter_target_lang,
                    )
                )
            elif current.name == "monologue":
                self._client._config.system_prompt = (
                    self.app.config.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT
                )
            else:
                self._client._config.system_prompt = (
                    self.app.config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
                )

            self._client._config.skip_emotion_extraction = overrides.get(
                "skip_emotion_extraction", False
            )
            self._client._config.monologue_mode = current.name == "monologue"
            if "temperature" in overrides:
                self._client._config.temperature = overrides["temperature"]
            else:
                self._client._config.temperature = self.app.config.ollama_temperature
            if "max_history" in overrides:
                self._client._config.max_history = overrides["max_history"]
            else:
                self._client._config.max_history = self.app.config.ollama_max_history
            if "enable_vlm" in overrides:
                self._client._config.enable_vlm = overrides["enable_vlm"]
            else:
                self._client._config.enable_vlm = self.app.config.enable_vlm
            self._client._history.clear()

        logger.info(f"Conversation mode switched to: {mode}")
```

- [ ] **Step 5: Replace _audio_loop mode branches**

In `_audio_loop`, replace the SPEAKING state mode check (around line 1346-1350):

```python
            # Before:
            if self._state == ConvState.SPEAKING and not self._interpreter_mode:
                if self._monologue_mode:
                    await self._bg_listen(chunk, streaming_stt)
                    continue

            # After:
            if self._state == ConvState.SPEAKING and self._mode_manager:
                action = self._mode_manager.current.on_speaking_audio(chunk, self._mode_manager._ctx)
                if action == "bg_listen":
                    mode = self._mode_manager.current
                    if hasattr(mode, 'bg_listen'):
                        await mode.bg_listen(chunk, streaming_stt, self)
                    continue
                elif action == "ignore":
                    continue
                # action == "barge_in" → fall through to existing barge-in logic
```

Replace the TRANSCRIBING/THINKING mode check (around line 1417-1426):

```python
            # Before:
            if self._state in (ConvState.TRANSCRIBING, ConvState.THINKING) and not self._interpreter_mode:
                if self._monologue_mode:
                    await self._bg_listen(chunk, streaming_stt)
                    continue
                else:
                    ...

            # After:
            if self._state in (ConvState.TRANSCRIBING, ConvState.THINKING):
                if self._mode_manager:
                    action = self._mode_manager.current.on_speaking_audio(chunk, self._mode_manager._ctx)
                    if action == "bg_listen" and hasattr(self._mode_manager.current, 'bg_listen'):
                        await self._mode_manager.current.bg_listen(chunk, streaming_stt, self)
                        continue
                if self._mode_manager and self._mode_manager.current.name != "interpreter":
                    if streaming_stt:
                        await asyncio.to_thread(self._stt.feed_chunk, chunk)
                    continue
```

Replace the interpreter continuous ASR block (around line 1429-1460):

```python
            # Before:
            if self._interpreter_mode and streaming_stt:
                ...
                if self._interp_sequencer:
                    self._interp_sequencer.submit(text)
                continue

            # After:
            if self._mode_manager and self._mode_manager.current.name == "interpreter" and streaming_stt:
                partial = await asyncio.to_thread(self._stt.feed_chunk, chunk)
                if partial and partial.text:
                    logger.debug(f"Partial: \"{partial.text}\" (final={partial.is_final}, stable={partial.is_stable})")
                    self.app.events.emit("asr_partial", {"text": partial.text})

                    should_translate = False
                    if partial.is_final and partial.text.strip():
                        should_translate = True
                    elif partial.is_stable and partial.text.strip():
                        if partial.text != interp_last_stable:
                            interp_last_stable = partial.text
                            interp_stable_since = time.monotonic()
                        elif time.monotonic() - interp_stable_since > 3.0:
                            should_translate = True
                            logger.info("Interpreter stable timeout fallback")
                    else:
                        interp_last_stable = ""

                    if should_translate:
                        text = partial.text
                        self._stt._partial_text = ""
                        self._stt._final_text = ""
                        interp_last_stable = ""
                        self.app.events.emit("asr_final", {"text": text})
                        logger.info(f"Interpreter translate: \"{text}\"")
                        self._mode_manager.current.preprocess_utterance(text, self._mode_manager._ctx)
                continue
```

- [ ] **Step 6: Replace _process_and_send_inner mode branches**

In `_process_and_send_inner` (around line 1727-1733), replace:

```python
        # Before:
        if self._monologue_mode:
            text = self._compose_monologue_prompt(text)
        else:
            ctx = self._get_vision_context()
            if ctx:
                text = f"[Faces: {ctx}]\n{text}"

        # After:
        if self._mode_manager:
            result = self._mode_manager.current.preprocess_utterance(text, self._mode_manager._ctx)
            if result is None:
                return  # mode handles sending itself
            text = result
```

- [ ] **Step 7: Replace remaining scattered mode checks**

Replace emotion/thinking guards. Search for `not self._monologue_mode` and replace with `self._mode_manager.current.play_emotions`:

At line ~646:
```python
        # Before:
        if self.app.config.play_emotions and not self._monologue_mode:
        # After:
        if self.app.config.play_emotions and (not self._mode_manager or self._mode_manager.current.play_emotions):
```

At line ~1739:
```python
        # Before:
        if self.app.config.play_emotions and not self._monologue_mode:
            self.app.emotions.queue_emotion("thinking")
        # After:
        if self.app.config.play_emotions and (not self._mode_manager or self._mode_manager.current.play_emotions):
            self.app.emotions.queue_emotion("thinking")
```

At line ~1750:
```python
        # Before:
        if self.app.config.play_emotions and not self._monologue_mode:
        # After:
        if self.app.config.play_emotions and (not self._mode_manager or self._mode_manager.current.play_emotions):
```

Replace monologue last_speech_time update (line ~1737):
```python
        # Before:
        self._last_speech_time = time.monotonic()
        # After:
        if self._mode_manager and hasattr(self._mode_manager.current, 'last_speech_time'):
            self._mode_manager.current.last_speech_time = time.monotonic()
```

Replace the `_process_and_send_raw` monologue check (line ~1783):
```python
        # Before:
        if not self._monologue_mode or self._narration_mode:
        # After:
        if (self._mode_manager and self._mode_manager.current.name != "monologue") or self._narration_mode:
```

- [ ] **Step 8: Remove old methods that moved to Mode classes**

Delete from conversation_plugin.py:
- `_compose_monologue_prompt` (lines 684-722)
- `_bg_listen` (lines 1578-1636)
- `_monologue_timer` (lines 1771-1810)

Keep `_process_and_send_raw` — it's still called by the monologue trigger event handler.

- [ ] **Step 9: Wire up monologue_trigger event**

Add a subscription in `start()` to handle the `monologue_trigger` event emitted by MonologueMode's timer:

```python
        # After ModeManager initialization:
        self.app.events.subscribe("monologue_trigger", self._on_monologue_trigger)
```

Add the handler method:

```python
    def _on_monologue_trigger(self, data: dict) -> None:
        """Handle monologue auto-trigger from MonologueMode timer."""
        from .conversation_plugin import ConvState

        if self._state != ConvState.IDLE or not self._client or self._narration_mode:
            return
        prompt = data.get("prompt", "")
        if prompt:
            self._spawn_task(
                self._process_and_send_raw(prompt),
                name="conversation.monologue_auto",
            )
```

- [ ] **Step 10: Update switch_backend to use ModeManager**

In `switch_backend` (around line 928-955), replace the mode if/elif/else config block:

```python
        # Before:
        if self._interpreter_mode:
            system_prompt = ...
            history, temperature = 0, 0.3
        elif self._monologue_mode:
            ...
        else:
            ...

        # After:
        if self._mode_manager:
            current = self._mode_manager.current
            overrides = current.get_ollama_config(config)
            # Resolve system prompt (same logic as _do_switch_mode)
            if current.name == "interpreter":
                from ..llm import INTERPRETER_SYSTEM_PROMPT
                system_prompt = config.interpreter_prompt or INTERPRETER_SYSTEM_PROMPT.format(
                    source_lang=config.interpreter_source_lang,
                    target_lang=config.interpreter_target_lang,
                )
            elif current.name == "monologue":
                system_prompt = config.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT
            else:
                system_prompt = config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
            history = overrides.get("max_history", config.ollama_max_history)
            temperature = overrides.get("temperature", config.ollama_temperature)
        else:
            system_prompt = config.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT
            history = config.ollama_max_history
            temperature = config.ollama_temperature
```

- [ ] **Step 11: Update _drain_queue import**

The `_drain_queue` function at the bottom of conversation_plugin.py is still used by the plugin (e.g., in `_fire_interrupt`). Keep it. The copy in `modes/interpreter.py` is for the interpreter's own use — no cross-import needed.

- [ ] **Step 12: Run full test suite**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/ -v --tb=short`
Expected: All tests pass. If TestSwitchMode tests fail, proceed to Step 13.

- [ ] **Step 13: Update TestSwitchMode tests**

The existing tests in `tests/test_conversation_plugin.py` check `plugin._monologue_mode` and `plugin._interpreter_mode` which no longer exist. Update them:

Replace references to `plugin._monologue_mode` with `plugin._mode_manager.current.name == "monologue"`.
Replace references to `plugin._interpreter_mode` with `plugin._mode_manager.current.name == "interpreter"`.

But note: `switch_mode` now calls `asyncio.ensure_future` so tests need to be async and await the switch. Since `switch_mode` is sync → `_do_switch_mode` is async, tests that call `switch_mode` synchronously need to also run the event loop.

For tests that call `plugin.switch_mode(...)` synchronously without a running event loop, change them to call `await plugin._do_switch_mode(...)` directly:

```python
    @pytest.mark.asyncio
    async def test_conversation_mode_enables_barge_in(self, standalone_app):
        plugin = ConversationPlugin(standalone_app)
        # Initialize ModeManager manually for unit tests
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
        from reachy_claw.modes import ConversationMode, MonologueMode, InterpreterMode
        plugin._mode_manager.register(ConversationMode())
        plugin._mode_manager.register(MonologueMode())
        plugin._mode_manager.register(InterpreterMode())

        standalone_app.config.barge_in_enabled = False
        await plugin._do_switch_mode("conversation")
        assert standalone_app.config.barge_in_enabled is True
        assert plugin._mode_manager.current.name == "conversation"
```

Apply the same pattern to all TestSwitchMode tests. Create a helper fixture to reduce boilerplate:

```python
@pytest.fixture
def mode_plugin(standalone_app):
    """ConversationPlugin with ModeManager initialized for testing."""
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
    from reachy_claw.modes import ConversationMode, MonologueMode, InterpreterMode
    plugin._mode_manager.register(ConversationMode())
    plugin._mode_manager.register(MonologueMode())
    plugin._mode_manager.register(InterpreterMode())
    return plugin
```

- [ ] **Step 14: Run full test suite again**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 15: Commit**

```bash
git add src/reachy_claw/plugins/conversation_plugin.py tests/test_conversation_plugin.py
git commit -m "refactor: replace switch_mode if/elif with ModeManager delegation"
```

---

### Task 7: Final cleanup and verification

**Files:**
- All modified files

- [ ] **Step 1: Search for any remaining old flag references**

Run: `cd /Users/harvest/project/clawd-reachy-mini && grep -rn '_monologue_mode\|_interpreter_mode\|_interp_sequencer' src/reachy_claw/`

Expected: No matches in `conversation_plugin.py`. If any remain, update them to use `self._mode_manager.current.name` or `self._mode_manager.current`.

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/harvest/project/clawd-reachy-mini && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Verify dashboard mode switching still works**

Check that `dashboard_plugin.py` calls `switch_mode()` — it should still work since the method signature is unchanged:

Run: `grep -n 'switch_mode' src/reachy_claw/plugins/dashboard_plugin.py`

The dashboard calls `conversation_plugin.switch_mode(mode)` which is still the same sync entry point. No changes needed.

- [ ] **Step 4: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: cleanup remaining old mode flag references"
```
