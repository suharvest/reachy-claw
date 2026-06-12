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
    spawn_task: Callable
    get_vision_context: Callable
    capture_frame: Callable


class Mode(ABC):
    """Base class for a conversation mode.

    Subclasses declare config overrides as class attributes (None = use
    app.config default) and implement enter/exit for sub-task lifecycle.
    """

    name: str = "unnamed"

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
        with defaults.
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
        Return None to indicate the mode handles sending itself.
        """
        return text

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        """Behavior when audio arrives during SPEAKING state.

        Returns: "barge_in" | "bg_listen" | "ignore".
        """
        return "barge_in" if self.barge_in else "ignore"


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

        if self._current:
            await self._current.exit(self._ctx)

        if new_mode.barge_in is not None:
            self._ctx.app.config.barge_in_enabled = new_mode.barge_in

        self._current = new_mode
        await new_mode.enter(self._ctx)

        self._ctx.app.config.conversation_mode = name
        self._ctx.events.emit("mode_change", {"mode": name, "prev": prev_name})
        logger.info("Mode switched: %s → %s", prev_name, name)
