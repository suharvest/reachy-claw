"""ConversationMode — standard interactive dialogue."""

from __future__ import annotations

from typing import Any

from ..mode import Mode, ModeContext


class ConversationMode(Mode):
    """Standard back-and-forth conversation with barge-in support."""

    name = "conversation"
    barge_in = True
    play_emotions = True

    def preprocess_utterance(self, text: str, ctx: ModeContext) -> str | None:
        vision_ctx = ctx.get_vision_context()
        if vision_ctx:
            return f"[Faces: {vision_ctx}]\n{text}"
        return text

    def on_speaking_audio(self, chunk: Any, ctx: ModeContext) -> str:
        return "barge_in"
