"""Unit tests for the SLV ConversationPlugin stream-safe TTS tag stripper.

Proves that emotion / [Faces: ...] tags split across LLM stream tokens are
DROPPED before reaching the SLV engine ``send_text`` (so they are never
spoken aloud), while still queuing the emotion (motion / dashboard keep
working). Reuses the fake-SLV harness style from
``tests/e2e_slv_plugin_local.py`` but runs fully offline (no ollama / no WS).
"""
from __future__ import annotations

import pytest

from reachy_claw.config import Config
from reachy_claw.app import ReachyClawApp
from reachy_claw.plugins.conversation_plugin_slv import ConversationPlugin


class FakeSLV:
    """Stub SLVClient: records every send_text chunk; no real WS."""

    def __init__(self):
        self.text_chunks: list[str] = []

    async def send_text(self, text: str):
        if text:
            self.text_chunks.append(text)


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
