"""Tag-stripping tests for the client-loop migration.

Two sinks must drop the prompt-injected ``[Faces: ...]`` vision-context tag
(echoed by smaller edge LLMs) while still firing ``_on_emotion`` for ``[happy]``
style tags:

  * conversation._TtsTagFilter — what gets SPOKEN (TTS).
  * dashboard.DashboardPlugin._strip — what shows in the transcript.

Ported from legacy/reachy_claw/llm.py (_RESPONSE_STRIP_RE / _StreamingBracketStripper).
"""

from __future__ import annotations

import asyncio

from reachy_voice.conversation import _TtsTagFilter
from reachy_voice.dashboard import DashboardPlugin, DashboardHub


def _run_tts_filter(tokens: list[str], *, emit_emotions: bool = True) -> tuple[str, list[str]]:
    spoken: list[str] = []
    emotions: list[str] = []

    async def send(t: str) -> None:
        spoken.append(t)

    f = _TtsTagFilter(send, emotions.append, emit_emotions=emit_emotions)

    async def main() -> None:
        for tok in tokens:
            await f(tok)

    asyncio.run(main())
    return "".join(spoken), emotions


# ── TTS filter: [Faces:] dropped silently, [emotion] still fires ───────


def test_tts_drops_faces_tag_keeps_emotion():
    spoken, emotions = _run_tts_filter(["[Faces: Alice] Hi there. [happy]"])
    assert "Faces" not in spoken
    assert "Alice" not in spoken
    assert spoken.strip() == "Hi there."
    assert emotions == ["happy"]  # emotion tag still fired


def test_tts_faces_tag_split_across_tokens():
    # The vision tag arrives across several tokens; it must still be removed.
    spoken, emotions = _run_tts_filter(
        ["Hello ", "[Faces: ", "Alice, Bob", "] friend", " [happy]"]
    )
    assert "Faces" not in spoken
    assert "Alice" not in spoken and "Bob" not in spoken
    assert "Hello" in spoken and "friend" in spoken
    assert emotions == ["happy"]


def test_tts_faces_only_no_emotion():
    spoken, emotions = _run_tts_filter(["[Faces: Alice]"])
    assert spoken == ""
    assert emotions == []


def test_tts_plain_emotion_unaffected():
    spoken, emotions = _run_tts_filter(["欢迎光临。", "[happy]"])
    assert spoken == "欢迎光临。"
    assert emotions == ["happy"]


# ── dashboard transcript: [Faces:] and [emotion] both stripped ────────


def _run_dash_strip(tokens: list[str]) -> str:
    plugin = DashboardPlugin(app=object(), hub=DashboardHub())
    return "".join(plugin._strip(t) for t in tokens)


def test_dashboard_strips_faces_and_emotion():
    out = _run_dash_strip(["[Faces: Alice] Hi there.[happy]"])
    assert "Faces" not in out
    assert "Alice" not in out
    assert "happy" not in out
    assert "Hi there." in out


def test_dashboard_faces_split_across_tokens():
    out = _run_dash_strip(["Hi ", "[Faces: ", "Bob", "] there", "[curious]"])
    assert "Faces" not in out and "Bob" not in out and "curious" not in out
    assert "Hi" in out and "there" in out
