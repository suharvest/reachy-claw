"""Unit tests for the conversation layer — each locks in a lesson learned on
the real robot during bring-up."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reachy_voice.config import Config, load_config
from reachy_voice.conversation import (
    SUPPORTED_LANGUAGES,
    ReachyCompanionApp,
    _TtsTagFilter,
    _strip_wake_word,
    build_ovs_config,
    build_system_prompt,
)


# ── TTS tag filter: tags must never be spoken, emotions must fire ──────


def _run_filter(tokens: list[str], *, emit_emotions: bool = True) -> tuple[str, list[str]]:
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


def test_tag_single_token_stripped():
    spoken, emotions = _run_filter(["欢迎光临。", "[happy]"])
    assert spoken == "欢迎光临。"
    assert emotions == ["happy"]


def test_tag_split_across_tokens_stripped():
    # LLM streams tags in fragments; the filter must hold back and strip.
    spoken, emotions = _run_filter(["你好。", " [", "hap", "py", "]"])
    assert "[" not in spoken and "happy" not in spoken
    assert emotions == ["happy"]


def test_tag_mid_stream():
    spoken, emotions = _run_filter(["对", "[curious]", "好的"])
    assert spoken == "对好的"
    assert emotions == ["curious"]


def test_no_tag_passthrough():
    spoken, emotions = _run_filter(["纯文本", "没有标签"])
    assert spoken == "纯文本没有标签"
    assert emotions == []


def test_bare_emotion_control_line_stripped():
    spoken, emotions = _run_filter(["Emotion: happy\n", "你好。"])
    assert spoken == "你好。"
    assert emotions == []


def test_bare_emotion_prefix_stripped_across_tokens():
    spoken, emotions = _run_filter(["Emotion", " happy，", "你好。"])
    assert spoken == "你好。"
    assert emotions == []


def test_emotion_tags_do_not_trigger_motion_by_default():
    spoken, emotions = _run_filter(["好的[happy]"], emit_emotions=False)
    assert spoken == "好的"
    assert emotions == []


def test_emotion_callback_error_never_breaks_tts():
    spoken: list[str] = []

    async def send(t: str) -> None:
        spoken.append(t)

    def boom(_: str) -> None:
        raise RuntimeError("motion exploded")

    f = _TtsTagFilter(send, boom)
    asyncio.run(f("好的[happy]"))
    assert "".join(spoken) == "好的"


# ── language lock (robot lesson: model drifted to English without it) ──


def test_default_language_is_zh():
    assert Config().language == "zh"


def test_zh_prompt_has_hard_lock():
    sp = build_system_prompt(Config(language="zh"))
    assert "只能用简体中文回复" in sp


def test_default_prompt_does_not_request_motion_tools():
    sp = build_system_prompt(Config(language="zh"))
    assert "call play_emotion" not in sp
    assert "不要输出任何动作" in sp


def test_tool_prompt_only_when_tools_enabled():
    sp = build_system_prompt(Config(language="zh", tools_enabled=True))
    assert "call play_emotion" in sp


def test_en_prompt_has_hard_lock():
    sp = build_system_prompt(Config(language="en"))
    assert "Reply ONLY in English" in sp


def test_supported_languages():
    assert set(SUPPORTED_LANGUAGES) == {"zh", "en"}


def test_wake_word_strips_spaced_mini():
    woke, rest = _strip_wake_word("你好 mini，看到什么？", "你好mini")
    assert woke is True
    assert rest == "看到什么"


# ── ovs config mapping (locks the on-robot-validated audio/VAD settings) ──


def test_ovs_config_uses_server_vad():
    # Robot lesson (2026-06-16): server VAD (silero) stays PRIMARY, but its
    # Paraformer endpoint fires nondeterministically on trailing silence and
    # the turn hangs in THINKING until the watchdog (卡死). Client VAD now
    # drives EOS as a fallback, and the mic is dropped while the robot speaks
    # so its own TTS echo can't open a never-ending server-VAD segment.
    ovs = build_ovs_config(Config())
    assert ovs.slv_config["vad"] == "silero"
    assert ovs.client_vad_drive_eos is True
    assert ovs.mic_drop_while_speaking is True
    assert ovs.playback_drain_enabled is True


def test_ovs_config_mic_makeup_gain():
    # Robot lesson: the Reachy USB mic is quiet; without gain ASR hallucinated.
    cfg = Config()
    ovs = build_ovs_config(cfg)
    assert ovs.mic_makeup_gain == pytest.approx(cfg.audio_volume) and ovs.mic_makeup_gain > 1


def test_ovs_config_language_drives_slv():
    ovs = build_ovs_config(Config(language="en"))
    assert ovs.slv_config["asr_language"] == "en"
    assert ovs.slv_config["tts_language"] == "en"


@pytest.mark.asyncio
async def test_server_loop_updates_visual_context_before_response_create():
    calls: list[tuple[str, object]] = []

    class _SLV:
        async def update_session(self, session):
            calls.append(("session.update", session))

        async def create_response(self):
            calls.append(("response.create", None))

    app = object.__new__(ReachyCompanionApp)
    app.config = SimpleNamespace(
        system_prompt="base",
        server_loop_enabled=lambda: True,
    )
    app.base_prompt = "Reachy prompt"
    app.vision = SimpleNamespace(faces_context=lambda: "Alice (happy)")
    app.session_reset_idle_s = 0
    app._idle_reset_task = None
    app.slv = _SLV()

    await app.on_user_utterance("你好", "zh")

    assert calls == [
        (
            "session.update",
            {"instructions": "Reachy prompt\n[Faces: Alice (happy)]"},
        ),
        ("response.create", None),
    ]


@pytest.mark.asyncio
async def test_utterance_ignored_until_wake_word():
    calls: list[tuple[str, object]] = []

    class _SLV:
        async def update_session(self, session):
            calls.append(("session.update", session))

        async def create_response(self):
            calls.append(("response.create", None))

    app = object.__new__(ReachyCompanionApp)
    app.config = SimpleNamespace(system_prompt="base", server_loop_enabled=lambda: True)
    app.base_prompt = "Reachy prompt"
    app.vision = None
    app.session_reset_idle_s = 0
    app._idle_reset_task = None
    app.wake_word = "你好mini"
    app.wake_session_timeout_s = 18
    app._awake_until = 0
    app.slv = _SLV()

    await app.on_user_utterance("今天天气怎么样", "zh")

    assert calls == []


@pytest.mark.asyncio
async def test_wake_word_opens_turn():
    calls: list[tuple[str, object]] = []

    class _SLV:
        async def update_session(self, session):
            calls.append(("session.update", session))

        async def create_response(self):
            calls.append(("response.create", None))

    app = object.__new__(ReachyCompanionApp)
    app.config = SimpleNamespace(system_prompt="base", server_loop_enabled=lambda: True)
    app.base_prompt = "Reachy prompt"
    app.vision = None
    app.session_reset_idle_s = 0
    app._idle_reset_task = None
    app.wake_word = "你好mini"
    app.wake_session_timeout_s = 18
    app._awake_until = 0
    app.slv = _SLV()

    await app.on_user_utterance("你好mini，介绍一下你自己", "zh")

    assert calls == [
        ("session.update", {"instructions": "Reachy prompt"}),
        ("response.create", None),
    ]


@pytest.mark.asyncio
async def test_visual_request_uses_snapshot_analysis(monkeypatch):
    spoken: list[str] = []

    async def fake_describe(cfg):
        assert cfg.language == "zh"
        return "我看到桌面上有一个白色物体。"

    class _SLV:
        async def send_text(self, text):
            spoken.append(text)

        async def flush_tts(self):
            spoken.append("<flush>")

    monkeypatch.setattr("reachy_voice.conversation.describe_current_view", fake_describe)
    app = object.__new__(ReachyCompanionApp)
    app.config = SimpleNamespace(system_prompt="base", server_loop_enabled=lambda: True)
    app.reachy_config = Config(language="zh")
    app.base_prompt = "Reachy prompt"
    app.vision = None
    app.session_reset_idle_s = 0
    app._idle_reset_task = None
    app.wake_word = "你好mini"
    app.wake_session_timeout_s = 18
    app._awake_until = 0
    app.slv = _SLV()

    await app.on_user_utterance("你好mini，你看到什么？", "zh")

    assert spoken == ["我看到桌面上有一个白色物体。", "<flush>"]


def test_env_override_language(monkeypatch):
    monkeypatch.setenv("REACHY_LANGUAGE", "en")
    assert load_config().language == "en"


def test_env_can_roll_back_server_loop(monkeypatch):
    monkeypatch.setenv("REACHY_SERVER_LOOP", "0")
    assert load_config().server_loop is False
