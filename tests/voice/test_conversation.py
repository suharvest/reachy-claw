"""Unit tests for the conversation layer — each locks in a lesson learned on
the real robot during bring-up."""

from __future__ import annotations

import asyncio

import pytest

from reachy_voice.config import Config, load_config
from reachy_voice.conversation import (
    SUPPORTED_LANGUAGES,
    _TtsTagFilter,
    build_ovs_config,
    build_system_prompt,
)


# ── TTS tag filter: tags must never be spoken, emotions must fire ──────


def _run_filter(tokens: list[str]) -> tuple[str, list[str]]:
    spoken: list[str] = []
    emotions: list[str] = []

    async def send(t: str) -> None:
        spoken.append(t)

    f = _TtsTagFilter(send, emotions.append)

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
    assert "Reply ONLY in Chinese" in sp


def test_en_prompt_has_hard_lock():
    sp = build_system_prompt(Config(language="en"))
    assert "Reply ONLY in English" in sp


def test_supported_languages():
    assert set(SUPPORTED_LANGUAGES) == {"zh", "en"}


# ── ovs config mapping (locks the on-robot-validated audio/VAD settings) ──


def test_ovs_config_uses_server_vad():
    # Robot lesson: client-driven EOS returned empty finals; server VAD works.
    ovs = build_ovs_config(Config())
    assert ovs.slv_config["vad"] == "silero"
    assert ovs.client_vad_drive_eos is False


def test_ovs_config_mic_makeup_gain():
    # Robot lesson: the Reachy USB mic is quiet; without gain ASR hallucinated.
    cfg = Config()
    ovs = build_ovs_config(cfg)
    assert ovs.mic_makeup_gain == pytest.approx(cfg.audio_volume) and ovs.mic_makeup_gain > 1


def test_ovs_config_language_drives_slv():
    ovs = build_ovs_config(Config(language="en"))
    assert ovs.slv_config["asr_language"] == "en"
    assert ovs.slv_config["tts_language"] == "en"


def test_env_override_language(monkeypatch):
    monkeypatch.setenv("REACHY_LANGUAGE", "en")
    assert load_config().language == "en"
