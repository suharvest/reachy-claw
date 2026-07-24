"""Minimal configuration for the Reachy Voice app.

Only the fields the core listen→think→speak→emote loop needs. Values come from
(in order of precedence): explicit YAML file → environment variables → defaults.
Deliberately tiny compared to the old reachy-claw config (~100 fields); add more
only when a feature actually needs it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # ── SLV V2V engine (streaming ASR + TTS over WebSocket) ──────────
    v2v_url: str = "ws://localhost:8621/v2v/stream"
    # Local default: Reachy owns the LLM text stream and sanitizes it before TTS.
    # Server-loop can be re-enabled for providers that support clean tool calls,
    # but the current local Jetson image can leak control text into speech.
    server_loop: bool = False
    realtime_protocol_version: int = 2
    # Local edge-llm currently cannot render OpenAI tool schemas for this model
    # image, returning tools_render_failed and dropping otherwise valid ASR
    # turns. Keep tools opt-in until the local model image supports schemas.
    tools_enabled: bool = False
    # Locked exhibition language: "zh" or "en". Drives ASR + TTS + a hard
    # "reply only in <language>" prompt instruction. Switchable at runtime via
    # the settings UI / POST /language. Deliberately NOT "auto": a curated
    # exhibition wants one consistent language/voice/persona.
    language: str = "zh"

    # ── edge-LLM (OpenAI-compatible) ─────────────────────────────────
    edge_llm_url: str = "http://localhost:11435/v1"
    edge_llm_model: str = "Qwen/Qwen3-4B-AWQ"
    edge_llm_max_tokens: int = 256
    # Session context budget (input tokens). The session trims oldest turns to
    # stay under this; each trim invalidates the prefix KV cache, so a big budget
    # keeps the session pinned at the trim boundary and cold-prefills the whole
    # context EVERY turn (ovs default 7000 → ~5s turns). Smaller = snappier.
    session_max_input_tokens: int = 3500
    # Reset the conversation after this many seconds of no user speech, so each
    # visitor starts with a fresh (small → warm → <1s) context. 0 disables.
    session_reset_idle_s: float = 90.0
    # Wake gate. Empty disables the gate; default requires "你好 mini" before
    # a visitor turn is allowed to reach the model.
    wake_word: str = "你好mini"
    wake_session_timeout_s: float = 18.0

    # ── Audio ────────────────────────────────────────────────────────
    # NB trailing colon: the Reachy camera re-enumerates its own
    # "Reachy Mini Camera: USB Audio" input node, and sounddevice's device
    # resolver matches by in-order word-subsequence — so the bare
    # "Reachy Mini Audio" matched BOTH ("...Camera: USB Audio" ends in "Audio"),
    # raising "Multiple input devices found" and crash-looping the mic pump
    # (video kept working, speaker went silent). Only the real card carries an
    # "Audio:" token (camera is "Camera:"), so the colon resolves it uniquely,
    # independent of the unstable hw:X index.
    audio_device: str = "Reachy Mini Audio:"
    sample_rate: int = 16000
    audio_volume: float = 3.5
    input_channel: int = 0
    # TTS speech rate (keepPitch) forwarded to the SLV in slv_config; MOSS
    # has no native speed knob, so the SLV time-stretches the streamed PCM.
    # 1.0 = off. Tunable via the runtime YAML or REACHY_TTS_SPEED.
    tts_speed: float = 1.2

    # ── Client-side VAD (turn boundaries) ────────────────────────────
    vad_backend: str = "silero"
    client_vad_preroll_ms: int = 700
    client_vad_silence_ms: int = 550
    client_vad_threshold: float = 0.3

    # ── LLM stream timeouts ──────────────────────────────────────────
    llm_first_token_timeout_s: float = 8.0
    llm_stream_idle_timeout_s: float = 6.0
    post_tts_echo_ignore_s: float = 0.6

    # ── Vision (remote face/emotion tracker, separate container) ─────
    vision_url: str = "tcp://127.0.0.1:8631"            # ZMQ PUB (faces/emotions)
    vision_mjpeg: str = "http://127.0.0.1:8630/stream"  # camera MJPEG (dashboard)
    vlm_base_url: str = "http://localhost:11435/v1"
    vlm_model: str = "Qwen/Qwen3-4B-AWQ"
    vlm_timeout_s: float = 20.0

    # ── Conversation profile (prompts/tools live as data files) ──────
    profile: str = "exhibition"

    # ── Settings dashboard ───────────────────────────────────────────
    settings_url: str = "http://0.0.0.0:8042"

    @property
    def profile_dir(self) -> Path:
        return Path(__file__).parent / "profiles" / self.profile

    def system_prompt(self) -> str:
        """Load the profile's instructions (system prompt) data file."""
        p = self.profile_dir / "instructions.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""


# Map config field -> environment variable (mirrors CLAWD_* convention loosely).
_ENV = {
    "v2v_url": "REACHY_V2V_URL",
    "server_loop": "REACHY_SERVER_LOOP",
    "realtime_protocol_version": "REACHY_REALTIME_PROTOCOL_VERSION",
    "tools_enabled": "REACHY_TOOLS_ENABLED",
    "edge_llm_url": "REACHY_EDGE_LLM_URL",
    "edge_llm_model": "REACHY_EDGE_LLM_MODEL",
    "audio_device": "REACHY_AUDIO_DEVICE",
    "input_channel": "REACHY_INPUT_CHANNEL",
    "tts_speed": "REACHY_TTS_SPEED",
    "session_max_input_tokens": "REACHY_SESSION_MAX_INPUT_TOKENS",
    "session_reset_idle_s": "REACHY_SESSION_RESET_IDLE_S",
    "wake_word": "REACHY_WAKE_WORD",
    "wake_session_timeout_s": "REACHY_WAKE_SESSION_TIMEOUT_S",
    "vlm_base_url": "REACHY_VLM_BASE_URL",
    "vlm_model": "REACHY_VLM_MODEL",
    "vlm_timeout_s": "REACHY_VLM_TIMEOUT_S",
    "profile": "REACHY_PROFILE",
    "language": "REACHY_LANGUAGE",
}


def _coerce(value: str, typ: Any) -> Any:
    # With `from __future__ import annotations`, dataclass field.type is a
    # string (e.g. "int"), so match both the type object and its name.
    if typ in (bool, "bool"):
        return value.lower() in ("1", "true", "yes", "on")
    if typ in (int, "int"):
        return int(value)
    if typ in (float, "float"):
        return float(value)
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build a Config from defaults, optional YAML, then environment overrides."""
    data: dict[str, Any] = {}
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            known = {f.name for f in fields(Config)}
            data = {k: v for k, v in loaded.items() if k in known}

    cfg = Config(**data)

    type_by_name = {f.name: f.type for f in fields(Config)}
    for field_name, env_name in _ENV.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            setattr(cfg, field_name, _coerce(raw, type_by_name.get(field_name, str)))
    return cfg
