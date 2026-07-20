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
    # Canonical Realtime V2 server-loop.  The gateway owns the local/cloud
    # provider and the LLM/tool loop; Reachy remains a provider-neutral device
    # client.  Disable only as a temporary fallback to the legacy client-loop.
    server_loop: bool = True
    realtime_protocol_version: int = 2
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
    # Keeps the always-on / open-mic UX (no wake word).
    session_reset_idle_s: float = 90.0

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

    # ── Attention & gaze (Wave B) ────────────────────────────────────
    # The robot follows + engages a visitor who comes CLOSE and lingers — not
    # someone merely walking past. "Close" = face bbox covering at least this
    # fraction of the frame; "lingers" = present continuously for stable_s.
    attention_enabled: bool = True
    attention_min_area: float = 0.018   # face bbox area / frame area to count as "close"
    #   calibrated on the robot: passer-by faces ≈0.007, a visitor at interaction
    #   distance ≈0.025-0.033 — 0.018 splits them.
    attention_stable_s: float = 1.2     # must stay this long before we engage (greet)
    attention_cooldown_s: float = 15.0  # don't re-greet the same lingering visitor
    gaze_max_yaw: float = 35.0          # deg of body turn at the image edge (slew-limited for safety)
    gaze_max_pitch: float = 20.0        # deg of head pitch at the image edge
    gaze_lost_s: float = 3.5            # HOLD aim this long through detection gaps, then re-center
    gaze_deadzone: float = 0.03         # ignore tiny off-centre (normalised) to avoid jitter
    gaze_invert_x: bool = False         # flip if the head turns away from the visitor
    gaze_invert_y: bool = False

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
    "edge_llm_url": "REACHY_EDGE_LLM_URL",
    "edge_llm_model": "REACHY_EDGE_LLM_MODEL",
    "audio_device": "REACHY_AUDIO_DEVICE",
    "input_channel": "REACHY_INPUT_CHANNEL",
    "tts_speed": "REACHY_TTS_SPEED",
    "session_max_input_tokens": "REACHY_SESSION_MAX_INPUT_TOKENS",
    "session_reset_idle_s": "REACHY_SESSION_RESET_IDLE_S",
    "profile": "REACHY_PROFILE",
    "language": "REACHY_LANGUAGE",
    # attention/gaze tunables — handy to adjust on the robot without a rebuild
    "attention_enabled": "REACHY_ATTENTION_ENABLED",
    "attention_min_area": "REACHY_ATTENTION_MIN_AREA",
    "gaze_max_yaw": "REACHY_GAZE_MAX_YAW",
    "gaze_invert_x": "REACHY_GAZE_INVERT_X",
    "gaze_invert_y": "REACHY_GAZE_INVERT_Y",
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
