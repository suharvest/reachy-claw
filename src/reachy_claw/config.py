"""Configuration for Reachy Mini OpenClaw interface."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config file search paths (first found wins)
CONFIG_SEARCH_PATHS = [
    Path("reachy-claw.yaml"),  # current directory
    Path("reachy-claw.yml"),
    Path.home() / ".reachy-claw" / "config.yaml",
    Path.home() / ".reachy-claw" / "config.yml",
]


@dataclass
class Config:
    """Configuration for the Reachy Mini interface."""

    # Desktop-robot WebSocket endpoint
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 18790
    gateway_token: str | None = None
    gateway_path: str = "/desktop-robot"

    # Reachy Mini connection
    reachy_connection_mode: str = "auto"  # "auto", "localhost_only", "network"
    reachy_media_backend: str = "default"  # "default" or "gstreamer"
    reachy_spawn_daemon: bool = True  # auto-spawn daemon for USB-connected Lite
    reachy_serialport: str = "auto"  # serial port for Lite, or "auto"
    reachy_daemon_port: int = 38001  # daemon FastAPI port (SDK connects here)

    # Speech-to-text
    stt_backend: str = "paraformer-streaming"  # "paraformer-streaming", "whisper", "faster-whisper", "openai", "sensevoice"
    whisper_model: str = "base"  # "tiny", "base", "small", "medium", "large"
    openai_api_key: str | None = None

    # Text-to-speech
    tts_backend: str = "matcha"  # "matcha", "kokoro", "elevenlabs", "macos-say", "piper", "none"
    tts_voice: str | None = None
    tts_model: str | None = None

    # Remote ASR/TTS service (Jetson Docker)
    speech_service_url: str = "http://localhost:8000"

    # Backend-specific settings (auto-populated from registry)
    # These fields are also generated dynamically, but we keep commonly
    # used ones here for IDE autocompletion and backward compatibility.
    sensevoice_language: str = "auto"
    matcha_speaker_id: int = 0
    matcha_speed: float = 1.2
    matcha_pitch_shift: float = 0.0
    matcha_clone_seed: int = 42
    kokoro_speaker_id: int = 3  # zf_001 (中文女声)
    kokoro_speed: float = 1.2
    kokoro_pitch_shift: float = 0.0
    kokoro_clone_seed: int = 42

    # Voice cloning (qwen3 backend)
    cloned_voice_name: str | None = None  # name of cloned voice (without .bin)

    # VAD
    vad_backend: str = "silero"  # "silero", "energy"
    silero_threshold: float = 0.3

    # Audio settings
    audio_device: str | None = None
    audio_volume: float = 1.0  # playback gain multiplier (e.g. 2.0 = double volume)
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration: float = 0.7
    max_recording_duration: float = 30.0

    # Gateway session
    gateway_warmup: bool = True  # send warmup message on connect to pre-init OpenClaw session
    gateway_keepalive_s: int = 60  # ping interval to prevent session timeout (0=disabled)

    # Barge-in
    barge_in_enabled: bool = True
    barge_in_energy_threshold: float = 0.02
    barge_in_confirm_frames: int = 2  # consecutive VAD-positive frames before interrupt (~128ms)
    barge_in_silero_threshold: float = 0.5  # stricter VAD threshold during playback
    barge_in_cooldown_ms: int = 300  # ignore barge-in for N ms after TTS starts

    # LLM backend (local Ollama, replaces gateway when set)
    llm_backend: str = "gateway"  # "gateway" (OpenClaw) or "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:0.8b"
    ollama_system_prompt: str = ""  # empty = use default
    ollama_monologue_prompt: str = ""  # empty = use default MONOLOGUE_SYSTEM_PROMPT
    ollama_temperature: float = 0.7
    ollama_max_history: int = 3  # conversation turns to keep (0 = stateless)

    # VLM (Vision Language Model)
    enable_vlm: bool = False
    vlm_model: str = "qwen3.5:2b"  # vision model (empty = use ollama_model)
    vlm_prompt: str = "Describe what you see in this image briefly."

    # Behavior
    wake_word: str | None = None
    play_emotions: bool = True
    idle_animations: bool = True
    standalone_mode: bool = False

    # Motion settings
    motion_emotion_intensity: float = 1.0
    motion_head_tracking_smoothing: float = 0.35
    motion_head_tracking_poll_interval: float = 0.05
    motion_idle_animation_interval: float = 5.0

    # Vision / face tracking
    vision_tracker_type: str = "mediapipe"  # "mediapipe", "remote", "none"
    vision_camera_source: str = "auto"  # "auto" (SDK if available, else OpenCV), "sdk", "opencv"
    vision_camera_index: int = 0
    vision_max_yaw: float = 55.0
    vision_body_yaw_gain: float = 3.0  # degrees per frame per unit offset (closed-loop)
    vision_max_pitch: float = 25.0
    vision_pitch_offset: float = 8.0  # degrees, positive = look up (compensate camera angle)
    vision_max_roll: float = 15.0
    vision_smoothing_alpha: float = 0.35
    vision_deadzone: float = 0.01
    vision_face_lost_delay: float = 2.0

    # Vision TRT service (used when vision_tracker_type == "remote")
    vision_service_url: str = "tcp://127.0.0.1:8631"
    vision_emotion_threshold: float = 0.35
    vision_emotion_cooldown: float = 2.0
    vision_emotion_sustain: float = 5.0  # resend same emotion after N seconds
    vision_identity_threshold: float = 0.4

    # Conversation mode
    conversation_mode: str = "conversation"  # "conversation" | "monologue" | "interpreter"
    monologue_interval: float = 5.0  # seconds between auto-triggered monologues
    interpreter_source_lang: str = "Chinese"  # source language for interpreter mode
    interpreter_target_lang: str = "English"  # target language for interpreter mode
    interpreter_prompt: str = ""  # empty = use default INTERPRETER_SYSTEM_PROMPT

    # Dashboard (exhibition UI)
    dashboard_enabled: bool = False
    dashboard_port: int = 8640

    # Plugin enable flags
    enable_face_tracker: bool = True  # auto-skips if deps missing
    enable_motion: bool = True

    # Paths
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".reachy-claw" / "cache"
    )

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.gateway_token:
            self.gateway_token = os.environ.get("OPENCLAW_TOKEN")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get(
                "OPENCLAW_OPENAI_TOKEN"
            ) or os.environ.get("OPENAI_API_KEY")

        env_url = os.environ.get("SPEECH_SERVICE_URL")
        if env_url:
            self.speech_service_url = env_url

    @property
    def desktop_robot_url(self) -> str:
        """WebSocket URL for the desktop-robot channel."""
        return f"ws://{self.gateway_host}:{self.gateway_port}{self.gateway_path}"

    @property
    def gateway_url(self) -> str:
        return self.desktop_robot_url


# ── YAML config file mapping ──────────────────────────────────────────
# Nested YAML keys → flat Config field names

_YAML_FIELD_MAP: dict[tuple[str, str], str] = {
    ("gateway", "host"): "gateway_host",
    ("gateway", "port"): "gateway_port",
    ("gateway", "token"): "gateway_token",
    ("gateway", "path"): "gateway_path",
    ("reachy", "connection_mode"): "reachy_connection_mode",
    ("reachy", "media_backend"): "reachy_media_backend",
    ("reachy", "serialport"): "reachy_serialport",
    ("reachy", "spawn_daemon"): "reachy_spawn_daemon",
    ("reachy", "daemon_port"): "reachy_daemon_port",
    ("stt", "backend"): "stt_backend",
    ("stt", "whisper_model"): "whisper_model",
    ("stt", "openai_api_key"): "openai_api_key",
    ("stt", "speech_service_url"): "speech_service_url",
    ("stt", "sensevoice_language"): "sensevoice_language",
    ("tts", "backend"): "tts_backend",
    ("tts", "voice"): "tts_voice",
    ("tts", "model"): "tts_model",
    ("tts", "speech_service_url"): "speech_service_url",
    ("vad", "backend"): "vad_backend",
    ("vad", "threshold"): "silero_threshold",
    ("audio", "device"): "audio_device",
    ("audio", "volume"): "audio_volume",
    ("audio", "sample_rate"): "sample_rate",
    ("audio", "silence_threshold"): "silence_threshold",
    ("audio", "silence_duration"): "silence_duration",
    ("audio", "max_recording_duration"): "max_recording_duration",
    ("gateway", "warmup"): "gateway_warmup",
    ("gateway", "keepalive_s"): "gateway_keepalive_s",
    ("barge_in", "enabled"): "barge_in_enabled",
    ("barge_in", "energy_threshold"): "barge_in_energy_threshold",
    ("barge_in", "confirm_frames"): "barge_in_confirm_frames",
    ("barge_in", "silero_threshold"): "barge_in_silero_threshold",
    ("barge_in", "cooldown_ms"): "barge_in_cooldown_ms",
    ("behavior", "wake_word"): "wake_word",
    ("behavior", "play_emotions"): "play_emotions",
    ("behavior", "idle_animations"): "idle_animations",
    ("behavior", "standalone_mode"): "standalone_mode",
    ("llm", "backend"): "llm_backend",
    ("llm", "model"): "ollama_model",
    ("llm", "base_url"): "ollama_base_url",
    ("llm", "system_prompt"): "ollama_system_prompt",
    ("llm", "monologue_prompt"): "ollama_monologue_prompt",
    ("llm", "temperature"): "ollama_temperature",
    ("llm", "max_history"): "ollama_max_history",
    ("vlm", "enabled"): "enable_vlm",
    ("vlm", "model"): "vlm_model",
    ("vlm", "prompt"): "vlm_prompt",
    ("motion", "emotion_intensity"): "motion_emotion_intensity",
    ("motion", "head_tracking_smoothing"): "motion_head_tracking_smoothing",
    ("motion", "head_tracking_poll_interval"): "motion_head_tracking_poll_interval",
    ("motion", "idle_animation_interval"): "motion_idle_animation_interval",
    ("vision", "tracker"): "vision_tracker_type",
    ("vision", "camera_source"): "vision_camera_source",
    ("vision", "camera_index"): "vision_camera_index",
    ("vision", "max_yaw"): "vision_max_yaw",
    ("vision", "max_pitch"): "vision_max_pitch",
    ("vision", "pitch_offset"): "vision_pitch_offset",
    ("vision", "max_roll"): "vision_max_roll",
    ("vision", "smoothing_alpha"): "vision_smoothing_alpha",
    ("vision", "deadzone"): "vision_deadzone",
    ("vision", "face_lost_delay"): "vision_face_lost_delay",
    ("vision", "service_url"): "vision_service_url",
    ("vision", "emotion_threshold"): "vision_emotion_threshold",
    ("vision", "emotion_cooldown"): "vision_emotion_cooldown",
    ("vision", "identity_threshold"): "vision_identity_threshold",
    ("conversation", "mode"): "conversation_mode",
    ("conversation", "monologue_interval"): "monologue_interval",
("dashboard", "enabled"): "dashboard_enabled",
    ("dashboard", "port"): "dashboard_port",
    ("plugins", "face_tracker"): "enable_face_tracker",
    ("plugins", "motion"): "enable_motion",
}

# Environment variable → Config field name
_ENV_FIELD_MAP: dict[str, str] = {
    "OPENCLAW_HOST": "gateway_host",
    "OPENCLAW_PORT": "gateway_port",
    "OPENCLAW_PATH": "gateway_path",
    "OPENCLAW_TOKEN": "gateway_token",
    "STT_BACKEND": "stt_backend",
    "WHISPER_MODEL": "whisper_model",
    "TTS_BACKEND": "tts_backend",
    "WAKE_WORD": "wake_word",
    "SPEECH_SERVICE_URL": "speech_service_url",
    "SENSEVOICE_LANGUAGE": "sensevoice_language",
    "VAD_BACKEND": "vad_backend",
}


def _find_config_file(explicit_path: str | Path | None = None) -> Path | None:
    """Find the config file. Explicit path > env var > search paths."""
    if explicit_path:
        p = Path(explicit_path)
        return p if p.is_file() else None

    env_path = os.environ.get("REACHY_CLAW_CONFIG")
    if env_path:
        p = Path(env_path)
        return p if p.is_file() else None

    for candidate in CONFIG_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_yaml(config: Config, data: dict[str, Any]) -> None:
    """Apply YAML data onto a Config instance."""
    from reachy_claw.backend_registry import get_yaml_mappings

    all_mappings = {**_YAML_FIELD_MAP, **get_yaml_mappings()}

    for (section, key), field_name in all_mappings.items():
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        if key in section_data:
            value = section_data[key]
            current = getattr(config, field_name, None)
            if current is not None and not isinstance(current, type(None)):
                try:
                    value = type(current)(value)
                except (ValueError, TypeError):
                    pass
            setattr(config, field_name, value)

    # Support nested backend config blocks:
    #   tts:
    #     kokoro:
    #       speaker_id: 3
    #       speed: 1.2
    # Maps to config fields: kokoro_speaker_id, kokoro_speed
    from reachy_claw.backend_registry import (
        get_tts_info, get_stt_info, get_vad_info,
        get_tts_names, get_stt_names, get_vad_names,
    )
    for section, get_names, get_info in [
        ("tts", get_tts_names, get_tts_info),
        ("stt", get_stt_names, get_stt_info),
        ("vad", get_vad_names, get_vad_info),
    ]:
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        for backend_name in get_names():
            block = section_data.get(backend_name)
            if not isinstance(block, dict):
                continue
            info = get_info(backend_name)
            if info is None:
                continue
            for field_name in info.settings_fields:
                if field_name not in block:
                    continue
                config_key = f"{backend_name}_{field_name}"
                value = block[field_name]
                current = getattr(config, config_key, None)
                if current is not None and not isinstance(current, type(None)):
                    try:
                        value = type(current)(value)
                    except (ValueError, TypeError):
                        pass
                setattr(config, config_key, value)


def _apply_env(config: Config) -> None:
    """Apply environment variables onto a Config instance."""
    from reachy_claw.backend_registry import get_env_mappings

    all_mappings = {**_ENV_FIELD_MAP, **get_env_mappings()}

    for env_var, field_name in all_mappings.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        current = getattr(config, field_name, None)
        if isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes")
        setattr(config, field_name, value)


def _get_overrides_path(config_dir: Path | None) -> Path:
    """Return path to runtime-overrides.yaml.

    Prefers DATA_DIR env (Docker persistent volume), then config dir,
    then ~/.reachy-claw/ as fallback.
    """
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        p = Path(os.path.expanduser(data_dir))
        return p / "runtime-overrides.yaml"
    if config_dir:
        return config_dir / "runtime-overrides.yaml"
    return Path.home() / ".reachy-claw" / "runtime-overrides.yaml"


# Reverse lookup: Config field name → (section, key)
_FIELD_TO_YAML: dict[str, tuple[str, str]] = {v: k for k, v in _YAML_FIELD_MAP.items()}


def save_runtime_overrides(config: Config, fields: list[str]) -> None:
    """Save specific config fields to runtime-overrides.yaml.

    Merges into existing overrides file so different handlers
    don't clobber each other.
    """
    overrides_path = _get_overrides_path(getattr(config, "_config_dir", None))
    overrides_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing overrides
    existing: dict[str, Any] = {}
    if overrides_path.is_file():
        try:
            existing = _load_yaml_file(overrides_path)
        except Exception:
            pass

    # Merge requested fields
    for field_name in fields:
        value = getattr(config, field_name, None)
        if value is None:
            continue
        # Convert Path to str for YAML
        if isinstance(value, Path):
            value = str(value)
        yaml_key = _FIELD_TO_YAML.get(field_name)
        if yaml_key:
            section, key = yaml_key
            existing.setdefault(section, {})[key] = value
        else:
            # Store as flat key in _extra section
            existing.setdefault("_extra", {})[field_name] = value

    with open(overrides_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
    logger.info("Saved runtime overrides to %s (fields: %s)", overrides_path, fields)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration: defaults → YAML file → overrides → environment variables.

    Priority (highest wins): environment variables > overrides > YAML file > defaults.

    Args:
        config_path: Explicit path to a YAML config file. If None,
            searches REACHY_CLAW_CONFIG env var, then default locations.
    """
    config = Config()

    # Layer 1: YAML file
    found = _find_config_file(config_path)
    config_dir: Path | None = None
    if found:
        logger.info(f"Loading config from {found}")
        data = _load_yaml_file(found)
        _apply_yaml(config, data)
        config_dir = found.parent

    # Layer 2: Runtime overrides (saved by dashboard)
    overrides_path = _get_overrides_path(config_dir)
    if overrides_path.is_file():
        logger.info(f"Loading runtime overrides from {overrides_path}")
        overrides = _load_yaml_file(overrides_path)
        _apply_yaml(config, overrides)
        # Also apply _extra flat keys (dynamic attributes like dashboard_volume)
        extra = overrides.get("_extra")
        if isinstance(extra, dict):
            for field_name, value in extra.items():
                setattr(config, field_name, value)

    # Store config dir for later save_runtime_overrides calls
    config._config_dir = config_dir  # type: ignore[attr-defined]

    # Layer 3: Environment variables (override everything)
    _apply_env(config)

    return config
