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
    kokoro_speaker_id: int = 3  # zf_001 (中文女声)
    kokoro_speed: float = 1.2

    # VAD
    vad_backend: str = "silero"  # "silero", "energy"
    silero_threshold: float = 0.3

    # Audio settings
    audio_device: str | None = None
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
    barge_in_confirm_frames: int = 3  # consecutive VAD-positive frames before interrupt (~200ms)

    # Behavior
    wake_word: str | None = None
    play_emotions: bool = True
    idle_animations: bool = True
    standalone_mode: bool = False

    # Motion settings
    motion_emotion_intensity: float = 0.7
    motion_head_tracking_smoothing: float = 0.3
    motion_head_tracking_poll_interval: float = 0.1
    motion_idle_animation_interval: float = 5.0

    # Vision / face tracking
    vision_tracker_type: str = "mediapipe"  # "mediapipe", "none"
    vision_camera_index: int = 0
    vision_max_yaw: float = 25.0
    vision_max_pitch: float = 15.0
    vision_smoothing_alpha: float = 0.3
    vision_deadzone: float = 0.02
    vision_face_lost_delay: float = 2.0

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
    ("audio", "sample_rate"): "sample_rate",
    ("audio", "silence_threshold"): "silence_threshold",
    ("audio", "silence_duration"): "silence_duration",
    ("audio", "max_recording_duration"): "max_recording_duration",
    ("gateway", "warmup"): "gateway_warmup",
    ("gateway", "keepalive_s"): "gateway_keepalive_s",
    ("barge_in", "enabled"): "barge_in_enabled",
    ("barge_in", "energy_threshold"): "barge_in_energy_threshold",
    ("barge_in", "confirm_frames"): "barge_in_confirm_frames",
    ("behavior", "wake_word"): "wake_word",
    ("behavior", "play_emotions"): "play_emotions",
    ("behavior", "idle_animations"): "idle_animations",
    ("behavior", "standalone_mode"): "standalone_mode",
    ("motion", "emotion_intensity"): "motion_emotion_intensity",
    ("motion", "head_tracking_smoothing"): "motion_head_tracking_smoothing",
    ("motion", "head_tracking_poll_interval"): "motion_head_tracking_poll_interval",
    ("motion", "idle_animation_interval"): "motion_idle_animation_interval",
    ("vision", "tracker"): "vision_tracker_type",
    ("vision", "camera_index"): "vision_camera_index",
    ("vision", "max_yaw"): "vision_max_yaw",
    ("vision", "max_pitch"): "vision_max_pitch",
    ("vision", "smoothing_alpha"): "vision_smoothing_alpha",
    ("vision", "deadzone"): "vision_deadzone",
    ("vision", "face_lost_delay"): "vision_face_lost_delay",
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


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration: defaults → YAML file → environment variables.

    Priority (highest wins): environment variables > YAML file > defaults.

    Args:
        config_path: Explicit path to a YAML config file. If None,
            searches REACHY_CLAW_CONFIG env var, then default locations.
    """
    config = Config()

    # Layer 1: YAML file
    found = _find_config_file(config_path)
    if found:
        logger.info(f"Loading config from {found}")
        data = _load_yaml_file(found)
        _apply_yaml(config, data)

    # Layer 2: Environment variables (override YAML)
    _apply_env(config)

    return config
