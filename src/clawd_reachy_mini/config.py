"""Configuration for Reachy Mini OpenClaw interface."""

import os
from dataclasses import dataclass, field
from pathlib import Path


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

    # Speech-to-text
    stt_backend: str = "whisper"  # "whisper", "faster-whisper", "openai"
    whisper_model: str = "base"  # "tiny", "base", "small", "medium", "large"
    openai_api_key: str | None = None

    # Text-to-speech
    tts_backend: str = "elevenlabs"  # "elevenlabs", "macos-say", "piper", "none"
    tts_voice: str | None = None
    tts_model: str | None = None

    # Audio settings
    audio_device: str | None = None
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
    max_recording_duration: float = 30.0

    # Barge-in
    barge_in_enabled: bool = True
    barge_in_energy_threshold: float = 0.02

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
        default_factory=lambda: Path.home() / ".clawd-reachy-mini" / "cache"
    )

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.gateway_token:
            self.gateway_token = os.environ.get("OPENCLAW_TOKEN")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get(
                "OPENCLAW_OPENAI_TOKEN"
            ) or os.environ.get("OPENAI_API_KEY")

    @property
    def desktop_robot_url(self) -> str:
        """WebSocket URL for the desktop-robot channel."""
        return f"ws://{self.gateway_host}:{self.gateway_port}{self.gateway_path}"

    @property
    def gateway_url(self) -> str:
        return self.desktop_robot_url


def load_config() -> Config:
    """Load configuration from environment and defaults."""
    return Config(
        gateway_host=os.environ.get("OPENCLAW_HOST", "127.0.0.1"),
        gateway_port=int(os.environ.get("OPENCLAW_PORT", "18790")),
        gateway_path=os.environ.get("OPENCLAW_PATH", "/desktop-robot"),
        stt_backend=os.environ.get("STT_BACKEND", "whisper"),
        whisper_model=os.environ.get("WHISPER_MODEL", "base"),
        tts_backend=os.environ.get("TTS_BACKEND", "elevenlabs"),
        wake_word=os.environ.get("WAKE_WORD"),
    )
