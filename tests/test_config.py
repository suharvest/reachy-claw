"""Tests for runtime configuration loading."""

from __future__ import annotations

from pathlib import Path

from clawd_reachy_mini.config import Config, load_config


def test_config_loads_tokens_from_environment(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TOKEN", "gateway-token")
    monkeypatch.setenv("OPENCLAW_OPENAI_TOKEN", "openai-token")

    config = Config()

    assert config.gateway_token == "gateway-token"
    assert config.openai_api_key == "openai-token"
    assert config.cache_dir.exists()


def test_load_config_reads_environment_overrides(monkeypatch):
    monkeypatch.setenv("OPENCLAW_HOST", "10.0.0.9")
    monkeypatch.setenv("OPENCLAW_PORT", "19999")
    monkeypatch.setenv("STT_BACKEND", "faster-whisper")
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WAKE_WORD", "hey reachy")

    config = load_config()

    assert config.gateway_host == "10.0.0.9"
    assert config.gateway_port == 19999
    assert config.stt_backend == "faster-whisper"
    assert config.whisper_model == "small"
    assert config.wake_word == "hey reachy"
    assert config.gateway_url == "ws://10.0.0.9:19999/desktop-robot"


def test_config_has_motion_defaults():
    config = Config()
    assert config.motion_emotion_intensity == 0.7
    assert config.motion_head_tracking_smoothing == 0.3
    assert config.motion_idle_animation_interval == 5.0


def test_config_has_vision_defaults():
    config = Config()
    assert config.vision_tracker_type == "mediapipe"
    assert config.vision_camera_index == 0
    assert config.enable_face_tracker is True
    assert config.enable_motion is True


# ── YAML config file tests ──────────────────────────────────────────


def test_load_config_from_yaml(tmp_path, monkeypatch):
    """YAML file values override defaults."""
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text(
        """\
gateway:
  host: 10.0.0.5
  port: 9999
stt:
  backend: sensevoice
  whisper_model: large
tts:
  backend: kokoro
  kokoro_speed: 1.5
behavior:
  wake_word: hey robot
  play_emotions: false
vision:
  tracker: none
  camera_index: 2
plugins:
  face_tracker: false
"""
    )

    config = load_config(cfg_file)

    assert config.gateway_host == "10.0.0.5"
    assert config.gateway_port == 9999
    assert config.stt_backend == "sensevoice"
    assert config.whisper_model == "large"
    assert config.tts_backend == "kokoro"
    assert config.kokoro_speed == 1.5
    assert config.wake_word == "hey robot"
    assert config.play_emotions is False
    assert config.vision_tracker_type == "none"
    assert config.vision_camera_index == 2
    assert config.enable_face_tracker is False


def test_env_overrides_yaml(tmp_path, monkeypatch):
    """Environment variables take priority over YAML."""
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text(
        """\
gateway:
  host: 10.0.0.5
stt:
  backend: whisper
"""
    )
    monkeypatch.setenv("OPENCLAW_HOST", "192.168.1.100")
    monkeypatch.setenv("STT_BACKEND", "openai")

    config = load_config(cfg_file)

    assert config.gateway_host == "192.168.1.100"  # env wins
    assert config.stt_backend == "openai"  # env wins


def test_load_config_missing_yaml_uses_defaults():
    """Non-existent explicit path is ignored gracefully."""
    config = load_config("/nonexistent/clawd.yaml")

    assert config.gateway_host == "127.0.0.1"
    assert config.stt_backend == "whisper"


def test_load_config_partial_yaml(tmp_path):
    """Partial YAML only overrides specified fields."""
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text("tts:\n  backend: piper\n")

    config = load_config(cfg_file)

    assert config.tts_backend == "piper"
    assert config.stt_backend == "whisper"  # untouched default
    assert config.gateway_host == "127.0.0.1"  # untouched default


def test_load_config_auto_detect_clawd_yaml(tmp_path, monkeypatch):
    """Auto-detect clawd.yaml in current directory."""
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text("gateway:\n  port: 12345\n")
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.gateway_port == 12345


def test_load_config_env_clawd_config(tmp_path, monkeypatch):
    """CLAWD_CONFIG env var points to config file."""
    cfg_file = tmp_path / "my-config.yaml"
    cfg_file.write_text("tts:\n  backend: none\n")
    monkeypatch.setenv("CLAWD_CONFIG", str(cfg_file))

    config = load_config()

    assert config.tts_backend == "none"
