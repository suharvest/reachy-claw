"""Tests for runtime configuration loading."""

from __future__ import annotations

from pathlib import Path

from reachy_claw.config import Config, load_config, save_runtime_overrides


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
    assert config.motion_emotion_intensity == 1.0
    assert config.motion_head_tracking_smoothing == 0.35
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
    cfg_file = tmp_path / "reachy-claw.yaml"
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
    cfg_file = tmp_path / "reachy-claw.yaml"
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
    config = load_config("/nonexistent/reachy-claw.yaml")

    assert config.gateway_host == "127.0.0.1"
    assert config.stt_backend == "paraformer-streaming"


def test_load_config_partial_yaml(tmp_path):
    """Partial YAML only overrides specified fields."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("tts:\n  backend: piper\n")

    config = load_config(cfg_file)

    assert config.tts_backend == "piper"
    assert config.stt_backend == "paraformer-streaming"  # untouched default
    assert config.gateway_host == "127.0.0.1"  # untouched default


def test_load_config_auto_detect_reachy_claw_yaml(tmp_path, monkeypatch):
    """Auto-detect reachy-claw.yaml in current directory."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("gateway:\n  port: 12345\n")
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.gateway_port == 12345


def test_load_config_env_reachy_claw_config(tmp_path, monkeypatch):
    """REACHY_CLAW_CONFIG env var points to config file."""
    cfg_file = tmp_path / "my-config.yaml"
    cfg_file.write_text("tts:\n  backend: none\n")
    monkeypatch.setenv("REACHY_CLAW_CONFIG", str(cfg_file))

    config = load_config()

    assert config.tts_backend == "none"


def test_nested_backend_config(tmp_path):
    """Backend settings can be nested under the backend name."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text(
        """\
tts:
  backend: kokoro
  kokoro:
    speaker_id: 7
    speed: 1.5
"""
    )

    config = load_config(cfg_file)

    assert config.tts_backend == "kokoro"
    assert config.kokoro_speaker_id == 7
    assert config.kokoro_speed == 1.5


# ── Runtime overrides persistence ────────────────────────────────────


def test_save_and_load_runtime_overrides(tmp_path):
    """save_runtime_overrides persists fields that load_config restores."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("conversation:\n  mode: conversation\n")

    # Load, change, save
    config = load_config(cfg_file)
    assert config.conversation_mode == "conversation"

    config.conversation_mode = "monologue"
    config.ollama_system_prompt = "You are a test robot."
    save_runtime_overrides(config, ["conversation_mode", "ollama_system_prompt"])

    # Verify overrides file was created
    overrides_path = tmp_path / "runtime-overrides.yaml"
    assert overrides_path.is_file()

    # Reload — overrides should apply
    config2 = load_config(cfg_file)
    assert config2.conversation_mode == "monologue"
    assert config2.ollama_system_prompt == "You are a test robot."


def test_runtime_overrides_merge_not_clobber(tmp_path):
    """Multiple save calls merge fields, not overwrite the whole file."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("conversation:\n  mode: conversation\n")

    config = load_config(cfg_file)
    config.conversation_mode = "monologue"
    save_runtime_overrides(config, ["conversation_mode"])

    # Save a different field
    config.ollama_system_prompt = "custom prompt"
    save_runtime_overrides(config, ["ollama_system_prompt"])

    # Both should survive
    config2 = load_config(cfg_file)
    assert config2.conversation_mode == "monologue"
    assert config2.ollama_system_prompt == "custom prompt"


def test_runtime_overrides_extra_fields(tmp_path):
    """Fields not in _YAML_FIELD_MAP are stored in _extra section."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("")

    config = load_config(cfg_file)
    config.dashboard_volume = 60  # type: ignore[attr-defined]
    save_runtime_overrides(config, ["dashboard_volume"])

    config2 = load_config(cfg_file)
    assert getattr(config2, "dashboard_volume", None) == 60


def test_env_overrides_runtime_overrides(tmp_path, monkeypatch):
    """Environment variables still take priority over runtime overrides."""
    cfg_file = tmp_path / "reachy-claw.yaml"
    cfg_file.write_text("")

    config = load_config(cfg_file)
    config.stt_backend = "whisper"
    save_runtime_overrides(config, ["stt_backend"])

    monkeypatch.setenv("STT_BACKEND", "openai")
    config2 = load_config(cfg_file)
    assert config2.stt_backend == "openai"  # env wins
