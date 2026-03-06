"""Tests for CLI argument handling and config layering in main.py."""

from __future__ import annotations

import sys

from clawd_reachy_mini.main import create_config, parse_args


def test_create_config_preserves_yaml_when_cli_omits_values(tmp_path, monkeypatch):
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text(
        """\
gateway:
  host: 10.9.8.7
  port: 19990
  path: /robot-path
behavior:
  play_emotions: false
  idle_animations: false
  standalone_mode: true
vision:
  tracker: none
  camera_index: 3
"""
    )

    monkeypatch.setattr(sys, "argv", ["clawd-reachy", "--config", str(cfg_file)])
    args = parse_args()
    config = create_config(args)

    assert config.gateway_host == "10.9.8.7"
    assert config.gateway_port == 19990
    assert config.gateway_path == "/robot-path"
    assert config.play_emotions is False
    assert config.idle_animations is False
    assert config.standalone_mode is True
    assert config.vision_tracker_type == "none"
    assert config.vision_camera_index == 3


def test_create_config_cli_overrides_yaml_when_explicit(tmp_path, monkeypatch):
    cfg_file = tmp_path / "clawd.yaml"
    cfg_file.write_text(
        """\
gateway:
  host: 10.1.1.1
vision:
  tracker: none
  camera_index: 3
"""
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clawd-reachy",
            "--config",
            str(cfg_file),
            "--gateway-host",
            "127.0.0.1",
            "--tracker-type",
            "mediapipe",
            "--camera-index",
            "0",
            "--no-emotions",
        ],
    )
    args = parse_args()
    config = create_config(args)

    assert config.gateway_host == "127.0.0.1"
    assert config.vision_tracker_type == "mediapipe"
    assert config.vision_camera_index == 0
    assert config.play_emotions is False
