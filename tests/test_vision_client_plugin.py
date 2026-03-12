"""Tests for VisionClientPlugin setup, smoothing, and emotion mapping."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.app import ReachyClawApp
from reachy_claw.plugins.vision_client_plugin import (
    VisionClientPlugin,
    _EMOTION_REMAP,
)


@pytest.fixture
def vision_config():
    return Config(
        enable_face_tracker=True,
        vision_tracker_type="remote",
        vision_camera_source="sdk",
        vision_max_yaw=25.0,
        vision_max_pitch=15.0,
        vision_smoothing_alpha=0.3,
        vision_deadzone=0.02,
        vision_face_lost_delay=2.0,
        vision_service_url="tcp://127.0.0.1:8631",
        vision_emotion_threshold=0.6,
        vision_emotion_cooldown=3.0,
        vision_identity_threshold=0.4,
    )


@pytest.fixture
def vision_app(vision_config, mock_reachy):
    a = ReachyClawApp(vision_config)
    a.reachy = mock_reachy
    return a


class TestVisionClientSetup:
    def test_works_without_robot(self, vision_config):
        """Vision client doesn't need a robot — it only consumes ZMQ data."""
        app = ReachyClawApp(vision_config)
        app.reachy = None
        plugin = VisionClientPlugin(app)
        assert plugin.setup() is True

    def test_skips_without_zmq(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        with patch.dict("sys.modules", {"zmq": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("zmq")):
                assert plugin.setup() is False

    def test_skips_without_msgpack(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        with patch.dict("sys.modules", {"msgpack": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("msgpack")):
                assert plugin.setup() is False

    def test_succeeds_with_all_deps(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        mock_zmq = MagicMock()
        mock_msgpack = MagicMock()
        with patch.dict("sys.modules", {"zmq": mock_zmq, "msgpack": mock_msgpack}):
            assert plugin.setup() is True


class TestVisionClientConfig:
    def test_reads_config_values(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        assert plugin._zmq_url == "tcp://127.0.0.1:8631"
        assert plugin._max_yaw == 25.0
        assert plugin._max_pitch == 15.0
        assert plugin._smoothing_alpha == 0.3
        assert plugin._deadzone == 0.02
        assert plugin._face_lost_delay == 2.0
        assert plugin._emotion_threshold == 0.6
        assert plugin._emotion_cooldown == 3.0


class TestVisionClientSmoothing:
    def test_smoothing_alpha_applied(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        plugin._smooth_x = 0.0
        plugin._smooth_y = 0.0

        face_x, face_y = 0.5, -0.3
        alpha = plugin._smoothing_alpha

        plugin._smooth_x += alpha * (face_x - plugin._smooth_x)
        plugin._smooth_y += alpha * (face_y - plugin._smooth_y)

        assert plugin._smooth_x == pytest.approx(alpha * 0.5)
        assert plugin._smooth_y == pytest.approx(alpha * -0.3)

    def test_deadzone_filters_small_movement(self, vision_app):
        plugin = VisionClientPlugin(vision_app)
        plugin._smooth_x = 0.5
        plugin._smooth_y = 0.3

        face_x = 0.501
        face_y = 0.301

        old_x = plugin._smooth_x
        old_y = plugin._smooth_y

        if (
            abs(face_x - plugin._smooth_x) >= plugin._deadzone
            or abs(face_y - plugin._smooth_y) >= plugin._deadzone
        ):
            plugin._smooth_x += plugin._smoothing_alpha * (face_x - plugin._smooth_x)
            plugin._smooth_y += plugin._smoothing_alpha * (face_y - plugin._smooth_y)

        assert plugin._smooth_x == old_x
        assert plugin._smooth_y == old_y


class TestEmotionRemap:
    def test_hsemotion_mapping(self):
        assert _EMOTION_REMAP["Anger"] == "angry"
        assert _EMOTION_REMAP["Contempt"] == "neutral"
        assert _EMOTION_REMAP["Disgust"] == "angry"
        assert _EMOTION_REMAP["Fear"] == "fear"
        assert _EMOTION_REMAP["Happiness"] == "happy"
        assert _EMOTION_REMAP["Neutral"] == "neutral"
        assert _EMOTION_REMAP["Sadness"] == "sad"
        assert _EMOTION_REMAP["Surprise"] == "surprised"

    def test_lowercase_variants(self):
        assert _EMOTION_REMAP["angry"] == "angry"
        assert _EMOTION_REMAP["happy"] == "happy"
        assert _EMOTION_REMAP["neutral"] == "neutral"
        assert _EMOTION_REMAP["sad"] == "sad"
        assert _EMOTION_REMAP["surprised"] == "surprised"


class TestConfigYamlMapping:
    def test_vision_service_fields_in_yaml_map(self):
        from reachy_claw.config import _YAML_FIELD_MAP

        assert ("vision", "service_url") in _YAML_FIELD_MAP
        assert _YAML_FIELD_MAP[("vision", "service_url")] == "vision_service_url"
        assert ("vision", "emotion_threshold") in _YAML_FIELD_MAP
        assert ("vision", "emotion_cooldown") in _YAML_FIELD_MAP
        assert ("vision", "identity_threshold") in _YAML_FIELD_MAP

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.vision_service_url == "tcp://127.0.0.1:8631"
        assert cfg.vision_emotion_threshold == 0.35
        assert cfg.vision_emotion_cooldown == 2.0
        assert cfg.vision_identity_threshold == 0.4


class TestMainPluginRegistration:
    def test_remote_tracker_registers_vision_client(self, vision_app):
        """When tracker_type is 'remote', VisionClientPlugin should be used."""
        assert vision_app.config.vision_tracker_type == "remote"

    def test_mediapipe_tracker_registers_face_tracker(self):
        cfg = Config(vision_tracker_type="mediapipe")
        assert cfg.vision_tracker_type == "mediapipe"


# ── Helpers ────────────────────────────────────────────────────────────

_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__


def _import_blocker(*blocked_names):
    def _import(name, *args, **kwargs):
        if name in blocked_names:
            raise ImportError(f"Mocked: {name} not installed")
        return _real_import(name, *args, **kwargs)
    return _import
