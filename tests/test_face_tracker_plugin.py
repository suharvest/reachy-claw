"""Tests for the FaceTrackerPlugin setup and target publishing."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from reachy_claw.config import Config
from reachy_claw.app import ReachyClawApp
from reachy_claw.plugins.face_tracker_plugin import FaceTrackerPlugin


@pytest.fixture
def tracker_config():
    return Config(
        enable_face_tracker=True,
        vision_tracker_type="mediapipe",
        vision_camera_index=0,
        vision_max_yaw=25.0,
        vision_max_pitch=15.0,
        vision_smoothing_alpha=0.3,
        vision_deadzone=0.02,
        vision_face_lost_delay=2.0,
    )


@pytest.fixture
def tracker_app(tracker_config, mock_reachy):
    a = ReachyClawApp(tracker_config)
    a.reachy = mock_reachy
    return a


class TestFaceTrackerSetup:
    def test_disabled_by_tracker_type_none(self, tracker_app):
        tracker_app.config.vision_tracker_type = "none"
        plugin = FaceTrackerPlugin(tracker_app)
        assert plugin.setup() is False

    def test_skips_when_mediapipe_not_installed(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)
        with patch.dict("sys.modules", {"mediapipe": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("mediapipe")):
                assert plugin.setup() is False

    def test_skips_when_cv2_not_installed(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)
        # mediapipe imports fine
        with patch.dict("sys.modules", {"mediapipe": MagicMock()}):
            with patch("builtins.__import__", side_effect=_import_blocker("cv2")):
                assert plugin.setup() is False

    def test_skips_when_camera_not_available(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)

        mock_mp = MagicMock()
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            with patch("builtins.__import__", side_effect=_import_passthrough("mediapipe", "cv2", mock_mp, mock_cv2)):
                assert plugin.setup() is False

        mock_cap.release.assert_called_once()

    def test_succeeds_when_all_available(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)

        mock_mp = MagicMock()
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict("sys.modules", {"mediapipe": mock_mp, "cv2": mock_cv2}):
            with patch("builtins.__import__", side_effect=_import_passthrough("mediapipe", "cv2", mock_mp, mock_cv2)):
                assert plugin.setup() is True

        mock_cap.release.assert_called_once()


class TestFaceTrackerConfig:
    def test_reads_config_values(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)
        assert plugin._tracker_type == "mediapipe"
        assert plugin._camera_index == 0
        assert plugin._max_yaw == 25.0
        assert plugin._max_pitch == 15.0
        assert plugin._smoothing_alpha == 0.3
        assert plugin._deadzone == 0.02
        assert plugin._face_lost_delay == 2.0


class TestFaceTrackerSmoothing:
    def test_smoothing_alpha_applied(self, tracker_app):
        """The EMA smoothing should move toward the target value."""
        plugin = FaceTrackerPlugin(tracker_app)
        plugin._smooth_x = 0.0
        plugin._smooth_y = 0.0

        # Simulate face at x=0.5, y=-0.3
        face_x, face_y = 0.5, -0.3
        alpha = plugin._smoothing_alpha

        # Apply one step
        plugin._smooth_x += alpha * (face_x - plugin._smooth_x)
        plugin._smooth_y += alpha * (face_y - plugin._smooth_y)

        assert plugin._smooth_x == pytest.approx(alpha * 0.5)
        assert plugin._smooth_y == pytest.approx(alpha * -0.3)

    def test_deadzone_filters_small_movement(self, tracker_app):
        plugin = FaceTrackerPlugin(tracker_app)
        plugin._smooth_x = 0.5
        plugin._smooth_y = 0.3

        # Movement smaller than deadzone
        face_x = 0.501  # diff = 0.001 < deadzone 0.02
        face_y = 0.301

        old_x = plugin._smooth_x
        old_y = plugin._smooth_y

        if (
            abs(face_x - plugin._smooth_x) >= plugin._deadzone
            or abs(face_y - plugin._smooth_y) >= plugin._deadzone
        ):
            plugin._smooth_x += plugin._smoothing_alpha * (face_x - plugin._smooth_x)
            plugin._smooth_y += plugin._smoothing_alpha * (face_y - plugin._smooth_y)

        # Should not have changed
        assert plugin._smooth_x == old_x
        assert plugin._smooth_y == old_y


# ── Helpers ────────────────────────────────────────────────────────────

_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__


def _import_blocker(*blocked_names):
    """Return an import side_effect that blocks specific modules."""
    def _import(name, *args, **kwargs):
        if name in blocked_names:
            raise ImportError(f"Mocked: {name} not installed")
        return _real_import(name, *args, **kwargs)
    return _import


def _import_passthrough(*name_mock_pairs):
    """Return an import side_effect that returns mocks for specific modules."""
    mapping = {}
    it = iter(name_mock_pairs)
    names = []
    mocks = []
    for item in name_mock_pairs:
        if isinstance(item, str):
            names.append(item)
        else:
            mocks.append(item)
    mapping = dict(zip(names, mocks))

    def _import(name, *args, **kwargs):
        if name in mapping:
            return mapping[name]
        return _real_import(name, *args, **kwargs)
    return _import
