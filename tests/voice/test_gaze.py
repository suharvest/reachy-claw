"""SDK-calibrated FaceGaze: a face's normalised position maps to a sensible
head yaw/pitch (uses the real Reachy Mini Lite intrinsics from the SDK)."""

from __future__ import annotations

import pytest

from reachy_voice.gaze import FaceGaze

fg = FaceGaze()
pytestmark = pytest.mark.skipif(not fg.ok, reason="SDK Lite calibration unavailable")


def test_centre_face_is_straight_ahead():
    yaw, pitch = fg.yaw_pitch(0.5, 0.5)
    assert abs(yaw) < 1.0 and abs(pitch) < 1.0


def test_face_left_looks_left():
    yaw, _ = fg.yaw_pitch(0.2, 0.5)
    assert yaw > 5  # +yaw = left


def test_face_right_looks_right():
    yaw, _ = fg.yaw_pitch(0.8, 0.5)
    assert yaw < -5


def test_face_high_looks_up():
    _, pitch = fg.yaw_pitch(0.5, 0.2)
    assert pitch < -5  # -pitch = up (matches create_head_pose)


def test_face_low_looks_down():
    _, pitch = fg.yaw_pitch(0.5, 0.8)
    assert pitch > 5


def test_symmetry_left_right():
    yl, _ = fg.yaw_pitch(0.3, 0.5)
    yr, _ = fg.yaw_pitch(0.7, 0.5)
    assert abs(yl + yr) < 2.0  # roughly mirror-symmetric about centre
