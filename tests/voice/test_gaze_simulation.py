"""Offline kinematic simulation of the gaze loop.

Instead of trial-and-error on the real robot (slow, and a wrong sign can slam
the motors), this feeds synthetic face positions through the REAL
``AttentionTracker`` and checks the resulting aim with the REAL SDK
``create_head_pose`` — so a flipped sign is caught here, instantly, not on
hardware.

Ground truth from the SDK (forward axis = +X):
  • +pitch tilts the head DOWN (forward z < 0); -pitch looks UP.
  • +yaw turns LEFT (forward y > 0); -yaw turns RIGHT.
Camera image: x grows rightward, y grows downward, both normalised to [-1,1].
"""

from __future__ import annotations

import numpy as np
from reachy_mini.utils import create_head_pose

from reachy_voice.attention import AttentionTracker


def _gaze_for_face(cx: float, cy: float, size: float = 0.3):
    """Run one close face at (cx,cy) through the tracker; return (yaw, pitch)."""
    out: dict[str, float] = {}
    tr = AttentionTracker(
        on_gaze=lambda y, p: out.update(yaw=y, pitch=p),
        on_engage=lambda: None,
        min_area=0.01, stable_s=0.5, cooldown_s=10.0,
        max_yaw=35.0, max_pitch=20.0, deadzone=0.0,
    )
    h = size / 2.0
    tr.update([{"bbox": [cx - h, cy - h, cx + h, cy + h]}])
    return out.get("yaw"), out.get("pitch")


def _head_forward(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    rot = np.array(create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True))[:3, :3]
    return rot @ np.array([1.0, 0.0, 0.0])


# ── vertical: the bug we just fixed ───────────────────────────────────


def test_face_high_in_frame_makes_head_look_up():
    _, pitch = _gaze_for_face(0.5, 0.2)          # centred, high
    fz = _head_forward(0.0, pitch)[2]
    assert fz > 0.02, f"high face should look UP, got forward z={fz:+.2f}"


def test_face_low_in_frame_makes_head_look_down():
    _, pitch = _gaze_for_face(0.5, 0.8)          # centred, low
    fz = _head_forward(0.0, pitch)[2]
    assert fz < -0.02, f"low face should look DOWN, got forward z={fz:+.2f}"


def test_centred_face_is_roughly_level():
    _, pitch = _gaze_for_face(0.5, 0.5)
    assert abs(pitch) < 1e-6


# ── horizontal: body turn (must follow the visitor) ───────────────────


def test_face_left_turns_left():
    yaw, _ = _gaze_for_face(0.2, 0.5)            # left of centre
    # body_yaw = yaw; +yaw = LEFT per the SDK kinematics
    assert yaw > 0
    assert _head_forward(yaw, 0.0)[1] > 0.02     # forward points left (+y)


def test_face_right_turns_right():
    yaw, _ = _gaze_for_face(0.8, 0.5)            # right of centre
    assert yaw < 0
    assert _head_forward(yaw, 0.0)[1] < -0.02    # forward points right (-y)


# ── magnitude sanity (slew limiting handles rate; this checks mapping) ─


def test_corner_face_does_not_exceed_configured_maxima():
    yaw, pitch = _gaze_for_face(0.0, 0.0)        # extreme top-left
    assert abs(yaw) <= 35.0 + 1e-6
    assert abs(pitch) <= 20.0 + 1e-6
