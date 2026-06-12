"""Unit tests for AttentionTracker — the gate that makes the robot follow and
greet a close, lingering visitor while ignoring small/fleeting passers-by.

A fake clock drives time so dwell/cooldown are deterministic and offline.
"""

from __future__ import annotations

from reachy_voice.attention import AttentionTracker


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _face(cx, cy, size):
    """A face centred at (cx,cy) with a square bbox of side `size` (frame frac)."""
    h = size / 2.0
    return {"bbox": [cx - h, cy - h, cx + h, cy + h], "identity": None, "emotion": "neutral"}


def _make(clock, **kw):
    gazes: list = []
    engages: list = []
    params = dict(
        min_area=0.04, stable_s=1.0, cooldown_s=10.0,
        max_yaw=30.0, max_pitch=16.0, lost_s=1.0, deadzone=0.05,
    )
    params.update(kw)  # let individual tests override any tunable
    tr = AttentionTracker(
        on_gaze=lambda y, p: gazes.append((y, p)),
        on_engage=lambda: engages.append(clock.t),
        clock=clock, **params,
    )
    return tr, gazes, engages


# ── the gate ──────────────────────────────────────────────────────────


def test_small_passerby_is_ignored():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    # area 0.1*0.1 = 0.01 < min_area 0.04 → ignored
    for _ in range(20):
        tr.update([_face(0.5, 0.5, 0.1)])
        clock.advance(0.1)
    assert gazes == []
    assert engages == []


def test_close_face_is_tracked():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    # area 0.3*0.3 = 0.09 >= 0.04, centred-left so it should aim left
    tr.update([_face(0.2, 0.5, 0.3)])
    assert gazes, "a close face should produce a gaze command"
    yaw, pitch = gazes[-1]
    # face is left of centre (cx=0.2 → nx=-0.6); default (no invert) turns left → +yaw
    assert yaw > 0


def test_fleeting_close_face_does_not_engage():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    # close but only for 0.5s (< stable_s 1.0) then gone
    tr.update([_face(0.5, 0.5, 0.3)]); clock.advance(0.5)
    tr.update([_face(0.5, 0.5, 0.3)]); clock.advance(0.1)
    tr.update([])  # walked off
    assert engages == []


def test_close_and_lingering_engages_once_then_cooldown():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    # present continuously past stable_s
    for _ in range(20):  # 20 * 0.1s = 2s > stable 1.0s
        tr.update([_face(0.5, 0.5, 0.3)])
        clock.advance(0.1)
    assert len(engages) == 1, "should greet exactly once on first dwell"
    # keep lingering — must NOT re-greet during cooldown (10s)
    for _ in range(50):  # +5s, still < cooldown
        tr.update([_face(0.5, 0.5, 0.3)])
        clock.advance(0.1)
    assert len(engages) == 1


def test_lingering_visitor_greeted_only_once_then_leaves_and_returns():
    clock = _Clock()
    tr, gazes, engages = _make(clock, lost_s=1.0, cooldown_s=2.0)
    # stand close and LINGER well past stable + cooldown
    for _ in range(60):  # 12s present
        tr.update([_face(0.5, 0.5, 0.3)]); clock.advance(0.2)
    assert len(engages) == 1, "must greet a lingering visitor only ONCE"
    # leave (sustained absence resets the flag)
    for _ in range(10):
        tr.update([]); clock.advance(0.2)
    # return → greet again
    for _ in range(10):
        tr.update([_face(0.5, 0.5, 0.3)]); clock.advance(0.2)
    assert len(engages) == 2, "a returning visitor should be greeted again"


def test_recenters_after_face_lost():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    tr.update([_face(0.2, 0.5, 0.3)])  # tracked
    gazes.clear()
    tr.update([])                      # gone; within lost_s grace → no recenter yet
    assert gazes == []
    clock.advance(1.1)                 # past lost_s
    tr.update([])
    assert gazes[-1] == (None, None)   # recentred exactly once


def test_deadzone_suppresses_tiny_offset():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    # face almost centred (cx=0.51 → nx=0.02 < deadzone 0.05) → yaw 0
    tr.update([_face(0.51, 0.5, 0.3)])
    yaw, pitch = gazes[-1]
    assert yaw == 0.0


def test_invert_x_flips_turn_direction():
    clock = _Clock()
    tr, gazes, engages = _make(clock, invert_x=True)
    tr.update([_face(0.2, 0.5, 0.3)])  # left of centre
    yaw, _ = gazes[-1]
    assert yaw < 0  # inverted → opposite sign


def test_nearest_of_several_faces_is_chosen():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    far = _face(0.1, 0.5, 0.25)   # area 0.0625
    near = _face(0.9, 0.5, 0.4)   # area 0.16 (bigger = closer)
    tr.update([far, near])
    yaw, _ = gazes[-1]
    # near face is right of centre (cx=0.9) → turns right (-yaw) by default
    assert yaw < 0


def test_malformed_bbox_is_ignored():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    tr.update([{"bbox": None}, {"bbox": [0.1, 0.2]}, {}])
    assert gazes == [] and engages == []


# ── robustness to real (sparse, noisy) detection ──────────────────────


def test_short_detection_gap_does_not_recenter():
    # lost_s defaults large; a brief blank gap must NOT snap back to centre.
    clock = _Clock()
    tr, gazes, engages = _make(clock, lost_s=3.5)
    tr.update([_face(0.25, 0.4, 0.3)]); clock.advance(0.2)
    gazes.clear()
    for _ in range(4):                       # ~0.8s of NO detection (a blink)
        tr.update([]); clock.advance(0.2)
    assert (None, None) not in gazes, "must hold aim through a short gap, not recentre"


def test_long_absence_recenters_once():
    clock = _Clock()
    tr, gazes, engages = _make(clock, lost_s=3.5)
    tr.update([_face(0.25, 0.4, 0.3)]); clock.advance(0.2)
    for _ in range(25):                      # 5s of nothing > lost_s
        tr.update([]); clock.advance(0.2)
    assert gazes[-1] == (None, None)
    assert sum(1 for g in gazes if g == (None, None)) == 1  # exactly one recenter


def test_ema_smooths_noisy_bbox_jitter():
    # a face jittering ±0.05 around cx=0.3 must yield a much smoother gaze track.
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    jitter = [0.25, 0.35, 0.25, 0.35, 0.25, 0.35]
    for cx in jitter:
        tr.update([_face(cx, 0.5, 0.3)]); clock.advance(0.2)
    yaws = [g[0] for g in gazes]
    raw_swing = max(yaws) - min(yaws)
    # raw nx swing would be 0.2 → ~ -0.2*30 = 6° each way; EMA(0.4) damps it
    step = max(abs(yaws[i + 1] - yaws[i]) for i in range(len(yaws) - 1))
    assert step < raw_swing  # smoothed steps are smaller than the raw swing


def test_lock_stays_on_same_person_when_two_faces_present():
    clock = _Clock()
    tr, gazes, engages = _make(clock)
    left = _face(0.25, 0.5, 0.30)   # locked first
    right = _face(0.80, 0.5, 0.34)  # slightly larger (closer) but far from lock
    tr.update([left]); clock.advance(0.2)
    gazes.clear()
    for _ in range(3):
        tr.update([left, right]); clock.advance(0.2)
    # despite `right` being larger, lock keeps us on the left person → yaw stays +
    assert all(g[0] > 0 for g in gazes), "lock must not jump to the larger far face"
