"""Attention & gaze: decide who the robot looks at and when it engages.

Fed the per-frame face list from :class:`~reachy_voice.vision.VisionClient`,
this turns it into two things:

  * **gaze** — a continuous head/body aim toward the nearest *close* face, so
    the robot visibly follows a visitor (``on_gaze(yaw_deg, pitch_deg)``; called
    with ``(None, None)`` to recentre once the face is gone).

  * **engagement** — a one-shot ``on_engage()`` when a close face has lingered
    long enough to count as "this person wants to interact" (then a cooldown
    so the same visitor isn't greeted on a loop).

The whole point is the gate: a face that is small (far) or fleeting (walking
past) is ignored — only someone who comes up close and stays triggers the robot.
Pure logic, no SDK/threads, so it unit-tests offline. ``clock`` is injectable.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("reachy_voice.attention")


def _bbox_area(bbox) -> float:  # noqa: ANN001
    try:
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    except Exception:  # noqa: BLE001 — malformed/missing bbox
        return 0.0


def _bbox_center(bbox) -> tuple[float, float]:  # noqa: ANN001
    x1, y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


class AttentionTracker:
    def __init__(
        self,
        *,
        on_gaze: Callable[[float | None, float | None], None],
        on_engage: Callable[[], None],
        min_area: float = 0.035,
        stable_s: float = 1.2,
        cooldown_s: float = 15.0,
        max_yaw: float = 30.0,
        max_pitch: float = 16.0,
        lost_s: float = 3.5,
        deadzone: float = 0.05,
        smooth_alpha: float = 0.4,
        lock_radius: float = 0.25,
        invert_x: bool = False,
        invert_y: bool = False,
        gaze_fn: Callable[[float, float], tuple[float, float]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._on_gaze = on_gaze
        self._on_engage = on_engage
        # SDK-calibrated (cx,cy)->(yaw,pitch). None → fall back to the simple
        # proportional mapping (used by unit tests).
        self._gaze_fn = gaze_fn
        self._min_area = min_area
        self._stable_s = stable_s
        self._cooldown_s = cooldown_s
        self._max_yaw = max_yaw
        self._max_pitch = max_pitch
        self._lost_s = lost_s
        self._deadzone = deadzone
        self._alpha = smooth_alpha       # EMA weight for new detections (0..1)
        self._lock_radius = lock_radius  # keep the same face if within this of the lock
        self._sx = -1.0 if invert_x else 1.0
        self._sy = -1.0 if invert_y else 1.0
        import time as _time

        self._clock = clock or _time.monotonic
        self._present_since: float | None = None
        self._last_seen: float | None = None
        self._last_engage = -1e9
        # EMA-smoothed centre of the locked face (image coords); None = no lock.
        self._scx: float | None = None
        self._scy: float | None = None
        self._gaze_cleared = True
        self._greeted = False  # greet a visitor ONCE per arrival, not on a loop

    def update(self, faces: list[dict]) -> None:
        """Process one frame of detected faces.

        Detection is sparse and noisy on real hardware (faces blink in and out,
        bboxes jitter), so we DON'T snap to each raw frame. Instead we lock onto
        one person, EMA-smooth their centre, and — critically — HOLD the last aim
        through detection gaps, only recentring after a long sustained absence.
        That turns "blink → snap → recentre → snap" into a steady follow.
        """
        now = self._clock()
        target = self._pick_target(faces)

        if target is None:
            # No close face THIS frame. Don't react to the gap: hold the current
            # aim. Only after a long sustained absence do we drop the lock and
            # recentre (once) — covering blinks without twitching.
            self._present_since = None
            if (
                not self._gaze_cleared
                and self._last_seen is not None
                and (now - self._last_seen) >= self._lost_s
            ):
                self._on_gaze(None, None)
                self._gaze_cleared = True
                self._scx = self._scy = None
                self._greeted = False  # visitor left → a new arrival may greet
            return

        # EMA-smooth the locked face's centre (kills per-frame jitter).
        cx, cy = _bbox_center(target["bbox"])
        if self._scx is None:
            self._scx, self._scy = cx, cy
        else:
            self._scx += self._alpha * (cx - self._scx)
            self._scy += self._alpha * (cy - self._scy)

        # Aim from the SMOOTHED centre. Signs verified kinematically against
        # create_head_pose (see tests/test_gaze_simulation.py):
        #   • image x: face left of centre (nx<0) → turn body left (+yaw)  → -nx
        #   • image y: face high in frame (ny<0) → look UP (head pitch<0). +pitch
        #     tilts DOWN and image-y grows downward, so pitch takes the SAME sign
        #     as ny (NOT mirrored like yaw).
        if self._gaze_fn is not None:
            # SDK-calibrated gaze (camera intrinsics + look-at geometry).
            yaw, pitch = self._gaze_fn(self._scx, self._scy)
            yaw *= self._sx
            pitch *= self._sy
        else:
            nx = self._dead(2.0 * self._scx - 1.0)
            ny = self._dead(2.0 * self._scy - 1.0)
            yaw = -self._sx * nx * self._max_yaw
            pitch = self._sy * ny * self._max_pitch
        yaw = max(-self._max_yaw, min(self._max_yaw, yaw))
        pitch = max(-self._max_pitch, min(self._max_pitch, pitch))
        self._on_gaze(yaw, pitch)
        self._last_seen = now
        self._gaze_cleared = False

        # Engagement: greet ONCE per arrival — when this visitor has lingered
        # long enough and we haven't greeted them this visit. The flag resets
        # only after they leave (above), so a lingering person isn't greeted on
        # a loop (which both annoys and adds servo noise while they talk).
        if self._present_since is None:
            self._present_since = now
        elif (
            not self._greeted
            and (now - self._present_since) >= self._stable_s
            and (now - self._last_engage) >= self._cooldown_s
        ):
            self._greeted = True
            self._last_engage = now
            logger.info("attention: visitor engaged (close + lingering)")
            try:
                self._on_engage()
            except Exception:  # noqa: BLE001 — engagement must not break tracking
                logger.debug("on_engage failed", exc_info=True)

    # ── helpers ───────────────────────────────────────────────────────
    def _pick_target(self, faces: list[dict]) -> dict | None:
        """Choose which close face to follow. While we hold a lock, prefer the
        face nearest the locked position (stay on the same person across frames);
        otherwise take the largest (closest) one. Faces below ``min_area`` (far /
        passing by) are ignored entirely."""
        close = [f for f in (faces or []) if _bbox_area(f.get("bbox")) >= self._min_area]
        if not close:
            return None
        if self._scx is not None:
            locked = min(close, key=lambda f: self._dist2(f, self._scx, self._scy))
            if self._dist2(locked, self._scx, self._scy) <= self._lock_radius ** 2:
                return locked
        # no lock (or the locked person left the radius) → largest/closest face
        return max(close, key=lambda f: _bbox_area(f.get("bbox")))

    @staticmethod
    def _dist2(face: dict, sx: float, sy: float) -> float:
        cx, cy = _bbox_center(face["bbox"])
        return (cx - sx) ** 2 + (cy - sy) ** 2

    def _dead(self, v: float) -> float:
        return 0.0 if abs(v) < self._deadzone else v
