"""SDK-calibrated gaze: turn a face's pixel position into the head yaw/pitch that
looks at it — using the **SDK's own** Reachy Mini Lite camera intrinsics, the
SDK's ``undistort_points``, and the look-at direction math from
``ReachyMini.look_at_image``/``look_at_world``.

We can't call ``look_at_image`` directly (it requires the SDK's live camera, and
we run ``no_media`` with vision-trt owning the head camera). But all the pieces
are static constants, so we replicate the exact pipeline headlessly:

    pixel (u,v) ── undistort(K,D) ──▶ camera ray ── T_head_cam ──▶ head-frame
    direction ──▶ yaw/pitch (align head +X with the direction)

So the *gaze geometry is the SDK's*, not a hand-rolled proportional guess. The
output (yaw,pitch in degrees, +yaw=left, +pitch=down, matching create_head_pose)
feeds the motion compositor as before.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("reachy_voice.gaze")

# Head→camera transform from reachy_mini.ReachyMini.__init__ (static): the camera
# is rigidly mounted on the head, so this never changes.
_T_HEAD_CAM_T = (0.0437, 0.0, 0.0512)
_T_HEAD_CAM_R = ((0, 0, 1), (-1, 0, 0), (0, -1, 0))


class FaceGaze:
    """Maps a normalised face centre (cx,cy ∈ [0,1], from vision) to (yaw,pitch)
    degrees. Falls back to ``None`` if the SDK calibration can't be loaded (caller
    then keeps its own mapping)."""

    def __init__(self) -> None:
        self._ok = False
        try:
            import numpy as np
            from reachy_mini.media.camera_constants import ReachyMiniLiteCamSpecs
            from reachy_mini.media.camera_utils import undistort_points

            self._np = np
            self._undistort = undistort_points
            self._K = np.array(ReachyMiniLiteCamSpecs.K, dtype=float)
            self._D = np.array(ReachyMiniLiteCamSpecs.D, dtype=float)
            # Use the resolution K was calibrated at (principal point = centre);
            # vision gives normalised coords so only the ratio matters.
            self._W = 2.0 * self._K[0, 2]
            self._H = 2.0 * self._K[1, 2]
            self._R = np.array(_T_HEAD_CAM_R, dtype=float)
            self._t = np.array(_T_HEAD_CAM_T, dtype=float)
            # Calibrate out any residual centre offset (principal point ≠ exact
            # centre) so a centred face → (0,0).
            self._cy0, self._cp0 = self._raw(0.5, 0.5)
            self._ok = True
            logger.info("FaceGaze: SDK Lite calibration loaded (W=%.0f H=%.0f)",
                        self._W, self._H)
        except Exception as e:  # noqa: BLE001
            logger.warning("FaceGaze unavailable (%s); caller keeps its mapping", e)

    @property
    def ok(self) -> bool:
        return self._ok

    def _raw(self, cx: float, cy: float) -> tuple[float, float]:
        np = self._np
        u, v = cx * self._W, cy * self._H
        xn, yn = self._undistort(u, v, self._K, self._D)
        ray = np.array([xn, yn, 1.0])
        ray /= np.linalg.norm(ray)
        d = self._t + self._R @ ray          # direction in head frame
        d /= np.linalg.norm(d)
        yaw = math.degrees(math.atan2(d[1], d[0]))            # +yaw = left
        pitch = math.degrees(-math.atan2(d[2], math.hypot(d[0], d[1])))  # +pitch = down
        return yaw, pitch

    def yaw_pitch(self, cx: float, cy: float) -> tuple[float, float]:
        """(cx,cy) normalised → (yaw,pitch) degrees, centre-calibrated to 0."""
        yaw, pitch = self._raw(cx, cy)
        return yaw - self._cy0, pitch - self._cp0
