"""MotionPlugin -- robot expression execution and head tracking fusion.

Consumes the HeadTargetBus for fused head tracking (face + DOA),
processes the emotion queue, and plays idle animations.
"""

import asyncio
import logging
import time

import numpy as np

from ..plugin import Plugin

logger = logging.getLogger(__name__)


class MotionPlugin(Plugin):
    """Execute robot expressions and fused head tracking."""

    name = "motion"

    def __init__(self, app):
        super().__init__(app)
        # Head tracking EMA state
        self._current_yaw = 0.0
        self._current_pitch = 0.0
        self._smoothing = app.config.motion_head_tracking_smoothing
        self._min_angle_change = 3.0  # degrees
        self._last_applied_yaw = 0.0
        self._last_applied_pitch = 0.0
        self._neutral_decay = 0.05

        # Speech wobble offsets (roll, pitch, yaw) set by HeadWobbler
        self._speech_roll = 0.0
        self._speech_pitch = 0.0
        self._speech_yaw = 0.0

    def set_speech_offsets(self, offsets: tuple) -> None:
        """Called by HeadWobbler to set speech-driven head offsets."""
        self._speech_roll, self._speech_pitch, self._speech_yaw = offsets

    async def start(self):
        await asyncio.gather(
            self._motion_loop(),
            self._head_tracking_loop(),
        )

    async def _motion_loop(self):
        """Process queued expressions and idle animations."""
        logger.info("Motion loop started")
        last_idle = time.monotonic()
        config = self.app.config

        while self._running:
            expr = self.app.emotions.get_next_expression()
            if expr:
                self._execute_expression(expr)
                await asyncio.sleep(expr.head.duration if expr.head else 0.5)
                last_idle = time.monotonic()
            elif (
                config.idle_animations
                and not self.app.is_speaking
                and time.monotonic() - last_idle > config.motion_idle_animation_interval
            ):
                idle_expr = self.app.emotions.get_idle_expression()
                self._execute_expression(idle_expr)
                last_idle = time.monotonic()
                await asyncio.sleep(idle_expr.head.duration if idle_expr.head else 1.0)
            else:
                await asyncio.sleep(0.1)

        logger.info("Motion loop stopped")

    async def _head_tracking_loop(self):
        """Consume fused head targets and drive the robot head."""
        logger.info("Head tracking fusion loop started")
        poll_interval = self.app.config.motion_head_tracking_poll_interval

        while self._running:
            if self.app.is_speaking:
                # During speech, apply wobble offsets instead of tracking
                self._apply_speech_wobble()
                await asyncio.sleep(poll_interval)
                continue

            target = self.app.head_targets.get_fused_target()

            if target.source == "none":
                self._current_yaw += self._neutral_decay * (0.0 - self._current_yaw)
                self._current_pitch += self._neutral_decay * (0.0 - self._current_pitch)
            else:
                self._current_yaw += self._smoothing * (target.yaw - self._current_yaw)
                self._current_pitch += self._smoothing * (target.pitch - self._current_pitch)

            delta_yaw = abs(self._current_yaw - self._last_applied_yaw)
            delta_pitch = abs(self._current_pitch - self._last_applied_pitch)

            if delta_yaw >= self._min_angle_change or delta_pitch >= self._min_angle_change:
                self._set_head_pose(self._current_yaw, self._current_pitch)
                self._last_applied_yaw = self._current_yaw
                self._last_applied_pitch = self._current_pitch

            await asyncio.sleep(poll_interval)

        logger.info("Head tracking fusion loop stopped")

    def _apply_speech_wobble(self) -> None:
        """Apply speech-driven head wobble offsets."""
        reachy = self.app.reachy
        if not reachy:
            return

        roll = self._speech_roll
        pitch = self._speech_pitch
        yaw = self._speech_yaw

        if abs(roll) < 0.1 and abs(pitch) < 0.1 and abs(yaw) < 0.1:
            return

        try:
            from reachy_mini.utils import create_head_pose

            pose = create_head_pose(
                roll=roll, pitch=pitch, yaw=yaw, degrees=True
            )
            reachy.set_target_head_pose(pose)
        except Exception:
            pass

    def _set_head_pose(self, yaw: float, pitch: float) -> None:
        """Set head yaw and pitch for real-time tracking."""
        reachy = self.app.reachy
        if not reachy:
            return
        try:
            from reachy_mini.utils import create_head_pose

            pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
            reachy.set_target_head_pose(pose)
        except Exception:
            pass

    def _execute_expression(self, expr) -> None:
        """Execute a robot expression (head + antenna movement)."""
        reachy = self.app.reachy
        if not reachy:
            logger.info(f"[SIM] Expression: {expr.description}")
            return

        try:
            from reachy_mini.utils import create_head_pose

            kwargs = {}
            if expr.head:
                head_pose = create_head_pose(
                    yaw=expr.head.yaw,
                    roll=expr.head.roll,
                    degrees=True,
                )
                kwargs["head"] = head_pose
                kwargs["duration"] = expr.head.duration

            if expr.antenna:
                # SDK uses antennas=[right, left] in radians
                kwargs["antennas"] = [
                    np.radians(expr.antenna.right),
                    np.radians(expr.antenna.left),
                ]
                if "duration" not in kwargs:
                    kwargs["duration"] = expr.antenna.duration

            if kwargs:
                reachy.goto_target(**kwargs)

            logger.debug(f"Executed: {expr.description}")
        except Exception as e:
            logger.error(f"Failed to execute expression: {e}")
