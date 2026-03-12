"""MotionPlugin -- robot expression execution and head tracking fusion.

Consumes the HeadTargetBus for fused head tracking (face + DOA),
processes the emotion queue, and plays idle animations.

Motion separation:
  - Body rotation (base Z-axis): tracks person's horizontal position
  - Head (Stewart platform): mirrors person's head pose (pitch/roll)
  - Antennas: emotion-driven dynamic animations (sine-wave oscillation)
"""

import asyncio
import logging
import math
import time

import numpy as np

from ..motion.emotion_mapper import AntennaAnimation
from ..plugin import Plugin

logger = logging.getLogger(__name__)

_ANIM_HZ = 30  # antenna animation update rate


class MotionPlugin(Plugin):
    """Execute robot expressions and fused head tracking."""

    name = "motion"

    # Motor preset definitions: smoothing, deadband, poll_interval, body_smoothing, body_deadband
    MOTOR_PRESETS = {
        "sensitive": {"smoothing": 0.50, "deadband": 1.0, "poll": 0.03, "body_smoothing": 0.45, "body_deadband": 1.0},
        "moderate":  {"smoothing": 0.35, "deadband": 2.0, "poll": 0.05, "body_smoothing": 0.35, "body_deadband": 2.0},
        "smart":     {"smoothing": 0.20, "deadband": 3.0, "poll": 0.07, "body_smoothing": 0.25, "body_deadband": 3.0},
    }

    def __init__(self, app):
        super().__init__(app)
        # Motor enable/disable (sleep mode)
        self._motor_enabled = True
        self._motor_preset = "moderate"

        # Head tracking EMA state (Stewart platform — pitch/roll mirroring)
        self._current_yaw = 0.0
        self._current_pitch = 0.0
        self._current_roll = 0.0
        self._smoothing = app.config.motion_head_tracking_smoothing
        self._min_angle_change = 2.0  # degrees — head deadband
        self._last_applied_yaw = 0.0
        self._last_applied_pitch = 0.0
        self._last_applied_roll = 0.0
        self._neutral_decay = 0.05

        # Body rotation EMA state (base Z-axis — horizontal tracking)
        self._current_body_yaw = 0.0
        self._last_applied_body_yaw = 0.0
        self._body_smoothing = 0.35  # body EMA
        self._body_min_angle = 2.0  # degrees — body deadband

        # Speech wobble offsets (roll, pitch, yaw) set by HeadWobbler
        self._speech_roll = 0.0
        self._speech_pitch = 0.0
        self._speech_yaw = 0.0

        # Antenna animation state
        self._antenna_anim: AntennaAnimation | None = None
        self._antenna_anim_start: float = 0.0

    def set_speech_offsets(self, offsets: tuple) -> None:
        """Called by HeadWobbler to set speech-driven head offsets."""
        self._speech_roll, self._speech_pitch, self._speech_yaw = offsets

    def set_motor_enabled(self, enabled: bool) -> None:
        """Enable or disable motor output (sleep mode)."""
        self._motor_enabled = enabled
        logger.info("Motor %s", "enabled" if enabled else "disabled (sleep)")

    def apply_motor_preset(self, preset: str) -> None:
        """Apply a motor tracking preset (sensitive/moderate/smart)."""
        params = self.MOTOR_PRESETS.get(preset)
        if not params:
            logger.warning("Unknown motor preset: %s", preset)
            return
        self._motor_preset = preset
        self._smoothing = params["smoothing"]
        self._min_angle_change = params["deadband"]
        self._body_smoothing = params["body_smoothing"]
        self._body_min_angle = params["body_deadband"]
        # poll_interval is read each iteration from config, so update config
        self.app.config.motion_head_tracking_poll_interval = params["poll"]
        logger.info("Motor preset: %s (smoothing=%.2f, deadband=%.1f°, poll=%.3fs)",
                     preset, params["smoothing"], params["deadband"], params["poll"])

    def get_motor_state(self) -> dict:
        """Return current motor state for dashboard sync."""
        return {"enabled": self._motor_enabled, "preset": self._motor_preset}

    async def start(self):
        await asyncio.gather(
            self._motion_loop(),
            self._head_tracking_loop(),
            self._antenna_animation_loop(),
        )

    async def _motion_loop(self):
        """Process queued expressions and idle animations."""
        logger.info("Motion loop started")
        last_idle = time.monotonic()
        config = self.app.config

        while self._running:
            if not self._motor_enabled:
                await asyncio.sleep(0.2)
                continue
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
        """Consume fused head targets and drive body rotation + head pose."""
        logger.info("Head tracking fusion loop started")

        while self._running:
            poll_interval = self.app.config.motion_head_tracking_poll_interval

            if not self._motor_enabled:
                await asyncio.sleep(poll_interval)
                continue

            if self.app.is_speaking and self.app.config.conversation_mode != "monologue":
                # During speech in conversation mode, apply wobble offsets instead of tracking
                # In monologue mode, keep tracking the user's face while speaking
                self._apply_speech_wobble()
                await asyncio.sleep(poll_interval)
                continue

            target = self.app.head_targets.get_fused_target()

            if target.source == "none":
                # Decay all axes to neutral
                self._current_yaw += self._neutral_decay * (0.0 - self._current_yaw)
                self._current_pitch += self._neutral_decay * (0.0 - self._current_pitch)
                self._current_roll += self._neutral_decay * (0.0 - self._current_roll)
                self._current_body_yaw += self._neutral_decay * (0.0 - self._current_body_yaw)
            else:
                # Head (Stewart platform): pitch + roll mirroring
                self._current_yaw += self._smoothing * (target.yaw - self._current_yaw)
                self._current_pitch += self._smoothing * (target.pitch - self._current_pitch)
                self._current_roll += self._smoothing * (target.roll - self._current_roll)
                # Body rotation: horizontal person tracking (slower EMA to avoid jitter)
                self._current_body_yaw += self._body_smoothing * (target.body_yaw - self._current_body_yaw)

            # Update head pose if changed enough
            delta_yaw = abs(self._current_yaw - self._last_applied_yaw)
            delta_pitch = abs(self._current_pitch - self._last_applied_pitch)
            delta_roll = abs(self._current_roll - self._last_applied_roll)

            if delta_yaw >= self._min_angle_change or delta_pitch >= self._min_angle_change or delta_roll >= self._min_angle_change:
                self._set_head_pose(self._current_yaw, self._current_pitch, self._current_roll)
                self._last_applied_yaw = self._current_yaw
                self._last_applied_pitch = self._current_pitch
                self._last_applied_roll = self._current_roll

            # Update body rotation if changed enough
            delta_body = abs(self._current_body_yaw - self._last_applied_body_yaw)
            if delta_body >= self._body_min_angle:
                self._set_body_yaw(self._current_body_yaw)
                self._last_applied_body_yaw = self._current_body_yaw

            await asyncio.sleep(poll_interval)

        logger.info("Head tracking fusion loop stopped")

    async def _antenna_animation_loop(self):
        """Drive continuous antenna animations at 30Hz."""
        logger.info("Antenna animation loop started")
        interval = 1.0 / _ANIM_HZ

        while self._running:
            if not self._motor_enabled:
                await asyncio.sleep(interval)
                continue
            anim = self._antenna_anim
            if anim is None:
                await asyncio.sleep(interval)
                continue

            t = time.monotonic() - self._antenna_anim_start
            if t > anim.duration:
                # Animation finished — decay to neutral
                self._antenna_anim = None
                self._set_antennas(0.0, 0.0)
                await asyncio.sleep(interval)
                continue

            # Sine wave: each antenna oscillates around center
            phase_l = 2.0 * math.pi * anim.frequency * t
            phase_r = phase_l + anim.phase_offset

            # Ease-in over first 0.3s, ease-out over last 0.3s
            ease = 1.0
            if t < 0.3:
                ease = t / 0.3
            elif t > anim.duration - 0.3:
                ease = (anim.duration - t) / 0.3

            left = anim.center + anim.amplitude * ease * math.sin(phase_l)
            right = anim.center + anim.amplitude * ease * math.sin(phase_r)

            self._set_antennas(right, left)
            await asyncio.sleep(interval)

        logger.info("Antenna animation loop stopped")

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

    def _set_head_pose(self, yaw: float, pitch: float, roll: float = 0.0) -> None:
        """Set head yaw, pitch, and roll on the Stewart platform."""
        reachy = self.app.reachy
        if not reachy:
            return
        try:
            from reachy_mini.utils import create_head_pose

            pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
            reachy.set_target_head_pose(pose)
        except Exception:
            pass

    def _set_body_yaw(self, yaw_degrees: float) -> None:
        """Set body base rotation (Z-axis) for horizontal person tracking."""
        reachy = self.app.reachy
        if not reachy:
            return
        try:
            reachy.set_target_body_yaw(math.radians(yaw_degrees))
        except Exception:
            pass

    def _set_antennas(self, right_deg: float, left_deg: float) -> None:
        """Set antenna positions immediately (degrees → radians)."""
        reachy = self.app.reachy
        if not reachy:
            return
        try:
            reachy.set_target_antenna_joint_positions([
                math.radians(right_deg),
                math.radians(left_deg),
            ])
        except Exception:
            pass

    def _execute_expression(self, expr) -> None:
        """Execute a robot expression (head + antenna movement)."""
        reachy = self.app.reachy
        if not reachy:
            logger.info(f"[SIM] Expression: {expr.description}")
            return

        # Start antenna animation if present (takes priority over static antenna)
        if expr.antenna_anim:
            self._antenna_anim = expr.antenna_anim
            self._antenna_anim_start = time.monotonic()
            logger.info(
                f"Antenna anim: center={expr.antenna_anim.center:.0f}° "
                f"amp={expr.antenna_anim.amplitude:.0f}° "
                f"freq={expr.antenna_anim.frequency:.1f}Hz "
                f"dur={expr.antenna_anim.duration:.1f}s | {expr.description}"
            )

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

            # Static antenna (only if no animation)
            if expr.antenna and not expr.antenna_anim:
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
