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
from dataclasses import dataclass

import numpy as np

from ..motion.emotion_mapper import AntennaAnimation
from ..plugin import Plugin

logger = logging.getLogger(__name__)

_ANIM_HZ = 30  # antenna animation update rate


@dataclass
class _HeadAccent:
    """A transient, additive head offset layered on top of the gaze anchor.

    Emotion tags no longer command an absolute head pose (which fought the
    tracking loop and the speech wobbler). Instead each tag injects a short
    *accent* — a relative roll/pitch/yaw offset with an attack/hold/release
    envelope — that the compositor sums on top of "looking at the person" and
    that decays back to the anchor. This is how biological head motion reads:
    a surprised recoil rides on top of your gaze, then returns to it.
    """

    roll: float
    pitch: float
    yaw: float
    start: float
    attack: float
    hold: float
    release: float

    @property
    def total(self) -> float:
        return self.attack + self.hold + self.release


def _smoothstep(x: float) -> float:
    """Smooth 0→1 ease (Hermite); flattens the ends so accents don't snap."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


class MotionPlugin(Plugin):
    """Execute robot expressions and fused head tracking."""

    name = "motion"

    # Motor preset definitions
    #   smoothing:       EMA factor per step (higher = faster response)
    #   deadband:        min angle change to send command (degrees)
    #   poll:            tracking loop interval (seconds)
    #   body_smoothing:  EMA factor for body rotation
    #   body_deadband:   min body angle change (degrees)
    #   max_step:        max degrees per step — caps angular velocity to protect motors
    #   body_max_step:   max body degrees per step
    MOTOR_PRESETS = {
        "sensitive": {
            "smoothing": 0.50, "deadband": 1.0, "poll": 0.03,
            "body_smoothing": 0.45, "body_deadband": 1.0,
            "max_step": 8.0, "body_max_step": 5.0,
            "idle_animations": True,
        },
        "moderate": {
            "smoothing": 0.35, "deadband": 2.0, "poll": 0.05,
            "body_smoothing": 0.35, "body_deadband": 2.0,
            "max_step": 5.0, "body_max_step": 3.0,
            "idle_animations": False,
        },
        "smart": {
            "smoothing": 0.15, "deadband": 3.0, "poll": 0.10,
            "body_smoothing": 0.15, "body_deadband": 4.0,
            "max_step": 3.0, "body_max_step": 2.0,
            "idle_animations": False,
        },
    }

    def __init__(self, app):
        super().__init__(app)
        # Motor enable/disable (sleep mode) — restore from persisted config
        self._motor_enabled = getattr(app.config, "motor_enabled", True)
        app.motor_enabled = self._motor_enabled
        self._motor_preset = getattr(app.config, "motor_preset", "moderate")
        if self._motor_preset not in self.MOTOR_PRESETS:
            self._motor_preset = "moderate"

        # Head tracking EMA state (Stewart platform — pitch/roll mirroring)
        self._current_yaw = 0.0
        self._current_pitch = 0.0
        self._current_roll = 0.0
        self._smoothing = app.config.motion_head_tracking_smoothing
        self._min_angle_change = 2.0  # degrees — head deadband
        self._max_step = 5.0  # max degrees per tracking step (motor protection)
        self._last_applied_yaw = 0.0
        self._last_applied_pitch = 0.0
        self._last_applied_roll = 0.0
        self._neutral_decay = 0.05

        # Body rotation EMA state (base Z-axis — horizontal tracking)
        self._current_body_yaw = 0.0
        self._last_applied_body_yaw = 0.0
        self._body_smoothing = 0.35  # body EMA
        self._body_min_angle = 2.0  # degrees — body deadband
        self._body_max_step = 3.0  # max body degrees per step

        # Speech wobble offsets (roll, pitch, yaw) set by HeadWobbler
        self._speech_roll = 0.0
        self._speech_pitch = 0.0
        self._speech_yaw = 0.0

        # Antenna animation state
        self._antenna_anim: AntennaAnimation | None = None
        self._antenna_anim_start: float = 0.0

        # ── Head compositor ──────────────────────────────────────────────
        # The head is a single-writer resource. Face tracking gives the gaze
        # anchor (where to point); emotion accents, speech wobble and idle
        # micro-motion are additive layers the loop sums on top, clamps into a
        # safe envelope, and emits as ONE target. Nothing writes the head
        # directly, so the layers compose instead of stomping each other.
        self._head_accent: _HeadAccent | None = None
        cfg = app.config
        # Emotion accents are *relative* — scale the EMOTION_MAP yaw/pitch/roll
        # down so they read as a gesture on top of gaze, not a reorientation.
        self._accent_gain = float(getattr(cfg, "motion_emotion_accent_gain", 0.6))
        # Safe head envelope (deg): composed output is clamped into this so
        # stacked layers can never drive the platform past a comfortable range.
        self._head_yaw_limit = float(getattr(cfg, "motion_head_yaw_limit", 25.0))
        self._head_pitch_limit = float(getattr(cfg, "motion_head_pitch_limit", 18.0))
        self._head_roll_limit = float(getattr(cfg, "motion_head_roll_limit", 18.0))
        # Idle micro-motion (subtle drift so a quiet head isn't frozen). Off by
        # default — opt in once the booth feel is dialed in.
        self._idle_micro = bool(getattr(cfg, "motion_head_idle_micro", False))

    def set_speech_offsets(self, offsets: tuple) -> None:
        """Called by HeadWobbler to set speech-driven head offsets."""
        self._speech_roll, self._speech_pitch, self._speech_yaw = offsets

    # ── Head compositor layers ───────────────────────────────────────────

    def _inject_head_accent(self, head) -> None:
        """Register an emotion's head pose as a transient additive accent.

        ``head`` is an emotion_mapper.HeadPose. Its yaw/pitch/roll become the
        accent amplitude (scaled by ``_accent_gain``); its duration sets the
        hold. The accent eases in, holds, then releases back to the gaze
        anchor, so the emotion reads as a head *gesture* riding on top of
        tracking rather than an absolute pose that overrides it.
        """
        dur = max(0.2, float(getattr(head, "duration", 0.6)))
        g = self._accent_gain
        self._head_accent = _HeadAccent(
            roll=head.roll * g,
            pitch=head.pitch * g,
            yaw=head.yaw * g,
            start=time.monotonic(),
            attack=min(0.25, dur * 0.35),
            hold=dur,
            release=max(0.4, dur * 0.8),
        )

    def _sample_head_accent(self, now: float) -> tuple[float, float, float]:
        """Current enveloped emotion-accent offset; clears the accent when done."""
        acc = self._head_accent
        if acc is None:
            return (0.0, 0.0, 0.0)
        t = now - acc.start
        if t >= acc.total:
            self._head_accent = None
            return (0.0, 0.0, 0.0)
        if t < acc.attack:
            env = _smoothstep(t / acc.attack) if acc.attack > 0 else 1.0
        elif t < acc.attack + acc.hold:
            env = 1.0
        else:
            rt = (t - acc.attack - acc.hold) / acc.release if acc.release > 0 else 1.0
            env = _smoothstep(1.0 - rt)
        return (acc.roll * env, acc.pitch * env, acc.yaw * env)

    def _sample_idle_micro(self, now: float, speaking: bool) -> tuple[float, float, float]:
        """Tiny low-frequency offsets so a quiet head looks alive, not frozen.

        Disabled while speaking or during an accent (those layers already carry
        the motion) and gated behind ``_idle_micro``. Uses detuned sines (no
        RNG) so it stays deterministic for tests.
        """
        if not self._idle_micro or speaking or self._head_accent is not None:
            return (0.0, 0.0, 0.0)
        roll = 0.3 * math.sin(now * 0.29 + 0.7)
        pitch = 0.6 * math.sin(now * 0.55) + 0.3 * math.sin(now * 0.21)
        yaw = 0.5 * math.sin(now * 0.37 + 1.3)
        return (roll, pitch, yaw)

    def set_motor_enabled(self, enabled: bool) -> None:
        """Enable or disable motor output (sleep mode)."""
        self._motor_enabled = enabled
        self.app.motor_enabled = enabled
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
        self._max_step = params["max_step"]
        self._body_smoothing = params["body_smoothing"]
        self._body_min_angle = params["body_deadband"]
        self._body_max_step = params["body_max_step"]
        # poll_interval is read each iteration from config, so update config
        self.app.config.motion_head_tracking_poll_interval = params["poll"]
        self.app.config.idle_animations = params.get("idle_animations", False)
        logger.info("Motor preset: %s (smoothing=%.2f, deadband=%.1f°, max_step=%.1f°, poll=%.3fs, idle=%s)",
                     preset, params["smoothing"], params["deadband"], params["max_step"], params["poll"],
                     self.app.config.idle_animations)

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
        """Drive the head compositor: gaze anchor + additive expression layers.

        Computes the gaze anchor (face/DOA tracking + body yaw), then sums the
        additive layers on top — emotion accents, speech wobble, idle
        micro-motion — clamps into the safe envelope, and emits ONE target.
        This is the sole writer of the head, so the layers compose instead of
        stomping each other.
        """
        logger.info("Head tracking fusion loop started")

        while self._running:
            poll_interval = self.app.config.motion_head_tracking_poll_interval

            if not self._motor_enabled:
                await asyncio.sleep(poll_interval)
                continue

            now = time.monotonic()
            speaking = self.app.is_speaking
            # Conversation-mode speech pauses face tracking (don't chase faces
            # mid sentence); monologue mode keeps tracking the listener.
            conv_speech = speaking and self.app.config.conversation_mode != "monologue"

            if conv_speech:
                # Anchor decays to neutral; the wobble/accent layers below keep
                # the head expressive while the gaze target is parked.
                self._current_yaw += self._neutral_decay * (0.0 - self._current_yaw)
                self._current_pitch += self._neutral_decay * (0.0 - self._current_pitch)
                self._current_roll += self._neutral_decay * (0.0 - self._current_roll)
                self._current_body_yaw += self._neutral_decay * (0.0 - self._current_body_yaw)
            else:
                target = self.app.head_targets.get_fused_target()
                if target.source == "none":
                    # Decay all axes to neutral
                    self._current_yaw += self._neutral_decay * (0.0 - self._current_yaw)
                    self._current_pitch += self._neutral_decay * (0.0 - self._current_pitch)
                    self._current_roll += self._neutral_decay * (0.0 - self._current_roll)
                    self._current_body_yaw += self._neutral_decay * (0.0 - self._current_body_yaw)
                else:
                    # Head (Stewart platform): pitch + roll mirroring
                    # EMA step with velocity clamping to protect motors
                    def _clamp_step(current, target_val, smoothing, max_step):
                        step = smoothing * (target_val - current)
                        if abs(step) > max_step:
                            step = max_step if step > 0 else -max_step
                        return current + step

                    self._current_yaw = _clamp_step(self._current_yaw, target.yaw, self._smoothing, self._max_step)
                    self._current_pitch = _clamp_step(self._current_pitch, target.pitch, self._smoothing, self._max_step)
                    self._current_roll = _clamp_step(self._current_roll, target.roll, self._smoothing, self._max_step)
                    # Body rotation: horizontal person tracking
                    self._current_body_yaw = _clamp_step(
                        self._current_body_yaw, target.body_yaw, self._body_smoothing, self._body_max_step
                    )

            # ── Compose additive layers on top of the gaze anchor ───────
            a_roll, a_pitch, a_yaw = self._sample_head_accent(now)
            if conv_speech:
                w_roll, w_pitch, w_yaw = self._speech_roll, self._speech_pitch, self._speech_yaw
            else:
                w_roll = w_pitch = w_yaw = 0.0
            i_roll, i_pitch, i_yaw = self._sample_idle_micro(now, speaking)

            # A transient layer is live → tighten the emit deadband so accents
            # and wobble stay smooth even below the tracking deadband.
            transient_active = (
                self._head_accent is not None
                or (conv_speech and (abs(w_roll) + abs(w_pitch) + abs(w_yaw)) > 0.1)
                or bool(i_roll or i_pitch or i_yaw)
            )

            def _clip(v, lim):
                return lim if v > lim else (-lim if v < -lim else v)

            comp_yaw = _clip(self._current_yaw + a_yaw + w_yaw + i_yaw, self._head_yaw_limit)
            comp_pitch = _clip(self._current_pitch + a_pitch + w_pitch + i_pitch, self._head_pitch_limit)
            comp_roll = _clip(self._current_roll + a_roll + w_roll + i_roll, self._head_roll_limit)
            comp_body = self._current_body_yaw

            # ── Single emit: head and body, deadband-gated ──────────────
            emit_deadband = 0.1 if transient_active else self._min_angle_change
            delta_yaw = abs(comp_yaw - self._last_applied_yaw)
            delta_pitch = abs(comp_pitch - self._last_applied_pitch)
            delta_roll = abs(comp_roll - self._last_applied_roll)

            if delta_yaw >= emit_deadband or delta_pitch >= emit_deadband or delta_roll >= emit_deadband:
                self._set_head_pose(comp_yaw, comp_pitch, comp_roll)
                self._last_applied_yaw = comp_yaw
                self._last_applied_pitch = comp_pitch
                self._last_applied_roll = comp_roll

            # Update body rotation if changed enough
            delta_body = abs(comp_body - self._last_applied_body_yaw)
            if delta_body >= self._body_min_angle:
                self._set_body_yaw(comp_body)
                self._last_applied_body_yaw = comp_body

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

        # Emotion head pose → additive accent on top of gaze. The compositor
        # owns the head; emotions never command an absolute pose here (that is
        # what fought face tracking and the speech wobbler). Antennas only.
        if expr.head:
            self._inject_head_accent(expr.head)

        try:
            kwargs = {}
            # Static antenna (only if no animation)
            if expr.antenna and not expr.antenna_anim:
                kwargs["antennas"] = [
                    np.radians(expr.antenna.right),
                    np.radians(expr.antenna.left),
                ]
                kwargs["duration"] = expr.antenna.duration

            if kwargs:
                reachy.goto_target(**kwargs)

            logger.debug(f"Executed: {expr.description}")
        except Exception as e:
            logger.error(f"Failed to execute expression: {e}")
