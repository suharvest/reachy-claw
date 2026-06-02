"""ReachyMotionPlugin — owns the Reachy SDK handle + the motion/emotion
calculators, exposing REAL motion methods the tools plugin dispatches into.

Mirrors ``plugins/actuator_actions.ArmPlugin`` from voice_arm:

  setup()  (SYNC):
    * connect the Reachy SDK (config-driven host/port; Mac sim vs Jetson
      :38001) and stash the handle on ``app.reachy`` so the rest of the
      CompanionRobotApp surface (and any future vision/face plugin) can
      read it. Degrade GRACEFULLY if no robot is reachable — the app
      still boots and the tools return a structured ``no_robot`` ack.
    * build the EmotionMapper / dance tables.
  start()/stop() (ASYNC): no-op (no long-running motor loop in this
    slice; motion is driven synchronously per tool call).

Why a dedicated plugin (not the existing ``MotionPlugin``): the existing
``reachy_claw.plugins.motion_plugin.MotionPlugin`` is bound to
``ReachyClawApp`` (its own EventBus, ``app.emotions``, head-tracking
loops, vision fusion). The client-loop app is a ``CompanionRobotApp``
(ovs_agent BaseApp) with a different surface. Rather than edit the
top-level plugin (forbidden — additive only), this plugin REUSES the
exact motion MATH by importing the calculators:

  * ``motion.emotion_mapper.EmotionMapper`` — emotion → RobotExpression
  * ``motion.dances.DANCE_ROUTINES`` — dance step sequences
  * ``reachy_mini.utils.create_head_pose`` — the SDK pose builder

and ports the handler BODIES verbatim from
``plugins/conversation_plugin.py`` (_cmd_move_head / _cmd_move_antennas /
_cmd_play_emotion / _cmd_dance) so the head/antenna math is identical to
the proven on-device path. No math is rewritten here.
"""
from __future__ import annotations

import logging
import sys
import time as _time
from typing import Any

import numpy as np

from ovs_agent.plugin import Plugin

# REUSE reachy's existing motion calculators (import, do not rewrite).
from reachy_claw.motion.dances import AVAILABLE_DANCES, DANCE_ROUTINES
from reachy_claw.motion.emotion_mapper import EmotionMapper

logger = logging.getLogger(__name__)


class ReachyMotionPlugin(Plugin):
    """Owns the Reachy SDK handle + motion calculators; exposes real
    ``move_head`` / ``move_antennas`` / ``play_emotion`` / ``dance``
    methods. The tools plugin dispatches into these."""

    name = "reachy_motion"

    def __init__(self, app, config: dict | None = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        # EmotionMapper is the SAME calculator the on-device MotionPlugin
        # uses; we drain its queue synchronously inside play_emotion.
        intensity = float(self.cfg.get("emotion_intensity", 0.7))
        self.emotions = EmotionMapper(intensity=intensity)
        self._owns_reachy = False

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:  # SYNC (per Plugin.setup contract)
        """Connect the SDK (graceful no-robot fallback) and stash the
        handle on ``app.reachy``. Always returns True so the app boots
        even with no robot — the tools degrade to a structured ack."""
        self._connect_robot()
        if self.app.reachy is None:
            logger.warning(
                "ReachyMotionPlugin: no robot connected — motion tools "
                "will return {'ok': False, 'reason': 'no_robot'} acks. "
                "App boots regardless (graceful degrade)."
            )
        else:
            logger.info("ReachyMotionPlugin: robot connected, motion live")
        return True

    def _connect_robot(self) -> None:
        """Connect to the Reachy daemon. Config-driven host/port.

        Connection knobs (from ``metadata.reachy`` in config.yaml):
          * ``daemon_port`` — daemon FastAPI port. Mac sim uses the SDK
            default (8000); Jetson daemon listens on 38001.
          * ``connection_mode`` — "auto" | "localhost_only" | "network".
          * ``media_backend`` — forced to "no_media" on macOS (no gi).

        Mirrors ``reachy_claw.app.ReachyClawApp.connect_robot`` but
        WITHOUT the auto-spawn / reconnect machinery (the client-loop
        slice expects an already-running daemon — sim on Mac, or the
        on-device daemon on the Jetson)."""
        if self.app.reachy is not None:
            # Some other plugin (or the daemon harness) already supplied a
            # handle — respect it, don't double-connect.
            logger.info("ReachyMotionPlugin: app.reachy already set, reusing")
            return

        try:
            from reachy_mini import ReachyMini
        except ImportError:
            logger.warning("reachy-mini not installed; running without robot")
            self.app.reachy = None
            return

        kwargs: dict[str, Any] = {}
        conn_mode = self.cfg.get("connection_mode", "auto")
        if conn_mode != "auto":
            kwargs["connection_mode"] = conn_mode
        port = int(self.cfg.get("daemon_port", 8000))
        if port != 8000:
            kwargs["port"] = port
        media_backend = self.cfg.get("media_backend", "default")
        if media_backend != "default":
            kwargs["media_backend"] = media_backend
        # macOS lacks GStreamer/gi — force no_media so the SDK doesn't crash.
        if sys.platform == "darwin" and "media_backend" not in kwargs:
            kwargs["media_backend"] = "no_media"
        timeout = float(self.cfg.get("connect_timeout_s", 5.0))
        kwargs["timeout"] = timeout

        host = self.cfg.get("host", "localhost")
        logger.info(
            "ReachyMotionPlugin: connecting SDK host=%s port=%d mode=%s "
            "media=%s timeout=%.1fs",
            host, port, conn_mode, kwargs.get("media_backend", "default"),
            timeout,
        )
        try:
            reachy = ReachyMini(**kwargs)
            reachy.__enter__()
            try:
                reachy.enable_motors()
            except Exception:  # noqa: BLE001
                logger.debug("enable_motors() failed (sim may not need it)")
            self.app.reachy = reachy
            self._owns_reachy = True
            logger.info("ReachyMotionPlugin: connected to Reachy daemon")
        except Exception as e:  # noqa: BLE001 — graceful degrade on any failure
            logger.warning(
                "ReachyMotionPlugin: SDK connect failed (%s); no-robot mode", e
            )
            self.app.reachy = None

    async def stop(self) -> None:
        await super().stop()
        if self._owns_reachy and self.app.reachy is not None:
            try:
                self.app.reachy.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self.app.reachy = None
            self._owns_reachy = False

    # ── real motion methods (bodies ported from conversation_plugin) ──
    #
    # These run the BLOCKING SDK call (reachy.goto_target) directly. The
    # tools plugin wraps them in asyncio.to_thread so a slow daemon can't
    # freeze the event loop. Each returns a JSON-serialisable ack dict.

    def move_head(self, yaw: float, pitch: float, roll: float = 0.0,
                  duration: float = 1.0) -> dict[str, Any]:
        """Port of conversation_plugin._cmd_move_head. yaw/pitch/roll in
        DEGREES (clamped to the on-device safe ranges)."""
        reachy = self.app.reachy
        if not reachy:
            return {"ok": False, "reason": "no_robot",
                    "requested": {"yaw": yaw, "pitch": pitch, "roll": roll}}
        from reachy_mini.utils import create_head_pose

        yaw = max(-45, min(45, yaw))
        pitch = max(-30, min(30, pitch))
        roll = max(-30, min(30, roll))
        pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
        reachy.goto_target(head=pose, duration=duration)
        return {"ok": True, "moved_to": {"yaw": yaw, "pitch": pitch, "roll": roll}}

    def move_antennas(self, left: float, right: float,
                      duration: float = 0.5) -> dict[str, Any]:
        """Port of conversation_plugin._cmd_move_antennas. left/right in
        DEGREES. SDK antenna order is [right, left] (radians)."""
        reachy = self.app.reachy
        if not reachy:
            return {"ok": False, "reason": "no_robot",
                    "requested": {"left": left, "right": right}}
        reachy.goto_target(
            antennas=[np.radians(right), np.radians(left)],
            duration=duration,
        )
        return {"ok": True, "antennas": {"left": left, "right": right}}

    def play_emotion(self, emotion: str) -> dict[str, Any]:
        """Port of conversation_plugin._cmd_play_emotion, but RENDERED
        synchronously here (the client-loop app has no MotionPlugin
        draining ``app.emotions``). Maps the emotion via the SAME
        EmotionMapper calculator, then drives the head + antennas through
        the SDK exactly like MotionPlugin._execute_expression does."""
        if not emotion:
            return {"ok": False, "reason": "missing_emotion"}
        expr = self.emotions.map_emotion(emotion)
        if expr is None:
            return {"ok": False, "reason": "unknown_emotion", "emotion": emotion}

        reachy = self.app.reachy
        # Reflect the expressed emotion on the shared CompanionRobotApp slot.
        self.app.current_emotion = emotion
        if not reachy:
            return {"ok": False, "reason": "no_robot", "emotion": emotion,
                    "expression": expr.description}

        from reachy_mini.utils import create_head_pose

        kwargs: dict[str, Any] = {}
        if expr.head:
            kwargs["head"] = create_head_pose(
                yaw=expr.head.yaw, pitch=expr.head.pitch,
                roll=expr.head.roll, degrees=True,
            )
            kwargs["duration"] = expr.head.duration
        # Render the antenna pose. For animated emotions we apply the
        # animation's CENTER as a static target (the client-loop slice has
        # no per-frame antenna oscillation loop; the center captures the
        # emotion's resting antenna posture). Static antenna emotions use
        # their explicit left/right.
        if expr.antenna_anim:
            c = expr.antenna_anim.center
            kwargs["antennas"] = [np.radians(c), np.radians(c)]
            kwargs.setdefault("duration", expr.antenna_anim.duration)
        elif expr.antenna:
            kwargs["antennas"] = [
                np.radians(expr.antenna.right),
                np.radians(expr.antenna.left),
            ]
            kwargs.setdefault("duration", expr.antenna.duration)

        if kwargs:
            reachy.goto_target(**kwargs)
        return {"ok": True, "emotion": emotion, "expression": expr.description}

    def dance(self, dance_name: str) -> dict[str, Any]:
        """Port of conversation_plugin._cmd_dance. Reuses DANCE_ROUTINES."""
        reachy = self.app.reachy
        routine = DANCE_ROUTINES.get(dance_name)
        if not routine:
            return {"ok": False, "reason": "unknown_dance",
                    "dance": dance_name, "available": AVAILABLE_DANCES}
        if not reachy:
            return {"ok": False, "reason": "no_robot", "dance": dance_name,
                    "steps": len(routine.steps)}

        from reachy_mini.utils import create_head_pose

        for step in routine.steps:
            pose = create_head_pose(
                yaw=step.yaw, pitch=step.pitch, roll=step.roll, degrees=True,
            )
            antennas = [np.radians(step.antenna_right), np.radians(step.antenna_left)]
            reachy.goto_target(head=pose, antennas=antennas, duration=step.duration)
            _time.sleep(step.duration)
        return {"ok": True, "dance": dance_name, "steps": len(routine.steps)}


__all__ = ["ReachyMotionPlugin"]
