"""ReachyToolsPlugin — registers REAL Reachy companion tools into the
ovs_agent tool registry, mirroring ``plugins/actuator_actions.ArmPlugin``.

Lifecycle (ovs_agent.plugin.Plugin contract):
  setup()  (SYNC, returns bool):
    register the companion tools onto ``self.app.tool_registry``:
      * move_head(yaw, pitch, roll=0.0)
      * move_antennas(left, right)
      * play_emotion(emotion)
      * dance(dance_name)
  start()/stop() (ASYNC): no-op (the SDK handle + motion loops belong to
    ReachyMotionPlugin).

The handlers are REAL: each dispatches into the ReachyMotionPlugin, which
owns the SDK handle and the ported motion math (head pose / antenna /
emotion / dance). The blocking ``reachy.goto_target`` SDK call runs off
the event loop via ``asyncio.to_thread`` so a slow daemon can't freeze
the voice loop. If no robot is connected the motion plugin returns a
structured ``{'ok': False, 'reason': 'no_robot'}`` ack — the LLM hears a
clean failure rather than an exception.

We register via the registry's public ``registry.tool(...)`` decorator
(the same API ``register_arm_tools`` calls under the hood) because the
arm helper forces one-tool-per-action no-arg closures, which don't fit
typed head/emotion/dance args.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ovs_agent.plugin import Plugin

from reachy_claw.clientloop.motion_plugin import ReachyMotionPlugin

logger = logging.getLogger(__name__)


class ReachyToolsPlugin(Plugin):
    name = "reachy_tools"

    def __init__(self, app, config: dict | None = None,
                 motion: ReachyMotionPlugin | None = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        # The motion plugin owns the SDK + calculators. The app wires it in
        # explicitly (so both plugins share one handle); if not provided we
        # locate it on the app's registered plugin list at setup() time.
        self._motion = motion
        self._registered_tool_names: list[str] = []

    def _resolve_motion(self) -> ReachyMotionPlugin | None:
        if self._motion is not None:
            return self._motion
        for p in getattr(self.app, "plugins", []) or []:
            if isinstance(p, ReachyMotionPlugin):
                self._motion = p
                return p
        return None

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:  # SYNC (per Plugin.setup contract)
        motion = self._resolve_motion()
        if motion is None:
            logger.error(
                "ReachyToolsPlugin: no ReachyMotionPlugin found; tools "
                "disabled. Register ReachyMotionPlugin before this plugin."
            )
            return False
        self._motion = motion
        self._register_tools(self.app.tool_registry)
        logger.info(
            "ReachyToolsPlugin tools registered: %s",
            self._registered_tool_names,
        )
        return True

    def _register_tools(self, registry) -> None:
        motion = self._motion
        assert motion is not None

        @registry.tool(
            name="move_head",
            description=(
                "Point the robot's head at a target orientation, in "
                "DEGREES. yaw = left/right (positive = left, negative = "
                "right, range -45..45), pitch = up/down (positive = up, "
                "range -30..30), roll = tilt (range -30..30). Call this "
                "when the user asks the robot to look in a direction, e.g. "
                "'向左看' (look left), 'look up', '看右边'."
            ),
            preamble_text="好的。",
        )
        async def move_head(yaw: float, pitch: float,
                            roll: float = 0.0) -> dict[str, Any]:
            logger.info(
                "[reachy_tools] move_head yaw=%.1f pitch=%.1f roll=%.1f",
                yaw, pitch, roll,
            )
            return await asyncio.to_thread(motion.move_head, yaw, pitch, roll)

        @registry.tool(
            name="move_antennas",
            description=(
                "Move the robot's two antennae to target angles in "
                "DEGREES (positive = up). left and right are the two "
                "antennae. Call this for antenna-specific gestures, e.g. "
                "'抬起天线' (raise antennas), 'wiggle your ears'."
            ),
            preamble_text="好的。",
        )
        async def move_antennas(left: float, right: float) -> dict[str, Any]:
            logger.info(
                "[reachy_tools] move_antennas left=%.1f right=%.1f", left, right
            )
            return await asyncio.to_thread(motion.move_antennas, left, right)

        @registry.tool(
            name="play_emotion",
            description=(
                "Play an emotion expression on the robot (head pose + "
                "antennae). emotion is a slug: 'happy', 'sad', 'curious', "
                "'excited', 'thinking', 'confused', 'surprised', 'angry', "
                "'neutral'. Call this when the user asks the robot to "
                "express a feeling, e.g. '开心一点' (be happy), 'look sad'."
            ),
            preamble_text="好的。",
        )
        async def play_emotion(emotion: str) -> dict[str, Any]:
            logger.info("[reachy_tools] play_emotion emotion=%r", emotion)
            return await asyncio.to_thread(motion.play_emotion, emotion)

        @registry.tool(
            name="dance",
            description=(
                "Perform a short choreographed dance routine. dance_name "
                "is one of: 'celebrate', 'curious_look', 'lobster', 'nod', "
                "'wiggle'. Call this when the user asks the robot to dance, "
                "e.g. '跳个舞' (do a dance)."
            ),
            preamble_text="好的，我跳个舞。",
        )
        async def dance(dance_name: str) -> dict[str, Any]:
            logger.info("[reachy_tools] dance dance_name=%r", dance_name)
            return await asyncio.to_thread(motion.dance, dance_name)

        self._registered_tool_names = [
            "move_head", "move_antennas", "play_emotion", "dance",
        ]


__all__ = ["ReachyToolsPlugin"]
