"""ReachyMotionToolsPlugin — registers the 4 REAL Reachy motion tools into the
ovs_agent client-loop tool registry.

This is the THIN, app-specific slice of the reachy_voice client-loop migration:
ovs_agent owns the conversation loop, the ``tool_registry``, ``dispatch`` and the
client-loop ``runner.stream_with_tools``. All we add here are the 4 motion tool
BODIES, which dispatch into the single-writer ``MotionController`` compositor.

Pattern mirrors:
  * ``ovs_agent/apps/companion_robot/demo_tools.py`` — the canonical companion
    tool shape (move_head / play_emotion), but with REAL bodies replacing mocks.
  * ``ovs_agent/plugins/actuator_actions.ArmPlugin`` — register tools onto
    ``self.app.tool_registry`` in ``setup()`` (SYNC), no long-running loop.

Lifecycle (ovs_agent.plugin.Plugin contract):
  setup()  (SYNC, returns bool):
    register move_head / move_antennas / play_emotion / dance onto
    ``self.app.tool_registry``. Returns False (tools disabled) if no
    MotionController was supplied.
  start()/stop() (ASYNC): no-op — the SDK handle + compositor thread belong to
    the ConversationEngine / MotionController, not this plugin.

Why route through the compositor (and NOT call the SDK directly): the
``MotionController`` is a single-writer compositor (one motion thread). Tool
handlers run OFF that thread (``asyncio.to_thread``), so they call the
``command_*`` / ``play_dance`` enqueue methods, which hand the SDK work to the
compositor thread — preserving the single-writer invariant and not fighting
presence / gaze / emotion. ``play_emotion`` reuses the existing debounced
emotion path. See spec Section 5 (Compositor contention).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ovs_agent.plugin import Plugin

logger = logging.getLogger("reachy_voice.plugins.motion_tools")


class ReachyMotionToolsPlugin(Plugin):
    name = "reachy_motion_tools"

    def __init__(self, app, motion=None, config: dict | None = None) -> None:  # noqa: ANN001
        super().__init__(app)
        self.cfg = dict(config or {})
        self._motion = motion
        self._registered_tool_names: list[str] = []

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:  # SYNC (per Plugin.setup contract)
        motion = self._motion
        if motion is None:
            logger.error(
                "ReachyMotionToolsPlugin: no MotionController supplied; motion "
                "tools disabled. Pass motion=... at construction."
            )
            return False
        self._register_tools(self.app.tool_registry)
        logger.info(
            "ReachyMotionToolsPlugin tools registered: %s",
            self._registered_tool_names,
        )
        return True

    def _register_tools(self, registry) -> None:  # noqa: ANN001
        motion = self._motion
        assert motion is not None

        @registry.tool(
            name="move_head",
            description=(
                "Point the robot's head at a target orientation, in DEGREES. "
                "yaw = left/right (positive = left, negative = right, range "
                "-45..45), pitch = up/down (positive = up, range -30..30), "
                "roll = tilt (range -30..30). Call this when the user asks the "
                "robot to look in a direction, e.g. '向左看' (look left), "
                "'look up', '看右边'."
            ),
            preamble_text="好的。",
        )
        async def move_head(yaw: float, pitch: float,
                            roll: float = 0.0) -> dict[str, Any]:
            return await asyncio.to_thread(motion.command_head, yaw, pitch, roll)

        @registry.tool(
            name="move_antennas",
            description=(
                "Move the robot's two antennae to target angles in DEGREES "
                "(positive = up). left and right are the two antennae. Call "
                "this for antenna-specific gestures, e.g. '抬起天线' (raise "
                "antennas), 'wiggle your ears'."
            ),
            preamble_text="好的。",
        )
        async def move_antennas(left: float, right: float) -> dict[str, Any]:
            return await asyncio.to_thread(motion.command_antennas, left, right)

        @registry.tool(
            name="play_emotion",
            description=(
                "Play an emotion expression on the robot (head pose + "
                "antennae, via the official recorded-move library). emotion is "
                "a slug: 'happy', 'sad', 'curious', 'excited', 'thinking', "
                "'confused', 'surprised', 'angry', 'neutral', 'welcoming'. Call "
                "this when the user asks the robot to express a feeling, e.g. "
                "'开心一点' (be happy), 'look sad'."
            ),
            preamble_text="好的。",
        )
        async def play_emotion(emotion: str) -> dict[str, Any]:
            # play_emotion enqueues onto the compositor's debounced emotion
            # slot and returns None; surface an ack the LLM can read.
            motion.play_emotion(emotion)
            # Reflect on the shared CompanionRobotApp slot for any observer.
            try:
                self.app.current_emotion = emotion
            except Exception:  # noqa: BLE001 — defensive, never break the tool
                pass
            return {"ok": True, "emotion": emotion}

        @registry.tool(
            name="dance",
            description=(
                "Perform a short choreographed dance routine. dance_name is "
                "one of: 'simple_nod', 'yeah_nod', 'uh_huh_tilt', "
                "'head_tilt_roll', 'side_to_side_sway', 'side_glance_flick', "
                "'chin_lead'. Call this when the user asks the robot to dance, "
                "e.g. '跳个舞' (do a dance)."
            ),
            preamble_text="好的，我跳个舞。",
            timeout_s=20.0,  # dances can exceed the default 10s dispatch timeout
        )
        async def dance(dance_name: str) -> dict[str, Any]:
            return await asyncio.to_thread(motion.play_dance, dance_name)

        self._registered_tool_names = [
            "move_head", "move_antennas", "play_emotion", "dance",
        ]

    async def stop(self) -> None:
        # Unregister our tools so re-instantiating the app (tests, restarts)
        # doesn't accumulate stale closures on the shared default_registry.
        registry = getattr(self.app, "tool_registry", None)
        if registry is not None:
            for tname in self._registered_tool_names:
                try:
                    registry.unregister(tname)
                except Exception:  # noqa: BLE001
                    pass
        await super().stop()


__all__ = ["ReachyMotionToolsPlugin"]
