"""ReachyClawClientLoopApp — embodied Reachy voice agent on the
CLIENT-LOOP tool runner, mirroring ``apps/voice_arm/app.py``.

Structure parallels VoiceArmApp:
  * subclass the robot base (here ``CompanionRobotApp`` instead of
    ``MultiModeApp``, since Reachy is an embodied companion),
  * pull our config blocks out of ``config.metadata``,
  * ``self.register(ReachyToolsPlugin(...))`` so the move_head /
    play_emotion tools land in ``app.tool_registry`` before any turn.

The tool-calling LLM <-> tool loop is driven CLIENT-SIDE by ovs_agent's
``tools/runner.stream_with_tools`` (invoked from ``app_mode.py`` when
``tools_enabled`` + ``server_loop`` is false). The engine stays a plain
ASR/TTS pass-through. This is fundamentally different from the
server-loop ``tool_advertise`` / ``SERVER_TOOL_CALL`` path.

The CLI loader resolves ``apps.<name>.app:App``; we are loaded by an
explicit ``--config`` path instead (ovs-agent run reachy_clientloop
won't find us under ovs_agent/apps), so a small launcher in
``proof_clientloop.py`` constructs this class directly.
"""
from __future__ import annotations

import logging

from ovs_agent.apps.companion_robot.app import CompanionRobotApp

from reachy_claw.clientloop.motion_plugin import ReachyMotionPlugin
from reachy_claw.clientloop.tools_plugin import ReachyToolsPlugin

logger = logging.getLogger(__name__)


class ReachyClawClientLoopApp(CompanionRobotApp):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)

        meta = getattr(config, "metadata", {}) or {}
        reachy_cfg = dict(meta.get("reachy", {}) or {})

        # ReachyMotionPlugin owns the SDK handle (sets app.reachy) + the
        # motion calculators. Register it FIRST so its setup() connects the
        # robot before the tools plugin resolves it. Mirrors
        # VoiceArmApp.register(ArmPlugin) (the actuator-owning plugin).
        logger.info("ReachyClawClientLoopApp: registering ReachyMotionPlugin")
        self._motion_plugin = ReachyMotionPlugin(self, reachy_cfg)
        self.register(self._motion_plugin)

        # Register the Reachy tools plugin so the move_head / move_antennas
        # / play_emotion / dance tools are advertised on the very first
        # user utterance. It dispatches into the motion plugin above.
        logger.info("ReachyClawClientLoopApp: registering ReachyToolsPlugin")
        self.register(ReachyToolsPlugin(self, reachy_cfg, motion=self._motion_plugin))


# CLI loader expects an ``App`` symbol at module top level.
App = ReachyClawClientLoopApp


__all__ = ["ReachyClawClientLoopApp", "App"]
