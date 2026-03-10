"""Tests for the robot command handlers in ConversationPlugin."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_claw.config import Config
from reachy_claw.plugins.conversation_plugin import ConversationPlugin


@pytest.fixture
def app_with_robot(mock_reachy):
    from reachy_claw.app import ReachyClawApp

    config = Config(
        standalone_mode=True,
        idle_animations=False,
        play_emotions=True,
        enable_face_tracker=False,
        enable_motion=False,
        tts_backend="none",
        stt_backend="whisper",
    )
    a = ReachyClawApp(config)
    a.reachy = mock_reachy
    return a


@pytest.fixture
def plugin(app_with_robot):
    return ConversationPlugin(app_with_robot)


# ── Passive emotion channel ──────────────────────────────────────


class TestEmotionChannel:
    @pytest.mark.asyncio
    async def test_on_emotion_queues_expression(self, plugin):
        await plugin._on_emotion("happy")
        expr = plugin.app.emotions.get_next_expression()
        assert expr is not None

    @pytest.mark.asyncio
    async def test_on_emotion_does_nothing_when_disabled(self, plugin):
        plugin.app.config.play_emotions = False
        await plugin._on_emotion("happy")
        expr = plugin.app.emotions.get_next_expression()
        assert expr is None

    @pytest.mark.asyncio
    async def test_on_emotion_handles_unknown(self, plugin):
        """Unknown emotions should not crash — EmotionMapper returns None."""
        await plugin._on_emotion("nonexistent")
        expr = plugin.app.emotions.get_next_expression()
        assert expr is None


# ── Active robot commands ─────────────────────────────────────────


class TestRobotCommandDispatch:
    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, plugin):
        result = plugin._execute_robot_command("nonexistent", {})
        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    @pytest.mark.asyncio
    async def test_on_robot_command_sends_result(self, plugin):
        plugin._client = MagicMock()
        plugin._client.send_robot_result = AsyncMock()

        await plugin._on_robot_command("status", {}, "cmd-123")

        plugin._client.send_robot_result.assert_called_once()
        args = plugin._client.send_robot_result.call_args
        assert args[0][0] == "cmd-123"
        assert isinstance(args[0][1], dict)


class TestCmdMoveHead:
    def test_success(self, plugin):
        with patch("reachy_mini.utils.create_head_pose", return_value=np.eye(4), create=True):
            result = plugin._cmd_move_head({"yaw": 15, "pitch": 10, "roll": 5, "duration": 0.5})

        assert result["status"] == "success"
        assert result["position"]["yaw"] == 15
        assert result["position"]["pitch"] == 10
        assert result["position"]["roll"] == 5
        plugin.app.reachy.goto_target.assert_called_once()

    def test_clamps_to_limits(self, plugin):
        with patch("reachy_mini.utils.create_head_pose", return_value=np.eye(4), create=True):
            result = plugin._cmd_move_head({"yaw": 100, "pitch": -50, "roll": 60})

        assert result["position"]["yaw"] == 45
        assert result["position"]["pitch"] == -30
        assert result["position"]["roll"] == 30

    def test_no_robot(self, plugin):
        plugin.app.reachy = None
        result = plugin._cmd_move_head({"yaw": 10})
        assert result["status"] == "error"
        assert "No robot" in result["message"]


class TestCmdMoveAntennas:
    def test_success(self, plugin):
        result = plugin._cmd_move_antennas({"left": 30, "right": 20, "duration": 0.5})

        assert result["status"] == "success"
        assert result["antennas"]["left"] == 30
        assert result["antennas"]["right"] == 20

        # Verify SDK called with [right, left] in radians
        call_args = plugin.app.reachy.goto_target.call_args
        antennas = call_args[1]["antennas"]
        assert abs(antennas[0] - np.radians(20)) < 1e-6  # right
        assert abs(antennas[1] - np.radians(30)) < 1e-6  # left

    def test_no_robot(self, plugin):
        plugin.app.reachy = None
        result = plugin._cmd_move_antennas({"left": 10})
        assert result["status"] == "error"


class TestCmdPlayEmotion:
    def test_success(self, plugin):
        result = plugin._cmd_play_emotion({"emotion": "happy"})
        assert result["status"] == "success"
        assert result["emotion"] == "happy"

    def test_missing_emotion(self, plugin):
        result = plugin._cmd_play_emotion({})
        assert result["status"] == "error"
        assert "Missing" in result["message"]


class TestCmdDance:
    @patch("reachy_claw.plugins.conversation_plugin.time", create=True)
    def test_success(self, mock_time_module, plugin):
        # We need to patch time.sleep inside the module
        import reachy_claw.plugins.conversation_plugin as conv_mod
        original_sleep = None

        with patch("reachy_mini.utils.create_head_pose", return_value=np.eye(4), create=True), \
             patch.object(conv_mod, "_time", create=True):
            # The _cmd_dance imports time as _time locally — patch at call site
            import time as real_time
            with patch.object(real_time, "sleep"):
                result = plugin._cmd_dance({"dance_name": "nod"})

        assert result["status"] == "success"
        assert result["dance"] == "nod"
        assert result["steps"] > 0

    def test_unknown_dance(self, plugin):
        result = plugin._cmd_dance({"dance_name": "nonexistent"})
        assert result["status"] == "error"
        assert "Unknown dance" in result["message"]

    def test_no_robot(self, plugin):
        plugin.app.reachy = None
        result = plugin._cmd_dance({"dance_name": "nod"})
        assert result["status"] == "error"


class TestCmdCaptureImage:
    def test_no_robot(self, plugin):
        plugin.app.reachy = None
        result = plugin._cmd_capture_image({})
        assert result["status"] == "error"

    def test_no_media(self, plugin):
        plugin.app.reachy.media = None
        result = plugin._cmd_capture_image({})
        assert result["status"] == "error"
        assert "Media" in result["message"]

    def test_no_frame(self, plugin):
        plugin.app.reachy.media.get_frame.return_value = None
        result = plugin._cmd_capture_image({})
        assert result["status"] == "error"
        assert "No frame" in result["message"]


class TestCmdStatus:
    def test_connected(self, plugin):
        result = plugin._cmd_status({})
        assert result["connected"] is True
        assert "head_pose" in result

    def test_not_connected(self, plugin):
        plugin.app.reachy = None
        result = plugin._cmd_status({})
        assert result["connected"] is False
        assert "head_pose" not in result
