"""Tests for the MotionPlugin (expression execution + head tracking)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_claw.motion.emotion_mapper import (
    AntennaMotion,
    HeadPose,
    RobotExpression,
)
from reachy_claw.motion.head_target import HeadTarget
from reachy_claw.plugins.motion_plugin import MotionPlugin

# Patch target for the lazy import inside motion_plugin methods
_CHP = "reachy_mini.utils.create_head_pose"


# ── Expression execution ───────────────────────────────────────────────


class TestExecuteExpression:
    def test_expression_with_head_and_antenna(self, app, mock_reachy):
        plugin = MotionPlugin(app)
        expr = RobotExpression(
            head=HeadPose(yaw=10, pitch=5, roll=3, duration=0.5),
            antenna=AntennaMotion(left=20, right=-20, duration=0.3),
            description="test expression",
        )

        with patch(_CHP, return_value=np.eye(4)):
            plugin._execute_expression(expr)

        mock_reachy.goto_target.assert_called_once()
        call_kwargs = mock_reachy.goto_target.call_args.kwargs
        assert call_kwargs["duration"] == 0.5
        assert "antennas" in call_kwargs

    def test_expression_head_only(self, app, mock_reachy):
        plugin = MotionPlugin(app)
        expr = RobotExpression(
            head=HeadPose(yaw=10, duration=0.5),
            description="head only",
        )

        with patch(_CHP, return_value=np.eye(4)):
            plugin._execute_expression(expr)

        mock_reachy.goto_target.assert_called_once()
        call_kwargs = mock_reachy.goto_target.call_args.kwargs
        assert "antennas" not in call_kwargs

    def test_expression_antenna_only(self, app, mock_reachy):
        plugin = MotionPlugin(app)
        expr = RobotExpression(
            antenna=AntennaMotion(left=30, right=-30, duration=0.4),
            description="antenna only",
        )

        with patch(_CHP, return_value=np.eye(4)):
            plugin._execute_expression(expr)

        mock_reachy.goto_target.assert_called_once()
        call_kwargs = mock_reachy.goto_target.call_args.kwargs
        assert "antennas" in call_kwargs
        assert call_kwargs["duration"] == 0.4

    def test_expression_no_robot_logs_sim(self, app, caplog):
        app.reachy = None
        plugin = MotionPlugin(app)
        expr = RobotExpression(description="sim test")

        import logging

        with caplog.at_level(logging.INFO):
            plugin._execute_expression(expr)

        assert "[SIM]" in caplog.text


# ── Motion loop (emotion queue + idle) ─────────────────────────────────


class TestMotionLoop:
    @pytest.mark.asyncio
    async def test_processes_queued_emotion(self, app, mock_reachy):
        plugin = MotionPlugin(app)
        app.register(plugin)

        # Queue an emotion
        app.emotions.queue_emotion("surprised")

        # Run motion loop briefly
        plugin._running = True
        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._motion_loop())
            await asyncio.sleep(0.15)
            plugin._running = False
            await task

        # Should have executed the expression
        assert mock_reachy.goto_target.called

    @pytest.mark.asyncio
    async def test_idle_animation_fires_after_interval(self, app, mock_reachy):
        app.config.idle_animations = True
        app.config.motion_idle_animation_interval = 0.05  # very short
        app.is_speaking = False

        plugin = MotionPlugin(app)
        plugin._running = True

        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._motion_loop())
            await asyncio.sleep(0.2)
            plugin._running = False
            await task

        # Idle animation should have fired at least once
        assert mock_reachy.goto_target.called

    @pytest.mark.asyncio
    async def test_no_idle_during_speaking(self, app, mock_reachy):
        app.config.idle_animations = True
        app.config.motion_idle_animation_interval = 0.05
        app.is_speaking = True  # Speaking blocks idle

        plugin = MotionPlugin(app)
        plugin._running = True

        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._motion_loop())
            await asyncio.sleep(0.15)
            plugin._running = False
            await task

        # Should NOT have moved (speaking blocks idle + no queued emotions)
        assert not mock_reachy.goto_target.called


# ── Head tracking loop ─────────────────────────────────────────────────


class TestHeadTrackingLoop:
    @pytest.mark.asyncio
    async def test_tracks_face_target(self, app, mock_reachy):
        app.config.motion_head_tracking_poll_interval = 0.02
        plugin = MotionPlugin(app)
        plugin._running = True

        # Publish a strong face target
        app.head_targets.publish(
            HeadTarget(yaw=20.0, pitch=-10.0, confidence=0.9, source="face")
        )

        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._head_tracking_loop())
            await asyncio.sleep(0.15)
            plugin._running = False
            await task

        # Should have called set_target_head_pose at least once
        assert mock_reachy.set_target_head_pose.called

    @pytest.mark.asyncio
    async def test_freezes_during_speaking(self, app, mock_reachy):
        app.config.motion_head_tracking_poll_interval = 0.02
        app.is_speaking = True

        plugin = MotionPlugin(app)
        plugin._running = True

        app.head_targets.publish(
            HeadTarget(yaw=20.0, confidence=0.9, source="face")
        )

        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._head_tracking_loop())
            await asyncio.sleep(0.1)
            plugin._running = False
            await task

        # Head tracking frozen during speech
        assert not mock_reachy.set_target_head_pose.called

    @pytest.mark.asyncio
    async def test_decays_to_neutral_without_data(self, app, mock_reachy):
        app.config.motion_head_tracking_poll_interval = 0.02
        plugin = MotionPlugin(app)
        plugin._running = True
        # Set initial position away from neutral
        plugin._current_yaw = 20.0
        plugin._last_applied_yaw = 20.0

        with patch(_CHP, return_value=np.eye(4)):
            task = asyncio.create_task(plugin._head_tracking_loop())
            await asyncio.sleep(0.2)
            plugin._running = False
            await task

        # Should have decayed toward 0
        assert abs(plugin._current_yaw) < 20.0


# ── Speech wobble offsets ──────────────────────────────────────────────


class TestSpeechOffsets:
    def test_set_speech_offsets(self, app):
        plugin = MotionPlugin(app)
        plugin.set_speech_offsets((2.0, 1.5, 0.5))
        assert plugin._speech_roll == 2.0
        assert plugin._speech_pitch == 1.5
        assert plugin._speech_yaw == 0.5

    def test_apply_speech_wobble_no_robot(self, app):
        app.reachy = None
        plugin = MotionPlugin(app)
        plugin.set_speech_offsets((5.0, 3.0, 1.0))
        # Should not raise
        plugin._apply_speech_wobble()

    def test_apply_speech_wobble_small_offset_ignored(self, app, mock_reachy):
        plugin = MotionPlugin(app)
        plugin.set_speech_offsets((0.01, 0.01, 0.01))
        plugin._apply_speech_wobble()
        # Too small, should not call set_target_head_pose
        assert not mock_reachy.set_target_head_pose.called
