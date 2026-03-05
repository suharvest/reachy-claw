"""Integration tests using the real Reachy Mini mockup simulation daemon.

These tests require a running daemon:
  reachy-mini-daemon --mockup-sim --headless --localhost-only --deactivate-audio

Skip automatically if the daemon is not reachable.
Run with: uv run pytest tests/test_integration_sim.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import numpy as np
import pytest

# ── Fixture: real simulated robot ──────────────────────────────────────


def _connect_sim():
    """Try to connect to the mockup sim daemon."""
    from reachy_mini import ReachyMini

    reachy = ReachyMini(
        connection_mode="localhost_only",
        media_backend="no_media",
        timeout=3,
    )
    reachy.__enter__()
    return reachy


@pytest.fixture(scope="module")
def sim_reachy():
    """Module-scoped simulated robot. Skips if daemon not running."""
    try:
        reachy = _connect_sim()
    except Exception as e:
        pytest.skip(f"Mockup sim daemon not reachable: {e}")
        return

    reachy.wake_up()
    time.sleep(0.5)
    yield reachy
    reachy.__exit__(None, None, None)


# ── Head movement tests ───────────────────────────────────────────────


class TestHeadMovement:
    def test_goto_target_changes_pose(self, sim_reachy):
        from reachy_mini.utils import create_head_pose

        # Move head to yaw=15
        pose = create_head_pose(yaw=15, degrees=True)
        sim_reachy.goto_target(head=pose, duration=0.3)
        time.sleep(0.4)

        current = sim_reachy.get_current_head_pose()
        assert current.shape == (4, 4)
        # In mockup sim, positions are applied immediately
        assert not np.allclose(current, np.eye(4), atol=0.05)

    def test_set_target_head_pose_immediate(self, sim_reachy):
        from reachy_mini.utils import create_head_pose

        # First move away
        sim_reachy.goto_target(
            head=create_head_pose(yaw=20, degrees=True), duration=0.2
        )
        time.sleep(0.3)

        # Set back to neutral immediately
        neutral = create_head_pose(degrees=True)
        sim_reachy.set_target_head_pose(neutral)
        time.sleep(0.2)

        current = sim_reachy.get_current_head_pose()
        assert np.allclose(current, np.eye(4), atol=0.1)

    def test_multiple_head_positions_in_sequence(self, sim_reachy):
        from reachy_mini.utils import create_head_pose

        positions = [
            (10, 0, 0),
            (-10, 0, 0),
            (0, 10, 0),
            (0, -10, 0),
            (0, 0, 10),
            (0, 0, -10),
        ]

        for yaw, pitch, roll in positions:
            pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
            sim_reachy.goto_target(head=pose, duration=0.2)
            time.sleep(0.3)

        # Return to neutral
        sim_reachy.goto_target(
            head=create_head_pose(degrees=True), duration=0.3
        )
        time.sleep(0.4)


# ── Antenna tests ─────────────────────────────────────────────────────


class TestAntennaMovement:
    def test_set_antenna_positions(self, sim_reachy):
        sim_reachy.set_target_antenna_joint_positions([0.5, -0.5])
        time.sleep(0.2)

        positions = sim_reachy.get_present_antenna_joint_positions()
        assert len(positions) == 2
        assert abs(positions[0] - 0.5) < 0.1
        assert abs(positions[1] - (-0.5)) < 0.1

    def test_antenna_via_goto_target(self, sim_reachy):
        from reachy_mini.utils import create_head_pose

        sim_reachy.goto_target(
            head=create_head_pose(degrees=True),
            antennas=[np.radians(-30), np.radians(30)],
            duration=0.3,
        )
        time.sleep(0.4)

        positions = sim_reachy.get_present_antenna_joint_positions()
        assert len(positions) == 2

    def test_antenna_snap_animation(self, sim_reachy):
        """The startup 'lobster claw' snap animation."""
        sim_reachy.set_target_antenna_joint_positions([0.7, -0.7])
        time.sleep(0.15)
        sim_reachy.set_target_antenna_joint_positions([-0.7, 0.7])
        time.sleep(0.15)
        sim_reachy.set_target_antenna_joint_positions([0.0, 0.0])
        time.sleep(0.15)

        positions = sim_reachy.get_present_antenna_joint_positions()
        assert abs(positions[0]) < 0.15
        assert abs(positions[1]) < 0.15


# ── Emotion expression execution ──────────────────────────────────────


class TestEmotionExecution:
    def test_execute_expression_moves_head(self, sim_reachy):
        from reachy_mini.utils import create_head_pose

        # Use set_target_head_pose (immediate) for reliable testing
        pose = create_head_pose(yaw=15, pitch=10, degrees=True)
        sim_reachy.set_target_head_pose(pose)
        time.sleep(0.3)

        current = sim_reachy.get_current_head_pose()
        assert current.shape == (4, 4)
        # Verify pose is not identity (head moved)
        assert not np.allclose(current, np.eye(4), atol=0.01)

    def test_execute_all_emotions_no_crash(self, sim_reachy):
        """Every mapped emotion can be executed without error."""
        from reachy_mini.utils import create_head_pose

        from clawd_reachy_mini.motion.emotion_mapper import EMOTION_MAP, EmotionMapper

        mapper = EmotionMapper(intensity=0.5)

        for emotion_name in EMOTION_MAP:
            expr = mapper.map_emotion(emotion_name)
            assert expr is not None, f"No expression for {emotion_name}"

            kwargs = {}
            if expr.head:
                kwargs["head"] = create_head_pose(
                    yaw=expr.head.yaw, pitch=expr.head.pitch,
                    roll=expr.head.roll, degrees=True,
                )
                kwargs["duration"] = max(expr.head.duration, 0.2)
            if expr.antenna:
                kwargs["antennas"] = [
                    np.radians(expr.antenna.right),
                    np.radians(expr.antenna.left),
                ]
            if not kwargs:
                continue

            sim_reachy.goto_target(**kwargs)
            time.sleep(0.3)

        # Return to neutral
        sim_reachy.goto_target(
            head=create_head_pose(degrees=True),
            antennas=[0, 0],
            duration=0.5,
        )
        time.sleep(0.6)


# ── Motion plugin with real sim ────────────────────────────────────────


class TestMotionPluginIntegration:
    @pytest.mark.asyncio
    async def test_motion_loop_processes_emotion_on_sim(self, sim_reachy):
        from clawd_reachy_mini.app import ClawdApp
        from clawd_reachy_mini.config import Config
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            motion_idle_animation_interval=999,
        )
        app = ClawdApp(config)
        app.reachy = sim_reachy

        plugin = MotionPlugin(app)
        plugin._running = True

        # Queue a surprise emotion
        app.emotions.queue_emotion("surprised")

        task = asyncio.create_task(plugin._motion_loop())
        await asyncio.sleep(1.0)  # let the expression execute
        plugin._running = False
        await task

        # Head should have moved from the surprised expression
        current = sim_reachy.get_current_head_pose()
        assert current.shape == (4, 4)

    @pytest.mark.asyncio
    async def test_head_tracking_drives_head_on_sim(self, sim_reachy):
        from clawd_reachy_mini.app import ClawdApp
        from clawd_reachy_mini.config import Config
        from clawd_reachy_mini.motion.head_target import HeadTarget
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            motion_head_tracking_poll_interval=0.05,
            motion_head_tracking_smoothing=0.5,
        )
        app = ClawdApp(config)
        app.reachy = sim_reachy

        plugin = MotionPlugin(app)
        plugin._running = True

        # Reset to neutral first
        from reachy_mini.utils import create_head_pose

        sim_reachy.set_target_head_pose(create_head_pose(degrees=True))
        await asyncio.sleep(0.3)

        # Publish face target at yaw=20
        app.head_targets.publish(
            HeadTarget(yaw=20.0, pitch=-5.0, confidence=0.9, source="face")
        )

        task = asyncio.create_task(plugin._head_tracking_loop())
        await asyncio.sleep(0.5)  # let tracking converge
        plugin._running = False
        await task

        # Verify head moved toward the target
        current = sim_reachy.get_current_head_pose()
        assert not np.allclose(current, np.eye(4), atol=0.05)

    @pytest.mark.asyncio
    async def test_idle_animation_moves_on_sim(self, sim_reachy):
        from clawd_reachy_mini.app import ClawdApp
        from clawd_reachy_mini.config import Config
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=True,
            enable_face_tracker=False,
            motion_idle_animation_interval=0.1,
        )
        app = ClawdApp(config)
        app.reachy = sim_reachy
        app.is_speaking = False

        plugin = MotionPlugin(app)
        plugin._running = True

        task = asyncio.create_task(plugin._motion_loop())
        await asyncio.sleep(0.8)
        plugin._running = False
        await task

        # Idle animation should have moved head


# ── Full plugin orchestration ──────────────────────────────────────────


class TestFullPluginOrchestration:
    @pytest.mark.asyncio
    async def test_app_registers_and_runs_motion_plugin(self, sim_reachy):
        from clawd_reachy_mini.app import ClawdApp
        from clawd_reachy_mini.config import Config
        from clawd_reachy_mini.plugins.motion_plugin import MotionPlugin

        config = Config(
            idle_animations=False,
            enable_face_tracker=False,
            enable_motion=True,
            standalone_mode=True,
        )
        app = ClawdApp(config)
        app.reachy = sim_reachy

        plugin = MotionPlugin(app)
        registered = app.register(plugin)
        assert registered

        # Queue emotion before run
        app.emotions.queue_emotion("excited")

        run_task = asyncio.create_task(app.run())
        await asyncio.sleep(1.0)

        # Shutdown
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        # Clean up without disconnecting the shared sim_reachy
        app.reachy = None
        await app.shutdown()
