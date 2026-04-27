"""Tests for Plugin base on_rest_start / on_rest_end hooks."""

from __future__ import annotations

import pytest

from reachy_claw.event_bus import EventBus
from reachy_claw.plugin import Plugin


class _NoopPlugin(Plugin):
    name = "noop"

    async def start(self) -> None:
        pass


class _TrackingPlugin(Plugin):
    name = "tracking"

    def __init__(self, app):
        super().__init__(app)
        self.entered = False
        self.exited = False

    async def start(self) -> None:
        pass

    async def on_rest_start(self) -> None:
        self.entered = True

    async def on_rest_end(self) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_default_hooks_are_noop():
    p = _NoopPlugin(app=None)  # type: ignore[arg-type]
    # Must be coroutines that return without error
    await p.on_rest_start()
    await p.on_rest_end()


@pytest.mark.asyncio
async def test_subclass_can_override_hooks():
    p = _TrackingPlugin(app=None)  # type: ignore[arg-type]
    await p.on_rest_start()
    await p.on_rest_end()
    assert p.entered is True
    assert p.exited is True


# ── Task 9: ConversationPlugin pause/resume ─────────────────────────

@pytest.mark.asyncio
async def test_conversation_pauses_on_rest_event():
    """ConversationPlugin toggles _paused on rest events."""
    from reachy_claw.plugins.conversation_plugin import ConversationPlugin

    class _StubApp:
        events = EventBus()
        config = type("C", (), {})()

    app = _StubApp()
    plugin = ConversationPlugin(app)
    await plugin.on_rest_start()
    assert plugin._paused is True
    await plugin.on_rest_end()
    assert plugin._paused is False


# ── Task 10: FaceTracker / VisionClient / Motion pause/resume ───────

@pytest.mark.asyncio
async def test_face_tracker_pauses():
    from reachy_claw.plugins.face_tracker_plugin import FaceTrackerPlugin

    class _StubConfig:
        vision_tracker_type = "none"
        vision_camera_source = "opencv"
        vision_camera_index = 0
        vision_max_yaw = 45.0
        vision_max_pitch = 30.0
        vision_max_roll = 20.0
        vision_smoothing_alpha = 0.3
        vision_deadzone = 0.02
        vision_face_lost_delay = 2.0

    class _StubApp:
        events = EventBus()
        config = _StubConfig()
        reachy = None

    p = FaceTrackerPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False


@pytest.mark.asyncio
async def test_vision_client_pauses():
    from reachy_claw.plugins.vision_client_plugin import VisionClientPlugin

    class _StubConfig:
        vision_service_url = "tcp://127.0.0.1:8631"
        vision_max_yaw = 45.0
        vision_max_pitch = 30.0
        vision_pitch_offset = 5.0
        vision_max_roll = 20.0
        vision_smoothing_alpha = 0.3
        vision_deadzone = 0.02
        vision_face_lost_delay = 2.0
        vision_emotion_threshold = 0.5
        vision_emotion_cooldown = 3.0
        vision_body_yaw_gain = 0.5
        vision_emotion_sustain = 10.0

    class _StubApp:
        events = EventBus()
        config = _StubConfig()
        head_targets = type("H", (), {"publish": lambda *a, **kw: None})()
        emotions = type("E", (), {"queue_emotion": lambda *a: None})()

    p = VisionClientPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False


@pytest.mark.asyncio
async def test_motion_pauses():
    from reachy_claw.plugins.motion_plugin import MotionPlugin

    class _StubConfig:
        motor_enabled = True
        motor_preset = "moderate"
        motion_head_tracking_smoothing = 0.35
        motion_head_tracking_poll_interval = 0.05
        idle_animations = False

    class _StubApp:
        events = EventBus()
        config = _StubConfig()
        motor_enabled = True
        is_speaking = False
        reachy = None

    p = MotionPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False
