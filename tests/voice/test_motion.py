"""Unit tests for the Jetson-safe motion controller.

The current architecture deliberately removes visual servo tracking. Vision may
publish frames/faces for the dashboard and snapshot analysis, but it must not
drive body/head motors. Motion is limited to explicit speech/tool gestures on
the single compositor thread.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from reachy_voice.motion import (
    _EMOTION_MOVES,
    _FREEZE_STATES,
    _PRESENCE_MOVES,
    _REST_ANTENNAS,
    _slew,
    MotionController,
)


def test_freeze_states_cover_listening():
    assert "listening" in _FREEZE_STATES
    assert "thinking" in _FREEZE_STATES
    assert "speaking" not in _FREEZE_STATES
    assert "idle" not in _FREEZE_STATES


def test_set_conv_state():
    mc = MotionController(MagicMock())
    mc.set_conv_state("listening")
    assert mc._conv_state == "listening"
    mc.set_conv_state(None)
    assert mc._conv_state == "idle"


def test_slew_caps_step():
    assert _slew(0.0, 100.0, 1.0) == 1.0
    assert _slew(0.0, -100.0, 1.0) == -1.0
    assert _slew(5.0, 5.3, 1.0) == 5.3


def test_every_prompt_tag_maps_to_moves():
    prompt_tags = (
        "happy", "excited", "laughing", "curious", "thinking", "surprised",
        "amazed", "proud", "loving", "grateful", "welcoming", "confused",
        "shy", "sad",
    )
    for tag in prompt_tags:
        assert tag in _EMOTION_MOVES, f"prompt tag [{tag}] has no move mapping"
        assert _EMOTION_MOVES[tag], f"[{tag}] maps to an empty move list"


def test_play_emotion_debounced_while_move_in_flight():
    reachy = MagicMock()
    mc = MotionController(reachy)
    mc._playing = True
    mc.play_emotion("sad")
    assert mc._pending is None
    assert not reachy.cancel_move.called


def test_play_emotion_debounced_within_min_gap():
    import time as _t

    reachy = MagicMock()
    mc = MotionController(reachy)
    mc._playing = False
    mc._last_move_start = _t.monotonic()
    mc.play_emotion("happy")
    assert mc._pending is None


def test_play_emotion_accepted_when_idle_and_spaced():
    reachy = MagicMock()
    mc = MotionController(reachy)
    mc._playing = False
    mc._last_move_start = -1e9
    mc.play_emotion("happy")
    assert mc._pending == "happy"


def test_presence_disabled_even_if_moves_exist():
    mc = MotionController(MagicMock())
    mc._move_names = set(_PRESENCE_MOVES)
    assert mc._presence_pool() == []


def test_rest_antennas_are_radians_and_bounded():
    right, left = _REST_ANTENNAS
    assert abs(right) < math.radians(30) and abs(left) < math.radians(30)


def test_speaking_flag_follows_rms():
    audio = MagicMock()
    audio.play_rms.return_value = 0.2
    mc = MotionController(MagicMock(), audio=audio)
    assert mc._speaking() is True
    audio.play_rms.return_value = 0.0
    assert mc._speaking() is False


def test_speech_step_uses_goto_for_head_and_antennas_only():
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)

    def fake_pose(**kwargs):
        return ("pose", kwargs)

    mc._speech_goto_step(fake_pose)

    assert reachy.enable_motors.called
    assert reachy.goto_target.called
    _, kwargs = reachy.goto_target.call_args
    assert "head" in kwargs
    assert "antennas" in kwargs
    assert "duration" in kwargs
    assert "body_yaw" not in kwargs
    assert not reachy.set_target_body_yaw.called
    assert not reachy.set_target_head_pose.called


def test_command_head_enqueues_without_touching_sdk():
    reachy = MagicMock()
    mc = MotionController(reachy)

    result = mc.command_head(90, -90, 90, duration=0.7)

    assert result["ok"] is True
    assert result["moved_to"] == {"yaw": 45.0, "pitch": -30.0, "roll": 30.0}
    assert mc._pending_cmd == ("head", (45.0, -30.0, 30.0, 0.7))
    assert not reachy.goto_target.called


def test_command_antennas_enqueues_without_touching_sdk():
    reachy = MagicMock()
    mc = MotionController(reachy)

    result = mc.command_antennas(left=12, right=-8, duration=0.4)

    assert result["ok"] is True
    assert mc._pending_cmd == ("antennas", (12.0, -8.0, 0.4))
    assert not reachy.goto_target.called


def test_run_command_head_goes_through_single_writer(monkeypatch):
    reachy = MagicMock()
    mc = MotionController(reachy)

    def fake_pose(**kwargs):
        return ("pose", kwargs)

    import reachy_voice.motion as motion

    monkeypatch.setitem(
        __import__("sys").modules,
        "reachy_mini.utils",
        type("Utils", (), {"create_head_pose": staticmethod(fake_pose)}),
    )

    mc._run_command(("head", (5.0, -2.0, 1.0, 0.6)))

    assert reachy.enable_motors.called
    assert reachy.goto_target.called
    _, kwargs = reachy.goto_target.call_args
    assert kwargs["head"][1]["yaw"] == 5.0
    assert kwargs["duration"] == 0.6
    assert not reachy.set_target_body_yaw.called
    assert motion is not None


def test_link_loss_pauses_quietly():
    reachy = MagicMock()
    reachy.enable_motors.side_effect = ConnectionError("Lost connection")
    mc = MotionController(reachy, audio=None)

    def fake_pose(**kwargs):
        return ("pose", kwargs)

    mc._speech_goto_step(fake_pose)

    assert mc._link_down is True
    assert reachy.enable_motors.call_count == 1


def test_probe_link_recovers():
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)
    mc._link_down = True
    reachy.get_current_head_pose.return_value = object()
    assert mc._probe_link() is True
    reachy.get_current_head_pose.side_effect = ConnectionError("still down")
    assert mc._probe_link() is False


def test_no_robot_is_noop():
    mc = MotionController(None)
    mc.start()
    mc.play_emotion("happy")
    mc.stop()
    assert mc._thread is None
