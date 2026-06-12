"""Unit tests for the motion compositor: official-move expressions, periodic
presence, and arbitration (mock robot, stubbed libraries — runs offline).
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

from reachy_voice.motion import (
    _FREEZE_STATES,
    _MAX_BODY_STEP_DEG,
    _MAX_HEAD_STEP_DEG,
    _EMOTION_MOVES,
    _PRESENCE_MOVES,
    _slew,
    MotionController,
)


def test_freeze_states_cover_listening():
    # The robot must hold still (clean mic) while listening/thinking.
    assert "listening" in _FREEZE_STATES
    assert "thinking" in _FREEZE_STATES
    assert "speaking" not in _FREEZE_STATES  # may move while speaking
    assert "idle" not in _FREEZE_STATES


def test_set_conv_state():
    mc = MotionController(MagicMock())
    mc.set_conv_state("listening")
    assert mc._conv_state == "listening"
    mc.set_conv_state(None)  # defends against None
    assert mc._conv_state == "idle"


# ── motor safety: slew-rate limiting ──────────────────────────────────


def test_slew_caps_step():
    assert _slew(0.0, 100.0, 1.0) == 1.0      # big jump clamped to max_step
    assert _slew(0.0, -100.0, 1.0) == -1.0
    assert _slew(5.0, 5.3, 1.0) == 5.3        # small move passes through


def test_ambient_never_jumps_more_than_max_step_per_tick():
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)
    mc.set_gaze(35.0, 20.0)  # ask for a large gaze swing all at once

    captured = []

    def fake_pose(yaw, pitch, roll, degrees):
        captured.append((yaw, pitch, roll))
        return None

    prev = (0.0, 0.0, 0.0)
    prev_body = 0.0
    for _ in range(40):
        mc._tick_ambient(fake_pose)
        y, p, r = captured[-1]
        # head deltas within the per-tick cap (+ tiny epsilon for float noise)
        assert abs(y - prev[0]) <= _MAX_HEAD_STEP_DEG + 1e-6
        assert abs(p - prev[1]) <= _MAX_HEAD_STEP_DEG + 1e-6
        assert abs(r - prev[2]) <= _MAX_HEAD_STEP_DEG + 1e-6
        body = mc._cmd_body
        assert abs(body - prev_body) <= _MAX_BODY_STEP_DEG + 1e-6
        prev, prev_body = (y, p, r), body


# ── tag → official move mapping ───────────────────────────────────────


def test_every_prompt_tag_maps_to_moves():
    prompt_tags = (
        "happy", "excited", "laughing", "curious", "thinking", "surprised",
        "amazed", "proud", "loving", "grateful", "welcoming", "confused",
        "shy", "sad",
    )
    for tag in prompt_tags:
        assert tag in _EMOTION_MOVES, f"prompt tag [{tag}] has no move mapping"
        assert _EMOTION_MOVES[tag], f"[{tag}] maps to an empty move list"


def _mc_with_moves(reachy, names):
    """A MotionController whose 'official libraries' are stubbed with `names`."""
    mc = MotionController(reachy)
    mc._move_names = set(names)
    mc._resolve = lambda n: f"move:{n}" if n in mc._move_names else None
    return mc


def test_pick_move_only_returns_available_names():
    mc = _mc_with_moves(MagicMock(), {"cheerful1"})  # one of happy's candidates
    for _ in range(20):
        assert mc._pick_move("happy") == "cheerful1"


def test_pick_move_none_when_library_missing_tag():
    mc = _mc_with_moves(MagicMock(), set())
    assert mc._pick_move("happy") is None


# ── expression playback / arbitration ─────────────────────────────────


def test_emotion_plays_official_move_with_sound_off():
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"cheerful1", "enthusiastic1", "enthusiastic2"})
    mc._run_expression("happy")
    assert reachy.play_move.called
    _, kwargs = reachy.play_move.call_args
    assert kwargs.get("sound") is False  # must not fight the duplex audio stream


def test_emotion_falls_back_to_wag_without_library():
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, set())
    mc._run_expression("happy")
    assert not reachy.play_move.called
    assert reachy.set_target_antenna_joint_positions.called
    for call in reachy.set_target_antenna_joint_positions.call_args_list:
        for v in call.args[0]:
            assert abs(v) < math.radians(70)  # radians, not degrees


def test_play_emotion_debounced_while_move_in_flight():
    # A new emotion while a move is playing must be DROPPED (never a rapid
    # cancel+replay — that desyncs the daemon's serial bus).
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"cheerful1"})
    mc._playing = True
    mc.play_emotion("sad")
    assert mc._pending is None        # dropped, not queued
    assert not reachy.cancel_move.called  # never preempt


def test_play_emotion_debounced_within_min_gap():
    import time as _t
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"cheerful1"})
    mc._playing = False
    mc._last_move_start = _t.monotonic()  # a move just started
    mc.play_emotion("happy")
    assert mc._pending is None            # too soon → dropped


def test_play_emotion_accepted_when_idle_and_spaced():
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"cheerful1"})
    mc._playing = False
    mc._last_move_start = -1e9            # long ago
    mc.play_emotion("happy")
    assert mc._pending == "happy"


# ── periodic presence ─────────────────────────────────────────────────


def test_presence_pool_filters_to_available_moves():
    mc = _mc_with_moves(MagicMock(), {"curious1", "simple_nod", "not_a_presence_move"})
    pool = mc._presence_pool()
    assert set(pool) == {"curious1", "simple_nod"}
    assert all(m in _PRESENCE_MOVES for m in pool)


def test_play_presence_uses_official_move():
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"curious1", "simple_nod"})
    mc._play_presence()
    assert reachy.play_move.called
    _, kwargs = reachy.play_move.call_args
    assert kwargs.get("sound") is False


def test_presence_disabled_without_library():
    mc = _mc_with_moves(MagicMock(), set())
    assert mc._presence_pool() == []


# ── state-gated motion: STILL while listening ─────────────────────────


def test_idle_head_rests_at_neutral_when_nothing_to_do():
    # No gaze, no speech → head target is neutral, so after settling the deadband
    # holds it still (quiet servos). The first ticks may send (easing to neutral),
    # but it converges to ~0 and stops.
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)
    seen = []

    def fake_pose(yaw, pitch, roll, degrees):
        seen.append((yaw, pitch, roll))
        return None

    for _ in range(20):
        mc._tick_ambient(fake_pose)
    y, p, r = seen[-1]
    assert abs(y) < 0.5 and abs(p) < 0.5 and abs(r) < 0.5  # rests near neutral


def test_speech_wobble_only_with_audio():
    # head moves while TTS plays (wobble), driven purely by play_rms
    reachy = MagicMock()
    audio = MagicMock()
    audio.play_rms.return_value = 0.5
    mc = MotionController(reachy, audio=audio)
    moved = False
    for _ in range(60):
        captured = {}
        mc._tick_ambient(lambda yaw, pitch, roll, degrees: captured.update(r=roll))
        # the deadband may suppress a send on a given tick (lambda not called),
        # so use .get; over many ticks the wobble crosses the threshold.
        if abs(captured.get("r", 0.0)) > 1.0:
            moved = True
            break
        time.sleep(0.005)
    assert moved, "head should wobble while TTS plays"


def test_rest_antennas_are_radians_and_bounded():
    from reachy_voice.motion import _REST_ANTENNAS
    right, left = _REST_ANTENNAS
    assert abs(right) < math.radians(30) and abs(left) < math.radians(30)


def test_ambient_tick_drives_head_body_and_antennas_within_clamp():
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)

    def fake_pose(yaw, pitch, roll, degrees):
        assert abs(yaw) <= 20 and abs(pitch) <= 16 and abs(roll) <= 16
        return ("pose", yaw, pitch, roll)

    mc._tick_ambient(fake_pose)
    # mirrors async_play_move's proven separate calls
    assert reachy.set_target_head_pose.called
    assert reachy.set_target_body_yaw.called
    assert reachy.set_target_antenna_joint_positions.called


def test_speech_wobble_scales_with_audio_rms():
    reachy = MagicMock()
    audio = MagicMock()
    audio.play_rms.return_value = 0.5  # loud TTS
    mc = MotionController(reachy, audio=audio)
    seen = {}

    def fake_pose(yaw, pitch, roll, degrees):
        seen["roll"] = roll
        return None

    big = False
    for _ in range(30):
        mc._tick_ambient(fake_pose)
        if abs(seen["roll"]) > 1.0:
            big = True
            break
        time.sleep(0.005)
    assert big, "speech wobble did not amplify head motion under loud audio"


def test_speaking_flag_follows_rms():
    audio = MagicMock()
    audio.play_rms.return_value = 0.2
    mc = MotionController(MagicMock(), audio=audio)
    assert mc._speaking() is True
    audio.play_rms.return_value = 0.0
    assert mc._speaking() is False


# ── safety ────────────────────────────────────────────────────────────


def test_deadband_suppresses_redundant_sends_when_static():
    # With no gaze and no audio, the head target is ~static; after it settles,
    # the deadband must stop re-sending (slash bus traffic) — the lever against
    # motor-comms errors.
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)
    mc.set_gaze(None, None)  # idle

    def fake_pose(yaw, pitch, roll, degrees):
        return ("pose", yaw, pitch, roll)

    for _ in range(200):  # ~8s at 25Hz
        mc._tick_ambient(fake_pose)
    settled = reachy.set_target_head_pose.call_count
    # let it run more; with a tiny static-ish breathing, sends stay sparse
    for _ in range(50):
        mc._tick_ambient(fake_pose)
    extra = reachy.set_target_head_pose.call_count - settled
    # far fewer than one send per tick (deadband working)
    assert extra < 50, f"deadband not gating: {extra} sends in 50 idle ticks"


def test_large_gaze_change_still_sends():
    reachy = MagicMock()
    mc = MotionController(reachy, audio=None)

    def fake_pose(yaw, pitch, roll, degrees):
        return ("pose", yaw, pitch, roll)

    mc._tick_ambient(fake_pose)  # establish baseline send
    before = reachy.set_target_body_yaw.call_count
    mc.set_gaze(35.0, 0.0)       # big new aim → must drive the body
    for _ in range(20):
        mc._tick_ambient(fake_pose)
    assert reachy.set_target_body_yaw.call_count > before


def test_move_invalidates_sent_so_ambient_recommands():
    reachy = MagicMock()
    mc = _mc_with_moves(reachy, {"cheerful1"})

    def fake_pose(yaw, pitch, roll, degrees):
        return ("pose", yaw, pitch, roll)

    mc._tick_ambient(fake_pose)            # primes _sent_head
    assert mc._sent_head is not None
    mc._run_expression("happy")            # plays a move → must invalidate
    assert mc._sent_head is None


def test_link_loss_pauses_quietly_without_flooding():
    # When the SDK raises ConnectionError (daemon link dropped), the drive must
    # mark the link down ONCE and stop — never raise or hammer.
    reachy = MagicMock()
    reachy.set_target_head_pose.side_effect = ConnectionError("Lost connection")
    mc = MotionController(reachy, audio=None)

    def fake_pose(yaw, pitch, roll, degrees):
        return None

    for _ in range(10):
        mc._tick_ambient(fake_pose)  # must not raise
    assert mc._link_down is True


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
    mc.play_emotion("happy")  # must not raise
    mc.stop()
    assert mc._thread is None
