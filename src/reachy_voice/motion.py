"""Jetson-safe motion for Reachy Mini.

Vision is intentionally display/snapshot-only: it never drives body, head, or
antenna motors. This controller is the single writer for all remaining motion,
and it only sends slow speech/tool gestures through ``goto_target``.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time

logger = logging.getLogger("reachy_voice.motion")

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"

# LLM emotion tag → candidate official emotion moves (one picked at random each
# time, so a tag doesn't always look identical). Validated against the library
# at load; a tag left with no valid move falls back to the antenna wag.
_EMOTION_MOVES: dict[str, tuple[str, ...]] = {
    "happy":      ("cheerful1", "enthusiastic1", "enthusiastic2"),
    "excited":    ("enthusiastic1", "enthusiastic2", "electric1"),
    "laughing":   ("laughing1", "laughing2"),
    "proud":      ("proud1", "proud2", "proud3"),
    "loving":     ("loving1", "grateful1"),
    "grateful":   ("grateful1", "understanding1"),
    "amazed":     ("amazed1", "surprised1"),
    "surprised":  ("surprised1", "surprised2"),
    "curious":    ("curious1", "inquiring1", "inquiring2", "inquiring3"),
    "thinking":   ("thoughtful1", "thoughtful2", "uncertain1"),
    "confused":   ("confused1", "incomprehensible2", "uncertain1"),
    "welcoming":  ("welcoming1", "welcoming2", "come1"),
    "yes":        ("yes1", "understanding1"),
    "no":         ("no1",),
    "shy":        ("shy1",),
    "sad":        ("sad1", "sad2", "downcast1"),
    "lonely":     ("lonely1", "downcast1"),
    "tired":      ("tired1", "exhausted1"),
    "scared":     ("scared1", "fear1", "anxiety1"),
    "angry":      ("irritated1", "irritated2", "displeased1"),  # mild for an exhibition
    "neutral":    ("serenity1", "calming1"),
}

# Gentle official moves played spontaneously when idle, to "show presence".
# A mix of look-around / attentive emotions and small dances — nothing manic.
_PRESENCE_MOVES: tuple[str, ...] = (
    # subtle emotions
    "curious1", "inquiring1", "inquiring2", "inquiring3",
    "attentive1", "attentive2", "thoughtful1", "thoughtful2",
    "boredom1", "boredom2", "impatient1", "serenity1", "shy1",
    # small dances
    "simple_nod", "yeah_nod", "uh_huh_tilt", "head_tilt_roll",
    "side_to_side_sway", "side_glance_flick", "chin_lead",
)

# Move names from the dances library (used to validate the `dance` tool arg).
# A subset of _PRESENCE_MOVES are dances; this lists the canonical dance names
# the LLM may request. Validated against the loaded library at call time.
_DANCE_MOVES: frozenset[str] = frozenset({
    "simple_nod", "yeah_nod", "uh_huh_tilt", "head_tilt_roll",
    "side_to_side_sway", "side_glance_flick", "chin_lead",
})

_COMPOSITOR_HZ = 25.0

# Conversation states during which the robot must hold PERFECTLY STILL so the
# (AEC-less to mechanical noise) mic hears the visitor cleanly — servo motion
# while listening is what wrecked multi-turn ASR. Motion only happens while
# SPEAKING (its own TTS, user not talking) or IDLE (nobody / between visitors).
_FREEZE_STATES = frozenset({"listening", "thinking", "barged_in"})

# Minimum spacing between official-move STARTS. Triggering moves faster than this
# (rapid cancel_move + play_move on top of an in-flight move) desyncs the daemon's
# 100Hz serial command stream → all-motor "no response" fault. So a move always
# plays to completion and new triggers within this window are dropped — never
# overlapping/preempting.
_MIN_MOVE_GAP_S = 1.0
# Idle gap before a spontaneous presence move (randomised in this range each
# time so it never feels metronomic). Spaced out to limit motor-current events.
_PRESENCE_GAP_MIN = 25.0
_PRESENCE_GAP_MAX = 60.0

# Send-deadband: don't re-issue a target that barely changed. This is the main
# lever against "motor communication error" — it slashes serial-bus traffic
# (idle head goes from ~25 packets/s to a couple), and cuts needless current.
# Below these per-axis deltas (vs the last SENT target), the command is skipped.
_SEND_DEADBAND_DEG = 0.35
_SEND_DEADBAND_ANT_RAD = math.radians(1.5)

# Antennas rest at this fixed relaxed pose between moves (radians [right, left]).
# No continuous sway — that read as jitter and was needless bus traffic. The
# official recorded moves animate the antennas when something actually happens.
_REST_ANTENNAS = [math.radians(8.0), math.radians(8.0)]

# HARD SAFETY: retained for one-shot commands that need slew-style helpers.
_MAX_HEAD_STEP_DEG = 2.0  # per tick @25Hz ≈ 50°/s
_MAX_BODY_STEP_DEG = 1.6  # per tick @25Hz ≈ 40°/s

# Jetson-safe motor partition:
# - vision writes no motors
# - speech writes head + antennas only, at slow goto cadence
# - no ambient head wobble at compositor rate
_SPEECH_STEP_MIN_S = 0.80
_SPEECH_STEP_MAX_S = 1.20
_SPEECH_HEAD_YAW_DEG = 4.0
_SPEECH_HEAD_PITCH_DEG = 3.0
_SPEECH_HEAD_ROLL_DEG = 4.0
_SPEECH_ANTENNA_RAD = 0.23

# Fallback antenna wag (center°, amp°, freq Hz, phase, dur s) when no official
# move is available for a tag.
_WAG = (15.0, 18.0, 2.5, math.pi, 1.6)


def _slew(cur: float, target: float, max_step: float) -> float:
    """Move ``cur`` toward ``target`` by at most ``max_step`` (rate limiter)."""
    d = target - cur
    if d > max_step:
        d = max_step
    elif d < -max_step:
        d = -max_step
    return cur + d


def _max_delta(a, b) -> float:  # noqa: ANN001
    """Largest absolute per-element difference between two equal-length tuples."""
    return max(abs(x - y) for x, y in zip(a, b))


class MotionController:
    """Owns all motor output on one thread: a compositor that keeps the robot
    alive, preempted by official emotion moves, and spontaneously playing gentle
    official moves when idle."""

    def __init__(self, reachy: object | None, audio: object | None = None) -> None:
        self.reachy = reachy
        self.audio = audio                 # exposes play_rms() for speech wobble
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: str | None = None   # emotion tag awaiting playback
        # One-shot tool command awaiting playback on the compositor thread.
        # Tools (move_head/move_antennas/dance) run OFF the compositor (from
        # asyncio.to_thread in the tools plugin); they MUST NOT touch the SDK
        # directly — that fights the single-writer compositor. Instead they
        # enqueue a (kind, args) here and the compositor thread executes it,
        # preserving the single-writer invariant. Newest wins (a fresh command
        # supersedes an unconsumed one). Protected by ``_cmd_lock``.
        self._pending_cmd: tuple[str, tuple] | None = None
        self._cmd_lock = threading.Lock()
        self._playing = False              # an official move is in flight
        self._last_move_start = -1e9       # monotonic; debounce overlapping moves
        self._libs: list = []              # RecordedMoves datasets, in lookup order
        self._move_names: set[str] = set()
        self._t0 = 0.0
        self._last_activity = 0.0          # monotonic; speech/emotion reset it
        self._next_gap = _PRESENCE_GAP_MIN
        self._conv_state = "idle"          # listen→freeze, speak→move (set by engine)
        # last commanded targets (deg) — slew-limited each tick for motor safety
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._cmd_roll = 0.0
        self._cmd_body = 0.0
        self._last_speech_step = 0.0
        self._next_speech_step = 0.0
        # When the daemon link drops (e.g. a motor-bus fault), motion PAUSES and
        # probes for recovery once a second — it must never hammer the SDK at
        # tick rate and flood tracebacks, which would starve the conversation.
        self._link_down = False
        # Last targets actually SENT to the bus (deadband gating). None = unsent.
        self._sent_head: tuple[float, float, float] | None = None
        self._sent_body: float | None = None
        self._sent_ant: list[float] | None = None
        self._motors_enabled = False

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self.reachy is None:
            return
        self._stop.clear()
        self._t0 = self._last_activity = time.monotonic()
        self._next_gap = random.uniform(_PRESENCE_GAP_MIN, _PRESENCE_GAP_MAX)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="motion-compositor"
        )
        self._thread.start()
        logger.info("motion compositor started")

    def stop(self) -> None:
        self._stop.set()
        if self.reachy is not None:
            try:
                self.reachy.cancel_move()
            except Exception:  # noqa: BLE001
                pass
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    # ── public control ───────────────────────────────────────────────
    def play_emotion(self, emotion: str) -> None:
        """Request an expression burst. DEBOUNCED: if a move is already playing or
        one started < _MIN_MOVE_GAP_S ago, the request is dropped — never preempt
        with a rapid cancel+replay (that desyncs the daemon's serial bus)."""
        if self.reachy is None:
            return
        now = time.monotonic()
        self._last_activity = now
        if self._playing or (now - self._last_move_start) < _MIN_MOVE_GAP_S:
            logger.debug("emotion [%s] dropped (debounce: move in flight)", emotion)
            return
        self._pending = emotion

    def set_conv_state(self, state: str) -> None:
        """Conversation state from the engine. While listening/thinking the robot
        freezes (clean mic); it only moves while speaking or idle."""
        self._conv_state = state or "idle"

    # ── tool commands (compositor-safe; enqueue only) ─────────────────
    #
    # These three are invoked by the client-loop motion TOOLS off the
    # compositor thread (asyncio.to_thread). They DO NOT call the SDK here —
    # that would fight the single-writer compositor. They enqueue a one-shot
    # command which the compositor thread executes in ``_run_command`` between
    # ambient ticks, exactly like ``play_emotion`` queues ``_pending``.

    def command_head(
        self, yaw: float, pitch: float, roll: float = 0.0, duration: float = 1.0
    ) -> dict:
        """Tool: point the head at yaw/pitch/roll (DEGREES, clamped to the
        on-device safe envelope). Compositor-safe: enqueues a one-shot goto."""
        if self.reachy is None:
            return {"ok": False, "reason": "no_robot",
                    "requested": {"yaw": yaw, "pitch": pitch, "roll": roll}}
        yaw = max(-45.0, min(45.0, float(yaw)))
        pitch = max(-30.0, min(30.0, float(pitch)))
        roll = max(-30.0, min(30.0, float(roll)))
        self._enqueue_cmd("head", (yaw, pitch, roll, float(duration)))
        return {"ok": True, "moved_to": {"yaw": yaw, "pitch": pitch, "roll": roll}}

    def command_antennas(
        self, left: float, right: float, duration: float = 0.5
    ) -> dict:
        """Tool: move the two antennas to target angles (DEGREES, positive =
        up). SDK antenna order is [right, left] (radians). Compositor-safe."""
        if self.reachy is None:
            return {"ok": False, "reason": "no_robot",
                    "requested": {"left": left, "right": right}}
        self._enqueue_cmd("antennas", (float(left), float(right), float(duration)))
        return {"ok": True, "antennas": {"left": left, "right": right}}

    def play_dance(self, name: str) -> dict:
        """Tool: play a choreographed dance routine from the dances library.
        Compositor-safe: enqueues the move, which the compositor plays via
        ``play_move`` (same path as presence). Library load is async, so the
        valid set is only known once it's ready — an unknown name returns a
        structured ack listing what's available."""
        if self.reachy is None:
            return {"ok": False, "reason": "no_robot", "dance": name}
        if not name:
            return {"ok": False, "reason": "missing_dance"}
        if self._move_names and name not in self._move_names:
            return {"ok": False, "reason": "unknown_dance", "dance": name,
                    "available": sorted(self._dance_pool())}
        self._enqueue_cmd("dance", (name,))
        return {"ok": True, "dance": name}

    def _dance_pool(self) -> list[str]:
        """Dance-library move names currently loaded (subset of presence pool
        that are dances, plus any other loaded move)."""
        return [m for m in self._move_names if m in _DANCE_MOVES] or list(
            m for m in _PRESENCE_MOVES if m in self._move_names
        )

    def _enqueue_cmd(self, kind: str, args: tuple) -> None:
        now = time.monotonic()
        self._last_activity = now
        with self._cmd_lock:
            self._pending_cmd = (kind, args)

    # ── library ──────────────────────────────────────────────────────
    def _load_library(self) -> None:
        # The Jetson-safe motion stack deliberately does not load the official
        # full-body recorded move libraries. They drive body/head/antennas as one
        # animation, which breaks the motor partition needed to avoid twitching.
        self._libs = []
        self._move_names = set()

    def _resolve(self, name: str):  # noqa: ANN201 — returns a Move or None
        for lib in self._libs:
            try:
                if name in lib.list_moves():
                    return lib.get(name)
            except Exception:  # noqa: BLE001
                continue
        return None

    def _pick_move(self, emotion: str) -> str | None:
        cands = [m for m in _EMOTION_MOVES.get(emotion, ()) if m in self._move_names]
        return random.choice(cands) if cands else None

    # ── compositor thread ────────────────────────────────────────────
    def _loop(self) -> None:
        try:
            from reachy_mini.utils import create_head_pose
        except Exception:  # noqa: BLE001
            create_head_pose = None

        # Load libraries here (not in start()) so a cold HF cache download can't
        # block app startup — until ready, expressions wag and presence is off.
        self._load_library()
        logger.info(
            "motion library ready (%d official moves, presence %s)",
            len(self._move_names),
            "on" if self._presence_pool() else "off",
        )

        dt = 1.0 / _COMPOSITOR_HZ
        while not self._stop.is_set():
            # If the daemon link is down, do NOT drive at tick rate (that floods
            # tracebacks and starves the rest of the app). Back off and probe.
            if self._link_down:
                time.sleep(1.0)
                if self._probe_link():
                    logger.info("motion: daemon link recovered; resuming")
                    self._link_down = False
                    self._cmd_yaw = self._cmd_pitch = self._cmd_roll = self._cmd_body = 0.0
                    self._sent_head = self._sent_body = self._sent_ant = None
                continue

            listening = self._conv_state in _FREEZE_STATES
            speaking = self._conv_state == "speaking" or self._speaking()

            # 1) explicit emotion from the LLM (a designed move). Only burst while
            #    SPEAKING/idle — never a big fast move while listening (that's what
            #    polluted the mic). Hold it pending until listening ends.
            pending = self._pending
            if pending is not None and not listening:
                self._pending = None
                self._run_expression(pending)
                self._last_activity = time.monotonic()
                continue

            # 1b) one-shot tool command (move_head / move_antennas / dance).
            #     Executed on THIS (compositor) thread so the single-writer
            #     invariant holds. Debounced/gated like an expression — never
            #     while listening, never overlapping an in-flight move.
            if not listening and not self._playing:
                with self._cmd_lock:
                    cmd = self._pending_cmd
                    self._pending_cmd = None
                if cmd is not None and (
                    time.monotonic() - self._last_move_start
                ) >= _MIN_MOVE_GAP_S:
                    self._run_command(cmd)
                    self._last_activity = time.monotonic()
                    continue
                elif cmd is not None:
                    # too soon after the last move — requeue and let it land next
                    with self._cmd_lock:
                        if self._pending_cmd is None:
                            self._pending_cmd = cmd

            # SPEAKING owns the head + antennas. Do not run vision/body motion
            # here; this is the hard partition that prevents the "services fight"
            # behaviour seen on the live robot.
            if speaking:
                self._last_activity = time.monotonic()
                if create_head_pose is not None:
                    self._tick_speech(create_head_pose)
                time.sleep(dt)
                continue

            # While LISTENING/THINKING: no motor output.
            if listening:
                self._last_activity = time.monotonic()
                time.sleep(dt)
                continue

            # No idle/listening motor output. Vision must never recenter or track
            # through the motor stack.
            time.sleep(dt)

    def _mark_link_down(self) -> None:
        """Called from a drive method when an SDK call fails. Logs ONCE on the
        up→down transition (no per-tick flood) and pauses motion."""
        if not self._link_down:
            logger.warning("motion: lost daemon link — pausing motion until it recovers")
            self._link_down = True
            self._motors_enabled = False

    def _probe_link(self) -> bool:
        """Cheap read to test whether the daemon link is back."""
        try:
            self.reachy.get_current_head_pose()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure_motors_enabled(self) -> bool:
        """Enable torque lazily, only when a real motion command is about to be
        sent. Starting the app with torque enabled but no user-visible motion can
        still tickle the motor bus on this Jetson setup."""
        if self._motors_enabled:
            return True
        try:
            self.reachy.enable_motors()
            self._motors_enabled = True
            return True
        except Exception:  # noqa: BLE001
            logger.warning("enable_motors() failed; motion paused", exc_info=True)
            self._mark_link_down()
            return False

    def _speaking(self) -> bool:
        if self.audio is None:
            return False
        try:
            return float(self.audio.play_rms()) > 0.01
        except Exception:  # noqa: BLE001
            return False

    def _presence_pool(self) -> list[str]:
        # Disabled in the Jetson-safe architecture. Official presence moves are
        # full-body recorded animations; they can write body/head/antennas at the
        # same time and reintroduce the twitchy multi-owner behaviour.
        return []

    def _tick_speech(self, create_head_pose) -> None:  # noqa: ANN001
        now = time.monotonic()
        if now < self._next_speech_step:
            return
        self._speech_goto_step(create_head_pose, label="speech")

    def _speech_goto_step(self, create_head_pose, label: str = "speech") -> None:  # noqa: ANN001
        """Speech/emotion owns head + antennas only, with slow non-overlapping
        goto commands. No body_yaw is sent from this path."""
        now = time.monotonic()
        if self._playing or (now - self._last_move_start) < 0.25:
            return
        scale = 1.0
        if any(word in label for word in ("excited", "happy", "surprised", "dance")):
            scale = 1.1
        yaw = random.uniform(1.8, _SPEECH_HEAD_YAW_DEG) * random.choice((-1.0, 1.0)) * scale
        pitch = random.uniform(-_SPEECH_HEAD_PITCH_DEG, _SPEECH_HEAD_PITCH_DEG) * scale
        roll = random.uniform(1.8, _SPEECH_HEAD_ROLL_DEG) * random.choice((-1.0, 1.0)) * scale
        duration = random.uniform(_SPEECH_STEP_MIN_S, _SPEECH_STEP_MAX_S)
        head = create_head_pose(
            z=random.uniform(16.0, 18.0),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mm=True,
            degrees=True,
        )
        antennas = [
            random.uniform(0.10, _SPEECH_ANTENNA_RAD) * random.choice((-1.0, 1.0)),
            random.uniform(0.10, _SPEECH_ANTENNA_RAD) * random.choice((-1.0, 1.0)),
        ]
        try:
            if not self._ensure_motors_enabled():
                return
            self._playing = True
            self._last_move_start = now
            self.reachy.goto_target(head=head, antennas=antennas, duration=duration)
            self._sent_head = self._sent_ant = None
            self._last_speech_step = now
            self._next_speech_step = now + duration
        except Exception:  # noqa: BLE001
            self._mark_link_down()
        finally:
            self._playing = False


    def _run_command(self, cmd: tuple[str, tuple]) -> None:
        """Execute a one-shot tool command on the compositor thread. Mirrors
        ``_run_expression``: sets _playing, drives the SDK, forces an ambient
        resync afterward so the deadband re-sends from the new pose."""
        kind, args = cmd
        try:
            from reachy_mini.utils import create_head_pose
        except Exception:  # noqa: BLE001
            create_head_pose = None
        try:
            if not self._ensure_motors_enabled():
                return
            self._playing = True
            self._last_move_start = time.monotonic()
            if kind == "head" and create_head_pose is not None:
                yaw, pitch, roll, duration = args
                logger.info(
                    "tool move_head -> yaw=%.1f pitch=%.1f roll=%.1f", yaw, pitch, roll
                )
                pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
                self.reachy.goto_target(head=pose, duration=duration)
            elif kind == "antennas":
                left, right, duration = args
                logger.info("tool move_antennas -> left=%.1f right=%.1f", left, right)
                self.reachy.goto_target(
                    antennas=[math.radians(right), math.radians(left)],
                    duration=duration,
                )
            elif kind == "dance":
                (name,) = args
                logger.info("tool dance '%s' -> safe head/antenna steps", name)
                if create_head_pose is None:
                    return
                for _ in range(3):
                    if self._stop.is_set():
                        break
                    self._speech_goto_step(create_head_pose, label=f"dance:{name}")
                    time.sleep(random.uniform(_SPEECH_STEP_MIN_S, _SPEECH_STEP_MAX_S))
        except ConnectionError:
            self._mark_link_down()
        except Exception:  # noqa: BLE001 — cancelled or playback error
            logger.debug("tool command %r interrupted/failed", kind, exc_info=True)
        finally:
            self._playing = False
            self._sent_head = self._sent_body = self._sent_ant = None

    def _run_expression(self, emotion: str) -> None:
        try:
            from reachy_mini.utils import create_head_pose
        except Exception:  # noqa: BLE001
            create_head_pose = None
        if create_head_pose is None:
            logger.info("emotion [%s] -> antenna wag (no head pose helper)", emotion)
            self._wag()
            return
        logger.info("emotion [%s] -> safe speech head/antenna step", emotion)
        self._speech_goto_step(create_head_pose, label=emotion)

    def _wag(self) -> None:
        center, amp, freq, phase, dur = _WAG
        steps = max(1, int(dur * _COMPOSITOR_HZ))
        dt = 1.0 / _COMPOSITOR_HZ
        for i in range(steps):
            if self._stop.is_set() or self._pending is not None:
                break
            t = i * dt
            right = center + amp * math.sin(2 * math.pi * freq * t)
            left = center + amp * math.sin(2 * math.pi * freq * t + phase)
            try:
                self.reachy.set_target_antenna_joint_positions(
                    [math.radians(right), math.radians(left)]
                )
            except Exception:  # noqa: BLE001
                break
            time.sleep(dt)
