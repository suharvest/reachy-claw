"""Motion for Reachy Mini: official-move expressions + periodic "presence",
over a minimal always-alive baseline.

The SDK ships **no** idle behaviour — if you stop sending targets the robot
freezes. So we run one motion thread (single writer, nothing fights for the
head) that layers, in priority order:

  1. **Expression bursts** — the LLM ends each reply with a tag like ``[happy]``;
     we map it to one of Pollen's **official recorded moves**
     (``reachy-mini-emotions-library``, ~80 designed animations driving head +
     antennas + body together) and play it.

  2. **Periodic presence** — when nobody is interacting, every ~10-18s the robot
     spontaneously plays a *gentle* official move (a look-around / curious /
     nod / sway, drawn from the emotions **and** dances libraries) so it keeps
     "showing presence" instead of sitting still. This is the SDK's own move
     library doing the work, not hand-authored animation.

  3. **Speech wobble** — while TTS plays, the head bobs in time with the live
     speaker loudness (read off the duplex stream). Our audio bypasses the SDK
     media backend (``no_media``), so the SDK's built-in ``enable_wobbling`` can't
     see it — hence we drive a small wobble ourselves from the playback RMS.

  4. **Breathing** — a tiny continuous drift so the robot is never frozen
     between moves (the only thing the SDK gives nothing for).

Wave B will add a gaze layer via the SDK's ``look_at_image`` (face tracking).
If the official libraries can't load (offline + uncached), expressions fall back
to a simple antenna wag and presence is disabled, so motion never hard-fails.
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

_COMPOSITOR_HZ = 25.0  # ~old stable app's rate; lighter on the motor bus than 50

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

# HARD SAFETY: cap how far any joint target may move per compositor tick. At
# 50 Hz, 1.2°/tick = 60°/s — fast enough to look responsive, slow enough that a
# big gaze swing can't slam the motors and brown out the bus (which wedged the
# motor comms once). Every commanded value is slew-limited through these.
_MAX_HEAD_STEP_DEG = 2.0  # per tick @25Hz ≈ 50°/s
_MAX_BODY_STEP_DEG = 1.6  # per tick @25Hz ≈ 40°/s
# Of the horizontal gaze aim, the head turns up to this many degrees (leading the
# look); the body carries whatever is beyond it. Makes the head visibly "look".
_HEAD_YAW_SHARE_DEG = 14.0

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
        # gaze anchor (yaw, pitch in degrees) — set by the attention tracker.
        # _target is the desired aim (None = look ahead); _cur eases toward it
        # at compositor rate for smooth tracking. Horizontal aim drives body_yaw
        # (a natural "turn to face you"); vertical drives head pitch.
        self._gaze_target: tuple[float, float] | None = None
        self._gaze_cur = [0.0, 0.0]
        # last commanded targets (deg) — slew-limited each tick for motor safety
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._cmd_roll = 0.0
        self._cmd_body = 0.0
        # When the daemon link drops (e.g. a motor-bus fault), motion PAUSES and
        # probes for recovery once a second — it must never hammer the SDK at
        # tick rate and flood tracebacks, which would starve the conversation.
        self._link_down = False
        # Last targets actually SENT to the bus (deadband gating). None = unsent.
        self._sent_head: tuple[float, float, float] | None = None
        self._sent_body: float | None = None
        self._sent_ant: list[float] | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self.reachy is None:
            return
        # In SDK 1.8.0, set_target_* only drives the robot with torque ON. Enable
        # motors once here (the old 1.5 app did the same on connect). Harmless if
        # already enabled.
        try:
            self.reachy.enable_motors()
        except Exception:  # noqa: BLE001
            logger.debug("enable_motors() failed", exc_info=True)
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

    def set_gaze(self, yaw_deg: float | None, pitch_deg: float | None) -> None:
        """Aim at a tracked visitor (degrees). None → recentre. Thread-safe:
        called from the vision thread, consumed by the compositor."""
        self._gaze_target = None if yaw_deg is None else (yaw_deg, pitch_deg or 0.0)

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
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            self._libs = [RecordedMoves(EMOTIONS_DATASET), RecordedMoves(DANCES_DATASET)]
            self._move_names = set()
            for lib in self._libs:
                self._move_names |= set(lib.list_moves())
        except Exception as e:  # noqa: BLE001 — degrade to antenna wag
            logger.warning("official move libraries unavailable (%s); using wag", e)
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
                    self._gaze_cur = [0.0, 0.0]
                    self._cmd_yaw = self._cmd_pitch = self._cmd_roll = self._cmd_body = 0.0
                    self._sent_head = self._sent_body = self._sent_ant = None
                continue

            listening = self._conv_state in _FREEZE_STATES

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

            # While LISTENING: do NOT freeze (the robot should stay engaged), but
            # only allow the GENTLE, slow gaze track (look at / follow the visitor)
            # — slow servos are quiet, and the XMOS NS/AEC handles the residue. No
            # presence, no fast/large moves while listening.
            if listening:
                self._last_activity = time.monotonic()
                if create_head_pose is not None:
                    self._tick_ambient(create_head_pose, gentle=True)
                time.sleep(dt)
                continue

            now = time.monotonic()
            speaking = self._conv_state == "speaking" or self._speaking()
            tracking = self._gaze_target is not None
            if speaking or tracking:
                # someone is interacting / being followed — not "idle"
                self._last_activity = now

            # 2) periodic presence ONLY when truly idle (nobody, not speaking) —
            #    the "show off alone" behaviour; never while serving a visitor.
            if (
                self._conv_state == "idle"
                and not speaking
                and not tracking
                and self._presence_pool()
                and (now - self._last_activity) >= self._next_gap
            ):
                self._play_presence()
                self._last_activity = time.monotonic()
                self._next_gap = random.uniform(_PRESENCE_GAP_MIN, _PRESENCE_GAP_MAX)
                continue

            # 3) speech wobble (while speaking) + gaze (orient toward a visitor).
            #    No always-on breathing — when there's nothing to do this commands
            #    neutral and the deadband holds the head still (quiet servos).
            if create_head_pose is not None:
                self._tick_ambient(create_head_pose)
            time.sleep(dt)

    def _mark_link_down(self) -> None:
        """Called from a drive method when an SDK call fails. Logs ONCE on the
        up→down transition (no per-tick flood) and pauses motion."""
        if not self._link_down:
            logger.warning("motion: lost daemon link — pausing motion until it recovers")
            self._link_down = True

    def _probe_link(self) -> bool:
        """Cheap read to test whether the daemon link is back."""
        try:
            self.reachy.get_current_head_pose()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _speaking(self) -> bool:
        if self.audio is None:
            return False
        try:
            return float(self.audio.play_rms()) > 0.01
        except Exception:  # noqa: BLE001
            return False

    def _presence_pool(self) -> list[str]:
        return [m for m in _PRESENCE_MOVES if m in self._move_names]

    def _play_presence(self) -> None:
        pool = self._presence_pool()
        if not pool:
            return
        name = random.choice(pool)
        move = self._resolve(name)
        if move is None:
            return
        logger.info("presence -> official move '%s'", name)
        try:
            self._playing = True
            self._last_move_start = time.monotonic()
            self.reachy.play_move(move, initial_goto_duration=0.4, sound=False)
        except ConnectionError:  # daemon link dropped — pause, don't flood
            self._mark_link_down()
        except Exception:  # noqa: BLE001 — cancelled (someone spoke) or error
            logger.debug("presence move '%s' interrupted/failed", name, exc_info=True)
        finally:
            self._playing = False
            # the move repositioned the robot — force the next ambient send
            self._sent_head = self._sent_body = self._sent_ant = None

    def _tick_ambient(self, create_head_pose, gentle: bool = False) -> None:  # noqa: ANN001
        t = time.monotonic() - self._t0
        # No always-on breathing: the head rests at neutral unless it's speaking
        # (wobble) or following a visitor (gaze). Keeps servos quiet when idle.
        yaw = pitch = roll = 0.0
        # speech wobble — scales with live TTS loudness. Skipped in gentle mode
        # (listening) so the only motion is the slow gaze track.
        if not gentle and self.audio is not None:
            try:
                rms = float(self.audio.play_rms())
            except Exception:  # noqa: BLE001
                rms = 0.0
            if rms > 0.01:
                amp = min(1.0, rms * 6.0)
                roll += amp * 4.0 * math.sin(t * 3.0)
                pitch += amp * 3.0 * math.sin(t * 5.0 + 0.5)
                yaw += amp * 2.0 * math.sin(t * 2.0)
        # gaze: ease toward the tracked visitor (or back to centre). Gentle mode
        # (listening) eases slower → slower servos → quieter mic.
        ga = 0.05 if gentle else 0.12
        gtgt = self._gaze_target or (0.0, 0.0)
        self._gaze_cur[0] += ga * (gtgt[0] - self._gaze_cur[0])
        self._gaze_cur[1] += ga * (gtgt[1] - self._gaze_cur[1])
        # Horizontal aim is SHARED: the head turns first (up to its share, so it
        # visibly "looks" at you), the body carries the remainder. Total facing =
        # head_yaw + body_yaw = the desired aim — a natural head-leads-body turn.
        gh = self._gaze_cur[0]
        head_share = max(-_HEAD_YAW_SHARE_DEG, min(_HEAD_YAW_SHARE_DEG, gh))
        yaw += head_share
        body_yaw = gh - head_share
        pitch += self._gaze_cur[1]
        # clamp to a safe envelope
        yaw = max(-20.0, min(20.0, yaw))
        pitch = max(-20.0, min(20.0, pitch))
        roll = max(-16.0, min(16.0, roll))
        body_yaw = max(-40.0, min(40.0, body_yaw))
        # HARD SAFETY: slew-limit every target so no single tick slams a motor.
        # Gentle mode (listening) halves the step → slower, quieter servos.
        hstep = _MAX_HEAD_STEP_DEG * (0.5 if gentle else 1.0)
        bstep = _MAX_BODY_STEP_DEG * (0.5 if gentle else 1.0)
        self._cmd_yaw = _slew(self._cmd_yaw, yaw, hstep)
        self._cmd_pitch = _slew(self._cmd_pitch, pitch, hstep)
        self._cmd_roll = _slew(self._cmd_roll, roll, hstep)
        self._cmd_body = _slew(self._cmd_body, body_yaw, bstep)
        # Deadband-gated drive: only send targets that meaningfully changed since
        # the last send. Cuts serial-bus traffic ~10x when idle (the main lever
        # against motor-comms errors) without affecting visible motion. Uses the
        # same separate calls async_play_move relies on (proven to actuate).
        head = (self._cmd_yaw, self._cmd_pitch, self._cmd_roll)
        ant = _REST_ANTENNAS  # fixed relaxed pose; the deadband sends it once
        try:
            if self._sent_head is None or _max_delta(head, self._sent_head) >= _SEND_DEADBAND_DEG:
                self.reachy.set_target_head_pose(
                    create_head_pose(yaw=head[0], pitch=head[1], roll=head[2], degrees=True)
                )
                self._sent_head = head
            if self._sent_body is None or abs(self._cmd_body - self._sent_body) >= _SEND_DEADBAND_DEG:
                self.reachy.set_target_body_yaw(math.radians(self._cmd_body))
                self._sent_body = self._cmd_body
            if self._sent_ant is None or _max_delta(ant, self._sent_ant) >= _SEND_DEADBAND_ANT_RAD:
                self.reachy.set_target_antenna_joint_positions(ant)
                self._sent_ant = ant
        except Exception:  # noqa: BLE001 — link loss; pause (don't flood)
            self._mark_link_down()


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
                move = self._resolve(name)
                if move is None:
                    logger.info("tool dance '%s' -> no library move; skipped", name)
                    return
                logger.info("tool dance -> official move '%s'", name)
                self.reachy.play_move(move, initial_goto_duration=0.4, sound=False)
        except ConnectionError:
            self._mark_link_down()
        except Exception:  # noqa: BLE001 — cancelled or playback error
            logger.debug("tool command %r interrupted/failed", kind, exc_info=True)
        finally:
            self._playing = False
            self._sent_head = self._sent_body = self._sent_ant = None

    def _run_expression(self, emotion: str) -> None:
        name = self._pick_move(emotion)
        if name is not None:
            move = self._resolve(name)
            if move is not None:
                logger.info("emotion [%s] -> official move '%s'", emotion, name)
                try:
                    self._playing = True
                    self._last_move_start = time.monotonic()
                    # sound=False: the daemon's audio path would fight our duplex
                    # stream on the same USB card. initial_goto eases in from the
                    # current ambient pose so there's no snap.
                    self.reachy.play_move(move, initial_goto_duration=0.3, sound=False)
                    return
                except ConnectionError:  # daemon link dropped — pause, don't flood
                    self._mark_link_down()
                    return
                except Exception:  # noqa: BLE001 — cancelled or playback error
                    logger.debug("official move '%s' interrupted/failed", name, exc_info=True)
                    return
                finally:
                    self._playing = False
                    # the move repositioned the robot — force the next ambient send
                    self._sent_head = self._sent_body = self._sent_ant = None
        # fallback: antenna wag
        logger.info("emotion [%s] -> antenna wag (no official move)", emotion)
        self._wag()

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
