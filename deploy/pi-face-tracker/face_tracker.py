#!/usr/bin/env python3
"""Pi/ReComputer single-writer motion scheduler.

Subscribes to the vision-hailo ZMQ face stream and gently points Reachy Mini's
body/head pitch toward the locked face while the robot is quiet. While the voice
service is speaking, face tracking is paused and this scheduler drives only a
small head/antenna speech motion. This process is the only runtime writer to the
motors on Pi-class deployments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import signal
import threading
import time
from pathlib import Path
from typing import Any

from reachy_voice.attention import AttentionTracker
from reachy_voice.gaze import FaceGaze
from reachy_voice.vision import VisionClient


LOG_LEVEL = os.environ.get("REACHY_FACE_TRACKER_LOG_LEVEL", "INFO").upper()
VISION_URL = os.environ.get("REACHY_VISION_URL", "tcp://127.0.0.1:8631")
DAEMON_HOST = os.environ.get("REACHY_DAEMON_HOST", "127.0.0.1")
DAEMON_PORT = int(os.environ.get("REACHY_DAEMON_PORT", "8000"))
STATUS_FILE = Path(os.environ.get("REACHY_FACE_TRACKER_STATUS", "/tmp/reachy_face_tracker_status.json"))
DISABLE_FILE = Path(os.environ.get("REACHY_FACE_TRACKING_OFF", "/tmp/reachy_face_tracking_off"))
MOTION_OFF_FILE = Path(os.environ.get("REACHY_MOTION_OFF_FILE", "/tmp/reachy_motion_off"))
SPEAKING_FILE = Path(os.environ.get("REACHY_SPEAKING_FILE", "/tmp/reachy_speaking_on"))
FUSE_FILE = Path(os.environ.get("REACHY_MOTION_FUSE_FILE", "/tmp/reachy_motion_fuse"))

COMPOSITOR_HZ = float(os.environ.get("REACHY_FACE_TRACKER_HZ", "12"))
SEND_DEADBAND_DEG = float(os.environ.get("REACHY_FACE_SEND_DEADBAND_DEG", "0.25"))
MAX_HEAD_STEP_DEG = float(os.environ.get("REACHY_FACE_MAX_HEAD_STEP_DEG", "0.28"))
MAX_BODY_STEP_DEG = float(os.environ.get("REACHY_FACE_MAX_BODY_STEP_DEG", "0.25"))
MAX_ANTENNA_STEP_RAD = float(os.environ.get("REACHY_FACE_MAX_ANTENNA_STEP_RAD", "0.025"))
TRACK_HEAD_YAW_SHARE_DEG = float(os.environ.get("REACHY_TRACK_HEAD_YAW_SHARE_DEG", "5"))
MAX_HEAD_YAW_DEG = float(os.environ.get("REACHY_FACE_MAX_HEAD_YAW_DEG", "12"))
MAX_HEAD_PITCH_DEG = float(os.environ.get("REACHY_FACE_MAX_HEAD_PITCH_DEG", "10"))
MAX_HEAD_ROLL_DEG = float(os.environ.get("REACHY_FACE_MAX_HEAD_ROLL_DEG", "6"))
MAX_BODY_YAW_DEG = float(os.environ.get("REACHY_FACE_MAX_BODY_YAW_DEG", "18"))
SPEECH_HEAD_ENABLED = os.environ.get("REACHY_SPEECH_HEAD_ENABLED", "1") == "1"
SPEECH_PITCH_DEG = float(os.environ.get("REACHY_SPEECH_PITCH_DEG", "1.4"))
SPEECH_ROLL_DEG = float(os.environ.get("REACHY_SPEECH_ROLL_DEG", "0.8"))
SPEECH_YAW_DEG = float(os.environ.get("REACHY_SPEECH_YAW_DEG", "1.2"))
SPEECH_ANTENNA_RAD = float(os.environ.get("REACHY_SPEECH_ANTENNA_RAD", "0.16"))
SPEECH_POSE_MIN_S = float(os.environ.get("REACHY_SPEECH_POSE_MIN_S", "0.85"))
SPEECH_POSE_MAX_S = float(os.environ.get("REACHY_SPEECH_POSE_MAX_S", "1.6"))
ANTENNA_SEND_DEADBAND_RAD = float(os.environ.get("REACHY_ANTENNA_SEND_DEADBAND_RAD", "0.012"))
SPEECH_LIBRARY = (
    "curious",
    "agree",
    "bright",
    "thoughtful",
    "calm",
)


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reachy_face_tracker")


def _slew(cur: float, target: float, max_step: float) -> float:
    delta = target - cur
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return cur + delta


def _max_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _bbox_area(face: dict[str, Any]) -> float:
    try:
        x1, y1, x2, y2 = face.get("bbox") or [0, 0, 0, 0]
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    except Exception:
        return 0.0


class FaceTrackMotion:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None
        self._cur = [0.0, 0.0]
        self._cmd_yaw = 0.0
        self._cmd_pitch = 0.0
        self._cmd_roll = 0.0
        self._cmd_body = 0.0
        self._cmd_ant = [0.0, 0.0]
        self._sent_head: tuple[float, float, float] | None = None
        self._sent_body: float | None = None
        self._sent_ant: tuple[float, float] | None = None
        self._reachy = None
        self._create_head_pose = None
        self._last_error = ""
        self._last_drive = 0.0
        self._connected = False
        self._fused = False
        self._mode = "booting"
        self._speaking = False
        self._speech_next_pose_at = 0.0
        self._speech_profile = "calm"
        self._speech_body_hold = 0.0
        self._speech_goal_head = (0.0, 0.0, 0.0)
        self._speech_goal_ant = (0.0, 0.0)

    def set_gaze(self, yaw_deg: float | None, pitch_deg: float | None) -> None:
        with self._lock:
            self._target = None if yaw_deg is None else (float(yaw_deg), float(pitch_deg or 0.0))

    def stop(self) -> None:
        self._stop.set()
        reachy = self._reachy
        self._reachy = None
        self._connected = False
        if reachy is not None:
            try:
                reachy.__exit__(None, None, None)
            except Exception:
                logger.debug("Reachy SDK close failed", exc_info=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            target = self._target
        return {
            "connected": self._connected,
            "enabled": not MOTION_OFF_FILE.exists() and not FUSE_FILE.exists(),
            "face_tracking_enabled": not DISABLE_FILE.exists(),
            "speaking": SPEAKING_FILE.exists(),
            "fused": self._fused or FUSE_FILE.exists(),
            "mode": self._mode,
            "speech_profile": self._speech_profile if self._speaking else None,
            "target": None if target is None else {"yaw": round(target[0], 2), "pitch": round(target[1], 2)},
            "command": {
                "head_yaw": round(self._cmd_yaw, 2),
                "head_pitch": round(self._cmd_pitch, 2),
                "head_roll": round(self._cmd_roll, 2),
                "body_yaw": round(self._cmd_body, 2),
                "antenna_right": round(self._cmd_ant[0], 3),
                "antenna_left": round(self._cmd_ant[1], 3),
            },
            "last_drive_age_s": round(time.monotonic() - self._last_drive, 2) if self._last_drive else None,
            "last_error": self._last_error,
        }

    def run(self) -> None:
        dt = 1.0 / max(1.0, COMPOSITOR_HZ)
        while not self._stop.is_set():
            if self._reachy is None:
                self._connect()
                if self._reachy is None:
                    time.sleep(2.0)
                    continue
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._connected = False
                self._fused = True
                try:
                    FUSE_FILE.write_text(self._last_error + "\n", encoding="utf-8")
                    MOTION_OFF_FILE.write_text("disabled after motor error\n", encoding="utf-8")
                except Exception:
                    logger.debug("failed to write motion fuse files", exc_info=True)
                logger.warning("drive failed: %s", self._last_error)
                self._reachy = None
                self._sent_head = self._sent_body = self._sent_ant = None
                time.sleep(1.0)
            time.sleep(dt)

    def _connect(self) -> None:
        try:
            from reachy_mini import ReachyMini
            from reachy_mini.utils import create_head_pose

            self._reachy = ReachyMini(
                host=DAEMON_HOST,
                port=DAEMON_PORT,
                connection_mode="localhost_only",
                spawn_daemon=False,
                media_backend="no_media",
                timeout=8,
            )
            self._create_head_pose = create_head_pose
            try:
                self._reachy.enable_motors()
            except Exception:
                logger.debug("enable_motors failed", exc_info=True)
            self._connected = True
            self._last_error = ""
            logger.info("connected to Reachy daemon at %s:%s", DAEMON_HOST, DAEMON_PORT)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._connected = False
            self._reachy = None
            logger.warning("connect failed: %s", self._last_error)

    def _tick(self) -> None:
        with self._lock:
            target = self._target
        disabled = MOTION_OFF_FILE.exists() or FUSE_FILE.exists()
        if disabled:
            target = None
            speaking = False
        else:
            speaking = SPEAKING_FILE.exists()
        if DISABLE_FILE.exists():
            target = None

        if speaking:
            t = time.monotonic()
            self._mode = "speaking"
            self._cur[0] += 0.04 * (0.0 - self._cur[0])
            self._cur[1] += 0.04 * (0.0 - self._cur[1])
            if not self._speaking:
                self._start_speech_expression(t)
            head_yaw, head_pitch, head_roll, ant_right, ant_left = self._speech_expression(t)
            body_yaw = self._speech_body_hold
        else:
            self._speaking = False
            goal_yaw, goal_pitch = target or (0.0, 0.0)
            alpha = 0.10 if target is not None else 0.05
            self._cur[0] += alpha * (goal_yaw - self._cur[0])
            self._cur[1] += alpha * (goal_pitch - self._cur[1])
            self._mode = "tracking" if target is not None and not disabled else "idle"
            head_yaw = max(-TRACK_HEAD_YAW_SHARE_DEG, min(TRACK_HEAD_YAW_SHARE_DEG, self._cur[0]))
            body_yaw = self._cur[0] - head_yaw
            head_pitch = self._cur[1]
            head_roll = 0.0
            ant_right = 0.0
            ant_left = 0.0
        if disabled:
            self._mode = "disabled"

        head_yaw = max(-MAX_HEAD_YAW_DEG, min(MAX_HEAD_YAW_DEG, head_yaw))
        head_pitch = max(-MAX_HEAD_PITCH_DEG, min(MAX_HEAD_PITCH_DEG, head_pitch))
        head_roll = max(-MAX_HEAD_ROLL_DEG, min(MAX_HEAD_ROLL_DEG, head_roll))
        body_yaw = max(-MAX_BODY_YAW_DEG, min(MAX_BODY_YAW_DEG, body_yaw))

        self._cmd_yaw = _slew(self._cmd_yaw, head_yaw, MAX_HEAD_STEP_DEG)
        self._cmd_pitch = _slew(self._cmd_pitch, head_pitch, MAX_HEAD_STEP_DEG)
        self._cmd_roll = _slew(self._cmd_roll, head_roll, MAX_HEAD_STEP_DEG)
        self._cmd_body = _slew(self._cmd_body, body_yaw, MAX_BODY_STEP_DEG)
        self._cmd_ant[0] = _slew(self._cmd_ant[0], ant_right, MAX_ANTENNA_STEP_RAD)
        self._cmd_ant[1] = _slew(self._cmd_ant[1], ant_left, MAX_ANTENNA_STEP_RAD)

        head = (self._cmd_yaw, self._cmd_pitch, self._cmd_roll)
        antennas = (self._cmd_ant[0], self._cmd_ant[1])
        sent = False
        if self._sent_head is None or _max_delta(head, self._sent_head) >= SEND_DEADBAND_DEG:
            pose = self._create_head_pose(yaw=head[0], pitch=head[1], roll=head[2], degrees=True)
            self._reachy.set_target_head_pose(pose)
            self._sent_head = head
            sent = True
        if self._sent_body is None or abs(self._cmd_body - self._sent_body) >= SEND_DEADBAND_DEG:
            self._reachy.set_target_body_yaw(math.radians(self._cmd_body))
            self._sent_body = self._cmd_body
            sent = True
        if self._sent_ant is None or _max_delta(antennas, self._sent_ant) >= ANTENNA_SEND_DEADBAND_RAD:
            self._reachy.set_target_antenna_joint_positions([antennas[0], antennas[1]])
            self._sent_ant = antennas
            sent = True
        if sent:
            self._last_drive = time.monotonic()

    def _start_speech_expression(self, now: float) -> None:
        self._speaking = True
        self._speech_profile = random.choice(SPEECH_LIBRARY)
        self._speech_body_hold = self._cmd_body
        self._speech_next_pose_at = 0.0
        self._choose_speech_pose(now)

    def _speech_expression(self, now: float) -> tuple[float, float, float, float, float]:
        if now >= self._speech_next_pose_at:
            self._choose_speech_pose(now)
        if SPEECH_HEAD_ENABLED:
            head_yaw, head_pitch, head_roll = self._speech_goal_head
        else:
            head_yaw, head_pitch, head_roll = self._cmd_yaw, self._cmd_pitch, self._cmd_roll
        ant_right, ant_left = self._speech_goal_ant
        return head_yaw, head_pitch, head_roll, ant_right, ant_left

    def _choose_speech_pose(self, now: float) -> None:
        profile = random.choice(SPEECH_LIBRARY)
        self._speech_profile = profile
        yaw_sign = random.choice((-1.0, 1.0))
        roll_sign = random.choice((-1.0, 1.0))
        if profile == "agree":
            head = (
                0.25 * yaw_sign * SPEECH_YAW_DEG,
                random.uniform(-0.2, 1.0) * SPEECH_PITCH_DEG,
                0.25 * roll_sign * SPEECH_ROLL_DEG,
            )
            antennas = (0.55 * SPEECH_ANTENNA_RAD, 0.85 * SPEECH_ANTENNA_RAD)
        elif profile == "curious":
            head = (
                0.55 * yaw_sign * SPEECH_YAW_DEG,
                random.uniform(-0.8, 0.15) * SPEECH_PITCH_DEG,
                0.75 * roll_sign * SPEECH_ROLL_DEG,
            )
            antennas = (0.9 * SPEECH_ANTENNA_RAD, 0.35 * SPEECH_ANTENNA_RAD)
        elif profile == "bright":
            head = (
                0.35 * yaw_sign * SPEECH_YAW_DEG,
                -0.65 * SPEECH_PITCH_DEG,
                0.25 * roll_sign * SPEECH_ROLL_DEG,
            )
            antennas = (SPEECH_ANTENNA_RAD, 0.8 * SPEECH_ANTENNA_RAD)
        elif profile == "thoughtful":
            head = (
                0.45 * yaw_sign * SPEECH_YAW_DEG,
                0.65 * SPEECH_PITCH_DEG,
                0.65 * roll_sign * SPEECH_ROLL_DEG,
            )
            antennas = (-0.25 * SPEECH_ANTENNA_RAD, 0.55 * SPEECH_ANTENNA_RAD)
        else:
            head = (
                random.uniform(-0.25, 0.25) * SPEECH_YAW_DEG,
                random.uniform(-0.2, 0.35) * SPEECH_PITCH_DEG,
                random.uniform(-0.25, 0.25) * SPEECH_ROLL_DEG,
            )
            antennas = (0.35 * SPEECH_ANTENNA_RAD, 0.35 * SPEECH_ANTENNA_RAD)
        self._speech_goal_head = head
        self._speech_goal_ant = antennas
        self._speech_next_pose_at = now + random.uniform(SPEECH_POSE_MIN_S, SPEECH_POSE_MAX_S)


class FaceTrackerApp:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._motion = FaceTrackMotion()
        self._vision = VisionClient(VISION_URL)
        self._face_count = 0
        self._last_faces = 0.0
        self._last_target_area = 0.0
        fg = FaceGaze()
        self._attention = AttentionTracker(
            on_gaze=self._motion.set_gaze,
            on_engage=lambda: None,
            min_area=float(os.environ.get("REACHY_ATTENTION_MIN_AREA", "0.018")),
            stable_s=float(os.environ.get("REACHY_ATTENTION_STABLE_S", "1.2")),
            cooldown_s=float(os.environ.get("REACHY_ATTENTION_COOLDOWN_S", "15")),
            max_yaw=float(os.environ.get("REACHY_GAZE_MAX_YAW", "35")),
            max_pitch=float(os.environ.get("REACHY_GAZE_MAX_PITCH", "20")),
            lost_s=float(os.environ.get("REACHY_GAZE_LOST_S", "3.5")),
            deadzone=float(os.environ.get("REACHY_GAZE_DEADZONE", "0.03")),
            invert_x=os.environ.get("REACHY_GAZE_INVERT_X", "0") == "1",
            invert_y=os.environ.get("REACHY_GAZE_INVERT_Y", "0") == "1",
            gaze_fn=fg.yaw_pitch if fg.ok else None,
        )

    def stop(self) -> None:
        self._stop.set()
        self._motion.stop()
        self._vision.stop()

    def run(self) -> None:
        self._vision.set_listener(self._on_faces)
        if not self._vision.start():
            raise RuntimeError("vision client failed to start")
        motion_thread = threading.Thread(target=self._motion.run, name="face-track-motion", daemon=True)
        motion_thread.start()
        logger.info("face tracker started: vision=%s daemon=%s:%s", VISION_URL, DAEMON_HOST, DAEMON_PORT)
        while not self._stop.is_set():
            self._write_status()
            time.sleep(1.0)

    def _on_faces(self, payload: dict[str, Any]) -> None:
        faces = payload.get("faces") or []
        self._face_count = len(faces)
        if faces:
            self._last_faces = time.monotonic()
            self._last_target_area = max(_bbox_area(face) for face in faces)
        if DISABLE_FILE.exists():
            self._motion.set_gaze(None, None)
            return
        self._attention.update(faces)

    def _write_status(self) -> None:
        status = {
            "service": "reachy-motion-scheduler",
            "time": time.time(),
            "vision_url": VISION_URL,
            "face_count": self._face_count,
            "faces_fresh": (time.monotonic() - self._last_faces) < 2.5 if self._last_faces else False,
            "last_face_age_s": round(time.monotonic() - self._last_faces, 2) if self._last_faces else None,
            "largest_face_area": round(self._last_target_area, 4),
            "motion": self._motion.status(),
            "motion_off": MOTION_OFF_FILE.exists(),
            "fuse_file": FUSE_FILE.exists(),
        }
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATUS_FILE)


def main() -> None:
    app = FaceTrackerApp()
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    try:
        app.run()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
