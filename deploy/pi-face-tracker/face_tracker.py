#!/usr/bin/env python3
"""Pi/ReComputer speech motion player.

This service is the only app-level motor writer on Pi deployments. It no longer
does face tracking. Voice and dashboard code communicate intent through
``/tmp/reachy_motion_owner`` and this service plays slow, non-overlapping
head/antenna motions. Body yaw is never written here.
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LOG_LEVEL = os.environ.get("REACHY_FACE_TRACKER_LOG_LEVEL", "INFO").upper()
DAEMON_HOST = os.environ.get("REACHY_DAEMON_HOST", "127.0.0.1")
DAEMON_PORT = int(os.environ.get("REACHY_DAEMON_PORT", "8000"))
STATUS_FILE = Path(os.environ.get("REACHY_FACE_TRACKER_STATUS", "/tmp/reachy_face_tracker_status.json"))
MOTION_OFF_FILE = Path(os.environ.get("REACHY_MOTION_OFF_FILE", "/tmp/reachy_motion_off"))
SPEAKING_FILE = Path(os.environ.get("REACHY_SPEAKING_FILE", "/tmp/reachy_speaking_on"))
OWNER_FILE = Path(os.environ.get("REACHY_MOTION_OWNER_FILE", "/tmp/reachy_motion_owner"))
FUSE_FILE = Path(os.environ.get("REACHY_MOTION_FUSE_FILE", "/tmp/reachy_motion_fuse"))

HEAD_ENABLED = os.environ.get("REACHY_HEAD_MOTION_ENABLED", "1") == "1"
ANTENNA_ENABLED = os.environ.get("REACHY_ANTENNA_MOTION_ENABLED", "1") == "1"
USE_LIBRARY = os.environ.get("REACHY_USE_MOTION_LIBRARY", "1") == "1"
OWNER_STALE_S = float(os.environ.get("REACHY_MOTION_OWNER_STALE_S", "8.0"))
SETTLE_DURATION_S = float(os.environ.get("REACHY_MOTION_SETTLE_S", "1.0"))
STEP_MIN_S = float(os.environ.get("REACHY_SPEECH_MOTION_STEP_MIN_S", "1.6"))
STEP_MAX_S = float(os.environ.get("REACHY_SPEECH_MOTION_STEP_MAX_S", "2.4"))
MAX_YAW_DEG = float(os.environ.get("REACHY_SPEECH_MAX_HEAD_YAW_DEG", "3.2"))
MAX_PITCH_DEG = float(os.environ.get("REACHY_SPEECH_MAX_HEAD_PITCH_DEG", "0.9"))
MAX_ROLL_DEG = float(os.environ.get("REACHY_SPEECH_MAX_HEAD_ROLL_DEG", "2.0"))
MAX_ANTENNA_RAD = float(os.environ.get("REACHY_SPEECH_MAX_ANTENNA_RAD", "0.18"))

EMOTION_HINTS = {
    "happy": ("happy", "joy", "smile", "excited"),
    "excited": ("excited", "happy", "joy", "surprised"),
    "surprised": ("surprised", "wow", "shock"),
    "thinking": ("thinking", "curious", "confused"),
    "sad": ("sad", "sorry", "disappointed"),
    "angry": ("angry", "mad", "annoyed"),
    "speech": ("happy", "curious", "thinking", "agree"),
}


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reachy_speech_motion")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _read_owner(now: float) -> tuple[str, float | None, str]:
    if SPEAKING_FILE.exists():
        return "expression", None, "speech"
    try:
        raw = OWNER_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "idle", None, ""
    except Exception:
        return "idle", None, ""
    if not raw:
        return "idle", None, ""
    parts = raw.split()
    owner = parts[0].strip().lower()
    label = parts[2].strip().lower() if len(parts) > 2 else ""
    try:
        ts = float(parts[1]) if len(parts) > 1 else None
    except (TypeError, ValueError):
        ts = None
    if owner == "expression":
        if ts is not None and now - ts > OWNER_STALE_S:
            return "settle", now, label
        return "expression", ts, label or "speech"
    if owner == "settle":
        if ts is not None and now - ts <= SETTLE_DURATION_S:
            return "settle", ts, label
        return "idle", ts, ""
    return "idle", ts, label


class MotionLibrary:
    def __init__(self) -> None:
        self._dance_cls = None
        self._load_error = ""
        self._loaded = False

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not USE_LIBRARY:
            self._load_error = "disabled"
            return
        try:
            from reachy_mini_dances_library.dance_move import DanceMove

            self._dance_cls = DanceMove
        except Exception as exc:  # noqa: BLE001
            self._load_error += f"dance library unavailable: {type(exc).__name__}: {exc}"
            logger.warning("dance library unavailable: %s: %s", type(exc).__name__, exc)

    def choose(self, label: str) -> Any | None:
        self.load()
        if label.startswith("dance") and self._dance_cls is not None:
            try:
                return self._dance_cls(os.environ.get("REACHY_DANCE_MOVE", "groovy_sway_and_roll"))
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"dance move unavailable: {type(exc).__name__}: {exc}"
                logger.warning(self._load_error)
        return None


class SpeechMotionPlayer:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._reachy = None
        self._create_head_pose = None
        self._connected = False
        self._mode = "booting"
        self._owner = "idle"
        self._label = ""
        self._profile = None
        self._last_error = ""
        self._last_drive = 0.0
        self._last_library_error = ""
        self._command = {
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "body_yaw": 0.0,
            "antenna_right": 0.0,
            "antenna_left": 0.0,
        }
        self._library = MotionLibrary()

    def stop(self) -> None:
        self._stop.set()
        self._close()

    def status(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "enabled": not MOTION_OFF_FILE.exists() and not FUSE_FILE.exists(),
            "face_tracking_enabled": False,
            "speaking": SPEAKING_FILE.exists(),
            "owner": self._owner,
            "expression_label": self._label or None,
            "architecture": "speech_library_motion_v1",
            "active_motor_group": "head_antennas" if self._mode == "expression" else "none",
            "motor_groups": {
                "tracking": [],
                "expression": ["head_pose", "antenna_right", "antenna_left"],
                "vision_snapshot": [],
            },
            "fused": FUSE_FILE.exists(),
            "mode": self._mode,
            "speech_profile": self._profile,
            "target": None,
            "command": dict(self._command),
            "last_drive_age_s": round(time.monotonic() - self._last_drive, 2) if self._last_drive else None,
            "last_error": self._last_error,
            "motion_library_error": self._last_library_error or None,
        }

    def run(self) -> None:
        while not self._stop.is_set():
            if MOTION_OFF_FILE.exists() or FUSE_FILE.exists():
                self._mode = "disabled"
                self._owner = "disabled"
                self._close()
                time.sleep(0.5)
                continue
            owner, _ts, label = _read_owner(time.time())
            self._owner = owner
            if owner == "expression":
                self._play_expression(label or "speech")
            elif owner == "settle":
                self._settle(label)
                time.sleep(0.2)
            else:
                self._mode = "idle"
                self._label = ""
                self._profile = None
                time.sleep(0.2)

    def _connect(self) -> bool:
        if self._reachy is not None and self._connected:
            return True
        if not self._daemon_ready():
            self._connected = False
            self._reachy = None
            return False
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
            return True
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._connected = False
            self._reachy = None
            logger.warning("connect failed: %s", self._last_error)
            return False

    def _daemon_ready(self) -> bool:
        url = f"http://{DAEMON_HOST}:{DAEMON_PORT}/api/state/full"
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status < 400:
                    return True
        except urllib.error.HTTPError as exc:
            self._last_error = f"DaemonNotReady: {exc.code} from /api/state/full"
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"DaemonNotReady: {type(exc).__name__}: {exc}"
        return False

    def _close(self) -> None:
        reachy = self._reachy
        self._reachy = None
        self._connected = False
        if reachy is not None:
            try:
                reachy.__exit__(None, None, None)
            except Exception:
                logger.debug("Reachy SDK close failed", exc_info=True)

    def _play_expression(self, label: str) -> None:
        self._mode = "expression"
        self._label = label
        if not self._connect():
            time.sleep(0.5)
            return
        move = self._library.choose(label)
        self._last_library_error = self._library.load_error
        if move is not None:
            self._profile = f"library:{label}"
            self._play_library_step(move)
        else:
            self._profile = f"fallback:{label}"
            self._play_fallback_step(label)

    def _play_library_step(self, move: Any) -> None:
        duration = max(0.8, float(getattr(move, "duration", 1.6) or 1.6))
        t = random.uniform(0.0, duration)
        head, antennas, _body_yaw = move.evaluate(t)
        antennas = self._clamp_antennas(antennas)
        step_duration = random.uniform(STEP_MIN_S, STEP_MAX_S)
        self._send(head=head, antennas=antennas, duration=step_duration)
        self._sleep_interruptible(step_duration)

    def _play_fallback_step(self, label: str) -> None:
        scale = 1.0
        if label.startswith("dance"):
            scale = 1.05
        elif any(word in label for word in ("excited", "happy", "surprised")):
            scale = 1.03
        yaw = random.uniform(1.2, MAX_YAW_DEG) * random.choice((-1.0, 1.0)) * scale
        pitch = random.uniform(-MAX_PITCH_DEG, MAX_PITCH_DEG) * scale
        roll = random.uniform(0.4, MAX_ROLL_DEG) * random.choice((-1.0, 1.0)) * scale
        head = self._create_head_pose(
            z=17.0,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            mm=True,
            degrees=True,
        )
        antennas = [
            random.uniform(0.06, MAX_ANTENNA_RAD) * random.choice((-1.0, 1.0)),
            random.uniform(0.06, MAX_ANTENNA_RAD) * random.choice((-1.0, 1.0)),
        ]
        step_duration = random.uniform(STEP_MIN_S, STEP_MAX_S)
        self._send(
            head=head,
            antennas=antennas,
            duration=step_duration,
            command_head=(yaw, pitch, roll),
        )
        self._sleep_interruptible(step_duration)

    def _settle(self, label: str) -> None:
        self._mode = "settle"
        self._label = label
        if not self._connect():
            return
        head = self._create_head_pose(z=17.0, yaw=0.0, pitch=0.0, roll=0.0, mm=True, degrees=True)
        self._send(head=head, antennas=[0.0, 0.0], duration=0.9, command_head=(0.0, 0.0, 0.0))
        self._sleep_interruptible(0.9)
        self._mode = "idle"
        self._label = ""

    def _send(
        self,
        *,
        head: Any,
        antennas: list[float],
        duration: float,
        command_head: tuple[float, float, float] | None = None,
    ) -> None:
        if self._reachy is None:
            return
        try:
            kwargs: dict[str, Any] = {"duration": duration}
            if HEAD_ENABLED:
                kwargs["head"] = head
            if ANTENNA_ENABLED:
                kwargs["antennas"] = antennas
            self._reachy.goto_target(**kwargs)
            self._update_command(head=head, antennas=antennas, command_head=command_head)
            self._last_drive = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("motion send failed: %s", self._last_error)
            try:
                FUSE_FILE.write_text(self._last_error + "\n", encoding="utf-8")
                MOTION_OFF_FILE.write_text("disabled after motor error\n", encoding="utf-8")
            except Exception:
                logger.debug("failed to write fuse files", exc_info=True)
            self._close()

    def _update_command(
        self,
        *,
        head: Any,
        antennas: list[float],
        command_head: tuple[float, float, float] | None,
    ) -> None:
        if command_head is not None:
            yaw, pitch, roll = command_head
        else:
            yaw = getattr(head, "yaw", 0.0)
            pitch = getattr(head, "pitch", 0.0)
            roll = getattr(head, "roll", 0.0)
        self._command = {
            "head_yaw": round(float(yaw or 0.0), 2),
            "head_pitch": round(float(pitch or 0.0), 2),
            "head_roll": round(float(roll or 0.0), 2),
            "body_yaw": 0.0,
            "antenna_right": round(float(antennas[0]), 3),
            "antenna_left": round(float(antennas[1]), 3),
        }

    def _clamp_antennas(self, antennas: Any) -> list[float]:
        try:
            right = float(antennas[0])
            left = float(antennas[1])
        except Exception:
            right = random.uniform(0.12, MAX_ANTENNA_RAD)
            left = random.uniform(-MAX_ANTENNA_RAD, -0.12)
        return [
            _clamp(right, -MAX_ANTENNA_RAD, MAX_ANTENNA_RAD),
            _clamp(left, -MAX_ANTENNA_RAD, MAX_ANTENNA_RAD),
        ]

    def _sleep_interruptible(self, duration: float) -> None:
        end = time.monotonic() + max(0.1, duration)
        while not self._stop.is_set() and time.monotonic() < end:
            owner, _ts, _label = _read_owner(time.time())
            if owner != "expression":
                break
            time.sleep(0.05)


class SpeechMotionApp:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._player = SpeechMotionPlayer()

    def stop(self) -> None:
        self._stop.set()
        self._player.stop()

    def run(self) -> None:
        motion_thread = threading.Thread(target=self._player.run, name="speech-motion", daemon=True)
        motion_thread.start()
        logger.info("speech motion player started: daemon=%s:%s", DAEMON_HOST, DAEMON_PORT)
        while not self._stop.is_set():
            self._write_status()
            time.sleep(0.5)

    def _write_status(self) -> None:
        status = {
            "service": "reachy-speech-motion",
            "time": time.time(),
            "vision_url": None,
            "face_count": 0,
            "faces_fresh": False,
            "last_face_age_s": None,
            "largest_face_area": 0.0,
            "motion": self._player.status(),
            "motion_off": MOTION_OFF_FILE.exists(),
            "fuse_file": FUSE_FILE.exists(),
        }
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATUS_FILE)


def main() -> None:
    app = SpeechMotionApp()
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    try:
        app.run()
    finally:
        app.stop()


if __name__ == "__main__":
    main()
