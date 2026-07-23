"""Runtime capability probes for the unified Reachy dashboard."""

from __future__ import annotations

import socket
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Probe:
    key: str
    label: str
    kind: str
    host: str = "127.0.0.1"
    port: int | None = None
    url: str | None = None
    path: str | None = None


def tcp_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_ok(url: str, timeout: float = 0.8) -> tuple[bool, int | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500, response.status
    except Exception:
        return False, None


def default_probes(daemon_port: int = 38001, vision_http: str = "http://127.0.0.1:8630/") -> list[Probe]:
    return [
        Probe("dashboard", "Unified dashboard", "self"),
        Probe("robot_daemon", "Reachy daemon", "http", url=f"http://127.0.0.1:{daemon_port}/"),
        Probe("voice_gateway", "Voice gateway", "http", url="http://127.0.0.1:8621/health"),
        Probe("vision", "Vision stream", "http", url=vision_http),
        Probe("vision_zmq", "Vision ZMQ", "tcp", port=8631),
        Probe("face_tracker", "Face tracker", "file", path="/tmp/reachy_face_tracker_status.json"),
        Probe("edge_llm", "Edge LLM", "http", url="http://127.0.0.1:11435/v1/models"),
        Probe("openclaw", "OpenClaw", "http", url="http://127.0.0.1:18789/healthz"),
    ]


def probe_capabilities(
    *,
    device_profile: str = "jetson-full",
    daemon_port: int = 38001,
    vision_http: str = "http://127.0.0.1:8630/",
    app_state: str = "unknown",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a small capability matrix consumed by the dashboard UI.

    The dashboard should always boot; individual panels decide whether to show
    fully, degrade, or hide from this payload.
    """

    services: dict[str, dict[str, Any]] = {}
    for probe in default_probes(daemon_port=daemon_port, vision_http=vision_http):
        if probe.kind == "self":
            ok, status = True, 200
        elif probe.kind == "tcp" and probe.port is not None:
            ok, status = tcp_open(probe.host, probe.port), None
        elif probe.kind == "http" and probe.url:
            ok, status = http_ok(probe.url)
        elif probe.kind == "file" and probe.path:
            ok, status = Path(probe.path).is_file(), 200 if Path(probe.path).is_file() else None
        else:
            ok, status = False, None
        services[probe.key] = {
            "label": probe.label,
            "available": ok,
            "status": status,
            "url": probe.url,
            "port": probe.port,
            "path": probe.path,
        }

    robot = services["robot_daemon"]["available"]
    voice = services["voice_gateway"]["available"]
    vision = services["vision"]["available"]
    llm = services["edge_llm"]["available"] or services["openclaw"]["available"]
    face_tracker = services["face_tracker"]["available"]
    full_voice_app = device_profile == "jetson-full" and app_state not in {"missing", "unknown"}

    features = {
        "video": vision,
        "face_tracking": (vision and services["vision_zmq"]["available"]) or face_tracker,
        "voice_transcript": voice or full_voice_app,
        "conversation": (voice and (llm or device_profile == "pi-light")) or full_voice_app,
        "robot_motion": robot and (full_voice_app or face_tracker),
        "emotion_moves": robot and full_voice_app,
        "settings": True,
        "service_restart": full_voice_app,
        "diary": full_voice_app,
        "faces": vision,
        "voice_clone": full_voice_app,
    }

    payload: dict[str, Any] = {
        "device_profile": device_profile,
        "app_state": app_state,
        "services": services,
        "features": features,
    }
    if extra:
        payload.update(extra)
    return payload
