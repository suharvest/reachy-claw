#!/usr/bin/env python3
"""Unified Reachy dashboard adapter for lightweight devices.

This serves the same ``src/reachy_voice/static`` UI as the full Jetson app, but
maps whatever is available on a Pi-class device into the dashboard contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(os.environ.get("REACHY_REPO_ROOT", Path(__file__).resolve().parents[2]))
STATIC_DIR = Path(os.environ.get("REACHY_DASHBOARD_STATIC", ROOT / "src/reachy_voice/static"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from reachy_voice.capabilities import probe_capabilities
    from reachy_voice import model_config
except Exception:  # pragma: no cover - keeps the adapter usable when copied alone
    probe_capabilities = None
    model_config = None


DEVICE_PROFILE = os.environ.get("REACHY_DEVICE_PROFILE", "pi-light")
VOICE_SERVICE = os.environ.get("REACHY_VOICE_SERVICE", "reachy-mini-wake-qwen.service")
VISION_BASE = os.environ.get("REACHY_VISION_BASE", "http://127.0.0.1:8630")
DAEMON_PORT = int(os.environ.get("REACHY_DAEMON_PORT", "8000"))
FACE_TRACKER_STATUS = Path(os.environ.get("REACHY_FACE_TRACKER_STATUS", "/tmp/reachy_face_tracker_status.json"))
FACE_TRACKING_OFF = Path(os.environ.get("REACHY_FACE_TRACKING_OFF", "/tmp/reachy_face_tracking_off"))

app = FastAPI(title="Reachy Unified Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _http_json(url: str, timeout: float = 2.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _tail_voice_log(lines: int = 320) -> str:
    lines = max(20, min(int(lines or 320), 700))
    result = subprocess.run(
        ["journalctl", "--no-pager", "-u", VOICE_SERVICE, "-n", str(lines)],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    text = result.stdout or result.stderr or ""
    interesting = (
        "WAKE_", "QWEN_", "ASR:", "ASSISTANT:", "USER_", "VISION_",
        "DANCE_", "EMOTION_", "PLAYBACK_", "STARTUP_", "BACK_TO_WAKE",
    )
    keep = [line for line in text.splitlines() if any(token in line for token in interesting)]
    return "\n".join(keep[-lines:])


def _capabilities() -> dict[str, Any]:
    if probe_capabilities is not None:
        return probe_capabilities(
            device_profile=DEVICE_PROFILE,
            daemon_port=DAEMON_PORT,
            vision_http=f"{VISION_BASE}/",
            app_state="adapter",
            extra={"adapter": "unified-dashboard"},
        )
    return {
        "device_profile": DEVICE_PROFILE,
        "app_state": "adapter",
        "services": {},
        "features": {"video": True, "voice_transcript": True, "conversation": True},
    }


def _model_config() -> dict[str, Any]:
    if model_config is None:
        return {"mode": "local", "online": {}, "local": {}}
    return model_config.redacted_config()


def _save_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    if model_config is None:
        raise RuntimeError("model config module unavailable")
    return model_config.save_config(payload)


def _restart_voice_service() -> dict[str, Any]:
    command = ["sudo", "-n", "systemctl", "restart", VOICE_SERVICE]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    if result.returncode != 0 and ("sudo" in result.stderr.lower() or "password" in result.stderr.lower()):
        command = ["systemctl", "restart", VOICE_SERVICE]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    return {
        "ok": result.returncode == 0,
        "command": " ".join(command),
        "returncode": result.returncode,
        "stderr": result.stderr[-1200:],
    }


def _face_tracking_status() -> dict[str, Any]:
    if FACE_TRACKER_STATUS.is_file():
        try:
            data = json.loads(FACE_TRACKER_STATUS.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            data = {"service": "reachy-face-tracker", "error": f"{type(exc).__name__}: {exc}"}
    else:
        data = {"service": "reachy-face-tracker", "available": False}
    data["enabled"] = not FACE_TRACKING_OFF.exists()
    data["available"] = FACE_TRACKER_STATUS.is_file()
    return data


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/status")
def status() -> dict[str, str]:
    return {"app": "reachy-unified-dashboard", "state": "adapter", "profile": DEVICE_PROFILE}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return _capabilities()


@app.get("/api/face-tracking")
def face_tracking_status() -> dict[str, Any]:
    return _face_tracking_status()


@app.post("/api/face-tracking")
async def set_face_tracking(payload: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(payload.get("enabled", True))
    if enabled:
        try:
            FACE_TRACKING_OFF.unlink()
        except FileNotFoundError:
            pass
    else:
        FACE_TRACKING_OFF.write_text("disabled by dashboard\n", encoding="utf-8")
    return {"ok": True, "status": _face_tracking_status()}


@app.get("/api/runtime-log", response_class=PlainTextResponse)
def runtime_log(lines: int = 320) -> str:
    return _tail_voice_log(lines)


@app.get("/api/model-config")
def get_model_config() -> dict[str, Any]:
    return _model_config()


@app.post("/api/model-config")
async def save_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "config": _save_model_config(payload)}


@app.post("/api/model-switch")
async def switch_model_mode(payload: dict[str, Any]) -> JSONResponse:
    mode = str(payload.get("mode", "")).strip()
    if mode not in {"local", "online"}:
        return JSONResponse({"ok": False, "error": "mode must be local or online"}, status_code=400)
    cfg = _save_model_config({"mode": mode})
    restart = bool(payload.get("restart", False))
    restart_result = _restart_voice_service() if restart else {"ok": None, "skipped": True}
    return JSONResponse({
        "ok": True,
        "config": cfg,
        "restart": restart_result,
        "note": "mode saved; active conversations use the new provider on the next wake/session",
    })


@app.get("/api/vision")
def vision_status() -> Any:
    try:
        return {"ok": True, "body": _http_json(f"{VISION_BASE}/")}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=503)


@app.get("/stream")
def stream() -> StreamingResponse:
    def gen():
        with urllib.request.urlopen(f"{VISION_BASE}/stream", timeout=15) as response:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/captures/count")
def capture_count() -> Any:
    try:
        return _http_json(f"{VISION_BASE}/api/captures/count")
    except Exception:
        return {"count": 0}


@app.get("/api/captures/list")
def capture_list() -> Any:
    try:
        return _http_json(f"{VISION_BASE}/api/captures/list")
    except Exception:
        return {"captures": []}


@app.get("/api/ollama/models")
def ollama_models() -> dict[str, Any]:
    return {"models": [], "source": "unavailable"}


@app.get("/api/settings/{namespace}")
def settings_unavailable(namespace: str) -> JSONResponse:
    return JSONResponse({"error": f"{namespace} settings unavailable on {DEVICE_PROFILE}"}, status_code=503)


@app.websocket("/ws")
async def dashboard_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "state", "state": "listening"})
    await ws.send_json({
        "type": "robot_state",
        "mode": "adapter",
        "llm_backend": "qwen-online",
        "ollama_model": "",
        "ollama_url": "",
        "silero_threshold": 0.0,
        "vlm_enabled": True,
        "barge_in_enabled": False,
        "capture_count": 0,
    })
    await ws.send_json({"type": "conversation_language", "language": "zh", "asr_language": "zh", "tts_language": "zh"})

    last_log = ""
    last_event = ""
    while True:
        try:
            log = await asyncio.to_thread(_tail_voice_log, 320)
            if log != last_log:
                last_log = log
                await ws.send_json({"type": "runtime_log", "text": log})
                for line in log.splitlines()[-8:]:
                    event = _line_to_dashboard_event(line)
                    if event and event != last_event:
                        last_event = event
                        await ws.send_json(event)
            await asyncio.sleep(1.5)
        except WebSocketDisconnect:
            break


def _line_to_dashboard_event(line: str) -> dict[str, Any] | None:
    if "WAKE_DETECTED" in line:
        return {"type": "state", "state": "thinking"}
    m = re.search(r"ASR:\s*(.+)$", line)
    if m:
        return {"type": "asr_final", "text": m.group(1).strip()}
    m = re.search(r"WAKE_ASR:\s*(.+)$", line)
    if m:
        return {"type": "asr_partial", "text": m.group(1).strip()}
    m = re.search(r"ASSISTANT:\s*(.+)$", line)
    if m:
        run_id = f"adapter-{int(time.time())}"
        text = m.group(1).strip()
        return {"type": "llm_end", "run_id": run_id, "full_text": text, "emotion": "neutral"}
    if "QWEN_ACTIVE" in line:
        return {"type": "state", "state": "listening"}
    if "BACK_TO_WAKE" in line:
        return {"type": "state", "state": "idle"}
    return None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8042")))
