"""Vision producer skeleton — drop-in replacement for vision-trt.

Run alongside reachy-claw to feed it face/emotion telemetry without needing
the NVIDIA-specific vision-trt container. Bring your own inference (Hailo,
Coral, RKNN, MediaPipe, ONNX-CPU — anything that produces bboxes + emotions).

Wire-compatible with reachy-claw's VisionClientPlugin and dashboard. Listens on:
  - tcp://0.0.0.0:8631  ZMQ PUB topic "vision"  (msgpack)
  - http://0.0.0.0:8630 HTTP API + MJPEG /stream

Fill in the three TODO_* functions and you're done.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import msgpack
import numpy as np
import zmq
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
# Config (override via env vars)
# ─────────────────────────────────────────────────────────────────────────────
ZMQ_PUB_PORT = int(os.environ.get("ZMQ_PUB_PORT", 8631))
HTTP_PORT = int(os.environ.get("HTTP_PORT", 8630))
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "/dev/video0")
CAMERA_W = int(os.environ.get("CAMERA_W", 640))
CAMERA_H = int(os.environ.get("CAMERA_H", 480))
CAPTURE_DIR = Path(os.environ.get("CAPTURE_DIR", "/app/data/captures"))
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FPS = float(os.environ.get("TARGET_FPS", 15))
FACE_DB_DIR = Path(os.environ.get("FACE_DB_DIR", "/app/data/faces"))
FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
PER_IDENTITY_COOLDOWN = float(os.environ.get("PER_IDENTITY_COOLDOWN", 30.0))
ANONYMOUS_COOLDOWN = float(os.environ.get("ANONYMOUS_COOLDOWN", 5.0))

# ─────────────────────────────────────────────────────────────────────────────
# 1) TODO_INIT — load your models here (called once at startup)
# ─────────────────────────────────────────────────────────────────────────────
def TODO_init_models() -> dict:
    """Load detection / emotion / (optional) embedding models.

    Returns a dict you'll get back in TODO_infer_frame as `models`.
    Examples: {"detector": ..., "emotion": ..., "face_db": ...}
    """
    # Example with MediaPipe (pip install mediapipe):
    #   import mediapipe as mp
    #   det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    #   return {"detector": det}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 2) TODO_INFER — produce face list from one BGR frame (called per frame)
# ─────────────────────────────────────────────────────────────────────────────
def TODO_infer_frame(frame_bgr: np.ndarray, models: dict) -> list[dict]:
    """Run inference on a single frame.

    Coordinates MUST be normalized — reachy-claw and dashboard both assume this.
    Frame size is in `frame_bgr.shape` (h, w, 3) if you need to convert from pixels.

    Return a list of face dicts:

        {
            # REQUIRED for head tracking — without these, the robot won't follow you
            "center": [x, y],                  # normalized [-1, 1], 0 = frame center
            "bbox":   [x1, y1, x2, y2],        # normalized [0, 1] — REQUIRED

            # Strongly recommended
            "confidence": 0.0-1.0,
            "emotion": "happy"|"sad"|"angry"|"surprised"|"fear"|"neutral"|"disgust",
            "emotion_confidence": 0.0-1.0,

            # Optional — needed only if you do face recognition
            "identity": "name" | None,
            "identity_distance": 0.0-1.0,
            "embedding": [128 floats] | None,  # for /api/faces/enroll only

            # Optional — landmarks[0]=left_eye, [1]=right_eye used for head roll
            "landmarks": [[x, y], ...],         # normalized [0, 1], 5 SCRFD points
        }

    Pixel→normalized helper:
        h, w = frame_bgr.shape[:2]
        cx_px, cy_px = (x1+x2)/2, (y1+y2)/2
        center = [(cx_px / w) * 2 - 1, (cy_px / h) * 2 - 1]   # [-1, 1]
        bbox_n = [x1 / w, y1 / h, x2 / w, y2 / h]             # [0, 1]

    Empty list = no faces (totally fine).
    """
    # Stub — return nothing so reachy-claw treats it as "no face visible"
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 3) TODO_DETECT_SMILE — opt-in: trigger smile_capture event per face
# ─────────────────────────────────────────────────────────────────────────────
def TODO_should_capture_smile(face: dict, frame_bgr: np.ndarray) -> bool:
    """Return True when THIS FACE is worth saving (e.g. peak smile).

    Called per-face with per-identity dedup. The runner handles cooldown.
    Default: never capture.

    Args:
        face: Single face dict with center, bbox, emotion, confidence, identity
        frame_bgr: Original frame for optional advanced detection

    Returns:
        True if this face should trigger a capture (runner will check cooldown)
    """
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Below this line you shouldn't need to touch anything.
# ═════════════════════════════════════════════════════════════════════════════

class _State:
    last_frame: np.ndarray | None = None
    last_jpeg: bytes | None = None
    capture_count: int = 0
    last_capture_t: float = 0.0
    fps: float = 0.0
    inference_ms: float = 0.0
    face_db: Any | None = None  # FaceDatabase if available
    # Per-identity smile dedup
    identity_last_capture: dict[str, float] = {}
    anonymous_last_capture: float = 0.0


state = _State()
_capture_lock = threading.Lock()  # guards capture_count and CAPTURE_DIR writes


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {CAMERA_DEVICE}")
    return cap


def _existing_capture_count() -> int:
    return sum(1 for f in CAPTURE_DIR.iterdir() if f.suffix == ".jpg")


def _capture_loop() -> None:
    """Camera → inference → ZMQ PUB. Runs forever in a thread."""
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://0.0.0.0:{ZMQ_PUB_PORT}")
    print(f"[vision-stub] ZMQ PUB on tcp://0.0.0.0:{ZMQ_PUB_PORT}", flush=True)

    models = TODO_init_models()
    state.capture_count = _existing_capture_count()

    # Initialize face_db if face_db.py is importable
    try:
        from face_db import FaceDatabase
        state.face_db = FaceDatabase(str(FACE_DB_DIR))
        print(f"[vision-stub] Face DB initialized at {FACE_DB_DIR}", flush=True)
    except ImportError:
        state.face_db = None
        print("[vision-stub] Face DB not available (no face_db.py)", flush=True)

    cap = _open_camera()
    print(f"[vision-stub] camera {CAMERA_DEVICE} open at {CAMERA_W}x{CAMERA_H}", flush=True)

    frame_id = 0
    frame_dt = 1.0 / TARGET_FPS
    fps_t0 = time.monotonic()
    fps_count = 0

    while True:
        loop_start = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        t0 = time.monotonic()
        try:
            faces = TODO_infer_frame(frame, models)
        except Exception as e:
            print(f"[vision-stub] inference error: {e}", flush=True)
            faces = []
        state.inference_ms = (time.monotonic() - t0) * 1000

        # Smile capture with per-identity dedup
        capture_events: list[dict] = []
        now = time.monotonic()

        for face in faces:
            # Check if this face should trigger capture
            if TODO_should_capture_smile(face, frame):
                identity = face.get("identity")

                # Per-identity cooldown check
                if identity:
                    last = state.identity_last_capture.get(identity, 0)
                    if now - last < PER_IDENTITY_COOLDOWN:
                        continue  # Skip, cooldown not expired
                else:
                    if now - state.anonymous_last_capture < ANONYMOUS_COOLDOWN:
                        continue  # Skip, anonymous cooldown

                # Capture this face
                with _capture_lock:
                    state.capture_count += 1
                    count_now = state.capture_count
                    fname = f"smile_{int(time.time()*1000)}_{count_now}.jpg"

                    # Crop face with 20% padding
                    bbox = face.get("bbox", [0.3, 0.2, 0.7, 0.8])
                    h, w = frame.shape[:2]
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
                    pad_w = int((x2 - x1) * 0.2)
                    pad_h = int((y2 - y1) * 0.2)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)
                    crop = frame[y1:y2, x1:x2]

                    cv2.imwrite(str(CAPTURE_DIR / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

                    # Update cooldown timestamp
                    if identity:
                        state.identity_last_capture[identity] = now
                    else:
                        state.anonymous_last_capture = now

                capture_events.append({"event": "smile", "count": count_now, "file": fname, "identity": identity})

        # PUB to reachy-claw
        zmq_msg: dict[str, Any] = {"frame_id": frame_id, "faces": faces}
        if capture_events:
            zmq_msg["capture"] = capture_events[0]
            zmq_msg["new_captures"] = len(capture_events)
        pub.send_multipart([b"vision", msgpack.packb(zmq_msg, use_bin_type=True)])

        # Update preview JPEG for /stream
        ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok2:
            state.last_jpeg = jpg.tobytes()
        state.last_frame = frame

        # FPS bookkeeping
        fps_count += 1
        if fps_count >= 30:
            state.fps = fps_count / (time.monotonic() - fps_t0)
            fps_count = 0
            fps_t0 = time.monotonic()

        frame_id += 1
        sleep = frame_dt - (time.monotonic() - loop_start)
        if sleep > 0:
            time.sleep(sleep)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API — drop-in compatible with vision-trt (dashboard expects these)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="vision-stub")


@app.get("/")
async def root():
    return {"service": "vision-stub", "fps": round(state.fps, 1)}


@app.get("/api/captures/count")
async def captures_count():
    return {"count": state.capture_count}


@app.get("/api/captures/list")
async def captures_list(limit: int = Query(200), offset: int = Query(0)):
    files = sorted(
        [f.name for f in CAPTURE_DIR.iterdir() if f.suffix == ".jpg"],
        reverse=True,
    )
    total = len(files)
    return {"files": files[offset:offset + limit], "total": total}


@app.get("/api/captures/image/{filename}")
async def captures_image(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = CAPTURE_DIR / filename
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/captures")
async def captures_clear():
    with _capture_lock:
        deleted = 0
        for f in CAPTURE_DIR.iterdir():
            if f.suffix == ".jpg":
                f.unlink()
                deleted += 1
        state.capture_count = 0
    return {"status": "cleared", "deleted": deleted, "count": 0}


@app.get("/stream")
async def mjpeg_stream():
    """MJPEG live preview — dashboard's 'What I see' panel."""
    async def gen():
        while True:
            jpg = state.last_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1.0 / max(TARGET_FPS, 5))

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# Face DB endpoints (stub — enroll returns 503 unless TODO_infer_frame returns embedding)
@app.get("/api/faces")
async def faces_list():
    if state.face_db is None:
        return {"faces": []}
    return {"faces": state.face_db.list_faces()}


@app.post("/api/faces/enroll")
async def face_enroll(name: str = Query(...)):
    """Enroll current face for identity recognition.

    Returns 503 if embedding model not available in TODO_infer_frame.
    To enable: have TODO_infer_frame return `embedding` field, and implement
    enrollment logic in your TODO_init_models.
    """
    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not initialized")

    # No embedding available from stub inference
    raise HTTPException(
        status_code=503,
        detail="Embedding model not available — implement in TODO_infer_frame"
    )


@app.delete("/api/faces/{name}")
async def face_delete(name: str):
    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not initialized")

    deleted = state.face_db.delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Face '{name}' not found")

    return {"status": "deleted", "name": name}


def main() -> None:
    threading.Thread(target=_capture_loop, daemon=True).start()
    import uvicorn
    print(f"[vision-stub] HTTP on 0.0.0.0:{HTTP_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")


if __name__ == "__main__":
    main()
