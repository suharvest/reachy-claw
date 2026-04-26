"""Vision producer for Hailo-8 — drop-in replacement for vision-trt.

Runs on harvest-pi (Raspberry Pi 5 + Hailo-8) and provides:
- ZMQ PUB on tcp://0.0.0.0:8631 (msgpack face telemetry)
- HTTP API on 0.0.0.0:8630 (dashboard endpoints)
- MJPEG stream on /stream

Wire-compatible with reachy-claw's VisionClientPlugin.
"""

import asyncio
import json
import logging
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

from hailo_pipeline import init_pipeline, HailoPipeline, EMOTION_CLASSES
from face_db import FaceDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision-hailo")

# Config (env vars)
ZMQ_PUB_PORT = int(os.environ.get("ZMQ_PUB_PORT", 8631))
HTTP_PORT = int(os.environ.get("HTTP_PORT", 8630))
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "/dev/video0")
CAMERA_W = int(os.environ.get("CAMERA_W", 640))
CAMERA_H = int(os.environ.get("CAMERA_H", 480))
CAPTURE_DIR = Path(os.environ.get("CAPTURE_DIR", "/var/lib/vision-hailo/captures"))
FACE_DB_DIR = Path(os.environ.get("FACE_DB_DIR", "/var/lib/vision-hailo/faces"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", 15))
PER_IDENTITY_COOLDOWN = float(os.environ.get("PER_IDENTITY_COOLDOWN", 30.0))
ANONYMOUS_COOLDOWN = float(os.environ.get("ANONYMOUS_COOLDOWN", 5.0))

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
FACE_DB_DIR.mkdir(parents=True, exist_ok=True)


class State:
    last_frame: Optional[np.ndarray] = None
    last_jpeg: Optional[bytes] = None
    capture_count: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    pipeline: Optional[HailoPipeline] = None
    face_db: Optional[FaceDatabase] = None
    # Smile capture dedup
    identity_last_capture: dict[str, float] = {}
    anonymous_last_capture: float = 0.0


state = State()
_capture_lock = threading.Lock()


def _open_camera() -> cv2.VideoCapture:
    """Open camera device with libcamera compatibility."""
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try libcamera backend if default fails
    if not cap.isOpened():
        logger.warning(f"Failed to open {CAMERA_DEVICE}, trying libcamera...")
        # Pi camera is often on video19 for libcamera
        for dev in ["/dev/video19", "/dev/video0"]:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
            if cap.isOpened():
                logger.info(f"Opened camera on {dev}")
                return cap
        raise RuntimeError(f"Cannot open any camera device")

    logger.info(f"Camera {CAMERA_DEVICE} open at {CAMERA_W}x{CAMERA_H}")
    return cap


def _existing_capture_count() -> int:
    return sum(1 for f in CAPTURE_DIR.iterdir() if f.suffix == ".jpg")


def _emotion_remap(emotion: str) -> str:
    """Map HSEemotion to dashboard-expected strings.

    Dashboard expects: happy, sad, angry, surprised, fear, neutral, disgust
    HSEemotion outputs: Happiness, Neutral, Sadness, Surprise, Fear, Anger, Disgust, Contempt
    """
    mapping = {
        "Happiness": "happy",
        "Sadness": "sad",
        "Anger": "angry",
        "Surprise": "surprised",
        "Fear": "fear",
        "Neutral": "neutral",
        "Disgust": "disgust",
        "Contempt": "neutral",  # Map contempt to neutral for dashboard
    }
    return mapping.get(emotion, "neutral")


def _should_capture_smile(face: dict, now: float) -> bool:
    """Check if this face should trigger a smile capture (per-identity dedup)."""
    # Must be happy with high confidence
    if face.get("emotion") != "Happiness":
        return False
    if face.get("emotion_confidence", 0) < 0.6:
        return False

    identity = face.get("identity")
    if identity:
        # Known identity — check per-identity cooldown
        last = state.identity_last_capture.get(identity, 0)
        if now - last < PER_IDENTITY_COOLDOWN:
            return False
    else:
        # Anonymous — check anonymous cooldown
        if now - state.anonymous_last_capture < ANONYMOUS_COOLDOWN:
            return False

    return True


def _capture_smile(frame: np.ndarray, face: dict, now: float) -> dict:
    """Save smile capture with face crop + 20% padding."""
    with _capture_lock:
        state.capture_count += 1
        count_now = state.capture_count

        # Crop with 20% padding
        bbox = face.get("bbox", [0.3, 0.2, 0.7, 0.8])
        h, w = frame.shape[:2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        # Add 20% padding
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = frame[y1:y2, x1:x2]
        fname = f"smile_{int(time.time()*1000)}_{count_now}.jpg"
        cv2.imwrite(str(CAPTURE_DIR / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Update cooldown
        identity = face.get("identity")
        if identity:
            state.identity_last_capture[identity] = now
        else:
            state.anonymous_last_capture = now

    return {"event": "smile", "count": count_now, "file": fname}


def _capture_loop() -> None:
    """Camera → inference → ZMQ PUB loop."""
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://0.0.0.0:{ZMQ_PUB_PORT}")
    logger.info(f"ZMQ PUB on tcp://0.0.0.0:{ZMQ_PUB_PORT}")

    # Init pipeline
    state.pipeline = init_pipeline()
    state.face_db = FaceDatabase(str(FACE_DB_DIR))
    state.capture_count = _existing_capture_count()

    cap = _open_camera()

    frame_id = 0
    frame_dt = 1.0 / TARGET_FPS
    fps_t0 = time.monotonic()
    fps_count = 0

    try:
        while True:
            loop_start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                logger.warning("Camera read failed")
                time.sleep(0.05)
                continue

            # Inference
            t0 = time.monotonic()
            try:
                faces = state.pipeline.process_frame(frame)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                faces = []
            state.inference_ms = (time.monotonic() - t0) * 1000

            # Identity matching (if face_db has entries)
            if state.face_db and faces:
                # No embedding model, so identity matching is disabled
                # (would need ArcFace HEF for this)
                pass

            # Remap emotions for dashboard
            for face in faces:
                face["emotion"] = _emotion_remap(face.get("emotion", "Neutral"))

            # Smile capture with per-identity dedup
            now = time.monotonic()
            capture_events: list[dict] = []
            for face in faces:
                if _should_capture_smile(face, now):
                    capture_events.append(_capture_smile(frame, face, now))

            # Build ZMQ message
            zmq_msg: dict[str, Any] = {"frame_id": frame_id, "faces": faces}
            if capture_events:
                # Include first capture event
                zmq_msg["capture"] = capture_events[0]
                zmq_msg["new_captures"] = len(capture_events)

            pub.send_multipart([b"vision", msgpack.packb(zmq_msg, use_bin_type=True)])

            # Update preview JPEG
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
                logger.info(f"FPS: {state.fps:.1f}, inference: {state.inference_ms:.1f}ms")

            frame_id += 1
            sleep = frame_dt - (time.monotonic() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    finally:
        cap.release()
        if state.pipeline:
            state.pipeline.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="vision-hailo")


@app.get("/")
async def root():
    return {"service": "vision-hailo", "fps": round(state.fps, 1)}


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
    """MJPEG live preview."""
    async def gen():
        while True:
            jpg = state.last_jpeg
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1.0 / max(TARGET_FPS, 5))

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# Face DB endpoints
@app.get("/api/faces")
async def faces_list():
    if state.face_db is None:
        return {"faces": []}
    return {"faces": state.face_db.list_faces()}


@app.post("/api/faces/enroll")
async def face_enroll(name: str = Query(...)):
    """Enroll current face for identity recognition.

    Returns 503 if embedding model not available (ArcFace HEF missing).
    """
    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not initialized")

    # No embedding model available
    # Would need ArcFace HEF to extract embedding from current face
    raise HTTPException(
        status_code=503,
        detail="Embedding model not available — ArcFace HEF required"
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
    logger.info(f"HTTP on 0.0.0.0:{HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")


if __name__ == "__main__":
    main()