"""Vision producer for CM4 — MediaPipe detection + ONNX emotion (skip-frame).

CPU-only implementation for Raspberry Pi CM4 (no GPU/NPU).
- Face detection: MediaPipe Face Detection (every frame, ~10-15 FPS on CM4)
- Emotion recognition: ONNX FERPlus-8 (skip-frame, default every 5 frames)
- Smile capture: triggered on high-confidence happy emotion

Wire-compatible with vision-trt and reachy-claw's VisionClientPlugin.
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

# MediaPipe Face Detection
import mediapipe as mp

# ONNX emotion classifier
try:
    from emotion_onnx import EmotionONNX, HAS_ONNX
except ImportError:
    HAS_ONNX = False
    EmotionONNX = None


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
TARGET_FPS = float(os.environ.get("TARGET_FPS", 10))  # Lower default for CM4
FACE_DB_DIR = Path(os.environ.get("FACE_DB_DIR", "/app/data/faces"))
FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
PER_IDENTITY_COOLDOWN = float(os.environ.get("PER_IDENTITY_COOLDOWN", 30.0))
ANONYMOUS_COOLDOWN = float(os.environ.get("ANONYMOUS_COOLDOWN", 5.0))

# Skip-frame config for emotion inference
EMOTION_SKIP_FRAMES = int(os.environ.get("EMOTION_SKIP_FRAMES", 5))  # Run emotion every N frames
EMOTION_CONFIDENCE_THRESHOLD = float(os.environ.get("EMOTION_CONFIDENCE_THRESHOLD", 0.6))
SMILE_THRESHOLD = float(os.environ.get("SMILE_THRESHOLD", 0.75))  # Happy confidence threshold for capture

# Model paths
EMOTION_MODEL_PATH = os.environ.get("EMOTION_MODEL_PATH", "/app/models/emotion-ferplus-8.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# Model initialization
# ─────────────────────────────────────────────────────────────────────────────
def init_models() -> dict:
    """Load MediaPipe face detection + ONNX emotion classifier."""
    # MediaPipe Face Detection (short-range model for close-up faces)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # 0 = short-range (<2m), 1 = full-range (<5m)
        min_detection_confidence=0.5,
    )

    # ONNX emotion classifier (optional, skip if not available)
    emotion_model = None
    if HAS_ONNX:
        try:
            emotion_model = EmotionONNX(EMOTION_MODEL_PATH)
            print(f"[vision-cm4] Emotion model loaded: {EMOTION_MODEL_PATH}", flush=True)
        except Exception as e:
            print(f"[vision-cm4] Warning: emotion model not loaded: {e}", flush=True)

    return {
        "detector": detector,
        "emotion": emotion_model,
    }


def infer_frame(frame_bgr: np.ndarray, models: dict, state: dict) -> list[dict]:
    """Run face detection (every frame) + emotion (skip-frame).

    Args:
        frame_bgr: BGR numpy image from camera
        models: {"detector": MediaPipe, "emotion": EmotionONNX}
        state: {"frame_id": int, "last_emotions": {face_id: emotion_dict}}

    Returns:
        List of face dicts with normalized coordinates and emotion.
    """
    h, w = frame_bgr.shape[:2]

    # MediaPipe expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = models["detector"].process(frame_rgb)

    if not results.detections:
        return []

    faces = []
    emotion_model = models.get("emotion")

    for i, det in enumerate(results.detections):
        # Extract bbox (MediaPipe uses relative coordinates [0,1])
        bbox = det.location_data.relative_bounding_box
        x1 = bbox.xmin
        y1 = bbox.ymin
        x2 = x1 + bbox.width
        y2 = y1 + bbox.height

        # Clamp to [0, 1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)

        # Center in [-1, 1] (reachy-claw expects this)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        center = [cx * 2 - 1, cy * 2 - 1]  # [-1, 1]

        confidence = det.score[0] if det.score else 0.9

        # Extract landmarks (6 keypoints from MediaPipe)
        # MediaPipe provides: right_eye, left_eye, nose_tip, mouth_center,
        #                     right_ear_tragion, left_ear_tragion
        landmarks = []
        if det.location_data.relative_keypoints:
            for kp in det.location_data.relative_keypoints:
                landmarks.append([kp.x, kp.y])  # Already in [0, 1]

        # Face dict without emotion (will add below)
        face = {
            "center": center,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "landmarks": landmarks[:5] if landmarks else None,  # Limit to 5 for compatibility
        }

        # Emotion inference (skip-frame)
        face_id = f"{int(x1*1000)}_{int(y1*1000)}"  # Simple ID based on position

        if emotion_model is not None:
            # Check if we should run emotion this frame
            should_run_emotion = (
                state["frame_id"] % EMOTION_SKIP_FRAMES == 0
                or face_id not in state["last_emotions"]
            )

            if should_run_emotion:
                # Crop face from frame
                px1 = int(x1 * w)
                py1 = int(y1 * h)
                px2 = int(x2 * w)
                py2 = int(y2 * h)

                # Add 20% padding for better emotion recognition
                pad_w = int((px2 - px1) * 0.2)
                pad_h = int((py2 - py1) * 0.2)
                px1 = max(0, px1 - pad_w)
                py1 = max(0, py1 - pad_h)
                px2 = min(w, px2 + pad_w)
                py2 = min(h, py2 + pad_h)

                crop = frame_bgr[py1:py2, px1:px2]

                if crop.size > 0:
                    try:
                        emotion_result = emotion_model.infer(crop)
                        state["last_emotions"][face_id] = emotion_result
                    except Exception as e:
                        # Fallback to cached or neutral
                        pass

            # Use cached emotion if available
            cached = state["last_emotions"].get(face_id)
            if cached:
                face["emotion"] = cached["emotion"]
                face["emotion_confidence"] = cached["emotion_confidence"]
            else:
                # No emotion available (first few frames or model not loaded)
                face["emotion"] = "neutral"
                face["emotion_confidence"] = 0.0

        faces.append(face)

    return faces


def should_capture_smile(face: dict, frame_bgr: np.ndarray) -> bool:
    """Return True when face shows high-confidence happy emotion."""
    emotion = face.get("emotion", "")
    confidence = face.get("emotion_confidence", 0.0)

    if emotion == "happy" and confidence >= SMILE_THRESHOLD:
        return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Infrastructure (HTTP, ZMQ, camera loop) — copied from vision-stub
# ═════════════════════════════════════════════════════════════════════════════

class _State:
    last_frame: np.ndarray | None = None
    last_jpeg: bytes | None = None
    capture_count: int = 0
    last_capture_t: float = 0.0
    fps: float = 0.0
    inference_ms: float = 0.0
    face_db: Any | None = None
    frame_id: int = 0
    last_emotions: dict[str, dict] = {}  # face_id -> emotion_result
    identity_last_capture: dict[str, float] = {}
    anonymous_last_capture: float = 0.0


state = _State()
_capture_lock = threading.Lock()


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
    print(f"[vision-cm4] ZMQ PUB on tcp://0.0.0.0:{ZMQ_PUB_PORT}", flush=True)

    # Rest-control subscriber (pause/resume from reachy-claw)
    rest_ctrl_url = os.environ.get("REST_CTRL_URL", "").strip()
    ctrl_sub = None
    poller = None
    if rest_ctrl_url:
        ctrl_sub = ctx.socket(zmq.SUB)
        ctrl_sub.connect(rest_ctrl_url)
        ctrl_sub.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(ctrl_sub, zmq.POLLIN)
        print(f"[vision-cm4] Rest control SUB connected to {rest_ctrl_url}", flush=True)

    paused = False

    models = init_models()
    state.capture_count = _existing_capture_count()

    # Initialize face_db if available
    try:
        from face_db import FaceDatabase
        state.face_db = FaceDatabase(str(FACE_DB_DIR))
        print(f"[vision-cm4] Face DB initialized at {FACE_DB_DIR}", flush=True)
    except ImportError:
        state.face_db = None
        print("[vision-cm4] Face DB not available", flush=True)

    cap = _open_camera()
    print(f"[vision-cm4] Camera {CAMERA_DEVICE} open at {CAMERA_W}x{CAMERA_H}", flush=True)
    print(f"[vision-cm4] Emotion skip-frame: every {EMOTION_SKIP_FRAMES} frames", flush=True)

    frame_dt = 1.0 / TARGET_FPS
    fps_t0 = time.monotonic()
    fps_count = 0

    while True:
        loop_start = time.monotonic()

        # Drain rest-control messages
        if poller is not None:
            while True:
                socks = dict(poller.poll(timeout=0))
                if ctrl_sub not in socks:
                    break
                try:
                    msg = ctrl_sub.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                cmd = msg.get("cmd")
                if cmd == "pause":
                    if not paused:
                        print("[vision-cm4] REST: pausing inference", flush=True)
                    paused = True
                elif cmd == "resume":
                    if paused:
                        print("[vision-cm4] REST: resuming inference", flush=True)
                    paused = False

        if paused:
            time.sleep(0.5)
            continue

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        t0 = time.monotonic()
        try:
            faces = infer_frame(frame, models, state)
        except Exception as e:
            print(f"[vision-cm4] inference error: {e}", flush=True)
            faces = []
        state.inference_ms = (time.monotonic() - t0) * 1000

        # Smile capture
        capture_events: list[dict] = []
        now = time.monotonic()

        for face in faces:
            if should_capture_smile(face, frame):
                identity = face.get("identity")

                # Per-identity cooldown
                if identity:
                    last = state.identity_last_capture.get(identity, 0)
                    if now - last < PER_IDENTITY_COOLDOWN:
                        continue
                else:
                    if now - state.anonymous_last_capture < ANONYMOUS_COOLDOWN:
                        continue

                # Capture
                with _capture_lock:
                    state.capture_count += 1
                    count_now = state.capture_count
                    fname = f"smile_{int(time.time()*1000)}_{count_now}.jpg"

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

                    if identity:
                        state.identity_last_capture[identity] = now
                    else:
                        state.anonymous_last_capture = now

                capture_events.append({"event": "smile", "count": count_now, "file": fname, "identity": identity})

        # PUB to reachy-claw
        zmq_msg: dict[str, Any] = {"frame_id": state.frame_id, "faces": faces}
        if capture_events:
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

        state.frame_id += 1
        sleep = frame_dt - (time.monotonic() - loop_start)
        if sleep > 0:
            time.sleep(sleep)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API (compatible with vision-trt dashboard)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="vision-cm4")


@app.get("/")
async def root():
    return {
        "service": "vision-cm4",
        "fps": round(state.fps, 1),
        "inference_ms": round(state.inference_ms, 1),
        "emotion_skip_frames": EMOTION_SKIP_FRAMES,
    }


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


@app.get("/api/faces")
async def faces_list():
    if state.face_db is None:
        return {"faces": []}
    return {"faces": state.face_db.list_faces()}


@app.post("/api/faces/enroll")
async def face_enroll(name: str = Query(...)):
    """Enroll face — returns 503 (no embedding model in CPU version)."""
    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not initialized")
    raise HTTPException(status_code=503, detail="Embedding model not available in CPU-only mode")


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
    print(f"[vision-cm4] HTTP on 0.0.0.0:{HTTP_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")


if __name__ == "__main__":
    main()