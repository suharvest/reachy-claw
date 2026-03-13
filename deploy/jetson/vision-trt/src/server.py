"""Main server: GStreamer camera capture + inference pipeline + outputs.

Captures frames directly from the camera via GStreamer with hardware-
accelerated decode/resize/encode, runs TensorRT inference, and publishes:
1. ZMQ PUB → reachy-claw (structured data)
2. WebSocket → browser (JSON detections)
3. HTTP API → face DB CRUD + health + stats
4. Video stream → browser (MJPEG from HW encoder)
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lazy imports for modules that need GPU
_zmq = None
_msgpack = None


class SmileCaptureTracker:
    """Track smile events and save face crops with per-person cooldown."""

    def __init__(self, capture_dir: str,
                 confidence_threshold: float = 0.6, distance_threshold: float = 0.8,
                 stable_frames: int = 5):
        self._dir = os.path.join(capture_dir, "captures")
        os.makedirs(self._dir, exist_ok=True)
        self._conf_threshold = confidence_threshold
        self._dist_threshold = distance_threshold
        self._stable_frames = stable_frames  # consecutive happy frames before capture
        self._embeddings_path = os.path.join(self._dir, "embeddings.json")
        # Store all captured person embeddings — each person captured only once
        self._captured_embeddings: list[list[float]] = self._load_embeddings()
        # Smile streak: spatial bucket → consecutive happy frame count
        self._smile_streak: dict[int, int] = {}
        # Count existing files for persistence across restarts
        self.count = len([f for f in os.listdir(self._dir) if f.endswith(".jpg")])
        if self.count:
            logger.info(f"SmileCaptureTracker: found {self.count} existing captures, "
                        f"{len(self._captured_embeddings)} dedup embeddings")

    def _load_embeddings(self) -> list[list[float]]:
        """Load persisted dedup embeddings from disk."""
        if not os.path.exists(self._embeddings_path):
            return []
        try:
            with open(self._embeddings_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} persisted dedup embeddings")
            return data
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return []

    def _save_embeddings(self) -> None:
        """Persist dedup embeddings to disk."""
        try:
            with open(self._embeddings_path, "w") as f:
                json.dump(self._captured_embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")

    def _is_duplicate(self, embedding: list[float]) -> bool:
        """Check if this face embedding matches any previously captured person."""
        emb = np.array(embedding, dtype=np.float32)
        for cap_emb in self._captured_embeddings:
            dist = float(np.linalg.norm(emb - np.array(cap_emb, dtype=np.float32)))
            if dist < self._dist_threshold:
                return True
        return False

    def _crop_and_save(self, face, frame: np.ndarray) -> str | None:
        """Crop face from frame with padding and save as JPEG. Returns filename or None."""
        h, w = frame.shape[:2]
        x1 = max(0, int(face.bbox[0] * w))
        y1 = max(0, int(face.bbox[1] * h))
        x2 = min(w, int(face.bbox[2] * w))
        y2 = min(h, int(face.bbox[3] * h))

        # Add padding (20%)
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        self.count += 1
        fname = f"smile_{int(time.time())}_{self.count:04d}.jpg"
        fpath = os.path.join(self._dir, fname)
        cv2.imwrite(fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return fname

    @staticmethod
    def _face_bucket(face) -> int:
        """Spatial bucket for tracking a face across consecutive frames."""
        cx = (face.bbox[0] + face.bbox[2]) / 2
        cy = (face.bbox[1] + face.bbox[3]) / 2
        return int(cx * 10) * 100 + int(cy * 10)

    def check_and_capture(self, results, frame: np.ndarray) -> dict | None:
        """Check all faces for smile capture trigger. Returns capture info or None.

        Requires stable_frames consecutive frames of happiness before capturing.
        """
        if not results:
            self._smile_streak.clear()
            return None

        active_buckets: set[int] = set()
        captures = []

        for face in results:
            bucket = self._face_bucket(face)
            active_buckets.add(bucket)

            is_happy = (
                face.emotion == "Happiness"
                and (face.emotion_confidence or 0) >= self._conf_threshold
            )

            if is_happy:
                self._smile_streak[bucket] = self._smile_streak.get(bucket, 0) + 1

                if self._smile_streak[bucket] < self._stable_frames:
                    continue  # not stable yet

                # Reset streak so same face needs to re-stabilize
                self._smile_streak[bucket] = 0

                # Per-person dedup: same person is only captured once
                if face.embedding and self._is_duplicate(face.embedding):
                    continue

                fname = self._crop_and_save(face, frame)
                if not fname:
                    continue

                # Record this person's embedding permanently
                if face.embedding:
                    self._captured_embeddings.append(face.embedding)

                logger.info(f"Smile captured: {fname} (total: {self.count})")
                captures.append(fname)
            else:
                # Not happy → reset streak for this position
                self._smile_streak[bucket] = 0

        # Clean up stale buckets (face left the frame)
        for b in list(self._smile_streak):
            if b not in active_buckets:
                del self._smile_streak[b]

        if not captures:
            return None

        # Batch-persist after all captures in this frame
        self._save_embeddings()
        return {"count": self.count, "event": True, "file": captures[-1], "new_captures": len(captures)}

    def clear(self) -> int:
        """Delete all captures and reset count. Returns deleted count."""
        deleted = 0
        for f in os.listdir(self._dir):
            if f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(self._dir, f))
                    deleted += 1
                except OSError:
                    pass
        self.count = 0
        self._captured_embeddings.clear()
        self._save_embeddings()
        logger.info(f"Cleared {deleted} captures")
        return deleted


class VisionService:
    """Core vision inference service."""

    def __init__(self):
        from .config import config

        self.config = config
        self.pipeline = None
        self.face_db = None
        self.streamer = None
        self.capture = None
        self.smile_tracker = None
        self._zmq_pub = None
        self._zmq_ctx = None
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Stats
        self._fps = 0.0
        self._inference_ms = 0.0
        self._frame_count = 0
        self._last_stats_time = time.monotonic()
        self._frame_id = 0

    def init(self):
        """Initialize all components."""
        from .capture import GstCameraCapture
        from .config import config
        from .face_database import FaceDatabase
        from .stream import VideoStreamer

        global _zmq, _msgpack
        import zmq
        import msgpack
        _zmq = zmq
        _msgpack = msgpack

        # Face database
        self.face_db = FaceDatabase(config.DATA_DIR)

        # Smile capture tracker
        self.smile_tracker = SmileCaptureTracker(config.DATA_DIR)

        # Video streamer (simple JPEG buffer now)
        self.streamer = VideoStreamer(config.STREAM_PORT)

        # ZMQ publisher
        self._zmq_ctx = zmq.Context()
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(f"tcp://0.0.0.0:{config.ZMQ_PUB_PORT}")
        logger.info(f"ZMQ PUB bound on port {config.ZMQ_PUB_PORT}")

        # TRT engines first (may take 20-60s on first boot)
        # Must build before camera start — nvv4l2decoder takes GPU memory
        try:
            from .models import load_engines

            engines = load_engines(config.MODEL_DIR, config.ENGINE_DIR)
            if engines:
                from .inference import VisionPipeline

                self.pipeline = VisionPipeline(engines, self.face_db, config)
                logger.info(f"Vision pipeline ready ({len(engines)} engines)")
            else:
                logger.warning("No TRT engines loaded, running in passthrough mode")
        except Exception as e:
            logger.error(f"Failed to load TRT engines: {e}")
            logger.warning("Running in passthrough mode (no inference)")

        # Camera capture via GStreamer (after TRT engines to avoid GPU memory contention)
        self._start_camera()

    def _start_camera(self) -> bool:
        """Detect camera device and start capture. Returns True on success."""
        from .capture import GstCameraCapture, find_camera_device
        from .config import config

        # Auto-detect device, fall back to config
        device = find_camera_device() or config.CAMERA_DEVICE
        logger.info(f"Starting camera on {device}")

        self.capture = GstCameraCapture(
            device=device,
            cam_width=config.CAMERA_WIDTH,
            cam_height=config.CAMERA_HEIGHT,
            cam_fps=config.CAMERA_FPS,
            inf_width=config.INPUT_WIDTH,
            inf_height=config.INPUT_HEIGHT,
            stream_width=config.STREAM_WIDTH,
        )
        if self.capture.start():
            return True

        logger.error("Camera capture failed to start")
        self.capture = None
        return False

    def process_and_publish(self, frame: np.ndarray, frame_id: int):
        """Run inference and publish results to all outputs."""
        t0 = time.monotonic()

        # Run inference
        results = []
        if self.pipeline:
            results = self.pipeline.process_frame(frame)

        # Cache for debug snapshot endpoint
        self._last_frame = frame
        self._last_results = results

        t1 = time.monotonic()
        self._inference_ms = (t1 - t0) * 1000

        # Update stats
        self._frame_count += 1
        elapsed = t1 - self._last_stats_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_stats_time = t1

        # Check for smile capture
        capture_info = None
        if self.smile_tracker and results:
            capture_info = self.smile_tracker.check_and_capture(results, frame)

        # Build messages (cast numpy scalars to Python floats for msgpack)
        def _f(v):
            return float(v) if v is not None else 0.0

        zmq_msg = {
            "timestamp": t0,
            "frame_id": frame_id,
            "faces": [
                {
                    "center": [float(c) for c in r.center],
                    "bbox": [float(b) for b in r.bbox],
                    "landmarks": [[float(x) for x in pt] for pt in r.landmarks],
                    "confidence": _f(r.confidence),
                    "embedding": r.embedding,  # already list[float] via .tolist()
                    "emotion": r.emotion,
                    "emotion_confidence": _f(r.emotion_confidence),
                    "identity": r.identity,
                    "identity_distance": _f(r.identity_distance),
                }
                for r in results
            ],
        }
        if capture_info:
            zmq_msg["capture"] = capture_info

        ws_msg = {
            "frame_id": frame_id,
            "faces": [
                {
                    "bbox": r.bbox,
                    "landmarks": r.landmarks,
                    "emotion": r.emotion,
                    "emotion_confidence": r.emotion_confidence,
                    "identity": r.identity,
                }
                for r in results
            ],
            "stats": {
                "fps": round(self._fps, 1),
                "inference_ms": round(self._inference_ms, 1),
            },
        }

        # ZMQ PUB
        try:
            self._zmq_pub.send_multipart([
                b"vision",
                _msgpack.packb(zmq_msg, use_bin_type=True),
            ])
        except Exception as e:
            if self._frame_id <= 3 or self._frame_id % 1000 == 0:
                logger.warning(f"ZMQ pub error: {e}")

        # WebSocket broadcast
        ws_json = json.dumps(ws_msg)
        with self._ws_lock:
            dead = set()
            for ws in self._ws_clients:
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(ws_json),
                        self._loop,
                    )
                except Exception:
                    dead.add(ws)
            self._ws_clients -= dead

        # Push HW-encoded JPEG to streamer
        if self.streamer and self.capture:
            jpeg = self.capture.get_stream_jpeg()
            if jpeg:
                self.streamer.set_jpeg(jpeg)

    def run_loop(self):
        """Main inference loop (runs in thread)."""
        target_interval = 1.0 / self.config.TARGET_FPS
        logger.info(f"Inference loop started (target {self.config.TARGET_FPS} FPS)")

        camera_retry_interval = 30  # seconds between camera retry attempts
        last_camera_retry = 0.0
        no_frame_count = 0

        while self._running:
            t0 = time.monotonic()

            # Camera not available — retry periodically
            if not self.capture or not self.capture._running:
                if t0 - last_camera_retry >= camera_retry_interval:
                    last_camera_retry = t0
                    logger.info("Camera not available, attempting to start...")
                    self._start_camera()
                time.sleep(1.0)
                continue

            frame = self.capture.get_inference_frame()
            if frame is None:
                no_frame_count += 1
                # If no frames for 30s, camera may have disconnected
                if no_frame_count > self.config.TARGET_FPS * 30:
                    logger.warning("No frames for 30s, restarting camera...")
                    self.capture.close()
                    self.capture = None
                    no_frame_count = 0
                time.sleep(0.01)
                continue

            no_frame_count = 0
            self._frame_id += 1
            self.process_and_publish(frame, self._frame_id)

            elapsed = time.monotonic() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        """Cleanup."""
        self._running = False
        if self.capture:
            self.capture.close()
        if self._zmq_pub:
            self._zmq_pub.close()
        if self._zmq_ctx:
            self._zmq_ctx.term()
        if self.streamer:
            self.streamer.close()


# Global service instance
service = VisionService()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: init service + start inference thread."""
    service.init()
    service._running = True
    service._loop = asyncio.get_event_loop()

    thread = threading.Thread(target=service.run_loop, daemon=True)
    thread.start()
    logger.info("Vision service started")

    yield

    service.close()
    logger.info("Vision service stopped")


# ── FastAPI app ──────────────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Vision TRT Service", lifespan=lifespan)

# CORS — allow dashboard (different port) to access stream and APIs
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": service.pipeline is not None,
        "capture": service.capture is not None and service.capture._running,
        "fps": round(service._fps, 1),
    }


@app.get("/api/stats")
async def stats():
    inf_shape = None
    if service.capture:
        frame = service.capture.get_inference_frame()
        if frame is not None:
            inf_shape = list(frame.shape)  # [H, W, C]
    return {
        "fps": round(service._fps, 1),
        "inference_ms": round(service._inference_ms, 1),
        "pipeline_ready": service.pipeline is not None,
        "faces_registered": len(service.face_db.list_faces()) if service.face_db else 0,
        "inference_frame_shape": inf_shape,
    }


@app.get("/api/snapshot")
async def snapshot(target: str = "inference"):
    """Debug: return frame as JPEG with bbox overlay.

    ?target=inference (default): 640x640 inference frame
    ?target=stream: 640x360 stream frame with mapped overlay
    """
    from fastapi.responses import Response

    results = getattr(service, '_last_results', [])

    if target == "stream":
        # Get stream frame and map inference coords onto it
        jpeg = service.capture.get_stream_jpeg() if service.capture else None
        if not jpeg:
            return Response(content='{"error":"no stream"}', media_type="application/json", status_code=503)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        frame = getattr(service, '_last_frame', None)
        if frame is None:
            return Response(content='{"error":"no frame"}', media_type="application/json", status_code=503)
        frame = frame.copy()

    h, w = frame.shape[:2]
    for r in results:
        x1 = int(r.bbox[0] * w)
        y1 = int(r.bbox[1] * h)
        x2 = int(r.bbox[2] * w)
        y2 = int(r.bbox[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lx, ly in r.landmarks:
            cv2.circle(frame, (int(lx * w), int(ly * h)), 3, (0, 0, 255), -1)
        label = f"{r.emotion} {r.emotion_confidence:.0%}"
        cv2.putText(frame, label, (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Add frame info
    cv2.putText(frame, f"{w}x{h}", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    _, jpg = cv2.imencode(".jpg", frame)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


@app.get("/api/faces")
async def list_faces():
    if not service.face_db:
        return {"faces": []}
    return {"faces": service.face_db.list_faces()}


@app.delete("/api/captures")
async def clear_captures():
    if not service.smile_tracker:
        return {"error": "Smile tracker not initialized"}, 503
    deleted = service.smile_tracker.clear()
    return {"status": "cleared", "deleted": deleted, "count": 0}


@app.get("/api/captures/count")
async def capture_count():
    count = service.smile_tracker.count if service.smile_tracker else 0
    return {"count": count}


@app.post("/api/faces/enroll")
async def enroll_face(name: str = Query(...)):
    if not service.face_db or not service.pipeline:
        return {"error": "Service not ready"}, 503

    if not service.capture:
        return {"error": "Camera not available"}, 503

    frame = service.capture.get_inference_frame()
    if frame is None:
        return {"error": "No frame available"}, 400

    results = service.pipeline.process_frame(frame)
    if not results:
        return {"error": "No face detected"}, 400

    # Use the largest face
    primary = max(
        results,
        key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
    )
    if not primary.embedding:
        return {"error": "Face embedding extraction failed"}, 500

    embedding = np.array(primary.embedding, dtype=np.float32)
    service.face_db.enroll(name, embedding)
    return {"status": "enrolled", "name": name}


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    if not service.face_db:
        return {"error": "Service not ready"}, 503

    if service.face_db.delete(name):
        return {"status": "deleted", "name": name}
    return {"error": "Face not found"}, 404


@app.post("/api/faces/enroll-image")
async def enroll_face_from_image(name: str = Form(...), image: UploadFile = File(...)):
    """Register a face from an uploaded image file."""
    if not service.face_db or not service.pipeline:
        return {"error": "Service not ready"}, 503

    contents = await image.read()
    if not contents:
        return {"error": "Empty file"}, 400
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image"}, 400

    results = service.pipeline.process_frame(frame)
    if not results:
        return {"error": "No face detected in image"}, 400

    primary = max(
        results,
        key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
    )
    if not primary.embedding:
        return {"error": "Face embedding extraction failed"}, 500

    embedding = np.array(primary.embedding, dtype=np.float32)
    service.face_db.enroll(name, embedding)
    return {"status": "enrolled", "name": name}


@app.get("/api/faces/export")
async def export_faces():
    """Export all face data as a zip archive."""
    import io
    import zipfile

    if not service.face_db:
        return {"error": "Service not ready"}, 503

    buf = io.BytesIO()
    data_dir = service.face_db._dir
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in data_dir.iterdir():
            if fpath.suffix in (".json", ".npy"):
                zf.write(fpath, fpath.name)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=faces.zip"},
    )


@app.post("/api/faces/import")
async def import_faces(file: UploadFile = File(...)):
    """Import face data from a zip archive (replaces existing)."""
    import io
    import zipfile

    if not service.face_db:
        return {"error": "Service not ready"}, 503

    contents = await file.read()
    buf = io.BytesIO(contents)
    if not zipfile.is_zipfile(buf):
        return {"error": "Invalid zip file"}, 400

    data_dir = service.face_db._dir
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            # Only extract .json and .npy files, no path traversal
            if name != os.path.basename(name):
                continue
            if not name.endswith((".json", ".npy")):
                continue
            zf.extract(name, data_dir)

    # Reload database
    service.face_db._faces.clear()
    service.face_db._load()
    return {"status": "imported", "faces": service.face_db.list_faces()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with service._ws_lock:
        service._ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    finally:
        with service._ws_lock:
            service._ws_clients.discard(websocket)


@app.get("/stream")
async def mjpeg_stream():
    """MJPEG stream for browsers."""

    async def generate():
        try:
            while True:
                jpg = service.streamer.get_jpeg() if service.streamer else None
                if jpg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                await asyncio.sleep(0.1)  # ~10fps matches inference rate
        finally:
            if service.streamer and not service._ws_clients:
                service.streamer._has_clients = False

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/")
async def index():
    """Serve the frontend page."""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Vision TRT Service</h1><p>Frontend not found.</p>")


if __name__ == "__main__":
    import uvicorn

    from .config import config

    uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT)
