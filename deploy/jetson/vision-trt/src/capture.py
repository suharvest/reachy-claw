"""GStreamer camera capture with hardware-accelerated resize and encode.

HW pipeline (Jetson, GST_V4L2_USE_LIBV4L2=1):
  v4l2src (libnvv4l2 decodes MJPEG→YUY2) → tee
    ├→ nvvidconv YUY2→BGRx 640x640 → appsink (inference)
    └→ nvvidconv 640w NVMM → nvjpegenc → appsink (MJPEG stream)

  Camera pre-set to 1080p via V4L2 ioctl. nvvidconv (VIC HW) handles
  both colorspace conversion and resize — no CPU videoconvert needed.

CPU fallback (no NVIDIA plugins):
  v4l2src → videoscale → videoconvert → tee → appsinks
"""

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


def find_camera_device(name_pattern: str = "Reachy Mini") -> str | None:
    """Auto-detect video device by scanning /sys/class/video4linux/*/name.

    Returns the first /dev/videoN whose name contains `name_pattern` and
    actually supports video capture (VIDIOC_QUERYCAP check).
    """
    v4l_dir = "/sys/class/video4linux"
    if not os.path.isdir(v4l_dir):
        return None

    candidates = []
    for entry in sorted(os.listdir(v4l_dir)):
        name_file = os.path.join(v4l_dir, entry, "name")
        try:
            with open(name_file) as f:
                dev_name = f.read().strip()
        except OSError:
            continue
        if name_pattern.lower() in dev_name.lower():
            candidates.append(f"/dev/{entry}")

    # Try each candidate — pick the one that supports MJPEG capture
    for dev in candidates:
        try:
            import fcntl
            import struct

            fd = os.open(dev, os.O_RDWR)
            V4L2_PIX_FMT_MJPEG = 0x47504A4D
            VIDIOC_S_FMT = 0xC0D05605
            fmt = bytearray(208)
            struct.pack_into("I", fmt, 0, 1)  # V4L2_BUF_TYPE_VIDEO_CAPTURE
            struct.pack_into("I", fmt, 4, 640)
            struct.pack_into("I", fmt, 8, 480)
            struct.pack_into("I", fmt, 12, V4L2_PIX_FMT_MJPEG)
            fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
            actual_fmt = struct.unpack_from("I", fmt, 12)[0]
            os.close(fd)
            if actual_fmt == V4L2_PIX_FMT_MJPEG:
                logger.info(f"Auto-detected camera: {dev} ({dev_name})")
                return dev
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            continue

    # Fallback: return first candidate
    if candidates:
        logger.info(f"Using first matching camera: {candidates[0]}")
        return candidates[0]
    return None


class GstCameraCapture:
    """GStreamer camera capture with dual output: inference frames + MJPEG stream."""

    def __init__(
        self,
        device: str = "/dev/video0",
        cam_width: int = 1920,
        cam_height: int = 1080,
        cam_fps: int = 60,
        inf_width: int = 640,
        inf_height: int = 640,
        stream_width: int = 640,
    ):
        self._device = device
        self._cam_width = cam_width
        self._cam_height = cam_height
        self._cam_fps = cam_fps
        self._inf_width = inf_width
        self._inf_height = inf_height
        self._stream_width = stream_width

        self._pipeline = None
        self._inference_sink = None
        self._stream_sink = None
        self._Gst = None
        self._running = False
        self._hw_pipeline = False

        # Latest JPEG for MJPEG stream (thread-safe)
        self._latest_jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()

    def _set_camera_format(self):
        """Pre-set camera to MJPEG at desired resolution via V4L2 ioctl.

        Must be called BEFORE GStreamer opens v4l2src. This forces the camera
        to output MJPEG at 1920x1080 instead of the max 3840x2592, so
        libnvv4l2's transparent decode handles a much smaller frame.
        """
        import fcntl
        import os
        import struct

        fd = None
        try:
            fd = os.open(self._device, os.O_RDWR)
            V4L2_PIX_FMT_MJPEG = 0x47504A4D  # 'MJPG'
            VIDIOC_S_FMT = 0xC0D05605

            fmt = bytearray(208)
            struct.pack_into("I", fmt, 0, 1)  # V4L2_BUF_TYPE_VIDEO_CAPTURE
            struct.pack_into("I", fmt, 4, self._cam_width)
            struct.pack_into("I", fmt, 8, self._cam_height)
            struct.pack_into("I", fmt, 12, V4L2_PIX_FMT_MJPEG)

            fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
            actual_w = struct.unpack_from("I", fmt, 4)[0]
            actual_h = struct.unpack_from("I", fmt, 8)[0]
            logger.info(f"Camera format set to {actual_w}x{actual_h} MJPEG via ioctl")
        except Exception as e:
            logger.warning(f"Could not pre-set camera format: {e}")
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def start(self) -> bool:
        """Build and start the GStreamer pipeline.

        Tries HW-accelerated pipeline first, falls back to CPU.
        Returns True if pipeline started successfully.
        """
        # Pre-set camera to desired resolution before GStreamer opens it
        self._set_camera_format()

        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp  # noqa: F401
            Gst.init(None)
            self._Gst = Gst
        except Exception as e:
            logger.error(f"GStreamer not available: {e}")
            return False

        # Retry loop — camera device may not be ready immediately after boot
        import time
        attempt = 0
        while True:
            attempt += 1
            # Try HW pipeline first (nvvidconv + nvjpegenc)
            if self._try_hw_pipeline():
                self._hw_pipeline = True
                logger.info("Camera capture started (HW: nvvidconv + nvjpegenc)")
                return True

            # Fall back to CPU pipeline
            if self._try_cpu_pipeline():
                self._hw_pipeline = False
                logger.info("Camera capture started (CPU fallback: videoconvert)")
                return True

            wait = min(attempt * 3, 30)  # 3,6,9,...,30,30,30,...
            logger.warning(f"Camera start failed (attempt {attempt}), retrying in {wait}s")
            time.sleep(wait)
            self._set_camera_format()  # re-set format before retry

    def _try_hw_pipeline(self) -> bool:
        """HW pipeline: libnvv4l2 MJPEG decode → nvvidconv resize/colorspace.

        With GST_V4L2_USE_LIBV4L2=1, libnvv4l2 transparently decodes MJPEG
        and outputs raw YUY2. We use nvvidconv (VIC hardware) instead of CPU
        videoconvert for both colorspace conversion and resize in one step.
        Camera is pre-set to 1080p via V4L2 ioctl to minimize frame size.
        """
        Gst = self._Gst

        for name in ("nvvidconv", "nvjpegenc"):
            if Gst.ElementFactory.find(name) is None:
                logger.debug(f"HW plugin {name} not found")
                return False

        stream_height = int(self._cam_height * self._stream_width / self._cam_width)

        pipeline_str = (
            f"v4l2src device={self._device} "
            "! tee name=t "
            # Inference: nvvidconv does YUY2→BGRx + resize in one shot (VIC HW)
            "t. ! queue leaky=downstream max-size-buffers=1 "
            f"! nvvidconv ! video/x-raw,format=BGRx,width={self._inf_width},"
            f"height={self._inf_height} "
            "! appsink name=inference_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false "
            # Stream: nvvidconv resize + HW JPEG encode
            "t. ! queue leaky=downstream max-size-buffers=1 "
            f"! nvvidconv ! video/x-raw(memory:NVMM),width={self._stream_width},"
            f"height={stream_height} "
            "! nvjpegenc "
            "! appsink name=stream_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false"
        )

        return self._launch_pipeline(pipeline_str)

    def _try_cpu_pipeline(self) -> bool:
        """CPU fallback: videoscale + videoconvert."""
        pipeline_str = (
            f"v4l2src device={self._device} "
            f"! videoscale ! video/x-raw,width={self._inf_width},height={self._inf_height} "
            "! videoconvert ! video/x-raw,format=BGR "
            "! tee name=t "
            "t. ! queue leaky=downstream max-size-buffers=1 "
            "! appsink name=inference_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false "
            "t. ! queue leaky=downstream max-size-buffers=1 "
            "! jpegenc quality=70 "
            "! appsink name=stream_sink emit-signals=false "
            "drop=true max-buffers=1 sync=false"
        )

        return self._launch_pipeline(pipeline_str)

    def _launch_pipeline(self, pipeline_str: str) -> bool:
        """Parse and start a GStreamer pipeline."""
        Gst = self._Gst
        logger.info(f"Launching pipeline: {pipeline_str}")
        pipeline = None
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            self._inference_sink = pipeline.get_by_name("inference_sink")
            self._stream_sink = pipeline.get_by_name("stream_sink")

            if not self._inference_sink or not self._stream_sink:
                logger.error("Failed to get appsink elements")
                return False

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Pipeline failed to start")
                pipeline.set_state(Gst.State.NULL)
                return False

            # Wait for pipeline to reach PLAYING (10s timeout)
            _, state, _ = pipeline.get_state(10 * Gst.SECOND)
            logger.info(f"Pipeline state after preroll: {state.value_name}")

            # Check bus for errors
            bus = pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR)
            if msg:
                err, debug = msg.parse_error()
                logger.error(f"Pipeline bus error: {err.message}")
                logger.debug(f"Debug: {debug}")
                pipeline.set_state(Gst.State.NULL)
                return False

            if state != Gst.State.PLAYING:
                logger.warning(f"Pipeline stuck in {state.value_name}, aborting")
                pipeline.set_state(Gst.State.NULL)
                return False

            self._pipeline = pipeline
            self._running = True
            return True

        except Exception as e:
            logger.error(f"Pipeline launch failed: {e}")
            if pipeline is not None:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            return False

    def get_inference_frame(self) -> np.ndarray | None:
        """Pull a 640x640 BGR frame for inference (non-blocking).

        Returns None if no frame is available.
        """
        if not self._running or not self._inference_sink:
            return None

        Gst = self._Gst
        sample = self._inference_sink.try_pull_sample(100 * 1000000)  # 100ms in ns
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = struct.get_int("width")[1]
        h = struct.get_int("height")[1]

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None

        # HW pipeline outputs BGRx (4ch), CPU pipeline outputs BGR (3ch)
        channels = len(map_info.data) // (h * w)
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, channels)
        buf.unmap(map_info)
        if channels == 4:
            return frame[:, :, :3].copy()  # BGRx → BGR (strip alpha)
        return frame.copy()

    def get_stream_jpeg(self) -> bytes | None:
        """Pull a HW-encoded JPEG frame for MJPEG stream (non-blocking).

        Returns None if no frame is available.
        """
        if not self._running or not self._stream_sink:
            return None

        Gst = self._Gst
        sample = self._stream_sink.try_pull_sample(100 * 1000000)  # 100ms in ns
        if sample is None:
            return None

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None

        jpeg = bytes(map_info.data)
        buf.unmap(map_info)
        return jpeg

    def close(self):
        """Stop the pipeline and release resources."""
        self._running = False
        if self._pipeline and self._Gst:
            self._pipeline.send_event(self._Gst.Event.new_eos())
            self._pipeline.set_state(self._Gst.State.NULL)
            self._pipeline = None
            logger.info("Camera capture stopped")
