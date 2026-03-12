"""Central application context and plugin orchestrator.

ReachyClawApp holds shared state (config, robot, head targets, emotions)
and manages the lifecycle of registered plugins.
"""

import asyncio
import logging
from typing import List

from .config import Config
from .event_bus import EventBus
from .motion.emotion_mapper import EmotionMapper
from .motion.head_target import HeadTargetBus
from .plugin import Plugin

logger = logging.getLogger(__name__)


class ReachyClawApp:
    """Central application that orchestrates all plugins."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reachy = None  # ReachyMini instance, set by connect_robot() or externally
        self._owns_reachy = False  # True if we created the connection ourselves
        self.events = EventBus()
        self.head_targets = HeadTargetBus()
        self.emotions = EmotionMapper(
            intensity=config.motion_emotion_intensity,
        )

        # Shared flags for inter-plugin coordination
        self.is_speaking = False
        self.running = False
        self.healthy = False

        self._plugins: List[Plugin] = []

    @staticmethod
    def _patch_gstreamer() -> None:
        """Patch SDK GStreamer for Docker/Jetson.

        Fixes:
        1. Camera: DeviceMonitor uses ``device.path`` instead of ``api.v4l2.path``
        2. Audio: no PulseAudio on Jetson, use ALSA directly for Reachy Mini Audio
        3. macOS: disable audio init entirely (no GStreamer audio)
        """
        try:
            from reachy_mini.media.camera_gstreamer import GStreamerCamera
            from reachy_mini.media.media_manager import MediaManager
        except Exception:
            return

        import sys
        if sys.platform != "linux":
            MediaManager._init_audio = lambda self, *a, **kw: None
            logger.debug("Patched SDK MediaManager: audio init disabled (non-Linux)")
        else:
            # Patch GStreamer audio to use ALSA directly (no PulseAudio on Jetson)
            ReachyClawApp._patch_gstreamer_audio()
            logger.debug("Patched SDK GStreamer audio for ALSA")

        _orig = GStreamerCamera.get_video_device

        def _patched(self):
            result = _orig(self)
            # If the original found a valid path, use it
            if result[0]:
                return result

            # Retry with native GStreamer field name
            try:
                from gi.repository import Gst
                from typing import cast
                from reachy_mini.media.camera_constants import (
                    CameraSpecs,
                    ReachyMiniLiteCamSpecs,
                )

                # Gst.init already called by SDK at this point
                monitor = Gst.DeviceMonitor()
                monitor.add_filter("Video/Source")
                monitor.start()
                try:
                    for device in monitor.get_devices():
                        name = device.get_display_name()
                        props = device.get_properties()
                        if props and "Reachy" in name:
                            for field in ("device.path", "object.path"):
                                if props.has_field(field):
                                    path = props.get_string(field)
                                    logger.info(
                                        "GStreamer camera found via %s: %s", field, path
                                    )
                                    return str(path), cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                finally:
                    monitor.stop()
            except Exception as e:
                logger.debug("GStreamer camera patch failed: %s", e)

            return result

        GStreamerCamera.get_video_device = _patched

    @staticmethod
    def _patch_gstreamer_audio() -> None:
        """Patch SDK GStreamer audio to use ALSA directly on Jetson.

        The SDK assumes PulseAudio but Jetson uses bare ALSA.
        Find the Reachy Mini Audio card via ALSA and patch the pipeline init
        to use alsasink/alsasrc instead of pulsesink/pulsesrc.
        """
        try:
            from reachy_mini.media.audio_gstreamer import GStreamerAudio
        except Exception:
            return

        # Find ALSA card number for "Reachy Mini Audio" via /proc/asound/cards
        alsa_device = None
        try:
            import re
            cards_text = open("/proc/asound/cards").read()
            m = re.search(r"^\s*(\d+)\s.*Reachy Mini Audio", cards_text, re.MULTILINE)
            if m:
                alsa_device = f"hw:{m.group(1)},0"
        except Exception as e:
            logger.debug("ALSA card detection failed: %s", e)

        if not alsa_device:
            logger.warning("Reachy Mini Audio ALSA card not found, skipping audio patch")
            return

        logger.info("Patching SDK audio to use ALSA device: %s", alsa_device)

        _orig_record = GStreamerAudio._init_pipeline_record
        _orig_playback = GStreamerAudio._init_pipeline_playback

        def _patched_record(self, pipeline):
            """Use silent dummy source — mic is handled by sounddevice, not SDK."""
            try:
                import gi
                gi.require_version("Gst", "1.0")
                gi.require_version("GstApp", "1.0")
                from gi.repository import Gst, GstApp  # noqa: F811

                self._appsink_audio = Gst.ElementFactory.make("appsink")
                caps = Gst.Caps.from_string(
                    f"audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},"
                    "format=F32LE,layout=interleaved"
                )
                self._appsink_audio.set_property("caps", caps)

                # Use audiotestsrc with silence instead of real mic
                # to avoid competing with sounddevice for the ALSA capture device
                audiosrc = Gst.ElementFactory.make("audiotestsrc")
                audiosrc.set_property("wave", 4)  # 4 = silence
                audiosrc.set_property("is-live", True)
                audioconvert = Gst.ElementFactory.make("audioconvert")
                audioresample = Gst.ElementFactory.make("audioresample")

                for el in [audiosrc, audioconvert, audioresample, self._appsink_audio]:
                    pipeline.add(el)
                audiosrc.link(audioconvert)
                audioconvert.link(audioresample)
                audioresample.link(self._appsink_audio)
                logger.debug("SDK audio record pipeline: dummy (mic via sounddevice)")
            except Exception as e:
                logger.warning("Dummy record pipeline failed: %s, falling back", e)
                _orig_record(self, pipeline)

        def _patched_playback(self, pipeline):
            """Use alsasink instead of pulsesink/autoaudiosink."""
            try:
                import gi
                gi.require_version("Gst", "1.0")
                gi.require_version("GstApp", "1.0")
                from gi.repository import Gst, GstApp  # noqa: F811

                self._appsrc = Gst.ElementFactory.make("appsrc")
                self._appsrc.set_property("format", Gst.Format.TIME)
                self._appsrc.set_property("is-live", True)
                caps = Gst.Caps.from_string(
                    f"audio/x-raw,format=F32LE,channels={self.CHANNELS},"
                    f"rate={self.SAMPLE_RATE},layout=interleaved"
                )
                self._appsrc.set_property("caps", caps)

                audioconvert = Gst.ElementFactory.make("audioconvert")
                audioresample = Gst.ElementFactory.make("audioresample")
                audiosink = Gst.ElementFactory.make("alsasink")
                audiosink.set_property("device", alsa_device)

                for el in [self._appsrc, audioconvert, audioresample, audiosink]:
                    pipeline.add(el)
                self._appsrc.link(audioconvert)
                audioconvert.link(audioresample)
                audioresample.link(audiosink)
                logger.debug("SDK audio playback pipeline: alsasink device=%s", alsa_device)
            except Exception as e:
                logger.warning("ALSA playback pipeline failed: %s, falling back", e)
                _orig_playback(self, pipeline)

        GStreamerAudio._init_pipeline_record = _patched_record
        GStreamerAudio._init_pipeline_playback = _patched_playback

    def connect_robot(self) -> None:
        """Connect to the Reachy Mini robot."""
        try:
            from reachy_mini import ReachyMini
        except ImportError:
            logger.warning("reachy-mini not installed, running without robot")
            self.reachy = None
            return

        self._patch_gstreamer()

        kwargs = {}
        if self.config.reachy_connection_mode != "auto":
            kwargs["connection_mode"] = self.config.reachy_connection_mode
        if self.config.reachy_daemon_port != 8000:
            kwargs["port"] = self.config.reachy_daemon_port
        if self.config.reachy_media_backend != "default":
            kwargs["media_backend"] = self.config.reachy_media_backend

        # macOS lacks GStreamer/gi — force no_media to avoid crash
        import sys
        if sys.platform == "darwin" and "media_backend" not in kwargs:
            kwargs["media_backend"] = "no_media"

        # Auto-spawn daemon for USB-connected Lite
        if self.config.reachy_spawn_daemon:
            self._ensure_daemon(self.config.reachy_serialport)

        import time as _time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.reachy = ReachyMini(**kwargs)
                self.reachy.__enter__()
                self.reachy.enable_motors()
                self._owns_reachy = True
                logger.info("Connected to Reachy Mini (motors enabled)")
                return
            except ImportError as e:
                # GStreamer/gi not available on macOS — retry with no_media
                logger.warning(f"Media backend unavailable ({e}), retrying with no_media")
                try:
                    kwargs["media_backend"] = "no_media"
                    self.reachy = ReachyMini(**kwargs)
                    self.reachy.__enter__()
                    self.reachy.enable_motors()
                    self._owns_reachy = True
                    logger.info("Connected to Reachy Mini (no media, motors enabled)")
                    return
                except Exception as e2:
                    logger.error(f"Failed to connect to Reachy Mini: {e2}")
                    self.reachy = None
                return
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(
                        f"Daemon not ready (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    _time.sleep(wait)
                else:
                    logger.error(f"Failed to connect to Reachy Mini after {max_retries} attempts: {e}")
                    self.reachy = None
            except Exception as e:
                logger.error(f"Failed to connect to Reachy Mini: {e}")
                self.reachy = None
                return

    def _ensure_daemon(self, serialport: str = "auto") -> None:
        """Start Reachy Mini daemon if not already running."""
        import socket

        # Check if daemon is already running on port 7447
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", 7447)) == 0:
                logger.info("Reachy daemon already running on :7447")
                return

        logger.info(f"Starting Reachy daemon (serialport={serialport})...")
        import shutil
        import subprocess
        import time

        daemon_bin = shutil.which("reachy-mini-daemon")
        if not daemon_bin:
            logger.error("reachy-mini-daemon not found in PATH")
            return

        cmd = [daemon_bin, "--serialport", serialport, "--localhost-only", "--deactivate-audio"]
        self._daemon_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for daemon to be ready (zenoh on 7447)
        for i in range(30):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", 7447)) == 0:
                    logger.info(f"Reachy daemon ready ({(i+1)*0.5:.1f}s)")
                    return
            if self._daemon_proc.poll() is not None:
                logger.error("Reachy daemon process exited unexpectedly")
                return
            time.sleep(0.5)

        logger.error("Reachy daemon failed to start (timeout 15s)")

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin. Calls setup() and skips if it returns False."""
        try:
            if not plugin.setup():
                logger.info(f"Plugin '{plugin.name}' skipped (setup returned False)")
                return False
        except Exception as e:
            logger.warning(f"Plugin '{plugin.name}' setup failed: {e}")
            return False

        self._plugins.append(plugin)
        logger.info(f"Plugin '{plugin.name}' registered")
        return True

    async def run(self) -> None:
        """Start all registered plugins concurrently."""
        if not self._plugins:
            logger.warning("No plugins registered, nothing to run")
            return

        self.running = True
        plugin_names = [p.name for p in self._plugins]
        logger.info(f"Starting plugins: {', '.join(plugin_names)}")

        tasks = []
        for plugin in self._plugins:
            plugin._running = True
            tasks.append(asyncio.create_task(plugin.start(), name=plugin.name))

        self.healthy = True
        logger.info("App marked healthy")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Plugins cancelled")
        except Exception as e:
            logger.error(f"Plugin error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop all plugins in reverse registration order."""
        logger.info("Shutting down plugins...")
        self.running = False
        self.healthy = False
        for plugin in reversed(self._plugins):
            try:
                await plugin.stop()
                logger.debug(f"Plugin '{plugin.name}' stopped")
            except Exception as e:
                logger.warning(f"Error stopping plugin '{plugin.name}': {e}")

        if self.reachy:
            try:
                from reachy_mini.utils import create_head_pose

                self.reachy.goto_target(
                    head=create_head_pose(roll=0, pitch=0, yaw=0, degrees=True),
                    duration=0.5,
                )
                self.reachy.set_target_antenna_joint_positions([0.0, 0.0])
            except Exception:
                pass
            # Disable motors before disconnecting
            try:
                self.reachy.disable_motors()
            except Exception:
                pass
            # Only close the connection if we created it ourselves
            if self._owns_reachy:
                try:
                    self.reachy.__exit__(None, None, None)
                except Exception:
                    pass
            self.reachy = None

        self._plugins.clear()
        logger.info("Shutdown complete")
