"""Main entry point for Reachy Mini OpenClaw interface."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from reachy_claw.app import ReachyClawApp
from reachy_claw.config import Config, load_config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy SDK/library loggers
    logging.getLogger("reachy_mini.media.media_manager").setLevel(logging.ERROR)
    logging.getLogger("zenoh.handlers").setLevel(logging.CRITICAL)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reachy Mini interface for OpenClaw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML config file (default: auto-detect reachy-claw.yaml or ~/.reachy-claw/config.yaml)",
    )

    # Connection options
    parser.add_argument(
        "--gateway-host", default=None, help="OpenClaw desktop-robot host"
    )
    parser.add_argument(
        "--gateway-port", type=int, default=None, help="OpenClaw desktop-robot port"
    )
    parser.add_argument("--gateway-path", default=None, help="WebSocket path")
    parser.add_argument("--gateway-token", help="Authentication token")

    # Reachy options
    parser.add_argument(
        "--reachy-mode",
        choices=["auto", "localhost_only", "network"],
        default=None,
        help="Reachy Mini connection mode",
    )

    # STT options
    from reachy_claw.backend_registry import get_stt_names, get_tts_names, get_vad_names

    parser.add_argument(
        "--stt",
        choices=get_stt_names(),
        default=None,
        help="Speech-to-text backend (default: whisper)",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size (default: base)",
    )

    # TTS options
    parser.add_argument(
        "--tts",
        choices=get_tts_names(),
        default=None,
        help="Text-to-speech backend (default: elevenlabs)",
    )
    parser.add_argument("--tts-voice", help="TTS voice ID (backend-specific)")
    parser.add_argument("--tts-model", help="TTS model path (for Piper backend)")

    # VAD options
    parser.add_argument(
        "--vad",
        choices=get_vad_names(),
        default=None,
        help="Voice activity detection backend (default: silero)",
    )

    # Remote speech service options
    parser.add_argument(
        "--speech-url",
        default=None,
        help="URL of remote speech service (for sensevoice/kokoro backends)",
    )

    # Audio options
    parser.add_argument("--audio-device", help="Audio input device name")

    # Behavior options
    parser.add_argument("--wake-word", help="Wake word to activate listening")
    parser.add_argument(
        "--no-emotions", action="store_true", help="Disable emotion animations"
    )
    parser.add_argument(
        "--no-idle", action="store_true", help="Disable idle animations"
    )
    parser.add_argument(
        "--no-barge-in", action="store_true", help="Disable barge-in"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=None,
        help="Run in standalone mode without OpenClaw",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run a quick demo of robot capabilities"
    )

    # Vision / face tracking options
    parser.add_argument(
        "--no-face-tracking",
        action="store_true",
        help="Disable face tracking",
    )
    parser.add_argument(
        "--tracker-type",
        choices=["mediapipe", "remote", "none"],
        default=None,
        help="Face tracker backend",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera device index for face tracking",
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)

    if args.gateway_host is not None:
        config.gateway_host = args.gateway_host
    if args.gateway_port is not None:
        config.gateway_port = args.gateway_port
    if args.gateway_path is not None:
        config.gateway_path = args.gateway_path
    if args.gateway_token:
        config.gateway_token = args.gateway_token

    if args.reachy_mode is not None:
        config.reachy_connection_mode = args.reachy_mode
    if args.audio_device is not None:
        config.audio_device = args.audio_device
    if args.stt is not None:
        config.stt_backend = args.stt
    if args.whisper_model is not None:
        config.whisper_model = args.whisper_model
    if args.tts is not None:
        config.tts_backend = args.tts
    if args.vad is not None:
        config.vad_backend = args.vad
    if args.tts_voice is not None:
        config.tts_voice = args.tts_voice
    if args.tts_model is not None:
        config.tts_model = args.tts_model
    if args.wake_word is not None:
        config.wake_word = args.wake_word
    if args.no_emotions:
        config.play_emotions = False
    if args.no_idle:
        config.idle_animations = False
    if args.no_barge_in:
        config.barge_in_enabled = False
    if args.standalone is not None:
        config.standalone_mode = args.standalone

    # Remote speech service
    if args.speech_url:
        config.speech_service_url = args.speech_url

    # Vision / face tracking
    if args.no_face_tracking:
        config.enable_face_tracker = False
    if args.tracker_type is not None:
        config.vision_tracker_type = args.tracker_type
    if args.camera_index is not None:
        config.vision_camera_index = args.camera_index

    return config


async def run_demo() -> int:
    """Run a quick demo of robot capabilities."""
    logging.info("Starting Reachy Mini demo...")

    try:
        from reachy_mini import ReachyMini
        from reachy_mini.utils import create_head_pose
    except ImportError:
        logging.error("reachy-mini package not installed")
        return 1

    reachy = None
    try:
        reachy = ReachyMini()
        reachy.__enter__()
        logging.info("Connected to Reachy Mini!")

        logging.info("Waking up robot...")
        reachy.wake_up()
        await asyncio.sleep(1.0)

        # Nod yes
        logging.info("Moving head - nodding yes...")
        for _ in range(2):
            reachy.goto_target(
                head=create_head_pose(roll=0, pitch=10, degrees=True), duration=0.3
            )
            await asyncio.sleep(0.4)
            reachy.goto_target(
                head=create_head_pose(roll=0, pitch=-10, degrees=True), duration=0.3
            )
            await asyncio.sleep(0.4)

        reachy.goto_target(
            head=create_head_pose(roll=0, pitch=0, degrees=True), duration=0.5
        )
        await asyncio.sleep(0.6)

        # Shake no
        logging.info("Moving head - shaking no...")
        for _ in range(2):
            reachy.goto_target(
                head=create_head_pose(roll=10, pitch=0, degrees=True), duration=0.3
            )
            await asyncio.sleep(0.4)
            reachy.goto_target(
                head=create_head_pose(roll=-10, pitch=0, degrees=True), duration=0.3
            )
            await asyncio.sleep(0.4)

        reachy.goto_target(
            head=create_head_pose(roll=0, pitch=0, degrees=True), duration=0.5
        )
        await asyncio.sleep(0.6)

        # Antennas
        logging.info("Moving antennas...")
        reachy.set_target_antenna_joint_positions([30.0, -30.0])
        await asyncio.sleep(0.5)
        reachy.set_target_antenna_joint_positions([-30.0, 30.0])
        await asyncio.sleep(0.5)
        reachy.set_target_antenna_joint_positions([0.0, 0.0])
        await asyncio.sleep(0.5)

        logging.info("Demo completed successfully!")

    except Exception as e:
        logging.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if reachy:
            reachy.__exit__(None, None, None)

    return 0


async def async_main(config: Config) -> int:
    app = ReachyClawApp(config)

    # Start health endpoint for container orchestration
    from reachy_claw.healthcheck import start_health_server

    health_port = 8641 if config.dashboard_enabled else 8640
    asyncio.create_task(start_health_server(app, port=health_port))

    # Connect robot
    app.connect_robot()

    # Register plugins in order: motion -> face tracker -> conversation
    from reachy_claw.plugins.motion_plugin import MotionPlugin

    if config.enable_motion:
        app.register(MotionPlugin(app))

    if config.enable_face_tracker:
        if config.vision_tracker_type == "remote":
            from reachy_claw.plugins.vision_client_plugin import VisionClientPlugin

            app.register(VisionClientPlugin(app))
        else:
            from reachy_claw.plugins.face_tracker_plugin import FaceTrackerPlugin

            app.register(FaceTrackerPlugin(app))

    from reachy_claw.plugins.conversation_plugin import ConversationPlugin

    app.register(ConversationPlugin(app))

    if config.dashboard_enabled:
        from reachy_claw.plugins.dashboard_plugin import DashboardPlugin

        app.register(DashboardPlugin(app))

    # Daily interaction logger (for diary generation)
    from reachy_claw.plugins.daily_log_plugin import DailyLogPlugin

    app.register(DailyLogPlugin(app))

    # Handle shutdown signals (first = graceful, second = force)
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    _sig_count = 0

    def signal_handler():
        nonlocal _sig_count
        _sig_count += 1
        if _sig_count == 1:
            logging.info("Shutdown signal received")
            shutdown_event.set()
        else:
            logging.info("Force shutdown")
            import os
            os._exit(1)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        run_task = asyncio.create_task(app.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [run_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        return 1
    finally:
        try:
            await asyncio.wait_for(app.shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning("Shutdown timed out after 5s")

    return 0


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.demo:
        logging.info("Running Reachy Mini demo")
        exit_code = asyncio.run(run_demo())
        sys.exit(exit_code)

    config = create_config(args)

    if config.standalone_mode:
        logging.info("Starting Reachy Mini in standalone mode (no gateway)")
    else:
        logging.info("Starting Reachy Mini OpenClaw interface")
        logging.info(f"Server: {config.desktop_robot_url}")

    logging.info(f"STT: {config.stt_backend} ({config.whisper_model})")
    logging.info(f"TTS: {config.tts_backend}")
    logging.info(f"VAD: {config.vad_backend}")
    if config.wake_word:
        logging.info(f"Wake word: {config.wake_word}")
    if config.barge_in_enabled:
        logging.info("Barge-in: enabled")
    if config.enable_face_tracker:
        logging.info(f"Face tracking: {config.vision_tracker_type}")

    exit_code = asyncio.run(async_main(config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
