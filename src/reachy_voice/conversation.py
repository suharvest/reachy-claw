"""Core conversation loop: listen → think → speak.

This is deliberately thin. The whole listen→think→speak pipeline (mic capture,
client VAD, SLV streaming ASR/TTS, edge-LLM, session history, echo filtering) is
provided by the `openvoicestream-agent` (ovs_agent) framework's `CompanionRobotApp`
— which is purpose-built for embodied voice agents like Reachy. We just:

  1. map our minimal `Config` → an `ovs_agent.Config`,
  2. instantiate the companion app and hand it the live `reachy_mini` handle,
  3. run its event loop as a task until the outer ReachyMiniApp stops us.

Reachy registers structured motion tools on the app so local and cloud
Realtime providers drive head/antenna motion without spoken control tags.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Awaitable, Callable

from ovs_agent.apps.companion_robot.app import CompanionRobotApp
from ovs_agent.config import Config as OvsConfig

from reachy_voice.attention import AttentionTracker
from reachy_voice.audio import DuplexAudioIO
from reachy_voice.config import Config
from reachy_voice.dashboard import DashboardHub, DashboardPlugin
from reachy_voice.motion import MotionController
from reachy_voice.plugins import ReachyMotionToolsPlugin
from reachy_voice.vision import VisionClient

logger = logging.getLogger("reachy_voice.conversation")

_TAG_RE = re.compile(r"\[([a-zA-Z_]+)\]")
# Vision-context tag the prompt injects (``[Faces: Alice, Bob]``). Smaller edge
# LLMs sometimes echo it into their reply — strip it from TTS/history so it is
# never spoken. NOT an emotion tag (it has a space/colon, so _TAG_RE misses it).
_FACES_RE = re.compile(r"\[Faces:[^\]]*\]", re.IGNORECASE)


class _TtsTagFilter:
    """Wraps ``slv.send_text`` so emotion tags like ``[happy]`` are NOT spoken,
    and each emotion fires a callback (→ motion). Streaming-safe: text from an
    unclosed ``[`` is held back until the tag closes, so a tag split across
    tokens is still removed cleanly."""

    def __init__(
        self,
        send_text: Callable[[str], Awaitable[None]],
        on_emotion: Callable[[str], None],
    ) -> None:
        self._send = send_text
        self._on_emotion = on_emotion
        self._buf = ""

    async def __call__(self, token: str) -> None:
        self._buf += token
        # Hold back everything from the last unclosed '[' (a tag in progress).
        open_idx = self._buf.rfind("[")
        if open_idx != -1 and "]" not in self._buf[open_idx:]:
            head, self._buf = self._buf[:open_idx], self._buf[open_idx:]
        else:
            head, self._buf = self._buf, ""

        def _strip(m: re.Match) -> str:
            try:
                self._on_emotion(m.group(1).lower())
            except Exception:  # noqa: BLE001 — motion must never break TTS
                logger.debug("emotion callback failed", exc_info=True)
            return ""

        # Drop any echoed [Faces: ...] vision-context tag silently (no emotion
        # callback), THEN strip [emotion] tags (firing _on_emotion per tag).
        head = _FACES_RE.sub("", head)
        cleaned = _TAG_RE.sub(_strip, head)
        if cleaned:
            await self._send(cleaned)

SUPPORTED_LANGUAGES = ("zh", "en")

# Hard language lock appended to the profile prompt. Without this the model
# drifts (it replied in English even with asr=zh). A curated exhibition wants
# one consistent language.
_LANGUAGE_LOCK = {
    "zh": (
        "Reply ONLY in Chinese (简体中文). Never switch languages, even if the "
        "visitor speaks another language.\n"
        'Example spoken reply: "欢迎，很高兴见到你。"'
    ),
    "en": (
        "Reply ONLY in English. Never switch languages, even if the visitor "
        "speaks another language.\n"
        'Example spoken reply: "Welcome! Glad you stopped by."'
    ),
}


def build_system_prompt(cfg: Config) -> str:
    """Profile base prompt + the hard language-lock instruction."""
    lang = cfg.language if cfg.language in _LANGUAGE_LOCK else "zh"
    return f"{cfg.system_prompt()}\n\n{_LANGUAGE_LOCK[lang]}"


class ReachyCompanionApp(CompanionRobotApp):
    """CompanionRobotApp wired for Reachy Mini.

    Adds per-turn vision context: before each user turn, the system prompt is
    rebuilt as base prompt + ``[Faces: …]`` (who the robot currently sees, with
    emotions) — same convention the prompt's "Names in [Faces: ...]" line
    expects. ``vision``/``base_prompt`` are set by the ConversationEngine.
    """

    vision: VisionClient | None = None
    base_prompt: str = ""
    # Reset the session after this many seconds of no user speech (set by the
    # ConversationEngine from cfg.session_reset_idle_s). 0 disables.
    session_reset_idle_s: float = 0.0
    _idle_reset_task: asyncio.Task | None = None

    async def on_user_utterance(self, text: str, detected_language: str | None = None) -> None:
        prompt = self.base_prompt or self.config.system_prompt
        if self.vision is not None:
            faces = self.vision.faces_context()
            if faces:
                prompt = f"{prompt}\n[Faces: {faces}]"
        self.config.system_prompt = prompt  # resolved fresh each turn by ovs
        self._arm_session_idle_reset()  # a long silence ⇒ next visitor starts fresh
        if self.config.server_loop_enabled():
            # Reachy needs the current camera/visitor context to be committed
            # before the provider starts this turn.  The session therefore has
            # create_response=false; update the canonical prompt first, then
            # explicitly create the response.  This is provider-neutral and
            # works for local cascade, OpenAI Realtime, and Qwen adapters.
            await self.slv.update_session({"instructions": prompt})
            await self.slv.create_response()
            return
        await super().on_user_utterance(text, detected_language)

    def _arm_session_idle_reset(self) -> None:
        """(Re)start the no-speech idle timer; each utterance pushes it out.
        Keeps the conversation small/warm per visitor without a wake word."""
        if self.session_reset_idle_s <= 0:
            return
        task = self._idle_reset_task
        if task is not None and not task.done():
            task.cancel()
        try:
            self._idle_reset_task = asyncio.create_task(self._session_idle_reset())
        except RuntimeError:
            pass  # no running loop (e.g. unit tests) — idle reset just disabled

    async def _session_idle_reset(self) -> None:
        try:
            await asyncio.sleep(self.session_reset_idle_s)
            await self.reset_conversation()
            logger.info(
                "session reset after %.0fs idle — fresh context for the next visitor",
                self.session_reset_idle_s,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("session idle-reset failed", exc_info=True)


def build_ovs_config(cfg: Config) -> OvsConfig:
    """Translate our minimal config into the ovs_agent framework config."""
    return OvsConfig(
        # SLV V2V engine (streaming ASR + TTS)
        slv_url=cfg.v2v_url,
        slv_config={
            "asr_language": cfg.language,
            "tts_language": cfg.language,
            "sample_rate": cfg.sample_rate,
            # SERVER VAD — let SLV segment + finalize. Proven to transcribe
            # correctly; the client-eos mode (vad="none" + drive_eos) returned
            # empty finals on this stack.
            "vad": "silero",
            "vad_silence_ms": 500,
            "multi_utterance": True,
            # Manual response creation is intentional: Reachy injects the
            # latest visual context after transcription completes and only
            # then starts the provider response.
            "create_response": False,
            # Native voxedge speed (keepPitch via TTSRateShifter): the v2v server
            # reads cfg.get("tts_speed") and the TTS backend DSP-stretches the
            # streamed PCM. 1.0 = no-op pass-through.
            "tts_speed": cfg.tts_speed,
        },
        # edge-LLM (OpenAI-compatible)
        llm_backend="edge_llm",
        llm_base_url=cfg.edge_llm_url,
        llm_model=cfg.edge_llm_model,
        system_prompt=build_system_prompt(cfg),
        session_tokenizer_model=cfg.edge_llm_model,
        # Bound the context so the session doesn't sit at the trim boundary
        # cold-prefilling the whole history every turn (config-level, reliable —
        # the runtime "history" override alone doesn't drive the trim budget).
        session_max_input_tokens=cfg.session_max_input_tokens,
        # Audio (ovs_agent owns the device via sounddevice; outer app sets no_media)
        audio_input_device=cfg.audio_device or None,
        audio_output_device=cfg.audio_device or None,
        audio_input_sample_rate=cfg.sample_rate,
        # The Reachy USB mic captures quietly; boost it before VAD/ASR (the old
        # reachy-claw used a 3.5x gain for the same reason).
        mic_makeup_gain=cfg.audio_volume,
        # Drop mic audio while the robot is SPEAKING/THINKING so its own TTS
        # echo (speaker -> USB mic) can't open a server-VAD segment that never
        # cleanly ends. The robot hearing itself caused the garbled
        # 'See'/'Name'/'Your name' ASR finals (reproduced in the e2e echo
        # variant, 2026-06-16). Already enabled in voice_arm/voice_rebot_arm.
        mic_drop_while_speaking=True,
        # Server VAD (vad="silero" above) stays primary, but its Paraformer
        # endpoint fires NONDETERMINISTICALLY on trailing silence — when it
        # misses, the turn never gets an asr_final and hangs in THINKING until
        # the 20s watchdog ("卡死"). So the client VAD ALSO drives EOS as a
        # fallback: on client speech-end it sends asr_eos to force finalization.
        # Validated by the e2e harness A/B (2026-06-16): with server VAD still
        # on, drive_eos eliminated the stall with NO premature-cut downside.
        # (The old "empty finals" problem was the vad="none" mode, not this
        # supplement to server VAD.)
        client_vad_backend=cfg.vad_backend,
        client_vad_threshold=cfg.client_vad_threshold,
        client_vad_silence_ms=cfg.client_vad_silence_ms,
        client_vad_drive_eos=True,
        # response.done is remote generation completion; application turn
        # completion waits until the duplex speaker queue has actually drained.
        playback_drain_enabled=True,
        # Timeouts
        llm_first_token_timeout_s=cfg.llm_first_token_timeout_s,
        llm_stream_idle_timeout_s=cfg.llm_stream_idle_timeout_s,
        # Empty allowlist exposes every registered local motion tool. In the
        # default server-loop they are advertised through session.update; the
        # explicit client-loop fallback uses the same registry locally.
        default_mode="chat",
        server_loop=cfg.server_loop,
        realtime_protocol_version=cfg.realtime_protocol_version,
        tools_enabled=True,
        tools_default_allowlist=[],
        tools_max_iterations=3,
        log_level="INFO",
    )


class ConversationEngine:
    """Owns the ovs_agent companion app and its lifecycle.

    Stable seam for `main.py`: ``start()`` / ``stop()`` never change as features
    are added — they go onto the app/plugins instead.
    """

    def __init__(
        self,
        reachy_mini: object,
        config: Config,
        hub: DashboardHub | None = None,
    ) -> None:
        self.reachy = reachy_mini
        self.config = config
        self.hub = hub or DashboardHub()
        self._app: ReachyCompanionApp | None = None
        self._task: asyncio.Task | None = None
        self._motion = MotionController(reachy_mini)
        self._vision = VisionClient(config.vision_url)
        self._attention: AttentionTracker | None = None

    def _on_emotion(self, emotion: str) -> None:
        """Emotion tag from the LLM → motion + dashboard."""
        self._motion.play_emotion(emotion)
        self.hub.publish({"type": "emotion", "emotion": emotion})

    def _on_visitor_engaged(self) -> None:
        """A visitor came close and lingered → greet them (once, then cooldown)."""
        logger.info("visitor engaged — greeting")
        self._motion.play_emotion("welcoming")
        self.hub.publish({"type": "emotion", "emotion": "welcoming"})

    def _on_faces(self, payload: dict) -> None:
        """Vision frame → dashboard boxes + attention/gaze (runs on the vision thread)."""
        self.hub.publish({"type": "vision_faces", **payload})
        if self._attention is not None:
            self._attention.update(payload.get("faces") or [])

    async def start(self) -> None:
        ovs_cfg = build_ovs_config(self.config)
        self._app = ReachyCompanionApp(ovs_cfg)
        # Hand the live robot handle to the app so motion can use it.
        self._app.reachy = self.reachy
        # Idle session-reset (fresh context per visitor; keeps open-mic UX).
        self._app.session_reset_idle_s = self.config.session_reset_idle_s
        # The Reachy USB sound card breaks if separate input+output streams are
        # opened (capture degrades after the first TTS playback). Replace the
        # stock AudioIO with a single full-duplex stream.
        audio = DuplexAudioIO(
            device=self.config.audio_device or None,
            sr=self.config.sample_rate,
            input_channel=getattr(self.config, "input_channel", 0),
        )
        self._app.audio = audio
        # Motion: the compositor reads live TTS loudness off the duplex stream
        # for speech wobble. Start it now so the robot is "alive" immediately.
        # Kill-switch (env REACHY_MOTION=0 or file /tmp/reachy_motion_off) lets us
        # A/B whether motion is degrading ASR without rebuilding.
        motion_enabled = (
            os.environ.get("REACHY_MOTION", "1") != "0"
            and not os.path.exists("/tmp/reachy_motion_off")
        )
        if motion_enabled:
            self._motion.audio = audio
            self._motion.start()
        else:
            logger.warning("MOTION DISABLED (kill-switch) — voice-only diagnostic mode")
        # Legacy client-loop fallback may still emit old [emotion] tags.  A
        # native speech-to-speech response cannot be filtered after synthesis,
        # so Realtime V2 server-loop uses the structured play_emotion tool.
        if not self._app.config.server_loop_enabled():
            self._app.slv.send_text = _TtsTagFilter(
                self._app.slv.send_text, self._on_emotion
            )
        # Vision: faces → per-turn [Faces:] prompt context, dashboard boxes, and
        # attention/gaze (follow + greet a close, lingering visitor).
        self._app.base_prompt = build_system_prompt(self.config)
        if self._vision.start():
            self._app.vision = self._vision
            if getattr(self.config, "attention_enabled", True):
                from reachy_voice.gaze import FaceGaze

                fg = FaceGaze()  # SDK-calibrated face→yaw/pitch (Lite camera)
                self._attention = AttentionTracker(
                    on_gaze=self._motion.set_gaze,
                    on_engage=self._on_visitor_engaged,
                    min_area=self.config.attention_min_area,
                    stable_s=self.config.attention_stable_s,
                    cooldown_s=self.config.attention_cooldown_s,
                    max_yaw=self.config.gaze_max_yaw,
                    max_pitch=self.config.gaze_max_pitch,
                    lost_s=self.config.gaze_lost_s,
                    deadzone=self.config.gaze_deadzone,
                    invert_x=self.config.gaze_invert_x,
                    invert_y=self.config.gaze_invert_y,
                    gaze_fn=fg.yaw_pitch if fg.ok else None,
                )
            self._vision.set_listener(self._on_faces)
        # Conversation feed → dashboard (ASR/LLM/state events) AND → motion, so
        # motion freezes while listening (keeps the mic free of servo noise).
        self._app.register(
            DashboardPlugin(self._app, self.hub, on_state=self._motion.set_conv_state)
        )
        # Local motion tools (move_head/move_antennas/play_emotion/dance):
        # register BEFORE _app.run() so the tools are on the registry before the
        # first turn. Bodies dispatch into the single-writer compositor.
        self._app.register(
            ReachyMotionToolsPlugin(
                self._app,
                motion=self._motion,
                on_emotion=self._on_emotion,
            )
        )
        logger.info(
            "Starting conversation: SLV=%s  LLM=%s (%s)",
            ovs_cfg.slv_url, ovs_cfg.llm_base_url, ovs_cfg.llm_model,
        )
        self._task = asyncio.create_task(self._app.run(), name="ovs_app_run")

    async def set_language(self, lang: str) -> str:
        """Switch the locked language (zh/en) at runtime: re-locks the prompt,
        re-points SLV ASR/TTS, clears history, and reconnects SLV so the next
        turn uses the new language. Returns the applied language."""
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language {lang!r} (use {SUPPORTED_LANGUAGES})")
        self.config.language = lang
        app = self._app
        if app is None:
            return lang
        # next turn resolves config.system_prompt fresh (ovs reads it per turn)
        app.base_prompt = build_system_prompt(self.config)
        app.config.system_prompt = app.base_prompt
        app.slv.config["asr_language"] = lang
        app.slv.config["tts_language"] = lang
        # drop history + stale prefix cache so the new prompt/language take hold
        try:
            await app.reset_conversation()
        except Exception:  # noqa: BLE001
            logger.debug("session.reset() failed during language switch", exc_info=True)
        await app.slv.reconnect()
        if app.config.server_loop_enabled():
            await app._readvertise_after_reconnect()
        self.hub.publish(
            {
                "type": "conversation_language",
                "language": lang,
                "asr_language": lang,
                "tts_language": lang,
            }
        )
        logger.info("conversation language switched to %s", lang)
        return lang

    async def stop(self) -> None:
        self._vision.stop()
        self._motion.stop()
        if self._app is not None:
            self._app.request_shutdown()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:  # noqa: BLE001 — never let teardown raise
                logger.exception("conversation task errored during shutdown")
        logger.info("ConversationEngine stopped.")
