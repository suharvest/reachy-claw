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
import time
from collections.abc import Awaitable, Callable

from ovs_agent.apps.companion_robot.app import CompanionRobotApp
from ovs_agent.config import Config as OvsConfig

from reachy_voice.audio import DuplexAudioIO
from reachy_voice.config import Config
from reachy_voice.dashboard import DashboardHub, DashboardPlugin
from reachy_voice.motion import MotionController
from reachy_voice.plugins import ReachyMotionToolsPlugin
from reachy_voice.vision import VisionClient
from reachy_voice.vision_analysis import VisionAnalysisError, describe_current_view

logger = logging.getLogger("reachy_voice.conversation")

_TAG_RE = re.compile(r"\[([a-zA-Z_]+)\]")
# Vision-context tag the prompt injects (``[Faces: Alice, Bob]``). Smaller edge
# LLMs sometimes echo it into their reply — strip it from TTS/history so it is
# never spoken. NOT an emotion tag (it has a space/colon, so _TAG_RE misses it).
_FACES_RE = re.compile(r"\[Faces:[^\]]*\]", re.IGNORECASE)
_CONTROL_LINE_RE = re.compile(
    r"(?im)^\s*(?:emotion|mood|action|gesture|play[_ -]?emotion|expression)\s*"
    r"[:=：-]?\s*[a-z_ -]*\s*[.!?。！？,，;；]*\s*$"
)
_CONTROL_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:emotion|mood|action|gesture|play[_ -]?emotion|expression)\s*"
    r"[:=：-]?\s*(?:happy|sad|neutral|excited|curious|surprised|angry|fearful|"
    r"thinking|laugh(?:ing)?|smile|contemplative)?\s*[.!?。！？,，;；-]*\s*"
)
_BARE_EMOTION_WORD_RE = re.compile(
    r"(?is)^\s*(?:happy|sad|neutral|excited|curious|surprised|angry|fearful|"
    r"thinking|laugh(?:ing)?|smile|contemplative)\s*[.!?。！？,，;；-]*\s*"
)
_VISION_REQUEST_RE = re.compile(
    r"(看(一下|看)?|拍(张|一张)?照|摄像头|你.*看.*到|看到.*什么|"
    r"what.*(see|seeing)|take.*photo|camera)",
    re.IGNORECASE,
)
_STOP_SPEAKING_RE = re.compile(r"(停止说话|不要说话|别说话|闭嘴|stop speaking|be quiet)", re.IGNORECASE)


def _compact_text(text: str) -> str:
    return re.sub(r"[\s,，.。!！?？:：;；、-]+", "", text).lower()


def _wake_aliases(wake_word: str) -> set[str]:
    compact = _compact_text(wake_word)
    aliases = {
        compact,
        "你好mini",
        "你好迷你",
        "你好米妮",
        "哈喽mini",
        "hello mini",
        "hey mini",
        "mini",
        "迷你",
        "你好",
    }
    return {_compact_text(alias) for alias in aliases if alias}


def _strip_wake_word(text: str, wake_word: str) -> tuple[bool, str]:
    """Return (woke, remainder), matching wake words despite spaces/case."""
    if not wake_word:
        return True, text
    compact_text = _compact_text(text)
    aliases = _wake_aliases(wake_word)
    matched = next((alias for alias in aliases if alias and alias in compact_text), "")
    if not matched:
        return False, text

    # Also strip the common spaced/non-spaced surface forms for clean prompts.
    variants = {
        wake_word,
        wake_word.replace(" ", ""),
        "你好mini",
        "你好 mini",
        "你好迷你",
        "你好米妮",
        "哈喽mini",
        "hello mini",
        "hey mini",
        "mini",
        "迷你",
        "你好",
    }
    remainder = text
    for variant in variants:
        remainder = re.sub(re.escape(variant), "", remainder, flags=re.IGNORECASE)
    return True, remainder.strip(" ，,。.!！？?")


class _TtsTagFilter:
    """Wraps ``slv.send_text`` so emotion tags like ``[happy]`` are NOT spoken,
    and each emotion fires a callback (→ motion). Streaming-safe: text from an
    unclosed ``[`` is held back until the tag closes, so a tag split across
    tokens is still removed cleanly."""

    def __init__(
        self,
        send_text: Callable[[str], Awaitable[None]],
        on_emotion: Callable[[str], None],
        *,
        emit_emotions: bool = False,
    ) -> None:
        self._send = send_text
        self._on_emotion = on_emotion
        self._emit_emotions = emit_emotions
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
            if self._emit_emotions:
                try:
                    self._on_emotion(m.group(1).lower())
                except Exception:
                    logger.debug("emotion callback failed", exc_info=True)
            return ""

        # Drop any echoed [Faces: ...] context and naked control-text leaks
        # before they reach TTS. The local model sometimes emits fragments like
        # "Emotion happy" when tools are disabled; those are control metadata,
        # never visitor-facing speech.
        head = _FACES_RE.sub("", head)
        head = _CONTROL_LINE_RE.sub("", head)
        head = _CONTROL_PREFIX_RE.sub("", head)
        head = _BARE_EMOTION_WORD_RE.sub("", head)
        cleaned = _TAG_RE.sub(_strip, head)
        if cleaned:
            await self._send(cleaned)

    async def flush(self) -> None:
        if not self._buf:
            return
        tail, self._buf = self._buf, ""
        await self(tail)

SUPPORTED_LANGUAGES = ("zh", "en")

# Hard language lock appended to the profile prompt. Without this the model
# drifts (it replied in English even with asr=zh). A curated exhibition wants
# one consistent language.
_LANGUAGE_LOCK = {
    "zh": (
        "只能用简体中文回复。不要切换语言，即使访客使用其他语言。"
        "只说一句面向访客的话，例如：欢迎，很高兴见到你。"
    ),
    "en": (
        "Reply ONLY in English. Never switch languages, even if the visitor "
        "speaks another language.\n"
        'Example spoken reply: "Welcome! Glad you stopped by."'
    ),
}

_MOTION_TOOL_PROMPT = (
    "Before each spoken reply, call play_emotion exactly once with the emotion "
    "that best fits it. Never write bracketed emotion tags in the reply."
)
_NO_CONTROL_TEXT_PROMPT = (
    "不要输出任何动作、表情、状态、工具、JSON、XML、方括号标签或英文控制前缀。"
    "不要描述自己的表情。只输出一句可以直接说给访客听的中文。"
)


def build_system_prompt(cfg: Config) -> str:
    """Profile base prompt + the hard language-lock instruction."""
    lang = cfg.language if cfg.language in _LANGUAGE_LOCK else "zh"
    motion_instruction = (
        _MOTION_TOOL_PROMPT
        if bool(getattr(cfg, "tools_enabled", False))
        else _NO_CONTROL_TEXT_PROMPT
    )
    return f"{cfg.system_prompt()}\n{motion_instruction}\n\n{_LANGUAGE_LOCK[lang]}"


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
    wake_word: str = ""
    wake_session_timeout_s: float = 0.0
    _awake_until: float = 0.0

    def _is_awake(self) -> bool:
        if not self.wake_word:
            return True
        return time.monotonic() < self._awake_until

    def _mark_awake(self) -> None:
        timeout = max(1.0, float(self.wake_session_timeout_s or 0.0))
        self._awake_until = time.monotonic() + timeout

    async def _speak_direct(self, text: str) -> None:
        await self.slv.send_text(text)
        flush = getattr(self.slv, "flush_tts", None)
        if flush is not None:
            await flush()

    async def _cancel_unwoken_response(self) -> None:
        if not self.config.server_loop_enabled():
            return
        send_json = getattr(self.slv, "_send_json", None)
        if not callable(send_json):
            return
        for event in (
            {"type": "response.cancel"},
            {"type": "input_audio_buffer.clear"},
        ):
            try:
                await send_json(event)
            except Exception:
                logger.debug("failed to cancel unwoken realtime response", exc_info=True)

    async def _handle_visual_request(self) -> bool:
        try:
            reply = await describe_current_view(self.reachy_config)
        except VisionAnalysisError as exc:
            logger.warning("visual analysis failed: %s", exc)
            reply = "我现在还看不清楚画面，视觉分析服务可能没有准备好。"
        await self._speak_direct(reply)
        self._arm_session_idle_reset()
        return True

    async def on_user_utterance(self, text: str, detected_language: str | None = None) -> None:
        raw_text = (text or "").strip()
        woke, stripped = _strip_wake_word(raw_text, self.wake_word)
        if woke:
            self._mark_awake()
            raw_text = stripped or raw_text
        elif not self._is_awake():
            logger.info("ignoring utterance before wake word: %r", raw_text)
            await self._cancel_unwoken_response()
            return
        else:
            self._mark_awake()

        if _STOP_SPEAKING_RE.search(raw_text):
            await self._speak_direct("好的。")
            self._awake_until = 0.0
            return

        if not raw_text:
            await self._speak_direct("我在。")
            self._arm_session_idle_reset()
            return

        if _VISION_REQUEST_RE.search(raw_text):
            await self._handle_visual_request()
            return

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
        tools_enabled=bool(getattr(cfg, "tools_enabled", False)),
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

    def _on_emotion(self, emotion: str) -> None:
        """Emotion tag from the LLM → motion + dashboard."""
        self._motion.play_emotion(emotion)
        self.hub.publish({"type": "emotion", "emotion": emotion})

    def _on_faces(self, payload: dict) -> None:
        """Vision frame → dashboard boxes only.

        Vision no longer drives motors: no person engagement, no gaze tracking,
        and no automatic greeting. The camera stream remains available for the
        dashboard and for explicit visual analysis requests.
        """
        self.hub.publish({"type": "vision_faces", **payload})

    async def start(self) -> None:
        ovs_cfg = build_ovs_config(self.config)
        self._app = ReachyCompanionApp(ovs_cfg)
        if not bool(getattr(self.config, "tools_enabled", False)):
            from ovs_agent.tools import ToolRegistry
            self._app.tool_registry = ToolRegistry()
        # Hand the live robot handle to the app so motion can use it.
        self._app.reachy = self.reachy
        self._app.reachy_config = self.config
        self._app.wake_word = self.config.wake_word
        self._app.wake_session_timeout_s = self.config.wake_session_timeout_s
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
        # Filter all model text before it reaches TTS. This is intentionally
        # installed for client-loop mode, which we use on the local Jetson so
        # no server-side generated control text can bypass the sanitizer.
        if not self._app.config.server_loop_enabled():
            tts_filter = _TtsTagFilter(
                self._app.slv.send_text,
                self._on_emotion,
                emit_emotions=bool(getattr(self.config, "tools_enabled", False)),
            )
            original_flush_tts = self._app.slv.flush_tts

            async def _flush_tts() -> None:
                await tts_filter.flush()
                await original_flush_tts()

            self._app.slv.send_text = tts_filter
            self._app.slv.flush_tts = _flush_tts
        # Vision: publish face telemetry to the dashboard only. It must not
        # drive motors or inject person identity into turns.
        self._app.base_prompt = build_system_prompt(self.config)
        if self._vision.start():
            self._vision.set_listener(self._on_faces)
        # Conversation feed → dashboard (ASR/LLM/state events) AND → motion, so
        # motion freezes while listening (keeps the mic free of servo noise).
        self._app.register(
            DashboardPlugin(self._app, self.hub, on_state=self._motion.set_conv_state)
        )
        # Local edge-llm on this image currently 500s when OpenAI tool schemas
        # are present (tools_render_failed). Keep motion tools opt-in so normal
        # voice replies work in the default local mode.
        if bool(getattr(self.config, "tools_enabled", False)):
            self._app.register(
                ReachyMotionToolsPlugin(
                    self._app,
                    motion=self._motion,
                    on_emotion=self._on_emotion,
                )
            )
        else:
            logger.info("motion tools disabled; not advertising tool schemas")
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
        except Exception:
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
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:
                logger.exception("conversation task errored during shutdown")
        logger.info("ConversationEngine stopped.")
