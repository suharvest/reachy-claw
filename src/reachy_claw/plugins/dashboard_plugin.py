"""DashboardPlugin — exhibition dashboard serving HTML + WebSocket state."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..plugin import Plugin

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "dashboard_static"


class DashboardPlugin(Plugin):
    """Serves exhibition dashboard UI and broadcasts robot state via WebSocket."""

    name = "dashboard"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._site = None
        self._runner = None
        self._ws_clients: set = set()
        self._last_llm_emotion: str | None = None
        self._capture_count: int = 0
        self._audio_card: str | None = None
        self._voice_clone_supported: bool = False

    def setup(self) -> bool:
        try:
            import aiohttp  # noqa: F401
            from aiohttp import web  # noqa: F401
        except ImportError:
            logger.warning("aiohttp not installed — dashboard disabled")
            return False
        return True

    async def start(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/stream", self._handle_stream_proxy)
        # Diary API
        app.router.add_get("/api/diaries", self._handle_diary_list)
        app.router.add_get("/api/diary/{date}", self._handle_diary_get)
        # Capture gallery proxy (same-origin for vision-trt)
        app.router.add_get("/api/captures/list", self._proxy_captures_list)
        app.router.add_get("/api/captures/image/{filename}", self._proxy_captures_image)
        # Ollama models API (proxy to avoid CORS)
        app.router.add_get("/api/ollama/models", self._proxy_ollama_models)
        # AI commands endpoint (unified tool execution for skills)
        app.router.add_post("/api/ai/commands", self._handle_ai_command)
        app.router.add_static("/static", STATIC_DIR, show_index=False)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = self.app.config.dashboard_port
        self._site = web.TCPSite(self._runner, "0.0.0.0", port)
        await self._site.start()
        logger.info("Dashboard listening on http://0.0.0.0:%d", port)

        # Restore persisted volume or default to 80%
        startup_vol = getattr(self.app.config, "dashboard_volume", 80)
        if not isinstance(startup_vol, int) or startup_vol < 0:
            startup_vol = 80
        await self._set_volume(startup_vol)

        # Restore capture count from vision-trt
        await self._restore_capture_count()

        # Subscribe to EventBus
        bus = self.app.events
        bus.subscribe("asr_partial", self._on_asr_partial)
        bus.subscribe("asr_final", self._on_asr_final)
        bus.subscribe("llm_delta", self._on_llm_delta)
        bus.subscribe("llm_end", self._on_llm_end)
        bus.subscribe("state_change", self._on_state_change)
        bus.subscribe("emotion", self._on_emotion)
        bus.subscribe("observation", self._on_observation)
        bus.subscribe("vision_faces", self._on_vision_faces)
        bus.subscribe("smile_capture", self._on_smile_capture)

        # State polling loop (5Hz)
        while self._running:
            try:
                await asyncio.wait_for(
                    self._broadcast_robot_state(), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("broadcast_robot_state timed out, skipping")
            except Exception as e:
                logger.warning("broadcast_robot_state error: %s", e)
            await asyncio.sleep(0.2)

    async def stop(self) -> None:
        await super().stop()

        bus = self.app.events
        bus.unsubscribe("asr_partial", self._on_asr_partial)
        bus.unsubscribe("asr_final", self._on_asr_final)
        bus.unsubscribe("llm_delta", self._on_llm_delta)
        bus.unsubscribe("llm_end", self._on_llm_end)
        bus.unsubscribe("state_change", self._on_state_change)
        bus.unsubscribe("emotion", self._on_emotion)
        bus.unsubscribe("observation", self._on_observation)
        bus.unsubscribe("vision_faces", self._on_vision_faces)
        bus.unsubscribe("smile_capture", self._on_smile_capture)

        # Close all WebSocket connections
        for ws in list(self._ws_clients):
            await ws.close()
        self._ws_clients.clear()

        if self._runner:
            await self._runner.cleanup()

    # ── Config persistence ──────────────────────────────────────────────

    def _save_overrides(self, fields: list[str]) -> None:
        """Persist config fields to runtime-overrides.yaml."""
        try:
            from ..config import save_runtime_overrides
            save_runtime_overrides(self.app.config, fields)
        except Exception as e:
            logger.warning("Failed to save config overrides: %s", e)

    # ── HTTP handlers ─────────────────────────────────────────────────

    async def _handle_health(self, request):
        """Health endpoint on dashboard port for container orchestration."""
        from aiohttp import web
        import json

        if self.app.healthy:
            body = {"status": "ok", "robot_connected": self.app.reachy is not None}
            return web.Response(text=json.dumps(body), content_type="application/json")
        return web.Response(
            text=json.dumps({"status": "starting"}),
            content_type="application/json",
            status=503,
        )

    async def _handle_ai_command(self, request):
        """Unified tool execution endpoint for skills.

        POST /api/ai/commands
        Body: {"command": "reachy_move_head", "args": {"yaw": 10, "pitch": -5}}

        Strips known prefixes (reachy_, sensecraft_) to get the action name,
        then dispatches to the conversation plugin's robot command handler.
        """
        from aiohttp import web
        import json as _json

        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"status": "error", "message": "Invalid JSON"}, status=400,
            )
        command = data.get("command", "")
        args = data.get("args", {})
        if not command:
            return web.json_response(
                {"status": "error", "message": "Missing 'command' field"}, status=400,
            )

        # Strip prefix to get action name: reachy_move_head → move_head
        for prefix in ("reachy_", "sensecraft_"):
            if command.startswith(prefix):
                action = command[len(prefix):]
                break
        else:
            action = command

        # Find the conversation plugin and execute
        conv = self.app.get_plugin("ConversationPlugin")
        if not conv:
            return web.json_response(
                {"status": "error", "message": "ConversationPlugin not available"},
                status=503,
            )
        result = await asyncio.to_thread(conv._execute_robot_command, action, args)
        return web.json_response(result)

    async def _handle_index(self, request):
        from aiohttp import web

        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return web.Response(text="Dashboard HTML not found", status=404)
        return web.FileResponse(index_path)

    async def _handle_diary_list(self, request):
        """Return list of available diary dates, sorted descending."""
        from aiohttp import web

        diary_dir = self._diary_dir()
        if not diary_dir.exists():
            return web.json_response({"dates": []})

        dates = sorted(
            [f.stem for f in diary_dir.glob("*.json") if f.stem[:4].isdigit()],
            reverse=True,
        )
        return web.json_response({"dates": dates})

    async def _handle_diary_get(self, request):
        """Return diary JSON for a specific date."""
        from aiohttp import web

        date = request.match_info["date"]
        diary_dir = self._diary_dir()

        if date == "latest":
            files = sorted(diary_dir.glob("*.json"), reverse=True) if diary_dir.exists() else []
            if not files:
                return web.json_response({"error": "No diaries found"}, status=404)
            filepath = files[0]
        else:
            filepath = diary_dir / f"{date}.json"

        if not filepath.exists():
            return web.json_response({"error": f"No diary for {date}"}, status=404)

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_ws(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        logger.info("Dashboard WS client connected (%d total)", len(self._ws_clients))

        try:
            async for msg in ws:
                if msg.type == 1:  # TEXT
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(data)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"WS message error: {e}")
        finally:
            self._ws_clients.discard(ws)
            logger.info("Dashboard WS client disconnected (%d remaining)", len(self._ws_clients))

        return ws

    async def _handle_ws_message(self, data: dict) -> None:
        """Handle client → server WS messages."""
        msg_type = data.get("type")
        logger.info("WS message received: %s", msg_type)
        if msg_type == "set_mode":
            mode = data.get("mode", "conversation")
            if mode not in ("conversation", "monologue", "interpreter"):
                return
            # Save previous mode for potential restore
            prev_mode = self.app.config.conversation_mode
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "switch_mode"):
                conv.switch_mode(mode)
            self._save_overrides(["conversation_mode"])
            await self._broadcast({
                "type": "mode_changed",
                "mode": mode,
                "prev_mode": prev_mode,
            })

        elif msg_type == "get_mode":
            conv = self.app.get_plugin("conversation")
            mode = self.app.config.conversation_mode
            if conv and getattr(conv, "_mode_manager", None):
                mode = conv._mode_manager.current.name
            await self._broadcast({"type": "mode", "mode": mode})

        elif msg_type == "set_interpreter_langs":
            source = data.get("source", "Chinese")
            target = data.get("target", "English")
            self.app.config.interpreter_source_lang = source
            self.app.config.interpreter_target_lang = target
            # If currently in interpreter mode, re-apply with new languages
            conv = self.app.get_plugin("conversation")
            if conv and getattr(conv, "_mode_manager", None) and conv._mode_manager.current.name == "interpreter":
                conv.switch_mode("interpreter")
            self._save_overrides(["interpreter_source_lang", "interpreter_target_lang"])
            await self._broadcast({
                "type": "interpreter_langs_changed",
                "source": source,
                "target": target,
            })

        elif msg_type == "get_prompts":
            from ..llm import DEFAULT_SYSTEM_PROMPT, MONOLOGUE_SYSTEM_PROMPT, INTERPRETER_SYSTEM_PROMPT

            cfg = self.app.config
            interp_default = INTERPRETER_SYSTEM_PROMPT.format(
                source_lang=cfg.interpreter_source_lang,
                target_lang=cfg.interpreter_target_lang,
            )
            await self._broadcast({
                "type": "prompts",
                "conversation": cfg.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT,
                "monologue": cfg.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT,
                "interpreter": cfg.interpreter_prompt or interp_default,
            })

        elif msg_type == "set_prompt":
            from ..llm import DEFAULT_SYSTEM_PROMPT, MONOLOGUE_SYSTEM_PROMPT, INTERPRETER_SYSTEM_PROMPT, OllamaClient

            mode = data.get("mode")
            prompt = data.get("prompt", "").strip()
            if mode == "conversation":
                self.app.config.ollama_system_prompt = prompt
            elif mode == "monologue":
                self.app.config.ollama_monologue_prompt = prompt
            elif mode == "interpreter":
                self.app.config.interpreter_prompt = prompt
            else:
                return

            # Hot-apply: if OllamaClient is active in matching mode, update live
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "_client") and isinstance(conv._client, OllamaClient):
                current_mode = conv._mode_manager.current.name if getattr(conv, "_mode_manager", None) else "conversation"
                if mode == "interpreter" and current_mode == "interpreter":
                    interp_default = INTERPRETER_SYSTEM_PROMPT.format(
                        source_lang=self.app.config.interpreter_source_lang,
                        target_lang=self.app.config.interpreter_target_lang,
                    )
                    conv._client._config.system_prompt = prompt or interp_default
                elif (mode == "monologue" and current_mode == "monologue") or (mode == "conversation" and current_mode == "conversation"):
                    conv._client._config.system_prompt = prompt or (
                        MONOLOGUE_SYSTEM_PROMPT if mode == "monologue" else DEFAULT_SYSTEM_PROMPT
                    )

            field_map = {
                "conversation": "ollama_system_prompt",
                "monologue": "ollama_monologue_prompt",
                "interpreter": "interpreter_prompt",
            }
            self._save_overrides([field_map[mode]])
            await self._broadcast({"type": "prompt_saved", "mode": mode})

        elif msg_type == "clear_captures":
            await self._clear_captures()

        elif msg_type == "get_volume":
            vol = await self._get_volume()
            await self._broadcast({"type": "volume", "volume": vol})

        elif msg_type == "set_volume":
            vol = max(0, min(100, int(data.get("volume", 80))))
            await self._set_volume(vol)
            self.app.config.dashboard_volume = vol  # type: ignore[attr-defined]
            self._save_overrides(["dashboard_volume"])
            await self._broadcast({"type": "volume", "volume": vol})

        elif msg_type == "get_history":
            conv = self.app.get_plugin("conversation")
            turns = 0
            if conv and hasattr(conv, "_client") and hasattr(conv._client, "_config"):
                turns = conv._client._config.max_history
            await self._broadcast({"type": "history", "turns": turns})

        elif msg_type == "set_history":
            turns = max(0, min(20, int(data.get("turns", 0))))
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "_client") and hasattr(conv._client, "_config"):
                conv._client._config.max_history = turns
                conv._client._history.clear()  # reset on change
            self.app.config.ollama_max_history = turns
            self._save_overrides(["ollama_max_history"])
            await self._broadcast({"type": "history", "turns": turns})

        elif msg_type == "restart_services":
            await self._restart_services()

        elif msg_type == "set_motor":
            motion = self.app.get_plugin("motion")
            if motion:
                enabled = data.get("enabled", True)
                preset = data.get("preset", "moderate")
                motion.set_motor_enabled(enabled)
                if enabled:
                    motion.apply_motor_preset(preset)
                # Persist motor state
                self.app.config.motor_enabled = enabled  # type: ignore[attr-defined]
                self.app.config.motor_preset = preset  # type: ignore[attr-defined]
                self._save_overrides(["motor_enabled", "motor_preset"])
                await self._broadcast({
                    "type": "motor_state",
                    **motion.get_motor_state(),
                })

        elif msg_type == "get_motor":
            motion = self.app.get_plugin("motion")
            if motion:
                await self._broadcast({
                    "type": "motor_state",
                    **motion.get_motor_state(),
                })

        elif msg_type == "get_voice":
            await self._broadcast(self._get_voice_settings())

        elif msg_type == "set_voice":
            voice_name = data.get("voice_name")  # Cloned voice name (optional)
            sid = int(data.get("speaker_id", 3))
            pitch = float(data.get("pitch_shift", 0.0))
            speed = float(data.get("speed", 1.0))
            backend = self.app.config.tts_backend  # "matcha" or "kokoro"

            if voice_name:
                # Clone mode: load embedding and set to TTS instance
                embedding_path = self.app.config.cache_dir / "voices" / f"{voice_name}.bin"
                self.app.config.cloned_voice_name = voice_name
                # Directly set to running TTS instance
                conv = self.app.get_plugin("conversation")
                if conv and hasattr(conv, "_tts"):
                    tts = conv._tts
                    if hasattr(tts, "_backend"):
                        tts = tts._backend
                    if hasattr(tts, "set_cloned_voice"):
                        tts.set_cloned_voice(str(embedding_path), voice_name)
                    # Also update pitch/speed
                    if hasattr(tts, "_pitch_shift"):
                        tts._pitch_shift = pitch
                    if hasattr(tts, "_speed"):
                        tts._speed = speed
                setattr(self.app.config, f"{backend}_pitch_shift", pitch)
                setattr(self.app.config, f"{backend}_speed", speed)
                self._save_overrides(["cloned_voice_name", f"{backend}_pitch_shift", f"{backend}_speed"])
            else:
                # Speaker ID mode
                setattr(self.app.config, f"{backend}_speaker_id", sid)
                setattr(self.app.config, f"{backend}_pitch_shift", pitch)
                setattr(self.app.config, f"{backend}_speed", speed)
                self.app.config.cloned_voice_name = None
                # Hot-apply to running TTS backend
                conv = self.app.get_plugin("conversation")
                if conv and hasattr(conv, "_tts"):
                    tts = conv._tts
                    if hasattr(tts, "_backend"):
                        tts = tts._backend
                    if hasattr(tts, "_speaker_id"):
                        tts._speaker_id = sid
                    if hasattr(tts, "_pitch_shift"):
                        tts._pitch_shift = pitch
                    if hasattr(tts, "_speed"):
                        tts._speed = speed
                    # Clear cloned voice
                    if hasattr(tts, "set_cloned_voice"):
                        tts.set_cloned_voice(None)
                self._save_overrides([
                    f"{backend}_speaker_id",
                    f"{backend}_pitch_shift",
                    f"{backend}_speed",
                    "cloned_voice_name",
                ])
            await self._broadcast(self._get_voice_settings())

        elif msg_type == "get_tts_capabilities":
            logger.info("Received get_tts_capabilities request")
            await self._send_tts_capabilities()

        elif msg_type == "get_cloned_voices":
            logger.info("Received get_cloned_voices request")
            await self._send_cloned_voices()

        elif msg_type == "clone_voice":
            name = data.get("name", "").strip()
            audio_b64 = data.get("audio_b64", "")
            await self._handle_clone_voice(name, audio_b64)

        elif msg_type == "get_llm":
            cfg = self.app.config
            await self._broadcast({
                "type": "llm_settings",
                "backend": cfg.llm_backend,
                "model": cfg.ollama_model,
                "ollama_url": cfg.ollama_base_url,
                "gateway_host": cfg.gateway_host,
                "gateway_port": cfg.gateway_port,
            })

        elif msg_type == "set_llm":
            backend = data.get("backend", self.app.config.llm_backend)
            model = data.get("model")
            ollama_url = data.get("ollama_url")
            gateway_host = data.get("gateway_host")
            gateway_port = data.get("gateway_port")

            # Update URL config first (before backend switch)
            if ollama_url:
                self.app.config.ollama_base_url = ollama_url
            if gateway_host:
                self.app.config.gateway_host = gateway_host
            if gateway_port:
                self.app.config.gateway_port = gateway_port

            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "switch_backend"):
                try:
                    # Pass URL changes to switch_backend for live reconnection
                    await conv.switch_backend(
                        backend,
                        model,
                        ollama_url=ollama_url,
                        gateway_host=gateway_host,
                        gateway_port=gateway_port,
                    )
                except Exception as e:
                    logger.error("Backend switch failed: %s", e)
                    await self._broadcast({"type": "toast", "text": f"Switch failed: {e}", "error": True})
                    return

            fields = ["llm_backend"]
            if model:
                self.app.config.ollama_model = model
                fields.append("ollama_model")
            if ollama_url:
                fields.append("ollama_base_url")
            if gateway_host:
                fields.append("gateway_host")
            if gateway_port:
                fields.append("gateway_port")

            self._save_overrides(fields)
            await self._broadcast({
                "type": "llm_settings",
                "backend": self.app.config.llm_backend,
                "model": self.app.config.ollama_model,
                "ollama_url": self.app.config.ollama_base_url,
                "gateway_host": self.app.config.gateway_host,
                "gateway_port": self.app.config.gateway_port,
            })

        elif msg_type == "set_vlm":
            enabled = bool(data.get("enabled", False))
            self.app.config.enable_vlm = enabled
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "_client") and hasattr(conv._client, "_config"):
                conv._client._config.enable_vlm = enabled
                # Clear history so stale non-VLM replies don't pollute context
                conv._client._history.clear()
            self._save_overrides(["enable_vlm"])
            await self._broadcast({"type": "vlm_state", "enabled": enabled})

        elif msg_type == "get_vlm":
            await self._broadcast({"type": "vlm_state", "enabled": self.app.config.enable_vlm})

        elif msg_type == "set_bargein":
            enabled = bool(data.get("enabled", True))
            self.app.config.barge_in_enabled = enabled
            self._save_overrides(["barge_in_enabled"])
            await self._broadcast({"type": "bargein_state", "enabled": enabled})

        elif msg_type == "get_bargein":
            await self._broadcast({"type": "bargein_state", "enabled": self.app.config.barge_in_enabled})

        elif msg_type == "set_vad_threshold":
            val = float(data.get("value", 0.3))
            self.app.config.silero_threshold = val
            self._save_overrides(["silero_threshold"])
            await self._broadcast({"type": "vad_threshold", "value": val})

        elif msg_type == "set_energy_threshold":
            val = float(data.get("value", 0.02))
            self.app.config.barge_in_energy_threshold = val
            self._save_overrides(["barge_in_energy_threshold"])
            await self._broadcast({"type": "energy_threshold", "value": val})

        elif msg_type == "get_capture_info":
            await self._send_capture_info()

        elif msg_type == "diary_narrate_start":
            date = data.get("date", "")
            # Enter narration mode — suppress conversation/monologue/interpreter
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "enter_narration_mode"):
                conv.enter_narration_mode()
            asyncio.create_task(self._narrate_diary(date))

        elif msg_type == "diary_narrate_stop":
            self._narration_active = False
            # Interrupt TTS and exit narration mode
            conv = self.app.get_plugin("conversation")
            if conv:
                if hasattr(conv, "stop_speaking"):
                    asyncio.create_task(conv.stop_speaking())
                if hasattr(conv, "exit_narration_mode"):
                    conv.exit_narration_mode()
            # Notify frontend to close overlay
            await self._broadcast({"type": "diary_narrate_end"})

        elif msg_type == "send_message":
            # External text input (from SenseCraft etc.) - send to LLM directly
            text = data.get("text", "")
            if text.strip():
                conv = self.app.get_plugin("conversation")
                if conv and hasattr(conv, "_process_and_send"):
                    # Use the same path as ASR input
                    asyncio.create_task(conv._process_and_send(text.strip()))

    # ── Diary Narration ─────────────────────────────────────────────

    @staticmethod
    def _diary_dir() -> Path:
        """Return diary storage directory (DATA_DIR in Docker, ~/.reachy-claw/ locally)."""
        import os

        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            return Path(data_dir) / "diaries"
        return Path.home() / ".reachy-claw" / "diaries"

    _narration_active: bool = False

    async def _narrate_diary(self, date: str) -> None:
        """Read a diary aloud section by section via TTS."""
        diary_dir = self._diary_dir()
        filepath = diary_dir / f"{date}.json"
        if not filepath.exists():
            logger.warning("Narrate: no diary for %s", date)
            return

        try:
            diary = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Narrate: failed to read diary %s: %s", date, e)
            return

        self._narration_active = True
        sections = diary.get("sections", [])

        conv = self.app.get_plugin("conversation")

        for section in sections:
            if not self._narration_active:
                break

            section_id = section.get("id", "")
            content = section.get("content", "")
            if not content:
                continue

            # Notify frontend to highlight this section
            await self._broadcast({
                "type": "diary_narrate_focus",
                "section_id": section_id,
                "state": "speaking",
            })

            # Speak via TTS
            if conv and hasattr(conv, "speak_text"):
                try:
                    await conv.speak_text(content)
                except Exception as e:
                    logger.warning("Narrate TTS error: %s", e)

            if not self._narration_active:
                break

            await self._broadcast({
                "type": "diary_narrate_focus",
                "section_id": section_id,
                "state": "done",
            })

            # Pause between sections for rhythm
            await asyncio.sleep(2.0)

        self._narration_active = False
        # Restore normal conversation mode
        conv = self.app.get_plugin("conversation")
        if conv and hasattr(conv, "exit_narration_mode"):
            conv.exit_narration_mode()
        await self._broadcast({"type": "diary_narrate_end"})

    def _get_voice_settings(self) -> dict:
        """Return current voice settings for the active TTS backend."""
        backend = self.app.config.tts_backend
        return {
            "type": "voice_settings",
            "speaker_id": getattr(self.app.config, f"{backend}_speaker_id", 0),
            "pitch_shift": getattr(self.app.config, f"{backend}_pitch_shift", 0.0),
            "speed": getattr(self.app.config, f"{backend}_speed", 1.0),
            "voice_clone_supported": getattr(self, "_voice_clone_supported", False),
            "cloned_voice_name": getattr(self.app.config, "cloned_voice_name", None),
        }

    async def _send_tts_capabilities(self) -> None:
        """Query jetson-voice TTS capabilities and cache result."""
        import aiohttp
        url = f"{self.app.config.speech_service_url}/tts/capabilities"
        logger.info("Querying TTS capabilities from %s", url)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        capabilities = data.get("capabilities", [])
                        self._voice_clone_supported = "voice_clone" in capabilities
                        logger.info("TTS capabilities: %s, voice_clone=%s", capabilities, self._voice_clone_supported)
                        await self._broadcast({
                            "type": "tts_capabilities",
                            "voice_clone": self._voice_clone_supported,
                            "backend": data.get("backend"),
                        })
                        return
        except Exception as e:
            logger.warning("Failed to get TTS capabilities: %s", e)
        self._voice_clone_supported = False
        await self._broadcast({"type": "tts_capabilities", "voice_clone": False})

    async def _send_cloned_voices(self) -> None:
        """List cloned voice embeddings from cache/voices/ directory."""
        import os
        voices_dir = Path(os.environ.get("DATA_DIR", str(self.app.config.cache_dir))) / "voices"
        voices = []
        if voices_dir.is_dir():
            for f in voices_dir.iterdir():
                if f.suffix == ".bin":
                    voices.append({"name": f.stem, "filename": f.name})
        await self._broadcast({"type": "cloned_voices", "voices": voices})

    async def _handle_clone_voice(self, name: str, audio_b64: str) -> None:
        """Extract embedding from audio, save to cache/voices/ directory."""
        import base64
        import os
        import tempfile

        import aiohttp

        if not name or not audio_b64:
            await self._broadcast({"type": "clone_voice_result", "error": "Missing name or audio"})
            return

        # Sanitize name (alphanumeric + underscore only)
        import re
        name = name[:32]
        if not name or not name.strip():
            await self._broadcast({"type": "clone_voice_result", "error": "Please enter a voice name"})
            return

        audio_bytes = base64.b64decode(audio_b64)
        
        # Detect audio format from magic bytes
        audio_format = "webm"
        if len(audio_bytes) >= 4:
            if audio_bytes[:4] == b'RIFF':
                audio_format = "wav"
            elif audio_bytes[:3] == b'ID3':
                audio_format = "mp3"
            elif len(audio_bytes) >= 2 and audio_bytes[:2] == b'\xff\xfb':
                audio_format = "mp3"
            elif audio_bytes[:4] == b'ftyp':
                audio_format = "m4a"
            
        logger.info("Received audio: %d bytes, format=%s", len(audio_bytes), audio_format)

        # Convert to WAV if needed
        if audio_format != "wav":
            logger.info("Converting %s to WAV...", audio_format)
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name
            tmp_out_path = tmp_in_path.replace(f".{audio_format}", ".wav")
            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "24000", "-ac", "1", tmp_out_path],
                    capture_output=True, timeout=30
                )
                if result.returncode != 0:
                    logger.error("ffmpeg failed: %s", result.stderr.decode())
                    await self._broadcast({"type": "clone_voice_result", "error": "Audio conversion failed"})
                    os.unlink(tmp_in_path)
                    return
                with open(tmp_out_path, "rb") as f:
                    audio_bytes = f.read()
                logger.info("Converted to WAV: %d bytes", len(audio_bytes))
            finally:
                os.unlink(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)

        url = f"{self.app.config.speech_service_url}/tts/clone/embedding"

        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("file", audio_bytes, filename="audio.wav", content_type="audio/wav")
                async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        logger.error("Embedding API error: %s", err)
                        await self._broadcast({"type": "clone_voice_result", "error": err})
                        return
                    result = await resp.json()
                    embedding_b64 = result["speaker_embedding_b64"]
                    embedding_bytes = base64.b64decode(embedding_b64)

            # Save embedding to voices directory
            voices_dir = Path(os.environ.get("DATA_DIR", str(self.app.config.cache_dir))) / "voices"
            voices_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{name}.bin"
            filepath = voices_dir / filename
            filepath.write_bytes(embedding_bytes)

            logger.info("Saved cloned voice: %s (%d bytes)", filepath, len(embedding_bytes))
            await self._broadcast({
                "type": "clone_voice_result",
                "success": True,
                "voice": {"name": name, "filename": filename}
            })

        except Exception as e:
            logger.error("Voice clone failed: %s", e)
            await self._broadcast({"type": "clone_voice_result", "error": str(e)})

    async def _send_capture_info(self) -> None:
        """Send capture storage info (path + count) to dashboard."""
        import os
        # Use HOST_DATA_DIR for display (host-visible path), fall back to DATA_DIR
        data_dir = os.environ.get("HOST_DATA_DIR") or os.environ.get("DATA_DIR", "~/reachy-data")
        host_path = os.path.join(data_dir, "vision", "captures")

        # Get count from vision-trt API
        count = 0
        vision_url = self.app.config.vision_service_url
        host = vision_url.replace("tcp://", "").split(":")[0]
        api_url = f"http://{host}:8630/api/captures/count"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        count = data.get("count", 0)
        except Exception:
            pass

        await self._broadcast({
            "type": "capture_info",
            "path": host_path,
            "count": count,
        })

    async def _find_audio_card(self) -> str | None:
        """Find ALSA card number for Reachy Mini Audio by scanning /proc/asound/cards."""
        if self._audio_card is not None:
            return self._audio_card
        try:
            proc = await asyncio.create_subprocess_exec(
                "cat", "/proc/asound/cards",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            import re
            for line in stdout.decode().splitlines():
                if "Reachy Mini Audio" in line:
                    m = re.match(r"\s*(\d+)\s+\[", line)
                    if m:
                        self._audio_card = m.group(1)
                        logger.info("Found Reachy Mini Audio on card %s", self._audio_card)
                        return self._audio_card
        except Exception:
            pass
        return None

    async def _get_volume(self) -> int:
        """Read current ALSA volume for Reachy Mini Audio, return as UI 0-100."""
        card = await self._find_audio_card()
        if card is None:
            return 80
        try:
            proc = await asyncio.create_subprocess_exec(
                "amixer", "-c", card, "get", "PCM",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            import re
            m = re.search(r"\[(\d+)%\]", stdout.decode())
            alsa_vol = int(m.group(1)) if m else 80
            return self._alsa_to_ui(alsa_vol)
        except Exception:
            return 80

    @staticmethod
    def _ui_to_alsa(ui_percent: int) -> int:
        """Map UI 0-100 to useful ALSA range.

        ALSA 0-60% is inaudible on Reachy Mini Audio, so we map:
          UI  0   → ALSA 0   (mute)
          UI  1   → ALSA 60  (minimum audible)
          UI 100  → ALSA 100 (maximum)
        Linear mapping over the audible 60-100 ALSA range.
        """
        if ui_percent <= 0:
            return 0
        return int(60 + (ui_percent / 100.0) * 40)

    @staticmethod
    def _alsa_to_ui(alsa_percent: int) -> int:
        """Reverse map ALSA percentage back to UI 0-100."""
        if alsa_percent <= 0:
            return 0
        if alsa_percent <= 60:
            return 1
        return max(1, min(100, int((alsa_percent - 60) / 40.0 * 100)))

    async def _set_volume(self, ui_percent: int) -> None:
        """Set ALSA volume for Reachy Mini Audio."""
        card = await self._find_audio_card()
        if card is None:
            logger.warning("Cannot set volume: Reachy Mini Audio card not found")
            return
        alsa_vol = self._ui_to_alsa(ui_percent)
        try:
            await asyncio.create_subprocess_exec(
                "amixer", "-c", card, "set", "PCM", f"{alsa_vol}%",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            # Also set PCM,1 (mono channel)
            await asyncio.create_subprocess_exec(
                "amixer", "-c", card, "set", "PCM,1", f"{alsa_vol}%",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            logger.info("Volume set to %d%% (ALSA %d%%)", ui_percent, alsa_vol)
        except Exception as e:
            logger.warning("Failed to set volume: %s", e)

    async def _restore_capture_count(self) -> None:
        """Fetch existing capture count from vision-trt on startup."""
        vision_url = self.app.config.vision_service_url
        host = vision_url.replace("tcp://", "").split(":")[0]
        api_url = f"http://{host}:8630/api/captures/count"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._capture_count = data.get("count", 0)
                        if self._capture_count:
                            logger.info("Restored capture count: %d", self._capture_count)
        except Exception as e:
            logger.debug("Could not restore capture count: %s", e)

    def _vision_http_base(self) -> str:
        """Return vision-trt HTTP base URL."""
        vision_url = self.app.config.vision_service_url  # tcp://127.0.0.1:8631
        host = vision_url.replace("tcp://", "").split(":")[0]
        return f"http://{host}:8630"

    async def _proxy_captures_list(self, request):
        """Proxy capture list from vision-trt."""
        from aiohttp import web, ClientSession, ClientTimeout

        url = f"{self._vision_http_base()}/api/captures/list"
        try:
            async with ClientSession(timeout=ClientTimeout(total=5)) as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"files": [], "total": 0, "error": str(e)})

    async def _proxy_captures_image(self, request):
        """Proxy a single capture image from vision-trt."""
        from aiohttp import web, ClientSession, ClientTimeout

        filename = request.match_info["filename"]
        url = f"{self._vision_http_base()}/api/captures/image/{filename}"
        try:
            async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return web.Response(status=resp.status)
                    body = await resp.read()
                    return web.Response(
                        body=body,
                        content_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"},
                    )
        except Exception as e:
            return web.Response(text=str(e), status=502)

    async def _proxy_ollama_models(self, request):
        """Proxy Ollama /api/tags to get available models.

        Query param 'url' can override the base URL (for testing remote services).
        Returns model list or fallback defaults on failure.
        """
        from aiohttp import web, ClientSession, ClientTimeout
        import json

        # Allow URL override via query param, else use config
        url_override = request.query.get("url")
        base_url = url_override or self.app.config.ollama_base_url
        tags_url = f"{base_url.rstrip('/')}/api/tags"

        # Default fallback models
        default_models = ["qwen3.5:0.8b", "qwen3.5:2b-q4_K_M", "qwen3.5:4b"]

        timeout = ClientTimeout(total=5)
        try:
            async with ClientSession(timeout=timeout) as session:
                async with session.get(tags_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = []
                        for model in data.get("models", []):
                            name = model.get("name", "")
                            if name:
                                models.append(name)
                        if not models:
                            models = default_models
                        return web.Response(
                            text=json.dumps({"models": models, "source": "ollama"}),
                            content_type="application/json",
                        )
                    else:
                        return web.Response(
                            text=json.dumps({"models": default_models, "source": "fallback"}),
                            content_type="application/json",
                        )
        except Exception as e:
            logger.debug("Ollama models fetch failed: %s", e)
            return web.Response(
                text=json.dumps({"models": default_models, "source": "fallback"}),
                content_type="application/json",
            )

    async def _handle_stream_proxy(self, request):
        """Proxy MJPEG stream from vision-trt (same-origin for browser)."""
        from aiohttp import web, ClientSession, ClientTimeout

        vision_url = self.app.config.vision_service_url  # tcp://127.0.0.1:8631
        host = vision_url.replace("tcp://", "").split(":")[0]
        stream_url = f"http://{host}:8630/stream"

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"},
        )
        await response.prepare(request)

        no_timeout = ClientTimeout(total=None, connect=10, sock_read=10)
        try:
            async with ClientSession(timeout=no_timeout) as session:
                async with session.get(stream_url) as upstream:
                    async for chunk in upstream.content.iter_any():
                        await response.write(chunk)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        except Exception as e:
            logger.debug("Stream proxy error: %s", e)
        return response

    # ── Broadcasting ──────────────────────────────────────────────────

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        if not self._ws_clients:
            return
        payload = json.dumps(msg, ensure_ascii=False)
        closed = []
        for ws in self._ws_clients:
            try:
                await asyncio.wait_for(ws.send_str(payload), timeout=2.0)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self._ws_clients.discard(ws)

    async def _broadcast_robot_state(self) -> None:
        target = await asyncio.to_thread(self.app.head_targets.get_fused_target)
        emotion = self.app.emotions._last_emotion

        # Get emotion mapping info
        from ..motion.emotion_mapper import EMOTION_MAP
        mapping_info = None
        expressions = EMOTION_MAP.get(emotion)
        if expressions:
            expr = expressions[0]
            mapping_info = {
                "name": emotion,
                "antenna_target": {
                    "left": expr.antenna.left if expr.antenna else 0,
                    "right": expr.antenna.right if expr.antenna else 0,
                },
                "head_offset": {
                    "pitch": expr.head.pitch if expr.head else 0,
                    "roll": expr.head.roll if expr.head else 0,
                },
                "description": expr.description,
            }

        # Get current antenna positions from robot if available.
        # Run in thread with timeout — this is a synchronous gRPC call that
        # can block the event loop if reachy-daemon is unresponsive.
        antenna = {"left": 0.0, "right": 0.0}
        if self.app.reachy:
            try:
                import numpy as np
                positions = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.app.reachy.get_present_antenna_joint_positions
                    ),
                    timeout=1.0,
                )
                # SDK returns [right, left] in radians
                antenna = {
                    "left": float(np.degrees(positions[1])),
                    "right": float(np.degrees(positions[0])),
                }
            except Exception:
                pass

        await self._broadcast({
            "type": "robot_state",
            "head": {
                "yaw": round(target.yaw, 1),
                "pitch": round(target.pitch, 1),
                "roll": round(target.roll, 1),
            },
            "body_yaw": round(target.body_yaw, 1),
            "antenna": antenna,
            "emotion": emotion,
            "emotion_mapping": mapping_info,
            "speaking": self.app.is_speaking,
            "tracking": {
                "source": target.source,
                "confidence": round(target.confidence, 2),
            },
            "mode": self.app.config.conversation_mode,
            "vlm_enabled": self.app.config.enable_vlm,
            "barge_in_enabled": self.app.config.barge_in_enabled,
            "capture_count": self._capture_count,
            "silero_threshold": self.app.config.silero_threshold,
            "barge_in_energy_threshold": self.app.config.barge_in_energy_threshold,
            "llm_backend": self.app.config.llm_backend,
            "ollama_model": self.app.config.ollama_model,
            "ollama_url": self.app.config.ollama_base_url,
            "gateway_host": self.app.config.gateway_host,
            "gateway_port": self.app.config.gateway_port,
        })

    # ── EventBus callbacks ────────────────────────────────────────────

    async def _on_asr_partial(self, data: dict) -> None:
        await self._broadcast({
            "type": "asr_partial",
            "text": data.get("text", ""),
            "is_final": False,
        })

    async def _on_asr_final(self, data: dict) -> None:
        await self._broadcast({
            "type": "asr_final",
            "text": data.get("text", ""),
        })

    async def _on_llm_delta(self, data: dict) -> None:
        await self._broadcast({
            "type": "llm_delta",
            "text": data.get("text", ""),
            "run_id": data.get("run_id", ""),
        })

    async def _on_llm_end(self, data: dict) -> None:
        await self._broadcast({
            "type": "llm_end",
            "full_text": data.get("full_text", ""),
            "run_id": data.get("run_id", ""),
            "emotion": self._last_llm_emotion,
        })
        self._last_llm_emotion = None

    async def _on_state_change(self, data: dict) -> None:
        await self._broadcast({
            "type": "state",
            "state": data.get("state", "idle"),
        })

    async def _on_emotion(self, data: dict) -> None:
        self._last_llm_emotion = data.get("emotion", "neutral")
        await self._broadcast({
            "type": "emotion",
            "emotion": self._last_llm_emotion,
        })

    async def _on_observation(self, data: dict) -> None:
        await self._broadcast({
            "type": "observation",
            "text": data.get("text", ""),
        })

    async def _on_vision_faces(self, data: dict) -> None:
        await self._broadcast({
            "type": "vision_faces",
            "faces": data.get("faces", []),
        })

    async def _on_smile_capture(self, data: dict) -> None:
        self._capture_count = data.get("count", self._capture_count + 1)
        await self._broadcast({
            "type": "smile_capture",
            "count": self._capture_count,
            "file": data.get("file"),
        })

    async def _restart_services(self) -> None:
        """Restart Docker containers via Docker Engine API (Unix socket).

        Order matters: vision-trt must grab /dev/video0 before reachy-daemon
        starts (otherwise the daemon takes the camera and vision-trt fails).
        After restarting vision-trt we wait until its health check passes
        before restarting the next container.
        """
        import aiohttp

        sock_path = "/var/run/docker.sock"
        # Containers where we must wait for health before moving on. vision-trt
        # is the only one with a strict startup-order requirement (camera lock).
        containers = [
            ("vision-trt", True),
            ("reachy-daemon", False),
            ("reachy-claw", False),
        ]

        # When vision-trt runs on a different host, its container is not managed
        # by the local docker socket — skip it.
        vision_url = self.app.config.vision_service_url
        vision_host = vision_url.replace("tcp://", "").split(":")[0] if vision_url else ""
        if vision_host and vision_host not in ("127.0.0.1", "localhost", "::1"):
            logger.info(
                "vision-trt is remote (%s); skipping local restart", vision_host,
            )
            containers = [(n, w) for n, w in containers if n != "vision-trt"]

        await self._broadcast({"type": "restart_status", "status": "starting"})

        conn = aiohttp.UnixConnector(path=sock_path)
        try:
            async with aiohttp.ClientSession(connector=conn) as session:
                for name, wait_healthy in containers:
                    await self._broadcast({
                        "type": "restart_status",
                        "status": "restarting",
                        "container": name,
                    })
                    try:
                        url = f"http://localhost/containers/{name}/restart?t=10"
                        async with session.post(
                            url, timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 204:
                                logger.info("Restarted container: %s", name)
                            else:
                                body = await resp.text()
                                logger.warning(
                                    "Restart %s: HTTP %d — %s", name, resp.status, body
                                )
                                continue
                    except Exception as e:
                        logger.error("Failed to restart %s: %s", name, e)
                        continue

                    if wait_healthy:
                        await self._wait_container_healthy(session, name, timeout=60)
        except Exception as e:
            logger.error("Docker socket error: %s", e)
            await self._broadcast({
                "type": "restart_status",
                "status": "error",
                "error": str(e),
            })
            return

        await self._broadcast({"type": "restart_status", "status": "done"})

    async def _wait_container_healthy(self, session, name: str, timeout: int = 60) -> bool:
        """Poll Docker container state until Health=healthy or running-without-healthcheck.

        Returns True if ready (healthy or running), False if timed out.
        """
        import asyncio
        import aiohttp

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with session.get(
                    f"http://localhost/containers/{name}/json",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        state = data.get("State", {})
                        health = state.get("Health")
                        if health:
                            if health.get("Status") == "healthy":
                                logger.info("%s is healthy", name)
                                return True
                        elif state.get("Running"):
                            # No healthcheck defined — running is good enough
                            logger.info("%s is running (no healthcheck)", name)
                            return True
            except Exception as e:
                logger.debug("Health probe for %s: %s", name, e)
            await asyncio.sleep(2)

        logger.warning("%s did not become healthy within %ds", name, timeout)
        return False

    async def _clear_captures(self) -> None:
        """Call vision-trt API to clear captures, broadcast reset."""
        vision_url = self.app.config.vision_service_url
        host = vision_url.replace("tcp://", "").split(":")[0]
        api_url = f"http://{host}:8630/api/captures"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.delete(api_url) as resp:
                    if resp.status == 200:
                        self._capture_count = 0
        except Exception as e:
            logger.warning("Failed to clear captures: %s", e)
        await self._broadcast({"type": "capture_reset", "count": 0})
