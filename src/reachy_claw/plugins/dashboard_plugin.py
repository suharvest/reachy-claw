"""DashboardPlugin — exhibition dashboard serving HTML + WebSocket state."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..plugin import Plugin
from ..settings_schema import (
    NAMESPACES,
    keys_for_namespace,
    spec_for,
    validate as validate_setting,
)
from ..config import save_runtime_overrides
from .. import ha_client

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "dashboard_static"


_DIARY_DEFAULT_PROMPT_CACHE: str | None = None


def _diary_default_prompt() -> str:
    """Read the SYSTEM_PROMPT default from scripts/generate_diary.py.

    Single source of truth: the script's hardcoded constant. We parse the
    file once and cache. If the script can't be read (missing file etc.),
    return an empty string and let the script fall back to its built-in.
    """
    global _DIARY_DEFAULT_PROMPT_CACHE
    if _DIARY_DEFAULT_PROMPT_CACHE is not None:
        return _DIARY_DEFAULT_PROMPT_CACHE
    try:
        path = Path(__file__).resolve().parents[3] / "scripts" / "generate_diary.py"
        text = path.read_text(encoding="utf-8")
        m = re.search(r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', text, re.DOTALL)
        _DIARY_DEFAULT_PROMPT_CACHE = m.group(1).strip() if m else ""
    except Exception:
        _DIARY_DEFAULT_PROMPT_CACHE = ""
    return _DIARY_DEFAULT_PROMPT_CACHE

# ── Markdown Diary Parser ─────────────────────────────────────────────────────

_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

# Map Chinese headings to front-end section IDs
_HEADING_TO_ID = {
    "今天的心情": "mood_curve",
    "遇到的人": "faces",
    "想到的事": "thoughts",
    "summary": "summary",
    "mood_curve": "mood_curve",
    "conversations": "conversations",
    "faces": "faces",
    "thoughts": "thoughts",
    "environment": "environment",
}


def _parse_diary_markdown(md: str) -> dict:
    """Parse Markdown diary with YAML front matter into JSON shape for front-end.

    Front-end expects:
        {
          "title": str,
          "date": str,
          "sections": [{"id": str, "type": str, "content": str, "heading": str}]
        }

    Sections are extracted from ## headings in the body.
    """
    m = _FRONT_MATTER_RE.match(md)
    if not m:
        return {"title": "", "date": "", "sections": []}
    front = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)

    # Split on '## ' headings
    sections = []
    current_content = []
    current_heading = ""
    for line in body.splitlines():
        if line.startswith("## "):
            # Save previous section if it has content
            if current_heading or current_content:
                content_text = "\n".join(current_content).strip()
                heading_text = current_heading[3:].strip() if current_heading.startswith("## ") else current_heading.strip()
                # Map heading to ID, default to heading text lowercased
                section_id = _HEADING_TO_ID.get(heading_text, heading_text.lower().replace(" ", "_"))
                sections.append({
                    "id": section_id,
                    "heading": heading_text,
                    "type": "narrative",
                    "content": content_text,
                })
            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    # Final section
    if current_heading or current_content:
        content_text = "\n".join(current_content).strip()
        heading_text = current_heading[3:].strip() if current_heading.startswith("## ") else current_heading.strip()
        section_id = _HEADING_TO_ID.get(heading_text, heading_text.lower().replace(" ", "_"))
        sections.append({
            "id": section_id,
            "heading": heading_text,
            "type": "narrative",
            "content": content_text,
        })

    return {
        "title": front.get("title", ""),
        "date": front.get("date", ""),
        "weather": front.get("weather"),
        "stats": front.get("stats"),
        "captures": front.get("captures", []),
        "sections": sections,
    }


def _build_settings_handlers(app):
    from aiohttp import web

    async def get_handler(request):
        ns = request.match_info["namespace"]
        if ns not in NAMESPACES:
            return web.json_response({"error": "unknown namespace"}, status=404)
        out = {}
        for k in keys_for_namespace(ns):
            spec = spec_for(f"{ns}.{k}")
            out[k] = getattr(app.config, spec.config_field)
        return web.json_response(out)

    async def put_handler(request):
        ns = request.match_info["namespace"]
        if ns not in NAMESPACES:
            return web.json_response({"error": "unknown namespace"}, status=404)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "expected JSON object"}, status=400)

        # Validate everything first (no partial updates).
        fields_to_save: list[str] = []
        for k, v in body.items():
            qkey = f"{ns}.{k}"
            try:
                validate_setting(qkey, v)
            except KeyError:
                return web.json_response(
                    {"error": f"unknown key: {qkey}"}, status=400
                )
            except ValueError as e:
                return web.json_response(
                    {"error": f"invalid value for {qkey}: {e}"}, status=400
                )
            fields_to_save.append(spec_for(qkey).config_field)

        # Cross-field validation for rest window: degenerate equal start/end
        # (other than the 24:00 sentinel) means the window is permanently closed.
        # Reject explicitly so the user can't accidentally disable rest.
        if ns == "rest":
            new_start = body.get("window_start", getattr(app.config, "rest_window_start"))
            new_end = body.get("window_end", getattr(app.config, "rest_window_end"))
            if new_end != "24:00" and new_start == new_end:
                return web.json_response(
                    {"error": "rest.window_start must differ from rest.window_end (unless end is 24:00)"},
                    status=400,
                )

        # Apply.
        for k, v in body.items():
            spec = spec_for(f"{ns}.{k}")
            setattr(app.config, spec.config_field, v)
        save_runtime_overrides(app.config, fields_to_save)
        return web.json_response({"updated": list(body.keys())})

    return {"get": get_handler, "put": put_handler}


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

        # Settings API
        settings_handlers = _build_settings_handlers(self.app)
        app.router.add_get("/api/settings/{namespace}", settings_handlers["get"])
        app.router.add_put("/api/settings/{namespace}", settings_handlers["put"])

        # Diary trigger API
        diary_handlers = _build_diary_trigger_handlers(self.app, broadcast=self._broadcast)
        app.router.add_get("/api/diary/status", diary_handlers["status"])
        app.router.add_post("/api/diary/generate", diary_handlers["generate"])
        app.router.add_post("/api/diary/publish", diary_handlers["publish"])

        # Rest control API
        rest_handlers = _build_rest_status_handlers(self.app)
        app.router.add_get("/api/rest/status", rest_handlers["status"])
        app.router.add_post("/api/rest/force", rest_handlers["force"])

        # HA API
        ha_handlers = _build_ha_handlers(self.app)
        app.router.add_post("/api/ha/test", ha_handlers["test"])
        app.router.add_get("/api/ha/entities", ha_handlers["entities"])

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
        """Return list of available diary dates from SQLite, sorted descending."""
        from aiohttp import web

        dates = self.app.db.list_diary_dates()
        return web.json_response({"dates": dates})

    async def _handle_diary_get(self, request):
        """Return diary parsed from Markdown in SQLite for a specific date."""
        from aiohttp import web

        date = request.match_info["date"]
        row = self.app.db.get_diary(date)
        if row is None:
            return web.json_response({"error": f"No diary for {date}"}, status=404)
        parsed = _parse_diary_markdown(row["markdown"])
        parsed["date"] = date
        parsed["generated_at"] = row["generated_at"]
        return web.json_response(parsed)

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
                "diary": cfg.diary_system_prompt or _diary_default_prompt(),
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
            elif mode == "diary":
                self.app.config.diary_system_prompt = prompt
                self._save_overrides(["diary_system_prompt"])
                await self._broadcast({"type": "prompt_saved", "mode": "diary"})
                return
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


def _build_diary_trigger_handlers(app, broadcast):
    """Builds /api/diary/status, /generate, /publish handlers.

    `broadcast` is an async callable that takes a dict and pushes it to all WS clients.
    """
    from aiohttp import web
    import asyncio
    import sys
    import uuid
    from datetime import datetime, timedelta
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[3]
    GEN = REPO_ROOT / "scripts" / "generate_diary.py"
    PUB = REPO_ROOT / "scripts" / "publish_diary.py"
    SCAN_DAYS = 14

    # Prevent concurrent generate/publish for the same date.
    _date_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(date: str) -> asyncio.Lock:
        if date not in _date_locks:
            _date_locks[date] = asyncio.Lock()
        return _date_locks[date]

    async def status_handler(request):
        today = datetime.now()
        out = []
        for i in range(SCAN_DAYS):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            row = app.db.get_diary(d)
            out.append({
                "date": d,
                "generated": row is not None,
                "published": bool(row and row.get("published_at")),
            })
        return web.json_response({"dates": out, "scan_window_days": SCAN_DAYS})

    async def _run_script(script: Path, date: str, force: bool, *, env_extra=None) -> tuple[int, str]:
        import os
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        args = [sys.executable, str(script), "--date", date]
        if force:
            args.append("--force")
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )
        out, err = await proc.communicate()
        return proc.returncode, (err.decode(errors="replace") or out.decode(errors="replace"))

    async def _do_generate(date: str, force: bool, job_id: str, lock: asyncio.Lock):
        try:
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "generating", "date": date})
            env_extra = {}
            if getattr(app.config, "diary_system_prompt", "").strip():
                env_extra["DIARY_SYSTEM_PROMPT_OVERRIDE"] = app.config.diary_system_prompt
            rc, msg = await _run_script(GEN, date, force, env_extra=env_extra or None)
            if rc != 0:
                await broadcast({"type": "diary_job", "job_id": job_id, "phase": "error", "date": date, "error": msg})
                return
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "done", "date": date})
        finally:
            lock.release()

    async def _do_publish(date: str, force: bool, job_id: str, lock: asyncio.Lock):
        try:
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "publishing", "date": date})
            env_extra = {
                "SITE_REPO_URL": app.config.diary_site_repo_url,
                "SITE_DIARY_PATH": app.config.diary_site_diary_path,
                "SITE_BRANCH": app.config.diary_site_branch,
            }
            rc, msg = await _run_script(PUB, date, force, env_extra=env_extra)
            if rc != 0:
                await broadcast({"type": "diary_job", "job_id": job_id, "phase": "error", "date": date, "error": msg})
                return
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "done", "date": date})
        finally:
            lock.release()

    async def generate_handler(request):
        body = await request.json()
        date = body.get("date")
        force = bool(body.get("force"))
        if not date:
            return web.json_response({"error": "date required"}, status=400)
        if not force and app.db.get_diary(date) is not None:
            return web.json_response({"date": date, "status": "already-generated"})
        lock = _lock_for(date)
        if lock.locked():
            # Per-date concurrency guard: refuse a second generate while one is in flight.
            return web.json_response(
                {"date": date, "status": "in-progress"}, status=409
            )
        await lock.acquire()
        job_id = uuid.uuid4().hex[:12]
        asyncio.create_task(_do_generate(date, force, job_id, lock))
        return web.json_response({"job_id": job_id, "date": date}, status=202)

    async def publish_handler(request):
        body = await request.json()
        date = body.get("date")
        force = bool(body.get("force"))
        if not date:
            return web.json_response({"error": "date required"}, status=400)
        row = app.db.get_diary(date)
        if row is None:
            return web.json_response({"error": "diary not generated"}, status=404)
        if not force and row.get("published_at"):
            return web.json_response({"date": date, "status": "already-published"})
        lock = _lock_for(date)
        if lock.locked():
            return web.json_response(
                {"date": date, "status": "in-progress"}, status=409
            )
        await lock.acquire()
        job_id = uuid.uuid4().hex[:12]
        asyncio.create_task(_do_publish(date, force, job_id, lock))
        return web.json_response({"job_id": job_id, "date": date}, status=202)

    return {"status": status_handler, "generate": generate_handler, "publish": publish_handler}


def _build_rest_status_handlers(app):
    """GET /api/rest/status, POST /api/rest/force ({"action":"enter"|"exit"|"clear"})."""
    from aiohttp import web

    def _rest_plugin():
        if hasattr(app, "get_plugin"):
            return app.get_plugin("rest")
        for p in getattr(app, "_plugins", getattr(app, "plugins", [])):
            if getattr(p, "name", "") == "rest":
                return p
        return None

    async def status_handler(request):
        rp = _rest_plugin()
        if rp is None:
            return web.json_response({"error": "rest plugin not registered"}, status=503)
        # Introspect each plugin's _paused flag so the user can verify the
        # rest signal actually propagated to the relevant plugins.
        plugin_states = {}
        for p in getattr(app, "_plugins", []):
            if hasattr(p, "_paused"):
                plugin_states[p.name] = bool(p._paused)
        return web.json_response({
            "resting": rp.is_resting(),
            "started_at": rp._started_at,
            "force_state": rp._force_state,  # True / False / None
            "enabled": getattr(app.config, "rest_enabled", True),
            "window_start": getattr(app.config, "rest_window_start", "23:00"),
            "window_end": getattr(app.config, "rest_window_end", "24:00"),
            "timezone": getattr(app.config, "rest_timezone", "Asia/Shanghai"),
            "plugin_paused": plugin_states,
        })

    async def force_handler(request):
        rp = _rest_plugin()
        if rp is None:
            return web.json_response({"error": "rest plugin not registered"}, status=503)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        action = body.get("action")
        if action == "enter":
            await rp.force_rest()
        elif action == "exit":
            await rp.force_resume()
        elif action == "clear":
            rp.force_clear()
        else:
            return web.json_response(
                {"error": "action must be one of: enter, exit, clear"}, status=400
            )
        return web.json_response({"resting": rp.is_resting(), "force_state": rp._force_state})

    return {"status": status_handler, "force": force_handler}


def _build_ha_handlers(app):
    """HA Sensors API: connection probe + entity listing.

    Settings PUT/GET for ha.* is handled by the generic
    _build_settings_handlers via the namespace dispatch.
    """
    from aiohttp import web

    async def test_handler(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        url = body.get("url") if isinstance(body, dict) else None
        token = body.get("token") if isinstance(body, dict) else None
        url = url or app.config.ha_url
        token = token or app.config.ha_token
        if not url or not token:
            return web.json_response(
                {"ok": False, "status": 0, "message": "ha_url or ha_token not set"}
            )
        result = await ha_client.probe(url, token)
        return web.json_response(result)

    async def entities_handler(request):
        url = app.config.ha_url
        token = app.config.ha_token
        if not url or not token:
            return web.json_response(
                {"error": "ha_url or ha_token not configured"}, status=400
            )
        try:
            states = await ha_client.list_states(url, token)
        except ha_client.HAUnauthorized as e:
            return web.json_response({"error": f"unauthorized: {e}"}, status=502)
        except ha_client.HAUnreachable as e:
            return web.json_response({"error": f"unreachable: {e}"}, status=502)
        except ha_client.HABadResponse as e:
            return web.json_response({"error": f"bad response: {e}"}, status=502)

        groups: dict[str, list[dict]] = {}
        for s in states:
            eid = s.get("entity_id", "")
            if "." not in eid:
                continue
            domain = eid.split(".", 1)[0]
            attrs = s.get("attributes") or {}
            groups.setdefault(domain, []).append({
                "entity_id": eid,
                "state": s.get("state", ""),
                "friendly_name": attrs.get("friendly_name", ""),
            })
        out = []
        for domain in sorted(groups):
            entities = sorted(groups[domain], key=lambda e: e["entity_id"])
            out.append({"domain": domain, "count": len(entities), "entities": entities})
        return web.json_response({"groups": out})

    return {"test": test_handler, "entities": entities_handler}
