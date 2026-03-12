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
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/stream", self._handle_stream_proxy)
        app.router.add_static("/static", STATIC_DIR, show_index=False)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        port = self.app.config.dashboard_port
        self._site = web.TCPSite(self._runner, "0.0.0.0", port)
        await self._site.start()
        logger.info("Dashboard listening on http://0.0.0.0:%d", port)

        # Subscribe to EventBus
        bus = self.app.events
        bus.subscribe("asr_partial", self._on_asr_partial)
        bus.subscribe("asr_final", self._on_asr_final)
        bus.subscribe("llm_delta", self._on_llm_delta)
        bus.subscribe("llm_end", self._on_llm_end)
        bus.subscribe("state_change", self._on_state_change)
        bus.subscribe("emotion", self._on_emotion)
        bus.subscribe("observation", self._on_observation)

        # State polling loop (5Hz)
        while self._running:
            await self._broadcast_robot_state()
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

        # Close all WebSocket connections
        for ws in list(self._ws_clients):
            await ws.close()
        self._ws_clients.clear()

        if self._runner:
            await self._runner.cleanup()

    # ── HTTP handlers ─────────────────────────────────────────────────

    async def _handle_index(self, request):
        from aiohttp import web

        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return web.Response(text="Dashboard HTML not found", status=404)
        return web.FileResponse(index_path)

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
        if msg_type == "set_mode":
            mode = data.get("mode", "conversation")
            if mode not in ("conversation", "monologue"):
                return
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "switch_mode"):
                conv.switch_mode(mode)
            await self._broadcast({"type": "mode_changed", "mode": mode})

        elif msg_type == "get_prompts":
            from ..llm import DEFAULT_SYSTEM_PROMPT, MONOLOGUE_SYSTEM_PROMPT

            cfg = self.app.config
            await self._broadcast({
                "type": "prompts",
                "conversation": cfg.ollama_system_prompt or DEFAULT_SYSTEM_PROMPT,
                "monologue": cfg.ollama_monologue_prompt or MONOLOGUE_SYSTEM_PROMPT,
            })

        elif msg_type == "set_prompt":
            from ..llm import DEFAULT_SYSTEM_PROMPT, MONOLOGUE_SYSTEM_PROMPT, OllamaClient

            mode = data.get("mode")
            prompt = data.get("prompt", "").strip()
            if mode == "conversation":
                self.app.config.ollama_system_prompt = prompt
            elif mode == "monologue":
                self.app.config.ollama_monologue_prompt = prompt
            else:
                return

            # Hot-apply: if OllamaClient is active in matching mode, update live
            conv = self.app.get_plugin("conversation")
            if conv and hasattr(conv, "_client") and isinstance(conv._client, OllamaClient):
                is_monologue = getattr(conv, "_monologue_mode", False)
                if (mode == "monologue" and is_monologue) or (mode == "conversation" and not is_monologue):
                    conv._client._config.system_prompt = prompt or (
                        MONOLOGUE_SYSTEM_PROMPT if is_monologue else DEFAULT_SYSTEM_PROMPT
                    )

            await self._broadcast({"type": "prompt_saved", "mode": mode})

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

        no_timeout = ClientTimeout(total=None, connect=10, sock_read=None)
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
                await ws.send_str(payload)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self._ws_clients.discard(ws)

    async def _broadcast_robot_state(self) -> None:
        target = self.app.head_targets.get_fused_target()
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

        # Get current antenna positions from robot if available
        antenna = {"left": 0.0, "right": 0.0}
        if self.app.reachy:
            try:
                import numpy as np
                positions = self.app.reachy.get_present_antenna_joint_positions()
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
        })

    async def _on_state_change(self, data: dict) -> None:
        await self._broadcast({
            "type": "state",
            "state": data.get("state", "idle"),
        })

    async def _on_emotion(self, data: dict) -> None:
        await self._broadcast({
            "type": "emotion",
            "emotion": data.get("emotion", "neutral"),
        })

    async def _on_observation(self, data: dict) -> None:
        await self._broadcast({
            "type": "observation",
            "text": data.get("text", ""),
        })
