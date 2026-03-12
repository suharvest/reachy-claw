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
                pass  # Client → server messages not used currently
        finally:
            self._ws_clients.discard(ws)
            logger.info("Dashboard WS client disconnected (%d remaining)", len(self._ws_clients))

        return ws

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
                "roll": 0.0,
            },
            "antenna": antenna,
            "emotion": emotion,
            "emotion_mapping": mapping_info,
            "speaking": self.app.is_speaking,
            "tracking": {
                "source": target.source,
                "confidence": round(target.confidence, 2),
            },
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
