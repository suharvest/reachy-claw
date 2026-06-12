"""Minimal HTTP health endpoint for container orchestration.

Serves GET /health on a configurable port (default 8640).
Returns 200 when the app is healthy, 503 during startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import ReachyClawApp

logger = logging.getLogger(__name__)


async def start_health_server(app: ReachyClawApp, port: int = 8640) -> None:
    """Run a tiny HTTP server that reports app health."""

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Read the request line (we only care about GET /health)
            data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            request_line = data.split(b"\r\n", 1)[0].decode(errors="replace")

            if "GET /health" in request_line:
                if app.healthy:
                    body = json.dumps({
                        "status": "ok",
                        "robot_connected": app.reachy is not None,
                        "plugins": [p.name for p in app._plugins],
                    })
                    status = "200 OK"
                else:
                    body = json.dumps({"status": "starting"})
                    status = "503 Service Unavailable"
            else:
                body = "Not Found"
                status = "404 Not Found"

            response = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle_client, "0.0.0.0", port)
    logger.info("Health server listening on :%d", port)
    async with server:
        await server.serve_forever()
