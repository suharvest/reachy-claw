"""Tier-A dashboard APIs — pure-port from the legacy reachy-claw dashboard
plugin (``legacy/reachy_claw/plugins/dashboard_plugin.py``).

These are the self-contained dashboard endpoints the shared frontend
(``static/app.js``) calls: an Ollama-models proxy, a Docker-socket service
restarter, and the vision-trt capture proxies. NONE of them touch the
reachy-mini SDK, the conversation core, or any deferred feature — they are
HTTP reverse-proxy / Docker Engine socket calls only.

The logic lives here (decoupled from FastAPI / the websocket plumbing) so it
can be unit-tested with a mocked ``httpx`` transport and no hardware. ``main.py``
wires these into the settings app's HTTP routes and ``/ws`` message loop.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("reachy_voice.tier_a")

# vision-trt serves its HTTP API (captures + MJPEG) on this port; the ZMQ PUB
# (faces) is a different port. The dashboard talks to the HTTP side.
VISION_HTTP_PORT = 8630
DOCKER_SOCK = "/var/run/docker.sock"

# Ollama models the UI falls back to when the service is unreachable.
DEFAULT_OLLAMA_MODELS = ["qwen3.5:0.8b", "qwen3.5:2b-q4_K_M", "qwen3.5:4b"]

# Docker restart order. vision-trt must grab /dev/video0 before reachy-daemon
# starts, so it goes first and we wait for its healthcheck before advancing.
# (name, wait_healthy). reachy-VOICE is the app container here, NOT reachy-claw.
DEFAULT_RESTART_CONTAINERS: list[tuple[str, bool]] = [
    ("vision-trt", True),
    ("reachy-daemon", False),
    ("reachy-voice", False),
]

# Hosts that ARE the local docker host (so the container IS managed by the
# local socket). Anything else means vision-trt is remote → skip its restart.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", ""}

Broadcast = Callable[[dict], Awaitable[None]]


# ── host helpers ─────────────────────────────────────────────────────────
def vision_host(vision_mjpeg: str) -> str:
    """Extract the host from the configured ``vision_mjpeg`` URL.

    The legacy plugin derived this from ``vision_service_url`` (``tcp://...``);
    reachy_voice's config exposes ``vision_mjpeg`` (``http://host:8630/stream``)
    instead, so parse the host out of that.
    """
    host = urlsplit(vision_mjpeg).hostname
    return host or "127.0.0.1"


def vision_http_base(vision_mjpeg: str) -> str:
    """vision-trt HTTP API base, e.g. ``http://127.0.0.1:8630``."""
    return f"http://{vision_host(vision_mjpeg)}:{VISION_HTTP_PORT}"


# ── 1. Ollama models proxy ───────────────────────────────────────────────
async def fetch_ollama_models(
    base_url: str, *, client: httpx.AsyncClient | None = None
) -> dict:
    """Proxy ``{base_url}/api/tags`` → ``{"models": [...], "source": ...}``.

    Returns the model-name list on success, else a small default list. Never
    raises — the dashboard treats any failure as "use defaults".
    """
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=5.0)
    try:
        resp = await client.get(tags_url)
        if resp.status_code == 200:
            data = resp.json()
            models = [
                m["name"] for m in data.get("models", []) if m.get("name")
            ]
            if not models:
                models = list(DEFAULT_OLLAMA_MODELS)
            return {"models": models, "source": "ollama"}
        return {"models": list(DEFAULT_OLLAMA_MODELS), "source": "fallback"}
    except Exception as e:  # noqa: BLE001 — any error → defaults
        logger.debug("Ollama models fetch failed: %s", e)
        return {"models": list(DEFAULT_OLLAMA_MODELS), "source": "fallback"}
    finally:
        if owns:
            await client.aclose()


# ── 2. Docker service restart ────────────────────────────────────────────
def _resolve_containers(vision_mjpeg: str) -> list[tuple[str, bool]]:
    """Restart order, with vision-trt dropped when it runs on a remote host
    (its container isn't managed by the LOCAL docker socket)."""
    containers = list(DEFAULT_RESTART_CONTAINERS)
    host = vision_host(vision_mjpeg)
    if host not in _LOCAL_HOSTS:
        logger.info("vision-trt is remote (%s); skipping local restart", host)
        containers = [(n, w) for n, w in containers if n != "vision-trt"]
    return containers


async def _wait_container_healthy(
    client: httpx.AsyncClient, name: str, timeout: float, *, sleep, now
) -> bool:
    """Poll ``/containers/{name}/json`` until Health=healthy, or (no
    healthcheck) Running. Returns False on timeout. ``sleep``/``now`` are
    injected so tests don't wall-clock."""
    deadline = now() + timeout
    while now() < deadline:
        try:
            resp = await client.get(
                f"http://localhost/containers/{name}/json", timeout=5.0
            )
            if resp.status_code == 200:
                state = resp.json().get("State", {})
                health = state.get("Health")
                if health:
                    if health.get("Status") == "healthy":
                        logger.info("%s is healthy", name)
                        return True
                elif state.get("Running"):
                    logger.info("%s is running (no healthcheck)", name)
                    return True
        except Exception as e:  # noqa: BLE001
            logger.debug("Health probe for %s: %s", name, e)
        await sleep(2)
    logger.warning("%s did not become healthy within %ss", name, timeout)
    return False


async def restart_services(
    vision_mjpeg: str,
    broadcast: Broadcast,
    *,
    client: httpx.AsyncClient | None = None,
    sleep=None,
    now=None,
    health_timeout: float = 60.0,
) -> None:
    """Restart the docker containers in order via the Docker Engine UNIX
    socket, broadcasting ``restart_status`` messages the UI consumes:

      {"status": "starting"}
      {"status": "restarting", "container": <name>}   (per container)
      {"status": "done"}  | {"status": "error", "error": <str>}

    A container flagged ``wait_healthy`` is polled to health before the next
    one restarts (vision-trt must own /dev/video0 before reachy-daemon).
    """
    import asyncio

    if sleep is None:
        sleep = asyncio.sleep
    if now is None:
        loop = asyncio.get_event_loop()
        now = loop.time

    containers = _resolve_containers(vision_mjpeg)
    logger.info("Dashboard restart container list: %s", containers)

    await broadcast({"type": "restart_status", "status": "starting"})

    owns = client is None
    if client is None:
        client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=DOCKER_SOCK), timeout=30.0
        )
    try:
        for name, wait_healthy in containers:
            await broadcast(
                {"type": "restart_status", "status": "restarting", "container": name}
            )
            try:
                resp = await client.post(
                    f"http://localhost/containers/{name}/restart?t=10", timeout=30.0
                )
                if resp.status_code == 204:
                    logger.info("Restarted container: %s", name)
                else:
                    logger.warning(
                        "Restart %s: HTTP %d — %s", name, resp.status_code, resp.text
                    )
                    continue
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to restart %s: %s", name, e)
                continue

            if wait_healthy:
                await _wait_container_healthy(
                    client, name, health_timeout, sleep=sleep, now=now
                )
    except Exception as e:  # noqa: BLE001 — socket-level failure
        logger.error("Docker socket error: %s", e)
        await broadcast({"type": "restart_status", "status": "error", "error": str(e)})
        return
    finally:
        if owns:
            await client.aclose()

    await broadcast({"type": "restart_status", "status": "done"})


# ── 3. Captures (vision-trt :8630 proxy) ─────────────────────────────────
async def capture_info(
    vision_mjpeg: str, *, client: httpx.AsyncClient | None = None
) -> dict:
    """GET vision-trt ``/api/captures/count`` → a ``capture_info`` message the
    UI consumes: ``{"type": "capture_info", "path": <str>, "count": <int>}``.

    ``path`` is the host-visible captures dir (display only); the count comes
    from vision-trt and falls back to 0 on any error.
    """
    import os

    data_dir = os.environ.get("HOST_DATA_DIR") or os.environ.get(
        "DATA_DIR", "~/reachy-data"
    )
    host_path = os.path.join(data_dir, "vision", "captures")

    count = 0
    url = f"{vision_http_base(vision_mjpeg)}/api/captures/count"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=5.0)
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            count = resp.json().get("count", 0)
    except Exception as e:  # noqa: BLE001
        logger.debug("capture count fetch failed: %s", e)
    finally:
        if owns:
            await client.aclose()

    return {"type": "capture_info", "path": host_path, "count": count}


async def clear_captures(
    vision_mjpeg: str, *, client: httpx.AsyncClient | None = None
) -> dict:
    """DELETE vision-trt ``/api/captures`` → a ``capture_reset`` message:
    ``{"type": "capture_reset", "count": 0}``. Never raises."""
    url = f"{vision_http_base(vision_mjpeg)}/api/captures"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=5.0)
    try:
        await client.delete(url)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to clear captures: %s", e)
    finally:
        if owns:
            await client.aclose()
    return {"type": "capture_reset", "count": 0}


async def captures_list(
    vision_mjpeg: str, *, client: httpx.AsyncClient | None = None
) -> dict:
    """Reverse-proxy vision-trt ``/api/captures/list`` JSON. On error returns
    ``{"files": [], "total": 0, "error": <str>}`` (the UI guards both)."""
    url = f"{vision_http_base(vision_mjpeg)}/api/captures/list"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=5.0)
    try:
        resp = await client.get(url)
        return resp.json()
    except Exception as e:  # noqa: BLE001
        return {"files": [], "total": 0, "error": str(e)}
    finally:
        if owns:
            await client.aclose()


async def captures_image(
    vision_mjpeg: str, filename: str, *, client: httpx.AsyncClient | None = None
) -> tuple[int, bytes, dict]:
    """Reverse-proxy a single capture image from vision-trt
    ``/api/captures/image/{filename}``.

    Returns ``(status_code, body, headers)`` so the caller builds the HTTP
    response. A 200 carries the JPEG bytes with a long cache header; an
    upstream non-200 is forwarded with an empty body; a connection error
    becomes 502 with the error text.
    """
    url = f"{vision_http_base(vision_mjpeg)}/api/captures/image/{filename}"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            return resp.status_code, b"", {}
        return (
            200,
            resp.content,
            {
                "Content-Type": "image/jpeg",
                "Cache-Control": "public, max-age=86400",
            },
        )
    except Exception as e:  # noqa: BLE001
        return 502, str(e).encode(), {"Content-Type": "text/plain"}
    finally:
        if owns:
            await client.aclose()


# ── FastAPI route registration ───────────────────────────────────────────
def register_http_routes(settings_app, config_getter: Callable[[], object]) -> None:
    """Register the Tier-A HTTP GET routes on a FastAPI app.

    ``config_getter`` returns the live config (so routes always read current
    ``edge_llm_url`` / ``vision_mjpeg``). Factored out of ``main.py`` so the
    exact route bodies are exercised in tests without the reachy-mini SDK.
    """
    from fastapi.responses import Response

    @settings_app.get("/api/ollama/models")
    async def ollama_models(url: str = "") -> dict:
        # ?url= overrides the LLM base (the UI passes the current Ollama URL,
        # without the trailing /v1). Fall back to the configured edge_llm_url.
        cfg = config_getter()
        base = url or getattr(cfg, "edge_llm_url", "")
        return await fetch_ollama_models(base)

    @settings_app.get("/api/captures/list")
    async def captures_list_route() -> dict:
        return await captures_list(config_getter().vision_mjpeg)

    @settings_app.get("/api/captures/image/{filename}")
    async def captures_image_route(filename: str):  # noqa: ANN202
        status, body, headers = await captures_image(
            config_getter().vision_mjpeg, filename
        )
        return Response(content=body, status_code=status, headers=headers)
