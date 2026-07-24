"""Explicit camera snapshot analysis for "what do you see?" requests."""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from reachy_voice.config import Config
from reachy_voice import speech_runtime, tier_a

logger = logging.getLogger("reachy_voice.vision_analysis")

DEFAULT_ONLINE_VISION_MODEL = "qwen3.5-omni-flash"
DEFAULT_ONLINE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VISION_PROMPT_ZH = "请用一句简短中文描述机器人摄像头现在看到的主要内容。不要编造看不到的细节。"
VISION_PROMPT_EN = "Briefly describe in one sentence what the robot camera sees. Do not invent details."


class VisionAnalysisError(RuntimeError):
    """Raised when snapshot capture or VLM analysis fails."""


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _prompt(language: str) -> str:
    return VISION_PROMPT_EN if language == "en" else VISION_PROMPT_ZH


async def fetch_snapshot_jpeg(
    cfg: Config, *, client: httpx.AsyncClient | None = None
) -> bytes:
    """Fetch one JPEG frame from vision-trt's snapshot endpoint."""
    url = f"{tier_a.vision_http_base(cfg.vision_mjpeg)}/api/snapshot?target=stream"
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=float(cfg.vlm_timeout_s))
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise VisionAnalysisError(f"snapshot HTTP {resp.status_code}")
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type.lower() and not resp.content.startswith(b"\xff\xd8"):
            raise VisionAnalysisError("snapshot response is not an image")
        return resp.content
    except VisionAnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VisionAnalysisError(f"snapshot failed: {exc}") from exc
    finally:
        if owns:
            await client.aclose()


def _online_provider() -> tuple[str, str, str | None]:
    settings = speech_runtime.read_settings()
    model = str(settings.get("qwen_vision_model") or DEFAULT_ONLINE_VISION_MODEL)
    api_key = None
    key_path = speech_runtime.dashscope_key_path()
    if key_path.exists():
        api_key = key_path.read_text(encoding="utf-8").strip()
    base_url = str(settings.get("qwen_compat_url") or DEFAULT_ONLINE_COMPAT_URL)
    return base_url, model, api_key


def _local_provider(cfg: Config) -> tuple[str, str, str | None]:
    return cfg.vlm_base_url or cfg.edge_llm_url, cfg.vlm_model or cfg.edge_llm_model, None


async def analyze_snapshot(
    cfg: Config,
    image_jpeg: bytes,
    *,
    mode: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send a JPEG snapshot to the active local/online vision model."""
    settings = speech_runtime.read_settings()
    selected = (mode or settings.get("mode") or "local").strip().lower()
    if selected == "online":
        base_url, model, api_key = _online_provider()
        if not api_key:
            raise VisionAnalysisError("online vision mode requires DashScope API key")
    else:
        base_url, model, api_key = _local_provider(cfg)

    image_b64 = base64.b64encode(image_jpeg).decode("ascii")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(cfg.language)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 120,
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=float(cfg.vlm_timeout_s))
    try:
        resp = await client.post(_chat_url(base_url), json=payload, headers=headers)
        if resp.status_code >= 400:
            raise VisionAnalysisError(f"VLM HTTP {resp.status_code}: {resp.text[:160]}")
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if isinstance(text, list):
            text = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in text
            )
        text = str(text).strip()
        if not text:
            raise VisionAnalysisError("VLM returned empty response")
        return text
    except VisionAnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VisionAnalysisError(f"VLM failed: {exc}") from exc
    finally:
        if owns:
            await client.aclose()


async def describe_current_view(cfg: Config) -> str:
    """Capture one frame and return a visitor-facing visual description."""
    image = await fetch_snapshot_jpeg(cfg)
    return await analyze_snapshot(cfg, image)
