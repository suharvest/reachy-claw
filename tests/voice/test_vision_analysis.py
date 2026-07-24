from __future__ import annotations

import json

import httpx
import pytest

from reachy_voice.config import Config
from reachy_voice import speech_runtime
from reachy_voice.vision_analysis import (
    VisionAnalysisError,
    analyze_snapshot,
    fetch_snapshot_jpeg,
)


@pytest.mark.asyncio
async def test_fetch_snapshot_jpeg_from_vision_service():
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"\xff\xd8jpg", headers={"content-type": "image/jpeg"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = Config(vision_mjpeg="http://192.168.50.189:8630/stream")

    image = await fetch_snapshot_jpeg(cfg, client=client)

    assert image == b"\xff\xd8jpg"
    assert seen == ["http://192.168.50.189:8630/api/snapshot?target=stream"]


@pytest.mark.asyncio
async def test_analyze_snapshot_local_posts_openai_compatible_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_VOICE_DATA_DIR", str(tmp_path))
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "我看到一个人。"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = Config(vlm_base_url="http://localhost:11435/v1", vlm_model="local-vlm")

    text = await analyze_snapshot(cfg, b"\xff\xd8jpg", mode="local", client=client)

    assert text == "我看到一个人。"
    assert seen["url"] == "http://localhost:11435/v1/chat/completions"
    assert seen["payload"]["model"] == "local-vlm"
    content = seen["payload"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_analyze_snapshot_online_requires_key(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_VOICE_DATA_DIR", str(tmp_path))
    speech_runtime.save_settings({"mode": "local"})

    with pytest.raises(VisionAnalysisError, match="DashScope API key"):
        await analyze_snapshot(Config(), b"\xff\xd8jpg", mode="online")
