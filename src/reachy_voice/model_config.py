"""Provider mode configuration shared by the unified dashboard adapters."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def config_dir() -> Path:
    return Path(os.environ.get("REACHY_CONFIG_DIR", str(Path.home() / ".config/reachy")))


def config_file() -> Path:
    return config_dir() / "voice_gateway.json"


def key_file() -> Path:
    return config_dir() / "dashscope_api_key"


def default_config() -> dict[str, Any]:
    key_path = str(key_file())
    return {
        "mode": "local",
        "local": {
            "asr": "paraformer-streaming",
            "tts": "matcha",
            "llm": "qwen3-1.7b-gguf",
            "enabled": True,
        },
        "online": {
            "provider": "qwen",
            "endpoint": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
            "compat_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "model": "qwen3.5-omni-flash-realtime",
            "vision_model": "qwen3.5-omni-flash",
            "voice": "Serena",
            "api_key_file": key_path,
        },
        "reachy": {
            "wake_word": "你好mini",
            "idle_timeout_seconds": 18,
            "max_session_seconds": 300,
            "speech_motion": True,
            "vision_enabled": True,
        },
        "updated_at": None,
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config() -> dict[str, Any]:
    path = config_file()
    if not path.exists():
        return default_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return deep_merge(default_config(), data)
    except Exception:
        pass
    return default_config()


def masked_key() -> str:
    path = key_file()
    if not path.exists():
        return ""
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}***{key[-4:]}"


def redacted_config() -> dict[str, Any]:
    cfg = load_config()
    cfg.setdefault("online", {})["api_key"] = masked_key()
    cfg.setdefault("online", {})["api_key_file"] = str(key_file())
    return cfg


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = deep_merge(load_config(), payload)
    api_key = str(payload.get("api_key", "") or "").strip()
    config_dir().mkdir(parents=True, exist_ok=True)
    if api_key:
        key_file().write_text(api_key, encoding="utf-8")
        key_file().chmod(0o600)
    cfg.setdefault("online", {})["api_key_file"] = str(key_file())
    cfg["updated_at"] = int(time.time())
    config_file().write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    config_file().chmod(0o600)
    return redacted_config()


def set_mode(mode: str) -> dict[str, Any]:
    if mode not in {"local", "online"}:
        raise ValueError("mode must be local or online")
    return save_config({"mode": mode})
