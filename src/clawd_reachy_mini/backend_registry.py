"""Auto-discovery registry for STT/TTS backends.

Adding a new backend requires only:
  1. Create a class inheriting TTSBackend / STTBackend
  2. Decorate it with @register_tts("name") / @register_stt("name")
  3. (Optional) Add a Settings inner class for backend-specific config

The registry auto-generates config fields, YAML mappings, env var
mappings, and CLI choices — no manual wiring needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Metadata about a registered backend."""

    name: str
    cls: type
    kind: str  # "tts" or "stt"
    settings_fields: dict[str, Any]  # field_name → default_value


# Global registries
_tts_registry: dict[str, BackendInfo] = {}
_stt_registry: dict[str, BackendInfo] = {}


def _extract_settings(cls: type) -> dict[str, Any]:
    """Extract settings fields from a backend's Settings inner class."""
    settings_cls = getattr(cls, "Settings", None)
    if settings_cls is None:
        return {}

    result = {}
    # Support both dataclass and plain class with annotations
    if hasattr(settings_cls, "__dataclass_fields__"):
        for f in fields(settings_cls):
            result[f.name] = f.default if f.default is not f.default_factory else f.default_factory()
    else:
        annotations = getattr(settings_cls, "__annotations__", {})
        for name, _type in annotations.items():
            default = getattr(settings_cls, name, None)
            result[name] = default
    return result


def register_tts(name: str):
    """Decorator to register a TTS backend."""
    def decorator(cls):
        _tts_registry[name] = BackendInfo(
            name=name,
            cls=cls,
            kind="tts",
            settings_fields=_extract_settings(cls),
        )
        cls._backend_name = name
        return cls
    return decorator


def register_stt(name: str):
    """Decorator to register an STT backend."""
    def decorator(cls):
        _stt_registry[name] = BackendInfo(
            name=name,
            cls=cls,
            kind="stt",
            settings_fields=_extract_settings(cls),
        )
        cls._backend_name = name
        return cls
    return decorator


def get_tts_names() -> list[str]:
    """Return all registered TTS backend names."""
    _ensure_loaded()
    return list(_tts_registry.keys())


def get_stt_names() -> list[str]:
    """Return all registered STT backend names."""
    _ensure_loaded()
    return list(_stt_registry.keys())


def get_tts_info(name: str) -> BackendInfo | None:
    _ensure_loaded()
    return _tts_registry.get(name)


def get_stt_info(name: str) -> BackendInfo | None:
    _ensure_loaded()
    return _stt_registry.get(name)


def get_all_backend_settings() -> dict[str, dict[str, Any]]:
    """Return all backend settings: {kind_name_field: default}.

    Used by config.py to auto-generate Config fields.
    """
    _ensure_loaded()
    result = {}
    for registry in (_tts_registry, _stt_registry):
        for info in registry.values():
            for field_name, default in info.settings_fields.items():
                # Prefix with backend name to avoid collisions
                config_key = f"{info.name}_{field_name}"
                result[config_key] = default
    return result


def get_yaml_mappings() -> dict[tuple[str, str], str]:
    """Auto-generate YAML section/key → config field mappings for backends."""
    _ensure_loaded()
    result = {}
    for registry, section in [(_tts_registry, "tts"), (_stt_registry, "stt")]:
        for info in registry.values():
            for field_name in info.settings_fields:
                config_key = f"{info.name}_{field_name}"
                yaml_key = f"{info.name}_{field_name}"
                result[(section, yaml_key)] = config_key
    return result


def get_env_mappings() -> dict[str, str]:
    """Auto-generate ENV_VAR → config field mappings for backends."""
    _ensure_loaded()
    result = {}
    for registry in (_tts_registry, _stt_registry):
        for info in registry.values():
            for field_name in info.settings_fields:
                config_key = f"{info.name}_{field_name}"
                env_var = f"{info.name}_{field_name}".upper()
                result[env_var] = config_key
    return result


# ── Lazy loading ─────────────────────────────────────────────────────

_loaded = False


def _ensure_loaded():
    """Import backend modules so decorators run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Import modules that contain @register_tts / @register_stt decorators
    import clawd_reachy_mini.stt  # noqa: F401
    import clawd_reachy_mini.tts  # noqa: F401
