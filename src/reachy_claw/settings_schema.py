"""Registry of dashboard-tunable settings.

Single source of truth for:
  - which settings can be changed from the dashboard
  - what types they have
  - how to validate incoming values
  - how dashboard namespace+key maps to flat Config field name

Used by the settings API (dashboard_plugin) to enumerate, validate,
and apply incoming PUT requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class SettingSpec:
    namespace: str          # "rest" or "diary"
    key: str                # e.g. "window_start"
    config_field: str       # flat field name in Config dataclass
    type_: type             # bool / str
    extra_validate: Callable[[Any], None] | None = None


_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")  # 00:00..23:59


def _validate_hhmm(v: Any) -> None:
    """Accept 00:00..23:59 plus the special end-of-day sentinel "24:00"."""
    if not isinstance(v, str):
        raise ValueError(f"expected HH:MM (24h), got {type(v).__name__}")
    if v == "24:00":
        return
    if not _HHMM.match(v):
        raise ValueError(f"expected HH:MM (00:00..23:59 or 24:00), got {v!r}")


def _validate_tz(v: Any) -> None:
    if not isinstance(v, str):
        raise ValueError(f"timezone must be a string, got {type(v).__name__}")
    try:
        ZoneInfo(v)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"unknown IANA timezone: {v!r}") from e


_HA_URL = re.compile(r"^https?://")
_ENTITY_ID = re.compile(r"^[a-z_]+\.[a-zA-Z0-9_]+$")


def _validate_ha_url(v: Any) -> None:
    if not isinstance(v, str):
        raise ValueError(f"ha.url must be a string, got {type(v).__name__}")
    if v == "":
        return
    if not _HA_URL.match(v):
        raise ValueError(f"ha.url must start with http:// or https://, got {v!r}")


def _validate_str_list(v: Any) -> None:
    if not isinstance(v, list):
        raise ValueError(f"expected list, got {type(v).__name__}")
    for i, item in enumerate(v):
        if not isinstance(item, str):
            raise ValueError(f"item {i} must be str, got {type(item).__name__}")


def _validate_entity_id_list(v: Any) -> None:
    _validate_str_list(v)
    for i, item in enumerate(v):
        if not _ENTITY_ID.match(item):
            raise ValueError(f"item {i}: invalid HA entity_id {item!r}")


_SPECS: list[SettingSpec] = [
    SettingSpec("rest", "enabled", "rest_enabled", bool),
    SettingSpec("rest", "window_start", "rest_window_start", str, _validate_hhmm),
    SettingSpec("rest", "window_end", "rest_window_end", str, _validate_hhmm),
    SettingSpec("rest", "timezone", "rest_timezone", str, _validate_tz),
    SettingSpec("diary", "auto_publish", "diary_auto_publish", bool),
    SettingSpec("diary", "privacy_linter", "diary_privacy_linter", bool),
    SettingSpec("diary", "site_repo_url", "diary_site_repo_url", str),
    SettingSpec("diary", "site_diary_path", "diary_site_diary_path", str),
    SettingSpec("diary", "site_branch", "diary_site_branch", str),
    SettingSpec("ha", "url", "ha_url", str, _validate_ha_url),
    SettingSpec("ha", "token", "ha_token", str),
    SettingSpec("ha", "entities", "ha_entities", list, _validate_entity_id_list),
]

REGISTRY: dict[str, SettingSpec] = {
    f"{s.namespace}.{s.key}": s for s in _SPECS
}

NAMESPACES: tuple[str, ...] = tuple(sorted({s.namespace for s in _SPECS}))


def keys_for_namespace(namespace: str) -> list[str]:
    return [s.key for s in _SPECS if s.namespace == namespace]


def spec_for(qualified_key: str) -> SettingSpec:
    return REGISTRY[qualified_key]


def validate(qualified_key: str, value: Any) -> None:
    spec = REGISTRY[qualified_key]  # raises KeyError if unknown
    # bool first (bool is subclass of int, so isinstance(True, int) is True)
    if spec.type_ is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{qualified_key}: expected bool, got {type(value).__name__}")
    elif not isinstance(value, spec.type_):
        raise ValueError(f"{qualified_key}: expected {spec.type_.__name__}, got {type(value).__name__}")
    if spec.extra_validate:
        spec.extra_validate(value)
