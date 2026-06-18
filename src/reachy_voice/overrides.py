"""Runtime settings overrides — operator tweaks made live from the dashboard.

Two jobs:

  * **persist** a handful of operator-tunable settings to a JSON file so they
    survive a restart/redeploy (``OverridesStore``), and
  * **apply** them to the *running* engine — writing the **real ovs_agent
    objects** that take effect, not a config field nothing reads (the SLV no-op
    trap, see memory ``slv-dashboard-settings-wiring``).

This module imports **no SDK / GStreamer / PortAudio** (like ``tier_a``), so it
unit-tests on a plain dev box: the appliers/readers reach the live engine purely
by duck-typed attribute access over ``engine._app.{config, session,
_client_vad}``.

The same appliers/readers are the single source of truth for three call sites —
startup replay, the runtime WS handler, and the dashboard snapshot — so the
snapshot can never drift from what's actually in effect.

``language`` is persisted in the same store but is **not** a live setting here:
it is genuinely async (it reconnects SLV), so ``main.py`` applies it via the
existing ``engine.set_language()`` coroutine path.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("reachy_voice.overrides")

OVERRIDES_FILENAME = "overrides.json"

# "history" is a turn count in the UI; ovs_agent trims context by a *token*
# budget (Session.max_input_tokens) — there is no turn cap. 7000 tokens (the ovs
# default) ≈ the legacy slider's 20-turn max, so 350 tokens/turn is a clean,
# honest mapping (default = 20 turns).
TOKENS_PER_TURN = 350
MAX_HISTORY_TURNS = 20
DEFAULT_HISTORY_TURNS = 20
DEFAULT_VAD_THRESHOLD = 0.3


# ── file location ────────────────────────────────────────────────────────────
def data_dir() -> Path:
    """Writable dir for runtime state. Bind-mounted on the robot so it survives
    restart + rebuild (see deploy/jetson/voice/docker-compose.yml)."""
    return Path(os.environ.get("REACHY_VOICE_DATA_DIR", "/data"))


def overrides_path() -> Path:
    return data_dir() / OVERRIDES_FILENAME


# ── persistence ──────────────────────────────────────────────────────────────
class OverridesStore:
    """A tiny JSON dict persisted atomically. A missing or corrupt file degrades
    to "no overrides" — it must never raise on read."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path is not None else overrides_path()
        self._data: dict[str, Any] = self.load()

    def load(self) -> dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("overrides file %s is corrupt — ignoring", self._path)
            return {}
        return dict(data) if isinstance(data, dict) else {}

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._write()

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(tmp, self._path)  # atomic on POSIX


# ── turns <-> tokens ─────────────────────────────────────────────────────────
def turns_to_tokens(turns: int) -> int:
    t = max(0, min(MAX_HISTORY_TURNS, int(turns)))
    return t * TOKENS_PER_TURN


def tokens_to_turns(tokens: int | None) -> int:
    if tokens is None:  # ovs "unlimited" default -> show the max
        return DEFAULT_HISTORY_TURNS
    return max(0, min(MAX_HISTORY_TURNS, round(tokens / TOKENS_PER_TURN)))


# ── live appliers (sync, guarded; safe to call from any thread) ──────────────
def _ovs_app(engine: Any) -> Any:
    """The ovs_agent app object (engine._app), or None if the engine isn't
    ready / has the wrong shape."""
    return getattr(engine, "_app", None)


def apply_bargein(engine: Any, value: Any) -> bool:
    cfg = getattr(_ovs_app(engine), "config", None)
    if cfg is None:
        return False
    cfg.barge_in_enabled = bool(value)  # read live per ASR partial / audio chunk
    return True


def apply_vad(engine: Any, value: Any) -> bool:
    app = _ovs_app(engine)
    vad = getattr(app, "_client_vad", None)
    cfg = getattr(app, "config", None)
    if vad is None or cfg is None:
        return False
    v = float(value)
    # the no-op trap: speech detection reads the VAD object's own threshold, so
    # BOTH it and the config (which the dashboard echoes) must change.
    vad.threshold = v
    cfg.client_vad_threshold = v
    return True


def apply_history(engine: Any, value: Any) -> bool:
    sess = getattr(_ovs_app(engine), "session", None)
    if sess is None:
        return False
    sess.max_input_tokens = turns_to_tokens(int(value))  # trims oldest whole turns
    return True


# ── readers (degrade to defaults when the engine isn't ready) ────────────────
def read_bargein(engine: Any) -> bool:
    cfg = getattr(_ovs_app(engine), "config", None)
    v = getattr(cfg, "barge_in_enabled", None)
    return True if v is None else bool(v)  # ovs: None == legacy always-on


def read_vad(engine: Any) -> float:
    app = _ovs_app(engine)
    v = getattr(getattr(app, "_client_vad", None), "threshold", None)
    if v is None:
        v = getattr(getattr(app, "config", None), "client_vad_threshold", None)
    return float(v) if v is not None else DEFAULT_VAD_THRESHOLD


def read_history(engine: Any) -> int:
    sess = getattr(_ovs_app(engine), "session", None)
    return tokens_to_turns(getattr(sess, "max_input_tokens", None))


# ── coercion ─────────────────────────────────────────────────────────────────
def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _clamp_unit(v: Any) -> float:
    return max(0.0, min(1.0, float(v)))


def _clamp_turns(v: Any) -> int:
    return max(0, min(MAX_HISTORY_TURNS, int(v)))


# ── registry: one row per live setting, shared by every call site ────────────
@dataclass(frozen=True)
class Setting:
    key: str
    apply: Callable[[Any, Any], bool]
    read: Callable[[Any], Any]
    coerce: Callable[[Any], Any]


LIVE_SETTINGS: dict[str, Setting] = {
    s.key: s
    for s in (
        Setting("bargein", apply_bargein, read_bargein, _as_bool),
        Setting("vad", apply_vad, read_vad, _clamp_unit),
        Setting("history", apply_history, read_history, _clamp_turns),
    )
}


def apply_saved(engine: Any, store: OverridesStore) -> None:
    """Replay the saved live settings onto the running engine at startup. Unknown
    keys (e.g. ``language``, handled elsewhere) are ignored; a bad single value
    is logged and skipped, never fatal."""
    saved = store.all()
    for key, setting in LIVE_SETTINGS.items():
        if key not in saved:
            continue
        try:
            ok = setting.apply(engine, setting.coerce(saved[key]))
            logger.info("override %s=%r applied=%s", key, saved[key], ok)
        except Exception:  # noqa: BLE001 — one bad override must not break startup
            logger.exception("failed to apply saved override %s", key)
