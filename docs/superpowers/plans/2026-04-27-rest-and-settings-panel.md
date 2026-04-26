# Rest Window + Settings Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable daily rest window that pauses ASR/TTS/LLM/Vision and runs housekeeping (diary generate+publish), driven by a new dashboard SETTINGS tab with manual diary triggers — all settings persisted via the existing `runtime-overrides.yaml` mechanism.

**Architecture:** New `RestPlugin` polls `app.config.rest_*` every 30s; on entry/exit it emits `rest_start`/`rest_end` events that other plugins subscribe to and self-pause. Housekeeping tasks are pluggable (Protocol-based registry); v1 ships `DiaryGenerateAndPublishTask`. Dashboard adds a SETTINGS tab with a section registry, plus settings API and async diary-trigger endpoints with WS progress events.

**Tech Stack:** Python 3.10+, dataclasses, asyncio, aiohttp, pyyaml, zoneinfo (stdlib), pytest, vanilla JS for dashboard frontend.

**Spec:** `docs/superpowers/specs/2026-04-27-rest-and-settings-panel-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|------|---------------|
| `src/reachy_claw/settings_schema.py` | Registry of dashboard-tunable settings: namespace+key → config field, type, validator |
| `src/reachy_claw/plugins/rest_plugin.py` | RestPlugin: schedule loop, event emission, housekeeping orchestration |
| `src/reachy_claw/plugins/housekeeping_tasks.py` | `HousekeepingTask` Protocol + `DiaryGenerateAndPublishTask` |
| `src/reachy_claw/plugins/dashboard_static/settings.js` | Settings tab renderer + section registry + diary history UI |
| `src/reachy_claw/plugins/dashboard_static/settings.css` | Settings tab styling |
| `tests/test_settings_schema.py` | Validators, namespace queries |
| `tests/test_settings_api.py` | GET/PUT endpoint behavior + persistence to runtime-overrides.yaml |
| `tests/test_rest_plugin.py` | Schedule loop, event emission with frozen clock |
| `tests/test_housekeeping_diary.py` | DiaryGenerateAndPublishTask runs scripts conditionally |
| `tests/test_diary_trigger_api.py` | POST /api/diary/generate + /publish with WS event flow |
| `tests/test_plugin_rest_hooks.py` | Plugin base hooks + per-plugin pause/resume integration tests |

### Modified files

| Path | Change |
|------|--------|
| `src/reachy_claw/config.py` | Add 9 new dataclass fields + KEY_MAP entries for `rest:` and `diary:` yaml sections |
| `reachy-claw.example.yaml` | Document the new `rest:` and `diary:` sections |
| `src/reachy_claw/plugin.py` | Add `on_rest_start` / `on_rest_end` async no-op base methods |
| `src/reachy_claw/plugins/conversation_plugin.py` | Implement pause/resume |
| `src/reachy_claw/plugins/face_tracker_plugin.py` | Implement pause/resume |
| `src/reachy_claw/plugins/vision_client_plugin.py` | Implement pause/resume |
| `src/reachy_claw/plugins/motion_plugin.py` | Implement pause/resume |
| `src/reachy_claw/plugins/dashboard_plugin.py` | Settings API + diary trigger endpoints + WS job events |
| `src/reachy_claw/plugins/dashboard_static/index.html` | Add SETTINGS tab |
| `src/reachy_claw/plugins/dashboard_static/app.js` | Tab switching wiring |
| `src/reachy_claw/app.py` | Register RestPlugin and `DiaryGenerateAndPublishTask` |

---

## Task 1: Add config fields + KEY_MAP entries

**Files:**
- Modify: `src/reachy_claw/config.py`
- Modify: `reachy-claw.example.yaml`
- Test: `tests/test_config.py` (extend)

- [ ] **Step 1.1: Write failing test that the new fields default correctly**

Append to `tests/test_config.py`:

```python
def test_rest_diary_defaults():
    from reachy_claw.config import Config
    c = Config()
    assert c.rest_enabled is True
    assert c.rest_window_start == "23:00"
    assert c.rest_window_end == "24:00"
    assert c.rest_timezone == "Asia/Shanghai"
    assert c.diary_auto_publish is True
    assert c.diary_privacy_linter is True
    assert c.diary_site_repo_url == ""
    assert c.diary_site_diary_path == "src/content/docs"
    assert c.diary_site_branch == "main"
```

- [ ] **Step 1.2: Run; expect failure (AttributeError)**

```
uv run pytest tests/test_config.py::test_rest_diary_defaults -v
```

- [ ] **Step 1.3: Add fields to Config dataclass**

In `src/reachy_claw/config.py`, locate the `@dataclass class Config:` block (around line 22-160). After the existing field group (e.g. after `audio_volume`), add:

```python
    # ── Rest window ─────────────────────────────────────────────
    rest_enabled: bool = True
    rest_window_start: str = "23:00"   # HH:MM
    rest_window_end: str = "24:00"     # HH:MM
    rest_timezone: str = "Asia/Shanghai"

    # ── Diary publishing ────────────────────────────────────────
    diary_auto_publish: bool = True
    diary_privacy_linter: bool = True
    diary_site_repo_url: str = ""
    diary_site_diary_path: str = "src/content/docs"
    diary_site_branch: str = "main"
```

- [ ] **Step 1.4: Add KEY_MAP entries**

In `src/reachy_claw/config.py`, find the `KEY_MAP` dict (around line 215-230). Append:

```python
    ("rest", "enabled"): "rest_enabled",
    ("rest", "window_start"): "rest_window_start",
    ("rest", "window_end"): "rest_window_end",
    ("rest", "timezone"): "rest_timezone",
    ("diary", "auto_publish"): "diary_auto_publish",
    ("diary", "privacy_linter"): "diary_privacy_linter",
    ("diary", "site_repo_url"): "diary_site_repo_url",
    ("diary", "site_diary_path"): "diary_site_diary_path",
    ("diary", "site_branch"): "diary_site_branch",
```

- [ ] **Step 1.5: Document yaml structure**

Append to `reachy-claw.example.yaml`:

```yaml

# ── Rest window ─────────────────────────────────────────────────
rest:
  enabled: true
  window_start: "23:00"
  window_end: "24:00"
  timezone: Asia/Shanghai

# ── Diary publishing ────────────────────────────────────────────
diary:
  auto_publish: true
  privacy_linter: true
  site_repo_url: ""              # required for publishing; e.g. git@github.com:org/site.git
  site_diary_path: src/content/docs
  site_branch: main
```

- [ ] **Step 1.6: Run all tests, expect PASS**

```
uv run pytest tests/test_config.py -v
```

- [ ] **Step 1.7: Commit**

```bash
git add src/reachy_claw/config.py reachy-claw.example.yaml tests/test_config.py
git commit -m "feat(config): add rest + diary publishing config fields"
```

---

## Task 2: Settings schema registry

**Files:**
- Create: `src/reachy_claw/settings_schema.py`
- Create: `tests/test_settings_schema.py`

- [ ] **Step 2.1: Write failing tests for the registry**

```python
# tests/test_settings_schema.py
"""Tests for the dashboard-tunable settings registry."""

from __future__ import annotations

import pytest

from reachy_claw.settings_schema import (
    SettingSpec,
    NAMESPACES,
    REGISTRY,
    keys_for_namespace,
    spec_for,
    validate,
)


def test_namespaces_are_rest_and_diary():
    assert set(NAMESPACES) == {"rest", "diary"}


def test_registry_has_expected_keys():
    expected = {
        "rest.enabled",
        "rest.window_start",
        "rest.window_end",
        "rest.timezone",
        "diary.auto_publish",
        "diary.privacy_linter",
        "diary.site_repo_url",
        "diary.site_diary_path",
        "diary.site_branch",
    }
    assert set(REGISTRY) == expected


def test_keys_for_namespace_returns_only_that_ns():
    assert set(keys_for_namespace("rest")) == {
        "enabled",
        "window_start",
        "window_end",
        "timezone",
    }
    assert set(keys_for_namespace("diary")) == {
        "auto_publish",
        "privacy_linter",
        "site_repo_url",
        "site_diary_path",
        "site_branch",
    }


def test_spec_for_returns_field_name_and_type():
    spec = spec_for("rest.window_start")
    assert spec.config_field == "rest_window_start"
    assert spec.type_ == str


def test_validate_accepts_valid_hhmm():
    validate("rest.window_start", "23:00")
    validate("rest.window_end", "07:30")


def test_validate_rejects_bad_hhmm():
    with pytest.raises(ValueError):
        validate("rest.window_start", "25:00")
    with pytest.raises(ValueError):
        validate("rest.window_start", "9-30")


def test_validate_rejects_bad_timezone():
    with pytest.raises(ValueError):
        validate("rest.timezone", "Not/A/Tz")


def test_validate_accepts_valid_timezone():
    validate("rest.timezone", "Asia/Shanghai")
    validate("rest.timezone", "UTC")


def test_validate_bool_type():
    validate("rest.enabled", True)
    validate("diary.auto_publish", False)
    with pytest.raises(ValueError):
        validate("rest.enabled", "true")  # string not bool
    with pytest.raises(ValueError):
        validate("rest.enabled", 1)


def test_validate_string_type():
    validate("diary.site_repo_url", "git@github.com:x/y.git")
    validate("diary.site_repo_url", "")  # empty allowed
    with pytest.raises(ValueError):
        validate("diary.site_repo_url", 123)


def test_validate_unknown_key_rejected():
    with pytest.raises(KeyError):
        validate("rest.unknown", "anything")
```

- [ ] **Step 2.2: Run; expect ImportError / AttributeError**

```
uv run pytest tests/test_settings_schema.py -v
```

- [ ] **Step 2.3: Implement the registry**

```python
# src/reachy_claw/settings_schema.py
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


_HHMM = re.compile(r"^([01]\d|2[0-4]):[0-5]\d$")


def _validate_hhmm(v: Any) -> None:
    if not isinstance(v, str) or not _HHMM.match(v):
        raise ValueError(f"expected HH:MM (24h), got {v!r}")


def _validate_tz(v: Any) -> None:
    if not isinstance(v, str):
        raise ValueError(f"timezone must be a string, got {type(v).__name__}")
    try:
        ZoneInfo(v)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"unknown IANA timezone: {v!r}") from e


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
```

- [ ] **Step 2.4: Run; expect PASS**

```
uv run pytest tests/test_settings_schema.py -v
```

- [ ] **Step 2.5: Commit**

```bash
git add src/reachy_claw/settings_schema.py tests/test_settings_schema.py
git commit -m "feat(settings): add settings_schema registry with validators"
```

---

## Task 3: Plugin base hooks for rest

**Files:**
- Modify: `src/reachy_claw/plugin.py`
- Test: `tests/test_plugin_rest_hooks.py` (create)

- [ ] **Step 3.1: Write failing test**

```python
# tests/test_plugin_rest_hooks.py
"""Tests for Plugin base on_rest_start / on_rest_end hooks."""

from __future__ import annotations

import asyncio

import pytest

from reachy_claw.plugin import Plugin


class _NoopPlugin(Plugin):
    name = "noop"

    async def start(self) -> None:
        pass


class _TrackingPlugin(Plugin):
    name = "tracking"

    def __init__(self, app):
        super().__init__(app)
        self.entered = False
        self.exited = False

    async def start(self) -> None:
        pass

    async def on_rest_start(self) -> None:
        self.entered = True

    async def on_rest_end(self) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_default_hooks_are_noop():
    p = _NoopPlugin(app=None)  # type: ignore[arg-type]
    # Must be coroutines that return without error
    await p.on_rest_start()
    await p.on_rest_end()


@pytest.mark.asyncio
async def test_subclass_can_override_hooks():
    p = _TrackingPlugin(app=None)  # type: ignore[arg-type]
    await p.on_rest_start()
    await p.on_rest_end()
    assert p.entered is True
    assert p.exited is True
```

- [ ] **Step 3.2: Run; expect failure**

```
uv run pytest tests/test_plugin_rest_hooks.py -v
```

- [ ] **Step 3.3: Add base hooks to Plugin**

Open `src/reachy_claw/plugin.py`. After the `stop()` method, add:

```python
    async def on_rest_start(self) -> None:
        """Called when the system enters its rest window.

        Override to pause expensive work (ASR/TTS/LLM/Vision). Default: no-op.
        """

    async def on_rest_end(self) -> None:
        """Called when the rest window ends. Override to resume work. Default: no-op."""
```

- [ ] **Step 3.4: Run; expect PASS**

```
uv run pytest tests/test_plugin_rest_hooks.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add src/reachy_claw/plugin.py tests/test_plugin_rest_hooks.py
git commit -m "feat(plugin): add on_rest_start/end base hooks (no-op default)"
```

---

## Task 4: RestPlugin — schedule loop + event emission

**Files:**
- Create: `src/reachy_claw/plugins/rest_plugin.py`
- Create: `tests/test_rest_plugin.py`

- [ ] **Step 4.1: Write failing test (frozen clock, manual loop tick)**

```python
# tests/test_rest_plugin.py
"""Tests for RestPlugin schedule logic and event emission."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from reachy_claw.event_bus import EventBus
from reachy_claw.plugins.rest_plugin import RestPlugin, _is_in_window


@dataclass
class _StubConfig:
    rest_enabled: bool = True
    rest_window_start: str = "23:00"
    rest_window_end: str = "24:00"
    rest_timezone: str = "Asia/Shanghai"


class _StubApp:
    def __init__(self):
        self.events = EventBus()
        self.config = _StubConfig()


def test_is_in_window_simple():
    tz = ZoneInfo("Asia/Shanghai")
    inside = datetime(2026, 4, 27, 23, 30, tzinfo=tz)
    outside = datetime(2026, 4, 27, 12, 0, tzinfo=tz)
    assert _is_in_window(inside, "23:00", "24:00") is True
    assert _is_in_window(outside, "23:00", "24:00") is False


def test_is_in_window_overnight():
    """Window 23:00-01:00 spans midnight."""
    tz = ZoneInfo("Asia/Shanghai")
    in_pre_midnight = datetime(2026, 4, 27, 23, 30, tzinfo=tz)
    in_post_midnight = datetime(2026, 4, 28, 0, 30, tzinfo=tz)
    outside = datetime(2026, 4, 28, 2, 0, tzinfo=tz)
    assert _is_in_window(in_pre_midnight, "23:00", "01:00") is True
    assert _is_in_window(in_post_midnight, "23:00", "01:00") is True
    assert _is_in_window(outside, "23:00", "01:00") is False


def test_is_in_window_24_means_end_of_day():
    """The legacy '24:00' notation means up to (but not including) the next 00:00."""
    tz = ZoneInfo("Asia/Shanghai")
    inside = datetime(2026, 4, 27, 23, 59, tzinfo=tz)
    outside = datetime(2026, 4, 28, 0, 0, tzinfo=tz)
    assert _is_in_window(inside, "23:00", "24:00") is True
    assert _is_in_window(outside, "23:00", "24:00") is False


@pytest.mark.asyncio
async def test_emits_rest_start_and_rest_end():
    app = _StubApp()
    received = []

    def handler(data):
        received.append(data)

    app.events.subscribe("rest_start", lambda d: received.append(("start", d)))
    app.events.subscribe("rest_end", lambda d: received.append(("end", d)))

    plugin = RestPlugin(app)  # type: ignore[arg-type]
    await plugin._enter_rest()
    await plugin._exit_rest()

    kinds = [r[0] for r in received]
    assert kinds == ["start", "end"]


@pytest.mark.asyncio
async def test_disabled_window_does_not_enter():
    app = _StubApp()
    app.config.rest_enabled = False
    plugin = RestPlugin(app)  # type: ignore[arg-type]
    # Even if "now" would normally be inside the window, disabled means no entry.
    assert plugin._should_rest_now(datetime(2026, 4, 27, 23, 30, tzinfo=ZoneInfo("Asia/Shanghai"))) is False
```

- [ ] **Step 4.2: Run; expect failure**

```
uv run pytest tests/test_rest_plugin.py -v
```

- [ ] **Step 4.3: Implement RestPlugin (schedule + events; housekeeping comes in Task 5)**

```python
# src/reachy_claw/plugins/rest_plugin.py
"""RestPlugin — daily rest window orchestrator.

Polls app.config every 30s; when "now" enters the rest window, emits
`rest_start` (other plugins self-pause via on_rest_start) and runs registered
HousekeepingTasks. When the window ends, emits `rest_end`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, time as dtime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..plugin import Plugin

if TYPE_CHECKING:
    from .housekeeping_tasks import HousekeepingTask

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30
HARD_OVERRUN_CAP_S = 600  # housekeeping may run up to 10 min past end_time


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(hour=int(h) % 24, minute=int(m))  # "24:00" → 00:00 next-day handled separately


def _is_in_window(now: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    """Whether `now` (timezone-aware) falls in [start, end) on the local clock.

    Special-cases:
      - end == "24:00" means end-of-day: the window is [start, 24:00) i.e.
        any time at or after start, on this calendar day, in the local tz.
      - Otherwise, if end <= start, the window wraps midnight.
    """
    start = _parse_hhmm(start_hhmm)
    cur = now.timetz().replace(tzinfo=None)

    if end_hhmm == "24:00":
        return cur >= start

    end = _parse_hhmm(end_hhmm)
    if start <= end:
        return start <= cur < end
    # wraparound: in window if cur >= start OR cur < end
    return cur >= start or cur < end


class RestPlugin(Plugin):
    name = "rest"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._resting = False
        self._tasks: list[HousekeepingTask] = []

    def register_task(self, task: "HousekeepingTask") -> None:
        self._tasks.append(task)

    def _should_rest_now(self, now: datetime) -> bool:
        if not getattr(self.app.config, "rest_enabled", True):
            return False
        return _is_in_window(
            now,
            self.app.config.rest_window_start,
            self.app.config.rest_window_end,
        )

    def _now(self) -> datetime:
        try:
            tz = ZoneInfo(self.app.config.rest_timezone)
        except ZoneInfoNotFoundError:
            logger.warning(
                "Invalid rest_timezone %r; falling back to UTC",
                self.app.config.rest_timezone,
            )
            tz = ZoneInfo("UTC")
        return datetime.now(tz=tz)

    async def _enter_rest(self) -> None:
        if self._resting:
            return
        self._resting = True
        logger.info("Entering rest window")
        self.app.events.emit("rest_start", {"started_at": int(time.time())})
        # Run housekeeping tasks sequentially in the background.
        asyncio.create_task(self._run_housekeeping())

    async def _exit_rest(self) -> None:
        if not self._resting:
            return
        self._resting = False
        logger.info("Exiting rest window")
        self.app.events.emit("rest_end", {"ended_at": int(time.time())})

    async def _run_housekeeping(self) -> None:
        for task in self._tasks:
            self.app.events.emit("housekeeping_task_start", {"name": task.name})
            ok = True
            error = None
            try:
                await asyncio.wait_for(task.run(self.app), timeout=HARD_OVERRUN_CAP_S)
            except asyncio.TimeoutError:
                ok = False
                error = "timed out"
                logger.warning("Housekeeping task %r timed out", task.name)
            except Exception as e:  # noqa: BLE001
                ok = False
                error = str(e)
                logger.warning("Housekeeping task %r failed: %s", task.name, e)
            self.app.events.emit(
                "housekeeping_task_end", {"name": task.name, "ok": ok, "error": error}
            )

    async def start(self) -> None:
        self._running = True
        while self._running:
            should = self._should_rest_now(self._now())
            if should and not self._resting:
                await self._enter_rest()
            elif not should and self._resting:
                await self._exit_rest()
            try:
                await asyncio.sleep(POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return
```

- [ ] **Step 4.4: Run; expect PASS**

```
uv run pytest tests/test_rest_plugin.py -v
```

- [ ] **Step 4.5: Commit**

```bash
git add src/reachy_claw/plugins/rest_plugin.py tests/test_rest_plugin.py
git commit -m "feat(rest): RestPlugin with schedule loop and rest_start/end events"
```

---

## Task 5: HousekeepingTask + DiaryGenerateAndPublishTask

**Files:**
- Create: `src/reachy_claw/plugins/housekeeping_tasks.py`
- Create: `tests/test_housekeeping_diary.py`

- [ ] **Step 5.1: Write failing test (subprocess mocking)**

```python
# tests/test_housekeeping_diary.py
"""Tests for the diary housekeeping task."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_claw.plugins.housekeeping_tasks import DiaryGenerateAndPublishTask


@dataclass
class _StubConfig:
    diary_auto_publish: bool = True
    diary_site_repo_url: str = "git@github.com:org/site.git"


class _StubApp:
    def __init__(self):
        self.config = _StubConfig()


@pytest.mark.asyncio
async def test_runs_generate_then_publish_when_auto_publish_true():
    app = _StubApp()
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        # Expect 2 subprocess calls: generate then publish
        assert mock_create.call_count == 2
        gen_args = mock_create.call_args_list[0].args
        pub_args = mock_create.call_args_list[1].args
        assert "generate_diary.py" in " ".join(gen_args)
        assert "publish_diary.py" in " ".join(pub_args)


@pytest.mark.asyncio
async def test_skips_publish_when_auto_publish_false():
    app = _StubApp()
    app.config.diary_auto_publish = False
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        assert mock_create.call_count == 1  # only generate


@pytest.mark.asyncio
async def test_skips_publish_when_site_url_empty():
    app = _StubApp()
    app.config.diary_site_repo_url = ""
    task = DiaryGenerateAndPublishTask()
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))

    with patch(
        "reachy_claw.plugins.housekeeping_tasks.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ) as mock_create:
        await task.run(app)
        assert mock_create.call_count == 1  # only generate, publish skipped
```

- [ ] **Step 5.2: Run; expect failure**

```
uv run pytest tests/test_housekeeping_diary.py -v
```

- [ ] **Step 5.3: Implement housekeeping tasks**

```python
# src/reachy_claw/plugins/housekeeping_tasks.py
"""Housekeeping tasks run during the rest window.

A HousekeepingTask is anything with `.name: str` and `.run(app)` coroutine.
v1 ships DiaryGenerateAndPublishTask. New tasks (DBVacuumTask,
CoverImageGenerateTask, etc.) just register with RestPlugin.register_task().
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


class HousekeepingTask(Protocol):
    name: str

    async def run(self, app) -> None: ...


class DiaryGenerateAndPublishTask:
    """Runs generate_diary.py for today, then publish_diary.py if auto_publish."""

    name = "diary_generate_and_publish"

    async def run(self, app) -> None:
        date = datetime.now().strftime("%Y-%m-%d")
        gen_script = REPO_ROOT / "scripts" / "generate_diary.py"
        pub_script = REPO_ROOT / "scripts" / "publish_diary.py"

        await self._run_subprocess(
            sys.executable,
            str(gen_script),
            "--date",
            date,
            label="generate_diary",
        )

        if not getattr(app.config, "diary_auto_publish", True):
            logger.info("Skipping publish: diary_auto_publish is false")
            return
        if not getattr(app.config, "diary_site_repo_url", "").strip():
            logger.warning("Skipping publish: diary_site_repo_url is empty")
            return

        env = os.environ.copy()
        env["SITE_REPO_URL"] = app.config.diary_site_repo_url
        env["SITE_DIARY_PATH"] = app.config.diary_site_diary_path
        env["SITE_BRANCH"] = app.config.diary_site_branch

        await self._run_subprocess(
            sys.executable,
            str(pub_script),
            "--date",
            date,
            label="publish_diary",
            env=env,
        )

    @staticmethod
    async def _run_subprocess(*args: str, label: str, env: dict | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} failed (rc={proc.returncode}): {stderr.decode(errors='replace')}"
            )
        logger.info("%s OK: %s", label, stdout.decode(errors="replace").strip()[:200])
```

- [ ] **Step 5.4: Run; expect PASS**

```
uv run pytest tests/test_housekeeping_diary.py -v
```

- [ ] **Step 5.5: Commit**

```bash
git add src/reachy_claw/plugins/housekeeping_tasks.py tests/test_housekeeping_diary.py
git commit -m "feat(rest): housekeeping registry + DiaryGenerateAndPublishTask"
```

---

## Task 6: Wire up RestPlugin + register diary task in app

**Files:**
- Modify: `src/reachy_claw/app.py`

- [ ] **Step 6.1: Read app.py to find plugin registration**

```bash
grep -n "self.plugins\.append\|RestPlugin\|DailyLogPlugin" src/reachy_claw/app.py | head
```

Note where existing plugins are registered (likely a `_register_plugins` method).

- [ ] **Step 6.2: Add RestPlugin registration**

Open `src/reachy_claw/app.py`. Find the plugin registration block (where `DailyLogPlugin` is appended). After the existing plugins (so RestPlugin starts last), add:

```python
        from .plugins.rest_plugin import RestPlugin
        from .plugins.housekeeping_tasks import DiaryGenerateAndPublishTask

        rest = RestPlugin(self)
        rest.register_task(DiaryGenerateAndPublishTask())
        self.plugins.append(rest)
```

- [ ] **Step 6.3: Smoke test that the app boots**

```bash
uv run python -c "from reachy_claw.app import ReachyClawApp; print('import ok')"
```

Expected: `import ok` (no traceback). If existing tests in `tests/test_main.py` exercise app construction, they should still pass:

```bash
uv run pytest tests/test_main.py -v
```

- [ ] **Step 6.4: Commit**

```bash
git add src/reachy_claw/app.py
git commit -m "feat(app): register RestPlugin with diary housekeeping task"
```

---

## Task 7: Settings API — GET / PUT endpoints

**Files:**
- Modify: `src/reachy_claw/plugins/dashboard_plugin.py`
- Create: `tests/test_settings_api.py`

- [ ] **Step 7.1: Write failing tests**

```python
# tests/test_settings_api.py
"""Tests for /api/settings/<namespace> endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web

from reachy_claw.config import Config


@pytest.fixture
def app_with_dashboard(tmp_path, monkeypatch):
    """Construct a minimal aiohttp app exposing settings endpoints
    bound to a temp config + temp DATA_DIR."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from reachy_claw.plugins.dashboard_plugin import (
        _build_settings_handlers,  # see implementation in Step 7.3
    )

    config = Config()

    class _StubApp:
        pass

    stub = _StubApp()
    stub.config = config

    aio = web.Application()
    handlers = _build_settings_handlers(stub)
    aio.router.add_get("/api/settings/{namespace}", handlers["get"])
    aio.router.add_put("/api/settings/{namespace}", handlers["put"])
    return aio, stub, tmp_path


@pytest.mark.asyncio
async def test_get_rest_namespace(app_with_dashboard, aiohttp_client):
    aio, stub, _ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.get("/api/settings/rest")
    assert r.status == 200
    body = await r.json()
    assert body["enabled"] is True
    assert body["window_start"] == "23:00"


@pytest.mark.asyncio
async def test_get_unknown_namespace_404(app_with_dashboard, aiohttp_client):
    aio, *_ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.get("/api/settings/nope")
    assert r.status == 404


@pytest.mark.asyncio
async def test_put_updates_config_and_persists(app_with_dashboard, aiohttp_client, tmp_path):
    aio, stub, dd = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put(
        "/api/settings/rest",
        json={"window_start": "22:30", "enabled": False},
    )
    assert r.status == 200
    body = await r.json()
    assert set(body["updated"]) == {"window_start", "enabled"}
    # In-process config mutated:
    assert stub.config.rest_window_start == "22:30"
    assert stub.config.rest_enabled is False
    # runtime-overrides.yaml on disk has the values:
    overrides_path = dd / "runtime-overrides.yaml"
    assert overrides_path.exists()
    content = overrides_path.read_text()
    assert "window_start" in content or "rest_window_start" in content


@pytest.mark.asyncio
async def test_put_rejects_invalid_value(app_with_dashboard, aiohttp_client):
    aio, stub, _ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put("/api/settings/rest", json={"window_start": "25:99"})
    assert r.status == 400


@pytest.mark.asyncio
async def test_put_rejects_unknown_key(app_with_dashboard, aiohttp_client):
    aio, *_ = app_with_dashboard
    client = await aiohttp_client(aio)
    r = await client.put("/api/settings/rest", json={"made_up": 1})
    assert r.status == 400
```

- [ ] **Step 7.2: Add `pytest-aiohttp` if missing**

```bash
grep -i "pytest-aiohttp" pyproject.toml || uv add --dev pytest-aiohttp
```

- [ ] **Step 7.3: Implement handlers + builder**

In `src/reachy_claw/plugins/dashboard_plugin.py`, add near the top (after existing imports):

```python
from ..settings_schema import (
    NAMESPACES,
    keys_for_namespace,
    spec_for,
    validate as validate_setting,
)
from ..config import save_runtime_overrides
```

Add a builder factory (so tests can construct handlers without the full plugin):

```python
def _build_settings_handlers(app):
    from aiohttp import web

    async def get_handler(request):
        ns = request.match_info["namespace"]
        if ns not in NAMESPACES:
            return web.json_response({"error": "unknown namespace"}, status=404)
        out = {}
        for k in keys_for_namespace(ns):
            spec = spec_for(f"{ns}.{k}")
            out[k] = getattr(app.config, spec.config_field)
        return web.json_response(out)

    async def put_handler(request):
        ns = request.match_info["namespace"]
        if ns not in NAMESPACES:
            return web.json_response({"error": "unknown namespace"}, status=404)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "expected JSON object"}, status=400)

        # Validate everything first (no partial updates).
        fields_to_save: list[str] = []
        for k, v in body.items():
            qkey = f"{ns}.{k}"
            try:
                validate_setting(qkey, v)
            except KeyError:
                return web.json_response(
                    {"error": f"unknown key: {qkey}"}, status=400
                )
            except ValueError as e:
                return web.json_response(
                    {"error": f"invalid value for {qkey}: {e}"}, status=400
                )
            fields_to_save.append(spec_for(qkey).config_field)

        # Apply.
        for k, v in body.items():
            spec = spec_for(f"{ns}.{k}")
            setattr(app.config, spec.config_field, v)
        save_runtime_overrides(app.config, fields_to_save)
        return web.json_response({"updated": list(body.keys())})

    return {"get": get_handler, "put": put_handler}
```

In the dashboard plugin's route registration section (find where `_handle_diary_*` routes are added), append:

```python
        handlers = _build_settings_handlers(self.app)
        app_router.router.add_get("/api/settings/{namespace}", handlers["get"])
        app_router.router.add_put("/api/settings/{namespace}", handlers["put"])
```

(Adapt variable names — `app_router` is whatever the dashboard uses; copy the pattern from existing `add_get` calls in the same method.)

- [ ] **Step 7.4: Run; expect PASS**

```
uv run pytest tests/test_settings_api.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_plugin.py tests/test_settings_api.py pyproject.toml uv.lock
git commit -m "feat(dashboard): /api/settings GET/PUT backed by runtime-overrides"
```

---

## Task 8: Diary trigger API — async + WS progress

**Files:**
- Modify: `src/reachy_claw/plugins/dashboard_plugin.py`
- Create: `tests/test_diary_trigger_api.py`

- [ ] **Step 8.1: Write failing tests**

```python
# tests/test_diary_trigger_api.py
"""Tests for POST /api/diary/generate, /api/diary/publish, GET /api/diary/status."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from reachy_claw.config import Config
from reachy_claw.storage.db import Database


@pytest.fixture
def app_with_diary_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from reachy_claw.plugins.dashboard_plugin import _build_diary_trigger_handlers

    db = Database(tmp_path / "reachy.db")
    db.init()

    class _StubApp:
        pass

    stub = _StubApp()
    stub.config = Config()
    stub.db = db
    # tracks emitted ws messages for assertion
    stub.ws_emissions = []

    async def fake_broadcast(msg):
        stub.ws_emissions.append(msg)

    aio = web.Application()
    handlers = _build_diary_trigger_handlers(stub, broadcast=fake_broadcast)
    aio.router.add_get("/api/diary/status", handlers["status"])
    aio.router.add_post("/api/diary/generate", handlers["generate"])
    aio.router.add_post("/api/diary/publish", handlers["publish"])
    return aio, stub, tmp_path


@pytest.mark.asyncio
async def test_status_returns_dates(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    today = datetime.now().strftime("%Y-%m-%d")
    stub.db.save_diary(date=today, markdown="m", llm_model="m", prompt_version="v1")
    client = await aiohttp_client(aio)
    r = await client.get("/api/diary/status")
    assert r.status == 200
    body = await r.json()
    found = next((d for d in body["dates"] if d["date"] == today), None)
    assert found is not None
    assert found["generated"] is True
    assert found["published"] is False


@pytest.mark.asyncio
async def test_generate_already_done_returns_200_skip(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    stub.db.save_diary(
        date="2026-04-26", markdown="m", llm_model="m", prompt_version="v1"
    )
    client = await aiohttp_client(aio)
    r = await client.post(
        "/api/diary/generate", json={"date": "2026-04-26", "force": False}
    )
    assert r.status == 200
    body = await r.json()
    assert body["status"] == "already-generated"


@pytest.mark.asyncio
async def test_generate_kicks_off_subprocess_and_emits_ws(app_with_diary_api, aiohttp_client):
    aio, stub, _ = app_with_diary_api
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.communicate = AsyncMock(return_value=(b"ok", b""))
    client = await aiohttp_client(aio)
    with patch(
        "reachy_claw.plugins.dashboard_plugin.asyncio.create_subprocess_exec",
        AsyncMock(return_value=fake_proc),
    ):
        r = await client.post(
            "/api/diary/generate", json={"date": "2026-04-26", "force": True}
        )
        assert r.status == 202
        body = await r.json()
        assert "job_id" in body
        # let the background task complete
        await asyncio.sleep(0.05)

    phases = [m["phase"] for m in stub.ws_emissions if m.get("type") == "diary_job"]
    assert "generating" in phases
    assert "done" in phases or "error" in phases
```

- [ ] **Step 8.2: Run; expect failure**

```
uv run pytest tests/test_diary_trigger_api.py -v
```

- [ ] **Step 8.3: Implement diary trigger handlers**

Append to `src/reachy_claw/plugins/dashboard_plugin.py`:

```python
def _build_diary_trigger_handlers(app, broadcast):
    """Builds /api/diary/status, /generate, /publish handlers.

    `broadcast` is an async callable that takes a dict and pushes it to all WS clients.
    """
    from aiohttp import web
    import asyncio
    import sys
    import uuid
    from datetime import datetime, timedelta
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[3]
    GEN = REPO_ROOT / "scripts" / "generate_diary.py"
    PUB = REPO_ROOT / "scripts" / "publish_diary.py"
    SCAN_DAYS = 14

    # Prevent concurrent generate/publish for the same date.
    _date_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(date: str) -> asyncio.Lock:
        if date not in _date_locks:
            _date_locks[date] = asyncio.Lock()
        return _date_locks[date]

    async def status_handler(request):
        today = datetime.now()
        out = []
        for i in range(SCAN_DAYS):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            row = app.db.get_diary(d)
            out.append({
                "date": d,
                "generated": row is not None,
                "published": bool(row and row.get("published_at")),
            })
        return web.json_response({"dates": out, "scan_window_days": SCAN_DAYS})

    async def _run_script(script: Path, date: str, force: bool, *, env_extra=None) -> tuple[int, str]:
        import os
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        args = [sys.executable, str(script), "--date", date]
        if force:
            args.append("--force")
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )
        out, err = await proc.communicate()
        return proc.returncode, (err.decode(errors="replace") or out.decode(errors="replace"))

    async def _do_generate(date: str, force: bool, job_id: str):
        lock = _lock_for(date)
        async with lock:
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "generating", "date": date})
            rc, msg = await _run_script(GEN, date, force)
            if rc != 0:
                await broadcast({"type": "diary_job", "job_id": job_id, "phase": "error", "date": date, "error": msg})
                return
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "done", "date": date})

    async def _do_publish(date: str, force: bool, job_id: str):
        lock = _lock_for(date)
        async with lock:
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "publishing", "date": date})
            env_extra = {
                "SITE_REPO_URL": app.config.diary_site_repo_url,
                "SITE_DIARY_PATH": app.config.diary_site_diary_path,
                "SITE_BRANCH": app.config.diary_site_branch,
            }
            rc, msg = await _run_script(PUB, date, force, env_extra=env_extra)
            if rc != 0:
                await broadcast({"type": "diary_job", "job_id": job_id, "phase": "error", "date": date, "error": msg})
                return
            await broadcast({"type": "diary_job", "job_id": job_id, "phase": "done", "date": date})

    async def generate_handler(request):
        body = await request.json()
        date = body.get("date")
        force = bool(body.get("force"))
        if not date:
            return web.json_response({"error": "date required"}, status=400)
        if not force and app.db.get_diary(date) is not None:
            return web.json_response({"date": date, "status": "already-generated"})
        job_id = uuid.uuid4().hex[:12]
        asyncio.create_task(_do_generate(date, force, job_id))
        return web.json_response({"job_id": job_id, "date": date}, status=202)

    async def publish_handler(request):
        body = await request.json()
        date = body.get("date")
        force = bool(body.get("force"))
        if not date:
            return web.json_response({"error": "date required"}, status=400)
        row = app.db.get_diary(date)
        if row is None:
            return web.json_response({"error": "diary not generated"}, status=404)
        if not force and row.get("published_at"):
            return web.json_response({"date": date, "status": "already-published"})
        job_id = uuid.uuid4().hex[:12]
        asyncio.create_task(_do_publish(date, force, job_id))
        return web.json_response({"job_id": job_id, "date": date}, status=202)

    return {"status": status_handler, "generate": generate_handler, "publish": publish_handler}
```

In the dashboard plugin's route registration, add:

```python
        diary_handlers = _build_diary_trigger_handlers(self.app, broadcast=self._broadcast)
        app_router.router.add_get("/api/diary/status", diary_handlers["status"])
        app_router.router.add_post("/api/diary/generate", diary_handlers["generate"])
        app_router.router.add_post("/api/diary/publish", diary_handlers["publish"])
```

(`self._broadcast` is the existing dashboard WS broadcast helper. If named differently, use the actual name.)

- [ ] **Step 8.4: Run; expect PASS**

```
uv run pytest tests/test_diary_trigger_api.py -v
```

- [ ] **Step 8.5: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_plugin.py tests/test_diary_trigger_api.py
git commit -m "feat(dashboard): /api/diary trigger endpoints + WS job events"
```

---

## Task 9: Pause/resume — ConversationPlugin

**Files:**
- Modify: `src/reachy_claw/plugins/conversation_plugin.py`
- Test: extend `tests/test_plugin_rest_hooks.py`

The conversation plugin owns ASR worker, TTS playback queue, and LLM dispatch. On rest_start, set a flag `_paused=True` checked at every entry point that produces output (TTS) or starts new work (LLM). On rest_end, unset.

- [ ] **Step 9.1: Identify pause points**

```bash
grep -n "asr_final\|llm_end\|self.tts\|tts_play\|_running\|class ConversationPlugin" src/reachy_claw/plugins/conversation_plugin.py | head -20
```

Locate (a) the ASR loop entry, (b) the TTS playback dispatch, (c) the LLM call site.

- [ ] **Step 9.2: Add `_paused` state and gating**

In `ConversationPlugin.__init__`, add `self._paused = False`. Then in:

- The ASR-final handler: if `self._paused`, log and return early before sending to LLM.
- The TTS dispatch / playback: if `self._paused`, drop the utterance silently.
- The LLM call site: same — drop if paused.

Then implement the lifecycle hooks:

```python
    async def on_rest_start(self) -> None:
        self._paused = True
        logger.info("ConversationPlugin paused for rest")
        # Drain any in-flight TTS to silence the speaker quickly.
        # If a `_tts_queue` exists, clear it:
        try:
            while not self._tts_queue.empty():
                self._tts_queue.get_nowait()
        except (AttributeError, Exception):
            pass

    async def on_rest_end(self) -> None:
        self._paused = False
        logger.info("ConversationPlugin resumed")
```

(Adapt queue name to what's actually present. The pattern is: clear pending TTS work; let the running ASR loop see `_paused` and skip.)

- [ ] **Step 9.3: Subscribe to rest events in `start()`**

In `start()`, after existing event subscriptions, add:

```python
        bus.subscribe("rest_start", lambda _d: asyncio.create_task(self.on_rest_start()))
        bus.subscribe("rest_end", lambda _d: asyncio.create_task(self.on_rest_end()))
```

And matching `unsubscribe` in `stop()`.

- [ ] **Step 9.4: Write integration test**

Append to `tests/test_plugin_rest_hooks.py`:

```python
@pytest.mark.asyncio
async def test_conversation_pauses_on_rest_event(monkeypatch):
    # Construct minimal ConversationPlugin and assert _paused toggles.
    from reachy_claw.event_bus import EventBus
    from reachy_claw.plugins.conversation_plugin import ConversationPlugin

    class _StubApp:
        events = EventBus()
        config = type("C", (), {})()
        # Add any other minimal attrs the plugin's __init__ needs;
        # see the plugin source for required fields.

    app = _StubApp()
    plugin = ConversationPlugin(app)
    # Manually invoke the handler (bypass the event loop subscription).
    await plugin.on_rest_start()
    assert plugin._paused is True
    await plugin.on_rest_end()
    assert plugin._paused is False
```

If `ConversationPlugin.__init__` requires more app attributes than `_StubApp` provides, add them as no-op stubs (`config.<field> = None`, etc.) — the goal is to instantiate and toggle the flag, not exercise full ASR.

- [ ] **Step 9.5: Run; expect PASS**

```
uv run pytest tests/test_plugin_rest_hooks.py -v
```

- [ ] **Step 9.6: Commit**

```bash
git add src/reachy_claw/plugins/conversation_plugin.py tests/test_plugin_rest_hooks.py
git commit -m "feat(conversation): pause ASR/LLM/TTS on rest events"
```

---

## Task 10: Pause/resume — FaceTracker, VisionClient, Motion

**Files:**
- Modify: `src/reachy_claw/plugins/face_tracker_plugin.py`
- Modify: `src/reachy_claw/plugins/vision_client_plugin.py`
- Modify: `src/reachy_claw/plugins/motion_plugin.py`
- Test: extend `tests/test_plugin_rest_hooks.py`

For each plugin, add a `_paused` flag, gate the hot-loop body on it, and subscribe to `rest_start`/`rest_end`. Pattern is the same as Task 9.

- [ ] **Step 10.1: FaceTrackerPlugin**

In `face_tracker_plugin.py`:
- Add `self._paused = False` in `__init__`.
- In the per-frame loop, at the top: `if self._paused: await asyncio.sleep(0.5); continue`.
- Implement `on_rest_start` (set flag) / `on_rest_end` (clear flag).
- In `start()`, subscribe; in `stop()`, unsubscribe (same pattern as Task 9.3).

- [ ] **Step 10.2: VisionClientPlugin**

Same pattern. The hot loop is the ZMQ recv loop — gate with `_paused`.

- [ ] **Step 10.3: MotionPlugin**

Same pattern. The hot path is the head-target update loop. When paused, skip the update (head stays where it is).

- [ ] **Step 10.4: Add tests for each**

Append to `tests/test_plugin_rest_hooks.py` (one test per plugin):

```python
@pytest.mark.asyncio
async def test_face_tracker_pauses():
    from reachy_claw.event_bus import EventBus
    from reachy_claw.plugins.face_tracker_plugin import FaceTrackerPlugin

    class _StubApp:
        events = EventBus()
        config = type("C", (), {})()

    p = FaceTrackerPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False


@pytest.mark.asyncio
async def test_vision_client_pauses():
    from reachy_claw.event_bus import EventBus
    from reachy_claw.plugins.vision_client_plugin import VisionClientPlugin

    class _StubApp:
        events = EventBus()
        config = type("C", (), {})()

    p = VisionClientPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False


@pytest.mark.asyncio
async def test_motion_pauses():
    from reachy_claw.event_bus import EventBus
    from reachy_claw.plugins.motion_plugin import MotionPlugin

    class _StubApp:
        events = EventBus()
        config = type("C", (), {})()

    p = MotionPlugin(_StubApp())
    await p.on_rest_start(); assert p._paused is True
    await p.on_rest_end(); assert p._paused is False
```

If any plugin's `__init__` needs additional stubs to instantiate, add them inline. The goal is the flag round-trip; full plugin behavior is exercised by existing tests.

- [ ] **Step 10.5: Run; expect PASS**

```
uv run pytest tests/test_plugin_rest_hooks.py -v
```

- [ ] **Step 10.6: Commit**

```bash
git add src/reachy_claw/plugins/face_tracker_plugin.py src/reachy_claw/plugins/vision_client_plugin.py src/reachy_claw/plugins/motion_plugin.py tests/test_plugin_rest_hooks.py
git commit -m "feat(plugins): pause/resume vision + motion on rest events"
```

---

## Task 11: Dashboard SETTINGS tab — UI

**Files:**
- Modify: `src/reachy_claw/plugins/dashboard_static/index.html`
- Modify: `src/reachy_claw/plugins/dashboard_static/app.js`
- Create: `src/reachy_claw/plugins/dashboard_static/settings.js`
- Create: `src/reachy_claw/plugins/dashboard_static/settings.css`

- [ ] **Step 11.1: Read existing tab structure**

```bash
grep -n "tab\|LIVE\|DIARY" src/reachy_claw/plugins/dashboard_static/index.html | head
grep -n "tab\|setActiveTab" src/reachy_claw/plugins/dashboard_static/app.js | head
```

Note the existing tab markup pattern + click handler.

- [ ] **Step 11.2: Add SETTINGS tab to index.html**

In `index.html`, find the tab nav block (where LIVE / DIARY are defined). Add a third button:

```html
<button class="tab-btn" data-tab="settings">SETTINGS</button>
```

Find the tab container area and add:

```html
<div id="settings-page" class="tab-page" style="display:none">
  <div id="settings-root"></div>
</div>
```

Also add `<link rel="stylesheet" href="settings.css">` and `<script type="module" src="settings.js"></script>` near the existing CSS/JS includes.

- [ ] **Step 11.3: Wire tab switching in app.js**

Find the existing tab switch logic (likely a `setActiveTab(name)` function or a click handler). Extend it so `data-tab="settings"` shows `#settings-page` and triggers `window.renderSettings(document.getElementById('settings-root'))` once on first activation. Pattern (adapt to existing code):

```js
// Inside the tab click handler:
if (tab === 'settings' && !window._settingsRendered) {
  window.renderSettings(document.getElementById('settings-root'));
  window._settingsRendered = true;
}
```

- [ ] **Step 11.4: Implement settings.js with section registry**

```js
// src/reachy_claw/plugins/dashboard_static/settings.js
"use strict";

const SECTIONS = [];

function registerSection(section) { SECTIONS.push(section); }

async function fetchSettings(ns) {
  const r = await fetch(`/api/settings/${ns}`);
  if (!r.ok) throw new Error(`GET /api/settings/${ns} → ${r.status}`);
  return r.json();
}

async function putSettings(ns, body) {
  const r = await fetch(`/api/settings/${ns}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `PUT /api/settings/${ns} → ${r.status}`);
  }
  return r.json();
}

// ── Section: Rest Window ──────────────────────────────────────────
registerSection({
  id: "rest-window",
  title: "休整时段 / Rest Window",
  async render(div) {
    const cur = await fetchSettings("rest");
    div.innerHTML = `
      <label>开始 <input type="time" data-k="window_start" value="${cur.window_start}"></label>
      <label>结束 <input type="time" data-k="window_end" value="${cur.window_end}"></label>
      <label>时区 <input type="text" data-k="timezone" value="${cur.timezone}"></label>
      <label><input type="checkbox" data-k="enabled" ${cur.enabled ? "checked" : ""}> 启用</label>
      <button class="btn-save">保存</button>
      <span class="msg"></span>
    `;
    div.querySelector(".btn-save").addEventListener("click", async () => {
      const body = {};
      div.querySelectorAll("[data-k]").forEach(el => {
        const k = el.dataset.k;
        body[k] = el.type === "checkbox" ? el.checked : el.value;
      });
      const msg = div.querySelector(".msg");
      try {
        await putSettings("rest", body);
        msg.textContent = "✓ 已保存";
        msg.className = "msg ok";
      } catch (e) {
        msg.textContent = "✗ " + e.message;
        msg.className = "msg err";
      }
    });
  },
});

// ── Section: Diary Publishing ─────────────────────────────────────
registerSection({
  id: "diary-publishing",
  title: "日记发布 / Diary Publishing",
  async render(div) {
    const cur = await fetchSettings("diary");
    div.innerHTML = `
      <label><input type="checkbox" data-k="auto_publish" ${cur.auto_publish ? "checked" : ""}> 自动每日发布</label>
      <label><input type="checkbox" data-k="privacy_linter" ${cur.privacy_linter ? "checked" : ""}> 隐私 linter</label>
      <label>站点 repo <input type="text" data-k="site_repo_url" value="${cur.site_repo_url}" placeholder="git@github.com:org/site.git"></label>
      <label>路径 <input type="text" data-k="site_diary_path" value="${cur.site_diary_path}"></label>
      <label>分支 <input type="text" data-k="site_branch" value="${cur.site_branch}"></label>
      <button class="btn-save">保存</button>
      <span class="msg"></span>
      <h4>历史</h4>
      <table class="diary-history"><tbody></tbody></table>
    `;
    div.querySelector(".btn-save").addEventListener("click", async () => {
      const body = {};
      div.querySelectorAll("[data-k]").forEach(el => {
        const k = el.dataset.k;
        body[k] = el.type === "checkbox" ? el.checked : el.value;
      });
      const msg = div.querySelector(".msg");
      try {
        await putSettings("diary", body);
        msg.textContent = "✓ 已保存";
        msg.className = "msg ok";
      } catch (e) {
        msg.textContent = "✗ " + e.message;
        msg.className = "msg err";
      }
    });
    await renderDiaryHistory(div.querySelector(".diary-history tbody"));
  },
});

async function renderDiaryHistory(tbody) {
  const r = await fetch("/api/diary/status");
  const { dates } = await r.json();
  tbody.innerHTML = "";
  for (const d of dates) {
    const tr = document.createElement("tr");
    const status = d.published ? "✓" : (d.generated ? "⚠" : "✗");
    const action = d.published
      ? { label: "重新生成", op: () => trigger("generate", d.date, true) }
      : (d.generated
          ? { label: "发布", op: () => trigger("publish", d.date, false) }
          : { label: "生成+发布", op: async () => { await trigger("generate", d.date, false); await trigger("publish", d.date, false); } });
    tr.innerHTML = `<td>${d.date}</td><td>${status}</td><td><button>${action.label}</button></td>`;
    tr.querySelector("button").addEventListener("click", async () => {
      tr.querySelector("button").disabled = true;
      try { await action.op(); } catch (e) { alert(e.message); }
      tr.querySelector("button").disabled = false;
      // Refresh row.
      await new Promise(r => setTimeout(r, 800));
      await renderDiaryHistory(tbody);
    });
    tbody.appendChild(tr);
  }
}

async function trigger(kind, date, force) {
  const r = await fetch(`/api/diary/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, force }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `POST /api/diary/${kind} → ${r.status}`);
  }
  return r.json();
}

window.renderSettings = function(root) {
  root.innerHTML = "<h2>SETTINGS</h2>";
  for (const s of SECTIONS) {
    const wrap = document.createElement("section");
    wrap.className = "settings-section";
    wrap.innerHTML = `<h3>${s.title}</h3><div class="content"></div>`;
    root.appendChild(wrap);
    s.render(wrap.querySelector(".content")).catch(e => {
      wrap.querySelector(".content").innerHTML = `<span class="msg err">加载失败：${e.message}</span>`;
    });
  }
};
```

- [ ] **Step 11.5: Implement settings.css**

```css
/* src/reachy_claw/plugins/dashboard_static/settings.css */
.settings-section {
  background: rgba(0,0,0,0.45);
  border: 1px solid #2a2a2a;
  border-radius: 6px;
  padding: 16px;
  margin: 12px 0;
}
.settings-section h3 {
  margin: 0 0 12px;
  color: #A3E635;
  font-family: 'Outfit', sans-serif;
}
.settings-section label {
  display: block;
  margin: 6px 0;
  font-size: 14px;
}
.settings-section input[type=text],
.settings-section input[type=time] {
  background: #111; color: #ddd; border: 1px solid #333;
  padding: 4px 8px; border-radius: 4px; margin-left: 6px;
}
.btn-save {
  background: #A3E635; color: #000; border: none;
  padding: 6px 14px; border-radius: 4px; cursor: pointer;
  margin-top: 10px;
}
.msg.ok { color: #A3E635; margin-left: 10px; }
.msg.err { color: #f87171; margin-left: 10px; }
.diary-history { width: 100%; border-collapse: collapse; margin-top: 10px; }
.diary-history td {
  padding: 6px 10px; border-bottom: 1px solid #2a2a2a;
  font-family: monospace;
}
.diary-history button {
  background: transparent; color: #A3E635;
  border: 1px solid #A3E635; padding: 4px 10px;
  cursor: pointer; border-radius: 4px;
}
.diary-history button:disabled { opacity: 0.5; cursor: wait; }
```

- [ ] **Step 11.6: Manual smoke check**

There are no automated tests for static UI in this project — verify manually:

```bash
# Start the dashboard locally (whatever the project's dev command is — likely)
uv run reachy-claw --no-robot      # or check README.md / scripts/run_*.sh
# Open http://localhost:<dashboard_port>/ in a browser, click SETTINGS tab.
# Confirm: rest section loads current values, save button works, diary history table populates.
```

If you can't run it locally, at minimum ensure:

```bash
# JS file parses (no syntax errors)
node -c src/reachy_claw/plugins/dashboard_static/settings.js
# HTML/CSS files exist and reference each other correctly
ls -la src/reachy_claw/plugins/dashboard_static/settings.{js,css}
grep -c "settings.js\|settings.css" src/reachy_claw/plugins/dashboard_static/index.html  # expect 2
```

- [ ] **Step 11.7: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_static/settings.js src/reachy_claw/plugins/dashboard_static/settings.css src/reachy_claw/plugins/dashboard_static/index.html src/reachy_claw/plugins/dashboard_static/app.js
git commit -m "feat(dashboard): SETTINGS tab with rest + diary sections + history"
```

---

## Task 12: Final integration + manual end-to-end

- [ ] **Step 12.1: Run the full test suite**

```
uv run pytest -x --ignore=tests/test_vision_client_plugin.py
```

Expected: green (the zmq-related skip stays as before).

- [ ] **Step 12.2: Manual end-to-end on a developer machine**

This step is dispatched to a remote agent or run manually — not from the main dev thread:

```
1. Set rest window to "now+2min" through "now+7min" via the dashboard SETTINGS tab.
2. Watch logs: at the start time, expect "Entering rest window" + a `rest_start` event.
3. Confirm ASR stops responding (speak — robot should ignore).
4. Confirm `housekeeping_task_start` for diary_generate_and_publish appears in logs.
5. At the end time, expect `rest_end` + ASR resumes.
6. From SETTINGS → 历史, click [生成+发布] for an old date that's never been processed.
   Confirm a row's status updates to ✓ after the WS job event arrives.
7. Click [重新生成] for a published date. Confirm a new commit appears in the site repo.

EVIDENCE: dashboard screenshot of SETTINGS tab; backend log excerpt showing rest_start/end + housekeeping events; site repo `git log` showing the manual-publish commit.
```

---

## Self-Review

**Spec coverage:**
- Rest window scheduling (Goal 1) → Task 4 ✓
- Plugin pause/resume (Goal 1) → Tasks 3, 9, 10 ✓
- Housekeeping registry + diary task → Task 5 ✓
- Settings persistence via runtime-overrides → Tasks 1, 2, 7 ✓
- Settings API → Task 7 ✓
- Diary trigger API + WS events → Task 8 ✓
- Dashboard SETTINGS tab + section registry + history UI → Task 11 ✓
- App wiring (RestPlugin registered, task registered) → Task 6 ✓
- Manual E2E → Task 12 ✓

**Placeholder scan:** No "TBD" / "implement later" / "add appropriate error handling" remains.

**Type/name consistency:**
- `app.config.rest_window_start` (Task 1) is read in `_should_rest_now` (Task 4) and `_build_settings_handlers` (Task 7) — matches.
- `events.emit` not `events.publish` — verified against EventBus actual API.
- `app.db` (assumed available from prior diary feature) used in Task 8's status handler.
- Plugin `_paused` flag name used identically in Tasks 9 and 10.
- `save_runtime_overrides(config, fields)` signature used in Task 7 matches `config.py`.

No issues found.
