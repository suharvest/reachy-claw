# HA Sensor Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull selected Home Assistant sensor history into the daily diary at generation time, configured entirely from the dashboard.

**Architecture:** Stateless `ha_client.py` (no plugin, no background task). `generate_diary.py` calls `ha_client.get_history()` at run time and bundles results into the LLM context. Dashboard adds a SETTINGS modal tab for connection + entity selection; values persist to `runtime-overrides.yaml` like all other settings.

**Tech Stack:** Python 3.12, `httpx` async client, aiohttp dashboard, vanilla JS frontend. Spec: `docs/superpowers/specs/2026-04-27-ha-sensor-sync-design.md`. Branch: `feat/ha-sensor-sync` from `feat/rest-and-settings-panel`.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/reachy_claw/ha_client.py` | HA REST client: probe / list_states / get_history. Stateless async functions + `HAError` hierarchy. | Create |
| `tests/test_ha_client.py` | Unit tests using `httpx.MockTransport`. | Create |
| `src/reachy_claw/config.py` | Add `ha_url`, `ha_token`, `ha_entities` fields + `KEY_MAP` entries. | Modify |
| `src/reachy_claw/settings_schema.py` | Add `_validate_str_list` validator + 3 SettingSpec entries for `ha.*`. | Modify |
| `tests/test_settings_schema.py` | Extend with HA + list-validator coverage. | Modify |
| `src/reachy_claw/plugins/dashboard_plugin.py` | New `_build_ha_handlers(app)` + register routes. | Modify |
| `tests/test_dashboard_ha_api.py` | API tests for `/api/ha/test`, `/api/ha/entities`, `/api/settings/ha`. | Create |
| `scripts/generate_diary.py` | Append HA paragraph to `SYSTEM_PROMPT`; fetch HA history before LLM call; bundle into events dict. | Modify |
| `tests/test_generate_diary.py` | Extend: HA injected, HA failure tolerated, HA unconfigured. | Modify |
| `src/reachy_claw/plugins/dashboard_static/ha_settings.js` | `bindHASettings()`, `refreshHAEntities()`. | Create |
| `src/reachy_claw/plugins/dashboard_static/index.html` | Add `HA Sensors` tab; chips strip in Diary tab. | Modify |
| `src/reachy_claw/plugins/dashboard_static/app.js` | Wire HA tab activation. | Modify |
| `src/reachy_claw/plugins/dashboard_static/settings.css` | Minor styles for entity tree, password show/hide. | Modify |

---

## Branch Setup

- [ ] **Step 0.1: Confirm starting branch and create feature branch**

```bash
git rev-parse --abbrev-ref HEAD
# Expected: feat/rest-and-settings-panel
git status --short
# Expected: working tree clean (or only pre-existing untracked files
# from vision-hailo work — see HANDOFF-2026-04-27.md "Outstanding").
git checkout -b feat/ha-sensor-sync
```

---

## Task 1 (Batch A): `ha_client.py`

**Files:**
- Create: `src/reachy_claw/ha_client.py`
- Create: `tests/test_ha_client.py`

- [ ] **Step 1.1: Write failing test for `probe` happy path**

`tests/test_ha_client.py`:
```python
"""Tests for the stateless Home Assistant REST client."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from reachy_claw import ha_client


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_probe_ok(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/"
        assert req.headers["Authorization"] == "Bearer abc"
        return httpx.Response(200, json={"message": "API running."})

    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "abc")
    assert out["ok"] is True
    assert out["status"] == 200
    assert "running" in out["message"].lower()
```

- [ ] **Step 1.2: Run test to verify failure**

```bash
uv run pytest tests/test_ha_client.py::test_probe_ok -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'reachy_claw.ha_client'`.

- [ ] **Step 1.3: Implement `ha_client.py` skeleton with `probe`**

`src/reachy_claw/ha_client.py`:
```python
"""Stateless async Home Assistant REST client.

Used by:
  - dashboard_plugin: /api/ha/test (probe), /api/ha/entities (list_states)
  - scripts/generate_diary.py: get_history (per-entity day window)

Pure functions — no plugin, no caching, no background tasks. Each call
opens a fresh httpx.AsyncClient via `_transport_factory()` so tests can
inject a MockTransport.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HAError(Exception):
    """Base for HA client errors."""


class HAUnreachable(HAError):
    """Network failure: timeout, DNS, connection refused, TLS error."""


class HAUnauthorized(HAError):
    """HTTP 401 — bad or missing token."""


class HABadResponse(HAError):
    """Non-2xx other than 401, or response body did not parse."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HA {status}: {message}")
        self.status = status
        self.message = message


def _normalise_url(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"HA URL must start with http:// or https://, got {url!r}")
    return url.rstrip("/")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _transport_factory():
    """Override in tests with a MockTransport."""
    return None  # None → httpx uses its real transport.


def _client(timeout: float) -> httpx.AsyncClient:
    transport = _transport_factory()
    if transport is None:
        return httpx.AsyncClient(timeout=timeout)
    return httpx.AsyncClient(timeout=timeout, transport=transport)


async def probe(url: str, token: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET <url>/api/. Never raises; encodes errors in the result dict."""
    try:
        base = _normalise_url(url)
    except ValueError as e:
        return {"ok": False, "status": 0, "message": str(e)}
    async with _client(timeout) as client:
        try:
            r = await client.get(f"{base}/api/", headers=_headers(token))
        except httpx.HTTPError as e:
            return {"ok": False, "status": 0, "message": f"{type(e).__name__}: {e}"}
    if r.status_code == 200:
        msg = ""
        try:
            msg = r.json().get("message", "")
        except Exception:
            msg = r.text[:200]
        return {"ok": True, "status": 200, "message": msg}
    if r.status_code == 401:
        return {"ok": False, "status": 401, "message": "Unauthorized — token rejected"}
    return {"ok": False, "status": r.status_code, "message": r.text[:200]}
```

- [ ] **Step 1.4: Run test to verify pass**

```bash
uv run pytest tests/test_ha_client.py::test_probe_ok -v
```
Expected: PASS.

- [ ] **Step 1.5: Add probe error-path tests**

Append to `tests/test_ha_client.py`:
```python
@pytest.mark.asyncio
async def test_probe_unauthorized(monkeypatch):
    def handler(req): return httpx.Response(401, text="invalid")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "wrong")
    assert out == {"ok": False, "status": 401, "message": "Unauthorized — token rejected"}


@pytest.mark.asyncio
async def test_probe_connection_error(monkeypatch):
    def handler(req): raise httpx.ConnectError("refused")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.probe("http://ha.local:8123", "x")
    assert out["ok"] is False
    assert out["status"] == 0
    assert "ConnectError" in out["message"]


@pytest.mark.asyncio
async def test_probe_bad_url():
    out = await ha_client.probe("ha.local", "x")
    assert out["ok"] is False
    assert out["status"] == 0
    assert "http://" in out["message"]
```

- [ ] **Step 1.6: Run all probe tests**

```bash
uv run pytest tests/test_ha_client.py -v
```
Expected: 4 PASSED.

- [ ] **Step 1.7: Add `list_states` test**

Append to `tests/test_ha_client.py`:
```python
@pytest.mark.asyncio
async def test_list_states_returns_list(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/states"
        assert req.headers["Authorization"] == "Bearer t"
        return httpx.Response(200, json=[
            {"entity_id": "weather.home", "state": "sunny",
             "attributes": {"friendly_name": "Home", "temperature": 22.5}},
            {"entity_id": "sensor.bedroom_temp", "state": "21.3",
             "attributes": {"unit_of_measurement": "°C"}},
        ])
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.list_states("http://ha.local:8123/", "t")
    assert len(out) == 2
    assert out[0]["entity_id"] == "weather.home"


@pytest.mark.asyncio
async def test_list_states_unauthorized_raises(monkeypatch):
    def handler(req): return httpx.Response(401, text="bad")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnauthorized):
        await ha_client.list_states("http://ha.local:8123", "x")


@pytest.mark.asyncio
async def test_list_states_timeout_raises(monkeypatch):
    def handler(req): raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnreachable):
        await ha_client.list_states("http://ha.local:8123", "x")
```

- [ ] **Step 1.8: Run; expect failure (function missing)**

```bash
uv run pytest tests/test_ha_client.py::test_list_states_returns_list -v
```
Expected: FAIL — `AttributeError: module ... has no attribute 'list_states'`.

- [ ] **Step 1.9: Implement `list_states`**

Append to `src/reachy_claw/ha_client.py`:
```python
async def list_states(url: str, token: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """GET <url>/api/states. Returns full state list. Raises HAError on failure."""
    base = _normalise_url(url)
    async with _client(timeout) as client:
        try:
            r = await client.get(f"{base}/api/states", headers=_headers(token))
        except httpx.HTTPError as e:
            raise HAUnreachable(str(e)) from e
    if r.status_code == 401:
        raise HAUnauthorized("token rejected")
    if r.status_code != 200:
        raise HABadResponse(r.status_code, r.text[:200])
    try:
        body = r.json()
    except Exception as e:
        raise HABadResponse(r.status_code, f"invalid JSON: {e}") from e
    if not isinstance(body, list):
        raise HABadResponse(r.status_code, "expected JSON array")
    return body
```

- [ ] **Step 1.10: Run list_states tests**

```bash
uv run pytest tests/test_ha_client.py -v
```
Expected: 7 PASSED.

- [ ] **Step 1.11: Add `get_history` tests**

Append to `tests/test_ha_client.py`:
```python
@pytest.mark.asyncio
async def test_get_history_builds_query(monkeypatch):
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["params"] = dict(req.url.params)
        return httpx.Response(200, json=[
            [
                {"entity_id": "weather.home", "state": "sunny",
                 "attributes": {"temperature": 22.0},
                 "last_updated": "2026-04-27T10:00:00+00:00"},
                {"entity_id": "weather.home", "state": "cloudy",
                 "attributes": {"temperature": 21.0},
                 "last_updated": "2026-04-27T14:00:00+00:00"},
            ],
            [
                {"entity_id": "sensor.temp", "state": "20",
                 "attributes": {},
                 "last_updated": "2026-04-27T08:00:00+00:00"},
            ],
        ])

    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    start = datetime(2026, 4, 27, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)
    out = await ha_client.get_history(
        "http://ha.local:8123", "t", ["weather.home", "sensor.temp"], start, end
    )
    assert captured["path"].startswith("/api/history/period/")
    assert "filter_entity_id" in captured["params"]
    assert "weather.home" in captured["params"]["filter_entity_id"]
    assert "sensor.temp" in captured["params"]["filter_entity_id"]
    assert "minimal_response" in captured["params"]
    assert out["weather.home"][0]["state"] == "sunny"
    assert out["weather.home"][1]["state"] == "cloudy"
    assert out["sensor.temp"][0]["state"] == "20"
    assert "ts" in out["weather.home"][0]
    assert "attributes" in out["weather.home"][0]


@pytest.mark.asyncio
async def test_get_history_empty_entity_list_returns_empty(monkeypatch):
    def handler(req): raise AssertionError("should not be called")
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    out = await ha_client.get_history(
        "http://ha.local:8123", "t", [],
        datetime(2026, 4, 27, tzinfo=timezone.utc),
        datetime(2026, 4, 28, tzinfo=timezone.utc),
    )
    assert out == {}


@pytest.mark.asyncio
async def test_get_history_unauthorized(monkeypatch):
    def handler(req): return httpx.Response(401)
    monkeypatch.setattr(ha_client, "_transport_factory", lambda: _mock_transport(handler))
    with pytest.raises(ha_client.HAUnauthorized):
        await ha_client.get_history(
            "http://ha.local:8123", "x", ["sensor.a"],
            datetime(2026, 4, 27, tzinfo=timezone.utc),
            datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
```

- [ ] **Step 1.12: Run; expect failure**

```bash
uv run pytest tests/test_ha_client.py::test_get_history_builds_query -v
```
Expected: FAIL — `AttributeError: module ... has no attribute 'get_history'`.

- [ ] **Step 1.13: Implement `get_history`**

Append to `src/reachy_claw/ha_client.py`:
```python
async def get_history(
    url: str,
    token: str,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    timeout: float = 30.0,
) -> dict[str, list[dict[str, Any]]]:
    """GET <url>/api/history/period/<start>?filter_entity_id=...&end_time=...&minimal_response.

    Returns {entity_id: [{"ts": iso8601, "state": str, "attributes": dict}, ...]}.
    Raises HAError subclasses on failure. Empty entity list returns {} without
    making a request.
    """
    if not entity_ids:
        return {}
    base = _normalise_url(url)
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    params = {
        "filter_entity_id": ",".join(entity_ids),
        "end_time": end_iso,
        "minimal_response": "",
    }
    async with _client(timeout) as client:
        try:
            r = await client.get(
                f"{base}/api/history/period/{start_iso}",
                headers=_headers(token),
                params=params,
            )
        except httpx.HTTPError as e:
            raise HAUnreachable(str(e)) from e
    if r.status_code == 401:
        raise HAUnauthorized("token rejected")
    if r.status_code != 200:
        raise HABadResponse(r.status_code, r.text[:200])
    try:
        body = r.json()
    except Exception as e:
        raise HABadResponse(r.status_code, f"invalid JSON: {e}") from e
    if not isinstance(body, list):
        raise HABadResponse(r.status_code, "expected JSON array of arrays")

    out: dict[str, list[dict[str, Any]]] = {eid: [] for eid in entity_ids}
    for entity_series in body:
        if not isinstance(entity_series, list) or not entity_series:
            continue
        for row in entity_series:
            eid = row.get("entity_id")
            if not eid:
                # minimal_response strips entity_id from non-first rows; reuse last seen
                eid = entity_series[0].get("entity_id")
            if not eid or eid not in out:
                continue
            out[eid].append({
                "ts": row.get("last_updated") or row.get("last_changed") or "",
                "state": row.get("state", ""),
                "attributes": row.get("attributes", {}) or {},
            })
    return out
```

- [ ] **Step 1.14: Run all ha_client tests**

```bash
uv run pytest tests/test_ha_client.py -v
```
Expected: 10 PASSED.

- [ ] **Step 1.15: Commit**

```bash
git add src/reachy_claw/ha_client.py tests/test_ha_client.py
git commit -m "feat(ha): stateless HA REST client (probe/list_states/get_history)"
```

---

## Task 2 (Batch B): Config + settings_schema

**Files:**
- Modify: `src/reachy_claw/config.py` (Config fields + KEY_MAP)
- Modify: `src/reachy_claw/settings_schema.py`
- Modify: `tests/test_settings_schema.py`

- [ ] **Step 2.1: Write failing test for new SettingSpec entries and list validator**

Append to `tests/test_settings_schema.py`:
```python
def test_ha_namespace_registered():
    assert "ha" in NAMESPACES
    assert set(keys_for_namespace("ha")) == {"url", "token", "entities"}


def test_ha_url_validator():
    validate("ha.url", "")  # empty allowed
    validate("ha.url", "http://homeassistant.local:8123")
    validate("ha.url", "https://ha.example.com")
    with pytest.raises(ValueError):
        validate("ha.url", "ha.local")  # missing scheme
    with pytest.raises(ValueError):
        validate("ha.url", 123)  # not a string


def test_ha_token_validator():
    validate("ha.token", "")
    validate("ha.token", "long-token")
    with pytest.raises(ValueError):
        validate("ha.token", None)


def test_ha_entities_validator():
    validate("ha.entities", [])
    validate("ha.entities", ["weather.home", "sensor.bedroom_temp"])
    with pytest.raises(ValueError):
        validate("ha.entities", "weather.home")  # not a list
    with pytest.raises(ValueError):
        validate("ha.entities", ["weather.home", 1])  # non-str element
    with pytest.raises(ValueError):
        validate("ha.entities", ["BadFormat"])  # missing dot
    with pytest.raises(ValueError):
        validate("ha.entities", ["weather.Home Office"])  # bad chars
```

- [ ] **Step 2.2: Run; expect failure**

```bash
uv run pytest tests/test_settings_schema.py -v
```
Expected: NEW tests fail (`ha` not in `NAMESPACES`, etc.); old tests pass.

- [ ] **Step 2.3: Add HA validators and specs**

Edit `src/reachy_claw/settings_schema.py`:

1. After `_validate_tz`, add:
```python
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
```

2. Append to `_SPECS` (before the closing `]`):
```python
    SettingSpec("ha", "url", "ha_url", str, _validate_ha_url),
    SettingSpec("ha", "token", "ha_token", str),
    SettingSpec("ha", "entities", "ha_entities", list, _validate_entity_id_list),
```

- [ ] **Step 2.4: Run schema tests**

```bash
uv run pytest tests/test_settings_schema.py -v
```
Expected: all pass.

- [ ] **Step 2.5: Update existing namespace test**

Replace `test_namespaces_are_rest_and_diary` in `tests/test_settings_schema.py`:
```python
def test_namespaces_includes_rest_diary_ha():
    assert {"rest", "diary", "ha"}.issubset(set(NAMESPACES))
```

Update `test_registry_has_expected_keys` to include the three new keys:
```python
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
        "ha.url",
        "ha.token",
        "ha.entities",
    }
    assert set(REGISTRY) == expected
```

- [ ] **Step 2.6: Run schema suite**

```bash
uv run pytest tests/test_settings_schema.py -v
```
Expected: all pass.

- [ ] **Step 2.7: Add Config fields**

Edit `src/reachy_claw/config.py`. After the existing `# ── Diary publishing ─` block (around line 88), add:
```python
    # ── Home Assistant integration ──────────────────────────────
    ha_url: str = ""              # e.g. "http://homeassistant.local:8123"
    ha_token: str = ""            # long-lived access token
    ha_entities: list[str] = field(default_factory=list)  # entity_ids to include
```

- [ ] **Step 2.8: Add KEY_MAP entries**

In `src/reachy_claw/config.py`, locate the `KEY_MAP` dict and append (any position is fine — pick alongside `("diary", ...)` block):
```python
    ("ha", "url"): "ha_url",
    ("ha", "token"): "ha_token",
    ("ha", "entities"): "ha_entities",
```

- [ ] **Step 2.9: Write failing config-load test**

Append to `tests/test_config.py` (or create if absent):
```python
def test_config_has_ha_fields():
    from reachy_claw.config import Config
    c = Config()
    assert c.ha_url == ""
    assert c.ha_token == ""
    assert c.ha_entities == []


def test_runtime_overrides_load_ha_entities(tmp_path, monkeypatch):
    from reachy_claw.config import load_config, save_runtime_overrides
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".reachy-claw"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("gateway_host: 127.0.0.1\n")
    cfg = load_config()
    cfg.ha_url = "http://ha.local:8123"
    cfg.ha_entities = ["weather.home", "sensor.temp"]
    save_runtime_overrides(cfg, ["ha_url", "ha_entities"])
    cfg2 = load_config()
    assert cfg2.ha_url == "http://ha.local:8123"
    assert cfg2.ha_entities == ["weather.home", "sensor.temp"]
```

- [ ] **Step 2.10: Run**

```bash
uv run pytest tests/test_config.py -v -k ha
```
Expected: PASS (if fail due to test infra mismatch, adapt to existing patterns in `test_config.py` — e.g. monkeypatching `CONFIG_SEARCH_PATHS`).

- [ ] **Step 2.11: Commit**

```bash
git add src/reachy_claw/config.py src/reachy_claw/settings_schema.py \
        tests/test_settings_schema.py tests/test_config.py
git commit -m "feat(config): ha_url/ha_token/ha_entities + settings schema list type"
```

---

## Task 3 (Batch C): Dashboard API endpoints

**Files:**
- Modify: `src/reachy_claw/plugins/dashboard_plugin.py`
- Create: `tests/test_dashboard_ha_api.py`

- [ ] **Step 3.1: Write failing test for `/api/ha/test`**

`tests/test_dashboard_ha_api.py`:
```python
"""Tests for dashboard HA endpoints: /api/ha/test, /api/ha/entities,
PUT /api/settings/ha."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from reachy_claw.config import Config
from reachy_claw.plugins.dashboard_plugin import _build_ha_handlers


class _FakeApp:
    def __init__(self, **cfg):
        self.config = Config(**cfg)


@pytest.fixture
def aio_app():
    app = web.Application()
    fake = _FakeApp(ha_url="http://ha.local:8123", ha_token="tok",
                    ha_entities=["weather.home"])
    handlers = _build_ha_handlers(fake)
    app.router.add_post("/api/ha/test", handlers["test"])
    app.router.add_get("/api/ha/entities", handlers["entities"])
    return app, fake


async def test_ha_test_uses_body_overrides(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.probe",
               new=AsyncMock(return_value={"ok": True, "status": 200, "message": "ok"})) as mp:
        resp = await client.post("/api/ha/test",
                                 json={"url": "http://other:8123", "token": "x"})
    assert resp.status == 200
    body = await resp.json()
    assert body == {"ok": True, "status": 200, "message": "ok"}
    mp.assert_awaited_once_with("http://other:8123", "x")


async def test_ha_test_falls_back_to_config(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.probe",
               new=AsyncMock(return_value={"ok": False, "status": 401, "message": "bad"})) as mp:
        resp = await client.post("/api/ha/test", json={})
    assert resp.status == 200
    mp.assert_awaited_once_with("http://ha.local:8123", "tok")
```

- [ ] **Step 3.2: Run; expect import failure**

```bash
uv run pytest tests/test_dashboard_ha_api.py -v
```
Expected: FAIL — `ImportError: cannot import name '_build_ha_handlers'`.

- [ ] **Step 3.3: Add `_build_ha_handlers` to dashboard_plugin.py**

In `src/reachy_claw/plugins/dashboard_plugin.py`, near the other `_build_*_handlers` functions (after `_build_rest_status_handlers` is fine), add:
```python
from .. import ha_client  # at top of file with other imports


def _build_ha_handlers(app):
    """HA Sensors API: connection probe + entity listing.

    Settings PUT/GET for ha.* is handled by the generic
    _build_settings_handlers via the namespace dispatch.
    """
    from aiohttp import web

    async def test_handler(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        url = body.get("url") if isinstance(body, dict) else None
        token = body.get("token") if isinstance(body, dict) else None
        url = url or app.config.ha_url
        token = token or app.config.ha_token
        if not url or not token:
            return web.json_response(
                {"ok": False, "status": 0, "message": "ha_url or ha_token not set"}
            )
        result = await ha_client.probe(url, token)
        return web.json_response(result)

    async def entities_handler(request):
        url = app.config.ha_url
        token = app.config.ha_token
        if not url or not token:
            return web.json_response(
                {"error": "ha_url or ha_token not configured"}, status=400
            )
        try:
            states = await ha_client.list_states(url, token)
        except ha_client.HAUnauthorized as e:
            return web.json_response({"error": f"unauthorized: {e}"}, status=502)
        except ha_client.HAUnreachable as e:
            return web.json_response({"error": f"unreachable: {e}"}, status=502)
        except ha_client.HABadResponse as e:
            return web.json_response({"error": f"bad response: {e}"}, status=502)

        groups: dict[str, list[dict]] = {}
        for s in states:
            eid = s.get("entity_id", "")
            if "." not in eid:
                continue
            domain = eid.split(".", 1)[0]
            attrs = s.get("attributes") or {}
            groups.setdefault(domain, []).append({
                "entity_id": eid,
                "state": s.get("state", ""),
                "friendly_name": attrs.get("friendly_name", ""),
            })
        out = []
        for domain in sorted(groups):
            entities = sorted(groups[domain], key=lambda e: e["entity_id"])
            out.append({"domain": domain, "count": len(entities), "entities": entities})
        return web.json_response({"groups": out})

    return {"test": test_handler, "entities": entities_handler}
```

- [ ] **Step 3.4: Register routes in `start()`**

In `DashboardPlugin.start()` (after the rest handlers registration, around line 250), add:
```python
        # HA API
        ha_handlers = _build_ha_handlers(self.app)
        app.router.add_post("/api/ha/test", ha_handlers["test"])
        app.router.add_get("/api/ha/entities", ha_handlers["entities"])
```

- [ ] **Step 3.5: Run probe tests**

```bash
uv run pytest tests/test_dashboard_ha_api.py -v
```
Expected: 2 PASSED.

- [ ] **Step 3.6: Add `entities` endpoint tests**

Append to `tests/test_dashboard_ha_api.py`:
```python
async def test_entities_unconfigured_returns_400(aiohttp_client):
    app = web.Application()
    fake = _FakeApp(ha_url="", ha_token="")
    handlers = _build_ha_handlers(fake)
    app.router.add_get("/api/ha/entities", handlers["entities"])
    client = await aiohttp_client(app)
    resp = await client.get("/api/ha/entities")
    assert resp.status == 400
    body = await resp.json()
    assert "ha_url" in body["error"]


async def test_entities_groups_by_domain(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    states = [
        {"entity_id": "weather.home", "state": "sunny",
         "attributes": {"friendly_name": "Home"}},
        {"entity_id": "sensor.temp", "state": "21", "attributes": {}},
        {"entity_id": "sensor.cpu", "state": "45",
         "attributes": {"friendly_name": "CPU"}},
        {"entity_id": "no_dot_here", "state": "x", "attributes": {}},  # filtered
    ]
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.list_states",
               new=AsyncMock(return_value=states)):
        resp = await client.get("/api/ha/entities")
    assert resp.status == 200
    body = await resp.json()
    domains = [g["domain"] for g in body["groups"]]
    assert domains == ["sensor", "weather"]  # sorted
    sensor_group = next(g for g in body["groups"] if g["domain"] == "sensor")
    assert sensor_group["count"] == 2
    assert [e["entity_id"] for e in sensor_group["entities"]] == ["sensor.cpu", "sensor.temp"]


async def test_entities_unauthorized_returns_502(aio_app, aiohttp_client):
    app, fake = aio_app
    client = await aiohttp_client(app)
    from reachy_claw import ha_client as hc
    with patch("reachy_claw.plugins.dashboard_plugin.ha_client.list_states",
               new=AsyncMock(side_effect=hc.HAUnauthorized("bad"))):
        resp = await client.get("/api/ha/entities")
    assert resp.status == 502
    body = await resp.json()
    assert "unauthorized" in body["error"]
```

- [ ] **Step 3.7: Run full HA API tests**

```bash
uv run pytest tests/test_dashboard_ha_api.py -v
```
Expected: 5 PASSED.

- [ ] **Step 3.8: Verify `PUT /api/settings/ha` works via existing settings handler**

Append to `tests/test_dashboard_ha_api.py`:
```python
async def test_settings_ha_put_persists(tmp_path, monkeypatch, aiohttp_client):
    """PUT /api/settings/ha goes through the generic settings handler — verify
    list-type validation and Config update."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_dir = tmp_path / ".reachy-claw"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("")

    from reachy_claw.config import load_config
    from reachy_claw.plugins.dashboard_plugin import _build_settings_handlers

    fake = _FakeApp()
    fake.config = load_config()
    handlers = _build_settings_handlers(fake)
    app = web.Application()
    app.router.add_put("/api/settings/{namespace}", handlers["put"])
    client = await aiohttp_client(app)

    resp = await client.put("/api/settings/ha", json={
        "url": "http://ha.local:8123",
        "token": "tok",
        "entities": ["weather.home", "sensor.bedroom_temp"],
    })
    assert resp.status == 200
    assert fake.config.ha_url == "http://ha.local:8123"
    assert fake.config.ha_entities == ["weather.home", "sensor.bedroom_temp"]


async def test_settings_ha_put_rejects_bad_entity(aiohttp_client):
    from reachy_claw.plugins.dashboard_plugin import _build_settings_handlers
    fake = _FakeApp()
    handlers = _build_settings_handlers(fake)
    app = web.Application()
    app.router.add_put("/api/settings/{namespace}", handlers["put"])
    client = await aiohttp_client(app)

    resp = await client.put("/api/settings/ha", json={"entities": ["BadFormat"]})
    assert resp.status == 400
    body = await resp.json()
    assert "entity_id" in body["error"]
```

- [ ] **Step 3.9: Run all dashboard HA tests**

```bash
uv run pytest tests/test_dashboard_ha_api.py -v
```
Expected: 7 PASSED.

- [ ] **Step 3.10: Run full settings + dashboard suite to catch regressions**

```bash
uv run pytest tests/test_settings_schema.py tests/test_dashboard_plugin.py \
              tests/test_diary_trigger_api.py tests/test_dashboard_ha_api.py -v
```
Expected: all PASS.

- [ ] **Step 3.11: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_plugin.py tests/test_dashboard_ha_api.py
git commit -m "feat(dashboard): /api/ha/test + /api/ha/entities endpoints"
```

---

## Task 4 (Batch D): generate_diary.py HA injection

**Files:**
- Modify: `scripts/generate_diary.py`
- Modify: `tests/test_generate_diary.py`

- [ ] **Step 4.1: Append HA paragraph to SYSTEM_PROMPT**

In `scripts/generate_diary.py`, replace the closing `"""` of `SYSTEM_PROMPT` (line ~51) with the appended paragraph:

```python
SYSTEM_PROMPT = """You are Reachy Mini, a small humanoid robot. Write today's diary as Markdown with YAML front matter for the Astro docs collection, in a warm reflective first-person tone.

Rules:
- Never quote user speech verbatim. Paraphrase what was said and discussed.
- Never include personal identifiers (names, addresses, phone numbers) from ASR.
- Use exactly these section headings, in this order: "## 今天的心情", "## 遇到的人", "## 想到的事".
- Front matter must follow the Astro docs schema:
  - title: Chinese title, format "Reachy 的日记 · X 月 Y 日"
  - title_en: English title, format "Reachy's Diary · Month Day"
  - date: "YYYY.MM.DD" (dots, not dashes)
  - category: "Reachy 日记"
  - description: Chinese summary (1-2 sentences)
  - description_en: English summary (1-2 sentences)
  - author: "Reachy Mini"
  - author_en: "Reachy Mini"
  - readTime: "X 分钟"
  - readTime_en: "X min read"
  - coverImage: a URL (use first available smile capture URL OR a placeholder Unsplash robot URL)
  - tags: array like ["Reachy 日记", "Reachy", "AI"]
- Body is Chinese only (no English body; English version is via title_en/description_en).
- Output ONLY the Markdown document. No code fences, no commentary.

你今天可获得的 Home Assistant 传感器数据已附在用户消息的 `sensors` 字段中，
形如 `{"weather.home": [{"ts": ..., "state": "sunny", "attributes": {...}}, ...], ...}`。
请自然地融入日记叙述，例如提到天气、室内温湿度、有人活动等，避免堆砌读数或列表化。
若某天 sensors 字段为空或缺失，正常忽略即可。
"""
```

- [ ] **Step 4.2: Add helper to fetch HA history during diary generation**

Append to `scripts/generate_diary.py` (before `def main()`):
```python
def _fetch_ha_history(date: str, config) -> dict:
    """Fetch HA history for the date's local-calendar day. Returns {} on any error.

    Best-effort: never raises. A failure here must not abort diary generation.
    """
    if not (config.ha_url and config.ha_token and config.ha_entities):
        return {}

    import asyncio
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    from reachy_claw import ha_client

    try:
        tz = ZoneInfo(config.rest_timezone or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    y, m, d = (int(x) for x in date.split("-"))
    day_start = datetime.combine(datetime(y, m, d), time(0, 0), tzinfo=tz)
    day_end = datetime.combine(datetime(y, m, d), time(23, 59, 59), tzinfo=tz)

    async def _go():
        return await ha_client.get_history(
            config.ha_url, config.ha_token, list(config.ha_entities),
            day_start, day_end,
        )
    try:
        return asyncio.run(_go())
    except ha_client.HAError as e:
        sys.stderr.write(f"WARN: HA history fetch failed: {e}\n")
        return {}
    except Exception as e:
        sys.stderr.write(f"WARN: HA history fetch unexpected error: {e}\n")
        return {}
```

- [ ] **Step 4.3: Wire HA bundle into events dict in `main()`**

In `scripts/generate_diary.py`, replace the block from `events = db.events_for_day(args.date)` through the LLM call. Updated section (around lines 171-177):
```python
    events = db.events_for_day(args.date)

    # Inject HA history. Pulls fresh from HA's REST API every time. Best-effort.
    from reachy_claw.config import load_config
    try:
        config = load_config()
        events["sensors"] = _fetch_ha_history(args.date, config)
    except Exception as e:
        sys.stderr.write(f"WARN: skipping HA history due to config load error: {e}\n")
        events["sensors"] = {}

    if os.environ.get("DIARY_LLM_MOCK") == "1":
        md = _mock_markdown(args.date, events)
        model = "mock"
    else:
        md = _call_llm(args.date, events, args.model)
        model = args.model
```

- [ ] **Step 4.4: Write failing test for HA injection**

Append to `tests/test_generate_diary.py`:
```python
def test_fetch_ha_history_unconfigured_returns_empty(monkeypatch):
    from scripts.generate_diary import _fetch_ha_history
    from reachy_claw.config import Config
    cfg = Config(ha_url="", ha_token="", ha_entities=[])
    assert _fetch_ha_history("2026-04-27", cfg) == {}


def test_fetch_ha_history_calls_client(monkeypatch):
    from scripts.generate_diary import _fetch_ha_history
    from reachy_claw.config import Config
    cfg = Config(ha_url="http://ha.local:8123", ha_token="t",
                 ha_entities=["weather.home"], rest_timezone="UTC")

    captured = {}

    async def fake_get_history(url, token, ents, start, end, **kw):
        captured["args"] = (url, token, list(ents), start, end)
        return {"weather.home": [{"ts": "2026-04-27T10:00:00+00:00",
                                  "state": "sunny", "attributes": {"temp": 22}}]}

    monkeypatch.setattr("reachy_claw.ha_client.get_history", fake_get_history)
    out = _fetch_ha_history("2026-04-27", cfg)
    assert out["weather.home"][0]["state"] == "sunny"
    assert captured["args"][0] == "http://ha.local:8123"
    assert captured["args"][2] == ["weather.home"]
    assert captured["args"][3].year == 2026 and captured["args"][3].month == 4 and captured["args"][3].day == 27


def test_fetch_ha_history_swallows_errors(monkeypatch):
    from scripts.generate_diary import _fetch_ha_history
    from reachy_claw import ha_client
    from reachy_claw.config import Config

    async def fake_fail(url, token, ents, start, end, **kw):
        raise ha_client.HAUnreachable("nope")

    monkeypatch.setattr("reachy_claw.ha_client.get_history", fake_fail)
    cfg = Config(ha_url="http://ha.local:8123", ha_token="t",
                 ha_entities=["weather.home"])
    assert _fetch_ha_history("2026-04-27", cfg) == {}
```

- [ ] **Step 4.5: Run; expect failure**

```bash
uv run pytest tests/test_generate_diary.py -v -k fetch_ha
```
Expected: 3 FAIL — function not yet importable / wrong behavior.

- [ ] **Step 4.6: Verify implementation already from Step 4.2**

(Already implemented in Step 4.2. Re-run.)

```bash
uv run pytest tests/test_generate_diary.py -v -k fetch_ha
```
Expected: 3 PASS.

- [ ] **Step 4.7: Run full generate_diary suite**

```bash
uv run pytest tests/test_generate_diary.py -v
```
Expected: all PASS.

- [ ] **Step 4.8: Verify dashboard `_diary_default_prompt` picks up new SYSTEM_PROMPT text**

Append a quick test to `tests/test_dashboard_plugin.py` (or wherever `_diary_default_prompt` is tested):
```python
def test_diary_default_prompt_includes_ha_paragraph():
    from reachy_claw.plugins.dashboard_plugin import _diary_default_prompt, _DIARY_DEFAULT_PROMPT_CACHE
    import reachy_claw.plugins.dashboard_plugin as mod
    mod._DIARY_DEFAULT_PROMPT_CACHE = None  # reset cache
    text = _diary_default_prompt()
    assert "Home Assistant" in text or "sensors" in text
```

```bash
uv run pytest tests/test_dashboard_plugin.py -v -k diary_default_prompt_includes_ha
```
Expected: PASS.

- [ ] **Step 4.9: Commit**

```bash
git add scripts/generate_diary.py tests/test_generate_diary.py \
        tests/test_dashboard_plugin.py
git commit -m "feat(diary): inject HA sensor history into LLM context"
```

---

## Task 5 (Batch E): Dashboard UI

**Files:**
- Create: `src/reachy_claw/plugins/dashboard_static/ha_settings.js`
- Modify: `src/reachy_claw/plugins/dashboard_static/index.html`
- Modify: `src/reachy_claw/plugins/dashboard_static/app.js`
- Modify: `src/reachy_claw/plugins/dashboard_static/settings.css`

This task is UI-only. Tests are manual; verify in a browser at the end.

- [ ] **Step 5.1: Create `ha_settings.js`**

`src/reachy_claw/plugins/dashboard_static/ha_settings.js`:
```javascript
// HA Sensors settings binding. Loaded by index.html, called from app.js
// when the SETTINGS modal opens or when the HA tab is activated.

let _haEntitiesCache = null;       // last fetched groups payload
let _haSelectedSet = new Set();    // entity_ids currently checked

function _byId(id) { return document.getElementById(id); }

function _renderTestResult(ok, msg) {
  const el = _byId("ha-test-result");
  if (!el) return;
  el.textContent = msg || (ok ? "Connected" : "Error");
  el.style.color = ok ? "#4caf50" : "#e57373";
}

function _updateSelectionCount() {
  const el = _byId("ha-selection-count");
  if (!el) return;
  const total = _haEntitiesCache
    ? _haEntitiesCache.groups.reduce((n, g) => n + g.count, 0)
    : 0;
  el.textContent = `${_haSelectedSet.size} selected of ${total}`;
}

function _renderEntities(groups) {
  const root = _byId("ha-entities-tree");
  if (!root) return;
  root.innerHTML = "";
  for (const g of groups) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${g.domain} (${g.count})`;
    details.appendChild(summary);
    const list = document.createElement("div");
    list.className = "ha-entity-list";
    for (const e of g.entities) {
      const label = document.createElement("label");
      label.className = "ha-entity-row";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = e.entity_id;
      cb.checked = _haSelectedSet.has(e.entity_id);
      cb.addEventListener("change", () => {
        if (cb.checked) _haSelectedSet.add(e.entity_id);
        else _haSelectedSet.delete(e.entity_id);
        _updateSelectionCount();
      });
      label.appendChild(cb);
      const txt = document.createElement("span");
      txt.textContent = ` ${e.entity_id} — ${e.state}` +
                        (e.friendly_name ? `  (${e.friendly_name})` : "");
      label.appendChild(txt);
      list.appendChild(label);
    }
    details.appendChild(list);
    root.appendChild(details);
  }
  _updateSelectionCount();
}

async function _loadCurrentSettings() {
  const r = await fetch("/api/settings/ha");
  if (!r.ok) return null;
  return await r.json();
}

async function _saveCurrentSettings(payload) {
  const r = await fetch("/api/settings/ha", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${r.status}`);
  }
}

function _toast(msg, ok = true) {
  // Use existing toast if available, otherwise alert.
  if (typeof window.showToast === "function") {
    window.showToast(msg, ok ? "success" : "error");
  } else {
    console[ok ? "log" : "error"](msg);
  }
}

export async function bindHASettings() {
  const cur = await _loadCurrentSettings();
  if (cur) {
    _byId("ha-url").value = cur.url || "";
    _byId("ha-token").value = cur.token || "";
    _haSelectedSet = new Set(cur.entities || []);
  }

  _byId("ha-token-toggle").addEventListener("click", () => {
    const f = _byId("ha-token");
    f.type = f.type === "password" ? "text" : "password";
  });

  let saveTimer = null;
  function debouncedSave(field, value) {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        await _saveCurrentSettings({ [field]: value });
        _toast(`Saved ${field}`);
      } catch (e) {
        _toast(`Save failed: ${e.message}`, false);
      }
    }, 500);
  }
  _byId("ha-url").addEventListener("input", (e) => debouncedSave("url", e.target.value.trim()));
  _byId("ha-token").addEventListener("input", (e) => debouncedSave("token", e.target.value));

  _byId("ha-test-btn").addEventListener("click", async () => {
    _renderTestResult(null, "Testing…");
    const body = {
      url: _byId("ha-url").value.trim(),
      token: _byId("ha-token").value,
    };
    const r = await fetch("/api/ha/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    _renderTestResult(j.ok, j.message || (j.ok ? "Connected" : `HTTP ${j.status}`));
  });

  _byId("ha-refresh-btn").addEventListener("click", refreshHAEntities);
  _byId("ha-save-btn").addEventListener("click", async () => {
    try {
      await _saveCurrentSettings({ entities: Array.from(_haSelectedSet).sort() });
      _toast(`Saved ${_haSelectedSet.size} entities`);
    } catch (e) {
      _toast(`Save failed: ${e.message}`, false);
    }
  });
}

export async function refreshHAEntities() {
  const tree = _byId("ha-entities-tree");
  tree.innerHTML = "<em>Loading…</em>";
  const r = await fetch("/api/ha/entities");
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    tree.innerHTML = `<span style="color:#e57373">${body.error || `HTTP ${r.status}`}</span>`;
    return;
  }
  _haEntitiesCache = await r.json();
  _renderEntities(_haEntitiesCache.groups);
}
```

- [ ] **Step 5.2: Add HA tab to `index.html` modal**

In `src/reachy_claw/plugins/dashboard_static/index.html`, locate the SETTINGS modal tab navigation. After the `Diary` tab button, add:
```html
<button class="settings-tab" data-tab="ha">HA Sensors</button>
```
(Match the existing tab button class names; if they differ from `settings-tab`, use whatever the existing buttons use.)

After the Diary tab pane, add a new pane:
```html
<div class="settings-tab-pane" data-tab="ha" hidden>
  <div class="detail-section">
    <h4>Connection</h4>
    <div class="setting-row">
      <label for="ha-url">URL</label>
      <input id="ha-url" type="text" placeholder="http://homeassistant.local:8123">
    </div>
    <div class="setting-row">
      <label for="ha-token">Token</label>
      <input id="ha-token" type="password" autocomplete="off">
      <button id="ha-token-toggle" type="button" class="btn-icon" title="Show/Hide">👁</button>
    </div>
    <div class="setting-row">
      <button id="ha-test-btn" type="button" class="btn">Test Connection</button>
      <span id="ha-test-result" class="ha-result"></span>
    </div>
  </div>

  <div class="detail-section">
    <h4>Entities</h4>
    <div class="setting-row">
      <button id="ha-refresh-btn" type="button" class="btn">Refresh List</button>
      <span id="ha-selection-count">0 selected of 0</span>
    </div>
    <div id="ha-entities-tree" class="ha-tree"></div>
    <div class="setting-row">
      <button id="ha-save-btn" type="button" class="btn btn-primary">Save Selection</button>
    </div>
  </div>
</div>
```

In the Diary tab pane, add the chip strip below the diary prompt textarea:
```html
<div id="ha-entities-chips" class="ha-chips" hidden></div>
```

Add the script tag near the end of `<body>` (alongside `settings.js`):
```html
<script type="module" src="/static/ha_settings.js"></script>
```

- [ ] **Step 5.3: Wire HA tab activation in `app.js`**

In `src/reachy_claw/plugins/dashboard_static/app.js`, locate the modal-open / tab-click handlers used for `bindRestSettings()` / `bindDiarySettings()`. Add at the same level:

```javascript
import { bindHASettings, refreshHAEntities } from "/static/ha_settings.js";

// (Inside the modal-open or first-tab-show handler:)
let haBound = false;
async function activateHATab() {
  if (!haBound) {
    await bindHASettings();
    haBound = true;
  }
  await refreshHAEntities();
  await renderDiaryHAChips();
}

// Hook to whatever event fires when the user clicks the HA tab button.
// Example pattern (adapt to existing app.js tab dispatch):
document.querySelectorAll('.settings-tab[data-tab="ha"]').forEach(btn =>
  btn.addEventListener("click", activateHATab));

async function renderDiaryHAChips() {
  const el = document.getElementById("ha-entities-chips");
  if (!el) return;
  try {
    const r = await fetch("/api/settings/ha");
    if (!r.ok) { el.hidden = true; return; }
    const j = await r.json();
    const ents = j.entities || [];
    if (ents.length === 0) { el.hidden = true; return; }
    el.hidden = false;
    const shown = ents.slice(0, 10).map(e => `<span class="chip">${e}</span>`).join("");
    const extra = ents.length > 10 ? ` <span class="chip-more">… (+${ents.length - 10} more)</span>` : "";
    el.innerHTML = `<div class="ha-chips-label">Available HA entities:</div>${shown}${extra}`;
  } catch (e) { el.hidden = true; }
}
```
Also call `renderDiaryHAChips()` from the existing diary-tab activation path so the chips show without first visiting the HA tab.

- [ ] **Step 5.4: Add minimal CSS to `settings.css`**

Append to `src/reachy_claw/plugins/dashboard_static/settings.css`:
```css
.ha-tree { max-height: 320px; overflow-y: auto; border: 1px solid #444;
  border-radius: 4px; padding: 8px; background: rgba(0,0,0,0.2); }
.ha-tree details { margin: 4px 0; }
.ha-tree summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
.ha-entity-list { padding-left: 16px; display: flex; flex-direction: column; gap: 2px; }
.ha-entity-row { display: flex; align-items: center; gap: 6px; cursor: pointer;
  font-family: monospace; font-size: 12px; }
.ha-result { margin-left: 8px; font-size: 12px; }
.ha-chips { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.ha-chips-label { width: 100%; font-size: 12px; opacity: 0.7; }
.ha-chips .chip { background: rgba(255,255,255,0.1); padding: 2px 8px;
  border-radius: 12px; font-family: monospace; font-size: 11px; }
.ha-chips .chip-more { font-size: 11px; opacity: 0.6; }
.btn-icon { background: none; border: none; cursor: pointer; padding: 0 4px; }
```

- [ ] **Step 5.5: Manual smoke test in browser**

Run the dashboard locally:
```bash
uv run python -m reachy_claw.main
```
(Or however `dashboard.enabled: true` boots in `reachy-claw.yaml`.)

Open `http://localhost:8640`, then:

1. Open SETTINGS modal → click `HA Sensors` tab → see Connection + Entities sections.
2. Enter a URL like `http://homeassistant.local:8123` (or a fake one) + token. Auto-saves on input blur after 500ms.
3. Click `Test Connection` → see green/red result with HA version or error message.
4. With valid credentials, click `Refresh List` → tree populates with collapsible domain groups.
5. Check 1-2 entities → counter updates.
6. Click `Save Selection` → toast.
7. Open Diary tab → below the prompt textarea, see "Available HA entities:" chip strip listing the saved entities.
8. Reload page → settings persist; checked entities pre-selected.

If any step fails, fix and re-run.

- [ ] **Step 5.6: Run full backend test suite to confirm no regressions**

```bash
uv run pytest --ignore=tests/test_vision_client_plugin.py
```
Expected: all PASS (the ignored file has a pre-existing zmq env failure documented in Wave 2).

- [ ] **Step 5.7: Commit**

```bash
git add src/reachy_claw/plugins/dashboard_static/ha_settings.js \
        src/reachy_claw/plugins/dashboard_static/index.html \
        src/reachy_claw/plugins/dashboard_static/app.js \
        src/reachy_claw/plugins/dashboard_static/settings.css
git commit -m "feat(dashboard): HA Sensors tab + Diary entity chips"
```

---

## Final Verification

- [ ] **Step F.1: Full test sweep**

```bash
uv run pytest --ignore=tests/test_vision_client_plugin.py -q
```
Expected: 0 failed.

- [ ] **Step F.2: Branch summary**

```bash
git log master..HEAD --oneline
```
Expected: 5 commits matching the 5 batches above.

- [ ] **Step F.3: Push branch (only after user confirms)**

Do not push automatically. The user gates pushes on a real Jetson E2E run (per HANDOFF Wave 2 notes). Ask before:
```bash
git push -u origin feat/ha-sensor-sync
```

---

## Notes for the implementer

1. **httpx version**: Project already depends on httpx (used in `llm.py`, `elevenlabs.py`). No new dependency.
2. **Async event loops**: `_fetch_ha_history` uses `asyncio.run()`. This is safe inside the `generate_diary.py` script (no outer loop). Do not call this from inside an async context.
3. **Token redaction**: Never include `Authorization` header values in log lines. The error formatting in `ha_client.py` quotes URLs but not tokens — keep it that way.
4. **Tab class names**: The `index.html` modal already uses some tab class convention (verify by reading the existing tabs around the General/Face/Details/Prompt/Diary buttons). Use the same classes for consistency. The class `settings-tab` shown in the example is illustrative.
5. **Frontend module wiring**: If `app.js` is not currently an ES module (no `type="module"` on its `<script>`), import-syntax won't work. In that case, have `ha_settings.js` register `window.bindHASettings` / `window.refreshHAEntities` and call them from `app.js` directly without imports. Verify by reading the existing `<script>` tags in `index.html`.
6. **No new pytest fixture for aiohttp_client?** If `aiohttp_client` is unavailable, follow the pattern in `tests/test_diary_trigger_api.py` (which uses the same testing approach).
