# HA Sensor Sync + Config Panel — Design

**Wave 3** of the Reachy diary feature stack. Stacked on `feat/rest-and-settings-panel`.

**Branch:** `feat/ha-sensor-sync` (to be created from `feat/rest-and-settings-panel`).

---

## Goal

Let the diary LLM weave Home Assistant sensor data (weather, room temperature, occupancy, etc.) into the daily Reachy diary, configured entirely through the dashboard. No hardcoded semantic mapping — the user's Diary Prompt decides interpretation.

## Non-Goals

- No long-term sensor archive on Reachy (HA recorder is the system of record).
- No real-time sensor reactions (e.g., motion-triggered behaviour). This is offline diary input only.
- No write-back to HA (Reachy doesn't control devices).
- No semantic mapping ("which entity is weather", "which is occupancy") in code.

---

## Architecture

**Pull-on-demand, no background sync.** When `generate_diary.py` runs (nightly via housekeeping, or manually via dashboard), it fetches the day's history for the configured entities directly from HA's REST API and bundles it into the LLM context.

Three components:

1. `src/reachy_claw/ha_client.py` — stateless async HA REST client (probe, list_states, get_history).
2. `dashboard_plugin.py` — three new endpoints for connection test, entity listing, and settings persistence.
3. `dashboard_static/` — new "HA Sensors" tab in the SETTINGS modal; minor enhancement to the existing Diary tab.

No new plugin, no asyncio task, no SQLite writes. The existing empty `sensors` table is left in place for future local-sensor use.

---

## Resolved design questions

| # | Question | Decision |
|---|----------|----------|
| Q1 | Token storage location | `runtime-overrides.yaml` (same as other settings) |
| Q2 | Polling cadence under rest | N/A — no polling, pull-on-demand only |
| Q3 | Attribute flattening | Pass full `attributes` JSON through to LLM |
| Q4 | Diary Prompt template | Default SYSTEM_PROMPT gets a guidance paragraph + dashboard shows selected entity list |

---

## Data Model

**No schema changes.** `sensors` table (created in Wave 1) remains empty in Wave 3. Reserved for future local-sensor sources.

---

## Component 1 — `ha_client.py`

New file. Pure stateless async functions using `httpx.AsyncClient`.

```python
class HAError(Exception): ...
class HAUnreachable(HAError): ...     # network / timeout / DNS
class HAUnauthorized(HAError): ...    # 401
class HABadResponse(HAError): ...     # other 4xx/5xx, parse failure

async def probe(url: str, token: str, *, timeout: float = 5.0) -> dict:
    """GET <url>/api/ — returns {"ok": bool, "status": int, "message": str}.
    Never raises; status field reflects errors. Used by /api/ha/test."""

async def list_states(url: str, token: str, *, timeout: float = 10.0) -> list[dict]:
    """GET <url>/api/states — returns full HA states list.
    Raises HAError subclasses on failure. Used by /api/ha/entities."""

async def get_history(
    url: str, token: str, entity_ids: list[str],
    start: datetime, end: datetime,
    *, timeout: float = 30.0,
) -> dict[str, list[dict]]:
    """GET <url>/api/history/period/<start>?filter_entity_id=...&end_time=...
    (we deliberately do NOT pass `minimal_response` because it strips
    `attributes` from non-first rows, which would violate the "full
    attributes JSON to LLM" decision from spec Q3=B).
    Returns {entity_id: [{"ts": iso8601_str, "state": str, "attributes": dict}, ...]}.
    Raises HAError subclasses on failure. Used by generate_diary.py."""
```

**Behaviour:**
- URL normalised: must start with `http://` or `https://`; trailing `/` stripped.
- `Authorization: Bearer <token>` header. Token never logged.
- All 401 → `HAUnauthorized`. Connection error / timeout → `HAUnreachable`. Other → `HABadResponse(status, message)`.

---

## Component 2 — Config Fields

`src/reachy_claw/config.py` `Config` dataclass adds:

```python
ha_url: str = ""              # e.g. "http://homeassistant.local:8123"
ha_token: str = ""            # long-lived access token
ha_entities: list[str] = field(default_factory=list)   # entity_ids to include; empty = HA disabled
```

`KEY_MAP` adds entries for namespace `"ha"`: `("ha", "url")`, `("ha", "token")`, `("ha", "entities")`.

Persistence: `runtime-overrides.yaml` (existing mechanism). No restart required for HA changes — `generate_diary.py` reads fresh config each run.

---

## Component 3 — Settings Schema

`src/reachy_claw/settings_schema.py` adds 3 specs:

| Key | Type | Validator |
|-----|------|-----------|
| `ha.url` | str | Empty OR matches `^https?://`. Trailing `/` stripped on save. |
| `ha.token` | str | Empty allowed. Stored as-is. |
| `ha.entities` | list[str] | New `_validate_str_list` validator. Each entry must match `^[a-z_]+\.[a-zA-Z0-9_]+$` (HA entity_id pattern). |

The validator registry (`_VALIDATORS`) gains a `"str_list"` type. Integrated into the existing `validate(namespace, key, value)` dispatch.

---

## Component 4 — Dashboard API Endpoints

In `dashboard_plugin.py`, new `_build_ha_handlers(app)` registers:

### `POST /api/ha/test`
- Body: `{"url"?: str, "token"?: str}`. Fields override saved config for this probe; missing fields fall back to current `app.config`.
- Always returns HTTP 200; result encoded in body:
  ```json
  {"ok": true, "status": 200, "message": "HA 2026.4.0"}
  {"ok": false, "status": 401, "message": "Invalid token"}
  {"ok": false, "status": 0, "message": "Connection refused"}
  ```

### `GET /api/ha/entities`
- Uses saved `ha_url` + `ha_token`.
- 400 if either is empty: `{"error": "ha_url or ha_token not configured"}`.
- Calls `list_states()`. On `HAError` → 502 + structured body.
- On success, groups entities by domain (prefix before `.`):
  ```json
  {
    "groups": [
      {"domain": "weather", "count": 1, "entities": [
        {"entity_id": "weather.home", "state": "sunny", "friendly_name": "Home"}
      ]},
      {"domain": "sensor", "count": 45, "entities": [...]}
    ]
  }
  ```
  Domains sorted alphabetically; entities within each domain sorted by `entity_id`.

### `PUT /api/settings/ha`
- Already routed by existing `_build_settings_handlers` namespace dispatch. No new handler code, but the schema entries above must be registered.
- Persists to `runtime-overrides.yaml`. Returns `{"updated": [<keys>]}` per the existing settings handler convention.

---

## Component 5 — Dashboard UI (HA Sensors Tab)

### Modal tab order (after Wave 3)

`General | Face | Details | Prompt | Diary | HA Sensors`

### HA Sensors tab content

Two `detail-section` blocks (style-consistent with Diary tab):

**Connection section:**
- URL: `<input type="text">`, debounced auto-save on blur.
- Token: `<input type="password">` + show/hide toggle button. Auto-save on blur.
- Test Connection button → calls `/api/ha/test` → inline result indicator (✓ green / ✗ red) + message text.

**Entities section:**
- Refresh List button → calls `/api/ha/entities` → re-renders the tree.
- Header line: `N selected of M` (live count).
- Tree: collapsible domain groups, each group header `▼ domain (count)`. All groups collapsed by default.
- Within group: `<label><input type="checkbox" value="<entity_id>"> entity_id — state (friendly_name)</label>`.
- Previously saved `ha_entities` are pre-checked.
- Save Selection button → `PUT /api/settings/ha` body `{"entities": [<checked>]}`. Toast on success. **No "Restart required" prompt** — pull-on-demand picks up new selection on next diary run.

### Diary tab enhancement (Q4 part C)

Below the Diary Prompt textarea, an info row:
> **Available HA entities:** `weather.home`, `sensor.bedroom_temp`, … _(12)_

- Rendered when `ha_entities` is non-empty.
- Read-only chips. Truncates to ~10 with `… (+N more)` if list is long.
- Source: `GET /api/settings/ha` (or whatever endpoint already returns the namespace).

### New frontend file

`src/reachy_claw/plugins/dashboard_static/ha_settings.js`. Exports:
- `bindHASettings()` — wires inputs, buttons, debounced auto-save.
- `refreshHAEntities()` — calls `/api/ha/entities`, renders the tree, syncs checkbox state from saved selection.

Loaded by `index.html`, called by `app.js` when modal opens (parallel to existing `bindRestSettings()` / `bindDiarySettings()`).

---

## Component 6 — Diary Prompt + generate_diary.py

### SYSTEM_PROMPT addition (Q4 part B)

Append to the existing `SYSTEM_PROMPT` constant in `scripts/generate_diary.py`:

```
你今天可获得的 Home Assistant 传感器数据已附在用户消息的 `sensors` 字段中，
形如 `{"weather.home": [{"ts": ..., "state": "sunny", "attributes": {...}}, ...], ...}`。
请自然地融入日记叙述，例如提到天气、室内温湿度、有人活动等，避免堆砌读数或列表化。
若某天 sensors 字段为空或缺失，正常忽略即可。
```

This propagates automatically into the dashboard Diary Prompt textarea default (Wave 2 made it a single source of truth via `_diary_default_prompt()`).

### Data collection

`events_for_day(date)` (or its caller) is extended:

1. If `config.ha_url`, `config.ha_token`, `config.ha_entities` are all non-empty:
   - Compute `[start, end]` covering the local calendar day in `config.rest_timezone` (use `zoneinfo.ZoneInfo` + `datetime.combine`).
   - Call `await ha_client.get_history(url, token, entities, start, end)`.
   - On any `HAError` → `logger.warning("HA history fetch failed: %s", e)`, set `sensors = {}`, continue.
2. Else: `sensors = {}`.
3. Bundle the existing dict with `"sensors": {entity_id: [{ts: iso8601, state, attributes}, ...]}` and JSON-dump into the LLM user prompt.

Diary generation never fails because of HA — sensors are best-effort.

---

## Error Handling Summary

| Scenario | Behaviour |
|---|---|
| HA URL empty | Diary still generates (no sensors). Dashboard `/api/ha/entities` 400. |
| HA token empty | Same as above. |
| HA token wrong (401) | Diary: warning logged, no sensors. Dashboard test/entities → user-visible error. |
| HA unreachable (timeout, DNS, conn refused) | Same as above. |
| HA returns malformed JSON | `HABadResponse`, same handling. |
| `ha_entities` is empty list | Skipped — no fetch, no sensors in bundle. |

---

## Testing

| Test file | Coverage |
|-----------|----------|
| `tests/test_ha_client.py` (new) | `httpx` mocked. probe 200/401/timeout. list_states parse. get_history time window construction + multi-entity grouping. Token redaction in logs. |
| `tests/test_dashboard_ha_api.py` (new) | `/api/ha/test` 200 with all variants of body. `/api/ha/entities` 400 / 200 grouping / 502 on HAError. `PUT /api/settings/ha` list-type accepts `[]`, valid list, rejects non-list / non-string elements / bad entity_id pattern. |
| `tests/test_settings_schema.py` (extend) | `_validate_str_list` accepts `[]`, `["a.b"]`; rejects `"a"`, `[1]`, `["bad name"]`. |
| `tests/test_generate_diary.py` (extend) | With HA configured + mocked client → bundle has `sensors` key with expected shape. With HA error → bundle has `sensors: {}` and diary still generated. With HA unconfigured → `sensors` key absent or `{}`. |
| Frontend | Manual verification: modal tab opens, list refreshes, checkboxes persist, test connection feedback. |

Target: full test suite passes (excluding the pre-existing `test_vision_client_plugin.py` zmq env failure already documented in Wave 2).

---

## Implementation Order (suggested batches for plan)

1. **A**: `ha_client.py` + `tests/test_ha_client.py`. Pure unit, no integration.
2. **B**: Config fields (`config.py`, `KEY_MAP`) + settings_schema list-type validator + `tests/test_settings_schema.py` extension.
3. **C**: Dashboard API endpoints (`/api/ha/test`, `/api/ha/entities`) + `tests/test_dashboard_ha_api.py`.
4. **D**: `generate_diary.py` SYSTEM_PROMPT addition + history fetch integration + `tests/test_generate_diary.py` extension.
5. **E**: Dashboard UI — `ha_settings.js`, modal HTML extensions, Diary tab chip strip, `app.js` wiring. Manual verify.

Each batch ends with a focused commit.

---

## Out of Scope (explicit)

- Sensor data persistence on Reachy.
- Backfilling diaries older than HA recorder retention (default 10 days).
- HA service calls / device control from Reachy.
- Push-mode (HA → Reachy webhooks). Pull-only.
- Per-attribute selection UI (Q3 was answered B = full JSON, not C = per-attribute).
- Secrets file with `chmod 0600` (Q1 was answered A = yaml override).
- Real-time sensor-driven motion or speech triggers.
