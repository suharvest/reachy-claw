# Rest Window + Settings Panel Design

**Date:** 2026-04-27
**Status:** Approved (pending user review)
**Depends on:** [2026-04-26-diary-archive-and-publish-design.md](./2026-04-26-diary-archive-and-publish-design.md)

## Goals

1. **Rest window** — pause TTS / ASR / LLM / Vision processing during a configurable daily window so the system can run "housekeeping" tasks (diary generation + publish, future: DB vacuum, cover image generation, index rebuilds) without resource contention.
2. **Settings panel on the dashboard** — a new `SETTINGS` top-level tab with extensible sections, first iteration covering the rest window and diary publishing, including manual diary generate/publish triggers for missed days.
3. **Settings persistence** — rest window and diary preferences stored via the existing `runtime-overrides.yaml` mechanism (same as `dashboard_volume`), so dashboard is the source of truth and changes are durable across restarts.

## Non-Goals

- Friendly "I'm resting" voice replies during the window. Rest is total silence.
- Wake-word override during rest. The robot ignores all input until the window ends.
- Authentication on the settings panel. Dashboard is local-network only.
- Multi-window schedules (e.g., morning + night rest). Single daily window in v1.
- Cron-style arbitrary schedules. Just `start_time + end_time + timezone`.

## Architecture

```
┌────────────────────────── Jetson (clawd-reachy-mini) ──────────────────────────┐
│                                                                                │
│  Settings (runtime-overrides.yaml)  ◄── PUT /api/settings/<ns> updates this   │
│  ├─ rest_window_start    "23:00"                                               │
│  ├─ rest_window_end      "24:00"                                               │
│  ├─ rest_enabled         true                                                  │
│  ├─ diary_auto_publish   true                                                  │
│  └─ ...                                                                        │
│  (Same mechanism the dashboard already uses for `dashboard_volume`.)           │
│                                                                                │
│  RestPlugin                                                                    │
│   ├─ reads app.config.rest_window_* every minute                               │
│   ├─ at start: events.publish("rest_start") → all plugins gate                │
│   ├─ during:  runs registered HousekeepingTasks (diary gen+publish, ...)      │
│   └─ at end:  events.publish("rest_end") → all plugins resume                 │
│                                                                                │
│  Subscribers (gate themselves):                                                │
│    ConversationPlugin → stop ASR worker, drain TTS queue, refuse LLM          │
│    FaceTrackerPlugin  → stop MediaPipe loop                                   │
│    VisionClientPlugin → close ZMQ subscription                                │
│    MotionPlugin       → freeze head, mute emotion mapper                      │
│    DailyLogPlugin     → keeps logging (records "rest" as a thought)           │
│                                                                                │
│  Dashboard endpoints:                                                          │
│    GET/PUT /api/settings/<namespace>                                           │
│    GET     /api/diary/status   (per-date: generated? published?)              │
│    POST    /api/diary/generate (manual trigger; supports backfill)            │
│    POST    /api/diary/publish  (manual trigger; supports backfill)            │
└────────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ WebSocket / fetch
                                      │
┌────────────────────────── Dashboard frontend ──────────────────────────────────┐
│  Tabs: [LIVE]  [DIARY]  [SETTINGS]                                             │
│   └─ Settings sections (each is a self-contained component):                   │
│      ┌─ Rest Window ─────────────────────────────┐                             │
│      │ Start [23:00]  End [24:00]  TZ [Asia/SH]  │                             │
│      │ ☑ Enabled                          [Save] │                             │
│      └───────────────────────────────────────────┘                             │
│      ┌─ Diary Publishing ────────────────────────┐                             │
│      │ ☑ Auto publish daily  ☑ Privacy linter    │                             │
│      │ Site repo: [git@...]                       │                             │
│      │ Diary path: [src/content/docs]             │                             │
│      │ ── History ───────────────────────────────│                             │
│      │ 2026-04-26 ✓ Published   [Regenerate]    │                             │
│      │ 2026-04-25 ✗ Missing     [Generate+Publish]│                             │
│      │ 2026-04-24 ⚠ Unpublished [Publish]        │                             │
│      └───────────────────────────────────────────┘                             │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Settings Persistence

**Reuse the existing `runtime-overrides.yaml` mechanism** (already used by the dashboard for things like `dashboard_volume`). No new SQLite table; no schema migration. This keeps a single, consistent runtime-config story across all dashboard-tunable settings.

How it works (already implemented in the codebase):

```
On startup:
  reachy-claw.yaml        → bootstrap defaults (deploy-time, immutable in container)
  runtime-overrides.yaml  → user-tunable values written by the dashboard
  Merged into app.config

Dashboard changes a setting:
  self.app.config.rest_window_start = "23:00"        # immediate, in-process
  save_runtime_overrides(["rest_window_start", ...])  # persists to runtime-overrides.yaml
```

`save_runtime_overrides()` already exists in `src/reachy_claw/config.py`. It writes to `~/.reachy-claw/runtime-overrides.yaml` (or `$DATA_DIR/runtime-overrides.yaml`), which is on the persistent volume.

### New config fields (added to ReachyClawConfig dataclass)

| Field | Default | Type | Persists in runtime-overrides? |
|-------|---------|------|----|
| `rest_enabled` | `True` | bool | yes |
| `rest_window_start` | `"23:00"` | str (HH:MM) | yes |
| `rest_window_end` | `"24:00"` | str (HH:MM) | yes |
| `rest_timezone` | `"Asia/Shanghai"` | str (IANA tz name) | yes |
| `diary_auto_publish` | `True` | bool | yes |
| `diary_privacy_linter` | `True` | bool | yes |
| `diary_site_repo_url` | `""` | str | yes |
| `diary_site_diary_path` | `"src/content/docs"` | str | yes |
| `diary_site_branch` | `"main"` | str | yes |

These appear in `reachy-claw.yaml` (as bootstrap defaults under a new `rest:` and `diary:` section, mapped via the existing key-flattening logic in `config.py`) AND can be overridden at runtime via the dashboard.

The yaml `rest:` section in `reachy-claw.example.yaml`:

```yaml
rest:
  enabled: true
  window_start: "23:00"
  window_end: "24:00"
  timezone: Asia/Shanghai

diary:
  auto_publish: true
  privacy_linter: true
  site_repo_url: ""           # required for publishing; e.g. git@github.com:org/site.git
  site_diary_path: src/content/docs
  site_branch: main
```

The existing `KEY_MAP` in `config.py` (currently maps `("audio", "volume") → "audio_volume"`) is extended with the new mappings.

### Settings schema registry

For dashboard validation only, a new module `src/reachy_claw/settings_schema.py` enumerates the known settings with their types, validators, and namespace grouping. This is the single source of truth for "which settings are tunable from the dashboard, and what shape are their values". Used by the settings API to:
- enumerate keys per namespace (for GET)
- validate values (for PUT)
- map dashboard namespace+key (e.g. `rest.window.start`) → flat config field name (`rest_window_start`)

## RestPlugin

### Lifecycle

`src/reachy_claw/plugins/rest_plugin.py`

```python
class RestPlugin(Plugin):
    name = "rest"

    async def start(self) -> None:
        while self._running:
            now_local = self._now_in_tz()
            should_rest = self._is_in_window(now_local)
            if should_rest and not self._resting:
                await self._enter_rest()
            elif not should_rest and self._resting:
                await self._exit_rest()
            await asyncio.sleep(30)
```

### Events emitted

```python
# When entering the rest window
events.publish("rest_start", {"started_at": ts, "estimated_end": ts_end})

# Each housekeeping task lifecycle (purely informational, for dashboard)
events.publish("housekeeping_task_start", {"name": "diary_publish"})
events.publish("housekeeping_task_end", {"name": "diary_publish", "ok": True, "error": None})

# When leaving the rest window
events.publish("rest_end", {"ended_at": ts})
```

### Housekeeping task registry

```python
class HousekeepingTask(Protocol):
    name: str
    async def run(self, app) -> None: ...

class RestPlugin(Plugin):
    def register_task(self, task: HousekeepingTask) -> None: ...
```

V1 ships with one built-in task: `DiaryGenerateAndPublishTask` (calls `generate_diary.py` then `publish_diary.py` for today's date). It checks `diary.auto_publish` setting; if false, it generates but doesn't publish.

Future tasks (out of scope): `DBVacuumTask`, `CoverImageGenerateTask`, `LogRotateTask`. Adding one is just a `register_task()` call; no code change to RestPlugin itself.

Tasks run **sequentially**, each wrapped in try/except so one failure doesn't block others. Total task budget: rest window length minus 5 minutes safety margin. Tasks should self-time-budget; if they take longer than the safety margin, they're allowed to overrun (rest window stays active until tasks complete OR end_time + 10min hard cap).

### Settings reload

RestPlugin reads from `app.config` every loop tick (30s). Dashboard updates mutate `app.config` in-process AND write `runtime-overrides.yaml`, so changes apply within a minute without restart.

## Plugin Pause/Resume

Each plugin defines `_on_rest_start` and `_on_rest_end` handlers and subscribes during `start()`. The base `Plugin` class gets a helper:

```python
class Plugin:
    async def on_rest_start(self) -> None:
        """Override to pause work. Default: no-op."""

    async def on_rest_end(self) -> None:
        """Override to resume. Default: no-op."""
```

`RestPlugin` invokes these via the event bus rather than direct calls (preserves loose coupling).

### Per-plugin behavior

**ConversationPlugin** (the heaviest):
- On rest_start: signal ASR worker to stop reading mic; cancel any in-flight LLM request; drain TTS playback queue; ignore new utterances.
- On rest_end: restart ASR; flush any cached state; resume normal flow.

**FaceTrackerPlugin** / **VisionClientPlugin**:
- On rest_start: cancel the per-frame coroutine; release camera/ZMQ.
- On rest_end: re-open and resume.

**MotionPlugin**:
- On rest_start: send "neutral pose" command, freeze head_target_bus output.
- On rest_end: resume normal motion.

**DailyLogPlugin / DashboardPlugin**:
- No-op. They keep working — dashboard still serves UI, daily log still records the rest event itself.

### Remote vision containers (vision-hailo, vision-stub)

The vision producers run in separate Docker containers (different process, different host possibly) and can't share the in-process EventBus. They subscribe to a **ZMQ control topic** that mirrors the local rest events:

- **RestPlugin opens a ZMQ PUB socket** on `tcp://0.0.0.0:18791` (configurable via `app.config.rest_control_port`, default `18791`).
- On `_enter_rest()` / `_exit_rest()`, it sends a JSON message: `{"cmd": "pause"}` / `{"cmd": "resume"}`.
- Each vision producer's main loop opens a `SUB` socket connecting to `tcp://<reachy_host>:18791` (env `REST_CTRL_URL`), polls non-blocking each iteration; if `pause` received, the producer skips camera read + inference and emits no detections until `resume`.

This makes vision containers symmetric with in-process plugins ("subscribe to rest events, gate the hot loop"). Frees the NPU during housekeeping (relevant for future cover-image generation tasks). Container code change is small (~30 lines per producer) and isolated.

A new config field `rest_control_port` (default `18791`) is added to `Config` and to the `rest:` yaml section. Producers default to `tcp://reachy:18791` (or whatever the producer container's network resolves to); operationally configured via `REST_CTRL_URL` env in `docker-compose.yml`.

## Dashboard API

### Settings endpoints

```
GET  /api/settings/<namespace>
  → 200 {key1: value1, key2: value2, ...}   (all keys with that namespace prefix)

PUT  /api/settings/<namespace>
  body: {key1: value1, ...}
  → 200 {updated: [keys]}
  → 400 if any key is unknown / value type mismatches expected
```

Allowed namespaces in v1: `rest`, `diary`. Unknown namespace → 404. Validation rules (from `settings_schema.py`):
- `rest.window_start`, `rest.window_end` must match `HH:MM` (24-hour).
- `rest.timezone` must be a valid IANA tz name.
- `rest.enabled`, `diary.auto_publish`, `diary.privacy_linter` must be bool.
- `diary.site_repo_url`, `diary.site_diary_path`, `diary.site_branch` must be strings.

The settings API maps the dashboard's namespace+key (e.g. `rest.window_start`) to the flat config field name (`rest_window_start`). On PUT it validates, sets `app.config.<field>`, and calls `save_runtime_overrides([...])`. Unknown keys are rejected — keeps the schema explicit and catches typos.

### Diary trigger endpoints

```
GET  /api/diary/status
  → 200 {
      "dates": [
        {"date": "2026-04-26", "generated": true, "published": true},
        {"date": "2026-04-25", "generated": false, "published": false},
        ...
      ],
      "scan_window_days": 14
    }

POST /api/diary/generate
  body: {"date": "2026-04-26", "force": false}
  → 202 {"job_id": "..."}                    (async; runs in background task)
  → 200 {"date": "...", "status": "already-generated"}   (if exists and !force)

POST /api/diary/publish
  body: {"date": "2026-04-26", "force": false}
  → 202 {"job_id": "..."}
  → 200 {"date": "...", "status": "already-published"}
```

Job progress is broadcast over the existing dashboard WebSocket as:
```json
{"type": "diary_job", "job_id": "...", "phase": "generating|publishing|done|error", "date": "2026-04-26", "error": "..."}
```

The frontend listens and updates the row's status icon live.

### Status calculation

`/api/diary/status` looks at the last 14 days (`scan_window_days`). For each date:
- `generated` = row exists in `diaries` table
- `published` = `published_at IS NOT NULL`

Dashboard renders this as the diary history list with action buttons.

## Frontend

### Files

- New: `src/reachy_claw/plugins/dashboard_static/settings.js`
- New: `src/reachy_claw/plugins/dashboard_static/settings.css`
- Modified: `src/reachy_claw/plugins/dashboard_static/index.html` (add SETTINGS tab)
- Modified: `src/reachy_claw/plugins/dashboard_static/app.js` (tab switching: SETTINGS shows settings.js's container)

### Settings page structure

`settings.js` exports a `renderSettings(container)` function. Internally it iterates a registered list of section components; each section is a self-contained `{ id, title, render(div), save() }` object.

V1 sections:
1. `RestWindowSection` — time inputs, enable toggle, save button
2. `DiaryPublishingSection` — toggles for auto_publish + privacy_linter, repo config text fields, history list with action buttons

Adding a new section in a future iteration = adding one entry to the section registry. No core changes.

### History list interaction

For each date in `/api/diary/status`:
- Both `generated` and `published` true → green ✓ + `[Regenerate]` button (calls `/api/diary/generate?force=1`)
- `generated` true, `published` false → yellow ⚠ + `[Publish]` button
- Both false → red ✗ + `[Generate + Publish]` button (sequential POST)

Clicking a button:
1. Show inline spinner on that row
2. POST to corresponding endpoint
3. Listen for matching `diary_job` WS event
4. Update icon + button text on completion

## File Changes

### New

- `src/reachy_claw/plugins/rest_plugin.py` — schedule poller + housekeeping registry
- `src/reachy_claw/plugins/housekeeping_tasks.py` — `DiaryGenerateAndPublishTask`, base `HousekeepingTask` protocol
- `src/reachy_claw/settings_schema.py` — known-key registry, validators, dashboard↔config field mapping
- `src/reachy_claw/plugins/dashboard_static/settings.js`
- `src/reachy_claw/plugins/dashboard_static/settings.css`
- `tests/test_rest_plugin.py`
- `tests/test_settings_schema.py` — registry validation rules
- `tests/test_settings_api.py` — GET/PUT endpoint behavior + persistence
- `tests/test_diary_trigger_api.py`

### Modified

- `src/reachy_claw/config.py` — add new fields to `ReachyClawConfig` dataclass + `KEY_MAP` entries for `rest:` and `diary:` yaml sections
- `reachy-claw.example.yaml` — document the new `rest:` and `diary:` sections
- `src/reachy_claw/plugin.py` — add `on_rest_start` / `on_rest_end` no-op base hooks
- `src/reachy_claw/plugins/conversation_plugin.py` — implement pause/resume
- `src/reachy_claw/plugins/face_tracker_plugin.py` — implement pause/resume
- `src/reachy_claw/plugins/vision_client_plugin.py` — implement pause/resume
- `src/reachy_claw/plugins/motion_plugin.py` — implement pause/resume
- `src/reachy_claw/plugins/dashboard_plugin.py` — settings + diary trigger endpoints + WS job events
- `src/reachy_claw/plugins/dashboard_static/index.html` — SETTINGS tab
- `src/reachy_claw/plugins/dashboard_static/app.js` — tab switching wiring
- `src/reachy_claw/app.py` — register `RestPlugin` and add `DiaryGenerateAndPublishTask`
- `deploy/vision-hailo/producer.py` — ZMQ SUB control + paused flag gating the main loop
- `deploy/vision-hailo/docker-compose.yml` — add `REST_CTRL_URL` env
- `deploy/vision-stub/producer.py` — same ZMQ SUB control
- `deploy/vision-stub/docker-compose.yml` — same env

## Implementation Order

1. **Config fields + settings_schema registry** — add new fields to `ReachyClawConfig`, write the schema registry, extend `KEY_MAP` for the new yaml sections. Unit-testable.
2. **Settings API endpoints** — GET/PUT backed by `app.config` + `save_runtime_overrides()`, with validation via the schema registry.
3. **`Plugin.on_rest_start/end` base hooks** — no-op defaults; tests confirm subscription wiring.
4. **RestPlugin shell** — schedule loop, event emission. No housekeeping yet.
5. **Per-plugin pause/resume** — one plugin at a time, each with its own integration test.
6. **Housekeeping registry + DiaryGenerateAndPublishTask** — runs `generate_diary.py` and `publish_diary.py` as subprocesses.
7. **Diary trigger API endpoints** — POST generate / publish, async with WS job events.
8. **Dashboard settings tab UI** — section registry + the two v1 sections.
9. **Dashboard diary history UI** — wired to `/api/diary/status`.
10. **End-to-end manual test** — set rest window to "now+2min" through 5 minutes; observe pause, observe housekeeping task run, observe resume.

## Testing

- Settings schema registry: unit tests for validators (HH:MM format, IANA tz, bool coercion).
- Settings API: integration tests for GET/PUT, validation rejection, and that PUT both mutates `app.config` and writes `runtime-overrides.yaml`.
- Config loading: round-trip test that yaml `rest:` / `diary:` sections map to the new dataclass fields and that `runtime-overrides.yaml` overlays them on startup.
- RestPlugin: unit tests with a frozen-time clock, asserting event emission.
- Per-plugin pause/resume: integration tests publishing fake `rest_start/end` events and asserting plugin state.
- Housekeeping task: subprocess mocking; verify it runs `generate_diary.py` then `publish_diary.py` only when `diary.auto_publish=true`.
- Diary trigger API: 202-then-WS-event flow with mocked subprocesses.
- Frontend: manual testing.

## Risks

- **Restart races** — if a plugin is mid-pause when the rest window ends, resume might race against pause completion. Mitigation: each plugin's pause/resume guarded by an asyncio.Lock; resume awaits the same lock.
- **Long-running housekeeping** — diary generation with a real LLM might take minutes. Mitigation: the rest window is a soft floor; housekeeping is allowed to overrun by up to 10 minutes (hard cap), at which point it's cancelled and logged.
- **Settings drift** — `reachy-claw.yaml` defaults vs `runtime-overrides.yaml` overlays. Mitigation: this is the same merge logic already used for `dashboard_volume`; runtime-overrides wins if present, yaml is the fallback. No new mechanism, no new drift surface.
- **`runtime-overrides.yaml` non-atomic write (pre-existing)** — `save_runtime_overrides()` opens the file with `"w"` and streams YAML; a crash mid-write corrupts the file and would block next startup. This is a pre-existing risk inherited from the volume-persistence path, not something this branch introduces. Out of scope here — fix as a separate small PR (write to `.tmp` then `os.replace`). Documented as a follow-up.
- **Manual generate during rest** — if the user hits "Generate now" while housekeeping is already running it for the same date, we get two LLM calls. Mitigation: a per-date `asyncio.Lock` in the diary trigger handler. The second call returns immediately with **HTTP 409** (`{"status": "in-progress"}`) rather than queueing — this avoids stacking duplicate LLM bills if the user clicks the button twice.
- **Auto-publish without a configured site repo** — would cause every rest cycle to fail loudly. Mitigation: skip the publish step (with a logged warning) when `diary.site_repo_url` is empty; generation still runs.

## Open Questions / Deferred

- Cover image generation (NanoBanana) is **not** in v1. It can be added later as a new HousekeepingTask without changing RestPlugin or the settings panel.
- HA sensor sync settings panel — separate branch, separate spec.
- Multi-window rest schedules (e.g., short morning rest + long night rest) — defer until requested.
