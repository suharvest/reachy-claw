# Design — Dashboard runtime overrides + vision-trt self-heal (Lines 1 + 2)

**Date:** 2026-06-14
**Branch:** `feat/runtime-overrides-and-vision-selfheal`
**Scope:** the follow-up roadmap's **Line 1** (finish dashboard API wiring) + **Line 2**
(robustness / self-heal), trimmed to what brings real, substantive improvement.

## Goal

1. Make the genuinely-useful dashboard tuning controls **actually work** on the
   live SLV/ovs_agent stack, and make those tweaks **persist across restart and
   redeploy**.
2. Make the deferred/not-yet-built controls **visibly disabled** (not hidden) so
   operators can see the roadmap but nothing silently no-ops or 404s.
3. Make `vision-trt` **self-heal** the camera-bind race instead of retrying forever.

## Non-goals (explicitly out of scope this session)

- `set_motor` amplitude presets — no current need, no real improvement → **disabled in UI**.
- `set_llm` switching, conversation modes, prompts, voice/clone, VLM, diary, Home
  Assistant, rest window — **disabled in UI**; the migration of the valuable ones
  (diary / modes / voice-clone / HA / multi-LLM) is **Line 3**.
- Deleting `legacy/reachy_claw/` — kept as the Line 3 migration reference; deleted
  *after* Line 3.
- Real on-robot verification / deploy — this session delivers code + a PR. The
  Jetson smoke deploy is a separate step (and `vision-trt` ships as a *different*
  container image than the voice app).

## Hard guardrails carried in

- **No dependency on `legacy/`.** Legacy `dashboard_plugin.py` is read only to learn
  *what a feature is supposed to do*; every handler here is self-contained in
  `src/reachy_voice/` and writes to the **real ovs_agent objects**. Copying legacy's
  "set a config field" pattern is exactly the SLV no-op trap
  (memory `slv-dashboard-settings-wiring.md`) — avoided by targeting live objects.
- **Feature branch + PR to `suharvest/reachy-claw` over ssh**; never local-merge to
  master; keep `3.10` in the CI matrix (branch-protection status check).

---

## Architecture

### New module: `src/reachy_voice/overrides.py` (SDK-free, unit-testable)

Like `tier_a.py`, this imports **no SDK / GStreamer / PortAudio**, so it unit-tests
on a plain dev box. It reaches the live engine purely by duck-typed attribute
access, so tests pass a fake engine object.

Two responsibilities:

1. **`OverridesStore`** — persistence. A JSON dict at
   `$REACHY_VOICE_DATA_DIR/overrides.json` (default `/data`).
   - `load() -> dict` — returns `{}` if the file is missing or corrupt (never raises).
   - `get(key, default)`, `all() -> dict`.
   - `set(key, value)` — mutates the in-memory dict and writes the file
     **atomically** (write temp + `os.replace`), creating the data dir if needed.

2. **Live appliers / readers** — the single source of truth for each setting, shared
   by startup-replay, the runtime WS handler, and the dashboard snapshot (so the
   snapshot can never drift from what's actually in effect):

   | key | apply(engine, value) writes | read(engine) returns |
   |---|---|---|
   | `bargein` | `engine._app.config.barge_in_enabled = bool` | bool (None→True, ovs default) |
   | `vad` | `engine._app._client_vad.threshold` **and** `engine._app.config.client_vad_threshold` | float (live VAD obj first) |
   | `history` | `engine._app.session.max_input_tokens` (see *History mapping*) | int turns |

   Each applier/reader is defensively guarded (`getattr` chains, try/except at the
   call site) so a missing attribute or one bad saved value degrades gracefully and
   never breaks engine start.

   These three appliers are **plain synchronous attribute assignments** — atomic
   under the GIL — so they are safe to call directly from the dashboard's event-loop
   thread without hopping to the engine loop.

   `language` is **not** in this sync table because it is genuinely async (it calls
   `engine.set_language()`, which reconnects SLV). It is still persisted in the same
   `OverridesStore`; see *Language handling*.

### History mapping

The dashboard control speaks "turns"; ovs_agent trims context by
`session.max_input_tokens` (tokens), not turn count. To avoid faking the control we
map turns → a token budget with a single documented constant
(`TOKENS_PER_TURN`, ~ a few hundred), clamped to a sane floor. The exact constant /
whether ovs `Session` also exposes a turn cap is confirmed by reading
`ovs_agent/session.py` during implementation; if a turn-count lever exists we use it
directly. The principle: wire to the **real** lever and convert units honestly.

### Language handling

- **Persisted** in `OverridesStore` under `language`.
- **Startup:** if an override exists, set `config.language` **before** the engine
  starts (`main.py` `_main`, before `ConversationEngine(...).start()`), so the engine
  boots in the right language with **no** extra SLV reconnect.
- **Runtime (WS `set_conversation_language`):** persist to the store, then drive the
  existing `engine.set_language()` coroutine via `run_coroutine_threadsafe(...,
  self._loop)` (the same path `POST /language` already uses), then echo
  `conversation_language` to all dashboards via the hub.

### Precedence

`dataclass defaults < YAML < env < runtime overrides`. Runtime overrides are the
operator's most recent live intent, so they win — applied last (post-start for the
sync trio; pre-start for language).

---

## Wiring (`src/reachy_voice/main.py`)

1. **`_recv` loop** (currently `main.py:167-194`) — add `elif kind == ...` branches:
   - `set_bargein` → store.set + apply_bargein + hub echo `{"type":"bargein_state",...}`.
   - `set_vad_threshold` → store.set + apply_vad + hub echo `{"type":"vad_threshold",...}`.
   - `set_history` / `get_history` → store.set + apply_history (+ read) + hub/ws
     `{"type":"history","turns":...}`.
   - `get_conversation_language` / `set_conversation_language` → snapshot reply /
     persist + `set_language` coroutine + echo.
   - `send_message` → inject text via the engine's `slv.send_text(...)` (+ optional
     `flush_tts`) using `run_coroutine_threadsafe` — the exact path `/debug/say`
     already uses (`main.py:283-302`).
2. **`_dashboard_snapshot`** (`main.py:366-391`) — replace the hardcoded
   `barge_in_enabled: False` and `silero_threshold` literal with the `read_*`
   functions (falling back to config when the engine isn't ready yet). `vlm_enabled`
   stays `False` (deferred/disabled).
3. **Startup replay** (`_main`, around `main.py:400-403`) — construct the
   `OverridesStore` on `self`; apply the `language` override before `engine.start()`;
   after `engine.start()`, apply the saved `bargein` / `vad` / `history` overrides.

No new third-party deps; all stdlib + existing patterns.

---

## Frontend — disable deferred controls (`src/reachy_voice/static/`)

A single, centralized pass (one helper + one CSS class) marks the deferred controls
`disabled` and non-interactive with a "暂未支持 / Coming soon" `title` tooltip:

- motor panel (toggle + presets), LLM backend/model "Apply", mode buttons, prompts
  tab, voice sliders + clone, VLM toggle, diary, Home Assistant, rest window.

The exact element ids/selectors are read out of the static HTML/JS during
implementation and collected into one list, so re-enabling a control later (when its
Line-3 backend lands) is a one-line edit. Controls we *did* wire (volume, language,
VAD, bargein, history, send_message, captures, restart) stay fully enabled. Static
files take effect on a hard browser reload (no Python restart).

---

## Line 2 — `vision-trt` self-heal (`deploy/jetson/vision-trt/src/capture.py`)

The camera-start retry loop (`capture.py:170-187`) is currently `while True` with
exponential backoff and **no bound** — on the boot race where the container starts
before `/dev/video0`, it can spin forever. Add `MAX_CAMERA_ATTEMPTS`
(env-configurable, default ~20 ≈ a few minutes); when exceeded, `logger.critical(...)`
then `os._exit(1)` (`os` is already imported, `capture.py:16`). The compose
`restart: unless-stopped` policy (`deploy/jetson/docker-compose.yml:53`) then
recreates the container, which re-binds the now-present device — the same manual fix
(`docker restart vision-trt`) the operator did by hand, now automatic.

This ships in the **vision-trt** image (separate build/redeploy from the voice app).

---

## Deploy change (`deploy/jetson/voice/docker-compose.yml`)

Add a **host bind-mount** to a stable absolute Jetson path so overrides survive both
`docker restart` and image rebuild/redeploy, and stay hand-inspectable over SSH:

```yaml
    volumes:
      - /home/recomputer/reachy-voice-data:/data   # NEW — runtime overrides (durable)
    environment:
      - REACHY_VOICE_DATA_DIR=/data                 # NEW
```

Applying it needs one `docker compose up` (not just a restart) on the robot — part of
the separate deploy step, not this PR.

---

## Testing & verification

- **`tests/voice/test_overrides.py`** (SDK-free, runs in CI):
  - `OverridesStore`: round-trip save/load (`tmp_path`); missing file → `{}`; corrupt
    JSON → `{}` (no raise); atomic write replaces cleanly.
  - appliers: against a `SimpleNamespace` fake engine, assert `bargein` sets the
    config field; `vad` sets **both** the VAD object and config; `history` maps
    turns→tokens onto `session.max_input_tokens`.
  - readers: return live values; degrade to defaults when attributes/engine absent.
  - coercion / clamping of out-of-range inputs.
- `uv run ruff check .` clean.
- `import reachy_voice.main` still can't run off-Jetson (GStreamer) — the WS-branch
  wiring and the frontend are verified by a **Jetson smoke deploy** (separate step):
  open `:8042`, toggle bargein / move the VAD + memory sliders / switch language /
  send a text message, confirm the robot's behavior changes and that the values
  survive a `docker restart`; confirm deferred controls render disabled.

## Risks / mitigations

- *ovs internal attribute names drift* (`_client_vad`, `session.max_input_tokens`):
  appliers are guarded and covered by tests against the names confirmed this session;
  a rename degrades to a no-op + log, not a crash.
- *History unit mismatch*: mapped honestly to the real token lever; relabel rather
  than fake if conversion proves too coarse.
- *Bind-mount path missing on a fresh Jetson*: Docker creates the host dir on first
  `up`; the app also `mkdir`s the in-container data dir defensively.
