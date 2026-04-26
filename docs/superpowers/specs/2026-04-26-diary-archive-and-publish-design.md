# Diary Archive & Static Site Publishing Design

**Date:** 2026-04-26
**Status:** Approved (pending user review)
**Supersedes (partial):** [2026-03-25-daily-diary-design.md](./2026-03-25-daily-diary-design.md) — replaces the jsonl-based logging and adds a publishing pipeline. The dashboard, narration, and barge-in features from the prior spec remain unchanged.

## Goals

1. **Persist daily interaction data** (ASR transcripts with timestamps, emotions, faces/smiles, thoughts, weather) in a queryable, long-term-friendly store so the daily diary generator can use richer history.
2. **Generate a daily diary as a Markdown document** that is suitable for both the local dashboard and external publishing.
3. **Publish each day's diary to an existing public Hugo site** via GitHub + GitHub Actions, with content frozen on a per-day basis and form (templates/CSS) iterated independently.

## Non-Goals

- Recording or archiving raw audio waveforms. Only ASR transcripts + timestamps are persisted.
- Re-generating historical diaries when templates change. Each day's Markdown is immutable once published.
- Backfilling diaries for days before this feature ships.
- Search, tagging, or RSS in the first iteration (Hugo can add later without changing source format).

## Architecture

```
┌────────────────────────── Jetson (clawd-reachy-mini) ──────────────────────────┐
│                                                                                │
│  Runtime plugins ─────────► SQLite (~/.reachy-claw/reachy.db)                  │
│  (ASR, emotions, faces,    asr_events / emotions / faces / thoughts / weather  │
│   thoughts, weather)       diaries                                             │
│                                                                                │
│                            ▲                                                   │
│                            │                                                   │
│  Daily 23:00 OpenClaw skill `daily-diary`                                      │
│   1. query day's data ─────┘                                                   │
│   2. LLM → Markdown (with front matter)                                        │
│   3. write `diaries` row + export .md                                          │
│   4. publish_diary.py → git push to site repo                                  │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────── GitHub: <TBD: site repo> ─────────────────────────────────┐
│  content/<TBD: diary section path>/2026-04-26.md                              │
│  static/captures/2026-04-26/*.jpg                                             │
│                                                                               │
│  GitHub Actions (on push to that path):                                       │
│   - hugo --minify                                                             │
│   - deploy to gh-pages → GitHub Pages                                         │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Data Layer: SQLite

### Why SQLite

- Single-file, no daemon, fits Jetson's footprint.
- Long-term scaling: years of events query in milliseconds with an index on `ts`.
- Replaces the jsonl design from the prior spec — one source of truth, easier to query for diary generation, easier to back up.

### Schema

File: `~/.reachy-claw/reachy.db`. Created on first run if absent. Schema versioned via `PRAGMA user_version`.

```sql
CREATE TABLE asr_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,           -- unix epoch seconds
  role TEXT NOT NULL,            -- 'user' | 'reachy'
  text TEXT NOT NULL,
  emotion TEXT
);
CREATE INDEX idx_asr_events_ts ON asr_events(ts);

CREATE TABLE emotions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  value REAL NOT NULL,
  label TEXT
);
CREATE INDEX idx_emotions_ts ON emotions(ts);

CREATE TABLE faces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  count INTEGER NOT NULL,
  smile_count INTEGER DEFAULT 0,
  capture_path TEXT
);
CREATE INDEX idx_faces_ts ON faces(ts);

CREATE TABLE thoughts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  text TEXT NOT NULL,
  emotion TEXT
);
CREATE INDEX idx_thoughts_ts ON thoughts(ts);

CREATE TABLE weather (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  temp_c REAL,
  humidity REAL,
  condition TEXT,                -- 'sunny', 'cloudy', etc.
  location TEXT
);
CREATE INDEX idx_weather_ts ON weather(ts);

CREATE TABLE diaries (
  date TEXT PRIMARY KEY,         -- 'YYYY-MM-DD'
  markdown TEXT NOT NULL,        -- full Markdown with front matter
  generated_at INTEGER NOT NULL,
  llm_model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  published_at INTEGER           -- NULL until pushed to site repo
);
```

### Storage Module

New file: `src/reachy_claw/storage/db.py`

Responsibilities:
- Open/initialize the DB (create tables, set `user_version`).
- Provide thin write helpers: `record_asr(ts, role, text, emotion)`, `record_face(...)`, etc.
- Provide read helpers for the diary collector: `events_for_day(date) -> dict[table, list[row]]`.
- Provide diary CRUD: `save_diary(date, markdown, ...)`, `get_diary(date)`, `mark_published(date)`.

Connection model: one `sqlite3.Connection` per process, opened with `check_same_thread=False` and serialized via a lock OR each plugin creates its own connection (sqlite handles concurrent writers via WAL mode). **Decision:** enable `PRAGMA journal_mode=WAL` and let each plugin open its own connection — simplest correct approach.

### Sampling

Same strategy as prior spec, now applied to SQLite inserts:
- `emotions`: insert at most once per minute, or on label change.
- `faces`: insert at most once per minute, or when a new face appears or a smile is captured.
- `asr_events`: insert every event (low volume, full fidelity needed for diary).
- `thoughts`: insert every event.
- `weather`: insert every poll (e.g., hourly from Home Assistant or external API).

### Plugin changes

`DailyLogPlugin` is rewritten to write SQLite via `storage/db.py` instead of jsonl files. The existing event subscriptions stay the same. Other plugins (`conversation_plugin`, `face_tracker_plugin`, `motion_plugin`) gain a small hook to call `db.record_*` at appropriate points — no behavioral changes, just an additional sink.

The dashboard's existing diary endpoints are updated to read from the `diaries` SQLite table instead of `~/.reachy-claw/diaries/*.json`. The dashboard front-end (diary.js / diary.css) stays as-is — only the data source changes. The on-disk JSON files from the prior design are deprecated; a one-time migration script imports any existing files into the `diaries` table.

### Maintenance

- WAL checkpoint on each diary-skill run.
- Monthly `VACUUM` (cron in OpenClaw or manual).
- Backup: SQLite file is rsync'd to host weekly (out of scope for this spec; documented as ops note).

## Diary Generation

### Trigger

OpenClaw skill `daily-diary` runs at 23:00 local time daily (existing scheduler). Manual invocation supported via skill CLI.

### Pipeline

1. Read all rows from SQLite where `ts` falls in the target day's local-time window.
2. Build a structured JSON blob (the same shape as the prior spec's `sections`, but consumed directly by the LLM rather than stored).
3. Call LLM (`dashscope/kimi-k2.5` via OpenClaw) with the diary prompt + structured data.
4. LLM returns **Markdown with Hugo front matter** (not JSON — Markdown is the storage and publish format).
5. Save the Markdown to `diaries.markdown`. Set `generated_at`, `llm_model`, `prompt_version`.
6. Hand off to `publish_diary.py` (next section).

### Markdown Format

```markdown
---
title: "A Day of Many Smiles"
date: 2026-04-26
weather:
  condition: "sunny"
  temp_c: 24.5
  humidity: 60
  location: "Shenzhen"
stats:
  faces_seen: 42
  smiles: 15
  conversations: 8
captures:
  - path: "captures/2026-04-26/001.jpg"
    time: "10:15"
  - path: "captures/2026-04-26/002.jpg"
    time: "14:22"
meta:
  llm_model: "dashscope/kimi-k2.5"
  prompt_version: "v1"
---

## 今天的心情

今天是热闹的一天……

## 遇到的人

今天有 42 个人来到我面前……

## 想到的事

……
```

Front matter carries structured data (renderable by Hugo templates as cards/widgets). Body is the LLM's first-person narrative. Section headings are fixed (`今天的心情` / `遇到的人` / `想到的事` and any future additions); the LLM is instructed to use these exact headings so templates can rely on them.

### Privacy / Content Rules

The system prompt explicitly requires:
- Reachy narrates in first person, in a warm reflective tone.
- **Never quote user speech verbatim.** Paraphrase what people said and what was discussed.
- Never include personally identifying details from ASR (names, addresses, phone numbers) unless they were already public context (e.g., a public guest's name announced in conversation).
- Smile/face captures may be included as image references, since these are explicit interactions where users smiled at the robot.

Raw ASR transcripts remain in `asr_events` on the Jetson and are never sent to the site repo.

### Immutability

- Once `diaries.published_at IS NOT NULL`, the row is treated as frozen. Re-runs of the skill for the same date are skipped by default.
- Manual override: `--force` flag deletes the row and pushes a fresh Markdown (used only for fixing errors).
- Template/CSS changes affect rendering but never re-generate historical Markdown. New front-matter fields appear only in newer diaries; templates use Hugo's `with` to skip when absent.

## Publishing Pipeline

### Site Repo

The site is an existing Hugo project on GitHub. Final values to be filled in at implementation time:

- **Site repo URL:** `<TBD: provided by user>`
- **Content path for diary section:** `<TBD: provided by user>` (e.g., `content/diary/`)
- **Static asset path for captures:** `static/captures/` (default; adjust if site convention differs)
- **GitHub Actions branch for Pages deploy:** typically `gh-pages` (defer to existing site convention)

These are the only project-specific values; everything else in the design is generic.

### Authentication: Deploy Key

A new SSH deploy key is generated and added to the site repo's GitHub deploy keys (with **write access** so the Jetson can push). The private key lives at `~/.ssh/diary_publish_ed25519` on the Jetson with `chmod 600`. SSH config:

```
# ~/.ssh/config
Host github-diary-site
  HostName github.com
  User git
  IdentityFile ~/.ssh/diary_publish_ed25519
  IdentitiesOnly yes
```

The Jetson clones via `git@github-diary-site:<owner>/<repo>.git`, isolating this key from any other GitHub identity on the host.

### publish_diary.py

New script: `scripts/publish_diary.py`

Inputs: `--date YYYY-MM-DD` (default: today). Behavior:

1. Read `diaries.markdown` for the date from SQLite. If `published_at` is set and `--force` not passed, exit cleanly with "already published".
2. Ensure local clone exists at `~/.reachy-claw/site-repo/`. Clone on first run; otherwise `git pull --rebase`.
3. Write Markdown to `<repo>/<diary content path>/YYYY-MM-DD.md`.
4. Copy referenced capture images from `~/.reachy-claw/captures/YYYY-MM-DD/` to `<repo>/static/captures/YYYY-MM-DD/`.
5. `git add` the new files, `git commit -m "diary: 2026-04-26"`, `git push`.
6. On success, set `diaries.published_at = now()`.
7. On failure (network, push rejection, etc.), log error and exit non-zero. Diary row in SQLite stays unpublished — the next daily run, or a manual retry, will try again. Do **not** roll back the Markdown row; only retry the push.

The script is invoked as the last step of the `daily-diary` OpenClaw skill. It can also be run manually for backfill of a specific day.

### GitHub Actions

The site repo's existing Hugo build workflow handles deployment. **No new workflow file is created in this spec** — we rely on the site repo's current setup. Implementation step will verify:

- Workflow triggers on push to the diary content path (or to `main`).
- Builds with Hugo and deploys to Pages.
- If the existing workflow does not cover the diary path or needs adjustment, it is updated in the site repo (out of clawd-reachy-mini's tree).

## Dashboard Impact

Minimal. The dashboard's diary page reads from the `diaries` table now; the dashboard endpoints (`/api/diaries`, `/api/diary/{date}`, `/api/diary/latest`) keep their contract (return JSON shape compatible with the prior spec — derived from the Markdown and front matter). Front-end code is untouched.

Narration mode (from prior spec) continues to work: it reads diary content from the dashboard endpoints, which now serve the Markdown-derived structure. The narration WS protocol is unchanged.

## File Changes

### New

- `src/reachy_claw/storage/__init__.py`
- `src/reachy_claw/storage/db.py` — SQLite open, schema init, read/write helpers
- `src/reachy_claw/storage/migrations.py` — schema versioning
- `scripts/publish_diary.py` — push to site repo
- `scripts/migrate_jsonl_to_sqlite.py` — one-time import of any prior jsonl/diary JSON
- `docs/ops/diary-publish-setup.md` — deploy key generation, SSH config, first-run instructions

### Modified

- `src/reachy_claw/plugins/daily_log_plugin.py` — write SQLite instead of jsonl
- `src/reachy_claw/plugins/conversation_plugin.py` — call `db.record_asr(...)`
- `src/reachy_claw/plugins/face_tracker_plugin.py` — call `db.record_face(...)`
- `src/reachy_claw/plugins/dashboard_plugin.py` — diary endpoints read from SQLite
- OpenClaw `daily-diary` skill — call publish step after generation

### Removed (deprecated, after migration)

- `~/.reachy-claw/daily-logs/*.jsonl` (kept on disk for one release cycle, then deleted)
- `~/.reachy-claw/diaries/*.json` (replaced by `diaries` table)

## Implementation Order

1. **`storage/db.py` + schema + tests** — pure unit-testable layer.
2. **Plugin wiring** — `DailyLogPlugin` and other plugins call `db.record_*`. Run alongside dashboard end-to-end to confirm data lands.
3. **Migration script** — import any existing jsonl/JSON to SQLite.
4. **Dashboard endpoint update** — switch to SQLite reads. Verify dashboard still renders correctly.
5. **Diary skill: Markdown output** — adapt LLM prompt to emit Markdown + front matter; save to `diaries`.
6. **`publish_diary.py`** — develop against a throwaway test repo first, then point at the real site repo.
7. **OpenClaw skill integration** — chain generate + publish in the daily skill.
8. **Site repo template** — add a Hugo template that renders the diary front matter (stats card, capture gallery, weather widget) — this work happens in the site repo, not clawd-reachy-mini.

## Testing

- `storage/db.py`: unit tests for schema init, sampling logic, day-window queries.
- Plugin integration: spin up the app with a temp DB path, fire mock events, assert rows.
- Migration script: feed fixture jsonl/JSON files, assert SQLite contents match.
- Dashboard endpoints: integration tests against a seeded test DB.
- `publish_diary.py`: integration test against a local bare git repo (no network needed) to verify clone/pull/commit/push semantics.
- Diary skill: manual E2E once on the Jetson with the real site repo, behind a `--dry-run` first.

## Open Questions / TBD

These are documented as values to provide at implementation kickoff, not blockers for the design:

1. **Site repo URL** — to be supplied by user.
2. **Diary content path within the site** — e.g., `content/diary/`, `content/journal/`, etc. — to be supplied by user.
3. **Whether the existing site Hugo workflow needs adjustment** — verify at integration step.

## Risks

- **Push conflicts** — if anything else writes to the same site repo, the Jetson's local clone may diverge. Mitigation: `git pull --rebase` before commit; on conflict, abort and alert.
- **LLM occasionally quoting ASR verbatim despite the prompt** — Mitigation: post-generation linter that flags any substring of an ASR event found verbatim in the Markdown body, blocking publish until reviewed.
- **SQLite file growth** — at expected event rates (low thousands/day), tens of MB per year. Negligible. Document the monthly VACUUM as ops hygiene.
- **Deploy key compromise** — limited blast radius (one repo, write only). Rotation procedure documented in `docs/ops/diary-publish-setup.md`.
