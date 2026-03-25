# Daily Diary Feature Design

## Overview

Extend the Emotion Mirror Dashboard with a daily diary page. A scheduled task collects the day's interaction data, an OpenClaw skill invokes LLM to generate a first-person narrative diary, and the Dashboard displays it with date switching. Reachy Mini can narrate the diary aloud while dynamically controlling the Dashboard UI, with barge-in support for user questions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Scheduled Task (cron / OpenClaw scheduler)             │
│  └─ OpenClaw "daily-diary" skill                        │
│     ├─ 1. Run data collector script (Python)            │
│     ├─ 2. Send structured data to LLM                   │
│     └─ 3. Store diary JSON file                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Storage: ~/.reachy-claw/diaries/YYYY-MM-DD.json        │
└─────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
┌───────────────────┐    ┌────────────────────────┐
│  Dashboard API    │    │  Narration Mode        │
│  GET /api/diaries │    │  WS: narrate_control   │
│  GET /api/diary/  │    │  TTS + UI sync         │
│  {date}           │    │  Barge-in + Q&A        │
└───────────────────┘    └────────────────────────┘
```

## Data Flow

### 1. Data Collection

A Python script collects the day's data from existing sources:

- **Emotion log**: Timestamped emotion values from robot_state broadcasts (already in memory during runtime; needs to be persisted to a daily log file)
- **Conversation log**: ASR final transcripts + LLM responses (already logged or can be captured from WS events)
- **Face/smile stats**: Face detection counts, smile capture counts and paths
- **LLM thoughts**: Thought card texts from monologue/conversation mode

Future (optional, gracefully skipped if unavailable):
- **Home Assistant sensors**: Temperature, humidity, weather, location via HA REST API
- **Recamera / Rerouter**: Image and audio analysis summaries

The collector outputs a structured JSON blob of raw daily data.

### 2. Diary Generation (OpenClaw Skill)

An OpenClaw skill `daily-diary`:
- Triggered by scheduler (e.g., daily at 23:00) or manually
- Calls the data collection script
- Sends the collected data + a system prompt to LLM
- System prompt configurable (default: warm first-person narrative style)
- LLM returns a **section-structured** diary (not plain text)

### 3. Storage Format

One JSON file per day: `~/.reachy-claw/diaries/YYYY-MM-DD.json`

```json
{
  "version": 1,
  "date": "2026-03-25",
  "generated_at": "2026-03-25T23:05:00+08:00",
  "title": "A Day of Many Smiles",
  "sections": [
    {
      "id": "summary",
      "type": "narrative",
      "content": "Today was a lively day..."
    },
    {
      "id": "mood_curve",
      "type": "chart",
      "content": "My emotions went on quite a journey today...",
      "data": [{"t": "09:00", "v": 55}, {"t": "10:00", "v": 72}, ...]
    },
    {
      "id": "conversations",
      "type": "highlights",
      "content": "I had 8 conversations today. Here are the memorable ones...",
      "items": [
        {"time": "14:30", "user": "What do you think about art?", "reply": "...", "emotion": "curious"}
      ]
    },
    {
      "id": "faces",
      "type": "stats",
      "content": "42 people stopped by today, and 15 of them smiled at me!",
      "data": {"faces_seen": 42, "smiles_captured": 15, "peak_hour": "15:00"}
    },
    {
      "id": "smile_wall",
      "type": "gallery",
      "content": "Here are some of the smiles I collected...",
      "items": [{"path": "captures/2026-03-25/001.jpg", "time": "10:15"}, ...]
    },
    {
      "id": "thoughts",
      "type": "highlights",
      "content": "Some things I found myself thinking about...",
      "items": [{"time": "11:00", "text": "...", "emotion": "contemplative"}]
    },
    {
      "id": "environment",
      "type": "sensors",
      "content": "The room was a comfortable 24 degrees...",
      "data": {"temperature": 24.5, "humidity": 60, "weather": "sunny", "location": "Shenzhen"}
    }
  ],
  "meta": {
    "data_sources": ["emotion_log", "asr_log", "face_stats", "llm_thoughts"],
    "llm_model": "dashscope/kimi-k2.5",
    "prompt_version": "v1"
  }
}
```

Key design decisions:
- **Section-based structure** enables narration mode to map speech to UI regions
- Each section has an `id` (anchor for UI navigation), `type` (rendering hint), `content` (narrative text), and optional `data`/`items`
- `environment` section is optional — gracefully omitted when HA is not configured
- `version` field for future schema evolution

## Dashboard Changes

### Navigation

Add a top-level tab bar to `index.html`:

```
[  LIVE  ]  [  DIARY  ]
```

- **LIVE**: Current Emotion Mirror (default)
- **DIARY**: Daily diary viewer

Implemented as two `<div>` containers with CSS `display: none/block` toggling. No router needed — keep it vanilla JS.

### Diary Page Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ◀  2026-03-24  │  2026-03-25 (Today)  │  ▶               │
│                  date navigation bar                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "A Day of Many Smiles"          🔊 Narrate         │   │
│  │  March 25, 2026                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Summary ──────────────────────────────────────────┐   │
│  │  Today was a lively day...                          │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Mood ─────────────────────────────────────────────┐   │
│  │  [Line chart: emotion over time]                    │   │
│  │  "My emotions went on quite a journey..."           │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Conversations ────────────────────────────────────┐   │
│  │  Chat bubble snippets                               │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Faces ────────────────────────────────────────────┐   │
│  │  [42 people] [15 smiles] [Peak: 3pm]               │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Smile Wall ───────────────────────────────────────┐   │
│  │  [Photo grid with staggered reveal]                 │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Reflections ──────────────────────────────────────┐   │
│  │  Thought cards                                      │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Environment (optional) ───────────────────────────┐   │
│  │  24.5°C  60% humidity  ☀ Sunny  📍 Shenzhen        │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Styling

- Consistent with existing dark theme (#000 bg, #A3E635 accent, Outfit font)
- Sections use card-style containers with subtle borders
- Smooth scroll-into-view animations for narration focus
- Active section highlight (glow border) during narration

### Backend API (new endpoints in dashboard_plugin.py)

```
GET /api/diaries          → {"dates": ["2026-03-25", "2026-03-24", ...]}
GET /api/diary/2026-03-25 → {diary JSON}
GET /api/diary/latest     → redirect to most recent diary
```

Reads from `~/.reachy-claw/diaries/` directory. No database.

## Narration Mode

### Concept

Reachy Mini becomes a presenter. The Dashboard is its screen. When narration starts:

1. Dashboard enters "narration mode" (dimmed non-active sections, spotlight on current)
2. Robot reads each section's `content` via TTS
3. Dashboard scrolls to and highlights the corresponding section
4. Between sections, brief pause + transition animation

### WebSocket Protocol

New message types on the existing Dashboard WS:

**Client → Server:**
```json
{"type": "diary_narrate_start", "date": "2026-03-25"}
{"type": "diary_narrate_stop"}
```

**Server → Client (from narration engine):**
```json
{"type": "diary_narrate_focus", "section_id": "mood_curve", "state": "speaking"}
{"type": "diary_narrate_focus", "section_id": "mood_curve", "state": "done"}
{"type": "diary_narrate_navigate", "action": "switch_date", "date": "2026-03-24"}
{"type": "diary_narrate_navigate", "action": "switch_tab", "tab": "live"}
{"type": "diary_narrate_end"}
```

### Barge-in and Q&A

Leverages existing barge-in infrastructure:

1. User speaks during narration → ASR captures question
2. Narration pauses (TTS stops)
3. Question + current diary context sent to LLM
4. LLM responds with:
   - Answer text (TTS'd by Reachy Mini)
   - Optional navigation command (e.g., `switch_date`, `focus_section`)
5. Dashboard reacts to navigation commands
6. After answer, narration can resume or user continues conversation

The LLM receives the full diary JSON as context, so it can answer questions like:
- "How many people did you see yesterday?" → switch to yesterday + focus faces section
- "Tell me more about that conversation" → focus conversations section
- "What was your happiest moment?" → focus mood curve peak

### LLM Tool/Function for UI Control

During narration Q&A, the LLM has access to a tool:

```json
{
  "name": "dashboard_navigate",
  "parameters": {
    "action": "focus_section | switch_date | switch_tab",
    "section_id": "string (optional)",
    "date": "string (optional)",
    "tab": "live | diary (optional)"
  }
}
```

This gets translated to WebSocket commands by the narration engine.

## Daily Data Logging (Prerequisite)

Current runtime data lives only in memory. For diary generation, we need persistent daily logs.

### New: DailyLogPlugin

A lightweight plugin that subscribes to existing events and appends to daily log files:

```
~/.reachy-claw/daily-logs/YYYY-MM-DD/
  emotions.jsonl     # {ts, emotion, value} per state broadcast (sampled, not every 50ms)
  conversations.jsonl # {ts, role, text, emotion} per ASR final / LLM end
  faces.jsonl         # {ts, count, names} per face detection event (sampled)
  thoughts.jsonl      # {ts, text, emotion} per LLM thought
```

Sampling strategy:
- Emotions: Log once per minute (or on change)
- Faces: Log once per minute (or on new face)
- Conversations/thoughts: Log every event (low volume)

This plugin runs silently alongside existing plugins. The data collection script reads these logs.

## OpenClaw Skill

### Skill Definition

```yaml
name: daily-diary
description: Generate Reachy Mini's daily diary from interaction logs
trigger: scheduled (23:00 daily) or manual
```

### Skill Flow

1. Execute data collection script: `python collect_daily_data.py --date today`
2. Script outputs structured JSON to stdout
3. Skill sends to LLM with diary generation prompt
4. LLM returns section-structured diary
5. Skill saves to `~/.reachy-claw/diaries/YYYY-MM-DD.json`

### Diary Generation Prompt (default, configurable)

```
You are Reachy Mini, a small humanoid robot. Write your daily diary based on the following interaction data.

Structure your diary into sections. Each section should have a narrative paragraph written in first person, warm and reflective tone.

Sections to include:
- summary: Overall day summary (2-3 sentences)
- mood_curve: Commentary on your emotional journey today
- conversations: Pick 2-3 most interesting conversations
- faces: Reflect on the people you met
- thoughts: Share your deeper reflections

Output as JSON matching the diary schema.
```

## File Changes Summary

### New Files
- `src/reachy_claw/plugins/daily_log_plugin.py` — Event logger plugin
- `src/reachy_claw/plugins/dashboard_static/diary.js` — Diary page JS
- `src/reachy_claw/plugins/dashboard_static/diary.css` — Diary page styles
- `scripts/collect_daily_data.py` — Data collection script
- OpenClaw skill definition (in openclaw extensions)

### Modified Files
- `src/reachy_claw/plugins/dashboard_static/index.html` — Add tab nav + diary page container
- `src/reachy_claw/plugins/dashboard_static/app.js` — Tab switching logic, narration WS handlers
- `src/reachy_claw/plugins/dashboard_static/style.css` — Tab nav styles, narration mode styles
- `src/reachy_claw/plugins/dashboard_plugin.py` — New API endpoints, narration WS messages
- `src/reachy_claw/app.py` — Register DailyLogPlugin

### Not Modified
- Existing Dashboard functionality (zero regression risk)
- OpenClaw core framework

## Implementation Order

1. **DailyLogPlugin** — Start logging daily data (prerequisite for everything)
2. **Data collection script** — Read logs, output structured JSON
3. **Diary JSON generation** — LLM integration (can test with mock data first)
4. **Dashboard diary page** — UI rendering + date switching
5. **Dashboard API endpoints** — Serve diary data
6. **Narration mode** — WS protocol + UI sync
7. **Barge-in Q&A** — LLM tool for dashboard navigation
8. **OpenClaw skill** — Wrap it all into a scheduled skill

## Testing Strategy

- **DailyLogPlugin**: Unit tests for event sampling, file rotation
- **Data collection**: Unit tests with fixture log files
- **Dashboard API**: Integration tests for endpoints
- **Diary page**: Manual testing with mock diary JSON files
- **Narration mode**: Manual E2E testing with robot
