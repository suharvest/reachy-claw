# AGENTS.md — reachy-voice

Guidance for AI agents working in **this** repository. This is the **reachy-voice
exhibition voice-companion app** for a **Reachy Mini Lite** running on a Jetson.
It is an **on-robot Python app** built on the **Reachy Mini Python SDK ≥ 1.8.0**
and the **ovs_agent** (OpenVoiceStream) framework.

> This is **not** a JavaScript / Web / WebRTC app and not a fresh scaffold. The
> upstream SDK `AGENTS.md` describes a JS "golden path" for new web apps — that
> path does **not** apply here. Only the SDK facts in *SDK reference* below carry
> over. Don't introduce JS apps, WebRTC, `mountHost()`, HF Spaces, or the
> `reachy-mini-app-assistant` scaffolding into this repo.

## What this repo is

- `src/reachy_voice/` — **the app** (the only thing that ships):
  - `main.py` — entry point (`ReachyVoiceApp`, a `ReachyMiniApp` subclass) + the dashboard HTTP/WS server + `/debug/*` endpoints.
  - `conversation.py` — `ConversationEngine` wrapping ovs_agent `CompanionRobotApp` (SLV ASR/TTS + edge-LLM + tools).
  - `motion.py` — 25 Hz single-writer compositor: gaze tracking, speech wobble, idle presence, and official recorded-move expressions.
  - `attention.py` + `gaze.py` — engagement-gated face tracking; `FaceGaze` projects pixels→yaw/pitch via SDK camera intrinsics.
  - `vision.py` — ZMQ consumer of the vision-trt face/emotion stream.
  - `audio.py` — duplex `sounddevice` I/O (subclasses ovs_agent `AudioIO`).
  - `dashboard.py`, `config.py`, `tier_a.py` (dashboard API endpoints).
- `legacy/reachy_claw/` — the **retired** old app (a full plugin platform on the old SDK with SDK media monkey-patching). **Not packaged, not imported.** Kept only as a reference to port features from (diary, skills, Home Assistant, voice-cloning, conversation modes, multi-LLM backends). Do not build on it.
- `deploy/jetson/` — per-component deploy. `voice/` is this app's `Dockerfile` + `docker-compose.yml` + `entrypoint.sh`; siblings: `vision-trt/`, `reachy/`, `edge-llm/`, …
- `tests/voice/` — the active test suite. The `reachy_claw` tests under `tests/` root target legacy code and are **not run** (CI is scoped to `tests/voice`).

## Runtime architecture (5 containers on the Jetson)

The app **orchestrates**; it never touches hardware directly — everything goes
through the daemon and the service containers:

| Container | Port(s) | Role |
|---|---|---|
| **reachy-daemon** | `:38001` | Official `reachy_mini` daemon — drives motors, exposes the SDK websocket (`/ws/sdk`, `/api/state/full`). **Not our code; don't modify.** |
| **vision-trt** | `:8630` HTTP, `:8631` ZMQ | Camera → TensorRT face/emotion → publishes faces over ZMQ (msgpack, topic `vision`). The app consumes `:8631`. |
| **deploy-speech (SLV)** | `:8621` | ASR + TTS over one WebSocket (V2V). |
| **edge-llm-chat-service** | `:11435` | Local LLM, OpenAI-compatible. |
| **reachy-voice** | `:8042` | **This app** (dashboard + orchestration). |

Data flow:
- **listen→think→speak:** mic → SLV(ASR) → edge-LLM → SLV(TTS) → speaker
- **see→track:** camera → vision-trt → ZMQ → `attention`/`gaze` → SDK → motors
- **emote:** LLM reply `[emotion]` tags → `motion` plays the official move libraries

## Build / test / lint

```bash
uv sync --extra dev
uv run pytest tests/voice -v      # active suite; some tests skip off-Jetson
uv run ruff check .
```

- Python **≥ 3.11** (CI matrix 3.11/3.12).
- The app imports GStreamer (`gi`) transitively via the SDK, so **`import reachy_voice.main` does not work on a plain dev box** (no `gi`/PortAudio) — real verification happens **on the Jetson**. Tests guard SDK/GStreamer/PortAudio imports and skip when absent.

## Deploy (live exhibition — follow exactly)

- The app is **baked into a Docker image** built `FROM reachy-claw:slv-v7`, run via `deploy/jetson/voice/docker-compose.yml` (host net, privileged, `restart: unless-stopped`). **Build context = repo root**; the Dockerfile COPYs `src/reachy_voice`, `hf-hub/`, and the entrypoint.
- **`hf-hub/`** (~107 MB official emotion/dance move cache) must be **pre-staged** in the build context. It is **gitignored — never commit it**.
- **Hot-patch** one file: `docker cp` into the container + `docker restart reachy-voice` (Python needs restart; static files under `reachy_voice/static/` just need a hard browser reload). Hot-patches are **lost on image rebuild** — the durable source of truth is the repo.
- Do **not** `docker compose up` in a way that recreates from a stale image and wipes hot-patches. On the live robot, prefer a targeted restart.
- **Code delivery:** feature branch + **PR to `suharvest/reachy-claw`** (origin, over **ssh**). Never local-merge to master. Push over **ssh** — the `gh` https token lacks the `workflow` scope, so workflow-file changes are rejected over https.

## SDK conventions this app relies on (Python SDK ≥ 1.8.0)

- **Connect:** `ReachyMini(..., request_media_backend="no_media")`. The app runs **no_media** — the camera is owned by vision-trt and audio by our duplex `sounddevice` stream. **Do not** use the SDK media/camera/audio path.
- **Daemon is on `:38001`** here (not the SDK default `:8000`).
- **Continuous motion (25 Hz compositor):** `set_target_head_pose(create_head_pose(...))`, `set_target_body_yaw(rad)`, `set_target_antenna_joint_positions([right, left] in rad)`; call `enable_motors()` once on start (1.8.0 requires torque on for `set_target_*`).
- **Expressions:** `play_move(move, sound=False)` from the official recorded-move libraries (`pollen-robotics/reachy-mini-emotions-library`, `reachy-mini-dances-library`, baked into the image's HF cache).
- Rule of thumb: `goto_target()` for one-shot smooth gestures (≥ 0.5 s); `set_target_*` for real-time loops (10 Hz+).

## Hard constraints

- **Live exhibition robot** — keep it running. Verify changes via the app's `/debug/{motion,vision,gaze,emotion,say,face,state}` endpoints and the dashboard; don't make the user repeatedly interact to test.
- The robot must **track while listening**, not freeze; the hardware handles mic noise.
- **Motion kill-switch:** env `REACHY_MOTION=0` or file `/tmp/reachy_motion_off` disables **all** motion at startup (read once, survives `docker restart`). If the robot is silent, check for it.
- Never access the audio/reSpeaker USB from outside the app while the app is running.

## SDK reference (upstream)

- Official SDK + app docs: <https://github.com/pollen-robotics/reachy_mini>. Its `AGENTS.md` targets **new JS/Web apps** — not applicable here beyond these facts.
- **Hardware:** head = 6-DOF Stewart platform, body = 1-DOF rotation, 2 antennas (usable as buttons). SDK-enforced safety ranges (auto-clamped): head pitch/roll **±40°**, head yaw **±180°**, body yaw **±160°**, max head–body yaw delta **65°**. (This app clamps more conservatively: head ±20°, body ±40°.)
- Daemon REST/interactive docs: `http://localhost:38001/docs`.
