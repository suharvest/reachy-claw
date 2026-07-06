# Deployment state audit — SLV exhibition Reachy (192.168.1.113)

**Snapshot: 2026-06-18.** Point-in-time comparison of what actually runs on the
robot vs what this repo declares. Purpose: make divergence visible so each
workstream's owner can pull its deployment definition into version control.
This document changes nothing on the robot and commits no other team's work.

## Container ↔ repo divergence

| Container | Running image (created) | Repo declares | State |
|---|---|---|---|
| `reachy-voice` | `reachy-voice:v0.4.0` (06-18 03:44) | `deploy/jetson/voice/docker-compose.yml` → `v0.4.0` | ✅ **In sync** — container source is byte-identical to `master` (sha256-verified 06-18). |
| `deploy-speech-1` | `seeed-local-voice:prod-unified-v8-moss` (06-18 03:49) + bind-mount `~/moss_tts_nano_patched.py` → `voxedge/.../moss_tts_nano.py` | `deploy/jetson/voice/docker-compose.slv-backend.yml` → `jetson-v1.16-race-fixes-slim` (3 wks old) | ⚠️ **Diverged — active workstream (NOT captured here).** MOSS TTS timbre/speed work; see below. |
| `edge-llm-chat-service` | `edge-llm-chat-service:v0.8.0-gdn-mtp-merged` (06-18 02:07) | `deploy/jetson/edge-llm/docker-compose.yml` → `${EDGE_LLM_IMAGE:-…:latest}` (tag not pinned) | ⚠️ Rebuilt 06-18; specific tag not pinned in repo. Likely a separate active workstream. |
| `reachy-daemon` | `…/reachy-daemon:v1.0` (04-21) | `deploy/jetson/reachy/docker-compose.yml` → `v1.0` | ✅ Tag matches. NOTE: PR #10 added `Dockerfile.daemon` pinning `reachy-mini==1.8.1`; that image has **not** been rebuilt/deployed (daemon 1.x interops with app 1.8.x fine — see memory `robot-usb-devices-recovery`). |
| `vision-trt` | `…/vision-trt:v2.1` (04-21) | `deploy/jetson/reachy/docker-compose.yml` → `v2.1` | ✅ Tag matches; full build context **is** tracked at `deploy/jetson/vision-trt/`. Running image predates the `capture.py` camera-race self-heal (merged in PR #9, not yet rebuilt). Currently healthy. |

The **only** git-external file bind-mounted into any container is the MOSS TTS
patch above. No other container carries an ad-hoc hot-patch.

## What this repo already captures (no action needed)

- `reachy-voice` app: source (`src/reachy_voice/`), `Dockerfile`, compose (with the
  `/data` overrides bind-mount), and the guarded `preflight.sh` / `deploy.sh`.
- `vision-trt`: full build context — `Dockerfile`, `src/`, `static/`, entrypoint.
- `reachy-daemon`: `Dockerfile.daemon` (+ PR #10's `reachy-mini==1.8.1` pin),
  composes, vendor wheels.
- Host integration: `deploy/jetson/udev/` rules, `deploy/jetson/kiosk/`,
  systemd units.

## Diverged-and-deliberately-not-captured (other workstreams)

These run on the robot but are **not** in git. They belong to active workstreams
owned by other developers; capturing them here would freeze a moving target and
step on that work. Their owners should commit them when they settle.

### MOSS TTS backend (`deploy-speech-1`) — voice timbre
- Image `seeed-local-voice:prod-unified-v8-moss` (repo declares `jetson-v1.16-…`).
- Live hot-patch `~/moss_tts_nano_patched.py` over the voxedge MOSS backend; was
  being edited as recently as 06-18 16:16 (container restarted ~30s later).
- Supporting on-robot scratch: `patch_moss_speed{,2,3}.py`, `tts_speed_test.py`,
  `moss_wrapper_unit.py`, `add_voice_config_env.sh`, `moss-v8-build/`,
  `moss-mix1-slim.tar` (1.1 GB model); image tags `prod-unified-v9..v14` + `…-moss`.
- Backend env of note: `TTS_DEFAULT_SID=0`, `TTS_DEFAULT_SPEED=1.2`, MOSS TTS.

> **⚠️ Risk for the SLV/TTS owner:** none of this is in git. If `deploy-speech`
> is ever recreated from `docker-compose.slv-backend.yml` (which points at the
> 3-week-old `jetson-v1.16` image), all MOSS work is lost. Recommend the owner
> commits the patched module + an updated compose/image reference once stable.

### edge-llm (`edge-llm-chat-service`)
- Running `v0.8.0-gdn-mtp-merged`, rebuilt 06-18 02:07; repo leaves the tag as an
  env-overridable `latest`. If this is a settled artifact, its owner should pin
  the tag in `deploy/jetson/edge-llm/docker-compose.yml`.

## Boundary note (reachy-voice ↔ MOSS)

The `reachy-voice` app and the MOSS timbre work do not interact at the code or
git level: PR #9 touched no SLV/TTS file; the runtime overrides
(`overrides.py`) carry no voice/timbre/speed/SID parameter; the dashboard
"restart services" action targets only `vision-trt`/`reachy-daemon`/`reachy-voice`
(never `deploy-speech`). `reachy-voice` is purely a client of the SLV backend
over `:8621` — it sends text + language and consumes whatever timbre the backend
is configured with.
