# Reachy Claw

**Sub-200ms voice assistant for [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) — fully local pipeline powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [OpenClaw](https://github.com/ArturSkowronski/openclaw).**

[![CI](https://github.com/suharvest/reachy-claw/actions/workflows/ci.yml/badge.svg)](https://github.com/suharvest/reachy-claw/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Reachy Mini](https://img.shields.io/badge/robot-Reachy%20Mini-orange.svg)](https://www.pollen-robotics.com/reachy-mini/)
[![sherpa-onnx](https://img.shields.io/badge/speech-sherpa--onnx-green.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![OpenClaw](https://img.shields.io/badge/AI-OpenClaw-purple.svg)](https://github.com/ArturSkowronski/openclaw)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="media/hero.png" alt="Reachy Claw — talk to your robot" width="640" />
</p>

Talk to your robot, and it talks back — with emotions, head movements, and face tracking. No cloud, no subscription, everything runs on your own hardware.

Reachy Claw connects [OpenClaw](https://github.com/ArturSkowronski/openclaw) (AI brain) with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (voice engine) to give a [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) desktop robot the ability to have real conversations. You speak, it listens, thinks, and responds — all in under 200ms. Works on Jetson, RK3588, or any CUDA device you have.

### Latency breakdown (Jetson Orin NX, CUDA)

| Stage | Engine | Latency |
|-------|--------|---------|
| Speech detection (VAD) | Silero VAD | ~10ms |
| Speech-to-text | Paraformer streaming (sherpa-onnx) | ~50ms TTFT |
| Text-to-speech | Matcha-TTS + Vocos (sherpa-onnx) | ~60ms TTFT |
| **ASR + TTS combined** | | **~110ms** |

Full voice-to-voice latency depends on LLM inference time (not included above).

## Key Features

- **Fully local pipeline** — Paraformer ASR + Matcha TTS via sherpa-onnx, runs on Jetson / RK3588 / any CUDA device, no cloud required
- **OpenClaw integration** — streaming LLM responses, tool use, and multi-turn conversation via [OpenClaw](https://github.com/ArturSkowronski/openclaw) gateway
- **Emotion-driven motion** — 14 distinct emotions mapped to head movements and antenna expressions
- **Face tracking** — MediaPipe-powered gaze following so the robot looks at whoever is speaking
- **Streaming TTS** — sentence-level streaming for low-latency responses
- **Barge-in support** — interrupt the robot mid-sentence, just like a real conversation
- **Pluggable backends** — swap STT/TTS/VAD without changing code (Paraformer, Matcha, Whisper, ElevenLabs, Piper, and more)
- **Plugin architecture** — Motion, Conversation, and FaceTracker as independent plugins
- **Flexible deployment** — standalone mode, simulator, direct connection, or via Reachy Mini daemon

## Table of Contents

- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Edge Speech Service](#edge-speech-service)
- [Running as a Reachy Mini App](#running-as-a-reachy-mini-app)
- [Configuration](#configuration)
- [Speech Backends](#speech-backends)
- [Installation](#installation)
- [Scripts](#scripts)
- [OpenClaw Skill](#openclaw-skill-action-skill)
- [Development](#development)
- [Key Files](#key-files)
- [Acknowledgements](#acknowledgements)

## Quickstart

```bash
git clone https://github.com/suharvest/reachy-claw.git
cd reachy-claw
uv sync --extra dev --extra audio
uv run reachy-claw --gateway-host 127.0.0.1
```

With edge speech service (Jetson):

```bash
uv run reachy-claw \
  --stt paraformer-streaming \
  --tts matcha \
  --speech-url http://<jetson-ip>:8000
```

Standalone mode (no gateway, echoes what it heard):

```bash
uv run reachy-claw --standalone
```

Robot demo mode:

```bash
uv run reachy-claw --demo
```

## Architecture

```text
                        ┌──────────────────────┐
                        │    Local Hardware     │
                        └──────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│  ReachyClawApp (app.py)         All running locally              │
│  ├── MotionPlugin     — emotions, head tracking, idle anims     │
│  ├── FaceTrackerPlugin — MediaPipe face detection → HeadTarget  │
│  └── ConversationPlugin — STT → Gateway → TTS conversation loop │
│                                                                 │
│  Shared state:                                                  │
│  • HeadTargetBus  — fuses face/DOA/neutral head targets         │
│  • EmotionMapper  — 14 emotions, queue with debounce            │
│  • HeadWobbler    — speech-driven head micro-movements          │
└─────────────────────────────────────────────────────────────────┘
          │                │                        │
          ▼                ▼                        ▼
   Reachy Mini SDK   Edge Speech Service      OpenClaw Gateway
   (head, antennas)  (sherpa-onnx, CUDA)      (LLM, tools, streaming)
                     Paraformer ASR            WebSocket :18790
                     Matcha TTS + Vocos
```

Data flow:

```text
Microphone → VAD (Silero) → Speech detected?
  → STT (Paraformer streaming) → text
  → OpenClaw Gateway (WebSocket) → AI response (streaming)
  → Sentence splitter → TTS (Matcha, streaming) → Speaker
                       → EmotionMapper → Robot head/antennas

Camera → MediaPipe → HeadTarget → HeadTargetBus → Robot head
TTS audio → HeadWobbler → speech roll/pitch/yaw → Robot head

Barge-in: VAD detects speech during playback → interrupt TTS → new turn
```

## Edge Speech Service

The speech service is maintained as a standalone project: **[Jetson Voice](https://github.com/suharvest/jetson-local-voice)** — a Docker-based voice stack that runs Paraformer ASR and Matcha TTS on Jetson (or any CUDA device) with sherpa-onnx.

```bash
# On your edge device (e.g. Jetson Orin NX, JetPack 6.2)
git clone https://github.com/suharvest/jetson-local-voice.git
cd jetson-local-voice
docker compose build && docker compose up -d
curl http://localhost:8000/health
```

Then point reachy-claw at it:

```bash
uv run reachy-claw \
  --stt paraformer-streaming \
  --tts matcha \
  --speech-url http://<device-ip>:8000
```

See the [Jetson Voice README](https://github.com/suharvest/jetson-local-voice) for full API docs, benchmarks, model comparison, and patched sherpa-onnx details.

## Running as a Reachy Mini App

This project can run in two ways:

### Direct (development / standalone)

```bash
uv run reachy-claw --gateway-host 192.168.1.100
```

The app manages the Reachy Mini connection itself.

### Via Reachy Mini Daemon (production)

The project registers as a Reachy Mini app via the `reachy_mini_apps` entry point. Install it into the daemon's environment:

```bash
pip install /path/to/reachy-claw
```

Then start via the daemon API:

```bash
# List available apps
curl http://localhost:8000/apps/list-available

# Start
curl http://localhost:8000/apps/start-app/reachy_claw

# Stop
curl http://localhost:8000/apps/stop-current-app
```

Or run directly as a Reachy Mini app (daemon must be running):

```bash
python -m reachy_claw.reachy_app
```

In daemon mode, the Reachy Mini connection is managed by the daemon and passed to the app.

## Configuration

Configuration is layered (highest priority wins):

**CLI args > Environment variables > YAML config file > Defaults**

### YAML config file

Copy the example and edit:

```bash
cp reachy-claw.example.yaml reachy-claw.yaml
```

The app auto-detects config files in this order:
1. `./reachy-claw.yaml` or `./reachy-claw.yml` (current directory)
2. `~/.reachy-claw/config.yaml`

Or specify explicitly:

```bash
reachy-claw --config /path/to/config.yaml
# or
export REACHY_CLAW_CONFIG=/path/to/config.yaml
```

Example `reachy-claw.yaml` (edge deployment with Jetson):

```yaml
stt:
  backend: paraformer-streaming
  speech_service_url: http://192.168.1.50:8000

tts:
  backend: matcha
  speech_service_url: http://192.168.1.50:8000
  matcha:
    speaker_id: 0
    speed: 1.2

vad:
  backend: silero

behavior:
  wake_word: hey reachy
  play_emotions: true

vision:
  tracker: mediapipe
  camera_index: 0
```

See `reachy-claw.example.yaml` for the full list of 70+ options.

### Environment variables

| Variable | Description |
|---|---|
| `OPENCLAW_HOST` | Gateway host (default: `127.0.0.1`) |
| `OPENCLAW_PORT` | Gateway port (default: `18790`) |
| `OPENCLAW_TOKEN` | Gateway auth token |
| `OPENCLAW_PATH` | WebSocket path (default: `/desktop-robot`) |
| `STT_BACKEND` | STT backend (default: `paraformer-streaming`) |
| `TTS_BACKEND` | TTS backend (default: `matcha`) |
| `SPEECH_SERVICE_URL` | Remote speech service URL |
| `WHISPER_MODEL` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `WAKE_WORD` | Wake word to activate listening |
| `OPENCLAW_OPENAI_TOKEN` / `OPENAI_API_KEY` | OpenAI API key (for `--stt openai`) |
| `REACHY_CLAW_CONFIG` | Path to YAML config file |

Backend-specific env vars are auto-generated from each backend's `Settings` class (e.g. `MATCHA_SPEAKER_ID`, `MATCHA_SPEED`, `SENSEVOICE_LANGUAGE`).

ElevenLabs TTS:
- `REACHY_ELEVENLABS_API_KEY` or `ELEVENLABS_API_KEY` (required)
- `REACHY_ELEVENLABS_VOICE_ID` or `ELEVENLABS_VOICE_ID` (optional)
- `REACHY_ELEVENLABS_MODEL_ID` or `ELEVENLABS_MODEL_ID` (optional)
- `REACHY_ELEVENLABS_OUTPUT_FORMAT` or `ELEVENLABS_OUTPUT_FORMAT` (optional)

### CLI options

```text
-c, --config          Path to YAML config file
-v, --verbose         Debug logging
--gateway-host        OpenClaw host (default: 127.0.0.1)
--gateway-port        OpenClaw port (default: 18790)
--gateway-token       Auth token
--reachy-mode         auto | localhost_only | network
--stt                 (choices auto-detected from registry)
--whisper-model       tiny | base | small | medium | large
--tts                 (choices auto-detected from registry)
--tts-voice           Voice ID (backend-specific)
--tts-model           Model path (for Piper)
--speech-url          Remote speech service URL
--audio-device        Input device name
--vad                 silero | energy
--wake-word           Wake phrase
--no-emotions         Disable emotion animations
--no-idle             Disable idle animations
--no-barge-in         Disable barge-in
--no-face-tracking    Disable face tracking
--tracker-type        mediapipe | none
--camera-index        Camera device index
--standalone          Run without gateway
--demo                Run robot movement demo and exit
```

## Speech Backends

Backends are discovered automatically via the `@register_tts` / `@register_stt` / `@register_vad` decorators in `backend_registry.py`.

### STT backends

| Name | Type | Description |
|---|---|---|
| **`paraformer-streaming`** | Remote (sherpa-onnx) | **Default.** Streaming ASR, bilingual zh+en, ~50ms TTFT |
| `sensevoice` | Remote (sherpa-onnx) | Offline ASR, 5 languages (zh/en/ja/ko/yue) |
| `whisper` | Local | OpenAI Whisper |
| `faster-whisper` | Local | CTranslate2-optimized Whisper |
| `openai` | Cloud | OpenAI Whisper API |

### TTS backends

| Name | Type | Description |
|---|---|---|
| **`matcha`** | Remote (sherpa-onnx) | **Default.** Matcha-TTS + Vocos, best Chinese quality, ~60ms TTFT |
| `kokoro` | Remote (sherpa-onnx) | Kokoro TTS, multilingual |
| `elevenlabs` | Cloud | ElevenLabs API |
| `macos-say` | Local | macOS built-in `say` command |
| `piper` | Local | Piper neural TTS |
| `none` | -- | Dummy (prints text, no audio) |

### VAD backends

| Name | Type | Description |
|---|---|---|
| **`silero`** | Local (ONNX) | **Default.** Silero VAD, accurate speech detection |
| `energy` | Local | Simple RMS energy threshold |

### Adding a new backend

Create a class in `tts.py` or `stt.py` with the decorator -- that's it:

```python
@register_tts("my-backend")
class MyTTS(TTSBackend):
    """My custom TTS backend."""

    # Optional: declare backend-specific config fields
    class Settings:
        api_key: str = ""
        voice_type: str = "default"

    def __init__(self, base_url="http://localhost:8000", api_key="", voice_type="default"):
        self._base_url = base_url
        self._api_key = api_key
        self._voice_type = voice_type

    async def synthesize(self, text: str) -> str:
        # Call your API, return path to temp audio file
        ...
```

This automatically provides:
- `--tts my-backend` CLI option
- `tts.my_backend_api_key` / `tts.my_backend_voice_type` in YAML config
- `MY_BACKEND_API_KEY` / `MY_BACKEND_VOICE_TYPE` environment variables
- `config.my_backend_api_key` / `config.my_backend_voice_type` config attributes

## Installation

### Prerequisites

- Python 3.10+
- Reachy Mini SDK (`reachy-mini`)
- For edge speech: a CUDA device running the [Jetson Voice](https://github.com/suharvest/jetson-local-voice) speech service (see [Edge Speech Service](#edge-speech-service))

### Install the main app

```bash
uv sync
```

Development install:

```bash
uv sync --extra dev
```

### Optional extras

- local mic input: `uv sync --extra audio`
- local faster transcription: `uv sync --extra local-stt`
- OpenAI cloud transcription: `uv sync --extra cloud-stt`
- Reachy vision extras: `uv sync --extra vision`
- MediaPipe face tracking: `uv sync --extra mediapipe-vision`
- MuJoCo simulator: `uv sync --extra sim`

## Scripts

| Script | Purpose |
|---|---|
| `scripts/run_sim.sh` | Launch MuJoCo simulator + gateway check + start app (no face tracking) |
| `scripts/run_real.sh` | Pre-flight checks (gateway, Jetson, Reachy gRPC) + start with edge speech |
| `scripts/run_sim_daemon.py` | MuJoCo simulator daemon (macOS, GStreamer shim) |

## OpenClaw Skill (`action-skill/`)

The action skill provides tool wrappers for LLM-driven robot control via OpenClaw:

- connect/disconnect
- head movement + antenna movement
- emotions and dance
- image capture
- robot speech (TTS)
- status checks

Skill docs: `action-skill/SKILL.md`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv tool run ruff check .
```

Action skill tests:

```bash
cd action-skill
uv sync --extra dev
uv run pytest
```

## Key Files

| File | Purpose |
|---|---|
| `src/reachy_claw/main.py` | CLI entrypoint |
| `src/reachy_claw/app.py` | ReachyClawApp orchestrator |
| `src/reachy_claw/reachy_app.py` | Reachy Mini daemon app adapter |
| `src/reachy_claw/gateway.py` | OpenClaw WebSocket protocol |
| `src/reachy_claw/backend_registry.py` | Auto-discovery registry for STT/TTS/VAD backends |
| `src/reachy_claw/stt.py` | STT backend implementations |
| `src/reachy_claw/tts.py` | TTS backend implementations |
| `src/reachy_claw/vad.py` | VAD backend implementations |
| `src/reachy_claw/plugins/` | Motion, conversation, face tracker plugins |
| `src/reachy_claw/motion/` | EmotionMapper, HeadTargetBus, HeadWobbler |
| `src/reachy_claw/vision/` | MediaPipe face tracker |
| `src/reachy_claw/config.py` | Configuration (YAML + env + defaults) |
| `reachy-claw.example.yaml` | Example configuration file |
| `deploy/jetson/` | Symlink to [Jetson Voice](https://github.com/suharvest/jetson-local-voice) edge speech service |
| `action-skill/` | OpenClaw skill package |

## Acknowledgements

This project was originally based on [clawd-reachy-mini](https://github.com/ArturSkowronski/clawd-reachy-mini) by [Artur Skowronski](https://github.com/ArturSkowronski).

- [OpenClaw](https://github.com/ArturSkowronski/openclaw) -- AI gateway framework
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) -- speech inference engine (Paraformer ASR, Matcha TTS, Silero VAD)
- [Pollen Robotics](https://www.pollen-robotics.com/) -- Reachy Mini hardware and SDK
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) -- face detection and tracking
- [OpenAI Whisper](https://github.com/openai/whisper) -- speech-to-text
- [ElevenLabs](https://elevenlabs.io/) / [Piper](https://github.com/rhasspy/piper) -- text-to-speech engines
