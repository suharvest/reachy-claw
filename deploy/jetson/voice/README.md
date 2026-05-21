# seeed-local-voice (Jetson reference compose)

This directory vendors the reference `docker-compose.yml` for the
[`seeed-local-voice`](https://github.com/Seeed-Studio) (a.k.a. OpenVoiceStream)
container so the Jetson voice stack is reproducible from this repo. The image
itself is **not** built here — only the compose + env template are mirrored.

The container exposes the V2V (voice-to-voice) WebSocket endpoint that
clawd-reachy-mini's `ConversationPlugin` (`backend: edge_llm_v2v`) consumes
at `ws://localhost:8621/v2v/stream`.

## Quick start

```bash
cd deploy/jetson/voice
cp .env.example .env
# Generate a fresh admin key:
#   sed -i "s|REPLACE_WITH_RANDOM_HEX|$(openssl rand -hex 16)|" .env
docker compose pull   # or `docker compose build` if you build locally
docker compose up -d speech
docker compose logs -f speech
```

Host port **8621 -> container 8000**. `deploy/jetson/reachy-claw.jetson.yaml`
already targets `ws://localhost:8621/v2v/stream`.

## Profile

`jetson-qwen3asr-matcha-nx` is the verified Orin NX profile:

- ASR: Qwen3 streaming (cuda backend)
- TTS: Matcha (cuda backend, multilingual)
- VAD: Silero, 400 ms tail
- Memory cap: 6 GB

Other profiles (e.g. `jetson-multilang-highperf-nx`) are listed in the
upstream `seeed-local-voice` repo. Don't mix profiles with the wrong
hardware tier — Orin AGX has its own variants.

## First-run notes

- Model artifacts (~3.2 GB: Qwen3 ASR + Matcha TTS + Silero VAD) download on
  first start into the `speech-models` named volume and persist across restarts.
- Inside China, keep `HF_ENDPOINT=https://hf-mirror.com` to avoid HF rate
  limits.
- If you have pre-staged artifacts, mount them at `/opt/models` instead of
  using the named volume.
- Healthy startup ends with the WebSocket handshake on `/v2v/stream`
  accepting; verify with:
  ```bash
  curl -s http://localhost:8621/health
  ```

## Tie-in with reachy-claw

`deploy/jetson/reachy-claw.jetson.yaml` is already pinned to
`v2v.url: ws://localhost:8621/v2v/stream`. No change needed there once
this compose is up and the model files are loaded.
