# Changelog

All notable changes to **reachy-claw** are documented here. The image
tag (e.g. `v1.21`) refers to `sensecraft-missionpack.seeed.cn/solution/reachy-claw:<tag>`
used in `deploy/jetson/reachy/docker-compose.yml`.

## v1.21 — 2026-05-22

V2V (edge_llm_v2v) audio-pipeline reliability + dashboard prompt fixes.

### Fixed
- **Long-utterance TTS audio loss.** TTS bursts on Jetson out-pace the
  ALSA playback rate; the duplex stream's playback queue and the
  upstream `_audio_queue` both used drop-on-full, so multi-second
  utterances had scattered chunks silently dropped (manifested as
  garbled/missing speech, worse the longer the sentence).
  End-to-end backpressure now cascades from the ALSA period rate all
  the way up to the V2V WebSocket recv loop — no chunk loss.
- **TTS tail clipped on natural turn end.** `_on_v2v_tts_done` called
  `drain_playback()` immediately on tts_done, discarding everything
  still queued in the duplex stream. Replaced with a poll that waits
  for the queue to drain before tearing down. Barge-in/interrupt
  paths keep the immediate drain.
- **Dashboard `system_prompt` had no effect in edge_llm_v2v.** The
  EdgeLLMClient was hardcoded to `DEFAULT_SYSTEM_PROMPT` at startup
  and the dashboard's `set_prompt` hot-apply branch only matched
  `OllamaClient`. Startup now reads `config.ollama_system_prompt`
  (TOML `[llm] system_prompt`), and the hot-apply branch updates the
  running EdgeLLMClient + invalidates the TRT prefix-cache so the
  next turn picks up the new prompt without a restart.
- **Emotion tags leaking into TTS.** The per-delta
  `_EMOTION_RE.sub("", delta)` regex couldn't match tags split across
  streaming tokens (e.g. `[`, `curious`, `]` arrive in three deltas).
  Replaced with a stateful char-by-char buffer: `[…]` containing only
  `[\w_]+` is dropped, anything else (CJK, spaces, punctuation) is
  flushed as plain text.

### Reproducing v1.21

The image is built on Jetson (ARM64) from the project root:

```bash
# On Jetson (or any aarch64 host with Docker)
cd /path/to/reachy-claw
docker build -f deploy/jetson/Dockerfile.reachy-claw \
  -t sensecraft-missionpack.seeed.cn/solution/reachy-claw:v1.21 .
```

To pull and run with this version:

```bash
cd deploy/jetson/reachy
# docker-compose.yml is already pinned to v1.21
docker compose --profile openclaw up -d   # or your usual profile
```

The reachy-claw container expects:
- `deploy-speech-1` (OVS, jetson-v1.14-hotswap-20260522 or newer) at `localhost:8621`
- `edge-llm-chat-service` (Qwen3-4B-AWQ or compatible) at `localhost:11435`
- `reachy-daemon` at `localhost:38001`

See `deploy/jetson/reachy/docker-compose.yml` for the full stack.

## v1.20 — 2026-04-24

Baseline V2V + duplex-stream release. Client-side silero VAD gate,
duplex `sd.Stream` for USB AEC, sounddevice fallback for NO_MEDIA,
multi-turn abort recovery.

(Earlier history is in `git log`.)
