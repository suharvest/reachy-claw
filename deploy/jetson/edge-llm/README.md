# edge-llm-chat-service (Jetson reference compose)

This directory vendors the reference `docker-compose.yml` for the external
[`edge-llm-chat-service`](https://github.com/seeed-studio) container so that
clawd-reachy-mini's Jetson deployment is reproducible without cloning the
upstream project. The image itself is **not** built from this repo — only the
compose + env template are mirrored here.

The service exposes an OpenAI-compatible HTTP endpoint that `reachy-claw`
(`llm.backend: edge_llm_v2v`) calls for chat completions.

## Quick start

```bash
cd deploy/jetson/edge-llm
cp .env.example .env
# edit .env to set EDGELLM_WORKSPACE to a writable directory on this host
docker compose pull
docker compose up -d edge-llm
docker compose logs -f edge-llm   # watch warmup
```

The container listens on host port **11435** (mapped to container `:8000`).
Port 11435 is intentional — 11434 is reserved for ollama on the same host.

## Key environment variables

| Var | Purpose |
|---|---|
| `EDGE_LLM_IMAGE` | Image reference. Override to use a local build. |
| `EDGELLM_ENGINE_REPO` | Hugging Face repo holding the pre-built TensorRT engine. |
| `EDGELLM_EXPECTED_ENGINE_COMMIT` | Engine commit pin — must match the artifacts. |
| `EDGELLM_EXPECTED_TENSORRT` | Required TensorRT version on the host. |
| `EDGELLM_WORKSPACE` | Persistent host path mounted to `/workspace`. Stores the downloaded engine. |
| `HF_ENDPOINT` | Set to `https://hf-mirror.com` for faster downloads inside China. |

## First-run notes

- On first start the container downloads ~3-5 GB of engine + tokenizer
  artifacts into `EDGELLM_WORKSPACE`. Plan disk + bandwidth accordingly.
- The Dockerfile sets `start-period=900s` on the healthcheck to cover the
  download + TensorRT warmup. Don't be alarmed if `docker ps` shows
  `(health: starting)` for ~10 min on a cold boot.
- After warmup, verify with:
  ```bash
  curl -s http://localhost:11435/v1/models | jq
  ```

## Tie-in with reachy-claw

`deploy/jetson/reachy-claw.jetson.yaml` is already pinned to
`edge_llm_url: http://localhost:11435`. No change needed there once this
compose is up.
