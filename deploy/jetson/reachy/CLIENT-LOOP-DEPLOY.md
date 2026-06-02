# Client-loop reachy-claw deploy (SLV / ovs_agent)

Reproduce the client-loop migration deployment on a Jetson (`recomputer-desktop`).
The conversation + tool-calling brain runs via **ovs_agent** (`conversation_plugin_slv`);
the engine (speech service) stays a pass-through ASR/TTS; reachy's autonomous
plugins (motion / face_tracker / vision / rest / daily_log / dashboard) are unchanged.

## Artifacts
- **Image**: `sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v2`
  (full `ReachyClawApp`, CMD `reachy-claw -v`, `conversation_backend: slv` default, ovs_agent baked).
- **Build**: `deploy/jetson/reachy/Dockerfile.reachy-claw.slv` (original Dockerfile + ovs_agent wheel vendoring via `--find-links`). Wheel in `deploy/jetson/reachy/vendor/`.
  ```bash
  # refresh the ovs_agent wheel if seeed-local-voice/agent changed:
  (cd ../seeed-local-voice/agent && uv build) && cp ../seeed-local-voice/agent/dist/openvoicestream_agent-*.whl deploy/jetson/reachy/vendor/
  docker buildx build --builder multiarch --platform linux/arm64 --push \
    -f deploy/jetson/reachy/Dockerfile.reachy-claw.slv \
    -t sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v2 .
  ```
- **Device config**: `reachy-claw.jetson.slv.yaml` (conversation.backend=slv, server VAD silero, edge-llm Qwen3 + `/no_think`, engine :8621, daemon :38001).
- **Compose override**: `docker-compose.slv.yml`.

## Deploy (single-service swap, non-destructive)
```bash
# on the Jetson, in /home/recomputer/reachy-deploy/reachy/
tar czf /home/recomputer/backup-reachy-deploy-$(date +%s).tar.gz -C /home/recomputer/reachy-deploy reachy   # backup
docker pull sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v2
# place reachy-claw.jetson.slv.yaml + docker-compose.slv.yml in this dir, then:
docker compose -p reachy -f docker-compose.yml -f docker-compose.slv.yml up -d --no-deps reachy-claw
```
- **`--no-deps`** is required: the base `depends_on: reachy-daemon` would otherwise try to recreate the running daemon and abort on a name conflict.
- **Omit `docker-compose.dev.yml`**: its source bind-mount (`reachy-claw-src/src/reachy_claw`) would shadow the image's baked `clientloop`/`conversation_plugin_slv` code.
- The override uses an isolated `reachy-data-slv` data dir to avoid a stale `runtime-overrides.yaml` clobbering the config.

## Verify
```bash
docker logs reachy-claw | grep -E "ConversationPlugin\(SLV\)|SLV connected|Plugins:"
curl -s localhost:8640/health     # dashboard / app health
```
Full-turn smoke without a mic (dashboard WS text injection):
```bash
# send {"type":"send_message","text":"向左看"} to ws://localhost:8640/ws
# → expect on-device Qwen3 tool_call move_head → _cmd_move_head (no_robot if motors off) → engine TTS
```

## Motors / physical robot
`REACHY_MOTOR_ENABLED=false` by default — `recomputer-desktop` has no physical Reachy
(no USB-serial servo bus). Attach a Reachy Mini, set `REACHY_MOTOR_ENABLED=true`, and the
same `move_head`/`dance`/`play_emotion` tool handlers drive real `goto_target` calls.

## Rollback
```bash
# back to the prior image:
sed -i 's/:slv-v2/:clientloop-test/' docker-compose.slv.yml   # or restore a backup tar
docker compose -p reachy -f docker-compose.yml -f docker-compose.slv.yml up -d --no-deps reachy-claw
```
