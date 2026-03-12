#!/usr/bin/env bash
# Sync all deploy files and source code to Jetson.
# Usage: ./deploy/jetson/sync.sh [--restart]
#   --restart   also restart reachy-claw and vision-trt after sync

set -euo pipefail

JETSON="recomputer@100.67.111.58"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESTART=false
[[ "${1:-}" == "--restart" ]] && RESTART=true

echo "=== Syncing to $JETSON ==="

# 1. Compose + config
rsync -avz -v \
  "$SCRIPT_DIR/reachy/docker-compose.yml" \
  "$SCRIPT_DIR/reachy/reachy-claw.jetson.yaml" \
  "$JETSON:~/reachy-deploy/reachy/"

# 2. Vision-TRT source
rsync -avz --delete -v \
  "$SCRIPT_DIR/vision-trt/src/" \
  "$JETSON:~/reachy-deploy/vision-trt/src/"

# 3. Reachy-claw source
rsync -avz --delete -v \
  --exclude='__pycache__' \
  "$PROJECT_ROOT/src/reachy_claw/" \
  "$JETSON:~/reachy-claw-src/src/reachy_claw/"

echo "=== Sync complete ==="

if $RESTART; then
  echo "=== Restarting containers ==="
  # Stop reachy-claw first (releases /dev/video0 for vision-trt)
  ssh "$JETSON" "cd ~/reachy-deploy/reachy && \
    docker compose stop reachy-claw 2>&1 && \
    docker compose restart vision-trt 2>&1 && \
    sleep 3 && \
    docker compose start reachy-claw 2>&1"
  echo "=== Restart complete ==="
fi
