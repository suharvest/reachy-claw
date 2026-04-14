#!/usr/bin/env bash
# Sync all deploy files and source code to Jetson.
# Usage: ./deploy/jetson/sync.sh [--restart]
#   --restart   also restart reachy-claw and vision-trt after sync
#
# Source code is mounted over the image via docker-compose.dev.yml,
# so rsync + restart is enough for quick iteration (no rebuild needed).

set -euo pipefail

JETSON="recomputer@100.67.111.58"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.dev.yml"

RESTART=false
[[ "${1:-}" == "--restart" ]] && RESTART=true

echo "=== Syncing to $JETSON ==="

# 1. Compose + config + dev override
rsync -avz -v \
  "$SCRIPT_DIR/reachy/docker-compose.yml" \
  "$SCRIPT_DIR/reachy/docker-compose.dev.yml" \
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

# 4. Skills (dereference symlinks with -L)
rsync -avzL --delete -v \
  "$PROJECT_ROOT/skills/" \
  "$JETSON:~/reachy-claw-src/skills/"

echo "=== Sync complete ==="

if $RESTART; then
  echo "=== Restarting containers (dev mode) ==="
  ssh "$JETSON" "cd ~/reachy-deploy/reachy && \
    $COMPOSE down --remove-orphans reachy-claw 2>&1 && \
    $COMPOSE --profile vision up -d vision-trt 2>&1 && \
    sleep 3 && \
    $COMPOSE up -d reachy-claw 2>&1"
  echo "=== Restart complete ==="
fi
