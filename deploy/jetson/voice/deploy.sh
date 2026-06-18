#!/usr/bin/env bash
# Guarded deploy for reachy-voice: pre-flight → build → recreate → verify.
#
# This is the ONLY blessed way to recreate the live voice container. It refuses
# to touch the running container unless the robot is actually reachable (see
# preflight.sh for why), and it verifies the result instead of assuming success.
#
# USAGE (on the robot, from the build dir that holds src/ + hf-hub):
#   bash deploy/jetson/voice/deploy.sh
#   FORCE=1 bash deploy/jetson/voice/deploy.sh   # skip the gate (NOT recommended)
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMPOSE="$HERE/docker-compose.yml"
CTR="reachy-voice"

echo "── 1/4 pre-flight gate ──────────────────────────────"
if [ "${FORCE:-0}" = "1" ]; then
  echo "FORCE=1 set — SKIPPING the hardware gate. You own the consequences."
else
  if ! bash "$HERE/preflight.sh"; then
    echo "Aborting deploy: pre-flight failed. The running container is untouched."
    exit 1
  fi
fi

echo "── 2/4 snapshot current state (for rollback) ────────"
PREV_IMG="$(docker inspect "$CTR" --format '{{.Config.Image}}' 2>/dev/null || echo none)"
echo "currently running image: $PREV_IMG"
echo "available reachy-voice tags: $(docker images reachy-voice --format '{{.Tag}}' | tr '\n' ' ')"

echo "── 3/4 build + recreate ─────────────────────────────"
docker compose -f "$COMPOSE" up -d --build

echo "── 4/4 verify (give the app time to connect) ────────"
NEW_IMG="$(awk -F'image:' '/image:[[:space:]]*reachy-voice/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$COMPOSE")"
sleep 12
STATUS="$(docker inspect "$CTR" --format '{{.State.Status}}' 2>/dev/null || echo missing)"
RESTARTS="$(docker inspect "$CTR" --format '{{.RestartCount}}' 2>/dev/null || echo '?')"
HAS_DATA="$(docker inspect "$CTR" --format '{{range .Mounts}}{{if eq .Destination "/data"}}yes{{end}}{{end}}' 2>/dev/null)"
HTTP="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' http://localhost:8042/ 2>/dev/null || echo 000)"

echo "container status : $STATUS  (restarts: $RESTARTS)"
echo "/data mounted    : ${HAS_DATA:-no}"
echo "dashboard :8042  : $HTTP"

if [ "$STATUS" = "running" ] && [ "$HTTP" = "200" ]; then
  echo -e "\033[32mDEPLOY OK — $NEW_IMG is up and serving.\033[0m"
  exit 0
fi
echo -e "\033[31mDEPLOY UNHEALTHY.\033[0m Recent logs:"
docker logs --tail 20 "$CTR" 2>&1 || true
echo
echo "Rollback: set 'image: $PREV_IMG' in $COMPOSE and run:"
echo "  docker compose -f $COMPOSE up -d   # no --build → reuses the old image"
exit 1
