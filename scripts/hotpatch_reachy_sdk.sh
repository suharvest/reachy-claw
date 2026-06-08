#!/usr/bin/env bash
# hotpatch_reachy_sdk.sh — in-place reachy-mini SDK bump on the running robot.
#
# ⚠️ READ THIS FIRST
# This is an EPHEMERAL hot-patch: it pip-installs reachy-mini==1.8.0 INSIDE the
# running client container. It is LOST the moment the container is recreated
# (docker compose up/pull, host reboot with --pull, image change). The PERMANENT
# fix is to rebuild the reachy-claw image with the updated pyproject and redeploy
# (see "PERMANENT PATH" at the bottom).
#
# Scope: only the CLIENT container (reachy-claw). The reachy-mini API used by the
# app is stable across 1.4->1.8, so the separate reachy-daemon container does NOT
# need to change as long as it is >=1.5 (WebSocket transport). verify_reachy_sdk.sh
# reports both versions — run it first and confirm before patching the client.
#
# Requires: the robot can reach a PyPI index that has arm64 wheels for
# reachy-mini 1.8.0 + gstreamer-bundle 1.28.3 (internal mirror or pypi.org).
#
# Usage (run ON the robot, 192.168.1.113):
#   ./hotpatch_reachy_sdk.sh                 # bump client to 1.8.0, restart, re-verify
#   TARGET=1.8.0 CLAW=reachy-claw ./hotpatch_reachy_sdk.sh
#   INDEX_URL=https://pypi.org/simple ./hotpatch_reachy_sdk.sh

set -euo pipefail

CLAW="${CLAW:-reachy-claw}"
TARGET="${TARGET:-1.8.0}"
INDEX_ARG=""
[[ -n "${INDEX_URL:-}" ]] && INDEX_ARG="--index-url ${INDEX_URL}"

here="$(cd "$(dirname "$0")" && pwd)"
resolve() { docker ps --format '{{.Names}}' | grep -E "(^|[-_])$1([-_]|$)" | head -1; }
CLAW_C="$(resolve "$CLAW")"
[[ -z "$CLAW_C" ]] && { echo "client container '$CLAW' not running"; exit 1; }
echo "client container: $CLAW_C   target: reachy-mini==$TARGET"

# ── 0. Baseline (so you can roll back / compare) ─────────────────────
echo "── baseline verify ──"
bash "$here/verify_reachy_sdk.sh" || echo "(baseline has warnings/failures — review before proceeding)"
OLD_VER="$(docker exec "$CLAW_C" python3 -c 'import reachy_mini;print(reachy_mini.__version__)' 2>/dev/null || echo '?')"
echo "current client reachy-mini: $OLD_VER   (rollback: pip install reachy-mini==$OLD_VER)"

read -r -p "Proceed to pip-install reachy-mini==$TARGET in $CLAW_C? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 0; }

# ── 1. Upgrade inside the client container ───────────────────────────
echo "── installing reachy-mini==$TARGET ──"
docker exec "$CLAW_C" python3 -m pip install --no-input $INDEX_ARG "reachy-mini==$TARGET"

NEW_VER="$(docker exec "$CLAW_C" python3 -c 'import reachy_mini;print(reachy_mini.__version__)' 2>/dev/null || echo '?')"
echo "client reachy-mini now: $NEW_VER"
[[ "$NEW_VER" != "$TARGET" ]] && { echo "✗ install did not land $TARGET (got $NEW_VER) — NOT restarting"; exit 1; }

# ── 2. Restart the client app so it re-imports the new SDK ───────────
# Per deploy convention: restart the container, do NOT `compose up` (that would
# recreate from the image and wipe this ephemeral pip change).
echo "── restarting $CLAW_C ──"
docker restart "$CLAW_C"
echo "waiting for app to come back…"; sleep 8

# ── 3. Re-verify (read-only) ─────────────────────────────────────────
echo "── post-patch verify ──"
bash "$here/verify_reachy_sdk.sh"

cat <<EOF

Done (EPHEMERAL). Client now on reachy-mini $NEW_VER.
Rollback:  docker exec $CLAW_C python3 -m pip install reachy-mini==$OLD_VER && docker restart $CLAW_C

PERMANENT PATH (recommended): rebuild + redeploy the image so 1.8.0 is baked in.
  # on a build host with the repo checked out (arm64/Jetson builder):
  docker build -f deploy/jetson/reachy/Dockerfile.reachy-claw.slv \\
    -t sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v8 .
  docker push  sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v8
  # then bump the image tag in docker-compose.slv.yml and on the robot:
  docker compose -p reachy -f docker-compose.yml -f docker-compose.slv.yml pull reachy-claw
  docker compose -p reachy -f docker-compose.yml -f docker-compose.slv.yml up -d --no-deps reachy-claw
EOF
