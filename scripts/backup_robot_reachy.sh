#!/usr/bin/env bash
# backup_robot_reachy.sh — NON-DESTRUCTIVE backup of the running robot before an
# SDK-upgrade redeploy. Captures the current images, a live snapshot of the
# running containers (incl. any ephemeral changes), the exact package versions,
# and the on-device deploy config. It NEVER stops, restarts, or recreates
# anything — the running environment is untouched.
#
# Output: a timestamped dir with image tarballs + a manifest + a restore note.
#
# Run ON the robot (192.168.1.113).
#
# Usage:
#   ./backup_robot_reachy.sh
#   OUT=/mnt/usb/reachy-backup DEPLOY_DIR=~/clawd/deploy/jetson/reachy ./backup_robot_reachy.sh

set -euo pipefail

CLAW="${CLAW:-reachy-claw}"
DAEMON="${DAEMON:-reachy-daemon}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$HOME/reachy-backup-$STAMP}"
DEPLOY_DIR="${DEPLOY_DIR:-}"

resolve() { docker ps --format '{{.Names}}' | grep -E "(^|[-_])$1([-_]|$)" | head -1; }
CLAW_C="$(resolve "$CLAW")"
DAEMON_C="$(resolve "$DAEMON")"

mkdir -p "$OUT"
echo "backup dir: $OUT"
MANIFEST="$OUT/manifest.txt"
{
  echo "Reachy robot backup — $STAMP"
  echo "host: $(hostname)   docker: $(docker --version)"
  echo
} > "$MANIFEST"

# ── free-space sanity (image saves can be several GB) ────────────────
echo "free space at $OUT:"; df -h "$OUT" | tee -a "$MANIFEST"

record_versions() {
  local c="$1" label="$2"
  [[ -z "$c" ]] && { echo "  $label: <container not running>" | tee -a "$MANIFEST"; return; }
  local img; img="$(docker inspect --format '{{.Config.Image}}' "$c" 2>/dev/null)"
  echo "  $label container=$c image=$img" | tee -a "$MANIFEST"
  docker exec "$c" python3 -c \
    'import reachy_mini,sys
mods=["reachy_mini","websockets","numpy","httpx"]
for m in mods:
    try:
        mod=__import__(m); print(f"    {m}={getattr(mod,\"__version__\",\"?\")}")
    except Exception as e:
        print(f"    {m}=<n/a {e}>")' 2>/dev/null | tee -a "$MANIFEST" || true
}

echo "── recording versions ──" | tee -a "$MANIFEST"
record_versions "$CLAW_C"   "client"
record_versions "$DAEMON_C" "daemon"

# ── 1. save the CURRENT images (the immutable rollback baseline) ─────
echo "── saving current images (docker save) ──"
save_image_of() {
  local c="$1" name="$2"
  [[ -z "$c" ]] && return
  local img; img="$(docker inspect --format '{{.Config.Image}}' "$c")"
  echo "  saving image $img → $OUT/image-$name.tar (this can take a while)…"
  docker save "$img" -o "$OUT/image-$name.tar"
  echo "image-$name.tar  ⇐  $img" >> "$MANIFEST"
}
save_image_of "$CLAW_C"   "claw"
save_image_of "$DAEMON_C" "daemon"

# ── 2. snapshot the RUNNING containers (captures ephemeral state) ────
# docker commit is read-only w.r.t. the running container; it just snapshots
# the writable layer (e.g. any prior in-container pip changes).
echo "── snapshotting running containers (docker commit) ──"
commit_snap() {
  local c="$1" name="$2"
  [[ -z "$c" ]] && return
  local tag="reachy-backup/$name:$STAMP"
  docker commit "$c" "$tag" >/dev/null
  docker save "$tag" -o "$OUT/snapshot-$name.tar"
  echo "snapshot-$name.tar  ⇐  live commit of $c  (tag $tag)" >> "$MANIFEST"
  echo "  $c → $tag"
}
commit_snap "$CLAW_C"   "claw"
commit_snap "$DAEMON_C" "daemon"

# ── 3. back up on-device deploy config + compose + runtime overrides ─
echo "── backing up deploy config ──"
if [[ -n "$DEPLOY_DIR" && -d "$DEPLOY_DIR" ]]; then
  tar czf "$OUT/deploy-config.tgz" -C "$DEPLOY_DIR" . 2>/dev/null \
    && echo "deploy-config.tgz  ⇐  $DEPLOY_DIR" >> "$MANIFEST" \
    && echo "  saved $DEPLOY_DIR"
else
  echo "  DEPLOY_DIR not set or missing — skipping compose/config tar."
  echo "  (set DEPLOY_DIR=/path/to/deploy/jetson/reachy to include it)"
  echo "deploy-config: SKIPPED (DEPLOY_DIR unset)" >> "$MANIFEST"
fi
# also grab any live runtime-overrides from the data dirs (best-effort)
for d in "$HOME"/reachy-data*/reachy-claw; do
  [[ -f "$d/runtime-overrides.yaml" ]] && cp "$d/runtime-overrides.yaml" \
     "$OUT/runtime-overrides.$(basename "$(dirname "$d")").yaml" 2>/dev/null || true
done

# ── restore note ─────────────────────────────────────────────────────
cat > "$OUT/RESTORE.md" <<EOF
# Restore from this backup ($STAMP)

## Roll back to the previous IMAGES (fast, normal rollback)
The original image tags are unchanged unless you overwrote them. If they're
gone, reload from the tarballs:

    docker load -i image-claw.tar
    docker load -i image-daemon.tar
    # then point docker-compose.slv.yml back at the original tag and:
    docker compose -p reachy -f docker-compose.yml -f docker-compose.slv.yml up -d --no-deps reachy-claw

## Restore the EXACT running state (incl. ephemeral changes)
    docker load -i snapshot-claw.tar     # tag: reachy-backup/claw:$STAMP
    docker load -i snapshot-daemon.tar   # tag: reachy-backup/daemon:$STAMP
    # run those tags in place of the normal images if needed.

## Versions captured: see manifest.txt
EOF

echo "── done ──"
echo "manifest:"; sed 's/^/   /' "$MANIFEST"
echo
echo "Backup complete (running containers untouched): $OUT"
echo "Contents:"; ls -lh "$OUT"
