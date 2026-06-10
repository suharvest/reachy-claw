#!/usr/bin/env bash
# verify_reachy_sdk.sh — robot-side diagnostic for the reachy-mini SDK upgrade.
#
# READ-ONLY by default: reports the reachy-mini version in each container,
# daemon reachability, client<->daemon connectivity, and the robot data loop
# (sensor read). Pass --motion to additionally run a tiny, gentle head nod as
# a live actuation test (the robot WILL move ~10 degrees).
#
# Run ON the robot (192.168.1.113). Safe to run before AND after an upgrade to
# compare. Exit code is non-zero if any critical check fails.
#
# Usage:
#   ./verify_reachy_sdk.sh                  # read-only checks
#   ./verify_reachy_sdk.sh --motion         # + gentle head-nod actuation test
#   CLAW=reachy-claw DAEMON=reachy-daemon PORT=38001 ./verify_reachy_sdk.sh

set -uo pipefail

CLAW="${CLAW:-reachy-claw}"
DAEMON="${DAEMON:-reachy-daemon}"
PORT="${PORT:-38001}"
DO_MOTION=0
[[ "${1:-}" == "--motion" ]] && DO_MOTION=1

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
hr()    { printf -- '────────────────────────────────────────\n'; }

fail=0
note_fail() { red "  ✗ $*"; fail=1; }
note_ok()   { green "  ✓ $*"; }

# Resolve a container even if the project prefixes names (e.g. reachy-reachy-claw-1)
resolve() {
  local want="$1"
  docker ps --format '{{.Names}}' | grep -E "(^|[-_])${want}([-_]|$)" | head -1
}

hr; echo "Reachy SDK diagnostic  ($(date '+%F %T'))"; hr

CLAW_C="$(resolve "$CLAW")"
DAEMON_C="$(resolve "$DAEMON")"
echo "client container : ${CLAW_C:-<not found>}"
echo "daemon container : ${DAEMON_C:-<not found>}"
[[ -z "$CLAW_C"   ]] && note_fail "client container '$CLAW' not running"
[[ -z "$DAEMON_C" ]] && note_fail "daemon container '$DAEMON' not running"

# ── reachy-mini version in each container ────────────────────────────
ver_in() { docker exec "$1" python3 -c \
  'import reachy_mini,sys; sys.stdout.write(getattr(reachy_mini,"__version__","?"))' 2>/dev/null; }

hr; echo "reachy-mini versions"
CLAW_VER="";  DAEMON_VER=""
if [[ -n "$CLAW_C"   ]]; then CLAW_VER="$(ver_in "$CLAW_C")";     echo "  client : ${CLAW_VER:-<import failed>}"; fi
if [[ -n "$DAEMON_C" ]]; then DAEMON_VER="$(ver_in "$DAEMON_C")"; echo "  daemon : ${DAEMON_VER:-<import failed>}"; fi
if [[ -n "$CLAW_VER" && -n "$DAEMON_VER" ]]; then
  if [[ "$CLAW_VER" == "$DAEMON_VER" ]]; then note_ok "client and daemon SDK versions match ($CLAW_VER)"
  else yellow "  ⚠ client ($CLAW_VER) and daemon ($DAEMON_VER) differ — OK if both >=1.5 (WebSocket), but watch for a version-mismatch warning in logs"; fi
fi

# ── transport deps (client) ─────────────────────────────────────────
if [[ -n "$CLAW_C" ]]; then
  hr; echo "client transport deps"
  docker exec "$CLAW_C" python3 -c \
    'import websockets,numpy,httpx;print("  websockets",websockets.__version__);print("  numpy",numpy.__version__);print("  httpx",httpx.__version__)' 2>/dev/null \
    || note_fail "could not import websockets/numpy/httpx in client"
fi

# ── daemon FastAPI/WebSocket reachable on PORT ───────────────────────
hr; echo "daemon reachability on :$PORT"
if curl -fsS --max-time 3 "http://localhost:$PORT/" >/dev/null 2>&1; then
  note_ok "daemon HTTP responds on :$PORT"
else
  # fall back to a raw TCP probe from inside the client container
  if [[ -n "$CLAW_C" ]] && docker exec "$CLAW_C" python3 -c \
      "import socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(s.connect_ex(('localhost',$PORT)))" 2>/dev/null; then
    note_ok "daemon TCP port :$PORT open (HTTP curl unavailable)"
  else
    note_fail "daemon NOT reachable on :$PORT"
  fi
fi

# ── client<->daemon data loop (read-only sensor read) ────────────────
if [[ -n "$CLAW_C" ]]; then
  hr; echo "data loop: connect + read current head pose (read-only)"
  docker exec "$CLAW_C" python3 - "$PORT" <<'PY'
import sys
port = int(sys.argv[1])
try:
    from reachy_mini import ReachyMini
    with ReachyMini(host="localhost", port=port, connection_mode="localhost_only",
                    media_backend="no_media") as r:
        pose = r.get_current_head_pose()
        ant  = r.get_present_antenna_joint_positions()
        import numpy as np
        assert np.asarray(pose).shape == (4, 4), f"bad pose shape {np.asarray(pose).shape}"
        print("  ✓ connected; head pose 4x4 OK; antennas =", list(ant))
except Exception as e:
    print("  ✗ data-loop read failed:", repr(e)); sys.exit(1)
PY
  [[ $? -ne 0 ]] && note_fail "client could not read sensor state from daemon"
fi

# ── optional gentle actuation test (robot moves!) ────────────────────
if [[ $DO_MOTION -eq 1 && -n "$CLAW_C" ]]; then
  hr; yellow "actuation test: gentle head nod (robot WILL move ~10°)"
  docker exec "$CLAW_C" python3 - "$PORT" <<'PY'
import sys, time
port = int(sys.argv[1])
try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    with ReachyMini(host="localhost", port=port, connection_mode="localhost_only",
                    media_backend="no_media") as r:
        r.enable_motors()
        for pitch in (10, -10, 0):
            r.goto_target(head=create_head_pose(roll=0, pitch=pitch, yaw=0, degrees=True), duration=0.4)
            time.sleep(0.5)
        print("  ✓ nod completed without error")
except Exception as e:
    print("  ✗ actuation failed:", repr(e)); sys.exit(1)
PY
  [[ $? -ne 0 ]] && note_fail "actuation test failed"
fi

# ── recent client errors ─────────────────────────────────────────────
if [[ -n "$CLAW_C" ]]; then
  hr; echo "recent client errors (last 200 log lines, grep)"
  docker logs --tail 200 "$CLAW_C" 2>&1 | grep -iE "error|traceback|failed|exception|mismatch" | tail -12 \
    || echo "  (none)"
fi

hr
if [[ $fail -eq 0 ]]; then green "RESULT: all critical checks passed"; else red "RESULT: one or more critical checks FAILED"; fi
exit $fail
