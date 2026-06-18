#!/usr/bin/env bash
# Pre-flight hardware + daemon health gate for reachy-voice deploys.
#
# WHY THIS EXISTS
#   reachy-voice talks to the robot through the reachy-mini daemon. Any deploy
#   that restarts or recreates the container forces the app to RECONNECT to that
#   daemon. If the Reachy USB (motor on /dev/ttyACM0, audio card) is currently
#   down, that reconnect fails and the container drops into a crash-loop — even
#   though nothing in the deploy is wrong. (Seen 2026-06-17: the motor USB hub
#   died at 10:42; a 16:16 `compose up` then looked like "the deploy broke it"
#   when the hardware had been gone for 5.5h.) An old container can mask a dead
#   USB by holding a stale connection; a fresh one cannot.
#
#   So: NEVER restart/recreate the voice container without first proving the
#   robot is actually reachable. Run this gate; a non-zero exit means DO NOT
#   deploy — fix the hardware first.
#
# USAGE (on the robot)
#   bash deploy/jetson/voice/preflight.sh [image]
#     image  optional reachy-voice image tag for the live SDK probe;
#            default: read from this dir's docker-compose.yml.
#   echo $?   # 0 = safe to deploy, non-zero = abort
#
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
COMPOSE="$HERE/docker-compose.yml"
DAEMON_PORT="${REACHY_DAEMON_PORT:-38001}"
MOTOR_TTY="/dev/ttyACM0"
AUDIO_USB_ID="38fb:1001"   # Reachy USB audio (per robot-usb-devices-recovery)

# Image for the live SDK handshake probe (arg > compose > running container).
IMAGE="${1:-}"
[ -z "$IMAGE" ] && IMAGE="$(awk -F'image:' '/image:[[:space:]]*reachy-voice/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$COMPOSE" 2>/dev/null)"
[ -z "$IMAGE" ] && IMAGE="$(docker inspect reachy-voice --format '{{.Config.Image}}' 2>/dev/null)"

FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗ %s\033[0m\n' "$1"; FAIL=1; }
warn() { printf '  \033[33m! %s\033[0m\n' "$1"; }

echo "== reachy-voice pre-flight gate =="
echo "   image for probe: ${IMAGE:-<none found>}"

# ── 1. Motor controller serial — the exact thing whose absence crashes us ──
if [ -e "$MOTOR_TTY" ]; then
  ok "motor serial $MOTOR_TTY present"
else
  bad "motor serial $MOTOR_TTY MISSING — Reachy USB unplugged/powered-off. Re-seat the Reachy↔Jetson USB cable or power-cycle the robot, then re-run."
fi

# ── 2. Reachy USB audio card (mic + speaker) ──
if lsusb 2>/dev/null | grep -qi "$AUDIO_USB_ID"; then
  ok "audio USB $AUDIO_USB_ID present"
elif grep -qiE 'usb|reachy|uac' /proc/asound/cards 2>/dev/null; then
  ok "USB audio card present in /proc/asound/cards"
else
  bad "Reachy audio USB ($AUDIO_USB_ID) MISSING — mic/speaker will be dead"
fi

# ── 3. reachy-daemon container running (+ healthy if it reports health) ──
DSTATE="$(docker inspect reachy-daemon \
  --format '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{else}}nohealth{{end}}' 2>/dev/null)"
case "$DSTATE" in
  running/healthy|running/nohealth) ok "reachy-daemon container: $DSTATE" ;;
  running/*)                        bad "reachy-daemon unhealthy: $DSTATE" ;;
  "")                               bad "reachy-daemon container not found" ;;
  *)                                bad "reachy-daemon not running: $DSTATE" ;;
esac

# ── 4. Daemon HTTP answers on its port ──
HTTP="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' "http://localhost:${DAEMON_PORT}/" 2>/dev/null || echo 000)"
if [ "$HTTP" = "200" ]; then
  ok "daemon HTTP :${DAEMON_PORT} → 200"
else
  bad "daemon HTTP :${DAEMON_PORT} → $HTTP"
fi

# ── 5. Decisive: a REAL SDK handshake that STAYS UP ──
#   HTTP 200 is not enough — the daemon can accept the SDK websocket and then
#   drop it ~5s later when the motor bus is gone. Connect exactly like the app
#   (ReachyMini → ws://localhost:PORT/ws/sdk, no_media) and require it to hold.
if [ "$FAIL" -eq 0 ] && [ -n "$IMAGE" ]; then
  PROBE=$(docker run --rm --network host --entrypoint python3 "$IMAGE" -c "
import sys, time
try:
    from reachy_mini import ReachyMini
except Exception as e:
    print('INCONCLUSIVE import:', e); sys.exit(50)
try:
    rm = ReachyMini(host='localhost', port=${DAEMON_PORT}, request_media_backend='no_media')
except TypeError as e:
    print('INCONCLUSIVE signature:', e); sys.exit(50)
except Exception as e:
    print('CONNECT_FAILED:', type(e).__name__, e); sys.exit(1)
# Held the connection open; make sure it does not drop in the first seconds
# (the motor-gone failure closed it ~5s in).
time.sleep(6)
try:
    rm.__exit__(None, None, None) if hasattr(rm,'__exit__') else None
except Exception:
    pass
print('HANDSHAKE_OK'); sys.exit(0)
" 2>&1 | grep -vE 'gstreamer|Reachy USB card not found|INFO|Warning|Deprecat' | tail -3)
  RC=${PIPESTATUS[0]:-$?}
  case "$RC" in
    0)  ok "daemon SDK handshake held ≥6s (robot reachable)" ;;
    50) warn "SDK probe inconclusive (tooling): $PROBE — relying on checks 1–4" ;;
    *)  bad "daemon SDK handshake FAILED: $PROBE" ;;
  esac
elif [ -z "$IMAGE" ]; then
  warn "no image found for SDK probe — relying on checks 1–4"
else
  warn "skipped SDK probe (a hard check above already failed)"
fi

echo "=================================="
if [ "$FAIL" -eq 0 ]; then
  echo -e "\033[32mPRE-FLIGHT OK — safe to deploy.\033[0m"
  exit 0
else
  echo -e "\033[31mPRE-FLIGHT FAILED — do NOT restart/recreate the voice container.\033[0m"
  echo "Fix the hardware/daemon above first; a deploy now would only crash-loop."
  exit 1
fi
