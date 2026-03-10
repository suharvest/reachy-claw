#!/usr/bin/env bash
# Launch Reachy Mini simulator + conversation app in one shot.
# Gateway (OpenClaw) must already be running on :18790.
#
# Usage:
#   ./scripts/run_sim.sh              # MuJoCo GUI + kokoro TTS + sensevoice STT
#   ./scripts/run_sim.sh --tts none   # no TTS (text-only pipeline test)
#   ./scripts/run_sim.sh --standalone # no gateway, echo-only mode
#   SIM_MODE=mockup-sim ./scripts/run_sim.sh  # headless mock (no GUI, for CI)
#
# Ctrl-C stops both processes.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Pre-flight checks ──────────────────────────────────────────────────

if ! lsof -i :18790 &>/dev/null; then
  echo "⚠  Gateway not detected on :18790"
  echo "   Start it first:  cd ~/project/openclaw && nvm use 22 && node scripts/run-node.mjs gateway"
  echo "   (continuing anyway — will fail on gateway connect)"
fi

# Check sim daemon isn't already running
if lsof -i :50051 &>/dev/null 2>&1; then
  echo "⚠  Sim daemon may already be running (gRPC port 50051 in use)"
  echo "   Kill it first or this script will fail to bind"
fi

# ── Cleanup on exit ────────────────────────────────────────────────────

SIM_PID=""
cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null && wait "$SIM_PID" 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Launch sim daemon (background) ─────────────────────────────────────

SIM_MODE="${SIM_MODE:-sim}"  # "sim" for MuJoCo GUI, "mockup-sim" for headless mock
SIM_API_PORT="${SIM_API_PORT:-18222}"
echo "Starting Reachy Mini simulator (mode: $SIM_MODE, API port: $SIM_API_PORT)..."
.venv/bin/mjpython scripts/run_sim_daemon.py \
  --"$SIM_MODE" --localhost-only --deactivate-audio \
  --fastapi-port "$SIM_API_PORT" &
SIM_PID=$!

# Wait for sim to be ready (gRPC on 50051)
echo -n "Waiting for sim daemon"
for i in $(seq 1 30); do
  if lsof -i :50051 &>/dev/null 2>&1; then
    echo " ready!"
    break
  fi
  echo -n "."
  sleep 0.5
done

if ! kill -0 "$SIM_PID" 2>/dev/null; then
  echo " FAILED (sim daemon exited)"
  exit 1
fi

sleep 1  # extra settle time for gRPC

# ── Launch reachy-claw (foreground) ──────────────────────────────

echo ""
echo "Starting conversation app..."
echo "  Press Ctrl-C to stop everything"
echo ""

# Use config file defaults; user can override via $@
# NOTE: do NOT use exec here — bash must stay alive for the EXIT trap
# to clean up the sim daemon background process.
uv run reachy-claw \
  --no-face-tracking \
  -v \
  "$@"
