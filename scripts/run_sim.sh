#!/usr/bin/env bash
# Launch Reachy Mini simulator + conversation app in one shot.
# Gateway (OpenClaw) must already be running on :18790.
#
# Usage:
#   ./scripts/run_sim.sh              # MuJoCo GUI + Reachy Voice app
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
if lsof -i :"${SIM_API_PORT:-18222}" &>/dev/null 2>&1; then
  echo "⚠  Sim daemon may already be running (API port ${SIM_API_PORT:-18222} in use)"
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
SIM_PYTHON=".venv/bin/python"
if [ "$SIM_MODE" = "sim" ]; then
  SIM_PYTHON=".venv/bin/mjpython"
  if [ ! -x "$SIM_PYTHON" ]; then
    echo "MuJoCo GUI mode requires $SIM_PYTHON (install the simulator extra)."
    exit 1
  fi
fi
"$SIM_PYTHON" scripts/run_sim_daemon.py \
  --"$SIM_MODE" --localhost-only --no-media --autostart \
  --fastapi-port "$SIM_API_PORT" &
SIM_PID=$!

# Wait for the daemon HTTP API to be ready.
echo -n "Waiting for sim daemon"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$SIM_API_PORT/api/daemon/status" &>/dev/null; then
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
NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1" \
no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1" \
REACHY_DAEMON_HOST=localhost REACHY_DAEMON_PORT="$SIM_API_PORT" \
  .venv/bin/python scripts/run_sim_app.py "$@"
