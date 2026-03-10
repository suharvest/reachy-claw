#!/usr/bin/env bash
# One-click launch for real Reachy Mini robot.
#
# Prerequisites:
#   1. Reachy Mini powered on and reachable (gRPC on :50051)
#   2. OpenClaw gateway running on :18790
#   3. Jetson speech service running on 100.67.111.58:8000
#
# Usage:
#   ./scripts/run_real.sh                    # default (paraformer streaming ASR + matcha TTS)
#   ./scripts/run_real.sh --tts none         # no TTS
#   ./scripts/run_real.sh --standalone       # no gateway, echo mode
#   SPEECH_URL=http://192.168.1.50:8000 ./scripts/run_real.sh  # custom Jetson IP
#
# Ctrl-C to stop (second Ctrl-C force-kills).

set -euo pipefail
cd "$(dirname "$0")/.."

SPEECH_URL="${SPEECH_URL:-http://100.67.111.58:8000}"
GATEWAY_PORT="${GATEWAY_PORT:-18790}"

# ── Pre-flight checks ──────────────────────────────────────────────────

echo "=== Reachy Mini (Real Robot) ==="

# Check gateway
if lsof -i :${GATEWAY_PORT} &>/dev/null; then
  echo "✓ Gateway detected on :${GATEWAY_PORT}"
else
  echo "⚠  Gateway not detected on :${GATEWAY_PORT}"
  echo "   Start it:  cd ~/project/openclaw && nvm use 22 && node scripts/run-node.mjs gateway"
fi

# Check Jetson speech service
if curl -s --connect-timeout 2 "${SPEECH_URL}/health" &>/dev/null; then
  echo "✓ Speech service reachable at ${SPEECH_URL}"
else
  echo "⚠  Speech service not reachable at ${SPEECH_URL}"
  echo "   Start it:  ssh recomputer 'cd jetson-voice && docker compose up -d'"
fi

# Check Reachy Mini (gRPC)
if lsof -i :50051 &>/dev/null 2>&1; then
  echo "✓ Reachy Mini gRPC detected on :50051"
else
  echo "⚠  Reachy Mini not detected on :50051 (will try network discovery)"
fi

echo ""

# ── Launch ──────────────────────────────────────────────────────────────

echo "Starting conversation app..."
echo "  STT: paraformer-streaming → ${SPEECH_URL}"
echo "  TTS: matcha → ${SPEECH_URL}"
echo "  Press Ctrl-C to stop"
echo ""

uv run reachy-claw \
  --stt paraformer-streaming \
  --tts matcha \
  --speech-url "${SPEECH_URL}" \
  -v \
  "$@"
