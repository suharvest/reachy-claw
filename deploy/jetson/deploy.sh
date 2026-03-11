#!/usr/bin/env bash
# One-click deploy: reachy-claw stack to Jetson.
#
# Two modes:
#   ./deploy.sh                    # Ollama mode (default, no OpenClaw)
#   ./deploy.sh --openclaw         # OpenClaw mode (starts gateway container)
#
# Prerequisites:
#   - Speech service already running on Jetson (port 8621)
#   - SSH access to Jetson (key-based auth recommended)
#   - Ollama mode: Ollama installed on Jetson (ollama.com)
#   - OpenClaw mode: configure LLM after deploy with configure-llm.sh
#
# Options:
#   --openclaw          Deploy with OpenClaw gateway
#   --setup-openclaw    Run OpenClaw first-time setup (extension install + config)
#   JETSON_HOST=x       Custom SSH host (default: recomputer)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Config ────────────────────────────────────────────────────────────
JETSON_HOST="${JETSON_HOST:-recomputer}"
JETSON_USER="${JETSON_USER:-recomputer}"
DEPLOY_DIR="${DEPLOY_DIR:-~/reachy-deploy}"
USE_OPENCLAW=false
SETUP_OPENCLAW=false

for arg in "$@"; do
    case "$arg" in
        --openclaw) USE_OPENCLAW=true ;;
        --setup-openclaw) USE_OPENCLAW=true; SETUP_OPENCLAW=true ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
die()   { err "$@"; exit 1; }

ssh_cmd() { ssh "$JETSON_USER@$JETSON_HOST" "$@"; }

# ── Pre-flight ────────────────────────────────────────────────────────
info "Checking SSH connection to $JETSON_USER@$JETSON_HOST..."
ssh_cmd "echo ok" >/dev/null 2>&1 || die "Cannot SSH to $JETSON_HOST"
ok "SSH connected"

if [ "$USE_OPENCLAW" = true ]; then
    info "Mode: OpenClaw (gateway + reachy)"
else
    info "Mode: Ollama (reachy only, no gateway)"
fi

# ── Sync deploy files ─────────────────────────────────────────────────
info "Syncing deploy files..."
ssh_cmd "mkdir -p $DEPLOY_DIR"
rsync -az \
    "$SCRIPT_DIR/reachy/docker-compose.yml" \
    "$SCRIPT_DIR/reachy/reachy-claw.jetson.yaml" \
    "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/"

# Sync OpenClaw helper scripts (always — they're small and useful later)
rsync -az \
    "$SCRIPT_DIR/openclaw/setup.sh" \
    "$SCRIPT_DIR/openclaw/configure-llm.sh" \
    "$JETSON_USER@$JETSON_HOST:$DEPLOY_DIR/"
ssh_cmd "chmod +x $DEPLOY_DIR/setup.sh $DEPLOY_DIR/configure-llm.sh"
ok "Files synced"

# ── Stop old containers ───────────────────────────────────────────────
info "Stopping old containers..."
ssh_cmd "cd $DEPLOY_DIR && docker compose --profile openclaw down 2>/dev/null" || true

# Also clean up legacy separate openclaw compose if it exists
ssh_cmd "cd $DEPLOY_DIR/openclaw && docker compose down 2>/dev/null" || true

# ── Start stack ───────────────────────────────────────────────────────
if [ "$USE_OPENCLAW" = true ]; then
    info "Pulling and starting (with OpenClaw gateway)..."
    ssh_cmd "cd $DEPLOY_DIR && docker compose --profile openclaw pull && docker compose --profile openclaw up -d"
else
    info "Pulling and starting (Ollama mode)..."
    ssh_cmd "cd $DEPLOY_DIR && docker compose pull && docker compose up -d"
fi
ok "Containers started"

# ── OpenClaw first-time setup ─────────────────────────────────────────
if [ "$SETUP_OPENCLAW" = true ]; then
    info "Running OpenClaw first-time setup (extension + config)..."
    ssh_cmd "cd $DEPLOY_DIR && bash setup.sh"
    ok "OpenClaw setup complete"
fi

# ── Smoke tests ───────────────────────────────────────────────────────
info "=== Running smoke tests ==="
sleep 5

check_service() {
    local name="$1" cmd="$2"
    if ssh_cmd "$cmd" >/dev/null 2>&1; then
        ok "$name"
    else
        err "$name is not responding"
    fi
}

check_service "Speech service (:8621)" "curl -sf http://localhost:8621/health"
check_service "Reachy daemon (:38001)" "curl -sf http://localhost:38001/"
if [ "$USE_OPENCLAW" = true ]; then
    check_service "OpenClaw gateway (:18789)" "curl -sf http://localhost:18789/healthz"
else
    check_service "Ollama (:11434)" "curl -sf http://localhost:11434/api/tags"
fi

echo ""
info "Container status:"
ssh_cmd "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'reachy|openclaw|speech|voice'" || true

echo ""
ok "Deployment complete!"
echo ""

if [ "$USE_OPENCLAW" = true ]; then
    echo "Next steps (OpenClaw mode):"
    echo "  1. Configure LLM (if not done):"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && ./configure-llm.sh dashscope <api-key>'"
    echo ""
    echo "  2. View logs:"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && docker compose --profile openclaw logs -f'"
else
    echo "Next steps (Ollama mode):"
    echo "  1. Make sure Ollama is running with the right model:"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'ollama pull qwen3.5:2b-q4_K_M'"
    echo ""
    echo "  2. View logs:"
    echo "     ssh $JETSON_USER@$JETSON_HOST 'cd $DEPLOY_DIR && docker compose logs -f'"
    echo ""
    echo "  To switch to OpenClaw mode later:"
    echo "     ./deploy.sh --setup-openclaw"
fi
echo ""
