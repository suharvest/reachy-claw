#!/usr/bin/env bash
# Auto-launch Reachy dashboard in fullscreen mode.
# Waits for the dashboard to be ready before opening the browser.
#
# Install: ./deploy/jetson/kiosk/install.sh

DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:8640}"
MAX_WAIT=300  # seconds

echo "Waiting for dashboard at $DASHBOARD_URL ..."
for ((i=1; i<=MAX_WAIT/5; i++)); do
    curl -s --max-time 2 "$DASHBOARD_URL/" > /dev/null 2>&1 && break
    sleep 5
done

echo "Launching browser: $DASHBOARD_URL"
# Detect browser
BROWSER=""
for cmd in chromium-browser chromium google-chrome firefox; do
    if command -v "$cmd" > /dev/null 2>&1; then
        BROWSER="$cmd"
        break
    fi
done

if [ -z "$BROWSER" ]; then
    echo "ERROR: No browser found. Install chromium: sudo apt install chromium-browser"
    exit 1
fi

case "$BROWSER" in
    *chromium*|*chrome*)
        exec "$BROWSER" --start-fullscreen --no-first-run --disable-translate --disable-infobars \
            --disable-session-crashed-bubble --noerrdialogs \
            --disable-features=TranslateUI \
            "$DASHBOARD_URL"
        ;;
    *firefox*)
        exec "$BROWSER" --start-fullscreen "$DASHBOARD_URL"
        ;;
esac
