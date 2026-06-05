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

echo "Launching kiosk: $DASHBOARD_URL"

reset_chromium_restore_state() {
    local base="${HOME}/.var/app/org.chromium.Chromium/config/chromium"
    [ -d "$base" ] || base="${HOME}/.config/chromium"
    [ -d "$base" ] || return 0

    rm -f \
        "$base/Default/Last Session" \
        "$base/Default/Last Tabs" \
        "$base/Default/Current Session" \
        "$base/Default/Current Tabs" 2>/dev/null || true

    python3 - "$base" <<'PY' || true
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
for path in (base / "Local State", base / "Default" / "Preferences"):
    if not path.exists():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    profile = data.setdefault("profile", {})
    profile["exit_type"] = "Normal"
    profile["exited_cleanly"] = True
    data.setdefault("browser", {})["show_home_button"] = False
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
PY
}

# Detect browser. Prefer Flatpak Chromium on Jetson because chromium-browser is
# often a snap shim that cannot launch from this kiosk session.
BROWSER=""
if command -v flatpak > /dev/null 2>&1; then
    if flatpak info org.chromium.Chromium > /dev/null 2>&1; then
        BROWSER="flatpak-chromium"
    fi
fi
for cmd in chromium-browser chromium google-chrome firefox; do
    [ -z "$BROWSER" ] || break
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
    flatpak-chromium)
        reset_chromium_restore_state
        exec flatpak run org.chromium.Chromium \
            --kiosk --no-first-run --disable-translate --disable-infobars \
            --disable-session-crashed-bubble --noerrdialogs \
            --password-store=basic \
            --disable-features=TranslateUI \
            "$DASHBOARD_URL"
        ;;
    *chromium*|*chrome*)
        reset_chromium_restore_state
        exec "$BROWSER" --kiosk --no-first-run --disable-translate --disable-infobars \
            --disable-session-crashed-bubble --noerrdialogs \
            --password-store=basic \
            --disable-features=TranslateUI \
            "$DASHBOARD_URL"
        ;;
    *firefox*)
        exec "$BROWSER" --start-fullscreen "$DASHBOARD_URL"
        ;;
esac
