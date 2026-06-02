#!/bin/bash
# Install Reachy Mini udev rules on the Jetson host.
#
# Rules installed (all *.rules in this dir):
#   99-reachy-camera.rules           — disable USB autosuspend (camera/audio, VID 38fb)
#   99-reachy-daemon-reconnect.rules — auto-restart reachy-daemon when the servo
#                                      serial bus (VID 1a86/55d3 -> ttyACM0) appears
#
# Usage:
#   sudo ./deploy/jetson/udev/install.sh
#
# Idempotent: safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    echo "Must run as root (sudo)." >&2
    exit 1
fi

for rule in "$SCRIPT_DIR"/*.rules; do
    target="/etc/udev/rules.d/$(basename "$rule")"
    cp "$rule" "$target"
    chmod 644 "$target"
    echo "Installed: $target"
done

udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --subsystem-match=tty

# udev rules only fire on device ADD events — for devices already enumerated
# at install time, apply the effects directly so the fix takes effect now
# without requiring a replug.
echo "Applying to currently-attached Reachy USB devices..."
# (a) autosuspend off for 38fb camera/audio
for dir in /sys/bus/usb/devices/*/idVendor; do
    if [ "$(cat "$dir" 2>/dev/null)" = "38fb" ]; then
        devdir=$(dirname "$dir")
        echo on > "$devdir/power/control" 2>/dev/null || true
    fi
done
# (b) if the servo serial is already present but reachy-daemon started before it,
#     restart the daemon now so it picks up the port (the rule handles future plugs).
if ls /dev/ttyACM* >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx reachy-daemon; then
        echo "Reachy serial present — restarting reachy-daemon to (re)scan the port..."
        docker restart reachy-daemon >/dev/null 2>&1 || true
    fi
fi

echo "Current state for Reachy Mini USB devices (VID 38fb):"
for dir in /sys/bus/usb/devices/*/idVendor; do
    [ "$(cat "$dir" 2>/dev/null)" = "38fb" ] || continue
    devdir=$(dirname "$dir")
    product=$(cat "$devdir/product" 2>/dev/null || echo "?")
    pid=$(cat "$devdir/idProduct" 2>/dev/null || echo "?")
    ctrl=$(cat "$devdir/power/control" 2>/dev/null || echo "?")
    delay=$(cat "$devdir/power/autosuspend_delay_ms" 2>/dev/null || echo "?")
    echo "  38fb:$pid ($product)  control=$ctrl  delay_ms=$delay"
done
