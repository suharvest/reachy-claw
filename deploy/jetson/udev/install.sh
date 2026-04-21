#!/bin/bash
# Install udev rules on the Jetson host.
#
# Usage:
#   sudo ./deploy/jetson/udev/install.sh
#
# Idempotent: safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET=/etc/udev/rules.d/99-reachy-camera.rules

if [ "$EUID" -ne 0 ]; then
    echo "Must run as root (sudo)." >&2
    exit 1
fi

cp "$SCRIPT_DIR/99-reachy-camera.rules" "$TARGET"
chmod 644 "$TARGET"

udevadm control --reload-rules
udevadm trigger --subsystem-match=usb

# udev rules only fire on device add events — for devices already enumerated
# at install time, set power/control=on directly so the fix takes effect now
# without requiring a replug.
echo "Applying to currently-attached Reachy USB devices..."
for dir in /sys/bus/usb/devices/*/idVendor; do
    if [ "$(cat "$dir" 2>/dev/null)" = "38fb" ]; then
        devdir=$(dirname "$dir")
        echo on > "$devdir/power/control" 2>/dev/null || true
    fi
done

echo "Installed: $TARGET"
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
