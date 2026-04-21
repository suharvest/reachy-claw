#!/bin/sh
set -e

DEV="${CAMERA_DEVICE:-/dev/video0}"

# Kill any stale process holding the camera FD (e.g. previous crashed instance).
# Our own container is fresh, so the only holders would be leaks from a prior
# run that didn't tear down cleanly.
if [ -e "$DEV" ]; then
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "$DEV" 2>/dev/null || true
        # Give kernel a beat to release the FD.
        sleep 1
    fi
fi

exec "$@"
