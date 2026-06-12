#!/bin/sh
# Reachy Voice entrypoint.
#
# The Reachy Mini USB sound card resets its PCM playback mixer to -23dB on every
# host reboot (inaudible). It exposes TWO PCM controls ('PCM',0 and 'PCM',1) —
# the SECOND one holds the -23dB attenuation, so both must be pinned to 100%
# (plain `sset PCM` only touches the first → speaker stays quiet). Find the card
# by name (index moves: hw:0 vs hw:2) and max both before starting.
CARD=$(awk '/Reachy Mini Audio/{print $1; exit}' /proc/asound/cards 2>/dev/null)
if [ -n "$CARD" ]; then
    amixer -c "$CARD" sset "'PCM',0" 100% unmute >/dev/null 2>&1
    amixer -c "$CARD" sset "'PCM',1" 100% unmute >/dev/null 2>&1
    amixer -c "$CARD" sset PCM 100% unmute >/dev/null 2>&1
    echo "entrypoint: card $CARD PCM(0,1) -> 100%"
else
    echo "entrypoint: Reachy USB card not found in /proc/asound/cards (continuing)"
fi

exec python3 -m reachy_voice.main
