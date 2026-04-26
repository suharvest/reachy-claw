# READ FIRST — course correction (added after main spec)

## Emotion HEF is already provided — DO NOT search for or convert one

User has supplied the emotion model. It is already staged in TWO places:

1. **In this repo**: `deploy/vision-hailo/models/hsemotion_b0.hef`
   - md5: `fb155867327611d9a38ae038899463e5`
   - Size: 7.9 MB
   - 8-class HSEmotion EfficientNet-B0 — outputs match consumer's expected
     emotion strings exactly (`Anger Contempt Disgust Fear Happiness Neutral
     Sadness Surprise`)

2. **On the Pi (harvest-rpi) for convenience**: `/tmp/hsemotion_b0.hef`
   - Already pushed via `fleet push`. Re-push from repo if /tmp gets cleared.

## What this means for the spec

Spec §7 item 2 ("Emotion classification — preferred X, fallback Y...") is
**resolved**: use this HEF. No need to:
- Search Hailo Model Zoo
- Try ONNX-CPU fallback
- Convert from PyTorch
- Worry about class-count mismatch

Just load `hsemotion_b0.hef` for emotion inference, feed it 224×224 face crops
(or whatever input size hailortcli reports — VERIFY by running
`hailortcli parse-hef /tmp/hsemotion_b0.hef` on the Pi during probing). 8
softmax outputs → argmax → map index to the 8 class labels above.

## Still open: face detection HEF (SCRFD)

Detection HEF is NOT pre-supplied. You still need to:
- Find/download SCRFD-2.5g HEF from Hailo Model Zoo
- OR use any face detector HEF you find on the Pi already

If you don't find one easily during probing (step ≤20), STOP and report —
don't go down a custom-conversion rabbit hole.

## Still open: ArcFace (embedding) HEF

Optional per spec §7 item 3 — only needed for the `/api/faces/enroll` endpoint
to work. If not easily available, return 503 from enroll as spec says.
