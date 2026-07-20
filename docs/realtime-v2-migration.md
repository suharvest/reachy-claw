# Reachy Voice — Realtime V2 migration

The active `src/reachy_voice` application is a provider-neutral Realtime V2
device client. It connects only to the Seeed V2V gateway; local cascade,
OpenAI Realtime, and Qwen Realtime are gateway deployment choices.

## Turn lifecycle

1. Reachy streams microphone PCM through `input_audio_buffer`.
2. The gateway emits transcription events but does not auto-create a response
   (`create_response: false`).
3. Reachy snapshots the current face/visitor context and sends a partial
   `session.update` containing the new instructions.
4. Reachy sends `response.create`.
5. Audio is played as it arrives. `response.output_audio.done` means remote
   audio generation ended; `response.done` is the one terminal response event.
   The application-level assistant-done hook waits for the local speaker queue
   to drain.

This ordering prevents server VAD from starting a cloud response before the
current camera context has reached the model.

## Motion and emotion

Emotion tags are not part of the native audio path because a speech-to-speech
provider may synthesize them before the device sees a transcript. The prompt
therefore asks the model to call `play_emotion` once per reply. Reachy exposes
`move_head`, `move_antennas`, `play_emotion`, and `dance` as canonical function
tools and executes them locally through the single-writer motion compositor.

## Configuration

The production defaults are:

```yaml
v2v_url: ws://localhost:8621/v2v/stream
server_loop: true
realtime_protocol_version: 2
```

Equivalent environment variables are `REACHY_V2V_URL`,
`REACHY_SERVER_LOOP`, and `REACHY_REALTIME_PROTOCOL_VERSION`.

Provider credentials, model names, and provider endpoints belong to the V2V
gateway. Switching providers must not require rebuilding or changing Reachy.

For a time-bounded rollback, set `REACHY_SERVER_LOOP=0`; this restores the
legacy local client-loop while retaining the Realtime V2 transport. Protocol V1
is supported only as a migration escape hatch and is not used by production.

## Packaging

`reachy-voice:v0.5.0` explicitly installs the vendored
`openvoicestream-agent` 0.2.0 wheel instead of relying on the opaque copy in the
older `reachy-claw:slv-v7` base image.
