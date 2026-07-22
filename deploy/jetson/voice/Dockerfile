# Reachy Voice — production image.
#
# Pragmatic base: the existing reachy-claw:slv-v7 image already carries every
# runtime dependency this app needs on Jetson arm64 (reachy-mini 1.8.0,
# ovs_agent, sounddevice/portaudio, onnxruntime, alsa-utils). We bake the
# reachy-voice source and the pinned Realtime V2 agent on top — a seconds-long
# build on the robot, zero
# dependency risk. A fully standalone image (python-slim + pip install .) can
# replace this later; the app itself doesn't care.
#
# Build (on the Jetson, repo root as context):
#   docker build -f deploy/jetson/voice/Dockerfile -t reachy-voice:v0.6.0 .
#
# v0.6.0 is the first canonical Realtime V2 SERVER-LOOP image. It is a distinct
# lineage from the already-published v0.5.0 (client-loop): keep the version
# bumped so the two never collide in the registry.

FROM sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v7

LABEL org.opencontainers.image.title="reachy-voice" \
      org.opencontainers.image.version="0.6.0"

# Do not inherit the opaque ovs_agent bundled in slv-v7.  The application and
# lock file pin the canonical Realtime V2 client explicitly.
COPY deploy/jetson/reachy/vendor/openvoicestream_agent-0.2.0-py3-none-any.whl /tmp/vendor/
RUN pip install --no-cache-dir --no-deps --upgrade \
    /tmp/vendor/openvoicestream_agent-0.2.0-py3-none-any.whl \
    && rm -rf /tmp/vendor

# Package now lives under src/ after the reachy-claw <- reachy-voice merge; it is
# still mounted at /opt/reachy-voice/reachy_voice so PYTHONPATH resolution is
# unchanged (python -m reachy_voice.main).
COPY src/reachy_voice /opt/reachy-voice/reachy_voice
COPY deploy/jetson/voice/entrypoint.sh /opt/reachy-voice/entrypoint.sh
RUN chmod +x /opt/reachy-voice/entrypoint.sh

# Bake the official emotion + dance move libraries into the image's HF cache so
# motion playback never needs a network round-trip and survives container
# recreation (otherwise they download to the container's writable layer on first
# use and are lost on `compose down/up`). ~107MB of designed moves + sounds.
#
# NOTE: this Jetson's `docker build` network cannot reach the HF Hub, so instead
# of downloading here we PRE-SEED the cache into the build context (run once from
# a container that already has it):
#   docker cp reachy-voice:/root/.cache/huggingface/hub ./hf-hub
COPY hf-hub /root/.cache/huggingface/hub

ENV PYTHONPATH=/opt/reachy-voice
WORKDIR /opt/reachy-voice

ENTRYPOINT ["/opt/reachy-voice/entrypoint.sh"]
