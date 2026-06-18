# Reachy Voice — production image.
#
# Pragmatic base: the existing reachy-claw:slv-v7 image already carries every
# runtime dependency this app needs on Jetson arm64 (reachy-mini 1.8.0,
# ovs_agent, sounddevice/portaudio, onnxruntime, alsa-utils). We bake the
# reachy-voice source on top — a seconds-long build on the robot, zero
# dependency risk. A fully standalone image (python-slim + pip install .) can
# replace this later; the app itself doesn't care.
#
# Build (on the Jetson, repo root as context):
#   docker build -f deploy/jetson/voice/Dockerfile -t reachy-voice:v0.2.0 .

FROM sensecraft-missionpack.seeed.cn/solution/reachy-claw:slv-v7

LABEL org.opencontainers.image.title="reachy-voice" \
      org.opencontainers.image.version="0.4.0"

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
