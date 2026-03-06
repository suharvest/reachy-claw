"""Kokoro TTS service using sherpa-onnx (CUDA accelerated)."""

from __future__ import annotations

import io
import logging
import os
import struct
import time

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/opt/models/kokoro-multi-lang-v1_1")
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "cuda")
TTS_NUM_THREADS = int(os.environ.get("TTS_NUM_THREADS", "4"))
DEFAULT_SPEAKER_ID = int(os.environ.get("TTS_DEFAULT_SID", "3"))  # zf_001

_tts_instance = None


def get_tts():
    """Lazy-initialize Kokoro TTS model."""
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    import sherpa_onnx

    model_file = os.path.join(MODEL_DIR, "model.onnx")
    voices_file = os.path.join(MODEL_DIR, "voices.bin")
    tokens_file = os.path.join(MODEL_DIR, "tokens.txt")
    data_dir = os.path.join(MODEL_DIR, "espeak-ng-data")

    lexicon_files = ",".join([
        os.path.join(MODEL_DIR, "lexicon-us-en.txt"),
        os.path.join(MODEL_DIR, "lexicon-zh.txt"),
    ])

    logger.info("Loading Kokoro TTS from %s (provider=%s)", MODEL_DIR, TTS_PROVIDER)
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=model_file,
                voices=voices_file,
                tokens=tokens_file,
                data_dir=data_dir,
                lexicon=lexicon_files,
            ),
            provider=TTS_PROVIDER,
            num_threads=TTS_NUM_THREADS,
        ),
    )
    _tts_instance = sherpa_onnx.OfflineTts(config)
    logger.info("Kokoro TTS loaded.")
    return _tts_instance


def samples_to_wav(samples: list, sample_rate: int) -> bytes:
    """Convert float32 samples to WAV bytes (16-bit PCM)."""
    buf = io.BytesIO()
    num_samples = len(samples)
    data_size = num_samples * 2

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))

    for s in samples:
        s = max(-1.0, min(1.0, s))
        buf.write(struct.pack("<h", int(s * 32767)))

    return buf.getvalue()


def synthesize(
    text: str,
    speaker_id: int | None = None,
    speed: float = 1.0,
    **kwargs,
) -> tuple[bytes, dict]:
    """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
    if speaker_id is None:
        speaker_id = DEFAULT_SPEAKER_ID

    tts = get_tts()
    start = time.time()
    audio = tts.generate(text, sid=speaker_id, speed=speed)
    elapsed = time.time() - start

    duration = len(audio.samples) / audio.sample_rate
    wav_bytes = samples_to_wav(audio.samples, audio.sample_rate)

    meta = {
        "duration": round(duration, 3),
        "inference_time": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
        "sample_rate": audio.sample_rate,
    }
    return wav_bytes, meta


def preload() -> None:
    """Pre-load TTS model."""
    tts = get_tts()
    # Warmup inference
    audio = tts.generate("hello", sid=DEFAULT_SPEAKER_ID, speed=1.0)
    logger.info("TTS warmup done: %d samples", len(audio.samples))


def get_sample_rate() -> int:
    """Return the model's audio sample rate."""
    return get_tts().sample_rate


def is_ready() -> bool:
    return _tts_instance is not None
