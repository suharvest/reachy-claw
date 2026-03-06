"""Streaming Paraformer ASR service using sherpa-onnx OnlineRecognizer."""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("STREAMING_ASR_MODEL_DIR", "/opt/models/paraformer-streaming")
ASR_PROVIDER = os.environ.get("STREAMING_ASR_PROVIDER", "cuda")
ASR_NUM_THREADS = int(os.environ.get("STREAMING_ASR_NUM_THREADS", "4"))

_recognizer = None


def get_recognizer():
    """Lazy-init the streaming Paraformer OnlineRecognizer."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    import sherpa_onnx

    encoder = os.path.join(MODEL_DIR, "encoder.onnx")
    decoder = os.path.join(MODEL_DIR, "decoder.onnx")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    logger.info("Loading streaming Paraformer from %s (provider=%s)", MODEL_DIR, ASR_PROVIDER)
    _recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        provider=ASR_PROVIDER,
        num_threads=ASR_NUM_THREADS,
    )
    logger.info("Streaming Paraformer loaded.")
    return _recognizer


def create_stream():
    """Create a new online stream for one utterance."""
    recognizer = get_recognizer()
    return recognizer.create_stream()


def feed_and_decode(stream, samples: np.ndarray, sample_rate: int = 16000):
    """Feed audio samples and decode. Returns (text, is_final)."""
    recognizer = get_recognizer()

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    if np.abs(samples).max() > 1.0:
        samples = samples / 32768.0

    stream.accept_waveform(sample_rate, samples)

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text = recognizer.get_result(stream).strip()
    # sherpa-onnx OnlineRecognizer: result includes partial until endpoint
    is_endpoint = recognizer.is_endpoint(stream)

    return text, is_endpoint


def finalize(stream, sample_rate: int = 16000) -> str:
    """Finalize the stream (flush remaining audio). Returns final text.

    With the patched sherpa-onnx (EOF fix), input_finished() alone is
    sufficient — the patch forces IsReady() to return true for partial
    final chunks and CIF force-fires residual tokens.  No silence
    padding is needed (and padding can cause hallucinations).
    """
    recognizer = get_recognizer()

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return recognizer.get_result(stream).strip()


def preload() -> None:
    """Pre-load model."""
    try:
        get_recognizer()
    except Exception as e:
        logger.warning(f"Streaming ASR preload failed (model may not be installed): {e}")


def is_ready() -> bool:
    return _recognizer is not None
