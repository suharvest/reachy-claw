"""FastAPI speech service: SenseVoice ASR + Kokoro TTS."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jetson Speech Service", version="1.0.0")


class TTSRequest(BaseModel):
    text: str
    sid: int | None = None
    speed: float = 1.0


@app.on_event("startup")
async def startup():
    import tts_service

    logger.info("Pre-loading TTS model...")
    tts_service.preload()

    # Pre-load streaming ASR (primary ASR backend)
    try:
        import streaming_asr_service
        streaming_asr_service.preload()
    except Exception as e:
        logger.info(f"Streaming ASR not available: {e}")

    # SenseVoice (offline ASR) is lazy-loaded on first /asr request
    # to avoid wasting GPU memory when only streaming is used.

    logger.info("Speech service ready.")


@app.get("/health")
async def health():
    import asr_service, tts_service

    result = {
        "asr": asr_service.is_ready(),
        "tts": tts_service.is_ready(),
    }
    try:
        import streaming_asr_service
        result["streaming_asr"] = streaming_asr_service.is_ready()
    except ImportError:
        result["streaming_asr"] = False
    return result


@app.post("/asr")
async def asr(
    file: UploadFile = File(...),
    language: str = Query("auto"),
):
    import asr_service

    audio_bytes = await file.read()
    text = asr_service.transcribe_audio(audio_bytes, language=language)
    return {"text": text}


@app.post("/tts")
async def tts(req: TTSRequest):
    import tts_service

    wav_bytes, meta = tts_service.synthesize(
        text=req.text,
        speaker_id=req.sid,
        speed=req.speed,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": str(meta["duration"]),
            "X-Inference-Time": str(meta["inference_time"]),
            "X-RTF": str(meta["rtf"]),
        },
    )


@app.options("/tts/stream")
async def tts_stream_options():
    """Allow clients to probe for streaming support."""
    return Response(status_code=200)


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream TTS as raw PCM: first 4 bytes = sample_rate (uint32 LE), then int16 PCM chunks."""
    import tts_service

    def generate():
        import struct

        tts = tts_service.get_tts()
        sid = req.sid if req.sid is not None else tts_service.DEFAULT_SPEAKER_ID
        sample_rate_sent = False

        def callback(samples, progress):
            """Called by sherpa-onnx as audio is generated."""
            nonlocal sample_rate_sent
            # Return 1 to continue, 0 to stop
            return 1

        audio = tts.generate(
            req.text, sid=sid, speed=req.speed, callback=callback,
        )

        # sherpa-onnx callback doesn't yield partial chunks in all builds,
        # so we chunk the final output into ~256ms pieces for streaming effect
        # that still reduces time-to-first-byte vs batch WAV encoding
        sr = audio.sample_rate
        yield struct.pack("<I", sr)

        chunk_size = 4096  # ~256ms at 16kHz
        samples = audio.samples
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            pcm = struct.pack(f"<{len(chunk)}h", *(
                int(max(-1.0, min(1.0, s)) * 32767) for s in chunk
            ))
            yield pcm

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.websocket("/asr/stream")
async def asr_stream(
    ws: WebSocket,
    language: str = "auto",
    sample_rate: int = 16000,
):
    """Streaming ASR via WebSocket.

    Protocol:
      Client sends: raw int16 PCM bytes (audio chunks)
      Client sends: empty bytes b"" to signal end of audio
      Server sends: JSON {"text": "...", "is_final": bool, "is_stable": bool}
    """
    import json

    import numpy as np

    await ws.accept()

    try:
        import streaming_asr_service
    except ImportError:
        await ws.send_json({"error": "streaming ASR not available"})
        await ws.close()
        return

    stream = streaming_asr_service.create_stream()
    prev_text = ""

    try:
        while True:
            data = await ws.receive_bytes()

            if len(data) == 0:
                # End of audio — finalize
                final_text = streaming_asr_service.finalize(stream)
                await ws.send_json({
                    "text": final_text,
                    "is_final": True,
                    "is_stable": True,
                })
                break

            # Convert int16 bytes to float32
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            text, is_endpoint = streaming_asr_service.feed_and_decode(
                stream, samples, sample_rate
            )

            if is_endpoint:
                await ws.send_json({
                    "text": text,
                    "is_final": True,
                    "is_stable": True,
                })
                # Reset stream for potential next utterance
                stream = streaming_asr_service.create_stream()
                prev_text = ""
            elif text and text != prev_text:
                # Send partial result
                # Mark as stable if text only grew (no corrections)
                is_stable = text.startswith(prev_text) if prev_text else False
                await ws.send_json({
                    "text": text,
                    "is_final": False,
                    "is_stable": is_stable,
                })
                prev_text = text

    except WebSocketDisconnect:
        logger.debug("ASR stream client disconnected")
    except Exception as e:
        logger.error(f"ASR stream error: {e}")
        try:
            await ws.close()
        except Exception:
            pass
