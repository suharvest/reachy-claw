"""END-TO-END client-loop PROOF *through the real SLV engine* (throwaway).

Unlike ``proof_clientloop.py`` (which injects ASR text directly and stubs
TTS, exercising only the runner in isolation), this launcher runs the
REAL ``ReachyClawClientLoopApp`` exactly as ``ovs-agent run`` would —
connected to a live pass-through SLV engine on :8621 — and injects ONE
spoken turn by feeding a WAV into the app's mic source. The full chain
therefore goes THROUGH the engine WS transport:

    WAV -> (mic pump) -> engine ASR -> engine emits asr_final over WS
        -> app dispatch -> run_default_dialogue_turn
        -> CLIENT-LOOP runner (stream_with_tools) carrying our tools
        -> LLM emits tool_call -> ReachyToolsPlugin handler FIRES
        -> tool_result -> LLM re-issues -> final assistant text
        -> app sends text to engine -> engine TTS -> audio frames back

We do NOT advertise tools to the engine (that would be the server-loop
path). The engine never sees a tool; it only does ASR + TTS. The LLM +
tool loop lives entirely client-side in the app.

Injection mechanism: we swap ``app.audio`` for a ``_WavFedAudioIO`` whose
``start_capture()`` async-generator yields the WAV PCM in mic-sized
chunks followed by trailing silence (so the engine's silero VAD endpoints
and emits asr_final). Everything downstream is the unmodified app.

Run:
  LLM_MODEL=qwen2.5:14b \
  uv run python -m reachy_claw.clientloop.proof_engine_e2e \
    --wav /tmp/utt.wav
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path
from typing import AsyncIterator

from ovs_agent.config import load_config

from reachy_claw.clientloop.app import ReachyClawClientLoopApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("proof_engine_e2e")

_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _load_wav_pcm16(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    assert sw == 2, f"need 16-bit PCM, got sampwidth={sw}"
    assert ch == 1, f"need mono, got channels={ch}"
    return frames, sr


class _WavFedAudioIO:
    """Drop-in stand-in for ``AudioIO`` that feeds a WAV instead of a mic.

    Only the surface the app's mic pump / SLV client touch is implemented:
      * ``start_capture()`` async-gen yields PCM chunks (then silence).
      * ``chunk_ms`` / ``input_sr`` attributes read by the mic pump.
      * playback methods are no-ops (we never open a speaker; TTS frames
        are observed via the SLV client logs / event bus instead).
    """

    def __init__(self, pcm: bytes, input_sr: int, chunk_ms: int = 100) -> None:
        self._pcm = pcm
        self.input_sr = input_sr
        self.output_sr = 24000
        self.chunk_ms = chunk_ms
        self._chunk_bytes = int(input_sr * chunk_ms / 1000) * 2  # 16-bit mono
        self._fed = False

    async def start_capture(self) -> AsyncIterator[bytes]:
        # Small settle so the app has connected + sent its config frame and
        # released the mic-forward gate before audio starts flowing.
        await asyncio.sleep(1.0)
        log.info(
            ">>> INJECTING WAV into mic source: %d pcm bytes @ %d Hz "
            "(%d-byte chunks)", len(self._pcm), self.input_sr, self._chunk_bytes,
        )
        idx = 0
        while idx < len(self._pcm):
            yield self._pcm[idx:idx + self._chunk_bytes]
            idx += self._chunk_bytes
            await asyncio.sleep(self.chunk_ms / 1000.0)  # real-time pacing
        self._fed = True
        log.info(">>> WAV fully fed; now streaming CONTINUOUS silence (keeps mic-watchdog happy, no restart)")
        # Continuous low-level silence. Two jobs:
        #  1. The app's energy client-VAD sees speech->silence and sends ONE
        #     asr_eos to the engine, which (running without server VAD here)
        #     finalizes -> single asr_final. Silence RMS (0.0) is well below
        #     the energy VAD threshold (0.012) so it never re-triggers a new
        #     speech segment / second utterance.
        #  2. Steady chunk flow keeps the mic-pump watchdog from declaring the
        #     capture stale and RESTARTING it (which would re-run this feeder
        #     and re-inject the WAV, corrupting the single-turn proof).
        # The supervisor shuts the app down once tool_result + tts_started.
        silence = b"\x00\x00" * (self._chunk_bytes // 2)
        while True:
            yield silence
            await asyncio.sleep(self.chunk_ms / 1000.0)

    # ── playback surface (no-ops — we don't open a speaker) ──
    def set_source_sample_rate(self, sr: int) -> None:  # noqa: ANN001
        return None

    async def play(self, *a, **k) -> None:  # noqa: ANN002, ANN003
        return None

    def stop_playback(self) -> None:
        return None

    def arm_for_next_turn(self) -> None:
        return None

    @property
    def is_playing(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


class _ProofApp(ReachyClawClientLoopApp):
    """Real app + WAV-fed mic + auto-shutdown once TTS completes."""

    def __init__(self, config, pcm: bytes, wav_sr: int) -> None:  # noqa: ANN001
        super().__init__(config)
        # Swap the mic for the WAV feeder. input_sr must match what the
        # config tells the engine (16000).
        self.audio = _WavFedAudioIO(pcm, input_sr=16000)
        self._proof = {
            "asr_final": None,
            "tool_call": None,
            "tool_result": None,
            "tts_started": False,
        }

        # Subscribe to the client-loop tool events the runner emits via
        # app_mode (proves the tool fired CLIENT-SIDE, not server-side).
        self.events.subscribe("tool_call_started", self._on_tc_started)
        self.events.subscribe("tool_call_completed", self._on_tc_completed)

    def _on_tc_started(self, payload) -> None:  # noqa: ANN001
        self._proof["tool_call"] = payload
        log.info("########## CLIENT-LOOP TOOL_CALL STARTED: %s", payload)

    def _on_tc_completed(self, payload) -> None:  # noqa: ANN001
        self._proof["tool_result"] = payload
        log.info("########## CLIENT-LOOP TOOL_RESULT: %s", payload)


async def run(wav: str, config_path: Path, timeout_s: float) -> int:
    pcm, wav_sr = _load_wav_pcm16(wav)
    log.info("loaded wav: %s pcm_bytes=%d sr=%d", wav, len(pcm), wav_sr)
    if wav_sr != 16000:
        log.warning("WAV sr=%d != 16000; engine config expects 16k", wav_sr)

    cfg = load_config(config_path)

    # qwen2.5:14b on local CPU/Metal needs ~5-15s per LLM round, and the
    # client-loop does TWO rounds (tool_call + re-issue). The stock
    # watchdog/probe timeouts are tuned for a fast cloud LLM and would
    # abort the turn. Bump them for this local-model proof (throwaway).
    cfg.thinking_timeout_s = 120.0
    cfg.llm_first_token_timeout_s = 60.0
    cfg.llm_stream_idle_timeout_s = 60.0
    cfg.llm_availability_probe_timeout_s = 30.0
    cfg.llm_availability_probe_interval_s = 300.0

    # Turn OFF the app's client VAD. With client VAD on, the mic pump only
    # forwards audio *during* detected speech and stops transmitting on the
    # speech->silence edge (without sending asr_eos), so the engine's
    # server-side silero VAD never receives the trailing silence and never
    # endpoints -> the ASR turn wedges until the engine's 45s deadline.
    # With client VAD OFF the mic pump forwards ALL audio (WAV + trailing
    # silence) raw, and the engine's silero VAD does the endpointing and
    # emits a single asr_final. This is the correct division of labour for
    # a server-VAD (vad: "silero") pass-through engine.
    cfg.client_vad_backend = "off"

    log.info(
        "config: slv_url=%s llm=%s/%s tools_enabled=%s server_loop=%s "
        "(thinking_timeout=%.0fs first_token=%.0fs probe_timeout=%.0fs)",
        cfg.slv_url, cfg.llm_backend, cfg.llm_model,
        cfg.tools_enabled, getattr(cfg, "server_loop", False),
        cfg.thinking_timeout_s, cfg.llm_first_token_timeout_s,
        cfg.llm_availability_probe_timeout_s,
    )

    app = _ProofApp(cfg, pcm, wav_sr)

    # Instrument the SLV client to flag asr_final + tts_started as they
    # cross the WS transport (irrefutable "went through the engine").

    orig_handle_json = app.slv._handle_json

    async def _wrapped_handle_json(raw: str) -> None:
        await orig_handle_json(raw)

    app.slv._handle_json = _wrapped_handle_json  # keep signature parity

    # Watch the event queue indirectly: subscribe to dispatch by wrapping
    # on_user_utterance to capture the asr_final text the engine produced.
    orig_on_utt = app.on_user_utterance

    async def _wrapped_on_utt(text, *a, **k):  # noqa: ANN001, ANN002, ANN003
        app._proof["asr_final"] = text
        log.info("########## ENGINE asr_final -> app.on_user_utterance: %r", text)
        return await orig_on_utt(text, *a, **k)

    app.on_user_utterance = _wrapped_on_utt

    # Detect tts_started off the SLV event stream by wrapping _handle_json
    # to peek for the tts_started type before delegating.
    import json as _json

    async def _peeking_handle_json(raw: str) -> None:
        try:
            evt = _json.loads(raw)
            t = evt.get("type")
            if t == "tts_started":
                app._proof["tts_started"] = True
                log.info("########## ENGINE tts_started: %r", evt.get("sentence", "")[:80])
            elif t == "asr_final":
                log.info("########## ENGINE asr_final frame (raw): %r", evt.get("text"))
        except Exception:
            pass
        await orig_handle_json(raw)

    app.slv._handle_json = _peeking_handle_json

    # Auto-shutdown: once we've seen tool_result AND tts_started, give a
    # short grace for audio frames then trigger shutdown.
    async def _supervisor() -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            p = app._proof
            if p["tool_result"] is not None and p["tts_started"]:
                log.info(">>> proof signals satisfied; grace 2s for TTS frames then shutdown")
                await asyncio.sleep(2.0)
                break
            await asyncio.sleep(0.25)
        else:
            log.warning(">>> supervisor TIMEOUT (%.0fs) — shutting down", timeout_s)
        if app._shutdown_evt is not None:
            app._shutdown_evt.set()

    sup_task = asyncio.create_task(_supervisor())
    try:
        await app.run()
    finally:
        sup_task.cancel()
        try:
            await sup_task
        except asyncio.CancelledError:
            pass

    # ── verdict ──
    p = app._proof
    log.info("=" * 70)
    log.info("PROOF SIGNALS (all must be true):")
    log.info("  engine asr_final text     = %r", p["asr_final"])
    log.info("  client-loop tool_call     = %s", p["tool_call"])
    log.info("  client-loop tool_result   = %s", p["tool_result"])
    log.info("  engine tts_started        = %s", p["tts_started"])
    proven = (
        bool(p["asr_final"])
        and p["tool_call"] is not None
        and p["tool_result"] is not None
        and p["tts_started"]
    )
    log.info("=" * 70)
    log.info(
        "VERDICT: full client-loop pipeline THROUGH THE ENGINE %s",
        "PROVEN" if proven else "NOT-PROVEN",
    )
    return 0 if proven else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16k mono PCM16 WAV to inject")
    ap.add_argument("--config", type=Path, default=_CONFIG)
    ap.add_argument("--timeout", type=float, default=90.0)
    a = ap.parse_args()
    return asyncio.run(run(a.wav, a.config, a.timeout))


if __name__ == "__main__":
    sys.exit(main())
