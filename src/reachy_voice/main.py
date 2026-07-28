"""Reachy Voice app entry point.

A thin `ReachyMiniApp` shell: the SDK owns lifecycle, the daemon connection, and
the settings webserver. All the voice logic lives in `conversation.ConversationEngine`.

The daemon launches this via the `reachy_mini_apps` entry point (`python -m
reachy_voice.main`); `wrapped_run()` connects to the robot and calls `run()`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import threading
import traceback
from urllib.parse import urlparse

# NOTE: FastAPI resolves endpoint type hints via the module globals; with
# `from __future__ import annotations` these names MUST be importable at module
# level or websocket/dependency injection silently fails with HTTP 403.
from fastapi import Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from reachy_mini import ReachyMini, ReachyMiniApp

from reachy_voice import overrides, speech_runtime, tier_a
from reachy_voice.config import load_config
from reachy_voice.conversation import ConversationEngine
from reachy_voice.dashboard import DashboardHub

logger = logging.getLogger("reachy_voice")


# ── speaker volume via the ALSA mixer (the dashboard slider drives this) ──
def _alsa_card() -> str:
    try:
        with open("/proc/asound/cards") as f:
            for line in f:
                if "Reachy Mini Audio" in line:
                    return line.split()[0]
    except Exception:  # noqa: BLE001
        pass
    return "0"


# This card's PCM dB curve is steep — below ~50% raw it's near-silent (47% raw
# ≈ -32dB). So map the slider's full 0-100 onto the AUDIBLE raw band 50-100%
# (slider 0 = mute), so the whole slider is usable instead of dead in the bottom.
def _slider_to_raw(slider: int) -> int:
    return 0 if slider <= 0 else 50 + round(slider * 0.5)


def _raw_to_slider(raw: int) -> int:
    return 0 if raw <= 0 else max(0, min(100, round((raw - 50) * 2)))


def get_volume() -> int:
    """Current speaker volume as a 0-100 SLIDER value (reads the PCM mixer)."""
    try:
        out = subprocess.run(
            ["amixer", "-c", _alsa_card(), "sget", "PCM"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        m = re.search(r"\[(\d+)%\]", out)
        return _raw_to_slider(int(m.group(1))) if m else 100
    except Exception:  # noqa: BLE001
        return 100


def set_volume(slider: int) -> int:
    """Set speaker volume from a 0-100 SLIDER value on BOTH PCM controls (the
    card has two; the 2nd holds the attenuation). Returns the slider value."""
    slider = max(0, min(100, int(slider)))
    raw = _slider_to_raw(slider)
    card = _alsa_card()
    for ctrl in ("'PCM',0", "'PCM',1", "PCM"):
        try:
            subprocess.run(
                ["amixer", "-c", card, "sset", ctrl, f"{raw}%", "unmute"],
                capture_output=True, timeout=3,
            )
        except Exception:  # noqa: BLE001
            pass
    return slider

# This robot runs the reachy-mini daemon on a non-default FastAPI port (38001 on
# the Jetson deploy; the SDK default is 8000). The app framework's localhost
# detection checks 8000, so we point it at the real port via env (overridable).
DAEMON_HOST = os.environ.get("REACHY_DAEMON_HOST", "localhost")
DAEMON_PORT = int(os.environ.get("REACHY_DAEMON_PORT", "38001"))


class ReachyVoiceApp(ReachyMiniApp):
    # Serve the settings UI (reachy_voice/static/) at this URL — the SDK mounts it.
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # We own the audio device via a sounddevice duplex stream (for ALSA echo
    # cancellation), so keep the SDK's gstreamer off it — matches the working
    # SLV deployment (`media_backend: no_media`).
    request_media_backend: str | None = "no_media"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Wire ALL routes (incl. the /ws websocket) BEFORE wrapped_run starts
        # uvicorn — websocket routes added after server startup get 403'd.
        self._config = load_config(os.environ.get("REACHY_VOICE_CONFIG"))
        self._engine = None
        self._loop = None
        self._hub = DashboardHub()
        # Operator tweaks (bargein / VAD / memory / language) persisted to a
        # bind-mounted file so they survive restart + redeploy. A missing file
        # (fresh robot / dev box) just means "no overrides".
        self._overrides = overrides.OverridesStore()
        self._engine_state = "starting"
        self._wire_settings()

    @staticmethod
    def _check_daemon_on_localhost(port: int = DAEMON_PORT, timeout: float = 0.5) -> bool:
        # Override the base check (which defaults to 8000) to probe the real
        # daemon port, so connection_mode resolves to "localhost_only".
        import socket

        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except OSError:
            return False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        asyncio.run(self._main(reachy_mini, self._config, stop_event))

    # ── settings dashboard (static UI is auto-served; here are the endpoints) ──
    def _wire_settings(self) -> None:
        if self.settings_app is None:
            return

        @self.settings_app.get("/status")
        def status() -> dict[str, str]:
            return {
                "app": "reachy-voice",
                "state": getattr(self, "_engine_state", "?"),
                "language": getattr(self._config, "language", "?"),
                "error": getattr(self, "error", ""),
            }

        @self.settings_app.post("/language")
        def set_language(language: str = Body(..., embed=True)) -> dict[str, str]:
            # Settings UI / API switch. The engine runs in another thread's event
            # loop, so hop onto it with run_coroutine_threadsafe.
            if self._engine is None or self._loop is None:
                return {"error": "engine not ready"}
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._engine.set_language(language.strip().lower()), self._loop
                )
                return {"language": fut.result(timeout=15)}
            except Exception as e:  # noqa: BLE001 — report back to the UI
                return {"error": str(e)}

        @self.settings_app.get("/api/speech/provider")
        def get_speech_provider() -> dict:
            return speech_runtime.read_settings()

        @self.settings_app.post("/api/speech/provider")
        def set_speech_provider(payload: dict = Body(...)) -> dict:
            try:
                return speech_runtime.save_settings(payload)
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}

        # ── live dashboard feed (bidirectional): broadcast events out, AND
        #    handle control messages in (set_volume / get_volume) ──
        @self.settings_app.websocket("/ws")
        async def ws_feed(ws: WebSocket) -> None:
            await ws.accept()
            q = self._hub.subscribe()

            async def _send() -> None:
                for msg in self._dashboard_snapshot():
                    await ws.send_json(msg)
                while True:
                    await ws.send_json(await q.get())

            async def _recv() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    data = await ws.receive_json()
                    kind = data.get("type")
                    if kind == "get_volume":
                        # amixer is a blocking subprocess — run it off the loop.
                        vol = await loop.run_in_executor(None, get_volume)
                        await ws.send_json({"type": "volume", "volume": vol})
                    elif kind == "set_volume":
                        v = await loop.run_in_executor(
                            None, set_volume, int(data.get("volume", 0))
                        )
                        self._hub.publish({"type": "volume", "volume": v})  # echo to all
                    # ── Tier-A: docker restart + vision-trt captures ──
                    elif kind == "restart_services":
                        # Long-running (polls health); fire-and-forget so the
                        # recv loop keeps servicing the socket. Progress streams
                        # out as restart_status broadcasts.
                        asyncio.create_task(self._restart_services())
                    elif kind == "get_capture_info":
                        self._hub.publish(
                            await tier_a.capture_info(self._config.vision_mjpeg)
                        )
                    elif kind == "clear_captures":
                        self._hub.publish(
                            await tier_a.clear_captures(self._config.vision_mjpeg)
                        )
                    # ── Tier-B: ovs_agent runtime tuning (persisted overrides) ──
                    elif kind == "set_bargein":
                        val = await loop.run_in_executor(
                            None, self._set_override, "bargein", data.get("enabled", True)
                        )
                        self._hub.publish({"type": "bargein_state", "enabled": val})
                    elif kind == "set_vad_threshold":
                        val = await loop.run_in_executor(
                            None, self._set_override, "vad", data.get("value", 0.3)
                        )
                        self._hub.publish({"type": "vad_threshold", "value": val})
                    elif kind == "set_history":
                        val = await loop.run_in_executor(
                            None, self._set_override, "history", data.get("turns", 0)
                        )
                        self._hub.publish({"type": "history", "turns": val})
                    elif kind == "get_history":
                        await ws.send_json(
                            {"type": "history", "turns": overrides.read_history(self._engine)}
                        )
                    elif kind == "set_vlm":
                        enabled = bool(data.get("enabled", True))
                        self._hub.publish({"type": "vlm_state", "enabled": enabled})
                    elif kind == "get_conversation_language":
                        await ws.send_json(self._language_msg())
                    elif kind == "set_conversation_language":
                        lang = str(data.get("language", "")).strip().lower()
                        if lang:
                            await loop.run_in_executor(
                                None, self._overrides.set, "language", lang
                            )
                            if self._engine is not None and self._loop is not None:
                                # async (reconnects SLV) — hop to the engine loop
                                asyncio.run_coroutine_threadsafe(
                                    self._engine.set_language(lang), self._loop
                                )
                            self._hub.publish(self._language_msg(lang))
                    elif kind == "send_message":
                        # External text (e.g. SenseCraft): inject as a USER turn so
                        # the LLM replies — NOT direct TTS (that's /debug/say).
                        text = str(data.get("text", "")).strip()
                        if text and self._engine is not None and self._loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._inject_user_text(self._engine, text), self._loop
                            )

            send_task = asyncio.create_task(_send())
            recv_task = asyncio.create_task(_recv())
            try:
                done, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                for t in pending:  # await cancellation so it doesn't leak
                    try:
                        await t
                    except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                        pass
                for t in done:
                    t.exception()  # retrieve so asyncio doesn't log "never retrieved"
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                self._hub.unsubscribe(q)

        # ── camera: proxy the vision service's MJPEG stream ──
        @self.settings_app.get("/stream")
        async def stream():
            import httpx

            upstream = getattr(self._config, "vision_mjpeg", "http://127.0.0.1:8630/stream")

            async def gen():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", upstream) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                gen(), media_type="multipart/x-mixed-replace; boundary=frame"
            )

        # ── Tier-A HTTP proxies (ollama models + vision-trt captures) ──
        # Registered via a shared helper so the route bodies live in one place
        # (also unit-testable without the SDK — see tests/voice/test_tier_a_routes).
        tier_a.register_http_routes(self.settings_app, lambda: self._config)

        # ── debug API: drive motion / TTS programmatically (no voice needed) ──
        def _motion():
            eng = self._engine
            return getattr(eng, "_motion", None) if eng is not None else None

        @self.settings_app.get("/debug/motion")
        def debug_motion() -> dict:
            m = _motion()
            if m is None:
                return {"error": "engine not ready"}
            return {
                "conv_state": m._conv_state,
                "cmd_deg": [round(m._cmd_yaw, 2), round(m._cmd_pitch, 2),
                            round(m._cmd_roll, 2), round(m._cmd_body, 2)],
                "playing": m._playing,
                "link_down": m._link_down,
                "moves_loaded": len(m._move_names),
            }

        @self.settings_app.post("/debug/state")
        def debug_state(state: str = Body(..., embed=True)) -> dict:
            m = _motion()
            if m is None:
                return {"error": "engine not ready"}
            m.set_conv_state(state)
            return {"conv_state": state}

        @self.settings_app.post("/debug/emotion")
        def debug_emotion(emotion: str = Body(..., embed=True)) -> dict:
            m = _motion()
            if m is None:
                return {"error": "engine not ready"}
            m.play_emotion(emotion)
            return {"emotion": emotion}

        @self.settings_app.post("/debug/say")
        def debug_say(text: str = Body(..., embed=True)) -> dict:
            # Inject text straight into the SLV TTS (makes the robot speak) so we
            # can exercise the speak path + audio + motion without ASR.
            eng = self._engine
            if eng is None or self._loop is None:
                return {"error": "engine not ready"}

            async def _say() -> None:
                slv = eng._app.slv
                await slv.send_text(text)
                flush = getattr(slv, "flush_tts", None)
                if flush is not None:
                    await flush()

            try:
                asyncio.run_coroutine_threadsafe(_say(), self._loop).result(timeout=20)
                return {"said": text}
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}

        @self.settings_app.get("/debug/vision")
        def debug_vision() -> dict:
            # What vision sees right now. Vision is dashboard-only; it no
            # longer feeds attention/gaze or drives motors.
            eng = self._engine
            if eng is None:
                return {"error": "engine not ready"}
            vision = getattr(eng, "_vision", None)
            faces = []
            for f in (vision.snapshot() if vision is not None else []):
                b = f.get("bbox")
                if b and len(b) == 4:
                    area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    faces.append({
                        "area": round(area, 4),
                        "center": [round((b[0] + b[2]) / 2, 2), round((b[1] + b[3]) / 2, 2)],
                        "id": f.get("identity"),
                    })
            return {
                "vision_fresh": bool(vision.faces_fresh()) if vision is not None else False,
                "faces_seen": len(faces),
                "faces": faces,
                "motor_control": "disabled",
            }

    async def _restart_services(self) -> None:
        """Restart the docker containers (vision-trt → reachy-daemon →
        reachy-voice) via the Docker socket, streaming restart_status to all
        connected dashboards. The hub's publish() is sync + thread-safe, so
        wrap it as the async broadcast tier_a expects."""

        async def broadcast(msg: dict) -> None:
            self._hub.publish(msg)

        await tier_a.restart_services(self._config.vision_mjpeg, broadcast)

    # ── runtime overrides (bargein / VAD / memory) ──
    def _set_override(self, key: str, raw_value: object) -> object:
        """Coerce → persist → apply a live override; returns the coerced value
        so the WS handler can echo it to all dashboards. Runs in an executor
        (it does a small blocking file write). Safe before the engine is ready:
        the value still persists and is replayed at startup."""
        setting = overrides.LIVE_SETTINGS[key]
        value = setting.coerce(raw_value)
        self._overrides.set(key, value)
        setting.apply(self._engine, value)
        return value

    def _language_msg(self, lang: str | None = None) -> dict:
        lang = lang or getattr(self._config, "language", "zh")
        return {
            "type": "conversation_language",
            "language": lang,
            "asr_language": lang,
            "tts_language": lang,
        }

    @staticmethod
    async def _inject_user_text(engine: object, text: str) -> None:
        """Feed text in as if the user said it (ASR → LLM → TTS path)."""
        app = getattr(engine, "_app", None)
        if app is not None:
            await app.on_user_utterance(text)

    def _dashboard_snapshot(self) -> list[dict]:
        """Messages sent to a freshly-connected dashboard so its panels and
        status dots reflect current config immediately (the original UI guards
        every field, so a partial snapshot is fine). Live values come from the
        running engine via the same readers the WS setters use, so the snapshot
        can't drift from what's actually in effect."""
        cfg = self._config
        eng = self._engine
        return [
            self._language_msg(),
            {
                "type": "robot_state",
                "mode": "chat",
                "llm_backend": "edge_llm",
                "ollama_model": getattr(cfg, "edge_llm_model", ""),
                "ollama_url": getattr(cfg, "edge_llm_url", ""),
                "silero_threshold": overrides.read_vad(eng),
                "vlm_enabled": True,
                "barge_in_enabled": overrides.read_bargein(eng),
                "capture_count": 0,
            },
            {"type": "speech_provider", **speech_runtime.read_settings()},
            {"type": "history", "turns": overrides.read_history(eng)},
            {"type": "volume", "volume": get_volume()},  # real speaker volume
        ]

    # ── async runtime: start the engine, idle until the daemon stops us ──
    async def _main(
        self,
        reachy_mini: ReachyMini,
        config: object,
        stop_event: threading.Event,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        # Replay a persisted language override BEFORE connecting, so SLV boots
        # in the right language with no extra reconnect.
        saved_lang = self._overrides.get("language")
        if saved_lang:
            config.language = str(saved_lang).strip().lower()
        engine = ConversationEngine(reachy_mini, config, hub=self._hub)
        self._engine = engine
        await engine.start()
        # Replay the live tuning overrides (bargein / VAD / memory) onto the
        # now-running engine.
        overrides.apply_saved(engine, self._overrides)
        self._engine_state = "running"
        logger.info(
            "Reachy Voice running (lang=%s, settings at %s)",
            getattr(config, "language", "?"), self.custom_app_url,
        )
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
        finally:
            self._engine_state = "stopping"
            await engine.stop()


def main() -> None:
    """Console entry point. Identical to running `python -m reachy_voice.main`."""
    logging.basicConfig(
        level=os.environ.get("REACHY_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = ReachyVoiceApp()
    try:
        # Pass the daemon host/port through to ReachyMini (forwarded by
        # wrapped_run as **kwargs) so we connect to localhost:38001, not the
        # SDK default reachy-mini.local:8000.
        app.wrapped_run(host=DAEMON_HOST, port=DAEMON_PORT)
    except Exception:
        app.error = traceback.format_exc()
        if os.environ.get("REACHY_DASHBOARD_FALLBACK", "1") != "1":
            raise
        if app.settings_app is None or app.custom_app_url is None:
            raise
        app._engine_state = "dashboard_only"
        logger.error(
            "Robot runtime failed; serving dashboard-only fallback on %s",
            app.custom_app_url,
            exc_info=True,
        )
        import uvicorn

        url = urlparse(app.custom_app_url)
        uvicorn.run(
            app.settings_app,
            host=url.hostname or "0.0.0.0",
            port=url.port or 8042,
            log_level=os.environ.get("UVICORN_LOG_LEVEL", "info").lower(),
        )
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
