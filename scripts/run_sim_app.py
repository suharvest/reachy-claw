#!/usr/bin/env python3
"""Run Reachy Voice on macOS with media imports stubbed for --no-media sim."""

import atexit
import os
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock


gi_mod = types.ModuleType("gi")
gi_mod.require_version = lambda *args, **kwargs: None  # type: ignore[attr-defined]
gi_repo = types.ModuleType("gi.repository")
gi_repo.Gst = MagicMock()  # type: ignore[attr-defined]
gi_repo.GLib = MagicMock()  # type: ignore[attr-defined]
gi_mod.repository = gi_repo  # type: ignore[attr-defined]

sys.modules.setdefault("gi", gi_mod)
sys.modules.setdefault("gi.repository", gi_repo)
sys.modules.setdefault("gi.repository.Gst", gi_repo.Gst)
sys.modules.setdefault("gi.repository.GLib", gi_repo.GLib)

scripted_wav = os.environ.get("REACHY_SCRIPTED_AUDIO_WAV")
if scripted_wav:
    sys.path.insert(0, str(Path(__file__).parents[1] / "tests" / "e2e"))
    from fake_audio import ScriptedAudioIO  # noqa: E402
    import reachy_voice.conversation as conversation  # noqa: E402

    scripted_audio = ScriptedAudioIO(scripted_wav)
    conversation.DuplexAudioIO = lambda **_kwargs: scripted_audio

    @atexit.register
    def _report_audio() -> None:
        print(
            "SCRIPTED_AUDIO_RESULT "
            f"tts_bytes={len(scripted_audio.captured_tts)} "
            f"output_sr={scripted_audio.output_sr}",
            flush=True,
        )

from reachy_voice.main import main  # noqa: E402


if __name__ == "__main__":
    main()
