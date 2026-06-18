"""Tests for the runtime-overrides persistence layer.

``overrides.py`` is deliberately SDK-free (like ``tier_a``), so it runs in CI:
the appliers/readers reach the live engine purely by duck-typed attribute
access, which lets us drive them with a ``SimpleNamespace`` fake engine that
mirrors the real object graph
(``engine._app.{config, session, _client_vad}``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reachy_voice import overrides


# ── fake engine mirroring the real ovs_agent object graph ────────────────────
def fake_engine(*, barge=None, vad=0.3, max_tokens=7000):
    app = SimpleNamespace(
        config=SimpleNamespace(barge_in_enabled=barge, client_vad_threshold=vad),
        _client_vad=SimpleNamespace(threshold=vad),
        session=SimpleNamespace(max_input_tokens=max_tokens),
    )
    return SimpleNamespace(_app=app)


# ── OverridesStore ───────────────────────────────────────────────────────────
def test_store_roundtrip(tmp_path):
    p = tmp_path / "overrides.json"
    store = overrides.OverridesStore(p)
    store.set("bargein", True)
    store.set("vad", 0.42)
    # a fresh store reading the same file sees the persisted values
    assert overrides.OverridesStore(p).all() == {"bargein": True, "vad": 0.42}


def test_store_missing_file_is_empty(tmp_path):
    assert overrides.OverridesStore(tmp_path / "nope.json").all() == {}


def test_store_corrupt_file_is_empty(tmp_path):
    p = tmp_path / "overrides.json"
    p.write_text("{not valid json", encoding="utf-8")
    # must not raise — a corrupt file degrades to "no overrides"
    assert overrides.OverridesStore(p).all() == {}


def test_store_set_creates_parent_dir(tmp_path):
    p = tmp_path / "sub" / "dir" / "overrides.json"
    overrides.OverridesStore(p).set("history", 12)
    assert json.loads(p.read_text(encoding="utf-8")) == {"history": 12}


def test_store_set_is_atomic_overwrite(tmp_path):
    p = tmp_path / "overrides.json"
    store = overrides.OverridesStore(p)
    store.set("vad", 0.3)
    store.set("vad", 0.7)
    # no temp/partial files left behind; file is valid and current
    assert list(tmp_path.iterdir()) == [p]
    assert json.loads(p.read_text(encoding="utf-8")) == {"vad": 0.7}


def test_data_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHY_VOICE_DATA_DIR", str(tmp_path))
    assert overrides.overrides_path() == tmp_path / overrides.OVERRIDES_FILENAME


def test_data_dir_default(monkeypatch):
    monkeypatch.delenv("REACHY_VOICE_DATA_DIR", raising=False)
    assert str(overrides.overrides_path()).startswith("/data/")


# ── turns <-> tokens mapping ─────────────────────────────────────────────────
def test_turns_tokens_roundtrip():
    # default 7000 tokens <-> 20 turns (the legacy slider's max)
    assert overrides.tokens_to_turns(7000) == 20
    assert overrides.turns_to_tokens(20) == 7000


def test_tokens_to_turns_handles_none():
    # ovs default before any override can be None ("unlimited") -> show the max
    assert overrides.tokens_to_turns(None) == overrides.DEFAULT_HISTORY_TURNS


# ── appliers ─────────────────────────────────────────────────────────────────
def test_apply_bargein():
    eng = fake_engine(barge=None)
    assert overrides.apply_bargein(eng, False) is True
    assert eng._app.config.barge_in_enabled is False


def test_apply_vad_sets_both_live_obj_and_config():
    eng = fake_engine(vad=0.3)
    overrides.apply_vad(eng, 0.55)
    # the no-op trap: the VAD object's own threshold is what speech detection
    # reads, so BOTH must change
    assert eng._app._client_vad.threshold == 0.55
    assert eng._app.config.client_vad_threshold == 0.55


def test_apply_history_maps_turns_to_tokens():
    eng = fake_engine(max_tokens=7000)
    overrides.apply_history(eng, 10)
    assert eng._app.session.max_input_tokens == 3500


def test_appliers_guard_missing_engine():
    # engine not ready / wrong shape -> no raise, returns False
    assert overrides.apply_bargein(None, True) is False
    assert overrides.apply_vad(SimpleNamespace(), 0.5) is False
    assert overrides.apply_history(SimpleNamespace(_app=SimpleNamespace()), 5) is False


# ── readers ──────────────────────────────────────────────────────────────────
def test_read_bargein_none_is_true():
    # ovs resolver: barge_in_enabled None means the legacy always-on default
    assert overrides.read_bargein(fake_engine(barge=None)) is True
    assert overrides.read_bargein(fake_engine(barge=False)) is False


def test_read_vad_prefers_live_obj():
    eng = fake_engine(vad=0.3)
    eng._app._client_vad.threshold = 0.6  # live value diverged from config
    assert overrides.read_vad(eng) == 0.6


def test_read_history_returns_turns():
    assert overrides.read_history(fake_engine(max_tokens=3500)) == 10


def test_readers_degrade_without_engine():
    assert overrides.read_bargein(None) is True            # default
    assert overrides.read_vad(None) == pytest.approx(0.3)  # default
    assert overrides.read_history(None) == overrides.DEFAULT_HISTORY_TURNS


# ── coercion / clamping ──────────────────────────────────────────────────────
def test_coercion_clamps_vad_to_unit_interval():
    s = overrides.LIVE_SETTINGS["vad"]
    assert s.coerce(1.5) == 1.0
    assert s.coerce(-0.2) == 0.0


def test_coercion_clamps_history_turns():
    s = overrides.LIVE_SETTINGS["history"]
    assert s.coerce(99) == 20
    assert s.coerce(-3) == 0


def test_coercion_bargein_truthy():
    s = overrides.LIVE_SETTINGS["bargein"]
    assert s.coerce("true") is True
    assert s.coerce(0) is False


# ── apply_saved replays the store onto the engine ────────────────────────────
def test_apply_saved_replays_known_keys(tmp_path):
    p = tmp_path / "overrides.json"
    store = overrides.OverridesStore(p)
    store.set("bargein", False)
    store.set("vad", 0.7)
    store.set("history", 4)
    store.set("language", "en")  # not a live setting; must be ignored here
    store.set("bogus", 123)      # unknown key; must be ignored

    eng = fake_engine()
    overrides.apply_saved(eng, store)

    assert eng._app.config.barge_in_enabled is False
    assert eng._app._client_vad.threshold == 0.7
    assert eng._app.session.max_input_tokens == overrides.turns_to_tokens(4)
