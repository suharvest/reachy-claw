"""Tests for the dashboard-tunable settings registry."""

from __future__ import annotations

import pytest

from reachy_claw.settings_schema import (
    NAMESPACES,
    REGISTRY,
    keys_for_namespace,
    spec_for,
    validate,
)


def test_namespaces_includes_rest_diary_ha():
    assert {"rest", "diary", "ha"}.issubset(set(NAMESPACES))


def test_registry_has_expected_keys():
    expected = {
        "rest.enabled",
        "rest.window_start",
        "rest.window_end",
        "rest.timezone",
        "diary.auto_publish",
        "diary.privacy_linter",
        "diary.site_repo_url",
        "diary.site_diary_path",
        "diary.site_branch",
        "ha.url",
        "ha.token",
        "ha.entities",
    }
    assert set(REGISTRY) == expected


def test_keys_for_namespace_returns_only_that_ns():
    assert set(keys_for_namespace("rest")) == {
        "enabled",
        "window_start",
        "window_end",
        "timezone",
    }
    assert set(keys_for_namespace("diary")) == {
        "auto_publish",
        "privacy_linter",
        "site_repo_url",
        "site_diary_path",
        "site_branch",
    }


def test_spec_for_returns_field_name_and_type():
    spec = spec_for("rest.window_start")
    assert spec.config_field == "rest_window_start"
    assert spec.type_ == str


def test_validate_accepts_valid_hhmm():
    validate("rest.window_start", "23:00")
    validate("rest.window_end", "07:30")


def test_validate_rejects_bad_hhmm():
    with pytest.raises(ValueError):
        validate("rest.window_start", "25:00")
    with pytest.raises(ValueError):
        validate("rest.window_start", "9-30")
    # 24:30 must be rejected — only the exact 24:00 sentinel is allowed.
    with pytest.raises(ValueError):
        validate("rest.window_start", "24:30")
    with pytest.raises(ValueError):
        validate("rest.window_end", "24:01")


def test_validate_accepts_24_00_sentinel():
    validate("rest.window_end", "24:00")


def test_validate_rejects_bad_timezone():
    with pytest.raises(ValueError):
        validate("rest.timezone", "Not/A/Tz")


def test_validate_accepts_valid_timezone():
    validate("rest.timezone", "Asia/Shanghai")
    validate("rest.timezone", "UTC")


def test_validate_bool_type():
    validate("rest.enabled", True)
    validate("diary.auto_publish", False)
    with pytest.raises(ValueError):
        validate("rest.enabled", "true")  # string not bool
    with pytest.raises(ValueError):
        validate("rest.enabled", 1)


def test_validate_string_type():
    validate("diary.site_repo_url", "git@github.com:x/y.git")
    validate("diary.site_repo_url", "")  # empty allowed
    with pytest.raises(ValueError):
        validate("diary.site_repo_url", 123)


def test_validate_unknown_key_rejected():
    with pytest.raises(KeyError):
        validate("rest.unknown", "anything")


# ── HA namespace ────────────────────────────────────────────────────


def test_ha_namespace_registered():
    assert "ha" in NAMESPACES
    assert set(keys_for_namespace("ha")) == {"url", "token", "entities"}


def test_ha_url_validator():
    validate("ha.url", "")  # empty allowed
    validate("ha.url", "http://homeassistant.local:8123")
    validate("ha.url", "https://ha.example.com")
    with pytest.raises(ValueError):
        validate("ha.url", "ha.local")  # missing scheme
    with pytest.raises(ValueError):
        validate("ha.url", 123)  # not a string


def test_ha_token_validator():
    validate("ha.token", "")
    validate("ha.token", "long-token")
    with pytest.raises(ValueError):
        validate("ha.token", None)


def test_ha_entities_validator():
    validate("ha.entities", [])
    validate("ha.entities", ["weather.home", "sensor.bedroom_temp"])
    with pytest.raises(ValueError):
        validate("ha.entities", "weather.home")  # not a list
    with pytest.raises(ValueError):
        validate("ha.entities", ["weather.home", 1])  # non-str element
    with pytest.raises(ValueError):
        validate("ha.entities", ["BadFormat"])  # missing dot
    with pytest.raises(ValueError):
        validate("ha.entities", ["weather.Home Office"])  # bad chars
