"""Stateless async Home Assistant REST client.

Used by:
  - dashboard_plugin: /api/ha/test (probe), /api/ha/entities (list_states)
  - scripts/generate_diary.py: get_history (per-entity day window)

Pure functions — no plugin, no caching, no background tasks. Each call
opens a fresh httpx.AsyncClient via `_transport_factory()` so tests can
inject a MockTransport.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HAError(Exception):
    """Base for HA client errors."""


class HAUnreachable(HAError):
    """Network failure: timeout, DNS, connection refused, TLS error."""


class HAUnauthorized(HAError):
    """HTTP 401 — bad or missing token."""


class HABadResponse(HAError):
    """Non-2xx other than 401, or response body did not parse."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HA {status}: {message}")
        self.status = status
        self.message = message


def _normalise_url(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"HA URL must start with http:// or https://, got {url!r}")
    return url.rstrip("/")


def _safe_url_for_log(url: str) -> str:
    """Strip query string + userinfo so secrets never appear in log lines."""
    try:
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        # Drop userinfo from netloc.
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return urlunsplit((parts.scheme, f"{host}{port}", parts.path, "", ""))
    except Exception:
        return "<redacted>"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _transport_factory():
    """Override in tests with a MockTransport."""
    return None  # None → httpx uses its real transport.


def _client(timeout: float) -> httpx.AsyncClient:
    transport = _transport_factory()
    if transport is None:
        return httpx.AsyncClient(timeout=timeout)
    return httpx.AsyncClient(timeout=timeout, transport=transport)


async def probe(url: str, token: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """GET <url>/api/. Never raises; encodes errors in the result dict."""
    try:
        base = _normalise_url(url)
    except ValueError as e:
        return {"ok": False, "status": 0, "message": str(e)}
    async with _client(timeout) as client:
        try:
            r = await client.get(f"{base}/api/", headers=_headers(token))
        except httpx.HTTPError as e:
            return {"ok": False, "status": 0, "message": f"{type(e).__name__}: {e}"}
    if r.status_code == 200:
        msg = ""
        try:
            msg = r.json().get("message", "")
        except Exception:
            msg = r.text[:200]
        return {"ok": True, "status": 200, "message": msg}
    if r.status_code == 401:
        return {"ok": False, "status": 401, "message": "Unauthorized — token rejected"}
    return {"ok": False, "status": r.status_code, "message": r.text[:200]}


async def list_states(url: str, token: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """GET <url>/api/states. Returns full state list. Raises HAError on failure."""
    base = _normalise_url(url)
    async with _client(timeout) as client:
        try:
            r = await client.get(f"{base}/api/states", headers=_headers(token))
        except httpx.HTTPError as e:
            raise HAUnreachable(f"{type(e).__name__} contacting {_safe_url_for_log(base)}") from e
    if r.status_code == 401:
        raise HAUnauthorized("token rejected")
    if r.status_code != 200:
        raise HABadResponse(r.status_code, r.text[:200])
    try:
        body = r.json()
    except Exception as e:
        raise HABadResponse(r.status_code, f"invalid JSON: {e}") from e
    if not isinstance(body, list):
        raise HABadResponse(r.status_code, "expected JSON array")
    return body


async def get_history(
    url: str,
    token: str,
    entity_ids: list[str],
    start: datetime,
    end: datetime,
    *,
    timeout: float = 30.0,
) -> dict[str, list[dict[str, Any]]]:
    """GET <url>/api/history/period/<start>?filter_entity_id=...&end_time=...&minimal_response.

    Returns {entity_id: [{"ts": iso8601, "state": str, "attributes": dict}, ...]}.
    Raises HAError subclasses on failure. Empty entity list returns {} without
    making a request.
    """
    if not entity_ids:
        return {}
    base = _normalise_url(url)
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    # NOTE: we deliberately omit `minimal_response`. HA's REST API treats it
    # as a flag (no value), but httpx always emits `key=value` for params,
    # producing `&minimal_response=` which HA may reject; more importantly,
    # `minimal_response` strips `attributes` from non-first rows in each
    # entity's series, which contradicts the spec's Q3=B decision to pass
    # full attributes JSON to the LLM. Bandwidth cost is fine: this fires
    # once per diary generation (nightly).
    params = {
        "filter_entity_id": ",".join(entity_ids),
        "end_time": end_iso,
    }
    async with _client(timeout) as client:
        try:
            r = await client.get(
                f"{base}/api/history/period/{start_iso}",
                headers=_headers(token),
                params=params,
            )
        except httpx.HTTPError as e:
            raise HAUnreachable(_safe_url_for_log(base)) from e
    if r.status_code == 401:
        raise HAUnauthorized("token rejected")
    if r.status_code != 200:
        raise HABadResponse(r.status_code, r.text[:200])
    try:
        body = r.json()
    except Exception as e:
        raise HABadResponse(r.status_code, f"invalid JSON: {e}") from e
    if not isinstance(body, list):
        raise HABadResponse(r.status_code, "expected JSON array of arrays")

    out: dict[str, list[dict[str, Any]]] = {eid: [] for eid in entity_ids}
    for entity_series in body:
        if not isinstance(entity_series, list) or not entity_series:
            continue
        for row in entity_series:
            eid = row.get("entity_id")
            if not eid:
                # minimal_response strips entity_id from non-first rows; reuse last seen
                eid = entity_series[0].get("entity_id")
            if not eid or eid not in out:
                continue
            out[eid].append({
                "ts": row.get("last_updated") or row.get("last_changed") or "",
                "state": row.get("state", ""),
                "attributes": row.get("attributes", {}) or {},
            })
    return out
