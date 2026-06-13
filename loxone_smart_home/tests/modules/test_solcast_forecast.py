"""Tests for the Solcast PV forecast client (parsing + config gating)."""

import asyncio
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp
import pytest

from modules.growatt import solcast_forecast as sf_module
from modules.growatt.solcast_forecast import SolcastForecast


PRAGUE = ZoneInfo("Europe/Prague")


def test_disabled_without_credentials():
    assert SolcastForecast(api_key="", rooftop_id="").enabled is False
    assert SolcastForecast(api_key="k", rooftop_id="").enabled is False
    assert SolcastForecast(api_key="", rooftop_id="r").enabled is False
    assert SolcastForecast(api_key="k", rooftop_id="r").enabled is True


def test_whitespace_credentials_treated_as_empty():
    f = SolcastForecast(api_key="  ", rooftop_id="\t")
    assert f.enabled is False


@pytest.mark.asyncio
async def test_fetch_disabled_returns_empty():
    f = SolcastForecast(api_key="", rooftop_id="")
    assert await f.fetch_hourly_today_tomorrow() == {}


def test_parse_converts_kw_to_kwh_and_buckets_local():
    # Summer (CEST = UTC+2). period_end 10:00Z → local 12:00,
    # interval START 11:30 → hour 11.
    payload = {
        "forecasts": [
            {"period_end": "2026-05-29T10:00:00.0000000Z", "pv_estimate": 4.0},
            {"period_end": "2026-05-29T10:30:00.0000000Z", "pv_estimate": 6.0},
        ]
    }
    r = SolcastForecast._parse_forecasts(payload, PRAGUE)
    assert r == {"2026-05-29": {11: 2.0, 12: 3.0}}


def test_parse_sums_two_half_hours_in_same_hour():
    # Two intervals that both land in local hour 11.
    payload = {
        "forecasts": [
            {"period_end": "2026-05-29T09:30:00Z", "pv_estimate": 2.0},  # start 11:00
            {"period_end": "2026-05-29T10:00:00Z", "pv_estimate": 2.0},  # start 11:30
        ]
    }
    r = SolcastForecast._parse_forecasts(payload, PRAGUE)
    assert r["2026-05-29"][11] == pytest.approx(2.0)  # (2*0.5)+(2*0.5)


def test_parse_skips_malformed_entries():
    payload = {
        "forecasts": [
            {"pv_estimate": 4.0},  # no period_end → skipped
            {"period_end": "not-a-date", "pv_estimate": 4.0},  # bad date → skipped
            {"period_end": "2026-05-29T10:00:00Z", "pv_estimate": 4.0},  # valid
        ]
    }
    r = SolcastForecast._parse_forecasts(payload, PRAGUE)
    assert sum(sum(h.values()) for h in r.values()) == pytest.approx(2.0)


def test_parse_empty_payload():
    assert SolcastForecast._parse_forecasts({}, PRAGUE) == {}
    assert SolcastForecast._parse_forecasts(None, PRAGUE) == {}


def _quantile_payload():
    return {
        "forecasts": [
            {
                "period_end": "2026-05-29T10:00:00Z",  # start 11:00 local (CEST)
                "pv_estimate": 2.0,
                "pv_estimate10": 1.0,
                "pv_estimate90": 3.0,
            }
        ]
    }


def test_parse_selects_p10_and_p90():
    p10 = SolcastForecast._parse_forecasts(_quantile_payload(), PRAGUE, "p10")
    p50 = SolcastForecast._parse_forecasts(_quantile_payload(), PRAGUE, "p50")
    p90 = SolcastForecast._parse_forecasts(_quantile_payload(), PRAGUE, "p90")
    # kWh = kW * 0.5h for the single 30-min interval (bucketed at local hour 11).
    assert p10["2026-05-29"][11] == pytest.approx(0.5)
    assert p50["2026-05-29"][11] == pytest.approx(1.0)
    assert p90["2026-05-29"][11] == pytest.approx(1.5)


def test_parse_quantile_falls_back_to_p50_when_field_absent():
    payload = {
        "forecasts": [
            {"period_end": "2026-05-29T10:00:00Z", "pv_estimate": 2.0}  # no p10
        ]
    }
    r = SolcastForecast._parse_forecasts(payload, PRAGUE, "p10")
    assert r["2026-05-29"][11] == pytest.approx(1.0)  # fell back to pv_estimate


def test_constructor_normalises_unknown_quantile():
    assert SolcastForecast(api_key="k", rooftop_id="r", quantile="bogus").quantile == "p50"
    assert SolcastForecast(api_key="k", rooftop_id="r", quantile="p90").quantile == "p90"


def test_multiple_rooftop_ids_parsed():
    f = SolcastForecast(api_key="k", rooftop_id="sw-uuid, se-uuid")
    assert f.rooftop_ids == ["sw-uuid", "se-uuid"]
    assert f.rooftop_id == "sw-uuid"  # back-compat: first id
    assert f.enabled is True


@pytest.mark.asyncio
async def test_multi_site_budget_blocks_partial_refresh(tmp_path):
    # Two sites cost 2 calls; with only 1 left in the budget, the refresh must
    # not start (else we'd spend it on a one-array forecast) → returns cache.
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="a,b", quota_path=tmp_path / "q.json")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._req_day = datetime.now(timezone.utc).date()
    f._req_count = f._max_requests_per_day - 1  # room for 1, need 2
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}
    assert f._req_count == f._max_requests_per_day - 1  # untouched


@pytest.mark.asyncio
async def test_auth_failure_disables_without_network(tmp_path):
    # Once a permanent auth error is recorded, fetch must short-circuit to the
    # cache and never hit the network again (protecting the daily budget).
    # quota_path is isolated so the test never reads/writes the real
    # /app/config_state file when run in-container.
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=tmp_path / "q.json")
    f._auth_failed = True
    f._cached = {"2026-05-29": {12: 3.0}}
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}


@pytest.mark.asyncio
async def test_daily_budget_exhausted_returns_cache(tmp_path):
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=tmp_path / "q.json")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._req_day = datetime.now(timezone.utc).date()
    f._req_count = f._max_requests_per_day  # budget spent
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}


@pytest.mark.asyncio
async def test_interval_throttle_returns_cache(tmp_path):
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=tmp_path / "q.json")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._last_attempt = datetime.now(timezone.utc)  # just attempted
    out = await f.fetch_hourly_today_tomorrow(min_refresh_interval_hours=3)
    assert out == {"2026-05-29": {12: 3.0}}


# --- Quota persistence + connection-failure refunds -------------------------


class _FakeResponse:
    """Minimal stand-in for an aiohttp response (HTTP response WAS received)."""

    def __init__(self, status=200, body="", payload=None):
        self.status = status
        self._body = body
        self._payload = payload

    async def text(self):
        return self._body

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stand-in for aiohttp.ClientSession; either returns a response or raises."""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_session(monkeypatch, session):
    monkeypatch.setattr(
        sf_module.aiohttp, "ClientSession", lambda *a, **k: session
    )


def test_quota_state_load_tolerates_missing_and_corrupt(tmp_path):
    missing = tmp_path / "nope" / "solcast_quota.json"
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=missing)
    assert f._req_count == 0 and f._last_attempt is None and not f._auth_failed

    corrupt = tmp_path / "solcast_quota.json"
    corrupt.write_text("{not json!!")
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=corrupt)
    assert f._req_count == 0 and f._last_attempt is None and not f._auth_failed

    wrong_shape = tmp_path / "solcast_quota2.json"
    wrong_shape.write_text("[1, 2, 3]")
    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=wrong_shape)
    assert f._req_count == 0


@pytest.mark.asyncio
async def test_http_error_burns_budget_and_persists(monkeypatch, tmp_path):
    # A received HTTP response (here 500) must still count against the budget,
    # and the spent state must survive a "restart" (new instance, same path).
    path = tmp_path / "solcast_quota.json"
    session = _FakeSession(response=_FakeResponse(status=500, body="boom"))
    _patch_session(monkeypatch, session)

    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=path)
    await f.fetch_hourly_today_tomorrow()
    assert session.calls == 1
    assert f._req_count == 1  # burned, no refund
    assert f._last_attempt is not None

    saved = json.loads(path.read_text())
    assert saved["req_count"] == 1
    assert saved["utc_day"] == datetime.now(timezone.utc).date().isoformat()
    assert saved["last_attempt_iso"]
    assert saved["auth_failed"] is False

    # "Restart": a fresh instance restores the spent budget AND the interval
    # throttle, so it serves the cache without touching the network.
    session2 = _FakeSession(response=_FakeResponse(status=500, body="boom"))
    _patch_session(monkeypatch, session2)
    f2 = SolcastForecast(api_key="k", rooftop_id="r", quota_path=path)
    assert f2._req_count == 1
    assert f2._last_attempt is not None
    f2._cached = {"2026-05-29": {12: 3.0}}
    out = await f2.fetch_hourly_today_tomorrow(min_refresh_interval_hours=3)
    assert out == {"2026-05-29": {12: 3.0}}
    assert session2.calls == 0  # interval throttle persisted across restart


@pytest.mark.asyncio
async def test_connect_failure_refunds_budget(monkeypatch, tmp_path):
    # The request provably never went out (DNS/refused →
    # ClientConnectorError) → Solcast's real quota was not touched, so the
    # reserved unit must be refunded (a LAN outage must not zero the day's
    # budget). The interval throttle still applies.
    from unittest.mock import MagicMock

    path = tmp_path / "solcast_quota.json"
    session = _FakeSession(
        exc=aiohttp.ClientConnectorError(MagicMock(), OSError("dns boom"))
    )
    _patch_session(monkeypatch, session)

    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=path)
    await f.fetch_hourly_today_tomorrow()
    assert session.calls == 1
    assert f._req_count == 0  # refunded
    assert f._last_attempt is not None  # interval throttle NOT refunded
    assert json.loads(path.read_text())["req_count"] == 0


@pytest.mark.asyncio
async def test_timeout_burns_budget(monkeypatch, tmp_path):
    # A timeout may have reached Solcast and been counted by its real
    # ledger — refunding locally could let us exceed the actual quota, so
    # timeouts burn the reserved unit.
    path = tmp_path / "solcast_quota.json"
    session = _FakeSession(exc=asyncio.TimeoutError())
    _patch_session(monkeypatch, session)

    f = SolcastForecast(api_key="k", rooftop_id="r", quota_path=path)
    await f.fetch_hourly_today_tomorrow()
    assert f._req_count == 1  # NOT refunded


@pytest.mark.asyncio
async def test_auth_failure_latch_persists_for_same_key_only(monkeypatch, tmp_path):
    # 401 latches auth_failed and persists it; a restart with the SAME key
    # stays latched (no network), but a CHANGED key gets a fresh chance.
    path = tmp_path / "solcast_quota.json"
    session = _FakeSession(response=_FakeResponse(status=401, body="bad key"))
    _patch_session(monkeypatch, session)

    f = SolcastForecast(api_key="bad-key", rooftop_id="r", quota_path=path)
    await f.fetch_hourly_today_tomorrow()
    assert f._auth_failed is True
    assert json.loads(path.read_text())["auth_failed"] is True

    # Same key after restart → still latched, never hits the network.
    session2 = _FakeSession(response=_FakeResponse(status=401, body="bad key"))
    _patch_session(monkeypatch, session2)
    f2 = SolcastForecast(api_key="bad-key", rooftop_id="r", quota_path=path)
    assert f2._auth_failed is True
    await f2.fetch_hourly_today_tomorrow()
    assert session2.calls == 0

    # Different (fixed) key → latch dropped, fetch attempts again.
    f3 = SolcastForecast(api_key="good-key", rooftop_id="r", quota_path=path)
    assert f3._auth_failed is False


@pytest.mark.asyncio
async def test_successful_fetch_persists_spent_budget(monkeypatch, tmp_path):
    path = tmp_path / "solcast_quota.json"
    payload = {
        "forecasts": [
            {"period_end": "2026-05-29T10:00:00Z", "pv_estimate": 4.0},
        ]
    }
    session = _FakeSession(response=_FakeResponse(status=200, payload=payload))
    _patch_session(monkeypatch, session)

    f = SolcastForecast(api_key="k", rooftop_id="a,b", quota_path=path)
    out = await f.fetch_hourly_today_tomorrow(local_tz=PRAGUE)
    assert out  # parsed something
    assert session.calls == 2  # one per site
    assert f._req_count == 2  # whole batch reserved and kept
    assert json.loads(path.read_text())["req_count"] == 2
