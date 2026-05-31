"""Tests for the Solcast PV forecast client (parsing + config gating)."""

from zoneinfo import ZoneInfo

import pytest

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
async def test_multi_site_budget_blocks_partial_refresh():
    # Two sites cost 2 calls; with only 1 left in the budget, the refresh must
    # not start (else we'd spend it on a one-array forecast) → returns cache.
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="a,b")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._req_day = datetime.now(timezone.utc).date()
    f._req_count = f._max_requests_per_day - 1  # room for 1, need 2
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}
    assert f._req_count == f._max_requests_per_day - 1  # untouched


@pytest.mark.asyncio
async def test_auth_failure_disables_without_network():
    # Once a permanent auth error is recorded, fetch must short-circuit to the
    # cache and never hit the network again (protecting the daily budget).
    f = SolcastForecast(api_key="k", rooftop_id="r")
    f._auth_failed = True
    f._cached = {"2026-05-29": {12: 3.0}}
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}


@pytest.mark.asyncio
async def test_daily_budget_exhausted_returns_cache():
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="r")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._req_day = datetime.now(timezone.utc).date()
    f._req_count = f._max_requests_per_day  # budget spent
    out = await f.fetch_hourly_today_tomorrow()
    assert out == {"2026-05-29": {12: 3.0}}


@pytest.mark.asyncio
async def test_interval_throttle_returns_cache():
    from datetime import datetime, timezone
    f = SolcastForecast(api_key="k", rooftop_id="r")
    f._cached = {"2026-05-29": {12: 3.0}}
    f._last_attempt = datetime.now(timezone.utc)  # just attempted
    out = await f.fetch_hourly_today_tomorrow(min_refresh_interval_hours=3)
    assert out == {"2026-05-29": {12: 3.0}}
