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
