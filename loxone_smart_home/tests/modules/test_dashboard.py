"""Unit tests for the monitoring dashboard's pure helpers.

The dashboard (modules/growatt/dashboard.py) is the public production page on
port 5555. Most of it is aiohttp wiring, but the price-row builder and the
actuals query helper contain real arithmetic/timezone/cache logic worth
locking down so a refactor of the shared helpers can't silently change output.
"""
from datetime import datetime
from types import SimpleNamespace

import pytest

from modules.growatt import dashboard


class _FakeConfig:
    distribution_tariff_high = 0.913
    distribution_tariff_low = 0.116
    low_tariff_hours = "0-10,11-12,13-14,15-17,18-24"


def _fake_ctrl():
    return SimpleNamespace(config=_FakeConfig())


def test_build_price_rows_basic_economics_and_status():
    ctrl = _fake_ctrl()
    price_items = {
        ("10:00", "10:15"): 3.0,
        ("11:00", "11:15"): 5.0,
    }.items()
    rows = dashboard._build_price_rows(
        ctrl, price_items, "today",
        charging={("10:00", "10:15")},
        pre_discharge=set(),
        discharge={("11:00", "11:15")},
        sell_production=set(),
        soc_lookup={},
        sell_fee=0.5,
        batt_amort=2.0,
        inv_off_threshold=-2.1,
        cur_block=("10:00", "10:15"),
    )
    assert [r["start"] for r in rows] == ["10:00", "11:00"]
    charging_row, discharge_row = rows
    # Status reflects the per-action block-sets.
    assert charging_row["status"] == "charging"
    assert charging_row["is_current"] is True
    assert discharge_row["status"] == "discharge"
    assert discharge_row["is_current"] is False
    # net_sell = price - distribution - sell_fee - batt_amort (verified against
    # the row's own resolved distribution, so it's not brittle to the tariff
    # schedule). Distribution must be one of the two configured tariffs.
    assert charging_row["distribution_czk"] in (0.91, 0.12)  # high/low, rounded
    assert charging_row["net_sell_czk"] == round(
        3.0 - charging_row["distribution_czk"] - 0.5 - 2.0, 2
    )
    assert charging_row["day"] == "today"


def test_build_price_rows_inverter_off_only_when_not_charging():
    ctrl = _fake_ctrl()
    price_items = {
        ("02:00", "02:15"): -3.0,  # below threshold, not charging → inverter_off
        ("03:00", "03:15"): -3.0,  # below threshold but charging → charging wins
    }.items()
    rows = dashboard._build_price_rows(
        ctrl, price_items, "today",
        charging={("03:00", "03:15")},
        pre_discharge=set(), discharge=set(), sell_production=set(),
        soc_lookup={}, sell_fee=0.5, batt_amort=2.0,
        inv_off_threshold=-2.1, cur_block=None,
    )
    assert rows[0]["status"] == "inverter_off"
    assert rows[0]["is_inverter_off"] is True
    assert rows[1]["status"] == "charging"
    assert rows[1]["is_inverter_off"] is False


def test_build_price_rows_projection_lookup_uses_day_prefix():
    ctrl = _fake_ctrl()
    soc_lookup = {"tomorrow:10:00": {"soc": 55.0, "action": "charge"}}
    rows = dashboard._build_price_rows(
        ctrl, {("10:00", "10:15"): 1.0}.items(), "tomorrow",
        charging=set(), pre_discharge=set(), discharge=set(), sell_production=set(),
        soc_lookup=soc_lookup, sell_fee=0.5, batt_amort=2.0,
        inv_off_threshold=-2.1, cur_block=None,
    )
    assert rows[0]["projected_soc"] == 55.0
    assert rows[0]["projected_action"] == "charge"


class _FakeRecord:
    def __init__(self, t, v):
        self._t, self._v = t, v

    def get_time(self):
        return self._t

    def get_value(self):
        return self._v


class _FakeTable:
    def __init__(self, records):
        self.records = records


class _FakeInflux:
    def __init__(self, records):
        self._records = records

    async def query(self, q):
        return [_FakeTable(self._records)]


def _actuals_ctrl(records):
    return SimpleNamespace(
        influxdb_client=_FakeInflux(records),
        settings=SimpleNamespace(influxdb=SimpleNamespace(bucket_solar="solar")),
    )


@pytest.mark.asyncio
async def test_today_local_actuals_buckets_today_and_applies_transform():
    today = datetime.now().date()
    # Naive timestamps (no tzinfo) are used verbatim; one today, one yesterday.
    rec_today = _FakeRecord(datetime(today.year, today.month, today.day, 9, 0), 2000.0)
    rec_old = _FakeRecord(datetime(2000, 1, 1, 9, 0), 9999.0)
    ctrl = _actuals_ctrl([rec_today, rec_old])
    payload = await dashboard._today_local_actuals(
        ctrl, field="InputPower", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: round(w / 1000.0, 3),
        result_key="hourly", cache_attr="_test_cache", ttl=300,
    )
    # Only today's record kept; 2000 W → 2.0 kWh; yesterday dropped.
    assert payload == {"hourly": {"09:00": 2.0}}


@pytest.mark.asyncio
async def test_today_local_actuals_no_client_returns_empty():
    payload = await dashboard._today_local_actuals(
        SimpleNamespace(influxdb_client=None),
        field="SOC", every="15m", agg_fn="last", time_fmt="%H:%M",
        value_fn=lambda s: round(s, 1),
        result_key="blocks", cache_attr="_c", ttl=90,
    )
    assert payload == {"blocks": {}}


@pytest.mark.asyncio
async def test_today_local_actuals_uses_cache_within_ttl():
    ctrl = _actuals_ctrl([])
    ctrl._test_cache = (datetime.now(), {"hourly": {"cached": 1.0}})
    payload = await dashboard._today_local_actuals(
        ctrl, field="InputPower", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: w, result_key="hourly", cache_attr="_test_cache", ttl=300,
    )
    assert payload == {"hourly": {"cached": 1.0}}
