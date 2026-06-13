"""Regression tests for the branch-review fixes.

Self-contained: defines its own minimal fake InfluxDB records so it needs no
shared fixtures and no optional deps (PuLP/skforecast). Covers:
- timezone bucketing in the shared `_carry_forward_hourly` helper
- solar calibration recency weighting (chronological ordering)
- new config validators (solar_arrays, low_tariff_hours)
"""

from datetime import datetime, time, timedelta, timezone

import pytest

from config.settings import GrowattConfig
from modules.growatt.consumption_forecast import _carry_forward_hourly
from modules.growatt.solar_forecast import DailyForecast, SolarArray, SolarForecast

try:
    from zoneinfo import ZoneInfo
    PRAGUE = ZoneInfo("Europe/Prague")
except Exception:  # pragma: no cover
    PRAGUE = None


class _Rec:
    """Minimal stand-in for an InfluxDB record."""

    def __init__(self, t: datetime, value):
        self._t = t
        self._v = value

    def get_time(self):
        return self._t

    def get_value(self):
        return self._v


class _Table:
    def __init__(self, records):
        self.records = records


class _Client:
    """Async InfluxDB client returning a fixed table list for any query."""

    def __init__(self, tables):
        self._tables = tables

    async def query(self, _query):
        return self._tables


# --- timezone bucketing in the shared carry-forward helper -----------------

@pytest.mark.skipif(PRAGUE is None, reason="zoneinfo unavailable")
def test_carry_forward_converts_change_times_to_local():
    """A state change at 22:30 UTC belongs to hour 23 *local* (Prague, +1 in
    winter), so the carry-forward must key it by the local hour, not 22 (UTC)."""
    change = _Rec(datetime(2026, 1, 15, 22, 30, tzinfo=timezone.utc), 0)
    # Both the UTC-hour and local-hour keys are offered; only the local one
    # should receive the state when local_tz is supplied.
    hour_keys = ["2026-01-15-22", "2026-01-15-23"]

    out = _carry_forward_hourly([_Table([change])], hour_keys, local_tz=PRAGUE)

    assert out.get("2026-01-15-23") == 0       # local hour gets the 0-state
    assert out.get("2026-01-15-22") != 0       # UTC hour does not (None/absent)


def test_carry_forward_without_tz_uses_utc_hour():
    """Backwards-compatible: no local_tz → naive/UTC bucketing (unchanged)."""
    change = _Rec(datetime(2026, 1, 15, 22, 30, tzinfo=timezone.utc), 0)
    out = _carry_forward_hourly(
        [_Table([change])], ["2026-01-15-22", "2026-01-15-23"], local_tz=None
    )
    assert out.get("2026-01-15-22") == 0       # UTC hour gets it without tz


# --- solar calibration recency weighting -----------------------------------

async def test_calibration_weights_recent_days_chronologically():
    """`calibrate_from_actuals` weights newer days more (weight=i+1). The
    actuals dict must be iterated chronologically regardless of the order the
    InfluxDB records arrive in, else the weighting is meaningless."""
    fc = SolarForecast(
        arrays=[SolarArray("a", 35, 180, 10.0)],
        latitude=49.0, longitude=14.5,
    )
    # Forecast (denominator) is 10 kWh for each day.
    for day in ("2026-03-01", "2026-03-02", "2026-03-03"):
        d = datetime.strptime(day, "%Y-%m-%d").date()
        fc._consensus[day] = DailyForecast(date=d, total_kwh=10.0, source="model+api")

    # Actuals returned in REVERSED (non-chronological) order to prove sorting:
    #   newest 03-03 ratio 1.5 (15/10, clamped), 03-02 ratio 1.0, 03-01 ratio 0.3.
    recs = [
        _Rec(datetime(2026, 3, 3, 12, tzinfo=timezone.utc), 15.0),
        _Rec(datetime(2026, 3, 2, 12, tzinfo=timezone.utc), 10.0),
        _Rec(datetime(2026, 3, 1, 12, tzinfo=timezone.utc), 3.0),
    ]
    client = _Client([_Table(recs)])

    await fc.calibrate_from_actuals(client, bucket="solar", days=30)

    # Chronological weighting: ratios [0.3, 1.0, 1.5] with weights [1, 2, 3]
    #   = (0.3*1 + 1.0*2 + 1.5*3) / 6 = 6.8 / 6 ≈ 1.133.
    # The buggy insertion-order weighting would give ≈0.733 — assert we are
    # clearly on the chronological side.
    assert fc.confidence == pytest.approx(1.133, abs=0.02)


# --- config validators ------------------------------------------------------

def test_solar_arrays_validator_accepts_default():
    default = GrowattConfig.model_fields["solar_arrays"].default
    assert GrowattConfig.validate_solar_arrays(default) == default


@pytest.mark.parametrize("bad", ["not json", '[{"name": 1}]', "{}"])
def test_solar_arrays_validator_rejects_bad(bad):
    with pytest.raises(ValueError):
        GrowattConfig.validate_solar_arrays(bad)


def test_solar_arrays_validator_allows_empty():
    # An empty array is valid: "no arrays configured" (solar forecast disabled).
    # The gated consumer decides what to do; startup must not hard-fail.
    assert GrowattConfig.validate_solar_arrays("[]") == "[]"


def test_low_tariff_hours_validator_accepts_default():
    default = GrowattConfig.model_fields["low_tariff_hours"].default
    assert GrowattConfig.validate_low_tariff_hours(default) == default


@pytest.mark.parametrize("bad", ["9-9", "0-25", "a-b", "17-9", "5"])
def test_low_tariff_hours_validator_rejects_bad(bad):
    with pytest.raises(ValueError):
        GrowattConfig.validate_low_tariff_hours(bad)


# --- MILP: clamp live SOC into [min_soc, max_soc] --------------------------

pulp = pytest.importorskip("pulp")


def test_milp_clamps_soc_above_max_instead_of_infeasible():
    """A live SOC above max_soc must be clamped before pinning soc[0]; an
    out-of-bounds start used to make the MILP infeasible (silent greedy
    fallback every tick). After the fix the solve succeeds and every block's
    SOC stays within the configured window."""
    from modules.growatt.milp_optimizer import MILPBatteryOptimizer

    base = datetime(2026, 5, 31, 0, 0)
    blocks = [(base + timedelta(minutes=15 * i), 2.0 + (i % 4)) for i in range(8)]

    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={h: 1.0 for h in range(24)},
        distribution_func=lambda h: 1.0,
        battery_capacity_kwh=10.0,
        current_soc=100.0,   # above max_soc=90 → must be clamped
        min_soc=20.0,
        max_soc=90.0,
        charge_rate_kw=2.5,
        discharge_rate_kw=2.5,
        discharge_power_pct=25.0,
        efficiency=0.85,
    )

    assert decisions, "optimizer returned no decisions"
    # Clamp means the starting SOC is reported at the ceiling, not 100.
    assert decisions[0].soc_before <= 90.0 + 1e-6
    for d in decisions:
        assert 20.0 - 1e-6 <= d.soc_after <= 90.0 + 1e-6


def test_block_key_uses_2400_sentinel_for_midnight_end():
    """Regression: the last block of the day (23:45) must key its end as
    '24:00', not '00:00', so it matches _current_prices / current_block_key and
    is actually actuated. (Pre-existing bug found in review.)"""
    from datetime import datetime
    from modules.growatt_controller import _block_key
    # Block ending at next-day midnight → '24:00' sentinel.
    assert _block_key(
        datetime(2026, 6, 1, 23, 45), datetime(2026, 6, 2, 0, 0)
    ) == ("23:45", "24:00")
    # Ordinary intraday block → plain HH:MM.
    assert _block_key(
        datetime(2026, 6, 1, 10, 0), datetime(2026, 6, 1, 10, 15)
    ) == ("10:00", "10:15")
    # A genuine same-day 00:00 start (not a midnight end) is unaffected.
    assert _block_key(
        datetime(2026, 6, 1, 0, 0), datetime(2026, 6, 1, 0, 15)
    ) == ("00:00", "00:15")
