"""Tests for the ML (skforecast) consumption forecaster.

Exercises the real skforecast training path when installed; otherwise the
availability/fallback behaviour is still checked and the training tests skip.
"""

import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from modules.growatt.ml_consumption_forecast import (
    MLConsumptionForecast,
    SKFORECAST_AVAILABLE,
    ML_MODEL_SCHEMA_VERSION,
)


class FakeRecord:
    def __init__(self, t, v):
        self._t = t
        self._v = v

    def get_time(self):
        return self._t

    def get_value(self):
        return self._v


class FakeTable:
    def __init__(self, records):
        self.records = records


class FakeInfluxClient:
    """Returns synthetic consumption/temperature based on query content."""

    def __init__(self, days=21):
        self.days = days

    async def query(self, q):
        start = datetime(2026, 1, 1, 0, 0, 0)
        n = self.days * 24
        if "INVPowerToLocalLoad" in q:
            recs = []
            for i in range(n):
                t = start + timedelta(hours=i)
                # Daily sinusoid in Watts, always positive (build filters v>0).
                w = 500 + 300 * math.sin(2 * math.pi * (t.hour / 24.0))
                recs.append(FakeRecord(t, w))
            return [FakeTable(recs)]
        if "temperature_outside" in q:
            recs = []
            for i in range(n):
                t = start + timedelta(hours=i)
                temp = 5 + 5 * math.sin(2 * math.pi * (t.hour / 24.0))
                recs.append(FakeRecord(t, temp))
            return [FakeTable(recs)]
        if "inverter_state" in q:
            # No change records → carry-forward defaults all hours to on.
            return [FakeTable([])]
        return []


class GappyInfluxClient(FakeInfluxClient):
    """FakeInfluxClient with an outage hole in the middle.

    The hole spans GAP_HOURS from GAP_START. The gap policy measures
    contiguous NaN runs on the RAW hourly series (before any filling):
    a >24h gap TRUNCATES training to the contiguous data after the most
    recent such gap (declining only when <336h remain); ≤24h is bridged
    by interpolate(limit=24).
    """

    GAP_START_HOURS = 24 * 10
    GAP_HOURS = 24 * 4

    async def query(self, q):
        tables = await super().query(q)
        if "inverter_state" in q or not tables:
            return tables
        start = datetime(2026, 1, 1, 0, 0, 0)
        gap_lo = start + timedelta(hours=self.GAP_START_HOURS)
        gap_hi = gap_lo + timedelta(hours=self.GAP_HOURS)
        recs = [
            r for r in tables[0].records
            if not (gap_lo <= r.get_time() < gap_hi)
        ]
        return [FakeTable(recs)]


def make_settings():
    return SimpleNamespace(
        influxdb=SimpleNamespace(bucket_solar="solar", bucket_loxone="loxone")
    )


def test_needs_rebuild_when_untrained():
    m = MLConsumptionForecast()
    assert m.needs_rebuild() is True
    assert m.is_trained is False


def test_available_flag_matches_import():
    assert MLConsumptionForecast().available == SKFORECAST_AVAILABLE


@pytest.mark.asyncio
async def test_build_returns_false_without_skforecast(monkeypatch):
    import modules.growatt.ml_consumption_forecast as mod
    monkeypatch.setattr(mod, "SKFORECAST_AVAILABLE", False)
    m = MLConsumptionForecast()
    ok = await m.build_model(FakeInfluxClient(), make_settings())
    assert ok is False


@pytest.mark.asyncio
async def test_build_returns_false_with_too_little_data():
    m = MLConsumptionForecast()
    # 3 days < 2-week minimum → skip.
    ok = await m.build_model(FakeInfluxClient(days=3), make_settings())
    assert ok is False
    assert m.is_trained is False


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_build_and_predict():
    m = MLConsumptionForecast()
    ok = await m.build_model(FakeInfluxClient(days=21), make_settings())
    assert ok is True
    assert m.is_trained is True
    assert m._schema_version == ML_MODEL_SCHEMA_VERSION
    assert m.needs_rebuild() is False

    forecast_temps = {h: 6.0 for h in range(24)}
    target = (m._last_train_index_end + timedelta(hours=1)).date()
    preds = m.predict_hourly(forecast_temps, target, steps=24)
    assert isinstance(preds, dict)
    assert len(preds) > 0
    # Predictions should be in a sane kWh range (synthetic load ~0.2-0.8 kWh).
    for hour, kwh in preds.items():
        assert 0 <= hour <= 23
        assert 0.0 <= kwh < 5.0


def test_predict_without_training_returns_empty():
    m = MLConsumptionForecast()
    assert m.predict_hourly({h: 10.0 for h in range(24)}, datetime(2026, 1, 1).date()) == {}


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_predict_covers_target_date_after_training():
    # The recursive forecaster can only forecast forward from training-end.
    # Predicting the day AFTER train-end must still return a full set of hours
    # (the stale-anchor bug returned {} for any date past train-end+1).
    m = MLConsumptionForecast()
    ok = await m.build_model(FakeInfluxClient(days=21), make_settings())
    assert ok is True
    next_day = (m._last_train_index_end + timedelta(days=1)).date()
    preds = m.predict_hourly({h: 6.0 for h in range(24)}, next_day)
    assert len(preds) >= 20  # essentially all 24 hours of the next day


def test_predict_bails_when_model_too_stale():
    # A target far beyond the training window would require an enormous
    # recursive horizon — predict_hourly must bail (return {}) rather than
    # compound error over hundreds of steps. Tested at the guard seam (no
    # skforecast fit needed): a stale anchor + far target must short-circuit
    # BEFORE the forecaster is ever consulted.
    m = MLConsumptionForecast()
    # Simulate a trained-but-stale model without invoking skforecast.
    m._last_train_index_end = datetime(2026, 1, 21, 23, 0, 0)

    class _BoomForecaster:
        def predict(self, *a, **k):  # must never be reached
            raise AssertionError("forecaster consulted despite stale guard")

    m._forecaster = _BoomForecaster()
    far = (m._last_train_index_end + timedelta(days=30)).date()
    assert m.predict_hourly({h: 6.0 for h in range(24)}, far) == {}


def test_predict_bails_for_date_before_training():
    m = MLConsumptionForecast()
    m._last_train_index_end = datetime(2026, 1, 21, 23, 0, 0)
    m._forecaster = object()  # sentinel; should not be used
    past = (m._last_train_index_end - timedelta(days=2)).date()
    assert m.predict_hourly({h: 6.0 for h in range(24)}, past) == {}


def test_needs_rebuild_default_one_day():
    # Recursive AR must refresh often; default max age is 1 day.
    import modules.growatt.ml_consumption_forecast as mod
    m = MLConsumptionForecast()
    # Untrained → always rebuild.
    assert m.needs_rebuild() is True


@pytest.mark.asyncio
async def test_queries_use_start_labeled_windows(monkeypatch):
    """aggregateWindow must be start-labeled (timeSrc: "_start").

    The Flux default stamps each window with its STOP time, shifting every
    hour key +1h and delaying the whole predicted load profile.
    """
    pytest.importorskip("pandas")
    import modules.growatt.ml_consumption_forecast as mod
    # Force past the availability gate so the queries are actually issued;
    # the empty results bail out long before skforecast is touched.
    monkeypatch.setattr(mod, "SKFORECAST_AVAILABLE", True)

    captured = []

    class RecordingClient:
        async def query(self, q):
            captured.append(q)
            return []

    m = MLConsumptionForecast()
    ok = await m.build_model(RecordingClient(), make_settings())
    assert ok is False  # no data — but the queries were issued

    agg_queries = [q for q in captured if "aggregateWindow" in q]
    assert len(agg_queries) == 2  # consumption + temperature
    for q in agg_queries:
        assert 'timeSrc: "_start"' in q


@pytest.mark.asyncio
async def test_needs_rebuild_backoff_after_failed_build():
    """A failed training attempt must not be retried on every poll."""
    m = MLConsumptionForecast()
    # Fails regardless of environment: without skforecast at the availability
    # gate, with skforecast at the 2-week-minimum data check.
    ok = await m.build_model(FakeInfluxClient(days=3), make_settings())
    assert ok is False

    # Within the backoff window: no retry, despite being untrained.
    assert m.needs_rebuild() is False

    # After the window expires the rebuild is wanted again.
    m._last_build_attempt = datetime.now() - timedelta(
        seconds=m.failed_build_backoff_seconds + 1
    )
    assert m.needs_rebuild() is True


def test_needs_rebuild_success_cadence_unchanged():
    """A SUCCESSFUL build keeps the existing 1-day retrain cadence."""
    m = MLConsumptionForecast()
    # Simulate a successful build: attempt recorded, then training completed.
    m._forecaster = object()
    m._schema_version = ML_MODEL_SCHEMA_VERSION
    m._last_build_attempt = datetime.now() - timedelta(seconds=10)
    m._last_train_time = datetime.now()
    assert m.needs_rebuild() is False

    # Two days later it retrains (backoff must not mask a stale success).
    m._last_build_attempt = datetime.now() - timedelta(days=2)
    m._last_train_time = datetime.now() - timedelta(days=2, seconds=-10)
    assert m.needs_rebuild() is True


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_build_truncates_at_long_gap_and_trains():
    """A >24h contiguous RAW gap no longer declines outright — it truncates
    to the contiguous data AFTER the gap and trains on that.

    Declining outright disabled the ML engine for up to a year after a
    single long outage anywhere in the 365-day window. With the gap at day
    10 of 30 (4-day hole), 16 days (384h ≥ 336h) survive after it.
    """
    from unittest.mock import MagicMock
    logger = MagicMock()
    m = MLConsumptionForecast(logger=logger)
    ok = await m.build_model(GappyInfluxClient(days=30), make_settings())
    assert ok is True
    assert m.is_trained is True
    # The truncation is logged (what was dropped, what remains).
    assert any(
        "truncated" in str(c.args[0])
        for c in logger.info.call_args_list
    )
    # Training end is the last hour of the synthetic series (Jan 30 23:00 —
    # 30 days of hourly data starting Jan 1 00:00).
    assert m._last_train_index_end == datetime(2026, 1, 30, 23, 0, 0)


class ThirtyHourGapClient(GappyInfluxClient):
    """30h gap "months ago": plenty of contiguous data remains after it."""
    GAP_HOURS = 30
    GAP_START_HOURS = 24 * 5  # post-gap: 30d*24 - 150h = 570h ≥ 336h


class RecentThirtyHourGapClient(GappyInfluxClient):
    """30h gap ending ~5 days before the end of the window: only 120h of
    contiguous data survive after it — below the 336h minimum."""
    GAP_HOURS = 30
    GAP_START_HOURS = 30 * 24 - 30 - 24 * 5  # gap ends 5 days before end


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_build_trains_on_data_after_old_30h_gap():
    """A 30h gap months back must NOT disable the engine — training uses
    the post-gap contiguous data (≥336h remain)."""
    m = MLConsumptionForecast()
    ok = await m.build_model(ThirtyHourGapClient(days=30), make_settings())
    assert ok is True
    assert m.is_trained is True


@pytest.mark.asyncio
async def test_build_declines_on_recent_30h_gap(monkeypatch):
    """A 30h gap 5 days ago leaves only 120h of post-gap data (<336h) —
    truncation can't help, training declines.

    Also pins the raw-measurement semantics: the 30h outage must be
    DETECTED as a long gap on the RAW series (a partial interpolate first
    would under-report it as 24h).
    """
    pytest.importorskip("pandas")
    import modules.growatt.ml_consumption_forecast as mod
    # The decline happens before any skforecast use; force past the
    # availability gate so the gap logic is exercised even without it.
    monkeypatch.setattr(mod, "SKFORECAST_AVAILABLE", True)

    m = MLConsumptionForecast()
    ok = await m.build_model(RecentThirtyHourGapClient(days=30), make_settings())
    assert ok is False
    assert m.is_trained is False


class MisalignedFirstWindowClient(FakeInfluxClient):
    """Mimics Flux's real labelling: the FIRST window of a range(start:-365d)
    is partial and labelled at the non-hour-aligned range start (e.g. :51:09),
    while every later window is on the hour. The oldest series index is then
    off-hour, which used to anchor asfreq("1h") off the grid → every real point
    missed its slot → a fabricated >24h gap → ML wrongly declined. Flooring the
    index to the hour fixes it."""

    async def query(self, q):
        tables = await super().query(q)
        if "inverter_state" in q or not tables:
            return tables
        recs = tables[0].records
        if recs:
            # Shift the first record off the hour by 51m09.629938s.
            t0 = recs[0].get_time().replace(minute=51, second=9, microsecond=629938)
            recs[0] = FakeRecord(t0, recs[0].get_value())
        return [FakeTable(recs)]


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_build_trains_despite_offhour_first_window():
    """Regression: a non-hour-aligned first timestamp (Flux partial range-start
    window) must not manufacture a spurious >24h gap — index is hour-floored, so
    asfreq aligns and training succeeds on the full window."""
    m = MLConsumptionForecast()
    ok = await m.build_model(MisalignedFirstWindowClient(days=21), make_settings())
    assert ok is True
    assert m.is_trained is True


class ModerateGapClient(GappyInfluxClient):
    GAP_HOURS = 20


@pytest.mark.skipif(not SKFORECAST_AVAILABLE, reason="skforecast not installed")
@pytest.mark.asyncio
async def test_build_trains_through_moderate_gap():
    """Gaps ≤24h are bridged (interpolate(limit=24)) and train fine.

    The old interpolate(limit=6) + ffill/bfill(limit=3) could only bridge
    12h, so 13-24h gaps declined with a misleading message.
    """
    m = MLConsumptionForecast()
    ok = await m.build_model(ModerateGapClient(days=30), make_settings())
    assert ok is True
    assert m.is_trained is True
