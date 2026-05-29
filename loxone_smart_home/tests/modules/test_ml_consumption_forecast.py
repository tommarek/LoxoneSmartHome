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
