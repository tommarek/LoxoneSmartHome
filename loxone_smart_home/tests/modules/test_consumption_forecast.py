"""Tests for the temperature-aware consumption forecast module."""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

from modules.growatt.consumption_forecast import ConsumptionModel, ConsumptionForecast


class TestConsumptionModel:

    def test_temp_to_bucket_normal(self) -> None:
        # 0°C → bucket (0+20)/2 = 10
        assert ConsumptionModel.temp_to_bucket(0.0) == 10
        # 20°C → bucket (20+20)/2 = 20
        assert ConsumptionModel.temp_to_bucket(20.0) == 20
        # -10°C → bucket (-10+20)/2 = 5
        assert ConsumptionModel.temp_to_bucket(-10.0) == 5

    def test_temp_to_bucket_clamped(self) -> None:
        # Below -20 gets clamped
        assert ConsumptionModel.temp_to_bucket(-30.0) == 0
        # Above 40 gets clamped
        assert ConsumptionModel.temp_to_bucket(50.0) == 30

    def test_build_computes_medians(self) -> None:
        model = ConsumptionModel()
        # Add data: winter morning (bucket 10 = 0°C, hour 8, weekday)
        key = (10, 8, False)
        model.bins[key] = [2.0, 2.5, 3.0, 2.2, 2.8]

        model.build()

        assert key in model.medians
        assert model.medians[key] == pytest.approx(2.5)  # Median of sorted list

    def test_build_outlier_removal(self) -> None:
        model = ConsumptionModel()
        key = (10, 8, False)
        # Normal values around 2.0-3.0, one extreme outlier at 20.0
        model.bins[key] = [2.0, 2.5, 2.2, 2.8, 3.0, 2.3, 2.7, 20.0]

        model.build()

        # Outlier (20.0) should be removed, median should be ~2.5
        assert model.medians[key] < 3.5

    def test_build_few_samples_no_outlier_removal(self) -> None:
        model = ConsumptionModel()
        key = (10, 8, False)
        model.bins[key] = [2.0, 10.0]  # Only 2 samples

        model.build()

        # With < 3 samples, no IQR removal, just median
        assert model.medians[key] == pytest.approx(6.0)

    def test_predict_exact_match(self) -> None:
        model = ConsumptionModel()
        model.medians = {(10, 8, False): 2.5}
        model.hourly_fallback = {8: 2.0}
        model.global_median = 1.5

        result = model.predict(temperature=0.0, hour=8, is_weekend=False)
        assert result == 2.5

    def test_predict_fallback_opposite_daytype(self) -> None:
        model = ConsumptionModel()
        # Only have weekend data for this bucket/hour
        model.medians = {(10, 8, True): 1.8}
        model.hourly_fallback = {8: 2.0}
        model.global_median = 1.5

        # Asking for weekday, falls back to weekend
        result = model.predict(temperature=0.0, hour=8, is_weekend=False)
        assert result == 1.8

    def test_predict_fallback_adjacent_bucket(self) -> None:
        model = ConsumptionModel()
        # Only have data for bucket 11 (2°C), not 10 (0°C)
        model.medians = {(11, 8, False): 2.3}
        model.hourly_fallback = {8: 2.0}
        model.global_median = 1.5

        result = model.predict(temperature=0.0, hour=8, is_weekend=False)
        assert result == 2.3

    def test_predict_fallback_hourly(self) -> None:
        model = ConsumptionModel()
        model.medians = {}  # No bucket matches
        model.hourly_fallback = {8: 2.0}
        model.global_median = 1.5

        result = model.predict(temperature=0.0, hour=8, is_weekend=False)
        assert result == 2.0

    def test_predict_fallback_global(self) -> None:
        model = ConsumptionModel()
        model.medians = {}
        model.hourly_fallback = {}
        model.global_median = 1.5

        result = model.predict(temperature=0.0, hour=8, is_weekend=False)
        assert result == 1.5


class TestConsumptionForecast:

    def test_needs_rebuild_when_no_model(self) -> None:
        cf = ConsumptionForecast()
        assert cf.needs_rebuild() is True

    def test_predict_hourly_correct_mapping(self) -> None:
        cf = ConsumptionForecast()
        model = ConsumptionModel()
        # Set up simple model: cold weather = more consumption
        for hour in range(24):
            model.medians[(5, hour, False)] = 3.0  # Cold weekday: 3 kWh/hr
            model.medians[(5, hour, True)] = 2.0   # Cold weekend: 2 kWh/hr
            model.medians[(20, hour, False)] = 1.0  # Warm weekday: 1 kWh/hr
        model.hourly_fallback = {h: 2.0 for h in range(24)}
        model.global_median = 2.0
        model.built_at = datetime.now()
        cf._model = model
        cf._last_model_build = datetime.now()

        # Cold weekday
        temps = {h: -10.0 for h in range(24)}  # bucket 5
        result = cf.predict_hourly(temps, date(2026, 4, 14))  # Monday
        assert all(v == 3.0 for v in result.values())
        assert len(result) == 24

    def test_predict_daily_sums_hours(self) -> None:
        cf = ConsumptionForecast()
        model = ConsumptionModel()
        for hour in range(24):
            model.medians[(10, hour, False)] = 2.0
        model.hourly_fallback = {h: 2.0 for h in range(24)}
        model.global_median = 2.0
        model.built_at = datetime.now()
        cf._model = model

        result = cf.predict_daily(0.0, date(2026, 4, 14))  # weekday
        assert result == pytest.approx(48.0)  # 24 hours * 2.0 kWh

    def test_predict_daily_no_model_returns_default(self) -> None:
        cf = ConsumptionForecast()
        result = cf.predict_daily(10.0, date(2026, 4, 14))
        assert result == 24.0  # Default 1 kWh/hr * 24

    def test_model_summary_no_model(self) -> None:
        cf = ConsumptionForecast()
        summary = cf.get_model_summary()
        assert summary["status"] == "not built"

    @pytest.mark.asyncio
    async def test_build_model_from_influxdb(self) -> None:
        """Test model building with mocked InfluxDB data."""
        cf = ConsumptionForecast(logger=MagicMock())

        # Create mock consumption records (hourly power in watts)
        consumption_records = []
        temp_records = []
        for day_offset in range(30):
            for hour in range(24):
                dt = datetime(2026, 3, 1 + day_offset % 28, hour, 0)

                # Consumption: ~2000W average (higher in morning/evening)
                watts = 1500 + (500 if hour in (7, 8, 18, 19) else 0)
                c_rec = MagicMock()
                c_rec.get_time.return_value = dt
                c_rec.get_value.return_value = float(watts)
                consumption_records.append(c_rec)

                # Temperature: ~5°C
                t_rec = MagicMock()
                t_rec.get_time.return_value = dt
                t_rec.get_value.return_value = 5.0
                temp_records.append(t_rec)

        c_table = MagicMock()
        c_table.records = consumption_records
        t_table = MagicMock()
        t_table.records = temp_records

        mock_client = AsyncMock()
        mock_client.query = AsyncMock(side_effect=[[c_table], [t_table]])

        mock_settings = MagicMock()
        mock_settings.influxdb.bucket_solar = "solar"
        mock_settings.influxdb.bucket_loxone = "loxone"

        success = await cf.build_model(mock_client, mock_settings)
        assert success is True
        assert cf.model is not None
        assert cf.model.data_points > 0
        assert len(cf.model.medians) > 0
