"""Tests for the solar production forecast module."""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from modules.growatt.solar_forecast import (
    SolarForecast, SolarArray, SolarProductionModel, DailyForecast,
)


@pytest.fixture
def solar_arrays() -> list:
    return [
        SolarArray(name="terasa", declination=35, azimuth=234, kwp=7.0),
        SolarArray(name="ulice", declination=35, azimuth=134, kwp=6.5),
    ]


@pytest.fixture
def forecast(solar_arrays) -> SolarForecast:
    return SolarForecast(
        arrays=solar_arrays,
        latitude=49.0,
        longitude=14.5,
        confidence=0.7,
        logger=MagicMock(),
    )


class TestSolarForecastInit:

    def test_from_config(self) -> None:
        config = MagicMock()
        config.solar_arrays = '[{"name":"a","declination":30,"azimuth":180,"kwp":5.0}]'
        config.latitude = 49.0
        config.longitude = 14.5
        config.solar_forecast_confidence = 0.8

        sf = SolarForecast.from_config(config)
        assert len(sf.arrays) == 1
        assert sf.arrays[0].name == "a"
        assert sf.arrays[0].kwp == 5.0
        assert sf.confidence == 0.8

    def test_from_config_two_arrays(self) -> None:
        config = MagicMock()
        config.solar_arrays = (
            '[{"name":"terasa","declination":35,"azimuth":234,"kwp":7.0},'
            '{"name":"ulice","declination":35,"azimuth":134,"kwp":6.5}]'
        )
        config.latitude = 49.0
        config.longitude = 14.5
        config.solar_forecast_confidence = 0.7

        sf = SolarForecast.from_config(config)
        assert len(sf.arrays) == 2
        assert sum(a.kwp for a in sf.arrays) == 13.5


class TestWeatherCalculation:

    def test_basic_ghi_calculation(self, forecast) -> None:
        """Test that GHI converts to reasonable kWh production."""
        weather_data = {
            "hourly": [
                {"time": "2026-04-11T10:00:00", "shortwave_radiation": 500, "temperature_2m": 0},
                {"time": "2026-04-11T11:00:00", "shortwave_radiation": 800, "temperature_2m": 0},
                {"time": "2026-04-11T12:00:00", "shortwave_radiation": 900, "temperature_2m": 0},
            ]
        }
        result = forecast.calculate_from_weather(weather_data)
        assert "2026-04-11" in result
        day = result["2026-04-11"]
        assert day.total_kwh > 0
        # 13.5 kWp * 0.80 perf_ratio * 1.0 temp_factor * (500+800+900)/1000 = 23.76 kWh
        # temp_factor=1.0 at 0°C ambient (cell=25°C=STC)
        assert abs(day.total_kwh - 23.76) < 0.1

    def test_zero_radiation_excluded(self, forecast) -> None:
        weather_data = {
            "hourly": [
                {"time": "2026-04-11T02:00:00", "shortwave_radiation": 0},
                {"time": "2026-04-11T12:00:00", "shortwave_radiation": 600},
            ]
        }
        result = forecast.calculate_from_weather(weather_data)
        day = result["2026-04-11"]
        # Only hour 12 contributes
        assert len(day.hourly) == 1
        assert 12 in day.hourly

    def test_empty_weather_data(self, forecast) -> None:
        result = forecast.calculate_from_weather({"hourly": []})
        assert result == {}


class TestConsensus:

    def test_both_sources_agree(self, forecast) -> None:
        """When API and weather agree within 20%, use average."""
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=40.0,
                hourly={10: 5.0, 11: 8.0, 12: 10.0}, source="api",
            )
        }
        forecast._weather_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=42.0,
                hourly={10: 5.5, 11: 8.5, 12: 9.5}, source="weather",
            )
        }
        result = forecast.build_consensus()
        day = result["2026-04-11"]
        # Within 20% → average
        assert day.hourly[10] == pytest.approx(5.25, abs=0.01)
        assert day.source == "consensus"

    def test_sources_diverge_uses_average(self, forecast) -> None:
        """When sources diverge and no model available, use average."""
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="api",
            )
        }
        forecast._weather_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=30.0,
                hourly={12: 5.0}, source="weather",
            )
        }
        result = forecast.build_consensus()
        # No model available → average all sources
        assert result["2026-04-11"].hourly[12] == pytest.approx(7.5, abs=0.01)

    def test_model_underpredicts_uses_average(self, forecast) -> None:
        """When model predicts much less than other sources, use average (sparse bin)."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=10.0,
                hourly={12: 1.0}, source="model",
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=40.0,
                hourly={12: 8.0}, source="api",
            )
        }
        forecast._weather_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=35.0,
                hourly={12: 7.0}, source="weather",
            )
        }
        result = forecast.build_consensus()
        # Model=1.0, avg_others=7.5 → model much lower → use other sources only
        assert result["2026-04-11"].hourly[12] == pytest.approx(7.5, abs=0.01)

    def test_model_overpredicts_trusts_model(self, forecast) -> None:
        """When model predicts more than other sources, trust it (real data)."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="model",
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=30.0,
                hourly={12: 5.0}, source="api",
            )
        }
        forecast._weather_forecast = {}
        result = forecast.build_consensus()
        # Model=10.0, avg_others=5.0 → model higher → trust model
        assert result["2026-04-11"].hourly[12] == 10.0

    def test_single_source_used_directly(self, forecast) -> None:
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=35.0,
                hourly={12: 8.0}, source="api",
            )
        }
        forecast._weather_forecast = {}
        result = forecast.build_consensus()
        assert result["2026-04-11"].total_kwh == 35.0


class TestConfidenceDiscount:

    def test_production_discounted(self, forecast) -> None:
        forecast._consensus = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={10: 5.0, 12: 10.0}, source="consensus",
            )
        }
        assert forecast.get_expected_production_kwh(date(2026, 4, 11)) == pytest.approx(35.0)

    def test_hourly_discounted(self, forecast) -> None:
        forecast._consensus = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={10: 5.0, 12: 10.0}, source="consensus",
            )
        }
        hourly = forecast.get_hourly_production(date(2026, 4, 11))
        assert hourly[10] == pytest.approx(3.5)
        assert hourly[12] == pytest.approx(7.0)

    def test_missing_date_returns_zero(self, forecast) -> None:
        forecast._consensus = {}
        assert forecast.get_expected_production_kwh(date(2026, 4, 11)) == 0.0
        assert forecast.get_hourly_production(date(2026, 4, 11)) == {}


class TestCalibration:

    @pytest.mark.asyncio
    async def test_calibration_adjusts_confidence(self, forecast) -> None:
        """Calibrate from actuals when forecast history exists."""
        # Set up API forecast history for comparison (need 3+ days for weighted avg)
        forecast._api_forecast = {
            "2026-04-10": DailyForecast(
                date=date(2026, 4, 10), total_kwh=50.0, hourly={}, source="api",
            ),
            "2026-04-09": DailyForecast(
                date=date(2026, 4, 9), total_kwh=45.0, hourly={}, source="api",
            ),
            "2026-04-08": DailyForecast(
                date=date(2026, 4, 8), total_kwh=40.0, hourly={}, source="api",
            ),
        }

        # Mock InfluxDB returning actual production
        mock_client = AsyncMock()
        mock_record_1 = MagicMock()
        mock_record_1.get_time.return_value = datetime(2026, 4, 10)
        mock_record_1.get_value.return_value = 48.0

        mock_record_2 = MagicMock()
        mock_record_2.get_time.return_value = datetime(2026, 4, 9)
        mock_record_2.get_value.return_value = 40.0

        mock_record_3 = MagicMock()
        mock_record_3.get_time.return_value = datetime(2026, 4, 8)
        mock_record_3.get_value.return_value = 36.0

        mock_table = MagicMock()
        mock_table.records = [mock_record_1, mock_record_2, mock_record_3]
        mock_client.query = AsyncMock(return_value=[mock_table])

        await forecast.calibrate_from_actuals(mock_client, "solar", days=7)

        # Confidence should be adjusted (not still 0.7)
        # Day 1: 36/40 = 0.90, Day 2: 40/45 = 0.89, Day 3: 48/50 = 0.96
        # Weighted: (0.90*1 + 0.89*2 + 0.96*3) / 6 = 0.924
        assert forecast.confidence != 0.7
        assert 0.8 < forecast.confidence < 1.0

    @pytest.mark.asyncio
    async def test_calibration_fallback_no_forecast(self, forecast) -> None:
        """Use actual vs theoretical max when no forecast history."""
        forecast._api_forecast = {}

        mock_record = MagicMock()
        mock_record.get_time.return_value = datetime(2026, 4, 10)
        mock_record.get_value.return_value = 45.0  # 13.5 kWp * 5 = 67.5 max

        mock_table = MagicMock()
        mock_table.records = [mock_record]
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=[mock_table])

        await forecast.calibrate_from_actuals(mock_client, "solar", days=7)
        # 45 / 67.5 ≈ 0.667
        assert 0.6 < forecast.confidence < 0.75

    @pytest.mark.asyncio
    async def test_calibration_no_data(self, forecast) -> None:
        """No data available — confidence unchanged."""
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=[])

        original = forecast.confidence
        await forecast.calibrate_from_actuals(mock_client, "solar", days=7)
        assert forecast.confidence == original


class TestSolarProductionModel:

    def test_predict_3d_hit(self) -> None:
        """3D bin is the top level and is used when populated."""
        model = SolarProductionModel(total_kwp=13.5)
        # rad_b=20 (500 W/m²), cloud_b=2 (20%), alt_b=9 (45°)
        model.add_sample(20, 2, 9, 5.0)
        model.add_sample(20, 2, 9, 5.5)
        model.build()
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=45)
        assert result == pytest.approx(5.25, abs=0.1)
        assert model.hit_level_counts["3d"] == 1

    def test_predict_2d_fallback(self) -> None:
        """Falls back to 2D when 3D bin is empty for the queried altitude."""
        model = SolarProductionModel(total_kwp=13.5)
        # Only populate altitude bucket 9 (45°)
        model.add_sample(20, 2, 9, 5.0)
        model.build()
        # Query altitude 30° (bucket 6) — misses 3D, hits 2D
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=30)
        assert result > 0
        assert model.hit_level_counts["2d"] == 1

    def test_predict_interpolation_fallback(self) -> None:
        """Falls back to interpolation when 2D exact is empty."""
        model = SolarProductionModel(total_kwp=13.5)
        # Populate rad bucket 18 and 22 with cloud bucket 2
        model.add_sample(18, 2, 9, 4.0)
        model.add_sample(22, 2, 9, 6.0)
        model.build()
        # Query rad bucket 20 (ghi=500), cloud bucket 2 (20%), altitude bucket 4 (20°)
        # No 3D match at (20,2,4), no 2D exact at (20,2), interpolates between (18,2) and (22,2)
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=20)
        assert result > 0
        assert model.hit_level_counts["interpolate"] == 1

    def test_no_5d_or_4d_bins(self) -> None:
        """5D and 4D bin attributes no longer exist."""
        model = SolarProductionModel()
        assert not hasattr(model, "bins_5d")
        assert not hasattr(model, "median_5d")
        assert not hasattr(model, "bins_4d")
        assert not hasattr(model, "median_4d")


class TestConsensusFloatEquality:

    def test_model_same_value_as_other_source(self, forecast) -> None:
        """When model and another source produce the same float value,
        both are correctly identified by source identity."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=40.0,
                hourly={12: 8.0}, source="model",
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=40.0,
                hourly={12: 8.0}, source="api",
            )
        }
        forecast._weather_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=35.0,
                hourly={12: 7.0}, source="weather",
            )
        }
        result = forecast.build_consensus()
        # avg_others = (8.0 + 7.0) / 2 = 7.5
        # divergence = |8.0 - 7.5| / 7.75 ≈ 0.065 < 0.3 → average all
        assert result["2026-04-11"].hourly[12] == pytest.approx(
            (8.0 + 8.0 + 7.0) / 3, abs=0.01
        )


class TestTemperatureDerating:

    def test_hot_day_derating(self, forecast) -> None:
        """Hot summer day: 35°C ambient → cell 60°C → ~14% derating."""
        weather_data = {
            "hourly": [
                {"time": "2026-07-15T12:00:00", "shortwave_radiation": 800, "temperature_2m": 35},
            ]
        }
        result = forecast.calculate_from_weather(weather_data)
        day = result["2026-07-15"]
        # base: 800 * 13.5 * 0.80 / 1000 = 8.64
        # temp_factor at 35°C: cell=60°C, factor = 1 + (-0.004)*(60-25) = 0.86
        # expected: 8.64 * 0.86 = 7.43
        assert day.total_kwh == pytest.approx(7.43, abs=0.1)

    def test_cold_day_near_stc(self, forecast) -> None:
        """Cold day: 5°C ambient → cell 30°C → slight derating from STC."""
        weather_data = {
            "hourly": [
                {"time": "2026-01-15T12:00:00", "shortwave_radiation": 300, "temperature_2m": 5},
            ]
        }
        result = forecast.calculate_from_weather(weather_data)
        day = result["2026-01-15"]
        # base: 300 * 13.5 * 0.80 / 1000 = 3.24
        # temp_factor at 5°C: cell=30°C, factor = 1 + (-0.004)*(30-25) = 0.98
        # expected: 3.24 * 0.98 = 3.175
        assert day.total_kwh == pytest.approx(3.18, abs=0.1)


class TestReliableForecast:

    def test_winter_low_forecast_is_reliable(self, forecast) -> None:
        """3 kWh in December is reliable for a 13.5 kWp system."""
        forecast._consensus = {
            "2026-12-15": DailyForecast(
                date=date(2026, 12, 15), total_kwh=3.0,
                hourly={12: 2.0, 13: 1.0}, source="consensus",
            )
        }
        assert forecast.has_reliable_forecast() is True

    def test_winter_very_low_forecast_unreliable(self, forecast) -> None:
        """0.5 kWh in December is unreliable even in winter."""
        forecast._consensus = {
            "2026-12-15": DailyForecast(
                date=date(2026, 12, 15), total_kwh=0.5,
                hourly={12: 0.5}, source="consensus",
            )
        }
        assert forecast.has_reliable_forecast() is False

    def test_summer_low_forecast_unreliable(self, forecast) -> None:
        """3 kWh in June is unreliable for a 13.5 kWp system."""
        forecast._consensus = {
            "2026-06-15": DailyForecast(
                date=date(2026, 6, 15), total_kwh=3.0,
                hourly={12: 2.0, 13: 1.0}, source="consensus",
            )
        }
        # min_kwh = 13.5 * 0.3 = 4.05, so 3.0 < 4.05 → unreliable
        assert forecast.has_reliable_forecast() is False

    def test_summer_normal_forecast_reliable(self, forecast) -> None:
        """25 kWh in June is reliable."""
        forecast._consensus = {
            "2026-06-15": DailyForecast(
                date=date(2026, 6, 15), total_kwh=25.0,
                hourly={10: 3.0, 11: 5.0, 12: 7.0, 13: 6.0, 14: 4.0},
                source="consensus",
            )
        }
        assert forecast.has_reliable_forecast() is True

    def test_empty_consensus_unreliable(self, forecast) -> None:
        forecast._consensus = {}
        assert forecast.has_reliable_forecast() is False


class TestConsensusPersistence:

    @pytest.mark.asyncio
    async def test_save_consensus_to_influxdb(self, forecast) -> None:
        """Consensus forecasts are written to solar_forecast_history."""
        forecast._consensus = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=40.0,
                hourly={10: 5.0, 12: 8.0}, source="model+api",
            )
        }
        mock_client = AsyncMock()
        mock_client.write_point = AsyncMock()
        await forecast.save_consensus_to_influxdb(mock_client, "solar")

        mock_client.write_point.assert_called_once()
        call_kwargs = mock_client.write_point.call_args.kwargs
        assert call_kwargs["measurement"] == "solar_forecast_history"
        assert call_kwargs["fields"]["total_kwh"] == 40.0
        assert call_kwargs["tags"]["forecast_date"] == "2026-04-11"

    @pytest.mark.asyncio
    async def test_calibration_uses_influxdb_when_memory_empty(self, forecast) -> None:
        """After restart, calibration loads historical forecasts from InfluxDB."""
        forecast._api_forecast = {}
        forecast._consensus = {}
        forecast._model_forecast = {}

        # Mock actual production: 3 days (needed for weighted avg)
        mock_actual_records = []
        for day_offset, actual_kwh in [(3, 36.0), (2, 40.0), (1, 48.0)]:
            rec = MagicMock()
            rec.get_time.return_value = datetime(2026, 4, 12 - day_offset)
            rec.get_value.return_value = actual_kwh
            mock_actual_records.append(rec)

        actual_table = MagicMock()
        actual_table.records = mock_actual_records

        # Mock historical forecast from InfluxDB (solar_forecast_history)
        history_records = []
        for day_offset, forecast_kwh in [(3, 40.0), (2, 45.0), (1, 50.0)]:
            date_str = f"2026-04-{12 - day_offset:02d}"
            for field_name, field_val in [("total_kwh", forecast_kwh), ("source", "model+api")]:
                rec = MagicMock()
                rec.values = {"forecast_date": date_str}
                rec.get_field.return_value = field_name
                rec.get_value.return_value = field_val
                history_records.append(rec)

        history_table = MagicMock()
        history_table.records = history_records

        mock_client = AsyncMock()
        # First call: actual production query; second call: forecast history query
        mock_client.query = AsyncMock(side_effect=[[actual_table], [history_table]])

        await forecast.calibrate_from_actuals(mock_client, "solar", days=7)

        # Should have used InfluxDB history, not fallen back to theoretical max
        # Ratios: 36/40=0.9, 40/45=0.889, 48/50=0.96
        # Confidence should be in a reasonable calibrated range
        assert 0.8 < forecast.confidence < 1.0


class TestIntradayCalibration:
    """Live actual-vs-forecast scaling that the controller applies to the
    rest of today's solar forecast before passing it to the optimizer."""

    def _make_forecast_with_today(self, forecast, today_hourly: dict) -> None:
        """Plant a model-source consensus forecast for today with the given
        per-hour kWh values, so get_hourly_production(today) returns them."""
        from datetime import date as _date
        from modules.growatt.solar_forecast import DailyForecast
        today_obj = _date(2026, 5, 29)
        forecast._consensus[today_obj.strftime("%Y-%m-%d")] = DailyForecast(
            date=today_obj,
            total_kwh=sum(today_hourly.values()),
            hourly=today_hourly,
            source="model",
        )
        return today_obj

    def _mock_query(self, hourly_actuals_w: dict, target_date):
        """Build an AsyncMock that returns InputPower records at the given
        hours (watts) on target_date."""
        records = []
        for hour, watts in hourly_actuals_w.items():
            rec = MagicMock()
            t = datetime(target_date.year, target_date.month, target_date.day, hour, 0)
            rec.get_time.return_value = t
            rec.get_value.return_value = float(watts)
            records.append(rec)
        table = MagicMock()
        table.records = records
        client = AsyncMock()
        client.query = AsyncMock(return_value=[table])
        return client

    @pytest.mark.asyncio
    async def test_under_forecast_returns_ratio_above_1(self, forecast) -> None:
        """Reality > forecast → ratio > 1, scaling future hours UP."""
        today = self._make_forecast_with_today(
            forecast,
            # Cloudy-morning forecast: 0.5 kWh/hour through 09:00, then sun
            {h: 0.5 for h in range(6, 10)} | {h: 3.0 for h in range(10, 18)},
        )
        # Reality was 7× higher than forecast for the completed hours.
        client = self._mock_query({6: 3500, 7: 3500, 8: 3500}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=9, target_date=today,
        )
        assert ratio is not None
        # actual_sum=10.5, forecast_sum=1.5 → 7.0, clamped to 3.0
        assert ratio == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_over_forecast_returns_ratio_below_1(self, forecast) -> None:
        """Reality < forecast → ratio < 1, scaling future hours DOWN."""
        today = self._make_forecast_with_today(
            forecast, {h: 3.0 for h in range(6, 18)},
        )
        # Reality is half the forecast.
        client = self._mock_query({6: 1500, 7: 1500, 8: 1500}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=9, target_date=today,
        )
        assert ratio == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_clamped_to_low_bound(self, forecast) -> None:
        """Crazy-low actuals can't drive a runaway downward correction."""
        today = self._make_forecast_with_today(
            forecast, {h: 3.0 for h in range(6, 18)},
        )
        # Actual is 1/15 of forecast → raw ratio 0.067, must clamp to 0.5.
        # Need ≥0.2 kWh/h to pass the dawn/dusk usable filter (200W mean).
        client = self._mock_query({6: 200, 7: 200, 8: 200}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=9, target_date=today,
        )
        assert ratio == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_dawn_hours_excluded(self, forecast) -> None:
        """Pre-sunrise hours (zero actual) get filtered out even if forecast
        had non-zero values for them. Otherwise calibration would always go
        DOWN during early-morning calls."""
        today = self._make_forecast_with_today(
            forecast,
            {5: 0.5, 6: 0.5, 7: 1.0, 8: 2.0, 9: 3.0},
        )
        # Hour 5 is pre-dawn (zero actual); hour 6+ sun rising and matches
        # forecast roughly. Without the filter, ratio would include hour 5's
        # 0/0.5=0 → dragged down. With the filter, only hours 6 and 7 count.
        client = self._mock_query({5: 0, 6: 500, 7: 1000}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=8, target_date=today,
        )
        assert ratio is not None
        # Hours 6, 7 considered: actual 0.5+1.0=1.5, forecast 0.5+1.0=1.5 → 1.0
        assert ratio == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_completed_hours(self, forecast) -> None:
        """current_hour == 0 → nothing completed today, can't calibrate."""
        today = self._make_forecast_with_today(
            forecast, {h: 1.0 for h in range(6, 18)},
        )
        client = self._mock_query({}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=0, target_date=today,
        )
        assert ratio is None

    @pytest.mark.asyncio
    async def test_returns_none_when_forecast_tiny(self, forecast) -> None:
        """Pre-dawn forecast is too small to ratio against — return None."""
        today = self._make_forecast_with_today(
            forecast, {h: 0.01 for h in range(24)},
        )
        client = self._mock_query({5: 5, 6: 5}, today)

        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=7, target_date=today,
        )
        assert ratio is None  # forecast sum < min_forecast_kwh
