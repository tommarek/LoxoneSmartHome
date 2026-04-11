"""Tests for the solar production forecast module."""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from modules.growatt.solar_forecast import SolarForecast, SolarArray, DailyForecast


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
                {"time": "2026-04-11T10:00:00", "shortwave_radiation": 500},
                {"time": "2026-04-11T11:00:00", "shortwave_radiation": 800},
                {"time": "2026-04-11T12:00:00", "shortwave_radiation": 900},
            ]
        }
        result = forecast.calculate_from_weather(weather_data)
        assert "2026-04-11" in result
        day = result["2026-04-11"]
        assert day.total_kwh > 0
        # 13.5 kWp * 0.80 perf_ratio * (500+800+900)/1000 = 23.76 kWh
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

    def test_sources_diverge_uses_conservative(self, forecast) -> None:
        """When sources diverge >20%, use the lower value."""
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
        # Divergence = |10-5| / 7.5 = 66% > 20% → conservative (min)
        assert result["2026-04-11"].hourly[12] == 5.0

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
        # Set up API forecast history for comparison
        forecast._api_forecast = {
            "2026-04-10": DailyForecast(
                date=date(2026, 4, 10), total_kwh=50.0, hourly={}, source="api",
            ),
            "2026-04-09": DailyForecast(
                date=date(2026, 4, 9), total_kwh=45.0, hourly={}, source="api",
            ),
        }

        # Mock InfluxDB returning actual production
        mock_client = AsyncMock()
        mock_record_1 = MagicMock()
        mock_record_1.get_time.return_value = datetime(2026, 4, 10)
        mock_record_1.get_value.return_value = 48.0  # Close to forecast

        mock_record_2 = MagicMock()
        mock_record_2.get_time.return_value = datetime(2026, 4, 9)
        mock_record_2.get_value.return_value = 40.0

        mock_table = MagicMock()
        mock_table.records = [mock_record_1, mock_record_2]
        mock_client.query = AsyncMock(return_value=[mock_table])

        await forecast.calibrate_from_actuals(mock_client, "solar", days=7)

        # Confidence should be adjusted (not still 0.7)
        # Day 1: 40/45 = 0.89, Day 2: 48/50 = 0.96
        # Weighted: (0.89*1 + 0.96*2) / 3 = 0.937
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
