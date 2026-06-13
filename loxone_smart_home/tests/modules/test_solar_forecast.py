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

    def test_model_overpredicts_supported_bin_trusts_model(self, forecast) -> None:
        """When a well-supported (3d/2d) model bin predicts more, trust it."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="model",
                hourly_levels={12: "3d"},
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
        # Model=10.0, avg_others=5.0 → diverging, but bin well-supported → model
        assert result["2026-04-11"].hourly[12] == 10.0

    def test_model_overpredicts_sparse_bin_uses_others(self, forecast) -> None:
        """A diverging HIGHER model value from a sparse bin no longer wins —
        the old direction-based rule was a structural over-forecast bias."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="model",
                hourly_levels={12: "global"},
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
        assert result["2026-04-11"].hourly[12] == pytest.approx(5.0)

    def test_model_underpredicts_supported_bin_trusts_model(self, forecast) -> None:
        """A diverging LOWER model value from a well-supported bin wins —
        trust real installation data in BOTH directions."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=20.0,
                hourly={12: 4.0}, source="model",
                hourly_levels={12: "2d"},
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
        # Model=4.0, avg_others=7.5 → diverging, bin well-supported → model
        assert result["2026-04-11"].hourly[12] == pytest.approx(4.0)

    def test_sparse_2d_bin_underpredicting_uses_others(self, forecast) -> None:
        """A diverging LOWER value from a 1-sample 2D bin ("2d_sparse") must
        NOT override the other sources — one anomalous near-zero hour in a
        bin would otherwise kill Solcast+API+weather for that hour."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=5.0,
                hourly={12: 0.5}, source="model",
                hourly_levels={12: "2d_sparse"},
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
        # Model=0.5, avg_others=7.5 → diverging, sparse bin → avg_others
        assert result["2026-04-11"].hourly[12] == pytest.approx(7.5)

    def test_sparse_2d_bin_overpredicting_uses_others(self, forecast) -> None:
        """A diverging HIGHER value from a sparse 2D bin loses too."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="model",
                hourly_levels={12: "2d_sparse"},
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
        assert result["2026-04-11"].hourly[12] == pytest.approx(5.0)

    def test_supported_2d_bin_overpredicting_trusts_model(self, forecast) -> None:
        """A well-supported "2d" hit (≥ MIN_2D_TRUST_SAMPLES) still wins on
        divergence in the HIGHER direction (lower direction covered above)."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=50.0,
                hourly={12: 10.0}, source="model",
                hourly_levels={12: "2d"},
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
        assert result["2026-04-11"].hourly[12] == pytest.approx(10.0)

    def test_below_horizon_zero_vetoes_other_sources(self, forecast) -> None:
        """A genuine below-horizon model zero still zeroes the consensus."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=20.0,
                hourly={4: 0.0, 12: 10.0}, source="model",
                hourly_levels={4: "below_horizon", 12: "3d"},
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=30.0,
                hourly={4: 0.5, 12: 9.0}, source="api",
            )
        }
        forecast._weather_forecast = {}
        result = forecast.build_consensus()
        assert result["2026-04-11"].hourly[4] == 0

    def test_non_horizon_model_zero_does_not_veto(self, forecast) -> None:
        """A model zero that did NOT come from the below-horizon early return
        (e.g. a sparse/global bin) must not discard positive other sources."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=20.0,
                hourly={9: 0.0}, source="model",
                hourly_levels={9: "global"},
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=30.0,
                hourly={9: 2.0}, source="api",
            )
        }
        forecast._weather_forecast = {}
        result = forecast.build_consensus()
        # Diverging zero from a weak bin → use the other sources
        assert result["2026-04-11"].hourly[9] == pytest.approx(2.0)

    def test_no_radiation_zero_does_not_veto(self, forecast) -> None:
        """GHI=0 with the sun UP (a missing/zero radiation entry at a sunny
        hour) reports "no_radiation", which must NOT veto — the positive
        Solcast/API values survive via the normal divergence path (whose
        denominator (model+others)/2 is positive when others are)."""
        forecast._model_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=20.0,
                hourly={11: 0.0}, source="model",
                hourly_levels={11: "no_radiation"},
            )
        }
        forecast._solcast_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=30.0,
                hourly={11: 6.0}, source="solcast",
            )
        }
        forecast._api_forecast = {
            "2026-04-11": DailyForecast(
                date=date(2026, 4, 11), total_kwh=28.0,
                hourly={11: 5.0}, source="api",
            )
        }
        forecast._weather_forecast = {}
        result = forecast.build_consensus()
        # Model 0 (no_radiation) vs avg(6.0, 5.0)=5.5 → keep the average.
        assert result["2026-04-11"].hourly[11] == pytest.approx(5.5)

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
        assert forecast.confidence == pytest.approx(0.924, abs=0.01)

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
    async def test_truncated_final_window_dropped(self, forecast) -> None:
        """With timeSrc:"_start" the final daily window is truncated by the
        -1d range stop; its partial-day total must not enter the recency-
        weighted ratios (it would land on the heaviest weight)."""
        from datetime import timedelta, timezone
        now = datetime.now(timezone.utc)

        def _day_start(days_back: int) -> datetime:
            d = now - timedelta(days=days_back)
            return d.replace(hour=0, minute=0, second=0, microsecond=0)

        def _rec(t0, value):
            rec = MagicMock()
            rec.get_time.return_value = t0
            rec.get_value.return_value = value
            return rec

        # 3 complete days (ratio 1.0 each) + the truncated final window whose
        # partial total would otherwise contribute a wild ratio.
        records = []
        forecast._api_forecast = {}
        for days_back in (4, 3, 2):
            t0 = _day_start(days_back)
            forecast._api_forecast[t0.strftime("%Y-%m-%d")] = DailyForecast(
                date=t0.date(), total_kwh=40.0, hourly={}, source="api",
            )
            records.append(_rec(t0, 40.0))
        t_partial = _day_start(1)  # window end > range stop → incomplete
        forecast._api_forecast[t_partial.strftime("%Y-%m-%d")] = DailyForecast(
            date=t_partial.date(), total_kwh=40.0, hourly={}, source="api",
        )
        records.append(_rec(t_partial, 5.0))  # partial-day total

        table = MagicMock()
        table.records = records
        client = AsyncMock()
        client.query = AsyncMock(return_value=[table])

        await forecast.calibrate_from_actuals(client, "solar", days=7)
        # Surviving ratios are all 1.0; the partial day (ratio 5/40 → clamped
        # 0.3 at the heaviest weight) would have dragged confidence to ≈0.72.
        assert forecast.confidence == pytest.approx(1.0, abs=0.01)

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
        """3D bin is the top level and is used when well-populated (≥4)."""
        model = SolarProductionModel(total_kwp=13.5)
        # rad_b=20 (500 W/m²), cloud_b=2 (20%), alt_b=9 (45°)
        model.add_sample(20, 2, 9, 5.0)
        model.add_sample(20, 2, 9, 5.5)
        model.add_sample(20, 2, 9, 5.0)
        model.add_sample(20, 2, 9, 5.5)
        model.build()
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=45)
        assert result == pytest.approx(5.25, abs=0.1)
        assert model.hit_level_counts["3d"] == 1

    def test_3d_bin_below_min_support_falls_back_to_2d(self) -> None:
        """A 3D bin with <4 samples isn't published — its samples still
        contribute to the 2D level, which serves the prediction (as
        "2d_sparse": 3 samples is below the 2D trust threshold too)."""
        model = SolarProductionModel(total_kwp=13.5)
        model.add_sample(20, 2, 9, 5.0)
        model.add_sample(20, 2, 9, 5.5)
        model.add_sample(20, 2, 9, 5.0)  # only 3 samples → below support
        model.build()
        assert (20, 2, 9) not in model.median_3d
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=45)
        assert result == pytest.approx(5.0, abs=0.1)
        assert model.hit_level_counts["3d"] == 0
        assert model.hit_level_counts["2d_sparse"] == 1

    def test_predict_with_level_below_horizon(self) -> None:
        """Sun below the horizon reports the veto level; GHI=0 with the sun
        UP reports the distinct non-veto "no_radiation" level, so the
        consensus can tell a genuine night zero from a missing/zero
        radiation entry at a sunny hour."""
        model = SolarProductionModel(total_kwp=13.5)
        # Sun down → genuine zero, regardless of GHI.
        assert model.predict_with_level(ghi=500, sun_altitude=0) == (
            0.0, "below_horizon"
        )
        assert model.predict_with_level(ghi=0, sun_altitude=0) == (
            0.0, "below_horizon"
        )
        # GHI=0 but sun up (default altitude 45°) → no_radiation, not a veto.
        assert model.predict_with_level(ghi=0) == (0.0, "no_radiation")

    def test_2d_trust_level_depends_on_sample_count(self) -> None:
        """A 2D bin is published from a single sample, but only reports the
        consensus-trusted "2d" level at ≥ MIN_2D_TRUST_SAMPLES; sparser hits
        report "2d_sparse" with the SAME value."""
        sparse = SolarProductionModel(total_kwp=13.5)
        sparse.add_sample(20, 2, 9, 5.0)  # 1 sample
        sparse.build()
        # Query altitude bucket 6 → misses 3D, hits 2D (20, 2).
        val, level = sparse.predict_with_level(
            ghi=500, cloud_cover=20, sun_altitude=30
        )
        assert val == pytest.approx(5.0, abs=0.1)
        assert level == "2d_sparse"
        assert sparse.hit_level_counts["2d_sparse"] == 1
        assert sparse.hit_level_counts["2d"] == 0

        supported = SolarProductionModel(total_kwp=13.5)
        for kwh in (5.0, 5.5, 5.0, 5.5):  # ≥ MIN_2D_TRUST_SAMPLES, spread
            supported.add_sample(20, 2, 9, kwh)
        supported.build()
        val, level = supported.predict_with_level(
            ghi=500, cloud_cover=20, sun_altitude=30
        )
        assert val == pytest.approx(5.25, abs=0.1)
        assert level == "2d"
        assert supported.hit_level_counts["2d"] == 1

    def test_hit_level_counts_reset(self) -> None:
        """reset_hit_level_counts zeroes all counters."""
        model = SolarProductionModel(total_kwp=13.5)
        model.hit_level_counts["3d"] = 7
        model.hit_level_counts["global"] = 3
        model.reset_hit_level_counts()
        assert all(v == 0 for v in model.hit_level_counts.values())

    def test_predict_2d_fallback(self) -> None:
        """Falls back to 2D when 3D bin is empty for the queried altitude
        (a single-sample bin still serves the value, as "2d_sparse")."""
        model = SolarProductionModel(total_kwp=13.5)
        # Only populate altitude bucket 9 (45°)
        model.add_sample(20, 2, 9, 5.0)
        model.build()
        # Query altitude 30° (bucket 6) — misses 3D, hits 2D
        result = model.predict(ghi=500, cloud_cover=20, sun_altitude=30)
        assert result > 0
        assert model.hit_level_counts["2d_sparse"] == 1

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

    def test_quantile_default_is_median(self) -> None:
        from modules.growatt.solar_forecast import SolarProductionModel
        import statistics
        vals = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0]
        m = SolarProductionModel(quantile=0.5)
        assert m._compute_quantile(vals) == pytest.approx(statistics.median(vals))

    def test_higher_quantile_recovers_potential(self) -> None:
        from modules.growatt.solar_forecast import SolarProductionModel
        # Right-skewed bin (curtailment historically removed the sunniest highs).
        vals = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0]
        m50 = SolarProductionModel(quantile=0.5)
        m75 = SolarProductionModel(quantile=0.75)
        assert m75._compute_quantile(vals) > m50._compute_quantile(vals)

    def test_no_5d_or_4d_bins(self) -> None:
        """5D and 4D bin attributes no longer exist."""
        model = SolarProductionModel()
        assert not hasattr(model, "bins_5d")
        assert not hasattr(model, "median_5d")
        assert not hasattr(model, "bins_4d")
        assert not hasattr(model, "median_4d")


class TestPredictFromModelTelemetry:

    def test_hit_level_counts_reset_per_run(self, forecast) -> None:
        """Each predict_from_model run starts from zeroed counters so the
        telemetry shows the live prediction mix, not history since build."""
        model = SolarProductionModel(total_kwp=13.5)
        model.build()
        # Inflated by a previous run / pass-2 curtailment probing
        model.hit_level_counts["3d"] = 99
        forecast._production_model = model

        result = forecast.predict_from_model([
            {"time": "2026-06-12T12:00:00", "shortwave_radiation": 500,
             "cloudcover": 20, "temperature_2m": 20},
        ])

        # Counters reflect only this run (1 prediction, empty model → global)
        assert model.hit_level_counts["3d"] == 0
        assert sum(model.hit_level_counts.values()) == 1
        # The per-hour level is recorded for consensus gating
        assert result["2026-06-12"].hourly_levels[12] == "global"


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

    def test_stale_only_consensus_unreliable(self, forecast) -> None:
        """A consensus holding ONLY past-dated entries (never evicted) must
        not report a reliable forecast for today."""
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        forecast._consensus = {
            yesterday.strftime("%Y-%m-%d"): DailyForecast(
                date=yesterday, total_kwh=25.0,
                hourly={10: 3.0, 12: 7.0}, source="consensus",
            )
        }
        assert forecast.has_reliable_forecast() is False

    def test_stale_entry_plus_current_entry_reliable(self, forecast) -> None:
        """A stale entry alongside a healthy today entry doesn't break it."""
        from datetime import timedelta
        today = date.today()
        yesterday = today - timedelta(days=1)
        forecast._consensus = {
            yesterday.strftime("%Y-%m-%d"): DailyForecast(
                date=yesterday, total_kwh=0.1, hourly={}, source="consensus",
            ),
            today.strftime("%Y-%m-%d"): DailyForecast(
                date=today, total_kwh=25.0,
                hourly={10: 3.0, 12: 7.0}, source="consensus",
            ),
        }
        assert forecast.has_reliable_forecast() is True


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


class TestFluxTimeSrc:
    """All aggregateWindow pipelines must label windows by their START.

    The Flux default (timeSrc:"_stop") stamps [08:00,09:00) as 09:00,
    shifting every hour-keyed pipeline (training, calibration) by +1h."""

    @pytest.mark.asyncio
    async def test_training_queries_use_start_timesrc(self, forecast) -> None:
        queries: list = []

        solar_rec = MagicMock()
        solar_rec.get_time.return_value = datetime(2026, 5, 1, 12)
        solar_rec.get_value.return_value = 1000.0
        solar_table = MagicMock()
        solar_table.records = [solar_rec]

        async def _q(q: str):
            queries.append(q)
            # Return one solar record so the function proceeds past the
            # solar check and issues the weather queries too.
            return [solar_table] if "InputPower" in q else []

        client = AsyncMock()
        client.query = AsyncMock(side_effect=_q)
        settings = MagicMock()
        settings.influxdb.bucket_solar = "solar"
        settings.influxdb.bucket_weather = "weather_forecast"

        ok = await forecast.build_production_model(client, settings)
        assert ok is False  # no weather data — but queries were issued

        agg_queries = [q for q in queries if "aggregateWindow" in q]
        # solar + SOC + load + 6 weather fields
        assert len(agg_queries) >= 9
        for q in agg_queries:
            assert 'timeSrc: "_start"' in q

    @pytest.mark.asyncio
    async def test_daily_calibration_query_uses_start_timesrc(self, forecast) -> None:
        queries: list = []

        async def _q(q: str):
            queries.append(q)
            return []

        client = AsyncMock()
        client.query = AsyncMock(side_effect=_q)
        await forecast.calibrate_from_actuals(client, "solar", days=7)

        daily = [q for q in queries if "aggregateWindow(every: 1d" in q]
        assert daily, "expected the daily-actuals aggregateWindow query"
        for q in daily:
            assert 'timeSrc: "_start"' in q


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

    def _mock_query(self, hourly_actuals_w: dict, target_date,
                    soc=50.0, load_w=500.0, inverter_on=1, export_enabled=1):
        """Build an AsyncMock whose query() is FIELD-AWARE.

        The calibration now reads InputPower AND the curtailment signals (SOC,
        load, export_enabled, inverter_on). Returning the right data per field
        (default: battery not full, inverter on, export on → nothing curtailed)
        keeps these ratio tests exercising the real curtailment-aware path.
        """
        def _rec(hour, value):
            rec = MagicMock()
            rec.get_time.return_value = datetime(
                target_date.year, target_date.month, target_date.day, hour, 0
            )
            rec.get_value.return_value = float(value)
            return rec

        def _table(records):
            t = MagicMock()
            t.records = records
            return t

        async def _query(q: str):
            if 'InputPower' in q:
                return [_table([_rec(h, w) for h, w in hourly_actuals_w.items()])]
            if '"SOC"' in q:
                return [_table([_rec(h, soc) for h in hourly_actuals_w])]
            if 'INVPowerToLocalLoad' in q:
                return [_table([_rec(h, load_w) for h in hourly_actuals_w])]
            if 'export_enabled' in q:
                return [_table([_rec(0, export_enabled)])]
            if 'inverter_on' in q:
                return [_table([_rec(0, inverter_on)])]
            return []

        client = AsyncMock()
        client.query = AsyncMock(side_effect=_query)
        return client

    @pytest.mark.asyncio
    async def test_intraday_queries_use_start_timesrc(self, forecast) -> None:
        """The intraday actuals/SOC/load queries label windows by start."""
        today = self._make_forecast_with_today(
            forecast, {h: 3.0 for h in range(6, 18)},
        )
        queries: list = []
        inner = self._mock_query({6: 1500, 7: 1500, 8: 1500}, today)

        async def _q(q: str):
            queries.append(q)
            return await inner.query(q)

        client = AsyncMock()
        client.query = AsyncMock(side_effect=_q)
        await forecast.compute_intraday_calibration(
            client, "solar", current_hour=9, target_date=today,
        )
        agg_queries = [q for q in queries if "aggregateWindow" in q]
        assert agg_queries, "expected aggregateWindow queries"
        for q in agg_queries:
            assert 'timeSrc: "_start"' in q

    @pytest.mark.asyncio
    async def test_curtailed_hours_excluded_from_calibration(self, forecast) -> None:
        """Battery-full + low-load hours (MPPT throttled) must NOT drag the
        calibration ratio down — they reflect a dispatch decision, not the
        panels' potential. With all recent hours curtailed, calibration backs
        off (returns None) rather than suppressing the forecast."""
        today = self._make_forecast_with_today(
            forecast, {h: 3.0 for h in range(6, 18)},
        )
        # Actual far below forecast BUT battery full (100%) and load tiny →
        # this is curtailment, not real underproduction.
        client = self._mock_query(
            {6: 600, 7: 600, 8: 600}, today, soc=100.0, load_w=100.0,
        )
        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=9, target_date=today,
        )
        assert ratio is None, "curtailed actuals must not suppress the forecast"

        # Sanity: the SAME low actuals, but battery NOT full → genuine
        # underproduction → calibration DOES scale down (clamped to 0.5).
        client2 = self._mock_query(
            {6: 600, 7: 600, 8: 600}, today, soc=55.0, load_w=100.0,
        )
        ratio2 = await forecast.compute_intraday_calibration(
            client2, "solar", current_hour=9, target_date=today,
        )
        assert ratio2 == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_single_usable_hour_skips_calibration(self, forecast) -> None:
        """After curtailment filtering, a lone surviving (early-morning) hour
        must NOT drive a day-wide rescale — calibration backs off to None."""
        today = self._make_forecast_with_today(
            forecast, {h: 3.0 for h in range(6, 18)},
        )
        # Only one completed usable hour (07:00); battery not full so it isn't
        # curtailed, but a single noisy sample is too thin to calibrate on.
        client = self._mock_query({7: 2230}, today, soc=55.0, load_w=100.0)
        ratio = await forecast.compute_intraday_calibration(
            client, "solar", current_hour=8, target_date=today,
        )
        assert ratio is None

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

    def test_persistence_lifts_current_hour_when_live_above_forecast(self, forecast) -> None:
        """Live PV power >> forecast → current hour anchored to live, no
        downward effect."""
        # Forecast says cloudy: 0.5 kWh/h. Reality: 3.65 kW (sunny).
        model = {h: 0.5 for h in range(6, 20)}
        out = forecast.apply_live_persistence(
            model, live_power_kw=3.65, current_hour=8, blend_horizon_hours=2,
        )
        # Hour 8 (α=1): blended = 1.0*3.65 + 0.0*0.5 = 3.65; max with 0.5 → 3.65
        assert out[8] == pytest.approx(3.65)
        # Hour 9 (α=0.5): blended = 0.5*3.65 + 0.5*0.5 = 2.075; max → 2.075
        assert out[9] == pytest.approx(2.075)
        # Hour 10 (α=0): blended = 0*3.65 + 1*0.5 = 0.5; max(0.5,0.5)=0.5
        assert out[10] == pytest.approx(0.5)
        # Hour 11+ untouched
        assert out[11] == 0.5

    def test_persistence_current_hour_scaled_by_remaining_minutes(self, forecast) -> None:
        """Half-elapsed current hour anchors only its remaining half to live
        power, so an instantaneous kW peak isn't booked as a full hour of kWh."""
        model = {h: 0.5 for h in range(6, 20)}
        out = forecast.apply_live_persistence(
            model, live_power_kw=3.65, current_hour=8, blend_horizon_hours=2,
            minutes_elapsed=30,
        )
        # Hour 8 (α=1, remaining=0.5): 0.5*0.5 (elapsed, model) + 3.65*0.5 (live)
        # = 0.25 + 1.825 = 2.075 — well below the 3.65 the old full-hour code gave.
        assert out[8] == pytest.approx(2.075)
        # Future hours are full hours ahead → unaffected by minutes_elapsed.
        assert out[9] == pytest.approx(2.075)
        assert out[10] == pytest.approx(0.5)

    def test_persistence_never_lowers_forecast(self, forecast) -> None:
        """If model already predicts higher than live, keep the model
        prediction — don't let a cloud-pass briefly suppress the forecast."""
        # Forecast: 5 kWh/h. Live: only 1 kW (cloud-pass).
        model = {h: 5.0 for h in range(6, 20)}
        out = forecast.apply_live_persistence(
            model, live_power_kw=1.0, current_hour=8, blend_horizon_hours=2,
        )
        # Even at α=1 for current hour: max(model=5.0, blended=1.0) → keep 5.0
        assert out[8] == 5.0
        assert out[9] == 5.0
        assert out[10] == 5.0

    def test_persistence_skipped_when_live_below_threshold(self, forecast) -> None:
        """Nighttime or fault → live near 0 → no persistence applied at all."""
        model = {h: 5.0 for h in range(6, 20)}
        out = forecast.apply_live_persistence(
            model, live_power_kw=0.1, current_hour=8,
        )
        # Output identical to input (no anchoring)
        assert out == model

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


class TestZeroProductionCoverage:
    """A zero/near-zero (<NEAR_ZERO_WATTS) training sample is only legitimate
    (snow cover, standby) when RECORDED inverter state covers that hour.
    Hours predating state recording default to "on", but that convention must
    not vouch for (near-)zeros: historic manual-off/bypass hours — including
    ones with a 5-50 W telemetry blip — would inject false ~0 kWh samples
    into high-GHI bins and drag the medians down. Real positive samples keep
    the default-on behaviour."""

    ZERO_DT = datetime(2026, 7, 1, 12)
    POSITIVE_SAMPLES = 30 * 8  # 30 June days x hours 9-16

    def _client(
        self, inverter_records: list, probe_watts: float = 0.0,
        probe_ghi: float = 500.0,
    ):
        solar_records = []
        ghi_records = []

        def _rec(dt, value):
            r = MagicMock()
            r.get_time.return_value = dt
            r.get_value.return_value = value
            return r

        for day in range(1, 31):
            for hour in range(9, 17):
                dt = datetime(2026, 6, day, hour)
                solar_records.append(_rec(dt, 3000.0))
                ghi_records.append(_rec(dt, 500.0))
        # The probe: a zero/near-zero hour WITH radiation, daytime.
        solar_records.append(_rec(self.ZERO_DT, probe_watts))
        ghi_records.append(_rec(self.ZERO_DT, probe_ghi))

        def _table(recs):
            t = MagicMock()
            t.records = recs
            return t

        async def _q(q: str):
            if '"InputPower"' in q:
                return [_table(solar_records)]
            if "shortwave_radiation" in q:
                return [_table(ghi_records)]
            if "inverter_on" in q:
                return [_table(inverter_records)] if inverter_records else []
            return []  # SOC, load, export, other weather fields

        client = AsyncMock()
        client.query = AsyncMock(side_effect=_q)
        return client

    async def _build(
        self, solar_arrays, inverter_records, probe_watts: float = 0.0,
        probe_ghi: float = 500.0,
    ):
        forecast = SolarForecast(
            arrays=solar_arrays, latitude=49.0, longitude=14.5,
            confidence=0.7, logger=MagicMock(),
        )
        settings = MagicMock()
        settings.influxdb.bucket_solar = "solar"
        settings.influxdb.bucket_weather = "weather_forecast"
        ok = await forecast.build_production_model(
            self._client(
                inverter_records, probe_watts=probe_watts, probe_ghi=probe_ghi
            ),
            settings,
        )
        assert ok is True
        return forecast._production_model

    def _always_on_record(self):
        on_record = MagicMock()
        # Predates every training hour → carry-forward covers them all as ON.
        on_record.get_time.return_value = datetime(2026, 5, 1, 0, 0)
        on_record.get_value.return_value = 1
        return on_record

    @pytest.mark.asyncio
    async def test_zero_hour_without_state_coverage_excluded(self, solar_arrays) -> None:
        """No inverter_state records at all → the 0 W hour has no coverage
        and must NOT enter the training set; positives still train."""
        model = await self._build(solar_arrays, [])
        assert model.data_points == self.POSITIVE_SAMPLES

    @pytest.mark.asyncio
    async def test_zero_hour_with_state_coverage_included(self, solar_arrays) -> None:
        """The SAME 0 W hour, but recorded state says the inverter was ON
        the whole time → a real snow/standby observation, kept."""
        model = await self._build(solar_arrays, [self._always_on_record()])
        assert model.data_points == self.POSITIVE_SAMPLES + 1

    @pytest.mark.asyncio
    async def test_near_zero_hour_without_state_coverage_excluded(self, solar_arrays) -> None:
        """A 30 W hour (telemetry blip during an otherwise-off hour) is a
        near-zero, not real production — without recorded state coverage it
        must be excluded exactly like an exact 0 W sample."""
        model = await self._build(solar_arrays, [], probe_watts=30.0)
        assert model.data_points == self.POSITIVE_SAMPLES

    @pytest.mark.asyncio
    async def test_near_zero_hour_with_state_coverage_included(self, solar_arrays) -> None:
        """The SAME 30 W hour with recorded ON state is a legitimate
        observation (snow/standby with sensor noise) and is kept."""
        model = await self._build(
            solar_arrays, [self._always_on_record()], probe_watts=30.0
        )
        assert model.data_points == self.POSITIVE_SAMPLES + 1

    @pytest.mark.asyncio
    async def test_low_ghi_near_zero_trains_without_coverage(self, solar_arrays) -> None:
        """A 30 W hour at LOW radiation (GHI 80 <= NEAR_ZERO_HIGH_GHI) is
        normal dawn/dusk/deep-overcast output — it must stay trainable even
        WITHOUT inverter-state coverage. Requiring coverage there would
        wholesale-drop every dim hour predating state recording and bias the
        lowest radiation bins high."""
        model = await self._build(
            solar_arrays, [], probe_watts=30.0, probe_ghi=80.0
        )
        assert model.data_points == self.POSITIVE_SAMPLES + 1
