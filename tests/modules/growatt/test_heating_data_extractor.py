"""Tests for heating data extractor module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytz

from loxone_smart_home.config.settings import Settings, GrowattConfig, RoomConfig
from loxone_smart_home.modules.home.heating.data_extractor import HeatingDataExtractor
from loxone_smart_home.modules.home.heating.models import RoomState


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)

    # Mock GrowattConfig
    growatt_config = MagicMock(spec=GrowattConfig)

    # Create proper room mocks
    room1 = MagicMock()
    room1.name = "living_room"
    room1.T_set = 21.0

    room2 = MagicMock()
    room2.name = "bedroom"
    room2.T_set = 20.0

    growatt_config.rooms = [room1, room2]
    growatt_config.influx_temperature_bucket = "loxone"
    growatt_config.influx_temperature_measurement = "temperature"
    growatt_config.influx_room_field_prefix = "room_"
    growatt_config.influx_outdoor_field = "temperature_outside"
    growatt_config.irradiance_source = "forecast"

    settings.growatt = growatt_config

    # Mock InfluxDB config
    influxdb_config = MagicMock()
    influxdb_config.bucket_weather = "weather_forecast"
    settings.influxdb = influxdb_config

    return settings


@pytest.fixture
def mock_influxdb_client():
    """Create mock InfluxDB client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock()


@pytest.fixture
def data_extractor(mock_influxdb_client, mock_settings, mock_logger):
    """Create HeatingDataExtractor instance for testing."""
    return HeatingDataExtractor(mock_influxdb_client, mock_settings, mock_logger)


class MockInfluxDBRecord:
    """Mock InfluxDB record for testing."""

    def __init__(self, time, value, field=None):
        self._time = time
        self._value = value
        self._field = field

    def get_time(self):
        return self._time

    def get_value(self):
        return self._value

    def get_field(self):
        return self._field


class MockInfluxDBTable:
    """Mock InfluxDB table for testing."""

    def __init__(self, records):
        self.records = records


@pytest.mark.asyncio
class TestHeatingDataExtractor:
    """Test cases for HeatingDataExtractor."""

    async def test_extract_room_temperatures(self, data_extractor, mock_influxdb_client):
        """Test extraction of room temperatures."""
        # Mock query responses
        mock_influxdb_client.query = AsyncMock()

        # First room query result
        mock_influxdb_client.query.side_effect = [
            [MockInfluxDBTable([MockInfluxDBRecord(datetime.now(), 20.5)])],  # living_room
            [MockInfluxDBTable([MockInfluxDBRecord(datetime.now(), 19.8)])],  # bedroom
        ]

        room_states = await data_extractor.extract_room_temperatures()

        assert len(room_states) == 2
        assert room_states[0].name == "living_room"
        assert room_states[0].T_now == 20.5
        assert room_states[1].name == "bedroom"
        assert room_states[1].T_now == 19.8

    async def test_extract_room_temperatures_fallback(self, data_extractor, mock_influxdb_client):
        """Test room temperature extraction with fallback to setpoint."""
        # Mock empty query response
        mock_influxdb_client.query = AsyncMock(return_value=[])

        room_states = await data_extractor.extract_room_temperatures()

        # Should use setpoint as fallback
        assert len(room_states) == 2
        assert room_states[0].T_now == 21.0  # living_room setpoint
        assert room_states[1].T_now == 20.0  # bedroom setpoint

    async def test_extract_energy_prices(self, data_extractor, mock_influxdb_client):
        """Test extraction of energy prices."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=pytz.UTC)

        # Mock query response with price data
        records = [
            MockInfluxDBRecord(base_time + timedelta(hours=i), 2.0 + i * 0.1)
            for i in range(24)
        ]
        mock_influxdb_client.query = AsyncMock(return_value=[MockInfluxDBTable(records)])

        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 16, 0, 0)

        prices = await data_extractor.extract_energy_prices(start_date, end_date)

        assert len(prices) == 24
        # Check first and last prices
        first_hour = list(prices.keys())[0]
        assert prices[first_hour] == 2.0

    async def test_extract_weather_forecast(self, data_extractor, mock_influxdb_client):
        """Test extraction of weather forecast data."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=pytz.UTC)

        # Mock temperature records
        temp_records = [
            MockInfluxDBRecord(base_time + timedelta(hours=i), 5.0 + i * 0.5)
            for i in range(24)
        ]

        # Mock irradiance records
        irr_records = [
            MockInfluxDBRecord(base_time + timedelta(hours=i), 100.0 * i)
            for i in range(24)
        ]

        # Set up side effects for two queries
        mock_influxdb_client.query = AsyncMock()
        mock_influxdb_client.query.side_effect = [
            [MockInfluxDBTable(temp_records)],  # Temperature query
            [MockInfluxDBTable(irr_records)],   # Irradiance query
        ]

        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 16, 0, 0)

        temperatures, irradiances = await data_extractor.extract_weather_forecast(start_date, end_date)

        assert len(temperatures) == 24
        assert len(irradiances) == 24

        # Check first values
        first_hour = list(temperatures.keys())[0]
        assert temperatures[first_hour] == 5.0
        assert irradiances[first_hour] == 0.0

    async def test_extract_outdoor_temperature(self, data_extractor, mock_influxdb_client):
        """Test extraction of current outdoor temperature."""
        # Mock query response
        mock_influxdb_client.query = AsyncMock(
            return_value=[MockInfluxDBTable([MockInfluxDBRecord(datetime.now(), 8.5)])]
        )

        temperature = await data_extractor.extract_outdoor_temperature()

        assert temperature == 8.5

    async def test_extract_outdoor_temperature_no_data(self, data_extractor, mock_influxdb_client):
        """Test outdoor temperature extraction with no data."""
        # Mock empty query response
        mock_influxdb_client.query = AsyncMock(return_value=[])

        temperature = await data_extractor.extract_outdoor_temperature()

        assert temperature is None

    def test_aggregate_to_hourly(self, data_extractor):
        """Test aggregation of data to hourly resolution."""
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=data_extractor.local_tz)

        # Create 15-minute data
        data = {}
        for i in range(24):
            for j in range(4):  # 4 x 15 minutes = 1 hour
                timestamp = base_time + timedelta(hours=i, minutes=j*15)
                data[timestamp] = 10.0 + i + j * 0.1

        df = data_extractor.aggregate_to_hourly(data, method="mean")

        assert len(df) == 24
        # First hour should average 10.0, 10.1, 10.2, 10.3 = 10.15
        assert df.iloc[0]["value"] == pytest.approx(10.15, rel=1e-3)

    def test_validate_data_completeness(self, data_extractor):
        """Test data completeness validation."""
        # Create complete data
        base_time = datetime(2024, 1, 15, 0, 0, tzinfo=data_extractor.local_tz)
        data = {base_time + timedelta(hours=i): 10.0 + i for i in range(24)}
        df = pd.DataFrame(list(data.items()), columns=["datetime", "value"])
        df.set_index("datetime", inplace=True)

        # Should be valid
        assert data_extractor.validate_data_completeness(df, required_hours=24)

        # Create data with gaps
        data_with_gaps = {base_time + timedelta(hours=i): 10.0 + i for i in range(20)}
        df_gaps = pd.DataFrame(list(data_with_gaps.items()), columns=["datetime", "value"])
        df_gaps.set_index("datetime", inplace=True)

        # Should be invalid (missing 4 hours = 16.7% > 10% threshold)
        assert not data_extractor.validate_data_completeness(df_gaps, required_hours=24)

    def test_localize_datetime(self, data_extractor):
        """Test datetime localization."""
        # Test naive datetime
        naive_dt = datetime(2024, 1, 15, 10, 0)
        localized = data_extractor._localize_datetime(naive_dt)
        assert localized.tzinfo == data_extractor.local_tz

        # Test already localized datetime
        utc_dt = datetime(2024, 1, 15, 10, 0, tzinfo=pytz.UTC)
        converted = data_extractor._localize_datetime(utc_dt)
        assert converted.tzinfo.zone == "Europe/Prague"

    def test_to_utc(self, data_extractor):
        """Test datetime conversion to UTC."""
        # Test naive datetime
        naive_dt = datetime(2024, 1, 15, 10, 0)
        utc_dt = data_extractor._to_utc(naive_dt)
        assert utc_dt.tzinfo == pytz.UTC

        # Test Prague datetime
        prague_dt = data_extractor.local_tz.localize(datetime(2024, 1, 15, 10, 0))
        utc_dt = data_extractor._to_utc(prague_dt)
        assert utc_dt.tzinfo == pytz.UTC
        # Prague is UTC+1 in winter, so 10:00 Prague = 09:00 UTC
        assert utc_dt.hour == 9