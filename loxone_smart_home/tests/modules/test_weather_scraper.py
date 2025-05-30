"""Tests for weather scraper module."""

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from config.settings import Settings
from modules.weather_scraper import HourlyData, WeatherScraper


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.weather.weather_service = "openmeteo"
    settings.weather.latitude = 49.00642
    settings.weather.longitude = 14.51994
    settings.weather.openmeteo_url = "https://api.open-meteo.com/v1/forecast"
    settings.weather.aladin_url_base = "http://www.nts2.cz:443/meteo/aladin/"
    settings.weather.openweathermap_url = "https://api.openweathermap.org/data/2.5/onecall"
    settings.weather.openweathermap_api_key = "test-api-key"
    settings.weather.update_interval = 1800
    settings.weather.retry_delay = 60
    settings.mqtt.topic_weather = "weather"
    settings.influxdb.bucket_weather = "weather_forecast"
    return settings


@pytest.fixture
def mock_mqtt_client() -> MagicMock:
    """Create mock MQTT client."""
    client = MagicMock()
    client.publish = AsyncMock()
    return client


@pytest.fixture
def mock_influxdb_client() -> MagicMock:
    """Create mock InfluxDB client."""
    client = MagicMock()
    client.write_point = AsyncMock()
    return client


@pytest.fixture
def weather_scraper(
    mock_mqtt_client: MagicMock,
    mock_influxdb_client: MagicMock,
    mock_settings: Settings,
) -> WeatherScraper:
    """Create weather scraper instance."""
    return WeatherScraper(mock_mqtt_client, mock_influxdb_client, mock_settings)


@pytest.fixture
def mock_openmeteo_response() -> Dict[str, Any]:
    """Mock OpenMeteo API response."""
    current_time = int(
        datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).timestamp()
    )
    return {
        "hourly": {
            "time": [current_time, current_time + 3600, current_time + 7200],
            "temperature_2m": [15.5, 16.2, 17.1],
            "relativehumidity_2m": [65, 62, 60],
            "precipitation": [0.0, 0.1, 0.0],
            "windspeed_10m": [5.2, 6.1, 5.8],
            "winddirection_10m": [180, 185, 190],
            "cloudcover": [25, 30, 35],
            "surface_pressure": [1013.25, 1013.0, 1012.75],
        },
        "daily": {
            "time": [current_time],
            "sunrise": [current_time + 21600],  # 6 AM
            "sunset": [current_time + 64800],  # 6 PM
            "precipitation_sum": [0.1],
            "rain_sum": [0.1],
        },
    }


@pytest.fixture
def mock_air_quality_response() -> Dict[str, Any]:
    """Mock air quality API response."""
    return {
        "hourly": {
            "pm10": [15.0, 16.0, 14.5],
            "pm2_5": [8.0, 8.5, 7.8],
            "uv_index": [2.5, 3.0, 3.5],
        }
    }


@pytest.fixture
def mock_aladin_response() -> Dict[str, Any]:
    """Mock Aladin API response."""
    return {
        "nowCasting": {"nowUtc": int(datetime.now(timezone.utc).timestamp())},
        "parameterValues": {
            "TEMPERATURE": [15.0, 15.5, 16.0],
            "PRECIPITATION_TOTAL": [0.0, 0.1, 0.0],
            "WIND_SPEED": [5.0, 5.5, 6.0],
            "WIND_DIRECTION": [180, 185, 190],
            "CLOUDS_TOTAL": [25, 30, 35],
            "HUMIDITY": [65, 63, 60],
            "PRESSURE": [1013, 1013, 1012],
        },
    }


@pytest.fixture
def mock_openweathermap_response() -> Dict[str, Any]:
    """Mock OpenWeatherMap API response."""
    current_time = int(datetime.now(timezone.utc).timestamp())
    return {
        "current": {
            "dt": current_time,
            "temp": 15.5,
            "humidity": 65,
            "pressure": 1013,
            "wind_speed": 5.2,
            "wind_deg": 180,
            "clouds": 25,
        },
        "hourly": [
            {
                "dt": current_time,
                "temp": 15.5,
                "humidity": 65,
                "pressure": 1013,
                "wind_speed": 5.2,
                "wind_deg": 180,
                "clouds": 25,
            },
            {
                "dt": current_time + 3600,
                "temp": 16.0,
                "humidity": 63,
                "pressure": 1013,
                "wind_speed": 5.5,
                "wind_deg": 185,
                "clouds": 30,
                "rain": {"1h": 0.1},
            },
        ],
    }


def create_mock_session(mock_response: AsyncMock) -> MagicMock:
    """Create a properly mocked aiohttp session."""
    mock_get = MagicMock()
    mock_get.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get.__aexit__ = AsyncMock()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_get
    return mock_session


class TestWeatherScraper:
    """Test weather scraper functionality."""

    @pytest.mark.asyncio
    async def test_init(
        self,
        mock_mqtt_client: MagicMock,
        mock_influxdb_client: MagicMock,
        mock_settings: Settings,
    ) -> None:
        """Test weather scraper initialization."""
        scraper = WeatherScraper(mock_mqtt_client, mock_influxdb_client, mock_settings)

        assert scraper.name == "WeatherScraper"
        assert scraper.mqtt_client == mock_mqtt_client
        assert scraper.influxdb_client == mock_influxdb_client
        assert scraper.settings == mock_settings
        assert scraper._task is None
        assert scraper._session is None

    @pytest.mark.asyncio
    async def test_start_stop(self, weather_scraper: WeatherScraper) -> None:
        """Test starting and stopping the weather scraper."""
        with patch.object(weather_scraper, "_run_periodic"):
            # Start the scraper
            await weather_scraper.start()

            assert weather_scraper._session is not None
            assert isinstance(weather_scraper._session, ClientSession)
            assert weather_scraper._task is not None

            # Stop the scraper
            await weather_scraper.stop()

            assert weather_scraper._task.cancelled()

    @pytest.mark.asyncio
    async def test_fetch_openmeteo(
        self,
        weather_scraper: WeatherScraper,
        mock_openmeteo_response: Dict[str, Any],
        mock_air_quality_response: Dict[str, Any],
        mock_mqtt_client: MagicMock,
        mock_influxdb_client: MagicMock,
    ) -> None:
        """Test fetching weather from OpenMeteo."""
        weather_scraper.settings.weather.weather_service = "openmeteo"

        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.json = AsyncMock()
        mock_response.json.side_effect = [mock_openmeteo_response, mock_air_quality_response]

        weather_scraper._session = create_mock_session(mock_response)

        # Fetch weather
        await weather_scraper.fetch_weather()

        # Verify HTTP calls
        assert weather_scraper._session.get.call_count == 2

        # Verify MQTT publish
        mock_mqtt_client.publish.assert_called_once()
        topic, payload = mock_mqtt_client.publish.call_args[0]
        assert topic == "weather"
        data = json.loads(payload)
        assert data["source"] == "openmeteo"
        assert "hourly" in data
        assert "daily" in data

        # Verify InfluxDB writes
        assert mock_influxdb_client.write_point.call_count >= 1

    @pytest.mark.asyncio
    async def test_fetch_aladin(
        self,
        weather_scraper: WeatherScraper,
        mock_aladin_response: Dict[str, Any],
        mock_mqtt_client: MagicMock,
        mock_influxdb_client: MagicMock,
    ) -> None:
        """Test fetching weather from Aladin."""
        weather_scraper.settings.weather.weather_service = "aladin"

        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_aladin_response)

        weather_scraper._session = create_mock_session(mock_response)

        # Fetch weather
        await weather_scraper.fetch_weather()

        # Verify HTTP call
        weather_scraper._session.get.assert_called_once()

        # Verify MQTT publish
        mock_mqtt_client.publish.assert_called_once()
        topic, payload = mock_mqtt_client.publish.call_args[0]
        assert topic == "weather"
        data = json.loads(payload)
        assert data["source"] == "aladin"
        assert "hourly" in data
        assert len(data["hourly"]) == 3

        # Verify InfluxDB write
        mock_influxdb_client.write_point.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_openweathermap(
        self,
        weather_scraper: WeatherScraper,
        mock_openweathermap_response: Dict[str, Any],
        mock_mqtt_client: MagicMock,
        mock_influxdb_client: MagicMock,
    ) -> None:
        """Test fetching weather from OpenWeatherMap."""
        weather_scraper.settings.weather.weather_service = "openweathermap"

        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=mock_openweathermap_response)

        weather_scraper._session = create_mock_session(mock_response)

        # Fetch weather
        await weather_scraper.fetch_weather()

        # Verify HTTP call
        weather_scraper._session.get.assert_called_once()

        # Verify MQTT publish
        mock_mqtt_client.publish.assert_called_once()
        topic, payload = mock_mqtt_client.publish.call_args[0]
        assert topic == "weather"
        data = json.loads(payload)
        assert data["source"] == "openweathermap"
        assert "hourly" in data

        # Check that current weather replaced first hour
        assert data["hourly"][0]["temp"] == 15.5  # From current
        assert data["hourly"][1]["precip"] == 0.1  # Rain from hourly

        # Verify InfluxDB write
        mock_influxdb_client.write_point.assert_called()

    @pytest.mark.asyncio
    async def test_hourly_data_format(self) -> None:
        """Test HourlyData named tuple format."""
        data = HourlyData(
            temp=20.5, precip=0.0, wind=5.2, wind_direction=180, clouds=25, rh=65, pressure=1013.25
        )

        data_dict = data._asdict()
        assert data_dict["temp"] == 20.5
        assert data_dict["precip"] == 0.0
        assert data_dict["wind"] == 5.2
        assert data_dict["wind_direction"] == 180
        assert data_dict["clouds"] == 25
        assert data_dict["rh"] == 65
        assert data_dict["pressure"] == 1013.25

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        weather_scraper: WeatherScraper,
        mock_mqtt_client: MagicMock,
    ) -> None:
        """Test error handling in weather fetching."""
        # Test with no session
        weather_scraper._session = None
        await weather_scraper.fetch_weather()

        # Should not crash and not publish
        mock_mqtt_client.publish.assert_not_called()

        # Test with HTTP error
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))
        weather_scraper._session = mock_session

        await weather_scraper.fetch_weather()

        # Should handle error gracefully
        mock_mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_periodic_run(self, weather_scraper: WeatherScraper) -> None:
        """Test periodic weather fetching."""
        weather_scraper.settings.weather.update_interval = 0.1  # type: ignore[assignment]
        weather_scraper._running = True  # Set running flag

        fetch_count = 0

        async def mock_fetch() -> None:
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count >= 2:
                weather_scraper._running = False

        with patch.object(weather_scraper, "fetch_weather", mock_fetch):
            await weather_scraper._run_periodic()

        assert fetch_count >= 2

    @pytest.mark.asyncio
    async def test_influxdb_write(
        self,
        weather_scraper: WeatherScraper,
        mock_influxdb_client: MagicMock,
    ) -> None:
        """Test writing weather data to InfluxDB."""
        test_data = {
            "temperature_2m": 20.5,
            "humidity": 65,
            "pressure": 1013.25,
            "time": 1234567890,  # Should be skipped
            "status": "ok",  # Non-numeric, should be skipped
        }

        await weather_scraper._write_influx_point(
            test_data, int(datetime.now(timezone.utc).timestamp()), "test", "hour"
        )

        mock_influxdb_client.write_point.assert_called_once()
        call_args = mock_influxdb_client.write_point.call_args

        # Check fields were converted to float
        fields = call_args.kwargs["fields"]
        assert fields["temperature_2m"] == 20.5
        assert fields["humidity"] == 65.0
        assert fields["pressure"] == 1013.25
        assert "time" not in fields
        assert "status" not in fields

        # Check tags
        tags = call_args.kwargs["tags"]
        assert tags["room"] == "outside"
        assert tags["type"] == "hour"
        assert tags["source"] == "test"

    @pytest.mark.asyncio
    async def test_no_api_key_openweathermap(
        self,
        weather_scraper: WeatherScraper,
        mock_mqtt_client: MagicMock,
    ) -> None:
        """Test OpenWeatherMap without API key."""
        weather_scraper.settings.weather.weather_service = "openweathermap"
        weather_scraper.settings.weather.openweathermap_api_key = None
        weather_scraper._session = AsyncMock()

        await weather_scraper.fetch_weather()

        # Should not publish without API key
        mock_mqtt_client.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_service(
        self,
        weather_scraper: WeatherScraper,
        mock_mqtt_client: MagicMock,
    ) -> None:
        """Test with unknown weather service."""
        weather_scraper.settings.weather.weather_service = "unknown"
        weather_scraper._session = AsyncMock()

        await weather_scraper.fetch_weather()

        # Should not publish for unknown service
        mock_mqtt_client.publish.assert_not_called()
