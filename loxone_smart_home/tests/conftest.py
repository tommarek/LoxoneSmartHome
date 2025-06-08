"""Pytest configuration and shared fixtures."""

import asyncio
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from config.settings import Settings


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    with patch.dict(
        "os.environ",
        {
            "INFLUXDB_TOKEN": "test-token",
            "INFLUXDB_ORG": "test-org",
            "MQTT_BROKER": "localhost",
            "LOXONE_HOST": "192.168.1.100",
        },
    ):
        return Settings(influxdb_token="test-token")


@pytest_asyncio.fixture
async def mock_mqtt_client(mock_settings: Settings) -> AsyncGenerator[MagicMock, None]:
    """Create a mock MQTT client."""
    client = MagicMock()
    client.settings = mock_settings
    client.client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.publish = AsyncMock()
    client.subscribe = AsyncMock()

    yield client


@pytest.fixture
def mock_influxdb_client(mock_settings: Settings) -> MagicMock:
    """Create a mock InfluxDB client."""
    client = MagicMock()
    client.settings = mock_settings
    client.client = MagicMock()
    client.write_api = MagicMock()
    client.write_point = AsyncMock()
    client.write_points = AsyncMock()
    client.query = AsyncMock()

    return client


@pytest.fixture
def sample_udp_data() -> bytes:
    """Sample UDP data for testing."""
    return (
        b"2024-01-15 10:30:00;temperature;21.5;living_room;sensor;tag1_value;tag2_value"
    )


@pytest.fixture
def sample_mqtt_message() -> Dict[str, Any]:
    """Sample MQTT message for testing."""
    return {
        "power": 2500.0,
        "battery_soc": 75,
        "grid_export": 1200.0,
        "pv_production": 3700.0,
    }


@pytest.fixture
def sample_weather_data() -> Dict[str, Any]:
    """Sample weather data for testing."""
    return {
        "temperature": 15.5,
        "humidity": 65,
        "pressure": 1013.25,
        "wind_speed": 5.2,
        "wind_direction": 180,
        "precipitation": 0.0,
        "cloud_cover": 25,
    }


@pytest.fixture
def sample_energy_prices() -> List[Dict[str, Any]]:
    """Sample energy prices for testing."""
    return [
        {"hour": 0, "price": 2.5},
        {"hour": 1, "price": 2.3},
        {"hour": 2, "price": 2.1},
        {"hour": 3, "price": 1.9},
        {"hour": 4, "price": 1.8},
        {"hour": 5, "price": 2.0},
        {"hour": 6, "price": 2.4},
        {"hour": 7, "price": 3.2},
        {"hour": 8, "price": 3.5},
        {"hour": 9, "price": 3.1},
        {"hour": 10, "price": 2.8},
        {"hour": 11, "price": 2.6},
        {"hour": 12, "price": 2.4},
        {"hour": 13, "price": 2.5},
        {"hour": 14, "price": 2.7},
        {"hour": 15, "price": 2.9},
        {"hour": 16, "price": 3.3},
        {"hour": 17, "price": 3.8},
        {"hour": 18, "price": 4.2},
        {"hour": 19, "price": 3.9},
        {"hour": 20, "price": 3.4},
        {"hour": 21, "price": 3.0},
        {"hour": 22, "price": 2.7},
        {"hour": 23, "price": 2.5},
    ]
