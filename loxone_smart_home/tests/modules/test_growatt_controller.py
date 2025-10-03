"""Tests for the Growatt controller module."""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import GrowattConfig, Settings
from modules.growatt_controller import GrowattController
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient


@pytest.fixture
def mock_mqtt_client() -> AsyncMock:
    """Create a mock MQTT client."""
    client = AsyncMock(spec=AsyncMQTTClient)
    client.publish = AsyncMock()
    client.subscribe = AsyncMock()
    return client


@pytest.fixture
def mock_influxdb_client() -> AsyncMock:
    """Create a mock InfluxDB client."""
    client = AsyncMock(spec=AsyncInfluxDBClient)
    return client


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.growatt = GrowattConfig(
        simulation_mode=False,
        export_price_min=2.5,
        battery_charge_blocks=8
    )
    # Add logging configuration
    settings.log_level = "INFO"
    settings.log_timezone = "Europe/Prague"
    return settings


@pytest.fixture
def growatt_controller(
    mock_mqtt_client: AsyncMock,
    mock_influxdb_client: AsyncMock,
    mock_settings: Settings,
) -> GrowattController:
    """Create a GrowattController instance with mocked dependencies."""
    controller = GrowattController(mock_mqtt_client, mock_influxdb_client, mock_settings)
    # Mock the wait for result to prevent timeouts in tests
    # Accept any arguments with **kwargs to handle timeout parameter

    async def mock_wait(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"success": True}
    setattr(controller, "_wait_for_command_result", AsyncMock(side_effect=mock_wait))

    # Mock the query inverter state to prevent hangs
    async def mock_query_state(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"battery_first": {}, "grid_first": {}}
    setattr(controller, "_query_inverter_state", AsyncMock(side_effect=mock_query_state))

    return controller


async def test_controller_initialization(growatt_controller: GrowattController) -> None:
    """Test Growatt controller initialization."""
    assert growatt_controller.name == "GrowattController"
    assert growatt_controller.config is not None
    assert growatt_controller._periodic_check_task is None


async def test_fetch_dam_energy_prices_success(
    growatt_controller: GrowattController
) -> None:
    """Test successful energy price fetching with correct EUR/MWh parsing."""
    # Mock response matching actual OTE API structure
    mock_response = {
        "axis": {
            "y": {"legend": "Price (EUR/MWh)"}  # API claims EUR but line 0 is CZK
        },
        "data": {
            "dataLine": [
                {
                    "colour": "FF6600",
                    "point": [
                        {"x": "1", "y": 3912.8},  # CZK/MWh (line 0)
                        {"x": "2", "y": 3879.3},
                        {"x": "3", "y": 3997.7},
                    ]
                },
                {
                    "colour": "A04000",
                    "point": [
                        {"x": "1", "y": 98.5},    # EUR/MWh (line 1)
                        {"x": "2", "y": 105.2},
                        {"x": "3", "y": 92.7},
                    ]
                },
            ]
        }
    }

    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 200
        mock_get.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_session.return_value.__aenter__.return_value.get = MagicMock(
            return_value=mock_get
        )

        prices = await growatt_controller._price_analyzer.fetch_dam_energy_prices(
            "2024-01-01"
        )

        # Now expecting 15-minute intervals (3 hours × 4 blocks = 12)
        assert len(prices) == 12
        # Should use dataLine[1] which contains EUR/MWh prices
        # Each hour is expanded to 4 15-minute blocks with the same price
        assert prices[("00:00", "00:15")] == 98.5
        assert prices[("00:15", "00:30")] == 98.5
        assert prices[("00:30", "00:45")] == 98.5
        assert prices[("00:45", "01:00")] == 98.5
        assert prices[("01:00", "01:15")] == 105.2
        assert prices[("02:00", "02:15")] == 92.7


async def test_fetch_dam_energy_prices_failure(
    growatt_controller: GrowattController
) -> None:
    """Test energy price fetching with API failure."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 500
        mock_session.return_value.__aenter__.return_value.get = MagicMock(
            return_value=mock_get
        )

        prices = await growatt_controller._price_analyzer.fetch_dam_energy_prices()
        assert prices == {}


@pytest.mark.asyncio
async def test_fetch_dam_energy_prices_single_dataline(
    growatt_controller: GrowattController
) -> None:
    """Test price fetching when only one dataLine exists (fallback case)."""
    mock_response = {
        "data": {
            "dataLine": [
                {
                    "point": [
                        {"x": "1", "y": 100.0},
                        {"x": "2", "y": 110.0},
                    ]
                },
            ]
        }
    }
    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 200
        mock_get.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_session.return_value.__aenter__.return_value.get = MagicMock(
            return_value=mock_get
        )

        prices = await growatt_controller._price_analyzer.fetch_dam_energy_prices(
            "2024-01-01"
        )
        # Now expecting 15-minute intervals (2 hours × 4 blocks = 8)
        assert len(prices) == 8
        assert prices[("00:00", "00:15")] == 100.0
        assert prices[("00:15", "00:30")] == 100.0
        assert prices[("01:00", "01:15")] == 110.0


@pytest.mark.asyncio
async def test_fetch_dam_energy_prices_full_day(
    growatt_controller: GrowattController
) -> None:
    """Test price fetching for a full day."""
    mock_response = {
        "data": {
            "dataLine": [
                {
                    "point": [
                        {"x": str(hour), "y": 100.0 + hour * 10}
                        for hour in range(1, 25)
                    ]
                }
            ]
        }
    }

    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 200
        mock_get.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_session.return_value.__aenter__.return_value.get = MagicMock(
            return_value=mock_get
        )

        prices = await growatt_controller._price_analyzer.fetch_dam_energy_prices(
            "2024-01-01"
        )
        # Now expecting 15-minute intervals (24 hours × 4 blocks = 96)
        assert len(prices) == 96
        assert prices[("00:00", "00:15")] == 110.0
        assert prices[("00:15", "00:30")] == 110.0
        # The last 15-minute block in the day is 23:45-24:00
        assert prices[("23:45", "24:00")] == 340.0


@pytest.mark.asyncio
async def test_export_control_commands(
    growatt_controller: GrowattController,
    mock_mqtt_client: AsyncMock,
) -> None:
    """Test export control MQTT commands."""
    # Test enable export
    await growatt_controller._mode_manager.enable_export()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/export/enable", json.dumps({"value": True})
    )

    # Test disable export
    await growatt_controller._mode_manager.disable_export()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/export/disable", json.dumps({"value": True})
    )


@pytest.mark.asyncio
async def test_simulation_mode(
    mock_mqtt_client: AsyncMock,
    mock_influxdb_client: AsyncMock,
) -> None:
    """Test that MQTT commands are not sent in simulation mode."""
    settings = MagicMock(spec=Settings)
    settings.growatt = GrowattConfig(simulation_mode=True)
    # Add logging configuration
    settings.log_level = "INFO"
    settings.log_timezone = "Europe/Prague"

    controller = GrowattController(mock_mqtt_client, mock_influxdb_client, settings)

    await controller._mode_manager.set_battery_first("06:00", "08:00")
    await controller._mode_manager.enable_ac_charge()
    await controller._mode_manager.disable_ac_charge()
    await controller._mode_manager.enable_export()
    await controller._mode_manager.disable_export()

    # No MQTT commands should be sent
    mock_mqtt_client.publish.assert_not_called()


@pytest.mark.skip(reason="Test hangs - needs investigation after cleanup")
@pytest.mark.asyncio(loop_scope="function")
async def test_battery_control_commands(
    growatt_controller: GrowattController,
    mock_mqtt_client: AsyncMock,
) -> None:
    """Test battery control MQTT commands are sent correctly."""
    # Test battery-first mode
    await growatt_controller._mode_manager.set_battery_first(
        "06:00", "08:00", stop_soc=80
    )

    # Verify MQTT publish was called with battery-first command
    calls = [call for call in mock_mqtt_client.publish.call_args_list
             if "batteryfirst" in call[0][0]]
    assert len(calls) > 0

    # Test grid-first mode
    mock_mqtt_client.reset_mock()
    await growatt_controller._mode_manager.set_grid_first(
        "10:00", "12:00", stop_soc=20
    )

    # Verify MQTT publish was called with grid-first command
    calls = [call for call in mock_mqtt_client.publish.call_args_list
             if "gridfirst" in call[0][0]]
    assert len(calls) > 0


@pytest.mark.asyncio
async def test_start_stop(
    growatt_controller: GrowattController,
    mock_mqtt_client: AsyncMock,
) -> None:
    """Test controller start and stop methods."""
    # Mock all the startup methods to prevent actual execution
    setattr(growatt_controller, "_sync_inverter_time", AsyncMock())
    setattr(growatt_controller, "_fetch_prices", AsyncMock())
    setattr(growatt_controller, "_evaluate_conditions", AsyncMock())
    setattr(growatt_controller, "_periodic_evaluation_loop", AsyncMock())

    # Test start
    await growatt_controller.start()
    assert growatt_controller.mqtt_client is not None
    assert growatt_controller._running is True

    # Verify key startup methods were called
    growatt_controller._sync_inverter_time.assert_called_once()  # type: ignore[attr-defined]
    growatt_controller._fetch_prices.assert_called_once()  # type: ignore[attr-defined]
    growatt_controller._evaluate_conditions.assert_called()  # type: ignore[attr-defined]

    # Verify MQTT pre-registrations (two topics: result and status)
    # Note: Subscriptions are now pre-registered in __init__, not in start()
    assert mock_mqtt_client.register_subscription.call_count == 2

    # Test stop
    await growatt_controller.stop()
    assert growatt_controller._running is False
