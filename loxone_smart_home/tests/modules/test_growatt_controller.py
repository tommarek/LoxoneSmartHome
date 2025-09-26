"""Tests for the Growatt controller module."""

import asyncio
import json
import zoneinfo
from datetime import datetime, timedelta
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
        export_price_threshold=2.5,
        battery_charge_hours=2,
        individual_cheapest_hours=8,
        schedule_hour=23,
        schedule_minute=59,
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
    controller._wait_for_command_result = AsyncMock(return_value={"success": True})
    return controller


@pytest.mark.asyncio
async def test_controller_initialization(growatt_controller: GrowattController) -> None:
    """Test Growatt controller initialization."""
    assert growatt_controller.name == "GrowattController"
    assert growatt_controller.config is not None
    assert growatt_controller._scheduled_tasks == []
    assert growatt_controller._daily_schedule_task is None


@pytest.mark.asyncio
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

        assert len(prices) == 3
        # Should use dataLine[1] which contains EUR/MWh prices
        assert prices[("00:00", "01:00")] == 98.5
        assert prices[("01:00", "02:00")] == 105.2
        assert prices[("02:00", "03:00")] == 92.7


@pytest.mark.asyncio
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
        assert len(prices) == 2
        assert prices[("00:00", "01:00")] == 100.0
        assert prices[("01:00", "02:00")] == 110.0


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
        assert len(prices) == 24
        assert prices[("00:00", "01:00")] == 110.0
        # The last hour in the day is 23:00-24:00
        assert prices[("23:00", "24:00")] == 340.0


@pytest.mark.asyncio
async def test_find_cheapest_consecutive_hours(
    growatt_controller: GrowattController
) -> None:
    """Test finding cheapest consecutive hours."""
    prices = {
        ("00:00", "01:00"): 1500.0,
        ("01:00", "02:00"): 1200.0,
        ("02:00", "03:00"): 1100.0,
        ("03:00", "04:00"): 1300.0,
        ("04:00", "05:00"): 1400.0,
    }

    result = growatt_controller._price_analyzer.find_cheapest_consecutive_hours(
        prices, x=2
    )
    assert len(result) == 2
    assert result[0] == ("01:00", "02:00", 1200.0)
    assert result[1] == ("02:00", "03:00", 1100.0)


@pytest.mark.asyncio
async def test_find_n_cheapest_hours(growatt_controller: GrowattController) -> None:
    """Test finding N cheapest individual hours."""
    prices = {
        ("00:00", "01:00"): 1500.0,
        ("01:00", "02:00"): 1200.0,
        ("02:00", "03:00"): 1100.0,
        ("03:00", "04:00"): 1300.0,
        ("04:00", "05:00"): 1400.0,
    }

    result = growatt_controller._price_analyzer.find_n_cheapest_hours(prices, n=3)
    assert len(result) == 3
    assert result[0][2] == 1100.0  # Cheapest
    assert result[1][2] == 1200.0  # Second cheapest
    assert result[2][2] == 1300.0  # Third cheapest


@pytest.mark.asyncio
async def test_categorize_prices_into_quadrants(
    growatt_controller: GrowattController
) -> None:
    """Test price categorization into quadrants."""
    prices = {
        ("00:00", "01:00"): 1000.0,  # Cheapest
        ("01:00", "02:00"): 1500.0,  # Cheap
        ("02:00", "03:00"): 2000.0,  # Expensive
        ("03:00", "04:00"): 3000.0,  # Most Expensive
    }

    quadrants = growatt_controller._price_analyzer.categorize_prices_into_quadrants(
        prices
    )

    # The algorithm divides into 4 equal intervals
    # Range is 2000 (3000-1000), interval is 500
    # Cheapest: < 1500, Cheap: < 2000, Expensive: < 2500, Most Expensive: >= 2500
    assert len(quadrants["Cheapest"]) == 1
    assert len(quadrants["Cheap"]) == 1
    assert len(quadrants["Expensive"]) == 1
    assert len(quadrants["Most Expensive"]) == 1
    assert quadrants["Cheapest"][0][2] == 1000.0
    assert quadrants["Most Expensive"][0][2] == 3000.0


@pytest.mark.asyncio
async def test_group_contiguous_hours(growatt_controller: GrowattController) -> None:
    """Test grouping contiguous hours."""
    hours = [
        ("00:00", "01:00", 1000.0),
        ("01:00", "02:00", 1050.0),  # Within 20% of previous
        ("02:00", "03:00", 1100.0),  # Within 20% of previous
        ("04:00", "05:00", 2000.0),  # Not contiguous
        ("05:00", "06:00", 2100.0),  # Within 20% of previous
    ]

    groups = growatt_controller._price_analyzer.group_contiguous_hours(hours)

    assert len(groups) == 2
    assert groups[0] == ("00:00", "03:00")
    assert groups[1] == ("04:00", "06:00")


# Test removed: test_battery_control_commands
# This test was sending commands to the inverter which should not be done in tests


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


@pytest.mark.asyncio
async def test_calculate_and_schedule_next_day_no_prices(
    growatt_controller: GrowattController,
) -> None:
    """Test scheduling with no price data available."""
    with patch.object(
        growatt_controller._price_analyzer, "fetch_dam_energy_prices", return_value={}
    ):
        with patch.object(
            growatt_controller._price_analyzer, "generate_mock_prices", return_value={}
        ):
            with patch.object(
                growatt_controller, "_schedule_fallback_mode"
            ) as mock_fallback:
                await growatt_controller._calculate_and_schedule_next_day()
                mock_fallback.assert_called_once()


@pytest.mark.asyncio
async def test_calculate_and_schedule_next_day_with_prices(
    growatt_controller: GrowattController,
) -> None:
    """Test scheduling with price data available."""
    mock_prices = {
        ("00:00", "01:00"): 1.0,
        ("01:00", "02:00"): 2.0,
        ("02:00", "03:00"): 3.0,
        ("03:00", "04:00"): 4.0,
    }

    with patch.object(
        growatt_controller._price_analyzer,
        "fetch_dam_energy_prices",
        return_value=mock_prices
    ):
        with patch.object(
            growatt_controller, "_schedule_battery_control"
        ) as mock_battery:
            with patch.object(growatt_controller, "_schedule_export_control"):
                await growatt_controller._calculate_and_schedule_next_day()

                mock_battery.assert_called_once()
                # Check that the battery control was called with the correct arguments
                args = mock_battery.call_args[0]
                assert isinstance(args[0], set)  # all_cheap_hours
                # cheapest_consecutive can be a set or list depending on implementation
                assert isinstance(args[1], (set, list))  # cheapest_consecutive
                assert isinstance(args[2], dict)  # hourly_prices
                assert isinstance(args[3], float)  # eur_czk_rate


@pytest.mark.asyncio
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
    growatt_controller._sync_inverter_time = AsyncMock()
    growatt_controller._reset_inverter_state = AsyncMock()
    growatt_controller._schedule_daily_calculation = AsyncMock()
    growatt_controller._apply_current_state = AsyncMock()

    # Test start
    await growatt_controller.start()
    assert growatt_controller.mqtt_client is not None
    assert growatt_controller._running is True

    # Verify key startup methods were called
    growatt_controller._sync_inverter_time.assert_called_once()
    growatt_controller._reset_inverter_state.assert_called_once()
    growatt_controller._schedule_daily_calculation.assert_called_once()
    growatt_controller._apply_current_state.assert_called_once()

    # Verify MQTT subscription
    mock_mqtt_client.subscribe.assert_called_once()

    # Test stop
    await growatt_controller.stop()
    assert growatt_controller._running is False
    assert len(growatt_controller._scheduled_tasks) == 0