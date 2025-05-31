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
    return GrowattController(mock_mqtt_client, mock_influxdb_client, mock_settings)


@pytest.mark.asyncio
async def test_controller_initialization(growatt_controller: GrowattController) -> None:
    """Test Growatt controller initialization."""
    assert growatt_controller.name == "GrowattController"
    assert growatt_controller.config is not None
    assert growatt_controller._scheduled_tasks == []
    assert growatt_controller._daily_schedule_task is None


@pytest.mark.asyncio
async def test_fetch_dam_energy_prices_success(growatt_controller: GrowattController) -> None:
    """Test successful energy price fetching."""
    mock_response = {
        "data": {
            "dataLine": [
                {},
                {
                    "point": [
                        {"x": "1", "y": "1500.50"},
                        {"x": "2", "y": "1600.25"},
                        {"x": "3", "y": "1400.00"},
                    ]
                },
            ]
        }
    }

    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 200
        mock_get.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=mock_get)

        prices = await growatt_controller._fetch_dam_energy_prices("2024-01-01")

        assert len(prices) == 3
        assert prices[("00:00", "01:00")] == 1500.50
        assert prices[("01:00", "02:00")] == 1600.25
        assert prices[("02:00", "03:00")] == 1400.00


@pytest.mark.asyncio
async def test_fetch_dam_energy_prices_failure(growatt_controller: GrowattController) -> None:
    """Test energy price fetching with API failure."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value.status = 500
        mock_session.return_value.__aenter__.return_value.get = MagicMock(return_value=mock_get)

        prices = await growatt_controller._fetch_dam_energy_prices()
        assert prices == {}


@pytest.mark.asyncio
async def test_find_cheapest_consecutive_hours(growatt_controller: GrowattController) -> None:
    """Test finding cheapest consecutive hours."""
    prices = {
        ("00:00", "01:00"): 1500.0,
        ("01:00", "02:00"): 1200.0,
        ("02:00", "03:00"): 1100.0,
        ("03:00", "04:00"): 1300.0,
        ("04:00", "05:00"): 1400.0,
    }

    result = growatt_controller._find_cheapest_consecutive_hours(prices, x=2)
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

    result = growatt_controller._find_n_cheapest_hours(prices, n=3)
    assert len(result) == 3
    assert result[0][2] == 1100.0  # Cheapest
    assert result[1][2] == 1200.0  # Second cheapest
    assert result[2][2] == 1300.0  # Third cheapest


@pytest.mark.asyncio
async def test_categorize_prices_into_quadrants(growatt_controller: GrowattController) -> None:
    """Test price categorization into quadrants."""
    prices = {
        ("00:00", "01:00"): 1000.0,  # Cheapest
        ("01:00", "02:00"): 1500.0,  # Cheap
        ("02:00", "03:00"): 2000.0,  # Expensive
        ("03:00", "04:00"): 3000.0,  # Most Expensive
    }

    quadrants = growatt_controller._categorize_prices_into_quadrants(prices)

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

    groups = growatt_controller._group_contiguous_hours(hours)

    assert len(groups) == 2
    assert groups[0] == ("00:00", "03:00")
    assert groups[1] == ("04:00", "06:00")


@pytest.mark.asyncio
async def test_battery_control_commands(
    growatt_controller: GrowattController,
    mock_mqtt_client: AsyncMock,
) -> None:
    """Test battery control MQTT commands."""
    # Test set battery first
    await growatt_controller._set_battery_first("06:00", "08:00")
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/batteryfirst/set/timeslot",
        json.dumps({"start": "06:00", "stop": "08:00", "enabled": True, "slot": 1}),
    )

    # Test enable AC charge
    await growatt_controller._enable_ac_charge()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/batteryfirst/set/acchargeenabled", json.dumps({"value": True})
    )

    # Test disable AC charge
    await growatt_controller._disable_ac_charge()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/batteryfirst/set/acchargeenabled", json.dumps({"value": False})
    )

    # Test disable battery first
    await growatt_controller._disable_battery_first()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/batteryfirst/set/timeslot",
        json.dumps({"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}),
    )


@pytest.mark.asyncio
async def test_export_control_commands(
    growatt_controller: GrowattController,
    mock_mqtt_client: AsyncMock,
) -> None:
    """Test export control MQTT commands."""
    # Test enable export
    await growatt_controller._enable_export()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/export/enable", json.dumps({"value": True})
    )

    # Test disable export
    await growatt_controller._disable_export()
    mock_mqtt_client.publish.assert_called_with(
        "energy/solar/command/export/disable", json.dumps({"value": False})
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

    await controller._set_battery_first("06:00", "08:00")
    await controller._enable_ac_charge()
    await controller._disable_ac_charge()
    await controller._enable_export()
    await controller._disable_export()

    # No MQTT commands should be sent
    mock_mqtt_client.publish.assert_not_called()


@pytest.mark.asyncio
async def test_calculate_and_schedule_next_day_no_prices(
    growatt_controller: GrowattController,
) -> None:
    """Test scheduling with no price data available."""
    with patch.object(growatt_controller, "_fetch_dam_energy_prices", return_value={}):
        with patch.object(growatt_controller, "_schedule_fallback_mode") as mock_fallback:
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

    with patch.object(growatt_controller, "_fetch_dam_energy_prices", return_value=mock_prices):
        with patch.object(growatt_controller, "_schedule_battery_control") as mock_battery:
            with patch.object(growatt_controller, "_schedule_export_control"):
                await growatt_controller._calculate_and_schedule_next_day()

                mock_battery.assert_called_once()
                # Check that the battery control was called with the correct arguments
                args = mock_battery.call_args[0]
                assert isinstance(args[0], set)  # all_cheap_hours
                assert isinstance(args[1], set)  # cheapest_consecutive
                assert args[2] == mock_prices  # hourly_prices


@pytest.mark.asyncio
async def test_schedule_export_control(
    growatt_controller: GrowattController,
) -> None:
    """Test export control scheduling."""
    hourly_prices = {
        ("00:00", "01:00"): 80.0,
        ("01:00", "02:00"): 90.0,
        ("02:00", "03:00"): 120.0,  # Above threshold
        ("03:00", "04:00"): 130.0,  # Above threshold
        ("04:00", "05:00"): 85.0,
    }

    with patch.object(growatt_controller, "_schedule_at_time") as mock_schedule:
        await growatt_controller._schedule_export_control(hourly_prices)

        # Should schedule enable at 02:00 and disable at 04:00
        calls = mock_schedule.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "02:00"  # Enable time
        assert calls[1][0][0] == "04:00"  # Disable time


@pytest.mark.asyncio
async def test_start_stop(growatt_controller: GrowattController) -> None:
    """Test controller start and stop."""
    with patch.object(growatt_controller, "_schedule_daily_calculation") as mock_schedule:
        await growatt_controller.start()
        mock_schedule.assert_called_once()

    # Create a dummy task to test cancellation
    async def dummy_coro() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    task1 = asyncio.create_task(dummy_coro())
    task2 = asyncio.create_task(dummy_coro())

    growatt_controller._scheduled_tasks = [task1]
    growatt_controller._daily_schedule_task = task2

    with patch.object(growatt_controller, "_disable_battery_first") as mock_disable:
        await growatt_controller.stop()
        mock_disable.assert_called_once()

    # Give tasks time to be cancelled
    await asyncio.sleep(0.1)

    # Tasks should be cancelled after stop
    assert task1.cancelled() or task1.done()
    assert task2.cancelled() or task2.done()


@pytest.mark.asyncio
async def test_daily_calculation_loop(growatt_controller: GrowattController) -> None:
    """Test the daily calculation loop."""
    growatt_controller._running = True

    # Simulate the loop - first sleep returns immediately, second raises CancelledError
    sleep_count = 0

    async def mock_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            return  # First sleep (delay calculation) passes
        else:
            raise asyncio.CancelledError()  # Second sleep (after calculation) cancels

    with patch.object(growatt_controller, "_calculate_and_schedule_next_day") as mock_calc:
        with patch("asyncio.sleep", side_effect=mock_sleep):
            try:
                await growatt_controller._daily_calculation_loop()
            except asyncio.CancelledError:
                pass  # Expected

        # Should have calculated once after first sleep
        mock_calc.assert_called_once()


@pytest.mark.asyncio
async def test_schedule_at_time(growatt_controller: GrowattController) -> None:
    """Test scheduling a task at a specific time."""

    async def dummy_task() -> str:
        return "completed"

    # Schedule for immediate execution using Prague timezone
    local_tz = zoneinfo.ZoneInfo("Europe/Prague")
    past_time = (datetime.now(local_tz) - timedelta(hours=1)).strftime("%H:%M")

    with patch("asyncio.sleep") as mock_sleep:
        await growatt_controller._schedule_at_time(past_time, dummy_task)
        # Should have waited for next day occurrence
        assert mock_sleep.call_args[0][0] > 0  # Positive delay
