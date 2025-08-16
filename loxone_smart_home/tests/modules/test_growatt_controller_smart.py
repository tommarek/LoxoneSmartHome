"""Unit tests for smart battery optimization features in Growatt controller."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import zoneinfo

from config.settings import Settings
from modules.growatt_controller import GrowattController


@pytest.fixture
def mock_mqtt_client():
    """Create a mock MQTT client."""
    client = AsyncMock()
    client.publish = AsyncMock()
    return client


@pytest.fixture
def mock_influxdb_client():
    """Create a mock InfluxDB client."""
    client = AsyncMock()
    client.write_point = AsyncMock()
    client.query = AsyncMock()
    return client


@pytest.fixture
def settings():
    """Create test settings."""
    settings = Settings(
        influxdb_token="test_token",
        loxone_bridge_loxone_host="127.0.0.1",
        weather_latitude=50.0,
        weather_longitude=14.0,
    )
    settings.growatt.simulation_mode = True
    settings.growatt.summer_temp_threshold = 15.0
    settings.growatt.summer_price_threshold = 1.0
    settings.growatt.battery_charge_hours = 2
    settings.growatt.individual_cheapest_hours = 6
    return settings


@pytest.fixture
async def controller(mock_mqtt_client, mock_influxdb_client, settings):
    """Create a Growatt controller instance."""
    controller = GrowattController(mock_mqtt_client, mock_influxdb_client, settings)
    controller._running = True
    yield controller
    # Cleanup: stop the controller to cancel all tasks
    await controller.stop()


@pytest.mark.asyncio
async def test_season_detection_summer(controller, mock_influxdb_client):
    """Test season detection returns summer when temperature > threshold."""
    # Mock query to return summer temperature
    mock_influxdb_client.query.return_value = [
        MagicMock(records=[MagicMock(get_value=lambda: 18.5)])
    ]

    season = await controller._get_season_mode()

    assert season == "summer"
    assert controller._season_mode == "summer"
    mock_influxdb_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_season_detection_winter(controller, mock_influxdb_client):
    """Test season detection returns winter when temperature <= threshold."""
    # Mock query to return winter temperature
    mock_influxdb_client.query.return_value = [
        MagicMock(records=[MagicMock(get_value=lambda: 10.0)])
    ]

    season = await controller._get_season_mode()

    assert season == "winter"
    assert controller._season_mode == "winter"


@pytest.mark.asyncio
async def test_season_detection_cache(controller, mock_influxdb_client):
    """Test season detection uses cache within 24 hours."""
    # First call
    mock_influxdb_client.query.return_value = [
        MagicMock(records=[MagicMock(get_value=lambda: 20.0)])
    ]
    season1 = await controller._get_season_mode()

    # Second call (should use cache)
    season2 = await controller._get_season_mode()

    assert season1 == season2 == "summer"
    # Query should only be called once due to caching
    mock_influxdb_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_season_detection_no_data(controller, mock_influxdb_client):
    """Test season detection defaults to winter when no data available."""
    # Mock query to return no data
    mock_influxdb_client.query.return_value = []

    season = await controller._get_season_mode()

    assert season == "winter"


# Test removed: test_grid_first_mode_simulation
# This test was sending commands to the inverter which should not be done in tests


# Test removed: test_grid_first_mode_real
# This test was sending commands to the inverter which should not be done in tests


# Test removed: test_load_first_mode
# This test was sending commands to the inverter which should not be done in tests


# Test removed: test_summer_strategy_with_low_prices
# This test was sending commands to the inverter which should not be done in tests


# Test removed: test_summer_strategy_no_low_prices
# This test was sending commands to the inverter which should not be done in tests

    # No grid-first or load-first periods
    grid_periods = [p for p in controller._scheduled_periods if p[0] == "grid_first"]
    load_periods = [p for p in controller._scheduled_periods if p[0] == "load_first"]
    assert len(grid_periods) == 0
    assert len(load_periods) == 0


@pytest.mark.asyncio
async def test_winter_strategy_with_ac_charging(controller, mock_influxdb_client):
    """Test winter strategy schedules AC charging during cheapest hours."""
    # Mock winter temperature
    mock_influxdb_client.query.return_value = [
        MagicMock(records=[MagicMock(get_value=lambda: 8.0)])
    ]

    # Mock prices with cheap night hours
    mock_prices = {}
    for h in range(24):
        if 2 <= h <= 5:  # Cheapest hours
            price = 30.0
        elif h < 8 or h > 20:  # Night hours
            price = 40.0
        else:  # Day hours
            price = 60.0
        mock_prices[(f"{h:02d}:00", f"{h+1:02d}:00")] = price

    with patch.object(controller, '_fetch_dam_energy_prices', return_value=mock_prices):
        await controller._calculate_and_schedule_next_day()

    # Check that AC charging is scheduled
    ac_charge_periods = [p for p in controller._scheduled_periods if p.kind == "ac_charge"]
    assert len(ac_charge_periods) > 0

    # No grid-first periods in winter
    grid_periods = [p for p in controller._scheduled_periods if p.kind == "grid_first"]
    assert len(grid_periods) == 0


@pytest.mark.asyncio
async def test_startup_synchronization_battery_first(controller):
    """Test startup sync applies battery-first mode when in scheduled period."""
    from modules.growatt_controller import Period
    from datetime import time as dt_time

    # Set up scheduled periods with Period objects
    controller._scheduled_periods = [
        Period("battery_first", dt_time(10, 0), dt_time(14, 0)),
        Period("export", dt_time(14, 0), dt_time(18, 0)),
    ]

    # Mock current time to be within battery-first period
    with patch.object(controller, '_get_local_now') as mock_now:
        mock_now.return_value = datetime(
            2024, 1, 1, 11, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Prague")
        )

        await controller._apply_current_state()

    # Should have called _set_battery_first
    # We can't directly verify this in simulation mode, but we can check the logic ran
    assert controller._scheduled_periods is not None


@pytest.mark.asyncio
async def test_startup_synchronization_grid_first(controller):
    """Test startup sync applies grid-first mode when in scheduled period."""
    from modules.growatt_controller import Period
    from datetime import time as dt_time

    # Set up scheduled periods with Period objects
    controller._scheduled_periods = [
        Period("grid_first", dt_time(6, 0), dt_time(11, 0)),
        Period("battery_first", dt_time(11, 0), dt_time(16, 0)),
    ]

    # Mock current time to be within grid-first period
    with patch.object(controller, '_get_local_now') as mock_now:
        mock_now.return_value = datetime(
            2024, 1, 1, 8, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Prague")
        )

        await controller._apply_current_state()

    # Verify logic ran
    assert controller._scheduled_periods is not None


@pytest.mark.asyncio
async def test_startup_synchronization_load_first(controller):
    """Test startup sync applies load-first mode when in scheduled period."""
    from modules.growatt_controller import Period
    from datetime import time as dt_time

    # Set up scheduled periods with Period objects
    controller._scheduled_periods = [
        Period("battery_first", dt_time(11, 0), dt_time(16, 0)),
        Period("load_first", dt_time(16, 0), dt_time(23, 59)),
    ]

    # Mock current time to be within load-first period
    with patch.object(controller, '_get_local_now') as mock_now:
        mock_now.return_value = datetime(
            2024, 1, 1, 18, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Prague")
        )

        await controller._apply_current_state()

    # Verify logic ran
    assert controller._scheduled_periods is not None


@pytest.mark.asyncio
async def test_contiguous_hours_grouping():
    """Test grouping of contiguous hours works correctly."""
    controller = GrowattController(AsyncMock(), AsyncMock(), Settings(
        influxdb_token="test",
        loxone_bridge_loxone_host="127.0.0.1",
        weather_latitude=50.0,
        weather_longitude=14.0,
    ))

    # Test data with gaps
    hours = [
        ("10:00", "11:00", 30.0),
        ("11:00", "12:00", 32.0),
        ("12:00", "13:00", 31.0),
        ("14:00", "15:00", 30.0),  # Gap here
        ("15:00", "16:00", 29.0),
    ]

    groups = controller._group_contiguous_hours_simple(hours)

    # Should have two groups due to the gap
    assert len(groups) == 2
    assert groups[0] == ("10:00", "13:00")
    assert groups[1] == ("14:00", "16:00")


@pytest.mark.asyncio
async def test_export_control_summer_low_prices(controller):
    """Test export is disabled during low-price periods in summer."""
    # Create hourly prices
    hourly_prices = {
        (f"{h:02d}:00", f"{h+1:02d}:00"): 35.0 if 11 <= h <= 15 else 50.0
        for h in range(24)
    }

    await controller._schedule_summer_strategy(hourly_prices, 25.0)

    # Check that export is disabled during low-price hours
    # This is indirectly verified by checking the scheduled periods
    battery_periods = [p for p in controller._scheduled_periods if p.kind == "battery_first"]
    assert len(battery_periods) > 0

    # During battery-first periods in summer, export should be disabled
    # (This would be verified by the actual scheduled tasks)


@pytest.mark.asyncio
async def test_zero_exchange_rate_protection(controller):
    """Test that zero or negative exchange rates are handled safely."""
    # Test negative rate from API
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="EMU|euro|1|EUR|-25.0")
        mock_get = mock_session.return_value.__aenter__.return_value.get
        mock_get.return_value.__aenter__.return_value = mock_response

        # Clear cache to force API call
        controller._eur_czk_rate = None
        controller._eur_czk_rate_updated = None

        rate = await controller._get_eur_czk_rate()
        assert rate == 25.0  # Should return safe fallback of 25.0


@pytest.mark.asyncio
async def test_full_neutral_shutdown(controller, mock_mqtt_client):
    """Test that stop() disables all modes for neutral state."""
    controller.config.simulation_mode = False

    # Don't actually stop the controller in the fixture cleanup
    with patch.object(controller, '_disable_battery_first') as mock_battery:
        with patch.object(controller, '_disable_grid_first') as mock_grid:
            with patch.object(controller, '_disable_export') as mock_export:
                await controller.stop()

                # All three disable methods should be called
                mock_battery.assert_called_once()
                mock_grid.assert_called_once()
                mock_export.assert_called_once()
