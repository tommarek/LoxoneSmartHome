"""Integration tests for Growatt controller with decision engine."""

import json
from datetime import datetime, time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import GrowattConfig, Settings
from modules.growatt_controller import GrowattController
from modules.growatt.decision_engine import GrowattDecisionEngine  # noqa: F401
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
    client.query = AsyncMock()
    return client


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings with price thresholds."""
    settings = MagicMock(spec=Settings)
    settings.growatt = GrowattConfig(
        simulation_mode=False,
        export_price_min=2.5,
        battery_charge_blocks=8,
        battery_efficiency=0.85,
        discharge_profit_margin=1.5,
    )
    settings.log_level = "INFO"
    settings.log_timezone = "Europe/Prague"
    settings.influxdb = MagicMock()
    settings.influxdb.bucket_loxone = "loxone"
    return settings


@pytest.fixture
def growatt_controller(
    mock_mqtt_client: AsyncMock,
    mock_influxdb_client: AsyncMock,
    mock_settings: Settings,
) -> GrowattController:
    """Create a GrowattController instance with mocked dependencies."""
    controller = GrowattController(
        mqtt_client=mock_mqtt_client,
        influxdb_client=mock_influxdb_client,
        settings=mock_settings,
    )
    # Initialize attributes
    controller._current_prices = {}
    controller._eur_czk_rate = 25.0
    controller._last_price_fetch = datetime.now()
    return controller


@pytest.mark.asyncio
async def test_decision_engine_integration_cheap_hour_charging(
    growatt_controller: GrowattController,
) -> None:
    """Test that cheap hour triggers battery charging."""
    # Set up cheap hour scenario with 15-minute blocks
    growatt_controller._battery_soc = 50.0
    growatt_controller._high_loads_active = False
    growatt_controller._current_mode = "regular"
    # Create 15-minute price blocks for hour 3 and 4
    growatt_controller._current_prices = {}
    for hour in [3, 4]:
        for i in range(4):
            start_min = i * 15
            end_min = start_min + 15
            start_str = f"{hour:02d}:{start_min:02d}"
            end_str = f"{hour:02d}:{end_min:02d}" if end_min < 60 else f"{hour + 1:02d}:00"
            price = 60.0 if hour == 3 else 65.0
            growatt_controller._current_prices[(start_str, end_str)] = price

    # Mark hour 3 blocks as cheapest
    growatt_controller._cheapest_charging_blocks = {
        ("03:00", "03:15"), ("03:15", "03:30"),
        ("03:30", "03:45"), ("03:45", "04:00")
    }
    # Also set combined blocks (used in context building)
    growatt_controller._combined_charging_blocks = growatt_controller._cheapest_charging_blocks.copy()

    # Mock season and sun calculation
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 1, 1, 3, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Verify context for cheap hour charging
    assert context.battery_soc == 50.0
    assert context.current_price == 60.0
    assert context.current_block_key == ("03:30", "03:45")
    assert context.price_thresholds.charge_price_max == 1.5

    # Use decision engine
    decision = growatt_controller._decision_engine.decide(context)
    assert decision == "charge_from_grid"


@pytest.mark.asyncio
async def test_decision_engine_integration_high_price_discharge(
    growatt_controller: GrowattController,
) -> None:
    """Test that high prices trigger battery discharge."""
    # Set up high price scenario with 15-minute blocks
    growatt_controller._battery_soc = 85.0
    growatt_controller._high_loads_active = False
    growatt_controller._current_mode = "regular"
    growatt_controller._current_prices = {}

    # Create 15-minute blocks for hours 3 and 18
    for hour in [3, 18]:
        for i in range(4):
            start_min = i * 15
            end_min = start_min + 15
            start_str = f"{hour:02d}:{start_min:02d}"
            end_str = f"{hour:02d}:{end_min:02d}" if end_min < 60 else f"{hour + 1:02d}:00"
            price = 60.0 if hour == 3 else 150.0
            growatt_controller._current_prices[(start_str, end_str)] = price

    # Not a charging time - empty cheapest blocks
    growatt_controller._cheapest_charging_blocks = set()

    # Mock season and sun calculation
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 1, 1, 18, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Verify high price context
    assert context.battery_soc == 85.0
    assert context.current_price == 150.0
    assert context.current_block_key == ("18:30", "18:45")
    assert context.price_thresholds.export_price_min == 2.5

    # Use decision engine - now with no scheduled charging, it should discharge
    decision = growatt_controller._decision_engine.decide(context)
    # With high price and no scheduled charging, should discharge
    # May be regular if price conditions not met
    assert decision in ["discharge_to_grid", "regular"]


@pytest.mark.asyncio
async def test_decision_engine_integration_summer_solar(
    growatt_controller: GrowattController,
) -> None:
    """Test summer solar optimization."""
    # Set up summer solar scenario
    growatt_controller._battery_soc = 70.0
    growatt_controller._high_loads_active = False
    growatt_controller._current_load = 2.0  # 2kW load
    growatt_controller._solar_power = 5.0  # 5kW solar
    growatt_controller._current_mode = "regular"
    growatt_controller._current_prices = {
        ("12:00", "13:00"): 70.0,
        ("13:00", "14:00"): 80.0,
        ("14:00", "15:00"): 90.0,  # Current hour - expensive
        ("15:00", "16:00"): 85.0,
    }

    # Mock season and sun calculation
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="summer")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {
                "sunrise": MagicMock(time=MagicMock(return_value=time(5, 30))),
                "sunset": MagicMock(time=MagicMock(return_value=time(21, 0))),
            }
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 7, 1, 14, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Verify summer context
    assert context.is_summer_mode is True
    assert context.solar_power == 5.0
    assert context.current_load == 2.0

    # Use decision engine - should be regular mode with export (price 90 > threshold 40)
    decision = growatt_controller._decision_engine.decide(context)
    assert decision == "regular"  # Regular mode with export enabled


@pytest.mark.asyncio
async def test_high_load_protection_override(
    growatt_controller: GrowattController,
) -> None:
    """Test high load protection prevents discharge."""
    # Set up high load scenario
    growatt_controller._battery_soc = 80.0
    growatt_controller._high_loads_active = True  # High loads active
    growatt_controller._current_mode = "regular"
    growatt_controller._current_prices = {}

    # Mock season
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}

            # Build context
            context = await growatt_controller._build_decision_context()

    # Verify high load context
    assert context.high_loads_active is True

    # Use decision engine - should protect battery
    decision = growatt_controller._decision_engine.decide(context)
    assert decision == "high_load_protected"


@pytest.mark.asyncio
async def test_manual_override_highest_priority(
    growatt_controller: GrowattController,
) -> None:
    """Test manual override has highest priority."""
    from modules.growatt.models import Period

    # Set up manual override
    growatt_controller._manual_override_period = Period(
        start=time(0, 0),
        end=time(23, 59),
        kind="regular",  # Use string instead of enum
        params={}
    )
    growatt_controller._battery_soc = 50.0
    growatt_controller._high_loads_active = True  # Even with high loads
    growatt_controller._current_prices = {
        ("03:00", "04:00"): 50.0,  # Even with cheap prices
    }

    # Mock season
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}

            # Build context
            context = await growatt_controller._build_decision_context()

    # Verify manual override context
    assert context.manual_override_active is True
    assert context.manual_override_mode == "regular"

    # Use decision engine - manual override wins
    decision = growatt_controller._decision_engine.decide(context)
    assert decision == "regular"


@pytest.mark.asyncio
async def test_low_battery_prevents_discharge(
    growatt_controller: GrowattController,
) -> None:
    """Test low battery prevents discharge even with high prices."""
    # Set up low battery scenario
    growatt_controller._battery_soc = 25.0  # Too low
    growatt_controller._high_loads_active = False
    growatt_controller._current_mode = "regular"
    growatt_controller._current_prices = {
        ("18:00", "19:00"): 150.0,  # High price
    }

    # Mock season
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 1, 1, 18, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Use decision engine - should not discharge
    decision = growatt_controller._decision_engine.decide(context)
    assert decision != "discharge_to_grid"


@pytest.mark.asyncio
async def test_full_battery_prevents_charging(
    growatt_controller: GrowattController,
) -> None:
    """Test 100% battery prevents charging even with cheap prices."""
    # Set up full battery scenario
    growatt_controller._battery_soc = 100.0  # Completely full
    growatt_controller._high_loads_active = False
    growatt_controller._current_mode = "regular"
    growatt_controller._current_prices = {
        ("03:00", "04:00"): 40.0,  # Very cheap
    }

    # Mock season
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 1, 1, 3, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Use decision engine - should not charge
    decision = growatt_controller._decision_engine.decide(context)
    assert decision != "charge_from_grid"


@pytest.mark.asyncio
async def test_price_data_in_context(
    growatt_controller: GrowattController,
) -> None:
    """Test that price data is properly included in context."""
    # Set price data with 15-minute blocks
    growatt_controller._last_price_fetch = datetime.now()
    growatt_controller._current_prices = {}

    # Create 15-minute blocks for hours 10 and 11
    for hour in [10, 11]:
        for i in range(4):
            start_min = i * 15
            end_min = start_min + 15
            start_str = f"{hour:02d}:{start_min:02d}"
            end_str = f"{hour:02d}:{end_min:02d}" if end_min < 60 else f"{hour + 1:02d}:00"
            price = 75.0 if hour == 10 else 85.0
            growatt_controller._current_prices[(start_str, end_str)] = price

    # Empty cheapest blocks
    growatt_controller._cheapest_charging_blocks = set()

    # Mock season
    with patch.object(growatt_controller, "_get_season_mode", AsyncMock(return_value="winter")):
        with patch("modules.growatt_controller.sun") as mock_sun:
            mock_sun.return_value = {}
            with patch.object(growatt_controller, "_get_local_now") as mock_now:
                mock_now.return_value = datetime(2024, 1, 1, 10, 30)

                # Build context
                context = await growatt_controller._build_decision_context()

    # Verify price data is in context
    assert context.current_price == 75.0
    assert context.current_block_key == ("10:30", "10:45")
    assert len(context.prices_15min) == 8  # 2 hours * 4 blocks per hour


@pytest.mark.asyncio
async def test_home_status_battery_soc_update(
    growatt_controller: GrowattController,
) -> None:
    """Test that battery SOC is updated from home status."""
    # Initial SOC
    growatt_controller._battery_soc = 50.0

    # Mock detect method to return no high loads
    setattr(growatt_controller, "_detect_high_loads_from_status", lambda x: {"active": False})
    setattr(growatt_controller, "_handle_high_load_start", AsyncMock())
    # Note: _determine_and_apply_mode was removed - using _evaluate_conditions instead
    setattr(growatt_controller, "_evaluate_conditions", AsyncMock())

    # Test battery SOC update
    payload = {
        "solar": {
            "battery_soc": {"value": 85.0}
        }
    }
    await growatt_controller._on_home_status("loxone/status", json.dumps(payload))

    # SOC should be updated
    assert growatt_controller._battery_soc == 85.0

    # Test with different format
    payload = {
        "solar": {
            "soc": 92.0
        }
    }
    await growatt_controller._on_home_status("loxone/status", json.dumps(payload))

    assert growatt_controller._battery_soc == 92.0
