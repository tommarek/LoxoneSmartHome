"""Unit tests for the Growatt decision engine."""

import pytest
from datetime import datetime, time
from typing import Dict, Tuple
from unittest.mock import MagicMock

from modules.growatt.decision_engine import (
    GrowattDecisionEngine,
    DecisionContext,
    PriceThresholds,
    PriceRankingData,
    MODE_DEFINITIONS,
    Priority
)


def create_15min_prices(hour_prices: Dict[int, float]) -> Dict[Tuple[str, str], float]:
    """Helper to create 15-minute price blocks from hourly prices.

    Args:
        hour_prices: Dict mapping hour (0-23) to price

    Returns:
        Dict with 15-minute block keys
    """
    prices_15min = {}
    for hour, price in hour_prices.items():
        for i in range(4):  # 4 blocks per hour
            start_min = i * 15
            end_min = start_min + 15
            if hour == 23 and end_min == 60:
                start_str = f"{hour:02d}:{start_min:02d}"
                end_str = "24:00"
            else:
                start_str = f"{hour:02d}:{start_min:02d}"
                if end_min < 60:
                    end_str = f"{hour:02d}:{end_min:02d}"
                else:
                    end_str = f"{(hour + 1) % 24:02d}:00"
            prices_15min[(start_str, end_str)] = price
    return prices_15min


@pytest.fixture
def decision_engine() -> GrowattDecisionEngine:
    """Create a decision engine instance for testing."""
    logger = MagicMock()
    return GrowattDecisionEngine(logger)


@pytest.fixture
def base_context() -> DecisionContext:
    """Create a base context for testing."""
    return DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime.now(),
        manual_override_mode=None,
        scheduled_mode=None,
        current_mode=None
    )


def test_decision_engine_initialization(decision_engine: GrowattDecisionEngine) -> None:
    """Test that decision engine initializes correctly."""
    assert decision_engine is not None
    assert len(decision_engine.decision_tree) == 6  # 6 decision nodes (consolidated export control)
    assert decision_engine._last_decision is None


def test_manual_override_highest_priority(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that manual override has the highest priority."""
    # Set all conditions to trigger
    base_context.manual_override_active = True
    base_context.manual_override_mode = "regular"
    base_context.high_loads_active = True
    base_context.scheduled_mode = "charge_from_grid"

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "regular"  # Export is now always price-based
    assert explanation["priority"]["level"] == Priority.MANUAL_OVERRIDE
    assert "manual override" in explanation["reason"].lower()


def test_battery_charging_overrides_high_load(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that battery charging overrides high load protection."""
    base_context.high_loads_active = True
    base_context.scheduled_mode = "charge_from_grid"
    base_context.is_battery_charging_scheduled = True  # Must set this explicitly

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    # With high loads and cheap price, should use battery_first_ac_charge
    assert decision == "battery_first_ac_charge"
    assert explanation["priority"]["level"] == Priority.SCHEDULED_BATTERY_CHARGING
    assert "high load" in explanation["reason"].lower()


def test_high_load_protection_triggers(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that high load protection triggers when conditions are met."""
    base_context.high_loads_active = True
    base_context.scheduled_mode = "regular"  # Not battery charging

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "high_load_protected"
    assert explanation["priority"]["level"] == Priority.HIGH_LOAD_PROTECTION
    assert "high loads" in explanation["reason"].lower()

    # Verify mode definition
    mode_def = MODE_DEFINITIONS["high_load_protected"]
    assert mode_def["stop_soc"] == 100  # Should prevent discharge
    assert mode_def["inverter_mode"] == "load_first"


def test_scheduled_mode_ignored_for_price_decisions(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that scheduled modes are ignored in favor of price-based decisions."""
    base_context.scheduled_mode = "sell_production"  # Legacy field

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    # Should fall back to regular mode without price data or conditions
    assert decision == "regular"
    assert explanation["priority"]["level"] == Priority.DEFAULT_MODE
    # Expect default mode reason, not scheduled
    assert "special" in explanation["reason"].lower() or "regular" in explanation["reason"].lower()


def test_default_mode_fallback(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that default mode is used when no conditions match."""
    # All conditions false/None - should use default
    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "regular"
    assert explanation["priority"]["level"] == Priority.DEFAULT_MODE
    assert "no special conditions" in explanation["reason"].lower()


def test_mode_transitions(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test mode transitions as conditions change."""
    # Start with regular mode
    base_context.scheduled_mode = "regular"
    assert decision_engine.decide(base_context) == "regular"

    # High load triggers protection
    base_context.high_loads_active = True
    assert decision_engine.decide(base_context) == "high_load_protected"

    # Battery charging with high load -> battery_first_ac_charge
    base_context.scheduled_mode = "charge_from_grid"
    base_context.is_battery_charging_scheduled = True  # Must set this explicitly
    assert decision_engine.decide(base_context) == "battery_first_ac_charge"

    # Manual override takes precedence over everything
    base_context.manual_override_active = True
    base_context.manual_override_mode = "regular"
    assert decision_engine.decide(base_context) == "regular"

    # Clear manual override, should go back to battery_first_ac_charge (high load + charging)
    base_context.manual_override_active = False
    assert decision_engine.decide(base_context) == "battery_first_ac_charge"

    # Clear high loads, should still be charge_from_grid
    base_context.high_loads_active = False
    assert decision_engine.decide(base_context) == "charge_from_grid"

    # Change scheduled mode to regular
    base_context.scheduled_mode = "regular"
    base_context.is_battery_charging_scheduled = False  # No longer charging
    assert decision_engine.decide(base_context) == "regular"


def test_mode_definitions_complete() -> None:
    """Test that all required mode definitions exist."""
    required_modes = [
        "regular",
        "regular",
        "high_load_protected",
        "charge_from_grid",
        "battery_first_ac_charge",
        "discharge_to_grid",
        "sell_production"
    ]

    for mode in required_modes:
        assert mode in MODE_DEFINITIONS
        mode_def = MODE_DEFINITIONS[mode]
        assert "description" in mode_def
        assert "inverter_mode" in mode_def
        assert "stop_soc" in mode_def or mode_def.get("stop_soc") == "configurable"
        # Export is now always handled separately based on price
        assert "ac_charge" in mode_def


def test_visualization_output(decision_engine: GrowattDecisionEngine) -> None:
    """Test that visualization produces output."""
    viz = decision_engine.visualize_tree()
    assert viz is not None
    assert len(viz) > 0
    assert "Decision Tree" in viz
    assert "Manual Override" in viz
    assert "Battery Charging" in viz
    assert "High Load Protection" in viz


def test_test_scenario_method(decision_engine: GrowattDecisionEngine) -> None:
    """Test the test_scenario convenience method."""
    result = decision_engine.test_scenario(
        manual_override=False,
        high_loads=True,
        scheduled_mode="regular",
        battery_soc=45.0
    )

    assert "scenario" in result
    assert "result" in result
    assert result["result"]["decision"] == "high_load_protected"
    assert result["scenario"]["high_loads"] is True
    assert result["scenario"]["battery_soc"] == 45.0


def test_explanation_without_decision(decision_engine: GrowattDecisionEngine) -> None:
    """Test explanation when no decision has been made yet."""
    explanation = decision_engine.explain_decision()
    assert explanation["decision"] == "unknown"
    assert "no decision" in explanation["reason"].lower()


def test_explanation_with_context(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test explanation with a provided context."""
    base_context.high_loads_active = True
    base_context.scheduled_mode = "regular"

    # Explain without making decision first
    decision_engine._last_decision = None
    decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision(base_context)

    assert explanation["decision"] == "high_load_protected"
    assert explanation["context"]["high_loads"] is True
    assert explanation["mode_details"]["stop_soc"] == 100


def test_charge_from_grid_detection(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that charge_from_grid is properly detected."""
    # Create context with charge_from_grid to trigger __post_init__
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime.now(),
        manual_override_mode=None,
        scheduled_mode="charge_from_grid",
        current_mode=None
    )

    # Should be detected in post_init
    assert context.is_battery_charging_scheduled is True

    decision = decision_engine.decide(context)
    assert decision == "charge_from_grid"


def test_priority_ordering() -> None:
    """Test that priorities are correctly ordered."""
    assert Priority.MANUAL_OVERRIDE < Priority.SCHEDULED_BATTERY_CHARGING
    assert Priority.SCHEDULED_BATTERY_CHARGING < Priority.HIGH_LOAD_PROTECTION
    assert Priority.HIGH_LOAD_PROTECTION < Priority.SCHEDULED_MODE
    assert Priority.SCHEDULED_MODE < Priority.DEFAULT_MODE


def test_all_composite_modes_supported(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that all composite modes from the original system are supported."""
    composite_modes = [
        "regular",
        "sell_production",
        "regular",
        "charge_from_grid",
        "discharge_to_grid"
    ]

    for mode in composite_modes:
        base_context.scheduled_mode = mode
        decision = decision_engine.decide(base_context)
        # Should either be the mode itself or overridden by higher priority
        assert decision in MODE_DEFINITIONS


def test_price_based_charging_decision(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that cheap prices trigger battery charging."""
    # Create 15-minute price data
    prices_15min = {}
    # Add blocks for 3:00-4:00 (4 blocks of 15 minutes each)
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"03:{start_min:02d}"
        end_str = f"03:{end_min:02d}" if end_min < 60 else "04:00"
        prices_15min[(start_str, end_str)] = 60.0  # Cheap price

    # Mark current block as one of the cheapest
    current_block = ("03:00", "03:15")
    cheapest_blocks = {current_block, ("03:15", "03:30"), ("03:30", "03:45"), ("03:45", "04:00")}

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),  # 3 AM
        manual_override_mode=None,
        current_mode="regular",
        current_price=60.0,  # Cheap price
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=3.0,  # 120.0 EUR/MWh
            discharge_price_min=6.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.87
        )
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    assert decision == "charge_from_grid"
    assert "cheap" in explanation["reason"].lower()


def test_price_based_discharge_decision(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that very high prices with sufficient spread trigger battery discharge."""
    # Create 15-minute price data
    prices_15min = {}
    # Add cheap blocks (3:00-4:00)
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"03:{start_min:02d}"
        end_str = f"03:{end_min:02d}" if end_min < 60 else "04:00"
        prices_15min[(start_str, end_str)] = 30.0  # Cheap price

    # Add expensive blocks (18:00-19:00)
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"18:{start_min:02d}"
        end_str = f"18:{end_min:02d}" if end_min < 60 else "19:00"
        prices_15min[(start_str, end_str)] = 150.0  # High price

    current_block = ("18:00", "18:15")

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=80.0,  # Good SOC for discharge
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),  # 6 PM peak
        current_price=150.0,  # 3.75 CZK/kWh
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=set(),  # Not a charging block
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.2,
            battery_efficiency=0.87
        ),
        price_ranking=PriceRankingData(
            current_rank=2,
            total_blocks=2,
            percentile=100.0,
            blocks_cheaper_count=1,
            blocks_more_expensive_count=0,
            daily_min=30.0,  # 0.75 CZK/kWh
            daily_max=150.0,
            daily_avg=90.0,
            daily_median=90.0,
            daily_spread=120.0,
            price_quadrant="Most Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        ),
        is_summer_mode=True  # Set to summer to prevent charging
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # 3.75 CZK/kWh > max(2.0 min, 0.75*1.2 margin) = max(2.0, 0.9) = 2.0, so should discharge
    assert decision == "discharge_to_grid"
    assert "25% power" in explanation["reason"].lower()


def test_low_battery_prevents_discharge(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that low battery SOC prevents discharge even with high prices."""
    prices_15min = {}
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"18:{start_min:02d}"
        end_str = f"18:{end_min:02d}" if end_min < 60 else "19:00"
        prices_15min[(start_str, end_str)] = 150.0  # High price

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=15.0,  # Too low for discharge (below 20% threshold)
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=150.0,  # High price
        current_block_key=("18:00", "18:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    decision = decision_engine.decide(context)
    # Should not discharge with low battery
    assert decision != "discharge_to_grid"


def test_full_battery_prevents_charging(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that full battery prevents charging even with cheap prices."""
    prices_15min = {}
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"03:{start_min:02d}"
        end_str = f"03:{end_min:02d}" if end_min < 60 else "04:00"
        prices_15min[(start_str, end_str)] = 50.0  # Very cheap

    current_block = ("03:00", "03:15")
    cheapest_blocks = {current_block}  # Mark as cheapest block

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=100.0,  # Completely full
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=50.0,  # Very cheap
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=1.0,  # 40 EUR/MWh = 1 CZK/kWh
            discharge_price_min=3.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    decision = decision_engine.decide(context)
    # Should not charge when at 100%
    assert decision != "charge_from_grid"
    assert decision != "battery_first_ac_charge"


def test_export_enabled_above_threshold(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test export is enabled when price above threshold."""
    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=100.0,  # Full battery to avoid charging decision
        current_mode="regular",
        current_load=2.0,  # 2kW load
        solar_power=5.0,   # 5kW solar (exceeds load)
        current_time=datetime(2024, 7, 1, 14, 0),  # Summer afternoon
        sunrise=time(5, 30),
        sunset=time(21, 0),
        is_summer_mode=True,
        current_price=50.0,  # Above export threshold but below cheap threshold
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=1.0,  # 40 EUR/MWh = 1 CZK/kWh
            discharge_price_min=3.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    decision = decision_engine.decide(context)
    decision_engine.explain_decision()

    # Should choose regular mode (export is handled separately)
    assert decision == "regular"  # Regular mode (export handled separately)
    # Export control is handled separately, not in mode decision


def test_low_price_export_disabled(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that prices below threshold disable export."""
    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=100.0,  # Full battery to avoid charging
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 12, 0),
        current_price=30.0,  # Below export threshold
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=1.0,  # 40 EUR/MWh = 1 CZK/kWh
            discharge_price_min=3.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    decision = decision_engine.decide(context)
    decision_engine.explain_decision()

    # Should choose regular mode (export is handled separately)
    assert decision == "regular"  # Export is now always price-based
    # Export control is handled separately, not in mode decision


def test_price_validation_invalid_data(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that invalid price data is handled correctly."""
    prices_15min = {}
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"03:{start_min:02d}"
        end_str = f"03:{end_min:02d}" if end_min < 60 else "04:00"
        prices_15min[(start_str, end_str)] = -50.0  # Invalid negative price

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=-50.0,  # Invalid negative price
        current_block_key=("03:00", "03:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=3.0,  # 120 EUR/MWh = 3 CZK/kWh
            discharge_price_min=3.5,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    # Should fall back to regular mode with invalid prices
    decision = decision_engine.decide(context)
    assert decision == "regular"


def test_time_range_overnight(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test overnight time range handling."""
    prices_15min = {}
    # Add overnight 15-minute blocks
    for hour in [23, 0, 1, 2, 3]:
        for i in range(4):
            start_min = i * 15
            end_min = start_min + 15
            if hour == 23 and end_min == 60:
                start_str = f"{hour:02d}:{start_min:02d}"
                end_str = "24:00"
            else:
                start_str = f"{hour:02d}:{start_min:02d}"
                if end_min < 60:
                    end_str = f"{hour:02d}:{end_min:02d}"
                else:
                    end_str = f"{(hour + 1) % 24:02d}:00"
            prices_15min[(start_str, end_str)] = 60.0

    current_block = ("02:00", "02:15")
    # Mark this as a cheapest block for charging
    cheapest_blocks = {current_block}

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 2, 0),  # 2 AM
        current_price=60.0,
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=3.0,  # 120 EUR/MWh = 3 CZK/kWh
            discharge_price_min=3.5,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    decision = decision_engine.decide(context)
    # Should charge during overnight cheap hours
    assert decision == "charge_from_grid"


def test_profit_margin_calculation(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test profit margin prevents unprofitable discharge."""
    prices_15min = create_15min_prices({
        3: 100.0,  # Charge price too high
        18: 125.0,  # Current price
    })

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=80.0,
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=125.0,  # Above export threshold but not profitable enough
        current_block_key=("18:00", "18:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=3.0,  # 120 EUR/MWh = 3 CZK/kWh
            discharge_price_min=3.5,
            discharge_profit_margin=1.5,  # High margin requirement
            battery_efficiency=0.87
        )
    )

    decision = decision_engine.decide(context)
    # Should not discharge if profit margin isn't met
    assert decision != "discharge_to_grid"


def test_mode_change_detection(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test mode change detection."""
    # First decision
    assert decision_engine.has_mode_changed("regular")
    # Same mode - no change
    assert not decision_engine.has_mode_changed("regular")
    # Different mode - change detected
    assert decision_engine.has_mode_changed("charge_from_grid")
    # Back to regular - change detected
    assert decision_engine.has_mode_changed("regular")


def test_consecutive_cheap_hours_grouping(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that charging blocks are selected based on cheapest prices (non-consecutive OK)."""
    prices_15min = create_15min_prices({
        2: 60.0,   # Cheap
        3: 65.0,   # Cheap
        4: 70.0,   # Cheap
        5: 85.0,   # Expensive
        6: 60.0,   # Cheap again (non-consecutive)
        7: 90.0,   # Expensive
        8: 55.0,   # Cheapest
    })

    # The 8 cheapest blocks should be from hours 8, 2, 6, 3 (in price order)
    # That's 4 blocks from hour 8, 4 blocks from hour 2
    cheapest_blocks = set()
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        # Hour 8 blocks (cheapest at 55.0)
        h8_end = f"08:{end_min:02d}" if end_min < 60 else "09:00"
        cheapest_blocks.add((f"08:{start_min:02d}", h8_end))
        # Hour 2 blocks (second cheapest at 60.0)
        h2_end = f"02:{end_min:02d}" if end_min < 60 else "03:00"
        cheapest_blocks.add((f"02:{start_min:02d}", h2_end))

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 8, 0),
        current_price=55.0,
        current_block_key=("08:00", "08:15"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80 EUR/MWh = 2 CZK/kWh
            export_price_min=3.0,  # 120 EUR/MWh = 3 CZK/kWh
            discharge_price_min=3.5,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    # Should charge when in a cheapest block
    decision = decision_engine.decide(context)
    assert decision == "charge_from_grid"


def test_invalid_time_format_handling(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that invalid time formats are handled gracefully."""
    # Test the _is_hour_in_range method with invalid formats
    result = decision_engine._is_hour_in_range("invalid", "02:00", "04:00")
    assert result is False

    result = decision_engine._is_hour_in_range("03:00", "invalid", "04:00")
    assert result is False

    result = decision_engine._is_hour_in_range("03:00", "02:00", "invalid")
    assert result is False


def test_mode_change_tracking(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that mode changes are properly tracked."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime.now(),
        manual_override_mode=None,
        current_mode="regular"
    )

    # First decision
    mode1 = decision_engine.decide(context)
    assert decision_engine.has_mode_changed(mode1) is True  # First is always a change

    # Same mode again
    assert decision_engine.has_mode_changed(mode1) is False

    # Different mode
    mode2 = "charge_from_grid" if mode1 != "charge_from_grid" else "regular"
    assert decision_engine.has_mode_changed(mode2) is True


def test_no_price_data_fallback(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test behavior when no price data is available."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime.now(),
        manual_override_mode=None,
        current_mode="regular",
        current_price=0.0,  # No price data
        current_block_key=None,
        prices_15min={},    # Empty prices
        cheapest_blocks=set(),
        price_thresholds=None  # No thresholds
    )

    decision = decision_engine.decide(context)
    # Should fall back to regular mode without price data
    assert decision == "regular"


def test_calculate_price_ranking(decision_engine: GrowattDecisionEngine) -> None:
    """Test price ranking calculation."""
    # Create 15-minute prices - we need at least a few different price levels
    prices_15min = {}
    prices_15min.update(create_15min_prices({0: 50.0}))  # 4 blocks
    prices_15min.update(create_15min_prices({1: 40.0}))  # 4 blocks - Cheapest
    prices_15min.update(create_15min_prices({2: 60.0}))  # 4 blocks
    prices_15min.update(create_15min_prices({3: 80.0}))  # 4 blocks
    prices_15min.update(create_15min_prices({4: 100.0}))  # 4 blocks - Most expensive
    prices_15min.update(create_15min_prices({5: 70.0}))  # 4 blocks
    # Total: 24 blocks

    # Test cheapest block (from hour 1)
    ranking = decision_engine.calculate_price_ranking(("01:00", "01:15"), prices_15min)
    assert ranking is not None
    assert ranking.current_rank == 1
    assert ranking.percentile == 0.0
    assert ranking.price_quadrant == "Cheapest"
    assert ranking.is_relatively_cheap is True
    assert ranking.blocks_cheaper_count == 0
    # Total 24 blocks, this is 1st, so 23 are more expensive
    assert ranking.blocks_more_expensive_count == 23

    # Test most expensive block (from hour 4)
    ranking = decision_engine.calculate_price_ranking(("04:00", "04:15"), prices_15min)
    assert ranking is not None
    assert ranking.current_rank == 21  # 20 blocks are cheaper, this is 21st
    assert ranking.percentile >= 80.0  # Should be near 100%
    assert ranking.price_quadrant == "Most Expensive"
    assert ranking.is_relatively_expensive is True
    assert ranking.blocks_cheaper_count == 20
    # There are 3 other blocks from hour 4 with the same price, so they're also at rank 21
    assert ranking.blocks_more_expensive_count == 3

    # Test middle block (from hour 2)
    ranking = decision_engine.calculate_price_ranking(("02:00", "02:15"), prices_15min)
    assert ranking is not None
    assert ranking.current_rank == 9  # 8 blocks cheaper (4 from hour 1, 4 from hour 0)
    assert 30 < ranking.percentile < 40  # Around 35%
    assert ranking.price_quadrant == "Cheap"


def test_percentile_based_charging_decision(decision_engine: GrowattDecisionEngine) -> None:
    """Test that charging decisions use percentile ranking when available."""
    prices_15min = create_15min_prices({
        0: 80.0,
        1: 40.0,  # Cheapest - should charge
        2: 90.0,
        3: 100.0,
    })

    # Mark hour 1 blocks as cheapest
    cheapest_blocks = {
        ("01:00", "01:15"), ("01:15", "01:30"),
        ("01:30", "01:45"), ("01:45", "02:00")
    }

    # Create ranking for cheap block
    price_ranking = PriceRankingData(
        current_rank=1,
        total_blocks=16,  # 4 hours * 4 blocks
        percentile=0.0,
        blocks_cheaper_count=0,
        blocks_more_expensive_count=15,
        daily_min=40.0,
        daily_max=100.0,
        daily_avg=77.5,
        daily_median=85.0,
        daily_spread=60.0,
        price_quadrant="Cheapest",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 1, 30),
        current_price=40.0,
        current_block_key=("01:30", "01:45"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=1.2,  # 50.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=price_ranking
    )

    decision = decision_engine.decide(context)
    # Should charge because in cheapest blocks
    assert decision == "charge_from_grid"


def test_percentile_based_export_decision(decision_engine: GrowattDecisionEngine) -> None:
    """Test that export decisions use absolute threshold only (percentile info is for context)."""
    prices_15min = create_15min_prices({
        0: 30.0,  # Below threshold
        1: 40.0,  # At threshold
        2: 60.0,  # Above threshold
        3: 80.0,  # Well above threshold
    })

    # Test 1: Price below absolute threshold - should NOT export
    price_ranking = PriceRankingData(
        current_rank=4,
        total_blocks=4,
        percentile=0.0,  # Cheapest hour
        blocks_cheaper_count=3,
        blocks_more_expensive_count=0,
        daily_min=30.0,
        daily_max=80.0,
        daily_avg=52.5,
        daily_median=50.0,
        daily_spread=50.0,
        price_quadrant="Cheap",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=100.0,  # Full battery to prevent charging decision
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=30.0,  # Below threshold
        current_block_key=("00:30", "00:45"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=0.5,  # 20.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=price_ranking
    )

    decision = decision_engine.decide(context)
    decision_engine.explain_decision()

    # Should choose regular mode (export is handled separately)
    assert decision == "regular"  # Export is now always price-based
    # Export control is handled separately, not in mode decision

    # Test 2: Price above absolute threshold - should export
    context.current_price = 60.0
    context.current_time = datetime(2024, 1, 1, 2, 30)

    decision = decision_engine.decide(context)
    decision_engine.explain_decision()

    # Should choose regular mode (export is handled separately)
    assert decision == "regular"
    # Export control is handled separately, not in mode decision


def test_winter_charging_logic(decision_engine: GrowattDecisionEngine) -> None:
    """Test that winter mode only charges during 2 cheapest hours."""
    prices_15min = create_15min_prices({
        0: 20.0,  # Cheapest hour
        1: 25.0,  # Second cheapest
        2: 35.0,  # Third
        3: 40.0,  # Fourth
        4: 50.0,  # Most expensive
    })

    # Mark hours 0 and 1 blocks as cheapest (8 blocks total = 2 hours)
    cheapest_blocks = set()
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        # Hour 0 blocks
        h0_end = f"00:{end_min:02d}" if end_min < 60 else "01:00"
        cheapest_blocks.add((f"00:{start_min:02d}", h0_end))
        # Hour 1 blocks
        h1_end = f"01:{end_min:02d}" if end_min < 60 else "02:00"
        cheapest_blocks.add((f"01:{start_min:02d}", h1_end))

    # Test 1: Winter mode, rank 1 (cheapest hour) - should charge
    price_ranking = PriceRankingData(
        current_rank=1,
        total_blocks=5,
        percentile=0.0,
        blocks_cheaper_count=0,
        blocks_more_expensive_count=4,
        daily_min=20.0,
        daily_max=50.0,
        daily_avg=34.0,
        daily_median=35.0,
        daily_spread=30.0,
        price_quadrant="Cheapest",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,  # Not full
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=20.0,
        current_block_key=("00:30", "00:45"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # 30.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=price_ranking,
        is_summer_mode=False  # Winter mode
    )

    decision = decision_engine.decide(context)
    # Should charge because rank 1 <= 2 cheapest hours
    assert decision == "charge_from_grid"

    # Test 2: Winter mode, rank 3 - should NOT charge
    # Need to create a new context to trigger __post_init__ with the new values
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 2, 30),
        current_price=35.0,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # 30.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=3,
            total_blocks=5,
            percentile=50.0,
            blocks_cheaper_count=2,
            blocks_more_expensive_count=2,
            daily_min=20.0,
            daily_max=50.0,
            daily_avg=34.0,
            daily_median=35.0,
            daily_spread=30.0,
            price_quadrant="Cheap",
            is_relatively_cheap=True,
            is_relatively_expensive=False
        ),
        is_summer_mode=False
    )

    decision = decision_engine.decide(context)
    # Should NOT charge because rank 3 > 2 cheapest hours (winter logic takes precedence)
    # Export should be disabled because price (35) < threshold (40)
    assert decision == "regular"  # Export is now always price-based

    # Test 3: Summer mode - should NEVER charge from AC
    # Test with cheapest hour to ensure no charging happens
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=20.0,  # Cheapest price
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # 30.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=1,  # Cheapest hour
            total_blocks=5,
            percentile=0.0,  # Lowest percentile
            blocks_cheaper_count=0,
            blocks_more_expensive_count=4,
            daily_min=20.0,
            daily_max=50.0,
            daily_avg=34.0,
            daily_median=35.0,
            daily_spread=30.0,
            price_quadrant="Cheapest",
            is_relatively_cheap=True,
            is_relatively_expensive=False
        ),
        is_summer_mode=True  # Summer mode
    )

    decision = decision_engine.decide(context)
    # Should NOT charge in summer mode, even during cheapest hour
    # Export should be disabled because price (20) < threshold (40)
    assert decision == "regular"  # Export is now always price-based

    # Test 4: Summer mode with moderately expensive hour - still no charging
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 30),
        current_price=40.0,  # At threshold
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # 30.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=4,
            total_blocks=5,
            percentile=75.0,  # Below discharge threshold
            blocks_cheaper_count=3,
            blocks_more_expensive_count=1,
            daily_min=20.0,
            daily_max=50.0,
            daily_avg=34.0,
            daily_median=35.0,
            daily_spread=30.0,
            price_quadrant="Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        ),
        is_summer_mode=True  # Summer mode
    )

    decision = decision_engine.decide(context)
    # Should NOT charge in summer mode, regardless of price
    # Export should be enabled because price (40) >= threshold (40)
    assert decision == "regular"


def test_price_spread_discharge_logic(decision_engine: GrowattDecisionEngine) -> None:
    """Test smart discharge based on price spread relative to daily minimum."""
    # Create a day with significant price variation
    prices_15min = create_15min_prices({
        0: 32.0,   # 0.8 CZK/kWh
        1: 36.0,   # 0.9 CZK/kWh
        2: 40.0,   # 1.0 CZK/kWh
        6: 80.0,   # 2.0 CZK/kWh
        7: 120.0,  # 3.0 CZK/kWh - should discharge
        8: 140.0,  # 3.5 CZK/kWh - should discharge
    })

    # For this test, no blocks are marked as cheapest (not charging time)
    cheapest_blocks = set()

    # Test 1: High price with sufficient spread - should discharge
    # 3.0 CZK/kWh > max(2.0 min, 0.8*3.0 margin) = max(2.0, 2.4) = 2.4 ✓
    price_ranking = PriceRankingData(
        current_rank=5,
        total_blocks=6,
        percentile=83.3,
        blocks_cheaper_count=4,
        blocks_more_expensive_count=1,
        daily_min=32.0,  # 0.8 CZK/kWh
        daily_max=140.0,
        daily_avg=73.0,
        daily_median=60.0,
        daily_spread=108.0,
        price_quadrant="Most Expensive",
        is_relatively_cheap=False,
        is_relatively_expensive=True
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 7, 30),
        current_price=120.0,  # 3.0 CZK/kWh
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=3.0,  # Changed from 1.5 to match test intent
            battery_efficiency=0.85
        ),
        price_ranking=price_ranking
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # Should discharge: 3.0 >= max(2.0, 0.8*3.0) = 2.4 ✓
    assert decision == "discharge_to_grid"
    assert "25% power" in explanation["reason"].lower()

    # Test 2: Price below spread requirement - should NOT discharge
    # 2.0 CZK/kWh < max(2.0 min, 0.8*3.0 margin) = max(2.0, 2.4) = 2.4 ✗
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 6, 30),
        current_price=80.0,  # 2.0 CZK/kWh
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=3.0,  # Changed from 1.5 to match test intent
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=4,
            total_blocks=6,
            percentile=66.7,
            blocks_cheaper_count=3,
            blocks_more_expensive_count=2,
            daily_min=32.0,  # 0.8 CZK/kWh
            daily_max=140.0,
            daily_avg=73.0,
            daily_median=60.0,
            daily_spread=108.0,
            price_quadrant="Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        )
    )

    decision = decision_engine.decide(context)
    # Should NOT discharge: 2.0 < 2.4 required ✗
    assert decision == "regular"  # Just export, no discharge

    # Test 3: Price below absolute threshold - should NOT discharge
    # 1.0 CZK/kWh < max(2.0 min, 0.8*3.0 margin) = max(2.0, 2.4) = 2.4 ✗
    # NOTE: With new cheapest-blocks logic, won't charge unless block is in cheapest set
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 2, 30),
        current_price=40.0,  # 1.0 CZK/kWh
        current_block_key=("02:00", "02:15"),  # Current block
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,  # Empty - not a charging block
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh (not used with new logic)
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=3.0,  # Changed from 1.5 for consistency
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=3,
            total_blocks=6,
            percentile=50.0,
            blocks_cheaper_count=2,
            blocks_more_expensive_count=3,
            daily_min=32.0,
            daily_max=140.0,
            daily_avg=73.0,
            daily_median=60.0,
            daily_spread=108.0,
            price_quadrant="Cheap",
            is_relatively_cheap=True,
            is_relatively_expensive=False
        )
    )

    decision = decision_engine.decide(context)
    # Should NOT charge (block not in cheapest set) and should NOT discharge (price too low)
    assert decision == "regular"  # Just export, no charge, no discharge

    # Test 4: Flat price day - should NOT discharge even at high prices
    flat_prices_15min = create_15min_prices({
        0: 76.0,  # 1.9 CZK/kWh
        1: 80.0,  # 2.0 CZK/kWh
        2: 84.0,  # 2.1 CZK/kWh
        3: 88.0,  # 2.2 CZK/kWh
    })

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 30),
        current_price=88.0,  # 2.2 CZK/kWh
        current_block_key=("03:30", "03:45"),
        prices_15min=flat_prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # 80.0 EUR/MWh
            export_price_min=1.0,  # 40.0 EUR/MWh
            discharge_price_min=2.0,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        ),
        price_ranking=PriceRankingData(
            current_rank=4,
            total_blocks=4,
            percentile=100.0,
            blocks_cheaper_count=3,
            blocks_more_expensive_count=0,
            daily_min=76.0,  # 1.9 CZK/kWh
            daily_max=88.0,
            daily_avg=82.0,
            daily_median=82.0,
            daily_spread=12.0,
            price_quadrant="Most Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        )
    )

    decision = decision_engine.decide(context)
    # Should NOT discharge: 2.2 >= 2.0 threshold BUT 2.2 < 1.9*3 = 5.7
    assert decision == "regular"  # No discharge on flat days
