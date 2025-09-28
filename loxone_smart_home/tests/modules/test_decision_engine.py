"""Unit tests for the Growatt decision engine."""

import pytest
from datetime import datetime, time
from unittest.mock import MagicMock

from modules.growatt.decision_engine import (
    GrowattDecisionEngine,
    DecisionContext,
    PriceThresholds,
    PriceRankingData,
    MODE_DEFINITIONS,
    Priority
)


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
    base_context.manual_override_mode = "regular_no_export"
    base_context.high_loads_active = True
    base_context.scheduled_mode = "charge_from_grid"

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "regular_no_export"
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
    base_context.manual_override_mode = "regular_no_export"
    assert decision_engine.decide(base_context) == "regular_no_export"

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
        "regular_no_export",
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
        assert "export" in mode_def
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
        "regular_no_export",
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
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),  # 3 AM
        manual_override_mode=None,
        current_mode="regular",
        current_price=60.0,  # Cheap price
        hourly_prices={
            ("03:00", "04:00"): 60.0,
            ("04:00", "05:00"): 65.0,
            ("05:00", "06:00"): 70.0,
        },
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0,
            charge_efficiency=0.87
        )
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    assert decision == "charge_from_grid"
    assert "cheap" in explanation["reason"].lower()


def test_price_based_discharge_decision(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that very high prices trigger battery discharge."""
    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=80.0,  # Good SOC for discharge
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),  # 6 PM peak
        current_price=150.0,  # High price (3.75x export threshold)
        hourly_prices={
            ("03:00", "04:00"): 30.0,  # Cheap charge price
            ("18:00", "19:00"): 150.0,  # Current high price
        },
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=40.0,  # Export at 40, discharge at 120+ (3x)
            charge_efficiency=0.87,
            min_profit_margin=1.2
        )
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # 150 is > 120 (3x40), so should discharge
    assert decision == "discharge_to_grid"
    assert "discharge" in explanation["reason"].lower() or "profit" in explanation["reason"].lower()


def test_low_battery_prevents_discharge(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that low battery SOC prevents discharge even with high prices."""
    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=15.0,  # Too low for discharge (below 20% threshold)
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=150.0,  # High price
        hourly_prices={("18:00", "19:00"): 150.0},
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0
        )
    )

    decision = decision_engine.decide(context)
    # Should not discharge with low battery
    assert decision != "discharge_to_grid"


def test_full_battery_prevents_charging(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that full battery prevents charging even with cheap prices."""
    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=100.0,  # Completely full
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=50.0,  # Very cheap
        hourly_prices={("03:00", "04:00"): 50.0},
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=40.0
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
            cheap_threshold=80.0,
            export_threshold=40.0  # Export above 40 EUR/MWh
        )
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # Should enable export when price > threshold
    assert decision == "regular"  # Regular mode with export enabled
    assert "export enabled" in explanation["reason"].lower()


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
            cheap_threshold=80.0,
            export_threshold=40.0  # Export above 40 EUR/MWh
        )
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # Should disable export when price < threshold
    assert decision == "regular_no_export"
    assert "export disabled" in explanation["reason"].lower()


def test_price_validation_invalid_data(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that invalid price data is handled correctly."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=-50.0,  # Invalid negative price
        hourly_prices={("03:00", "04:00"): -50.0},
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0
        )
    )

    # Should fall back to regular mode with invalid prices
    decision = decision_engine.decide(context)
    assert decision == "regular"


def test_time_range_overnight(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test overnight time range handling."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 2, 0),  # 2 AM
        current_price=60.0,
        hourly_prices={
            ("23:00", "00:00"): 60.0,
            ("00:00", "01:00"): 60.0,
            ("01:00", "02:00"): 60.0,
            ("02:00", "03:00"): 60.0,
            ("03:00", "04:00"): 60.0,
        },
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0
        )
    )

    decision = decision_engine.decide(context)
    # Should charge during overnight cheap hours
    assert decision == "charge_from_grid"


def test_profit_margin_calculation(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test profit margin prevents unprofitable discharge."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=80.0,
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=125.0,  # Above export threshold but not profitable enough
        hourly_prices={
            ("03:00", "04:00"): 100.0,  # Charge price too high
            ("18:00", "19:00"): 125.0,  # Current price
        },
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0,
            charge_efficiency=0.87,
            min_profit_margin=1.5  # High margin requirement
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
    """Test that consecutive cheap hours are properly grouped."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=60.0,
        hourly_prices={
            ("02:00", "03:00"): 60.0,
            ("03:00", "04:00"): 65.0,
            ("04:00", "05:00"): 70.0,
            ("05:00", "06:00"): 85.0,  # Above threshold
            ("06:00", "07:00"): 60.0,  # New cheap period
        },
        price_thresholds=PriceThresholds(
            cheap_threshold=80.0,
            export_threshold=120.0
        )
    )

    # Access private method for testing
    cheap_periods = decision_engine._find_consecutive_cheap_hours(context)

    # Should have 2 separate periods
    assert len(cheap_periods) == 2
    # First period: 02:00-05:00
    assert cheap_periods[0][0] == "02:00"
    assert cheap_periods[0][1] == "05:00"
    # Second period: 06:00-07:00
    assert cheap_periods[1][0] == "06:00"
    assert cheap_periods[1][1] == "07:00"


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
        hourly_prices={},    # Empty prices
        price_thresholds=None  # No thresholds
    )

    decision = decision_engine.decide(context)
    # Should fall back to regular mode without price data
    assert decision == "regular"


def test_calculate_price_ranking(decision_engine: GrowattDecisionEngine) -> None:
    """Test price ranking calculation."""
    hourly_prices = {
        ("00:00", "01:00"): 50.0,
        ("01:00", "02:00"): 40.0,  # Cheapest
        ("02:00", "03:00"): 60.0,
        ("03:00", "04:00"): 80.0,
        ("04:00", "05:00"): 100.0,  # Most expensive
        ("05:00", "06:00"): 70.0,
    }

    # Test cheapest hour
    ranking = decision_engine.calculate_price_ranking(("01:00", "02:00"), hourly_prices)
    assert ranking is not None
    assert ranking.current_rank == 1
    assert ranking.percentile == 0.0
    assert ranking.price_quadrant == "Cheapest"
    assert ranking.is_relatively_cheap is True
    assert ranking.hours_cheaper_count == 0
    assert ranking.hours_more_expensive_count == 5

    # Test most expensive hour
    ranking = decision_engine.calculate_price_ranking(("04:00", "05:00"), hourly_prices)
    assert ranking is not None
    assert ranking.current_rank == 6
    assert ranking.percentile == 100.0
    assert ranking.price_quadrant == "Most Expensive"
    assert ranking.is_relatively_expensive is True
    assert ranking.hours_cheaper_count == 5
    assert ranking.hours_more_expensive_count == 0

    # Test middle hour
    ranking = decision_engine.calculate_price_ranking(("02:00", "03:00"), hourly_prices)
    assert ranking is not None
    assert ranking.current_rank == 3
    assert 35 < ranking.percentile < 45  # Around 40%
    assert ranking.price_quadrant == "Cheap"


def test_percentile_based_charging_decision(decision_engine: GrowattDecisionEngine) -> None:
    """Test that charging decisions use percentile ranking when available."""
    hourly_prices = {
        ("00:00", "01:00"): 80.0,
        ("01:00", "02:00"): 40.0,  # Bottom 25% - should charge
        ("02:00", "03:00"): 90.0,
        ("03:00", "04:00"): 100.0,
    }

    # Create ranking for cheap hour
    price_ranking = PriceRankingData(
        current_rank=1,
        total_hours=4,
        percentile=0.0,
        hours_cheaper_count=0,
        hours_more_expensive_count=3,
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
        hourly_prices=hourly_prices,
        price_thresholds=PriceThresholds(
            cheap_threshold=50.0,  # Would not charge with absolute threshold
            export_threshold=40.0,
            charge_percentile_threshold=25.0  # Should charge with percentile
        ),
        price_ranking=price_ranking
    )

    decision = decision_engine.decide(context)
    # Should charge because percentile (0%) <= threshold (25%)
    assert decision == "charge_from_grid"


def test_percentile_based_export_decision(decision_engine: GrowattDecisionEngine) -> None:
    """Test that export decisions use percentile ranking when available."""
    hourly_prices = {
        ("00:00", "01:00"): 40.0,
        ("01:00", "02:00"): 50.0,
        ("02:00", "03:00"): 60.0,  # 75th percentile - should export
        ("03:00", "04:00"): 80.0,
    }

    # Create ranking for moderately expensive hour (not expensive enough for discharge)
    price_ranking = PriceRankingData(
        current_rank=3,
        total_hours=4,
        percentile=66.7,  # Above export threshold but below discharge
        hours_cheaper_count=2,
        hours_more_expensive_count=1,
        daily_min=40.0,
        daily_max=80.0,
        daily_avg=57.5,
        daily_median=55.0,
        daily_spread=40.0,
        price_quadrant="Expensive",
        is_relatively_cheap=False,
        is_relatively_expensive=True
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=100.0,  # Full battery to avoid charging
        current_time=datetime(2024, 1, 1, 2, 30),
        current_price=60.0,  # Testing 02:00-03:00 hour
        hourly_prices=hourly_prices,
        price_thresholds=PriceThresholds(
            cheap_threshold=70.0,
            export_threshold=90.0,  # Would not export with absolute threshold
            export_percentile_threshold=60.0,  # Should export with percentile
            charge_percentile_threshold=25.0,
            discharge_percentile_threshold=95.0  # Set high to avoid discharge
        ),
        price_ranking=price_ranking
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # Should export because percentile (100%) >= threshold (60%)
    assert decision == "regular"
    assert "expensive hour" in explanation["reason"].lower()
