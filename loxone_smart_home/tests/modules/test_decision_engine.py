"""Unit tests for the Growatt decision engine."""

import pytest
from unittest.mock import MagicMock

from modules.growatt.decision_engine import (
    GrowattDecisionEngine,
    DecisionContext,
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
        manual_override_mode=None,
        high_loads_active=False,
        scheduled_mode=None,
        battery_soc=50.0,
        current_mode=None
    )


def test_decision_engine_initialization(decision_engine: GrowattDecisionEngine) -> None:
    """Test that decision engine initializes correctly."""
    assert decision_engine is not None
    assert len(decision_engine.decision_tree) == 5  # 5 priority levels
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

    assert decision == "charge_from_grid"
    assert explanation["priority"]["level"] == Priority.SCHEDULED_BATTERY_CHARGING
    assert "battery charging" in explanation["reason"].lower()


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


def test_scheduled_mode_applies(
    decision_engine: GrowattDecisionEngine,
    base_context: DecisionContext
) -> None:
    """Test that scheduled modes apply when no overrides are active."""
    base_context.scheduled_mode = "sell_production"

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "sell_production"
    assert explanation["priority"]["level"] == Priority.SCHEDULED_MODE
    assert "scheduled" in explanation["reason"].lower()


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

    # Battery charging overrides high load
    base_context.scheduled_mode = "charge_from_grid"
    base_context.is_battery_charging_scheduled = True  # Must set this explicitly
    assert decision_engine.decide(base_context) == "charge_from_grid"

    # Manual override takes precedence over everything
    base_context.manual_override_active = True
    base_context.manual_override_mode = "regular_no_export"
    assert decision_engine.decide(base_context) == "regular_no_export"

    # Clear manual override, should go back to charge_from_grid
    base_context.manual_override_active = False
    assert decision_engine.decide(base_context) == "charge_from_grid"

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
    mode = decision_engine.decide(base_context)
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
        manual_override_mode=None,
        high_loads_active=False,
        scheduled_mode="charge_from_grid",
        battery_soc=50.0,
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