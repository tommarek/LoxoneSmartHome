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
        current_mode=None
    )


def test_decision_engine_initialization(decision_engine: GrowattDecisionEngine) -> None:
    """Test that decision engine initializes correctly."""
    assert decision_engine is not None
    assert len(decision_engine.decision_tree) == 9  # 9 decision nodes
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
    base_context.is_battery_charging_scheduled = True  # Battery charging scheduled

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
    # Not battery charging (is_battery_charging_scheduled defaults to False)

    decision = decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision()

    assert decision == "high_load_protected"
    assert explanation["priority"]["level"] == Priority.HIGH_LOAD_PROTECTION
    assert "high loads" in explanation["reason"].lower()

    # Verify mode definition
    mode_def = MODE_DEFINITIONS["high_load_protected"]
    assert mode_def["stop_soc"] == "max_soc"  # Should prevent discharge
    assert mode_def["inverter_mode"] == "load_first"


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
    assert decision_engine.decide(base_context) == "regular"

    # High load triggers protection
    base_context.high_loads_active = True
    assert decision_engine.decide(base_context) == "high_load_protected"

    # Battery charging with high load -> battery_first_ac_charge
    base_context.is_battery_charging_scheduled = True
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

    # Change to regular (no battery charging)
    base_context.is_battery_charging_scheduled = False
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

    # Explain without making decision first
    decision_engine._last_decision = None
    decision_engine.decide(base_context)
    explanation = decision_engine.explain_decision(base_context)

    assert explanation["decision"] == "high_load_protected"
    assert explanation["context"]["high_loads"] is True
    assert explanation["mode_details"]["stop_soc"] == "max_soc"


def test_charge_from_grid_detection(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Test that charge_from_grid is properly detected via cheapest blocks."""
    # Create context with current block in cheapest blocks
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime.now(),
        manual_override_mode=None,
        current_mode=None,
        current_block_key=("10:00", "10:15"),
        cheapest_blocks={("10:00", "10:15"), ("11:00", "11:15")}
    )

    # Should be detected in post_init based on current_block_key in cheapest_blocks
    assert context.is_battery_charging_scheduled is True

    decision = decision_engine.decide(context)
    assert decision == "charge_from_grid"


def test_optimizer_owns_charging_no_seasonal_gate(
    decision_engine: GrowattDecisionEngine
) -> None:
    """When the optimizer is active it owns charge economics: the rule-based
    summer/price gate does NOT apply, so a block the optimizer scheduled is
    charged regardless of season/price. Correctness against bad-price charging
    lives in the MILP (objective-driven reserve, no peak grid-charge), not in a
    seasonal heuristic — so the decision engine simply actuates the model's set."""
    block = ("13:00", "13:15")
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 6, 1, 13, 0),
        current_mode="regular",
        current_price=1.2,  # positive; summer gate would block, optimizer won't
        current_block_key=block,
        cheapest_blocks={block},  # optimizer's chosen charge block
        is_summer_mode=True,
        optimizer_active=True,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0, export_price_min=0.8, discharge_price_min=6.0,
            discharge_profit_margin=1.5, battery_efficiency=0.87,
            summer_charge_price_max=0.0,
        ),
    )
    assert context.is_battery_charging_scheduled is True
    assert decision_engine.decide(context) == "charge_from_grid"


def test_summer_gate_still_blocks_rulebased_charge(
    decision_engine: GrowattDecisionEngine
) -> None:
    """Without the optimizer, the rule-based summer policy still applies: a
    positive-price block is NOT charged from grid in summer (only <= 0 CZK)."""
    block = ("18:30", "18:45")
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 6, 1, 18, 30),
        current_mode="regular",
        current_price=3.53,
        current_block_key=block,
        cheapest_blocks={block},
        is_summer_mode=True,
        optimizer_active=False,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0, export_price_min=3.0, discharge_price_min=6.0,
            discharge_profit_margin=1.5, battery_efficiency=0.87,
            summer_charge_price_max=0.0,
        ),
    )
    assert context.is_battery_charging_scheduled is False
    assert decision_engine.decide(context) != "charge_from_grid"


def test_optimizer_charge_skipped_when_block_not_scheduled(
    decision_engine: GrowattDecisionEngine
) -> None:
    """With the optimizer active, a block NOT in its charge set is not charged,
    even in winter (no rule-based fallback sneaks in)."""
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 10, 0),
        current_mode="regular",
        current_price=1.0,
        current_block_key=("10:00", "10:15"),
        cheapest_blocks={("03:00", "03:15")},  # different block scheduled
        is_summer_mode=False,
        optimizer_active=True,
    )
    assert context.is_battery_charging_scheduled is False


def test_optimizer_active_suppresses_price_based_discharge(
    decision_engine: GrowattDecisionEngine
) -> None:
    """When the optimizer drives, the rule-based price-based discharge (4B) must
    NOT fire for a block the optimizer left as hold — otherwise the inverter
    exports to grid while the chart (optimizer schedule) shows it idle.
    Regression for 'graph shows discharge disabled but battery still discharging'."""
    block = ("20:30", "20:45")
    thresholds = PriceThresholds(
        charge_price_max=2.0, export_price_min=1.0, discharge_price_min=5.0,
        discharge_profit_margin=1.5, battery_efficiency=0.87,
    )
    ctx = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=61.0,
        current_time=datetime(2024, 6, 1, 20, 30),
        current_mode="regular",
        current_price=8.65,  # well above discharge threshold → 4B would fire
        current_block_key=block,
        prices_15min={block: 8.65, ("03:00", "03:15"): 1.0},
        price_thresholds=thresholds,
        min_soc=20.0,
        optimizer_active=True,
        optimizer_discharge_blocks=set(),  # optimizer did NOT schedule this block
    )
    assert decision_engine.decide(ctx) != "discharge_to_grid"


def test_optimizer_scheduled_discharge_still_fires(
    decision_engine: GrowattDecisionEngine
) -> None:
    """A block the optimizer DID schedule for discharge still discharges (4A)."""
    block = ("21:00", "21:15")
    ctx = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=61.0,
        current_time=datetime(2024, 6, 1, 21, 0),
        current_mode="regular",
        current_price=9.0,
        current_block_key=block,
        min_soc=20.0,
        optimizer_active=True,
        optimizer_discharge_blocks={block},
    )
    assert decision_engine.decide(ctx) == "discharge_to_grid"


def _hold_ctx(block, **overrides) -> DecisionContext:
    base = dict(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=44.0,
        current_time=datetime(2024, 6, 1, 2, 0),
        current_mode="regular",
        current_price=3.5,
        current_block_key=block,
        min_soc=20.0,
        max_soc=90.0,
        optimizer_active=True,
        optimizer_hold_blocks={block},
    )
    base.update(overrides)
    return DecisionContext(**base)


def test_battery_hold_node_selected(decision_engine: GrowattDecisionEngine) -> None:
    """A block the optimizer scheduled to hold actuates the battery_hold mode
    (preserve battery, serve house from grid) instead of regular/load_first."""
    block = ("02:00", "02:15")
    assert decision_engine.decide(_hold_ctx(block)) == "battery_hold"


def test_battery_hold_holds_flat_settings() -> None:
    """battery_hold uses load_first + AC-charge off (no discharge), but pins
    stop_soc to the CURRENT SOC, not max_soc — in load_first the inverter
    charges toward stop_soc, so max_soc would grid-charge the battery overnight
    instead of holding it flat."""
    bh = MODE_DEFINITIONS["battery_hold"]
    assert bh["inverter_mode"] == "load_first"
    assert bh["stop_soc"] == "current_soc"  # flat hold: no discharge, no charge
    assert bh["ac_charge"] is False


def test_battery_hold_yields_to_manual_override(
    decision_engine: GrowattDecisionEngine
) -> None:
    block = ("02:00", "02:15")
    ctx = _hold_ctx(block, manual_override_active=True, manual_override_mode="regular")
    assert decision_engine.decide(ctx) == "regular"


def test_battery_hold_yields_to_high_load_protection(
    decision_engine: GrowattDecisionEngine
) -> None:
    """During high loads, high_load_protected (priority 3) fires first — it holds
    the battery identically, so the hold block is still protected."""
    block = ("02:00", "02:15")
    ctx = _hold_ctx(block, high_loads_active=True)
    assert decision_engine.decide(ctx) == "high_load_protected"


def test_battery_hold_yields_to_discharge(
    decision_engine: GrowattDecisionEngine
) -> None:
    """If a block is (defensively) flagged for both discharge and hold, the
    higher-priority discharge node wins."""
    block = ("21:00", "21:15")
    ctx = _hold_ctx(
        block, current_time=datetime(2024, 6, 1, 21, 0), current_price=9.0,
        battery_soc=61.0, optimizer_discharge_blocks={block},
    )
    assert decision_engine.decide(ctx) == "discharge_to_grid"


def test_no_battery_hold_when_block_not_scheduled(
    decision_engine: GrowattDecisionEngine
) -> None:
    """A block NOT in the hold set falls through to regular (load_first)."""
    ctx = _hold_ctx(("02:00", "02:15"), current_block_key=("03:00", "03:15"))
    assert decision_engine.decide(ctx) == "regular"


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
        decision = decision_engine.decide(base_context)
        # Should be determined by decision logic based on context
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
        prices_15min[(start_str, end_str)] = 1.5  # Cheap price (CZK/kWh)

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
        current_price=1.5,  # Cheap price (CZK/kWh)
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=3.0,  # CZK/kWh
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
        prices_15min[(start_str, end_str)] = 0.75  # Cheap price (CZK/kWh)

    # Add expensive blocks (18:00-19:00)
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        start_str = f"18:{start_min:02d}"
        end_str = f"18:{end_min:02d}" if end_min < 60 else "19:00"
        prices_15min[(start_str, end_str)] = 3.75  # High price (CZK/kWh)

    current_block = ("18:00", "18:15")

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=80.0,  # Good SOC for discharge
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),  # 6 PM peak
        current_price=3.75,  # CZK/kWh
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=set(),  # Not a charging block
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.75,  # CZK/kWh
            daily_max=3.75,
            daily_avg=2.25,
            daily_median=2.25,
            daily_spread=3.0,
            price_quadrant="Most Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        ),
        is_summer_mode=True  # Set to summer to prevent charging
    )

    decision = decision_engine.decide(context)
    explanation = decision_engine.explain_decision()

    # 3.75 CZK/kWh > max(2.0 min, 0.75 * 1.2 margin) = max(2.0, 0.9) = 2.0, so should discharge
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
        prices_15min[(start_str, end_str)] = 3.75  # High price (CZK/kWh)

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=15.0,  # Too low for discharge (below 20% threshold)
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=3.75,  # High price (CZK/kWh)
        current_block_key=("18:00", "18:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        prices_15min[(start_str, end_str)] = 1.25  # Very cheap (CZK/kWh)

    current_block = ("03:00", "03:15")
    cheapest_blocks = {current_block}  # Mark as cheapest block

    context = DecisionContext(
        manual_override_active=False,
        manual_override_mode=None,
        high_loads_active=False,
        battery_soc=100.0,  # Completely full
        current_mode="regular",
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=1.25,  # Very cheap (CZK/kWh)
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        current_price=1.25,  # Above export threshold but below cheap threshold (CZK/kWh)
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        current_price=0.75,  # Below export threshold (CZK/kWh)
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        prices_15min[(start_str, end_str)] = -1.25  # Invalid negative price (CZK/kWh)

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 0),
        current_price=-1.25,  # Invalid negative price (CZK/kWh)
        current_block_key=("03:00", "03:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=3.0,  # CZK/kWh
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
            prices_15min[(start_str, end_str)] = 1.5  # CZK/kWh

    current_block = ("02:00", "02:15")
    # Mark this as a cheapest block for charging
    cheapest_blocks = {current_block}

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 2, 0),  # 2 AM
        current_price=1.5,  # CZK/kWh
        current_block_key=current_block,
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=3.0,  # CZK/kWh
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
        3: 2.5,  # Charge price too high (CZK/kWh)
        18: 3.125,  # Current price (CZK/kWh)
    })

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=80.0,
        current_time=datetime(2024, 1, 1, 18, 0),
        current_price=3.125,  # Above export threshold but not profitable enough (CZK/kWh)
        current_block_key=("18:00", "18:15"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=3.0,  # CZK/kWh
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
        2: 1.5,    # Cheap (CZK/kWh)
        3: 1.625,  # Cheap (CZK/kWh)
        4: 1.75,   # Cheap (CZK/kWh)
        5: 2.125,  # Expensive (CZK/kWh)
        6: 1.5,    # Cheap again (non-consecutive) (CZK/kWh)
        7: 2.25,   # Expensive (CZK/kWh)
        8: 1.375,  # Cheapest (CZK/kWh)
    })

    # The 8 cheapest blocks should be from hours 8, 2, 6, 3 (in price order)
    # That's 4 blocks from hour 8, 4 blocks from hour 2
    cheapest_blocks = set()
    for i in range(4):
        start_min = i * 15
        end_min = start_min + 15
        # Hour 8 blocks (cheapest at 1.375 CZK/kWh)
        h8_end = f"08:{end_min:02d}" if end_min < 60 else "09:00"
        cheapest_blocks.add((f"08:{start_min:02d}", h8_end))
        # Hour 2 blocks (second cheapest at 1.5 CZK/kWh)
        h2_end = f"02:{end_min:02d}" if end_min < 60 else "03:00"
        cheapest_blocks.add((f"02:{start_min:02d}", h2_end))

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 8, 0),
        current_price=1.375,  # CZK/kWh
        current_block_key=("08:00", "08:15"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=3.0,  # CZK/kWh
            discharge_price_min=3.5,
            discharge_profit_margin=1.5,
            battery_efficiency=0.85
        )
    )

    # Should charge when in a cheapest block
    decision = decision_engine.decide(context)
    assert decision == "charge_from_grid"


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
    prices_15min.update(create_15min_prices({0: 1.25}))  # 4 blocks (CZK/kWh)
    prices_15min.update(create_15min_prices({1: 1.0}))   # 4 blocks - Cheapest (CZK/kWh)
    prices_15min.update(create_15min_prices({2: 1.5}))   # 4 blocks (CZK/kWh)
    prices_15min.update(create_15min_prices({3: 2.0}))   # 4 blocks (CZK/kWh)
    prices_15min.update(create_15min_prices({4: 2.5}))   # 4 blocks - Most expensive (CZK/kWh)
    prices_15min.update(create_15min_prices({5: 1.75}))  # 4 blocks (CZK/kWh)
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
        0: 2.0,   # CZK/kWh
        1: 1.0,   # Cheapest - should charge (CZK/kWh)
        2: 2.25,  # CZK/kWh
        3: 2.5,   # CZK/kWh
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
        daily_min=1.0,
        daily_max=2.5,
        daily_avg=1.9375,
        daily_median=2.125,
        daily_spread=1.5,
        price_quadrant="Cheapest",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 1, 30),
        current_price=1.0,  # CZK/kWh
        current_block_key=("01:30", "01:45"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=1.2,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        0: 0.75,  # Below threshold (CZK/kWh)
        1: 1.0,   # At threshold (CZK/kWh)
        2: 1.5,   # Above threshold (CZK/kWh)
        3: 2.0,   # Well above threshold (CZK/kWh)
    })

    # Test 1: Price below absolute threshold - should NOT export
    price_ranking = PriceRankingData(
        current_rank=4,
        total_blocks=4,
        percentile=0.0,  # Cheapest hour
        blocks_cheaper_count=3,
        blocks_more_expensive_count=0,
        daily_min=0.75,
        daily_max=2.0,
        daily_avg=1.3125,
        daily_median=1.25,
        daily_spread=1.25,
        price_quadrant="Cheap",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=100.0,  # Full battery to prevent charging decision
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=0.75,  # Below threshold (CZK/kWh)
        current_block_key=("00:30", "00:45"),
        prices_15min=prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=0.5,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
    context.current_price = 1.5  # CZK/kWh
    context.current_time = datetime(2024, 1, 1, 2, 30)

    decision = decision_engine.decide(context)
    decision_engine.explain_decision()

    # Should choose regular mode (export is handled separately)
    assert decision == "regular"
    # Export control is handled separately, not in mode decision


def test_winter_charging_logic(decision_engine: GrowattDecisionEngine) -> None:
    """Test that winter mode only charges during 2 cheapest hours."""
    prices_15min = create_15min_prices({
        0: 0.5,    # Cheapest hour (CZK/kWh)
        1: 0.625,  # Second cheapest (CZK/kWh)
        2: 0.875,  # Third (CZK/kWh)
        3: 1.0,    # Fourth (CZK/kWh)
        4: 1.25,   # Most expensive (CZK/kWh)
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
        daily_min=0.5,
        daily_max=1.25,
        daily_avg=0.85,
        daily_median=0.875,
        daily_spread=0.75,
        price_quadrant="Cheapest",
        is_relatively_cheap=True,
        is_relatively_expensive=False
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,  # Not full
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=0.5,  # CZK/kWh
        current_block_key=("00:30", "00:45"),
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        current_price=0.875,  # CZK/kWh
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.5,
            daily_max=1.25,
            daily_avg=0.85,
            daily_median=0.875,
            daily_spread=0.75,
            price_quadrant="Cheap",
            is_relatively_cheap=True,
            is_relatively_expensive=False
        ),
        is_summer_mode=False
    )

    decision = decision_engine.decide(context)
    # Should NOT charge because rank 3 > 2 cheapest hours (winter logic takes precedence)
    # Export should be disabled because price (0.875) < threshold (1.0)
    assert decision == "regular"  # Export is now always price-based

    # Test 3: Summer mode - should NEVER charge from AC
    # Test with cheapest hour to ensure no charging happens
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 0, 30),
        current_price=0.5,  # Cheapest price (CZK/kWh)
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.5,
            daily_max=1.25,
            daily_avg=0.85,
            daily_median=0.875,
            daily_spread=0.75,
            price_quadrant="Cheapest",
            is_relatively_cheap=True,
            is_relatively_expensive=False
        ),
        is_summer_mode=True  # Summer mode
    )

    decision = decision_engine.decide(context)
    # Should NOT charge in summer mode, even during cheapest hour
    # Export should be disabled because price (0.5) < threshold (1.0)
    assert decision == "regular"  # Export is now always price-based

    # Test 4: Summer mode with moderately expensive hour - still no charging
    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 30),
        current_price=1.0,  # At threshold (CZK/kWh)
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=0.8,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.5,
            daily_max=1.25,
            daily_avg=0.85,
            daily_median=0.875,
            daily_spread=0.75,
            price_quadrant="Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        ),
        is_summer_mode=True  # Summer mode
    )

    decision = decision_engine.decide(context)
    # Should NOT charge in summer mode, regardless of price
    # Export should be enabled because price (1.0) >= threshold (1.0)
    assert decision == "regular"


def test_price_spread_discharge_logic(decision_engine: GrowattDecisionEngine) -> None:
    """Test smart discharge based on price spread relative to daily minimum."""
    # Create a day with significant price variation
    prices_15min = create_15min_prices({
        0: 0.8,   # CZK/kWh
        1: 0.9,   # CZK/kWh
        2: 1.0,   # CZK/kWh
        6: 2.0,   # CZK/kWh
        7: 3.0,   # CZK/kWh - should discharge
        8: 3.5,   # CZK/kWh - should discharge
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
        daily_min=0.8,  # CZK/kWh
        daily_max=3.5,
        daily_avg=1.8667,
        daily_median=1.5,
        daily_spread=2.7,
        price_quadrant="Most Expensive",
        is_relatively_cheap=False,
        is_relatively_expensive=True
    )

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 7, 30),
        current_price=3.0,  # CZK/kWh
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
        current_price=2.0,  # CZK/kWh
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.8,  # CZK/kWh
            daily_max=3.5,
            daily_avg=1.8667,
            daily_median=1.5,
            daily_spread=2.7,
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
        current_price=1.0,  # CZK/kWh
        current_block_key=("02:00", "02:15"),  # Current block
        prices_15min=prices_15min,
        cheapest_blocks=cheapest_blocks,  # Empty - not a charging block
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh (not used with new logic)
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=0.8,
            daily_max=3.5,
            daily_avg=1.8667,
            daily_median=1.5,
            daily_spread=2.7,
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
        0: 1.9,  # CZK/kWh
        1: 2.0,  # CZK/kWh
        2: 2.1,  # CZK/kWh
        3: 2.2,  # CZK/kWh
    })

    context = DecisionContext(
        manual_override_active=False,
        high_loads_active=False,
        battery_soc=50.0,
        current_time=datetime(2024, 1, 1, 3, 30),
        current_price=2.2,  # CZK/kWh
        current_block_key=("03:30", "03:45"),
        prices_15min=flat_prices_15min,
        cheapest_blocks=set(),
        price_thresholds=PriceThresholds(
            charge_price_max=2.0,  # CZK/kWh
            export_price_min=1.0,  # CZK/kWh
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
            daily_min=1.9,  # CZK/kWh
            daily_max=2.2,
            daily_avg=2.05,
            daily_median=2.05,
            daily_spread=0.3,
            price_quadrant="Most Expensive",
            is_relatively_cheap=False,
            is_relatively_expensive=True
        )
    )

    decision = decision_engine.decide(context)
    # Should NOT discharge: 2.2 >= 2.0 threshold BUT 2.2 < 1.9 * 3 = 5.7
    assert decision == "regular"  # No discharge on flat days


# ===== Phase 1 Tests: Summer Mode, Distribution Tariff =====


class TestSummerModeCharging:
    """Tests for summer mode with price-gated charging (Phase 1.1)."""

    def _make_context(self, price_czk_kwh: float, is_summer: bool,
                      in_cheapest: bool = True) -> DecisionContext:
        block_key = ("13:00", "13:15")
        cheapest = {block_key} if in_cheapest else set()
        return DecisionContext(
            manual_override_active=False,
            high_loads_active=False,
            battery_soc=50.0,
            current_time=datetime(2026, 7, 15, 13, 0),
            current_price=price_czk_kwh,
            current_block_key=block_key,
            cheapest_blocks=cheapest,
            is_summer_mode=is_summer,
            price_thresholds=PriceThresholds(
                charge_price_max=1.5,
                export_price_min=1.0,
                discharge_price_min=5.0,
                discharge_profit_margin=4.0,
                battery_efficiency=0.85,
                summer_charge_price_max=0.0,
            ),
        )

    def test_summer_negative_price_charges(self) -> None:
        """Summer + negative price + in cheapest → should charge."""
        ctx = self._make_context(price_czk_kwh=-1.0, is_summer=True)
        # -1.0 CZK/kWh < 0.0 threshold
        assert ctx.is_battery_charging_scheduled is True

    def test_summer_positive_price_no_charge(self) -> None:
        """Summer + positive price → should NOT charge."""
        ctx = self._make_context(price_czk_kwh=2.0, is_summer=True)
        # 2.0 CZK/kWh > 0.0 threshold
        assert ctx.is_battery_charging_scheduled is False

    def test_summer_zero_price_charges(self) -> None:
        """Summer + exactly 0 price → should charge (≤ threshold)."""
        ctx = self._make_context(price_czk_kwh=0.0, is_summer=True)
        assert ctx.is_battery_charging_scheduled is True

    def test_summer_not_in_cheapest_no_charge(self) -> None:
        """Summer + negative price but NOT in cheapest blocks → no charge."""
        ctx = self._make_context(price_czk_kwh=-1.0, is_summer=True, in_cheapest=False)
        assert ctx.is_battery_charging_scheduled is False

    def test_winter_cheapest_block_charges(self) -> None:
        """Winter mode: in cheapest blocks → charges (regression test)."""
        ctx = self._make_context(price_czk_kwh=2.0, is_summer=False)
        assert ctx.is_battery_charging_scheduled is True

    def test_winter_not_in_cheapest_no_charge(self) -> None:
        """Winter mode: not in cheapest → no charge (regression test)."""
        ctx = self._make_context(price_czk_kwh=2.0, is_summer=False, in_cheapest=False)
        assert ctx.is_battery_charging_scheduled is False


class TestDistributionTariff:
    """Tests for distribution tariff logic (Phase 1.2)."""

    def _make_thresholds(self) -> PriceThresholds:
        return PriceThresholds(
            charge_price_max=1.5,
            export_price_min=1.0,
            discharge_price_min=5.0,
            discharge_profit_margin=4.0,
            battery_efficiency=0.85,
            distribution_tariff_high=1.5,
            distribution_tariff_low=0.5,
            low_tariff_hours="0-6,22-24",
        )

    def test_low_tariff_during_night(self) -> None:
        thresholds = self._make_thresholds()
        for hour in [0, 1, 3, 5, 22, 23]:
            rate = GrowattDecisionEngine._get_distribution_tariff(hour, thresholds)
            assert rate == 0.5, f"Hour {hour} should be low tariff"

    def test_high_tariff_during_day(self) -> None:
        thresholds = self._make_thresholds()
        for hour in [6, 7, 12, 18, 21]:
            rate = GrowattDecisionEngine._get_distribution_tariff(hour, thresholds)
            assert rate == 1.5, f"Hour {hour} should be high tariff"

    def test_discharge_blocked_by_self_consumption_value(
        self, decision_engine: GrowattDecisionEngine
    ) -> None:
        """Discharge should be blocked when self-consumption value exceeds spot price."""
        # Spot: 4.0 CZK/kWh, cheapest: 1.0, margin threshold: 4.0
        # Self-consumption floor: 1.0/0.85 + 1.5 = 2.68
        # All thresholds: max(5.0, 4.0, 2.68) = 5.0
        # Current 4.0 < 5.0 → no discharge
        prices = create_15min_prices({h: 1.0 for h in range(24)})  # 1.0 CZK/kWh base
        prices[("18:00", "18:15")] = 4.0  # 4.0 CZK/kWh for current block

        ctx = DecisionContext(
            manual_override_active=False,
            high_loads_active=False,
            battery_soc=80.0,
            current_time=datetime(2026, 4, 11, 18, 0),
            current_price=4.0,  # CZK/kWh
            current_block_key=("18:00", "18:15"),
            prices_15min=prices,
            price_thresholds=self._make_thresholds(),
        )

        decision = decision_engine.decide(ctx)
        assert decision == "regular"  # 4.0 < 5.0, no discharge

    def test_discharge_allowed_above_all_thresholds(
        self, decision_engine: GrowattDecisionEngine
    ) -> None:
        """Discharge allowed when price exceeds ALL thresholds including self-consumption."""
        # Spot: 7.0 CZK/kWh, cheapest: 1.0
        # margin: 1.0 * 4.0 = 4.0
        # self-consumption: 1.0/0.85 + 1.5 = 2.68
        # effective: max(5.0, 4.0, 2.68) = 5.0
        # 7.0 >= 5.0 → discharge
        prices = create_15min_prices({h: 1.0 for h in range(24)})  # CZK/kWh
        prices[("18:00", "18:15")] = 7.0  # 7.0 CZK/kWh

        ctx = DecisionContext(
            manual_override_active=False,
            high_loads_active=False,
            battery_soc=80.0,
            current_time=datetime(2026, 4, 11, 18, 0),
            current_price=7.0,  # CZK/kWh
            current_block_key=("18:00", "18:15"),
            prices_15min=prices,
            price_thresholds=self._make_thresholds(),
        )

        decision = decision_engine.decide(ctx)
        assert decision == "discharge_to_grid"


class TestSellProductionDecision:
    """Tests for the optimizer-scheduled sell_production decision node."""

    def test_sell_production_node_selected(self, decision_engine: GrowattDecisionEngine) -> None:
        """Block flagged by optimizer for sell_production picks that mode."""
        block = ("08:00", "08:15")
        ctx = DecisionContext(
            manual_override_active=False,
            high_loads_active=False,
            battery_soc=70.0,
            current_time=datetime(2026, 4, 11, 8, 0),
            current_price=1.5,
            current_block_key=block,
            optimizer_sell_production_blocks={block},
        )
        assert decision_engine.decide(ctx) == "sell_production"

    def test_sell_production_blocked_by_high_loads(
        self, decision_engine: GrowattDecisionEngine
    ) -> None:
        """High-load protection takes precedence over sell_production."""
        block = ("08:00", "08:15")
        ctx = DecisionContext(
            manual_override_active=False,
            high_loads_active=True,
            battery_soc=70.0,
            current_time=datetime(2026, 4, 11, 8, 0),
            current_price=1.5,
            current_block_key=block,
            optimizer_sell_production_blocks={block},
        )
        assert decision_engine.decide(ctx) == "high_load_protected"

    def test_sell_production_yields_to_manual_override(
        self, decision_engine: GrowattDecisionEngine
    ) -> None:
        """Manual override beats sell_production."""
        block = ("08:00", "08:15")
        ctx = DecisionContext(
            manual_override_active=True,
            manual_override_mode="regular",
            high_loads_active=False,
            battery_soc=70.0,
            current_time=datetime(2026, 4, 11, 8, 0),
            current_price=1.5,
            current_block_key=block,
            optimizer_sell_production_blocks={block},
        )
        assert decision_engine.decide(ctx) == "regular"

    def test_no_sell_production_when_block_not_scheduled(
        self, decision_engine: GrowattDecisionEngine
    ) -> None:
        """When the current block is not in the optimizer set, sell_production does not fire."""
        ctx = DecisionContext(
            manual_override_active=False,
            high_loads_active=False,
            battery_soc=70.0,
            current_time=datetime(2026, 4, 11, 8, 0),
            current_price=1.5,
            current_block_key=("08:00", "08:15"),
            optimizer_sell_production_blocks={("10:00", "10:15")},
        )
        assert decision_engine.decide(ctx) == "regular"
