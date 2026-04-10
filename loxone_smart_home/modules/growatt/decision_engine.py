"""Decision tree engine for Growatt mode selection.

This module implements a priority-based decision tree for determining which
mode the Growatt inverter should be in based on various conditions like
manual overrides, high loads, scheduled periods, etc.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging
import statistics


class Priority(IntEnum):
    """Priority levels for decision rules (lower number = higher priority)."""
    MANUAL_OVERRIDE = 1         # Highest priority
    SCHEDULED_BATTERY_CHARGING = 2
    HIGH_LOAD_PROTECTION = 3
    SCHEDULED_MODE = 4
    DEFAULT_MODE = 5            # Lowest priority


@dataclass
class PriceThresholds:
    """Simple price thresholds for decision making (all in CZK/kWh)."""
    charge_price_max: float  # Charge when price below this
    export_price_min: float  # Export solar when price above this
    discharge_price_min: float  # Discharge battery when price above this
    discharge_profit_margin: float  # Required profit margin (e.g., 1.5 = 50% profit)
    battery_efficiency: float  # Battery round-trip efficiency
    target_soc: float = 100.0  # Target SOC when charging from grid
    summer_charge_price_max: float = 0.0  # Max CZK/kWh to charge in summer (0 = only negative)
    distribution_tariff_high: float = 1.5  # High tariff distribution cost CZK/kWh
    distribution_tariff_low: float = 0.5  # Low tariff distribution cost CZK/kWh
    low_tariff_hours: str = "0-6,22-24"  # Comma-separated hour ranges for low tariff


@dataclass
class PriceRankingData:
    """Price ranking information for current 15-minute block within daily context."""
    current_rank: int  # 1 = cheapest block of day
    total_blocks: int  # Total 15-minute blocks in price data (96 for full day)
    percentile: float  # 0-100, lower = cheaper
    blocks_cheaper_count: int  # How many blocks are cheaper
    blocks_more_expensive_count: int  # How many blocks are more expensive
    daily_min: float  # Cheapest price today
    daily_max: float  # Most expensive price today
    daily_avg: float  # Average price today
    daily_median: float  # Median price today
    daily_spread: float  # Max - Min price spread
    price_quadrant: str  # "Cheapest", "Cheap", "Expensive", "Most Expensive"
    is_relatively_cheap: bool = False  # True if in cheaper half of day
    is_relatively_expensive: bool = False  # True if in expensive half of day


@dataclass
class DecisionContext:
    """Context information for making mode decisions."""
    # Required fields (no defaults)
    manual_override_active: bool
    high_loads_active: bool
    battery_soc: float
    current_time: datetime

    # Optional fields (with defaults)
    manual_override_mode: Optional[str] = None
    current_mode: Optional[str] = None
    current_load: float = 0.0  # Current home load in kW
    solar_power: float = 0.0  # Current solar generation in kW
    current_price: float = 0.0  # Current 15-minute block price EUR/MWh
    current_block_key: Optional[Tuple[str, str]] = None  # Current 15-minute block
    prices_15min: Dict[Tuple[str, str], float] = field(default_factory=dict)  # 96 15-min blocks
    # Set of cheapest charging blocks
    cheapest_blocks: Set[Tuple[str, str]] = field(default_factory=set)
    price_thresholds: Optional[PriceThresholds] = None
    price_ranking: Optional[PriceRankingData] = None  # Price ranking within day
    sunrise: Optional[time] = None
    sunset: Optional[time] = None
    is_summer_mode: bool = False
    is_battery_charging_scheduled: bool = False

    def __post_init__(self) -> None:
        """Derive additional context after initialization."""
        if self.is_summer_mode:
            # Summer: only charge from grid when price is below summer threshold
            # (e.g., negative prices = get paid to consume)
            if self.current_block_key and self.cheapest_blocks:
                in_cheapest = self.current_block_key in self.cheapest_blocks
                current_czk = self.current_price * 25 / 1000  # EUR/MWh to CZK/kWh
                max_price = (
                    self.price_thresholds.summer_charge_price_max
                    if self.price_thresholds else 0.0
                )
                self.is_battery_charging_scheduled = in_cheapest and current_czk <= max_price
            else:
                self.is_battery_charging_scheduled = False
        # Check if current block is in the cheapest charging blocks
        elif self.current_block_key and self.cheapest_blocks:
            self.is_battery_charging_scheduled = (
                self.current_block_key in self.cheapest_blocks
            )
        # No fallback to price threshold - only charge during cheapest blocks
        else:
            self.is_battery_charging_scheduled = False


@dataclass
class DecisionNode:
    """A single node in the decision tree."""
    name: str
    priority: Priority
    condition: Callable[[DecisionContext], bool]
    action: Union[str, Callable[[DecisionContext], str]]
    explanation: Union[str, Callable[[DecisionContext], str]]
    stop_on_match: bool = True

    def evaluate(self, context: DecisionContext) -> Optional[str]:
        """Evaluate this node against the context.

        Returns:
            The mode to apply if condition matches, None otherwise
        """
        if self.condition(context):
            if callable(self.action):
                return self.action(context)
            return self.action
        return None

    def get_action_str(self, context: DecisionContext) -> str:
        """Get the action as a string for logging."""
        if callable(self.action):
            return self.action(context)
        return self.action


# Mode definitions - what each mode means in terms of inverter settings
# NOTE: Export control is ALWAYS price-based, not mode-dependent
MODE_DEFINITIONS = {
    "regular": {
        "description": "Normal operation - load first",
        "inverter_mode": "load_first",
        "stop_soc": 20,  # Default minimum SOC
        "ac_charge": False
    },
    "high_load_protected": {
        "description": "High load protection - prevent battery discharge",
        "inverter_mode": "load_first",
        "stop_soc": 100,  # Prevent any discharge
        "ac_charge": False
    },
    "charge_from_grid": {
        "description": "Charging battery from grid to 100%",
        "inverter_mode": "battery_first",
        "stop_soc": 100,  # Charge to full
        "ac_charge": True
    },
    "battery_first_ac_charge": {
        "description": "Battery-first with AC charging (high load + cheap price)",
        "inverter_mode": "battery_first",
        "stop_soc": 100,  # Charge to full and prevent discharge
        "ac_charge": True,
        "high_load_compatible": True  # Can run with high loads
    },
    "discharge_to_grid": {
        "description": "Discharging battery to grid",
        "inverter_mode": "grid_first",
        "stop_soc": "configurable",  # From parameters
        "ac_charge": False
    },
    "sell_production": {
        "description": "Sell only solar production",
        "inverter_mode": "grid_first",
        "stop_soc": 100,  # Don't discharge battery
        "ac_charge": False
    }
}


class GrowattDecisionEngine:
    """Decision tree engine for Growatt mode selection."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the decision engine.

        Args:
            logger: Optional logger for debugging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.decision_tree = self._build_decision_tree()
        self._last_decision: Optional[Dict[str, Any]] = None
        self._last_mode: Optional[str] = None  # Track last mode to detect changes

    def _build_decision_tree(self) -> List[DecisionNode]:
        """Build the priority-ordered decision tree.

        Returns:
            List of decision nodes in priority order
        """
        return [
            # Priority 1: Manual Override - User takes full control
            DecisionNode(
                name="Manual Override",
                priority=Priority.MANUAL_OVERRIDE,
                condition=lambda ctx: ctx.manual_override_active,
                action=lambda ctx: ctx.manual_override_mode or "regular",
                explanation="User manual override is active"
            ),

            # Priority 2A: Cheap Hour + High Load - Battery-first with AC charging
            DecisionNode(
                name="High Load Battery Charging",
                priority=Priority.SCHEDULED_BATTERY_CHARGING,
                condition=lambda ctx: (
                    ctx.high_loads_active
                    and ctx.battery_soc < 100
                    and ctx.is_battery_charging_scheduled
                ),
                action="battery_first_ac_charge",
                explanation=(
                    "Scheduled charging hour with high loads - "
                    "battery-first mode with AC charging"
                )
            ),

            # Priority 2B: Cheap Hour Charging - Charge battery during cheapest blocks
            DecisionNode(
                name="Scheduled Battery Charging",
                priority=Priority.SCHEDULED_BATTERY_CHARGING,
                condition=lambda ctx: (
                    ctx.battery_soc < 100
                    and ctx.is_battery_charging_scheduled
                ),
                action="charge_from_grid",
                explanation="Cheapest electricity blocks - charging battery to 100%"
            ),

            # Priority 3: High Load Protection - Prevent ALL battery discharge
            DecisionNode(
                name="High Load Protection",
                priority=Priority.HIGH_LOAD_PROTECTION,
                condition=lambda ctx: (
                    ctx.high_loads_active
                    and not ctx.is_battery_charging_scheduled
                ),
                action="high_load_protected",
                explanation=(
                    "High loads active - load-first mode with 100% stop SOC (no discharge)"
                )
            ),

            # Priority 4: Check for battery discharge opportunity
            DecisionNode(
                name="Battery Discharge Control",
                priority=Priority.SCHEDULED_MODE,
                condition=lambda ctx: self._should_discharge_battery(ctx),
                action="discharge_to_grid",
                explanation=lambda ctx: (
                    f"Battery discharge profitable: "
                    f"{ctx.current_price:.1f} EUR/MWh meets spread requirement, "
                    f"discharging at 25% power"
                )
            ),

            # Priority 5: Default Mode - No special conditions
            DecisionNode(
                name="Default Mode",
                priority=Priority.DEFAULT_MODE,
                condition=lambda _: True,  # Always matches as fallback
                action="regular",
                explanation="No special conditions - regular operation"
            )
        ]

    def decide(self, context: DecisionContext) -> str:
        """Make a decision based on the current context.

        Args:
            context: Current system context

        Returns:
            The mode to apply
        """
        for node in self.decision_tree:
            mode = node.evaluate(context)
            if mode is not None:
                # Store the decision for explanation
                self._last_decision = {
                    "mode": mode,
                    "node": node,
                    "context": context
                }

                # Log the decision
                self.logger.debug(
                    f"Decision: {mode} (Priority {node.priority}: {node.name})"
                )

                return mode

        # Should never reach here due to default fallback
        self.logger.error("No decision node matched! Using regular mode as emergency fallback")
        return "regular"

    def explain_decision(self, context: Optional[DecisionContext] = None) -> Dict[str, Any]:
        """Explain the most recent decision or evaluate a new context.

        Args:
            context: Optional context to evaluate. If None, explains last decision.

        Returns:
            Dictionary with decision explanation
        """
        if context is not None:
            # Make a new decision to explain
            mode = self.decide(context)

        if self._last_decision is None:
            return {
                "decision": "unknown",
                "reason": "No decision has been made yet",
                "priority": None
            }

        node = self._last_decision["node"]
        mode = self._last_decision["mode"]
        ctx = self._last_decision["context"]

        mode_def = MODE_DEFINITIONS.get(mode, {})

        # Get explanation (may be callable)
        explanation_text = node.explanation
        if callable(node.explanation):
            explanation_text = node.explanation(ctx)

        return {
            "decision": mode,
            "reason": explanation_text,
            "priority": {
                "level": node.priority,
                "name": node.priority.name
            },
            "mode_details": {
                "description": mode_def.get("description", "Unknown mode"),
                "inverter_mode": mode_def.get("inverter_mode"),
                "stop_soc": mode_def.get("stop_soc"),
                "export": mode_def.get("export"),
                "ac_charge": mode_def.get("ac_charge")
            },
            "context": {
                "manual_override": ctx.manual_override_active,
                "high_loads": ctx.high_loads_active,
                "battery_soc": ctx.battery_soc,
                "battery_charging_scheduled": ctx.is_battery_charging_scheduled
            }
        }

    def get_mode_definition(self, mode: str) -> Dict[str, Any]:
        """Get the definition of a specific mode.

        Args:
            mode: Mode name

        Returns:
            Mode definition dictionary
        """
        return MODE_DEFINITIONS.get(mode, {})

    def visualize_tree(self) -> str:
        """Generate a text visualization of the decision tree.

        Returns:
            String representation of the tree
        """
        lines = ["Decision Tree (Priority Order):"]
        lines.append("=" * 60)

        for node in self.decision_tree:
            indent = "  " * (node.priority - 1)
            lines.append(f"{indent}{node.priority}. {node.name}")
            lines.append(f"{indent}   → {node.explanation}")
            if callable(node.action):
                lines.append(f"{indent}   Action: Dynamic (based on context)")
            else:
                lines.append(f"{indent}   Action: {node.action}")
            lines.append("")

        return "\n".join(lines)

    def test_scenario(self,
                      manual_override: bool = False,
                      high_loads: bool = False,
                      battery_soc: float = 50.0) -> Dict[str, Any]:
        """Test a specific scenario for debugging.

        Args:
            manual_override: Whether manual override is active
            high_loads: Whether high loads are active
            battery_soc: Current battery SOC

        Returns:
            Decision explanation for the scenario
        """
        context = DecisionContext(
            manual_override_active=manual_override,
            high_loads_active=high_loads,
            battery_soc=battery_soc,
            current_time=datetime.now(),
            manual_override_mode="regular" if manual_override else None,
            current_mode=None
        )

        self.decide(context)
        explanation = self.explain_decision()

        return {
            "scenario": {
                "manual_override": manual_override,
                "high_loads": high_loads,
                "battery_soc": battery_soc
            },
            "result": explanation
        }

    # Price-aware decision helper methods

    @staticmethod
    def _get_distribution_tariff(hour: int, thresholds: PriceThresholds) -> float:
        """Get the distribution tariff for a given hour based on low/high tariff schedule.

        Args:
            hour: Hour of day (0-23)
            thresholds: Price thresholds containing tariff config

        Returns:
            Distribution tariff in CZK/kWh
        """
        for range_str in thresholds.low_tariff_hours.split(","):
            range_str = range_str.strip()
            if "-" in range_str:
                parts = range_str.split("-")
                start_h, end_h = int(parts[0]), int(parts[1])
                if start_h <= hour < end_h:
                    return thresholds.distribution_tariff_low
        return thresholds.distribution_tariff_high

    def _should_discharge_battery(self, context: DecisionContext) -> bool:
        """Determine if battery should discharge based on simple price thresholds.

        Discharge only when:
        1. Battery SOC > 20%
        2. No high loads active
        3. Current price meets BOTH requirements:
           a) Above absolute minimum (discharge_price_min)
           b) At least N× the cheapest hour (discharge_profit_margin)

        The effective threshold is: max(discharge_price_min, cheapest_hour × margin)

        Args:
            context: Current decision context

        Returns:
            True if battery should discharge to grid
        """
        # Don't discharge if battery is too low
        if context.battery_soc <= 20:
            return False

        # Don't discharge during high loads
        if context.high_loads_active:
            return False

        # No price data means no price-based discharging
        if not context.price_thresholds or context.current_price <= 0:
            return False

        # Convert current price to CZK/kWh (EUR/MWh * 25 / 1000)
        current_price_czk = context.current_price * 25 / 1000

        # Find the absolute cheapest 15-minute block price
        if not context.prices_15min:
            return False

        cheapest_block_price = min(context.prices_15min.values())
        cheapest_block_czk = cheapest_block_price * 25 / 1000

        # Calculate required price based on profit margin
        required_by_margin = cheapest_block_czk * context.price_thresholds.discharge_profit_margin

        # Self-consumption opportunity cost: if we discharge now, we need to
        # recharge later (at cheapest price + its distribution tariff) and we
        # lose the distribution tariff savings from self-consuming now
        current_hour = context.current_time.hour
        current_dist = self._get_distribution_tariff(current_hour, context.price_thresholds)
        # Cost to recharge 1 kWh (accounting for round-trip efficiency)
        recharge_cost = cheapest_block_czk / context.price_thresholds.battery_efficiency
        self_consumption_floor = recharge_cost + current_dist

        # Effective threshold is the highest of all requirements
        effective_threshold = max(
            context.price_thresholds.discharge_price_min,
            required_by_margin,
            self_consumption_floor
        )

        if current_price_czk >= effective_threshold:
            self.logger.info(
                f"Discharge profitable: {current_price_czk:.2f} CZK/kWh ≥ "
                f"{effective_threshold:.2f} CZK/kWh "
                f"(absolute min: {context.price_thresholds.discharge_price_min:.2f}, "
                f"margin: {cheapest_block_czk:.2f} × "
                f"{context.price_thresholds.discharge_profit_margin:.1f} = "
                f"{required_by_margin:.2f}, "
                f"self-consumption: {self_consumption_floor:.2f} "
                f"[recharge {recharge_cost:.2f} + dist {current_dist:.2f}])"
            )
            return True
        else:
            self.logger.debug(
                f"No discharge: {current_price_czk:.2f} CZK/kWh < "
                f"{effective_threshold:.2f} CZK/kWh required "
                f"(margin: {required_by_margin:.2f}, "
                f"self-consumption: {self_consumption_floor:.2f})"
            )
            return False

    def _is_hour_in_range(self, hour: str, start: str, end: str) -> bool:
        """Check if an hour is within a time range.

        Args:
            hour: Hour to check (HH:MM)
            start: Range start (HH:MM)
            end: Range end (HH:MM)

        Returns:
            True if hour is in range
        """
        try:
            # Handle both HH:MM and HH:00 formats
            h_time = datetime.strptime(hour[:5], "%H:%M").time()
            s_time = datetime.strptime(start[:5], "%H:%M").time()
            e_time = datetime.strptime(end[:5], "%H:%M").time()

            if s_time <= e_time:
                return s_time <= h_time < e_time
            else:
                # Handles overnight ranges
                return h_time >= s_time or h_time < e_time
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Invalid time format in range check: {e}")
            return False

    def has_mode_changed(self, new_mode: str) -> bool:
        """Check if the mode has changed from the last decision.

        Args:
            new_mode: The newly decided mode

        Returns:
            True if mode has changed
        """
        changed = self._last_mode != new_mode
        self._last_mode = new_mode
        return changed

    def _validate_price_data(self, hourly_prices: Dict[Tuple[str, str], float]) -> bool:
        """Validate price data integrity.

        Args:
            hourly_prices: Hour price mapping to validate

        Returns:
            True if data is valid
        """
        if not hourly_prices:
            return False

        # Check for reasonable price range (0-1000 EUR/MWh)
        for price in hourly_prices.values():
            if not 0 <= price <= 1000:
                self.logger.warning(f"Suspicious price detected: {price} EUR/MWh")
                return False

        return True

    def calculate_price_ranking(
        self,
        current_block_key: Tuple[str, str],
        prices_15min: Dict[Tuple[str, str], float]
    ) -> Optional[PriceRankingData]:
        """Calculate price ranking data for current 15-minute block.

        Args:
            current_block_key: Current 15-minute block key (start, end)
            prices_15min: All 15-minute prices for the day (96 blocks)

        Returns:
            PriceRankingData or None if insufficient data
        """
        if not prices_15min or current_block_key not in prices_15min:
            return None

        current_price = prices_15min[current_block_key]
        all_prices = list(prices_15min.values())

        # Calculate statistics
        daily_min = min(all_prices)
        daily_max = max(all_prices)
        daily_avg = sum(all_prices) / len(all_prices)
        daily_median = statistics.median(all_prices)
        daily_spread = daily_max - daily_min

        # Calculate rank (1 = cheapest)
        sorted_prices = sorted(all_prices)
        current_rank = sorted_prices.index(current_price) + 1

        # Handle duplicate prices by finding first occurrence
        for i, price in enumerate(sorted_prices):
            if abs(price - current_price) < 0.001:  # Float comparison tolerance
                current_rank = i + 1
                break

        # Calculate counts
        blocks_cheaper_count = current_rank - 1
        blocks_more_expensive_count = len(all_prices) - current_rank

        # Calculate percentile (0 = cheapest, 100 = most expensive)
        percentile = (current_rank - 1) / max(1, len(all_prices) - 1) * 100

        # Determine price quadrant
        if percentile <= 25:
            price_quadrant = "Cheapest"
        elif percentile <= 50:
            price_quadrant = "Cheap"
        elif percentile <= 75:
            price_quadrant = "Expensive"
        else:
            price_quadrant = "Most Expensive"

        return PriceRankingData(
            current_rank=current_rank,
            total_blocks=len(all_prices),
            percentile=percentile,
            blocks_cheaper_count=blocks_cheaper_count,
            blocks_more_expensive_count=blocks_more_expensive_count,
            daily_min=daily_min,
            daily_max=daily_max,
            daily_avg=daily_avg,
            daily_median=daily_median,
            daily_spread=daily_spread,
            price_quadrant=price_quadrant,
            is_relatively_cheap=percentile <= 50,
            is_relatively_expensive=percentile > 50
        )
