"""Decision tree engine for Growatt mode selection.

This module implements a priority-based decision tree for determining which
mode the Growatt inverter should be in based on various conditions like
manual overrides, high loads, scheduled periods, etc.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Union
import logging


class Priority(IntEnum):
    """Priority levels for decision rules (lower number = higher priority)."""
    MANUAL_OVERRIDE = 1         # Highest priority
    SCHEDULED_BATTERY_CHARGING = 2
    HIGH_LOAD_PROTECTION = 3
    SCHEDULED_MODE = 4
    DEFAULT_MODE = 5            # Lowest priority


@dataclass
class DecisionContext:
    """Context information for making mode decisions."""
    manual_override_active: bool
    manual_override_mode: Optional[str]
    high_loads_active: bool
    scheduled_mode: Optional[str]
    battery_soc: float
    current_mode: Optional[str]
    is_battery_charging_scheduled: bool = False  # True if charge_from_grid is scheduled now

    def __post_init__(self):
        """Derive additional context after initialization."""
        # Check if the scheduled mode is battery charging
        if self.scheduled_mode == "charge_from_grid":
            self.is_battery_charging_scheduled = True


@dataclass
class DecisionNode:
    """A single node in the decision tree."""
    name: str
    priority: Priority
    condition: Callable[[DecisionContext], bool]
    action: Union[str, Callable[[DecisionContext], str]]
    explanation: str
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
MODE_DEFINITIONS = {
    "regular": {
        "description": "Normal operation - load first with export",
        "inverter_mode": "load_first",
        "stop_soc": 20,  # Default minimum SOC
        "export": True,
        "ac_charge": False
    },
    "regular_no_export": {
        "description": "Normal operation without export",
        "inverter_mode": "load_first",
        "stop_soc": 20,
        "export": False,
        "ac_charge": False
    },
    "high_load_protected": {
        "description": "High load protection - prevent battery discharge",
        "inverter_mode": "load_first",
        "stop_soc": 100,  # Prevent any discharge
        "export": True,   # Allow solar export if available
        "ac_charge": False
    },
    "charge_from_grid": {
        "description": "Charging battery from grid",
        "inverter_mode": "battery_first",
        "stop_soc": "configurable",  # From parameters
        "export": True,   # Allow excess solar to be exported
        "ac_charge": True
    },
    "discharge_to_grid": {
        "description": "Discharging battery to grid",
        "inverter_mode": "grid_first",
        "stop_soc": "configurable",  # From parameters
        "export": True,
        "ac_charge": False
    },
    "sell_production": {
        "description": "Sell only solar production",
        "inverter_mode": "grid_first",
        "stop_soc": 100,  # Don't discharge battery
        "export": True,
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

            # Priority 2: Scheduled Battery Charging - Overrides high load protection
            # This ensures we can charge during cheap hours even with high loads
            DecisionNode(
                name="Scheduled Battery Charging",
                priority=Priority.SCHEDULED_BATTERY_CHARGING,
                condition=lambda ctx: ctx.is_battery_charging_scheduled,
                action="charge_from_grid",
                explanation="Battery charging period active - overrides high load protection"
            ),

            # Priority 3: High Load Protection - Prevent battery discharge
            # Only applies when NOT charging from grid
            DecisionNode(
                name="High Load Protection",
                priority=Priority.HIGH_LOAD_PROTECTION,
                condition=lambda ctx: (
                    ctx.high_loads_active and not ctx.is_battery_charging_scheduled
                ),
                action="high_load_protected",
                explanation=(
                    "High loads detected - preventing battery discharge with load-first @ 100% SOC"
                )
            ),

            # Priority 4: Other Scheduled Modes
            DecisionNode(
                name="Scheduled Mode",
                priority=Priority.SCHEDULED_MODE,
                condition=lambda ctx: ctx.scheduled_mode is not None,
                action=lambda ctx: ctx.scheduled_mode,
                explanation="Applying scheduled mode for current time period"
            ),

            # Priority 5: Default Mode - No special conditions
            DecisionNode(
                name="Default Mode",
                priority=Priority.DEFAULT_MODE,
                condition=lambda ctx: True,  # Always matches as fallback
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

        return {
            "decision": mode,
            "reason": node.explanation,
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
                "scheduled_mode": ctx.scheduled_mode,
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
                      scheduled_mode: Optional[str] = None,
                      battery_soc: float = 50.0) -> Dict[str, Any]:
        """Test a specific scenario for debugging.

        Args:
            manual_override: Whether manual override is active
            high_loads: Whether high loads are active
            scheduled_mode: Current scheduled mode
            battery_soc: Current battery SOC

        Returns:
            Decision explanation for the scenario
        """
        context = DecisionContext(
            manual_override_active=manual_override,
            manual_override_mode="regular" if manual_override else None,
            high_loads_active=high_loads,
            scheduled_mode=scheduled_mode,
            battery_soc=battery_soc,
            current_mode=None
        )

        self.decide(context)
        explanation = self.explain_decision()

        return {
            "scenario": {
                "manual_override": manual_override,
                "high_loads": high_loads,
                "scheduled_mode": scheduled_mode,
                "battery_soc": battery_soc
            },
            "result": explanation
        }
