"""Inverter state tracking for preventing redundant commands."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class InverterState:
    """Complete inverter configuration state.

    This immutable class represents the complete state of inverter settings,
    allowing comparison to detect actual changes and prevent redundant commands.
    """

    # Core mode settings
    inverter_mode: str  # "load_first", "battery_first", or "grid_first"
    stop_soc: int  # Stop state-of-charge percentage (5-100)
    power_rate: int  # Power rate percentage (10-100)

    # Time window (for battery/grid first modes)
    time_start: str  # Start time in "HH:MM" format
    time_stop: str   # Stop time in "HH:MM" format

    # Additional controls
    ac_charge_enabled: bool  # Whether AC charging from grid is enabled
    export_enabled: bool  # Whether export to grid is enabled

    # Metadata
    timestamp: datetime  # When this state was created
    source: str  # Source of state change: "evaluation", "manual", "schedule", etc.

    def significant_changes(self, other: Optional['InverterState']) -> List[str]:
        """Return list of significant changes that require commands.

        Args:
            other: Previous state to compare against

        Returns:
            List of human-readable change descriptions
        """
        if other is None:
            return ["Initial state configuration"]

        changes = []

        # Check mode changes
        if self.inverter_mode != other.inverter_mode:
            changes.append(f"Mode: {other.inverter_mode} → {self.inverter_mode}")

        # Check SOC changes
        if self.stop_soc != other.stop_soc:
            changes.append(f"Stop SOC: {other.stop_soc}% → {self.stop_soc}%")

        # Check power rate changes
        if self.power_rate != other.power_rate:
            changes.append(f"Power rate: {other.power_rate}% → {self.power_rate}%")

        # Check time window changes (only relevant for battery/grid first)
        if self.inverter_mode in ["battery_first", "grid_first"]:
            if self.time_start != other.time_start or self.time_stop != other.time_stop:
                changes.append(
                    f"Time window: {other.time_start}-{other.time_stop} → "
                    f"{self.time_start}-{self.time_stop}"
                )

        # Check AC charging changes
        if self.ac_charge_enabled != other.ac_charge_enabled:
            ac_state = "enabled" if self.ac_charge_enabled else "disabled"
            changes.append(f"AC charging: {ac_state}")

        # Check export changes
        if self.export_enabled != other.export_enabled:
            export_state = "enabled" if self.export_enabled else "disabled"
            changes.append(f"Export: {export_state}")

        return changes

    def mode_params_changed(self, other: Optional['InverterState']) -> bool:
        """Check if core mode parameters changed (requiring mode command).

        Args:
            other: Previous state to compare against

        Returns:
            True if mode-related parameters changed
        """
        if other is None:
            return True

        # Any of these changes require sending mode command
        return (
            self.inverter_mode != other.inverter_mode
            or self.stop_soc != other.stop_soc
            or self.power_rate != other.power_rate
            or self.time_start != other.time_start
            or self.time_stop != other.time_stop
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for JSON storage/API responses.

        Returns:
            Dictionary representation of the state
        """
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InverterState':
        """Create InverterState from dictionary.

        Args:
            data: Dictionary containing state data

        Returns:
            InverterState instance
        """
        # Convert timestamp string back to datetime
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def summary(self) -> str:
        """Get a concise summary of the state for logging.

        Returns:
            Single-line summary string
        """
        mode_str = f"{self.inverter_mode}@{self.stop_soc}%"

        if self.inverter_mode in ["battery_first", "grid_first"]:
            mode_str += f" {self.time_start}-{self.time_stop}"
            if self.power_rate != 100:
                mode_str += f" {self.power_rate}%pwr"

        export_str = "EXP:ON" if self.export_enabled else "EXP:OFF"
        ac_str = "AC:ON" if self.ac_charge_enabled else "AC:OFF"

        return f"{mode_str} {export_str} {ac_str}"
