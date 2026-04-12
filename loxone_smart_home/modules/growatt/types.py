"""Type definitions and constants for Growatt controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dt_time
from enum import Enum
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable


class GrowattLogLevel(Enum):
    """Logging levels for Growatt controller with progressive detail."""

    SUMMARY = "SUMMARY"  # High-level state changes only
    DETAIL = "DETAIL"    # Include price summaries and key decisions
    VERBOSE = "VERBOSE"  # Full price tables and debug info (current behavior)
    DEBUG = "DEBUG"      # All debug messages including raw data

    def __ge__(self, other: "GrowattLogLevel") -> bool:
        """Allow comparison of log levels."""
        order = [GrowattLogLevel.SUMMARY, GrowattLogLevel.DETAIL,
                 GrowattLogLevel.VERBOSE, GrowattLogLevel.DEBUG]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: "GrowattLogLevel") -> bool:
        """Allow comparison of log levels."""
        order = [GrowattLogLevel.SUMMARY, GrowattLogLevel.DETAIL,
                 GrowattLogLevel.VERBOSE, GrowattLogLevel.DEBUG]
        return order.index(self) > order.index(other)


# Type definitions
PeriodType = Literal[
    "battery_first", "grid_first", "load_first", "ac_charge", "export"
]

# Mode precedence (highest → lowest).
# load_first is the fallback/default if none match.
MODE_PRECEDENCE: Tuple[PeriodType, ...] = (
    "battery_first", "grid_first", "load_first"
)

# Use a single end-of-day barrier everywhere
EOD_HHMM = "23:59"
EOD_HHMMSS = "23:59:55"
EOD_DTTIME = dt_time(23, 59, 55)

# Common time jitter constants for readability
MIDNIGHT_JITTER = "00:00:05"  # 5 seconds after midnight
# 10 seconds after midnight for disables
MIDNIGHT_DISABLE_JITTER = "00:00:10"


@dataclass(frozen=True)
class Period:
    """Represents a scheduled period with proper time types."""

    kind: PeriodType
    start: dt_time
    end: dt_time
    params: Optional[Dict[str, Any]] = None

    def contains_time(self, t: dt_time) -> bool:
        """Check if a time falls within this period, handling midnight wrap."""
        if self.start <= self.end:
            # Normal case: period doesn't wrap midnight
            return self.start <= t <= self.end
        else:
            # Wrap case: period spans midnight
            return t >= self.start or t <= self.end


@runtime_checkable
class InverterAdapter(Protocol):
    """Interface for inverter communication methods needed by ModeManager.

    Decouples ModeManager from GrowattController, enabling independent testing.
    The controller implements this protocol implicitly via structural typing.
    """

    def _to_device_hhmm(self, s: str) -> str:
        """Convert time string to HH:MM format required by device."""
        ...

    def _ensure_future_start(
        self,
        start_str: str,
        stop_str: str,
        min_future_minutes: int = 1,
        preserve_duration: bool = False,
        immediate_activation: bool = False,
    ) -> Tuple[str, str]:
        """Ensure start time is appropriate for inverter scheduling."""
        ...

    async def _wait_for_command_result(
        self, command_type: str, timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for command result via command queue."""
        ...

    async def _query_inverter_state(self) -> Dict[str, Any]:
        """Query current inverter state."""
        ...
