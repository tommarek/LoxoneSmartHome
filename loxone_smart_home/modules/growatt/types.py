"""Type definitions and constants for Growatt controller."""

from dataclasses import dataclass
from datetime import time as dt_time
from typing import Any, Dict, Literal, Optional, Tuple


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

    def to_string_tuple(self) -> Tuple[str, str, str]:
        """Convert to string tuple for logging."""
        return (
            self.kind,
            self.start.strftime("%H:%M"),
            self.end.strftime("%H:%M"),
        )
