"""Data models and types for Growatt controller."""

from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from typing import Any, Dict, Literal, Optional, Tuple, Union

PeriodType = Literal[
    "regular", "sell_production", "charge_from_grid", "discharge_to_grid"
]

MODE_PRECEDENCE: Tuple[str, ...] = ("battery_first", "grid_first", "load_first")

EOD_HHMM = "23:59"
EOD_HHMMSS = "23:59:55"
EOD_DTTIME = dt_time(23, 59, 55)

MIDNIGHT_JITTER = "00:00:05"
MIDNIGHT_DISABLE_JITTER = "00:00:10"


@dataclass(frozen=True)
class Period:
    """Represents a scheduled period with proper time types."""
    kind: PeriodType
    start: dt_time
    end: dt_time
    params: Optional[Dict[str, Any]] = None
    manual: bool = False
    source: str = "auto"

    def contains_time(self, t: dt_time) -> bool:
        """Check if a time falls within this period, handling midnight wrap."""
        if self.start <= self.end:
            return self.start <= t < self.end
        else:
            return t >= self.start or t < self.end

    def to_string_tuple(self) -> Tuple[str, str, str]:
        """Convert to legacy string tuple format for logging."""
        return (self.kind, self.start.strftime("%H:%M"), self.end.strftime("%H:%M"))


def is_24(s: str) -> bool:
    """Check if time string represents 24:00 (end of day)."""
    return s in ("24:00", "24:00:00")


def fmt_hhmm(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M")


def fmt_hhmmss(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM:SS string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M:%S")
