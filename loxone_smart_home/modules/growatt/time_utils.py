"""Time utilities for Growatt controller."""

from datetime import datetime, time as dt_time, timedelta
from typing import Union

from .types import EOD_HHMM, EOD_HHMMSS


def is_24(s: str) -> bool:
    """Check if time string represents 24:00 (end of day)."""
    return s in ("24:00", "24:00:00")


def fmt_hhmm(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M")


def fmt_hhmmss(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM:SS string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M:%S")


def parse_time_any(s: str) -> dt_time:
    """Accept HH:MM, HH:MM:SS, and '24:00' => 00:00 (next-day semantics)."""
    if is_24(s):
        return dt_time(0, 0)
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    raise ValueError(f"Invalid time string: {s}")


def parse_hhmm(s: str) -> dt_time:
    """Parse HH:MM or HH:MM:SS string to time object."""
    return parse_time_any(s)


def normalize_end_time(s: str) -> str:
    """Only for device/scheduler emission.

    Collapse '24:00' to end-of-day barrier.
    """
    return EOD_HHMMSS if is_24(s) else s


def normalize_for_schedule(s: str) -> str:
    """Normalize any time string for scheduling calls.
    Converts 24:00 to EOD_HHMMSS and ensures HH:MM:SS format.
    """
    if is_24(s):
        return EOD_HHMMSS
    # Normalize HH:MM to HH:MM:SS for consistency
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M:%S")
        except ValueError:
            pass
    # If we can't parse it, raise an error
    raise ValueError(f"Invalid time string for schedule: {s}")


def bump_time(hhmm: str, seconds: int) -> str:
    """Add seconds to a time string systematically.

    Args:
        hhmm: Time in HH:MM or HH:MM:SS format
        seconds: Number of seconds to add

    Returns:
        Time string in HH:MM:SS format
    """
    try:
        t = datetime.strptime(hhmm, "%H:%M:%S")
    except ValueError:
        t = datetime.strptime(hhmm, "%H:%M")
    t = (t + timedelta(seconds=seconds)).time()
    return t.strftime("%H:%M:%S")


def to_device_hhmm(s: str) -> str:
    """Convert time string to HH:MM format required by device.

    The firmware strictly requires HH:MM format (no seconds) for
    batteryfirst/set/timeslot and gridfirst/set/timeslot commands.
    """
    # Handle EOD sentinels and 24:00
    if s in ("24:00", "24:00:00", EOD_HHMMSS):
        return EOD_HHMM  # "23:59"
    # Truncate to HH:MM if longer
    return s[:5] if len(s) >= 5 else s
