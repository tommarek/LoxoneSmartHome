"""Shared scheduling logic for battery charge/discharge optimization.

This module contains pure functions for calculating optimal charge/discharge schedules
that can be used by both the Growatt controller and the web API.
"""

from datetime import datetime
from typing import Dict, List, Set, Tuple


def calculate_optimal_schedule(
    price_blocks: List[Tuple[datetime, float]],
    charge_blocks_count: int = 8,
    discharge_threshold_czk: float = 3.0
) -> Tuple[Set[datetime], Set[datetime], float, float]:
    """Calculate optimal charge/discharge schedule from price blocks.

    Args:
        price_blocks: List of (timestamp, price_czk_kwh) tuples
        charge_blocks_count: Number of cheapest blocks to charge (default: 8 = 2 hours)
        discharge_threshold_czk: Price threshold for discharge in CZK/kWh (default: 3.0)

    Returns:
        Tuple of:
        - Set of timestamps to charge battery
        - Set of timestamps to discharge battery
        - Charge threshold (max price of charging blocks)
        - Discharge threshold (same as input)

    This implements the same logic as GrowattController._calculate_cross_day_optimal_schedule:
    1. Find globally cheapest N blocks across entire window for charging
    2. Mark blocks above discharge_threshold_czk for discharge
    3. Everything else is normal mode
    """
    if not price_blocks:
        return set(), set(), 0.0, discharge_threshold_czk

    # === STEP 1: Find globally cheapest charging blocks ===
    sorted_by_price = sorted(price_blocks, key=lambda x: x[1])
    cheapest_blocks = sorted_by_price[:charge_blocks_count]

    # Extract timestamps for charging
    charge_times = set(timestamp for timestamp, _ in cheapest_blocks)

    # Calculate charge threshold (highest price among charging blocks)
    charge_threshold = max(price for _, price in cheapest_blocks) if cheapest_blocks else 0.0

    # === STEP 2: Find discharge blocks ===
    discharge_times = set(
        timestamp for timestamp, price in price_blocks
        if price >= discharge_threshold_czk
    )

    return charge_times, discharge_times, charge_threshold, discharge_threshold_czk


def determine_block_mode(
    timestamp: datetime,
    price: float,
    charge_times: Set[datetime],
    discharge_times: Set[datetime]
) -> Tuple[str, str]:
    """Determine mode and icon for a single time block.

    Args:
        timestamp: Block timestamp
        price: Block price in CZK/kWh
        charge_times: Set of timestamps marked for charging
        discharge_times: Set of timestamps marked for discharge

    Returns:
        Tuple of (mode, icon) where:
        - mode: "charge", "discharge", or "normal"
        - icon: "🔋", "⚡", or "-"
    """
    if timestamp in charge_times:
        return "charge", "🔋"
    elif timestamp in discharge_times:
        return "discharge", "⚡"
    else:
        return "normal", "-"
