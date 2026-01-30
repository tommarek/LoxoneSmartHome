"""Shared scheduling logic for battery charge/discharge optimization.

This module contains pure functions for calculating optimal charge/discharge schedules
that can be used by both the Growatt controller and the web API.
"""

from datetime import datetime
from typing import Dict, List, Set, Tuple


def calculate_optimal_schedule(
    price_blocks: List[Tuple[datetime, float]],
    charge_blocks_count: int = 8,
    discharge_threshold_czk: float = 3.0,
    discharge_profit_margin: float = 4.0
) -> Tuple[Set[datetime], Set[datetime], float, float]:
    """Calculate optimal charge/discharge schedule from price blocks.

    Args:
        price_blocks: List of (timestamp, price_czk_kwh) tuples
        charge_blocks_count: Number of cheapest blocks to charge (default: 8 = 2 hours)
        discharge_threshold_czk: Absolute minimum price for discharge in CZK/kWh (default: 3.0)
        discharge_profit_margin: Required multiplier over cheapest block (default: 4.0)

    Returns:
        Tuple of:
        - Set of timestamps to charge battery
        - Set of timestamps to discharge battery
        - Charge threshold (max price of charging blocks)
        - Effective discharge threshold used

    Discharge threshold is: max(discharge_threshold_czk, cheapest_block * discharge_profit_margin)
    This matches the runtime decision engine logic in GrowattDecisionEngine._should_discharge_battery.
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
    # Use effective threshold: max(absolute minimum, cheapest block * profit margin)
    cheapest_price = min(price for _, price in price_blocks)
    required_by_margin = cheapest_price * discharge_profit_margin
    effective_threshold = max(discharge_threshold_czk, required_by_margin)

    discharge_times = set(
        timestamp for timestamp, price in price_blocks
        if price >= effective_threshold
    )

    return charge_times, discharge_times, charge_threshold, effective_threshold


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
