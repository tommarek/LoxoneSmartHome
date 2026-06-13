"""Shared scheduling helper.

`calculate_dynamic_block_count` sizes the charge-window block count from the
price spread; used by the controller. (The greedy `calculate_optimal_schedule`
and `determine_block_mode` were removed when the system went MILP-only.)
"""

from typing import List


def calculate_dynamic_block_count(
    prices_czk: List[float],
    min_blocks: int = 4,
    max_blocks: int = 16,
    self_consumption_value: float = 2.5,
    battery_efficiency: float = 0.85,
) -> int:
    """Calculate how many blocks to charge based on price profitability.

    For each block (sorted cheapest first), include it if charging at that
    price and later self-consuming saves money compared to buying at
    self_consumption_value. Accounts for round-trip efficiency loss.

    Args:
        prices_czk: All block prices in CZK/kWh
        min_blocks: Minimum blocks to charge (floor)
        max_blocks: Maximum blocks to charge (ceiling)
        self_consumption_value: Average value of self-consumed kWh (CZK/kWh)
            This is the avoided cost: avg spot price during consumption + distribution tariff
        battery_efficiency: Round-trip efficiency (0.85 = 15% loss)

    Returns:
        Number of blocks to charge
    """
    if not prices_czk:
        return min_blocks

    sorted_prices = sorted(prices_czk)
    count = 0
    for price in sorted_prices:
        # Net value: what we save by charging now and self-consuming later
        # minus the efficiency loss
        net_value = self_consumption_value * battery_efficiency - price
        if net_value > 0 and count < max_blocks:
            count += 1
        else:
            break

    return max(min_blocks, count)
