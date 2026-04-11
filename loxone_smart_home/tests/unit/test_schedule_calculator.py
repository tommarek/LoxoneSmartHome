"""Tests for schedule calculator utility functions."""

import pytest
from datetime import datetime
from typing import List

from utils.schedule_calculator import (
    calculate_optimal_schedule,
    calculate_dynamic_block_count,
    determine_block_mode,
)


class TestCalculateDynamicBlockCount:
    """Tests for dynamic charge block count calculation."""

    def test_empty_prices_returns_min(self) -> None:
        result = calculate_dynamic_block_count([], min_blocks=4, max_blocks=16)
        assert result == 4

    def test_all_prices_below_value_returns_max(self) -> None:
        # All prices are 0 CZK/kWh, self-consumption value is 3.0
        # Net value = 3.0 * 0.85 - 0.0 = 2.55 > 0 for every block
        prices = [0.0] * 20
        result = calculate_dynamic_block_count(
            prices, min_blocks=4, max_blocks=16,
            self_consumption_value=3.0, battery_efficiency=0.85,
        )
        assert result == 16  # Capped at max

    def test_all_prices_above_value_returns_min(self) -> None:
        # All prices are 10 CZK/kWh, self-consumption value is 3.0
        # Net value = 3.0 * 0.85 - 10.0 = -7.45 < 0
        prices = [10.0] * 20
        result = calculate_dynamic_block_count(
            prices, min_blocks=4, max_blocks=16,
            self_consumption_value=3.0, battery_efficiency=0.85,
        )
        assert result == 4

    def test_mixed_prices_correct_count(self) -> None:
        # self_consumption_value * efficiency = 3.0 * 0.85 = 2.55
        # Blocks cheaper than 2.55 are profitable
        prices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        result = calculate_dynamic_block_count(
            prices, min_blocks=2, max_blocks=10,
            self_consumption_value=3.0, battery_efficiency=0.85,
        )
        # Sorted: 0.5, 1.0, 1.5, 2.0, 2.5 are all < 2.55 → 5 blocks
        assert result == 5

    def test_negative_prices_all_included(self) -> None:
        # Negative prices always profitable (charging gets paid)
        prices = [-2.0, -1.0, -0.5, 0.0, 0.5]
        result = calculate_dynamic_block_count(
            prices, min_blocks=1, max_blocks=10,
            self_consumption_value=1.0, battery_efficiency=0.85,
        )
        # threshold = 1.0 * 0.85 = 0.85
        # -2.0, -1.0, -0.5, 0.0, 0.5 are all < 0.85 → 5 blocks
        assert result == 5

    def test_flat_prices_at_threshold(self) -> None:
        # All prices exactly at breakeven
        breakeven = 3.0 * 0.85  # 2.55
        prices = [breakeven] * 10
        result = calculate_dynamic_block_count(
            prices, min_blocks=4, max_blocks=16,
            self_consumption_value=3.0, battery_efficiency=0.85,
        )
        # Net value = 0, not > 0, so no blocks profitable → min
        assert result == 4

    def test_min_blocks_floor_respected(self) -> None:
        # Only 1 block profitable, but min is 4
        prices = [0.0, 10.0, 10.0, 10.0, 10.0]
        result = calculate_dynamic_block_count(
            prices, min_blocks=4, max_blocks=16,
            self_consumption_value=3.0, battery_efficiency=0.85,
        )
        assert result == 4


class TestCalculateOptimalSchedule:
    """Tests for the existing schedule calculation."""

    def test_empty_input(self) -> None:
        charge, discharge, ct, dt = calculate_optimal_schedule([])
        assert charge == set()
        assert discharge == set()

    def test_picks_cheapest_blocks(self) -> None:
        now = datetime(2026, 4, 10, 0, 0)
        blocks = [
            (now.replace(hour=h), float(h)) for h in range(24)
        ]
        charge, _, _, _ = calculate_optimal_schedule(blocks, charge_blocks_count=4)
        # Hours 0-3 are cheapest (price 0.0, 1.0, 2.0, 3.0)
        charge_hours = {t.hour for t in charge}
        assert charge_hours == {0, 1, 2, 3}

    def test_discharge_threshold(self) -> None:
        now = datetime(2026, 4, 10, 0, 0)
        # Cheapest = 1.0, margin 4.0 → threshold = max(5.0, 1.0*4.0) = 5.0
        blocks = [
            (now.replace(hour=0), 1.0),
            (now.replace(hour=12), 6.0),
            (now.replace(hour=18), 3.0),
        ]
        _, discharge, _, eff_threshold = calculate_optimal_schedule(
            blocks, discharge_threshold_czk=5.0, discharge_profit_margin=4.0,
        )
        assert now.replace(hour=12) in discharge
        assert now.replace(hour=18) not in discharge
        assert eff_threshold == 5.0
