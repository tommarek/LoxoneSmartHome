"""Tests for the greedy battery optimizer."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from modules.growatt.optimizer import BatteryOptimizer


@pytest.fixture
def optimizer() -> BatteryOptimizer:
    return BatteryOptimizer(logger=MagicMock())


def make_blocks(prices: list, start_hour: int = 0) -> list:
    """Create price blocks from a list of prices (one per hour)."""
    base = datetime(2026, 4, 11, start_hour, 0)
    return [(base + timedelta(hours=i), p) for i, p in enumerate(prices)]


def const_dist(rate: float):
    """Return a constant distribution tariff function."""
    return lambda h: rate


class TestOptimizerBasic:

    def test_empty_input(self, optimizer) -> None:
        charge, discharge, decisions = optimizer.optimize(
            blocks=[], solar_hourly={}, consumption_hourly={},
            distribution_func=const_dist(1.0),
        )
        assert charge == set()
        assert discharge == set()
        assert decisions == []

    def test_cheap_blocks_charge_expensive_discharge(self, optimizer) -> None:
        """Cheapest blocks should charge, most expensive should discharge.

        Uses a 2-day window (48 blocks) so the optimizer can see cheap recharge
        opportunities tomorrow — otherwise it won't discharge today if it can't
        recharge cheaply in the remaining blocks.
        """
        # Day 1: cheap night, expensive evening
        # Day 2: cheap night again (recharge opportunity after discharge)
        prices = (
            [-1.0] * 6   # day1 0-5: very cheap
            + [2.0] * 6  # day1 6-11: moderate
            + [1.0] * 6  # day1 12-17: moderate-low
            + [8.0] * 6  # day1 18-23: very expensive
            + [-1.0] * 6 # day2 0-5: cheap again (recharge after discharge)
            + [2.0] * 18 # day2 rest: moderate
        )
        blocks = make_blocks(prices)

        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=50.0,
            min_soc=20.0,
            max_soc=100.0,
        )

        charge_hours = {t.hour for t in charge}
        discharge_hours = {t.hour for t in discharge}

        # Cheap hours should have charging
        assert len(charge) > 0
        # Expensive hours (18-23 day1) should have discharging since
        # tomorrow has cheap recharge blocks
        assert len(discharge_hours & set(range(18, 24))) > 0

    def test_soc_never_below_min(self, optimizer) -> None:
        """Battery should never discharge below min_soc."""
        # All expensive prices — optimizer wants to discharge everything
        prices = [10.0] * 24
        blocks = make_blocks(prices)

        _, _, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=30.0,
            min_soc=20.0,
        )

        for d in decisions:
            assert d.soc_after >= 19.9  # Allow tiny float imprecision

    def test_soc_never_above_max(self, optimizer) -> None:
        """Battery should never charge above max_soc."""
        # All cheap prices — optimizer wants to charge everything
        prices = [0.0] * 24
        blocks = make_blocks(prices)

        _, _, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=90.0,
            max_soc=100.0,
        )

        for d in decisions:
            assert d.soc_after <= 100.1

    def test_summarize(self, optimizer) -> None:
        prices = [0.5] * 8 + [3.0] * 8 + [6.0] * 8
        blocks = make_blocks(prices)

        _, _, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
        )

        summary = optimizer.summarize(decisions)
        assert summary["total_blocks"] == 24
        assert summary["charge_blocks"] + summary["discharge_blocks"] + summary["hold_blocks"] == 24


class TestOptimizerSolar:

    def test_solar_reduces_charging_need(self, optimizer) -> None:
        """With solar production, fewer grid charge blocks needed."""
        prices = [1.0] * 24
        blocks = make_blocks(prices)

        # No solar
        charge_no_solar, _, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=20.0,
        )

        # Heavy solar midday
        solar = {h: 3.0 for h in range(9, 16)}
        charge_with_solar, _, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=20.0,
        )

        # With solar, should need fewer or equal charge blocks
        assert len(charge_with_solar) <= len(charge_no_solar)


class TestOptimizerDistribution:

    def test_distribution_affects_discharge_decision(self, optimizer) -> None:
        """Higher distribution tariff makes self-consumption more valuable,
        reducing willingness to discharge."""
        prices = [4.0] * 24  # Moderately high price
        blocks = make_blocks(prices)

        # Low distribution — discharge more attractive
        _, discharge_low, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(0.5),
            current_soc=80.0,
        )

        # High distribution — self-consumption more valuable
        _, discharge_high, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(3.0),
            current_soc=80.0,
        )

        # With high distribution, self-consumption is worth more,
        # so fewer discharge blocks (or equal)
        assert len(discharge_high) <= len(discharge_low)
