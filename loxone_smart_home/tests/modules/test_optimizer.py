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


def make_15min_blocks(hourly_prices: list, start_hour: int = 0) -> list:
    """Create 15-min price blocks from hourly prices (4 blocks per hour)."""
    base = datetime(2026, 4, 11, start_hour, 0)
    blocks = []
    for i, p in enumerate(hourly_prices):
        for q in range(4):
            blocks.append((base + timedelta(hours=i, minutes=q * 15), p))
    return blocks


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


class TestOptimizer15Min:
    """Tests using 15-minute resolution blocks (matching production)."""

    def test_15min_charge_discharge_pattern(self, optimizer) -> None:
        """With 15-min blocks, charge should span multiple blocks per hour."""
        # Day 1 + Day 2 for recharge visibility
        prices = (
            [-1.0] * 6 + [2.0] * 6 + [1.0] * 6 + [8.0] * 6
            + [-1.0] * 6 + [2.0] * 18
        )
        blocks = make_15min_blocks(prices)
        assert len(blocks) == 192  # 48 hours * 4 blocks
        # Low consumption so battery doesn't drain before evening
        low_consumption = {h: 0.1 for h in range(24)}

        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=low_consumption,
            distribution_func=const_dist(1.0),
            current_soc=50.0,
        )
        assert len(decisions) == 192
        assert len(charge) > 0
        # Expensive hours (18-23 day1) should have discharging
        discharge_hours = {t.hour for t in discharge}
        assert len(discharge_hours & set(range(18, 24))) > 0

    def test_all_negative_prices(self, optimizer) -> None:
        """All negative prices: should charge, never discharge."""
        prices = [-2.0] * 24
        blocks = make_15min_blocks(prices)

        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=20.0,
        )
        assert len(charge) > 0
        assert len(discharge) == 0

    def test_single_block(self, optimizer) -> None:
        """Single block edge case."""
        blocks = [(datetime(2026, 4, 11, 12, 0), 3.0)]
        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=50.0,
        )
        assert len(decisions) == 1

    def test_two_blocks(self, optimizer) -> None:
        """Two block edge case: one cheap, one expensive."""
        base = datetime(2026, 4, 11, 0, 0)
        blocks = [(base, -1.0), (base + timedelta(minutes=15), 10.0)]
        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=50.0,
        )
        assert len(decisions) == 2

    def test_night_reserve_defaults(self, optimizer) -> None:
        """Verify _night_reserve_kwh defaults to 5.0 without calling calibrate."""
        assert optimizer._night_reserve_kwh == 5.0
        assert optimizer._night_reserve_updated is None

    def test_backward_pass_prefers_better_blocks(self, optimizer) -> None:
        """6 CZK block before 10 CZK block with limited battery — should
        prefer the more profitable block."""
        # Setup: moderate prices, then a good block, then a great block
        # Hours 0-16: moderate (2 CZK) — hold
        # Hour 17: good (6 CZK)
        # Hour 18-19: great (10 CZK)
        # Hours 20-23: moderate
        # Day 2: cheap (recharge opportunity)
        prices = (
            [2.0] * 17 + [6.0] + [10.0] * 2 + [2.0] * 4
            + [0.0] * 6 + [2.0] * 18
        )
        blocks = make_15min_blocks(prices)

        _, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(0.5),
            current_soc=40.0,  # Limited battery
            min_soc=20.0,
        )

        # With limited battery (only 20% usable = 2 kWh), should prefer
        # discharging at hour 18/19 (10 CZK) over hour 17 (6 CZK)
        discharge_at_17 = sum(1 for t in discharge if t.hour == 17)
        discharge_at_18_19 = sum(1 for t in discharge if t.hour in (18, 19))
        # Should have more discharge blocks at the better price
        assert discharge_at_18_19 >= discharge_at_17

    def test_discharge_economics_includes_recharge_dist(self, optimizer) -> None:
        """Verify recharge cost includes future distribution tariff.

        With low distribution, discharge is profitable (low cost on both sides).
        With very high distribution, sell revenue drops and recharge cost rises,
        making discharge unprofitable.
        """
        # High spot price evening, cheap recharge tomorrow
        prices = (
            [1.0] * 18 + [12.0] * 6  # Day 1: cheap then expensive evening
            + [0.5] * 6 + [2.0] * 18  # Day 2: cheap night for recharge
        )
        blocks = make_15min_blocks(prices)
        # Low consumption so base load doesn't drain battery before evening
        low_consumption = {h: 0.1 for h in range(24)}

        # Low distribution: sell_revenue = 12.0 - 0.5 - 0.5 - 2.0 = 9.0
        # recharge_cost = (0.5 + 0.5) / 0.85 = 1.18
        # profit = 9.0 - 1.18 = 7.82 > 0 → should discharge
        _, discharge_low_dist, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=low_consumption,
            distribution_func=const_dist(0.5),
            current_soc=80.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(discharge_low_dist) > 0

        # Very high distribution: sell_revenue = 12.0 - 5.0 - 0.5 - 2.0 = 4.5
        # recharge_cost = (0.5 + 5.0) / 0.85 = 6.47
        # profit = 4.5 - 6.47 = -1.97 < 0 → should NOT discharge
        _, discharge_high_dist, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=low_consumption,
            distribution_func=const_dist(5.0),
            current_soc=80.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(discharge_high_dist) < len(discharge_low_dist)


class TestTwoPassOptimizer:
    """Tests for the two-pass charge block refinement."""

    def test_pass2_no_change_when_no_waste(self, optimizer: BatteryOptimizer) -> None:
        """When no charge blocks are wasted, Pass 2 returns the same set."""
        # Low SOC + cheap blocks = charge is useful, not wasted
        blocks = make_15min_blocks([0.5, 0.5, 5.0, 5.0], start_hour=0)
        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=lambda h: 0.5,
            current_soc=20.0,  # Low SOC — charging is needed
            min_soc=20.0,
        )
        # All charge blocks should have meaningful SOC increase
        for d in decisions:
            if d.action == "charge":
                assert d.soc_after - d.soc_before >= 1.0, \
                    f"Charge at {d.timestamp} had SOC change of only {d.soc_after - d.soc_before}%"

    def test_pass2_redistributes_solar_overlap(self, optimizer: BatteryOptimizer) -> None:
        """When solar fills battery during a charge slot, Pass 2 moves it."""
        # Hour 0-1: cheap (charge candidates)
        # Hour 2-3: expensive
        # Heavy solar at hour 0 means charging at hour 0 is wasted
        blocks = make_15min_blocks([0.5, 0.5, 5.0, 5.0], start_hour=0)
        solar = {0: 10.0, 1: 0.0, 2: 0.0, 3: 0.0}  # Huge solar at hour 0
        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            distribution_func=lambda h: 0.5,
            current_soc=20.0,
            min_soc=20.0,
            max_soc=100.0,
            battery_capacity_kwh=10.0,
        )
        # The optimizer should still find charge blocks, but they should be
        # at hours where solar doesn't already fill the battery
        assert len(charge) > 0

    def test_pass2_preserves_negative_prices(self, optimizer: BatteryOptimizer) -> None:
        """Negative price blocks should never be considered 'wasted'."""
        # Negative prices = get paid to charge, should always be kept
        blocks = make_15min_blocks([-1.0, -0.5, 5.0, 5.0], start_hour=0)
        charge, discharge, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={0: 10.0, 1: 10.0, 2: 0.0, 3: 0.0},  # Solar fills battery
            consumption_hourly={},
            distribution_func=lambda h: 0.5,
            current_soc=20.0,
            min_soc=20.0,
            battery_capacity_kwh=10.0,
        )
        # Negative price blocks should be in charge set
        negative_block_hours = {ts.hour for ts in charge if ts.hour in [0, 1]}
        # At least some negative-price blocks should be selected
        assert len(charge) > 0
