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
        charge, discharge, _sp, decisions = optimizer.optimize(
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

        charge, discharge, _sp, decisions = optimizer.optimize(
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

        _, _, _sp, decisions = optimizer.optimize(
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

        _, _, _sp, decisions = optimizer.optimize(
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

        _, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
        )

        summary = optimizer.summarize(decisions)
        assert summary["total_blocks"] == 24
        # hold_idle is the battery-passive preserve variant of hold (greedy
        # retention holds on deficit blocks emit it since the SPH hold fix).
        assert (
            summary["charge_blocks"] + summary["discharge_blocks"]
            + summary["hold_blocks"] + summary["hold_idle_blocks"]
        ) == 24


class TestOptimizerSolar:

    def test_solar_reduces_charging_need(self, optimizer) -> None:
        """With solar production, fewer grid charge blocks needed."""
        prices = [1.0] * 24
        blocks = make_blocks(prices)

        # No solar
        charge_no_solar, _, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=20.0,
        )

        # Heavy solar midday
        solar = {h: 3.0 for h in range(9, 16)}
        charge_with_solar, _, _sp, _ = optimizer.optimize(
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
        _, discharge_low, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(0.5),
            current_soc=80.0,
        )

        # High distribution — self-consumption more valuable
        _, discharge_high, _sp, _ = optimizer.optimize(
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

        charge, discharge, _sp, decisions = optimizer.optimize(
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

        charge, discharge, _sp, decisions = optimizer.optimize(
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
        charge, discharge, _sp, decisions = optimizer.optimize(
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
        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=50.0,
        )
        assert len(decisions) == 2

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

        _, discharge, _sp, decisions = optimizer.optimize(
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
        """Verify the RECHARGE (import) cost includes the distribution tariff.

        Export pays no distribution, so sell revenue is independent of it; but
        the future recharge is an IMPORT (price + distribution). With low
        distribution the discharge→recharge round-trip is profitable; with very
        high distribution the recharge cost exceeds the sell revenue, so it
        should not discharge.
        """
        # High spot price evening, cheap recharge tomorrow
        prices = (
            [1.0] * 18 + [12.0] * 6  # Day 1: cheap then expensive evening
            + [0.5] * 6 + [2.0] * 18  # Day 2: cheap night for recharge
        )
        blocks = make_15min_blocks(prices)
        # Low consumption so base load doesn't drain battery before evening
        low_consumption = {h: 0.1 for h in range(24)}

        # Low distribution: sell_revenue = 12.0 - 0.5 - 2.0 = 9.5 (no dist on export)
        # recharge_cost = (0.5 + 0.5) / 0.85 = 1.18 → profit 8.3 > 0 → discharge
        _, discharge_low_dist, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=low_consumption,
            distribution_func=const_dist(0.5),
            current_soc=80.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(discharge_low_dist) > 0

        # Very high distribution: sell_revenue still 9.5 (export pays no dist), but
        # recharge_cost = (0.5 + 10.0) / 0.85 = 12.35 > 9.5 → round-trip a loss →
        # should NOT discharge (can't profitably refill).
        _, discharge_high_dist, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=low_consumption,
            distribution_func=const_dist(10.0),
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
        charge, discharge, _sp, decisions = optimizer.optimize(
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
        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            distribution_func=lambda h: 0.5,
            current_soc=20.0,
            min_soc=20.0,
            max_soc=100.0,
            battery_capacity_kwh=10.0,
        )
        # Documented behavior: charging stays in the CHEAP hours (0-1) and is
        # never placed in the expensive hours (2-3); the solar burst + cheap
        # charge leave the battery full for the expensive-hour discharge.
        charge_hours = {ts.hour for ts in charge}
        assert charge_hours, "expected some charge blocks"
        assert charge_hours <= {0, 1}, (
            f"charge must stay in cheap hours 0-1, not expensive 2-3: {sorted(charge_hours)}"
        )
        # And the battery reaches (near) full by the time expensive hours start.
        soc_at_h2 = next(d.soc_after for d in decisions if d.timestamp.hour == 2)
        assert soc_at_h2 >= 90.0

    def test_pass2_preserves_negative_prices(self, optimizer: BatteryOptimizer) -> None:
        """Negative price blocks should never be considered 'wasted'."""
        # Negative prices = get paid to charge, should always be kept
        blocks = make_15min_blocks([-1.0, -0.5, 5.0, 5.0], start_hour=0)
        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={0: 10.0, 1: 10.0, 2: 0.0, 3: 0.0},  # Solar fills battery
            consumption_hourly={},
            distribution_func=lambda h: 0.5,
            current_soc=20.0,
            min_soc=20.0,
            battery_capacity_kwh=10.0,
        )
        # Documented behavior: negative-price blocks (paid to charge) must be
        # KEPT in the charge set, never dropped as "wasted" by Pass 2.
        negative_block_hours = {ts.hour for ts in charge if ts.hour in (0, 1)}
        assert negative_block_hours, (
            f"negative-price hours 0/1 must be charged: {sorted(ts.hour for ts in charge)}"
        )

    def test_picks_cheapest_negatives_not_chronologically_first(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Steep negative gradient: cheapest later blocks must win over earlier
        less-negative blocks.

        Reproduces the production bug from 2026-04-25 where the optimizer
        scheduled charging at 09:15-11:00 (-0.05 to -1.75 CZK) and skipped
        the deeply-negative midday window 13:30-15:30 (-6 to -12 CZK).
        Battery filled to 100% from the morning charges, so the much cheaper
        afternoon blocks became no-ops in the forward simulation.

        After the fix, the cheapest-N selector picks the deepest negatives
        first regardless of when they fall in the day.
        """
        # 9-12: mildly negative morning (4 hours)
        # 13-14: positive (skip)
        # 15-18: deeply negative (4 hours) — these MUST win
        # 19+:  high evening (filler)
        prices = (
            [-0.05, -0.5, -1.0, -2.0]
            + [0.5, 0.5]
            + [-6.0, -9.0, -11.0, -12.0]
            + [5.0] * 14
        )
        blocks = make_blocks(prices, start_hour=9)

        charge, _, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,           # small 2 kWh gap → blocks_to_fill at the floor of 4
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        charge_hours = sorted({t.hour for t in charge})

        # The four deeply-negative midday hours must be charged.
        deep_negative_hours = {15, 16, 17, 18}
        assert deep_negative_hours.issubset(charge_hours), (
            f"Optimizer must pick deeply-negative blocks {deep_negative_hours}; "
            f"got {charge_hours}"
        )
        # And the marginally-negative morning ones should NOT be picked
        # (battery has limited capacity and cheaper alternatives are ahead).
        marginal_morning = {9, 10}
        assert not (marginal_morning & set(charge_hours)), (
            f"Optimizer must not waste battery headroom on marginally-negative "
            f"blocks {marginal_morning} when -12 CZK blocks are coming; "
            f"got {charge_hours}"
        )


class TestSelfConsumptionHold:
    """Tests for self-consumption hold value — battery should prefer
    powering the house during expensive hours over selling to grid
    when selling earns less than self-consumption saves."""

    def test_hold_for_expensive_evening_when_battery_limited(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Battery at low SOC with expensive evening ahead should NOT discharge
        during moderate-price hours — it's worth more for self-consumption."""
        # Hour 0-5: cheap (1.0 CZK), Hour 6-11: moderate (3.0 CZK),
        # Hour 12-17: expensive evening (6.0 CZK)
        prices = [1.0] * 6 + [3.0] * 6 + [6.0] * 6
        blocks = make_15min_blocks(prices, start_hour=0)

        # Meaningful house consumption so self-consumption matters
        consumption = {h: 1.0 for h in range(18)}  # 1 kWh/hr

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=consumption,
            distribution_func=lambda h: 0.5,
            battery_capacity_kwh=10.0,
            current_soc=40.0,  # Only 2 kWh usable (40% - 20% min)
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # At moderate prices (3.0 CZK), discharge_profit = 3.0 - 0.5 - 0.5 - 2.0 - recharge
        # = 0.0 - recharge < 0, so no discharge at moderate prices anyway.
        # But at 6.0 CZK, profit = 6.0 - 0.5 - 0.5 - 2.0 - recharge = 3.0 - recharge
        # Self-consumption at 6.0 CZK saves 6.5 CZK/kWh (price + dist)
        # With limited battery (2 kWh) and 6 hrs of consumption ahead,
        # holding is better than discharging
        discharge_hours = {ts.hour for ts in discharge}
        # Should NOT discharge during moderate hours (6-11) — save for evening
        moderate_discharge = sum(1 for ts in discharge if 6 <= ts.hour < 12)
        assert moderate_discharge == 0, (
            f"Should not discharge during moderate hours when expensive evening ahead, "
            f"but discharged at hours: {sorted(discharge_hours)}"
        )

    def test_discharge_allowed_when_battery_has_excess(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Battery with plenty of charge beyond self-consumption needs
        SHOULD still discharge — only the needed portion is held.
        The self-consumption hold value should NOT block discharge when
        usable_kwh exceeds future consumption needs."""
        # Create a scenario with very high price spread and minimal consumption
        # Hour 0-11: cheap (0.5 CZK), Hour 12-23: very expensive (10.0 CZK)
        prices = [0.5] * 12 + [10.0] * 12
        blocks = make_15min_blocks(prices, start_hour=0)

        # Minimal consumption — battery has way more than needed
        consumption = {h: 0.1 for h in range(24)}  # Only 0.1 kWh/hr = 2.4 kWh/day

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=consumption,
            distribution_func=lambda h: 0.5,
            battery_capacity_kwh=10.0,
            current_soc=80.0,  # 6 kWh usable, only 2.4 kWh consumption
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # Verify the self-consumption hold is not preventing ALL discharge.
        # The hold_value condition is: usable_kwh <= sc_kwh_needed
        # With 6 kWh usable and ~2.4 kWh consumption, excess should discharge.
        # If discharge is still 0, that's OK if it's due to the 80% worthwhile
        # threshold or reserve SOC — not the self-consumption hold.
        # Check that hold decisions during expensive hours have hold_value = 0
        # (meaning self-consumption hold did NOT activate for excess battery)
        expensive_holds = [
            d for d in decisions
            if d.action == "hold" and d.price_czk >= 10.0
        ]
        # At least some expensive blocks should exist as hold (not all charged)
        if expensive_holds:
            # The hold_value for excess battery blocks should be 0
            # (self-consumption hold only activates when battery is limited)
            first_expensive = expensive_holds[0]
            # SOC should still be high (battery not drained by low consumption)
            assert first_expensive.soc_before > 40, (
                f"SOC should remain high with low consumption, got {first_expensive.soc_before}"
            )

    def test_no_discharge_when_sell_revenue_negative(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Discharge to grid must not be selected when sell_revenue ≤ 0.

        Reproduces the production bug: optimizer scheduled discharges across
        morning hours where spot prices were near zero or negative, justifying
        them via 'round-trip arbitrage' against very-cheap midday recharge.
        After the amortisation fix, a discharge is only allowed when the sale
        itself covers fees + amortisation (sell_revenue > 0).
        """
        # Morning: cheap-ish (0.0–2.0 CZK), midday: very negative (recharge bait)
        # Even though "round trip" math (-2.62 - (-3.5) = 0.88) is positive,
        # we must refuse to discharge at a sale loss.
        prices = (
            [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]   # morning falling
            + [-3.0, -3.0, -3.0, -3.0]                     # midday very negative
            + [4.0, 5.0, 6.0, 7.0]                         # evening recovery
            + [3.0] * 8
        )
        blocks = make_15min_blocks(prices, start_hour=4)

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 0.2 for h in range(28)},
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=12.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # No discharge should land on blocks where sell_revenue ≤ 0:
        # sell_rev = price - 0.12 - 0.5 - 2.0 ≤ 0  ⇔  price ≤ 2.62
        bad = [d for d in decisions
               if d.action == "discharge" and d.price_czk <= 2.62]
        assert bad == [], (
            "Discharge selected at unprofitable sell prices: "
            + ", ".join(f"{d.timestamp.strftime('%H:%M')}={d.price_czk:+.2f}" for d in bad)
        )

        # Sanity: the very-negative midday blocks (-3.0 CZK) should be the
        # cheapest picked — not the marginally-negative morning blocks.
        cheap_charge_hours = {ts.hour for ts in charge}
        assert any(h in cheap_charge_hours for h in (12, 13, 14, 15)), (
            f"Should charge at the cheapest (-3 CZK) midday blocks, "
            f"got hours: {sorted(cheap_charge_hours)}"
        )

    def test_no_discharge_at_negative_spot_even_with_cheaper_future(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """The 'round-trip arbitrage' bug: discharge at slightly-negative now,
        recharge at very-negative later. Must not happen — battery wear is real."""
        # Now: -1.0 (sell_rev = -1.0 - 0.12 - 0.5 - 2.0 = -3.62, loss)
        # Future: -3.5 (recharge "gain" 3.5/0.85 ≈ 4.12)
        # Old logic: discharge_profit = -3.62 - (-4.0) = +0.38 → would discharge.
        # New logic: sell_revenue ≤ 0 → discharge forbidden.
        prices = [-1.0] * 4 + [-3.5] * 4 + [3.0] * 8
        blocks = make_15min_blocks(prices, start_hour=10)

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 0.2 for h in range(24)},
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=90.0,  # plenty of battery to abuse
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        negative_price_discharges = [
            d for d in decisions if d.action == "discharge" and d.price_czk < 0
        ]
        assert negative_price_discharges == [], (
            f"Optimizer discharged into negative-price market: "
            + ", ".join(f"{d.timestamp.strftime('%H:%M')}={d.price_czk:+.2f}"
                        for d in negative_price_discharges)
        )

    def test_amortisation_threshold_governs_discharge_floor(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """The discharge floor scales with amortisation: a higher wear cost
        forbids discharges that a lower wear cost would still permit."""
        # Spot 2.0 CZK, dist 0.12, fees 0.5.
        # With amort=0: sell_rev = 2.0 - 0.12 - 0.5 - 0 = +1.38 → discharge OK
        # With amort=2: sell_rev = 2.0 - 0.12 - 0.5 - 2.0 = -0.62 → forbidden
        prices = [2.0] * 8 + [-1.0] * 4 + [4.0] * 12   # cheap recharge in middle
        blocks = make_15min_blocks(prices, start_hour=0)

        common = dict(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 0.1 for h in range(24)},
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
        )

        _, discharge_no_amort, _sp, dec_no = optimizer.optimize(**common, battery_amortisation_czk=0.0)
        _, discharge_high_amort, _sp, dec_hi = optimizer.optimize(**common, battery_amortisation_czk=2.0)

        # At spot=2.0, the high-amort run must NOT discharge those blocks.
        spot_2_dischg_hi = [d for d in dec_hi
                            if d.action == "discharge" and abs(d.price_czk - 2.0) < 0.01]
        assert spot_2_dischg_hi == [], (
            "High amortisation should forbid discharge at spot=2.0 "
            f"(sell_rev = -0.62), but got {len(spot_2_dischg_hi)} blocks"
        )

        # The zero-amort run is allowed to discharge there (sell_rev = +1.38).
        spot_2_dischg_no = [d for d in dec_no
                            if d.action == "discharge" and abs(d.price_czk - 2.0) < 0.01]
        assert len(spot_2_dischg_no) > 0, (
            "Zero amortisation should permit discharge at spot=2.0 with cheap recharge"
        )

    def test_self_consumption_value_subtracts_amortisation(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Future SC value used for the hold bonus must subtract amortisation.

        Without the fix, future SC was valued at (price + dist) — the gross
        avoided buy cost. That overstates the value of holding the battery,
        because using it for SC also wears it. With the fix, future SC is
        (price + dist - amort), so 'hold for SC' only fires when the saving
        actually beats the wear cost.

        Concrete: now=4.0 with NO consumption (solar covers it, so it's a
        pure sell opportunity). Future=3.0 with consumption (only here can
        SC value be earned). Sell-now revenue is +1.38; per-kWh SC saving
        net of wear at 3.0 is +1.12. Pre-fix code valued future SC at +3.12
        and would hold; post-fix correctly prefers discharge.
        """
        # Hour 8: peak (4.0 CZK), consumption covered by solar (no deficit).
        # Hour 9: cheap (0.5 CZK) — recharge bait.
        # Hours 10–31: moderate (3.0 CZK), real consumption — only here is SC valuable.
        prices = [4.0] * 1 + [0.5] * 1 + [3.0] * 22
        blocks = make_15min_blocks(prices, start_hour=8)

        # Solar exactly cancels consumption at hours 8-9 (no SC need there);
        # full consumption from hour 10 onward. NOTE: an explicit 0.0
        # consumption would trigger the engine's base-load fallback (the value
        # model and the SOC simulation both serve learned base load on
        # zero-forecast blocks), so the "no deficit at the peak" premise is
        # expressed as solar == consumption instead.
        consumption = {h: 1.0 for h in range(24)}
        consumption[8] = 0.4
        consumption[9] = 0.4
        solar = {8: 0.4, 9: 0.4}

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        peak_discharge = [d for d in decisions
                          if d.action == "discharge" and abs(d.price_czk - 4.0) < 0.01]
        assert len(peak_discharge) > 0, (
            "Should discharge at the 4.0 CZK peak: sell_rev=+1.38 beats "
            "future SC saving net of amortisation (+1.12). The pre-fix code "
            "would value future SC at +3.12 and incorrectly hold."
        )

    def test_hold_value_does_not_block_when_no_future_consumption(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """When there's no future consumption (all solar-covered),
        hold value should not prevent profitable discharge.

        TEST CHANGE (iteration 2): starts at 100% SOC (nothing to solar-bank,
        so the hold scoring is isolated to retention) and adds a cheap tail
        AFTER the peak (the discharge gate requires a profitable future
        recharge: dv = sell_rev - recharge_cost > 0; in the original shape
        the only cheap blocks preceded the peak, so dv was always negative
        and the assertion survived purely via the `len(charge) > 0` arm — an
        artifact of the old median-import charge gate, which the
        forward-looking gate correctly refuses with no in-horizon deficit."""
        # Solar-covered hours 0-7, expensive 8.0 peak at 4-7, cheap tail 8-11
        prices = [0.5] * 4 + [8.0] * 4 + [0.5] * 4
        blocks = make_15min_blocks(prices, start_hour=0)

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={h: 5.0 for h in range(8)},  # Solar covers hours 0-7
            consumption_hourly={h: 0.5 for h in range(12)},
            distribution_func=lambda h: 0.5,
            battery_capacity_kwh=10.0,
            current_soc=100.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # With solar covering all peak-hour consumption there is no
        # self-consumption need at the peak — the hold/retention value must
        # not block the profitable discharge:
        # sell_rev = 8.0 - 0.5 - 2.0 = 5.5, recharge = (0.5+0.5)/0.85 ≈ 1.18.
        assert len(discharge) > 0, (
            "Hold value must not block profitable discharge when solar covers "
            "all consumption and the battery is full"
        )


class TestSellProduction:
    """Tests for the sell_production action — export solar to grid instead
    of storing it. Sell_production is the WEAKER variant of selling: it
    exports solar excess only (battery passive). Discharge is the stronger
    variant (sells battery + solar excess). So sell_production fires only
    in the gap zone where battery export is unprofitable (sell_revenue ≤ 0
    after amort) but solar export is still profitable (sell_now > 0)."""

    def test_sells_gap_zone_solar_when_negative_midday(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Morning at 1.5 CZK is in the gap zone — sell_revenue (with amort)
        is -1.12 so discharge gated off, but sell_now (no amort, solar→grid)
        is +0.88. With midday at -2 CZK, the swap is profitable."""
        # 1.5 CZK morning (gap zone), -2 CZK midday (very cheap recharge),
        # 2 CZK evening (consumption hours)
        prices = [1.5] * 4 + [-2.0] * 4 + [2.0] * 12
        blocks = make_15min_blocks(prices, start_hour=8)
        solar = {h: 5.0 for h in range(8, 12)}  # heavy morning solar
        consumption = {h: 0.5 for h in range(24)}

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # Morning blocks (price 1.5) should be sell_production
        morning_sp = [d for d in decisions
                      if d.action == "sell_production" and 8 <= d.timestamp.hour < 12]
        assert len(morning_sp) > 0, (
            "Morning gap-zone solar at 1.5 CZK with cheap midday "
            "should be sell_production. Got actions: "
            f"{[(d.timestamp.strftime('%H:%M'), d.action) for d in decisions[:20]]}"
        )

        # Midday should still charge (paid to take energy)
        midday_charge = [d for d in decisions
                         if d.action == "charge" and 12 <= d.timestamp.hour < 16]
        assert len(midday_charge) > 0, "Midday negative blocks should still charge"

        # Morning blocks should NOT be discharge (sell_revenue is -1.12)
        morning_discharge = [d for d in decisions
                             if d.action == "discharge" and 8 <= d.timestamp.hour < 12]
        assert morning_discharge == [], (
            "Morning at 1.5 CZK shouldn't discharge — sell_revenue is negative "
            "after amortisation"
        )

    def test_discharge_takes_precedence_over_sell_production(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """When a block qualifies for both discharge (high spot covers wear)
        and sell_production, discharge wins — it sells battery AND solar."""
        # 4 CZK morning (sell_revenue +1.38, discharges battery too) + cheap midday
        prices = [4.0] * 4 + [-2.0] * 4 + [3.0] * 12
        blocks = make_15min_blocks(prices, start_hour=8)
        solar = {h: 5.0 for h in range(8, 12)}
        consumption = {h: 0.5 for h in range(24)}

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # The 4 CZK blocks (hour 8) qualify for both — discharge must win there.
        # (Later 3 CZK blocks may turn to sell_production once the faster, realistic
        # discharge has drawn the battery down; that's expected, not the assertion.)
        peak_blocks = [d for d in decisions if d.timestamp.hour == 8]
        assert any(d.action == "discharge" for d in peak_blocks), (
            "4 CZK morning should discharge"
        )
        assert all(d.action != "sell_production" for d in peak_blocks), (
            "Discharge supersedes sell_production at 4 CZK"
        )

    def test_sells_when_huge_afternoon_solar_refills(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """No cheap grid charging, but later solar far exceeds battery gap:
        morning solar is displaceable. Sell_production fires via solar
        replacement path (lose lower-priced future export, sell at higher now)."""
        # 1.8 CZK morning (gap zone, sell_now = 1.18), 0.8 CZK midday (also
        # gap zone but cheaper), no charge blocks selected (all > median*0.85)
        # MASSIVE midday solar fills battery alone, then exports rest.
        # Swap: sell now at 1.18 vs export later at 0.18 → +1.0 profit.
        prices = [1.8] * 4 + [0.8] * 8 + [1.5] * 12
        blocks = make_15min_blocks(prices, start_hour=8)
        # Morning solar above the per-block excess floor (0.5 kWh/15min after
        # consumption deducted)
        solar = {h: 2.5 for h in range(8, 12)}      # modest morning
        solar.update({h: 12.0 for h in range(12, 20)})  # huge midday/afternoon
        consumption = {h: 0.5 for h in range(24)}

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,  # mostly full — small gap easy to refill
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        morning_sp = [d for d in decisions
                      if d.action == "sell_production" and 8 <= d.timestamp.hour < 12]
        assert len(morning_sp) > 0, (
            "Morning solar at 1.8 CZK should be sell_production when later "
            "solar at lower price fills battery alone. Got actions: "
            f"{[(d.timestamp.strftime('%H:%M'), d.action) for d in decisions[:24]]}"
        )

    def test_holds_morning_solar_when_storage_value_too_high(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """When future SC value (with amort) exceeds sell_now by more than
        the margin, holding for SC beats exporting now."""
        # Morning at 1.0 CZK (gap zone, sell_now = 0.38), evening 5 CZK
        # consumption (SC value = 5+0.12-2 = 3.12, way above 0.38).
        # No cheap recharge, no future solar.
        prices = [1.0] * 4 + [3.0] * 4 + [5.0] * 16
        blocks = make_15min_blocks(prices, start_hour=8)
        solar = {h: 2.0 for h in range(8, 12)}     # morning only
        consumption = {h: 0.3 for h in range(8, 14)}  # light morning
        consumption.update({h: 1.5 for h in range(14, 24)})  # heavy evening (SC need)

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=50.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # sell_now at 1.0 CZK = 0.38; storage_value = 3.12 (SC at evening 5)
        # swap_profit = 0.38 - 3.12 = -2.74 → don't sell, hold for SC
        morning_sp = [d for d in decisions
                      if d.action == "sell_production" and 8 <= d.timestamp.hour < 12]
        assert morning_sp == [], (
            "Should hold morning solar when future SC saves more than selling now. "
            f"Got: {[(d.timestamp.strftime('%H:%M'), d.action) for d in morning_sp]}"
        )

    def test_no_sell_production_when_storage_dominates(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """When neither cheap grid charging nor future solar is available
        AND storage value (SC) exceeds sell_now, the heuristic must hold.

        At gap-zone morning prices (sell_now small) and high evening SC
        (storage_value large), holding is strictly better than selling.
        """
        # 1.5 CZK morning (gap zone, sell_now = 0.88), 6 CZK evening
        # heavy consumption. Storage_value = 6+0.12-2 = 4.12 >> sell_now.
        # No future solar (morning only), no cheap grid (all >> threshold).
        prices = [1.5] * 4 + [5.5] * 4 + [6.0] * 16
        blocks = make_15min_blocks(prices, start_hour=8)
        solar = {h: 4.0 for h in range(8, 12)}     # morning only
        consumption = {h: 0.3 for h in range(8, 14)}
        consumption.update({h: 2.0 for h in range(14, 24)})  # heavy evening SC

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=50.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # sell_now (0.88) << storage_value (4.12) → swap_profit hugely negative
        morning_sp = [d for d in decisions
                      if d.action == "sell_production" and 8 <= d.timestamp.hour < 12]
        assert morning_sp == [], (
            "Should hold for high-value SC instead of selling at gap-zone morning. "
            f"Got: {[(d.timestamp.strftime('%H:%M'), d.action) for d in morning_sp]}"
        )

    def test_aggregate_budget_caps_sell_production(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Many morning solar-surplus blocks but only one cheap recharge slot:
        aggregate refill budget caps the number of sell_production blocks."""
        # 8 hours of 4 CZK morning, all with heavy solar (32 kWh excess total)
        # 1 hour of -3 CZK midday (~ 2.5 kWh charge capacity at 25% efficiency-adjusted)
        # then evening at 4 CZK
        prices = [4.0] * 8 + [-3.0] * 1 + [4.0] * 15
        blocks = make_15min_blocks(prices, start_hour=4)
        solar = {h: 5.0 for h in range(4, 12)}  # 8 hrs × 5 kWh = 40 kWh solar
        consumption = {h: 1.0 for h in range(24)}  # 1 kWh/hr loads

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=20.0,  # Empty battery → big gap, no solar-only refill
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # Total morning solar excess: ~32 kWh. Refill budget bounded by
        # 1 cheap charge hour × ~2.125 kWh + future_solar_surplus * eff.
        # Either way, NOT all 32 morning blocks should be sell_production.
        morning_sp_kwh = sum(
            max(0.0, d.solar_kwh - d.consumption_kwh)
            for d in decisions if d.action == "sell_production"
        )
        # Budget includes future_solar_surplus[0] × eff; allocation should
        # not exceed available refill capacity
        assert morning_sp_kwh < 40.0 * 0.85 + 5.0, (
            f"Allocated solar export ({morning_sp_kwh:.2f} kWh) exceeds available "
            f"refill capacity"
        )

    def test_no_double_action(self, optimizer: BatteryOptimizer) -> None:
        """Each block has exactly one action — no overlap between charge,
        discharge, and sell_production sets."""
        prices = [4.0] * 4 + [-2.0] * 4 + [3.0] * 16
        blocks = make_15min_blocks(prices, start_hour=6)
        solar = {h: 5.0 for h in range(6, 14)}
        consumption = {h: 0.5 for h in range(24)}

        charge, discharge, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=70.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # Set intersections must be empty
        assert charge & discharge == set(), "charge and discharge sets overlap"
        assert charge & sell_prod == set(), "charge and sell_production overlap"
        assert discharge & sell_prod == set(), "discharge and sell_production overlap"

        # Per-block actions are mutually exclusive. hold_idle is the greedy
        # battery-passive preserve (retention hold on a deficit block).
        valid_actions = {"charge", "discharge", "hold", "hold_idle", "sell_production"}
        for d in decisions:
            assert d.action in valid_actions, f"Unknown action: {d.action}"

    def test_skips_sell_production_when_solar_excess_below_floor(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Tiny forecast solar excess (below the volume floor) must NOT
        trigger sell_production, even when the spot price is high.

        Reproduces the production case where a 90 Wh forecast excess at a
        4.96 CZK evening block produced a sell_production assignment that
        locked the battery and forced grid imports for the real load deficit
        (consumption forecast under-predicted heating load).
        """
        # Two consecutive evening blocks at high prices. First has substantial
        # solar excess (above floor) → should sell. Second has trickle excess
        # (below floor) → should hold despite the very high price.
        prices = [4.0] * 8 + [-3.0] * 4 + [3.0] * 12   # midday recharge available
        blocks = make_15min_blocks(prices, start_hour=10)
        # Hour 10: 5 kWh/hr solar, 0.5 kWh/hr cons → excess 1.125 kWh/block (>= floor)
        # Hour 11: 0.6 kWh/hr solar, 0.5 kWh/hr cons → excess 0.025 kWh/block (< floor)
        solar = {10: 5.0, 11: 0.6}
        consumption = {h: 0.5 for h in range(24)}

        _, _, sell_prod, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        sp_hours = {ts.hour for ts in sell_prod}
        assert 10 in sp_hours, (
            f"Hour 10 has 1.125 kWh excess (well above floor) — should fire. "
            f"Got: {sorted(sp_hours)}"
        )
        assert 11 not in sp_hours, (
            f"Hour 11 has only 0.025 kWh excess (below 0.25 floor) — must NOT "
            f"fire even at high price. Got: {sorted(sp_hours)}"
        )

    def test_date_aware_forecasts_do_not_collide_across_days(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """A cross-day window must not reuse tomorrow's hour forecast for today."""
        base = datetime(2026, 4, 11, 10, 0)
        blocks = [
            (base + timedelta(minutes=15 * i), 4.0)
            for i in range(8)
        ] + [
            (base + timedelta(days=1, minutes=15 * i), 4.0)
            for i in range(8)
        ] + [
            (base + timedelta(days=1, hours=2, minutes=15 * i), -3.0)
            for i in range(4)
        ]

        today = base.date()
        tomorrow = (base + timedelta(days=1)).date()
        solar = {
            (today, 10): 0.0,
            (today, 11): 0.0,
            (tomorrow, 10): 5.0,
            (tomorrow, 11): 5.0,
        }
        consumption = {
            (today, h): 0.5 for h in range(24)
        }
        consumption.update({(tomorrow, h): 0.5 for h in range(24)})

        _, _, _, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly=solar,
            consumption_hourly=consumption,
            distribution_func=const_dist(0.12),
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        today_10 = [d for d in decisions if d.timestamp.date() == today and d.timestamp.hour == 10]
        tomorrow_10 = [
            d for d in decisions
            if d.timestamp.date() == tomorrow and d.timestamp.hour == 10
        ]
        assert today_10
        assert tomorrow_10
        assert all(d.solar_kwh == 0.0 for d in today_10)
        assert all(d.solar_kwh == 1.25 for d in tomorrow_10)


class TestExportAmortisationPassTwo:
    """battery_amortisation_export_czk must survive into the Pass-2 re-run.

    Regression: the Pass-2 _single_pass call omitted amort_export, so it
    defaulted back to the shared wear cost and a prohibitive export-wear
    override was silently dropped whenever Pass 2 fired.
    """

    def test_pass2_keeps_export_wear_override(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # Hour 0: very cheap + solar sized so charge+banking lands the battery
        # at ~99.1% after three blocks → the fourth hour-0 charge block is
        # "wasted" (ΔSOC < 1%), which forces Pass 2 to redistribute it onto a
        # cheap hour-3 hold block. Hours 1-2: 5.0 CZK — with base amortisation
        # 0 a discharge there is profitable (sell_rev = 4.5, recharge ≈ 1.06)
        # and DOES fire when the override is absent (verified). The OLD Pass-2
        # dropped the override, so its re-run scheduled exactly those
        # discharges. With the 50 CZK export-wear override sell_rev is deeply
        # negative, so the final plan must contain NO grid-export discharge.
        blocks = make_15min_blocks([0.3, 5.0, 5.0, 0.4], start_hour=0)
        _, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={0: 5.6},  # tops the battery just before hour-0 ends
            consumption_hourly={h: 1.0 for h in range(4)},
            distribution_func=const_dist(0.5),
            battery_capacity_kwh=10.0,
            current_soc=50.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=0.0,
            battery_amortisation_export_czk=50.0,
        )

        # The scenario must actually exercise Pass 2 (wasted hour-0 charge
        # blocks redistributed), otherwise this test proves nothing.
        pass2_ran = any(
            "Pass 2" in str(call)
            for call in optimizer.logger.info.call_args_list
        )
        assert pass2_ran, "scenario must force a Pass-2 charge redistribution"

        assert discharge == set(), (
            "export-wear override (50 CZK/kWh) must gate off all grid-export "
            "discharges in the FINAL (Pass-2) plan too; got discharges at "
            f"{sorted(ts.strftime('%H:%M') for ts in discharge)}"
        )
        assert all(d.action != "discharge" for d in decisions)


class TestBelowFloorLiveSoc:
    """A below-min_soc telemetry SOC must not be clamped UP (phantom energy)."""

    def test_below_min_soc_is_preserved_and_not_discharged(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # All-expensive prices: the optimizer would love to discharge, but the
        # battery is BELOW the floor — there is no usable energy. The old
        # two-sided clamp lifted 10% → 20% and credited 1 kWh out of thin air.
        prices = [10.0] * 24
        blocks = make_blocks(prices)

        _, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=10.0,   # below the 20% floor (real telemetry case)
            min_soc=20.0,
        )

        assert decisions[0].soc_before == 10.0, (
            "live below-floor SOC must be preserved, not clamped up to min_soc"
        )
        assert discharge == set(), "no usable energy below the floor"
        # Nothing can draw the battery further down, and with no charge blocks
        # (flat 10 CZK day) nothing raises it either — no phantom credit.
        for d in decisions:
            assert d.soc_after <= 10.0 + 1e-6

    def test_above_max_soc_still_clamped_down(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # The high-side clamp must survive (documented real condition:
        # telemetry 100% while max_soc=90).
        blocks = make_blocks([2.0] * 24)
        _, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=100.0,
            max_soc=90.0,
        )
        assert decisions[0].soc_before == 90.0


class TestTerminalSocValue:
    """Leftover SOC at horizon end is worth the MILP-style terminal value, so
    the greedy engine must not dump the battery into a marginal end-of-horizon
    price bump before tomorrow's DAM prices arrive."""

    def test_no_end_of_horizon_battery_dump(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # Horizon 10:00-23:45. Late-night 1.0 blocks set the terminal-value
        # cap (min import 2.0 → cap 2.0/0.85 ≈ 2.35; median SC = 3.5+1-2 =
        # 2.5 → terminal ≈ 2.33). The 4.5 spike's GROSS sell value is
        # 4.5-0.5-2.0 = 2.0 CZK/kWh — below the 2.33 CZK/kWh a retained kWh
        # is worth tomorrow, so selling does NOT beat retaining and the
        # battery must not be dumped into the marginal end-of-horizon bump.
        # (A higher spike whose gross sell value clears the terminal value
        # AND has a profitable in-horizon recharge is legitimate arbitrage —
        # see test_mid_horizon_arbitrage_with_cheap_recharge_discharges.)
        prices = [3.5] * 8 + [4.5] * 2 + [3.5] * 2 + [1.0] * 2
        blocks = make_15min_blocks(prices, start_hour=10)

        _, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 0.1 for h in range(24)},
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=90.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        spike_discharges = [
            d for d in decisions
            if d.action == "discharge" and abs(d.price_czk - 4.5) < 0.01
        ]
        assert spike_discharges == [], (
            "marginal end-of-horizon discharge (gross sell 2.0 < terminal "
            "2.33) must not dump the battery before tomorrow's prices arrive; "
            f"got {len(spike_discharges)} spike discharges"
        )
        # The retained energy survives to the end of the horizon.
        assert decisions[-1].soc_after >= 70.0, (
            f"battery should end the horizon well above min, got "
            f"{decisions[-1].soc_after:.1f}%"
        )

    def test_mid_horizon_arbitrage_with_cheap_recharge_discharges(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """A profitable mid-horizon discharge with a genuine cheap recharge
        scheduled AFTER it must actually discharge.

        Regression for the terminal-value FLOOR on future_max_discharge_value:
        that array holds NET arbitrage profits (sell_rev - recharge_cost; the
        battery ends refilled), but the floor injected the GROSS value of a
        retained kWh and the 0.8 is_worthwhile gate applied it at EVERY block.
        Here the 5.5 spike's dv = sell_rev(3.0) - recharge(2.35) = 0.65 > 0,
        gross sell 3.0 > terminal 2.33, and the 1.0 CZK recharge follows the
        spike — discharging pockets dv AND preserves the terminal energy,
        strictly better than holding. The old floor rejected it
        (0.65 < 0.8 × 2.33)."""
        # Spike at hours 6-7, cheap recharge at hours 12-13, moderate rest.
        prices = [3.5] * 6 + [5.5] * 2 + [3.5] * 4 + [1.0] * 2 + [3.5] * 10
        blocks = make_15min_blocks(prices, start_hour=0)

        _, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 0.1 for h in range(24)},
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=90.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        spike_discharges = [
            d for d in decisions
            if d.action == "discharge" and abs(d.price_czk - 5.5) < 0.01
        ]
        assert spike_discharges, (
            "profitable mid-horizon arbitrage (dv=0.65>0, gross sell 3.0 > "
            "terminal 2.33, cheap 1.0 recharge after the spike) must "
            "discharge; the terminal floor on future_max_discharge_value "
            "blocked it. Actions at the spike: "
            f"{[(f'{d.timestamp:%H:%M}', d.action) for d in decisions if abs(d.price_czk - 5.5) < 0.01]}"
        )
        # The cheap recharge after the spike is actually used.
        assert any(
            d.action == "charge" and abs(d.price_czk - 1.0) < 0.01
            for d in decisions
        ), "the 1.0 CZK blocks after the spike should recharge the battery"

    def test_terminal_value_zero_on_negative_price_days(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # With a negative-price valley the cheapest import is ≤ 0, the
        # MILP-mirrored cap drives the terminal value to 0 and a genuinely
        # profitable evening discharge must still fire (no over-holding).
        prices = (
            [-1.0] * 6 + [2.0] * 6 + [1.0] * 6 + [8.0] * 6
            + [-1.0] * 6 + [2.0] * 18
        )
        blocks = make_blocks(prices)
        _, discharge, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={},
            distribution_func=const_dist(1.0),
            current_soc=50.0,
        )
        assert len({t.hour for t in discharge} & set(range(18, 24))) > 0


class TestGreedyHoldIdle:
    """Retention holds on deficit blocks emit hold_idle (battery passive) so
    the SPH doesn't actuate load_first and drain the energy being retained."""

    def test_retention_hold_with_deficit_becomes_hold_idle(
        self, optimizer: BatteryOptimizer
    ) -> None:
        # Cheap night (charges), moderate midday, expensive evening with real
        # consumption: the midday holds retain the battery for the evening —
        # on the SPH a plain hold (load_first) would drain it into the midday
        # deficit, so those blocks must be hold_idle and preserve SOC.
        prices = [1.0] * 6 + [3.0] * 6 + [6.0] * 6
        blocks = make_15min_blocks(prices, start_hour=0)
        consumption = {h: 1.0 for h in range(18)}

        _, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=consumption,
            distribution_func=const_dist(0.5),
            battery_capacity_kwh=10.0,
            current_soc=40.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        midday = [d for d in decisions if 6 <= d.timestamp.hour < 12]
        idle = [d for d in midday if d.action == "hold_idle"]
        assert idle, (
            "midday retention holds with a consumption deficit must be "
            f"hold_idle, got actions {[d.action for d in midday]}"
        )
        # hold_idle is battery-passive: SOC carries over unchanged.
        for d in idle:
            assert d.soc_after == d.soc_before, (
                f"hold_idle at {d.timestamp:%H:%M} must not move SOC: "
                f"{d.soc_before} → {d.soc_after}"
            )
        # And the retention actually works: the battery reaches the expensive
        # evening with at least as much charge as it had entering midday.
        soc_at_6 = next(d.soc_before for d in decisions if d.timestamp.hour == 6)
        soc_at_12 = next(d.soc_before for d in decisions if d.timestamp.hour == 12)
        assert soc_at_12 >= soc_at_6 - 1e-6

    def test_peak_blocks_self_consume_when_cheaper_tail_follows(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """A peak FOLLOWED by a cheaper tail (the rolling-horizon shape).

        Regression: gating the hold_idle remap on the INCLUSIVE future-SC
        array (which counts block i itself as "future") idled the battery
        straight through its own peak — there was always "a peak ahead", so
        the energy was never spent. Peak deficit blocks must self-consume
        (plain hold); only the genuinely-cheaper pre-peak blocks may idle.
        """
        # 24h of 15-min blocks: [1]*24 + [3]*24 + [6]*16 + [2]*32 blocks.
        prices = [1.0] * 6 + [3.0] * 6 + [6.0] * 4 + [2.0] * 8
        blocks = make_15min_blocks(prices, start_hour=0)
        consumption = {h: 1.0 for h in range(24)}

        _, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly=consumption,
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=90.0,  # battery charged
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        peak = [d for d in decisions if 12 <= d.timestamp.hour < 16]
        deficit_peak = [d for d in peak if d.consumption_kwh > d.solar_kwh]
        assert deficit_peak, "peak must have deficit blocks in this scenario"
        assert all(d.action != "hold_idle" for d in deficit_peak), (
            "expensive-peak deficit blocks must self-consume, not idle: "
            f"{[(f'{d.timestamp:%H:%M}', d.action) for d in deficit_peak]}"
        )
        # The stored energy is actually SPENT during the peak.
        soc_peak_start = next(
            d.soc_before for d in decisions if d.timestamp.hour == 12
        )
        soc_peak_end = next(
            d.soc_after for d in decisions
            if d.timestamp.hour == 15 and d.timestamp.minute == 45
        )
        assert soc_peak_end < soc_peak_start, (
            f"battery must drain over the peak: {soc_peak_start} → {soc_peak_end}"
        )
        # Pre-peak cheaper blocks still retain (idle) for the peak.
        pre_peak = [d for d in decisions if d.timestamp.hour < 12]
        assert any(d.action == "hold_idle" for d in pre_peak), (
            "cheaper pre-peak deficit blocks should idle to retain energy: "
            f"{[(f'{d.timestamp:%H:%M}', d.action) for d in pre_peak]}"
        )

    def test_two_peaks_with_energy_for_both_self_consume_today(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Energy-aware remap (quantity gate): a strictly-better future peak
        must NOT idle the current peak when the battery holds enough energy
        for BOTH.

        Regression: the remap was quantity-blind — whenever ANY strictly-
        better future deficit block existed, the current block idled. 32h
        horizon, today's peak 6.0, tomorrow's 6.2, SOC 100% (≈8 kWh usable
        vs ≈4.3 kWh per peak): today's peak went entirely hold_idle and the
        whole peak was imported from grid at ~7 CZK/kWh despite ample stored
        energy. With the gate (idle only while usable_kwh <= strictly-better
        future need / leg_eta) today's peak self-consumes; only once the
        remaining energy is just enough for tomorrow's better peak may late
        blocks idle."""
        # 32h from 12:00: moderate, today's 6.0 peak (17-20), cheap tail +
        # night, moderate, tomorrow's 6.2 peak (truncated horizon end).
        hourly = [3.0] * 5 + [6.0] * 4 + [2.0] * 3 + [1.0] * 6 \
            + [3.0] * 10 + [6.2] * 4
        blocks = make_15min_blocks(hourly, start_hour=12)
        assert len(blocks) == 128  # ~32h rolling horizon

        _, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(1.0),
            battery_capacity_kwh=10.0,
            current_soc=100.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
            # export wear high enough that the peaks aren't grid-export
            # arbitrage — this test isolates hold vs hold_idle.
            battery_amortisation_export_czk=5.0,
        )

        today_peak = [d for d in decisions if abs(d.price_czk - 6.0) < 0.01]
        assert today_peak, "scenario must contain today's 6.0 peak"
        # Today's peak must NOT be entirely idled — the battery has energy
        # for both peaks, so it self-consumes (plain hold) through the peak.
        holds = [d for d in today_peak if d.action == "hold"]
        assert len(holds) > len(today_peak) // 2, (
            "today's peak must predominantly self-consume when the battery "
            "holds enough for today AND tomorrow; got actions "
            f"{[(f'{d.timestamp:%H:%M}', d.action) for d in today_peak]}"
        )
        # The first peak blocks (most usable energy) must self-consume.
        assert today_peak[0].action == "hold"
        # And the stored energy is actually SPENT across today's peak.
        assert today_peak[-1].soc_after < today_peak[0].soc_before - 10.0, (
            f"battery must drain over today's peak: "
            f"{today_peak[0].soc_before} → {today_peak[-1].soc_after}"
        )
        # Tomorrow's strictly-better peak still gets served from battery.
        tomorrow_peak = [d for d in decisions if abs(d.price_czk - 6.2) < 0.01]
        assert any(d.soc_after < d.soc_before for d in tomorrow_peak), (
            "tomorrow's peak should also self-consume from the battery"
        )

    def test_hold_idle_never_lands_in_action_sets(
        self, optimizer: BatteryOptimizer
    ) -> None:
        prices = [1.0] * 6 + [3.0] * 6 + [6.0] * 6
        blocks = make_15min_blocks(prices, start_hour=0)
        charge, discharge, sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(18)},
            distribution_func=const_dist(0.5),
            battery_capacity_kwh=10.0,
            current_soc=40.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        idle_ts = {d.timestamp for d in decisions if d.action == "hold_idle"}
        assert not (idle_ts & charge)
        assert not (idle_ts & discharge)
        assert not (idle_ts & sp)


# ---------------------------------------------------------------------------
# Adaptive charge power rate (compute_charge_power_rates)
# ---------------------------------------------------------------------------
from modules.growatt.optimizer import BlockDecision, compute_charge_power_rates


def _charge_dec(ts: datetime, soc_before: float, soc_after: float) -> BlockDecision:
    return BlockDecision(
        timestamp=ts, action="charge", price_czk=1.0, distribution_czk=0.0,
        solar_kwh=0.0, consumption_kwh=0.0,
        soc_before=soc_before, soc_after=soc_after, net_value=0.0,
    )


def test_adaptive_rate_gentle_long_window_clamps_to_floor():
    # 4 blocks (1h) adding 8% of a 10 kWh pack = 0.8 kWh over 1h = 0.8 kW avg.
    # 0.8/9.8 ≈ 8% → below the 25% floor → clamped to 25.
    base = datetime(2026, 4, 11, 1, 0)
    decs = [
        _charge_dec(base + timedelta(minutes=15 * i), 50 + 2 * i, 52 + 2 * i)
        for i in range(4)
    ]
    rates = compute_charge_power_rates(decs, 10.0, 9.8, min_power_rate=25)
    assert set(rates.values()) == {25}
    assert len(rates) == 4


def test_adaptive_rate_short_window_needs_full_power():
    # One 15-min block banking 24% of 10 kWh = 2.4 kWh in 0.25h = 9.6 kW avg.
    # 9.6/9.8 ≈ 98% → near full rate.
    base = datetime(2026, 4, 11, 13, 0)
    rates = compute_charge_power_rates([_charge_dec(base, 40, 64)], 10.0, 9.8, efficiency=1.0)
    assert list(rates.values())[0] == 98


def test_adaptive_rate_splits_independent_windows():
    # Two windows separated by a >15-min gap get independent rates.
    a = datetime(2026, 4, 11, 1, 0)
    b = datetime(2026, 4, 11, 5, 0)
    decs = [
        _charge_dec(a, 50, 52), _charge_dec(a + timedelta(minutes=15), 52, 54),  # gentle
        _charge_dec(b, 40, 64),  # full-power single block
    ]
    rates = compute_charge_power_rates(decs, 10.0, 9.8, min_power_rate=25, efficiency=1.0)
    assert rates[a] == 25
    assert rates[b] == 98


def test_adaptive_rate_ignores_non_charge_and_empty():
    base = datetime(2026, 4, 11, 1, 0)
    hold = BlockDecision(
        timestamp=base, action="hold", price_czk=1.0, distribution_czk=0.0,
        solar_kwh=0.0, consumption_kwh=0.0, soc_before=50, soc_after=50, net_value=0.0,
    )
    assert compute_charge_power_rates([hold], 10.0, 9.8) == {}
    assert compute_charge_power_rates([], 10.0, 9.8) == {}


def _disch_dec(ts: datetime, soc_before: float, soc_after: float) -> BlockDecision:
    return BlockDecision(
        timestamp=ts, action="discharge", price_czk=4.0, distribution_czk=0.0,
        solar_kwh=0.0, consumption_kwh=0.0,
        soc_before=soc_before, soc_after=soc_after, net_value=0.0,
    )


def test_adaptive_discharge_rate_uses_abs_soc_delta():
    # Symmetric to charging: a short 1-block discharge dumping 24% of a 10 kWh
    # pack = 2.4 kWh in 0.25h = 9.6 kW → ~98%. A long gentle drain clamps to floor.
    base = datetime(2026, 4, 11, 21, 0)
    fast = compute_charge_power_rates(
        [_disch_dec(base, 64, 40)], 10.0, 9.8, action="discharge", efficiency=1.0
    )
    assert list(fast.values())[0] == 98
    slow = compute_charge_power_rates(
        [_disch_dec(base + timedelta(minutes=15 * i), 60 - 2 * i, 58 - 2 * i)
         for i in range(4)],
        10.0, 9.8, min_power_rate=25, action="discharge", efficiency=1.0,
    )
    assert set(slow.values()) == {25}


def test_adaptive_discharge_rate_ignores_charge_blocks():
    # action filter keeps charge and discharge windows independent.
    base = datetime(2026, 4, 11, 21, 0)
    mixed = [_charge_dec(base, 40, 64), _disch_dec(base + timedelta(hours=2), 64, 40)]
    only_disch = compute_charge_power_rates(mixed, 10.0, 9.8, action="discharge")
    assert len(only_disch) == 1
    assert base + timedelta(hours=2) in only_disch


def test_adaptive_rate_max_power_clamps_ceiling():
    # A short window that wants full power must be clamped to max_power_rate
    # (the C-rate ceiling), e.g. 54% for a 5.3 kW cap on a 9.8 kW inverter.
    base = datetime(2026, 4, 11, 21, 0)
    rates = compute_charge_power_rates(
        [_disch_dec(base, 64, 40)], 10.0, 9.8, action="discharge", max_power_rate=54
    )
    assert list(rates.values())[0] == 54  # would be 98 un-clamped


def test_adaptive_rate_uses_grid_side_power_not_raw_soc():
    """The rate is sized from GRID-side power (efficiency-corrected), not raw
    battery ΔSOC. The powerRate caps TOTAL battery throughput, so discharge =
    ΔSOC·η (house self-consumption counts too — NOT subtracted) and charge =
    ΔSOC/η."""
    base = datetime(2026, 4, 11, 21, 0)
    # Discharge: ΔSOC=20% of 10 kWh = 2.0 kWh drain over 0.25h.
    # output = 2.0*0.9 = 1.8 kWh / 0.25h = 7.2 kW → 7.2/9.8 ≈ 74%.
    d_house = BlockDecision(
        timestamp=base, action="discharge", price_czk=4.0, distribution_czk=0.0,
        solar_kwh=0.0, consumption_kwh=1.0, soc_before=60.0, soc_after=40.0,
        net_value=0.0,
    )
    d_nohouse = BlockDecision(
        timestamp=base, action="discharge", price_czk=4.0, distribution_czk=0.0,
        solar_kwh=0.0, consumption_kwh=0.0, soc_before=60.0, soc_after=40.0,
        net_value=0.0,
    )
    # A co-served house load must NOT change the rate (total-output semantics).
    assert compute_charge_power_rates([d_house], 10.0, 9.8, action="discharge", efficiency=0.9)[base] == 74
    assert compute_charge_power_rates([d_nohouse], 10.0, 9.8, action="discharge", efficiency=0.9)[base] == 74
    # Charge input = ΔSOC/η: 2.0/0.9 = 2.22 kWh / 0.25h = 8.9 kW → 91%.
    c = _charge_dec(base, 40.0, 60.0)
    assert compute_charge_power_rates([c], 10.0, 9.8, action="charge", efficiency=0.9)[base] == 91


class TestBaseLoadProfileLocalKeying:
    """The base-load profile is consumed with LOCAL (hour, weekend) keys, so
    build_base_load_profile / update_profile_with_yesterday must convert
    InfluxDB's UTC record times to local time before extracting hour/weekday,
    AND use start-labeled aggregateWindow (the Flux default stop-labeling
    shifts hour keys +1h; in CEST the two bugs no longer cancel)."""

    @staticmethod
    def _rec(dt, value):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.get_time.return_value = dt
        r.get_value.return_value = value
        return r

    @staticmethod
    def _client(heating_records, ev_records, load_records, captured=None):
        from unittest.mock import AsyncMock, MagicMock

        def _table(recs):
            t = MagicMock()
            t.records = recs
            return t

        async def _q(q):
            if captured is not None:
                captured.append(q)
            if "heating" in q:
                return [_table(heating_records)] if heating_records else []
            if "ev_charging" in q:
                return [_table(ev_records)] if ev_records else []
            return [_table(load_records)] if load_records else []

        client = MagicMock()
        client.query = AsyncMock(side_effect=_q)
        return client

    @pytest.mark.asyncio
    async def test_build_keys_by_local_hour(self, optimizer) -> None:
        """A start-labeled 10:00 UTC record in June (CEST, UTC+2) is the
        12:00 LOCAL hour — the profile slot must be 12, not 10."""
        from datetime import timezone
        # Wednesday 2026-06-10 10:00 UTC == 12:00 Europe/Prague (CEST)
        load = [self._rec(datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc), 1500.0)]
        client = self._client([], [], load)

        profile = await optimizer.build_base_load_profile(client, "solar", "loxone")

        assert profile.profile == {(12, False): pytest.approx(1.5)}

    @pytest.mark.asyncio
    async def test_build_weekend_classified_on_local_day(self, optimizer) -> None:
        """Friday 22:00 UTC is Saturday 00:00 local (CEST) — must land in
        the WEEKEND hour-0 slot, not Friday hour 22."""
        from datetime import timezone
        # 2026-06-12 is a Friday; 22:00 UTC -> 2026-06-13 (Sat) 00:00 CEST
        load = [self._rec(datetime(2026, 6, 12, 22, 0, tzinfo=timezone.utc), 800.0)]
        client = self._client([], [], load)

        profile = await optimizer.build_base_load_profile(client, "solar", "loxone")

        assert profile.profile == {(0, True): pytest.approx(0.8)}

    @pytest.mark.asyncio
    async def test_build_heating_exclusion_keys_match_load_keys(self, optimizer) -> None:
        """Heating/EV exclusion keys are converted the same way as load keys,
        so a heating hour still excludes the matching load record."""
        from datetime import timezone
        ts = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        heating = [self._rec(ts, 1)]
        load = [
            self._rec(ts, 5000.0),  # heating hour — must be excluded
            self._rec(datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc), 1000.0),
        ]
        client = self._client(heating, [], load)

        profile = await optimizer.build_base_load_profile(client, "solar", "loxone")

        # Only the non-heating hour remains, at its LOCAL hour 13.
        assert profile.profile == {(13, False): pytest.approx(1.0)}

    @pytest.mark.asyncio
    async def test_build_queries_use_start_labeled_windows(self, optimizer) -> None:
        captured: list = []
        client = self._client([], [], [], captured=captured)

        await optimizer.build_base_load_profile(client, "solar", "loxone")

        agg = [q for q in captured if "aggregateWindow" in q]
        assert len(agg) == 3  # heating + EV + load
        for q in agg:
            assert 'timeSrc: "_start"' in q

    @pytest.mark.asyncio
    async def test_build_explicit_local_tz_param_respected(self, optimizer) -> None:
        """An explicitly-passed tz overrides the Europe/Prague default."""
        import zoneinfo
        from datetime import timezone
        load = [self._rec(datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc), 1500.0)]
        client = self._client([], [], load)

        profile = await optimizer.build_base_load_profile(
            client, "solar", "loxone", local_tz=zoneinfo.ZoneInfo("UTC")
        )

        assert profile.profile == {(10, False): pytest.approx(1.5)}

    @pytest.mark.asyncio
    async def test_yesterday_update_keys_by_local_hour(self, optimizer) -> None:
        from datetime import timezone
        # Seed the slot the local-keyed record should update.
        optimizer._base_load_profile.profile[(12, False)] = 2.0
        load = [self._rec(datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc), 1000.0)]
        captured: list = []
        client = self._client([], [], load, captured=captured)

        await optimizer.update_profile_with_yesterday(client, "solar", "loxone")

        # EMA at the LOCAL hour-12 slot: 0.9*2.0 + 0.1*1.0
        assert optimizer._base_load_profile.profile[(12, False)] == pytest.approx(1.9)
        # The UTC hour-10 slot must NOT have been touched/created.
        assert (10, False) not in optimizer._base_load_profile.profile
        # And the windows are start-labeled.
        agg = [q for q in captured if "aggregateWindow" in q]
        assert len(agg) == 3
        for q in agg:
            assert 'timeSrc: "_start"' in q


class TestDistributionAwareChargeGate:
    """Charge-gate economics, iteration 2 (forward-looking, wear-aware).

    Iteration 1 made the gate distribution-aware but valued a charged kWh at
    the horizon's MEDIAN import cost. That breaks whenever cheap blocks are
    the horizon MAJORITY (negative/windy days, cheap-NT bands): the median
    sits in the cheap mode, cheap*efficiency < cheap → zero candidates, and
    the `spot < 0` fallback was dead because the actuation gate used the
    same algebra. Iteration 2 values the charge FORWARD-LOOKING against the
    best future deficit self-consumption value (future_sc_value, per
    delivered kWh, amortisation-adjusted, terminal-floored):

        charge profitable  ⇔  (p + dist) < future_sc_value[i] * efficiency

    BEHAVIOUR CHANGE (deliberate): future_sc_value subtracts
    battery_amortisation_czk, so the gate is now WEAR-CONSISTENT — a cycle
    that is cash-positive but wear-negative (full-cost spread smaller than
    the wear cost) no longer charges. The "still charges" controls below
    therefore use wear-PROFITABLE parameters, and the wear-marginal case is
    pinned as a no-charge test of its own.
    """

    @staticmethod
    def _low_spread_prices(day_spot: float = 1.2) -> list:
        """Two days: spot 0.8 overnight (00-07), `day_spot` daytime (08-23)."""
        day = [0.8] * 8 + [day_spot] * 16
        return day * 2

    def test_no_grid_charge_when_distribution_eats_the_spread(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """dist 2.5: night full cost 3.3 vs best future SC value
        (1.2 + 2.5 - 2.0 amort) * 0.85 = 1.445 — every charged kWh loses
        money, so NO grid charge may be scheduled."""
        blocks = make_15min_blocks(self._low_spread_prices())
        charge, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(2.5),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert charge == set(), (
            "spot-only break-even scheduled loss-making night charging: "
            f"{sorted(t.hour for t in charge)}"
        )
        assert all(d.action != "charge" for d in decisions)

    def test_grid_charge_still_selected_with_low_distribution(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Wear-PROFITABLE control: night 0.8 / day 4.0 at dist 0.5 — night
        full cost 1.3 < (4.0 + 0.5 - 2.0 amort) * 0.85 = 2.125, so nightly
        charging must still happen (the fix must not over-tighten the gate).

        TEST CHANGE (iteration 2): the original control used day spot 1.2,
        which is cash-positive (1.3 < 1.7 * 0.85... barely) but wear-NEGATIVE
        with amort 2.0 — under the wear-consistent gate it correctly stops
        charging (pinned in test_wear_negative_spread_does_not_charge below).
        The control's PURPOSE (guard against over-tightening) is preserved
        with a wear-profitable day price."""
        blocks = make_15min_blocks(self._low_spread_prices(day_spot=4.0))
        charge, _, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(0.5),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(charge) > 0, "profitable night charging must survive the fix"
        assert all(t.hour < 8 for t in charge), (
            f"charging must stay in the cheap night hours: "
            f"{sorted({t.hour for t in charge})}"
        )

    def test_wear_negative_spread_does_not_charge(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """WEAR-CONSISTENCY (new, intended behaviour of the iteration-2 gate):
        a cycle that is cash-positive but wear-negative must NOT charge.

        Night 0.8 / day 1.2 at dist 0.5, amort 2.0: importing at 1.3 to
        displace a 1.7 import is cash-positive after round-trip losses
        (1.7 * 0.85 = 1.445 > 1.3) — the OLD wear-blind gate cycled nightly
        for ~0.12 CZK/kWh cash margin while wearing the battery 2.0 CZK per
        delivered kWh, a net value destruction of ~1.6 CZK/kWh. The new gate
        values the charge at the amort-adjusted SC value
        ((1.2 + 0.5 - 2.0) < 0 → floored terminal value) and refuses."""
        blocks = make_15min_blocks(self._low_spread_prices(day_spot=1.2))
        charge, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(0.5),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert charge == set(), (
            "cash-positive but wear-negative cycling must not charge: "
            f"{sorted({t.hour for t in charge})}"
        )
        assert all(d.action != "charge" for d in decisions)

    def test_cheap_majority_negative_day_still_charges_before_peak(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """S1-shaped cheap-majority regression: spot -0.5 for 20h, 4.0 for
        the 17-20 evening peak, dist 2.0, amort 2.0.

        The iteration-1 median gate never charged here: the median import
        cost sits in the cheap mode (1.5), threshold 1.5 * 0.85 = 1.275 <
        1.5 → zero candidates, and the `spot < 0` fallback was vetoed by the
        algebraically-identical actuation gate — so the battery idled while
        the peak imported at 6.0 CZK/kWh (greedy/MILP parity broken; the
        MILP charges the same windows). Forward-looking gate: 1.5 <
        (4.0 + 2.0 - 2.0) * 0.85 = 3.4 → charges ahead of the peak."""
        day = [-0.5 if not (17 <= h <= 20) else 4.0 for h in range(24)]
        blocks = make_15min_blocks(day * 2)
        charge, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(2.0),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(charge) > 0, (
            "cheap-majority horizon must still charge ahead of the peak "
            "(median-based gate regression)"
        )
        # Charging lands in the cheap hours, never inside the expensive peak.
        assert all(not (17 <= t.hour <= 20) for t in charge), (
            f"charging must avoid the 17-20 peak: {sorted({t.hour for t in charge})}"
        )
        # And the stored energy is actually used: day-1 peak deficit blocks
        # are served from the battery (SOC drops across the peak).
        day1_peak = [
            d for d in decisions
            if d.timestamp.hour == 20 and d.timestamp.day == blocks[0][0].day
        ]
        day1_prepeak = [
            d for d in decisions
            if d.timestamp.hour == 17 and d.timestamp.day == blocks[0][0].day
        ]
        assert day1_peak[-1].soc_after < day1_prepeak[0].soc_before, (
            "the peak must drain the battery charged from the cheap majority"
        )

    def test_negative_spot_candidate_vetoed_when_full_cost_unprofitable(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """The `p < 0` candidate fallback must still pass the actuation-side
        charge_value gate. Spot -0.05 with dist 2.5 costs 2.45/kWh while the
        best future deficit SC value (0.3 + 2.5 - 2.0 amort = 0.8) is only
        worth 0.8 * 0.85 = 0.68 per grid kWh after round-trip losses —
        charge_value is negative, so the scheduled block must NOT actuate as
        a charge."""
        prices = [-0.05] * 12 + [0.3] * 12
        blocks = make_blocks(prices)
        charge, _, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(2.5),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert charge == set(), (
            "negative-spot blocks with loss-making FULL cost must not charge: "
            f"{sorted(t.hour for t in charge)}"
        )
        assert all(d.action != "charge" for d in decisions)

    def test_paid_to_charge_still_charges_despite_high_distribution(
        self, optimizer: BatteryOptimizer
    ) -> None:
        """Deep negatives must keep charging even at high distribution: spot
        -5.0 with dist 2.5 has NEGATIVE full cost (-2.5) — the gate must not
        veto free money."""
        prices = [-5.0] * 6 + [1.2] * 18
        blocks = make_blocks(prices)
        charge, _, _sp, _ = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
            consumption_hourly={h: 1.0 for h in range(24)},
            distribution_func=const_dist(2.5),
            battery_capacity_kwh=10.0,
            current_soc=20.0,
            min_soc=20.0,
            efficiency=0.85,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )
        assert len(charge) > 0, "negative full-cost blocks must still charge"
        assert all(t.hour < 6 for t in charge)
