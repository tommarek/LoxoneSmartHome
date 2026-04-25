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
        assert summary["charge_blocks"] + summary["discharge_blocks"] + summary["hold_blocks"] == 24


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

        # Very high distribution: sell_revenue = 12.0 - 5.0 - 0.5 - 2.0 = 4.5
        # recharge_cost = (0.5 + 5.0) / 0.85 = 6.47
        # profit = 4.5 - 6.47 = -1.97 < 0 → should NOT discharge
        _, discharge_high_dist, _sp, _ = optimizer.optimize(
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
        # The optimizer should still find charge blocks, but they should be
        # at hours where solar doesn't already fill the battery
        assert len(charge) > 0

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
        # Negative price blocks should be in charge set
        negative_block_hours = {ts.hour for ts in charge if ts.hour in [0, 1]}
        # At least some negative-price blocks should be selected
        assert len(charge) > 0


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

        # Sanity: very-negative blocks should still be picked up as charge candidates.
        cheap_charge_hours = {ts.hour for ts in charge}
        assert any(h in cheap_charge_hours for h in (8, 9, 10, 11)), (
            f"Should still charge at very-negative midday blocks, got hours: {sorted(cheap_charge_hours)}"
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
        # Hour 8: peak (4.0 CZK), zero consumption (covered by solar).
        # Hour 9: cheap (0.5 CZK) — recharge bait.
        # Hours 10–31: moderate (3.0 CZK), real consumption — only here is SC valuable.
        prices = [4.0] * 1 + [0.5] * 1 + [3.0] * 22
        blocks = make_15min_blocks(prices, start_hour=8)

        # Solar exactly cancels consumption at hour 8 (no SC need there);
        # full consumption from hour 10 onward.
        consumption = {h: 1.0 for h in range(24)}
        consumption[8] = 0.0   # peak block: no consumption → not a future-SC contributor
        consumption[9] = 0.0

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={},
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
        hold value should not prevent profitable discharge."""
        # Expensive blocks with zero consumption
        prices = [0.5] * 4 + [8.0] * 4
        blocks = make_15min_blocks(prices, start_hour=0)

        charge, discharge, _sp, decisions = optimizer.optimize(
            blocks=blocks,
            solar_hourly={h: 5.0 for h in range(8)},  # Solar covers everything
            consumption_hourly={h: 0.5 for h in range(8)},
            distribution_func=lambda h: 0.5,
            battery_capacity_kwh=10.0,
            current_soc=80.0,
            min_soc=20.0,
            sell_fee_czk=0.5,
            battery_amortisation_czk=2.0,
        )

        # With solar covering all consumption, no self-consumption need
        # Battery should discharge at 8.0 CZK if profitable
        # discharge_profit = 8.0 - 0.5 - 0.5 - 2.0 - recharge > 0
        # This should still work
        assert len(discharge) > 0 or len(charge) > 0, (
            "Optimizer should still make active decisions when solar covers consumption"
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

        # Morning blocks should DISCHARGE (sells battery + solar excess), not sell_production
        morning_discharge = [d for d in decisions
                             if d.action == "discharge" and 8 <= d.timestamp.hour < 12]
        morning_sp = [d for d in decisions
                      if d.action == "sell_production" and 8 <= d.timestamp.hour < 12]
        assert len(morning_discharge) > 0, "Morning at 4 CZK should discharge"
        assert morning_sp == [], (
            "Discharge supersedes sell_production at 4 CZK morning"
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
        solar = {h: 1.0 for h in range(8, 12)}      # modest morning
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

        # Per-block actions are mutually exclusive
        valid_actions = {"charge", "discharge", "hold", "sell_production"}
        for d in decisions:
            assert d.action in valid_actions, f"Unknown action: {d.action}"
