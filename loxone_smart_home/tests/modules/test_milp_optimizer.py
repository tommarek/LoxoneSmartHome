"""Tests for the MILP battery optimizer.

These exercise the real CBC solver when PuLP is installed; otherwise they
skip (the controller falls back to the greedy engine in that case).
"""

from datetime import datetime, timedelta
from typing import List, Tuple

import pytest

from modules.growatt.milp_optimizer import MILPBatteryOptimizer, PULP_AVAILABLE

pytestmark = pytest.mark.skipif(
    not PULP_AVAILABLE, reason="PuLP/CBC not installed"
)


def make_blocks(prices: List[float], start_hour: int = 0) -> List[Tuple[datetime, float]]:
    base = datetime(2025, 6, 1, start_hour, 0, 0)
    return [(base + timedelta(minutes=15 * i), p) for i, p in enumerate(prices)]


def flat_dist(tariff: float = 1.0):
    return lambda _hour: tariff


def test_returns_four_tuple_and_decision_per_block():
    blocks = make_blocks([2.0] * 8)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
        current_soc=50.0,
    )
    assert isinstance(charge, set)
    assert isinstance(discharge, set)
    assert isinstance(sp, set)
    assert len(decisions) == len(blocks)


def test_charges_when_cheap_then_discharges_when_expensive():
    # First 8 blocks dirt cheap (charge), last 8 very expensive (discharge).
    blocks = make_blocks([-0.5] * 8 + [15.0] * 8)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
        current_soc=30.0,
        min_soc=20.0,
        max_soc=100.0,
        charge_rate_kw=5.0,
        discharge_rate_kw=5.0,
        discharge_power_pct=100.0,
    )
    assert len(charge) > 0, "should charge during cheap blocks"
    assert len(discharge) > 0, "should discharge during expensive blocks"
    # All charges precede all discharges chronologically.
    assert max(charge) < min(discharge)


def test_never_discharges_when_revenue_negative():
    # Price below the amortisation+fee+dist floor everywhere → selling loses
    # money, so discharge must never be scheduled.
    blocks = make_blocks([0.5] * 12)
    opt = MILPBatteryOptimizer()
    _, discharge, _, _ = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
        current_soc=100.0,
        battery_amortisation_czk=2.0,
        sell_fee_czk=0.5,
    )
    assert discharge == set()


def test_solar_charges_battery_on_hold_blocks():
    # No grid charge needed: pure solar surplus should refill the battery so
    # SOC rises across hold blocks (the missing-solar-charge bug would keep
    # SOC flat/declining). Prices flat & low so discharge/sellprod inactive.
    blocks = make_blocks([1.0] * 8)
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={h: 8.0 for h in range(24)},   # 2 kWh surplus/block
        consumption_hourly={h: 0.0 for h in range(24)},
        distribution_func=flat_dist(1.0),
        current_soc=30.0,
        min_soc=20.0,
        max_soc=100.0,
    )
    # SOC should climb from solar with no grid charge action scheduled.
    assert decisions[-1].soc_after > decisions[0].soc_before + 1.0
    assert all(d.action != "charge" for d in decisions), \
        "should not buy grid charge when free solar is available"


def test_no_battery_drain_when_solar_covers_load():
    # Battery covers only the NET deficit (load - solar). When solar exactly
    # covers the load there is ZERO net deficit, so the battery must not be
    # drained to serve the house — SOC stays at/above the start. The old bug
    # drained by the FULL load regardless of solar, which would crater SOC.
    # Prices flat & low so there's no discharge/export incentive either.
    # (Deterministic: a zero-deficit scenario has no degenerate discharge
    # optimum, unlike quantitative drain comparisons under cheap recharge.)
    blocks = make_blocks([1.0] * 8)
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={h: 4.0 for h in range(24)},        # 1.0 kWh/block
        consumption_hourly={h: 4.0 for h in range(24)},  # 1.0 kWh/block
        distribution_func=flat_dist(1.0),
        current_soc=50.0, min_soc=20.0, max_soc=100.0,
        battery_capacity_kwh=10.0,
    )
    # No net deficit and no export incentive → battery is left alone.
    drop = decisions[0].soc_before - decisions[-1].soc_after
    assert drop <= 2.0, f"battery drained {drop:.1f}% despite zero net deficit"
    assert all(d.action != "discharge" for d in decisions)


def test_greedy_fallback_returns_full_schedule():
    # When the MILP solve is non-optimal/infeasible/raises, optimize()
    # delegates to _greedy_fallback. Exercise that seam directly (it is what
    # the non-optimal branch calls) so the test is deterministic and not
    # subject to CBC-subprocess behaviour under pytest output capture.
    blocks = make_blocks([-1.0] * 6 + [10.0] * 6)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt._greedy_fallback(
        blocks, {}, {}, flat_dist(1.0),
        battery_capacity_kwh=10.0, current_soc=50.0, min_soc=20.0,
        max_soc=100.0, charge_rate_kw=5.0, discharge_rate_kw=5.0,
        discharge_power_pct=100.0, efficiency=0.85, sell_fee_czk=0.5,
        battery_amortisation_czk=2.0,
    )
    # Greedy fallback must produce a full decision list, not an empty schedule,
    # and adopt it as the engine's own last-decisions (the fallback contract).
    assert len(decisions) == len(blocks)
    assert opt._last_decisions == decisions
    # It should at least exploit the dirt-cheap (-1 CZK) blocks to charge.
    assert len(charge) > 0


def test_soc_stays_within_bounds():
    blocks = make_blocks([-1.0] * 4 + [20.0] * 4 + [-1.0] * 4)
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
        current_soc=50.0,
        min_soc=20.0,
        max_soc=90.0,
        charge_rate_kw=10.0,
        discharge_rate_kw=10.0,
        discharge_power_pct=100.0,
    )
    for d in decisions:
        assert d.soc_after >= 20.0 - 1e-6
        assert d.soc_after <= 90.0 + 1e-6


def test_empty_blocks_returns_empty():
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=[],
        solar_hourly={},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
    )
    assert (charge, discharge, sp, decisions) == (set(), set(), set(), [])


def test_mutual_exclusion_no_block_does_two_things():
    blocks = make_blocks([-0.5] * 4 + [15.0] * 4)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, _ = opt.optimize(
        blocks=blocks,
        solar_hourly={h: 5.0 for h in range(24)},
        consumption_hourly={},
        distribution_func=flat_dist(1.0),
        current_soc=50.0,
        charge_rate_kw=5.0,
        discharge_rate_kw=5.0,
        discharge_power_pct=100.0,
    )
    # No timestamp may appear in more than one action set.
    assert not (charge & discharge)
    assert not (charge & sp)
    assert not (discharge & sp)
