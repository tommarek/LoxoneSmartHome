"""Tests for the MILP battery optimizer.

These exercise the real CBC solver when PuLP is installed; otherwise they
skip (the controller falls back to the greedy engine in that case).
"""

from datetime import datetime, time, timedelta
from typing import List, Tuple

import pytest

from modules.growatt.deferrable_loads import DeferrableLoad
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


def test_does_not_grid_charge_at_peak_to_meet_reserve():
    """Regression for the live peak-charge incident: an evening price peak
    followed by cheap night blocks, with steady load creating a reserve need.
    The price-aware reserve penalty must keep the optimizer from grid-charging
    at the peak — any grid charge must land in the cheap blocks, never the peak."""
    blocks = make_blocks([9.0] * 8 + [1.0] * 8, start_hour=18)  # peak then cheap
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={h: 1.0 for h in range(24)},  # steady load → reserve need
        distribution_func=flat_dist(1.0),
        current_soc=25.0, min_soc=20.0, max_soc=100.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
    )
    price_by_ts = dict(blocks)
    peak = [d for d in decisions if price_by_ts[d.timestamp] >= 9.0]
    # The incident signature was SOC RISING across the peak (grid-charged to ~80%).
    # The correct plan net-discharges (or holds) across the peak and refills from
    # the cheap blocks — so SOC must not rise across the peak window.
    assert peak[-1].soc_after <= peak[0].soc_before + 1.0, (
        "battery must not grid-charge UP during the evening peak for the reserve"
    )


def test_no_grid_export_below_export_floor():
    """STRICT export floor: with a positive battery-export margin but a spot
    price below the export floor, the optimizer schedules NO grid export."""
    blocks = make_blocks([4.0] * 8, start_hour=18)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={h: 8.0 for h in range(24)},
        consumption_hourly={h: 0.5 for h in range(24)},
        distribution_func=flat_dist(1.0),
        current_soc=90.0, min_soc=20.0, max_soc=100.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        battery_amortisation_czk=0.0,  # export would otherwise be very attractive
        export_price_min=5.0,          # floor ABOVE the 4.0 spot → no export
    )
    assert discharge == set(), "no battery export below the export floor"
    assert sp == set(), "no sell-production below the export floor"


def test_inverter_off_below_threshold():
    """STRICT PV-off: in a block whose spot price is below the inverter-off
    price (deeply negative), the inverter is off — no charge/discharge/sell —
    even though a negative price would otherwise make grid-charging attractive."""
    blocks = make_blocks([-3.0] * 4 + [5.0] * 4, start_hour=12)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={h: 8.0 for h in range(24)},
        consumption_hourly={h: 1.0 for h in range(24)},
        distribution_func=flat_dist(1.0),
        current_soc=50.0, min_soc=20.0, max_soc=100.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        inverter_off_price=-2.0,
    )
    price_by_ts = dict(blocks)
    for ts in (t for t in price_by_ts if price_by_ts[t] < -2.0):
        assert ts not in charge and ts not in discharge and ts not in sp, (
            "inverter must be off (no battery/export) below the off-price"
        )


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
    # SOC flat/declining). Prices flat & BELOW the export floor (0.35) so export
    # is inactive and the surplus must store rather than sell.
    blocks = make_blocks([0.2] * 8)
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


def _ev(energy_kwh=2.5, power_kw=5.0, start=0, end=2, interruptible=True):
    return DeferrableLoad(
        name="ev", energy_required_kwh=energy_kwh, power_kw=power_kw,
        earliest_start=time(start, 0), latest_end=time(end, 0),
        interruptible=interruptible,
    )


def test_deferrable_interruptible_picks_cheapest_blocks():
    # Cheapest blocks are indices 4,5 (01:00, 01:15). A 2.5 kWh @ 5 kW EV
    # needs 2 blocks and must land there.
    blocks = make_blocks([5.0, 5.0, 5.0, 5.0, -1.0, -1.0, 5.0, 5.0])
    opt = MILPBatteryOptimizer()
    opt.optimize(
        blocks=blocks, solar_hourly={}, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=50.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        deferrable_loads=[_ev()],
    )
    sched = opt._last_deferrable_schedules[0]
    assert sched.scheduled_blocks == 2
    assert {b[0] for b in sched.blocks} == {"01:00", "01:15"}


def test_deferrable_charges_into_solar_surplus():
    # Flat grid price everywhere, but block 4 has free solar surplus the full
    # battery can't absorb (so it would otherwise curtail). A high sell fee
    # kills export, so that surplus is genuinely free — the EV should grab it
    # instead of paying grid in an equally-priced block.
    blocks = make_blocks([5.0] * 8)
    # Key solar by the exact block timestamp so ONLY block 4 has surplus
    # (hour-keyed solar would apply to all four blocks in that hour).
    solar = {blocks[4][0]: 20.0}  # 5 kWh surplus in block 4 only
    opt = MILPBatteryOptimizer()
    opt.optimize(
        blocks=blocks, solar_hourly=solar, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=100.0, max_soc=100.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        sell_fee_czk=10.0,  # export uneconomic → surplus would be curtailed
        deferrable_loads=[_ev(energy_kwh=1.0)],  # 1 block
    )
    sched = opt._last_deferrable_schedules[0]
    assert sched.scheduled_blocks == 1
    assert sched.blocks[0][0] == blocks[4][0].strftime("%H:%M")


def test_deferrable_noninterruptible_runs_contiguously():
    # Cheapest blocks are non-adjacent (1 and 3,4). A non-interruptible 2-block
    # load must pick a contiguous run, not the globally-cheapest scattered set.
    blocks = make_blocks([5.0, -1.0, 5.0, -1.0, -1.0, 5.0, 5.0, 5.0])
    opt = MILPBatteryOptimizer()
    opt.optimize(
        blocks=blocks, solar_hourly={}, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=50.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        deferrable_loads=[_ev(interruptible=False)],
    )
    sched = opt._last_deferrable_schedules[0]
    assert sched.scheduled_blocks == 2
    starts = [datetime.strptime(b[0], "%H:%M") for b in sched.blocks]
    assert (starts[1] - starts[0]) == timedelta(minutes=15), "run must be contiguous"


def test_deferrable_partial_when_window_too_narrow():
    # Window admits only 2 blocks but the load needs 4 → partial + shortfall,
    # and the MILP must stay feasible (not fall back).
    blocks = make_blocks([1.0] * 8)
    ev = DeferrableLoad(
        name="ev", energy_required_kwh=5.0, power_kw=5.0,  # 4 blocks needed
        earliest_start=time(0, 0), latest_end=time(0, 30),  # only blocks 0,1
        interruptible=True,
    )
    opt = MILPBatteryOptimizer()
    c, d, sp, decisions = opt.optimize(
        blocks=blocks, solar_hourly={}, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=50.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        deferrable_loads=[ev],
    )
    assert len(decisions) == len(blocks)  # solved, no fallback to empty
    sched = opt._last_deferrable_schedules[0]
    assert sched.scheduled_blocks == 2
    assert sched.energy_shortfall_kwh > 0


def test_greedy_fallback_still_populates_deferrable_schedules():
    # If the MILP solve is infeasible/timed-out/raises it delegates to greedy,
    # which must still pre-schedule deferrable loads (else actuation goes dark).
    blocks = make_blocks([-1.0] * 6 + [10.0] * 6)
    opt = MILPBatteryOptimizer()
    opt._greedy_fallback(
        blocks, {}, {}, flat_dist(1.0),
        battery_capacity_kwh=10.0, current_soc=50.0, min_soc=20.0,
        max_soc=100.0, charge_rate_kw=5.0, discharge_rate_kw=5.0,
        discharge_power_pct=100.0, efficiency=0.85, sell_fee_czk=0.5,
        battery_amortisation_czk=2.0, deferrable_loads=[_ev()],
    )
    assert len(opt._last_deferrable_schedules) == 1
    assert opt._last_deferrable_schedules[0].scheduled_blocks == 2


def test_deferrable_placement_sticks_to_previous_plan_on_a_tie():
    # All blocks equal price → EV placement is a tie. With a previous plan and
    # a switch penalty, the load must keep its prior block rather than drift.
    blocks = make_blocks([5.0] * 8)  # 00:00..01:45, all equal
    opt = MILPBatteryOptimizer()
    prev_block = blocks[6][0]  # 01:30
    opt._prev_deferrable_runs = {"ev": {prev_block}}
    opt.optimize(
        blocks=blocks, solar_hourly={}, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=50.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        deferrable_loads=[_ev(energy_kwh=1.0)],  # 1 block
        switch_penalty_czk=0.05,
    )
    sched = opt._last_deferrable_schedules[0]
    assert sched.block_datetimes == [prev_block]


def test_switch_penalty_does_not_override_real_arbitrage():
    # Even when the previous plan claimed every block was idle, a tiny switch
    # penalty must not stop the optimizer exploiting a real cheap→expensive
    # spread (penalty << price spread). Use -0.5 (not -1.0) so import_cost stays
    # positive and terminal SOC value can't out-value exporting at the peak.
    blocks = make_blocks([-0.5] * 6 + [15.0] * 6)
    opt = MILPBatteryOptimizer()
    opt._prev_actions = {ts: "hold" for ts, _ in blocks}
    charge, discharge, _, _ = opt.optimize(
        blocks=blocks, solar_hourly={}, consumption_hourly={},
        distribution_func=flat_dist(1.0), current_soc=30.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        switch_penalty_czk=0.05,
    )
    assert len(charge) > 0 and len(discharge) > 0


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


def test_hold_idle_when_battery_preserved_and_grid_serves_load():
    """When the solver serves the house from GRID while keeping the battery
    above its floor for a higher-value later use, those blocks are classified
    "hold_idle" (actuated as battery_hold) — not plain "hold" (load_first,
    which would drain the battery). Scenario: cheap-to-import NOW (self-consuming
    is a loss vs amortisation) and an expensive discharge opportunity LATER, with
    the battery already full so it can't charge."""
    blocks = make_blocks([0.5] * 4 + [10.0] * 4, start_hour=0)
    opt = MILPBatteryOptimizer()
    charge, discharge, sp, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={h: 1.0 for h in range(24)},
        distribution_func=flat_dist(0.5),
        current_soc=90.0, min_soc=20.0, max_soc=90.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        battery_amortisation_czk=2.0, sell_fee_czk=0.5,
    )
    cheap_cutoff = blocks[4][0]
    cheap = [d for d in decisions if d.timestamp < cheap_cutoff]
    assert any(d.action == "hold_idle" for d in cheap), (
        f"expected a hold_idle block in the cheap window: "
        f"{[d.action for d in decisions]}"
    )
    # hold_idle blocks are not grid-facing — never in charge/discharge/sell sets.
    hold_ts = {d.timestamp for d in decisions if d.action == "hold_idle"}
    assert not (hold_ts & charge)
    assert not (hold_ts & discharge)
    assert not (hold_ts & sp)
    # And the battery is genuinely preserved (above min) on those blocks.
    for d in decisions:
        if d.action == "hold_idle":
            assert d.soc_before > 20.0


def test_self_consumption_stays_hold_not_hold_idle():
    """A block where the battery clearly, profitably serves the house
    (self-consumption) stays action "hold" (load_first), never "hold_idle".
    Prices low enough that EXPORT is never profitable (so the battery won't
    sell), but higher NOW than later so self-consuming the present load clearly
    beats holding (which would only buy cheaper grid later) → the now-blocks
    self-consume from the battery."""
    blocks = make_blocks([1.4] * 4 + [0.1] * 4, start_hour=0)
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={h: 1.0 for h in range(24)},
        distribution_func=flat_dist(1.0),
        current_soc=60.0, min_soc=20.0, max_soc=90.0,
        battery_amortisation_czk=1.0, sell_fee_czk=0.5,
    )
    expensive_cutoff = blocks[4][0]
    expensive = [d for d in decisions if d.timestamp < expensive_cutoff]
    # The expensive blocks self-consume from the battery → plain "hold".
    assert any(d.action == "hold" for d in expensive), (
        [d.action for d in decisions]
    )
    assert all(d.action != "hold_idle" for d in expensive), (
        [d.action for d in decisions]
    )


def test_summarize_reports_hold_idle_count():
    blocks = make_blocks([0.5] * 4 + [10.0] * 4, start_hour=0)
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks,
        solar_hourly={},
        consumption_hourly={h: 1.0 for h in range(24)},
        distribution_func=flat_dist(0.5),
        current_soc=90.0, min_soc=20.0, max_soc=90.0,
        charge_rate_kw=5.0, discharge_rate_kw=5.0, discharge_power_pct=100.0,
        battery_amortisation_czk=2.0,
    )
    s = opt.summarize(decisions)
    assert "hold_idle_blocks" in s
    assert s["hold_idle_blocks"] == sum(
        1 for d in decisions if d.action == "hold_idle"
    )


def _sp_scenario(current_soc):
    """Moderate morning (below battery-discharge break-even so the battery is
    HELD, not discharged), cheap midday, high evening peak; surplus solar.
    Morning 2.0 − fee 0.5 − amort 2.0 < 0 → no battery discharge, so the only
    grid-facing morning action is exporting the surplus SOLAR (sell_production)
    when full / banking it when not."""
    prices = [2.0] * 4 + [0.1] * 8 + [4.0] * 8
    blocks = make_blocks(prices, start_hour=8)
    solar = {h: 3.0 for h in range(8, 10)}
    solar.update({h: 10.0 for h in range(10, 16)})
    consumption = {h: 0.5 for h in range(24)}
    opt = MILPBatteryOptimizer()
    _, _, _, decisions = opt.optimize(
        blocks=blocks, solar_hourly=solar, consumption_hourly=consumption,
        distribution_func=flat_dist(0.5), current_soc=current_soc, min_soc=20.0,
        max_soc=100.0, charge_rate_kw=5.0, discharge_rate_kw=5.0,
        discharge_power_pct=100.0, sell_fee_czk=0.5, battery_amortisation_czk=2.0,
    )
    return decisions


def test_surplus_solar_exports_as_sell_production_when_full():
    """At/near FULL SOC the battery can't absorb more, so surplus solar exports
    as sell_production (battery passive, SOC not rising)."""
    decisions = _sp_scenario(current_soc=100.0)
    morning = [d for d in decisions if d.timestamp.hour == 8]
    assert any(d.action == "sell_production" for d in morning), (
        "full-battery morning surplus must export as sell_production: "
        f"{[(d.timestamp.strftime('%H:%M'), d.action) for d in decisions[:8]]}"
    )
    for d in morning:
        if d.action == "sell_production":
            assert d.soc_after <= d.soc_before + 0.5


def test_surplus_solar_can_sell_at_mid_soc():
    """sell_production is actuated as grid-first @ stop_soc=live SOC (battery
    passive), so surplus solar can be EXPORTED at ANY SOC — there is no near-full
    gate. With abundant solar the battery can't usefully bank, the mid-SOC plan
    sells the morning surplus rather than curtailing it."""
    decisions = _sp_scenario(current_soc=50.0)
    morning = [d for d in decisions if d.timestamp.hour == 8]
    assert any(d.action == "sell_production" for d in morning), (
        "mid-SOC surplus must be sellable (no near-full gate): "
        f"{[(d.timestamp.strftime('%H:%M'), d.action) for d in morning]}"
    )


def test_milp_discharge_rate_independent_of_discharge_power_pct():
    """Regression guard for the '4x too slow discharge' fix (commit 7f2f31a):
    the MILP discharge energy is discharge_rate_kw*0.25 per block and the energy
    model deliberately IGNORES discharge_power_pct. A high evening spike with the
    battery well above its floor must drain at the full rate at any pct, and the
    two pct settings must produce an identical SOC drop. If the throttle factor
    were re-introduced, pct=25 would drain ~4x slower and this test would fail."""
    prices = [1.0] * 4 + [9.0] * 4  # cheap, then a high spike
    blocks = make_blocks(prices, start_hour=18)
    opt = MILPBatteryOptimizer()

    def first_discharge_drop(pct):
        _, _, _, decisions = opt.optimize(
            blocks=blocks, solar_hourly={}, consumption_hourly={},
            distribution_func=flat_dist(0.5), current_soc=90.0, min_soc=20.0,
            max_soc=100.0, charge_rate_kw=5.0, discharge_rate_kw=5.0,
            discharge_power_pct=pct, sell_fee_czk=0.5, battery_amortisation_czk=1.0,
        )
        disch = [d for d in decisions if d.action == "discharge"]
        assert disch, f"expected a discharge block at pct={pct}"
        return disch[0].soc_before - disch[0].soc_after

    drop25 = first_discharge_drop(25.0)
    drop100 = first_discharge_drop(100.0)
    # discharge_power_pct must not change the modelled discharge rate.
    assert abs(drop25 - drop100) < 0.01, (drop25, drop100)
    # And it's the FULL rate: batt_to_grid_max=5.0*0.25=1.25 kWh → ~13% of a
    # 10 kWh pack per block; a re-introduced 25% throttle would give only ~3.4%.
    assert drop25 > 10.0, f"discharge drained only {drop25:.1f}% — rate throttled?"
