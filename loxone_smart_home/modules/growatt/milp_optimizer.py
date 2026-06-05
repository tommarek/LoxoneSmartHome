"""MILP-based battery scheduler.

A drop-in replacement for the greedy `BatteryOptimizer.optimize()`: same
inputs, same 4-tuple output. Uses PuLP to solve the full-horizon schedule as
a single mixed-integer linear program instead of block-by-block greedy
decisions.

Why bother:
- True global optimum (greedy can pick a small early gain that locks out a
  much bigger later gain).
- Cleanly expresses constraints (SOC bounds, power limits, reserve floor,
  sell-production minimum excess) instead of hand-coded gates.
- More stable schedule across re-evaluations.

Cost:
- PuLP + CBC adds a few seconds to runtime (run off the event loop by the
  controller). Falls back to the greedy engine if PuLP is missing or the
  solve is infeasible/times out.

THE ENERGY-FLOW MODEL (the important part).

Each 15-min block routes energy between four nodes — solar, grid, battery,
and house load — and we maximise net cash. Modelling every flow explicitly
(rather than a single net "drain") is what makes the economics correct:

  Per block i, decision flows (all kWh):
    solar_to_load, solar_to_batt, solar_to_grid, curtail   (split of solar[i])
    batt_to_load, batt_to_grid                              (battery out)
    grid_to_load, grid_charge                               (grid in)
    soc[i+1]                                                (battery state)

  Balances:
    solar:  solar_to_load + solar_to_batt + solar_to_grid + curtail == solar[i]
    load:   solar_to_load + batt_to_load + grid_to_load == consumption[i] + deferrable[i]
    SOC:    soc[i+1] == soc[i] + (grid_charge + solar_to_batt)*η_chg
                                - (batt_to_load + batt_to_grid)/η_dis
    power:  grid_charge + solar_to_batt <= charge_max
            batt_to_load + batt_to_grid <= discharge_max * (1 - is_charge)
    reserve: soc[i+1] >= effective_min_soc[i]

  Objective (maximise):
    + solar_to_grid * (spot - fee)               solar export (no dist, no wear)
    + batt_to_grid  * (spot - fee)               battery export gross (no dist)
    - (grid_to_load + grid_charge) * (spot+dist) grid import cost
    - (batt_to_load + batt_to_grid) * amort      battery wear (once, on the way out)
    - curtail * CURTAIL_PENALTY                   prefer banking free solar
    - switch_penalty * mode_changed              damp churn vs the last plan
    + soc[n] * terminal_value                     keep useful energy at horizon end

  Why this is correct for the Czech market:
    - Serving load from the battery removes a grid_to_load term worth
      (spot+dist); net of amort wear that is (spot+dist-amort) — exactly the
      greedy engine's self_consumption_value. So self-consumption is valued
      WITHOUT an explicit reward term.
    - Battery export nets to (spot-dist-fee-amort) = sell_revenue.
    - Distribution is paid on BOTH import and export (it appears in import
      cost and is subtracted from export revenue).
    - The grid_to_load path keeps the LP feasible at low SOC and makes the
      cost of emptying the battery (then re-importing) explicit, so the solver
      won't dump the battery to the grid when self-consumption is worth more.

DEFERRABLE LOADS (EMHASS-style co-optimization).

Controllable loads (EV, hot-water boost, …) are added as decision variables
rather than pre-scheduled on grid price alone: a per-block binary says
"run this load now", its draw enters the LOAD balance, and the solver places
the required blocks where total cost is lowest — automatically charging into
PV surplus and around battery dispatch. Interruptible loads use one binary
per in-window block; non-interruptible loads use a single contiguous-run
start variable. See `_add_deferrable_loads`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import pulp  # type: ignore
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

from .optimizer import (
    BlockDecision,
    SELL_PRODUCTION_MARGIN_CZK,
    SELL_PRODUCTION_MIN_EXCESS_KWH,
    _forecast_value,
)


# Curtailment penalty: small enough never to distort a real price decision
# (real prices are 1-15 CZK/kWh), large enough above CBC's optimality
# tolerance that banking free solar is robustly preferred over wasting it.
CURTAIL_PENALTY = 0.01

# Flows below this many kWh are treated as solver numerical noise, not a real
# action, when classifying a block's headline action for the dashboard.
ACTION_EPS = 5e-3

# Minimum surplus-solar export (kWh/block) for a battery-passive block to be
# actuated as sell_production (grid_first) instead of plain hold (load_first).
# Above this, exporting the surplus is worth switching the inverter to a true
# export mode so load_first doesn't quietly charge the battery instead; below
# it the amount is too small to bother (avoids mode churn). 0.1 kWh ≈ 400 W.
SP_EXPORT_FLOOR_KWH = 0.1

# Hardware reality (SPH grid-first): the inverter can only export-without-
# charging when the battery is FULL. Below (max_soc - this margin) it banks
# surplus solar into the battery instead, so the plan must NOT schedule
# sell_production there — it would be physically impossible (and mislabel a
# charge as a sale). Keep in sync with config.sell_production_min_soc_margin.
SP_MIN_SOC_MARGIN_PCT = 2.0

# A grid-EXPORT (discharge-to-grid) block must move at least this fraction of
# the grid-first power rate. The controller actuates discharge as on/off at a
# fixed powerRate, so without a floor the solver scatters many trivial sub-rate
# exports that the inverter would each run at full power (the shadow-run
# defect). A floor keeps every export block "meaningful" (≥half rate) so
# set-based actuation is faithful, while still allowing a partial final block
# so the solver never has to BUY grid power just to top up for a full export.
MIN_GRID_DISCHARGE_FRACTION = 0.5

# Per-block reserve-shortfall penalty (CZK per kWh). The overnight reserve is
# now EMERGENT from the objective itself: the objective already prices every
# future grid import, so the solver naturally keeps charge whenever importing
# later would be expensive, and sells/discharges when that is more profitable —
# which is exactly the right economics. A non-zero per-block penalty on top
# DOUBLE-COUNTS and, because the reserve floor can sit above SOC for many
# consecutive blocks, its per-block accumulation snowballs into "grid-charge at
# the evening PEAK to satisfy the reserve" — a guaranteed loss (the live
# incident). So this is 0: the only hard battery floor is min_soc (the soc var
# lower bound); the reserve is the model's economic decision, not a heuristic.
#
# IMPORTANT: keep this 0 until the reserve-floor loss convention is unified with
# the MILP SOC dynamics. The shared _compute_reserve_soc_per_block helper models
# losses with the FULL round-trip efficiency, while MILP SOC continuity uses the
# sqrt-split per-leg eta — so eff_min_kwh and the soc it constrains use different
# conventions. That mismatch is inert only because this penalty is 0; making it
# nonzero would apply a (~sqrt-scale) mis-scaled soft floor.
RESERVE_SHORTFALL_FLOOR = 0.0


def _sell_now_below_margin(price: float, dist: float, sell_fee: float) -> bool:
    """True when solar-export revenue (spot - fee) doesn't clear the swap-profit
    margin, so sell-production mode isn't worth the overhead. Export pays no
    distribution (dist kept in the signature for callers but not deducted)."""
    return (price - sell_fee) <= SELL_PRODUCTION_MARGIN_CZK


class MILPBatteryOptimizer:
    """Mixed-integer linear program for whole-day battery scheduling.

    Same signature as `BatteryOptimizer.optimize()` so the controller can
    swap engines via a config flag without other code changes.
    """

    # The controller checks this to decide whether to hand deferrable loads
    # to the optimizer (co-optimize) or pre-schedule them and overlay the
    # result onto the consumption forecast (greedy can't co-optimize).
    CO_OPTIMIZES_DEFERRABLE = True

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Re-use the greedy optimizer's reserve-SOC + bin-state helpers so
        # behaviour is consistent across engines.
        from .optimizer import BatteryOptimizer
        self._helper = BatteryOptimizer(logger=self.logger)
        self._last_decisions: List[BlockDecision] = []
        # Which engine produced _last_decisions: 'milp' (sqrt-split η per leg) or
        # 'greedy' (full η on one leg). The controller/dashboard use this to pick
        # the correct ΔSOC↔grid-side conversion for adaptive powerRate sizing —
        # a MILP solve that fell back to greedy carries greedy's SOC convention.
        self._last_engine: str = "milp"
        self._last_reserve_info: Dict = {}
        # Deferrable-load placements from the most recent solve, exposed to
        # the controller for dashboard + MQTT actuation. List[DeferrableLoadSchedule].
        self._last_deferrable_schedules: List[Any] = []
        # Headline action per block timestamp from the previous solve, used by
        # the switch-penalty term to damp churn across re-evaluations.
        self._prev_actions: Dict[datetime, str] = {}
        # Per-load set of block timestamps chosen last solve, so the switch
        # penalty can also damp deferrable-load re-planning (esp. relocating a
        # non-interruptible run) across ticks. {load_name: {datetime, ...}}.
        self._prev_deferrable_runs: Dict[str, Set[datetime]] = {}

    @property
    def _base_load_profile(self):
        return self._helper._base_load_profile

    @property
    def _last_reserve_info_dict(self):
        return self._helper._last_reserve_info

    def is_available(self) -> bool:
        return PULP_AVAILABLE

    async def build_base_load_profile(self, *args, **kwargs):
        return await self._helper.build_base_load_profile(*args, **kwargs)

    def summarize(self, decisions: List[BlockDecision]) -> Dict:
        return self._helper.summarize(decisions)

    def _greedy_fallback(self, blocks, solar_hourly, consumption_hourly,
                         distribution_func, battery_capacity_kwh, current_soc,
                         min_soc, max_soc, charge_rate_kw, discharge_rate_kw,
                         discharge_power_pct, efficiency, sell_fee_czk,
                         battery_amortisation_czk, deferrable_loads=None,
                         sell_production_min_soc_margin=SP_MIN_SOC_MARGIN_PCT,
                         battery_amortisation_export_czk=None):
        """Run the greedy engine and adopt its decisions as our own.

        Used when the MILP solve is infeasible, times out, or raises, so the
        controller always receives a usable schedule (the documented
        greedy-fallback contract). The greedy engine cannot co-optimize
        deferrable loads, so we pre-schedule them (cheapest in-window blocks)
        and overlay their draw onto the consumption forecast before solving,
        preserving deferrable behaviour even on the fallback path.
        """
        if deferrable_loads:
            consumption_hourly = self._overlay_deferrable_for_greedy(
                deferrable_loads, blocks, distribution_func, consumption_hourly
            )
        result = self._helper.optimize(
            blocks=blocks,
            solar_hourly=solar_hourly,
            consumption_hourly=consumption_hourly,
            distribution_func=distribution_func,
            battery_capacity_kwh=battery_capacity_kwh,
            current_soc=current_soc,
            min_soc=min_soc,
            max_soc=max_soc,
            charge_rate_kw=charge_rate_kw,
            discharge_rate_kw=discharge_rate_kw,
            discharge_power_pct=discharge_power_pct,
            efficiency=efficiency,
            sell_fee_czk=sell_fee_czk,
            battery_amortisation_czk=battery_amortisation_czk,
            battery_amortisation_export_czk=battery_amortisation_export_czk,
            sell_production_min_soc_margin=sell_production_min_soc_margin,
        )
        self._last_decisions = result[3]
        self._last_engine = "greedy"  # decisions now carry greedy's SOC convention
        # Keep the switch-penalty anchor in sync with what we actually returned,
        # so the NEXT solve doesn't penalise deviations from a stale MILP plan
        # that was never applied (the fallback replaced it).
        self._prev_actions = {d.timestamp: d.action for d in result[3]}
        return result

    def _overlay_deferrable_for_greedy(
        self, deferrable_loads, blocks, distribution_func, consumption_hourly
    ) -> Dict[Any, float]:
        """Pre-schedule deferrable loads greedily and overlay onto consumption.

        Returns a COPY of consumption_hourly with the scheduled per-block load
        added, and stores the resulting schedules in
        ``self._last_deferrable_schedules`` (same contract as the MILP path).
        """
        from .deferrable_loads import DeferrableLoadScheduler
        scheduler = DeferrableLoadScheduler(self.logger)
        schedules = scheduler.schedule_all(
            list(deferrable_loads), blocks, distribution_func
        )
        self._last_deferrable_schedules = schedules
        loads_by_name = {l.name: l for l in deferrable_loads}
        overlay = scheduler.consumption_overlay(schedules, loads_by_name)
        merged = dict(consumption_hourly)
        for key, extra in overlay.items():
            if isinstance(key, datetime):
                # Resolve the existing base with the same key precedence as
                # _forecast_value, using `is not None` so a real 0.0-kWh block
                # isn't treated as missing (which `or` would, then wrongly pull
                # in another day's hourly value).
                base = merged.get(key)
                if base is None:
                    base = merged.get((key.date(), key.hour))
                if base is None:
                    base = merged.get(key.hour, 0.0)
                merged[key] = base + extra
            else:
                merged[key] = merged.get(key, 0.0) + extra
        return merged

    def optimize(
        self,
        blocks: List[Tuple[datetime, float]],
        solar_hourly: Dict[Any, float],
        consumption_hourly: Dict[Any, float],
        distribution_func,
        battery_capacity_kwh: float = 10.0,
        current_soc: float = 50.0,
        min_soc: float = 20.0,
        max_soc: float = 100.0,
        charge_rate_kw: float = 2.5,
        discharge_rate_kw: float = 2.5,
        # Accepted for engine-signature parity but IGNORED by the MILP energy
        # model: batt_to_grid_max = discharge_rate_kw*0.25 (the '4x too slow'
        # fix, commit 7f2f31a). Guarded by test_milp_discharge_rate_independent…
        discharge_power_pct: float = 25.0,
        efficiency: float = 0.85,
        sell_fee_czk: float = 0.5,
        battery_amortisation_czk: float = 2.0,
        battery_amortisation_export_czk: Optional[float] = None,
        deferrable_loads: Optional[Sequence[Any]] = None,
        switch_penalty_czk: float = 0.0,
        export_price_min: Optional[float] = None,
        inverter_off_price: Optional[float] = None,
        sell_production_min_soc_margin: float = SP_MIN_SOC_MARGIN_PCT,
    ) -> Tuple[Set[datetime], Set[datetime], Set[datetime], List[BlockDecision]]:
        """Solve the full-horizon schedule via MILP (energy-flow model).

        Strict, engine-agnostic rules the plan must respect (so the plan never
        promises an action the controller's hardware gates will then block):
        - export_price_min: never export to grid (solar OR battery) in a block
          whose spot price is below this floor (the export/transmission fee) —
          never export at a net loss.
        - inverter_off_price: when spot price is below this, the inverter is OFF
          (PV disabled) — no solar, no charge/discharge, no export that block.
        """
        # Reset per-solve outputs so a fallback/early-return never leaks a
        # stale deferrable plan from a previous solve.
        self._last_deferrable_schedules = []

        if not PULP_AVAILABLE:
            self.logger.warning(
                "PuLP not installed — MILP unavailable, falling back to greedy"
            )
            return self._greedy_fallback(
                blocks, solar_hourly, consumption_hourly, distribution_func,
                battery_capacity_kwh, current_soc, min_soc, max_soc,
                charge_rate_kw, discharge_rate_kw, discharge_power_pct,
                efficiency, sell_fee_czk, battery_amortisation_czk,
                deferrable_loads, sell_production_min_soc_margin,
                battery_amortisation_export_czk=battery_amortisation_export_czk,
            )

        if not blocks:
            return set(), set(), set(), []

        # Clamp the live SOC into the configured [min, max] window before
        # pinning soc[0]. Telemetry can report e.g. 100% while max_soc=90;
        # an out-of-bounds start makes soc[0] == start_battery_kwh infeasible
        # against the soc variable bounds, silently forcing greedy fallback
        # every tick.
        current_soc = max(min_soc, min(max_soc, current_soc))

        n = len(blocks)
        charge_max = charge_rate_kw * 0.25
        # discharge_rate_kw is the ACTUAL grid-discharge power at the configured
        # discharge_power_rate (~2.5 kW at 25% on this inverter), confirmed from
        # telemetry. Both house self-consumption and grid export run at that rate
        # — do NOT throttle grid export by discharge_power_pct again (that
        # double-counted the power rate and modelled grid discharge 4x too slow).
        batt_out_max = discharge_rate_kw * 0.25
        batt_to_grid_max = batt_out_max
        max_battery_kwh = battery_capacity_kwh * max_soc / 100.0
        min_battery_kwh = battery_capacity_kwh * min_soc / 100.0
        # SOC at/above which sell_production (pure grid export, battery passive)
        # is physically possible — below it the inverter banks surplus solar.
        sp_min_kwh = battery_capacity_kwh * max(0.0, max_soc - sell_production_min_soc_margin) / 100.0
        start_battery_kwh = battery_capacity_kwh * current_soc / 100.0

        # Split the round-trip efficiency symmetrically across the two legs so
        # both charging and discharging incur loss (eta_chg * eta_dis ==
        # efficiency). The greedy engine applies all loss on the charge leg;
        # sqrt-splitting is closer to real inverter behaviour and makes the
        # cost of a charge→discharge cycle independent of which leg you price.
        eta = max(1e-3, efficiency) ** 0.5

        # Per-block price coefficients.
        prices = [p for _, p in blocks]
        dists = [distribution_func(ts.hour) for ts, _ in blocks]
        # Grid export gross (battery wear charged separately). Export pays NO
        # distribution — only the sell fee (FVE buyback = spot − fee). Distribution
        # applies to IMPORT only. (Was spot − dist − fee, which under-valued export
        # by the whole distribution tariff and made it hoard/curtail sellable solar.)
        export_now = [prices[i] - sell_fee_czk for i in range(n)]
        # Wear cost charged on battery→GRID export. Defaults to the shared wear
        # cost; an explicit export override raises the arbitrage hurdle without
        # touching battery→house self-consumption. Unset → identical to before.
        amort_export = (
            battery_amortisation_czk
            if battery_amortisation_export_czk is None
            else battery_amortisation_export_czk
        )
        # Battery export NET of wear (the "never export at a loss" gate).
        sell_revenue = [export_now[i] - amort_export for i in range(n)]
        # Grid import cost per kWh (paid on both load and battery charge).
        import_cost = [prices[i] + dists[i] for i in range(n)]

        # Per-block raw solar / consumption / structural surplus.
        solar_block: List[float] = []
        consumption_block: List[float] = []
        solar_excess: List[float] = []
        for ts, _ in blocks:
            s = _forecast_value(solar_hourly, ts) / 4.0
            c = _forecast_value(consumption_hourly, ts) / 4.0
            if c <= 0:
                # Mirror the greedy engine AND the shared reserve helper: a
                # missing/zero consumption entry falls back to the learned
                # base-load profile, so the MILP load balance never sees 0 house
                # load where the reserve floor (eff_min_kwh) assumed real base
                # load — an intra-MILP + cross-engine inconsistency otherwise.
                c = self._helper._base_load_profile.get(
                    ts.hour, ts.weekday() >= 5
                ) / 4.0
            solar_block.append(s)
            consumption_block.append(c)
            solar_excess.append(max(0.0, s - c))

        # Per-block effective minimum SOC (reserve), reused from the greedy
        # helper so both engines protect the same overnight energy.
        effective_min_socs = self._helper._compute_reserve_soc_per_block(
            blocks, prices, solar_hourly, battery_capacity_kwh,
            min_soc, max_soc, charge_max, efficiency,
        )
        eff_min_kwh = [
            battery_capacity_kwh * (s / 100.0) for s in effective_min_socs
        ]

        # === MILP model ===
        prob = pulp.LpProblem("battery_schedule", pulp.LpMaximize)

        grid_charge   = [pulp.LpVariable(f"gc_{i}", 0, charge_max) for i in range(n)]
        solar_to_batt = [pulp.LpVariable(f"sb_{i}", lowBound=0) for i in range(n)]
        solar_to_load = [pulp.LpVariable(f"sl_{i}", lowBound=0) for i in range(n)]
        solar_to_grid = [pulp.LpVariable(f"sg_{i}", lowBound=0) for i in range(n)]
        curtail       = [pulp.LpVariable(f"cu_{i}", lowBound=0) for i in range(n)]
        batt_to_load  = [pulp.LpVariable(f"bl_{i}", 0, batt_out_max) for i in range(n)]
        batt_to_grid  = [pulp.LpVariable(f"bg_{i}", 0, batt_to_grid_max) for i in range(n)]
        grid_to_load  = [pulp.LpVariable(f"gl_{i}", lowBound=0) for i in range(n)]

        is_charge = [pulp.LpVariable(f"ic_{i}", cat="Binary") for i in range(n)]
        is_disch  = [pulp.LpVariable(f"id_{i}", cat="Binary") for i in range(n)]
        is_sp     = [pulp.LpVariable(f"isp_{i}", cat="Binary") for i in range(n)]
        # Physical battery-charging indicator (covers BOTH grid charge and solar
        # banking). Makes total battery charge and total battery discharge
        # mutually exclusive per block — a single inverter can't do both at once.
        is_batt_chg = [pulp.LpVariable(f"bc_{i}", cat="Binary") for i in range(n)]
        # Battery near-full indicator: only when SOC ≥ sp_min_kwh can a load_first
        # (hold) block actually EXPORT surplus solar — below that the inverter
        # banks it into the battery. Gates solar_to_grid so the plan matches that
        # hardware reality (see the solar_to_grid bound below).
        near_full = [pulp.LpVariable(f"nf_{i}", cat="Binary") for i in range(n)]
        soc = [
            pulp.LpVariable(f"soc_{i}", min_battery_kwh, max_battery_kwh)
            for i in range(n + 1)
        ]
        # Soft-reserve shortfall (kWh the SOC falls below the reserve floor).
        # The physical floor (min_battery_kwh) stays hard via the soc bounds;
        # only the reserve margin above it is relaxable, at a steep penalty.
        reserve_short = [pulp.LpVariable(f"rs_{i}", lowBound=0) for i in range(n)]

        # Deferrable-load run variables + per-block draw expressions (kWh).
        # `deferrable_penalty` damps re-planning of load placement across ticks.
        deferrable_draw, deferrable_meta, deferrable_penalty = (
            self._add_deferrable_loads(
                prob, blocks, deferrable_loads, switch_penalty_czk
            )
        )

        prob += soc[0] == start_battery_kwh

        for i in range(n):
            s = solar_block[i]
            c = consumption_block[i]

            # Solar balance: every kWh of solar is used, banked, exported, or curtailed.
            prob += solar_to_load[i] + solar_to_batt[i] + solar_to_grid[i] + curtail[i] == s
            # Load balance: house draw + any deferrable load running this block,
            # served by solar, battery, and/or grid.
            prob += (
                solar_to_load[i] + batt_to_load[i] + grid_to_load[i]
                == c + deferrable_draw[i]
            )

            # Battery power limits, mutually exclusive via is_batt_chg: ALL
            # charge (grid + solar banking) is gated by is_batt_chg, ALL discharge
            # (to load + to grid) by (1 - is_batt_chg). A single inverter can't
            # charge and discharge at once — this also forbids the otherwise-legal
            # hold-block combo of solar_to_batt>0 AND batt_to_load>0 (physically
            # impossible). Grid charge is additionally capped by is_charge (the
            # grid-facing mode indicator) further below.
            prob += grid_charge[i] + solar_to_batt[i] <= charge_max * is_batt_chg[i]
            prob += batt_to_load[i] + batt_to_grid[i] <= batt_out_max * (1 - is_batt_chg[i])
            # sell_production actuates grid-first with the battery PASSIVE (it does
            # NOT serve the house). Forbid batt_to_load on is_sp blocks so the plan
            # can't bank self-consumption value the hardware won't realise (the
            # deficit is served from grid there) — keeps plan and actuation aligned.
            prob += batt_to_load[i] <= batt_out_max * (1 - is_sp[i])

            # SOC continuity: charge legs gain energy at η_chg; discharge legs
            # deplete the battery at 1/η_dis (wear is priced in the objective).
            prob += soc[i + 1] == (
                soc[i]
                + (grid_charge[i] + solar_to_batt[i]) * eta
                - (batt_to_load[i] + batt_to_grid[i]) / eta
            )
            # Soft reserve floor (penalised per-block by reserve_penalty).
            prob += soc[i + 1] >= eff_min_kwh[i] - reserve_short[i]

            # Activation linking → mode indicators (grid-facing actions).
            prob += grid_charge[i] <= charge_max * is_charge[i]
            # Grid EXPORT is semi-continuous: when a block discharges to grid it
            # moves between half and full the grid-first power rate — never a
            # trivial sub-rate trickle (see MIN_GRID_DISCHARGE_FRACTION). The
            # range (vs a hard ==) lets the last block of a run export a partial
            # amount, so the solver never buys grid power just to top up.
            prob += batt_to_grid[i] <= batt_to_grid_max * is_disch[i]
            prob += batt_to_grid[i] >= (
                MIN_GRID_DISCHARGE_FRACTION * batt_to_grid_max * is_disch[i]
            )
            # Surplus PV can spill to grid, but only where the HARDWARE actually
            # exports it — otherwise a plain hold block actuates load_first, which
            # banks the surplus into the battery (SOC rises) instead of exporting,
            # so a plan that exported it would diverge from reality. Surplus
            # export is allowed up to the block's surplus when the block is a
            # grid-facing mode (is_disch/is_sp) OR the battery is near full
            # (near_full → load_first genuinely spills the excess), PLUS the
            # always-physical overflow that exceeds the charge rate (can't be
            # banked fast enough even at mid-SOC). Hard cap stays at the surplus.
            phys_overflow = max(0.0, max(0.0, s - c) - charge_max)
            prob += solar_to_grid[i] <= (
                phys_overflow
                + max(s, 0.0) * (near_full[i] + is_disch[i] + is_sp[i])
            )
            prob += solar_to_grid[i] <= max(s, 0.0)
            # NOTE: no combined AC grid-export cap (solar_to_grid + batt_to_grid)
            # is modelled — the SPH battery DC rate (batt_to_grid_max) is already
            # conservative vs the larger inverter AC rating, and structural solar
            # surplus on a discharge block is small. If an inverter AC export
            # limit is ever configured, add `solar_to_grid[i] + batt_to_grid[i]
            # <= ac_export_max` here.
            # near_full may only be 1 when SOC is genuinely near full (≥sp_min),
            # so it can't be used to license mid-SOC export.
            prob += soc[i] >= sp_min_kwh - max_battery_kwh * (1 - near_full[i])
            # One grid-facing mode per block.
            prob += is_charge[i] + is_disch[i] + is_sp[i] <= 1

            # Never export from the battery at a loss.
            if sell_revenue[i] <= 0:
                prob += is_disch[i] == 0

            # STRICT RULE 1 — export floor: no grid export (solar OR battery)
            # when the spot price is below the export/transmission fee, so we
            # never export at a net loss. Engine-agnostic; mirrors the controller
            # export gate so the plan matches actuation.
            if export_price_min is not None and prices[i] < export_price_min:
                prob += solar_to_grid[i] == 0
                prob += batt_to_grid[i] == 0
                prob += is_disch[i] == 0
                prob += is_sp[i] == 0

            # STRICT RULE 2 — PV/inverter OFF below a deeply-negative price: the
            # whole inverter is off (PV disabled, battery idle), load served from
            # grid. Mirrors the controller inverter-off gate so the plan never
            # schedules solar/charge/discharge in a block the hardware powers down.
            if inverter_off_price is not None and prices[i] < inverter_off_price:
                prob += solar_to_load[i] == 0
                prob += solar_to_batt[i] == 0
                prob += solar_to_grid[i] == 0
                prob += grid_charge[i] == 0
                prob += batt_to_load[i] == 0
                prob += batt_to_grid[i] == 0
                prob += is_charge[i] == 0
                prob += is_disch[i] == 0
                prob += is_sp[i] == 0
            # Sell-production gates: only worthwhile above a volume floor AND a
            # swap margin (gate on the structural surplus s-c).
            if (
                solar_excess[i] < SELL_PRODUCTION_MIN_EXCESS_KWH
                or _sell_now_below_margin(prices[i], dists[i], sell_fee_czk)
            ):
                prob += is_sp[i] == 0
            else:
                prob += solar_to_grid[i] >= SELL_PRODUCTION_MIN_EXCESS_KWH * is_sp[i]
            # NOTE: no SOC gate on is_sp. sell_production is actuated as grid-first
            # with stop_soc pinned to the LIVE SOC (controller _build_desired_state),
            # so the battery stays passive and the surplus exports at ANY SOC — the
            # solver may freely choose export vs banking per-block on economics.

        # Terminal value of leftover SOC = avoided future import (spot+dist-amort),
        # the TRUE worth of stored energy. Capped below the cheapest grid-charge
        # break-even so terminal value can never alone justify buying grid power
        # to inflate SOC (banking free solar still benefits).
        sc_values = sorted(
            prices[i] + dists[i] - battery_amortisation_czk for i in range(n)
        )
        terminal_value_per_kwh = (
            max(0.0, sc_values[len(sc_values) // 2]) if sc_values else 0.0
        )
        min_charge_cost = min(import_cost) if import_cost else 0.0
        # Cap uses the ROUND-TRIP efficiency on purpose (not the per-leg `eta`):
        # the break-even for "charge cheap now, use later" is the full
        # charge→store→discharge cycle. Don't "fix" this to `eta`. Apply the cap
        # ALWAYS (not only when min_charge_cost > 0): on a deeply-negative-price
        # day min(import_cost) can be ≤ 0 while the median stays positive, and the
        # max(0.0, …) floor then correctly drives terminal value to 0 — otherwise
        # the cap is skipped and SOC-hoarding (grid-charge to inflate end SOC)
        # gets rewarded exactly when it shouldn't.
        terminal_value_per_kwh = max(0.0, min(
            terminal_value_per_kwh, min_charge_cost / max(1e-3, efficiency)
        ))
        # Slightly discount the (estimated, future) terminal value so a
        # CERTAIN present-block self-consumption saving always wins ties
        # against hoarding energy for an uncertain later benefit. Without this
        # the solver is indifferent between covering a present deficit from the
        # battery vs importing and keeping SOC, and may leave the battery idle.
        terminal_value_per_kwh *= 0.99

        objective = pulp.lpSum(
            solar_to_grid[i] * export_now[i]
            + batt_to_grid[i] * export_now[i]
            - (grid_to_load[i] + grid_charge[i]) * import_cost[i]
            - batt_to_load[i] * battery_amortisation_czk
            - batt_to_grid[i] * amort_export
            - curtail[i] * CURTAIL_PENALTY
            - reserve_short[i] * RESERVE_SHORTFALL_FLOOR
            for i in range(n)
        ) + soc[n] * terminal_value_per_kwh * eta
        # ^ multiply by the discharge-leg loss: stored soc[n] can only deliver
        #   soc[n]*eta to load/grid, so value it at the delivered rate (not the
        #   full stored kWh), matching the SOC-continuity discharge term.

        # Schedule-churn damping: a tiny cost for deviating from the previous
        # plan's headline action, so noisy price/forecast updates don't reshuffle
        # the schedule. Far below real spreads, so it only breaks ties.
        if switch_penalty_czk > 0 and self._prev_actions:
            objective -= self._switch_penalty_term(
                blocks, is_charge, is_disch, is_sp, switch_penalty_czk
            )
        # Same idea for deferrable-load placement (built in _add_deferrable_loads).
        if deferrable_penalty is not None:
            objective -= deferrable_penalty

        prob += objective

        # Solve (time-limited; off the event loop via the controller).
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
        try:
            status = prob.solve(solver)
            status_str = pulp.LpStatus[status]
        except Exception as e:
            self.logger.warning(
                f"MILP solve raised {e} — falling back to greedy engine"
            )
            return self._greedy_fallback(
                blocks, solar_hourly, consumption_hourly, distribution_func,
                battery_capacity_kwh, current_soc, min_soc, max_soc,
                charge_rate_kw, discharge_rate_kw, discharge_power_pct,
                efficiency, sell_fee_czk, battery_amortisation_czk,
                deferrable_loads, sell_production_min_soc_margin,
                battery_amortisation_export_czk=battery_amortisation_export_czk,
            )
        if status_str != "Optimal":
            # Infeasible / Unbounded / Undefined / timed-out without an optimum:
            # delegate to greedy so the controller still gets a usable plan.
            self.logger.warning(
                f"MILP solver returned status {status_str} — "
                f"falling back to greedy engine"
            )
            return self._greedy_fallback(
                blocks, solar_hourly, consumption_hourly, distribution_func,
                battery_capacity_kwh, current_soc, min_soc, max_soc,
                charge_rate_kw, discharge_rate_kw, discharge_power_pct,
                efficiency, sell_fee_czk, battery_amortisation_czk,
                deferrable_loads, sell_production_min_soc_margin,
                battery_amortisation_export_czk=battery_amortisation_export_czk,
            )

        # Materialise results.
        charge_times: Set[datetime] = set()
        discharge_times: Set[datetime] = set()
        sp_times: Set[datetime] = set()
        decisions: List[BlockDecision] = []

        def val(v):
            x = pulp.value(v)
            return float(x) if x is not None else 0.0

        soc_v = [val(s) for s in soc]
        new_actions: Dict[datetime, str] = {}
        for i, (ts, price) in enumerate(blocks):
            gc = val(grid_charge[i])
            bg = val(batt_to_grid[i])
            sg = val(solar_to_grid[i])
            sb = val(solar_to_batt[i])
            bl = val(batt_to_load[i])
            gl = val(grid_to_load[i])
            action = "hold"
            net_value = 0.0
            # Classify the headline action by the solver's own mutually-exclusive
            # mode indicators (is_charge/is_disch/is_sp, gated <=1), NOT by the
            # largest grid flow. Surplus PV spills to grid on ANY block (the
            # export bound is relaxed to all blocks), so a magnitude test like
            # `gc >= sg` would mislabel a genuine charge/discharge block as
            # hold whenever concurrent solar export exceeds the battery flow,
            # dropping it from charge_times/discharge_times and from actuation.
            if val(is_charge[i]) > 0.5 and gc > ACTION_EPS:
                action = "charge"
                charge_times.add(ts)
                net_value = -gc * import_cost[i]
            elif val(is_disch[i]) > 0.5 and bg > ACTION_EPS:
                action = "discharge"
                discharge_times.add(ts)
                net_value = bg * sell_revenue[i]
            elif val(is_sp[i]) > 0.5 and sg > ACTION_EPS:
                # A block the solver explicitly put into sell-production mode.
                action = "sell_production"
                sp_times.add(ts)
                net_value = sg * export_now[i]
            elif sg > SP_EXPORT_FLOOR_KWH and sb < ACTION_EPS:
                # Surplus solar is being EXPORTED while the battery is NOT charging
                # from it — actuate sell_production (grid-first @ stop_soc=live SOC,
                # battery passive) so the surplus is actually SOLD, at any SOC.
                # Plain "hold" maps to load_first, which would charge the surplus
                # into the battery before spilling to grid — silently storing it
                # instead of selling. Honour the solver's solar-flow decision.
                action = "sell_production"
                sp_times.add(ts)
                net_value = sg * export_now[i]
            elif bl > ACTION_EPS:
                # Battery self-consumption: no grid-facing action (stays "hold",
                # which maps to load_first and is what the inverter already does),
                # but credit the avoided import (net of battery wear) so net_value
                # isn't understated as 0 on genuinely active blocks.
                net_value = bl * (import_cost[i] - battery_amortisation_czk)
            elif gl > ACTION_EPS and soc_v[i] > min_battery_kwh + ACTION_EPS:
                # The solver is serving the house from the GRID while the battery
                # sits above its floor with usable charge — i.e. it deliberately
                # PRESERVES the battery this block (for a higher-value later use).
                # This is a true "hold" that must be actuated as battery-hold
                # (load_first + stop_soc=max_soc, no discharge), NOT plain
                # load_first, which would drain the battery for the house. Without
                # this the plan ("hold") and the hardware ("load_first drains")
                # diverge.
                action = "hold_idle"
            new_actions[ts] = action

            decisions.append(BlockDecision(
                timestamp=ts,
                action=action,
                price_czk=price,
                distribution_czk=dists[i],
                solar_kwh=solar_block[i],
                consumption_kwh=consumption_block[i],
                soc_before=(soc_v[i] / battery_capacity_kwh) * 100.0,
                soc_after=(soc_v[i + 1] / battery_capacity_kwh) * 100.0,
                net_value=net_value,
            ))

        self._last_decisions = decisions
        self._last_engine = "milp"
        self._prev_actions = new_actions
        self._last_deferrable_schedules = self._extract_deferrable_schedules(
            blocks, deferrable_meta, import_cost, val
        )
        if effective_min_socs:
            self._helper._last_reserve_info["effective_min_soc"] = round(
                effective_min_socs[0], 1
            )

        return charge_times, discharge_times, sp_times, decisions

    # ── Deferrable-load co-optimization helpers ──────────────────────────

    def _add_deferrable_loads(
        self,
        prob: Any,
        blocks: List[Tuple[datetime, float]],
        deferrable_loads: Optional[Sequence[Any]],
        switch_penalty_czk: float = 0.0,
    ) -> Tuple[List[Any], List[Dict[str, Any]], Any]:
        """Add deferrable-load decision variables and constraints to `prob`.

        Returns:
            (draw_per_block, meta, penalty) where draw_per_block[i] is a PuLP
            expression (or 0.0) for the deferrable kWh drawn in block i, meta is
            a list of per-load dicts used afterwards to build the schedules, and
            penalty is a (small) PuLP expression that damps re-planning the
            chosen blocks vs the previous solve (or None when not applicable).
        """
        n = len(blocks)
        draw_terms: List[List[Any]] = [[] for _ in range(n)]
        meta: List[Dict[str, Any]] = []
        penalty_terms: List[Any] = []
        if not deferrable_loads:
            return [0.0] * n, meta, None

        for li, load in enumerate(deferrable_loads):
            block_energy = load.power_kw * 0.25  # kWh delivered per running block
            requested = load.required_blocks()
            if block_energy <= 0 or requested <= 0:
                continue

            # In-window block indices (uses the load's own midnight-aware test).
            win = [i for i in range(n) if load.in_window(blocks[i][0].time())]
            if not win:
                meta.append({
                    "load": load, "requested": requested, "scheduled": 0,
                    "block_energy": block_energy, "run": {}, "win": [],
                })
                continue
            nb_eff = min(requested, len(win))

            run_by_block: Dict[int, Any] = {}
            placed = False
            if not load.interruptible:
                placed = self._add_contiguous_run(
                    prob, blocks, li, win, nb_eff, run_by_block
                )
                if not placed:
                    # No contiguous slot fits in the window — fall back to
                    # interruptible (scattered) scheduling. Warn, because the
                    # actuation layer will run this nominally non-interruptible
                    # load in pieces (mirrors the greedy scheduler's warning).
                    self.logger.warning(
                        f"Deferrable load {load.name!r} non-interruptible but no "
                        f"contiguous {nb_eff}-block run fits its window — "
                        f"scheduling interruptibly (scattered)"
                    )
            if not placed:
                # Interruptible (or non-interruptible with no contiguous slot):
                # one binary per in-window block, deliver exactly nb_eff blocks.
                run_vars = {
                    i: pulp.LpVariable(f"dl{li}_{i}", cat="Binary") for i in win
                }
                prob += pulp.lpSum(run_vars.values()) == nb_eff
                run_by_block = run_vars

            for i, var in run_by_block.items():
                draw_terms[i].append(var * block_energy)

            # Anti-churn: bias toward the blocks this load ran last solve, so a
            # (esp. non-interruptible) run isn't relocated by forecast jitter.
            # Tiny magnitude → only breaks ties, never overrides a real saving.
            prev = self._prev_deferrable_runs.get(load.name)
            if switch_penalty_czk > 0 and prev:
                for i, var in run_by_block.items():
                    if blocks[i][0] in prev:
                        penalty_terms.append(switch_penalty_czk * (1 - var))
                    else:
                        penalty_terms.append(switch_penalty_czk * var)

            meta.append({
                "load": load, "requested": requested, "scheduled": nb_eff,
                "block_energy": block_energy, "run": run_by_block, "win": win,
                # True when a NON-interruptible load was scattered because no
                # contiguous slot fit — its naive baseline must then use the
                # scattered (interruptible) form, not a contiguous run it never
                # actually ran, or savings_czk is mis-stated.
                "scattered": (not load.interruptible) and (not placed),
            })

        draw = [pulp.lpSum(terms) if terms else 0.0 for terms in draw_terms]
        penalty = pulp.lpSum(penalty_terms) if penalty_terms else None
        return draw, meta, penalty

    @staticmethod
    def _add_contiguous_run(
        prob: Any,
        blocks: List[Tuple[datetime, float]],
        li: int,
        win: List[int],
        nb_eff: int,
        run_by_block: Dict[int, Any],
    ) -> bool:
        """Model a non-interruptible load as a single contiguous run.

        Adds one binary start variable per feasible start position (a run of
        nb_eff consecutive, in-window, 15-min-adjacent blocks), constrains
        exactly one to be chosen, and fills run_by_block[i] with the affine
        expression "is block i covered by the chosen run". Returns False if no
        contiguous slot exists (caller then falls back to interruptible).
        """
        win_set = set(win)
        starts: List[int] = []
        for k in win:
            run_idx = [k + m for m in range(nb_eff)]
            if run_idx[-1] >= len(blocks):
                continue
            if any(j not in win_set for j in run_idx):
                continue
            contiguous = all(
                (blocks[j + 1][0] - blocks[j][0]) == timedelta(minutes=15)
                for j in run_idx[:-1]
            )
            if contiguous:
                starts.append(k)
        if not starts:
            return False

        start_vars = {
            k: pulp.LpVariable(f"dls{li}_{k}", cat="Binary") for k in starts
        }
        prob += pulp.lpSum(start_vars.values()) == 1
        for i in win:
            covering = [
                start_vars[k] for k in starts if k <= i < k + nb_eff
            ]
            run_by_block[i] = pulp.lpSum(covering) if covering else 0.0
        return True

    def _extract_deferrable_schedules(
        self,
        blocks: List[Tuple[datetime, float]],
        meta: List[Dict[str, Any]],
        import_cost: List[float],
        val,
    ) -> List[Any]:
        """Build DeferrableLoadSchedule objects from the solved run variables."""
        if not meta:
            self._prev_deferrable_runs = {}
            return []
        from .deferrable_loads import (
            DeferrableLoadSchedule,
            earliest_contiguous_run,
        )

        prev_runs: Dict[str, Set[datetime]] = {}
        schedules: List[Any] = []
        for m in meta:
            load = m["load"]
            block_energy = m["block_energy"]
            win = m["win"]
            run = m["run"]
            chosen_idx = sorted(i for i in win if val(run[i]) > 0.5) if run else []

            # NOTE: expected_cost_czk / naive_cost_czk price each block's energy
            # as 100% grid import (import_cost[i] * block_energy). This is a
            # grid-priced UPPER BOUND for reporting only — it ignores that the
            # deferrable draw may be served by PV surplus or battery in the solved
            # plan. It does NOT enter the objective/dispatch. The relative
            # savings_czk (naive − expected, same block_energy on both) stays
            # meaningful as a scheduling-shift metric; only the absolute figures
            # are overstated on PV/battery-served days.
            blocks_hhmm: List[Tuple[str, str]] = []
            block_dts: List[datetime] = []
            expected = 0.0
            for i in chosen_idx:
                ts = blocks[i][0]
                end = ts + timedelta(minutes=15)
                blocks_hhmm.append((ts.strftime("%H:%M"), end.strftime("%H:%M")))
                block_dts.append(ts)
                expected += import_cost[i] * block_energy

            # Naive baseline: run ASAP. For a non-interruptible load the
            # baseline must match the chosen plan's contiguity (earliest
            # contiguous run) — otherwise savings_czk is mis-stated against a
            # cheaper scattered baseline the load could never actually run.
            # k == nb_eff == min(requested, len(win)); on a SHORTFALL day
            # (window shorter than requested) the optimizer runs all available
            # blocks, so the ASAP baseline win[:k] spans the same blocks and
            # reported savings collapse to ~0 — correct: there is no scheduling
            # slack to exploit, so savings_czk reflects only the scheduled blocks.
            k = len(chosen_idx)
            # A non-interruptible load that was SCATTERED (no contiguous slot fit)
            # actually ran in pieces, so its baseline must be the scattered ASAP
            # form too — not a contiguous run it never performed.
            if load.interruptible or m.get("scattered"):
                naive_idx = win[:k]
            else:
                positions = earliest_contiguous_run([blocks[i][0] for i in win], k)
                naive_idx = [win[p] for p in positions]
            naive = sum(import_cost[i] * block_energy for i in naive_idx)

            prev_runs[load.name] = set(block_dts)
            shortfall = max(0, m["requested"] - len(chosen_idx)) * block_energy
            schedules.append(DeferrableLoadSchedule(
                load_name=load.name,
                blocks=blocks_hhmm,
                block_datetimes=block_dts,
                expected_cost_czk=round(expected, 2),
                naive_cost_czk=round(naive, 2),
                requested_blocks=m["requested"],
                scheduled_blocks=len(chosen_idx),
                energy_shortfall_kwh=round(shortfall, 3),
            ))
        self._prev_deferrable_runs = prev_runs
        return schedules

    def _switch_penalty_term(
        self, blocks, is_charge, is_disch, is_sp, penalty: float
    ) -> Any:
        """Linear penalty for deviating from the previous plan's headline action.

        For a block whose previous action was 'charge', the penalty is
        penalty*(1 - is_charge); becoming idle/other costs `penalty`. Idle
        blocks are penalised for becoming active. Only ties are affected
        because `penalty` is far below real price spreads.
        """
        terms = []
        for i, (ts, _) in enumerate(blocks):
            prev = self._prev_actions.get(ts)
            if prev == "charge":
                terms.append(penalty * (1 - is_charge[i]))
            elif prev == "discharge":
                terms.append(penalty * (1 - is_disch[i]))
            elif prev == "sell_production":
                terms.append(penalty * (1 - is_sp[i]))
            elif prev in ("hold", "hold_idle"):
                # Both are non-grid-facing; penalise only a switch INTO a
                # grid-facing action, same as a plain hold.
                terms.append(penalty * (is_charge[i] + is_disch[i] + is_sp[i]))
        return pulp.lpSum(terms)
