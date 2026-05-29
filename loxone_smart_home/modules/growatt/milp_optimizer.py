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
    load:   solar_to_load + batt_to_load + grid_to_load          == consumption[i]
    SOC:    soc[i+1] == soc[i] + (grid_charge + solar_to_batt)*η
                                - batt_to_load - batt_to_grid
    power:  grid_charge + solar_to_batt <= charge_max
            batt_to_load + batt_to_grid <= discharge_max * (1 - is_charge)
    reserve: soc[i+1] >= effective_min_soc[i]

  Objective (maximise):
    + solar_to_grid * (spot - dist - fee)        solar export (no wear)
    + batt_to_grid  * (spot - dist - fee)        battery export gross
    - (grid_to_load + grid_charge) * (spot+dist) grid import cost
    - (batt_to_load + batt_to_grid) * amort      battery wear (once, on the way out)
    - curtail * CURTAIL_PENALTY                   prefer banking free solar
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
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

try:
    import pulp  # type: ignore
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

from .optimizer import (
    BlockDecision,
    SELL_PRODUCTION_MARGIN_CZK,
    SELL_PRODUCTION_MIN_EXCESS_KWH,
)


# Curtailment penalty: small enough never to distort a real price decision
# (real prices are 1-15 CZK/kWh), large enough above CBC's optimality
# tolerance that banking free solar is robustly preferred over wasting it.
CURTAIL_PENALTY = 0.01

# Flows below this many kWh are treated as solver numerical noise, not a real
# action, when classifying a block's headline action for the dashboard.
ACTION_EPS = 5e-3


def _sell_now_below_margin(price: float, dist: float, sell_fee: float) -> bool:
    """True when solar-export revenue (spot - dist - fee) doesn't clear the
    swap-profit margin, so sell-production mode isn't worth the overhead."""
    return (price - dist - sell_fee) <= SELL_PRODUCTION_MARGIN_CZK


class MILPBatteryOptimizer:
    """Mixed-integer linear program for whole-day battery scheduling.

    Same signature as `BatteryOptimizer.optimize()` so the controller can
    swap engines via a config flag without other code changes.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Re-use the greedy optimizer's reserve-SOC + bin-state helpers so
        # behaviour is consistent across engines.
        from .optimizer import BatteryOptimizer
        self._helper = BatteryOptimizer(logger=self.logger)
        self._last_decisions: List[BlockDecision] = []
        self._last_reserve_info: Dict = {}

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
                         battery_amortisation_czk):
        """Run the greedy engine and adopt its decisions as our own.

        Used when the MILP solve is infeasible, times out, or raises, so the
        controller always receives a usable schedule (the documented
        greedy-fallback contract).
        """
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
        )
        self._last_decisions = result[3]
        return result

    def optimize(
        self,
        blocks: List[Tuple[datetime, float]],
        solar_hourly: Dict[int, float],
        consumption_hourly: Dict[int, float],
        distribution_func,
        battery_capacity_kwh: float = 10.0,
        current_soc: float = 50.0,
        min_soc: float = 20.0,
        max_soc: float = 100.0,
        charge_rate_kw: float = 2.5,
        discharge_rate_kw: float = 2.5,
        discharge_power_pct: float = 25.0,
        efficiency: float = 0.85,
        sell_fee_czk: float = 0.5,
        battery_amortisation_czk: float = 2.0,
    ) -> Tuple[Set[datetime], Set[datetime], Set[datetime], List[BlockDecision]]:
        """Solve the full-horizon schedule via MILP (energy-flow model)."""
        if not PULP_AVAILABLE:
            self.logger.warning(
                "PuLP not installed — MILP unavailable, falling back to greedy"
            )
            return self._greedy_fallback(
                blocks, solar_hourly, consumption_hourly, distribution_func,
                battery_capacity_kwh, current_soc, min_soc, max_soc,
                charge_rate_kw, discharge_rate_kw, discharge_power_pct,
                efficiency, sell_fee_czk, battery_amortisation_czk,
            )

        if not blocks:
            return set(), set(), set(), []

        n = len(blocks)
        charge_max = charge_rate_kw * 0.25
        # House self-consumption can use the full inverter; grid EXPORT is
        # throttled to discharge_power_pct (the Growatt grid-first power rate).
        batt_to_grid_max = discharge_rate_kw * (discharge_power_pct / 100.0) * 0.25
        batt_out_max = discharge_rate_kw * 0.25
        max_battery_kwh = battery_capacity_kwh * max_soc / 100.0
        min_battery_kwh = battery_capacity_kwh * min_soc / 100.0
        start_battery_kwh = battery_capacity_kwh * current_soc / 100.0

        # Per-block price coefficients.
        prices = [p for _, p in blocks]
        dists = [distribution_func(ts.hour) for ts, _ in blocks]
        # Grid export gross (battery wear charged separately): spot - dist - fee.
        export_now = [prices[i] - dists[i] - sell_fee_czk for i in range(n)]
        # Battery export NET of wear (the "never export at a loss" gate).
        sell_revenue = [export_now[i] - battery_amortisation_czk for i in range(n)]
        # Grid import cost per kWh (paid on both load and battery charge).
        import_cost = [prices[i] + dists[i] for i in range(n)]

        # Per-block raw solar / consumption / structural surplus.
        solar_block: List[float] = []
        consumption_block: List[float] = []
        solar_excess: List[float] = []
        for ts, _ in blocks:
            s = solar_hourly.get(ts.hour, 0.0) / 4.0
            c = consumption_hourly.get(ts.hour, 0.0) / 4.0
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
        soc = [
            pulp.LpVariable(f"soc_{i}", min_battery_kwh, max_battery_kwh)
            for i in range(n + 1)
        ]

        prob += soc[0] == start_battery_kwh

        for i in range(n):
            s = solar_block[i]
            c = consumption_block[i]

            # Solar balance: every kWh of solar is used, banked, exported, or curtailed.
            prob += solar_to_load[i] + solar_to_batt[i] + solar_to_grid[i] + curtail[i] == s
            # Load balance: house draw served by solar, battery, and/or grid.
            prob += solar_to_load[i] + batt_to_load[i] + grid_to_load[i] == c

            # Battery power limits. Inflow capped by charge rate; outflow capped
            # by full inverter power AND forbidden during a charge block (the
            # inverter can't charge and discharge at once).
            prob += grid_charge[i] + solar_to_batt[i] <= charge_max
            prob += batt_to_load[i] + batt_to_grid[i] <= batt_out_max * (1 - is_charge[i])

            # SOC continuity: charge legs gain energy at efficiency; discharge
            # legs deplete the battery 1:1 (wear is priced in the objective).
            prob += soc[i + 1] == (
                soc[i]
                + (grid_charge[i] + solar_to_batt[i]) * efficiency
                - batt_to_load[i] - batt_to_grid[i]
            )
            prob += soc[i + 1] >= eff_min_kwh[i]

            # Activation linking → mode indicators (grid-facing actions).
            prob += grid_charge[i] <= charge_max * is_charge[i]
            prob += batt_to_grid[i] <= batt_to_grid_max * is_disch[i]
            prob += solar_to_grid[i] <= max(s, 0.0) * is_sp[i]
            # One grid-facing mode per block.
            prob += is_charge[i] + is_disch[i] + is_sp[i] <= 1

            # Never export from the battery at a loss.
            if sell_revenue[i] <= 0:
                prob += is_disch[i] == 0
            # Sell-production gates: only worthwhile above a volume floor AND a
            # swap margin (gate on the structural surplus s-c).
            if (
                solar_excess[i] < SELL_PRODUCTION_MIN_EXCESS_KWH
                or _sell_now_below_margin(prices[i], dists[i], sell_fee_czk)
            ):
                prob += is_sp[i] == 0
            else:
                prob += solar_to_grid[i] >= SELL_PRODUCTION_MIN_EXCESS_KWH * is_sp[i]

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
        if min_charge_cost > 0:
            terminal_value_per_kwh = min(
                terminal_value_per_kwh, min_charge_cost / efficiency
            )
        # Slightly discount the (estimated, future) terminal value so a
        # CERTAIN present-block self-consumption saving always wins ties
        # against hoarding energy for an uncertain later benefit. Without this
        # the solver is indifferent between covering a present deficit from the
        # battery vs importing and keeping SOC, and may leave the battery idle.
        terminal_value_per_kwh *= 0.99

        prob += pulp.lpSum(
            solar_to_grid[i] * export_now[i]
            + batt_to_grid[i] * export_now[i]
            - (grid_to_load[i] + grid_charge[i]) * import_cost[i]
            - (batt_to_load[i] + batt_to_grid[i]) * battery_amortisation_czk
            - curtail[i] * CURTAIL_PENALTY
            for i in range(n)
        ) + soc[n] * terminal_value_per_kwh

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
        for i, (ts, price) in enumerate(blocks):
            gc = val(grid_charge[i])
            bg = val(batt_to_grid[i])
            sg = val(solar_to_grid[i])
            action = "hold"
            net_value = 0.0
            # Classify the headline action by the largest grid-facing flow,
            # ignoring sub-watt-hour solver noise.
            if gc > ACTION_EPS and gc >= bg and gc >= sg:
                action = "charge"
                charge_times.add(ts)
                net_value = -gc * import_cost[i]
            elif bg > ACTION_EPS and bg >= sg:
                action = "discharge"
                discharge_times.add(ts)
                net_value = bg * sell_revenue[i]
            elif sg > ACTION_EPS:
                action = "sell_production"
                sp_times.add(ts)
                net_value = sg * export_now[i]

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
        if effective_min_socs:
            self._helper._last_reserve_info["effective_min_soc"] = round(
                effective_min_socs[0], 1
            )

        return charge_times, discharge_times, sp_times, decisions
