# Battery dispatch — the MILP scheduler

Battery dispatch is a single **mixed-integer linear program (MILP)** solved over
the whole price horizon (today + tomorrow once OTE publishes, ~32 h of 15-minute
blocks). It is the system's only dispatch engine.

- Code: [`modules/growatt/milp_optimizer.py`](../loxone_smart_home/modules/growatt/milp_optimizer.py)
  (`MILPBatteryOptimizer`), with shared infrastructure in
  [`modules/growatt/optimizer.py`](../loxone_smart_home/modules/growatt/optimizer.py)
  (`BatteryOptimizer`: base-load profile, reserve floor, power-rate sizing).
- Solver: PuLP → CBC, run off the event loop via `asyncio.to_thread`.
- Output: `(charge_times, discharge_times, sell_production_times, decisions)`.

## Why an LP and not a rule set

A greedy/rule-based scheduler can take a small early gain that locks out a larger
later one (e.g. discharge into a mediocre evening price and then have nothing left
for a price spike two hours later). Expressing the whole day as one optimisation
gives a **true global optimum**, expresses constraints declaratively (SOC bounds,
power limits, reserve, export rules) instead of as hand-tuned gates, and produces
a **more stable schedule** across re-evaluations.

## Decision variables (per 15-minute block *i*)

All flows are energy in **kWh for that block** (a block is ¼ h, so kW·0.25):

| Variable | Meaning |
|---|---|
| `solar_to_load`, `solar_to_batt`, `solar_to_grid`, `curtail` | split of the block's solar production |
| `batt_to_load`, `batt_to_grid` | energy leaving the battery |
| `grid_to_load`, `grid_charge` | energy drawn from the grid |
| `soc[i+1]` | battery state of charge (kWh) at the end of the block |
| `is_charge[i]`, `is_disch[i]`, `is_sp[i]` | mutually-exclusive mode indicators (binary) |
| deferrable run binaries | "run controllable load X in block i" (see below) |

## Constraints (energy balances)

```
solar:   solar_to_load + solar_to_batt + solar_to_grid + curtail == solar[i]
load:    solar_to_load + batt_to_load  + grid_to_load           == consumption[i] + deferrable[i]
SOC:     soc[i+1] == soc[i] + (grid_charge + solar_to_batt)·η_chg
                              - (batt_to_load + batt_to_grid)/η_dis
power:   grid_charge + solar_to_batt          <= charge_max
         batt_to_load + batt_to_grid          <= discharge_max·(1 - is_charge[i])
reserve: soc[i+1] >= effective_min_soc[i] - reserve_short[i]      (soft)
bounds:  min_soc·cap/100 <= soc[i] <= max_soc·cap/100             (hard)
```

Modelling **every flow explicitly** (rather than a single net "battery drain") is
what makes the economics correct — it lets the objective price self-consumption,
export, and import as three distinct things.

### Efficiency convention — per-leg `√η`

Round-trip efficiency `η` (e.g. 0.85) is split across the two conversions: each
leg uses `η_leg = √η ≈ 0.922`. Charging *into* the battery multiplies by `η_leg`;
discharging *out* divides by `η_leg`. A round trip therefore loses `η_leg² = η`,
as it should. This same per-leg convention is used by the SOC continuity above,
by the reserve-floor helper, and by the power-rate sizing — so the kWh figures
everywhere agree.

## Objective (maximise net cash)

```
maximise  Σ_i [ + solar_to_grid[i] · (spot[i] - fee)          solar export (no dist, no wear)
                + batt_to_grid[i]  · (spot[i] - fee)          battery export gross
                - (grid_to_load[i] + grid_charge[i]) · (spot[i] + dist[i])   import cost
                - (batt_to_load[i] + batt_to_grid[i]) · amort  battery wear (once, on the way out)
                - curtail[i] · CURTAIL_PENALTY                 prefer banking free solar
                - mode_changed[i] · switch_penalty ]           damp churn vs the last plan
          + soc[n] · terminal_value                            value energy left at the horizon end
```

Why this is correct for the Czech (D57d) market:

- **Self-consumption is valued implicitly.** Serving load from the battery removes
  a `grid_to_load` term worth `(spot + dist)`; net of wear that is
  `(spot + dist − amort)`. There is no explicit "self-consumption reward" — it
  falls out of *not paying* the import term.
- **Export** nets to `(spot − fee − amort)` for the battery and `(spot − fee)` for
  solar. Distribution tariff is **not** paid on export, only on import.
- The explicit `grid_to_load` path keeps the LP feasible at low SOC and makes the
  cost of emptying the battery then re-importing visible, so the solver won't dump
  the battery to grid when self-consumption is worth more.

Key constants (`milp_optimizer.py`):

- `CURTAIL_PENALTY = 0.01` — tiny, just enough to prefer banking free solar.
- `SP_EXPORT_FLOOR_KWH = 0.1` — minimum surplus to actuate a passive block as a
  real export (`sell_production`) rather than plain hold.
- `MIN_GRID_DISCHARGE_FRACTION = 0.5` — a grid-export block must move ≥ half the
  power rate, because the inverter actuates discharge on/off at a fixed rate; a
  partial *final* block is still allowed so the solver never buys grid power just
  to top up a full export.

## The reserve-SOC floor

`effective_min_soc[i]` is a **soft** per-block floor that nudges the battery to
keep enough charge to bridge upcoming base load until the next recharge
opportunity. See [reserve-soc](#reserve-soc-floor-running-peak) below for the
maths.

### `RESERVE_SHORTFALL_FLOOR = 0.0`

The shortfall penalty is **zero**: the overnight reserve is *emergent from the
objective*. The objective already prices every future grid import, so the solver
keeps charge whenever importing later would be expensive and discharges when that
is more profitable — which is exactly the right economics. A non-zero per-block
penalty would double-count (the floor can sit above SOC for many consecutive
blocks, so its per-block accumulation snowballs into "grid-charge at the evening
peak to satisfy the reserve" — a guaranteed loss). The only **hard** battery
floor is `min_soc`, enforced as the `soc` variable's lower bound.

### Reserve floor (running-peak)

The shared helper `_compute_reserve_soc_per_block` estimates, for each block, how
much energy the battery must hold to cover base load until the next recharge
block (a block with > 1 kWh/h solar or a cheap-price top-up). It walks the blocks
from *now* to that recharge point and tracks a **running cumulative draw**,
taking the **peak**:

```
running += deficit/η_leg     (block where base load > solar)
running -= surplus·η_leg      (block where solar > base load)
reserve  = max over the window of running
```

The peak (not the net total) is what matters: a solar surplus *later* in the
window cannot retroactively cover a base-load deficit that occurs *earlier* — if
the battery would be drained before the surplus arrives, the earlier deficit is
the binding constraint. A single cheap-grid top-up opportunity in the window
reduces the required reserve by one block's charge.

## Deferrable loads (EMHASS-style co-optimisation)

Controllable loads (EV charge, hot-water boost, …) are **decision variables**, not
pre-scheduled on grid price alone. A per-block binary says "run load X now", its
draw enters the **load balance**, and the solver places the required blocks where
total cost is lowest — automatically charging into PV surplus and around battery
dispatch.

- Interruptible loads: one binary per in-window block.
- Non-interruptible loads: a single contiguous-run start variable.
- A small switch penalty damps churn of run placement across re-evaluations.

**Only list loads that are *not already* in the consumption history**
(`INVPowerToLocalLoad`); otherwise the forecast already learned them and the LP
would double-count, over-charging from grid. Windows are clamped to the first
window instance in the horizon so a load isn't scheduled past its current cycle's
deadline.

## Power-rate sizing (actuation)

The inverter is actuated at a `powerRate` percentage. `compute_charge_power_rates`
sizes that rate from the **grid-side** energy the rate governs, using the same
per-leg `√η` convention as the SOC model: a discharge of `ΔSOC` battery energy
delivers `ΔSOC·√η` to the grid; a charge of `ΔSOC` needs `ΔSOC/√η` of input.
Results are clamped to a configured C-rate ceiling.

## Safe fallback

When PuLP is missing, or the solve is infeasible or times out without an
incumbent, `optimize()` returns a minimal safe plan via `_safe_fallback`:

1. **Tier 1 — replay.** If the last good plan still covers the horizon, replay it
   verbatim. It is a recently-vetted MILP plan; replaying it through a brief solver
   hiccup beats forcing a hold. The decision-engine export/discharge/protection
   gates still apply on top, so this stays safe.
2. **Tier 2 — strictly conservative.** Otherwise hold every block and grid-charge
   only the cheapest blocks needed to lift SOC to the highest reserve floor —
   never discharging or exporting. Charge candidates are restricted to blocks
   *at or before* the reserve deadline, so cheap charging can't be deferred past
   the moment the reserve is actually needed.

`optimizer_enabled = False` means hold (no scheduling); the safety gates still
apply. There is no engine choice — MILP is the only engine.
