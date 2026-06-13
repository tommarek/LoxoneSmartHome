# Decision engine & inverter actuation

The MILP produces a *plan* (sets of charge / discharge / sell-production /
battery-hold blocks). The **decision engine** turns that plan, plus live
conditions, into a concrete inverter mode each evaluation tick — subject to the
hardware's behaviour and a chain of safety gates.

Code: [`modules/growatt/decision_engine.py`](../loxone_smart_home/modules/growatt/decision_engine.py),
[`mode_manager.py`](../loxone_smart_home/modules/growatt/mode_manager.py),
[`modules/growatt_controller.py`](../loxone_smart_home/modules/growatt_controller.py).

## Decision priority

Rules are evaluated highest-priority first; the first match wins:

| Priority | Rule | Effect |
|---|---|---|
| 1 | Manual override | User-forced mode for a duration |
| 2 | Scheduled battery charging | A planned charge block |
| 3 | High-load protection | EV/heating active → hold battery off discharge |
| 4 | Scheduled mode | The optimiser's plan for this block (discharge / sell / hold) |
| 5 | Default | Regular operation (`load_first`) |

## The SPH inverter quirks (critical)

The Growatt SPH does **not** behave the way the mode names suggest. Confirmed from
telemetry:

- **`battery_first` charges the battery up to `stop_soc` *from the grid*** — even
  with `ac_charge = False`. The flag does **not** gate grid-charge.
- **`load_first` discharges the battery** to cover any load deficit; `stop_soc` is
  **not** a discharge floor in this mode.
- **`grid_first` maintains `stop_soc`** (charges up to it, discharges down to it).

Consequences encoded in `_build_desired_state`:

- The "battery passive — no discharge **and** no grid-charge" modes
  (`sell_production`, `battery_hold`, `high_load_protected`) use **`battery_first`
  with `stop_soc` pinned to the *live* SOC** (floored, so it never rounds up and
  leaves grid-charge headroom). `load_first` / `stop_soc = max_soc` would *not* hold
  the battery. You cannot bank surplus solar during a hold without also grid-
  charging, so a hold exports surplus instead of storing it.
- Surplus-solar export is actuated as `sell_production` (grid_first @ `stop_soc` =
  live SOC), because plain `load_first` would silently charge the surplus into the
  battery before spilling to grid.

## Safety gates

On top of the chosen mode, gates run every tick:

- **Battery protection** — never discharge below `min_soc` (a hard SOC bound).
- **Export gating** (`EXP:OFF/ON`) — export is only enabled when the price clears
  the export threshold, so the system won't dump to grid at a loss regardless of
  what the plan says.
- **Inverter on/off** — the inverter is switched off via Modbus when the all-in
  price crosses a negative threshold (it's not worth running).

## High-load protection (EV / heating)

Stops the battery **discharging** while a big load runs, so stored energy isn't
drained into an EV charge or electric heating. It is a Priority-3 decision node plus
`and not high_loads_active` guards on every discharge/sell/hold node.

Two detection sources, merged:

- **Heating** — Loxone relays via the `loxone/status` MQTT cache (event-driven,
  `tag1=heating`).
- **EV** — teslamate → InfluxDB (`ev` measurement). Polled, because the EV state
  isn't in the MQTT cache.

Two hardware gotchas the detection handles:

1. `ev_charging` is written **on change**, so it ages out of a 30-minute query
   window mid-charge — detection must not rely on it being present every poll.
2. `ev_charging_power` is in **kW** but the threshold is in **W**, so the compare is
   `(power_kw·1000) > threshold_w`. (The dashboard's `ev_power` is kW — don't divide
   again.)

Note that protection composes with charging: during a scheduled cheap/negative-price
block while an EV charges, the system still grid-charges the battery (`battery_first`,
AC on) — protection only blocks *discharge*, not charge.
