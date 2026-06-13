# Economics — pricing & reporting

Two distinct things share the word "economics":

1. **Dispatch economics** — the per-block prices the MILP optimises against. These
   *drive* the plan. See [battery-dispatch.md](battery-dispatch.md).
2. **Reporting economics** — what the dashboard shows for realised cost, savings,
   and battery arbitrage. These are computed *after the fact* from meters and never
   feed dispatch.

This document covers the pricing model both use, then the reporting reconstruction.

## The price of a kWh, per 15-minute block

Czech retail under the D57d tariff has three components. All are computed **per
block** (not as a daily average) so negative spot prices and the VT/NT tariff
bands are priced correctly.

| Term | Symbol | Source |
|---|---|---|
| Spot price | `spot[i]` | OTE day-ahead market (DAM), CZK/kWh; negatives are kept, never clamped |
| Distribution tariff | `dist(hour)` | VT (high) or NT (low) depending on the hour, CZK/kWh |
| Export sell fee | `fee` | fixed CZK/kWh deducted from export revenue |
| Battery wear | `amort` | CZK/kWh of throughput (optional separate export wear) |

**Import cost** of a kWh: `spot[i] + dist(hour)`
**Export revenue** of a kWh: `max(0, spot[i] − fee)` — distribution is **not** paid
on export, and you are never charged to export (the floor at 0).

### VT/NT distribution bands

`dist(hour)` returns `distribution_tariff_low` (NT) when the hour falls in
`low_tariff_hours` (a comma-separated list of `start-end` ranges, e.g. the D57d
schedule `0-10,11-12,13-14,15-17,18-24`), else `distribution_tariff_high` (VT). The
list must contain at least one range — an empty value would silently bill every
hour at the high VT rate and is rejected at config-validation time.

### Battery wear

Self-consumption is valued at the avoided import `(spot + dist − amort)`; export at
`(spot − fee − amort)`. The wear `amort` is charged **once**, on the energy leaving
the battery. An optional `battery_amortisation_export_czk` applies an extra
export-only wear penalty, leaving self-consumption at the base wear cost.

## Reporting reconstruction (dashboard)

The dashboard rebuilds the day's economics from **meters and actuals**, not from
the optimiser's plan. Four headline numbers, each tagged with a source/accuracy
flag:

| Number | How |
|---|---|
| **Import cost / export revenue (today)** | Integrate the cumulative daily-reset meters `EnergyToUserToday` / `EnergyToGridToday` per 15-min block, pricing each block at its own spot + tariff. |
| **Battery arbitrage (net)** | From per-block power flows: split charge into solar (free) vs grid (priced at buy), and discharge into self-consumption (valued at buy) vs export (valued at sell). |
| **Saved today (vs no battery)** | Realised cost minus a no-battery baseline reconstructed from the same PV/load power. |
| **Expected today (full day)** | Saved-so-far (actuals) + the remaining plan's projected gain. |

### Gotchas the reconstruction handles

These are subtle and worth knowing when reading the code:

- **Cumulative meters reset at midnight.** The query window reaches back far enough
  (≈ −28 h) to include the previous midnight reset, and a synthetic `(midnight, 0)`
  baseline is seeded so the opening 00:00 block is counted. Day-difference math is
  per-field and **gap-aware**: a block delta is only counted when that meter field
  was observed in the immediately-preceding hour, so a telemetry gap neither
  inflates nor deflates the total.
- **Flux window labelling.** `aggregateWindow` labels each window by its `_stop`
  time by default, so the 23:45–00:00 window is stamped 00:00 *today*. Any
  hour-keyed series compared against the optimiser's start-keyed grid must request
  `timeSrc:"_start"`; the power-flow path filters out the 00:00 stop-label artifact
  so yesterday's last block doesn't leak into today.
- **Arbitrage export attribution.** When PV surplus and battery discharge occur in
  the same 15-min mean block, only the export that could *not* have come from solar
  is attributed to the battery (`grid_export − solar_export`), so cheap solar export
  isn't charged against the battery.
- **EUR/CZK rate.** A successfully fetched CNB rate is cached for 24 h; a fallback
  (CNB unreachable) is cached only briefly so a transient outage is retried within
  minutes rather than pinning the flat fallback for a whole day.

### Weekly / monthly reconstruction

`_period_economics` differences the cumulative meters per hour over a window aligned
to **local midnight** (so the oldest day is a full day, not a partial one priced at
a single tariff), using historical OTE prices. The no-battery baseline is only
accumulated for an hour when both the meter delta was counted and PV/load telemetry
exists, so a gap can't credit a whole hour's baseline as phantom savings.
