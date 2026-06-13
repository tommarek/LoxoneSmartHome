# Design & maths documentation

This folder explains the **approaches and mathematics** behind the energy side of
the service — battery dispatch, pricing/economics, and forecasting. It is the
conceptual companion to the code: the code comments say *what a line does*, these
documents say *why the model is shaped this way*.

For operational topics (deployment, the web/controller split, the InfluxDB
schema, testing) see the root [`CLAUDE.md`](../CLAUDE.md) and
[`DEPLOYMENT.md`](../loxone_smart_home/DEPLOYMENT.md).

## Contents

| Document | What it covers |
|---|---|
| [battery-dispatch.md](battery-dispatch.md) | The MILP scheduler: decision variables, energy-flow balances, the objective function, the efficiency convention, the reserve-SOC floor, deferrable-load co-optimisation, power-rate sizing, and the safe fallback. |
| [economics.md](economics.md) | How energy is priced per 15-minute block (spot + VT/NT distribution tariff, export fee, battery wear), and how the dashboard reconstructs realised cost / savings / arbitrage from meters. |
| [forecasting.md](forecasting.md) | The solar production model (binned hierarchy, multi-source consensus, intraday calibration) and the consumption forecast (temperature-binned and ML variants). |
| [decision-engine.md](decision-engine.md) | Inverter modes, the SPH hardware quirks that constrain them, the decision-priority gate chain, and high-load (EV/heating) protection. |

## The big picture

```
        prices (OTE/DAM)          weather (OpenMeteo/Solcast)
              │                            │
              ▼                            ▼
       ┌──────────────┐            ┌────────────────┐
       │ consumption  │            │ solar          │
       │ forecast     │            │ forecast       │
       └──────┬───────┘            └───────┬────────┘
              │  load[i] (kWh/block)       │  solar[i] (kWh/block)
              └───────────────┬────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  MILP scheduler      │  one MIP over the 15-min blocks
                   │  (battery-dispatch)  │  → charge / discharge / sell / hold
                   └──────────┬───────────┘
                              ▼
                   ┌─────────────────────┐
                   │  decision engine     │  maps the plan to an inverter mode,
                   │  + safety gates      │  applies SPH quirks + protections
                   └──────────┬───────────┘
                              ▼
                       Growatt SPH inverter
```

Everything downstream of the forecasts is driven by a single optimisation: the
MILP decides the *plan*, and the decision engine actuates it on the hardware
subject to the inverter's physical behaviour and a set of safety gates.
