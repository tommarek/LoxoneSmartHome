# Forecasting

The MILP needs two inputs per 15-minute block: expected **solar production**
(`solar[i]`) and expected **household consumption** (`consumption[i]`). Both are
forecast from history + weather.

## Solar production forecast

Code: [`modules/growatt/solar_forecast.py`](../loxone_smart_home/modules/growatt/solar_forecast.py),
[`solcast_forecast.py`](../loxone_smart_home/modules/growatt/solcast_forecast.py).

### Learned production model

`SolarProductionModel` is trained on ~730 days of historical production paired with
weather. It bins each hour by physical drivers and stores the median production per
bin, with a 3-level fallback hierarchy:

```
3-D bin: (radiation bucket, cloud bucket, sun-altitude bucket)
   ↓ too few samples
2-D bin: (radiation bucket, cloud bucket)
   ↓ too few samples
interpolate → global median
```

`hit_level_counts` tracks which level served each lookup, for observability and for
gating trust in the consensus (below).

Training details:

- **Two-pass.** Build a rough model, use it to identify *curtailed* hours
  (SOC = 100 % with low load — production was clipped, not weather-limited), drop
  those, and rebuild. Otherwise curtailment would teach the model to under-predict.
- **Outlier removal** uses interpolated IQR quartiles (the raw index form would put
  Q3 at the maximum for a minimum-size bin, so the upper fence would never trim
  anything for exactly the sparse, noisy bins that need it).
- A physics cap at nameplate capacity bounds runaway estimates.

### Multi-source consensus

`build_consensus` combines the learned model with up to three weather-based sources
(forecast.solar API, Solcast, OpenMeteo GHI-with-temperature-derating). It averages
when sources agree (< 30 % divergence). On divergence, trust is gated by the
model's per-hour bin level: a well-supported `3d`/`2d` hit is trusted in both
directions; an `interpolate`/`global` hit defers to the non-model average. A model
zero only vetoes other sources when it is a genuine below-horizon zero.

### Intraday calibration

Confidence is auto-tuned by comparing past forecasts against actuals. The consensus
is persisted to InfluxDB (`solar_forecast_history`) so calibration survives
restarts. Calibration excludes *censored* hours (export disabled / inverter off /
battery full) so a deliberately-curtailed hour doesn't drag the ratio down — the
hour-state reconstruction takes the **min** over the entering state and any
within-hour change, so a partially-disabled hour counts as disabled.

### Solcast quota

The free tier allows 10 requests/day per rooftop; the client throttles to ≤ 9/day
with a UTC-monotonic interval guard. Quota state is persisted next to the config
overrides and survives restarts. Any received HTTP response counts (timeouts too —
the request may have reached Solcast's ledger); only provably-unsent connect errors
are refunded. A partial multi-array fetch (e.g. a rate-limit after the first array)
keeps the previous full forecast rather than overwriting it with an under-count.

## Consumption forecast

Code: [`modules/growatt/consumption_forecast.py`](../loxone_smart_home/modules/growatt/consumption_forecast.py)
(binned, default), [`ml_consumption_forecast.py`](../loxone_smart_home/modules/growatt/ml_consumption_forecast.py)
(opt-in ML).

### Binned model (default)

A temperature-aware lookup: `(temp_bucket, hour, is_weekend) → median kWh`, with 30
temperature buckets (−20…40 °C in 2 °C steps), trained on ~365 days. Designed for an
all-electric house where heating load dominates in winter.

- IQR outlier removal (same interpolated-quartile approach as the solar model).
- A bin needs ≥ 4 samples to be trusted as an exact-key median; sparser bins fall
  back through: opposite day-type → adjacent temperature buckets → hourly median →
  global median. (Gating sparse bins out entirely would collapse a young install to
  the 1.0 kWh hard default, so their samples still feed the fallbacks.)
- Rebuilt weekly; a failed build backs off an hour rather than retrying every tick.

Tomorrow is predicted with OpenMeteo forecast temperatures, falling back to the
last-24 h actuals when no forecast is available.

### ML model (opt-in)

`CONSUMPTION_FORECAST_ENGINE=ml` swaps in a skforecast recursive forecaster (lagged
features + temperature exogenous input). It degrades gracefully to the binned model
when skforecast is missing or training declines. The schema version is bumped when
features change so a stale persisted model is invalidated.

## A note on time-series hour keys (Flux)

Every hour- or day-keyed `aggregateWindow` query that feeds a **model** or an
hour-keyed lookup must pass `timeSrc:"_start"`. Flux's default labels each window by
its `_stop` time, which shifts every hour key by +1. Display-only chart queries may
use the default stop label, but any new hour-keyed *consumer* needs `_start`.
