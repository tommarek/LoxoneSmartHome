"""ML-based consumption forecaster (autoregressive + exogenous covariates).

Drop-in alternative to the temperature-binned ConsumptionForecast that
uses a `skforecast` autoregressive recursive model with:
- past hourly load (lags)
- hour-of-day, day-of-week as exogenous covariates
- outdoor temperature as exogenous covariate

The advantage over binned-median: captures lag patterns (this Friday
≈ last Friday) and continuous (rather than discrete) temperature
response. Falls back silently to the binned model if skforecast isn't
installed or training fails.

Inspired by EMHASS's ML forecaster module.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
    from sklearn.ensemble import (  # type: ignore
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from skforecast.recursive import ForecasterRecursive  # type: ignore
    SKFORECAST_AVAILABLE = True
except Exception:
    try:
        # Older skforecast API (<0.13) used a different import path.
        from skforecast.ForecasterAutoreg import ForecasterAutoreg as ForecasterRecursive  # type: ignore
        from sklearn.ensemble import (  # type: ignore
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        import pandas as pd  # type: ignore
        SKFORECAST_AVAILABLE = True
    except Exception:
        SKFORECAST_AVAILABLE = False


# Bumped if the feature set or training logic changes incompatibly.
# v2: training/prediction now bucket InfluxDB UTC timestamps into local time.
# v3: inverter_on carry-forward also converts change records to local (was
#     comparing local hour-keys against UTC change timestamps).
# v4: hourly aggregates are start-labeled (timeSrc: "_start") — the default
#     stop-labeling shifted every hour key +1h; lags extended to 48h/168h.
ML_MODEL_SCHEMA_VERSION = 4


class MLConsumptionForecast:
    """Train and predict household consumption with a recursive AR model."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._forecaster = None
        self._last_train_time: Optional[datetime] = None
        self._last_train_index_end: Optional[datetime] = None
        # Set on EVERY build attempt (success or failure). The controller
        # polls needs_rebuild() every minute, so a failing build (cold start,
        # InfluxDB outage) must back off instead of re-running heavy 365-day
        # queries on every poll.
        self._last_build_attempt: Optional[datetime] = None
        self.failed_build_backoff_seconds: int = 3600
        self._exog_cols: List[str] = []
        self._schema_version: int = 0
        # The (forecaster, training-end) pair read by predict_hourly, which runs
        # in a worker thread (asyncio.to_thread). Published as one immutable
        # tuple in a single assignment so a concurrent prediction can never
        # observe a new forecaster paired with a stale training-end.
        self._predict_state: Optional[Tuple[Any, datetime]] = None
        # Local timezone the training hours were bucketed in (so prediction
        # uses the same wall-clock convention the optimizer expects).
        self._local_tz = None

    @property
    def is_trained(self) -> bool:
        return self._forecaster is not None

    @property
    def available(self) -> bool:
        return SKFORECAST_AVAILABLE

    def needs_rebuild(self, max_age_days: int = 1) -> bool:
        """Whether the model should be retrained.

        Defaults to a 1-day max age: a recursive AR forecaster anchored at its
        training-end cannot meaningfully "skip ahead" many days, so it must be
        refreshed often to keep the prediction horizon close to now.
        """
        if self._failed_attempt_recently():
            return False
        if self._forecaster is None or self._last_train_time is None:
            return True
        if self._schema_version != ML_MODEL_SCHEMA_VERSION:
            return True
        age = (datetime.now() - self._last_train_time).total_seconds()
        return age > max_age_days * 86400

    def _failed_attempt_recently(self) -> bool:
        """True if the last build attempt failed within the backoff window."""
        if self._last_build_attempt is None:
            return False
        if (
            self._last_train_time is not None
            and self._last_train_time >= self._last_build_attempt
        ):
            return False  # last attempt succeeded — normal cadence applies
        age = (datetime.now() - self._last_build_attempt).total_seconds()
        return age < self.failed_build_backoff_seconds

    @staticmethod
    def _build_exog(timestamps, temps_lookup: Dict[str, float]):
        """Build the exogenous feature dataframe aligned to `timestamps`.

        Hours missing from `temps_lookup` fall back to the mean of the
        available training temperatures. NOTE: over a full training year
        that mean is ≈10°C in CZ anyway — this mainly avoids a hardcoded
        constant; the real per-hour improvement is in the PREDICTION path,
        which uses the nearest available hour's temperature
        (_fallback_temperature). 10°C is used only when no temperatures are
        available at all.
        """
        import pandas as pd  # type: ignore
        if temps_lookup:
            default_temp = sum(temps_lookup.values()) / len(temps_lookup)
        else:
            default_temp = 10.0
        rows = []
        for ts in timestamps:
            key = ts.strftime("%Y-%m-%d-%H")
            rows.append({
                "hour": ts.hour,
                "weekday": ts.weekday(),
                "temp": temps_lookup.get(key, default_temp),
            })
        return pd.DataFrame(rows, index=timestamps)

    async def build_model(
        self, influxdb_client: Any, settings: Any, local_tz: Any = None
    ) -> bool:
        """Train the recursive AR model on the last 365 days of data.

        `local_tz`: timezone to bucket InfluxDB's UTC timestamps into, so the
        model's hour-of-day features and prediction index match the optimizer's
        local-hour grid. Without it the forecast is offset by the UTC->local
        difference (1-2h) and the consumption peak lands in the wrong block.
        """
        self._local_tz = local_tz
        # Record the attempt up front so needs_rebuild() can back off if this
        # attempt fails (missing deps, query error, too little data, ...).
        self._last_build_attempt = datetime.now()
        if not SKFORECAST_AVAILABLE:
            self.logger.info(
                "skforecast not installed — ML consumption forecaster disabled"
            )
            return False

        # Quantile to forecast: 0.5 = the default RandomForest, which predicts
        # the conditional MEAN (not a true median). A higher value forecasts
        # an upper bound on demand (quantile gradient boosting) so the
        # optimizer keeps a larger reserve. Read defensively so a missing/old
        # settings object just yields the default (mean) behaviour.
        try:
            quantile = float(
                getattr(
                    getattr(settings, "growatt", None),
                    "ml_consumption_quantile",
                    0.5,
                )
            )
        except (TypeError, ValueError):
            quantile = 0.5
        quantile = min(0.95, max(0.5, quantile))
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return False

        # Query consumption + temperature (same as binned model).
        bucket_solar = settings.influxdb.bucket_solar
        bucket_loxone = settings.influxdb.bucket_loxone
        consumption_query = f'''
from(bucket: "{bucket_solar}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
'''
        temperature_query = f'''
from(bucket: "{bucket_loxone}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "temperature" and r._field == "temperature_outside")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
'''
        inverter_on_query = f'''
from(bucket: "{bucket_solar}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "inverter_state" and r._field == "inverter_on")
  |> sort(columns: ["_time"])
'''
        try:
            consumption_result = await influxdb_client.query(consumption_query)
            temperature_result = await influxdb_client.query(temperature_query)
            inverter_on_result = await influxdb_client.query(inverter_on_query)
        except Exception as e:
            self.logger.warning(f"ML training query failed: {e}")
            return False

        if not consumption_result or not temperature_result:
            return False

        def _to_local(t: datetime) -> datetime:
            # InfluxDB timestamps are UTC-aware; convert to the optimizer's
            # local wall-clock then drop tzinfo so all keys/indices are naive
            # local (pandas asfreq needs a consistent naive index).
            if local_tz is not None and getattr(t, "tzinfo", None) is not None:
                t = t.astimezone(local_tz)
            return t.replace(tzinfo=None)

        # Parse to dicts keyed by "YYYY-MM-DD-HH" string for alignment.
        cons_by_hour: Dict[str, float] = {}
        cons_ts: Dict[str, datetime] = {}
        for table in consumption_result:
            for record in table.records:
                t = _to_local(record.get_time())
                key = t.strftime("%Y-%m-%d-%H")
                v = record.get_value()
                if isinstance(v, (int, float)) and v > 0:
                    cons_by_hour[key] = float(v) / 1000.0  # kWh
                    # Floor to the hour: this timestamp becomes the training
                    # series index, and asfreq("1h") needs it ON the hour. The
                    # raw record time is NOT aligned (the first -365d window is
                    # labelled at the non-aligned range start, e.g. :51:09), and
                    # a single off-hour anchor makes asfreq miss every real
                    # point → a fabricated >24h gap → ML wrongly declines.
                    cons_ts[key] = t.replace(minute=0, second=0, microsecond=0)

        temp_by_hour: Dict[str, float] = {}
        for table in temperature_result:
            for record in table.records:
                t = _to_local(record.get_time())
                key = t.strftime("%Y-%m-%d-%H")
                v = record.get_value()
                if isinstance(v, (int, float)):
                    temp_by_hour[key] = float(v)

        # Carry-forward inverter_on so we can exclude bypass hours.
        from .consumption_forecast import (
            _carry_forward_hourly, MAX_REALISTIC_HOURLY_KWH,
        )
        inverter_on_by_hour = _carry_forward_hourly(
            inverter_on_result, cons_by_hour.keys(), local_tz
        )

        # Build cleaned training series (intersection + filters).
        keys = sorted(set(cons_by_hour.keys()) & set(temp_by_hour.keys()))
        clean_keys: List[str] = []
        for k in keys:
            if inverter_on_by_hour.get(k, 1) == 0:
                continue  # inverter off → grid passthrough, bogus
            if cons_by_hour[k] > MAX_REALISTIC_HOURLY_KWH:
                continue  # outlier cap (manual-bypass artifact)
            clean_keys.append(k)

        if len(clean_keys) < 24 * 14:  # need ≥ 2 weeks for AR
            self.logger.info(
                f"ML training: only {len(clean_keys)} clean hours — "
                f"need ≥{24*14}, skipping"
            )
            return False

        # Build sorted time series + exog frame.
        timestamps = [cons_ts[k] for k in clean_keys]
        y_series = pd.Series(
            [cons_by_hour[k] for k in clean_keys],
            index=pd.DatetimeIndex(timestamps),
            name="consumption_kwh",
        )
        # skforecast needs a regular frequency for recursive forecasting.
        # Reindex to hourly and forward-fill small gaps; bail if too gappy.
        y_series = y_series.sort_index()
        # Hour-floored index can in principle collide (e.g. a partial first
        # window flooring onto an existing hour); keep the last so asfreq's
        # unique-index requirement holds.
        y_series = y_series[~y_series.index.duplicated(keep="last")]
        y_full = y_series.asfreq("1h")
        # Gap policy, measured on the RAW hourly series before any filling
        # (measuring after a partial interpolate under-reports the true
        # outage length): bridging a contiguous gap >24h would fabricate
        # training data (paired with default temps), but declining outright
        # would disable the ML engine for up to a YEAR after a single long
        # outage anywhere in the 365-day window (temp-sensor outage,
        # inverter-off stretch, vacation — common events), retried hourly
        # with the heavy 365-day queries. Instead, TRUNCATE to the
        # contiguous data AFTER the most recent >24h gap and train on that;
        # decline only if too little remains. Small gaps (≤24h) on the
        # surviving series are interpolated; short ragged edges (which
        # interpolation can't reach) get bounded ffill/bfill.
        min_train_hours = 24 * 14  # same floor as the clean-hours check
        na_mask = y_full.isna()
        if na_mask.any():
            # Contiguous NaN runs: id -> length in hours.
            run_ids = (~na_mask).cumsum()[na_mask]
            run_lengths = run_ids.value_counts()
            long_runs = run_lengths[run_lengths > 24]
            if not long_runs.empty:
                # End (last NaN hour) of the MOST RECENT >24h gap; keep only
                # the contiguous data after it.
                gap_end = run_ids[run_ids.isin(long_runs.index)].index.max()
                kept = y_full.index > gap_end
                dropped = int((~kept).sum())
                y_full = y_full[kept]
                if len(y_full) < min_train_hours:
                    self.logger.info(
                        f"ML training: only {len(y_full)}h of contiguous "
                        f"data after the most recent >24h gap (ends "
                        f"{gap_end}) — need ≥{min_train_hours}h, declining"
                    )
                    return False
                self.logger.info(
                    f"ML training: {len(long_runs)} contiguous gap(s) >24h "
                    f"(longest {int(run_lengths.max())}h) — truncated "
                    f"{dropped}h ending {gap_end}, training on the "
                    f"{len(y_full)}h after it"
                )
        if y_full.isna().mean() > 0.3:
            self.logger.info(
                f"ML training: too many gaps ({y_full.isna().mean():.0%}) "
                f"in hourly series — skipping"
            )
            return False
        na_mask = y_full.isna()
        if na_mask.any():
            y_full = y_full.interpolate(limit=24)
            # Bounded fills for the series edges (leading NaNs can't be
            # interpolated forward).
            y_full = y_full.ffill(limit=3).bfill(limit=3)
            if y_full.isna().any():
                self.logger.info(
                    "ML training: NaNs remain after bounded gap-filling "
                    "(unfillable series edge) — declining (recursive "
                    "forecaster needs a regular series)"
                )
                return False

        exog = self._build_exog(y_full.index, temp_by_hour)

        def _fit_blocking():
            if quantile > 0.5:
                # Native quantile regression for an upper-bound load forecast.
                regressor = GradientBoostingRegressor(
                    loss="quantile", alpha=quantile,
                    n_estimators=100, max_depth=3,
                    learning_rate=0.1, random_state=0,
                )
            else:
                regressor = RandomForestRegressor(
                    n_estimators=60, max_depth=10, n_jobs=-1, random_state=0
                )
            # Lags: the full last day, plus the same hour 2 days and 1 week
            # back — lags=24 alone cannot see last week's pattern.
            lags = list(range(1, 25)) + [48, 168]
            # skforecast renamed the first constructor arg from `regressor`
            # (<=0.13) to `estimator` (>=0.14). Support both so the module
            # works across the version range pinned in requirements.
            try:
                fc = ForecasterRecursive(estimator=regressor, lags=lags)
            except TypeError:
                fc = ForecasterRecursive(regressor=regressor, lags=lags)
            fc.fit(y=y_full, exog=exog)
            return fc

        try:
            # RandomForest.fit is CPU-bound and blocking; run it off the event
            # loop so MQTT handling and the 15-min decision cadence aren't
            # frozen for the (multi-second) training time.
            import asyncio
            forecaster = await asyncio.to_thread(_fit_blocking)
        except Exception as e:
            self.logger.warning(f"ML training failed: {e}")
            return False

        self._forecaster = forecaster
        self._last_train_time = datetime.now()
        self._last_train_index_end = y_full.index[-1].to_pydatetime()
        self._exog_cols = list(exog.columns)
        self._schema_version = ML_MODEL_SCHEMA_VERSION
        # Publish the worker-thread-visible pair LAST, as a single immutable
        # tuple, so predict_hourly observes forecaster and training-end together.
        self._predict_state = (forecaster, self._last_train_index_end)
        model_kind = (
            f"quantile GBR (α={quantile:.2f})" if quantile > 0.5
            else "mean RandomForest"
        )
        self.logger.info(
            f"ML consumption model trained: {len(y_full)} hours, "
            f"lags=1-24,48,168, {len(self._exog_cols)} exog features, "
            f"{model_kind}"
        )
        return True

    def predict_hourly(
        self,
        forecast_temps: Dict[int, float],
        target_date: date,
        steps: Optional[int] = None,
    ) -> Dict[int, float]:
        """Predict hourly consumption (kWh) for `target_date`.

        A recursive AR forecaster can only forecast the periods *immediately
        following* its training window, so to cover `target_date` we must
        forecast every hour from training-end+1h through the end of
        `target_date` and then keep the hours belonging to that date. If the
        gap is implausibly large (stale model), we bail to {} so the caller
        falls back — and log it, so a silently-dead ML engine is visible.

        Returns hour-of-target_date → kWh. Falls back to {} on failure.
        """
        # Read the model state as a SINGLE reference. predict_hourly runs in a
        # worker thread (asyncio.to_thread) while build_model may replace the
        # model on the event loop; reading the published (forecaster, train_end)
        # tuple once guarantees the pair is consistent — a separate-attribute
        # snapshot could observe a new forecaster with a stale training-end.
        state = self._predict_state
        if state is None:
            return {}
        forecaster, train_end = state
        try:
            import pandas as pd  # type: ignore
            start = train_end + timedelta(hours=1)
            # Forecast through 23:00 of target_date (last_train_index_end is
            # naive local, matching the optimizer's local-hour grid).
            target_end = datetime(
                target_date.year, target_date.month, target_date.day, 23, 0, 0
            )
            if target_end < start:
                # target_date is entirely before the training window — nothing
                # sensible to predict.
                self.logger.debug(
                    "ML predict: target_date precedes training window"
                )
                return {}
            need_steps = int((target_end - start).total_seconds() // 3600) + 1
            if steps is not None:
                need_steps = max(need_steps, steps)
            # Guard against a stale model forcing an enormous recursive horizon
            # (each step compounds error and cost). ~10 days is the ceiling.
            if need_steps > 24 * 10:
                self.logger.warning(
                    f"ML model stale: would need {need_steps} forecast steps "
                    f"to reach {target_date} — skipping (rebuild needed)"
                )
                return {}
            idx = pd.date_range(start=start, periods=need_steps, freq="1h")
            # Missing forecast hours fall back to the nearest available
            # hour's temperature (10°C only when no temps exist at all).
            from .consumption_forecast import _fallback_temperature
            temps_lookup = {
                f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}":
                    forecast_temps.get(
                        t.hour, _fallback_temperature(forecast_temps, t.hour)
                    )
                for t in idx
            }
            exog = self._build_exog(idx, temps_lookup)
            preds = forecaster.predict(steps=need_steps, exog=exog)
        except Exception as e:
            self.logger.debug(f"ML predict failed: {e}")
            return {}

        result: Dict[int, float] = {}
        for ts, val in preds.items():
            if ts.date() == target_date:
                result[ts.hour] = max(0.0, float(val))
        if not result:
            self.logger.warning(
                f"ML predict produced no hours for {target_date} "
                f"(train-end {train_end}) — falling back"
            )
        return result
