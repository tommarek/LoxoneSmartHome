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
ML_MODEL_SCHEMA_VERSION = 3


class MLConsumptionForecast:
    """Train and predict household consumption with a recursive AR model."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._forecaster = None
        self._last_train_time: Optional[datetime] = None
        self._last_train_index_end: Optional[datetime] = None
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
        if self._forecaster is None or self._last_train_time is None:
            return True
        if self._schema_version != ML_MODEL_SCHEMA_VERSION:
            return True
        age = (datetime.now() - self._last_train_time).total_seconds()
        return age > max_age_days * 86400

    @staticmethod
    def _build_exog(timestamps, temps_lookup: Dict[str, float]):
        """Build the exogenous feature dataframe aligned to `timestamps`."""
        import pandas as pd  # type: ignore
        rows = []
        for ts in timestamps:
            key = ts.strftime("%Y-%m-%d-%H")
            rows.append({
                "hour": ts.hour,
                "weekday": ts.weekday(),
                "temp": temps_lookup.get(key, 10.0),
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
        if not SKFORECAST_AVAILABLE:
            self.logger.info(
                "skforecast not installed — ML consumption forecaster disabled"
            )
            return False

        # Quantile to forecast: 0.5 = median (RandomForest, the default). A
        # higher value forecasts an upper bound on demand (quantile gradient
        # boosting) so the optimizer keeps a larger reserve. Read defensively
        # so a missing/old settings object just yields the median behaviour.
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
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
        temperature_query = f'''
from(bucket: "{bucket_loxone}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "temperature" and r._field == "temperature_outside")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
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
                    cons_ts[key] = t

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
        y_full = y_series.asfreq("1h")
        if y_full.isna().mean() > 0.3:
            self.logger.info(
                f"ML training: too many gaps ({y_full.isna().mean():.0%}) "
                f"in hourly series — skipping"
            )
            return False
        # Interpolate the small remaining gaps.
        y_full = y_full.interpolate(limit=6).ffill().bfill()

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
            # skforecast renamed the first constructor arg from `regressor`
            # (<=0.13) to `estimator` (>=0.14). Support both so the module
            # works across the version range pinned in requirements.
            try:
                fc = ForecasterRecursive(estimator=regressor, lags=24)
            except TypeError:
                fc = ForecasterRecursive(regressor=regressor, lags=24)
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
            else "median RandomForest"
        )
        self.logger.info(
            f"ML consumption model trained: {len(y_full)} hours, "
            f"lags=24, {len(self._exog_cols)} exog features, {model_kind}"
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
            temps_lookup = {
                f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}":
                    forecast_temps.get(t.hour, 10.0)
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
