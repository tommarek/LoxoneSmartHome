"""Temperature-aware consumption forecast using historical InfluxDB data.

Builds a lookup model: (temperature_bucket, hour, weekday) → expected kWh,
using all available historical data. Designed for an all-electric house
where heating load dominates in winter.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConsumptionModel:
    """Binned consumption lookup model.

    Keys: (temp_bucket_index, hour, is_weekend) → median kWh consumption.
    Temp buckets: -20 to 40°C in 2°C steps = 30 buckets.
    """
    # (temp_bucket, hour, is_weekend) -> list of historical kWh values
    bins: Dict[Tuple[int, int, bool], List[float]] = field(default_factory=dict)
    # After build: (temp_bucket, hour, is_weekend) -> median kWh
    medians: Dict[Tuple[int, int, bool], float] = field(default_factory=dict)
    # Fallback: hour -> median kWh (across all temps/days)
    hourly_fallback: Dict[int, float] = field(default_factory=dict)
    # Global fallback
    global_median: float = 1.0
    # Metadata
    data_points: int = 0
    date_range: str = ""
    built_at: Optional[datetime] = None

    TEMP_BUCKET_SIZE: int = 2  # °C per bucket
    TEMP_MIN: int = -20
    TEMP_MAX: int = 40
    # Minimum samples for a bin to be trusted as a median. A 1-3 sample
    # "median" is really just noise, yet an exact-key hit would be preferred
    # over much richer fallback data (opposite day type, adjacent buckets).
    MIN_BIN_SAMPLES: int = 4

    @staticmethod
    def temp_to_bucket(temp: float) -> int:
        """Convert temperature to bucket index (0-29, 30 buckets of 2°C)."""
        clamped = max(-20, min(40, temp))
        # min(.., 29) folds the exact-40°C edge into the top bucket so we keep
        # exactly 30 buckets (0-29) rather than spawning a lone bucket 30.
        return min(29, int((clamped + 20) / 2))

    def build(self) -> None:
        """Compute medians from raw bins, with IQR outlier removal."""
        self.medians = {}
        self.hourly_fallback = {}
        hourly_all: Dict[int, List[float]] = {}
        all_bin_medians: List[float] = []

        for key, values in self.bins.items():
            # IQR outlier removal
            sorted_vals = sorted(values)
            q1_idx = len(sorted_vals) // 4
            q3_idx = 3 * len(sorted_vals) // 4
            q1 = sorted_vals[q1_idx]
            q3 = sorted_vals[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [v for v in values if lower <= v <= upper]
            bin_median = statistics.median(filtered) if filtered else statistics.median(values)

            if len(values) >= self.MIN_BIN_SAMPLES:
                self.medians[key] = bin_median
            # Too few samples to trust as an exact-key median — leave the
            # key out of medians so the fallback chain (opposite day type →
            # adjacent temp buckets → hourly) handles it instead. The samples
            # still feed the hourly/global fallbacks below: on a young
            # install EVERY bin can be sparse, and gating them out entirely
            # would leave the fallbacks empty too, collapsing all predictions
            # to the 1.0 kWh hard default.

            # Collect for hourly + global fallbacks (sparse bins included)
            _, hour, _ = key
            if hour not in hourly_all:
                hourly_all[hour] = []
            hourly_all[hour].append(bin_median)
            all_bin_medians.append(bin_median)

        # Build hourly fallbacks
        for hour, vals in hourly_all.items():
            self.hourly_fallback[hour] = statistics.median(vals)

        # Global fallback
        self.global_median = statistics.median(all_bin_medians) if all_bin_medians else 1.0
        self.built_at = datetime.now()

    def predict(self, temperature: float, hour: int, is_weekend: bool) -> float:
        """Predict consumption for given conditions.

        Args:
            temperature: Outside temperature in °C
            hour: Hour of day (0-23)
            is_weekend: Whether it's a weekend day

        Returns:
            Expected consumption in kWh for a 1-hour period
        """
        bucket = self.temp_to_bucket(temperature)
        key = (bucket, hour, is_weekend)

        if key in self.medians:
            return self.medians[key]

        # Try same temp/hour but opposite day type
        alt_key = (bucket, hour, not is_weekend)
        if alt_key in self.medians:
            return self.medians[alt_key]

        # Try adjacent temperature buckets
        for delta in [1, -1, 2, -2]:
            adj_key = (bucket + delta, hour, is_weekend)
            if adj_key in self.medians:
                return self.medians[adj_key]

        # Hourly fallback
        if hour in self.hourly_fallback:
            return self.hourly_fallback[hour]

        return self.global_median


def _carry_forward_hourly(
    result: Any, hour_keys: Any, local_tz: Any = None
) -> Dict[str, int]:
    """Reconstruct effective hourly state from sparse state-change records.

    This is the canonical implementation; solar_forecast.py and
    ml_consumption_forecast.py import it from here for export/inverter-on
    reconstruction. State-change records span until the next change. For each
    hour key we want the effective state DURING that hour: min over
    (state-entering, any changes within the hour), so a partially-disabled hour
    still counts as disabled. Hours predating the first record stay missing —
    caller defaults to "enabled".
    """
    changes: List[Tuple[datetime, int]] = []
    if result:
        for table in result:
            for record in table.records:
                val = record.get_value()
                if isinstance(val, (int, float)):
                    ct = record.get_time()
                    if local_tz is not None and getattr(
                        ct, "tzinfo", None
                    ) is not None:
                        ct = ct.astimezone(local_tz)
                    changes.append((ct.replace(tzinfo=None), int(val)))
    # The forward-only pointer below assumes globally ascending time, but a Flux
    # query with varying tags returns one table PER tag set (sort() only orders
    # within a table), so the concatenated list is per-table runs interleaved in
    # time. Sort once here so correctness never depends on InfluxDB returning a
    # single, pre-sorted series.
    changes.sort(key=lambda c: c[0])
    out: Dict[str, int] = {}
    if not changes or not hour_keys:
        return out
    ci = 0
    last_state: Optional[int] = None
    for key in sorted(hour_keys):
        hour_start = datetime.strptime(key, "%Y-%m-%d-%H")
        hour_end = hour_start + timedelta(hours=1)
        while (
            ci < len(changes)
            and changes[ci][0].replace(tzinfo=None) < hour_start
        ):
            last_state = changes[ci][1]
            ci += 1
        states_in_hour: List[int] = []
        if last_state is not None:
            states_in_hour.append(last_state)
        while (
            ci < len(changes)
            and changes[ci][0].replace(tzinfo=None) < hour_end
        ):
            last_state = changes[ci][1]
            states_in_hour.append(last_state)
            ci += 1
        if states_in_hour:
            out[key] = min(states_in_hour)
    return out


def _fallback_temperature(forecast_temps: Dict[int, float], hour: int) -> float:
    """Temperature for an hour missing from the forecast.

    Uses the nearest available forecast hour's temperature — a constant
    10°C default would silently halve heating estimates in January. 10°C
    is used only when no temperatures are available at all.
    """
    if not forecast_temps:
        return 10.0
    # Circular hour distance: for a missing hour 0 with temps at 12-23,
    # hour 23 is 1h away (not 23h) and far more representative than noon.
    nearest = min(
        forecast_temps, key=lambda h: min(abs(h - hour), 24 - abs(h - hour))
    )
    return forecast_temps[nearest]


# Bumped when the build_model logic changes in a way that invalidates
# previously cached models (e.g., new training-data filters). On version
# mismatch the controller forces a fresh rebuild, even if the cached
# model is younger than rebuild_interval_days.
# v5: hourly aggregates are start-labeled (timeSrc: "_start") — the default
#     stop-labeling shifted every hour key +1h — and bins now need
#     MIN_BIN_SAMPLES samples to enter medians.
# v6: sparse (below-MIN_BIN_SAMPLES) bins feed the hourly/global fallbacks
#     again (v5 dropped them entirely, leaving young installs with empty
#     fallbacks), and a build with zero trusted bins is declined.
MODEL_SCHEMA_VERSION = 6

# Upper bound for a "real household" hourly consumption sample (kWh).
# INVPowerToLocalLoad reports grid passthrough when the inverter is in
# bypass (manual off, fault, etc.) — those readings can be 15-25 kW for
# hours, which is not real local consumption. Anything above this cap
# is filtered from training. Set generously enough to allow legitimate
# heating + EV stacking; tighten if your loads stay smaller.
MAX_REALISTIC_HOURLY_KWH = 10.0


class ConsumptionForecast:
    """Consumption forecaster using historical data and temperature correlation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._model: Optional[ConsumptionModel] = None
        self._model_version: int = 0  # 0 = no model yet
        self._last_model_build: Optional[datetime] = None
        # Set on EVERY build attempt (success or failure). The controller
        # polls needs_rebuild() every minute, so a failing build (cold start,
        # InfluxDB outage) must back off instead of re-running heavy 365-day
        # queries on every poll.
        self._last_build_attempt: Optional[datetime] = None
        self.failed_build_backoff_seconds: int = 3600
        # Rebuild model weekly
        self.rebuild_interval_days: int = 7

    @property
    def model(self) -> Optional[ConsumptionModel]:
        return self._model

    def needs_rebuild(self) -> bool:
        """Check if model needs rebuilding."""
        if self._failed_attempt_recently():
            return False
        if self._model is None or self._last_model_build is None:
            return True
        if self._model_version != MODEL_SCHEMA_VERSION:
            return True
        age = (datetime.now() - self._last_model_build).total_seconds()
        return age > self.rebuild_interval_days * 86400

    def _failed_attempt_recently(self) -> bool:
        """True if the last build attempt failed within the backoff window."""
        if self._last_build_attempt is None:
            return False
        if (
            self._last_model_build is not None
            and self._last_model_build >= self._last_build_attempt
        ):
            return False  # last attempt succeeded — normal cadence applies
        age = (datetime.now() - self._last_build_attempt).total_seconds()
        return age < self.failed_build_backoff_seconds

    async def build_model(
        self, influxdb_client: Any, settings: Any, local_tz: Any = None
    ) -> bool:
        """Build consumption model from historical InfluxDB data.

        Queries up to 365 days of hourly consumption + temperature data,
        bins by (temperature, hour, weekday/weekend), removes outliers,
        and stores medians as the prediction model.

        Args:
            influxdb_client: Async InfluxDB client
            settings: Application settings

        Returns:
            True if model was built successfully
        """
        # Record the attempt up front so needs_rebuild() can back off if this
        # attempt fails (query error, too little data, ...).
        self._last_build_attempt = datetime.now()
        try:
            self.logger.info("Building consumption model from historical data...")

            def _to_local(t: datetime) -> datetime:
                # Bucket InfluxDB's UTC timestamps into the optimizer's local
                # wall-clock so the (temp, hour, weekend) bins are keyed by the
                # same local hours predict_hourly is queried with. Without it
                # the consumption peak lands 1-2h off and weekend/weekday can
                # misclassify near midnight. Mirrors the ML model's _to_local.
                if local_tz is not None and getattr(t, "tzinfo", None) is not None:
                    t = t.astimezone(local_tz)
                return t

            # Query hourly consumption (local load power, averaged per hour)
            consumption_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "solar")
  |> filter(fn: (r) => r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> yield(name: "consumption")
'''
            # Query hourly outside temperature
            temperature_query = f'''
from(bucket: "{settings.influxdb.bucket_loxone}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "temperature")
  |> filter(fn: (r) => r._field == "temperature_outside")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> yield(name: "temperature")
'''
            # Inverter on/off state changes — used to exclude hours where the
            # inverter was off (INVPowerToLocalLoad reads grid passthrough
            # then, not real household load, so those samples poison the bins).
            inverter_on_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "inverter_state" and r._field == "inverter_on")
  |> sort(columns: ["_time"])
'''

            consumption_result = await influxdb_client.query(consumption_query)
            temperature_result = await influxdb_client.query(temperature_query)
            inverter_on_result = await influxdb_client.query(inverter_on_query)

            if not consumption_result or not temperature_result:
                self.logger.warning("No consumption or temperature data found")
                return False

            # Parse consumption data: timestamp -> watts
            consumption_by_hour: Dict[str, float] = {}  # "YYYY-MM-DD-HH" -> watts
            for table in consumption_result:
                for record in table.records:
                    v = record.get_value()
                    if not isinstance(v, (int, float)):
                        continue  # skip None/empty aggregates (would crash watts<=0)
                    t = _to_local(record.get_time())
                    key = t.strftime("%Y-%m-%d-%H")
                    consumption_by_hour[key] = float(v)

            # Parse temperature data: timestamp -> °C
            temp_by_hour: Dict[str, float] = {}
            for table in temperature_result:
                for record in table.records:
                    v = record.get_value()
                    if not isinstance(v, (int, float)):
                        continue
                    t = _to_local(record.get_time())
                    key = t.strftime("%Y-%m-%d-%H")
                    temp_by_hour[key] = float(v)

            # Resolve inverter on/off state per hour from sparse change records.
            inverter_on_by_hour = _carry_forward_hourly(
                inverter_on_result, consumption_by_hour.keys(), local_tz
            )

            # Build model bins
            model = ConsumptionModel()
            matched = 0
            inverter_off_excluded = 0
            outlier_excluded = 0

            for hour_key, watts in consumption_by_hour.items():
                if hour_key not in temp_by_hour:
                    continue
                if watts <= 0:
                    continue

                # Skip hours where the controller knew the inverter was off.
                if inverter_on_by_hour.get(hour_key, 1) == 0:
                    inverter_off_excluded += 1
                    continue

                temp = temp_by_hour[hour_key]
                # Convert watts average to kWh for 1 hour
                kwh = watts / 1000.0

                # Skip implausibly-large samples — almost always inverter
                # bypass (user toggled off via OIG UI without telling the
                # controller, or device fault). Grid passthrough then reports
                # 15-25kW for hours, contaminating bins for that
                # (temp, hour, weekday) slot.
                if kwh > MAX_REALISTIC_HOURLY_KWH:
                    outlier_excluded += 1
                    continue

                # Parse hour and day of week
                parts = hour_key.split("-")
                year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                try:
                    dt = date(year, month, day)
                except ValueError:
                    continue
                is_weekend = dt.weekday() >= 5

                bucket = ConsumptionModel.temp_to_bucket(temp)
                bin_key = (bucket, hour, is_weekend)
                if bin_key not in model.bins:
                    model.bins[bin_key] = []
                model.bins[bin_key].append(kwh)
                matched += 1

            if matched < 100:
                self.logger.warning(f"Only {matched} matched data points, need at least 100")
                return False

            model.data_points = matched
            earliest = min(consumption_by_hour.keys())
            latest = max(consumption_by_hour.keys())
            model.date_range = f"{earliest[:10]} to {latest[:10]}"

            # Build medians with outlier removal
            model.build()

            if not model.medians:
                # Possible on a young install: ≥100 matched points but every
                # (temp, hour, day-type) bin below MIN_BIN_SAMPLES. Installing
                # a model with zero trusted bins would report success while
                # every prediction rides on thin fallbacks — decline instead
                # and retry once more data has accumulated.
                self.logger.warning(
                    f"Consumption model declined: {matched} matched points but "
                    f"no bin reached MIN_BIN_SAMPLES={ConsumptionModel.MIN_BIN_SAMPLES} "
                    f"samples — too little data per (temp, hour, day-type) slot"
                )
                return False

            self._model = model
            self._model_version = MODEL_SCHEMA_VERSION
            self._last_model_build = datetime.now()

            self.logger.info(
                f"Consumption model built: {matched} data points, "
                f"{len(model.medians)} bins, "
                f"range {model.date_range} "
                f"(excluded {inverter_off_excluded} inverter-off hours, "
                f"{outlier_excluded} outliers >{MAX_REALISTIC_HOURLY_KWH:.0f}kWh)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to build consumption model: {e}", exc_info=True)
            return False

    def predict_hourly(
        self,
        forecast_temps: Dict[int, float],
        target_date: date,
    ) -> Dict[int, float]:
        """Predict hourly consumption for a given date and temperature forecast.

        Args:
            forecast_temps: hour (0-23) -> forecast temperature (°C)
            target_date: Date to predict for

        Returns:
            hour (0-23) -> predicted consumption in kWh
        """
        if not self._model:
            return {}

        is_weekend = target_date.weekday() >= 5
        result: Dict[int, float] = {}

        for hour in range(24):
            temp = forecast_temps.get(hour)
            if temp is None:
                temp = _fallback_temperature(forecast_temps, hour)
            result[hour] = max(0.0, self._model.predict(temp, hour, is_weekend))

        return result

    def predict_daily(
        self,
        avg_temperature: float,
        target_date: date,
    ) -> float:
        """Predict total daily consumption for a given date and avg temperature.

        Args:
            avg_temperature: Average temperature for the day (°C)
            target_date: Date to predict for

        Returns:
            Predicted total consumption in kWh
        """
        if not self._model:
            return 24.0  # Default: 1 kWh/hour

        is_weekend = target_date.weekday() >= 5
        total = 0.0
        for hour in range(24):
            total += max(0.0, self._model.predict(avg_temperature, hour, is_weekend))
        return total

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary for logging/API."""
        if not self._model:
            return {"status": "not built"}

        return {
            "status": "ready",
            "data_points": self._model.data_points,
            "bins": len(self._model.medians),
            "date_range": self._model.date_range,
            "built_at": self._model.built_at.isoformat() if self._model.built_at else None,
            "global_median_kwh": round(self._model.global_median, 2),
        }
