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
        hourly_all: Dict[int, List[float]] = {}

        for key, values in self.bins.items():
            if len(values) < 3:
                # Not enough data points, skip outlier removal
                self.medians[key] = statistics.median(values)
            else:
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
                self.medians[key] = statistics.median(filtered) if filtered else statistics.median(values)

            # Collect for hourly fallback
            _, hour, _ = key
            if hour not in hourly_all:
                hourly_all[hour] = []
            hourly_all[hour].append(self.medians[key])

        # Build hourly fallbacks
        for hour, vals in hourly_all.items():
            self.hourly_fallback[hour] = statistics.median(vals)

        # Global fallback
        all_medians = list(self.medians.values())
        self.global_median = statistics.median(all_medians) if all_medians else 1.0
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

    Mirror of solar_forecast._carry_forward_hourly. State-change records span
    until the next change. For each hour key we want the effective state
    DURING that hour: min over (state-entering, any changes within the hour),
    so a partially-disabled hour still counts as disabled. Hours predating
    the first record stay missing — caller defaults to "enabled".
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


# Bumped when the build_model logic changes in a way that invalidates
# previously cached models (e.g., new training-data filters). On version
# mismatch the controller forces a fresh rebuild, even if the cached
# model is younger than rebuild_interval_days.
MODEL_SCHEMA_VERSION = 4

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
        # Rebuild model weekly
        self.rebuild_interval_days: int = 7

    @property
    def model(self) -> Optional[ConsumptionModel]:
        return self._model

    def needs_rebuild(self) -> bool:
        """Check if model needs rebuilding."""
        if self._model is None or self._last_model_build is None:
            return True
        if self._model_version != MODEL_SCHEMA_VERSION:
            return True
        age = (datetime.now() - self._last_model_build).total_seconds()
        return age > self.rebuild_interval_days * 86400

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
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> yield(name: "consumption")
'''
            # Query hourly outside temperature
            temperature_query = f'''
from(bucket: "{settings.influxdb.bucket_loxone}")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == "temperature")
  |> filter(fn: (r) => r._field == "temperature_outside")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
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
                    t = _to_local(record.get_time())
                    key = t.strftime("%Y-%m-%d-%H")
                    consumption_by_hour[key] = record.get_value()

            # Parse temperature data: timestamp -> °C
            temp_by_hour: Dict[str, float] = {}
            for table in temperature_result:
                for record in table.records:
                    t = _to_local(record.get_time())
                    key = t.strftime("%Y-%m-%d-%H")
                    temp_by_hour[key] = record.get_value()

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
            temp = forecast_temps.get(hour, 10.0)  # Default 10°C if unknown
            result[hour] = self._model.predict(temp, hour, is_weekend)

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
            total += self._model.predict(avg_temperature, hour, is_weekend)
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
