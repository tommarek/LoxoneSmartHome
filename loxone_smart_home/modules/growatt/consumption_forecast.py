"""Temperature-aware consumption forecast using historical InfluxDB data.

Builds a lookup model: (temperature_bucket, hour, weekday) → expected kWh,
using all available historical data. Designed for an all-electric house
where heating load dominates in winter.
"""

import logging
import math
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
        """Convert temperature to bucket index."""
        clamped = max(-20, min(40, temp))
        return int((clamped + 20) / 2)

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


class ConsumptionForecast:
    """Consumption forecaster using historical data and temperature correlation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._model: Optional[ConsumptionModel] = None
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
        age = (datetime.now() - self._last_model_build).total_seconds()
        return age > self.rebuild_interval_days * 86400

    async def build_model(self, influxdb_client: Any, settings: Any) -> bool:
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

            consumption_result = await influxdb_client.query(consumption_query)
            temperature_result = await influxdb_client.query(temperature_query)

            if not consumption_result or not temperature_result:
                self.logger.warning("No consumption or temperature data found")
                return False

            # Parse consumption data: timestamp -> watts
            consumption_by_hour: Dict[str, float] = {}  # "YYYY-MM-DD-HH" -> watts
            for table in consumption_result:
                for record in table.records:
                    t = record.get_time()
                    key = t.strftime("%Y-%m-%d-%H")
                    consumption_by_hour[key] = record.get_value()

            # Parse temperature data: timestamp -> °C
            temp_by_hour: Dict[str, float] = {}
            for table in temperature_result:
                for record in table.records:
                    t = record.get_time()
                    key = t.strftime("%Y-%m-%d-%H")
                    temp_by_hour[key] = record.get_value()

            # Build model bins
            model = ConsumptionModel()
            matched = 0

            for hour_key, watts in consumption_by_hour.items():
                if hour_key not in temp_by_hour:
                    continue
                if watts <= 0:
                    continue

                temp = temp_by_hour[hour_key]
                # Convert watts average to kWh for 1 hour
                kwh = watts / 1000.0

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
            self._last_model_build = datetime.now()

            self.logger.info(
                f"Consumption model built: {matched} data points, "
                f"{len(model.medians)} bins, "
                f"range {model.date_range}"
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
