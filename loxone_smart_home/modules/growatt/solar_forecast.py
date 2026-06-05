"""Solar production forecast using learned model, forecast.solar API, and weather data.

Primary predictor: SolarProductionModel trained on 365 days of actual production
paired with weather observations. Falls back to forecast.solar API and OpenMeteo.
Supports multiple solar arrays with different orientations.
"""

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


def _effective_cloud_cover(
    cloud_low: float, cloud_mid: float, cloud_high: float,
) -> float:
    """Compute solar-effective cloud cover from low/mid/high layers.

    Low clouds are thick and block most radiation, mid clouds are thinner,
    high clouds (cirrus) barely affect solar production. Uses a transmittance
    model: each layer transmits a fraction of light based on its opacity.

    Returns:
        Effective cloud cover percentage (0-100).
    """
    transmittance = (
        (1.0 - 0.9 * cloud_low / 100)
        * (1.0 - 0.5 * cloud_mid / 100)
        * (1.0 - 0.15 * cloud_high / 100)
    )
    return max(0.0, min(100.0, (1.0 - transmittance) * 100))


def _sun_position(latitude: float, year: int, month: int, day: int, hour: int) -> Tuple[float, float]:
    """Compute sun azimuth and altitude for a given hour (mid-hour).

    Returns:
        (azimuth_degrees, altitude_degrees) where altitude <= 0 means below horizon.
    """
    try:
        day_of_year = datetime(year, month, day).timetuple().tm_yday
        decl = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
        hour_angle = (hour + 0.5 - 12) * 15
        lat_rad = math.radians(latitude)
        decl_rad = math.radians(decl)
        ha_rad = math.radians(hour_angle)
        sin_alt = (math.sin(lat_rad) * math.sin(decl_rad) +
                   math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
        altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))
        cos_az = ((math.sin(decl_rad) - math.sin(lat_rad) * sin_alt) /
                  max(0.001, math.cos(lat_rad) * math.cos(math.asin(max(-1, min(1, sin_alt))))))
        azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))
        if hour_angle > 0:
            azimuth = 360 - azimuth
        return azimuth, max(0, altitude)
    except Exception:
        return 180.0, 30.0


@dataclass
class SolarArray:
    """Configuration for a single solar panel array."""
    name: str
    declination: int  # Panel tilt in degrees (0=horizontal, 90=vertical)
    azimuth: int  # Panel azimuth in degrees (0=south, 90=west, -90=east, 180=north)
    kwp: float  # Installed peak capacity in kWp


@dataclass
class HourlyForecast:
    """Solar production forecast for a single hour."""
    hour: int  # 0-23
    watt_hours: float  # Expected production in Wh
    source: str  # "api", "weather", or "consensus"


@dataclass
class DailyForecast:
    """Solar production forecast for a full day."""
    date: date
    total_kwh: float
    hourly: Dict[int, float] = field(default_factory=dict)  # hour -> kWh
    source: str = "unknown"


@dataclass
class SolarProductionModel:
    """Learned solar production model trained on historical data.

    Features:
    - GHI radiation: 25 W/m² steps (41 buckets)
    - Cloud cover: 10% steps (11 buckets) — direct weather parameter
    - Sun altitude: 5° steps (19 buckets) — angle of incidence

    Cloud cover is ALWAYS included in predictions — never dropped in fallbacks.
    Fallback: 3D (rad, cloud, alt) → 2D (rad, cloud) → interpolate → global.
    Uses p50 (median) for prediction, physics cap as safety net.
    """
    # 3D bins: (rad, cloud, alt) -> [kwh]
    bins_3d: Dict[Tuple[int, int, int], List[float]] = field(default_factory=dict)
    median_3d: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    # 2D bins: (rad, cloud) -> [kwh] (fallback — always includes cloud!)
    bins_2d: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    median_2d: Dict[Tuple[int, int], float] = field(default_factory=dict)
    global_median: float = 0.0
    # Observability: which bin level is actually used
    hit_level_counts: Dict[str, int] = field(
        default_factory=lambda: {"3d": 0, "2d": 0, "interpolate": 0, "global": 0}
    )
    # Metadata
    total_kwp: float = 13.5
    data_points: int = 0
    curtailed_filtered: int = 0
    date_range: str = ""
    built_at: Optional[datetime] = None
    # Per-bin estimator quantile. 0.5 = median (default; unchanged behaviour).
    # Higher (e.g. 0.7) recovers PV *potential* when curtailment thins out a
    # bin's high-end clean samples — the surviving samples skew low because the
    # sunniest hours are the ones most likely to have been curtailed.
    quantile: float = 0.5

    @staticmethod
    def radiation_to_bucket(ghi: float) -> int:
        """GHI W/m² → bucket (25 W/m² steps, 0-40)."""
        return min(40, max(0, int(ghi / 25)))

    @staticmethod
    def cloud_to_bucket(cloud_pct: float) -> int:
        """Cloud cover % → bucket (10% steps, 0-10). 0=clear, 10=overcast."""
        return min(10, max(0, int(cloud_pct / 10)))

    @staticmethod
    def azimuth_to_bucket(azimuth_deg: float) -> int:
        """Sun azimuth → bucket (15° steps, 0-23)."""
        return int((azimuth_deg % 360) / 15)

    @staticmethod
    def altitude_to_bucket(altitude_deg: float) -> int:
        """Sun altitude → bucket (5° steps, 0-18)."""
        return min(18, max(0, int(altitude_deg / 5)))

    @staticmethod
    def temp_to_bucket(temp_c: float) -> int:
        """Temperature → bucket (5°C steps)."""
        return min(12, max(0, int((temp_c + 20) / 5)))

    def _compute_quantile(self, values: List[float]) -> float:
        """Per-bin estimate at `self.quantile`, after IQR outlier removal.

        `quantile=0.5` reproduces the median exactly (linear interpolation at
        0.5 == median for both odd and even n), so the default is behaviour-
        preserving; a higher quantile lifts the estimate toward potential.
        """
        if not values:
            return 0.0
        if len(values) < 3:
            return statistics.median(values)
        sorted_vals = sorted(values)
        q1 = sorted_vals[len(sorted_vals) // 4]
        q3 = sorted_vals[3 * len(sorted_vals) // 4]
        iqr = q3 - q1
        filtered = sorted(
            v for v in values if (q1 - 1.5 * iqr) <= v <= (q3 + 1.5 * iqr)
        )
        if not filtered:
            filtered = sorted_vals
        q = min(1.0, max(0.0, self.quantile))
        if q <= 0:
            return filtered[0]
        if q >= 1:
            return filtered[-1]
        # Linear-interpolated quantile.
        idx = q * (len(filtered) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(filtered) - 1)
        frac = idx - lo
        return filtered[lo] * (1 - frac) + filtered[hi] * frac

    def add_sample(self, rad_b: int, cloud_b: int, alt_b: int,
                   kwh: float) -> None:
        """Add a training sample to all bin levels."""
        k3 = (rad_b, cloud_b, alt_b)
        k2 = (rad_b, cloud_b)
        for bins, key in [(self.bins_3d, k3), (self.bins_2d, k2)]:
            if key not in bins:
                bins[key] = []
            bins[key].append(kwh)

    def build(self) -> None:
        """Compute median for all bin levels."""
        for bins, median_dict in [
            (self.bins_3d, self.median_3d), (self.bins_2d, self.median_2d),
        ]:
            median_dict.clear()
            for key, values in bins.items():
                if values:
                    med = self._compute_quantile(values)
                    median_dict[key] = med

        all_medians = [v for v in self.median_2d.values() if v > 0]
        self.global_median = statistics.median(all_medians) if all_medians else 0.0
        self.built_at = datetime.now()

    def _interpolate_2d(self, rad_b: int, cloud_b: int) -> Optional[float]:
        """Interpolate between nearest populated 2D bins."""
        # Try radiation axis first (same cloud cover)
        lower_rad, lower_val = None, None
        upper_rad, upper_val = None, None
        for dr in range(1, 10):
            if lower_val is None and (rad_b - dr, cloud_b) in self.median_2d:
                lower_rad, lower_val = rad_b - dr, self.median_2d[(rad_b - dr, cloud_b)]
            if upper_val is None and (rad_b + dr, cloud_b) in self.median_2d:
                upper_rad, upper_val = rad_b + dr, self.median_2d[(rad_b + dr, cloud_b)]
            if lower_val is not None and upper_val is not None:
                break
        if lower_val is not None and upper_val is not None:
            span = upper_rad - lower_rad
            weight = (rad_b - lower_rad) / span if span > 0 else 0.5
            return lower_val + (upper_val - lower_val) * weight
        if lower_val is not None or upper_val is not None:
            # Use `is not None` (not truthiness) so a legitimate 0.0 median
            # from the populated side isn't skipped in favour of the other.
            return lower_val if lower_val is not None else upper_val

        # Try cloud axis (same radiation bucket)
        lower_cloud, lower_cval = None, None
        upper_cloud, upper_cval = None, None
        for dc in range(1, 6):
            if lower_cval is None and (rad_b, cloud_b - dc) in self.median_2d:
                lower_cloud, lower_cval = cloud_b - dc, self.median_2d[(rad_b, cloud_b - dc)]
            if upper_cval is None and (rad_b, cloud_b + dc) in self.median_2d:
                upper_cloud, upper_cval = cloud_b + dc, self.median_2d[(rad_b, cloud_b + dc)]
            if lower_cval is not None and upper_cval is not None:
                break
        if lower_cval is not None and upper_cval is not None:
            span = upper_cloud - lower_cloud
            weight = (cloud_b - lower_cloud) / span if span > 0 else 0.5
            return lower_cval + (upper_cval - lower_cval) * weight
        if lower_cval is not None or upper_cval is not None:
            return lower_cval if lower_cval is not None else upper_cval
        return None

    def predict(
        self, ghi: float,
        sun_azimuth: float = 180, sun_altitude: float = 45,
        cloud_cover: float = 50, temperature: float = 15,
    ) -> float:
        """Predict kWh production. Cloud cover is ALWAYS used.

        Fallback: 3D (rad, cloud, alt) → 2D (rad, cloud) → interpolate → global.
        """
        if ghi <= 0 or sun_altitude <= 0:
            return 0.0

        rad_b = self.radiation_to_bucket(ghi)
        cloud_b = self.cloud_to_bucket(cloud_cover)
        alt_b = self.altitude_to_bucket(sun_altitude)

        # Physics cap: can't exceed nameplate capacity per hour
        physics_max = self.total_kwp

        def _cap(v: float) -> float:
            return min(v, physics_max)

        # 3D (rad, cloud, alt)
        k = (rad_b, cloud_b, alt_b)
        if k in self.median_3d:
            self.hit_level_counts["3d"] += 1
            return _cap(self.median_3d[k])

        # 2D exact (rad, cloud)
        k = (rad_b, cloud_b)
        if k in self.median_2d:
            self.hit_level_counts["2d"] += 1
            return _cap(self.median_2d[k])

        # 2D interpolation — find nearest on radiation/cloud axes
        interp = self._interpolate_2d(rad_b, cloud_b)
        if interp is not None:
            self.hit_level_counts["interpolate"] += 1
            return _cap(interp)

        self.hit_level_counts["global"] += 1
        return _cap(self.global_median)


class SolarForecast:
    """Solar production forecast combining learned model, API, and weather data."""

    FORECAST_SOLAR_URL = "https://api.forecast.solar/estimate"

    def __init__(
        self,
        arrays: List[SolarArray],
        latitude: float,
        longitude: float,
        confidence: float = 0.7,
        logger: Optional[logging.Logger] = None,
        model_quantile: float = 0.5,
    ):
        self.arrays = arrays
        self.latitude = latitude
        self.longitude = longitude
        self.confidence = confidence
        # Per-bin quantile for the learned production model (see
        # SolarProductionModel.quantile). 0.5 keeps the historical median.
        self.model_quantile = model_quantile
        self.logger = logger or logging.getLogger(__name__)

        # Cached forecasts
        self._api_forecast: Dict[str, DailyForecast] = {}  # date_str -> forecast
        self._weather_forecast: Dict[str, DailyForecast] = {}
        self._model_forecast: Dict[str, DailyForecast] = {}  # from learned model
        self._solcast_forecast: Dict[str, DailyForecast] = {}  # from Solcast API
        self._consensus: Dict[str, DailyForecast] = {}
        self._last_api_update: Optional[datetime] = None

        # Learned production model
        self._production_model: Optional[SolarProductionModel] = None

    @classmethod
    def from_config(
        cls, config: Any, logger: Optional[logging.Logger] = None,
        settings: Any = None,
    ) -> "SolarForecast":
        """Create SolarForecast from GrowattConfig + optional global Settings."""
        arrays_json = getattr(config, "solar_arrays", "[]")
        arrays_data = json.loads(arrays_json) if isinstance(arrays_json, str) else arrays_json
        arrays = [
            SolarArray(
                name=a["name"],
                declination=a["declination"],
                azimuth=a["azimuth"],
                kwp=a["kwp"],
            )
            for a in arrays_data
        ]
        # Prefer global settings lat/lon (from .env), fall back to config.
        # Use explicit None checks so a valid 0.0 coordinate isn't discarded.
        _s_lat = getattr(settings, "latitude", None)
        _s_lon = getattr(settings, "longitude", None)
        lat = _s_lat if _s_lat is not None else getattr(config, "latitude", 49.0)
        lon = _s_lon if _s_lon is not None else getattr(config, "longitude", 14.5)
        return cls(
            arrays=arrays,
            latitude=lat,
            longitude=lon,
            confidence=getattr(config, "solar_forecast_confidence", 0.7),
            logger=logger,
            model_quantile=getattr(config, "solar_model_quantile", 0.5),
        )

    # --- forecast.solar API ---

    async def fetch_api_forecast(self) -> Dict[str, DailyForecast]:
        """Fetch forecast from forecast.solar for each array and combine.

        Returns:
            Dict mapping date string (YYYY-MM-DD) to DailyForecast
        """
        combined_hourly: Dict[str, Dict[int, float]] = {}  # date_str -> {hour -> wh}

        async with aiohttp.ClientSession() as session:
            for array in self.arrays:
                try:
                    url = (
                        f"{self.FORECAST_SOLAR_URL}"
                        f"/{self.latitude}/{self.longitude}"
                        f"/{array.declination}/{array.azimuth}/{array.kwp}"
                    )
                    self.logger.debug(f"Fetching forecast.solar for {array.name}: {url}")

                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 429:
                            self.logger.warning(
                                "forecast.solar rate limit hit — keeping arrays "
                                "fetched so far this loop"
                            )
                            break  # don't discard already-fetched arrays
                        if resp.status != 200:
                            self.logger.warning(
                                f"forecast.solar returned {resp.status} for {array.name}"
                            )
                            continue

                        data = await resp.json()
                        # Use watt_hours_period (per-hour), NOT watt_hours (cumulative)
                        watt_hours = data.get("result", {}).get("watt_hours_period", {})

                        for timestamp_str, wh in watt_hours.items():
                            # forecast.solar returns naive "YYYY-MM-DD HH:MM:SS"
                            # timestamps already in the SITE's local timezone
                            # (derived from lat/lon) — which for this deployment
                            # is the Europe/Prague zone used for all scheduling.
                            # So the naive date/hour below IS the local grid, no
                            # conversion needed (cf. OpenMeteo timezone=auto and
                            # Solcast's explicit tz conversion). If forecast.solar
                            # ever changed its default to UTC this would shift by
                            # the local offset; that contract is assumed here.
                            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            date_str = dt.strftime("%Y-%m-%d")
                            hour = dt.hour

                            if date_str not in combined_hourly:
                                combined_hourly[date_str] = {}
                            combined_hourly[date_str][hour] = (
                                combined_hourly[date_str].get(hour, 0) + wh
                            )

                    self.logger.info(
                        f"Fetched forecast.solar for {array.name} "
                        f"({array.kwp} kWp, az={array.azimuth}°, tilt={array.declination}°)"
                    )

                except asyncio.TimeoutError:
                    self.logger.warning(f"forecast.solar timeout for {array.name}")
                except Exception as e:
                    self.logger.warning(f"forecast.solar error for {array.name}: {e}")

        # Build DailyForecast objects
        result: Dict[str, DailyForecast] = {}
        for date_str, hourly in combined_hourly.items():
            hourly_kwh = {h: wh / 1000.0 for h, wh in hourly.items()}
            total = sum(hourly_kwh.values())
            result[date_str] = DailyForecast(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                total_kwh=total,
                hourly=hourly_kwh,
                source="api",
            )

        if result:
            self._api_forecast = result
            self._last_api_update = datetime.now()
            total_all = sum(f.total_kwh for f in result.values())
            self.logger.info(
                f"Solar API forecast updated: {len(result)} days, "
                f"total {total_all:.1f} kWh across {len(self.arrays)} arrays"
            )

        return result

    # --- Weather-based calculation ---

    async def fetch_openmeteo_radiation(self) -> Dict[str, DailyForecast]:
        """Fetch solar radiation forecast directly from OpenMeteo API.

        This is a fallback when forecast.solar is rate-limited. OpenMeteo
        provides hourly shortwave_radiation forecasts for free with no rate limit.

        Returns:
            Dict mapping date string to DailyForecast
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={self.latitude}&longitude={self.longitude}"
            f"&hourly=shortwave_radiation,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,temperature_2m"
            f"&forecast_days=2"
            f"&timezone=auto"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"OpenMeteo API returned {resp.status}")
                        return {}

                    data = await resp.json()
                    hourly = data.get("hourly", {})
                    times = hourly.get("time", [])
                    radiation = hourly.get("shortwave_radiation", [])
                    cloud_low = hourly.get("cloudcover_low", [])
                    cloud_mid = hourly.get("cloudcover_mid", [])
                    cloud_high = hourly.get("cloudcover_high", [])
                    cloud_total = hourly.get("cloudcover", [])
                    temperatures = hourly.get("temperature_2m", [])

                    if not times or not radiation:
                        return {}

                    weather_data = {"hourly": []}
                    for i, t in enumerate(times):
                        # Compute effective cloud cover from layers
                        if cloud_low and i < len(cloud_low):
                            eff_cloud = _effective_cloud_cover(
                                cloud_low[i] or 0,
                                cloud_mid[i] if cloud_mid and i < len(cloud_mid) else 0,
                                cloud_high[i] if cloud_high and i < len(cloud_high) else 0,
                            )
                        else:
                            eff_cloud = cloud_total[i] if cloud_total and i < len(cloud_total) else 50

                        weather_data["hourly"].append({
                            "time": t,
                            "shortwave_radiation": radiation[i] if i < len(radiation) else 0,
                            "cloudcover": eff_cloud,
                            "temperature_2m": temperatures[i] if i < len(temperatures) else 15,
                        })

                    result = self.calculate_from_weather(weather_data)
                    if result:
                        self.logger.info(
                            f"OpenMeteo radiation forecast: "
                            f"{sum(f.total_kwh for f in result.values()):.1f} kWh "
                            f"across {len(result)} days"
                        )
                    return result

        except Exception as e:
            self.logger.warning(f"OpenMeteo radiation fetch failed: {e}")
            return {}

    # --- Learned production model ---

    async def build_production_model(
        self, influxdb_client: Any, settings: Any, local_tz: Any = None
    ) -> bool:
        """Train solar production model from ALL historical data.

        Queries data in monthly chunks to avoid InfluxDB timeout.
        Uses cloud cover (%) instead of derived clear sky ratio.
        Two-pass: build rough model, filter curtailed hours, rebuild.

        Args:
            influxdb_client: Async InfluxDB client
            settings: App settings (for bucket names)
            local_tz: Local timezone used to bucket InfluxDB's UTC timestamps
                into LOCAL hour keys. Critical: ``_sun_position`` treats its
                ``hour`` arg as local solar time (noon at 12), and prediction
                (``predict_from_model``) keys on local hours, so training MUST
                bucket on the same local clock — otherwise the sun-altitude
                bucket learned at a UTC hour is applied at a different physical
                instant, shifting the whole production curve by the UTC offset.
                Mirrors the consumption/ML forecasters' ``_to_local`` fix.

        Returns:
            True if model was built successfully
        """
        try:
            self.logger.info("Building solar production model from historical data...")

            def _to_local(t: datetime) -> datetime:
                # Convert tz-aware UTC timestamps to local before bucketing so
                # hour keys (and the sun position derived from them) match the
                # local-time basis used at prediction time. Naive timestamps are
                # assumed already-local and passed through.
                if local_tz is not None and getattr(t, "tzinfo", None) is not None:
                    return t.astimezone(local_tz)
                return t

            # Query hourly solar production
            solar_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "InputPower")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            # Query hourly SOC and load (for curtailment detection)
            soc_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "SOC")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            load_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            # Raw export_enabled and inverter_on state-change records from
            # controller-persisted state. We carry the last value forward
            # over hours that had no change (state intervals), so an hour
            # where the field was 0 the WHOLE time still counts as 0 even
            # though no point falls inside that hour's window.
            export_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "inverter_state" and r._field == "export_enabled")
  |> sort(columns: ["_time"])
'''
            inverter_on_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "inverter_state" and r._field == "inverter_on")
  |> sort(columns: ["_time"])
'''
            # Query weather fields separately (much faster than combined OR query)
            weather_bucket = settings.influxdb.bucket_weather

            solar_result = await influxdb_client.query(solar_query)
            soc_result = await influxdb_client.query(soc_query)
            load_result = await influxdb_client.query(load_query)
            export_result = await influxdb_client.query(export_query)
            inverter_on_result = await influxdb_client.query(inverter_on_query)

            if not solar_result:
                self.logger.warning("No solar production data found")
                return False

            # Parse solar data (skip None/empty aggregates — they'd poison the
            # numeric model build with TypeErrors downstream).
            solar_by_hour: Dict[str, float] = {}
            for table in solar_result:
                for record in table.records:
                    v = record.get_value()
                    if not isinstance(v, (int, float)):
                        continue
                    key = _to_local(record.get_time()).strftime("%Y-%m-%d-%H")
                    solar_by_hour[key] = float(v)

            soc_by_hour: Dict[str, float] = {}
            if soc_result:
                for table in soc_result:
                    for record in table.records:
                        v = record.get_value()
                        if not isinstance(v, (int, float)):
                            continue
                        key = _to_local(record.get_time()).strftime("%Y-%m-%d-%H")
                        soc_by_hour[key] = float(v)

            load_by_hour: Dict[str, float] = {}
            if load_result:
                for table in load_result:
                    for record in table.records:
                        v = record.get_value()
                        if not isinstance(v, (int, float)):
                            continue
                        key = _to_local(record.get_time()).strftime("%Y-%m-%d-%H")
                        load_by_hour[key] = float(v)

            # Carry-forward state intervals: state-change records span until the
            # next change; for each solar-data hour we want the effective state
            # DURING that hour. Shared with the consumption/ML forecasters — see
            # consumption_forecast._carry_forward_hourly for the min-over-changes
            # semantics (a partial-disabled hour counts as disabled; hours
            # predating the first record stay missing so callers default to "on").
            from .consumption_forecast import _carry_forward_hourly

            export_by_hour = _carry_forward_hourly(
                export_result, solar_by_hour.keys(), local_tz
            )
            inverter_on_by_hour = _carry_forward_hourly(
                inverter_on_result, solar_by_hour.keys(), local_tz
            )

            # Query each weather field separately for performance
            ghi_by_hour: Dict[str, float] = {}
            cloud_total_by_hour: Dict[str, float] = {}
            cloud_low_by_hour: Dict[str, float] = {}
            cloud_mid_by_hour: Dict[str, float] = {}
            cloud_high_by_hour: Dict[str, float] = {}
            temp_by_hour: Dict[str, float] = {}

            field_targets = {
                "shortwave_radiation": ghi_by_hour,
                "cloudcover": cloud_total_by_hour,
                "cloudcover_low": cloud_low_by_hour,
                "cloudcover_mid": cloud_mid_by_hour,
                "cloudcover_high": cloud_high_by_hour,
                "temperature_2m": temp_by_hour,
            }

            for field_name, target_dict in field_targets.items():
                try:
                    q = f'''
from(bucket: "{weather_bucket}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "weather_forecast")
  |> filter(fn: (r) => r._field == "{field_name}")
  |> filter(fn: (r) => r.type == "hour")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
                    result = await influxdb_client.query(q)
                    if result:
                        # Average across any duplicate series landing in the
                        # same local hour (consistent with the `mean`
                        # aggregateWindow above) — taking max biased cloud/
                        # radiation toward the most extreme contributing record.
                        sums: Dict[str, float] = {}
                        counts: Dict[str, int] = {}
                        for table in result:
                            for record in table.records:
                                key = _to_local(record.get_time()).strftime("%Y-%m-%d-%H")
                                val = record.get_value() or 0
                                sums[key] = sums.get(key, 0.0) + val
                                counts[key] = counts.get(key, 0) + 1
                        for key, total in sums.items():
                            target_dict[key] = total / counts[key]
                        self.logger.debug(f"Weather {field_name}: {len(target_dict)} hours loaded")
                except Exception as e:
                    self.logger.warning(f"Failed to load weather {field_name}: {e}")

            if not ghi_by_hour:
                self.logger.warning("No weather radiation data found")
                return False

            def _get_cloud(hour_key: str) -> float:
                """Get effective cloud cover from layered data, fall back to total."""
                if hour_key in cloud_low_by_hour:
                    return _effective_cloud_cover(
                        cloud_low_by_hour.get(hour_key, 0),
                        cloud_mid_by_hour.get(hour_key, 0),
                        cloud_high_by_hour.get(hour_key, 0),
                    )
                return cloud_total_by_hour.get(hour_key, 50.0)

            # Helper to add a data point to all bin levels
            def _add_to_model(m: SolarProductionModel, hour_key: str, kwh: float) -> None:
                ghi = ghi_by_hour.get(hour_key, 0)
                if ghi <= 0:
                    return
                parts = hour_key.split("-")
                year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

                cloud = _get_cloud(hour_key)
                temp = temp_by_hour.get(hour_key, 15.0)
                azimuth, altitude = _sun_position(self.latitude, year, month, day, hour)

                if altitude <= 0:
                    return  # Sun below horizon

                rad_b = SolarProductionModel.radiation_to_bucket(ghi)
                cloud_b = SolarProductionModel.cloud_to_bucket(cloud)
                alt_b = SolarProductionModel.altitude_to_bucket(altitude)

                m.add_sample(rad_b, cloud_b, alt_b, kwh)

            # PASS 1: Build rough model from all matched data
            total_kwp = sum(a.kwp for a in self.arrays)
            model = SolarProductionModel(total_kwp=total_kwp)
            matched = 0
            for hour_key, watts in solar_by_hour.items():
                if hour_key not in ghi_by_hour or watts <= 0:
                    continue
                if ghi_by_hour[hour_key] <= 0:
                    continue
                _add_to_model(model, hour_key, watts / 1000.0)
                matched += 1

            if matched < 100:
                self.logger.warning(f"Only {matched} matched hours, need 100+")
                return False

            model.build()

            # PASS 2: Filter curtailed hours and rebuild. Pass 1 stays at the
            # median for stable curtailment detection; the FINAL model uses the
            # configured quantile so the prediction can recover potential.
            model2 = SolarProductionModel(
                total_kwp=total_kwp, quantile=self.model_quantile
            )
            curtailed = 0
            for hour_key, watts in solar_by_hour.items():
                if hour_key not in ghi_by_hour or watts <= 0:
                    continue
                ghi = ghi_by_hour[hour_key]
                if ghi <= 0:
                    continue

                kwh = watts / 1000.0
                parts = hour_key.split("-")
                year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

                cloud = _get_cloud(hour_key)
                temp = temp_by_hour.get(hour_key, 15.0)
                azimuth, altitude = _sun_position(self.latitude, year, month, day, hour)

                soc = soc_by_hour.get(hour_key, 50.0)
                load = load_by_hour.get(hour_key, 0)
                expected = model.predict(
                    ghi,
                    sun_azimuth=azimuth,
                    sun_altitude=altitude,
                    cloud_cover=cloud,
                    temperature=temp,
                )

                # Filter 1a: controller had export disabled this hour → solar
                # was likely curtailed; data isn't representative of true
                # potential. Default 1 (enabled) for hours predating this
                # measurement so they fall back to the heuristic below.
                if export_by_hour.get(hour_key, 1) == 0:
                    curtailed += 1
                    continue

                # Filter 1b: controller had inverter physically off this hour
                # via Modbus register 0 (price-threshold gate). Production
                # was zero by definition — must not be in the training set.
                if inverter_on_by_hour.get(hour_key, 1) == 0:
                    curtailed += 1
                    continue

                # Filter 2 (legacy heuristic): only filter as curtailed when
                # battery is full AND load is low. When load is high, the
                # inverter produces at full capacity to serve the load even
                # at 100% SOC — this is real production data.
                load_kwh = load / 1000.0
                if (soc >= 100 and expected > 0 and kwh < expected * 0.6
                        and load_kwh < expected * 0.5):
                    curtailed += 1
                    continue

                _add_to_model(model2, hour_key, kwh)

            model2.data_points = matched - curtailed
            model2.curtailed_filtered = curtailed

            earliest = min(solar_by_hour.keys())
            latest = max(solar_by_hour.keys())
            model2.date_range = f"{earliest[:10]} to {latest[:10]}"

            model2.build()
            self._production_model = model2

            self.logger.info(
                f"Solar production model built: {model2.data_points} hours "
                f"({curtailed} curtailed filtered), "
                f"{len(model2.median_3d)} 3D / {len(model2.median_2d)} 2D bins, "
                f"range {model2.date_range}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to build production model: {e}", exc_info=True)
            return False

    def predict_from_model(
        self, weather_hourly: List[Dict[str, Any]]
    ) -> Dict[str, DailyForecast]:
        """Predict solar production using the learned model + weather forecast.

        Args:
            weather_hourly: List of dicts with "time", "shortwave_radiation",
                "cloudcover", "temperature_2m"

        Returns:
            Dict of date_str -> DailyForecast
        """
        if not self._production_model:
            return {}

        daily_hourly: Dict[str, Dict[int, float]] = {}

        for entry in weather_hourly:
            time_str = entry.get("time", "")
            ghi = entry.get("shortwave_radiation", 0)
            cloud_cover = entry.get("cloudcover", 50)
            temp = entry.get("temperature_2m", 15)

            if not time_str:
                continue

            try:
                dt = datetime.fromisoformat(time_str)
            except ValueError:
                continue

            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            azimuth, altitude = _sun_position(self.latitude, dt.year, dt.month, dt.day, hour)

            predicted_kwh = self._production_model.predict(
                ghi,
                sun_azimuth=azimuth,
                sun_altitude=altitude,
                cloud_cover=cloud_cover,
                temperature=temp or 15,
            )

            if date_str not in daily_hourly:
                daily_hourly[date_str] = {}
            daily_hourly[date_str][hour] = predicted_kwh

        result: Dict[str, DailyForecast] = {}
        for date_str, hourly in daily_hourly.items():
            total = sum(hourly.values())
            result[date_str] = DailyForecast(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                total_kwh=total,
                hourly=hourly,
                source="model",
            )

        if result:
            self._model_forecast = result

        return result

    def calculate_from_weather(
        self, weather_data: Dict[str, Any]
    ) -> Dict[str, DailyForecast]:
        """Calculate solar production from weather irradiance data.

        Uses shortwave_radiation (W/m²) from OpenMeteo weather forecast
        combined with panel specifications to estimate production.

        Args:
            weather_data: Dict with "hourly" key containing list of dicts,
                each with "time" (ISO timestamp) and "shortwave_radiation" (W/m²)

        Returns:
            Dict mapping date string to DailyForecast
        """
        hourly_data = weather_data.get("hourly", [])
        if not hourly_data:
            return {}

        total_kwp = sum(a.kwp for a in self.arrays)
        # Base performance ratio: accounts for inverter losses, wiring, etc.
        # Temperature derating is applied per-hour below.
        base_performance_ratio = 0.80

        combined_hourly: Dict[str, Dict[int, float]] = {}

        for entry in hourly_data:
            time_str = entry.get("time", "")
            ghi = entry.get("shortwave_radiation", 0)  # W/m² Global Horizontal Irradiance

            if not time_str or ghi <= 0:
                continue

            try:
                dt = datetime.fromisoformat(time_str)
            except ValueError:
                continue

            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            # Skip hours where sun is below horizon
            _, altitude = _sun_position(self.latitude, dt.year, dt.month, dt.day, hour)
            if altitude <= 0:
                continue

            # Temperature derating: silicon panels lose ~0.4%/°C above 25°C cell temp
            # OpenMeteo may return an explicit null for a missing hour; treat
            # only None as missing (0 °C is a valid, falsy temperature).
            temp = entry.get("temperature_2m")
            if temp is None:
                temp = 25
            cell_temp = temp + 25  # NOCT approximation: cell ≈ ambient + 25°C
            temp_factor = 1 + (-0.004) * (cell_temp - 25)

            # production = GHI * total_kwp * performance_ratio * temp_factor / 1000
            # GHI in W/m² for 1 hour → Wh/m², divided by 1000 W/m² (STC reference)
            # gives fraction of peak output
            production_kwh = ghi * total_kwp * base_performance_ratio * temp_factor / 1000.0

            if date_str not in combined_hourly:
                combined_hourly[date_str] = {}
            combined_hourly[date_str][hour] = (
                combined_hourly[date_str].get(hour, 0) + production_kwh
            )

        result: Dict[str, DailyForecast] = {}
        for date_str, hourly in combined_hourly.items():
            total = sum(hourly.values())
            result[date_str] = DailyForecast(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                total_kwh=total,
                hourly=hourly,
                source="weather",
            )

        if result:
            self._weather_forecast = result

        return result

    # --- Consensus ---

    def set_solcast_hourly(self, hourly_by_date: Dict[str, Dict[int, float]]) -> None:
        """Ingest Solcast forecast (date_str → hour → kWh) into the consensus pool.

        Called by the controller after a successful Solcast API fetch.
        Stored alongside model + forecast.solar; build_consensus() blends them.
        """
        from datetime import datetime as _dt
        new: Dict[str, DailyForecast] = {}
        for date_str, hourly in hourly_by_date.items():
            if not hourly:
                continue
            try:
                d = _dt.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            new[date_str] = DailyForecast(
                date=d,
                total_kwh=sum(hourly.values()),
                hourly=dict(hourly),
                source="solcast",
            )
        self._solcast_forecast = new

    def build_consensus(self) -> Dict[str, DailyForecast]:
        """Combine all forecast sources into a consensus.

        Priority: learned model (most accurate) > Solcast > forecast.solar API > weather GHI.
        When multiple sources available, average values that agree, use model
        when sources diverge (it's trained on actual installation data).
        """
        all_dates = (
            set(self._api_forecast.keys())
            | set(self._weather_forecast.keys())
            | set(self._model_forecast.keys())
            | set(self._solcast_forecast.keys())
        )
        result: Dict[str, DailyForecast] = {}

        for date_str in all_dates:
            model = self._model_forecast.get(date_str)
            api = self._api_forecast.get(date_str)
            weather = self._weather_forecast.get(date_str)
            solcast = self._solcast_forecast.get(date_str)

            # Collect all available sources for this date
            sources: List[DailyForecast] = []
            if model and model.total_kwh > 0:
                sources.append(model)
            if solcast and solcast.total_kwh > 0:
                sources.append(solcast)
            if api and api.total_kwh > 0:
                sources.append(api)
            if weather and weather.total_kwh > 0:
                sources.append(weather)

            if not sources:
                continue

            # If learned model available, it's the primary source
            # Average with other sources if they agree, otherwise trust model
            if model and model.total_kwh > 0:
                hourly: Dict[int, float] = {}
                all_hours = set()
                for s in sources:
                    all_hours |= set(s.hourly.keys())

                for hour in all_hours:
                    # Only trust a model 0 when the hour is genuinely PRESENT in
                    # the model (sun below horizon). A MISSING hour (e.g. a dropped
                    # weather entry) must NOT be conflated with a real zero — fall
                    # through to averaging the other sources instead of discarding
                    # their positive PV.
                    if hour in model.hourly and model.hourly[hour] <= 0:
                        hourly[hour] = 0
                        continue
                    model_val = model.hourly.get(hour, 0)

                    # Separate model from non-model sources by identity
                    other_vals = [
                        s.hourly.get(hour, 0)
                        for s in sources
                        if s.source != "model" and s.hourly.get(hour, 0) > 0
                    ]

                    if not other_vals:
                        hourly[hour] = model_val
                    else:
                        avg_others = sum(other_vals) / len(other_vals)
                        divergence = abs(model_val - avg_others) / ((model_val + avg_others) / 2)
                        if divergence <= 0.3:
                            # Agreement: average all sources
                            all_vals = [model_val] + other_vals
                            hourly[hour] = sum(all_vals) / len(all_vals)
                        elif model_val >= avg_others:
                            # Model predicts more: trust it (real installation data)
                            hourly[hour] = model_val
                        else:
                            # Model predicts much less: sparse bin, use other sources
                            hourly[hour] = avg_others

                total = sum(hourly.values())
                source_names = [s.source for s in sources]
                result[date_str] = DailyForecast(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    total_kwh=total,
                    hourly=hourly,
                    source="model+" + "+".join(s for s in source_names if s != "model"),
                )
            elif len(sources) >= 2:
                # No model, but multiple other sources — original consensus logic
                hourly = {}
                all_hours = set()
                for s in sources:
                    all_hours |= set(s.hourly.keys())
                for hour in all_hours:
                    vals = [s.hourly.get(hour, 0) for s in sources if s.hourly.get(hour, 0) > 0]
                    if vals:
                        hourly[hour] = sum(vals) / len(vals)
                    else:
                        hourly[hour] = 0
                total = sum(hourly.values())
                result[date_str] = DailyForecast(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    total_kwh=total,
                    hourly=hourly,
                    source="consensus",
                )
            else:
                result[date_str] = sources[0]

        self._consensus = result
        return result

    # --- Calibration from actuals ---

    async def calibrate_from_actuals(
        self,
        influxdb_client: Any,
        bucket: str,
        days: int = 30,
    ) -> None:
        """Auto-tune confidence factor by comparing past forecasts with actual production.

        Queries actual solar production (InputPower) from InfluxDB for recent days
        and compares with what forecast.solar would have predicted. Updates
        self.confidence to reflect real-world accuracy.

        Args:
            influxdb_client: Async InfluxDB client
            bucket: Solar InfluxDB bucket name
            days: Number of past days to compare (default 30)
        """
        try:
            # Get actual daily production using TodayGenerateEnergy (inverter's own
            # daily total). Exclude today (-1d stop) to avoid comparing a partial
            # day's actual against a full day's forecast.
            query = f'''
from(bucket: "{bucket}")
  |> range(start: -{days}d, stop: -1d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "TodayGenerateEnergy")
  |> aggregateWindow(every: 1d, fn: max, createEmpty: false)
  |> filter(fn: (r) => r._value > 1)
'''
            result = await influxdb_client.query(query)
            if not result:
                self.logger.debug("No actual solar data for calibration")
                return

            # Daily production totals (already in kWh from inverter)
            # Use max() in case of multiple entries per day
            actual_by_day: Dict[str, float] = {}
            for table in result:
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    val = record.get_value()
                    if not isinstance(val, (int, float)):
                        continue  # skip None/empty aggregate (would crash max cmp)
                    if day not in actual_by_day or val > actual_by_day[day]:
                        actual_by_day[day] = val

            if not actual_by_day:
                return

            # Compare with stored forecasts (if we have any for those days)
            # Try consensus (actual prediction path) first, then model, then API,
            # then persisted history from InfluxDB (survives restarts)
            persisted_history: Optional[Dict[str, DailyForecast]] = None
            ratios: list = []
            # Iterate chronologically so the recency weighting below
            # (weight = i + 1) actually tracks date order. `actual_by_day` is a
            # plain dict in InfluxDB-record order; day_str is "YYYY-MM-DD" so
            # lexicographic sort == chronological.
            for day_str, actual_kwh in sorted(actual_by_day.items()):
                forecast = (
                    self._consensus.get(day_str)
                    or self._model_forecast.get(day_str)
                    or self._api_forecast.get(day_str)
                )
                if not forecast:
                    # Lazy-load persisted history on first miss
                    if persisted_history is None:
                        persisted_history = await self._load_forecast_history(
                            influxdb_client, bucket, days=days,
                        )
                    forecast = persisted_history.get(day_str)
                if forecast and forecast.total_kwh > 1.0:
                    ratio = actual_kwh / forecast.total_kwh
                    # Clamp to reasonable range (0.3 - 1.5)
                    ratio = max(0.3, min(1.5, ratio))
                    ratios.append(ratio)
                    self.logger.debug(
                        f"Calibration {day_str}: actual={actual_kwh:.1f} kWh, "
                        f"forecast={forecast.total_kwh:.1f} kWh ({forecast.source}), "
                        f"ratio={ratio:.2f}"
                    )

            if len(ratios) < 3:
                # Not enough forecast history for reliable calibration.
                # Use actuals vs theoretical max as a conservative estimate.
                total_kwp = sum(a.kwp for a in self.arrays)
                if total_kwp > 0 and actual_by_day:
                    avg_actual = sum(actual_by_day.values()) / len(actual_by_day)
                    # Coarse bootstrap until real forecast-vs-actual history
                    # accumulates: compare average daily yield to a good-day
                    # peak (~5 kWh/kWp). This is deliberately conservative — on a
                    # cloudy-average site the ratio is small and the result sits
                    # at the 0.65 floor; it only rises above the floor on sites
                    # whose average day is itself sunny. `confidence` merely
                    # discounts NON-model sources, so a floor-biased estimate is
                    # acceptable here.
                    typical_max = total_kwp * 5  # ~5 kWh/kWp on a good day
                    estimated_confidence = min(1.0, avg_actual / typical_max)
                    self.confidence = max(0.65, estimated_confidence)
                    self.logger.info(
                        f"☀️ Calibration (insufficient forecast history, {len(ratios)} days): "
                        f"confidence={self.confidence:.2f} "
                        f"(avg actual: {avg_actual:.1f} kWh, typical max: {typical_max:.0f} kWh)"
                    )
                return

            # Weighted average: recent days count more
            # Last day weight=days, first day weight=1
            weighted_sum = 0.0
            weight_total = 0.0
            for i, ratio in enumerate(ratios):
                weight = i + 1  # Older=1, newer=len
                weighted_sum += ratio * weight
                weight_total += weight

            new_confidence = weighted_sum / weight_total if weight_total > 0 else 0.7
            # Clamp to reasonable range
            new_confidence = max(0.3, min(1.2, new_confidence))

            old_confidence = self.confidence
            self.confidence = new_confidence
            self.logger.info(
                f"☀️ Calibration: confidence {old_confidence:.2f} → {new_confidence:.2f} "
                f"(from {len(ratios)} days of actual vs forecast data)"
            )

        except Exception as e:
            self.logger.warning(f"Solar calibration failed: {e}")

    # --- Reliability check ---

    def has_reliable_forecast(self) -> bool:
        """Check if we have a reliable solar forecast to base decisions on.

        Returns False if:
        - No forecast data at all
        - Forecast is suspiciously low for the season and system size
        """
        if not self._consensus:
            return False

        from datetime import date as _date
        today = _date.today()
        total_kwp = sum(a.kwp for a in self.arrays)
        for forecast in self._consensus.values():
            # Skip stale past-dated entries that were never evicted from the
            # consensus pool — a yesterday forecast must not gate today's
            # reliability decision.
            if forecast.date < today:
                continue
            day_of_year = forecast.date.timetuple().tm_yday
            if 91 <= day_of_year <= 273:  # April-September
                min_kwh = total_kwp * 0.3
            else:  # October-March
                min_kwh = total_kwp * 0.1
            if total_kwp > 5 and forecast.total_kwh < min_kwh:
                return False

        return True

    # --- Public interface ---

    def get_expected_production_kwh(self, target_date: date) -> float:
        """Get total expected production for a date.

        If the consensus is from the learned model, use values directly
        (the model already reflects real-world installation performance).
        Only apply confidence discount for API/weather-only sources.

        Args:
            target_date: Date to forecast

        Returns:
            Expected production in kWh
        """
        date_str = target_date.strftime("%Y-%m-%d")
        forecast = self._consensus.get(date_str)
        if not forecast:
            return 0.0
        # Model-based forecasts are already calibrated to real production
        if forecast.source.startswith("model"):
            return forecast.total_kwh
        return forecast.total_kwh * self.confidence

    def get_hourly_production(self, target_date: date) -> Dict[int, float]:
        """Get hourly production forecast for a date.

        Model-based forecasts used directly; API/weather discounted by confidence.

        Args:
            target_date: Date to forecast

        Returns:
            Dict mapping hour (0-23) to expected kWh
        """
        date_str = target_date.strftime("%Y-%m-%d")
        forecast = self._consensus.get(date_str)
        if not forecast:
            return {}
        if forecast.source.startswith("model"):
            return dict(forecast.hourly)
        return {h: kwh * self.confidence for h, kwh in forecast.hourly.items()}

    @staticmethod
    def apply_live_persistence(
        hourly_forecast: Dict[int, float],
        live_power_kw: float,
        current_hour: int,
        min_live_kw: float = 0.5,
        blend_horizon_hours: int = 2,
        minutes_elapsed: int = 0,
    ) -> Dict[int, float]:
        """Anchor near-future forecast to live measured solar power.

        Inspired by EMHASS's "mixed one-observation persistence" approach:
        the very near future is dominated by what's actually happening RIGHT
        NOW (live PV power), smoothly transitioning to the model forecast
        for further-out hours.

        For each affected hour h, returns:
            max(forecast[h], α(h) * live_power_kw + (1-α(h)) * forecast[h])

        where α decays linearly from 1.0 at the current hour to 0.0 at
        `current_hour + blend_horizon_hours`. Using max() means we only
        ever *lift* the forecast — we don't trust a temporary cloud-pass
        (live dropped to 0 for a minute) to suppress an otherwise-sunny
        forecast. Calibration handles the over-prediction direction.

        `live_power_kw` is an *instantaneous* power reading (kW); a future
        full hour at that power yields ≈ that many kWh. The CURRENT hour is
        only partially ahead of us, so only its remaining fraction
        (`1 - minutes_elapsed/60`) can be anchored to live power — the elapsed
        part keeps the model's energy. Without this the snapshot kW is booked
        as a whole hour of kWh, systematically over-forecasting the current
        hour (worst near solar noon, where instantaneous kW peaks well above
        the hour's average energy).

        Disabled when live_power_kw < min_live_kw (likely nighttime or
        fault — model forecast handles those correctly).

        Args:
            hourly_forecast: model+API forecast, mutated COPY returned
            live_power_kw: instantaneous PV power from telemetry, in kW
            current_hour: hour-of-day (0-23) we're currently in (local)
            min_live_kw: skip blending if live below this (night/fault)
            blend_horizon_hours: how far ahead the live anchor decays to 0
            minutes_elapsed: minutes already elapsed in the current hour (0-59)

        Returns:
            Blended forecast dict (does not mutate input).
        """
        out = dict(hourly_forecast)
        if live_power_kw < min_live_kw or blend_horizon_hours <= 0:
            return out
        for h_offset in range(blend_horizon_hours + 1):
            h = current_hour + h_offset
            if h >= 24:
                break
            alpha = max(0.0, 1.0 - h_offset / blend_horizon_hours)
            model_kwh = out.get(h, 0.0)
            if h_offset == 0:
                # Current hour: only the remaining minutes can be anchored to
                # live power; the elapsed share stays at the model's energy.
                remaining = max(0.0, 1.0 - min(max(minutes_elapsed, 0), 60) / 60.0)
                blended = (
                    model_kwh * (1.0 - remaining)
                    + alpha * live_power_kw * remaining
                    + (1.0 - alpha) * model_kwh * remaining
                )
            else:
                blended = alpha * live_power_kw + (1.0 - alpha) * model_kwh
            out[h] = max(model_kwh, blended)
        return out

    async def _hourly_mean_today(
        self, influxdb_client: Any, bucket: str, field: str, current_hour: int,
        target_date: date, local_tz: Any, lookback_hours: int, scale: float = 1.0,
    ) -> Dict[int, float]:
        """Hourly-mean of a `solar` field for completed hours of today.

        Returns {hour: value*scale} for hours < current_hour on target_date.
        """
        out: Dict[int, float] = {}
        try:
            q = f'''
from(bucket: "{bucket}")
  |> range(start: -{lookback_hours + 1}h)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "{field}")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            result = await influxdb_client.query(q)
            if result:
                for table in result:
                    for record in table.records:
                        t = record.get_time()
                        if local_tz is not None and t.tzinfo is not None:
                            t = t.astimezone(local_tz)
                        if t.date() != target_date or t.hour >= current_hour:
                            continue
                        v = record.get_value()
                        if isinstance(v, (int, float)):
                            out[t.hour] = float(v) * scale
        except Exception as e:
            self.logger.debug(f"_hourly_mean_today({field}) failed: {e}")
        return out

    async def _state_by_hour_today(
        self, influxdb_client: Any, bucket: str, field: str, current_hour: int,
        target_date: date, local_tz: Any,
    ) -> Dict[int, int]:
        """Carry-forward an inverter_state on/off field to today's hours.

        State is recorded only on change, so we read the last 24h of records
        and, for each completed hour h of today, take the most recent value
        at/before the END of that hour. Missing → 1 (enabled/on), matching the
        training-filter convention.
        """
        out: Dict[int, int] = {}
        try:
            q = f'''
from(bucket: "{bucket}")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "inverter_state" and r._field == "{field}")
  |> sort(columns: ["_time"])
'''
            result = await influxdb_client.query(q)
            records: List[Tuple[datetime, int]] = []
            if result:
                for table in result:
                    for record in table.records:
                        t = record.get_time()
                        if local_tz is not None and t.tzinfo is not None:
                            t = t.astimezone(local_tz)
                        v = record.get_value()
                        if isinstance(v, (int, float)):
                            records.append((t, int(round(float(v)))))
            records.sort(key=lambda r: r[0])
            for h in range(current_hour):
                # End of hour h, local. Build a naive comparison via the record
                # timestamps (already local). Last record at/before this wins.
                last = 1
                for t, v in records:
                    if t.date() < target_date or (
                        t.date() == target_date and t.hour <= h
                    ):
                        last = v
                    else:
                        break
                out[h] = last
        except Exception as e:
            self.logger.debug(f"_state_by_hour_today({field}) failed: {e}")
        return out

    async def compute_intraday_calibration(
        self,
        influxdb_client: Any,
        bucket: str,
        current_hour: int,
        target_date: date,
        local_tz: Any = None,
        lookback_hours: int = 3,
        min_forecast_kwh: float = 0.3,
        clamp_low: float = 0.5,
        clamp_high: float = 3.0,
        exclude_curtailed: bool = True,
        min_usable_hours: int = 2,
    ) -> Optional[float]:
        """Return a scaling ratio that aligns today's forecast to recent actuals.

        Looks at the last `lookback_hours` COMPLETED hours of today: pulls
        actual hourly InputPower from InfluxDB and compares with what
        get_hourly_production() said for those same hours. Returns a single
        scalar that the caller should multiply against the remaining-day
        forecast.

        CURTAILMENT-AWARE: with `exclude_curtailed` (default True) any recent
        hour whose production was *deliberately limited* — inverter switched
        off by the price gate, export disabled, or the battery full with low
        load (MPPT throttled) — is dropped from the comparison. Those actuals
        reflect a dispatch DECISION, not the panels' potential; calibrating to
        them would scale the potential forecast down to censored output and,
        via the optimizer, cause needless grid charging. This mirrors the
        filters the production-model training already applies.

        Returns None when there isn't enough useful data (night, very early
        morning, freshly-deployed, or all-tiny forecasts) — caller leaves
        the forecast untouched in that case.

        Args:
            influxdb_client: AsyncInfluxDBClient instance
            bucket: solar bucket name
            current_hour: hour-of-day (0-23) we're currently in (incomplete)
            target_date: today's date
            lookback_hours: how many complete hours back to consider
            min_forecast_kwh: skip the calibration if forecast for all
                considered hours is below this (predawn / dusk noise)
            clamp_low, clamp_high: hard bounds on the ratio so a single
                outlier hour can't drive a wild correction

        Returns:
            Scaling factor in [clamp_low, clamp_high], or None.
        """
        # Pull last lookback_hours of actual hourly production.
        # We need data from completed hours only, so window ends at the
        # start of current_hour (today, local). For InfluxDB simplicity we
        # use a relative range — current_hour is an hour-of-day; we use
        # the wall time and trust aggregateWindow to bucket cleanly.
        if current_hour <= 0:
            return None  # nothing completed yet today
        try:
            # Query the last lookback_hours+1 hours so we have buffer; we'll
            # then filter to completed hours of today only.
            q = f'''
from(bucket: "{bucket}")
  |> range(start: -{lookback_hours + 1}h)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "InputPower")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            result = await influxdb_client.query(q)
            actual_by_hour: Dict[int, float] = {}
            if result:
                for table in result:
                    for record in table.records:
                        t = record.get_time()
                        # InfluxDB returns UTC timestamps; convert to local
                        # so hour/date match what the caller expects (the
                        # controller passes local-time current_hour and
                        # local-date target_date).
                        if local_tz is not None and t.tzinfo is not None:
                            t = t.astimezone(local_tz)
                        if t.date() != target_date:
                            continue
                        h = t.hour
                        if h >= current_hour:
                            continue  # incomplete current hour — skip
                        watts = record.get_value()
                        if isinstance(watts, (int, float)):
                            actual_by_hour[h] = float(watts) / 1000.0  # kWh

            if not actual_by_hour:
                return None

            forecast_today = self.get_hourly_production(target_date)

            # Pull the signals needed to detect curtailed hours over the same
            # window: SOC + house load (dense), and the export/inverter on-off
            # state (sparse state-change records → carried forward).
            soc_by_hour: Dict[int, float] = {}
            load_by_hour: Dict[int, float] = {}
            export_by_hour: Dict[int, int] = {}
            inverter_on_by_hour: Dict[int, int] = {}
            if exclude_curtailed:
                soc_by_hour = await self._hourly_mean_today(
                    influxdb_client, bucket, "SOC", current_hour, target_date,
                    local_tz, lookback_hours, scale=1.0,
                )
                load_by_hour = await self._hourly_mean_today(
                    influxdb_client, bucket, "INVPowerToLocalLoad", current_hour,
                    target_date, local_tz, lookback_hours, scale=1 / 1000.0,
                )
                export_by_hour = await self._state_by_hour_today(
                    influxdb_client, bucket, "export_enabled", current_hour,
                    target_date, local_tz,
                )
                inverter_on_by_hour = await self._state_by_hour_today(
                    influxdb_client, bucket, "inverter_on", current_hour,
                    target_date, local_tz,
                )

            def _is_curtailed(h: int) -> bool:
                if not exclude_curtailed:
                    return False
                # Inverter off (price gate) or export disabled → censored.
                if inverter_on_by_hour.get(h, 1) == 0:
                    return True
                if export_by_hour.get(h, 1) == 0:
                    return True
                # Battery full + low load → MPPT throttled below potential.
                fc = forecast_today.get(h, 0.0)
                if (
                    soc_by_hour.get(h, 50.0) >= 99.0
                    and fc > 0
                    and actual_by_hour[h] < fc * 0.6
                    and load_by_hour.get(h, 0.0) < fc * 0.5
                ):
                    return True
                return False

            # Drop pre-dawn / dusk hours (tiny actual/forecast → noisy ratio)
            # AND any hour whose output was deliberately curtailed.
            all_recent = sorted(actual_by_hour.keys())
            usable_hours = [
                h for h in all_recent
                if actual_by_hour[h] >= 0.2 and forecast_today.get(h, 0.0) >= 0.2
                and not _is_curtailed(h)
            ]
            n_curtailed = sum(
                1 for h in all_recent
                if actual_by_hour[h] >= 0.2 and forecast_today.get(h, 0.0) >= 0.2
                and _is_curtailed(h)
            )
            if n_curtailed:
                self.logger.info(
                    f"☀️  Intraday calibration: skipped {n_curtailed} curtailed "
                    f"hour(s) (battery full / inverter off / export disabled)"
                )

            if not usable_hours:
                # Everything recent was curtailed (e.g. sunny + battery full all
                # afternoon): the actuals don't reflect potential, so leave the
                # forecast untouched rather than suppress it.
                return None

            # Require a minimum number of usable hours. After curtailment
            # filtering, a single surviving hour is often an early-morning
            # (low-sun-angle) sample with a large relative forecast error; one
            # such hour would otherwise drive a wild (clamp-pinned) day-wide
            # rescale. Calibrating to a single noisy sample is worse than not
            # calibrating, so back off and leave the forecast as-is.
            if len(usable_hours) < min_usable_hours:
                self.logger.info(
                    f"☀️  Intraday calibration: only {len(usable_hours)} usable "
                    f"hour(s) after curtailment filtering (need "
                    f"{min_usable_hours}) — leaving forecast unscaled"
                )
                return None

            # Restrict to the most recent `lookback_hours` USABLE hours.
            considered = usable_hours[-lookback_hours:]

            total_actual = sum(actual_by_hour[h] for h in considered)
            total_forecast = sum(forecast_today.get(h, 0.0) for h in considered)

            if total_forecast < min_forecast_kwh:
                # Pre-dawn / very dim forecasts have huge ratio noise.
                return None

            ratio = total_actual / total_forecast
            clamped = max(clamp_low, min(clamp_high, ratio))
            self.logger.info(
                f"☀️  Intraday solar calibration: actual {total_actual:.2f} kWh "
                f"vs forecast {total_forecast:.2f} kWh over hours {considered} "
                f"→ ratio {ratio:.2f} (clamped {clamped:.2f})"
            )
            return clamped
        except Exception as e:
            self.logger.debug(f"Intraday calibration failed: {e}")
            return None

    # --- Persistence ---

    async def save_to_influxdb(self, influxdb_client: Any, bucket: str) -> None:
        """Persist current API forecast to InfluxDB so it survives restarts."""
        if not self._api_forecast or not influxdb_client:
            return

        try:
            for date_str, forecast in self._api_forecast.items():
                # Store hourly data as a JSON string field
                hourly_json = json.dumps({str(h): round(kwh, 3) for h, kwh in forecast.hourly.items()})
                await influxdb_client.write_point(
                    bucket=bucket,
                    measurement="solar_forecast_cache",
                    fields={
                        "total_kwh": forecast.total_kwh,
                        "hourly_json": hourly_json,
                        "source": forecast.source,
                    },
                    tags={"forecast_date": date_str},
                )
            self.logger.debug(f"Saved {len(self._api_forecast)} forecast days to InfluxDB")
        except Exception as e:
            self.logger.warning(f"Failed to persist forecast: {e}")

    async def save_consensus_to_influxdb(self, influxdb_client: Any, bucket: str) -> None:
        """Persist consensus forecasts to InfluxDB for post-restart calibration."""
        if not self._consensus or not influxdb_client:
            return

        try:
            for date_str, forecast in self._consensus.items():
                hourly_json = json.dumps({str(h): round(kwh, 3) for h, kwh in forecast.hourly.items()})
                await influxdb_client.write_point(
                    bucket=bucket,
                    measurement="solar_forecast_history",
                    fields={
                        "total_kwh": forecast.total_kwh,
                        "hourly_json": hourly_json,
                        "source": forecast.source,
                    },
                    tags={"forecast_date": date_str},
                )
            self.logger.debug(f"Saved {len(self._consensus)} consensus forecast days to InfluxDB")
        except Exception as e:
            self.logger.warning(f"Failed to persist consensus forecast: {e}")

    async def _load_forecast_history(
        self, influxdb_client: Any, bucket: str, days: int = 30,
    ) -> Dict[str, DailyForecast]:
        """Batch-load historical consensus forecasts from InfluxDB.

        Used by calibration to compare past forecasts with actuals after restarts.
        """
        try:
            query = f'''
from(bucket: "{bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "solar_forecast_history")
  |> last()
'''
            result = await influxdb_client.query(query)
            if not result:
                return {}

            by_date: Dict[str, Dict[str, Any]] = {}
            for table in result:
                for record in table.records:
                    date_tag = record.values.get("forecast_date", "")
                    if not date_tag:
                        continue
                    if date_tag not in by_date:
                        by_date[date_tag] = {}
                    by_date[date_tag][record.get_field()] = record.get_value()

            forecasts: Dict[str, DailyForecast] = {}
            for date_str, fields in by_date.items():
                total = fields.get("total_kwh", 0)
                if not total or total < 1:
                    continue
                hourly_json = fields.get("hourly_json", "{}")
                try:
                    hourly_raw = json.loads(hourly_json) if isinstance(hourly_json, str) else {}
                    hourly = {int(h): float(kwh) for h, kwh in hourly_raw.items()}
                except (json.JSONDecodeError, ValueError):
                    hourly = {}
                forecasts[date_str] = DailyForecast(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    total_kwh=float(total),
                    hourly=hourly,
                    source=str(fields.get("source", "cached_consensus")),
                )

            if forecasts:
                self.logger.debug(f"Loaded {len(forecasts)} historical forecast days from InfluxDB")
            return forecasts
        except Exception:
            return {}

    async def load_from_influxdb(self, influxdb_client: Any, bucket: str) -> bool:
        """Load cached forecast from InfluxDB (used on startup when API is rate-limited)."""
        if not influxdb_client:
            return False

        try:
            query = f'''
from(bucket: "{bucket}")
  |> range(start: -2d)
  |> filter(fn: (r) => r._measurement == "solar_forecast_cache")
  |> last()
'''
            result = await influxdb_client.query(query)
            if not result:
                return False

            # Parse results grouped by forecast_date tag
            by_date: Dict[str, Dict[str, Any]] = {}
            for table in result:
                for record in table.records:
                    date_tag = record.values.get("forecast_date", "")
                    if not date_tag:
                        continue
                    if date_tag not in by_date:
                        by_date[date_tag] = {}
                    field = record.get_field()
                    by_date[date_tag][field] = record.get_value()

            loaded = 0
            for date_str, fields in by_date.items():
                total = fields.get("total_kwh", 0)
                hourly_json = fields.get("hourly_json", "{}")
                if not total or total < 1:
                    continue
                try:
                    hourly_raw = json.loads(hourly_json) if isinstance(hourly_json, str) else {}
                    hourly = {int(h): float(kwh) for h, kwh in hourly_raw.items()}
                except (json.JSONDecodeError, ValueError):
                    hourly = {}

                self._api_forecast[date_str] = DailyForecast(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    total_kwh=float(total),
                    hourly=hourly,
                    source="cached",
                )
                loaded += 1

            if loaded > 0:
                self.logger.info(f"Loaded {loaded} cached forecast days from InfluxDB")
                return True
            return False

        except Exception as e:
            self.logger.debug(f"Could not load cached forecast: {e}")
            return False

    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get a summary of current forecast state for logging/API."""
        summary: Dict[str, Any] = {
            "arrays": [
                {"name": a.name, "kwp": a.kwp, "azimuth": a.azimuth, "tilt": a.declination}
                for a in self.arrays
            ],
            "total_kwp": sum(a.kwp for a in self.arrays),
            "confidence": self.confidence,
            "last_api_update": self._last_api_update.isoformat() if self._last_api_update else None,
            "forecasts": {},
        }
        for date_str, forecast in self._consensus.items():
            summary["forecasts"][date_str] = {
                "total_kwh": round(forecast.total_kwh, 1),
                "discounted_kwh": round(forecast.total_kwh * self.confidence, 1),
                "source": forecast.source,
                "peak_hour": max(forecast.hourly, key=forecast.hourly.get) if forecast.hourly else None,
                "peak_kwh": round(max(forecast.hourly.values()), 2) if forecast.hourly else 0,
            }
        return summary


# Required for type hints in the module
import asyncio  # noqa: E402 - needed for TimeoutError in fetch_api_forecast
