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

    Bins actual production by physics-based features:
    - radiation_bucket: GHI in 50 W/m² steps (primary energy driver)
    - clear_sky_bucket: direct/(direct+diffuse) ratio (cloud type)
    - sun_azimuth_bucket: sun compass direction in 30° steps (which array gets light)
    - sun_altitude_bucket: sun height angle in 10° steps (angle of incidence)
    - temp_bucket: outdoor temperature in 5°C steps (panel efficiency derating)

    Uses p75 for prediction — represents panel POTENTIAL on a good day.
    Two-pass training filters curtailed hours (SOC >= 90% + low output).
    """
    # Primary 5D bins: (rad, clear_sky, azimuth, altitude, temp) -> [kwh]
    bins: Dict[Tuple[int, int, int, int, int], List[float]] = field(default_factory=dict)
    p75: Dict[Tuple[int, int, int, int, int], float] = field(default_factory=dict)
    medians: Dict[Tuple[int, int, int, int, int], float] = field(default_factory=dict)
    # 3D fallback: (rad, azimuth, altitude) -> [kwh]
    mid_bins: Dict[Tuple[int, int, int], List[float]] = field(default_factory=dict)
    mid_p75: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    # 2D fallback: (rad, altitude) -> [kwh]
    simple_bins: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    simple_p75: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # 1D fallback: altitude -> median
    altitude_fallback: Dict[int, float] = field(default_factory=dict)
    global_median: float = 0.0
    # Metadata
    data_points: int = 0
    curtailed_filtered: int = 0
    date_range: str = ""
    built_at: Optional[datetime] = None

    @staticmethod
    def radiation_to_bucket(ghi: float) -> int:
        """GHI W/m² → bucket (50 W/m² steps, 0-20)."""
        return min(20, max(0, int(ghi / 50)))

    @staticmethod
    def clear_sky_to_bucket(direct: float, diffuse: float) -> int:
        """Clear sky ratio → bucket (0-4, 20% steps). 4 = fully clear."""
        total = direct + diffuse
        if total <= 0:
            return 0
        return min(4, int((direct / total) / 0.2))

    @staticmethod
    def azimuth_to_bucket(azimuth_deg: float) -> int:
        """Sun azimuth → bucket (30° steps, 0-11). 0=North, 3=East, 6=South, 9=West."""
        return int((azimuth_deg % 360) / 30)

    @staticmethod
    def altitude_to_bucket(altitude_deg: float) -> int:
        """Sun altitude → bucket (10° steps, 0-9). 0=horizon, 6=60°."""
        return min(9, max(0, int(altitude_deg / 10)))

    @staticmethod
    def temp_to_bucket(temp_c: float) -> int:
        """Temperature → bucket (5°C steps, clamped -4 to 8 = -20°C to 40°C)."""
        return min(12, max(0, int((temp_c + 20) / 5)))

    @staticmethod
    def _compute_p75(values: List[float]) -> Tuple[float, float]:
        """Compute median and p75 from a list of values with IQR outlier removal."""
        if not values:
            return 0.0, 0.0
        if len(values) < 3:
            return statistics.median(values), max(values)
        sorted_vals = sorted(values)
        q1 = sorted_vals[len(sorted_vals) // 4]
        q3 = sorted_vals[3 * len(sorted_vals) // 4]
        iqr = q3 - q1
        filtered = [v for v in values if (q1 - 1.5 * iqr) <= v <= (q3 + 1.5 * iqr)]
        if not filtered:
            filtered = values
        med = statistics.median(filtered)
        p75_idx = min(int(len(filtered) * 0.75), len(filtered) - 1)
        return med, filtered[p75_idx]

    def build(self) -> None:
        """Compute percentiles from all bin levels (5D → 3D → 2D → 1D)."""
        self.p75 = {}
        self.medians = {}
        self.mid_p75 = {}
        self.simple_p75 = {}
        self.altitude_fallback = {}
        alt_all: Dict[int, List[float]] = {}

        # 5D bins
        for key, values in self.bins.items():
            if not values:
                continue
            med, p75 = self._compute_p75(values)
            self.medians[key] = med
            self.p75[key] = p75

        # 3D bins (rad, azimuth, altitude)
        for key, values in self.mid_bins.items():
            if not values:
                continue
            _, p75 = self._compute_p75(values)
            self.mid_p75[key] = p75
            alt_bucket = key[2]  # altitude
            if alt_bucket not in alt_all:
                alt_all[alt_bucket] = []
            alt_all[alt_bucket].append(p75)

        # 2D bins (rad, altitude)
        for key, values in self.simple_bins.items():
            if not values:
                continue
            _, p75 = self._compute_p75(values)
            self.simple_p75[key] = p75

        # 1D altitude fallback
        for alt_b, vals in alt_all.items():
            self.altitude_fallback[alt_b] = statistics.median(vals) if vals else 0.0

        all_p75 = [v for v in self.p75.values() if v > 0]
        if not all_p75:
            all_p75 = [v for v in self.mid_p75.values() if v > 0]
        self.global_median = statistics.median(all_p75) if all_p75 else 0.0
        self.built_at = datetime.now()

    def predict(
        self, ghi: float,
        sun_azimuth: float = 180, sun_altitude: float = 45,
        direct_radiation: float = 0, diffuse_radiation: float = 0,
        temperature: float = 15,
    ) -> float:
        """Predict potential kWh production for given conditions.

        Fallback chain: 5D → 3D (rad, azimuth, alt) → 2D (rad, alt) → 1D (alt) → global.

        Args:
            ghi: Global Horizontal Irradiance (W/m²)
            sun_azimuth: Sun compass direction (degrees, 0=N, 90=E, 180=S, 270=W)
            sun_altitude: Sun height above horizon (degrees)
            direct_radiation: Direct radiation (W/m²)
            diffuse_radiation: Diffuse radiation (W/m²)
            temperature: Outdoor temperature (°C)
        """
        if ghi <= 0 or sun_altitude <= 0:
            return 0.0

        rad_b = self.radiation_to_bucket(ghi)
        clear_b = self.clear_sky_to_bucket(direct_radiation, diffuse_radiation)
        az_b = self.azimuth_to_bucket(sun_azimuth)
        alt_b = self.altitude_to_bucket(sun_altitude)
        temp_b = self.temp_to_bucket(temperature)

        # Physics cap: can't produce more than GHI × kWp × perf_ratio
        # This prevents garbage predictions from sparse bins
        total_kwp = 13.5  # TODO: get from config
        physics_max = ghi * total_kwp * 0.85 / 1000 / 4.0  # kWh per 15-min block

        def _capped(val: float) -> float:
            return min(val, physics_max)

        # 5D exact match
        key = (rad_b, clear_b, az_b, alt_b, temp_b)
        if key in self.p75:
            return _capped(self.p75[key])

        # 5D: try adjacent temperature first (least impact on production)
        for dt in [1, -1, 2, -2]:
            adj = (rad_b, clear_b, az_b, alt_b, temp_b + dt)
            if adj in self.p75:
                return _capped(self.p75[adj])

        # 5D: try adjacent clear sky BEFORE radiation
        # (clear sky ratio matters more — overcast vs clear at same GHI is huge)
        for dc in [1, -1]:
            adj = (rad_b, clear_b + dc, az_b, alt_b, temp_b)
            if adj in self.p75:
                return _capped(self.p75[adj])

        # 5D: try adjacent radiation (same clear sky)
        for dr in [1, -1]:
            adj = (rad_b + dr, clear_b, az_b, alt_b, temp_b)
            if adj in self.p75:
                return _capped(self.p75[adj])

        # 2D fallback (rad, altitude) — simple but reliable
        simple_key = (rad_b, alt_b)
        if simple_key in self.simple_p75:
            return _capped(self.simple_p75[simple_key])

        for dr in [1, -1, 2, -2]:
            adj = (rad_b + dr, alt_b)
            if adj in self.simple_p75:
                return _capped(self.simple_p75[adj])

        # 1D altitude fallback
        if alt_b in self.altitude_fallback:
            return _capped(self.altitude_fallback[alt_b])

        return _capped(self.global_median)


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
    ):
        self.arrays = arrays
        self.latitude = latitude
        self.longitude = longitude
        self.confidence = confidence
        self.logger = logger or logging.getLogger(__name__)

        # Cached forecasts
        self._api_forecast: Dict[str, DailyForecast] = {}  # date_str -> forecast
        self._weather_forecast: Dict[str, DailyForecast] = {}
        self._model_forecast: Dict[str, DailyForecast] = {}  # from learned model
        self._consensus: Dict[str, DailyForecast] = {}
        self._last_api_update: Optional[datetime] = None

        # Learned production model
        self._production_model: Optional[SolarProductionModel] = None

    @classmethod
    def from_config(cls, config: Any, logger: Optional[logging.Logger] = None) -> "SolarForecast":
        """Create SolarForecast from GrowattConfig."""
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
        return cls(
            arrays=arrays,
            latitude=getattr(config, "latitude", 49.0),
            longitude=getattr(config, "longitude", 14.5),
            confidence=getattr(config, "solar_forecast_confidence", 0.7),
            logger=logger,
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
                            self.logger.warning("forecast.solar rate limit hit, using cached data")
                            return self._api_forecast
                        if resp.status != 200:
                            self.logger.warning(
                                f"forecast.solar returned {resp.status} for {array.name}"
                            )
                            continue

                        data = await resp.json()
                        # Use watt_hours_period (per-hour), NOT watt_hours (cumulative)
                        watt_hours = data.get("result", {}).get("watt_hours_period", {})

                        for timestamp_str, wh in watt_hours.items():
                            # Format: "YYYY-MM-DD HH:MM:SS"
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
            f"&hourly=shortwave_radiation"
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

                    if not times or not radiation:
                        return {}

                    weather_data = {"hourly": []}
                    for t, ghi in zip(times, radiation):
                        weather_data["hourly"].append({
                            "time": t,
                            "shortwave_radiation": ghi or 0,
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
        self, influxdb_client: Any, settings: Any
    ) -> bool:
        """Train solar production model from historical data (365 days).

        Two-pass approach:
        1. Build rough model from all matched solar+weather hours
        2. Filter curtailed hours (high SOC + low production) and rebuild

        Args:
            influxdb_client: Async InfluxDB client
            settings: App settings (for bucket names)

        Returns:
            True if model was built successfully
        """
        try:
            self.logger.info("Building solar production model from historical data...")

            # Calculate max sun elevation per date using astral
            from astral import LocationInfo
            from astral.sun import sun
            loc = LocationInfo("home", "", "Europe/Prague", self.latitude, self.longitude)
            max_sun_elev_by_date: Dict[str, float] = {}
            base_date = datetime.now().date()
            for days_ago in range(730):
                d = base_date - timedelta(days=days_ago)
                try:
                    s = sun(loc.observer, date=d)
                    # Max elevation at solar noon
                    noon = s["noon"]
                    # Sun elevation = 90 - zenith. At noon, elevation is max.
                    # Approximate: latitude-based max elevation for date
                    import math as _math
                    day_of_year = d.timetuple().tm_yday
                    declination = 23.45 * _math.sin(_math.radians(360 / 365 * (day_of_year - 81)))
                    max_elev = 90 - abs(self.latitude - declination)
                    max_sun_elev_by_date[d.strftime("%Y-%m-%d")] = max_elev
                except Exception:
                    max_sun_elev_by_date[d.strftime("%Y-%m-%d")] = 45.0

            # Query hourly solar production
            solar_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "InputPower")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            # Query hourly SOC (for curtailment detection)
            soc_query = f'''
from(bucket: "{settings.influxdb.bucket_solar}")
  |> range(start: -730d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "SOC")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
'''
            # Query hourly GHI + direct + diffuse radiation
            # Query weather fields separately (much faster than combined OR query)
            weather_bucket = settings.influxdb.bucket_weather
            weather_fields = {
                "shortwave_radiation": "ghi",
                "direct_radiation": "direct",
                "diffuse_radiation": "diffuse",
                "temperature_2m": "temp",
            }

            solar_result = await influxdb_client.query(solar_query)
            soc_result = await influxdb_client.query(soc_query)

            if not solar_result:
                self.logger.warning("No solar production data found")
                return False

            # Parse solar data
            solar_by_hour: Dict[str, float] = {}
            for table in solar_result:
                for record in table.records:
                    key = record.get_time().strftime("%Y-%m-%d-%H")
                    solar_by_hour[key] = record.get_value()

            soc_by_hour: Dict[str, float] = {}
            if soc_result:
                for table in soc_result:
                    for record in table.records:
                        key = record.get_time().strftime("%Y-%m-%d-%H")
                        soc_by_hour[key] = record.get_value()

            # Query each weather field separately for performance
            ghi_by_hour: Dict[str, float] = {}
            direct_by_hour: Dict[str, float] = {}
            diffuse_by_hour: Dict[str, float] = {}
            temp_by_hour: Dict[str, float] = {}

            field_targets = {
                "shortwave_radiation": ghi_by_hour,
                "direct_radiation": direct_by_hour,
                "diffuse_radiation": diffuse_by_hour,
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
                        for table in result:
                            for record in table.records:
                                key = record.get_time().strftime("%Y-%m-%d-%H")
                                val = record.get_value() or 0
                                if key not in target_dict or val > target_dict[key]:
                                    target_dict[key] = val
                        self.logger.debug(f"Weather {field_name}: {len(target_dict)} hours loaded")
                except Exception as e:
                    self.logger.warning(f"Failed to load weather {field_name}: {e}")

            if not ghi_by_hour:
                self.logger.warning("No weather radiation data found")
                return False

            # Pre-compute sun position for each hour using astronomy
            import math as _math
            import zoneinfo
            local_tz = zoneinfo.ZoneInfo("Europe/Prague")

            def _sun_position(year: int, month: int, day: int, hour: int) -> Tuple[float, float]:
                """Compute sun azimuth and altitude for a given datetime.
                Returns (azimuth_degrees, altitude_degrees)."""
                try:
                    dt = datetime(year, month, day, hour, 30, tzinfo=local_tz)  # mid-hour
                    day_of_year = dt.timetuple().tm_yday
                    # Solar declination
                    decl = 23.45 * _math.sin(_math.radians(360 / 365 * (day_of_year - 81)))
                    # Hour angle (15° per hour from solar noon)
                    # Solar noon ≈ 12:00 local standard time (approximate)
                    hour_angle = (hour + 0.5 - 12) * 15
                    lat_rad = _math.radians(self.latitude)
                    decl_rad = _math.radians(decl)
                    ha_rad = _math.radians(hour_angle)
                    # Altitude
                    sin_alt = (_math.sin(lat_rad) * _math.sin(decl_rad) +
                               _math.cos(lat_rad) * _math.cos(decl_rad) * _math.cos(ha_rad))
                    altitude = _math.degrees(_math.asin(max(-1, min(1, sin_alt))))
                    # Azimuth
                    cos_az = ((_math.sin(decl_rad) - _math.sin(lat_rad) * sin_alt) /
                              (max(0.001, _math.cos(lat_rad) * _math.cos(_math.asin(max(-1, min(1, sin_alt)))))))
                    azimuth = _math.degrees(_math.acos(max(-1, min(1, cos_az))))
                    if hour_angle > 0:
                        azimuth = 360 - azimuth
                    return azimuth, max(0, altitude)
                except Exception:
                    return 180.0, 30.0

            # Helper to add a data point to all bin levels
            def _add_to_model(m: SolarProductionModel, hour_key: str, kwh: float) -> None:
                ghi = ghi_by_hour.get(hour_key, 0)
                if ghi <= 0:
                    return
                parts = hour_key.split("-")
                year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

                direct = direct_by_hour.get(hour_key, 0)
                diffuse = diffuse_by_hour.get(hour_key, 0)
                temp = temp_by_hour.get(hour_key, 15.0)
                azimuth, altitude = _sun_position(year, month, day, hour)

                if altitude <= 0:
                    return  # Sun below horizon

                rad_b = SolarProductionModel.radiation_to_bucket(ghi)
                clear_b = SolarProductionModel.clear_sky_to_bucket(direct, diffuse)
                az_b = SolarProductionModel.azimuth_to_bucket(azimuth)
                alt_b = SolarProductionModel.altitude_to_bucket(altitude)
                temp_b = SolarProductionModel.temp_to_bucket(temp)

                # 5D bin
                key5 = (rad_b, clear_b, az_b, alt_b, temp_b)
                if key5 not in m.bins:
                    m.bins[key5] = []
                m.bins[key5].append(kwh)

                # 3D fallback (rad, azimuth, altitude)
                key3 = (rad_b, az_b, alt_b)
                if key3 not in m.mid_bins:
                    m.mid_bins[key3] = []
                m.mid_bins[key3].append(kwh)

                # 2D fallback (rad, altitude)
                key2 = (rad_b, alt_b)
                if key2 not in m.simple_bins:
                    m.simple_bins[key2] = []
                m.simple_bins[key2].append(kwh)

            # PASS 1: Build rough model from all matched data
            model = SolarProductionModel()
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

            # PASS 2: Filter curtailed hours and rebuild
            model2 = SolarProductionModel()
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

                direct = direct_by_hour.get(hour_key, 0)
                diffuse = diffuse_by_hour.get(hour_key, 0)
                temp = temp_by_hour.get(hour_key, 15.0)
                azimuth, altitude = _sun_position(year, month, day, hour)

                soc = soc_by_hour.get(hour_key, 50.0)
                expected = model.predict(
                    ghi,
                    sun_azimuth=azimuth,
                    sun_altitude=altitude,
                    direct_radiation=direct,
                    diffuse_radiation=diffuse,
                    temperature=temp,
                )

                if soc >= 90 and expected > 0 and kwh < expected * 0.6:
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
                f"{len(model2.medians)} bins, "
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
                optionally "direct_radiation", "diffuse_radiation"

        Returns:
            Dict of date_str -> DailyForecast
        """
        if not self._production_model:
            return {}

        # Pre-compute max sun elevation per date
        from astral import LocationInfo
        from astral.sun import sun as astral_sun
        max_elev_cache: Dict[str, float] = {}

        daily_hourly: Dict[str, Dict[int, float]] = {}

        for entry in weather_hourly:
            time_str = entry.get("time", "")
            ghi = entry.get("shortwave_radiation", 0)
            direct = entry.get("direct_radiation", 0)
            diffuse = entry.get("diffuse_radiation", 0)
            temp = entry.get("temperature_2m", 15)

            if not time_str:
                continue

            try:
                dt = datetime.fromisoformat(time_str)
            except ValueError:
                continue

            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            # Compute sun position for this specific hour
            try:
                import math as _math
                day_of_year = dt.timetuple().tm_yday
                decl = 23.45 * _math.sin(_math.radians(360 / 365 * (day_of_year - 81)))
                hour_angle = (hour + 0.5 - 12) * 15
                lat_rad = _math.radians(self.latitude)
                decl_rad = _math.radians(decl)
                ha_rad = _math.radians(hour_angle)
                sin_alt = (_math.sin(lat_rad) * _math.sin(decl_rad) +
                           _math.cos(lat_rad) * _math.cos(decl_rad) * _math.cos(ha_rad))
                altitude = _math.degrees(_math.asin(max(-1, min(1, sin_alt))))
                cos_az = ((_math.sin(decl_rad) - _math.sin(lat_rad) * sin_alt) /
                          max(0.001, _math.cos(lat_rad) * _math.cos(_math.asin(max(-1, min(1, sin_alt))))))
                azimuth = _math.degrees(_math.acos(max(-1, min(1, cos_az))))
                if hour_angle > 0:
                    azimuth = 360 - azimuth
                altitude = max(0, altitude)
            except Exception:
                azimuth, altitude = 180.0, 30.0

            predicted_kwh = self._production_model.predict(
                ghi,
                sun_azimuth=azimuth,
                sun_altitude=altitude,
                direct_radiation=direct,
                diffuse_radiation=diffuse,
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
        # Performance ratio: accounts for inverter losses, wiring, temperature, etc.
        performance_ratio = 0.80

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

            # Simple model: production = GHI * total_kwp * performance_ratio / 1000
            # GHI in W/m² for 1 hour → Wh/m², divided by 1000 W/m² (STC reference)
            # gives fraction of peak output
            production_kwh = ghi * total_kwp * performance_ratio / 1000.0

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

    def build_consensus(self) -> Dict[str, DailyForecast]:
        """Combine all forecast sources into a consensus.

        Priority: learned model (most accurate) > forecast.solar API > weather GHI.
        When multiple sources available, average values that agree, use model
        when sources diverge (it's trained on actual installation data).
        """
        all_dates = (
            set(self._api_forecast.keys())
            | set(self._weather_forecast.keys())
            | set(self._model_forecast.keys())
        )
        result: Dict[str, DailyForecast] = {}

        for date_str in all_dates:
            model = self._model_forecast.get(date_str)
            api = self._api_forecast.get(date_str)
            weather = self._weather_forecast.get(date_str)

            # Collect all available sources for this date
            sources: List[DailyForecast] = []
            if model and model.total_kwh > 0:
                sources.append(model)
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
                    vals = [s.hourly.get(hour, 0) for s in sources if s.hourly.get(hour, 0) > 0]
                    if not vals:
                        hourly[hour] = 0
                    elif len(vals) == 1:
                        hourly[hour] = vals[0]
                    else:
                        # Model is first val if present
                        model_val = model.hourly.get(hour, 0)
                        if model_val > 0:
                            avg_others = sum(v for v in vals if v != model_val) / max(1, len(vals) - 1)
                            if avg_others > 0:
                                divergence = abs(model_val - avg_others) / ((model_val + avg_others) / 2)
                                if divergence <= 0.3:
                                    # Agreement: average all
                                    hourly[hour] = sum(vals) / len(vals)
                                else:
                                    # Divergence: trust model (trained on real data)
                                    hourly[hour] = model_val
                            else:
                                hourly[hour] = model_val
                        else:
                            hourly[hour] = sum(vals) / len(vals)

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
            days: Number of past days to compare (default 7)
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
                    if day not in actual_by_day or val > actual_by_day[day]:
                        actual_by_day[day] = val

            if not actual_by_day:
                return

            # Compare with stored forecasts (if we have any for those days)
            # Use the raw API forecast total (before confidence discount)
            ratios: list = []
            for day_str, actual_kwh in actual_by_day.items():
                forecast = self._api_forecast.get(day_str)
                if forecast and forecast.total_kwh > 1.0:
                    ratio = actual_kwh / forecast.total_kwh
                    # Clamp to reasonable range (0.3 - 1.5)
                    ratio = max(0.3, min(1.5, ratio))
                    ratios.append(ratio)
                    self.logger.debug(
                        f"Calibration {day_str}: actual={actual_kwh:.1f} kWh, "
                        f"forecast={forecast.total_kwh:.1f} kWh, ratio={ratio:.2f}"
                    )

            if len(ratios) < 3:
                # Not enough forecast history for reliable calibration.
                # Use actuals vs theoretical max as a conservative estimate.
                total_kwp = sum(a.kwp for a in self.arrays)
                if total_kwp > 0 and actual_by_day:
                    avg_actual = sum(actual_by_day.values()) / len(actual_by_day)
                    typical_max = total_kwp * 5  # ~5 kWh/kWp on a good day
                    estimated_confidence = min(1.0, avg_actual / typical_max)
                    # Floor at 0.65 — avg includes cloudy days but scheduling
                    # decisions should be optimistic enough for sunny ones
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
        - No forecast data at all (API rate-limited + no cache + no weather)
        - Total kWp > 5 but forecast < 5 kWh for a day (suspiciously low)
        """
        if not self._consensus:
            return False

        total_kwp = sum(a.kwp for a in self.arrays)
        for forecast in self._consensus.values():
            # A 13.5 kWp system should produce > 5 kWh on almost any day
            if total_kwp > 5 and forecast.total_kwh < 5:
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
