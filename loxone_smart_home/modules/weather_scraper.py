"""Weather scraper module - fetches weather data from multiple sources."""

import asyncio
import json
from collections import namedtuple
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from loxone_smart_home.config.settings import Settings
from loxone_smart_home.modules.base import BaseModule
from loxone_smart_home.utils.influxdb_client import SharedInfluxDBClient
from loxone_smart_home.utils.mqtt_client import SharedMQTTClient

# Named tuple for standardized hourly data
HourlyData = namedtuple(
    "HourlyData",
    [
        "temp",
        "precip",
        "wind",
        "wind_direction",
        "clouds",
        "rh",
        "pressure",
    ],
)


class WeatherScraper(BaseModule):
    """Weather scraper that fetches data from multiple weather APIs."""

    def __init__(
        self,
        mqtt_client: Optional[SharedMQTTClient],
        influxdb_client: Optional[SharedInfluxDBClient],
        settings: Settings,
    ) -> None:
        """Initialize the weather scraper."""
        super().__init__(
            name="WeatherScraper",
            mqtt_client=mqtt_client,
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Start the weather scraper."""
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run_periodic())
        self.logger.info("Weather scraper started")

    async def stop(self) -> None:
        """Stop the weather scraper."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        self.logger.info("Weather scraper stopped")

    async def _run_periodic(self) -> None:
        """Run periodic weather updates."""
        while self._running:
            try:
                await self.fetch_weather()
                await asyncio.sleep(self.settings.weather.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in weather scraper: {e}", exc_info=True)
                await asyncio.sleep(self.settings.weather.retry_delay)

    async def fetch_weather(self) -> None:
        """Fetch weather data from configured source."""
        if not self._session:
            self.logger.error("HTTP session not initialized")
            return

        # Determine which service to use from configuration
        service = self.settings.weather.weather_service

        try:
            if service == "openmeteo":
                data = await self._fetch_openmeteo()
            elif service == "aladin":
                data = await self._fetch_aladin()
            elif service == "openweathermap":
                data = await self._fetch_openweathermap()
            else:
                self.logger.error(f"Unknown weather service: {service}")
                return

            # Publish to MQTT
            if self.mqtt_client and data:
                mqtt_data = self._prepare_mqtt_data(data)
                await self.mqtt_client.publish(
                    self.settings.mqtt.topic_weather, json.dumps(mqtt_data)
                )
                self.logger.debug("Published weather data to MQTT")

            # Save to InfluxDB
            if self.influxdb_client and data:
                await self._save_to_influxdb(data)
                self.logger.debug("Saved weather data to InfluxDB")

        except Exception as e:
            self.logger.error(f"Failed to fetch weather data: {e}", exc_info=True)

    async def _fetch_openmeteo(self) -> Dict[str, Any]:
        """Fetch weather data from OpenMeteo API."""
        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        # Main weather data
        fields = (
            "temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,"
            "precipitation,rain,showers,snowfall,surface_pressure,cloudcover,"
            "cloudcover_low,cloudcover_mid,cloudcover_high,visibility,"
            "windspeed_10m,winddirection_10m,windgusts_10m,temperature_80m,"
            "shortwave_radiation,direct_radiation,diffuse_radiation,"
            "direct_normal_irradiance,terrestrial_radiation,"
            "shortwave_radiation_instant,direct_radiation_instant,"
            "diffuse_radiation_instant,direct_normal_irradiance_instant,"
            "terrestrial_radiation_instant"
        )
        daily_fields = (
            "sunrise,sunset,precipitation_sum,rain_sum,precipitation_hours,"
            "shortwave_radiation_sum"
        )

        url = self.settings.weather.openmeteo_url
        params = {
            "latitude": self.settings.weather.latitude,
            "longitude": self.settings.weather.longitude,
            "hourly": fields,
            "daily": daily_fields,
            "models": "best_match",
            "windspeed_unit": "ms",
            "timeformat": "unixtime",
            "timezone": "GMT",
        }

        async with self._session.get(url, params=params) as resp:
            js = await resp.json()

        # Air quality data
        air_fields = "pm10,pm2_5,ozone,aerosol_optical_depth,uv_index"
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            "latitude": self.settings.weather.latitude,
            "longitude": self.settings.weather.longitude,
            "hourly": air_fields,
            "timeformat": "unixtime",
            "timezone": "GMT",
        }

        async with self._session.get(air_url, params=air_params) as resp:
            js_air = await resp.json()

        # Merge air quality data
        js["hourly"].update(js_air["hourly"])

        # Process data
        hour_now = int(
            datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).timestamp()
        )

        # Build hourly data
        hourly_records: List[Dict[str, Any]] = []
        hourly_now: Dict[str, Any] = {}

        if "time" in js["hourly"]:
            for i, time_val in enumerate(js["hourly"]["time"]):
                record = {k: js["hourly"][k][i] for k in js["hourly"].keys()}
                if time_val >= hour_now:
                    hourly_records.append(record)
                if time_val == hour_now:
                    hourly_now = record

        day_now = int(
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )

        # Build daily data
        daily_records: List[Dict[str, Any]] = []
        today_data: Dict[str, Any] = {}

        if "time" in js["daily"]:
            for i, time_val in enumerate(js["daily"]["time"]):
                record = {k: js["daily"][k][i] for k in js["daily"].keys()}
                daily_records.append(record)
                if time_val == day_now:
                    today_data = record

        now = int(datetime.now(timezone.utc).replace(microsecond=0).timestamp())

        return {
            "source": "openmeteo",
            "timestamp": now,
            "now": hourly_now,
            "today": today_data,
            "hourly": hourly_records,
            "daily": daily_records,
        }

    async def _fetch_aladin(self) -> Dict[str, Any]:
        """Fetch weather data from Aladin API."""
        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        # Note: URL has changed from the original, using the base URL pattern
        url = f"{self.settings.weather.aladin_url_base}get_data.php"
        params = {
            "latitude": self.settings.weather.latitude,
            "longitude": self.settings.weather.longitude,
        }

        async with self._session.get(url, params=params) as resp:
            js = await resp.json()

        return {
            "source": "aladin",
            "timestamp": js["nowCasting"]["nowUtc"],
            "hourly": [
                HourlyData(*data)._asdict()
                for data in zip(
                    js["parameterValues"]["TEMPERATURE"],
                    js["parameterValues"]["PRECIPITATION_TOTAL"],
                    js["parameterValues"]["WIND_SPEED"],
                    js["parameterValues"]["WIND_DIRECTION"],
                    js["parameterValues"]["CLOUDS_TOTAL"],
                    js["parameterValues"]["HUMIDITY"],
                    js["parameterValues"]["PRESSURE"],
                )
            ],
        }

    async def _fetch_openweathermap(self) -> Dict[str, Any]:
        """Fetch weather data from OpenWeatherMap API."""
        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        if not self.settings.weather.openweathermap_api_key:
            raise ValueError("OpenWeatherMap API key not configured")

        url = self.settings.weather.openweathermap_url
        params = {
            "lat": self.settings.weather.latitude,
            "lon": self.settings.weather.longitude,
            "exclude": "minutely,daily,alerts",
            "units": "metric",
            "appid": self.settings.weather.openweathermap_api_key,
        }

        async with self._session.get(url, params=params) as resp:
            js = await resp.json()

        data = {
            "source": "openweathermap",
            "timestamp": js["hourly"][0]["dt"],
            "hourly": [
                HourlyData(
                    data["temp"],
                    data.get("rain", {}).get("1h", 0),
                    data["wind_speed"],
                    data["wind_deg"],
                    data["clouds"],
                    data["humidity"],
                    data["pressure"],
                )._asdict()
                for data in js["hourly"]
            ],
        }

        # Update first hour forecast to current weather
        data["hourly"][0] = HourlyData(
            js["current"]["temp"],
            js["current"].get("rain", {}).get("1h", 0),
            js["current"]["wind_speed"],
            js["current"]["wind_deg"],
            js["current"]["clouds"],
            js["current"]["humidity"],
            js["current"]["pressure"],
        )._asdict()

        return data

    def _prepare_mqtt_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare weather data for MQTT publishing."""
        mqtt_data = {
            "timestamp": raw_data["timestamp"],
            "source": raw_data["source"],
        }

        if raw_data["source"] == "openmeteo":
            mqtt_data["hourly"] = raw_data["hourly"]
            mqtt_data["daily"] = raw_data["daily"]
        else:
            mqtt_data["hourly"] = raw_data["hourly"]

        return mqtt_data

    async def _save_to_influxdb(self, data: Dict[str, Any]) -> None:
        """Save weather data to InfluxDB."""
        if not self.influxdb_client:
            return

        timestamp = data["timestamp"]
        source = data["source"]

        # Save current/now data
        if source == "openmeteo" and "now" in data and data["now"]:
            await self._write_influx_point(data["now"], timestamp, source, "hour")

        # Save daily data
        if source == "openmeteo" and "today" in data and data["today"]:
            await self._write_influx_point(data["today"], timestamp, source, "day")

        # For other sources, save first hourly as "now"
        if source in ["aladin", "openweathermap"] and data.get("hourly"):
            await self._write_influx_point(data["hourly"][0], timestamp, source, "hour")

    async def _write_influx_point(
        self, data: Dict[str, Any], timestamp: int, source: str, forecast_type: str
    ) -> None:
        """Write a single point to InfluxDB."""
        if not self.influxdb_client:
            return

        # Convert all values to float, skipping non-numeric values
        fields = {}
        for name, value in data.items():
            if name == "time":  # Skip timestamp field
                continue
            try:
                fields[name] = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                # Skip non-numeric fields
                self.logger.debug(f"Skipping non-numeric field {name}={value}")
                continue

        if not fields:
            self.logger.warning("No numeric fields to write to InfluxDB")
            return

        await self.influxdb_client.write_point(
            bucket=self.settings.influxdb.bucket_weather,
            measurement="weather_forecast",
            fields=fields,
            tags={
                "room": "outside",
                "type": forecast_type,
                "source": source,
            },
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
        )
