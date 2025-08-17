"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import aiohttp
from astral import LocationInfo
from astral.sun import sun

from config.settings import Settings
from modules.base import BaseModule
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient


PeriodType = Literal["battery_first", "grid_first", "load_first", "ac_charge", "export"]

# Mode precedence (highest → lowest). load_first is the fallback/default if none match.
MODE_PRECEDENCE: Tuple[PeriodType, ...] = ("battery_first", "grid_first", "load_first")

# Use a single end-of-day barrier everywhere
EOD_HHMM = "23:59"
EOD_HHMMSS = "23:59:55"
EOD_DTTIME = dt_time(23, 59, 55)

# Common time jitter constants for readability
MIDNIGHT_JITTER = "00:00:05"  # 5 seconds after midnight
MIDNIGHT_DISABLE_JITTER = "00:00:10"  # 10 seconds after midnight for disables


def _is_24(s: str) -> bool:
    """Check if time string represents 24:00 (end of day)."""
    return s in ("24:00", "24:00:00")


def _fmt_hhmm(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M")


def _fmt_hhmmss(t: Union[datetime, dt_time]) -> str:
    """Format datetime/time as HH:MM:SS string."""
    return (t if isinstance(t, dt_time) else t.time()).strftime("%H:%M:%S")


@dataclass(frozen=True)
class Period:
    """Represents a scheduled period with proper time types."""
    kind: PeriodType
    start: dt_time
    end: dt_time
    params: Optional[Dict[str, Any]] = None

    def contains_time(self, t: dt_time) -> bool:
        """Check if a time falls within this period, handling midnight wrap."""
        if self.start <= self.end:
            return self.start <= t < self.end
        else:
            return t >= self.start or t < self.end

    def to_string_tuple(self) -> Tuple[str, str, str]:
        """Convert to legacy string tuple format for logging."""
        return (self.kind, self.start.strftime("%H:%M"), self.end.strftime("%H:%M"))


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    def _select_primary_mode(self, modes: set[PeriodType]) -> PeriodType:
        """Select primary mode based on precedence rules."""
        for m in MODE_PRECEDENCE:
            if m in modes:
                return m
        return "load_first"  # Default mode

    def _validate_config(self) -> None:
        """Validate required config attributes and set defaults."""
        required = [
            "battery_first_topic",
            "grid_first_topic",
            "grid_first_stopsoc_topic",
            "grid_first_powerrate_topic",
            "ac_charge_topic",
            "export_enable_topic",
            "export_disable_topic",
            "schedule_hour",
            "schedule_minute",
            "export_price_threshold",
            "summer_price_threshold",
            "battery_charge_hours",
            "individual_cheapest_hours",
        ]
        missing = [k for k in required if not hasattr(self.config, k)]
        if missing:
            raise ValueError(f"Growatt config missing keys: {', '.join(missing)}")

        # Set sensible defaults for optional flags in a separate dict
        # (Pydantic models don't allow dynamic attribute assignment)
        self._optional_config = {
            "dst_merge_policy": getattr(self.config, "dst_merge_policy", "avg"),
            "suppress_export_during_ac_charge": getattr(
                self.config, "suppress_export_during_ac_charge", True
            ),
            "eur_czk_rate": getattr(self.config, "eur_czk_rate", 25.0),
            "temperature_avg_days": getattr(self.config, "temperature_avg_days", 3),
            "summer_temp_threshold": getattr(self.config, "summer_temp_threshold", 15.0),
            "simulation_mode": getattr(self.config, "simulation_mode", False),
        }

        # Validate and clamp schedule time
        if hasattr(self.config, "schedule_hour"):
            if not 0 <= self.config.schedule_hour <= 23:
                old_val = self.config.schedule_hour
                self.config.schedule_hour = max(0, min(23, self.config.schedule_hour))
                self.logger.warning(
                    f"Invalid schedule_hour, clamping to 0-23 "
                    f"(was {old_val} → now {self.config.schedule_hour})"
                )

        if hasattr(self.config, "schedule_minute"):
            if not 0 <= self.config.schedule_minute <= 59:
                old_val = self.config.schedule_minute
                self.config.schedule_minute = max(0, min(59, self.config.schedule_minute))
                self.logger.warning(
                    f"Invalid schedule_minute, clamping to 0-59 "
                    f"(was {old_val} → now {self.config.schedule_minute})"
                )

        # Validate battery_charge_hours
        if hasattr(self.config, "battery_charge_hours"):
            if not 0 <= self.config.battery_charge_hours <= 24:
                old_val = self.config.battery_charge_hours
                self.config.battery_charge_hours = max(
                    0, min(24, self.config.battery_charge_hours)
                )
                self.logger.warning(
                    f"Invalid battery_charge_hours, clamping to 0-24 "
                    f"(was {old_val} → now {self.config.battery_charge_hours})"
                )

        # Validate individual_cheapest_hours
        if hasattr(self.config, "individual_cheapest_hours"):
            if not 0 <= self.config.individual_cheapest_hours <= 24:
                old_val = self.config.individual_cheapest_hours
                self.config.individual_cheapest_hours = max(
                    0, min(24, self.config.individual_cheapest_hours)
                )
                self.logger.warning(
                    f"Invalid individual_cheapest_hours, clamping to 0-24 "
                    f"(was {old_val} → now {self.config.individual_cheapest_hours})"
                )

        # Validate price thresholds
        if hasattr(self.config, "summer_price_threshold"):
            if self.config.summer_price_threshold < 0:
                old_val = self.config.summer_price_threshold
                self.config.summer_price_threshold = max(0, self.config.summer_price_threshold)
                self.logger.warning(
                    f"Invalid summer_price_threshold, setting to 0 "
                    f"(was {old_val} → now {self.config.summer_price_threshold})"
                )

        if hasattr(self.config, "export_price_threshold"):
            if self.config.export_price_threshold < 0:
                old_val = self.config.export_price_threshold
                self.config.export_price_threshold = max(0, self.config.export_price_threshold)
                self.logger.warning(
                    f"Invalid export_price_threshold, setting to 0 "
                    f"(was {old_val} → now {self.config.export_price_threshold})"
                )

    def __init__(
        self,
        mqtt_client: AsyncMQTTClient,
        influxdb_client: AsyncInfluxDBClient,
        settings: Settings,
    ) -> None:
        """Initialize the Growatt controller."""
        super().__init__(
            name="GrowattController",
            service_name="GROWATT",
            mqtt_client=mqtt_client,
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self.config = settings.growatt

        # Initialize optional config dict before validation
        self._optional_config: Dict[str, Any] = {}

        # Validate config and set defaults
        self._validate_config()

        # Scheduled tasks
        self._scheduled_tasks: List[asyncio.Task[None]] = []
        self._daily_schedule_task: Optional[asyncio.Task[None]] = None

        # Track scheduled periods for startup state sync
        self._scheduled_periods: List[Period] = []

        # Local timezone (Prague/Czech Republic)
        self._local_tz = zoneinfo.ZoneInfo("Europe/Prague")

        # Location for sunrise/sunset calculations
        self._location = LocationInfo(
            name="Prague",
            region="Czech Republic",
            latitude=settings.weather.latitude,
            longitude=settings.weather.longitude,
            timezone="Europe/Prague"
        )

        # Season mode cache
        self._season_mode: Optional[str] = None
        self._season_mode_updated: Optional[datetime] = None

        # Running flag for daily loop
        self._running: bool = False

        # EUR/CZK exchange rate cache
        self._eur_czk_rate: Optional[float] = None
        self._eur_czk_rate_updated: Optional[datetime] = None

        # Energy price data cache
        self._current_prices: Dict[Tuple[str, str], float] = {}
        self._prices_date: Optional[str] = None
        self._prices_updated: Optional[datetime] = None

        # Track AC and Export states to prevent duplicate commands
        self._ac_enabled: Optional[bool] = None
        self._export_enabled: Optional[bool] = None

        # Track inverter clock drift
        self._clock_drift_seconds: float = 0

        # Track applied mode configurations to prevent flapping
        self._last_applied: Dict[str, Tuple[Any, ...]] = {}

        # Default scheduler function (will be set properly in _calculate_and_schedule_next_day)
        self._schedule_func = self._schedule_at_time

    def _get_local_now(self) -> datetime:
        """Get current time in local timezone."""
        return datetime.now(self._local_tz)

    def _parse_time_any(self, s: str) -> dt_time:
        """Accept HH:MM, HH:MM:SS, and '24:00' => 00:00 (next-day semantics)."""
        if _is_24(s):
            return dt_time(0, 0)
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                pass
        raise ValueError(f"Invalid time string: {s}")

    def _parse_hhmm(self, s: str) -> dt_time:
        """Parse HH:MM or HH:MM:SS string to time object."""
        return self._parse_time_any(s)

    def _normalize_end_time(self, s: str) -> str:
        """Only for device/scheduler emission. Collapse '24:00' to end-of-day barrier."""
        return EOD_HHMMSS if _is_24(s) else s

    def _normalize_for_schedule(self, s: str) -> str:
        """Normalize any time string for scheduling calls.

        Converts 24:00 to EOD_HHMMSS and ensures HH:MM:SS format.
        """
        if _is_24(s):
            return EOD_HHMMSS
        # Normalize HH:MM to HH:MM:SS for consistency
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(s, fmt).strftime("%H:%M:%S")
            except ValueError:
                pass
        # If we can't parse it, raise an error
        raise ValueError(f"Invalid time string for schedule: {s}")

    def _bump_time(self, hhmm: str, seconds: int) -> str:
        """Add seconds to a time string systematically.

        Args:
            hhmm: Time in HH:MM or HH:MM:SS format
            seconds: Number of seconds to add

        Returns:
            Time string in HH:MM:SS format
        """
        try:
            t = datetime.strptime(hhmm, "%H:%M:%S")
        except ValueError:
            t = datetime.strptime(hhmm, "%H:%M")
        t = (t + timedelta(seconds=seconds)).time()
        return t.strftime("%H:%M:%S")

    def _to_device_hhmm(self, s: str) -> str:
        """Convert time string to HH:MM format required by device timeslot endpoints.

        The firmware strictly requires HH:MM format (no seconds) for
        batteryfirst/set/timeslot and gridfirst/set/timeslot commands.
        """
        # Handle EOD sentinels and 24:00
        if s in ("24:00", "24:00:00", EOD_HHMMSS):
            return EOD_HHMM  # "23:59"
        # Truncate to HH:MM if longer
        return s[:5] if len(s) >= 5 else s

    async def _set_mode(self, mode: str, *args: Any) -> None:
        """Set the inverter mode (thin dispatcher to individual setters)."""
        if mode == "battery_first":
            await self._set_battery_first(*args)
        elif mode == "grid_first":
            await self._set_grid_first(*args)
        elif mode == "load_first":
            await self._set_load_first()
        else:
            self.logger.error(f"Unknown mode: {mode}")

    def _get_local_date_string(self, days_ahead: int = 1) -> str:
        """Get date string in local timezone for API calls."""
        local_date = self._get_local_now() + timedelta(days=days_ahead)
        return local_date.strftime("%Y-%m-%d")

    def _get_sunrise_time(self, days_ahead: int = 0) -> dt_time:
        """Get sunrise time for the specified date.

        Args:
            days_ahead: Number of days ahead (0 for today, 1 for tomorrow)

        Returns:
            Sunrise time as dt_time in local timezone, rounded to nearest 15 minutes
        """
        target_date = self._get_local_now().date() + timedelta(days=days_ahead)
        s = sun(self._location.observer, date=target_date, tzinfo=self._local_tz)
        sunrise = s['sunrise']

        # Round to nearest 15 minutes (not floor)
        total_minutes = sunrise.hour * 60 + sunrise.minute
        rounded_minutes = int((total_minutes + 7) // 15) * 15  # +7 for rounding to nearest
        hour, minute = divmod(rounded_minutes, 60)

        # Handle 24:00 edge case (should roll to 00:00 next day for display)
        if hour >= 24:
            hour -= 24

        sunrise = sunrise.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return sunrise.time()

    def _log_schedule_summary(self) -> None:
        """Log a detailed summary of the scheduled periods for the day."""
        if not self._scheduled_periods:
            self.logger.info("No periods scheduled for the day")
            return

        self.logger.info("═" * 80)
        self.logger.info("                        DAILY SCHEDULE SUMMARY")
        self.logger.info("═" * 80)

        # Add season mode and sunrise info
        if hasattr(self, '_season_mode') and self._season_mode:
            self.logger.info(f"Season Mode: {self._season_mode.upper()}")

        try:
            days_ahead = 1 if self._get_local_now().time() >= dt_time(23, 45) else 0
            sunrise = self._get_sunrise_time(days_ahead)
            self.logger.info(f"Sunrise Time: {sunrise.strftime('%H:%M')}")
        except Exception:
            pass

        self.logger.info("-" * 80)

        # Build comprehensive schedule view
        schedule_entries = []

        # Group periods by time
        time_slots: Dict[str, Dict[str, Any]] = {}
        for period in sorted(self._scheduled_periods, key=lambda x: x.start):
            start_str = period.start.strftime("%H:%M")
            # If the slot wraps (start > end) and end is 00:00, or ends at EOD, present as 24:00
            if ((period.start > period.end and period.end == dt_time(0, 0))
                    or period.end == EOD_DTTIME):
                end_str = "24:00"
            else:
                end_str = period.end.strftime("%H:%M")
            key = f"{start_str}-{end_str}"
            if key not in time_slots:
                time_slots[key] = {
                    "start": start_str,
                    "end": end_str,
                    "modes": set(),
                    "periods": [],  # Track actual period objects
                    "primary_mode": None
                }
            time_slots[key]["modes"].add(period.kind)
            time_slots[key]["periods"].append(period)

        # Determine primary mode using consistent precedence for all slots
        for key, slot in time_slots.items():
            slot["primary_mode"] = self._select_primary_mode(slot["modes"])

        # Create detailed schedule entries
        for time_key in sorted(time_slots.keys()):
            slot = time_slots[time_key]
            entry: Dict[str, Any] = {
                "time": f"{slot['start']} → {slot['end']}",
                "mode": slot["primary_mode"] or "Unknown",
                "details": []
            }

            # Add mode-specific details
            if slot["primary_mode"] == "grid_first":
                entry["mode"] = "GRID-FIRST"
                # Find the first grid_first period from this slot's contributing periods
                gf = next((p for p in slot["periods"] if p.kind == "grid_first" and p.params), None)
                if gf and gf.params:
                    stop_soc = gf.params.get("stop_soc", 20)
                    power_rate = gf.params.get("power_rate", 10)
                else:
                    stop_soc = 20
                    power_rate = 10
                entry["details"].append(f"StopSOC: {stop_soc}%")
                entry["details"].append(f"PowerRate: {power_rate}%")
                entry["details"].append("Priority: Sell to grid")
            elif slot["primary_mode"] == "battery_first":
                entry["mode"] = "BATTERY-FIRST"
                entry["details"].append("Priority: Charge battery")
            elif slot["primary_mode"] == "load_first":
                entry["mode"] = "LOAD-FIRST"
                entry["details"].append("Priority: Supply loads")

            # Add AC charging status
            if "ac_charge" in slot["modes"]:
                entry["details"].append("AC Charging: ENABLED")
            elif slot["primary_mode"] in ["battery_first", "grid_first"]:
                entry["details"].append("AC Charging: DISABLED")

            # Add export status
            if "export" in slot["modes"]:
                entry["details"].append("Export: ENABLED")
            else:
                entry["details"].append("Export: DISABLED")

            schedule_entries.append(entry)

        # Print schedule in readable format
        for entry in schedule_entries:
            self.logger.info(f"[{entry['time']}] {entry['mode']}")
            for detail in entry['details']:
                self.logger.info(f"  • {detail}")

        self.logger.info("═" * 80)

        # Add compact one-liner summary
        compact = []
        for entry in schedule_entries:
            mode = entry['mode'][:4].upper()  # First 4 chars
            details = []
            if "Export: ENABLED" in entry['details']:
                details.append("EXP")
            if "AC Charging: ENABLED" in entry['details']:
                details.append("AC")
            detail_str = "+".join(details) if details else ""
            if detail_str:
                compact.append(f"[{entry['time']} {mode}+{detail_str}]")
            else:
                compact.append(f"[{entry['time']} {mode}]")

        if compact:
            self.logger.info("Compact: " + " ".join(compact[:8]))  # First 8 periods
            if len(compact) > 8:
                self.logger.info("         " + " ".join(compact[8:]))  # Remaining periods

    async def _get_eur_czk_rate(self) -> float:
        """Get EUR to CZK exchange rate from Czech National Bank."""
        # Check cache first (refresh once per day)
        now = self._get_local_now()
        if (
            self._eur_czk_rate is not None
            and self._eur_czk_rate_updated is not None
            and (now - self._eur_czk_rate_updated).total_seconds() < 86400  # 24 hours
        ):
            return self._eur_czk_rate

        try:
            # Fetch current exchange rate from CNB
            url = ("https://www.cnb.cz/cs/financni-trhy/devizovy-trh/"
                   "kurzy-devizoveho-trhu/kurzy-devizoveho-trhu/denni_kurz.txt")
            self.logger.debug(f"Fetching EUR/CZK exchange rate from CNB: {url}")

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.warning(
                            f"Failed to fetch CNB exchange rate: HTTP {response.status}"
                        )
                        fallback = (
                            self._optional_config.get("eur_czk_rate", 25.0)
                        )
                        # Cache the fallback to avoid repeated API calls
                        self._eur_czk_rate = fallback
                        self._eur_czk_rate_updated = now
                        return fallback

                    # Read raw bytes and try different encodings
                    raw = await response.read()
                    try:
                        text = raw.decode("windows-1250")
                    except Exception:
                        text = raw.decode("utf-8", errors="replace")

            # Parse the text file - format: country|currency|quantity|code|rate
            # Example: EMU|euro|1|EUR|24,470
            for line in text.split('\n'):
                if '|EUR|' in line:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        # Convert Czech decimal format (comma) to Python float
                        rate_str = parts[4].replace(',', '.')
                        rate = float(rate_str)

                        # Guard against invalid rates
                        if rate <= 0:
                            raise ValueError(f"Invalid EUR/CZK rate from CNB: {rate}")

                        # Cache the rate
                        self._eur_czk_rate = rate
                        self._eur_czk_rate_updated = now

                        self.logger.info(f"Updated EUR/CZK exchange rate: {rate:.3f}")
                        return rate

            self.logger.warning("EUR not found in CNB exchange rate data")
            # Return default fallback (25 CZK per EUR is typical)
            fallback = self._optional_config.get("eur_czk_rate", 25.0)
            # Cache the fallback to avoid repeated API calls
            self._eur_czk_rate = fallback
            self._eur_czk_rate_updated = now
            return fallback

        except Exception as e:
            self.logger.error(f"Error fetching EUR/CZK exchange rate: {e}")
            # Return default fallback (25 CZK per EUR is typical)
            fallback = self._optional_config.get("eur_czk_rate", 25.0)
            # Cache the fallback to avoid repeated API calls on errors
            self._eur_czk_rate = fallback
            self._eur_czk_rate_updated = now
            return fallback

    async def _get_inverter_time(self) -> Optional[datetime]:
        """Query current time from the inverter via MQTT request/response.

        Protocol details (tested and working):
        - Request topic: energy/solar/command/datetime/get
        - Response topic: energy/solar/result
        - Response contains: {"value": "YYYY-MM-DD HH:MM:SS", "command": "datetime/get",
                             "success": true, "message": "..."}
        """
        if self._optional_config.get("simulation_mode", False):
            return self._get_local_now()

        try:
            assert self.mqtt_client is not None

            # Set up the request/response pattern
            request_topic = "energy/solar/command/datetime/get"
            # The inverter responds on a common result topic
            response_topic = "energy/solar/result"

            # Create a correlation ID for tracking this specific request
            import uuid
            correlation_id = f"datetime-{uuid.uuid4().hex[:8]}"

            # Create a future to wait for the response
            loop = asyncio.get_running_loop()
            response_future: asyncio.Future[datetime] = loop.create_future()

            # Handler for the response
            async def response_handler(_topic: str, payload: Any) -> None:
                try:
                    # Payload could be bytes or string
                    if isinstance(payload, bytes):
                        payload = payload.decode()
                    data = json.loads(payload)

                    # Check if this is our datetime response
                    if data.get("command") != "datetime/get":
                        return  # Not our response, ignore

                    # Check success
                    if not data.get("success", False):
                        error_msg = data.get("message", "Unknown error")
                        if not response_future.done():
                            response_future.set_exception(
                                ValueError(f"Inverter returned error: {error_msg}")
                            )
                        return

                    value = data.get("value")  # Format: "YYYY-MM-DD HH:MM:SS"

                    if value:
                        # Parse the datetime string
                        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

                        # Handle potential year issue: firmware may return 2-digit year
                        # If year < 100, it's likely years since 2000, so add 2000
                        if dt.year < 100:
                            dt = dt.replace(year=dt.year + 2000)
                            self.logger.debug(f"Adjusted 2-digit year to {dt.year}")

                        # Add timezone info
                        dt = dt.replace(tzinfo=self._local_tz)

                        if not response_future.done():
                            response_future.set_result(dt)
                            self.logger.debug(f"Received inverter time: {value}")
                    else:
                        if not response_future.done():
                            response_future.set_exception(ValueError("No value in response"))

                except Exception as e:
                    self.logger.error(f"Error parsing inverter time response: {e}")
                    if not response_future.done():
                        response_future.set_exception(e)

            # Subscribe to the result topic
            await self.mqtt_client.subscribe(response_topic, response_handler)

            try:
                # Send the request with correlation ID
                request_payload = {"correlationId": correlation_id}
                self.logger.debug(
                    f"Requesting inverter time via {request_topic} "
                    f"with correlationId: {correlation_id}"
                )
                await self.mqtt_client.publish(request_topic, json.dumps(request_payload))

                # Wait for response with timeout
                inverter_time = await asyncio.wait_for(response_future, timeout=5.0)
                return inverter_time

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Inverter time request timed out after 5 seconds. "
                    f"Expected response on {response_topic} with command=datetime/get"
                )
                return None
            finally:
                # Unsubscribe from result topic
                await self.mqtt_client.unsubscribe(response_topic)

        except Exception as e:
            self.logger.error(f"Failed to get inverter time: {e}")
            return None

    async def _sync_inverter_time(self) -> bool:
        """Synchronize inverter time with server time.

        Returns True if sync was successful or not needed.
        """
        if self._optional_config.get("simulation_mode", False):
            return True

        try:
            server_time = self._get_local_now()
            inverter_time = await self._get_inverter_time()

            if inverter_time is None:
                self.logger.warning("Could not get inverter time, assuming it's correct")
                self._clock_drift_seconds = 0
                return True

            # Calculate drift
            drift = (server_time - inverter_time).total_seconds()
            self._clock_drift_seconds = drift

            if abs(drift) > 30:  # More than 30 seconds drift
                self.logger.warning(
                    f"Inverter clock drift detected: {drift:.0f} seconds. "
                    f"Server: {server_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Inverter: {inverter_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # Optionally update inverter time
                if abs(drift) > 120:  # More than 2 minutes
                    self.logger.info("Updating inverter time to match server")
                    assert self.mqtt_client is not None
                    topic = "energy/solar/command/datetime/set"
                    payload = {"value": server_time.strftime("%Y-%m-%d %H:%M:%S")}
                    await self.mqtt_client.publish(topic, json.dumps(payload))
                    self._clock_drift_seconds = 0
                    return True
            else:
                self.logger.debug(f"Inverter time is in sync (drift: {drift:.0f}s)")

            return True

        except Exception as e:
            self.logger.error(f"Failed to sync inverter time: {e}", exc_info=True)
            self._clock_drift_seconds = 0
            return False

    def _ensure_future_start(
        self,
        start_str: str,
        stop_str: str,
        min_future_minutes: int = 1,
        preserve_duration: bool = False,
    ) -> Tuple[str, str]:
        """Ensure start time is at least min_future_minutes in the future.

        If preserve_duration=True, keep the original slot duration when bumping start.
        Returns adjusted (start, stop) in HH:MM format.
        """
        now = self._get_local_now()

        # Parse times (support HH:MM or HH:MM:SS)
        def _parse_any(s: str) -> datetime:
            if s in ("24:00", "24:00:00"):
                s = "00:00:00"
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    t = datetime.strptime(s, fmt).time()
                    return datetime.combine(now.date(), t, self._local_tz)
                except ValueError:
                    continue
            raise ValueError(f"Invalid time string: {s}")

        start_dt = _parse_any(start_str)
        stop_dt = _parse_any(stop_str)

        # Compute original duration (handle wrap)
        if stop_dt <= start_dt:
            stop_dt += timedelta(days=1)
        duration = stop_dt - start_dt

        # Round up to the next whole minute to avoid edge cases
        min_start = now.replace(microsecond=0, second=0) + timedelta(minutes=min_future_minutes)
        if start_dt <= min_start:
            adjusted_start_dt = min_start
            if preserve_duration:
                adjusted_stop_dt = adjusted_start_dt + duration
            else:
                adjusted_stop_dt = stop_dt  # keep original stop

            # Normalize back into today's clock (mod 24h)
            def _fmt(dt: datetime) -> str:
                t = dt.time()
                return t.strftime("%H:%M")

            msg = (
                f"Adjusting start time from {start_str} to {_fmt(adjusted_start_dt)} "
                f"(+{min_future_minutes}min from now)"
            )
            if preserve_duration:
                msg += f", stop adjusted from {stop_str} to {_fmt(adjusted_stop_dt)}"
            self.logger.debug(msg)
            return _fmt(adjusted_start_dt), _fmt(adjusted_stop_dt)

        # No change
        return (
            start_dt.time().strftime("%H:%M"),
            stop_dt.time().strftime("%H:%M"),
        )

    async def _get_season_mode(self) -> str:
        """Determine season mode based on 3-day temperature average."""
        # Check cache first (refresh once per day)
        now = self._get_local_now()
        if (
            self._season_mode is not None
            and self._season_mode_updated is not None
            and (now - self._season_mode_updated).total_seconds() < 86400  # 24 hours
        ):
            return self._season_mode

        try:
            # Query InfluxDB for temperature data
            if self.influxdb_client:
                # Build Flux query for last 3 days of temperature
                days = self._optional_config.get("temperature_avg_days", 3)
                query = f'''
                    from(bucket: "{self.settings.influxdb.bucket_loxone}")
                    |> range(start: -{days}d)
                    |> filter(fn: (r) => r._measurement == "temperature")
                    |> filter(fn: (r) => r._field == "temperature_outside")
                    |> mean()
                '''

                result = await self.influxdb_client.query(query)

                # Parse result to get average temperature
                avg_temp = None
                for table in result:
                    for record in table.records:
                        avg_temp = record.get_value()
                        break
                    if avg_temp is not None:
                        break

                if avg_temp is not None:
                    # Determine season based on temperature threshold
                    temp_threshold = self._optional_config.get("summer_temp_threshold", 15.0)
                    season = "summer" if avg_temp > temp_threshold else "winter"
                    self._season_mode = season
                    self._season_mode_updated = now
                    self.logger.info(
                        f"Season mode determined: {self._season_mode} "
                        f"(3-day avg temp: {avg_temp:.1f}°C, "
                        f"threshold: {self._optional_config.get('summer_temp_threshold', 15.0)}°C)"
                    )
                    return self._season_mode
                else:
                    self.logger.warning(
                        "No temperature data available; "
                        "keeping previous season or default to winter"
                    )
                    # Keep previous season if we had one, otherwise default to winter
                    if self._season_mode is None:
                        self._season_mode = "winter"
                        self._season_mode_updated = now
                    return self._season_mode

        except Exception as e:
            self.logger.error(f"Failed to determine season mode: {e}", exc_info=True)

        # Default to winter mode if unable to determine
        self._season_mode = "winter"
        self._season_mode_updated = now
        return self._season_mode

    async def _log_price_table(
        self, hourly_prices: Dict[Tuple[str, str], float], date: str, eur_czk_rate: float
    ) -> None:
        """Log hourly prices in a nice table format."""
        # Use provided rate for consistency

        self.logger.info(f"Energy prices for {date}:")
        self.logger.info("┌──────────┬────────────┬──────────────┐")
        self.logger.info("│   Hour   │ EUR/MWh    │   CZK/kWh    │")
        self.logger.info("├──────────┼────────────┼──────────────┤")

        # Sort hours by start time for proper display
        sorted_hours = sorted(hourly_prices.items(), key=lambda x: x[0][0])

        for (start_hour, end_hour), price_eur_mwh in sorted_hours:
            price_czk_kwh = price_eur_mwh * eur_czk_rate / 1000
            self.logger.info(
                f"│ {start_hour}-{end_hour} │ {price_eur_mwh:8.2f}   │   {price_czk_kwh:7.3f}    │"
            )

        self.logger.info("└──────────┴────────────┴──────────────┘")

    async def start(self) -> None:
        """Start the Growatt controller."""
        self._running = True  # Set the flag so daily loop runs

        # Sync inverter time on startup
        self.logger.info("Checking inverter time synchronization...")
        await self._sync_inverter_time()

        await self._schedule_daily_calculation()
        self.logger.info("Growatt controller started")

        # Check and apply current state based on active time periods
        # This happens after scheduling so we have the periods available
        await self._apply_current_state()

    async def stop(self) -> None:
        """Stop the Growatt controller."""
        self._running = False  # Clear the flag to stop daily loop

        # Cancel all scheduled tasks
        tasks_to_cancel = []
        for task in self._scheduled_tasks:
            if task and not task.done():
                task.cancel()
                tasks_to_cancel.append(task)

        if self._daily_schedule_task and not self._daily_schedule_task.done():
            self._daily_schedule_task.cancel()
            tasks_to_cancel.append(self._daily_schedule_task)

        # Wait for all tasks to complete cancellation with timeout
        if tasks_to_cancel:
            try:
                await asyncio.wait(tasks_to_cancel, timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks did not cancel within timeout")
            except Exception as e:
                self.logger.debug(f"Task cancellation exception (expected): {e}")

        # Clear task lists
        self._scheduled_tasks.clear()
        self._daily_schedule_task = None

        # Land fully neutral - disable all modes to ensure predictable state
        try:
            await self._disable_battery_first()
            await asyncio.sleep(0.5)  # Delay between shutdown commands
        except Exception as e:
            self.logger.error(f"Error disabling battery first on shutdown: {e}")

        try:
            await self._disable_grid_first()
            await asyncio.sleep(0.5)  # Delay between shutdown commands
        except Exception as e:
            self.logger.error(f"Error disabling grid first on shutdown: {e}")

        try:
            await self._disable_export()
            await asyncio.sleep(0.2)  # Small delay before AC disable
        except Exception as e:
            self.logger.error(f"Error disabling export on shutdown: {e}")

        try:
            await self._disable_ac_charge()
        except Exception as e:
            self.logger.error(f"Error disabling AC charge on shutdown: {e}")

        self.logger.info("Growatt controller stopped")

    async def _apply_current_state(self) -> None:
        """Apply the appropriate state based on current time and scheduled periods."""
        now = self._get_local_now()
        now_t = now.time()
        self.logger.info(
            f"Checking current state at {now.strftime('%H:%M')} for immediate application..."
        )

        if not self._scheduled_periods:
            self.logger.info("No scheduled periods found yet, skipping startup state sync")
            return

        # Collect all active period types at current time
        active: set[PeriodType] = set()
        for period in self._scheduled_periods:
            if period.contains_time(now_t):
                active.add(period.kind)
                self.logger.info(
                    f"Currently in {period.kind} period "
                    f"({period.start.strftime('%H:%M')}-{period.end.strftime('%H:%M')})"
                )

        # Decide desired state with deterministic precedence
        primary = self._select_primary_mode(active)
        desired_mode: Tuple[Any, ...]
        if primary == "grid_first":
            # Try to get params from an active scheduled grid-first period
            stop_soc = 20  # Default
            power_rate = 10  # Default
            for p in self._scheduled_periods:
                if p.kind == "grid_first" and p.contains_time(now_t) and p.params:
                    stop_soc = int(p.params.get("stop_soc", stop_soc))
                    power_rate = int(p.params.get("power_rate", power_rate))
                    break
            desired_mode = ("grid_first", stop_soc, power_rate)
        elif primary == "battery_first":
            desired_mode = ("battery_first",)
        else:
            desired_mode = ("load_first",)  # Default mode

        # Export and AC charging flags
        want_export = "export" in active
        want_ac = "ac_charge" in active

        # Apply mode
        self.logger.info(f"Applying mode: {desired_mode[0]}")
        # Use narrower window (current hour to next hour) to avoid overwriting whole day
        current_hour = now.strftime("%H:00")
        end = now + timedelta(hours=1)
        # Handle midnight crossing safely
        if end.date() != now.date():
            next_hour = EOD_HHMMSS  # Stop at end of day
        else:
            next_hour = end.strftime("%H:00")

        if desired_mode[0] == "battery_first":
            # Use _emit_device_window to handle midnight wrap properly
            # Don't preserve duration to avoid spillover past the hour
            await self._emit_device_window(
                self._set_battery_first, current_hour, next_hour, preserve_when_same_day=False
            )
        elif desired_mode[0] == "grid_first":
            # Use _emit_device_window to handle midnight wrap properly
            # Don't preserve duration to avoid spillover past the hour
            await self._emit_device_window(
                self._set_grid_first, current_hour, next_hour,
                desired_mode[1], desired_mode[2],
                preserve_when_same_day=False,
            )
        else:
            await self._set_load_first()

        # Small delay before AC charging control
        await asyncio.sleep(0.5)

        # AC charging (honor summer mode rule)
        season = await self._get_season_mode()
        if season == "summer" and want_ac:
            self.logger.info("Summer mode active, disabling AC charging despite schedule")
            want_ac = False

        if want_ac:
            await self._enable_ac_charge()
        else:
            await self._disable_ac_charge()

        # Small delay before export control
        await asyncio.sleep(0.5)

        # Export control
        if want_export:
            await self._enable_export()
        else:
            await self._disable_export()

        self.logger.info("Startup state synchronization complete")

    async def _fetch_dam_energy_prices(
        self, date: Optional[str] = None
    ) -> Dict[Tuple[str, str], float]:
        """Fetch energy prices from OTE DAM API (DST-safe)."""
        if date is None:
            date = self._get_local_date_string(days_ahead=1)

        url = (
            "https://www.ote-cr.cz/en/short-term-markets/electricity/"
            f"day-ahead-market/@@chart-data?report_date={date}"
        )
        self.logger.info(f"Fetching DAM energy prices for {date} from: {url}")

        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers={"User-Agent": "growatt-controller/1.0"}
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch DAM prices: HTTP {response.status}")
                        return {}

                    data = await response.json()

            # Parse the OTE API response (DST-safe)
            # dataLine[0] contains CZK/MWh prices (despite axis label saying EUR)
            # dataLine[1] contains actual EUR/MWh prices
            hourly_prices: Dict[Tuple[str, str], float] = {}
            if data.get("data", {}).get("dataLine"):
                lines = data["data"]["dataLine"]

                # Robustly identify EUR line (some days only have one line or order flips)
                def is_eur_line(line: Dict[str, Any]) -> bool:
                    """Check if this line contains EUR prices based on metadata."""
                    name = (line.get("name") or "").lower()
                    tooltip = (line.get("tooltip") or "").lower()
                    # Look for EUR indicators in metadata
                    return "eur/mwh" in name or "eur" in name or "eur/mwh" in tooltip

                # Find EUR line, fall back to last line if not found
                eur_line = next((ln for ln in lines if is_eur_line(ln)), None)
                if eur_line:
                    price_data = eur_line.get("point", [])
                    self.logger.debug("Using identified EUR line for prices")
                elif len(lines) >= 2:
                    # Default to second line (index 1) which is usually EUR
                    price_data = lines[1].get("point", [])
                    self.logger.debug("Using second line (usual EUR position) for prices")
                else:
                    # Only one line available
                    price_data = lines[0].get("point", [])
                    self.logger.warning("Only one dataLine found, prices might be in CZK")

                # DST-safe parsing using sequential datetime
                date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                base_dt = datetime.combine(date_obj, dt_time(0, 0), self._local_tz)

                # Sort by x value if present to ensure correct order
                def _pkey(p: Dict[str, Any]) -> float:
                    try:
                        return float(p.get("x", 0))
                    except Exception:
                        return 0.0
                price_data = sorted(price_data, key=_pkey)

                # DST merge policy for 25-hour days (fall back)
                merge_policy = self._optional_config.get("dst_merge_policy", "avg")

                def _merge_duplicate(existing: float, new: float) -> float:
                    """Merge duplicate hour prices during DST transitions."""
                    if merge_policy == "min":
                        return min(existing, new)
                    elif merge_policy == "max":
                        return max(existing, new)
                    elif merge_policy == "first":
                        return existing
                    elif merge_policy == "second":
                        return new
                    else:  # avg (default)
                        return (existing + new) / 2.0

                for i, point in enumerate(price_data):
                    try:
                        price = float(point["y"])
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid price value at index {i}: {point.get('y')}")
                        continue

                    # Skip non-finite prices (but keep negative prices - they're valid!)
                    if not (price == price) or price in (float("inf"), float("-inf")):
                        self.logger.warning(f"Skipping non-finite price at index {i}: {price}")
                        continue

                    # Use index for sequential time calculation (handles DST transitions)
                    start_dt = base_dt + timedelta(hours=i)
                    stop_dt = start_dt + timedelta(hours=1)

                    # Format times, handling midnight wrap
                    start_time = start_dt.strftime("%H:%M")
                    if stop_dt.date() != start_dt.date():
                        stop_time = "24:00"  # Next day
                    else:
                        stop_time = stop_dt.strftime("%H:%M")

                    key = (start_time, stop_time)
                    if key in hourly_prices:
                        # DST duplicate hour - merge according to policy
                        self.logger.debug(
                            f"DST duplicate hour {key} detected, merging with {merge_policy}"
                        )
                        hourly_prices[key] = _merge_duplicate(hourly_prices[key], price)
                    else:
                        hourly_prices[key] = price

            self.logger.info(f"Successfully fetched {len(hourly_prices)} DAM price points")
            return hourly_prices

        except Exception as e:
            self.logger.error(f"Error fetching DAM prices: {e}", exc_info=True)
            return {}

    def _generate_mock_prices(self, date: str) -> Dict[Tuple[str, str], float]:
        """Generate mock energy prices for testing when OTE data unavailable.
        
        Creates a realistic price pattern with:
        - Lower prices at night (2-6 AM)
        - Higher prices during peak hours (8-10 AM, 5-8 PM)
        - Medium prices during day
        """
        import random
        random.seed(date)  # Consistent prices for same date
        
        hourly_prices: Dict[Tuple[str, str], float] = {}
        base_price = 80.0  # EUR/MWh
        
        for hour in range(24):
            start = f"{hour:02d}:00"
            end = f"{(hour + 1) % 24:02d}:00" if hour < 23 else "24:00"
            
            # Night valley (2-6 AM) - cheapest
            if 2 <= hour < 6:
                price = base_price * random.uniform(0.5, 0.7)
            # Morning peak (8-10 AM) - expensive
            elif 8 <= hour < 10:
                price = base_price * random.uniform(1.3, 1.5)
            # Evening peak (17-20 PM) - most expensive
            elif 17 <= hour < 20:
                price = base_price * random.uniform(1.4, 1.6)
            # Night (22-2 AM) - cheap
            elif hour >= 22 or hour < 2:
                price = base_price * random.uniform(0.6, 0.8)
            # Day hours - medium
            else:
                price = base_price * random.uniform(0.9, 1.1)
                
            hourly_prices[(start, end)] = round(price, 2)
            
        self.logger.warning(f"Using mock prices for {date} (OTE data unavailable)")
        return hourly_prices

    def _find_cheapest_consecutive_hours(
        self, prices: Dict[Tuple[str, str], float], x: int = 2
    ) -> List[Tuple[str, str, float]]:
        """Find X consecutive hours with lowest total price."""
        if not prices:
            return []

        # Sort by start time "HH:MM"
        intervals = sorted(prices.keys(), key=lambda t: t[0])
        num_intervals = len(intervals)

        if num_intervals == 0:
            return []

        # Determine interval duration
        if num_intervals >= 2:
            start0 = datetime.strptime(intervals[0][0], "%H:%M")
            start1 = datetime.strptime(intervals[1][0], "%H:%M")
            interval_duration = int((start1 - start0).total_seconds() // 60)
        else:
            interval_duration = 60  # assume hourly if only one interval present

        if interval_duration not in (15, 60):
            self.logger.warning(f"Unknown interval duration: {interval_duration} minutes")
            return []

        intervals_per_hour = 60 // interval_duration
        intervals_needed = max(1, x * intervals_per_hour)  # Allow x=1

        if num_intervals < intervals_needed:
            return []

        cheapest_window: List[Tuple[str, str, float]] = []
        min_price_sum = float("inf")

        for i in range(num_intervals - intervals_needed + 1):
            window = intervals[i:i + intervals_needed]
            price_sum = sum(prices[k] for k in window)

            if price_sum < min_price_sum:
                min_price_sum = price_sum
                cheapest_window = [(s, e, prices[(s, e)]) for (s, e) in window]

        return cheapest_window

    def _find_n_cheapest_hours(
        self, prices: Dict[Tuple[str, str], float], n: int = 8
    ) -> List[Tuple[str, str, float]]:
        """Find N cheapest individual hours."""
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()], key=lambda x: x[2]
        )
        return sorted_prices[:n]

    def _categorize_prices_into_quadrants(
        self, prices: Dict[Tuple[str, str], float]
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Categorize prices into four quadrants."""
        price_values = list(prices.values())
        if not price_values:
            return {"Cheapest": [], "Cheap": [], "Expensive": [], "Most Expensive": []}

        min_price = min(price_values)
        max_price = max(price_values)

        # Handle flat pricing (all prices equal)
        if max_price - min_price < 1e-9:
            return {
                "Cheapest": [(s, e, p) for (s, e), p in prices.items()],
                "Cheap": [],
                "Expensive": [],
                "Most Expensive": []
            }

        interval = (max_price - min_price) / 4

        quadrants: Dict[str, List[Tuple[str, str, float]]] = {
            "Cheapest": [],
            "Cheap": [],
            "Expensive": [],
            "Most Expensive": [],
        }

        for (start, stop), price in prices.items():
            if price < min_price + interval:
                quadrants["Cheapest"].append((start, stop, price))
            elif price < min_price + 2 * interval:
                quadrants["Cheap"].append((start, stop, price))
            elif price < min_price + 3 * interval:
                quadrants["Expensive"].append((start, stop, price))
            else:
                quadrants["Most Expensive"].append((start, stop, price))

        return quadrants

    def _group_contiguous_hours(self, hours: List[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
        """Group contiguous hours into continuous ranges (with price similarity)."""
        if not hours:
            return []

        def t(s: str) -> datetime:
            # Keep logic in HH:MM; treat "24:00" as exclusive end-of-day for comparisons
            return datetime.strptime("00:00" if s == "24:00" else s, "%H:%M")

        sorted_hours = sorted(hours, key=lambda x: t(x[0]))

        groups: List[Tuple[str, str]] = []
        gs, ge, gp = sorted_hours[0]

        for s, e, p in sorted_hours[1:]:
            # contiguous if next start equals current end AND prices within 20%
            if t(s) == t(ge) and (abs(p - gp) < abs(gp * 0.2) if gp != 0 else p == 0):
                ge = e
            else:
                groups.append((gs, ge))
                gs, ge, gp = s, e, p

        groups.append((gs, ge))
        return groups

    def _group_contiguous_hours_simple(
        self, hours: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str]]:
        """Group contiguous hours into continuous ranges without price similarity check."""
        if not hours:
            return []

        # Sort by start time using proper time comparison
        def parse_time(s: str) -> datetime:
            return datetime.strptime("00:00" if s == "24:00" else s, "%H:%M")

        sorted_hours = sorted(hours, key=lambda x: parse_time(x[0]))

        groups: List[Tuple[str, str]] = []
        group_start, group_end = sorted_hours[0][0], sorted_hours[0][1]

        for start, end, _price in sorted_hours[1:]:
            # Compare times properly - contiguous if next start equals current end
            # Don't normalize 24:00 here - keep it for proper boundary handling
            if parse_time(start) == parse_time(group_end if group_end != "24:00" else "00:00"):
                group_end = end
            else:
                # Only append group, don't normalize yet
                groups.append((group_start, group_end))
                group_start, group_end = start, end

        groups.append((group_start, group_end))
        return groups

    async def _ensure_exclusive(self, primary: PeriodType) -> None:
        """Ensure modes are mutually exclusive at the device level."""
        if primary == "battery_first":
            await self._disable_grid_first()
            await asyncio.sleep(0.5)
        elif primary == "grid_first":
            await self._disable_battery_first()
            await asyncio.sleep(0.5)

    async def _set_battery_first(
        self, start_hour: str, stop_hour: str, stop_soc: int = 90, power_rate: int = 100,
        *, preserve_duration: bool = True, pre_scheduled: bool = False
    ) -> None:
        """Set battery-first mode for specified time window.

        Battery-first mode prioritizes charging battery from grid/solar.
        stop_soc: Battery level to stop charging at (default 90%)
        power_rate: Charge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        """
        # Only adjust time if not pre-scheduled (pre-scheduled commands are sent 1 min early)
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            # Preserve duration when bumping start so the window semantics stay intact
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour, preserve_duration=preserve_duration
            )

        # Guard against collapsed windows (start == stop)
        if adjusted_start == adjusted_stop:
            self.logger.debug(
                f"Battery-first window collapsed after bump ({adjusted_start}=={adjusted_stop}); "
                "skipping."
            )
            return

        # Validate and clamp parameters to safe ranges
        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        # Per-mode idempotence keyed by APPLIED params
        sig = ("battery_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        if self._last_applied.get("battery_first") == sig:
            self.logger.debug(
                f"Battery-first {adjusted_start}-{adjusted_stop} already applied, skipping"
            )
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%) at {current_time}"
            )
            self._last_applied["battery_first"] = sig
            return

        assert self.mqtt_client is not None

        # Ensure exclusive mode before setting
        await self._ensure_exclusive("battery_first")

        # Optional: Set battery-first parameters if different from defaults
        # These are rarely used as battery-first typically charges at max rate to max SOC
        if stop_soc != 90:
            # Set stop SOC (battery level to stop charging)
            stopsoc_topic = "energy/solar/command/batteryfirst/set/stopsoc"
            stopsoc_payload = {"value": stop_soc}
            self.logger.debug(f"Setting battery-first stopSOC to {stop_soc}%")
            await self.mqtt_client.publish(stopsoc_topic, json.dumps(stopsoc_payload))
            await asyncio.sleep(0.5)

        if power_rate != 100:
            # Set power rate (charge rate)
            powerrate_topic = "energy/solar/command/batteryfirst/set/powerrate"
            powerrate_payload = {"value": power_rate}
            self.logger.debug(f"Setting battery-first powerRate to {power_rate}%")
            await self.mqtt_client.publish(powerrate_topic, json.dumps(powerrate_payload))
            await asyncio.sleep(0.5)

        # Convert to HH:MM format required by device
        # Note: We use slot 1 for battery-first mode
        start_dev = self._to_device_hhmm(adjusted_start)
        stop_dev = self._to_device_hhmm(adjusted_stop)
        payload = {"start": start_dev, "stop": stop_dev, "enabled": True, "slot": 1}

        self.logger.debug(f"Enabling battery-first mode for {adjusted_start}-{adjusted_stop}")
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
            f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
            f"powerRate={power_rate}%) at {current_time} → "
            f"Topic: {self.config.battery_first_topic}"
        )
        self._last_applied["battery_first"] = sig

    async def _set_ac_charge(self, enabled: bool) -> None:
        """Set AC charging state (unified setter)."""
        if self._ac_enabled == enabled:
            return  # Already in desired state

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING {state} (simulated at {current_time})")
            self._ac_enabled = enabled
            return

        payload = {"value": 1 if enabled else 0}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        state = "ENABLED" if enabled else "DISABLED"
        self.logger.info(
            f"⚡ AC CHARGING {state} at {current_time} → Topic: {self.config.ac_charge_topic}"
        )
        self._ac_enabled = enabled

    async def _enable_ac_charge(self) -> None:
        """Enable AC charging during battery-first mode."""
        await self._set_ac_charge(True)

    async def _disable_ac_charge(self) -> None:
        """Disable AC charging."""
        await self._set_ac_charge(False)

    async def _disable_battery_first(self) -> None:
        """Disable battery-first mode."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE DISABLED (simulated at {current_time})"
            )
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.battery_first_topic}"
        )

    async def _set_export(self, enabled: bool) -> None:
        """Set export state (unified setter handling edge-triggered topics).

        NOTE: Export control is NOT handled by the Growatt inverter firmware.
        These topics are consumed by external systems:
        - Smart meter relay control
        - Loxone home automation
        - DSO (Distribution System Operator) limiter
        - Other energy management systems

        The edge-triggered design means:
        - export/enable topic triggers export ON
        - export/disable topic triggers export OFF
        - Both use {"value": true} as payload (topic determines action)
        """
        if self._export_enabled == enabled:
            return  # Already in desired state

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            emoji = '⬆️' if enabled else '⬇️'
            self.logger.info(
                f"{emoji} [SIMULATE] EXPORT {state} (simulated at {current_time})"
            )
            self._export_enabled = enabled
            return

        # Edge-triggered topics: enable and disable use different topics
        # Both use {"value": True} as the payload
        payload = {"value": True}
        assert self.mqtt_client is not None

        if enabled:
            topic = self.config.export_enable_topic
            await self.mqtt_client.publish(topic, json.dumps(payload))
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬆️ EXPORT ENABLED at {current_time} → Topic: {topic}")
        else:
            topic = self.config.export_disable_topic
            await self.mqtt_client.publish(topic, json.dumps(payload))
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬇️ EXPORT DISABLED at {current_time} → Topic: {topic}")

        self._export_enabled = enabled

    async def _enable_export(self) -> None:
        """Enable electricity export to grid."""
        await self._set_export(True)

    async def _disable_export(self) -> None:
        """Disable electricity export to grid."""
        await self._set_export(False)

    async def _set_grid_first(
        self, start_hour: str, stop_hour: str, stop_soc: int = 20, power_rate: int = 10,
        *, preserve_duration: bool = True, pre_scheduled: bool = False
    ) -> None:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc: Battery level to stop discharging at (default 20%)
        power_rate: Discharge rate in % (default 10%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        """
        # Only adjust time if not pre-scheduled (pre-scheduled commands are sent 1 min early)
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            # Ensure start time is in the future for inverter to trigger
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour, preserve_duration=preserve_duration
            )

        # Guard against collapsed windows (start == stop)
        if adjusted_start == adjusted_stop:
            self.logger.debug(
                f"Grid-first window collapsed after bump ({adjusted_start}=={adjusted_stop}); "
                "skipping."
            )
            return

        # Validate and clamp parameters to safe ranges
        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        # Per-mode idempotence keyed by APPLIED params
        sig = ("grid_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        if self._last_applied.get("grid_first") == sig:
            self.logger.debug(
                f"Grid-first {adjusted_start}-{adjusted_stop} "
                f"(stopSOC={stop_soc}, rate={power_rate}) already applied, skipping"
            )
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔌 [SIMULATE] GRID-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%, simulated at {current_time})"
            )
            self._last_applied["grid_first"] = sig
            return

        assert self.mqtt_client is not None

        # Ensure exclusive mode before setting
        await self._ensure_exclusive("grid_first")

        # First set the parameters before enabling the mode
        # Set stop SOC (battery level to stop discharging)
        stopsoc_payload = {"value": stop_soc}
        self.logger.debug(f"Setting grid-first stopSOC to {stop_soc}%")
        await self.mqtt_client.publish(
            self.config.grid_first_stopsoc_topic, json.dumps(stopsoc_payload)
        )

        # Small delay between commands
        await asyncio.sleep(0.5)

        # Set power rate (discharge rate)
        powerrate_payload = {"value": power_rate}
        self.logger.debug(f"Setting grid-first powerRate to {power_rate}%")
        await self.mqtt_client.publish(
            self.config.grid_first_powerrate_topic, json.dumps(powerrate_payload)
        )

        # Small delay before enabling the mode
        await asyncio.sleep(0.5)

        # Finally set the time slot to enable the mode
        # IMPORTANT: Both battery-first and grid-first MUST use slot 1!
        # The inverter prioritizes slot 1, so using slot 2 for grid-first prevents
        # proper export functionality when switching between modes.
        # Convert to HH:MM format required by device
        start_dev = self._to_device_hhmm(adjusted_start)
        stop_dev = self._to_device_hhmm(adjusted_stop)
        timeslot_payload = {
            "start": start_dev, "stop": stop_dev, "enabled": True, "slot": 1
        }
        self.logger.debug(f"Enabling grid-first mode for {adjusted_start}-{adjusted_stop}")
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(timeslot_payload))

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
            f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
            f"powerRate={power_rate}%) at {current_time} → "
            f"Topics: {self.config.grid_first_topic}, {self.config.grid_first_stopsoc_topic}, "
            f"{self.config.grid_first_powerrate_topic}"
        )
        self._last_applied["grid_first"] = sig

    async def _disable_grid_first(self) -> None:
        """Disable grid-first mode."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"🔌 [SIMULATE] GRID-FIRST MODE DISABLED (simulated at {current_time})")
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.grid_first_topic}"
        )

    async def _set_load_first(self) -> None:
        """Set load-first mode (disable both battery-first and grid-first).

        Load-first is the default mode where the inverter supplies loads
        from solar/battery without forcing grid or battery priority.
        """
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⚖️ [SIMULATE] LOAD-FIRST MODE SET (simulated at {current_time})")
            return

        # Disable both battery-first and grid-first with delay between
        await self._disable_battery_first()
        await asyncio.sleep(0.5)  # Delay between commands
        await self._disable_grid_first()

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"⚖️ LOAD-FIRST MODE SET (disabled battery & grid first) at {current_time}"
        )

    async def _calculate_and_schedule_next_day(self) -> None:
        """Calculate energy prices and schedule battery control for next day."""
        # Cancel and await previous scheduled tasks to prevent race conditions
        tasks_to_cancel = [t for t in self._scheduled_tasks if t and not t.done()]
        for task in tasks_to_cancel:
            task.cancel()

        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception:
                pass

        self._scheduled_tasks.clear()

        # Clear scheduled periods for new schedule
        self._scheduled_periods.clear()

        # Reset state tracking for new day
        self._last_applied.clear()
        self._ac_enabled = None
        self._export_enabled = None

        # Determine target date using local time
        now = self._get_local_now()
        current_time = now.time()
        cutoff_time = dt_time(23, 45)

        if current_time < cutoff_time:
            days_ahead = 0
            # Scheduling for today - use _schedule_today to skip past times
            self._schedule_func = self._schedule_today
        else:
            days_ahead = 1
            # Scheduling for tomorrow - use _schedule_at_time for next day
            self._schedule_func = self._schedule_at_time

        target_date = self._get_local_date_string(days_ahead=days_ahead)

        self.logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Scheduling energy prices for date: {target_date}")

        # Fetch energy prices from DAM
        hourly_prices = await self._fetch_dam_energy_prices(date=target_date)

        if not hourly_prices:
            self.logger.warning(
                "Failed to retrieve real energy prices. Using mock prices for testing."
            )
            # Use mock prices as fallback
            hourly_prices = self._generate_mock_prices(target_date)
            
            if not hourly_prices:
                self.logger.error(
                    "Failed to generate mock prices. Setting load-first mode and disabling export."
                )
                # Schedule fallback mode
                await self._schedule_fallback_mode()
                # Apply safe state immediately (not just at midnight)
                await self._set_load_first()
                await self._disable_export()
                self.logger.info("Applied safe state immediately due to price generation failure")
                return

        # Store prices in cache
        self._current_prices = hourly_prices
        self._prices_date = target_date
        self._prices_updated = datetime.now()

        # Get exchange rate once for consistency
        eur_czk_rate = await self._get_eur_czk_rate()

        # Log price table for visibility
        await self._log_price_table(hourly_prices, target_date, eur_czk_rate)

        self.logger.info(f"Energy prices for {target_date}: {len(hourly_prices)} hours")

        # Determine season mode
        season_mode = await self._get_season_mode()
        self.logger.info(f"Operating in {season_mode.upper()} mode for {target_date}")

        # Route to appropriate scheduling strategy based on season
        if season_mode == "summer":
            # Summer strategy: Use grid-first in morning, battery-first during low prices
            await self._schedule_summer_strategy(hourly_prices, eur_czk_rate)
        else:
            # Winter strategy: Traditional battery-first with AC charging
            await self._schedule_winter_strategy(hourly_prices, eur_czk_rate)

        # Log the complete schedule summary
        self._log_schedule_summary()

    async def _schedule_winter_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule winter battery strategy with AC charging during cheapest hours."""
        # Analyze prices (existing logic)
        cheapest_individual_hours = set(
            (start, stop)
            for start, stop, _ in self._find_n_cheapest_hours(
                hourly_prices, n=self.config.individual_cheapest_hours
            )
        )

        quadrants = self._categorize_prices_into_quadrants(hourly_prices)
        cheapest_quadrant_hours = set(
            (start, stop) for start, stop, _ in quadrants.get("Cheapest", [])
        )

        # Combine cheapest hours
        cheapest_individual_hours = cheapest_individual_hours.union(cheapest_quadrant_hours)

        cheapest_consecutive = set(
            (start, stop)
            for start, stop, _ in self._find_cheapest_consecutive_hours(
                hourly_prices, self.config.battery_charge_hours
            )
        )

        # Union all cheap hours
        all_cheap_hours = cheapest_individual_hours.union(cheapest_consecutive)

        # Calculate price statistics
        all_prices = list(hourly_prices.values())
        min_price = min(all_prices)
        max_price = max(all_prices)
        avg_price = sum(all_prices) / len(all_prices)

        # Calculate average prices for selected hours
        if cheapest_individual_hours:
            individual_avg = sum(hourly_prices[hour] for hour in cheapest_individual_hours) / len(
                cheapest_individual_hours
            )
        else:
            individual_avg = 0.0

        if cheapest_consecutive:
            consecutive_avg = sum(hourly_prices[hour] for hour in cheapest_consecutive) / len(
                cheapest_consecutive
            )
        else:
            consecutive_avg = 0.0

        # Use provided rate for consistency
        min_price_czk = min_price * eur_czk_rate / 1000
        max_price_czk = max_price * eur_czk_rate / 1000
        avg_price_czk = avg_price * eur_czk_rate / 1000
        individual_avg_czk = individual_avg * eur_czk_rate / 1000
        consecutive_avg_czk = consecutive_avg * eur_czk_rate / 1000

        self.logger.info(
            f"Winter mode price analysis: min={min_price:.2f} EUR/MWh "
            f"({min_price_czk:.3f} CZK/kWh), "
            f"max={max_price:.2f} EUR/MWh ({max_price_czk:.3f} CZK/kWh), "
            f"avg={avg_price:.2f} EUR/MWh ({avg_price_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Cheapest individual hours: {len(cheapest_individual_hours)} "
            f"(avg: {individual_avg:.2f} EUR/MWh = {individual_avg_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Cheapest consecutive hours: {len(cheapest_consecutive)} "
            f"(avg: {consecutive_avg:.2f} EUR/MWh = {consecutive_avg_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Total cheap hours: {len(all_cheap_hours)}, "
            f"Export threshold: {self.config.export_price_threshold:.2f} CZK/kWh"
        )

        # Schedule battery-first mode with AC charging for winter
        await self._schedule_battery_control(
            all_cheap_hours, cheapest_consecutive, hourly_prices, eur_czk_rate
        )

    async def _schedule_fallback_mode(self) -> None:
        """Schedule fallback mode when price data is unavailable."""
        # Schedule to disable battery-first at midnight with jitter
        task = asyncio.create_task(
            self._schedule_at_time(MIDNIGHT_JITTER, self._disable_battery_first)
        )
        self._scheduled_tasks.append(task)

        # Schedule to disable export at midnight with jitter
        task = asyncio.create_task(
            self._schedule_at_time(MIDNIGHT_DISABLE_JITTER, self._disable_export)
        )
        self._scheduled_tasks.append(task)

    def _schedule_action(self, time_str: str, coro_func: Any, *args: Any) -> asyncio.Task[None]:
        """Schedule an action using the appropriate scheduler (today vs tomorrow).

        Uses self._schedule_func which is set based on whether we're scheduling
        for today (use _schedule_today) or tomorrow (use _schedule_at_time).
        """
        # Check if this is a mode change command (via _emit_device_window)
        if coro_func == self._emit_device_window and len(args) >= 3:
            # For mode changes, schedule 1 minute early and mark as pre_scheduled
            adjusted_time = self._subtract_minutes_from_time(time_str, 1)
            # Create a wrapper that adds pre_scheduled=True

            async def wrapper() -> None:
                await self._emit_device_window(*args, pre_scheduled=True)
            return asyncio.create_task(self._schedule_func(adjusted_time, wrapper))
        else:
            # For other actions, schedule normally
            return asyncio.create_task(self._schedule_func(time_str, coro_func, *args))

    def _subtract_minutes_from_time(self, time_str: str, minutes: int) -> str:
        """Subtract minutes from a time string (HH:MM or HH:MM:SS).

        Handles midnight wrap-around (00:00 -> 23:59).
        """
        # Parse the time
        target_time = self._parse_time_any(time_str)
        now = self._get_local_now()

        # Create datetime for today with this time
        target_dt = datetime.combine(now.date(), target_time, self._local_tz)

        # Subtract minutes
        adjusted_dt = target_dt - timedelta(minutes=minutes)

        # Return as HH:MM:SS string
        return adjusted_dt.strftime("%H:%M:%S")

    async def _schedule_summer_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule summer battery strategy with grid-first and low-price storage.

        Export Logic:
        - DISABLED during low-price hours (< threshold, typically < 1 CZK/kWh)
        - ENABLED during all other periods (prices > threshold)
        - This maximizes profit by avoiding sales at low prices
        """
        # Convert threshold from CZK/kWh to EUR/MWh for comparison
        # Minimal guard against division by zero
        eur_czk_rate = max(eur_czk_rate, 1.0)
        threshold_eur_mwh = self.config.summer_price_threshold * 1000 / eur_czk_rate

        # Find hours below the summer threshold (typically <1 CZK/kWh)
        low_price_hours = [
            (start, stop, price)
            for (start, stop), price in hourly_prices.items()
            if price < threshold_eur_mwh
        ]

        if not low_price_hours:
            # No low-price hours - optimize based on time of day
            self.logger.info(
                f"Summer mode: No hours below "
                f"{self.config.summer_price_threshold:.2f} CZK/kWh."
            )

            # Get sunrise time
            now = self._get_local_now()
            days_ahead = 1 if now.time() >= dt_time(23, 45) else 0
            sunrise_time = self._get_sunrise_time(days_ahead)
            sunrise_str = sunrise_time.strftime("%H:%M")
            self.logger.info(f"Sunrise time: {sunrise_str}")

            # Schedule Load-First from midnight until sunrise
            if sunrise_time != dt_time(0, 0):
                self.logger.info(
                    f"Scheduling load-first from 00:00 to {sunrise_str} (overnight consumption)"
                )
                self._scheduled_periods.append(
                    Period("load_first", dt_time(0, 0), sunrise_time)
                )
                # Since no hours are below threshold, all prices are good - enable export
                self._scheduled_periods.append(
                    Period("export", dt_time(0, 0), sunrise_time)
                )

                task = self._schedule_action("00:00", self._set_load_first)
                self._scheduled_tasks.append(task)

                # Enable export for overnight period (prices above threshold)
                task = self._schedule_action(MIDNIGHT_JITTER, self._enable_export)
                self._scheduled_tasks.append(task)

            # Schedule Grid-First from sunrise to end of day (sell solar + battery at 10% rate)
            self.logger.info(
                f"Scheduling grid-first from {sunrise_str} to {EOD_HHMM} "
                f"(sell solar + battery, stopSOC=20%, powerRate=10%)"
            )
            self._scheduled_periods.append(
                Period("grid_first", sunrise_time, EOD_DTTIME,
                       params={"stop_soc": 20, "power_rate": 10})
            )
            self._scheduled_periods.append(
                Period("export", sunrise_time, EOD_DTTIME)
            )

            task = self._schedule_action(
                sunrise_str, self._emit_device_window,
                self._set_grid_first, sunrise_str, EOD_HHMM, 20, 10
            )
            self._scheduled_tasks.append(task)

            # Enable export after sunrise
            task = self._schedule_action(self._bump_time(sunrise_str, 5), self._enable_export)
            self._scheduled_tasks.append(task)
            return

        # Group low-price hours into contiguous periods
        low_price_groups = self._group_contiguous_hours_simple(low_price_hours)

        # Find the earliest and latest low-price times
        first_low_start = min(start for start, _ in low_price_groups)
        last_low_end = max(end for _, end in low_price_groups)

        self.logger.info(
            f"Summer mode: Found {len(low_price_hours)} hours below "
            f"{self.config.summer_price_threshold:.2f} CZK/kWh. "
            f"First low: {first_low_start}, Last low: {last_low_end}"
        )

        # Get sunrise time for the target day (tomorrow if scheduling after 23:45)
        now = self._get_local_now()
        days_ahead = 1 if now.time() >= dt_time(23, 45) else 0
        sunrise_time = self._get_sunrise_time(days_ahead)
        sunrise_str = sunrise_time.strftime("%H:%M")
        self.logger.info(f"Sunrise time: {sunrise_str}")

        # Schedule Load-First from midnight until sunrise
        if sunrise_time != dt_time(0, 0):
            self.logger.info(
                f"Scheduling load-first from 00:00 to {sunrise_str} (overnight consumption)"
            )
            self._scheduled_periods.append(
                Period("load_first", dt_time(0, 0), sunrise_time)
            )

            # Check if overnight prices are above threshold - if so, enable export
            sunrise_t = sunrise_time
            overnight_above_threshold = any(
                (datetime.strptime(start, "%H:%M").time() < sunrise_t)
                and (price >= threshold_eur_mwh)
                for ((start, _), price) in hourly_prices.items()
            )

            if overnight_above_threshold:
                self.logger.info("Overnight prices above threshold, enabling export")
                self._scheduled_periods.append(
                    Period("export", dt_time(0, 0), sunrise_time)
                )
                task = self._schedule_action(MIDNIGHT_JITTER, self._enable_export)
                self._scheduled_tasks.append(task)

            task = self._schedule_action("00:00", self._set_load_first)
            self._scheduled_tasks.append(task)

        # Schedule Grid-First from sunrise until first low price
        # Skip if low starts at/before sunrise
        sunrise_t = sunrise_time
        first_low_t = self._parse_hhmm(first_low_start)
        if first_low_t > sunrise_t:
            # Only schedule grid-first if there's a gap before first low price
            grid_first_end = first_low_start
            self.logger.info(
                f"Scheduling grid-first from {sunrise_str} to {grid_first_end} "
                f"(sell morning solar, stopSOC=20%, powerRate=10%)"
            )
            self._scheduled_periods.append(
                Period(
                    "grid_first",
                    sunrise_time,
                    self._parse_hhmm(grid_first_end),
                    params={"stop_soc": 20, "power_rate": 10}
                )
            )
            self._scheduled_periods.append(
                Period("export", sunrise_time, self._parse_hhmm(grid_first_end))
            )

            # Set grid-first at sunrise with stopSOC=20% and powerRate=10%
            task = self._schedule_action(
                sunrise_str, self._emit_device_window,
                self._set_grid_first, sunrise_str, grid_first_end, 20, 10
            )
            self._scheduled_tasks.append(task)

            # Enable export during morning high prices
            task = self._schedule_action(self._bump_time(sunrise_str, 5), self._enable_export)
            self._scheduled_tasks.append(task)

        # Schedule battery-first during each low-price period
        previous_end = None
        for group_start, group_end in low_price_groups:
            # If there's a gap from previous period, schedule load-first
            if previous_end:
                prev_end_t = self._parse_hhmm(previous_end)
                group_start_t = self._parse_hhmm(group_start)
                if prev_end_t < group_start_t:
                    self.logger.info(
                        f"Scheduling load-first from {previous_end} to {group_start} (price gap)"
                    )
                    self._scheduled_periods.append(
                        Period(
                            "load_first",
                            self._parse_hhmm(previous_end),
                            self._parse_hhmm(group_start)
                        )
                    )
                    self._scheduled_periods.append(
                        Period(
                            "export",
                            self._parse_hhmm(previous_end),
                            self._parse_hhmm(group_start)
                        )
                    )  # Track export period
                    task = self._schedule_action(previous_end, self._set_load_first)
                    self._scheduled_tasks.append(task)

                    # Re-enable export during the gap (prices above threshold)
                    enable_time = self._bump_time(previous_end, 5)
                    task = self._schedule_action(enable_time, self._enable_export)
                    self._scheduled_tasks.append(task)

            # Schedule battery-first for this low-price period
            self.logger.info(
                f"Scheduling battery-first from {group_start} to {group_end} "
                f"(store cheap solar, no AC charge)"
            )
            self._scheduled_periods.append(
                Period("battery_first", self._parse_hhmm(group_start), self._parse_hhmm(group_end))
            )

            # Switch to battery-first at start of low period
            task = self._schedule_action(
                group_start, self._emit_device_window,
                self._set_battery_first, group_start, group_end
            )
            self._scheduled_tasks.append(task)

            # Ensure AC charging is disabled (we only want solar charging)
            task = self._schedule_action(self._bump_time(group_start, 5), self._disable_ac_charge)
            self._scheduled_tasks.append(task)

            # Disable export during low prices (no point selling below operator costs)
            task = self._schedule_action(self._bump_time(group_start, 10), self._disable_export)
            self._scheduled_tasks.append(task)

            previous_end = group_end

        # Schedule load-first for evening (after last low price)
        if not _is_24(last_low_end):
            self.logger.info(
                f"Scheduling load-first from {last_low_end} to {EOD_HHMM} (use stored energy)"
            )
            self._scheduled_periods.append(
                Period("load_first", self._parse_hhmm(last_low_end), EOD_DTTIME)
            )
            self._scheduled_periods.append(
                Period("export", self._parse_hhmm(last_low_end), EOD_DTTIME)
            )  # Enable export for excess

            task = self._schedule_action(last_low_end, self._set_load_first)
            self._scheduled_tasks.append(task)

            # Enable export for evening (in case of excess energy)
            task = self._schedule_action(self._bump_time(last_low_end, 5), self._enable_export)
            self._scheduled_tasks.append(task)

    async def _schedule_battery_control(
        self,
        all_cheap_hours: set[Tuple[str, str]],
        cheapest_consecutive: set[Tuple[str, str]],
        hourly_prices: Dict[Tuple[str, str], float],
        eur_czk_rate: float,
    ) -> None:
        """Schedule battery control based on price analysis."""
        # Group battery-first hours into contiguous blocks
        battery_first_hours = [
            (start, stop, hourly_prices[(start, stop)]) for start, stop in all_cheap_hours
        ]
        battery_first_groups = self._group_contiguous_hours_simple(battery_first_hours)

        # Schedule battery-first mode for each block
        for group_start, group_end in battery_first_groups:
            self.logger.info(f"Scheduling battery-first mode from {group_start} to {group_end}")
            self._scheduled_periods.append(
                Period("battery_first", self._parse_hhmm(group_start), self._parse_hhmm(group_end))
            )
            task = self._schedule_action(
                group_start, self._emit_device_window,
                self._set_battery_first, group_start, group_end
            )
            self._scheduled_tasks.append(task)

        # Schedule AC charging during cheapest consecutive hours
        if cheapest_consecutive:
            ac_charge_hours = [
                (start, stop, hourly_prices[(start, stop)]) for start, stop in cheapest_consecutive
            ]
            ac_charge_groups = self._group_contiguous_hours_simple(ac_charge_hours)

            for start_time, stop_time in ac_charge_groups:
                # Calculate price statistics for this charging period
                start_t = self._parse_hhmm(start_time)
                stop_t = self._parse_hhmm(stop_time)
                charge_prices = [
                    price
                    for start, stop, price in ac_charge_hours
                    if self._parse_hhmm(start) >= start_t and self._parse_hhmm(stop) <= stop_t
                ]
                if charge_prices:
                    min_charge_price = min(charge_prices)
                    max_charge_price = max(charge_prices)
                    avg_charge_price = sum(charge_prices) / len(charge_prices)

                    # Convert to CZK/kWh for display (use provided rate)
                    min_charge_price_czk_kwh = min_charge_price * eur_czk_rate / 1000
                    max_charge_price_czk_kwh = max_charge_price * eur_czk_rate / 1000
                    avg_charge_price_czk_kwh = avg_charge_price * eur_czk_rate / 1000

                    self.logger.info(
                        f"Scheduling AC charge from {start_time} to {stop_time} "
                        f"(min: {min_charge_price:.2f}, avg: {avg_charge_price:.2f}, "
                        f"max: {max_charge_price:.2f} EUR/MWh = "
                        f"{min_charge_price_czk_kwh:.2f}-{avg_charge_price_czk_kwh:.2f}-"
                        f"{max_charge_price_czk_kwh:.2f} CZK/kWh)"
                    )
                else:
                    self.logger.info(
                        f"Scheduling AC charge from {start_time} to {stop_time} (no price data)"
                    )

                # Track AC charge period
                self._scheduled_periods.append(
                    Period("ac_charge", self._parse_hhmm(start_time), self._parse_hhmm(stop_time))
                )

                # Schedule AC charge start
                task = self._schedule_action(start_time, self._enable_ac_charge)
                self._scheduled_tasks.append(task)

                # Schedule AC charge stop (normalize if 24:00)
                stop_time_norm = self._normalize_end_time(stop_time)
                task = self._schedule_action(stop_time_norm, self._disable_ac_charge)
                self._scheduled_tasks.append(task)

        # Schedule export control
        await self._schedule_export_control(hourly_prices, eur_czk_rate)

        # Ensure clean end-of-day state
        self._schedule_end_of_day_cleanup()

    async def _schedule_export_control(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule export enable/disable based on price thresholds."""
        # Build AC charge windows for overlap checking
        ac_windows: List[Tuple[dt_time, dt_time]] = [
            (p.start, p.end) for p in self._scheduled_periods if p.kind == "ac_charge"
        ]

        def _segments(start: dt_time, end: dt_time) -> List[Tuple[dt_time, dt_time]]:
            """Split a possibly midnight-wrapping range into 1-2 non-wrapping segments."""
            if start <= end:
                return [(start, end)]
            # Wraps midnight: [start, EOD) U [00:00, end)
            return [(start, EOD_DTTIME), (dt_time(0, 0), end)]

        def _ranges_overlap(
            a_start: dt_time, a_end: dt_time, b_start: dt_time, b_end: dt_time
        ) -> bool:
            """Proper overlap test for possibly wrapping half-open time ranges."""
            for s1, e1 in _segments(a_start, a_end):
                for s2, e2 in _segments(b_start, b_end):
                    if s1 < e2 and e1 > s2:
                        return True
            return False

        def _overlaps_ac(start: dt_time, end: dt_time) -> bool:
            """Check if a time window overlaps with any AC charging period."""
            return any(
                _ranges_overlap(start, end, ac_start, ac_end)
                for ac_start, ac_end in ac_windows
            )

        def _subtract_one(
            a: Tuple[dt_time, dt_time], b: Tuple[dt_time, dt_time]
        ) -> List[Tuple[dt_time, dt_time]]:
            """Return A\\B for non-wrapping [start,end) dt_time ranges."""
            (as_, ae), (bs, be) = a, b
            # No overlap
            if ae <= bs or as_ >= be:
                return [a]
            # Full cover
            if bs <= as_ and be >= ae:
                return []
            pieces = []
            if bs > as_:
                pieces.append((as_, bs))
            if be < ae:
                pieces.append((be, ae))
            return pieces

        def subtract_wrap_range(
            a_start: dt_time, a_end: dt_time, b_start: dt_time, b_end: dt_time
        ) -> List[Tuple[dt_time, dt_time]]:
            """Return (possibly multiple) non-wrapping export slices = A \\ B, with wrap handled."""
            a_parts = _segments(a_start, a_end)
            b_parts = _segments(b_start, b_end)
            out: List[Tuple[dt_time, dt_time]] = []
            for ap in a_parts:
                cur = [ap]
                for bp in b_parts:
                    nxt: List[Tuple[dt_time, dt_time]] = []
                    for seg in cur:
                        nxt.extend(_subtract_one(seg, bp))
                    cur = nxt
                out.extend(cur)
            return out

        # Convert threshold from CZK/kWh to EUR/MWh for comparison with API data
        # API prices are in EUR/MWh, threshold is in CZK/kWh
        # Conversion: 1 MWh = 1000 kWh
        # Minimal guard against division by zero
        eur_czk_rate = max(eur_czk_rate, 1.0)
        threshold_eur_mwh = self.config.export_price_threshold * 1000 / eur_czk_rate

        export_hours = [
            (start, stop)
            for (start, stop), price in hourly_prices.items()
            if price >= threshold_eur_mwh
        ]

        if not export_hours:
            self.logger.info(
                f"No hours above export price threshold "
                f"({threshold_eur_mwh:.2f} EUR/MWh = "
                f"{self.config.export_price_threshold:.2f} CZK/kWh)"
            )
            # Schedule disable at midnight with small delay to avoid conflicts
            task = asyncio.create_task(
                self._schedule_at_time(MIDNIGHT_JITTER, self._disable_export)
            )
            self._scheduled_tasks.append(task)
            return

        # Group export hours into contiguous blocks (ignore price differences for export)
        export_hours_with_price = [
            (start, stop, hourly_prices[(start, stop)]) for start, stop in export_hours
        ]
        export_groups = self._group_contiguous_hours_simple(export_hours_with_price)

        self.logger.info(
            f"Found {len(export_groups)} export periods above "
            f"{threshold_eur_mwh:.2f} EUR/MWh threshold"
        )

        # Sort groups by start time to handle them in order
        export_groups.sort(key=lambda x: self._parse_hhmm(x[0]))

        # Track if we actually scheduled a 00:00 enable
        midnight_enable_scheduled = False

        for group_start, group_end in export_groups:
            # DO NOT normalize here; keep HH:MM or "24:00" for logic
            # group_end_norm will be used only when calling _schedule_at_time

            # Note: Gap-disable between export groups removed as redundant
            # Each export slice already schedules its own disable at slice end

            # Check if export overlaps with AC charging (if suppression enabled)
            suppress_export = self._optional_config.get("suppress_export_during_ac_charge", True)
            export_slices: List[Tuple[dt_time, dt_time]] = []

            if suppress_export and _overlaps_ac(
                self._parse_hhmm(group_start), self._parse_hhmm(group_end)
            ):
                # Carve out AC times from export window
                remaining = [(self._parse_hhmm(group_start), self._parse_hhmm(group_end))]
                for ac_start, ac_end in ac_windows:
                    next_remaining = []
                    for rs, re in remaining:
                        next_remaining.extend(subtract_wrap_range(rs, re, ac_start, ac_end))
                    remaining = next_remaining

                if remaining:
                    export_slices = remaining
                    self.logger.info(
                        f"Export {group_start}-{group_end} overlaps AC charge, "
                        f"scheduling {len(export_slices)} non-overlapping slice(s)"
                    )
                else:
                    self.logger.info(
                        f"Export {group_start}-{group_end} fully overlaps AC charge, skipping"
                    )
                    continue
            else:
                # No AC overlap, use full window
                export_slices = [(self._parse_hhmm(group_start), self._parse_hhmm(group_end))]

            # Schedule each export slice
            for slice_start_t, slice_end_t in export_slices:
                slice_start = slice_start_t.strftime("%H:%M")
                # Keep the exact EOD barrier if the slice ends at EOD_DTTIME
                if slice_end_t == EOD_DTTIME:
                    slice_end = EOD_HHMMSS
                else:
                    slice_end = self._normalize_end_time(slice_end_t.strftime("%H:%M"))

                # Calculate price statistics for this slice
                group_prices = [
                    price
                    for start, stop, price in export_hours_with_price
                    if (self._parse_hhmm(start) >= slice_start_t
                        and self._parse_hhmm(stop) <= slice_end_t)
                ]
                if group_prices:
                    min_price = min(group_prices)
                    max_price = max(group_prices)
                    avg_price = sum(group_prices) / len(group_prices)

                    # Convert prices to CZK/kWh for display
                    min_price_czk_kwh = min_price * eur_czk_rate / 1000
                    max_price_czk_kwh = max_price * eur_czk_rate / 1000
                    avg_price_czk_kwh = avg_price * eur_czk_rate / 1000

                    self.logger.info(
                        f"Scheduling export enable from {slice_start} to {slice_end} "
                        f"(min: {min_price:.2f}, avg: {avg_price:.2f}, "
                        f"max: {max_price:.2f} EUR/MWh = "
                        f"{min_price_czk_kwh:.2f}-{avg_price_czk_kwh:.2f}-"
                        f"{max_price_czk_kwh:.2f} CZK/kWh, "
                        f"threshold: {self.config.export_price_threshold:.2f})"
                    )
                else:
                    self.logger.info(
                        f"Scheduling export enable from {slice_start} to {slice_end} "
                        f"(no price data)"
                    )

                # Track export period
                self._scheduled_periods.append(
                    Period("export", slice_start_t, slice_end_t)
                )

                # Check if this is a midnight start
                if slice_start == "00:00":
                    midnight_enable_scheduled = True

                # Schedule export enable at start
                task = self._schedule_action(slice_start, self._enable_export)
                self._scheduled_tasks.append(task)

                # Schedule export disable at end (normalize to EOD sentinel if needed)
                disable_at = self._normalize_for_schedule(slice_end)
                task = self._schedule_action(disable_at, self._disable_export)
                self._scheduled_tasks.append(task)

        # Check if we need a midnight disable (only if we didn't schedule a 00:00 enable)
        if not midnight_enable_scheduled:
            # Use constant for better readability
            task = asyncio.create_task(
                self._schedule_at_time(MIDNIGHT_JITTER, self._disable_export)
            )
            self._scheduled_tasks.append(task)
            self.logger.info(f"Scheduled export disable at {MIDNIGHT_JITTER} for clean daily start")

    async def _schedule_at_time(self, time_str: str, coro_func: Any, *args: Any) -> None:
        """Schedule a coroutine to run at a specific time."""
        try:
            # Normalize time using unified handler
            time_str = self._normalize_for_schedule(time_str)

            # Support both HH:MM and HH:MM:SS formats
            # Parse time with unified handler
            target_time = self._parse_time_any(time_str)

            now = self._get_local_now()

            # Calculate next occurrence of target time in local timezone
            target_datetime = datetime.combine(now.date(), target_time, self._local_tz)
            if target_datetime <= now:
                target_datetime += timedelta(days=1)

            delay = (target_datetime - now).total_seconds()

            if delay > 0:
                await asyncio.sleep(delay)
            # Check if still running before executing
            if not self._running:
                return
            await coro_func(*args)

        except asyncio.CancelledError:
            # Expected on shutdown; keep quiet
            raise
        except Exception as e:
            self.logger.error(f"Error in scheduled task at {time_str}: {e}", exc_info=True)

    async def _emit_device_window(
        self,
        setter: Any,
        start_hhmm: str,
        stop_hhmm: str,
        *extra: Any,
        preserve_when_same_day: bool = True,
        pre_scheduled: bool = False,
    ) -> None:
        """Call the device setter once or split across midnight if start > stop.

        Some Growatt firmwares don't accept a single slot where start > stop.
        This helper splits such windows into two: [start → 23:59] and [00:00 → stop].
        Args:
            preserve_when_same_day: Whether to preserve duration for same-day windows.
                                   Set to False for immediate apply to avoid spillover.
            pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        """
        st = self._parse_hhmm(start_hhmm)
        en = self._parse_hhmm(stop_hhmm)
        if st <= en:
            # If we're aiming at the EOD barrier, never preserve duration on bumps.
            # Otherwise, allow the caller to choose.
            preserve = False if stop_hhmm in (EOD_HHMM, EOD_HHMMSS) else preserve_when_same_day
            await setter(
                start_hhmm, stop_hhmm, *extra,
                preserve_duration=preserve, pre_scheduled=pre_scheduled
            )
        else:
            # Split wrap: [start → 23:59] U [00:00 → stop]
            self.logger.debug(f"Splitting midnight-wrap window {start_hhmm}-{stop_hhmm}")
            # Keep hard EOD
            await setter(
                start_hhmm, EOD_HHMM, *extra,
                preserve_duration=False, pre_scheduled=pre_scheduled
            )
            await asyncio.sleep(0.5)  # Delay between split commands
            await setter(
                "00:00", stop_hhmm, *extra,
                preserve_duration=True, pre_scheduled=pre_scheduled
            )

    async def _schedule_today(self, time_str: str, coro_func: Any, *args: Any) -> None:
        """Schedule only if the time is still ahead *today* (no rollover).

        This prevents scheduling past times that would roll over to tomorrow.
        Use this ONLY when re-applying today's schedule (e.g., after a midday restart).
        Do NOT use for the nightly calculation which schedules tomorrow's prices.
        """
        # Normalize like _schedule_at_time (HH:MM -> HH:MM:SS, 24:00 -> EOD)
        time_str = self._normalize_for_schedule(time_str)

        try:
            target_time = self._parse_time_any(time_str)
        except Exception:
            self.logger.warning(f"Skipping invalid time string: {time_str}")
            return

        now = self._get_local_now().replace(microsecond=0)
        target_dt = datetime.combine(now.date(), target_time, self._local_tz)

        if target_dt <= now:
            self.logger.debug(f"Skipping past-time schedule for today: {time_str}")
            return

        delay = (target_dt - now).total_seconds()

        try:
            if delay > 0:
                await asyncio.sleep(delay)
            # Check if still running before executing
            if not self._running:
                return
            await coro_func(*args)
        except asyncio.CancelledError:
            # Expected on shutdown
            raise
        except Exception as e:
            self.logger.error(f"Error in scheduled task at {time_str}: {e}", exc_info=True)

    def _schedule_end_of_day_cleanup(self) -> None:
        """Ensure we land in a neutral state at end of day."""
        # Small jittered disables are safe & idempotent.
        self._scheduled_tasks.append(
            asyncio.create_task(self._schedule_at_time(EOD_HHMMSS, self._disable_export))
        )
        self._scheduled_tasks.append(
            asyncio.create_task(self._schedule_at_time(EOD_HHMMSS, self._disable_ac_charge))
        )

    async def _schedule_daily_calculation(self) -> None:
        """Run initial calculation now and then re-run daily at configured schedule hour:minute."""
        # Run initial calculation
        await self._calculate_and_schedule_next_day()

        # Schedule daily recalculation
        self._daily_schedule_task = asyncio.create_task(self._daily_calculation_loop())

    async def _daily_calculation_loop(self) -> None:
        """Run daily calculation loop."""
        while self._running:
            try:
                # Calculate time until next scheduled time in local timezone
                now = self._get_local_now()
                target_time = dt_time(self.config.schedule_hour, self.config.schedule_minute)
                target_datetime = datetime.combine(now.date(), target_time, self._local_tz)

                if target_datetime <= now:
                    target_datetime += timedelta(days=1)

                delay = (target_datetime - now).total_seconds()

                self.logger.info(
                    f"Next daily calculation at {target_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"({delay / 3600:.1f} hours)"
                )
                await asyncio.sleep(delay)

                # Run calculation
                await self._calculate_and_schedule_next_day()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in daily calculation loop: {e}", exc_info=True)
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)
