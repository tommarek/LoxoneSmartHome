"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
import zoneinfo
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple, Union

import aiohttp
from astral import LocationInfo
from astral.sun import sun

from config.settings import Settings
from modules.base import BaseModule
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient

# Import from refactored modules
from .growatt.models import (
    Period, PeriodType,
    EOD_HHMM, EOD_HHMMSS, EOD_DTTIME,
    is_24
)
from .growatt.price_analyzer import PriceAnalyzer
from .growatt.mode_manager import ModeManager
from .growatt.decision_engine import (
    GrowattDecisionEngine, DecisionContext, PriceThresholds, MODE_DEFINITIONS
)


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    # Class attributes for type checking
    _manual_override_period: Optional[Period]
    _manual_override_end_time: Optional[datetime]
    _manual_override_source: str
    _optional_config: Dict[str, Any]

    def _select_primary_mode(self, modes: set[PeriodType]) -> PeriodType:
        """Select primary mode from active modes.

        With composite modes, there should only be one active mode at a time.
        Manual overrides take precedence over everything else.
        """
        # Check for active manual override first
        if self._manual_override_period is not None:
            now = self._get_local_now()
            # Check if manual override is still valid
            if self._manual_override_end_time is not None:
                if now < self._manual_override_end_time:
                    return self._manual_override_period.kind
                else:
                    self.logger.info("Manual override expired, clearing")
                    self._manual_override_period = None
                    self._manual_override_end_time = None
                    self._manual_override_source = ""
            else:
                # No end time means manual override is permanent until cleared
                return self._manual_override_period.kind

        # With composite modes, just return the first (and should be only) mode
        if modes:
            return next(iter(modes))
        return "regular"  # Default composite mode

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

        # Note: Price thresholds validation not needed as Pydantic Field has gt=0 constraint

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

        # Optional config will be set during validation

        # Validate config and set defaults
        self._validate_config()

        # Event-driven state
        self._last_evaluation_hour: Optional[int] = None
        self._last_evaluation_reason: Optional[str] = None
        self._evaluation_lock = asyncio.Lock()
        self._periodic_check_task: Optional[asyncio.Task[None]] = None

        # Legacy compatibility - to be removed after full migration
        self._scheduled_tasks: list[Any] = []  # Empty list for backward compatibility
        self._scheduled_periods: list[Any] = []  # Empty list for backward compatibility
        self._scheduled_mode: Optional[str] = None  # Legacy mode tracking

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

        self._eur_czk_rate: Optional[float] = None
        self._eur_czk_rate_updated: Optional[datetime] = None

        self._current_prices: Dict[Tuple[str, str], float] = {}
        self._prices_date: Optional[str] = None
        self._prices_updated: Optional[datetime] = None

        # Mode states are now tracked in ModeManager

        self._clock_drift_seconds: float = 0

        self._last_applied: Dict[str, Tuple[Any, ...]] = {}

        self._schedule_func = self._schedule_at_time

        self._manual_override_period = None
        self._manual_override_end_time = None
        self._manual_override_source = ""

        self._last_command_results: Dict[str, Dict[str, Any]] = {}

        # Initialize refactored modules
        self._price_analyzer = PriceAnalyzer(self.logger, self._local_tz, self._optional_config)
        self._mode_manager = ModeManager(self)
        self._decision_engine = GrowattDecisionEngine(self.logger)

        # Home status monitoring for high load detection
        self._home_status: Dict[str, Any] = {}
        self._high_loads_active: bool = False
        self._home_status_topic = "loxone/status"
        self._current_mode: Optional[str] = None  # Track the currently applied mode
        self._battery_soc: float = 50.0  # Default battery SOC, updated from status
        self._last_battery_soc: float = 50.0  # Track SOC changes
        self._current_load: float = 0.0  # Current home load in kW
        self._solar_power: float = 0.0  # Current solar generation in kW
        self._last_price_fetch: Optional[datetime] = None  # Track price update time

    async def _wait_for_command_result(
        self, command_type: str, timeout: float = 3.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for and capture command result from energy/solar/result topic.

        Args:
            command_type: The command type to wait for (e.g., "batteryfirst/set/timeslot")
            timeout: How long to wait for the result

        Returns:
            The result dict if received, None if timeout
        """
        if self._optional_config.get("simulation_mode", False):
            return {"success": True, "message": "Simulated"}

        result_future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()

        async def result_handler(_topic: str, payload: Any) -> None:
            try:
                if isinstance(payload, bytes):
                    payload = payload.decode()
                data = json.loads(payload)

                # Check if this is our command result
                if data.get("command") == command_type:
                    if not result_future.done():
                        result_future.set_result(data)
                        # Store for later reference
                        self._last_command_results[command_type] = data

                        # Log the result
                        success = data.get("success", False)
                        message = data.get("message", "No message")
                        if success:
                            self.logger.info(f"✅ Command {command_type} succeeded: {message}")
                        else:
                            self.logger.error(f"❌ Command {command_type} FAILED: {message}")
                            self.logger.error(
                                f"📋 Full error response: {json.dumps(data, indent=2)}"
                            )

            except Exception as e:
                self.logger.error(f"Error parsing command result: {e}")

        # Subscribe to result topic
        assert self.mqtt_client is not None
        await self.mqtt_client.subscribe("energy/solar/result", result_handler)

        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(result_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.logger.warning(f"⏱️ Timeout waiting for {command_type} result after {timeout}s")
            return None
        finally:
            await self.mqtt_client.unsubscribe("energy/solar/result")

    async def _query_inverter_state(self) -> Dict[str, Any]:
        """Query current inverter state (battery-first and grid-first settings).

        Returns:
            Dict with current state of both modes
        """
        if self._optional_config.get("simulation_mode", False):
            return {
                "battery_first": (
                    self._mode_manager.get_battery_first_slots()
                    if hasattr(self, '_mode_manager') else {}
                ),
                "grid_first": {}  # No longer tracking grid-first slots
            }

        state = {}

        # Query battery-first state
        try:
            assert self.mqtt_client is not None

            # Set up result handler
            bf_future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()

            async def bf_handler(_topic: str, payload: Any) -> None:
                try:
                    if isinstance(payload, bytes):
                        payload = payload.decode()
                    data = json.loads(payload)
                    if data.get("command") == "batteryfirst/get":
                        if not bf_future.done():
                            bf_future.set_result(data)
                except Exception as e:
                    self.logger.error(f"Error parsing battery-first state: {e}")

            await self.mqtt_client.subscribe("energy/solar/result", bf_handler)

            # Send query
            self.logger.debug("Querying battery-first state...")
            await self.mqtt_client.publish("energy/solar/command/batteryfirst/get", "{}")

            # Wait for response
            try:
                bf_result = await asyncio.wait_for(bf_future, timeout=3.0)
                state["battery_first"] = bf_result

                # Log the state
                if bf_result.get("success"):
                    slots = bf_result.get("timeSlots", [])
                    for slot in slots:
                        if slot.get("enabled"):
                            self.logger.info(
                                f"📊 Battery-first slot {slot.get('slot')}: "
                                f"{slot.get('start')}-{slot.get('stop')} ENABLED"
                            )
                else:
                    self.logger.warning(
                        f"Failed to query battery-first state: {bf_result.get('message')}"
                    )
                    self.logger.debug(f"📋 Full query response: {json.dumps(bf_result, indent=2)}")

            except asyncio.TimeoutError:
                self.logger.warning("Timeout querying battery-first state")
                state["battery_first"] = {}
            finally:
                await self.mqtt_client.unsubscribe("energy/solar/result")

        except Exception as e:
            self.logger.error(f"Error querying battery-first state: {e}")
            state["battery_first"] = {}

        # Query grid-first state
        try:
            assert self.mqtt_client is not None

            # Set up result handler
            gf_future: asyncio.Future[Dict[str, Any]] = asyncio.get_running_loop().create_future()

            async def gf_handler(_topic: str, payload: Any) -> None:
                try:
                    if isinstance(payload, bytes):
                        payload = payload.decode()
                    data = json.loads(payload)
                    if data.get("command") == "gridfirst/get":
                        if not gf_future.done():
                            gf_future.set_result(data)
                except Exception as e:
                    self.logger.error(f"Error parsing grid-first state: {e}")

            await self.mqtt_client.subscribe("energy/solar/result", gf_handler)

            # Send query
            self.logger.debug("Querying grid-first state...")
            await self.mqtt_client.publish("energy/solar/command/gridfirst/get", "{}")

            # Wait for response
            try:
                gf_result = await asyncio.wait_for(gf_future, timeout=3.0)
                state["grid_first"] = gf_result

                # Log the state
                if gf_result.get("success"):
                    slots = gf_result.get("timeSlots", [])
                    for slot in slots:
                        if slot.get("enabled"):
                            self.logger.info(
                                f"📊 Grid-first slot {slot.get('slot')}: "
                                f"{slot.get('start')}-{slot.get('stop')} ENABLED, "
                                f"stopSOC={gf_result.get('stopSOC')}%, "
                                f"powerRate={gf_result.get('powerRate')}%"
                            )
                else:
                    self.logger.warning(
                        f"Failed to query grid-first state: {gf_result.get('message')}"
                    )
                    self.logger.debug(f"📋 Full query response: {json.dumps(gf_result, indent=2)}")

            except asyncio.TimeoutError:
                self.logger.warning("Timeout querying grid-first state")
                state["grid_first"] = {}
            finally:
                await self.mqtt_client.unsubscribe("energy/solar/result")

        except Exception as e:
            self.logger.error(f"Error querying grid-first state: {e}")
            state["grid_first"] = {}

        return state

    def _get_local_now(self) -> datetime:
        """Get current time in local timezone."""
        return datetime.now(self._local_tz)

    def _parse_time_any(self, s: str) -> dt_time:
        """Accept HH:MM, HH:MM:SS, and '24:00' => 00:00 (next-day semantics)."""
        if is_24(s):
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
        return EOD_HHMMSS if is_24(s) else s

    def _normalize_for_schedule(self, s: str) -> str:
        """Normalize any time string for scheduling calls.

        Converts 24:00 to EOD_HHMMSS and ensures HH:MM:SS format.
        """
        if is_24(s):
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
        t_with_delta = (t + timedelta(seconds=seconds)).time()
        return t_with_delta.strftime("%H:%M:%S")

    def _to_device_hhmm(self, s: str) -> str:
        """Convert time string to HH:MM format required by device timeslot endpoints.

        The firmware strictly requires HH:MM format (no seconds) for
        batteryfirst/set/timeslot and gridfirst/set/timeslot commands.
        """
        # Handle EOD sentinels and 24:00
        if s in ("24:00", "24:00:00", EOD_HHMMSS):
            # DEBUG
            self.logger.debug(f"🔍 DEBUG _to_device_hhmm: Converting {s!r} to {EOD_HHMM!r}")
            return EOD_HHMM  # "23:59"
        # Truncate to HH:MM if longer
        result = s[:5] if len(s) >= 5 else s
        # DEBUG
        if s != result:
            self.logger.debug(f"🔍 DEBUG _to_device_hhmm: Converted {s!r} to {result!r}")
        return result

    async def _set_mode(self, mode: str, *args: Any) -> None:
        """Set the inverter mode (thin dispatcher to individual setters)."""
        # These direct calls are no longer used - decision engine handles modes
        pass

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
        sunrise: datetime = s['sunrise']

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

            # Add mode-specific details for composite modes
            first_period: Optional[Period] = slot["periods"][0] if slot["periods"] else None

            if slot["primary_mode"] == "discharge_to_grid":
                entry["mode"] = "DISCHARGE-TO-GRID"
                if first_period and first_period.params:
                    stop_soc = first_period.params.get("stop_soc", 20)
                    power_rate = first_period.params.get("power_rate", 100)
                    entry["details"].append(f"Discharge to {stop_soc}% SOC")
                    entry["details"].append(f"Power rate: {power_rate}%")
                entry["details"].append("Export: ENABLED")

            elif slot["primary_mode"] == "charge_from_grid":
                entry["mode"] = "CHARGE-FROM-GRID"
                if first_period and first_period.params:
                    stop_soc = first_period.params.get("stop_soc", self.config.max_soc)
                    entry["details"].append(f"Charge to {stop_soc}% SOC")
                entry["details"].append("AC Charging: ENABLED")
                entry["details"].append("Export: ENABLED")  # Export now enabled during charging

            elif slot["primary_mode"] == "sell_production":
                entry["mode"] = "SELL-PRODUCTION"
                entry["details"].append("Sell solar only (no battery discharge)")
                entry["details"].append("Export: ENABLED")

            elif slot["primary_mode"] == "regular":
                entry["mode"] = "REGULAR"
                entry["details"].append("Normal operation")
                entry["details"].append("Export: ENABLED")

            elif slot["primary_mode"] == "regular_no_export":
                entry["mode"] = "REGULAR-NO-EXPORT"
                entry["details"].append("Normal operation")
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
                        fallback = float(
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
            fallback = float(self._optional_config.get("eur_czk_rate", 25.0))
            # Cache the fallback to avoid repeated API calls
            self._eur_czk_rate = fallback
            self._eur_czk_rate_updated = now
            return fallback

        except Exception as e:
            self.logger.error(f"Error fetching EUR/CZK exchange rate: {e}")
            # Return default fallback (25 CZK per EUR is typical)
            fallback = float(self._optional_config.get("eur_czk_rate", 25.0))
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
        self._running = True

        # Sync inverter time on startup
        self.logger.info("Checking inverter time synchronization...")
        await self._sync_inverter_time()

        # Subscribe to home status for high load detection
        if self.mqtt_client:
            self.logger.info(f"Subscribing to home status topic: {self._home_status_topic}")
            await self.mqtt_client.subscribe(self._home_status_topic, self._on_home_status)

        # Fetch initial prices
        await self._fetch_prices()

        # Start periodic evaluation loop
        self._periodic_check_task = asyncio.create_task(self._periodic_evaluation_loop())

        # Perform initial evaluation
        await self._evaluate_conditions("startup")

        self.logger.info("Growatt controller started with event-driven evaluation")

    async def stop(self) -> None:
        """Stop the Growatt controller."""
        self._running = False  # Clear the flag to stop evaluation loop

        # Cancel periodic check task
        if self._periodic_check_task and not self._periodic_check_task.done():
            self._periodic_check_task.cancel()
            try:
                await self._periodic_check_task
            except asyncio.CancelledError:
                pass
            self._periodic_check_task = None

        # Apply safe shutdown mode
        self.logger.info("🔄 Resetting inverter to safe shutdown state...")
        try:
            await self._evaluate_conditions("shutdown")
            self.logger.info("✅ Inverter shutdown state: regular mode applied")
        except Exception as e:
            self.logger.error(f"Error during shutdown reset: {e}", exc_info=True)

        self.logger.info("Growatt controller stopped")

    async def _determine_and_apply_mode(self) -> None:
        """Legacy adapter - redirects to new evaluation system.

        This method exists for backward compatibility with code that still
        calls the old method. It simply triggers a re-evaluation.
        """
        await self._evaluate_conditions("legacy_call")

    async def _on_home_status(self, _topic: str, payload: Any) -> None:
        """Handle home status updates from UDP listener.

        Args:
            topic: The MQTT topic (should be loxone/status)
            payload: The status JSON payload
        """
        try:
            # Parse the payload
            if isinstance(payload, bytes):
                payload = payload.decode()

            if isinstance(payload, str):
                data = json.loads(payload)
            else:
                data = payload

            self._home_status = data

            # Update battery SOC if available in status
            solar_data = data.get("solar", {})
            if solar_data:
                # Try to find battery SOC in various possible formats
                for key in ["battery_soc", "batterysoc", "soc", "battery_level"]:
                    if key in solar_data:
                        soc_data = solar_data[key]
                        if isinstance(soc_data, dict):
                            soc_value = soc_data.get("value", self._battery_soc)
                        else:
                            soc_value = soc_data
                        if isinstance(soc_value, (int, float)) and 0 <= soc_value <= 100:
                            self._battery_soc = float(soc_value)
                            break

            # Detect high loads from the raw data
            high_loads = self._detect_high_loads_from_status(data)
            was_high_load = self._high_loads_active

            self._high_loads_active = high_loads["active"]

            # Log significant changes
            if self._high_loads_active != was_high_load:
                if self._high_loads_active:
                    details = []
                    if high_loads.get("ev_charging"):
                        details.append(f"EV: {high_loads['ev_power']:.0f}W")
                    if high_loads.get("heating_active"):
                        details.append(
                            f"Heating: {len(high_loads.get('heating_relays', []))} relays"
                        )
                    self.logger.warning(
                        f"⚡ High loads detected! {', '.join(details)} - Checking if action needed"
                    )
                    # Handle high load start
                    await self._handle_high_load_start()
                else:
                    self.logger.info("✅ High loads cleared - Restoring scheduled operation")
                    # Use decision tree to determine what mode to apply now
                    await self._determine_and_apply_mode()

        except Exception as e:
            self.logger.error(f"Failed to process home status: {e}", exc_info=True)

    def _detect_high_loads_from_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Detect high load conditions from home status data.

        Args:
            status: The status data from UDP listener

        Returns:
            Dictionary with high load detection results
        """
        result: Dict[str, Any] = {
            "active": False,
            "ev_charging": False,
            "ev_power": 0,
            "heating_active": False,
            "heating_relays": []
        }

        # Check EV charging (power > 100W threshold)
        ev_data = status.get("ev", {})
        ev_power_total = 0
        for name, data in ev_data.items():
            if isinstance(data, dict) and "power" in name.lower():
                value = data.get("value", 0)
                if value > 100:  # EV charging threshold
                    ev_power_total += value

        if ev_power_total > 100:
            result["ev_charging"] = True
            result["ev_power"] = ev_power_total

        # Check heating relays (ANY relay with tag1 = "heating" that is ON)
        relay_data = status.get("relay", {})

        for name, data in relay_data.items():
            if isinstance(data, dict):
                value = data.get("value", 0)
                tag1 = data.get("tag1", "")

                # If ANY heating relay is ON, heating is active
                if value == 1 and tag1 == "heating":
                    result["heating_active"] = True
                    result["heating_relays"].append(name)

        # Set active flag if EV charging or heating is active
        result["active"] = result["ev_charging"] or result["heating_active"]

        return result

    async def _handle_high_load_start(self) -> None:
        """Handle high loads by applying decision tree logic."""
        try:
            self.logger.info("⚡ High loads detected - using decision tree to determine action")
            # The decision tree will handle this - just re-evaluate current state
            await self._determine_and_apply_mode()
        except Exception as e:
            self.logger.error(f"Failed to handle high load start: {e}", exc_info=True)

    async def _fetch_prices(self) -> None:
        """Fetch current energy prices and trigger evaluation."""
        try:
            # Determine target date
            target_date = self._get_local_date_string(days_ahead=0)

            # Fetch energy prices from DAM
            hourly_prices = await self._price_analyzer.fetch_dam_energy_prices(date=target_date)

            if not hourly_prices:
                self.logger.warning("Failed to retrieve energy prices, using mock prices")
                hourly_prices = self._price_analyzer.generate_mock_prices(target_date)

            if hourly_prices:
                # Store prices in cache
                self._current_prices = hourly_prices
                self._prices_date = target_date
                self._prices_updated = datetime.now()

                # Get exchange rate
                self._eur_czk_rate = await self._get_eur_czk_rate()

                # Log price table
                await self._log_price_table(hourly_prices, target_date, self._eur_czk_rate or 25.0)

                # Trigger re-evaluation with new prices
                await self._on_price_update()
            else:
                self.logger.error("Failed to get any prices")

        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}", exc_info=True)

    async def _on_price_update(self) -> None:
        """Handle new price data availability."""
        await self._evaluate_conditions("price_update")

    async def _on_load_threshold_crossed(self, high_loads: bool) -> None:
        """Handle load threshold crossing.

        Args:
            high_loads: True if loads are now high, False if normal
        """
        if self._high_loads_active != high_loads:
            self._high_loads_active = high_loads
            reason = "high_load_detected" if high_loads else "load_normalized"
            await self._evaluate_conditions(reason)

    async def _on_manual_override_change(self) -> None:
        """Handle manual override changes."""
        await self._evaluate_conditions("manual_override")

    async def _apply_current_state(self) -> None:
        """Apply the currently appropriate mode based on conditions.

        This is now handled by the evaluation system.
        """
        await self._evaluate_conditions("state_sync")

    # Price analysis methods moved to PriceAnalyzer module

    # Mode management methods moved to ModeManager module

    async def _evaluate_conditions(self, reason: str) -> None:
        """Re-evaluate decision tree when conditions change.

        Args:
            reason: What triggered the re-evaluation
        """
        async with self._evaluation_lock:
            try:
                self._last_evaluation_reason = reason

                # Build complete decision context with price data
                context = await self._build_decision_context()

                # Get decision from engine
                new_mode = self._decision_engine.decide(context)

                # Check if mode changed
                if self._decision_engine.has_mode_changed(new_mode):
                    # Get explanation for logging
                    explanation = self._decision_engine.explain_decision()
                    self.logger.info(
                        f"📊 Mode change ({reason}): {self._current_mode} → {new_mode} - "
                        f"Reason: {explanation['reason']} "
                        f"(Priority: {explanation['priority']['name']})"
                    )

                    # Apply the new mode
                    await self._apply_decided_mode(new_mode)
                    self._current_mode = new_mode
                else:
                    self.logger.debug(f"Evaluation ({reason}): Mode unchanged ({new_mode})")

            except Exception as e:
                self.logger.error(f"Failed to evaluate conditions: {e}", exc_info=True)
                # Don't change mode on error

    async def _build_decision_context(self) -> DecisionContext:
        """Build complete decision context with all current data.

        Returns:
            DecisionContext with current system state and price data
        """
        now = self._get_local_now()
        current_hour = now.strftime("%H:00")
        next_hour = (now.replace(minute=0) + timedelta(hours=1)).strftime("%H:00")
        current_hour_key = (current_hour, next_hour)

        # Get current price
        current_price = self._current_prices.get(current_hour_key, 0.0)

        # Calculate sunrise/sunset
        sun_times = sun(self._location.observer, date=now.date())
        sunrise = sun_times["sunrise"].time() if "sunrise" in sun_times else None
        sunset = sun_times["sunset"].time() if "sunset" in sun_times else None

        # Create price thresholds from config
        price_thresholds = PriceThresholds(
            cheap_threshold=self.config.cheap_price_threshold_eur,
            export_threshold=self.config.export_enable_threshold_eur,  # EUR/MWh threshold
            charge_efficiency=self.config.charge_efficiency,
            min_profit_margin=self.config.min_profit_margin,
            # Add percentile-based thresholds with defaults or from config
            charge_percentile_threshold=getattr(
                self.config, 'charge_percentile_threshold', 25.0
            ),
            # Winter mode settings
            winter_cheapest_hours=getattr(
                self.config, 'battery_charge_hours', 2
            ),
            # Smart discharge control
            discharge_min_price_czk=getattr(
                self.config, 'discharge_min_price_czk', 2.0
            ),
            discharge_price_multiplier=getattr(
                self.config, 'discharge_price_multiplier', 3.0
            )
        )

        # Calculate price ranking for current hour
        price_ranking = None
        if self._current_prices and current_hour_key in self._current_prices:
            price_ranking = self._decision_engine.calculate_price_ranking(
                current_hour_key, self._current_prices
            )
            if price_ranking:
                self.logger.debug(
                    f"Price ranking: rank {price_ranking.current_rank}/"
                    f"{price_ranking.total_hours} "
                    f"(percentile {price_ranking.percentile:.1f}%, "
                    f"{price_ranking.price_quadrant}), "
                    f"spread {price_ranking.daily_spread:.2f} EUR/MWh"
                )

        return DecisionContext(
            # Manual control
            manual_override_active=bool(self._manual_override_period),
            manual_override_mode=(
                self._manual_override_period.kind if self._manual_override_period else None
            ),
            # System state
            high_loads_active=self._high_loads_active,
            battery_soc=self._battery_soc,
            current_mode=self._current_mode,
            current_load=self._current_load,
            solar_power=self._solar_power,
            # Time and pricing
            current_time=now,
            current_price=current_price,
            hourly_prices=self._current_prices.copy(),
            price_thresholds=price_thresholds,
            price_ranking=price_ranking,  # Include price ranking
            # Solar schedule
            sunrise=sunrise,
            sunset=sunset,
            is_summer_mode=(await self._get_season_mode() == "summer")
        )

    async def _apply_decided_mode(self, mode: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Apply a mode based on its definition from MODE_DEFINITIONS.

        Args:
            mode: Mode name from decision engine
            params: Optional parameters for configurable modes
        """
        params = params or {}

        # Get mode definition
        mode_def = MODE_DEFINITIONS.get(mode)
        if not mode_def:
            self.logger.error(f"Unknown mode: {mode}, falling back to regular")
            mode = "regular"
            mode_def = MODE_DEFINITIONS["regular"]

        # Determine actual SOC value
        stop_soc_raw = mode_def.get("stop_soc", 20)
        stop_soc: int = 20
        if stop_soc_raw == "configurable":
            soc_param = params.get("stop_soc", self.config.max_soc)
            stop_soc = int(soc_param) if soc_param is not None else int(self.config.max_soc)
        elif isinstance(stop_soc_raw, (int, float, str)):
            stop_soc = int(stop_soc_raw)

        inverter_mode = mode_def.get("inverter_mode", "load_first")

        # Apply the inverter mode with appropriate SOC
        if inverter_mode == "load_first":
            await self._mode_manager.set_load_first(stop_soc=stop_soc)
        elif inverter_mode == "battery_first":
            power_rate = int(params.get("power_rate", 100))
            await self._mode_manager.set_battery_first(
                "00:00", "23:59", stop_soc=stop_soc, power_rate=power_rate
            )
        elif inverter_mode == "grid_first":
            power_rate = int(params.get(
                "power_rate",
                self.config.discharge_power_rate
                if hasattr(self.config, 'discharge_power_rate') else 100
            ))
            await self._mode_manager.set_grid_first(
                "00:00", "23:59", stop_soc=stop_soc, power_rate=power_rate
            )

        await asyncio.sleep(0.5)

        # Set export state
        if mode_def.get("export", True):
            await self._mode_manager.enable_export()
        else:
            await self._mode_manager.disable_export()

        # Set AC charging state
        if mode_def.get("ac_charge", False):
            await self._mode_manager.enable_ac_charge()
        else:
            await self._mode_manager.disable_ac_charge()

        # Track if we're in high load protected mode
        self._high_load_protected_mode_active = (mode == "high_load_protected")

        self.logger.info(
            f"✅ Applied mode: {mode.upper()} - "
            f"{mode_def.get('description', 'No description')} "
            f"({inverter_mode} @ {stop_soc}% SOC)"
        )

    # Event-driven evaluation methods

    async def _periodic_evaluation_loop(self) -> None:
        """Periodically check for condition changes that need re-evaluation."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = self._get_local_now()

                # Check for hour change (price changes)
                current_hour = now.hour
                if self._last_evaluation_hour != current_hour:
                    self._last_evaluation_hour = current_hour
                    await self._evaluate_conditions("hour_change")

                # Check for battery SOC change (>5%)
                if abs(self._battery_soc - self._last_battery_soc) >= 5:
                    self._last_battery_soc = self._battery_soc
                    await self._evaluate_conditions("battery_soc_change")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic evaluation: {e}", exc_info=True)

    async def set_manual_override(
        self,
        mode: PeriodType,
        duration_type: str,
        duration_value: Optional[Union[str, int]] = None,
        params: Optional[Dict[str, Any]] = None,
        source: str = "api"
    ) -> Dict[str, Any]:
        """Set manual mode override.

        Args:
            mode: One of the 5 composite modes
            duration_type: "immediate", "end_of_day", "duration_hours", or "until_time"
            duration_value: Value for duration (hours for duration_hours, "HH:MM" for until_time)
            params: Optional parameters (stop_soc, power_rate for discharge/charge modes)
            source: Source of the override ("api" or "dashboard")

        Returns:
            Dict with success status and details
        """
        # Validate mode
        valid_modes = [
            "regular", "sell_production", "regular_no_export",
            "charge_from_grid", "discharge_to_grid"
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

        # Calculate end time based on duration type
        now = self._get_local_now()

        if duration_type == "immediate":
            end_time = None  # No end time, runs until manually cleared
        elif duration_type == "end_of_day":
            end_time = now.replace(hour=23, minute=59, second=0, microsecond=0)
        elif duration_type == "duration_hours":
            hours = int(duration_value) if duration_value else 1
            # Limit to 24 hours maximum
            hours = min(24, max(1, hours))
            end_time = now + timedelta(hours=hours)
        elif duration_type == "until_time":
            if not duration_value:
                raise ValueError("Time value required for until_time duration")
            # Parse time string "HH:MM"
            if isinstance(duration_value, int):
                duration_value = str(duration_value)
            time_parts = duration_value.split(":")
            if len(time_parts) != 2:
                raise ValueError(f"Invalid time format: {duration_value}")
            target_time = now.replace(
                hour=int(time_parts[0]),
                minute=int(time_parts[1]),
                second=0,
                microsecond=0
            )
            # If time is in past, assume tomorrow
            if target_time <= now:
                target_time += timedelta(days=1)
            end_time = target_time
        else:
            raise ValueError(f"Invalid duration type: {duration_type}")

        # Set default params if not provided based on mode
        if params is None:
            params = {}

        # Set sensible defaults for modes that need params
        if mode == "charge_from_grid":
            params.setdefault("stop_soc", self.config.max_soc)
            params.setdefault("power_rate", 100)
        elif mode == "discharge_to_grid":
            params.setdefault("stop_soc", self.config.discharge_min_soc)
            params.setdefault("power_rate", self.config.discharge_power_rate)

        # Create manual override period
        self._manual_override_period = Period(
            kind=mode,
            start=dt_time(0, 0),  # Start time doesn't matter for manual override
            end=dt_time(23, 59, 59),  # End time doesn't matter, we use _manual_override_end_time
            params=params,
            manual=True,
            source=source
        )
        self._manual_override_end_time = end_time
        self._manual_override_source = source

        # Apply the manual override mode immediately
        try:
            await self._on_manual_override_change()

            # Log manual intervention
            end_str = end_time.strftime("%Y-%m-%d %H:%M") if end_time else "manual clear"
            self.logger.info(
                f"🎯 Manual override set: {mode} until {end_str} "
                f"(source: {source}, params: {params})"
            )

            return {
                "success": True,
                "mode": mode,
                "end_time": end_time.isoformat() if end_time else None,
                "source": source,
                "params": params
            }

        except Exception as e:
            # Clear override on error
            self._manual_override_period = None
            self._manual_override_end_time = None
            self._manual_override_source = ""
            self.logger.error(f"Failed to set manual override: {e}")
            raise

    async def clear_manual_override(self) -> Dict[str, Any]:
        """Clear manual override and return to automatic control."""
        had_override = self._manual_override_period is not None

        # Clear override
        self._manual_override_period = None
        self._manual_override_end_time = None
        self._manual_override_source = ""

        if had_override:
            # Re-evaluate automatic schedule
            self.logger.info("🔄 Manual override cleared, re-evaluating automatic schedule")
            await self._evaluate_conditions("manual_override_cleared")

            return {
                "success": True,
                "message": "Manual override cleared, returned to automatic control"
            }
        else:
            return {
                "success": True,
                "message": "No manual override was active"
            }

    def get_manual_override_status(self) -> Dict[str, Any]:
        """Get current manual override status."""
        if not self._manual_override_period:
            return {"active": False}

        # Check if override is still valid
        if self._manual_override_end_time:
            now = self._get_local_now()
            if now >= self._manual_override_end_time:
                # Override expired
                return {"active": False, "expired": True}

        return {
            "active": True,
            "mode": self._manual_override_period.kind,
            "end_time": (self._manual_override_end_time.isoformat()
                         if self._manual_override_end_time else None),
            "source": self._manual_override_source,
            "params": self._manual_override_period.params
        }

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
        hourly_prices = await self._price_analyzer.fetch_dam_energy_prices(date=target_date)

        if not hourly_prices:
            self.logger.warning(
                "Failed to retrieve real energy prices. Using mock prices for testing."
            )
            # Use mock prices as fallback
            hourly_prices = self._price_analyzer.generate_mock_prices(target_date)

            if not hourly_prices:
                self.logger.error(
                    "Failed to generate mock prices. Setting load-first mode and disabling export."
                )
                # Schedule fallback mode
                await self._schedule_fallback_mode()
                # Apply safe state immediately
                self._scheduled_mode = "regular_no_export"  # Safe mode without selling
                await self._determine_and_apply_mode()
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

        # Apply the current state immediately after scheduling
        # This ensures we're in the correct mode right after midnight calculation
        self.logger.info("Applying current state after scheduling...")
        await self._apply_current_state()

    async def _schedule_winter_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule winter battery strategy with AC charging during cheapest hours."""
        # Analyze prices (existing logic)
        cheapest_individual_hours = set(
            (start, stop)
            for start, stop, _ in self._price_analyzer.find_n_cheapest_hours(
                hourly_prices, n=self.config.individual_cheapest_hours
            )
        )

        quadrants = self._price_analyzer.categorize_prices_into_quadrants(hourly_prices)
        cheapest_quadrant_hours = set(
            (start, stop) for start, stop, _ in quadrants.get("Cheapest", [])
        )

        # Combine cheapest hours
        cheapest_individual_hours = cheapest_individual_hours.union(cheapest_quadrant_hours)

        cheapest_consecutive = set(
            (start, stop)
            for start, stop, _ in self._price_analyzer.find_cheapest_consecutive_hours(
                hourly_prices, self.config.battery_charge_hours
            )
        )

        # Union all cheap hours
        all_cheap_hours = cheapest_individual_hours.union(cheapest_consecutive)

        # Calculate price statistics
        all_prices = list(hourly_prices.values())
        if not all_prices:
            self.logger.warning("No prices available for scheduling")
            return
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
        # Schedule regular mode as fallback
        async def apply_fallback_mode() -> None:
            self._scheduled_mode = "regular"
            await self._determine_and_apply_mode()
        task = asyncio.create_task(
            self._schedule_at_time("00:00", apply_fallback_mode)
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
                    Period("regular", dt_time(0, 0), sunrise_time)  # Regular mode overnight
                )

                # Schedule regular mode at midnight
                async def apply_overnight_mode() -> None:
                    self._scheduled_mode = "regular"
                    await self._determine_and_apply_mode()
                task = self._schedule_action("00:00", apply_overnight_mode)
                self._scheduled_tasks.append(task)

            # Schedule Grid-First from sunrise to end of day
            # (sell solar + battery at 10% rate)
            self.logger.info(
                f"Scheduling grid-first from {sunrise_str} to {EOD_HHMM} "
                f"(sell solar + battery, stopSOC=20%, powerRate=10%)"
            )
            self._scheduled_periods.append(
                Period("discharge_to_grid", sunrise_time, EOD_DTTIME,
                       params={
                           "stop_soc": self.config.discharge_min_soc,
                           "power_rate": self.config.discharge_power_rate
                       })
            )

            # Schedule discharge mode at sunrise
            async def apply_discharge_mode() -> None:
                self._scheduled_mode = "discharge_to_grid"
                await self._determine_and_apply_mode()
            task = self._schedule_action(sunrise_str, apply_discharge_mode)
            self._scheduled_tasks.append(task)
            return

        # Group low-price hours into contiguous periods
        low_price_groups = self._price_analyzer.group_contiguous_hours_simple(low_price_hours)

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
                Period("regular", dt_time(0, 0), sunrise_time)  # Regular mode overnight
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
                # Overnight prices above threshold - use regular mode with export
                pass  # Export is inherent in regular mode

            # Schedule regular mode at midnight
            async def apply_overnight_mode() -> None:
                self._scheduled_mode = "regular"
                await self._determine_and_apply_mode()
            task = self._schedule_action("00:00", apply_overnight_mode)
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
                f"(sell solar only - battery preserved at 100%, powerRate=10%)"
            )
            self._scheduled_periods.append(
                Period(
                    "sell_production",  # Use sell_production instead of grid_first
                    sunrise_time,
                    self._parse_hhmm(grid_first_end),
                    params={
                        "stop_soc": 100,  # Preserve battery, only sell solar production
                        "power_rate": self.config.discharge_power_rate
                    }
                )
            )
            # Export is handled by the sell_production mode itself

            # Set grid-first at sunrise with stopSOC=100% to preserve battery
            task = self._schedule_action(
                sunrise_str, self._emit_device_window,
                self._mode_manager.set_grid_first, sunrise_str, grid_first_end, 100, 10
            )
            self._scheduled_tasks.append(task)

            # Enable export during morning high prices
            task = self._schedule_action(
                self._bump_time(sunrise_str, 5), self._mode_manager.enable_export
            )
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
                            "regular",  # Use regular instead of load_first
                            self._parse_hhmm(previous_end),
                            self._parse_hhmm(group_start)
                        )
                    )
                    # Removed export Period - export is controlled via mode
                    # settings - Track export period
                    task = self._schedule_action(previous_end, self._mode_manager.set_load_first)
                    self._scheduled_tasks.append(task)

                    # Re-enable export during the gap (prices above threshold)
                    enable_time = self._bump_time(previous_end, 5)
                    task = self._schedule_action(enable_time, self._mode_manager.enable_export)
                    self._scheduled_tasks.append(task)

            # Schedule battery-first for this low-price period
            self.logger.info(
                f"Scheduling battery-first from {group_start} to {group_end} "
                f"(store cheap solar, no AC charge)"
            )
            # During low prices, use regular_no_export (store solar, don't sell)
            self._scheduled_periods.append(
                Period(
                    "regular_no_export",
                    self._parse_hhmm(group_start),
                    self._parse_hhmm(group_end)
                )
            )

            # Schedule mode switch at start of low price period
            async def apply_low_price_mode() -> None:
                self._scheduled_mode = "regular_no_export"
                await self._determine_and_apply_mode()
            task = self._schedule_action(group_start, apply_low_price_mode)
            self._scheduled_tasks.append(task)

            previous_end = group_end

        # Schedule load-first for evening (after last low price)
        if not is_24(last_low_end):
            self.logger.info(
                f"Scheduling load-first from {last_low_end} to {EOD_HHMM} (use stored energy)"
            )
            self._scheduled_periods.append(
                Period("regular", self._parse_hhmm(last_low_end), EOD_DTTIME)  # Regular for evening
            )

            # Schedule evening mode
            async def apply_evening_mode() -> None:
                self._scheduled_mode = "regular"
                await self._determine_and_apply_mode()
            task = self._schedule_action(last_low_end, apply_evening_mode)
            self._scheduled_tasks.append(task)

    async def _schedule_battery_control(
        self,
        all_cheap_hours: set[Tuple[str, str]],
        cheapest_consecutive: set[Tuple[str, str]],
        hourly_prices: Dict[Tuple[str, str], float],
        eur_czk_rate: float,
    ) -> None:
        """Schedule battery control based on price analysis using composite modes."""
        # Schedule charge_from_grid during cheapest consecutive hours
        if cheapest_consecutive:
            charge_hours = [
                (start, stop, hourly_prices[(start, stop)]) for start, stop in cheapest_consecutive
            ]
            charge_groups = self._price_analyzer.group_contiguous_hours_simple(charge_hours)

            for start_time, stop_time in charge_groups:
                # Calculate price statistics for this charging period
                start_t = self._parse_hhmm(start_time)
                stop_t = self._parse_hhmm(stop_time)
                charge_prices = [
                    price
                    for start, stop, price in charge_hours
                    if self._parse_hhmm(start) >= start_t and self._parse_hhmm(stop) <= stop_t
                ]
                if charge_prices:
                    avg_charge_price = sum(charge_prices) / len(charge_prices)

                    # Convert to CZK/kWh for display
                    avg_charge_price_czk = avg_charge_price * eur_czk_rate / 1000

                    self.logger.info(
                        f"Scheduling CHARGE_FROM_GRID from {start_time} to {stop_time} "
                        f"(avg: {avg_charge_price:.2f} EUR/MWh = "
                        f"{avg_charge_price_czk:.2f} CZK/kWh)"
                    )

                # Schedule charge_from_grid composite mode
                self._scheduled_periods.append(
                    Period(
                        "charge_from_grid",
                        self._parse_hhmm(start_time),
                        self._parse_hhmm(stop_time),
                        params={"stop_soc": self.config.max_soc})
                )
                # Create closure to apply mode at scheduled time

                async def apply_charge_mode() -> None:
                    self._scheduled_mode = "charge_from_grid"
                    await self._determine_and_apply_mode()
                task = self._schedule_action(start_time, apply_charge_mode)
                self._scheduled_tasks.append(task)

        # Smart discharge economics: only discharge when profitable
        # Calculate average charging cost from scheduled charge periods
        avg_charge_price = 0.0
        if cheapest_consecutive:
            charge_prices = [hourly_prices[(s, e)] for s, e in cheapest_consecutive]
            avg_charge_price = sum(charge_prices) / len(charge_prices) if charge_prices else 0.0

            # Calculate minimum profitable discharge price
            # Account for battery round-trip efficiency and required profit margin
            min_discharge_price = (
                (avg_charge_price / self.config.battery_efficiency)
                * self.config.discharge_profit_margin
            )

            # Log economics calculation
            charge_czk = avg_charge_price * eur_czk_rate / 1000
            min_discharge_czk = min_discharge_price * eur_czk_rate / 1000
            self.logger.info(
                f"Battery economics: Charge cost={charge_czk:.2f} CZK/kWh, "
                f"Min discharge price={min_discharge_czk:.2f} CZK/kWh "
                f"(efficiency={self.config.battery_efficiency:.0%}, "
                f"margin={self.config.discharge_profit_margin-1:.0%})"
            )
        else:
            # No charging scheduled, use average price as baseline
            avg_price = sum(hourly_prices.values()) / len(hourly_prices)
            min_discharge_price = avg_price * 1.3  # Require 30% premium without known charge cost

        # Find most expensive hours that exceed minimum profitable price
        most_expensive_hours = self._price_analyzer.find_n_most_expensive_hours(
            hourly_prices, n=min(6, len(hourly_prices) // 4)  # Max 6 hours or 1/4 of day
        )

        # Filter for profitable discharge only
        discharge_hours = [
            (start, stop, price)
            for start, stop, price in most_expensive_hours
            if price > min_discharge_price and (start, stop) not in cheapest_consecutive
        ]

        if discharge_hours:
            # Group consecutive hours for cleaner scheduling
            discharge_groups = self._price_analyzer.group_contiguous_hours_simple(discharge_hours)
            for start_time, stop_time in discharge_groups:
                # Calculate average price for this discharge period
                period_prices = [p for s, e, p in discharge_hours
                                 if self._parse_hhmm(s) >= self._parse_hhmm(start_time)
                                 and self._parse_hhmm(e) <= self._parse_hhmm(stop_time)]
                avg_period_price = sum(period_prices) / len(period_prices) if period_prices else 0
                avg_period_czk = avg_period_price * eur_czk_rate / 1000

                # Calculate profit margin for this discharge period
                if avg_charge_price > 0:
                    effective_charge_cost = avg_charge_price / self.config.battery_efficiency
                    profit_ratio = avg_period_price / effective_charge_cost
                    profit_pct = (profit_ratio - 1) * 100
                    self.logger.info(
                        f"Scheduling DISCHARGE_TO_GRID from {start_time} to {stop_time} "
                        f"(price: {avg_period_price:.2f} EUR/MWh = {avg_period_czk:.2f} CZK/kWh, "
                        f"profit: {profit_pct:.1f}%, power: {self.config.discharge_power_rate}%)"
                    )
                else:
                    self.logger.info(
                        f"Scheduling DISCHARGE_TO_GRID from {start_time} to {stop_time} "
                        f"(price: {avg_period_price:.2f} EUR/MWh = {avg_period_czk:.2f} CZK/kWh)"
                    )
                self._scheduled_periods.append(
                    Period(
                        "discharge_to_grid",
                        self._parse_hhmm(start_time),
                        self._parse_hhmm(stop_time),
                        params={
                            "stop_soc": self.config.discharge_min_soc,
                            "power_rate": self.config.discharge_power_rate
                        })
                )
                # Create closure to apply mode at scheduled time

                async def apply_discharge_mode() -> None:
                    self._scheduled_mode = "discharge_to_grid"
                    await self._determine_and_apply_mode()
                task = self._schedule_action(start_time, apply_discharge_mode)
                self._scheduled_tasks.append(task)

        # Fill gaps with regular or regular_no_export based on price
        # Calculate threshold for export control
        threshold_eur_mwh = self.config.export_price_threshold * 1000 / eur_czk_rate

        all_scheduled_times = set()
        for period in self._scheduled_periods:
            # Add all hours covered by scheduled periods
            start_hour = period.start.hour
            end_hour = period.end.hour if period.end != dt_time(0, 0) else 24
            if start_hour <= end_hour:
                for h in range(start_hour, end_hour):
                    all_scheduled_times.add(f"{h:02d}:00")
            else:
                # Wraps midnight
                for h in range(start_hour, 24):
                    all_scheduled_times.add(f"{h:02d}:00")
                for h in range(0, end_hour):
                    all_scheduled_times.add(f"{h:02d}:00")

        # Group regular modes for unscheduled hours
        regular_hours = []
        no_export_hours = []

        for (start, stop), price in sorted(hourly_prices.items()):
            if start not in all_scheduled_times:
                if price < threshold_eur_mwh:
                    # Low price - don't export
                    no_export_hours.append((start, stop, price))
                else:
                    # Normal/high price - allow export
                    regular_hours.append((start, stop, price))

        # Group consecutive regular hours into multi-hour periods
        if regular_hours:
            regular_groups = self._price_analyzer.group_contiguous_hours_simple(regular_hours)
            for group_start, group_end in regular_groups:
                self._scheduled_periods.append(
                    Period(
                        "regular",
                        self._parse_hhmm(group_start),
                        self._parse_hhmm(group_end)
                    )
                )
            self.logger.info(
                f"Scheduled {len(regular_groups)} regular period(s) "
                f"covering {len(regular_hours)} hours"
            )

        # Group consecutive no-export hours into multi-hour periods
        if no_export_hours:
            no_export_groups = self._price_analyzer.group_contiguous_hours_simple(no_export_hours)
            for group_start, group_end in no_export_groups:
                self._scheduled_periods.append(
                    Period(
                        "regular_no_export",
                        self._parse_hhmm(group_start),
                        self._parse_hhmm(group_end)
                    )
                )
            self.logger.info(
                f"Scheduled {len(no_export_groups)} no-export period(s) "
                f"covering {len(no_export_hours)} hours"
            )

        # Export control is now handled by composite modes
        # No need for separate export scheduling

        # Ensure clean end-of-day state
        try:
            self._schedule_end_of_day_cleanup()
        except Exception as e:
            self.logger.error(f"Error scheduling end-of-day cleanup: {e}", exc_info=True)

    async def _schedule_export_control(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float  # noqa: F841
    ) -> None:
        """Schedule export control based on price analysis.

        DEPRECATED: Export control is now handled by composite modes in the decision engine.
        - "regular" and "sell_production" modes have export enabled
        - "regular_no_export" mode has export disabled
        - "charge_from_grid" and "discharge_to_grid" control export as needed

        This method is kept as a no-op for backward compatibility.
        """
        pass

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
            self.logger.error(
                f"Error in scheduled task at {time_str}: {e}", exc_info=True
            )

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
            self.logger.error(
                f"Error in scheduled task at {time_str}: {e}", exc_info=True
            )

    def _schedule_end_of_day_cleanup(self) -> None:
        """Ensure we land in a neutral state at end of day."""
        # Small jittered disables are safe & idempotent.
        self._scheduled_tasks.append(
            asyncio.create_task(
                self._schedule_at_time(EOD_HHMMSS, self._mode_manager.disable_export)
            )
        )
        self._scheduled_tasks.append(
            asyncio.create_task(
                self._schedule_at_time(EOD_HHMMSS, self._mode_manager.disable_ac_charge)
            )
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
