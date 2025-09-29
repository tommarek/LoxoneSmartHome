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
    EOD_HHMM, EOD_HHMMSS,
    is_24
)
from .growatt.price_analyzer import PriceAnalyzer
from .growatt.mode_manager import ModeManager
from .growatt.decision_engine import (
    GrowattDecisionEngine, DecisionContext, PriceThresholds, MODE_DEFINITIONS
)
from .growatt.inverter_state import InverterState


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    # Class attributes for type checking
    _manual_override_period: Optional[Period]
    _manual_override_end_time: Optional[datetime]
    _manual_override_source: str
    _optional_config: Dict[str, Any]

    # State tracking for preventing redundant commands
    _current_inverter_state: Optional[InverterState] = None
    _last_commanded_state: Optional[InverterState] = None
    _state_history: list[InverterState]
    _commands_sent_count: int = 0
    _commands_skipped_count: int = 0

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
            "export_price_min",
            "battery_charge_hours",
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

        # Initialize state tracking
        self._current_inverter_state = None
        self._last_commanded_state = None
        self._state_history = []
        self._commands_sent_count = 0
        self._commands_skipped_count = 0

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

        # Create price thresholds from config (all in CZK/kWh)
        price_thresholds = PriceThresholds(
            charge_price_max=getattr(self.config, 'charge_price_max', 1.5),
            export_price_min=getattr(self.config, 'export_price_min', 1.0),
            discharge_price_min=getattr(self.config, 'discharge_price_min', 3.0),
            discharge_profit_margin=getattr(self.config, 'discharge_profit_margin', 1.5),
            battery_efficiency=getattr(self.config, 'battery_efficiency', 0.85)
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

    async def _build_desired_state(
        self, mode: str, params: Optional[Dict[str, Any]] = None
    ) -> InverterState:
        """Build the complete desired inverter state.

        Args:
            mode: Mode name from decision engine
            params: Optional parameters for configurable modes

        Returns:
            Complete InverterState representing desired configuration
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

        # Determine power rate
        default_rate = 100
        if hasattr(self.config, 'discharge_power_rate'):
            default_rate = self.config.discharge_power_rate
        power_rate = int(params.get("power_rate", default_rate))

        # Map mode to inverter mode
        inverter_mode = str(mode_def.get("inverter_mode", "load_first"))

        # Determine export based on price
        export_enabled = True  # Default
        try:
            context = await self._build_decision_context()
            if context.price_thresholds and context.current_price > 0:
                current_price_czk = context.current_price * 25 / 1000  # EUR/MWh to CZK/kWh
                export_enabled = current_price_czk >= context.price_thresholds.export_price_min
        except Exception as e:
            self.logger.debug(f"Could not determine export state: {e}, defaulting to enabled")

        # AC charge from mode definition
        ac_charge_enabled = bool(mode_def.get("ac_charge", False))

        return InverterState(
            inverter_mode=inverter_mode,
            stop_soc=stop_soc,
            power_rate=power_rate,
            time_start="00:00",
            time_stop="23:59",
            ac_charge_enabled=ac_charge_enabled,
            export_enabled=export_enabled,
            timestamp=self._get_local_now(),
            source="evaluation"
        )

    async def _apply_state_changes(
        self,
        old_state: Optional[InverterState],
        new_state: InverterState
    ) -> None:
        """Apply only the changes between old and new state.

        Args:
            old_state: Previous state (None for initial)
            new_state: Desired new state
        """
        # Mode change (inverter_mode, stop_soc, power_rate, time window)
        mode_changed = (
            not old_state or
            old_state.inverter_mode != new_state.inverter_mode or
            old_state.stop_soc != new_state.stop_soc or
            old_state.power_rate != new_state.power_rate or
            old_state.time_start != new_state.time_start or
            old_state.time_stop != new_state.time_stop
        )

        if mode_changed:
            self._commands_sent_count += 1
            if new_state.inverter_mode == "load_first":
                await self._mode_manager.set_load_first(stop_soc=new_state.stop_soc)
            elif new_state.inverter_mode == "battery_first":
                await self._mode_manager.set_battery_first(
                    new_state.time_start,
                    new_state.time_stop,
                    stop_soc=new_state.stop_soc,
                    power_rate=new_state.power_rate
                )
            elif new_state.inverter_mode == "grid_first":
                await self._mode_manager.set_grid_first(
                    new_state.time_start,
                    new_state.time_stop,
                    stop_soc=new_state.stop_soc,
                    power_rate=new_state.power_rate
                )

            await asyncio.sleep(0.5)

        # Export change
        if not old_state or old_state.export_enabled != new_state.export_enabled:
            self._commands_sent_count += 1
            if new_state.export_enabled:
                await self._mode_manager.enable_export()
                self.logger.info("Export ENABLED (price-based)")
            else:
                await self._mode_manager.disable_export()
                self.logger.info("Export DISABLED (price-based)")

        # AC charge change
        if not old_state or old_state.ac_charge_enabled != new_state.ac_charge_enabled:
            self._commands_sent_count += 1
            await self._mode_manager.set_ac_charge(new_state.ac_charge_enabled)

    async def _apply_decided_mode(self, mode: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Apply a mode based on its definition from MODE_DEFINITIONS.

        Args:
            mode: Mode name from decision engine
            params: Optional parameters for configurable modes
        """
        # Build desired state from mode and current conditions
        desired_state = await self._build_desired_state(mode, params)

        # Check if anything actually changed
        if self._current_inverter_state and desired_state == self._current_inverter_state:
            self._commands_skipped_count += 1
            self.logger.debug(
                f"State unchanged after evaluation: {desired_state.summary()} "
                f"(skipped: {self._commands_skipped_count}, sent: {self._commands_sent_count})"
            )
            return

        # Identify what changed
        if self._current_inverter_state:
            changes = desired_state.significant_changes(self._current_inverter_state)
            self.logger.info(f"📝 Applying changes: {', '.join(changes)}")
        else:
            self.logger.info(f"📝 Applying initial state: {mode}")

        # Apply ONLY the changes needed
        await self._apply_state_changes(self._current_inverter_state, desired_state)

        # Update tracking
        self._current_inverter_state = desired_state
        self._state_history.append(desired_state)
        if len(self._state_history) > 10:
            self._state_history.pop(0)

        # Track if we're in high load protected mode
        self._high_load_protected_mode_active = (mode == "high_load_protected")

        self.logger.info(f"✅ State applied: {desired_state.summary()}")

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
            "regular", "sell_production",
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
        st = self._parse_time_any(start_hhmm)
        en = self._parse_time_any(stop_hhmm)
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
