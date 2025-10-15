"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
import zoneinfo
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from .growatt.command_queue import CommandQueue


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
            "battery_charge_blocks",
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

        # Note: All config validation is handled by Pydantic Field constraints

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
        self._cheapest_charging_blocks: Set[Tuple[str, str]] = set()
        self._pre_discharge_blocks: Set[Tuple[str, str]] = set()
        self._peak_to_precharge_map: Dict[str, List[Tuple[str, str, float]]] = {}
        self._combined_charging_blocks: Set[Tuple[str, str]] = set()
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
        # Track 15-minute block changes
        self._last_evaluation_block: Optional[Tuple[int, int]] = None
        self._current_load: float = 0.0  # Current home load in kW
        self._solar_power: float = 0.0  # Current solar generation in kW
        self._last_price_fetch: Optional[datetime] = None  # Track price update time

        # Next-day price fetching state
        self._next_day_prices_fetched: bool = False  # Flag: have we fetched next day's prices?
        self._next_day_prices: Dict[Tuple[str, str], float] = {}  # Store next day's prices
        self._next_day_prices_date: Optional[str] = None  # Date of next day prices
        self._price_fetch_task: Optional[asyncio.Task[None]] = None  # Retry task for price fetching
        self._last_price_fetch_attempt: Optional[datetime] = None  # Last fetch attempt time

        # Command queue for managing multiple concurrent commands
        self._command_queue = CommandQueue(self.logger)

        # Pre-register MQTT subscriptions before connection
        # This ensures subscriptions are active before the message loop starts
        if self.mqtt_client:
            self.mqtt_client.register_subscription("energy/solar/result", self._result_handler)
            self.logger.info("Pre-registered subscription for command results: energy/solar/result")

            self.mqtt_client.register_subscription(self._home_status_topic, self._on_home_status)
            self.logger.info(
                f"Pre-registered subscription for home status: {self._home_status_topic}"
            )

    async def _result_handler(self, _topic: str, payload: Any) -> None:
        """Persistent handler for energy/solar/result MQTT messages.

        This handler is always listening and matches responses to pending commands
        in the command queue. It handles multiple concurrent commands properly.
        """
        try:
            if isinstance(payload, bytes):
                payload = payload.decode()

            data = json.loads(payload)
            command = data.get("command")
            success = data.get("success", False)

            # Try to resolve a pending command in the queue
            resolved = await self._command_queue.resolve_command(command, data)

            # Log the result
            message = data.get("message", "No message")
            if resolved:
                if success:
                    self.logger.debug(f"✅ Command {command} succeeded: {message}")
                else:
                    self.logger.error(f"❌ Command {command} FAILED: {message}")
                    self.logger.error(f"📋 Full error response: {json.dumps(data, indent=2)}")
            else:
                # Orphaned response (no pending command)
                self.logger.debug(
                    f"Received orphaned response for {command}: {message} "
                    f"(no pending command or already resolved)"
                )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse MQTT result payload: {e}")
            self.logger.error(f"Raw payload: {payload}")
        except Exception as e:
            self.logger.error(f"Error in result handler: {e}", exc_info=True)

    async def _send_command_and_wait(
        self,
        command_topic: str,
        command_type: str,
        payload: Dict[str, Any],
        timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Send MQTT command and wait for response using command queue.

        This method uses the command queue to track pending commands,
        allowing for multiple concurrent commands and proper retry handling.

        Args:
            command_topic: MQTT topic to send command to
            command_type: Command type for response matching (e.g., "datetime/get")
            payload: Command payload dict
            timeout: How long to wait for response

        Returns:
            Response dict if received, None if timeout
        """
        if self._optional_config.get("simulation_mode", False):
            return {"success": True, "message": "Simulated"}

        assert self.mqtt_client is not None

        # Queue the command and get a future for its response
        command_id, future = await self._command_queue.add_command(command_type, timeout)

        try:
            # Send the command
            await self.mqtt_client.publish(command_topic, json.dumps(payload))

            # Wait for the response (the result handler will resolve the future)
            result = await future

            # Store for later reference
            self._last_command_results[command_type] = result
            return result

        except asyncio.TimeoutError:
            # Timeout is already logged by the command queue
            self.logger.warning(f"⏱️ Timeout waiting for {command_type} result after {timeout}s")
            return None

        except Exception as e:
            self.logger.error(f"Error sending command {command_type}: {e}")
            return None

    async def _wait_for_command_result(
        self, command_type: str, timeout: float = 3.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for command result via command queue.

        DEPRECATED: Use _send_command_and_wait() for new code.
        Kept for compatibility with mode_manager.py which sends commands separately.

        Args:
            command_type: The command type to wait for (e.g., "batteryfirst/set/timeslot")
            timeout: How long to wait for the result

        Returns:
            The result dict if received, None if timeout
        """
        if self._optional_config.get("simulation_mode", False):
            return {"success": True, "message": "Simulated"}

        # Queue the command expectation and get a future for its response
        command_id, future = await self._command_queue.add_command(command_type, timeout)

        try:
            # Wait for the response (the result handler will resolve the future)
            result = await future
            # Store for later reference
            self._last_command_results[command_type] = result
            return result
        except asyncio.TimeoutError:
            self.logger.warning(f"⏱️ Timeout waiting for {command_type} result after {timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Error waiting for {command_type} result: {e}")
            return None

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

        # Query battery-first state using unified method
        try:
            self.logger.debug("Querying battery-first state...")
            bf_result = await self._send_command_and_wait(
                command_topic="energy/solar/command/batteryfirst/get",
                command_type="batteryfirst/get",
                payload={},
                timeout=3.0
            )

            if bf_result:
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
            else:
                self.logger.warning("Timeout querying battery-first state")
                state["battery_first"] = {}

        except Exception as e:
            self.logger.error(f"Error querying battery-first state: {e}")
            state["battery_first"] = {}

        # Query grid-first state using unified method
        try:
            self.logger.debug("Querying grid-first state...")
            gf_result = await self._send_command_and_wait(
                command_topic="energy/solar/command/gridfirst/get",
                command_type="gridfirst/get",
                payload={},
                timeout=3.0
            )

            if gf_result:
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
            else:
                self.logger.warning("Timeout querying grid-first state")
                state["grid_first"] = {}

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
            # Send command using unified method
            result = await self._send_command_and_wait(
                command_topic="energy/solar/command/datetime/get",
                command_type="datetime/get",
                payload={},
                timeout=5.0
            )

            if not result:
                self.logger.warning("Inverter time request timed out or failed")
                return None

            # Check success
            if not result.get("success", False):
                error_msg = result.get("message", "Unknown error")
                self.logger.error(f"Inverter returned error: {error_msg}")
                return None

            # Parse the datetime value
            value = result.get("value")  # Format: "YYYY-MM-DD HH:MM:SS"
            if not value:
                self.logger.error("No value in inverter time response")
                return None

            # Parse the datetime string
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

            # Handle potential year issue: firmware may return 2-digit year
            # If year < 100, it's likely years since 2000, so add 2000
            if dt.year < 100:
                dt = dt.replace(year=dt.year + 2000)
                self.logger.debug(f"Adjusted 2-digit year to {dt.year}")

            # Add timezone info
            dt = dt.replace(tzinfo=self._local_tz)

            self.logger.debug(f"Received inverter time: {value}")
            return dt

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
                    f"⏰ Clock skew detected: {drift:.1f}s "
                    f"({'inverter behind' if drift > 0 else 'inverter ahead'}) - "
                    f"Server: {server_time.strftime('%H:%M:%S')}, "
                    f"Inverter: {inverter_time.strftime('%H:%M:%S')}"
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
                sync_status = (
                    'inverter behind' if drift > 0
                    else 'inverter ahead' if drift < 0
                    else 'perfect sync'
                )
                self.logger.info(
                    f"⏰ Clock synchronized: {drift:.1f}s skew ({sync_status})"
                )

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
        immediate_activation: bool = False,
    ) -> Tuple[str, str]:
        """Ensure start time is appropriate for inverter scheduling.

        Args:
            start_str: Start time string (HH:MM or HH:MM:SS)
            stop_str: Stop time string (HH:MM or HH:MM:SS)
            min_future_minutes: Minutes in future for normal scheduling (default 1)
            preserve_duration: Keep original slot duration when bumping start
            immediate_activation: If True, set start time in PAST for immediate activation
                                 (inverter only triggers modes when clock crosses start time)

        Returns:
            Adjusted (start, stop) tuple in HH:MM format
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

        # Format helper
        def _fmt(dt: datetime) -> str:
            t = dt.time()
            return t.strftime("%H:%M")

        if immediate_activation:
            # For immediate activation, inverter needs to SEE a future time on its clock
            # so it can cross that time and trigger the mode.
            #
            # Key insight: We must account for clock skew between server and inverter
            # Clock drift calculation: server_time minus inverter_time
            #   - Positive: inverter is BEHIND server (inverter time < server time)
            #   - Negative: inverter is AHEAD of server (inverter time > server time)
            #
            # Example with inverter AHEAD by 45 seconds:
            #   Server: 19:36:00, Inverter: 19:36:45, drift = -45s
            #   If we set start to 19:38 server time:
            #     - Command reaches inverter at ~19:36:45 inverter time
            #     - Inverter needs to wait ~1 minute until its 19:38 → mode activates!

            # Base buffer for command transmission + processing (configurable)
            buffer_minutes = self.config.clock_drift_buffer_minutes

            # Add extra time if inverter is AHEAD (negative drift)
            # If inverter is ahead, we need MORE buffer time
            # Example: inverter 45s ahead → add 1 minute extra
            if self._clock_drift_seconds < 0:
                # Inverter is ahead - add extra buffer
                drift_adjustment_minutes = int(abs(self._clock_drift_seconds) / 60) + 1
            else:
                # Inverter is behind or in sync - may need slightly less time
                # but keep at least 1 minute for safety
                drift_adjustment_minutes = max(1, int(self._clock_drift_seconds / 60))

            total_buffer = buffer_minutes + drift_adjustment_minutes

            # Set start time in the FUTURE (relative to server)
            adjusted_start_dt = now.replace(microsecond=0, second=0) + timedelta(
                minutes=total_buffer
            )

            # Keep the original stop time (should be 23:59 for all-day modes)
            adjusted_stop_dt = stop_dt

            direction = 'behind' if self._clock_drift_seconds > 0 else 'ahead'
            skew_desc = (
                f"{abs(self._clock_drift_seconds):.1f}s {direction}"
                if self._clock_drift_seconds != 0
                else "in sync"
            )
            self.logger.info(
                f"⏰ Immediate activation scheduled: start={_fmt(adjusted_start_dt)} "
                f"(now={now.strftime('%H:%M:%S')}, buffer={total_buffer}min) - "
                f"Mode activates in ~{total_buffer} min | Clock skew: {skew_desc}"
            )
            return _fmt(adjusted_start_dt), _fmt(adjusted_stop_dt)

        # Normal future scheduling logic (existing behavior)
        # Round up to the next whole minute to avoid edge cases
        min_start = now.replace(microsecond=0, second=0) + timedelta(minutes=min_future_minutes)
        if start_dt <= min_start:
            adjusted_start_dt = min_start
            if preserve_duration:
                adjusted_stop_dt = adjusted_start_dt + duration
            else:
                adjusted_stop_dt = stop_dt  # keep original stop

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
        self, prices_15min: Dict[Tuple[str, str], float], date: str, eur_czk_rate: float
    ) -> None:
        """Log 15-minute interval prices in a compact table format."""
        # Use provided rate for consistency

        self.logger.info(f"Energy prices for {date} (15-minute intervals):")

        # If we have 96 blocks, show in compact 4-column format (one per quarter-hour)
        if len(prices_15min) >= 90:
            # Sort blocks by time
            sorted_blocks = sorted(prices_15min.items(), key=lambda x: x[0][0])

            self.logger.info("┌─────────┬──────────┬──────────┬──────────┬──────────┐")
            self.logger.info("│  Hour   │  :00-:15 │  :15-:30 │  :30-:45 │  :45-:00 │")
            self.logger.info("├─────────┼──────────┼──────────┼──────────┼──────────┤")

            # Process 4 blocks at a time (one hour)
            for hour in range(24):
                hour_blocks = sorted_blocks[hour * 4:(hour + 1) * 4]
                if len(hour_blocks) < 4:
                    break

                # Get prices for each 15-minute block in CZK/kWh
                prices = []
                for (start, end), price_eur in hour_blocks:
                    price_czk = price_eur * eur_czk_rate / 1000
                    # Mark blocks: 🔋=regular charge, 🔌=pre-discharge charge, ⚡=discharge
                    if (hasattr(self, '_pre_discharge_blocks') and
                            (start, end) in self._pre_discharge_blocks):
                        prices.append(f"{price_czk:5.2f}🔌")
                    elif (start, end) in self._cheapest_charging_blocks:
                        prices.append(f"{price_czk:5.2f}🔋")
                    elif (hasattr(self, '_discharge_periods') and
                          (start, end) in self._discharge_periods):
                        prices.append(f"{price_czk:5.2f}⚡")
                    else:
                        prices.append(f"{price_czk:6.2f} ")

                # Pad if we don't have all 4 blocks
                while len(prices) < 4:
                    prices.append("   -   ")

                self.logger.info(
                    f"│ {hour:02d}:00   │ {prices[0]} │ {prices[1]} │ {prices[2]} │ {prices[3]} │"
                )

            self.logger.info("└─────────┴──────────┴──────────┴──────────┴──────────┘")

            # Show legend
            legend_items = []
            if self._cheapest_charging_blocks:
                legend_items.append("🔋=Charge")
            if hasattr(self, '_pre_discharge_blocks') and self._pre_discharge_blocks:
                legend_items.append("🔌=Pre-discharge")
            if hasattr(self, '_discharge_periods') and self._discharge_periods:
                legend_items.append("⚡=Discharge")
            if legend_items:
                self.logger.info(f"Legend: {', '.join(legend_items)}")

            # Show summary statistics
            all_prices = list(prices_15min.values())
            min_price = min(all_prices) * eur_czk_rate / 1000
            max_price = max(all_prices) * eur_czk_rate / 1000
            avg_price = sum(all_prices) / len(all_prices) * eur_czk_rate / 1000

            self.logger.info(
                f"Summary: Min={min_price:.3f} CZK/kWh, Max={max_price:.3f} CZK/kWh, "
                f"Avg={avg_price:.3f} CZK/kWh"
            )
        else:
            # Fallback for fewer blocks - show as list
            self.logger.info("┌──────────────┬────────────┬──────────────┐")
            self.logger.info("│   Period     │ EUR/MWh    │   CZK/kWh    │")
            self.logger.info("├──────────────┼────────────┼──────────────┤")

            sorted_blocks = sorted(prices_15min.items(), key=lambda x: x[0][0])
            for (start, end), price_eur_mwh in sorted_blocks:
                price_czk_kwh = price_eur_mwh * eur_czk_rate / 1000
                period = f"{start}-{end}"
                self.logger.info(
                    f"│ {period:12s} │ {price_eur_mwh:8.2f}   │   {price_czk_kwh:7.3f}    │"
                )

            self.logger.info("└──────────────┴────────────┴──────────────┘")

    async def start(self) -> None:
        """Start the Growatt controller."""
        self._running = True

        # Note: MQTT subscriptions are pre-registered in __init__ to ensure they're
        # active before the message loop starts (avoids asyncio-mqtt race conditions)

        # Sync inverter time on startup
        self.logger.info("Checking inverter time synchronization...")
        await self._sync_inverter_time()

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

        # Cancel all pending commands in the queue
        await self._command_queue.cancel_all()

        # Cancel periodic check task
        if self._periodic_check_task and not self._periodic_check_task.done():
            self._periodic_check_task.cancel()
            try:
                await self._periodic_check_task
            except asyncio.CancelledError:
                pass
            self._periodic_check_task = None

        # Cancel price fetch task
        if self._price_fetch_task and not self._price_fetch_task.done():
            self._price_fetch_task.cancel()
            try:
                await self._price_fetch_task
            except asyncio.CancelledError:
                pass
            self._price_fetch_task = None

        # Unsubscribe from command results
        if self.mqtt_client:
            try:
                await self.mqtt_client.unsubscribe("energy/solar/result")
                self.logger.info("Unsubscribed from command result topic")
            except Exception as e:
                self.logger.debug(f"Error unsubscribing from result topic: {e}")

        # Cancel any pending command future
        # Note: Pending commands are already cancelled above via _command_queue.cancel_all()

        # Apply safe shutdown mode
        self.logger.info("🔄 Resetting inverter to safe shutdown state...")
        try:
            await self._evaluate_conditions("shutdown")
            self.logger.info("✅ Inverter shutdown state: regular mode applied")
        except Exception as e:
            self.logger.error(f"Error during shutdown reset: {e}", exc_info=True)

        self.logger.info("Growatt controller stopped")

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
                    self.logger.debug(
                        f"⚡ High loads detected! {', '.join(details)} - Checking if action needed"
                    )
                    # Handle high load start
                    await self._handle_high_load_start()
                else:
                    self.logger.debug("✅ High loads cleared - Restoring scheduled operation")
                    # Re-evaluate conditions to determine what mode to apply now
                    await self._evaluate_conditions("high_load_cleared")

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
            self.logger.debug("⚡ High loads detected - using decision tree to determine action")
            # Re-evaluate current state with the decision tree
            await self._evaluate_conditions("high_load_detected")
        except Exception as e:
            self.logger.error(f"Failed to handle high load start: {e}", exc_info=True)

    async def _fetch_prices(self) -> None:
        """Fetch current energy prices and trigger evaluation."""
        try:
            # Determine target date
            target_date = self._get_local_date_string(days_ahead=0)

            # Fetch energy prices from DAM (now returns 15-minute intervals)
            prices_15min, status = await self._price_analyzer.fetch_dam_energy_prices(
                date=target_date
            )

            if not prices_15min:
                if status == "not_published":
                    self.logger.info(
                        f"Prices for {target_date} not published yet, using mock prices"
                    )
                else:
                    self.logger.warning("Failed to retrieve energy prices, using mock prices")
                prices_15min = self._price_analyzer.generate_mock_prices(target_date)

            if prices_15min:
                # Store prices in cache (now 15-minute intervals)
                self._current_prices = prices_15min
                self._prices_date = target_date
                self._prices_updated = datetime.now()

                # Get exchange rate
                self._eur_czk_rate = await self._get_eur_czk_rate()

                # Find cheapest blocks for regular charging (non-consecutive)
                charge_blocks = getattr(self.config, 'battery_charge_blocks', 8)
                charging_schedule, avg_price = self._price_analyzer.get_charging_schedule(
                    prices_15min, num_blocks=charge_blocks
                )

                if charging_schedule:
                    # Store cheapest blocks as a set for fast lookup
                    self._cheapest_charging_blocks = set(
                        (block[0], block[1]) for block in charging_schedule
                    )
                else:
                    self._cheapest_charging_blocks = set()

                # Calculate discharge periods and pre-discharge schedule
                rate = self._eur_czk_rate or 25.0
                discharge_periods = self._calculate_discharge_periods(prices_15min, rate)

                # Calculate pre-discharge charging schedule if we have discharge periods
                pre_discharge_schedule: List[Tuple[str, str, float]] = []
                self._peak_to_precharge_map = {}
                if discharge_periods:
                    pre_discharge_charge_blocks = getattr(
                        self.config, 'pre_discharge_charge_blocks', 8
                    )

                    pre_discharge_schedule, self._peak_to_precharge_map = (
                        self._price_analyzer.calculate_pre_discharge_schedule(
                            prices_15min,
                            discharge_periods,
                            pre_discharge_charge_blocks
                        )
                    )

                    # Store pre-discharge blocks
                    self._pre_discharge_blocks = set(
                        (block[0], block[1]) for block in pre_discharge_schedule
                    )
                else:
                    self._pre_discharge_blocks = set()

                # Combine regular and pre-discharge charging blocks
                self._combined_charging_blocks = (
                    self._cheapest_charging_blocks | self._pre_discharge_blocks
                )

                # Log comprehensive price analysis
                await self._log_price_analysis(
                    prices_15min,
                    target_date,
                    charging_schedule,
                    charge_blocks,
                    pre_discharge_schedule,
                    self._peak_to_precharge_map
                )

                # Trigger re-evaluation with new prices
                await self._on_price_update()
            else:
                self.logger.error("Failed to get any prices")

        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}", exc_info=True)

    def _calculate_discharge_periods(
        self,
        prices: Dict[Tuple[str, str], float],
        rate: float
    ) -> List[Tuple[str, str, float]]:
        """Calculate periods when battery discharge would be profitable.

        Args:
            prices: Dictionary of 15-minute price blocks
            rate: EUR to CZK conversion rate

        Returns:
            List of tuples (start_time, end_time, price_eur) for discharge periods
        """
        if not prices:
            return []

        discharge_periods = []

        # Get thresholds from config
        discharge_min_czk = self.config.discharge_price_min
        profit_margin = self.config.discharge_profit_margin

        # Find cheapest block price
        cheapest_price_eur = min(prices.values())
        cheapest_price_czk = cheapest_price_eur * rate / 1000

        # Calculate effective threshold
        required_by_margin = cheapest_price_czk * profit_margin
        effective_threshold_czk = max(discharge_min_czk, required_by_margin)

        # Find all periods above threshold
        for (start, end), price_eur in sorted(prices.items()):
            price_czk = price_eur * rate / 1000
            if price_czk >= effective_threshold_czk:
                discharge_periods.append((start, end, price_eur))

        return discharge_periods

    async def _log_price_analysis(
        self,
        prices: Dict[Tuple[str, str], float],
        date: str,
        charging_schedule: List[Tuple[str, str, float]],
        charge_blocks: int,
        pre_discharge_schedule: Optional[List[Tuple[str, str, float]]] = None,
        peak_to_precharge_map: Optional[Dict[str, List[Tuple[str, str, float]]]] = None
    ) -> None:
        """Log comprehensive price analysis including table and charging schedule.

        Args:
            prices: Dictionary of 15-minute price blocks
            date: Date string for the prices
            charging_schedule: List of cheapest charging blocks
            charge_blocks: Number of blocks configured for charging
            pre_discharge_schedule: Optional list of pre-discharge charging blocks
            peak_to_precharge_map: Optional mapping of peaks to their pre-charge blocks
        """
        # Ensure we have exchange rate
        if not self._eur_czk_rate:
            self._eur_czk_rate = await self._get_eur_czk_rate()

        rate = self._eur_czk_rate or 25.0

        # Calculate discharge periods
        discharge_periods = self._calculate_discharge_periods(prices, rate)

        # Store discharge periods for price table display
        self._discharge_periods = set(
            (period[0], period[1]) for period in discharge_periods
        )

        # Log price table (will now show discharge periods too)
        await self._log_price_table(prices, date, rate)

        if charging_schedule:
            # Calculate average price for charging blocks
            avg_price = sum(block[2] for block in charging_schedule) / len(charging_schedule)
            avg_czk = avg_price * rate / 1000
            charge_duration_hours = charge_blocks * 0.25  # 15 minutes = 0.25 hours

            # Log charging schedule
            self.logger.info("=" * 50)
            self.logger.info(
                f"🔋 CHARGING SCHEDULE "
                f"({charge_blocks} blocks = {charge_duration_hours:.1f} hours)"
            )
            self.logger.info(f"   Average price: {avg_czk:.3f} CZK/kWh")
            self.logger.info("   Charging blocks:")

            for start, end, price_eur in charging_schedule:
                price_czk = price_eur * rate / 1000
                self.logger.info(
                    f"     {start}-{end}: {price_czk:.3f} CZK/kWh"
                )

            # Calculate savings vs peak
            all_prices = list(prices.values())
            max_price = max(all_prices) if all_prices else avg_price
            savings_pct = (
                ((max_price - avg_price) / max_price * 100)
                if max_price > 0 else 0
            )
            self.logger.info(f"   Savings vs peak: {savings_pct:.0f}%")
            self.logger.info("=" * 50)

        # Log discharge schedule if there are discharge periods
        if discharge_periods:
            # Calculate average price for discharge blocks
            avg_discharge_price = sum(p[2] for p in discharge_periods) / len(discharge_periods)
            avg_discharge_czk = avg_discharge_price * rate / 1000
            discharge_duration_hours = len(discharge_periods) * 0.25

            # Get cheapest price for profit calculation
            cheapest_price_eur = min(prices.values())
            cheapest_price_czk = cheapest_price_eur * rate / 1000

            # Calculate profit margin
            profit_margin = avg_discharge_czk / cheapest_price_czk if cheapest_price_czk > 0 else 0

            self.logger.info("=" * 50)
            self.logger.info(
                f"⚡ DISCHARGE SCHEDULE "
                f"({len(discharge_periods)} blocks = {discharge_duration_hours:.1f} hours)"
            )
            self.logger.info(f"   Average price: {avg_discharge_czk:.3f} CZK/kWh")
            self.logger.info(f"   Cheapest block: {cheapest_price_czk:.3f} CZK/kWh")
            self.logger.info(f"   Profit margin: {profit_margin:.1f}x")
            self.logger.info(
                f"   Discharge threshold: {self.config.discharge_price_min:.2f} CZK/kWh"
            )
            self.logger.info("   Discharge blocks:")

            # Group consecutive blocks for cleaner display
            grouped_periods = []
            current_group = [discharge_periods[0]]

            for i in range(1, len(discharge_periods)):
                prev_end = current_group[-1][1]
                curr_start = discharge_periods[i][0]

                # Check if consecutive (end time of prev equals start time of current)
                if prev_end == curr_start:
                    current_group.append(discharge_periods[i])
                else:
                    # Start new group
                    grouped_periods.append(current_group)
                    current_group = [discharge_periods[i]]

            # Add the last group
            grouped_periods.append(current_group)

            # Display grouped periods
            for group in grouped_periods:
                start = group[0][0]
                end = group[-1][1]
                avg_group_price = sum(p[2] for p in group) / len(group)
                avg_group_czk = avg_group_price * rate / 1000
                if len(group) > 1:
                    self.logger.info(
                        f"     {start}-{end}: {avg_group_czk:.3f} CZK/kWh "
                        f"(avg of {len(group)} blocks)"
                    )
                else:
                    self.logger.info(
                        f"     {start}-{end}: {avg_group_czk:.3f} CZK/kWh"
                    )

            self.logger.info("=" * 50)

        # Log pre-discharge preparation schedule if available
        if pre_discharge_schedule and peak_to_precharge_map:
            # Calculate totals
            total_unique_blocks = len(self._combined_charging_blocks)
            total_hours = total_unique_blocks * 0.25
            pre_discharge_count = len(self._pre_discharge_blocks)

            self.logger.info("=" * 50)
            self.logger.info(
                f"🔌 PRE-DISCHARGE: {pre_discharge_count} blocks added "
                f"(total charging: {total_unique_blocks} blocks = {total_hours:.1f}h)"
            )
            self.logger.info("=" * 50)

    async def _on_price_update(self) -> None:
        """Handle new price data availability."""
        await self._evaluate_conditions("price_update")

    async def _fetch_next_day_prices_task(self) -> None:
        """Background task to fetch next day's prices with infinite retry.

        This task runs continuously, retrying with smart time-based backoff until:
        1. Prices are successfully fetched, OR
        2. Task is cancelled (e.g., at midnight when switching to next day)
        """
        try:
            # Get target date (tomorrow)
            tomorrow_date = self._get_local_date_string(days_ahead=1)

            self.logger.info(
                f"🔄 Starting next-day price fetch for {tomorrow_date} "
                f"(will retry until successful or cancelled)"
            )

            # Fetch with infinite retry using config parameters
            # This will only return empty dict if cancelled
            prices = await self._price_analyzer.fetch_dam_energy_prices_with_retry(
                target_date=tomorrow_date,
                initial_delay_minutes=self.config.price_fetch_retry_initial_delay,
                max_delay_minutes=self.config.price_fetch_retry_max_delay,
                max_attempts=self.config.price_fetch_retry_max_attempts  # Ignored by retry func
            )

            if prices:
                # Store next day's prices
                self._next_day_prices = prices
                self._next_day_prices_date = tomorrow_date
                self._next_day_prices_fetched = True

                # Simple confirmation log (no verbose table)
                self.logger.info(
                    f"✅ Successfully fetched and stored {len(prices)} price blocks "
                    f"for {tomorrow_date}"
                )
            else:
                # Empty dict means task was cancelled (e.g., midnight rollover)
                self.logger.debug(
                    f"Price fetch for {tomorrow_date} returned empty (task cancelled)"
                )
                self._next_day_prices_fetched = False

        except asyncio.CancelledError:
            # Task cancelled (normal at midnight rollover)
            self.logger.debug(
                "Price fetch task cancelled (normal at midnight rollover)"
            )
            self._next_day_prices_fetched = False
            raise  # Re-raise to properly handle cancellation

        except Exception as e:
            self.logger.error(
                f"Unexpected error in next-day price fetch task: {e}", exc_info=True
            )
            self._next_day_prices_fetched = False

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
                    # Log mode change with state details (done in _apply_decided_mode)

                    # Apply the new mode (will log consolidated one-liner)
                    await self._apply_decided_mode(new_mode, reason=reason, explanation=explanation)
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

        # Calculate current 15-minute block
        current_minute = now.minute
        block_start_minute = (current_minute // 15) * 15

        current_block_start = now.replace(minute=block_start_minute, second=0, microsecond=0)
        current_block_end = current_block_start + timedelta(minutes=15)

        # Format keys for 15-minute blocks
        start_str = current_block_start.strftime("%H:%M")
        if current_block_end.date() != current_block_start.date():
            end_str = "24:00"
        else:
            end_str = current_block_end.strftime("%H:%M")
        current_block_key = (start_str, end_str)

        # Get current price for 15-minute block
        current_price = self._current_prices.get(current_block_key, 0.0)

        # Calculate sunrise/sunset
        sun_times = sun(self._location.observer, date=now.date())
        sunrise = sun_times["sunrise"].time() if "sunrise" in sun_times else None
        sunset = sun_times["sunset"].time() if "sunset" in sun_times else None

        # Create price thresholds from config (all in CZK/kWh)
        price_thresholds = PriceThresholds(
            charge_price_max=getattr(self.config, 'charge_price_max', 1.5),
            export_price_min=getattr(self.config, 'export_price_min', 1.0),
            discharge_price_min=getattr(self.config, 'discharge_price_min', 3.0),
            discharge_profit_margin=getattr(self.config, 'discharge_profit_margin', 4.0),
            battery_efficiency=getattr(self.config, 'battery_efficiency', 0.85)
        )

        # Calculate price ranking for current 15-minute block
        price_ranking = None
        if self._current_prices and current_block_key in self._current_prices:
            price_ranking = self._decision_engine.calculate_price_ranking(
                current_block_key, self._current_prices
            )
            if price_ranking:
                self.logger.debug(
                    f"Price ranking: rank {price_ranking.current_rank}/"
                    f"{price_ranking.total_blocks} "
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
            current_block_key=current_block_key,
            prices_15min=self._current_prices.copy(),  # Now using 15-minute prices
            cheapest_blocks=self._combined_charging_blocks.copy(),  # Combined charging blocks
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
            # Use appropriate default based on mode type
            if mode == "discharge_to_grid":
                default_soc = self.config.discharge_min_soc  # Use 20% for discharge
            elif mode == "charge_from_grid":
                default_soc = self.config.max_soc  # Use 100% for charging
            else:
                default_soc = self.config.max_soc  # Default to max for other modes

            soc_param = params.get("stop_soc", default_soc)
            stop_soc = int(soc_param) if soc_param is not None else int(default_soc)
        elif isinstance(stop_soc_raw, (int, float, str)):
            stop_soc = int(stop_soc_raw)

        # Determine power rate
        # AC charge from mode definition
        ac_charge_enabled = bool(mode_def.get("ac_charge", False))

        # If AC charging is enabled, ALWAYS use 100% power rate to charge quickly within the slot
        if ac_charge_enabled:
            power_rate = 100
            self.logger.debug("AC charging enabled - setting power_rate to 100%")
        else:
            # For other modes, use configured or default rate
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

    async def _apply_state_changes_with_rollback(
        self,
        old_state: Optional[InverterState],
        new_state: InverterState
    ) -> None:
        """Apply state changes with rollback on failure.

        Args:
            old_state: Previous state (None for initial)
            new_state: Desired new state

        Raises:
            Exception: If state application fails after rollback attempt
        """
        try:
            await self._apply_state_changes(old_state, new_state)
        except Exception as e:
            self.logger.error(f"Failed to apply state changes: {e}")
            if old_state:
                self.logger.warning("Attempting to rollback to previous state...")
                try:
                    await self._apply_state_changes(None, old_state)
                    self.logger.info("✅ Rollback successful")
                except Exception as rollback_error:
                    self.logger.error(f"❌ Rollback failed: {rollback_error}")
            raise

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
            not old_state
            or old_state.inverter_mode != new_state.inverter_mode
            or old_state.stop_soc != new_state.stop_soc
            or old_state.power_rate != new_state.power_rate
            or old_state.time_start != new_state.time_start
            or old_state.time_stop != new_state.time_stop
        )

        if mode_changed:
            self._commands_sent_count += 1

            # Extract previous mode for optimization (only disable the mode that was active)
            previous_mode = old_state.inverter_mode if old_state else None

            # Log detailed mode transition explanation (debug only)
            if old_state:
                self.logger.debug(
                    f"🔄 Inverter mode transition: {old_state.inverter_mode} → "
                    f"{new_state.inverter_mode} (source: {new_state.source})"
                )
                # Log parameter changes
                param_changes = []
                if old_state.stop_soc != new_state.stop_soc:
                    param_changes.append(f"stopSOC: {old_state.stop_soc}% → {new_state.stop_soc}%")
                if old_state.power_rate != new_state.power_rate:
                    param_changes.append(
                        f"powerRate: {old_state.power_rate}% → {new_state.power_rate}%"
                    )
                if (old_state.time_start != new_state.time_start
                        or old_state.time_stop != new_state.time_stop):
                    param_changes.append(
                        f"timeWindow: {old_state.time_start}-{old_state.time_stop} → "
                        f"{new_state.time_start}-{new_state.time_stop}"
                    )
                if param_changes:
                    self.logger.debug(f"   Parameters: {', '.join(param_changes)}")
            else:
                self.logger.debug(
                    f"🔄 Initial inverter mode: {new_state.inverter_mode} "
                    f"(stopSOC={new_state.stop_soc}%, source: {new_state.source})"
                )

            # Detect if this is for immediate activation (all-day time window)
            # Time window of 00:00-23:59 indicates mode should be active NOW
            immediate_activation = (
                new_state.time_start == "00:00" and new_state.time_stop == "23:59"
            )

            if new_state.inverter_mode == "load_first":
                await self._mode_manager.set_load_first(
                    stop_soc=new_state.stop_soc,
                    previous_mode=previous_mode
                )
            elif new_state.inverter_mode == "battery_first":
                await self._mode_manager.set_battery_first(
                    new_state.time_start,
                    new_state.time_stop,
                    stop_soc=new_state.stop_soc,
                    power_rate=new_state.power_rate,
                    immediate_activation=immediate_activation,
                    previous_mode=previous_mode
                )
            elif new_state.inverter_mode == "grid_first":
                await self._mode_manager.set_grid_first(
                    new_state.time_start,
                    new_state.time_stop,
                    stop_soc=new_state.stop_soc,
                    power_rate=new_state.power_rate,
                    immediate_activation=immediate_activation,
                    previous_mode=previous_mode
                )

            await asyncio.sleep(0.5)

        # Export change
        if not old_state or old_state.export_enabled != new_state.export_enabled:
            self._commands_sent_count += 1
            if new_state.export_enabled:
                await self._mode_manager.enable_export()
                self.logger.debug(
                    f"⚡ Export to grid ENABLED (source: {new_state.source})"
                )
            else:
                await self._mode_manager.disable_export()
                self.logger.debug(
                    f"⚡ Export to grid DISABLED (source: {new_state.source})"
                )

        # AC charge change
        if not old_state or old_state.ac_charge_enabled != new_state.ac_charge_enabled:
            self._commands_sent_count += 1
            await self._mode_manager.set_ac_charge(new_state.ac_charge_enabled)
            if new_state.ac_charge_enabled:
                self.logger.debug(
                    f"🔌 AC charging from grid ENABLED (source: {new_state.source})"
                )
            else:
                self.logger.debug(
                    f"🔌 AC charging from grid DISABLED (source: {new_state.source})"
                )

    async def _apply_decided_mode(
        self,
        mode: str,
        params: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        explanation: Optional[Dict[str, Any]] = None
    ) -> None:
        """Apply a mode based on its definition from MODE_DEFINITIONS.

        Args:
            mode: Mode name from decision engine
            params: Optional parameters for configurable modes
            reason: What triggered the mode change
            explanation: Decision explanation from decision engine
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

        # Log consolidated one-liner with all essential info
        if self._current_inverter_state:
            old_mode = self._current_mode or "unknown"
            brief_reason = explanation['reason'].split(' - ')[0] if explanation else "mode change"
            self.logger.info(
                f"📊 Mode: {old_mode} → {mode} ({desired_state.summary()}) - {brief_reason}"
            )
        else:
            self.logger.info(f"📊 Initial mode: {mode} ({desired_state.summary()})")

        # Apply ONLY the changes needed (with rollback on failure)
        await self._apply_state_changes_with_rollback(self._current_inverter_state, desired_state)

        # Update tracking
        self._current_inverter_state = desired_state
        self._state_history.append(desired_state)
        if len(self._state_history) > 10:
            self._state_history.pop(0)

        # Track if we're in high load protected mode
        self._high_load_protected_mode_active = (mode == "high_load_protected")

    # Event-driven evaluation methods

    async def _periodic_evaluation_loop(self) -> None:
        """Periodically check for condition changes that need re-evaluation."""
        last_midnight_check = None
        last_fetch_check = None

        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = self._get_local_now()

                # Check if it's time to start fetching next day's prices
                if now.hour == self.config.price_fetch_hour and now.minute == 0:
                    current_fetch_check = now.date()
                    should_fetch = (
                        last_fetch_check != current_fetch_check
                        and not self._next_day_prices_fetched
                    )
                    if should_fetch:
                        last_fetch_check = current_fetch_check
                        self.logger.info(
                            f"⏰ {now.hour:02d}:00 - Starting next-day price fetch "
                            f"(configured hour: {self.config.price_fetch_hour})"
                        )

                        # Cancel any existing fetch task
                        if self._price_fetch_task and not self._price_fetch_task.done():
                            self._price_fetch_task.cancel()
                            try:
                                await self._price_fetch_task
                            except asyncio.CancelledError:
                                pass

                        # Start new fetch task in background
                        self._price_fetch_task = asyncio.create_task(
                            self._fetch_next_day_prices_task()
                        )

                # Check for midnight - update date display
                if now.hour == 0 and now.minute == 0:
                    # Only process once per midnight
                    current_midnight = now.date()
                    if last_midnight_check != current_midnight:
                        last_midnight_check = current_midnight
                        self.logger.info("🕛 Midnight - new day starting...")

                        # Cancel any ongoing fetch task (if exists)
                        if self._price_fetch_task and not self._price_fetch_task.done():
                            self.logger.debug(
                                "Cancelling previous price fetch task (midnight rollover)"
                            )
                            self._price_fetch_task.cancel()
                            try:
                                await self._price_fetch_task
                            except asyncio.CancelledError:
                                pass
                            self._price_fetch_task = None

                        # If we have next day's prices ready, activate them
                        if self._next_day_prices_fetched and self._next_day_prices:
                            # Move next-day prices to current prices
                            self._current_prices = self._next_day_prices
                            self._prices_date = self._next_day_prices_date
                            self._prices_updated = now

                            # Recalculate cheapest blocks with the new prices
                            charge_blocks = getattr(self.config, 'battery_charge_blocks', 8)
                            charging_schedule, avg_price = (
                                self._price_analyzer.get_charging_schedule(
                                    self._current_prices, num_blocks=charge_blocks
                                )
                            )
                            self._cheapest_charging_blocks = set(
                                (block[0], block[1]) for block in charging_schedule
                            )

                            # Calculate discharge and pre-discharge schedules
                            rate = self._eur_czk_rate or 25.0
                            discharge_periods = self._calculate_discharge_periods(
                                self._current_prices, rate
                            )

                            # Calculate pre-discharge schedule
                            pre_discharge_schedule: List[Tuple[str, str, float]] = []
                            self._peak_to_precharge_map = {}
                            if discharge_periods:
                                pre_discharge_charge_blocks = getattr(
                                    self.config, 'pre_discharge_charge_blocks', 8
                                )

                                pre_discharge_schedule, self._peak_to_precharge_map = (
                                    self._price_analyzer.calculate_pre_discharge_schedule(
                                        self._current_prices,
                                        discharge_periods,
                                        pre_discharge_charge_blocks
                                    )
                                )

                                self._pre_discharge_blocks = set(
                                    (block[0], block[1]) for block in pre_discharge_schedule
                                )
                            else:
                                self._pre_discharge_blocks = set()

                            # Combine regular and pre-discharge charging blocks
                            self._combined_charging_blocks = (
                                self._cheapest_charging_blocks | self._pre_discharge_blocks
                            )

                            self.logger.info(
                                f"✅ Activated prices for {self._prices_date} "
                                f"({len(self._current_prices)} blocks)"
                            )

                            # Display comprehensive price analysis for the new day
                            if self._prices_date:  # Type guard for mypy
                                self.logger.info(
                                    f"🕛 Energy prices for TODAY ({self._prices_date}):"
                                )
                                await self._log_price_analysis(
                                    self._current_prices,
                                    self._prices_date,
                                    charging_schedule,
                                    charge_blocks,
                                    pre_discharge_schedule,
                                    self._peak_to_precharge_map
                                )

                            # Trigger re-evaluation with new prices
                            await self._on_price_update()

                            # Clear next-day price storage and reset flag
                            self._next_day_prices = {}
                            self._next_day_prices_date = None
                            self._next_day_prices_fetched = False

                            # Start background fetch for NEW next day's prices (non-blocking)
                            # This task will retry indefinitely with smart time-based backoff
                            self._price_fetch_task = asyncio.create_task(
                                self._fetch_next_day_prices_task()
                            )
                            self.logger.info(
                                "🔄 Started background fetch for next day's prices "
                                "(will retry with smart backoff until successful)"
                            )
                        else:
                            # No next-day prices yet (normal case) - keep using current prices
                            current_date = self._get_local_date_string(days_ahead=0)

                            self.logger.info(
                                f"📊 Continuing with existing price data for {current_date} "
                                f"(new prices will be fetched at {self.config.price_fetch_hour}:00)"
                            )

                            # Update the display date to current date and show price table
                            if self._current_prices:
                                # Display the price table with the current date
                                self.logger.info(f"🕛 Energy prices for TODAY ({current_date}):")

                                # Recalculate schedules with existing prices
                                charge_blocks = getattr(self.config, 'battery_charge_blocks', 8)
                                charging_schedule, _ = self._price_analyzer.get_charging_schedule(
                                    self._current_prices, num_blocks=charge_blocks
                                )

                                # Calculate discharge periods
                                rate = self._eur_czk_rate or 25.0
                                discharge_periods = self._calculate_discharge_periods(
                                    self._current_prices, rate
                                )

                                # Calculate pre-discharge schedule
                                pre_discharge_schedule: List[Tuple[str, str, float]] = []
                                peak_to_precharge_map = {}
                                if discharge_periods:
                                    pre_discharge_charge_blocks = getattr(
                                        self.config, 'pre_discharge_charge_blocks', 8
                                    )

                                    pre_discharge_schedule, peak_to_precharge_map = (
                                        self._price_analyzer.calculate_pre_discharge_schedule(
                                            self._current_prices,
                                            discharge_periods,
                                            pre_discharge_charge_blocks
                                        )
                                    )

                                # Show the price analysis with current date
                                await self._log_price_analysis(
                                    self._current_prices,
                                    current_date,  # Use current date for display
                                    charging_schedule,
                                    charge_blocks,
                                    pre_discharge_schedule,
                                    peak_to_precharge_map
                                )

                            # Reset the fetch flag for the new day
                            self._next_day_prices_fetched = False
                            self._next_day_prices = {}
                            self._next_day_prices_date = None

                            # DON'T start fetching - wait for configured fetch hour (14:00)
                            self.logger.info(
                                f"⏰ Next-day prices will be fetched at "
                                f"{self.config.price_fetch_hour}:00 when OTE publishes them"
                            )

                # Check for 15-minute block change (price changes every 15 minutes)
                current_block = (now.hour, now.minute // 15)
                if self._last_evaluation_block != current_block:
                    self._last_evaluation_block = current_block
                    await self._evaluate_conditions("15min_block_change")

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
