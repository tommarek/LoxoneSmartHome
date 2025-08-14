"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
import zoneinfo
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config.settings import Settings
from modules.base import BaseModule
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient


class EnergyPriceData:
    """Data class for energy price information."""

    def __init__(self, prices: Dict[Tuple[str, str], float]) -> None:
        """Initialize energy price data."""
        self.prices = prices
        # Use Prague timezone for energy price data
        self.timestamp = datetime.now(zoneinfo.ZoneInfo("Europe/Prague"))


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

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

        # Scheduled tasks
        self._scheduled_tasks: List[asyncio.Task[None]] = []
        self._daily_schedule_task: Optional[asyncio.Task[None]] = None

        # Track scheduled periods for startup state sync
        self._scheduled_periods: List[Tuple[str, str, str]] = []

        # Local timezone (Prague/Czech Republic)
        self._local_tz = zoneinfo.ZoneInfo("Europe/Prague")

        # Season mode cache
        self._season_mode: Optional[str] = None
        self._season_mode_updated: Optional[datetime] = None

        # Running flag for daily loop
        self._running: bool = False

        # Track last applied mode signature to prevent flapping
        self._last_mode_sig: Optional[Tuple[str, Tuple[Any, ...]]] = None

        # EUR/CZK exchange rate cache
        self._eur_czk_rate: Optional[float] = None
        self._eur_czk_rate_updated: Optional[datetime] = None
        
        # Track AC and Export states to prevent duplicate commands
        self._ac_enabled: Optional[bool] = None
        self._export_enabled: Optional[bool] = None

    def _get_local_now(self) -> datetime:
        """Get current time in local timezone."""
        return datetime.now(self._local_tz)

    def _parse_hhmm(self, s: str) -> dt_time:
        """Parse HH:MM allowing '24:00' -> 00:00 next day semantics."""
        if s == "24:00":
            # treat as 00:00 (next day) for comparisons
            return dt_time(0, 0)
        return datetime.strptime(s, "%H:%M").time()

    def _normalize_end_time(self, s: str) -> str:
        """Return a safe end-time string for scheduling and parsing."""
        # Prefer 23:59 to avoid '24:00' which Python can't parse
        return "23:59" if s in ("24:00", "23:60") else s

    async def _set_mode(self, mode: str, *args: Any) -> None:
        """Set inverter mode with flapping guard."""
        sig = (mode, args)
        if self._last_mode_sig == sig:
            self.logger.debug(f"Mode {mode} with args {args} already applied, skipping")
            return

        if mode == "battery_first":
            await self._set_battery_first(*args)
        elif mode == "grid_first":
            await self._set_grid_first(*args)
        elif mode == "load_first":
            await self._set_load_first()

        self._last_mode_sig = sig

    def _get_local_date_string(self, days_ahead: int = 1) -> str:
        """Get date string in local timezone for API calls."""
        local_date = self._get_local_now() + timedelta(days=days_ahead)
        return local_date.strftime("%Y-%m-%d")
    
    def _log_schedule_summary(self) -> None:
        """Log a summary of the scheduled periods for the day."""
        if not self._scheduled_periods:
            self.logger.info("No periods scheduled for the day")
            return
            
        self.logger.info("═══════════════════════════════════════")
        self.logger.info("Daily Schedule Summary:")
        self.logger.info("───────────────────────────────────────")
        self.logger.info("Type            Start   End")
        self.logger.info("───────────────────────────────────────")
        
        for period_type, start, end in sorted(self._scheduled_periods, key=lambda x: x[1]):
            self.logger.info(f"{period_type:<15} {start}  →  {end}")
            
        self.logger.info("═══════════════════════════════════════")

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
                        return self.config.eur_czk_rate

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
            fallback = self.config.eur_czk_rate if self.config.eur_czk_rate > 0 else 25.0
            return fallback

        except Exception as e:
            self.logger.error(f"Error fetching EUR/CZK exchange rate: {e}")
            # Return default fallback (25 CZK per EUR is typical)
            fallback = self.config.eur_czk_rate if self.config.eur_czk_rate > 0 else 25.0
            # Cache the fallback to avoid repeated API calls on errors
            self._eur_czk_rate = fallback
            self._eur_czk_rate_updated = now
            return fallback

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
                days = self.config.temperature_avg_days
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
                    season = "summer" if avg_temp > self.config.summer_temp_threshold else "winter"
                    self._season_mode = season
                    self._season_mode_updated = now
                    self.logger.info(
                        f"Season mode determined: {self._season_mode} "
                        f"(3-day avg temp: {avg_temp:.1f}°C, "
                        f"threshold: {self.config.summer_temp_threshold}°C)"
                    )
                    return self._season_mode
                else:
                    self.logger.warning("No temperature data available, defaulting to winter mode")

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
        except Exception as e:
            self.logger.error(f"Error disabling battery first on shutdown: {e}")

        try:
            await self._disable_grid_first()
        except Exception as e:
            self.logger.error(f"Error disabling grid first on shutdown: {e}")

        try:
            await self._disable_export()
        except Exception as e:
            self.logger.error(f"Error disabling export on shutdown: {e}")

        self.logger.info("Growatt controller stopped")

    async def _apply_current_state(self) -> None:
        """Apply the appropriate state based on current time and scheduled periods."""
        current_time = self._get_local_now()
        current_hour = current_time.strftime("%H:%M")

        self.logger.info(f"Checking current state at {current_hour} for immediate application...")

        # We need to track scheduled periods to check if we're in one
        # This will be populated by the scheduling methods
        if not hasattr(self, "_scheduled_periods") or not self._scheduled_periods:
            self.logger.info("No scheduled periods found yet, skipping startup state sync")
            return

        # Track if we're in any export period
        in_export_period = False
        mode_applied = False

        # Check all scheduled periods
        for period_type, start, end in self._scheduled_periods:
            start_time = self._parse_hhmm(start)
            end_time = self._parse_hhmm(self._normalize_end_time(end))
            current_time_only = current_time.time()

            # Handle periods that don't cross midnight
            if start_time <= end_time:
                in_period = start_time <= current_time_only < end_time
            else:
                # Handle periods that cross midnight
                in_period = current_time_only >= start_time or current_time_only < end_time

            if in_period:
                self.logger.info(
                    f"Currently in {period_type} period ({start}-{end}), applying state..."
                )

                if period_type == "battery_first":
                    await self._set_mode("battery_first", start, end)
                    mode_applied = True
                    # Check if we're in summer mode to disable AC charging
                    season = await self._get_season_mode()
                    if season == "summer":
                        await self._disable_ac_charge()
                elif period_type == "grid_first":
                    await self._set_mode("grid_first", start, end, 100)  # Always 100% stopSOC
                    mode_applied = True
                elif period_type == "load_first":
                    await self._set_mode("load_first")
                    mode_applied = True
                elif period_type == "ac_charge":
                    await self._enable_ac_charge()
                elif period_type == "export":
                    await self._enable_export()
                    in_export_period = True

        # If no mode was applied, default to load-first
        if not mode_applied:
            self.logger.info("Not in any scheduled period, applying load-first mode...")
            await self._set_mode("load_first")

        # If not in any export period, disable export
        if not in_export_period:
            self.logger.info("Not in any export period, disabling export...")
            await self._disable_export()

        self.logger.info("Startup state synchronization complete")

    async def _fetch_dam_energy_prices(
        self, date: Optional[str] = None
    ) -> Dict[Tuple[str, str], float]:
        """Fetch energy prices from OTE DAM API."""
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

            # Parse the OTE API response
            # dataLine[0] contains CZK/MWh prices (despite axis label saying EUR)
            # dataLine[1] contains actual EUR/MWh prices
            hourly_prices: Dict[Tuple[str, str], float] = {}
            if data.get("data", {}).get("dataLine"):
                lines = data["data"]["dataLine"]
                if len(lines) >= 2:
                    # Use second line (index 1) which has EUR/MWh prices
                    price_data = lines[1].get("point", [])
                else:
                    # Fallback to first line if only one exists
                    price_data = lines[0].get("point", [])
                    self.logger.warning("Only one dataLine found, prices might be in CZK")

                for point in price_data:
                    hour = int(point["x"]) - 1
                    price = float(point["y"])
                    start_time = f"{hour:02d}:00"
                    stop_time = f"{hour + 1:02d}:00"
                    hourly_prices[(start_time, stop_time)] = price

            self.logger.info(f"Successfully fetched {len(hourly_prices)} DAM price points")
            return hourly_prices

        except Exception as e:
            self.logger.error(f"Error fetching DAM prices: {e}", exc_info=True)
            return {}

    def _find_cheapest_consecutive_hours(
        self, prices: Dict[Tuple[str, str], float], x: int = 2
    ) -> List[Tuple[str, str, float]]:
        """Find X consecutive hours with lowest total price."""
        if not prices:
            return []

        # Sort by start time "HH:MM"
        intervals = sorted(prices.keys(), key=lambda t: t[0])
        num_intervals = len(intervals)

        if num_intervals < 2:
            return []

        # Determine interval duration from the first two (now sorted)
        start0 = datetime.strptime(intervals[0][0], "%H:%M")
        start1 = datetime.strptime(intervals[1][0], "%H:%M")
        interval_duration = int((start1 - start0).total_seconds() // 60)

        if interval_duration not in (15, 60):
            self.logger.warning(f"Unknown interval duration: {interval_duration} minutes")
            return []

        intervals_per_hour = 60 // interval_duration
        intervals_needed = x * intervals_per_hour

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
        """Group contiguous hours into continuous ranges."""
        if not hours:
            return []

        sorted_hours = sorted(hours, key=lambda x: datetime.strptime(x[0], "%H:%M"))

        groups = []
        current_group_start = sorted_hours[0][0]
        current_group_end = sorted_hours[0][1]
        current_group_price = sorted_hours[0][2]

        for hour in sorted_hours[1:]:
            previous_end = datetime.strptime(self._normalize_end_time(current_group_end), "%H:%M")
            current_start = datetime.strptime(hour[0], "%H:%M")
            current_price = hour[2]

            # Group if contiguous and price difference < 20%
            if current_start == previous_end and abs(current_price - current_group_price) < abs(
                current_group_price * 0.2
            ):
                current_group_end = hour[1]
            else:
                groups.append((current_group_start, self._normalize_end_time(current_group_end)))
                current_group_start = hour[0]
                current_group_end = hour[1]
                current_group_price = current_price

        groups.append((current_group_start, self._normalize_end_time(current_group_end)))
        return groups

    def _group_contiguous_hours_simple(
        self, hours: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str]]:
        """Group contiguous hours into continuous ranges without price similarity check."""
        if not hours:
            return []

        sorted_hours = sorted(hours, key=lambda x: datetime.strptime(x[0], "%H:%M"))

        groups = []
        current_group_start = sorted_hours[0][0]
        current_group_end = sorted_hours[0][1]

        for hour in sorted_hours[1:]:
            previous_end = datetime.strptime(self._normalize_end_time(current_group_end), "%H:%M")
            current_start = datetime.strptime(hour[0], "%H:%M")

            # Group if contiguous (ignore price differences)
            if current_start == previous_end:
                current_group_end = hour[1]
            else:
                groups.append((current_group_start, self._normalize_end_time(current_group_end)))
                current_group_start = hour[0]
                current_group_end = hour[1]

        groups.append((current_group_start, self._normalize_end_time(current_group_end)))
        return groups

    async def _set_battery_first(self, start_hour: str, stop_hour: str) -> None:
        """Set battery-first mode for specified time window."""
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE SET: {start_hour}-{stop_hour} "
                f"(simulated at {current_time})"
            )
            return

        payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}

        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE SET: {start_hour}-{stop_hour} "
            f"(action at {current_time}) → Topic: {self.config.battery_first_topic}"
        )

    async def _enable_ac_charge(self) -> None:
        """Enable AC charging during battery-first mode."""
        if self._ac_enabled is True:
            return  # Already enabled, skip duplicate command
            
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING ENABLED (simulated at {current_time})")
            self._ac_enabled = True
            return

        payload = {"value": True}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"⚡ AC CHARGING ENABLED at {current_time} → Topic: {self.config.ac_charge_topic}"
        )
        self._ac_enabled = True

    async def _disable_ac_charge(self) -> None:
        """Disable AC charging."""
        if self._ac_enabled is False:
            return  # Already disabled, skip duplicate command
            
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING DISABLED (simulated at {current_time})")
            self._ac_enabled = False
            return

        payload = {"value": False}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"⚡ AC CHARGING DISABLED at {current_time} → Topic: {self.config.ac_charge_topic}"
        )
        self._ac_enabled = False

    async def _disable_battery_first(self) -> None:
        """Disable battery-first mode."""
        if self.config.simulation_mode:
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

    async def _enable_export(self) -> None:
        """Enable electricity export to grid."""
        if self._export_enabled is True:
            return  # Already enabled, skip duplicate command
            
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬆️ [SIMULATE] EXPORT ENABLED (simulated at {current_time})")
            self._export_enabled = True
            return

        payload = {"value": True}
        assert self.mqtt_client is not None
        # Use the enable topic for enabling export
        await self.mqtt_client.publish(self.config.export_enable_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"⬆️ EXPORT ENABLED at {current_time} → Topic: {self.config.export_enable_topic}"
        )
        self._export_enabled = True

    async def _disable_export(self) -> None:
        """Disable electricity export to grid."""
        if self._export_enabled is False:
            return  # Already disabled, skip duplicate command
            
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬇️ [SIMULATE] EXPORT DISABLED (simulated at {current_time})")
            self._export_enabled = False
            return

        # Use command topic pattern: both enable and disable topics get {"value": True}
        # This is intentional for command-based topics (edge-triggered)
        payload = {"value": True}
        assert self.mqtt_client is not None
        # Use the disable topic for disabling export
        await self.mqtt_client.publish(self.config.export_disable_topic, json.dumps(payload))
        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"⬇️ EXPORT DISABLED at {current_time} → Topic: {self.config.export_disable_topic}"
        )
        self._export_enabled = False

    async def _set_grid_first(self, start_hour: str, stop_hour: str, stop_soc: int = 100) -> None:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc is set to 100% to prevent battery discharge to grid.
        """
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔌 [SIMULATE] GRID-FIRST MODE SET: {start_hour}-{stop_hour} "
                f"(stopSOC={stop_soc}%, simulated at {current_time})"
            )
            return

        # Set the time slot
        timeslot_payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(timeslot_payload))

        # Set stop SOC to prevent battery discharge
        stopsoc_payload = {"value": stop_soc}
        await self.mqtt_client.publish(
            self.config.grid_first_stopsoc_topic, json.dumps(stopsoc_payload)
        )

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE SET: {start_hour}-{stop_hour} (stopSOC={stop_soc}%) "
            f"at {current_time} → Topics: {self.config.grid_first_topic}, "
            f"{self.config.grid_first_stopsoc_topic}"
        )

    async def _disable_grid_first(self) -> None:
        """Disable grid-first mode."""
        if self.config.simulation_mode:
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
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⚖️ [SIMULATE] LOAD-FIRST MODE SET (simulated at {current_time})")
            return

        # Disable both battery-first and grid-first
        await self._disable_battery_first()
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
        
        # Reset mode signature and state tracking for new day
        self._last_mode_sig = None
        self._ac_enabled = None
        self._export_enabled = None

        # Determine target date using local time
        now = self._get_local_now()
        current_time = now.time()
        cutoff_time = dt_time(23, 45)

        if current_time < cutoff_time:
            days_ahead = 0
        else:
            days_ahead = 1

        target_date = self._get_local_date_string(days_ahead=days_ahead)

        self.logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Scheduling energy prices for date: {target_date}")

        # Fetch energy prices from DAM
        hourly_prices = await self._fetch_dam_energy_prices(date=target_date)

        if not hourly_prices:
            self.logger.error(
                "Failed to retrieve energy prices. Setting load-first mode and disabling export."
            )
            # Schedule fallback mode
            await self._schedule_fallback_mode()
            # Apply safe state immediately (not just at midnight)
            await self._set_load_first()
            await self._disable_export()
            self.logger.info("Applied safe state immediately due to price fetch failure")
            return

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
        # Schedule to disable battery-first at midnight
        task = asyncio.create_task(self._schedule_at_time("00:00", self._disable_battery_first))
        self._scheduled_tasks.append(task)

        # Schedule to disable export at midnight
        task = asyncio.create_task(self._schedule_at_time("00:00", self._disable_export))
        self._scheduled_tasks.append(task)

    async def _schedule_summer_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule summer battery strategy with grid-first and low-price storage."""
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
            # No low-price hours - use battery-first without AC charging all day
            self.logger.info(
                f"Summer mode: No hours below "
                f"{self.config.summer_price_threshold:.2f} CZK/kWh. "
                f"Using battery-first without AC charging all day."
            )

            # Schedule battery-first for entire day without AC charging
            self._scheduled_periods.append(("battery_first", "00:00", "23:59"))
            task = asyncio.create_task(
                self._schedule_at_time("00:00", self._set_battery_first, "00:00", "23:59")
            )
            self._scheduled_tasks.append(task)

            # Ensure AC charging is disabled
            task = asyncio.create_task(
                self._schedule_at_time("00:00:05", self._disable_ac_charge)
            )
            self._scheduled_tasks.append(task)

            # Export should be enabled all day (no low prices to avoid)
            task = asyncio.create_task(
                self._schedule_at_time("00:00:10", self._enable_export)
            )
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

        # Schedule grid-first mode for morning (before first low price)
        if first_low_start != "00:00":
            self.logger.info(
                f"Scheduling grid-first from 00:00 to {first_low_start} (sell morning solar)"
            )
            self._scheduled_periods.append(("grid_first", "00:00", first_low_start))

            # Set grid-first at midnight with stopSOC=100%
            task = asyncio.create_task(
                self._schedule_at_time(
                    "00:00", self._set_grid_first, "00:00", first_low_start, 100
                )
            )
            self._scheduled_tasks.append(task)

            # Enable export during morning high prices
            task = asyncio.create_task(
                self._schedule_at_time("00:00:05", self._enable_export)
            )
            self._scheduled_tasks.append(task)

        # Schedule battery-first during each low-price period
        previous_end = None
        for group_start, group_end in low_price_groups:
            # If there's a gap from previous period, schedule load-first
            if previous_end and previous_end < group_start:
                self.logger.info(
                    f"Scheduling load-first from {previous_end} to {group_start} (price gap)"
                )
                self._scheduled_periods.append(("load_first", previous_end, group_start))
                task = asyncio.create_task(
                    self._schedule_at_time(previous_end, self._set_load_first)
                )
                self._scheduled_tasks.append(task)

                # Re-enable export during the gap
                task = asyncio.create_task(
                    self._schedule_at_time(f"{previous_end}:05", self._enable_export)
                )
                self._scheduled_tasks.append(task)

            # Schedule battery-first for this low-price period
            self.logger.info(
                f"Scheduling battery-first from {group_start} to {group_end} "
                f"(store cheap solar, no AC charge)"
            )
            self._scheduled_periods.append(("battery_first", group_start, group_end))

            # Switch to battery-first at start of low period
            task = asyncio.create_task(
                self._schedule_at_time(
                    group_start, self._set_battery_first, group_start, group_end
                )
            )
            self._scheduled_tasks.append(task)

            # Ensure AC charging is disabled (we only want solar charging)
            task = asyncio.create_task(
                self._schedule_at_time(f"{group_start}:05", self._disable_ac_charge)
            )
            self._scheduled_tasks.append(task)

            # Disable export during low prices (no point selling below operator costs)
            task = asyncio.create_task(
                self._schedule_at_time(f"{group_start}:10", self._disable_export)
            )
            self._scheduled_tasks.append(task)

            previous_end = group_end

        # Schedule load-first for evening (after last low price)
        if last_low_end != "24:00" and last_low_end != "23:59":
            self.logger.info(
                f"Scheduling load-first from {last_low_end} to 23:59 (use stored energy)"
            )
            self._scheduled_periods.append(("load_first", last_low_end, "23:59"))

            task = asyncio.create_task(
                self._schedule_at_time(last_low_end, self._set_load_first)
            )
            self._scheduled_tasks.append(task)

            # Re-enable export for evening high prices
            task = asyncio.create_task(
                self._schedule_at_time(f"{last_low_end}:05", self._enable_export)
            )
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
            self._scheduled_periods.append(("battery_first", group_start, group_end))
            task = asyncio.create_task(
                self._schedule_at_time(group_start, self._set_battery_first, group_start, group_end)
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
                charge_prices = [
                    price
                    for start, stop, price in ac_charge_hours
                    if start >= start_time and stop <= stop_time
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
                self._scheduled_periods.append(("ac_charge", start_time, stop_time))

                # Schedule AC charge start
                task = asyncio.create_task(
                    self._schedule_at_time(start_time, self._enable_ac_charge)
                )
                self._scheduled_tasks.append(task)

                # Schedule AC charge stop (normalize if 24:00)
                stop_time_norm = self._normalize_end_time(stop_time)
                task = asyncio.create_task(
                    self._schedule_at_time(stop_time_norm, self._disable_ac_charge)
                )
                self._scheduled_tasks.append(task)

        # Schedule export control
        await self._schedule_export_control(hourly_prices, eur_czk_rate)

    async def _schedule_export_control(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule export enable/disable based on price thresholds."""
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
            task = asyncio.create_task(self._schedule_at_time("00:00:05", self._disable_export))
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
        export_groups.sort(key=lambda x: datetime.strptime(x[0], "%H:%M"))

        # Track previous end time to schedule disable between periods
        previous_end: Optional[str] = None

        for group_start, group_end in export_groups:
            # Handle 24:00 edge case
            group_end = self._normalize_end_time(group_end)

            # If there's a gap between previous export period and this one, disable export
            if previous_end is not None and previous_end < group_start:
                self.logger.info(
                    f"Scheduling export disable between periods at {previous_end} "
                    f"(gap until {group_start})"
                )
                task = asyncio.create_task(
                    self._schedule_at_time(previous_end, self._disable_export)
                )
                self._scheduled_tasks.append(task)

            # Calculate price statistics for this time range
            group_prices = [
                price
                for start, stop, price in export_hours_with_price
                if start >= group_start and stop <= group_end
            ]
            if group_prices:
                min_price = min(group_prices)
                max_price = max(group_prices)
                avg_price = sum(group_prices) / len(group_prices)

                # Convert prices to CZK/kWh for display
                eur_czk_rate = await self._get_eur_czk_rate()
                min_price_czk_kwh = min_price * eur_czk_rate / 1000
                max_price_czk_kwh = max_price * eur_czk_rate / 1000
                avg_price_czk_kwh = avg_price * eur_czk_rate / 1000

                self.logger.info(
                    f"Scheduling export enable from {group_start} to {group_end} "
                    f"(min: {min_price:.2f}, avg: {avg_price:.2f}, "
                    f"max: {max_price:.2f} EUR/MWh = "
                    f"{min_price_czk_kwh:.2f}-{avg_price_czk_kwh:.2f}-"
                    f"{max_price_czk_kwh:.2f} CZK/kWh, "
                    f"threshold: {self.config.export_price_threshold:.2f})"
                )
            else:
                self.logger.info(
                    f"Scheduling export enable from {group_start} to {group_end} (no price data)"
                )

            # Track export period
            self._scheduled_periods.append(("export", group_start, group_end))

            # Schedule export enable at start
            task = asyncio.create_task(self._schedule_at_time(group_start, self._enable_export))
            self._scheduled_tasks.append(task)

            # Always schedule export disable at end, but handle midnight transitions
            # For periods ending at 23:59, schedule disable at 23:59:55 to ensure
            # it happens before any potential midnight enable/disable
            if group_end == "23:59":
                task = asyncio.create_task(self._schedule_at_time("23:59:55", self._disable_export))
            else:
                task = asyncio.create_task(self._schedule_at_time(group_end, self._disable_export))
            self._scheduled_tasks.append(task)

            # Update previous end time
            previous_end = group_end

        # Check if we need a midnight disable (only if no export period starts at 00:00)
        has_midnight_start = any(group[0] == "00:00" for group in export_groups)
        if not has_midnight_start:
            task = asyncio.create_task(self._schedule_at_time("00:00", self._disable_export))
            self._scheduled_tasks.append(task)
            self.logger.info("Scheduled export disable at 00:00 for clean daily start")

    async def _schedule_at_time(self, time_str: str, coro_func: Any, *args: Any) -> None:
        """Schedule a coroutine to run at a specific time."""
        try:
            # Normalize impossible time to end-of-day
            if time_str in ("24:00", "24:00:00"):
                time_str = "23:59:55"

            # Support both HH:MM and HH:MM:SS formats
            try:
                target_time = datetime.strptime(time_str, "%H:%M:%S").time()
            except ValueError:
                # Allow HH:MM and normalize 24:00 here too if it slips through
                if time_str == "24:00":
                    time_str = "23:59:55"
                    target_time = datetime.strptime(time_str, "%H:%M:%S").time()
                else:
                    target_time = datetime.strptime(time_str, "%H:%M").time()

            now = self._get_local_now()

            # Calculate next occurrence of target time in local timezone
            target_datetime = datetime.combine(now.date(), target_time, self._local_tz)
            if target_datetime <= now:
                target_datetime += timedelta(days=1)

            delay = (target_datetime - now).total_seconds()

            if delay > 0:
                await asyncio.sleep(delay)
                await coro_func(*args)

        except asyncio.CancelledError:
            # Expected on shutdown; keep quiet
            raise
        except Exception as e:
            self.logger.error(f"Error in scheduled task at {time_str}: {e}", exc_info=True)

    async def _schedule_daily_calculation(self) -> None:
        """Schedule daily calculation at 23:59."""
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

                self.logger.info(f"Next daily calculation scheduled in {delay / 3600:.1f} hours")
                await asyncio.sleep(delay)

                # Run calculation
                await self._calculate_and_schedule_next_day()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in daily calculation loop: {e}", exc_info=True)
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)
