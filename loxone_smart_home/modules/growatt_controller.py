"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

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


@dataclass(frozen=True)
class Period:
    """Represents a scheduled period with proper time types."""
    kind: PeriodType
    start: dt_time
    end: dt_time

    def contains_time(self, t: dt_time) -> bool:
        """Check if a time falls within this period, handling midnight wrap."""
        if self.start <= self.end:
            return self.start <= t < self.end
        else:
            return t >= self.start or t < self.end

    def to_string_tuple(self) -> Tuple[str, str, str]:
        """Convert to legacy string tuple format for logging."""
        return (self.kind, self.start.strftime("%H:%M"), self.end.strftime("%H:%M"))


class EnergyPriceData:
    """Data class for energy price information."""

    def __init__(self, prices: Dict[Tuple[str, str], float]) -> None:
        """Initialize energy price data."""
        self.prices = prices
        # Use Prague timezone for energy price data
        self.timestamp = datetime.now(zoneinfo.ZoneInfo("Europe/Prague"))


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    def _select_primary_mode(self, modes: set[PeriodType]) -> PeriodType:
        """Select primary mode based on precedence rules."""
        for m in MODE_PRECEDENCE:
            if m in modes:
                return m
        return "load_first"  # Default mode

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
        return "23:59" if s in ("24:00", "23:60", "24:00:00") else s

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
            end_str = period.end.strftime("%H:%M")
            key = f"{start_str}-{end_str}"
            if key not in time_slots:
                time_slots[key] = {
                    "start": start_str,
                    "end": end_str,
                    "modes": set(),
                    "primary_mode": None
                }
            time_slots[key]["modes"].add(period.kind)

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
                entry["details"].append("StopSOC: 20%")
                entry["details"].append("PowerRate: 10%")
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
                            self.config.eur_czk_rate if self.config.eur_czk_rate > 0 else 25.0
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
            desired_mode = ("grid_first", 20, 10)  # Default stopSOC=20%, powerRate=10%
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
            next_hour = "23:59"  # Stop at end of day
        else:
            next_hour = end.strftime("%H:00")

        if desired_mode[0] == "battery_first":
            await self._set_mode("battery_first", current_hour, next_hour)
        elif desired_mode[0] == "grid_first":
            await self._set_mode(
                "grid_first", current_hour, next_hour, desired_mode[1], desired_mode[2]
            )
        else:
            await self._set_mode("load_first")

        # AC charging (honor summer mode rule)
        season = await self._get_season_mode()
        if season == "summer" and want_ac:
            self.logger.info("Summer mode active, disabling AC charging despite schedule")
            want_ac = False

        if want_ac:
            await self._enable_ac_charge()
        else:
            await self._disable_ac_charge()

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
                if len(lines) >= 2:
                    # Use second line (index 1) which has EUR/MWh prices
                    price_data = lines[1].get("point", [])
                else:
                    # Fallback to first line if only one exists
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
                merge_policy = getattr(self.config, "dst_merge_policy", "avg")

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

    async def _set_ac_charge(self, enabled: bool) -> None:
        """Set AC charging state (unified setter)."""
        if self._ac_enabled == enabled:
            return  # Already in desired state

        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING {state} (simulated at {current_time})")
            self._ac_enabled = enabled
            return

        payload = {"value": enabled}
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

    async def _set_export(self, enabled: bool) -> None:
        """Set export state (unified setter handling edge-triggered topics)."""
        if self._export_enabled == enabled:
            return  # Already in desired state

        if self.config.simulation_mode:
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
        self, start_hour: str, stop_hour: str, stop_soc: int = 20, power_rate: int = 10
    ) -> None:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc: Battery level to stop discharging at (default 20%)
        power_rate: Discharge rate in % (default 10%)
        """
        if self.config.simulation_mode:
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔌 [SIMULATE] GRID-FIRST MODE SET: {start_hour}-{stop_hour} "
                f"(stopSOC={stop_soc}%, powerRate={power_rate}%, simulated at {current_time})"
            )
            return

        # Set the time slot
        timeslot_payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(timeslot_payload))

        # Set stop SOC (battery level to stop discharging)
        stopsoc_payload = {"value": stop_soc}
        await self.mqtt_client.publish(
            self.config.grid_first_stopsoc_topic, json.dumps(stopsoc_payload)
        )

        # Set power rate (discharge rate)
        powerrate_payload = {"value": power_rate}
        await self.mqtt_client.publish(
            self.config.grid_first_powerrate_topic, json.dumps(powerrate_payload)
        )

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE SET: {start_hour}-{stop_hour} "
            f"(stopSOC={stop_soc}%, powerRate={power_rate}%) at {current_time} → "
            f"Topics: {self.config.grid_first_topic}, {self.config.grid_first_stopsoc_topic}, "
            f"{self.config.grid_first_powerrate_topic}"
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
        # Schedule to disable battery-first at midnight with jitter
        task = asyncio.create_task(self._schedule_at_time("00:00:05", self._disable_battery_first))
        self._scheduled_tasks.append(task)

        # Schedule to disable export at midnight with jitter
        task = asyncio.create_task(self._schedule_at_time("00:00:10", self._disable_export))
        self._scheduled_tasks.append(task)

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

                task = asyncio.create_task(
                    self._schedule_at_time("00:00", self._set_load_first)
                )
                self._scheduled_tasks.append(task)

                # Enable export for overnight period (prices above threshold)
                task = asyncio.create_task(
                    self._schedule_at_time("00:00:05", self._enable_export)
                )
                self._scheduled_tasks.append(task)

            # Schedule Grid-First from sunrise to sunset (sell solar + battery at 10% rate)
            self.logger.info(
                f"Scheduling grid-first from {sunrise_str} to 23:59 "
                f"(sell solar + battery, stopSOC=20%, powerRate=10%)"
            )
            self._scheduled_periods.append(
                Period("grid_first", sunrise_time, dt_time(23, 59))
            )
            self._scheduled_periods.append(
                Period("export", sunrise_time, dt_time(23, 59))
            )

            task = asyncio.create_task(
                self._schedule_at_time(
                    sunrise_str, self._set_grid_first, sunrise_str, "23:59", 20, 10
                )
            )
            self._scheduled_tasks.append(task)

            # Enable export after sunrise
            task = asyncio.create_task(
                self._schedule_at_time(f"{sunrise_str}:05", self._enable_export)
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
                task = asyncio.create_task(
                    self._schedule_at_time("00:00:05", self._enable_export)
                )
                self._scheduled_tasks.append(task)

            task = asyncio.create_task(
                self._schedule_at_time("00:00", self._set_load_first)
            )
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
                    self._parse_hhmm(grid_first_end)
                )
            )
            self._scheduled_periods.append(
                Period("export", sunrise_time, self._parse_hhmm(grid_first_end))
            )

            # Set grid-first at sunrise with stopSOC=20% and powerRate=10%
            task = asyncio.create_task(
                self._schedule_at_time(
                    sunrise_str, self._set_grid_first, sunrise_str, grid_first_end, 20, 10
                )
            )
            self._scheduled_tasks.append(task)

            # Enable export during morning high prices
            task = asyncio.create_task(
                self._schedule_at_time(f"{sunrise_str}:05", self._enable_export)
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
                    task = asyncio.create_task(
                        self._schedule_at_time(previous_end, self._set_load_first)
                    )
                    self._scheduled_tasks.append(task)

                    # Re-enable export during the gap (prices above threshold)
                    task = asyncio.create_task(
                        self._schedule_at_time(f"{previous_end}:05", self._enable_export)
                    )
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
            self._scheduled_periods.append(
                Period("load_first", self._parse_hhmm(last_low_end), dt_time(23, 59))
            )
            self._scheduled_periods.append(
                Period("export", self._parse_hhmm(last_low_end), dt_time(23, 59))
            )  # Enable export for excess

            task = asyncio.create_task(
                self._schedule_at_time(last_low_end, self._set_load_first)
            )
            self._scheduled_tasks.append(task)

            # Enable export for evening (in case of excess energy)
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
            self._scheduled_periods.append(
                Period("battery_first", self._parse_hhmm(group_start), self._parse_hhmm(group_end))
            )
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
        # Build AC charge windows for overlap checking
        ac_windows: List[Tuple[dt_time, dt_time]] = [
            (p.start, p.end) for p in self._scheduled_periods if p.kind == "ac_charge"
        ]

        def _segments(start: dt_time, end: dt_time) -> List[Tuple[dt_time, dt_time]]:
            """Split a possibly midnight-wrapping range into 1-2 non-wrapping segments."""
            if start <= end:
                return [(start, end)]
            # Wraps midnight: [start, 23:59:59) U [00:00, end)
            return [(start, dt_time(23, 59, 59)), (dt_time(0, 0), end)]

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
        export_groups.sort(key=lambda x: self._parse_hhmm(x[0]))

        # Track previous end time to schedule disable between periods
        previous_end: Optional[str] = None
        midnight_enable_scheduled = False  # Track if we actually scheduled a 00:00 enable
        last_enable_started = False  # Track if we enabled the preceding window

        for group_start, group_end in export_groups:
            # Handle 24:00 edge case
            group_end = self._normalize_end_time(group_end)

            # If there's a gap and we previously enabled, schedule disable
            if previous_end is not None and last_enable_started:
                prev_end_t = self._parse_hhmm(previous_end)
                group_start_t = self._parse_hhmm(group_start)
                if prev_end_t < group_start_t:
                    self.logger.info(
                        f"Scheduling export disable between periods at {previous_end} "
                        f"(gap until {group_start})"
                    )
                    task = asyncio.create_task(
                        self._schedule_at_time(previous_end, self._disable_export)
                    )
                    self._scheduled_tasks.append(task)

            # Check if export overlaps with AC charging (if suppression enabled)
            suppress_export = getattr(self.config, "suppress_export_during_ac_charge", True)
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
                    previous_end = group_end
                    last_enable_started = False
                    continue
            else:
                # No AC overlap, use full window
                export_slices = [(self._parse_hhmm(group_start), self._parse_hhmm(group_end))]

            # Schedule each export slice
            scheduled_any_slice = False
            for slice_start_t, slice_end_t in export_slices:
                slice_start = slice_start_t.strftime("%H:%M")
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
                task = asyncio.create_task(self._schedule_at_time(slice_start, self._enable_export))
                self._scheduled_tasks.append(task)

                # Schedule export disable at end
                if slice_end == "23:59":
                    disable_at = "23:59:55"
                else:
                    disable_at = slice_end
                task = asyncio.create_task(self._schedule_at_time(disable_at, self._disable_export))
                self._scheduled_tasks.append(task)

                scheduled_any_slice = True

            # Update tracking variables
            last_enable_started = scheduled_any_slice
            previous_end = group_end

        # Check if we need a midnight disable (only if we didn't schedule a 00:00 enable)
        if not midnight_enable_scheduled:
            # Add jitter to avoid collision with other midnight actions
            task = asyncio.create_task(self._schedule_at_time("00:00:05", self._disable_export))
            self._scheduled_tasks.append(task)
            self.logger.info("Scheduled export disable at 00:00:05 for clean daily start")

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
