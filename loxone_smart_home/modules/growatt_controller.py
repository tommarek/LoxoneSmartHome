"""Growatt controller module - manages solar battery based on energy prices."""

import asyncio
import json
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from loxone_smart_home.config.settings import Settings
from loxone_smart_home.modules.base import BaseModule
from loxone_smart_home.utils.influxdb_client import SharedInfluxDBClient
from loxone_smart_home.utils.mqtt_client import SharedMQTTClient


class EnergyPriceData:
    """Data class for energy price information."""

    def __init__(self, prices: Dict[Tuple[str, str], float]) -> None:
        """Initialize energy price data."""
        self.prices = prices
        self.timestamp = datetime.now()


class GrowattController(BaseModule):
    """Growatt controller that manages battery charging based on energy prices."""

    def __init__(
        self,
        mqtt_client: SharedMQTTClient,
        influxdb_client: SharedInfluxDBClient,
        settings: Settings,
    ) -> None:
        """Initialize the Growatt controller."""
        super().__init__(
            name="GrowattController",
            mqtt_client=mqtt_client,
            influxdb_client=influxdb_client,
            settings=settings,
        )
        self.config = settings.growatt

        # Scheduled tasks
        self._scheduled_tasks: List[asyncio.Task[None]] = []
        self._daily_schedule_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the Growatt controller."""
        await self._schedule_daily_calculation()
        self.logger.info("Growatt controller started")

    async def stop(self) -> None:
        """Stop the Growatt controller."""
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()

        if self._daily_schedule_task:
            self._daily_schedule_task.cancel()

        # Disable battery first mode on shutdown
        try:
            await self._disable_battery_first()
        except Exception as e:
            self.logger.error(f"Error disabling battery first on shutdown: {e}")

        self.logger.info("Growatt controller stopped")

    async def _fetch_dam_energy_prices(
        self, date: Optional[str] = None
    ) -> Dict[Tuple[str, str], float]:
        """Fetch energy prices from OTE DAM API."""
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        url = (
            "https://www.ote-cr.cz/en/short-term-markets/electricity/"
            f"day-ahead-market/@@chart-data?report_date={date}"
        )
        self.logger.info(f"Fetching DAM energy prices for {date} from: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch DAM prices: HTTP {response.status}")
                        return {}

                    data = await response.json()

            hourly_prices = {}
            if data.get("data", {}).get("dataLine"):
                price_data = data["data"]["dataLine"][1]["point"]

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
        intervals = list(prices.keys())
        num_intervals = len(intervals)

        if num_intervals < 2:
            return []

        # Determine interval duration
        interval_duration = (
            datetime.strptime(intervals[1][0], "%H:%M")
            - datetime.strptime(intervals[0][0], "%H:%M")
        ).seconds // 60

        if interval_duration == 15:
            intervals_per_hour = 4
        elif interval_duration == 60:
            intervals_per_hour = 1
        else:
            self.logger.warning(f"Unknown interval duration: {interval_duration} minutes")
            return []

        intervals_needed = x * intervals_per_hour

        if num_intervals < intervals_needed:
            return []

        cheapest_window = []
        min_price_sum = float("inf")

        for i in range(num_intervals - intervals_needed + 1):
            window_intervals = intervals[i : i + intervals_needed]
            price_sum = sum(prices[interval] for interval in window_intervals)

            if price_sum < min_price_sum:
                min_price_sum = price_sum
                cheapest_window = [
                    (interval[0], interval[1], prices[interval]) for interval in window_intervals
                ]

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
            "Cheapest": [], "Cheap": [], "Expensive": [], "Most Expensive": []
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
            previous_end = datetime.strptime(current_group_end, "%H:%M")
            current_start = datetime.strptime(hour[0], "%H:%M")
            current_price = hour[2]

            # Group if contiguous and price difference < 20%
            if current_start == previous_end and abs(current_price - current_group_price) < abs(
                current_group_price * 0.2
            ):
                current_group_end = hour[1]
            else:
                groups.append((current_group_start, current_group_end))
                current_group_start = hour[0]
                current_group_end = hour[1]
                current_group_price = current_price

        groups.append((current_group_start, current_group_end))
        return groups

    async def _set_battery_first(self, start_hour: str, stop_hour: str) -> None:
        """Set battery-first mode for specified time window."""
        if self.config.simulation_mode:
            self.logger.info(
                f"[SIMULATE] Would set battery-first mode from {start_hour} to {stop_hour}"
            )
            return

        payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}

        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))
        self.logger.info(f"Battery-first mode scheduled from {start_hour} to {stop_hour}")

    async def _enable_ac_charge(self) -> None:
        """Enable AC charging during battery-first mode."""
        if self.config.simulation_mode:
            self.logger.info("[SIMULATE] Would enable AC charging")
            return

        payload = {"value": True}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))
        self.logger.info("AC charging enabled")

    async def _disable_ac_charge(self) -> None:
        """Disable AC charging."""
        if self.config.simulation_mode:
            self.logger.info("[SIMULATE] Would disable AC charging")
            return

        payload = {"value": False}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))
        self.logger.info("AC charging disabled")

    async def _disable_battery_first(self) -> None:
        """Disable battery-first mode."""
        if self.config.simulation_mode:
            self.logger.info("[SIMULATE] Would disable battery-first mode")
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))
        self.logger.info("Battery-first mode disabled")

    async def _enable_export(self) -> None:
        """Enable electricity export to grid."""
        if self.config.simulation_mode:
            self.logger.info("[SIMULATE] Would enable electricity export to grid")
            return

        payload = {"value": True}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.export_enable_topic, json.dumps(payload))
        self.logger.info("Enabled electricity export to grid")

    async def _disable_export(self) -> None:
        """Disable electricity export to grid."""
        if self.config.simulation_mode:
            self.logger.info("[SIMULATE] Would disable electricity export to grid")
            return

        payload = {"value": False}
        assert self.mqtt_client is not None
        await self.mqtt_client.publish(self.config.export_disable_topic, json.dumps(payload))
        self.logger.info("Disabled electricity export to grid")

    async def _calculate_and_schedule_next_day(self) -> None:
        """Calculate energy prices and schedule battery control for next day."""
        # Cancel previous scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()
        self._scheduled_tasks.clear()

        # Determine target date
        now = datetime.now()
        current_time = now.time()
        cutoff_time = dt_time(23, 45)

        if current_time < cutoff_time:
            days_ahead = 0
        else:
            days_ahead = 1

        target_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

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
            return

        self.logger.info(f"Energy prices for {target_date}: {len(hourly_prices)} hours")

        # Analyze prices
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

        self.logger.info(f"Cheapest individual hours: {len(cheapest_individual_hours)}")
        self.logger.info(f"Cheapest consecutive hours: {len(cheapest_consecutive)}")
        self.logger.info(f"Total cheap hours: {len(all_cheap_hours)}")

        # Schedule battery-first mode
        await self._schedule_battery_control(all_cheap_hours, cheapest_consecutive, hourly_prices)

    async def _schedule_fallback_mode(self) -> None:
        """Schedule fallback mode when price data is unavailable."""
        # Schedule to disable battery-first at midnight
        task = asyncio.create_task(self._schedule_at_time("00:00", self._disable_battery_first))
        self._scheduled_tasks.append(task)

        # Schedule to disable export at midnight
        task = asyncio.create_task(self._schedule_at_time("00:00", self._disable_export))
        self._scheduled_tasks.append(task)

    async def _schedule_battery_control(
        self,
        all_cheap_hours: set[Tuple[str, str]],
        cheapest_consecutive: set[Tuple[str, str]],
        hourly_prices: Dict[Tuple[str, str], float],
    ) -> None:
        """Schedule battery control based on price analysis."""
        # Group battery-first hours into contiguous blocks
        battery_first_hours = [
            (start, stop, hourly_prices[(start, stop)]) for start, stop in all_cheap_hours
        ]
        battery_first_groups = self._group_contiguous_hours(battery_first_hours)

        # Schedule battery-first mode for each block
        for group_start, group_end in battery_first_groups:
            self.logger.info(f"Scheduling battery-first mode from {group_start} to {group_end}")
            task = asyncio.create_task(
                self._schedule_at_time(group_start, self._set_battery_first, group_start, group_end)
            )
            self._scheduled_tasks.append(task)

        # Schedule AC charging during cheapest consecutive hours
        if cheapest_consecutive:
            ac_charge_hours = [
                (start, stop, hourly_prices[(start, stop)]) for start, stop in cheapest_consecutive
            ]
            ac_charge_groups = self._group_contiguous_hours(ac_charge_hours)

            for start_time, stop_time in ac_charge_groups:
                self.logger.info(f"Scheduling AC charge from {start_time} to {stop_time}")

                # Schedule AC charge start
                task = asyncio.create_task(
                    self._schedule_at_time(start_time, self._enable_ac_charge)
                )
                self._scheduled_tasks.append(task)

                # Schedule AC charge stop
                task = asyncio.create_task(
                    self._schedule_at_time(stop_time, self._disable_ac_charge)
                )
                self._scheduled_tasks.append(task)

        # Schedule export control
        await self._schedule_export_control(hourly_prices)

    async def _schedule_export_control(self, hourly_prices: Dict[Tuple[str, str], float]) -> None:
        """Schedule export enable/disable based on price thresholds."""
        export_hours = [
            (start, stop)
            for (start, stop), price in hourly_prices.items()
            if price >= self.config.export_price_threshold
        ]

        if not export_hours:
            self.logger.info("No hours above export price threshold")
            return

        # Group export hours into contiguous blocks
        export_hours_with_price = [
            (start, stop, hourly_prices[(start, stop)]) for start, stop in export_hours
        ]
        export_groups = self._group_contiguous_hours(export_hours_with_price)

        for group_start, group_end in export_groups:
            # Handle 24:00 edge case
            if group_end == "24:00":
                group_end = "23:59"

            self.logger.info(f"Scheduling export enable from {group_start} to {group_end}")

            # Schedule export enable
            task = asyncio.create_task(self._schedule_at_time(group_start, self._enable_export))
            self._scheduled_tasks.append(task)

            # Schedule export disable
            task = asyncio.create_task(self._schedule_at_time(group_end, self._disable_export))
            self._scheduled_tasks.append(task)

    async def _schedule_at_time(self, time_str: str, coro_func: Any, *args: Any) -> None:
        """Schedule a coroutine to run at a specific time."""
        try:
            target_time = datetime.strptime(time_str, "%H:%M").time()
            now = datetime.now()

            # Calculate next occurrence of target time
            target_datetime = datetime.combine(now.date(), target_time)
            if target_datetime <= now:
                target_datetime += timedelta(days=1)

            delay = (target_datetime - now).total_seconds()

            if delay > 0:
                await asyncio.sleep(delay)
                await coro_func(*args)

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
                # Calculate time until next 23:59
                now = datetime.now()
                target_time = dt_time(self.config.schedule_hour, self.config.schedule_minute)
                target_datetime = datetime.combine(now.date(), target_time)

                if target_datetime <= now:
                    target_datetime += timedelta(days=1)

                delay = (target_datetime - now).total_seconds()

                self.logger.info(f"Next daily calculation scheduled in {delay/3600:.1f} hours")
                await asyncio.sleep(delay)

                # Run calculation
                await self._calculate_and_schedule_next_day()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in daily calculation loop: {e}", exc_info=True)
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)
