"""Data aggregation service for efficient data processing."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .cache import CacheService


logger = logging.getLogger(__name__)


class DataAggregator:
    """Service for aggregating and processing time-series data."""

    def __init__(self, influxdb_client: Any, cache: CacheService):
        """Initialize data aggregator."""
        self.influxdb_client = influxdb_client
        self.cache = cache
        self._running = False
        self._aggregation_tasks: List[asyncio.Task[None]] = []

    async def run(self) -> None:
        """Run aggregation service."""
        self._running = True

        # Start different aggregation tasks
        tasks = [
            self._aggregate_15min_data(),
            self._aggregate_hourly_data(),
            self._aggregate_daily_data()
        ]

        self._aggregation_tasks = [asyncio.create_task(task) for task in tasks]

        # Wait for all tasks
        try:
            await asyncio.gather(*self._aggregation_tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop aggregation service."""
        self._running = False

        for task in self._aggregation_tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._aggregation_tasks, return_exceptions=True)

    async def _aggregate_15min_data(self) -> None:
        """Aggregate 15-minute data for real-time views."""
        while self._running:
            try:
                now = datetime.now()
                # Round to previous 15-minute mark
                minutes = (now.minute // 15) * 15
                current_block = now.replace(minute=minutes, second=0, microsecond=0)
                prev_block = current_block - timedelta(minutes=15)

                # Query recent data
                query = f'''
                from(bucket: "solar")
                  |> range(start: {prev_block.isoformat()}, stop: {current_block.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "power_flow")
                  |> aggregateWindow(every: 15m, fn: mean)
                '''

                result = await self.influxdb_client.query(query)

                if result:
                    # Process and cache aggregated data
                    aggregated = self._process_15min_aggregation(result)
                    cache_key = f"agg:15m:{current_block.isoformat()}"
                    await self.cache.set(cache_key, aggregated, ttl=3600)

                # Wait until next 15-minute mark
                next_block = current_block + timedelta(minutes=15)
                sleep_seconds = (next_block - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                else:
                    await asyncio.sleep(60)  # Default wait

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 15-minute aggregation: {e}")
                await asyncio.sleep(60)

    async def _aggregate_hourly_data(self) -> None:
        """Aggregate hourly data for daily charts."""
        while self._running:
            try:
                now = datetime.now()
                current_hour = now.replace(minute=0, second=0, microsecond=0)
                prev_hour = current_hour - timedelta(hours=1)

                # Query hourly data
                query = f'''
                from(bucket: "solar")
                  |> range(start: {prev_hour.isoformat()}, stop: {current_hour.isoformat()})
                  |> filter(fn: (r) => r["_measurement"] == "energy")
                  |> aggregateWindow(every: 1h, fn: sum)
                '''

                result = await self.influxdb_client.query(query)

                if result:
                    # Process and cache
                    aggregated = self._process_hourly_aggregation(result)
                    cache_key = f"agg:1h:{current_hour.isoformat()}"
                    await self.cache.set(cache_key, aggregated, ttl=86400)

                # Wait until next hour
                next_hour = current_hour + timedelta(hours=1)
                sleep_seconds = (next_hour - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                else:
                    await asyncio.sleep(300)  # Default 5 min wait

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in hourly aggregation: {e}")
                await asyncio.sleep(300)

    async def _aggregate_daily_data(self) -> None:
        """Aggregate daily data for monthly/yearly views."""
        while self._running:
            try:
                now = datetime.now()
                today = now.date()
                yesterday = today - timedelta(days=1)

                # Query daily totals
                query = f'''
                from(bucket: "solar")
                  |> range(start: {yesterday.isoformat()}T00:00:00Z,
                           stop: {today.isoformat()}T00:00:00Z)
                  |> filter(fn: (r) => r["_measurement"] == "energy")
                  |> aggregateWindow(every: 1d, fn: sum)
                '''

                result = await self.influxdb_client.query(query)

                if result:
                    # Process and cache
                    aggregated = self._process_daily_aggregation(result, yesterday)
                    cache_key = f"agg:1d:{yesterday.isoformat()}"
                    await self.cache.set(cache_key, aggregated, ttl=2592000)  # 30 days

                # Wait until next day
                tomorrow = today + timedelta(days=1)
                next_run = datetime.combine(tomorrow, datetime.min.time())
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                else:
                    await asyncio.sleep(3600)  # Default 1 hour wait

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daily aggregation: {e}")
                await asyncio.sleep(3600)

    def _process_15min_aggregation(self, result: Any) -> Dict[str, Any]:
        """Process 15-minute aggregation result."""
        # TODO: Implement actual processing based on InfluxDB result structure
        return {
            "timestamp": datetime.now().isoformat(),
            "production_avg": 2500,
            "consumption_avg": 2200,
            "grid_import_avg": 0,
            "grid_export_avg": 300,
            "battery_power_avg": 200,
            "battery_soc_end": 75
        }

    def _process_hourly_aggregation(self, result: Any) -> Dict[str, Any]:
        """Process hourly aggregation result."""
        # TODO: Implement actual processing
        return {
            "timestamp": datetime.now().isoformat(),
            "production_total": 2.5,  # kWh
            "consumption_total": 2.2,
            "grid_import_total": 0,
            "grid_export_total": 0.3,
            "battery_charge_total": 0.5,
            "battery_discharge_total": 0.2,
            "self_sufficiency": 100,
            "self_consumption": 88
        }

    def _process_daily_aggregation(self, result: Any, date: Any) -> Dict[str, Any]:
        """Process daily aggregation result."""
        # TODO: Implement actual processing
        return {
            "date": date.isoformat(),
            "production_total": 25.5,  # kWh
            "consumption_total": 22.0,
            "grid_import_total": 2.5,
            "grid_export_total": 6.0,
            "battery_cycles": 1.5,
            "peak_production": 3.8,  # kW
            "peak_consumption": 4.2,
            "self_sufficiency": 88.6,
            "self_consumption": 72.5,
            "cost_savings": 125.50,  # CZK
            "co2_avoided": 12.75  # kg
        }

    async def get_aggregated_range(
        self,
        start: datetime,
        end: datetime,
        resolution: str
    ) -> List[Dict[str, Any]]:
        """Get aggregated data for a time range."""
        cache_prefix = f"agg:{resolution}"
        aggregated_data: List[Dict[str, Any]] = []

        # Calculate time blocks based on resolution
        if resolution == "15m":
            delta = timedelta(minutes=15)
        elif resolution == "1h":
            delta = timedelta(hours=1)
        elif resolution == "1d":
            delta = timedelta(days=1)
        else:
            return aggregated_data

        current = start
        while current < end:
            # Try to get from cache
            cache_key = f"{cache_prefix}:{current.isoformat()}"
            cached_data = await self.cache.get(cache_key)

            if cached_data:
                aggregated_data.append(cached_data)
            else:
                # If not cached, we could trigger a query here
                pass

            current += delta

        return aggregated_data
