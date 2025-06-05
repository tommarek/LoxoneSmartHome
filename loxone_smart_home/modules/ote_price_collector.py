"""OTE price collector module - downloads and stores electricity prices from OTE.

Rate Limiting Strategy:
- 1 second delay between requests (configurable via OTE_REQUEST_DELAY)
- 5 second delay after errors (configurable via OTE_ERROR_DELAY)
- 3 retry attempts per failed request with exponential backoff
- Browser-like headers to avoid being flagged as a bot
- Connection pooling with max 2 connections per host
- Automatic resume from last successful download point
- Extended breaks (1 minute) after 5+ consecutive failures
- Graceful handling of rate limiting (HTTP 429) responses
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import aiohttp
import pytz

from config.settings import Settings
from modules.base import BaseModule
from utils.async_influxdb_client import AsyncInfluxDBClient


class OTEPriceCollector(BaseModule):
    """Collects electricity prices from OTE and stores them in InfluxDB."""

    def __init__(
        self,
        influxdb_client: AsyncInfluxDBClient,
        settings: Settings,
    ) -> None:
        """Initialize the OTE price collector."""
        super().__init__(
            name="OTEPriceCollector",
            service_name="OTE",
            influxdb_client=influxdb_client,
            settings=settings,
        )

        # OTE API configuration
        self.base_url = settings.ote.base_url
        self.time_resolution = settings.ote.time_resolution

        # Local timezone (Prague/Czech Republic)
        self._local_tz = pytz.timezone("Europe/Prague")

        # InfluxDB bucket for OTE prices
        self.bucket_name = "ote_prices"

        # Session for HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None

        # Background tasks
        self._daily_update_task: Optional[asyncio.Task[None]] = None

        # Track last successful update
        self._last_update_date: Optional[datetime] = None

    async def start(self) -> None:
        """Start the OTE price collector."""
        # Configure session with connection limits and timeouts
        connector = aiohttp.TCPConnector(
            limit=10,  # Total connection pool size
            limit_per_host=2,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=30,  # Socket read timeout
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "LoxoneSmartHome/1.0 (Data Collector)"},
        )

        # Check if we need to do initial historical data load
        if await self._needs_historical_load():
            self.logger.info("Starting historical data download for last 3 years...")
            await self._load_historical_data()

        # Start daily updates
        self._daily_update_task = asyncio.create_task(self._daily_update_loop())

        self.logger.info("OTE price collector started")

    async def stop(self) -> None:
        """Stop the OTE price collector."""
        # Cancel daily update task
        if self._daily_update_task and not self._daily_update_task.done():
            self._daily_update_task.cancel()
            try:
                await self._daily_update_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._session:
            await self._session.close()

        self.logger.info("OTE price collector stopped")

    async def _needs_historical_load(self) -> bool:
        """Check if we need to load historical data."""
        try:
            # Query the latest timestamp in the database
            if not self.influxdb_client:
                return True

            query = f"""
            from(bucket: "{self.bucket_name}")
                |> range(start: -3y)
                |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
                |> filter(fn: (r) => r["_field"] == "price")
                |> last()
            """

            result = await self.influxdb_client.query(query)

            # If we have data, check if it's recent
            if result and len(result) > 0:
                for table in result:
                    for record in table.records:
                        last_time = record.get_time()
                        if last_time:
                            # If we have data from yesterday or today, skip historical load
                            days_old = (datetime.now(timezone.utc) - last_time).days
                            if days_old <= 1:
                                self.logger.info(
                                    f"Found recent data from {days_old} days ago, "
                                    "skipping historical load"
                                )
                                return False

            self.logger.info("No recent data found, will load historical data")
            return True

        except Exception as e:
            self.logger.warning(f"Error checking existing data: {e}, will load historical data")
            return True

    async def _find_earliest_missing_date(self, start_date, end_date):
        """Find the earliest date that's missing data in the given range."""
        try:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                if not await self._has_data_for_date(date_str):
                    return current_date
                current_date += timedelta(days=1)
            return None  # All dates have data
        except Exception as e:
            self.logger.warning(f"Error finding missing dates: {e}, starting from beginning")
            return start_date

    async def _load_historical_data(self) -> None:
        """Load historical data for the last 3 years with robust rate limiting."""
        end_date = datetime.now(self._local_tz).date()
        start_date = end_date - timedelta(days=self.settings.ote.load_historical_days)

        # Find the earliest date we already have data for
        actual_start_date = await self._find_earliest_missing_date(start_date, end_date)
        if actual_start_date is None:
            self.logger.info("All historical data already present, skipping download")
            return

        current_date = actual_start_date
        total_days = (end_date - actual_start_date).days + 1
        processed_days = 0
        failed_days = 0
        consecutive_failures = 0

        self.logger.info(
            f"Loading historical data from {actual_start_date} to {end_date} ({total_days} days)"
        )
        self.logger.info(
            f"Rate limiting: {self.settings.ote.request_delay}s between requests, "
            f"{self.settings.ote.error_delay}s after errors"
        )

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            success = False

            # Retry mechanism for each date
            for attempt in range(3):  # Max 3 attempts per date
                try:
                    # Exponential backoff for consecutive failures
                    if consecutive_failures > 0:
                        backoff_delay = min(
                            self.settings.ote.request_delay * (2**consecutive_failures),
                            30,  # Max 30 seconds
                        )
                        self.logger.info(
                            f"Backing off for {backoff_delay:.1f}s after "
                            f"{consecutive_failures} consecutive failures"
                        )
                        await asyncio.sleep(backoff_delay)

                    # Fetch data for this date
                    prices = await self._fetch_prices_for_date_with_retry(date_str)

                    if prices is not None:  # Allow empty dict for missing data
                        # Store in InfluxDB
                        if prices:  # Only store if we have actual price data
                            await self._store_prices(prices, date_str)
                        processed_days += 1
                        success = True
                        consecutive_failures = 0  # Reset on success

                        # Log progress every 30 days
                        if processed_days % 30 == 0:
                            progress = (processed_days / total_days) * 100
                            self.logger.info(
                                f"Historical data progress: {processed_days}/{total_days} "
                                f"days ({progress:.1f}%), failed: {failed_days}"
                            )
                        break
                    else:
                        if attempt < 2:  # Not the last attempt
                            await asyncio.sleep(self.settings.ote.error_delay * (attempt + 1))

                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1}/3 failed for {date_str}: {e}")
                    if attempt < 2:  # Not the last attempt
                        await asyncio.sleep(self.settings.ote.error_delay * (attempt + 1))

            if not success:
                failed_days += 1
                consecutive_failures += 1
                self.logger.error(f"Failed to fetch data for {date_str} after 3 attempts")

                # Take a longer break after multiple consecutive failures
                if consecutive_failures >= 5:
                    self.logger.warning(
                        f"Taking extended break after {consecutive_failures} consecutive failures"
                    )
                    await asyncio.sleep(60)  # 1 minute break

            # Move to next day
            current_date += timedelta(days=1)

            # Standard delay between requests (even on success)
            await asyncio.sleep(self.settings.ote.request_delay)

        success_rate = (processed_days / total_days) * 100 if total_days > 0 else 0
        self.logger.info(
            f"Historical data load completed. "
            f"Success: {processed_days}/{total_days} days ({success_rate:.1f}%), "
            f"Failed: {failed_days}"
        )

    async def _fetch_prices_for_date_with_retry(
        self, date_str: str
    ) -> Optional[Dict[Tuple[str, str], float]]:
        """Fetch electricity prices with enhanced error handling and user-agent."""
        if not self._session:
            raise RuntimeError("HTTP session not initialized")

        url = (
            f"{self.base_url}?report_date={date_str}"
            f"&time_resolution={self.time_resolution}"
        )

        # Use headers to appear more like a regular browser
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "cs,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.ote-cr.cz/cs/kratkodobe-trhy/elektrina/denni-trh",
            "Connection": "keep-alive",
        }

        try:
            async with self._session.get(url, headers=headers, timeout=30) as response:
                if response.status == 429:  # Rate limited
                    self.logger.warning(f"Rate limited for {date_str}, will retry with backoff")
                    return None
                elif response.status == 404:
                    self.logger.debug(f"No data available for {date_str} (404)")
                    return {}  # Empty dict means no data for this date
                elif response.status != 200:
                    self.logger.warning(f"HTTP {response.status} for {date_str}: {response.reason}")
                    return None

                data = await response.json()
                return self._parse_ote_response(data, date_str)

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching prices for {date_str}")
            return None
        except Exception as e:
            self.logger.warning(f"Error fetching prices for {date_str}: {e}")
            return None

    def _parse_ote_response(
        self, data: dict, date_str: str
    ) -> Optional[Dict[Tuple[str, str], float]]:
        """Parse OTE API response data."""
        hourly_prices = {}

        if "data" in data and "dataLine" in data["data"] and len(data["data"]["dataLine"]) > 1:
            # The structure is: data.dataLine[1].point contains the price data
            price_data = data["data"]["dataLine"][1].get("point", [])

            for point in price_data:
                hour = int(point["x"]) - 1  # OTE uses 1-based hour indexing
                price = float(point["y"])  # EUR/MWh

                start_time = f"{hour:02d}:00"
                stop_time = f"{(hour + 1) % 24:02d}:00"

                hourly_prices[(start_time, stop_time)] = price

            # Validate reasonable hour count (23-25 for DST transitions, normally 24)
            if 23 <= len(hourly_prices) <= 25:
                self.logger.debug(f"Fetched {len(hourly_prices)} price points for {date_str}")
                return hourly_prices
            else:
                self.logger.warning(
                    f"Unexpected hour count for {date_str}: {len(hourly_prices)} points"
                )
                return hourly_prices  # Still return the data, just log warning
        else:
            self.logger.warning(f"Unexpected data structure for {date_str}")
            return None

    async def _fetch_prices_for_date(self, date_str: str) -> Optional[Dict[Tuple[str, str], float]]:
        """Fetch electricity prices for a specific date (legacy method for daily updates)."""
        return await self._fetch_prices_for_date_with_retry(date_str)

    async def _store_prices(self, prices: Dict[Tuple[str, str], float], date_str: str) -> None:
        """Store prices in InfluxDB."""
        if not self.influxdb_client:
            return

        # Parse the date and localize to Prague timezone
        date = datetime.strptime(date_str, "%Y-%m-%d")
        local_date = self._local_tz.localize(date)

        for (start_hour, _), price_eur_mwh in prices.items():
            # Create timestamp for this hour
            hour = int(start_hour.split(":")[0])
            timestamp = local_date.replace(hour=hour, minute=0, second=0, microsecond=0)

            # Convert to UTC for storage
            utc_timestamp = timestamp.astimezone(pytz.UTC)

            # Store the price
            await self.influxdb_client.write_point(
                bucket=self.bucket_name,
                measurement="electricity_prices",
                fields={
                    "price": price_eur_mwh,  # EUR/MWh
                    "price_czk_kwh": price_eur_mwh
                    * self.settings.ote.eur_czk_rate
                    / 1000,  # Convert to CZK/kWh
                },
                tags={
                    "market": "day_ahead",
                    "currency": "EUR_MWh",
                    "source": "OTE",
                    "resolution": "PT60M",
                },
                timestamp=utc_timestamp,
            )

    async def _update_today_and_tomorrow(self) -> None:
        """Update prices for today and tomorrow."""
        now = datetime.now(self._local_tz)

        # Update today's prices
        today_str = now.strftime("%Y-%m-%d")
        today_prices = await self._fetch_prices_for_date(today_str)
        if today_prices:
            await self._store_prices(today_prices, today_str)
            self.logger.info(f"Updated prices for today ({today_str}): {len(today_prices)} hours")

        # Update tomorrow's prices (usually available after 2 PM)
        if now.hour >= 14:
            tomorrow = now + timedelta(days=1)
            tomorrow_str = tomorrow.strftime("%Y-%m-%d")
            tomorrow_prices = await self._fetch_prices_for_date(tomorrow_str)
            if tomorrow_prices:
                await self._store_prices(tomorrow_prices, tomorrow_str)
                self.logger.info(
                    f"Updated prices for tomorrow ({tomorrow_str}): {len(tomorrow_prices)} hours"
                )

    async def _daily_update_loop(self) -> None:
        """Run daily updates."""
        while self._running:
            try:
                # Calculate time until configured update time
                now = datetime.now(self._local_tz)
                target_time = now.replace(
                    hour=self.settings.ote.update_hour,
                    minute=self.settings.ote.update_minute,
                    second=0,
                    microsecond=0,
                )

                if target_time <= now:
                    # If it's already past update time, schedule for tomorrow
                    target_time += timedelta(days=1)

                delay = (target_time - now).total_seconds()

                self.logger.info(
                    f"Next daily update scheduled in {delay/3600:.1f} hours at {target_time}"
                )

                # Wait until scheduled time
                await asyncio.sleep(delay)

                # Perform the update
                await self._update_today_and_tomorrow()

                # Also check if we need to fill any gaps
                await self._check_and_fill_gaps()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in daily update loop: {e}", exc_info=True)
                # Wait an hour before retrying
                await asyncio.sleep(3600)

    async def _check_and_fill_gaps(self) -> None:
        """Check for gaps in the data and fill them."""
        try:
            # Query for the last 7 days to find any gaps
            end_date = datetime.now(self._local_tz).date()
            start_date = end_date - timedelta(days=7)

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")

                # Check if we have data for this date
                if not await self._has_data_for_date(date_str):
                    self.logger.info(f"Found gap in data for {date_str}, fetching...")
                    prices = await self._fetch_prices_for_date(date_str)
                    if prices:
                        await self._store_prices(prices, date_str)

                current_date += timedelta(days=1)

        except Exception as e:
            self.logger.error(f"Error checking for data gaps: {e}")

    async def _has_data_for_date(self, date_str: str) -> bool:
        """Check if we have data for a specific date."""
        try:
            if not self.influxdb_client:
                return False

            # Parse date and create time range
            date = datetime.strptime(date_str, "%Y-%m-%d")
            local_date = self._local_tz.localize(date)
            start_time = local_date.astimezone(pytz.UTC)
            end_time = start_time + timedelta(days=1)

            query = f"""
            from(bucket: "{self.bucket_name}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
                |> filter(fn: (r) => r["_field"] == "price")
                |> count()
            """

            result = await self.influxdb_client.query(query)

            # Check if we have at least 20 hours of data (allowing for some missing hours)
            for table in result:
                for record in table.records:
                    count = record.get_value()
                    if count and count >= 20:
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking data for {date_str}: {e}")
            return False

    async def get_prices_for_date(self, date: datetime) -> Optional[Dict[str, float]]:
        """Public method to get prices for a specific date from the database."""
        try:
            if not self.influxdb_client:
                return None

            # Create time range for the query
            local_date = self._local_tz.localize(
                date.replace(hour=0, minute=0, second=0, microsecond=0)
            )
            start_time = local_date.astimezone(pytz.UTC)
            end_time = start_time + timedelta(days=1)

            query = f"""
            from(bucket: "{self.bucket_name}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
                |> filter(fn: (r) => r["_field"] == "price")
                |> sort(columns: ["_time"])
            """

            result = await self.influxdb_client.query(query)

            prices = {}
            for table in result:
                for record in table.records:
                    time = record.get_time()
                    price = record.get_value()
                    if time and price:
                        # Convert back to local time for the hour key
                        local_time = time.astimezone(self._local_tz)
                        hour_key = f"{local_time.hour:02d}:00"
                        prices[hour_key] = float(price)

            return prices if prices else None

        except Exception as e:
            self.logger.error(f"Error getting prices for {date}: {e}")
            return None
