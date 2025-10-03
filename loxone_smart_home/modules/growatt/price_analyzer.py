"""Energy price analysis and scheduling logic for Growatt controller."""

import random
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


class PriceAnalyzer:
    """Handles energy price fetching and analysis for Growatt battery scheduling."""

    def __init__(self, logger: Any, local_tz: Any, optional_config: Dict[str, Any]) -> None:
        """Initialize PriceAnalyzer.

        Args:
            logger: Logger instance for logging
            local_tz: Local timezone object
            optional_config: Optional configuration dictionary
        """
        self.logger = logger
        self._local_tz = local_tz
        self._optional_config = optional_config

    def _get_local_date_string(self, days_ahead: int = 1) -> str:
        """Get local date string in YYYY-MM-DD format."""
        from datetime import datetime
        local_now = datetime.now(self._local_tz)
        target_date = local_now + timedelta(days=days_ahead)
        return target_date.strftime("%Y-%m-%d")

    async def fetch_dam_energy_prices(
        self, date: Optional[str] = None
    ) -> Dict[Tuple[str, str], float]:
        """Fetch energy prices from OTE DAM API.

        Returns prices in 15-minute intervals. If API provides hourly data (24 points),
        it's expanded to 15-minute intervals (96 points).

        Returns:
            Dictionary with (start_time, end_time) tuples as keys and EUR/MWh prices as values.
            Times are in "HH:MM" format, with 15-minute intervals.
        """
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

            # Process data based on format (24 hourly or 96 15-minute intervals)
            prices_15min: Dict[Tuple[str, str], float] = {}
            if data.get("data", {}).get("dataLine"):
                lines = data["data"]["dataLine"]

                # Identify EUR line based on metadata
                def is_eur_line(line: Dict[str, Any]) -> bool:
                    """Check if this line contains EUR prices based on metadata."""
                    name = (line.get("name") or "").lower()
                    title = (line.get("title") or "").lower()
                    tooltip = (line.get("tooltip") or "").lower()
                    # Look for EUR indicators - check for 15min or 60min price lines
                    return (
                        "eur/mwh" in name or "eur" in name
                        or "eur/mwh" in tooltip or "price" in title
                    )

                # Find the appropriate price line
                eur_line = None
                for line in lines:
                    title = (line.get("title") or "").lower()
                    # Prefer 15min price line if available
                    if "15min price" in title and "eur/mwh" in title:
                        eur_line = line
                        self.logger.info("Using 15-minute price data")
                        break
                    # Otherwise use any EUR line
                    elif is_eur_line(line):
                        eur_line = line

                # Fallback to line[1] if no EUR line identified
                if not eur_line and len(lines) >= 2:
                    eur_line = lines[1]
                    self.logger.debug("Using second line (usual EUR position) for prices")
                elif not eur_line:
                    # Only one line available
                    eur_line = lines[0]
                    self.logger.warning("Only one dataLine found, prices might be in CZK")

                if eur_line:
                    price_data = eur_line.get("point", [])

                    # Sort price data by x value (hour or 15-min index)
                    date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                    base_dt = datetime.combine(date_obj, dt_time(0, 0), self._local_tz)

                    def _pkey(p: Dict[str, Any]) -> float:
                        try:
                            return float(p.get("x", 0))
                        except Exception:
                            return 0.0
                    price_data = sorted(price_data, key=_pkey)

                    num_points = len(price_data)
                    self.logger.info(f"Processing {num_points} price points from OTE API")

                    if 92 <= num_points <= 100:
                        # Native 15-minute data (96 normal, 92 spring DST, 100 fall DST)
                        if num_points != 96:
                            dst_type = "spring forward" if num_points == 92 else "fall back"
                            self.logger.info(
                                f"DST transition day detected: {num_points} blocks ({dst_type})"
                            )
                        self.logger.info(f"Processing {num_points} points as 15-minute intervals")
                        for i, point in enumerate(price_data):
                            try:
                                price = float(point["y"])
                            except (ValueError, TypeError):
                                self.logger.warning(
                                    f"Invalid price at index {i}: {point.get('y')}"
                                )
                                continue

                            # Skip non-finite prices (but keep negative prices)
                            if not (price == price) or price in (float("inf"), float("-inf")):
                                self.logger.warning(f"Skipping non-finite price at {i}: {price}")
                                continue

                            # Calculate 15-minute time slots
                            minutes = i * 15
                            start_dt = base_dt + timedelta(minutes=minutes)
                            stop_dt = start_dt + timedelta(minutes=15)

                            start_time = start_dt.strftime("%H:%M")
                            if stop_dt.date() != start_dt.date():
                                # Handle day boundary
                                if stop_dt.hour == 0:
                                    stop_time = "24:00"
                                else:
                                    stop_time = stop_dt.strftime("%H:%M")
                            else:
                                stop_time = stop_dt.strftime("%H:%M")

                            key = (start_time, stop_time)
                            prices_15min[key] = price

                    elif num_points <= 24:
                        # Hourly data or partial data - expand to 15-minute intervals
                        self.logger.info(
                            f"Processing {num_points} hourly points, "
                            f"expanding to 15-minute intervals"
                        )
                        for i, point in enumerate(price_data):
                            try:
                                price = float(point["y"])
                            except (ValueError, TypeError):
                                self.logger.warning(
                                    f"Invalid price at hour {i}: {point.get('y')}"
                                )
                                continue

                            # Skip non-finite prices
                            if not (price == price) or price in (float("inf"), float("-inf")):
                                self.logger.warning(f"Skipping non-finite price at {i}: {price}")
                                continue

                            # Create 4 15-minute intervals for each hour
                            hour_dt = base_dt + timedelta(hours=i)
                            for quarter in range(4):
                                minutes = quarter * 15
                                start_dt = hour_dt + timedelta(minutes=minutes)
                                stop_dt = start_dt + timedelta(minutes=15)

                                start_time = start_dt.strftime("%H:%M")
                                if stop_dt.date() != start_dt.date():
                                    if stop_dt.hour == 0:
                                        stop_time = "24:00"
                                    else:
                                        stop_time = stop_dt.strftime("%H:%M")
                                else:
                                    stop_time = stop_dt.strftime("%H:%M")

                                key = (start_time, stop_time)
                                # Use same price for all 15-min blocks in the hour
                                prices_15min[key] = price

                    elif num_points > 24 and num_points < 92:
                        # Unexpected format - between hourly and 15-min
                        self.logger.error(
                            f"Unexpected number of price points: {num_points}. "
                            f"Expected 24 (hourly) or 92-100 (15-minute)"
                        )
                        return {}

            # Validate data completeness
            if prices_15min and len(prices_15min) < 90:
                self.logger.warning(
                    f"Incomplete price data: {len(prices_15min)} blocks, expected 92-100"
                )

            self.logger.info(f"Successfully processed {len(prices_15min)} 15-minute price points")
            return prices_15min

        except Exception as e:
            self.logger.error(f"Error fetching DAM prices: {e}", exc_info=True)
            return {}

    async def fetch_dam_energy_prices_with_retry(
        self,
        target_date: Optional[str] = None,
        initial_delay_minutes: int = 5,
        max_delay_minutes: int = 60,
        max_attempts: int = 20
    ) -> Dict[Tuple[str, str], float]:
        """Fetch DAM energy prices with exponential backoff retry logic.

        Handles midnight rollover: if we start fetching before midnight and cross over,
        the target date is recalculated to ensure we're always fetching the correct day.

        Args:
            target_date: Target date in YYYY-MM-DD format. If None, calculates next day.
            initial_delay_minutes: Initial retry delay in minutes (default: 5)
            max_delay_minutes: Maximum retry delay in minutes (default: 60)
            max_attempts: Maximum retry attempts (default: 20)

        Returns:
            Dictionary with (start_time, end_time) tuples as keys and EUR/MWh prices as values.
            Returns empty dict if all retries exhausted.
        """
        import asyncio

        attempt = 0
        delay_minutes = initial_delay_minutes

        while attempt < max_attempts:
            attempt += 1

            # Recalculate target date on each attempt to handle midnight rollover
            # If target_date was explicitly provided, use it. Otherwise calculate "next day"
            if target_date is None:
                # Calculate next day based on current time
                # This ensures if we cross midnight during retries, we adjust the target
                now = datetime.now(self._local_tz)
                # If it's before the price fetch hour, we want tomorrow's prices
                # If it's after, we still want tomorrow's prices (they should be available)
                fetch_target = now + timedelta(days=1)
                current_target = fetch_target.strftime("%Y-%m-%d")
            else:
                current_target = target_date

            self.logger.info(
                f"📊 Attempt {attempt}/{max_attempts}: Fetching DAM prices for {current_target}"
            )

            # Attempt to fetch prices
            prices = await self.fetch_dam_energy_prices(date=current_target)

            if prices and len(prices) >= 90:
                # Success! We have valid price data (at least 90 blocks for a full day)
                self.logger.info(
                    f"✅ Successfully fetched {len(prices)} price points for {current_target} "
                    f"on attempt {attempt}"
                )
                return prices

            # Failed - log and prepare for retry
            if attempt < max_attempts:
                self.logger.warning(
                    f"⏳ No valid prices yet for {current_target} "
                    f"({len(prices)} blocks received, need ≥90). "
                    f"Retrying in {delay_minutes} minutes... "
                    f"(attempt {attempt}/{max_attempts})"
                )

                # Wait for the retry delay
                await asyncio.sleep(delay_minutes * 60)

                # Exponential backoff with cap
                delay_minutes = min(delay_minutes * 2, max_delay_minutes)
            else:
                # Max attempts reached
                self.logger.error(
                    f"❌ Failed to fetch prices for {current_target} after {max_attempts} attempts. "
                    f"Prices may not be published yet or API is unavailable."
                )

        return {}

    def generate_mock_prices(self, date: str) -> Dict[Tuple[str, str], float]:
        """Generate mock energy prices for testing when OTE data unavailable.

        Creates a realistic price pattern with:
        - Lower prices at night (2-6 AM)
        - Higher prices during peak hours (8-10 AM, 5-8 PM)
        - Medium prices during day

        Returns 96 15-minute intervals.
        """
        random.seed(date)  # Consistent prices for same date

        prices_15min: Dict[Tuple[str, str], float] = {}
        base_price = 80.0  # EUR/MWh

        for hour in range(24):
            # Determine base hourly price based on time of day
            if 2 <= hour < 6:
                # Night valley (2-6 AM) - cheapest
                hour_price = base_price * random.uniform(0.5, 0.7)
            elif 8 <= hour < 10:
                # Morning peak (8-10 AM) - expensive
                hour_price = base_price * random.uniform(1.3, 1.5)
            elif 17 <= hour < 20:
                # Evening peak (17-20 PM) - most expensive
                hour_price = base_price * random.uniform(1.4, 1.6)
            elif hour >= 22 or hour < 2:
                # Night (22-2 AM) - cheap
                hour_price = base_price * random.uniform(0.6, 0.8)
            else:
                # Day hours - medium
                hour_price = base_price * random.uniform(0.9, 1.1)

            # Create 4 15-minute intervals with slight variations
            for quarter in range(4):
                start_minutes = hour * 60 + quarter * 15
                start_hour = start_minutes // 60
                start_min = start_minutes % 60
                end_minutes = start_minutes + 15
                end_hour = end_minutes // 60
                end_min = end_minutes % 60

                start = f"{start_hour:02d}:{start_min:02d}"
                if end_hour >= 24:
                    end = "24:00"
                else:
                    end = f"{end_hour:02d}:{end_min:02d}"

                # Add slight variation within the hour (±5%)
                variation = random.uniform(0.95, 1.05)
                quarter_price = hour_price * variation

                prices_15min[(start, end)] = round(quarter_price, 2)

        self.logger.warning(f"Using mock prices for {date} (OTE data unavailable) - 96 intervals")
        return prices_15min

    def find_n_most_expensive_blocks(
        self, prices: Dict[Tuple[str, str], float], n: int = 16
    ) -> List[Tuple[str, str, float]]:
        """Find N most expensive 15-minute blocks.

        Args:
            prices: Dictionary of 15-minute interval prices
            n: Number of blocks to return (default 16 = 4 hours)

        Returns:
            List of (start, stop, price) tuples for most expensive blocks
        """
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()],
            key=lambda x: x[2], reverse=True
        )
        return sorted_prices[:n]

    def find_cheapest_blocks(
        self, prices: Dict[Tuple[str, str], float], num_blocks: int = 8
    ) -> List[Tuple[str, str, float]]:
        """Find the cheapest N 15-minute blocks (not necessarily consecutive).

        Args:
            prices: Dictionary of 15-minute interval prices
            num_blocks: Number of blocks to find (default 8 = 2 hours)

        Returns:
            List of (start, stop, price) tuples for cheapest blocks
        """
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()],
            key=lambda x: x[2]
        )
        return sorted_prices[:num_blocks]

    def get_charging_schedule(
        self, prices: Dict[Tuple[str, str], float], num_blocks: int = 8
    ) -> Tuple[List[Tuple[str, str, float]], float]:
        """Get the charging schedule with cheapest blocks sorted by time.

        Args:
            prices: Dictionary of 15-minute interval prices
            num_blocks: Number of blocks to charge (default 8 = 2 hours)

        Returns:
            Tuple of (sorted list of charging blocks, average price)
        """
        cheapest = self.find_cheapest_blocks(prices, num_blocks)
        if not cheapest:
            return [], 0.0

        # Sort by start time for display
        sorted_blocks = sorted(cheapest, key=lambda x: x[0])

        # Calculate average price
        avg_price = sum(block[2] for block in cheapest) / len(cheapest) if cheapest else 0

        return sorted_blocks, avg_price
