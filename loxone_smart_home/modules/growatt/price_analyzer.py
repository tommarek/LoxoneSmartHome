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
    ) -> Tuple[Dict[Tuple[str, str], float], str]:
        """Fetch energy prices from OTE DAM API.

        Returns prices in 15-minute intervals. If API provides hourly data (24 points),
        it's expanded to 15-minute intervals (96 points).

        Returns:
            Tuple of (prices_dict, status) where status is:
            - "success": Got valid price data (>=90 blocks)
            - "not_published": Valid API response but no price data (prices not published yet)
            - "partial": Valid response but incomplete data (<90 blocks)
            - "error": API error, malformed response, or other failure
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
                        return {}, "error"

                    data = await response.json()

            # Process data based on format (24 hourly or 96 15-minute intervals)
            prices_15min: Dict[Tuple[str, str], float] = {}
            dataLine = data.get("data", {}).get("dataLine", [])

            # Check if dataLine is empty (prices not published yet)
            if not dataLine:
                self.logger.info(
                    f"OTE API returned valid response but no price data for {date} - "
                    f"prices not published yet"
                )
                return {}, "not_published"

            if dataLine:
                lines = dataLine

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
                        return {}, "error"

            # Validate data completeness and determine status
            if not prices_15min:
                # No prices extracted despite having dataLine
                self.logger.warning("Failed to extract any valid prices from response")
                return {}, "error"
            elif len(prices_15min) < 90:
                # Partial data - might be in process of being published
                self.logger.warning(
                    f"Incomplete price data: {len(prices_15min)} blocks, expected 90-100 "
                    f"(might be publishing in progress)"
                )
                return prices_15min, "partial"
            else:
                # Full data set
                self.logger.info(
                    f"Successfully processed {len(prices_15min)} 15-minute price points"
                )
                return prices_15min, "success"

        except Exception as e:
            self.logger.error(f"Error fetching DAM prices: {e}", exc_info=True)
            return {}, "error"

    def _calculate_smart_delay(self, current_hour: int, status: str, error_delay: int) -> int:
        """Calculate retry delay based on time of day and status.

        Args:
            current_hour: Current hour (0-23)
            status: Status from fetch attempt ("not_published", "partial", "error")
            error_delay: Current exponential backoff delay for errors

        Returns:
            Delay in minutes until next retry
        """
        if status == "not_published":
            # Smart backoff based on when prices are typically published (~14:00)
            if current_hour < 13:
                # Before 13:00: Prices won't be ready yet, check hourly
                return 60
            elif 13 <= current_hour < 15:
                # 13:00-15:00: Publication window, check every 15 minutes
                return 15
            else:
                # After 15:00: Should be published, check every 5 minutes
                return 5

        elif status == "partial":
            # Data is being published, check frequently
            return 5

        else:  # "error" status (API issues, network problems)
            # Use exponential backoff for actual errors
            return error_delay

    async def fetch_dam_energy_prices_with_retry(
        self,
        target_date: Optional[str] = None,
        initial_delay_minutes: int = 5,
        max_delay_minutes: int = 60,
        max_attempts: int = 20  # Parameter kept for backward compatibility but not enforced
    ) -> Dict[Tuple[str, str], float]:
        """Fetch DAM energy prices with infinite retry until success.

        DAM prices are published daily at ~14:00, so this function retries indefinitely
        with smart time-based backoff until prices are successfully fetched. The only
        way this returns empty is if the task is cancelled (e.g., at midnight rollover).

        Retry strategy:
        - "not_published" before 13:00: Retry every 60 minutes (normal)
        - "not_published" 13:00-15:00: Retry every 15 minutes (publication window)
        - "not_published" after 15:00: Retry every 5 minutes (should be available)
        - "partial" data: Retry every 5 minutes (publishing in progress)
        - "error" (API issues): Exponential backoff 5min → 10min → 30min → 60min

        Args:
            target_date: Target date in YYYY-MM-DD format. If None, calculates next day.
            initial_delay_minutes: Initial retry delay for errors (default: 5)
            max_delay_minutes: Maximum retry delay for errors (default: 60)
            max_attempts: Kept for compatibility, but ignored (retries indefinitely)

        Returns:
            Dictionary with (start_time, end_time) tuples as keys and EUR/MWh prices as values.
            Returns empty dict only if task is cancelled (e.g., midnight rollover).
        """
        import asyncio

        attempt = 0
        error_delay_minutes = initial_delay_minutes
        last_log_hour = -1  # Track hour to avoid log spam

        # Retry indefinitely until we get prices or task is cancelled
        while True:
            attempt += 1

            # Recalculate target date on each attempt to handle midnight rollover
            # If target_date was explicitly provided, use it. Otherwise calculate "next day"
            if target_date is None:
                # Calculate next day based on current time
                # This ensures if we cross midnight during retries, we adjust the target
                now = datetime.now(self._local_tz)
                fetch_target = now + timedelta(days=1)
                current_target = fetch_target.strftime("%Y-%m-%d")
            else:
                current_target = target_date

            now = datetime.now(self._local_tz)
            current_hour = now.hour

            # Log attempt (but reduce log spam for repeated "not_published" before 14:00)
            if attempt == 1 or current_hour != last_log_hour or attempt % 10 == 0:
                self.logger.info(
                    f"📊 Attempt {attempt}: Fetching DAM prices for {current_target} "
                    f"at {now.strftime('%H:%M')}"
                )
                last_log_hour = current_hour

            # Attempt to fetch prices
            prices, status = await self.fetch_dam_energy_prices(date=current_target)

            if status == "success":
                # Success! We have valid price data
                self.logger.info(
                    f"✅ Successfully fetched {len(prices)} price blocks for {current_target} "
                    f"after {attempt} attempt(s)"
                )
                return prices

            # Calculate smart delay based on status and time of day
            delay_minutes = self._calculate_smart_delay(current_hour, status, error_delay_minutes)

            # Log retry with appropriate level based on status and time
            if status == "not_published":
                if current_hour < 14:
                    # Before 14:00: This is completely normal
                    log_level = "debug"
                    message = (
                        f"⏳ Prices for {current_target} not published yet "
                        f"(normal before 14:00). Checking again in {delay_minutes} min"
                    )
                elif current_hour < 16:
                    # 14:00-16:00: Slightly delayed but acceptable
                    log_level = "info"
                    message = (
                        f"⏳ Prices for {current_target} not published yet "
                        f"(publication may be delayed). Checking again in {delay_minutes} min"
                    )
                else:
                    # After 16:00: Unusual delay
                    log_level = "warning"
                    message = (
                        f"⚠️ Prices for {current_target} still not published at "
                        f"{now.strftime('%H:%M')} (unusual delay). "
                        f"Checking again in {delay_minutes} min"
                    )
            elif status == "partial":
                log_level = "info"
                message = (
                    f"⏳ Partial price data for {current_target} ({len(prices)} blocks). "
                    f"Retrying in {delay_minutes} min (publishing in progress)..."
                )
            else:  # "error"
                log_level = "warning"
                message = (
                    f"⚠️ Error fetching prices for {current_target}. "
                    f"Retrying in {delay_minutes} min (attempt {attempt})..."
                )

            # Log with appropriate level
            if log_level == "debug":
                self.logger.debug(message)
            elif log_level == "info":
                self.logger.info(message)
            else:  # warning
                self.logger.warning(message)

            # Wait before next retry
            try:
                await asyncio.sleep(delay_minutes * 60)
            except asyncio.CancelledError:
                self.logger.info(
                    f"Price fetch for {current_target} cancelled (likely midnight rollover)"
                )
                raise

            # Update exponential backoff for errors
            if status == "error":
                error_delay_minutes = min(error_delay_minutes * 2, max_delay_minutes)
            else:
                # Reset error delay for non-error statuses
                error_delay_minutes = initial_delay_minutes

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

    def identify_discharge_peaks(
        self,
        discharge_periods: List[Tuple[str, str, float]],
        threshold_multiplier: float = 1.5
    ) -> List[List[Tuple[str, str, float]]]:
        """Identify significant discharge peaks that warrant pre-charging.

        Groups consecutive discharge blocks and identifies peaks that are
        significantly above average discharge price.

        Args:
            discharge_periods: List of discharge periods (start, end, price_eur)
            threshold_multiplier: Multiplier for average to identify peaks

        Returns:
            List of peak groups, where each group is a list of consecutive blocks
        """
        if not discharge_periods:
            return []

        # Calculate average discharge price
        avg_discharge_price = sum(p[2] for p in discharge_periods) / len(discharge_periods)
        peak_threshold = avg_discharge_price * threshold_multiplier

        # Filter for significant peaks
        significant_periods = [p for p in discharge_periods if p[2] >= peak_threshold]

        if not significant_periods:
            return []

        # Group consecutive blocks into peaks
        peaks = []
        current_peak = [significant_periods[0]]

        for i in range(1, len(significant_periods)):
            prev_end = current_peak[-1][1]
            curr_start = significant_periods[i][0]

            # Check if consecutive (end time of prev equals start time of current)
            if prev_end == curr_start:
                current_peak.append(significant_periods[i])
            else:
                # Start new peak
                peaks.append(current_peak)
                current_peak = [significant_periods[i]]

        # Add the last peak
        peaks.append(current_peak)

        # Sort peaks by average price (highest first) and take top ones
        peaks_with_avg = [(peak, sum(p[2] for p in peak) / len(peak)) for peak in peaks]
        peaks_with_avg.sort(key=lambda x: x[1], reverse=True)

        # Return top peaks (limit to avoid over-scheduling)
        max_peaks = 3  # Configurable: maximum number of peaks to prepare for
        return [peak for peak, _ in peaks_with_avg[:max_peaks]]

    def find_pre_discharge_blocks(
        self,
        prices: Dict[Tuple[str, str], float],
        peak_start: str,
        num_blocks: int = 8,
        window_hours: int = 6
    ) -> List[Tuple[str, str, float]]:
        """Find cheapest blocks before a discharge peak for pre-charging.

        Args:
            prices: Dictionary of 15-minute interval prices
            peak_start: Start time of the discharge peak (HH:MM)
            num_blocks: Number of blocks to find (default 8 = 2 hours)
            window_hours: Hours to look back from peak start

        Returns:
            List of (start, stop, price) tuples for pre-charge blocks
        """
        # Parse peak start time
        peak_hour = int(peak_start.split(':')[0])
        peak_minute = int(peak_start.split(':')[1])

        # Calculate window start (handle day boundary)
        window_start_hour = peak_hour - window_hours
        if window_start_hour < 0:
            window_start_hour += 24

        # Filter blocks within the window
        blocks_in_window = []
        for (start, end), price in prices.items():
            start_hour = int(start.split(':')[0])
            start_minute = int(start.split(':')[1])

            # Check if block is within window (handle day wrap)
            if window_start_hour < peak_hour:
                # Normal case: window doesn't wrap midnight
                if (start_hour > window_start_hour or
                   (start_hour == window_start_hour and start_minute >= 0)):
                    if (start_hour < peak_hour or
                       (start_hour == peak_hour and start_minute < peak_minute)):
                        blocks_in_window.append((start, end, price))
            else:
                # Window wraps midnight (e.g., peak at 02:00, window starts at 20:00)
                if start_hour >= window_start_hour or start_hour < peak_hour:
                    blocks_in_window.append((start, end, price))

        # Sort by price and return cheapest blocks
        blocks_in_window.sort(key=lambda x: x[2])
        return blocks_in_window[:num_blocks]

    def calculate_pre_discharge_schedule(
        self,
        prices: Dict[Tuple[str, str], float],
        discharge_periods: List[Tuple[str, str, float]],
        peak_threshold: float = 1.5,
        charge_blocks: int = 8,
        window_hours: int = 6
    ) -> Tuple[List[Tuple[str, str, float]], Dict[str, List[Tuple[str, str, float]]]]:
        """Calculate complete pre-discharge charging schedule for all peaks.

        Args:
            prices: Dictionary of 15-minute interval prices
            discharge_periods: List of all discharge periods
            peak_threshold: Multiplier to identify significant peaks
            charge_blocks: Number of blocks to charge before each peak
            window_hours: Hours to look back for cheap blocks

        Returns:
            Tuple of:
            - Combined list of all pre-discharge charging blocks
            - Dictionary mapping peak period to its pre-charge blocks
        """
        # Identify significant peaks
        peaks = self.identify_discharge_peaks(discharge_periods, peak_threshold)

        all_pre_charge_blocks = []
        peak_to_precharge = {}

        for peak in peaks:
            # Get peak start time and average price
            peak_start = peak[0][0]
            peak_key = f"{peak[0][0]}-{peak[-1][1]}"  # Full peak period

            # Find pre-charge blocks for this peak
            pre_charge_blocks = self.find_pre_discharge_blocks(
                prices, peak_start, charge_blocks, window_hours
            )

            peak_to_precharge[peak_key] = pre_charge_blocks
            all_pre_charge_blocks.extend(pre_charge_blocks)

        # Remove duplicates from combined list (keep unique blocks)
        unique_blocks = list({(b[0], b[1]): b for b in all_pre_charge_blocks}.values())

        # Sort by time for display
        unique_blocks.sort(key=lambda x: x[0])

        return unique_blocks, peak_to_precharge
