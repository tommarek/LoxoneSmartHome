"""Energy price analysis and scheduling logic for Growatt controller."""

import random
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


class PriceAnalyzer:
    """Handles energy price fetching and analysis for Growatt battery scheduling."""

    def __init__(self, logger, local_tz, optional_config: Dict[str, Any]):
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

            # dataLine[0]: CZK/MWh, dataLine[1]: EUR/MWh
            hourly_prices: Dict[Tuple[str, str], float] = {}
            if data.get("data", {}).get("dataLine"):
                lines = data["data"]["dataLine"]

                # Identify EUR line
                def is_eur_line(line: Dict[str, Any]) -> bool:
                    """Check if this line contains EUR prices based on metadata."""
                    name = (line.get("name") or "").lower()
                    tooltip = (line.get("tooltip") or "").lower()
                    # Look for EUR indicators in metadata
                    return "eur/mwh" in name or "eur" in name or "eur/mwh" in tooltip

                eur_line = next((ln for ln in lines if is_eur_line(ln)), None)
                if eur_line:
                    price_data = eur_line.get("point", [])
                    self.logger.debug("Using identified EUR line for prices")
                elif len(lines) >= 2:
                    price_data = lines[1].get("point", [])
                    self.logger.debug("Using second line (usual EUR position) for prices")
                else:
                    # Only one line available
                    price_data = lines[0].get("point", [])
                    self.logger.warning("Only one dataLine found, prices might be in CZK")

                date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                base_dt = datetime.combine(date_obj, dt_time(0, 0), self._local_tz)

                def _pkey(p: Dict[str, Any]) -> float:
                    try:
                        return float(p.get("x", 0))
                    except Exception:
                        return 0.0
                price_data = sorted(price_data, key=_pkey)

                # DST merge policy for 25-hour days (fall back)
                merge_policy = self._optional_config.get("dst_merge_policy", "avg")

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

                    start_dt = base_dt + timedelta(hours=i)
                    stop_dt = start_dt + timedelta(hours=1)

                    start_time = start_dt.strftime("%H:%M")
                    if stop_dt.date() != start_dt.date():
                        stop_time = "24:00"  # Next day
                    else:
                        stop_time = stop_dt.strftime("%H:%M")

                    key = (start_time, stop_time)
                    if key in hourly_prices:
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

    def generate_mock_prices(self, date: str) -> Dict[Tuple[str, str], float]:
        """Generate mock energy prices for testing when OTE data unavailable.

        Creates a realistic price pattern with:
        - Lower prices at night (2-6 AM)
        - Higher prices during peak hours (8-10 AM, 5-8 PM)
        - Medium prices during day
        """
        random.seed(date)  # Consistent prices for same date

        hourly_prices: Dict[Tuple[str, str], float] = {}
        base_price = 80.0  # EUR/MWh

        for hour in range(24):
            start = f"{hour:02d}:00"
            end = f"{(hour + 1) % 24:02d}:00" if hour < 23 else "24:00"

            # Night valley (2-6 AM) - cheapest
            if 2 <= hour < 6:
                price = base_price * random.uniform(0.5, 0.7)
            # Morning peak (8-10 AM) - expensive
            elif 8 <= hour < 10:
                price = base_price * random.uniform(1.3, 1.5)
            # Evening peak (17-20 PM) - most expensive
            elif 17 <= hour < 20:
                price = base_price * random.uniform(1.4, 1.6)
            # Night (22-2 AM) - cheap
            elif hour >= 22 or hour < 2:
                price = base_price * random.uniform(0.6, 0.8)
            # Day hours - medium
            else:
                price = base_price * random.uniform(0.9, 1.1)

            hourly_prices[(start, end)] = round(price, 2)

        self.logger.warning(f"Using mock prices for {date} (OTE data unavailable)")
        return hourly_prices

    def find_cheapest_consecutive_hours(
        self, prices: Dict[Tuple[str, str], float], x: int = 2
    ) -> List[Tuple[str, str, float]]:
        """Find X consecutive hours with lowest total price."""
        if not prices:
            return []

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

    def find_n_cheapest_hours(
        self, prices: Dict[Tuple[str, str], float], n: int = 8
    ) -> List[Tuple[str, str, float]]:
        """Find N cheapest individual hours."""
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()], key=lambda x: x[2]
        )
        return sorted_prices[:n]

    def find_n_most_expensive_hours(
        self, prices: Dict[Tuple[str, str], float], n: int = 4
    ) -> List[Tuple[str, str, float]]:
        """Find N most expensive individual hours."""
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()],
            key=lambda x: x[2], reverse=True
        )
        return sorted_prices[:n]

    def categorize_prices_into_quadrants(
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

    def group_contiguous_hours(self, hours: List[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
        """Group contiguous hours into continuous ranges (with price similarity)."""
        if not hours:
            return []

        def t(s: str) -> datetime:
            # Keep logic in HH:MM; treat "24:00" as exclusive end-of-day for comparisons
            return datetime.strptime("00:00" if s == "24:00" else s, "%H:%M")

        sorted_hours = sorted(hours, key=lambda x: t(x[0]))

        groups: List[Tuple[str, str]] = []
        gs, ge, gp = sorted_hours[0]

        for s, e, p in sorted_hours[1:]:
            # contiguous if next start equals current end AND prices within 20%
            if t(s) == t(ge) and (abs(p - gp) < abs(gp * 0.2) if gp != 0 else p == 0):
                ge = e
            else:
                groups.append((gs, ge))
                gs, ge, gp = s, e, p

        groups.append((gs, ge))
        return groups

    def group_contiguous_hours_simple(
        self, hours: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str]]:
        """Group contiguous hours into continuous ranges without price similarity check."""
        if not hours:
            return []

        def parse_time(s: str) -> datetime:
            return datetime.strptime("00:00" if s == "24:00" else s, "%H:%M")

        sorted_hours = sorted(hours, key=lambda x: parse_time(x[0]))

        groups: List[Tuple[str, str]] = []
        group_start, group_end = sorted_hours[0][0], sorted_hours[0][1]

        for start, end, _price in sorted_hours[1:]:
            # Compare times properly - contiguous if next start equals current end
            # Don't normalize 24:00 here - keep it for proper boundary handling
            if parse_time(start) == parse_time(group_end if group_end != "24:00" else "00:00"):
                # Extend the current group
                group_end = end
            else:
                # Save current group and start new one
                groups.append((group_start, group_end))
                group_start, group_end = start, end

        # Add the last group
        groups.append((group_start, group_end))
        return groups