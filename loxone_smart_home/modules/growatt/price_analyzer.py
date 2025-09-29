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

    def find_n_most_expensive_hours(
        self, prices: Dict[Tuple[str, str], float], n: int = 4
    ) -> List[Tuple[str, str, float]]:
        """Find N most expensive individual hours."""
        sorted_prices = sorted(
            [(start, stop, price) for (start, stop), price in prices.items()],
            key=lambda x: x[2], reverse=True
        )
        return sorted_prices[:n]
