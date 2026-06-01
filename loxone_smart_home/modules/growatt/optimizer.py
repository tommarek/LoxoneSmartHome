"""Greedy forward-simulation optimizer for battery charge/discharge scheduling.

Replaces rule-based scheduling with a block-by-block simulation that considers
price curve, solar forecast, consumption forecast, and battery constraints to
minimize total electricity cost.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BlockDecision:
    """Decision for a single 15-minute block."""
    timestamp: datetime
    action: str  # "charge", "discharge", "hold", "sell_production"
    price_czk: float  # Spot price CZK/kWh
    distribution_czk: float  # Distribution tariff CZK/kWh
    solar_kwh: float  # Expected solar production this block
    consumption_kwh: float  # Expected consumption this block
    soc_before: float  # SOC % before action
    soc_after: float  # SOC % after action
    net_value: float  # Value of this action (positive = saves money)


def _forecast_value(forecast: Dict[Any, float], timestamp: datetime) -> float:
    """Read a forecast value for a block.

    Preferred keys are absolute enough for cross-day optimization:
    - datetime: exact block timestamp
    - (date, hour): date-aware hourly value
    - "YYYY-MM-DD-HH": date-aware hourly string

    Integer hour keys remain supported for existing single-day tests/callers.
    """
    exact = forecast.get(timestamp)
    if exact is not None:
        return exact

    date_hour = forecast.get((timestamp.date(), timestamp.hour))
    if date_hour is not None:
        return date_hour

    iso_hour = forecast.get(f"{timestamp.date().isoformat()}-{timestamp.hour:02d}")
    if iso_hour is not None:
        return iso_hour

    return forecast.get(timestamp.hour, 0.0)


# Margin (CZK/kWh) above which a sell-production swap is worthwhile.
# Smaller spreads aren't worth the mode change (risk noise from forecast error).
SELL_PRODUCTION_MARGIN_CZK = 0.3

# Minimum forecast solar excess (kWh per 15-min block) required to trigger
# sell_production. The mode locks the battery (stop_soc=max_soc), so on a
# tiny phantom-excess block we'd lose self-consumption AND import the real
# load deficit from grid at the same hour's spot — net loss when the
# forecast overshoots. 0.25 kWh per 15 min ≈ 1 kW continuous solar surplus,
# enough headroom for normal consumption-forecast error.
SELL_PRODUCTION_MIN_EXCESS_KWH = 0.25


@dataclass
class BaseLoadProfile:
    """Learned hourly non-heating base load profile.

    48 slots: (hour 0-23) × (weekday/weekend) → median kWh consumption
    when no heating relays are active. Used to calculate how much battery
    to reserve for self-consumption.
    """
    # (hour, is_weekend) -> kWh per hour
    profile: Dict[Tuple[int, bool], float] = field(default_factory=dict)
    data_points: int = 0
    built_at: Optional[datetime] = None

    def get(self, hour: int, is_weekend: bool) -> float:
        """Get expected non-heating consumption for an hour."""
        key = (hour % 24, is_weekend)
        if key in self.profile:
            return self.profile[key]
        # Fallback: try opposite day type
        alt = (hour % 24, not is_weekend)
        if alt in self.profile:
            return self.profile[alt]
        # Global fallback: ~500W average
        return 0.5

    def get_reserve_until(
        self, from_hour: int, until_hour: int, is_weekend: bool,
        battery_capacity_kwh: float, min_soc: float, efficiency: float,
    ) -> float:
        """Calculate SOC needed to cover base load from from_hour to until_hour.

        Returns effective minimum SOC percentage.
        """
        reserve_kwh = 0.0
        h = from_hour
        while h != until_hour:
            reserve_kwh += self.get(h, is_weekend)
            h = (h + 1) % 24
            if reserve_kwh > battery_capacity_kwh:
                break  # Can't reserve more than battery holds
        # Account for efficiency loss
        reserve_kwh = reserve_kwh / efficiency
        # Clamp to usable capacity
        max_reserve = battery_capacity_kwh * (100 - min_soc) / 100 * 0.8
        reserve_kwh = min(reserve_kwh, max_reserve)
        return min_soc + (reserve_kwh / battery_capacity_kwh) * 100

    def summary(self) -> str:
        """Human-readable summary of the profile."""
        if not self.profile:
            return "No profile data"
        lines = []
        for is_wknd in [False, True]:
            label = "Weekend" if is_wknd else "Weekday"
            vals = [self.get(h, is_wknd) for h in range(24)]
            total = sum(vals)
            peak_h = max(range(24), key=lambda h: self.get(h, is_wknd))
            lines.append(f"{label}: {total:.1f} kWh/day, peak {self.get(peak_h, is_wknd):.2f} kWh at {peak_h:02d}h")
        return "; ".join(lines)


class BatteryOptimizer:
    """Greedy optimizer for 15-minute block scheduling.

    Two-pass approach:
    1. Score each block for charge/discharge/hold value
    2. Forward simulate, picking best feasible action per block
    """

    # The greedy engine cannot co-optimize deferrable loads; the controller
    # pre-schedules them and overlays the draw onto the consumption forecast.
    CO_OPTIMIZES_DEFERRABLE = False

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._base_load_profile: BaseLoadProfile = BaseLoadProfile()
        self._profile_updated: Optional[datetime] = None
        self._last_reserve_info: Dict[str, Any] = {}
        self._last_decisions: List[BlockDecision] = []
        self._night_reserve_kwh: float = 5.0
        self._night_reserve_updated: Optional[datetime] = None

    async def build_base_load_profile(
        self, influxdb_client: Any, solar_bucket: str, loxone_bucket: str,
        days: int = 90
    ) -> BaseLoadProfile:
        """Build hourly non-heating base load profile from historical data.

        1. Query heating relay hours (tag1="heating", value=1)
        2. Query house load (INVPowerToLocalLoad)
        3. Exclude hours where ANY heating relay was ON
        4. Compute median per (hour, weekday/weekend)

        Args:
            influxdb_client: Async InfluxDB client
            solar_bucket: Solar bucket (for INVPowerToLocalLoad)
            loxone_bucket: Loxone bucket (for relay data)
            days: Days of history

        Returns:
            BaseLoadProfile with 48 slots
        """
        try:
            self.logger.info("Building base load profile from historical data...")

            # Query heating relay activity: which hours had heating ON
            heating_query = f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating")
  |> filter(fn: (r) => r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)
'''
            # Query EV charging: which hours had EV charging active
            ev_query = f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "ev" and r._field == "ev_charging")
  |> filter(fn: (r) => r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)
'''
            heating_result = await influxdb_client.query(heating_query)
            ev_result = await influxdb_client.query(ev_query)

            # Build set of hours where heating OR EV charging was active
            high_load_hours: set = set()  # "YYYY-MM-DD-HH" keys
            if heating_result:
                for table in heating_result:
                    for record in table.records:
                        high_load_hours.add(record.get_time().strftime("%Y-%m-%d-%H"))
            heating_count = len(high_load_hours)

            if ev_result:
                for table in ev_result:
                    for record in table.records:
                        high_load_hours.add(record.get_time().strftime("%Y-%m-%d-%H"))
            ev_count = len(high_load_hours) - heating_count

            self.logger.debug(f"Found {heating_count} heating + {ev_count} EV hours to exclude")

            # Query house load
            load_query = f'''
from(bucket: "{solar_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> filter(fn: (r) => r._value > 0)
'''
            load_result = await influxdb_client.query(load_query)
            if not load_result:
                self.logger.warning("No load data for base load profile")
                return self._base_load_profile

            # Bin by (hour, is_weekend), excluding heating hours
            import statistics
            bins: Dict[Tuple[int, bool], List[float]] = {}
            total = 0
            excluded = 0

            for table in load_result:
                for record in table.records:
                    t = record.get_time()
                    key = t.strftime("%Y-%m-%d-%H")

                    # Skip hours where heating was active
                    if key in high_load_hours:
                        excluded += 1
                        continue

                    hour = t.hour
                    is_weekend = t.weekday() >= 5
                    kwh = record.get_value() / 1000.0  # W -> kWh for 1 hour

                    bin_key = (hour, is_weekend)
                    if bin_key not in bins:
                        bins[bin_key] = []
                    bins[bin_key].append(kwh)
                    total += 1

            # Compute medians
            profile = BaseLoadProfile()
            for bin_key, values in bins.items():
                if values:
                    profile.profile[bin_key] = statistics.median(values)

            profile.data_points = total
            profile.built_at = datetime.now()
            self._base_load_profile = profile
            self._profile_updated = datetime.now()

            self.logger.info(
                f"Base load profile built: {total} non-heating hours "
                f"({excluded} heating+EV hours excluded), "
                f"{len(profile.profile)} slots. {profile.summary()}"
            )
            return profile

        except Exception as e:
            self.logger.error(f"Failed to build base load profile: {e}", exc_info=True)
            return self._base_load_profile

    async def update_profile_with_yesterday(
        self, influxdb_client: Any, solar_bucket: str, loxone_bucket: str
    ) -> None:
        """Update profile with yesterday's actual data using EMA (0.9 old + 0.1 new)."""
        try:
            # Query yesterday's heating + EV hours
            heating_result = await influxdb_client.query(f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating" and r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)
''')
            ev_result = await influxdb_client.query(f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "ev" and r._field == "ev_charging" and r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)
''')
            high_load_hours: set = set()
            if heating_result:
                for table in heating_result:
                    for record in table.records:
                        high_load_hours.add(record.get_time().strftime("%Y-%m-%d-%H"))
            if ev_result:
                for table in ev_result:
                    for record in table.records:
                        high_load_hours.add(record.get_time().strftime("%Y-%m-%d-%H"))

            # Query yesterday's load
            load_result = await influxdb_client.query(f'''
from(bucket: "{solar_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> filter(fn: (r) => r._value > 0)
''')
            if not load_result:
                return

            for table in load_result:
                for record in table.records:
                    t = record.get_time()
                    key = t.strftime("%Y-%m-%d-%H")
                    if key in high_load_hours:
                        continue

                    hour = t.hour
                    is_weekend = t.weekday() >= 5
                    kwh = record.get_value() / 1000.0
                    bin_key = (hour, is_weekend)

                    old = self._base_load_profile.get(hour, is_weekend)
                    # EMA: 90% old + 10% new
                    self._base_load_profile.profile[bin_key] = 0.9 * old + 0.1 * kwh

            self._profile_updated = datetime.now()
            self.logger.info(f"Base load profile updated with yesterday's data")

        except Exception as e:
            self.logger.debug(f"Profile update failed: {e}")

    async def calibrate_night_reserve(
        self, influxdb_client: Any, solar_bucket: str, days: int = 90
    ) -> float:
        """Calculate overnight base load from historical data.

        Queries last N days of INVPowerToLocalLoad during night hours (18-07),
        caps at 500W/hr to exclude heating (which disables battery discharge),
        and computes the average overnight energy need.

        Args:
            influxdb_client: Async InfluxDB client
            solar_bucket: Solar InfluxDB bucket name
            days: Days of history to analyze

        Returns:
            Overnight reserve in kWh
        """
        try:
            query = f'''
from(bucket: "{solar_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> filter(fn: (r) => r._value > 0 and r._value < 2000)
'''
            result = await influxdb_client.query(query)
            if not result:
                return self._night_reserve_kwh

            # Sum night hours (18-07), capped at 500W to exclude heating
            night_watts: Dict[int, List[float]] = {}
            for table in result:
                for record in table.records:
                    h = record.get_time().hour
                    if h >= 18 or h < 7:
                        w = min(record.get_value(), 500)  # Cap at 500W (non-heating)
                        if h not in night_watts:
                            night_watts[h] = []
                        night_watts[h].append(w)

            if not night_watts:
                return self._night_reserve_kwh

            # Average kWh per night hour, sum across 13 night hours
            total_kwh = 0
            for h, watts_list in night_watts.items():
                avg_w = sum(watts_list) / len(watts_list)
                total_kwh += avg_w / 1000  # W -> kWh for 1 hour

            # Clamp to reasonable range
            reserve = max(3.0, min(8.0, total_kwh))
            old = self._night_reserve_kwh
            self._night_reserve_kwh = reserve
            self._night_reserve_updated = datetime.now()

            self.logger.info(
                f"Night reserve calibrated: {reserve:.1f} kWh "
                f"(was {old:.1f}, from {days}d of overnight base load data)"
            )
            return reserve

        except Exception as e:
            self.logger.warning(f"Night reserve calibration failed: {e}")
            return self._night_reserve_kwh

    def _compute_reserve_soc_per_block(
        self,
        blocks: List[Tuple[datetime, float]],
        prices: List[float],
        solar_hourly: Dict[Any, float],
        battery_capacity_kwh: float,
        min_soc: float,
        max_soc: float,
        kwh_per_block: float,
        efficiency: float,
    ) -> List[float]:
        """Compute effective minimum SOC for each block position.

        For each block i, calculates the reserve needed to cover base load
        from block i until the next recharge opportunity (cheap price or solar).
        Blocks closer to recharge need less reserve.

        Returns:
            List of effective_min_soc values, one per block.
        """
        n = len(blocks)
        if n == 0:
            return []

        sorted_prices = sorted(prices)
        price_threshold = sorted_prices[n // 4] if n > 4 else sorted_prices[0]
        max_reserve = battery_capacity_kwh * (max_soc - min_soc) / 100 * 0.85
        effective_min_socs = [min_soc] * n

        # Precompute next-recharge index for each block (scan forward once)
        # next_recharge[i] = index of next recharge block after i, or n if none
        next_recharge_idx = [n] * n
        # Scan backward: propagate the nearest recharge index
        for j in range(n - 1, -1, -1):
            solar_j = _forecast_value(solar_hourly, blocks[j][0])
            if solar_j > 1.0 or prices[j] <= price_threshold:
                next_recharge_idx[j] = j
            elif j + 1 < n:
                next_recharge_idx[j] = next_recharge_idx[j + 1]

        for block_idx in range(n):
            block_ts = blocks[block_idx][0]

            # Reserve at this block must bridge net load from the NEXT block
            # until the next recharge. next_recharge_idx[block_idx+1] is the
            # first recharge at/after block_idx+1; the [block_idx+1, window_end)
            # horizon excludes the recharge block itself (it refills mid-block),
            # so when the very next block is already a recharge the reserve is
            # 0 — intended: the upcoming block tops the battery back up.
            recharge_j = next_recharge_idx[min(block_idx + 1, n - 1)] if block_idx + 1 < n else n
            window_end = recharge_j if recharge_j < n else n

            # Net energy the battery must hold to cover consumption until the
            # next recharge. Work in per-BLOCK units over the SAME
            # [block_idx+1, window_end) horizon for load, solar and grid credit
            # so they can't disagree. Per block, solar first offsets the
            # concurrent base load; only the deficit must come from the battery
            # (charged at a discharge loss → /efficiency) and only the surplus
            # charges it (at a charge loss → *efficiency). This avoids the old
            # double-count where full solar was credited as incoming charge while
            # the full load was still booked into the reserve.
            gross_reserve_kwh = 0.0
            incoming_kwh = 0.0
            has_cheap_topup = False
            for j in range(block_idx + 1, window_end):
                ts_j = blocks[j][0]
                base_block = self._base_load_profile.get(
                    ts_j.hour, ts_j.weekday() >= 5
                ) / 4.0
                solar_block = max(0.0, _forecast_value(solar_hourly, ts_j) / 4.0)
                net = base_block - solar_block
                if net > 0:
                    gross_reserve_kwh += net / efficiency
                else:
                    incoming_kwh += (-net) * efficiency

                # A cheap-ish grid block before the recharge gives ONE top-up
                # opportunity. Only note it here — crediting a fresh full charge
                # block per cheap block (the old behaviour) over-credited
                # incoming on price-flat nights, zeroing the reserve and letting
                # discharge drain the battery with nothing left for base load.
                price_rank = sum(1 for p in prices if p < prices[j]) / n if n > 0 else 0.5
                if price_rank < 0.35:
                    has_cheap_topup = True

            if has_cheap_topup:
                incoming_kwh += kwh_per_block * efficiency

            reserve_kwh = max(0, gross_reserve_kwh - incoming_kwh)
            reserve_kwh = min(reserve_kwh, max_reserve)
            effective_min_socs[block_idx] = min(
                max_soc - 5,
                min_soc + (reserve_kwh / battery_capacity_kwh) * 100
            )

        # Store first block's info for dashboard
        if effective_min_socs:
            recharge_j = next_recharge_idx[min(1, n - 1)] if n > 1 else n
            recharge_ts = blocks[recharge_j][0] if recharge_j < n else None
            self._last_reserve_info = {
                "next_recharge_ts": recharge_ts.isoformat() if recharge_ts else None,
                "next_recharge_reason": "auto",
                "effective_min_soc": round(effective_min_socs[0], 1),
            }

        return effective_min_socs

    def optimize(
        self,
        blocks: List[Tuple[datetime, float]],  # (timestamp, price_czk_kwh)
        solar_hourly: Dict[Any, float],  # hour/date-hour -> kWh solar production
        consumption_hourly: Dict[Any, float],  # hour/date-hour -> kWh consumption
        distribution_func,  # (hour) -> distribution tariff CZK/kWh
        battery_capacity_kwh: float = 10.0,
        current_soc: float = 50.0,
        min_soc: float = 20.0,
        max_soc: float = 100.0,
        charge_rate_kw: float = 2.5,  # Max charge rate
        discharge_rate_kw: float = 2.5,  # Max discharge rate (at 100% power)
        discharge_power_pct: float = 25.0,  # Actual discharge power %
        efficiency: float = 0.85,
        sell_fee_czk: float = 0.5,  # Fee per kWh sold to grid
        battery_amortisation_czk: float = 2.0,  # Battery wear cost per kWh
        export_price_min: Optional[float] = None,  # strict export floor (CZK/kWh)
        inverter_off_price: Optional[float] = None,  # strict PV-off price (CZK/kWh)
    ) -> Tuple[Set[datetime], Set[datetime], Set[datetime], List[BlockDecision]]:
        """Optimize charge/discharge schedule.

        `export_price_min` / `inverter_off_price` are accepted for engine parity
        with the MILP (the controller passes the same kwargs to either engine).
        The greedy engine is the fallback; the controller's hardware export and
        inverter-off gates enforce these strict rules at actuation time.

        Args:
            blocks: Price blocks as (timestamp, price_czk_kwh) sorted chronologically
            solar_hourly: Hour or date-hour -> expected solar kWh
            consumption_hourly: Hour or date-hour -> expected consumption kWh
            distribution_func: Callable(hour) -> distribution tariff CZK/kWh
            battery_capacity_kwh: Total battery capacity
            current_soc: Current battery state of charge %
            min_soc: Minimum allowed SOC %
            max_soc: Maximum allowed SOC %
            charge_rate_kw: Maximum charging power in kW
            discharge_rate_kw: Maximum discharge power in kW
            discharge_power_pct: Discharge power rate percentage
            efficiency: Round-trip battery efficiency

        Returns:
            Tuple of (charge_timestamps, discharge_timestamps,
                      sell_production_timestamps, all_decisions)
        """
        if not blocks:
            return set(), set(), set(), []

        # Battery energy parameters
        kwh_per_block = charge_rate_kw * 0.25  # 15 minutes
        discharge_kwh_per_block = discharge_rate_kw * (discharge_power_pct / 100) * 0.25

        # Pre-compute block values (needed for reserve calculation below)
        prices = [p for _, p in blocks]
        sorted_prices = sorted(prices)
        n = len(blocks)

        # Compute per-block effective minimum SOC (dynamic reserve)
        first_block_ts = blocks[0][0] if blocks else datetime.now()
        effective_min_socs = self._compute_reserve_soc_per_block(
            blocks, prices, solar_hourly, battery_capacity_kwh,
            min_soc, max_soc, kwh_per_block, efficiency,
        )

        # Store first block's reserve info for dashboard (backward compatibility)
        if effective_min_socs:
            self._last_reserve_info["effective_min_soc"] = round(effective_min_socs[0], 1)

        # Future cheapest price (for estimating recharge cost)
        # For each block, what's the cheapest price available in remaining blocks?
        # Also track which hour that cheapest price occurs at (for distribution tariff)
        future_min_price = [0.0] * n
        future_min_price_hour = [0] * n
        running_min = float('inf')
        running_min_hour = 0
        for i in range(n - 1, -1, -1):
            if prices[i] <= running_min:
                running_min = prices[i]
                running_min_hour = blocks[i][0].hour
            future_min_price[i] = running_min
            future_min_price_hour[i] = running_min_hour

        # Future best discharge value: for each block, what's the maximum
        # discharge profit achievable in remaining blocks? Used to avoid
        # spending battery on mediocre blocks when better ones are ahead.
        future_max_discharge_value = [0.0] * n
        running_max = float('-inf')
        for i in range(n - 1, -1, -1):
            dist_i = distribution_func(blocks[i][0].hour)
            sell_rev_i = prices[i] - dist_i - sell_fee_czk - battery_amortisation_czk
            future_dist_i = distribution_func(future_min_price_hour[i])
            recharge_cost_i = (future_min_price[i] + future_dist_i) / efficiency
            # Mirror the per-block gate: discharges with non-positive sell_revenue
            # are never selected, so they shouldn't inflate "best-future" benchmarks.
            if sell_rev_i > 0:
                dv = sell_rev_i - recharge_cost_i
                running_max = max(running_max, dv)
            future_max_discharge_value[i] = running_max

        # Future self-consumption value: for each block, what's the best
        # self-consumption savings available in remaining blocks? Used to
        # prevent discharging battery when retaining it for expensive
        # self-consumption hours saves more money.
        # Only counts hours where consumption > solar (battery needed).
        # Also tracks cumulative kWh of future battery-needed consumption
        # so we know when the battery has excess beyond self-consumption needs.
        future_sc_value = [0.0] * n
        future_sc_kwh = [0.0] * n  # Cumulative kWh needing battery from block i onward
        running_best_sc = 0.0
        running_sc_kwh = 0.0
        for i in range(n - 1, -1, -1):
            ts_i = blocks[i][0]
            h_i = ts_i.hour
            solar_i = _forecast_value(solar_hourly, ts_i) / 4.0
            cons_i = _forecast_value(consumption_hourly, ts_i) / 4.0
            net_cons = max(0, cons_i - solar_i)
            if net_cons > 0:
                # Net SC value after battery wear: avoided buy minus amortisation.
                # If non-positive, SC from battery is worse than just buying from
                # grid, so it can't justify holding battery for that block.
                sc_val = prices[i] + distribution_func(h_i) - battery_amortisation_czk
                if sc_val > running_best_sc:
                    running_best_sc = sc_val
                running_sc_kwh += net_cons
            future_sc_value[i] = running_best_sc
            future_sc_kwh[i] = running_sc_kwh

        # Pre-select charge blocks: pick cheapest N blocks where charging is
        # profitable vs buying from grid later. Don't charge at 1.28 CZK when
        # -2.0 blocks are available — better to sit at min SOC and buy from grid.
        # The "or p < 0" safety covers deeply-negative days where the median
        # itself is below zero and the multiplicative threshold would otherwise
        # filter out marginally-negative blocks we still want as fallback.
        median_price = sorted_prices[n // 2] if n > 0 else 2.0
        charge_threshold = median_price * efficiency  # Only charge below this
        profitable_charge_blocks = [
            (ts, p) for ts, p in blocks if p < charge_threshold or p < 0
        ]
        profitable_charge_blocks.sort(key=lambda x: x[1])  # Cheapest first

        # Calculate how much grid charging we actually need:
        # Total battery gap minus expected net solar charging (solar - consumption)
        battery_gap_kwh = battery_capacity_kwh * (max_soc - current_soc) / 100
        expected_solar_charge = 0.0
        for ts, _ in blocks:
            solar_h = _forecast_value(solar_hourly, ts) / 4.0  # kWh per 15-min block
            consumption_h = _forecast_value(consumption_hourly, ts) / 4.0
            net = solar_h - consumption_h
            if net > 0:
                expected_solar_charge += net * efficiency
        grid_needed_kwh = max(0, battery_gap_kwh - expected_solar_charge)

        # Convert to blocks needed (min 4 to always grab the cheapest/negative prices)
        kwh_per_charge = kwh_per_block * efficiency
        blocks_to_fill = max(4, int(grid_needed_kwh / kwh_per_charge) + 1) if kwh_per_charge > 0 else 4

        # Pick exactly the cheapest N needed. Do NOT union with all-negatives:
        # the forward simulation walks chronologically, so adding extra
        # candidates ahead of cheaper ones causes early candidates to consume
        # battery headroom that the deeper-negative later blocks then cannot
        # use. Sorting profitable_charge_blocks ascending and taking [:N] is
        # already the optimum because all negatives are in the candidate pool.
        max_charge = min(len(profitable_charge_blocks), blocks_to_fill)
        charge_block_set = set(ts for ts, _ in profitable_charge_blocks[:max_charge])

        # === SELL-PRODUCTION SCHEDULING ===
        # When morning solar would otherwise charge the battery but later in the
        # day either (a) cheaper grid energy or (b) more solar is available,
        # exporting morning solar straight to grid is more profitable than
        # storing it. See SELL_PRODUCTION_MARGIN_CZK and the swap-profit logic.
        sell_production_set = self._select_sell_production_blocks(
            blocks, charge_block_set, solar_hourly, consumption_hourly,
            distribution_func, battery_capacity_kwh, current_soc, max_soc,
            kwh_per_block, efficiency, sell_fee_czk,
            battery_amortisation_czk, future_sc_value,
        )

        # Pass 1: initial forward simulation
        charge_times, discharge_times, sell_production_times, decisions = self._single_pass(
            blocks, charge_block_set, sell_production_set,
            solar_hourly, consumption_hourly,
            distribution_func, battery_capacity_kwh, current_soc, min_soc,
            max_soc, kwh_per_block, discharge_kwh_per_block, efficiency,
            effective_min_socs, future_min_price, future_min_price_hour,
            future_max_discharge_value, future_sc_value, future_sc_kwh,
            sorted_prices, sell_fee_czk, battery_amortisation_czk,
            first_block_ts,
        )

        # Pass 2: refine charge block selection using actual SOC trajectory
        refined_set = self._refine_charge_blocks(
            blocks, charge_block_set, decisions, max_soc
        )
        if refined_set != charge_block_set:
            moved = len(charge_block_set - refined_set)
            self.logger.info(
                f"Pass 2: redistributed {moved} charge blocks "
                f"(wasted slots replaced with better candidates)"
            )
            charge_times, discharge_times, sell_production_times, decisions = self._single_pass(
                blocks, refined_set, sell_production_set,
                solar_hourly, consumption_hourly,
                distribution_func, battery_capacity_kwh, current_soc, min_soc,
                max_soc, kwh_per_block, discharge_kwh_per_block, efficiency,
                effective_min_socs, future_min_price, future_min_price_hour,
                future_max_discharge_value, future_sc_value, future_sc_kwh,
                sorted_prices, sell_fee_czk, battery_amortisation_czk,
                first_block_ts,
            )

        self._last_decisions = decisions
        return charge_times, discharge_times, sell_production_times, decisions

    def _select_sell_production_blocks(
        self,
        blocks: List[Tuple[datetime, float]],
        charge_block_set: Set[datetime],
        solar_hourly: Dict[Any, float],
        consumption_hourly: Dict[Any, float],
        distribution_func,
        battery_capacity_kwh: float,
        current_soc: float,
        max_soc: float,
        kwh_per_block: float,
        efficiency: float,
        sell_fee_czk: float,
        battery_amortisation_czk: float,
        future_sc_value: List[float],
    ) -> Set[datetime]:
        """Mark blocks where solar should be exported instead of stored.

        Note: sell_production is the WEAKER variant of selling — it exports
        solar excess only, leaving the battery passive. Discharge is the
        STRONGER variant: it sells battery AND solar excess. So sell_production
        is needed primarily in the gap zone where sell_revenue (battery export)
        is unprofitable due to amortisation but sell_now (solar export, no
        battery wear) is still profitable.

        Per-block swap profit:

            sell_now           = spot - dist - fees                  # solar→grid revenue
            grid_replacement   = (future_min_charge_spot + dist) / efficiency
            solar_replacement  = future_min_export_revenue if solar refills battery
            storage_value      = future_sc_value[i]  (already amort-adjusted)

            effective_loss     = min(storage_value, grid_replacement, solar_replacement)
            swap_profit        = sell_now - effective_loss

        eligible if swap_profit > MARGIN.

        Aggregate refill budget caps total exported solar to what can actually
        be replaced from cheap grid + future solar surplus.
        """
        n = len(blocks)
        if n == 0:
            return set()

        max_battery_kwh = battery_capacity_kwh * max_soc / 100
        current_battery_kwh = battery_capacity_kwh * current_soc / 100
        battery_gap_remaining = max(0.0, max_battery_kwh - current_battery_kwh)

        # Per-block sell-now revenue and solar excess
        sell_now = [0.0] * n
        solar_excess = [0.0] * n
        for i, (ts, p) in enumerate(blocks):
            h = ts.hour
            sell_now[i] = p - distribution_func(h) - sell_fee_czk
            s = _forecast_value(solar_hourly, ts) / 4.0
            c = _forecast_value(consumption_hourly, ts) / 4.0
            solar_excess[i] = max(0.0, s - c)

        # Future solar surplus suffix sums (kWh available from block i+1 onward)
        future_solar_surplus = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            future_solar_surplus[i] = future_solar_surplus[i + 1] + solar_excess[i]

        # Future cheapest grid charge cost STRICTLY after each block (per battery kWh).
        future_min_charge_after = [float('inf')] * (n + 1)
        for i in range(n - 1, -1, -1):
            future_min_charge_after[i] = future_min_charge_after[i + 1]
            if i + 1 < n:
                ts_next, p_next = blocks[i + 1]
                if ts_next in charge_block_set:
                    cost = (p_next + distribution_func(ts_next.hour)) / efficiency
                    if cost < future_min_charge_after[i]:
                        future_min_charge_after[i] = cost

        # Future minimum export revenue at solar-surplus blocks STRICTLY after
        # each block — the "cheapest" future solar moment to sell at (we'd
        # rather export at the higher-priced morning than at this lower future).
        future_min_export_revenue_after = [float('inf')] * (n + 1)
        for i in range(n - 1, -1, -1):
            future_min_export_revenue_after[i] = future_min_export_revenue_after[i + 1]
            if i + 1 < n and solar_excess[i + 1] > 0 and sell_now[i + 1] > 0:
                if sell_now[i + 1] < future_min_export_revenue_after[i]:
                    future_min_export_revenue_after[i] = sell_now[i + 1]

        # Score eligible blocks
        candidates: List[Tuple[float, int]] = []  # (-swap_profit, i) for sort
        for i in range(n):
            # Require a minimum solar surplus to absorb consumption-forecast
            # error — otherwise the mode change locks the battery and we end
            # up importing the real load deficit from grid.
            if (
                solar_excess[i] < SELL_PRODUCTION_MIN_EXCESS_KWH
                or sell_now[i] <= SELL_PRODUCTION_MARGIN_CZK
            ):
                continue

            grid_replacement = future_min_charge_after[i]
            solar_only_refills = (
                future_solar_surplus[i + 1] * efficiency >= battery_gap_remaining
            )
            solar_replacement = (
                future_min_export_revenue_after[i] if solar_only_refills
                else float('inf')
            )
            storage_value = future_sc_value[i] if i < len(future_sc_value) else 0.0
            effective_loss = min(storage_value, grid_replacement, solar_replacement)
            if effective_loss == float('inf'):
                # No refill option AND no SC value — fully forfeit storage value.
                # storage_value is already finite (≥ 0), so this only triggers
                # when SC value happens to be 0; treat as straightforward sell.
                effective_loss = 0.0

            swap_profit = sell_now[i] - effective_loss
            if swap_profit > SELL_PRODUCTION_MARGIN_CZK:
                candidates.append((-swap_profit, i))

        if not candidates:
            return set()

        # Aggregate refill budget: cheap grid charging capacity + future solar
        cheap_charge_kwh = sum(
            kwh_per_block * efficiency
            for ts, _ in blocks if ts in charge_block_set
        )
        refill_budget = cheap_charge_kwh + future_solar_surplus[0] * efficiency

        # Allocate top-down by swap profit
        candidates.sort()  # smallest -profit = highest profit first
        sell_production_set: Set[datetime] = set()
        allocated = 0.0
        for _neg_profit, i in candidates:
            if allocated + solar_excess[i] > refill_budget:
                continue
            sell_production_set.add(blocks[i][0])
            allocated += solar_excess[i]
        return sell_production_set

    def _refine_charge_blocks(
        self,
        blocks: List[Tuple[datetime, float]],
        original_charge_set: Set[datetime],
        decisions: List[BlockDecision],
        max_soc: float,
    ) -> Set[datetime]:
        """Identify wasted charge blocks and replace with better candidates.

        A charge block is "wasted" if the battery was nearly full (from solar)
        and the charge action barely increased SOC. These wasted slots are
        replaced with the cheapest available hold blocks that had room to charge.

        Returns the original set if no improvement is possible.
        """
        # Find wasted charge blocks: SOC barely changed (< 1%)
        # Never consider negative-price blocks as wasted (we get paid to charge)
        wasted = {
            d.timestamp for d in decisions
            if d.action == "charge"
            and (d.soc_after - d.soc_before) < 1.0
            and d.price_czk >= 0
        }
        if not wasted:
            return original_charge_set

        # Find candidate replacement blocks: hold blocks with room to charge
        candidates = sorted(
            [(d.timestamp, d.price_czk) for d in decisions
             if d.action == "hold" and d.soc_before < max_soc - 5],
            key=lambda x: x[1]  # Cheapest first
        )

        if not candidates:
            return original_charge_set

        # Replace wasted blocks with cheapest candidates
        new_set = original_charge_set - wasted
        for ts, _ in candidates[:len(wasted)]:
            new_set.add(ts)

        return new_set

    def _single_pass(
        self,
        blocks: List[Tuple[datetime, float]],
        charge_block_set: Set[datetime],
        sell_production_set: Set[datetime],
        solar_hourly: Dict[Any, float],
        consumption_hourly: Dict[Any, float],
        distribution_func,
        battery_capacity_kwh: float,
        current_soc: float,
        min_soc: float,
        max_soc: float,
        kwh_per_block: float,
        discharge_kwh_per_block: float,
        efficiency: float,
        effective_min_socs: List[float],
        future_min_price: List[float],
        future_min_price_hour: List[int],
        future_max_discharge_value: List[float],
        future_sc_value: List[float],
        future_sc_kwh: List[float],
        sorted_prices: List[float],
        sell_fee_czk: float,
        battery_amortisation_czk: float,
        first_block_ts: datetime,
    ) -> Tuple[Set[datetime], Set[datetime], Set[datetime], List[BlockDecision]]:
        """Run a single forward simulation pass with given charge block set."""
        n = len(blocks)
        soc = current_soc
        decisions: List[BlockDecision] = []
        charge_times: Set[datetime] = set()
        discharge_times: Set[datetime] = set()
        sell_production_times: Set[datetime] = set()

        for i, (timestamp, price_czk) in enumerate(blocks):
            hour = timestamp.hour
            dist = distribution_func(hour)

            # Solar and consumption for this 15-min block (quarter of hourly value)
            solar = _forecast_value(solar_hourly, timestamp) / 4.0
            consumption = _forecast_value(consumption_hourly, timestamp) / 4.0

            # Net solar: excess solar charges battery automatically
            net_solar = solar - consumption  # Positive = excess, negative = deficit

            # Current battery energy (per-block dynamic reserve)
            effective_min_soc = effective_min_socs[i] if i < len(effective_min_socs) else min_soc
            battery_kwh = battery_capacity_kwh * soc / 100
            max_battery_kwh = battery_capacity_kwh * max_soc / 100
            min_battery_kwh = battery_capacity_kwh * effective_min_soc / 100

            # Score each action
            # Self-consumption value = avoided buy price - battery wear cost.
            # Using battery to power loads incurs amortisation just like grid discharge.
            self_consumption_value = price_czk + dist - battery_amortisation_czk

            # Recharge cost estimate: cheapest future price + its distribution
            future_cheapest = future_min_price[i] if i < n else sorted_prices[0]

            # --- CHARGE value ---
            charge_cost = price_czk + dist
            median_price = sorted_prices[n // 2] if n > 0 else 0
            future_value = (median_price + dist) * efficiency
            charge_possible = min(kwh_per_block, (max_battery_kwh - battery_kwh))
            charge_value = (future_value - charge_cost) if charge_possible > 0 else float('-inf')

            # --- DISCHARGE value ---
            # sell_revenue already accounts for amortisation. Hard floor: never
            # discharge to grid when the sale itself is unprofitable. Round-trip
            # arbitrage with hypothetical future recharge does not justify
            # destroying value now — every discharged kWh wears the battery.
            sell_revenue = price_czk - dist - sell_fee_czk - battery_amortisation_czk
            future_recharge_hour = future_min_price_hour[i] if i < n else 0
            future_dist = distribution_func(future_recharge_hour)
            recharge_cost = (future_cheapest + future_dist) / efficiency
            discharge_profit = sell_revenue - recharge_cost
            discharge_possible = min(
                discharge_kwh_per_block,
                (battery_kwh - min_battery_kwh) * efficiency
            )
            if sell_revenue <= 0 or discharge_possible <= 0:
                discharge_value = float('-inf')
            else:
                discharge_value = discharge_profit

            # --- HOLD value ---
            hold_value = 0.0
            if net_solar > 0 and battery_kwh < max_battery_kwh:
                solar_charge = min(net_solar, max_battery_kwh - battery_kwh)
                hold_value += solar_charge * self_consumption_value * efficiency

            # Value of retaining battery for future self-consumption:
            # each kWh kept now can offset a future grid purchase at the best
            # upcoming price. Only applies when battery energy above reserve
            # is NEEDED for future consumption (not excess that could be sold).
            usable_kwh = battery_kwh - min_battery_kwh
            if usable_kwh > 0 and i < n:
                best_future_sc = future_sc_value[i]
                sc_kwh_needed = future_sc_kwh[i] / efficiency  # Account for losses
                if best_future_sc > recharge_cost and usable_kwh <= sc_kwh_needed:
                    # Battery is fully needed for self-consumption — don't discharge
                    hold_value = max(hold_value, best_future_sc - recharge_cost)

            # Pick best action
            soc_before = soc
            action = "hold"
            net_value = hold_value

            if timestamp in charge_block_set and charge_possible > 0:
                action = "charge"
                net_value = charge_value

            best_remaining = future_max_discharge_value[i] if i < n else 0
            is_worthwhile = (best_remaining <= 0) or (discharge_value >= best_remaining * 0.8)
            if discharge_value > 0 and discharge_value > net_value and is_worthwhile:
                if discharge_possible > 0 and soc > effective_min_soc:
                    action = "discharge"
                    net_value = discharge_value

            # Sell-production wins over charge/hold but never displaces a
            # profitable discharge — discharge is the stronger variant: it
            # sells battery AND solar excess. Sell_production only sells
            # solar excess (battery passive). So when discharge fires, it
            # already covers what sell_production would do.
            if timestamp in sell_production_set and action != "discharge":
                action = "sell_production"
                # Informational value: sell-now revenue × solar excess this block
                solar_excess_now = max(0.0, solar - consumption)
                net_value = (price_czk - dist - sell_fee_czk) * solar_excess_now

            # === SOC SIMULATION ===
            solar_available = solar
            house_load = consumption if consumption > 0 else (
                self._base_load_profile.get(hour, first_block_ts.weekday() >= 5) / 4.0
            )
            net_from_solar = solar_available - house_load

            if action == "charge":
                grid_charge = kwh_per_block
                soc += (grid_charge * efficiency) / battery_capacity_kwh * 100
                if net_from_solar > 0:
                    solar_to_batt = min(net_from_solar * efficiency,
                                        max_battery_kwh - battery_kwh)
                    soc += solar_to_batt / battery_capacity_kwh * 100
                soc = min(max_soc, soc)
            elif action == "discharge":
                # `discharge_possible` is the GRID-DELIVERED energy and already
                # respects the per-block dynamic reserve (battery_kwh-min)*eff,
                # so it can't push SOC below effective_min_soc. Delivering that
                # to the grid drains battery_delivered/efficiency of stored
                # energy (inverter/round-trip loss). Mirrors the MILP, which
                # drains battery by grid_export / eta.
                battery_drain = discharge_possible / efficiency
                if net_from_solar < 0:
                    # House deficit is also served from the battery (also lossy).
                    # This adds on top of the grid-export drain, so the combined
                    # drain can exceed (battery_kwh - min_battery_kwh); clamp the
                    # final SOC at the per-block dynamic reserve, not the hw floor.
                    battery_drain += abs(net_from_solar) / efficiency
                soc -= battery_drain / battery_capacity_kwh * 100
                soc = max(effective_min_soc, soc)
            elif action == "sell_production":
                # Solar excess flows to grid (battery does NOT charge from it).
                # Battery only drains if loads exceed solar (auto-SC), and only
                # down to the dynamic reserve — preserve overnight energy the
                # same way the discharge branch does (was: hardware floor).
                if net_from_solar < 0:
                    need = abs(net_from_solar) / efficiency  # battery-side, lossy
                    avail = max(0.0, battery_kwh - min_battery_kwh)
                    battery_drain = min(need, avail)
                    if battery_drain > 0:
                        soc -= battery_drain / battery_capacity_kwh * 100
                        soc = max(effective_min_soc, soc)
                # net_from_solar >= 0: solar covers loads, excess to grid, SOC unchanged.
            else:  # hold
                if net_from_solar > 0:
                    solar_to_batt = min(net_from_solar * efficiency,
                                        max_battery_kwh - battery_kwh)
                    if solar_to_batt > 0:
                        soc += solar_to_batt / battery_capacity_kwh * 100
                        soc = min(max_soc, soc)
                else:
                    # Pure self-consumption: serve the deficit from the battery
                    # down to the per-block DYNAMIC reserve (not the hardware
                    # floor) so holding preserves overnight energy. Battery-side
                    # draw is lossy, matching the value model.
                    need = abs(net_from_solar) / efficiency
                    avail = max(0.0, battery_kwh - min_battery_kwh)
                    draw = min(need, avail)
                    if draw > 0:
                        soc -= draw / battery_capacity_kwh * 100
                        soc = max(effective_min_soc, soc)

            decision = BlockDecision(
                timestamp=timestamp,
                action=action,
                price_czk=price_czk,
                distribution_czk=dist,
                solar_kwh=solar,
                consumption_kwh=consumption,
                soc_before=soc_before,
                soc_after=soc,
                net_value=net_value,
            )
            decisions.append(decision)

            if action == "charge":
                charge_times.add(timestamp)
            elif action == "discharge":
                discharge_times.add(timestamp)
            elif action == "sell_production":
                sell_production_times.add(timestamp)

        return charge_times, discharge_times, sell_production_times, decisions

    def summarize(self, decisions: List[BlockDecision]) -> Dict:
        """Summarize optimizer decisions for logging."""
        if not decisions:
            return {}

        charge_blocks = [d for d in decisions if d.action == "charge"]
        discharge_blocks = [d for d in decisions if d.action == "discharge"]
        hold_blocks = [d for d in decisions if d.action == "hold"]
        sell_production_blocks = [d for d in decisions if d.action == "sell_production"]

        return {
            "total_blocks": len(decisions),
            "sell_production_blocks": len(sell_production_blocks),
            "charge_blocks": len(charge_blocks),
            "discharge_blocks": len(discharge_blocks),
            "hold_blocks": len(hold_blocks),
            "avg_charge_price": (
                sum(d.price_czk for d in charge_blocks) / len(charge_blocks)
                if charge_blocks else 0
            ),
            "avg_discharge_price": (
                sum(d.price_czk for d in discharge_blocks) / len(discharge_blocks)
                if discharge_blocks else 0
            ),
            "final_soc": decisions[-1].soc_after if decisions else 0,
        }
