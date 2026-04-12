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
    action: str  # "charge", "discharge", "hold"
    price_czk: float  # Spot price CZK/kWh
    distribution_czk: float  # Distribution tariff CZK/kWh
    solar_kwh: float  # Expected solar production this block
    consumption_kwh: float  # Expected consumption this block
    soc_before: float  # SOC % before action
    soc_after: float  # SOC % after action
    net_value: float  # Value of this action (positive = saves money)


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

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._base_load_profile: BaseLoadProfile = BaseLoadProfile()
        self._profile_updated: Optional[datetime] = None
        self._last_reserve_info: Dict[str, Any] = {}
        self._last_decisions: List[BlockDecision] = []

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
            from datetime import date as date_type
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

    def optimize(
        self,
        blocks: List[Tuple[datetime, float]],  # (timestamp, price_czk_kwh)
        solar_hourly: Dict[int, float],  # hour -> kWh solar production
        consumption_hourly: Dict[int, float],  # hour -> kWh consumption
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
    ) -> Tuple[Set[datetime], Set[datetime], List[BlockDecision]]:
        """Optimize charge/discharge schedule.

        Args:
            blocks: Price blocks as (timestamp, price_czk_kwh) sorted chronologically
            solar_hourly: Hour -> expected solar kWh
            consumption_hourly: Hour -> expected consumption kWh
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
            Tuple of (charge_timestamps, discharge_timestamps, all_decisions)
        """
        if not blocks:
            return set(), set(), []

        # Battery energy parameters
        usable_capacity = battery_capacity_kwh * (max_soc - min_soc) / 100
        kwh_per_block = charge_rate_kw * 0.25  # 15 minutes
        discharge_kwh_per_block = discharge_rate_kw * (discharge_power_pct / 100) * 0.25

        # Pre-compute block values (needed for reserve calculation below)
        prices = [p for _, p in blocks]
        sorted_prices = sorted(prices)
        n = len(blocks)

        # Dynamic reserve: find next recharge opportunity (cheap price or solar)
        # then sum base load hours until that point
        first_block_ts = blocks[0][0] if blocks else datetime.now()
        is_weekend = first_block_ts.weekday() >= 5

        # Find next recharge: iterate blocks chronologically (today + tomorrow)
        price_threshold = sorted_prices[n // 4] if n > 4 else sorted_prices[0]
        next_recharge_ts = None
        next_recharge_reason = "none"
        hours_until_recharge = 0

        for i, (ts, price) in enumerate(blocks):
            if i == 0:
                continue  # Skip current block
            h = ts.hour
            solar_h = solar_hourly.get(h, 0)

            if solar_h > 1.0:
                next_recharge_ts = ts
                next_recharge_reason = f"solar ({solar_h:.1f} kWh)"
                break
            elif price <= price_threshold:
                next_recharge_ts = ts
                next_recharge_reason = f"cheap ({price:.2f} CZK)"
                break

        if next_recharge_ts:
            # Count actual hours between now and recharge
            delta = (next_recharge_ts - first_block_ts).total_seconds() / 3600
            hours_until_recharge = max(1, int(delta))
        else:
            # No recharge found in schedule — assume 24 hours (worst case)
            hours_until_recharge = 24

        # Sum base load for each hour until recharge
        gross_reserve_kwh = 0.0
        for offset in range(hours_until_recharge):
            future_hour = (first_block_ts.hour + offset) % 24
            future_weekend = (first_block_ts + timedelta(hours=offset)).weekday() >= 5
            gross_reserve_kwh += self._base_load_profile.get(future_hour, future_weekend)

        # Account for efficiency
        gross_reserve_kwh = gross_reserve_kwh / efficiency

        # Subtract incoming energy: scheduled charge blocks + solar before recharge
        incoming_charge_kwh = 0.0
        for ts, price in blocks:
            if next_recharge_ts and ts >= next_recharge_ts:
                break
            if ts <= first_block_ts:
                continue
            # Cheap blocks (bottom 35%) will charge the battery
            price_rank = sum(1 for p in prices if p < price) / n if n > 0 else 0.5
            if price_rank < 0.35:
                incoming_charge_kwh += kwh_per_block * efficiency
            # Solar production for this block (quarter of hourly)
            solar_block = solar_hourly.get(ts.hour, 0) / 4.0
            if solar_block > 0:
                incoming_charge_kwh += solar_block * efficiency

        reserve_kwh = max(0, gross_reserve_kwh - incoming_charge_kwh)

        # Clamp to usable capacity
        max_reserve = battery_capacity_kwh * (max_soc - min_soc) / 100 * 0.85
        reserve_kwh = min(reserve_kwh, max_reserve)
        effective_min_soc = min(max_soc - 5, min_soc + (reserve_kwh / battery_capacity_kwh) * 100)

        # Store for dashboard
        self._last_reserve_info = {
            "next_recharge_ts": next_recharge_ts.isoformat() if next_recharge_ts else None,
            "next_recharge_reason": next_recharge_reason,
            "hours_until_recharge": hours_until_recharge,
            "gross_reserve_kwh": round(gross_reserve_kwh, 1),
            "incoming_charge_kwh": round(incoming_charge_kwh, 1),
            "net_reserve_kwh": round(reserve_kwh, 1),
            "effective_min_soc": round(effective_min_soc, 1),
        }

        # Future cheapest price (for estimating recharge cost)
        # For each block, what's the cheapest price available in remaining blocks?
        future_min_price = [0.0] * n
        running_min = float('inf')
        for i in range(n - 1, -1, -1):
            running_min = min(running_min, prices[i])
            future_min_price[i] = running_min

        # Pre-select charge blocks: pick cheapest N blocks where charging is
        # profitable vs buying from grid later. Don't charge at 1.28 CZK when
        # -2.0 blocks are available — better to sit at min SOC and buy from grid.
        median_price = sorted_prices[n // 2] if n > 0 else 2.0
        charge_threshold = median_price * efficiency  # Only charge below this
        profitable_charge_blocks = [
            (ts, p) for ts, p in blocks if p < charge_threshold
        ]
        profitable_charge_blocks.sort(key=lambda x: x[1])  # Cheapest first

        # Calculate how much grid charging we actually need:
        # Total battery gap minus expected net solar charging (solar - consumption)
        battery_gap_kwh = battery_capacity_kwh * (max_soc - current_soc) / 100
        expected_solar_charge = 0.0
        for ts, _ in blocks:
            h = ts.hour
            solar_h = solar_hourly.get(h, 0) / 4.0  # kWh per 15-min block
            consumption_h = consumption_hourly.get(h, 0) / 4.0
            net = solar_h - consumption_h
            if net > 0:
                expected_solar_charge += net * efficiency
        grid_needed_kwh = max(0, battery_gap_kwh - expected_solar_charge)

        # Convert to blocks needed (min 4 to always grab the cheapest/negative prices)
        kwh_per_charge = kwh_per_block * efficiency
        blocks_to_fill = max(4, int(grid_needed_kwh / kwh_per_charge) + 1) if kwh_per_charge > 0 else 4

        # Also always include negative price blocks (get paid to charge)
        negative_blocks = set(ts for ts, p in blocks if p < 0)

        max_charge = min(len(profitable_charge_blocks), blocks_to_fill)
        charge_block_set = set(ts for ts, _ in profitable_charge_blocks[:max_charge]) | negative_blocks

        # Forward simulation
        soc = current_soc
        decisions: List[BlockDecision] = []
        charge_times: Set[datetime] = set()
        discharge_times: Set[datetime] = set()

        for i, (timestamp, price_czk) in enumerate(blocks):
            hour = timestamp.hour
            dist = distribution_func(hour)

            # Solar and consumption for this 15-min block (quarter of hourly value)
            solar = solar_hourly.get(hour, 0.0) / 4.0
            consumption = consumption_hourly.get(hour, 0.0) / 4.0

            # Net solar: excess solar charges battery automatically
            net_solar = solar - consumption  # Positive = excess, negative = deficit

            # Current battery energy
            battery_kwh = battery_capacity_kwh * soc / 100
            max_battery_kwh = battery_capacity_kwh * max_soc / 100
            min_battery_kwh = battery_capacity_kwh * effective_min_soc / 100

            # Score each action
            # Self-consumption value = price + distribution (what we save by using battery)
            self_consumption_value = price_czk + dist

            # Recharge cost estimate: cheapest future price + its distribution
            future_cheapest = future_min_price[i] if i < n else sorted_prices[0]

            # --- CHARGE value ---
            # Worth charging if future self-consumption value exceeds charge cost
            charge_cost = price_czk + dist  # Cost to charge now
            # Median price as estimate of future self-consumption value
            median_price = sorted_prices[n // 2] if n > 0 else 0
            future_value = (median_price + dist) * efficiency
            charge_possible = min(kwh_per_block, (max_battery_kwh - battery_kwh))
            charge_value = (future_value - charge_cost) if charge_possible > 0 else float('-inf')

            # --- DISCHARGE value ---
            # Real sell revenue = spot - distribution - sell_fee - battery_amortisation
            # Must exceed recharge cost to be profitable
            recharge_cost = future_cheapest / efficiency  # Cost to refill later
            sell_revenue = price_czk - dist - sell_fee_czk - battery_amortisation_czk
            discharge_profit = sell_revenue - recharge_cost  # Net profit per kWh
            discharge_possible = min(
                discharge_kwh_per_block,
                (battery_kwh - min_battery_kwh) * efficiency
            )
            discharge_value = discharge_profit if discharge_possible > 0 else float('-inf')

            # --- HOLD value ---
            # Battery stays put; solar charges, consumption draws from grid
            hold_value = 0.0

            # If there's excess solar and battery has room, solar charges battery
            # for free — this is value we get without doing anything
            if net_solar > 0 and battery_kwh < max_battery_kwh:
                # Solar fills battery for free → future self-consumption savings
                solar_charge = min(net_solar, max_battery_kwh - battery_kwh)
                # Don't need grid charging this block if solar covers it
                hold_value += solar_charge * self_consumption_value * efficiency

            # Pick best action
            soc_before = soc
            action = "hold"
            net_value = hold_value

            # Charge only during pre-selected cheapest blocks
            if timestamp in charge_block_set and charge_possible > 0:
                action = "charge"
                net_value = charge_value

            # Discharge when profitable (sell revenue > recharge cost + fees)
            if discharge_value > 0 and discharge_value > net_value:
                if discharge_possible > 0 and soc > effective_min_soc:
                    action = "discharge"
                    net_value = discharge_value

            # === REALISTIC SOC SIMULATION ===
            # Every block: house consumes, solar produces, charge/discharge adds/removes
            # Base load always drains battery (when no solar/grid covers it)

            # 1. Solar production this block
            solar_available = solar  # kWh this 15-min block

            # 2. House consumption this block
            # consumption is already per 15-min block (hourly / 4 from line above)
            house_load = consumption if consumption > 0 else (
                self._base_load_profile.get(hour, first_block_ts.weekday() >= 5) / 4.0
            )

            # 3. Net energy balance before charge/discharge command
            net_from_solar = solar_available - house_load  # positive = excess solar

            if action == "charge":
                # Grid charges battery at charge_rate, house still consumes
                # Net battery change = grid charge * efficiency + excess solar * efficiency - deficit from load
                grid_charge = kwh_per_block
                if net_from_solar > 0:
                    # Solar covers house + charges battery
                    solar_to_batt = min(net_from_solar * efficiency,
                                        (max_battery_kwh - battery_kwh))
                    soc += (grid_charge * efficiency + solar_to_batt) / battery_capacity_kwh * 100
                else:
                    # House draws some from grid charge, rest goes to battery
                    # Grid provides charge_rate, some goes to house, rest to battery
                    net_charge = grid_charge * efficiency + net_from_solar  # net_from_solar is negative
                    soc += max(0, net_charge) / battery_capacity_kwh * 100
                soc = min(max_soc, soc)

            elif action == "discharge":
                # Battery discharges at discharge_rate TO GRID + house consumes from battery
                grid_discharge = discharge_kwh_per_block
                if net_from_solar >= 0:
                    soc -= grid_discharge / battery_capacity_kwh * 100
                else:
                    total_drain = grid_discharge + abs(net_from_solar)
                    soc -= total_drain / battery_capacity_kwh * 100
                # Discharge stops at physical min SOC (not reserve floor)
                soc = max(min_soc, soc)

            else:  # hold
                if net_from_solar > 0:
                    solar_to_batt = min(net_from_solar * efficiency,
                                        max_battery_kwh - battery_kwh)
                    if solar_to_batt > 0:
                        soc += solar_to_batt / battery_capacity_kwh * 100
                        soc = min(max_soc, soc)
                else:
                    # House draws from battery down to physical min SOC
                    draw = min(abs(net_from_solar),
                               battery_capacity_kwh * soc / 100 - battery_capacity_kwh * min_soc / 100)
                    if draw > 0:
                        soc -= draw / battery_capacity_kwh * 100
                        soc = max(min_soc, soc)

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

        self._last_decisions = decisions
        return charge_times, discharge_times, decisions

    def summarize(self, decisions: List[BlockDecision]) -> Dict:
        """Summarize optimizer decisions for logging."""
        if not decisions:
            return {}

        charge_blocks = [d for d in decisions if d.action == "charge"]
        discharge_blocks = [d for d in decisions if d.action == "discharge"]
        hold_blocks = [d for d in decisions if d.action == "hold"]

        return {
            "total_blocks": len(decisions),
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
