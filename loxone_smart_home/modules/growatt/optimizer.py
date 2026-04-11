"""Greedy forward-simulation optimizer for battery charge/discharge scheduling.

Replaces rule-based scheduling with a block-by-block simulation that considers
price curve, solar forecast, consumption forecast, and battery constraints to
minimize total electricity cost.
"""

import logging
from dataclasses import dataclass
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


class BatteryOptimizer:
    """Greedy optimizer for 15-minute block scheduling.

    Two-pass approach:
    1. Score each block for charge/discharge/hold value
    2. Forward simulate, picking best feasible action per block
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        # Data-driven night reserve (updated daily from InfluxDB)
        self._night_reserve_kwh: float = 5.0  # Default, updated by calibrate_night_reserve
        self._night_reserve_updated: Optional[datetime] = None

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

        # Night reserve: keep enough battery for overnight self-consumption
        # Calibrated from historical data (non-heating base load 18:00-07:00)
        night_reserve_kwh = self._night_reserve_kwh
        night_reserve_soc = min_soc + (night_reserve_kwh / battery_capacity_kwh) * 100
        effective_min_soc = min(max_soc - 10, max(min_soc, night_reserve_soc))

        # Pre-compute block values
        prices = [p for _, p in blocks]
        sorted_prices = sorted(prices)

        # Future cheapest price (for estimating recharge cost)
        # For each block, what's the cheapest price available in remaining blocks?
        n = len(blocks)
        future_min_price = [0.0] * n
        running_min = float('inf')
        for i in range(n - 1, -1, -1):
            running_min = min(running_min, prices[i])
            future_min_price[i] = running_min

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
            # Worth discharging if current price exceeds opportunity cost
            recharge_cost = future_cheapest / efficiency  # Cost to refill later
            discharge_profit = price_czk - recharge_cost - dist  # Net profit per kWh
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

            if charge_value > discharge_value and charge_value > hold_value:
                # Charge is best — but only if price is in the cheapest portion
                # Use price ranking: only charge in bottom 30% of price blocks
                price_rank = sum(1 for p in prices if p < price_czk) / n if n > 0 else 0.5
                if price_rank < 0.35 and charge_possible > 0:
                    action = "charge"
                    net_value = charge_value
                    soc += (charge_possible * efficiency / battery_capacity_kwh) * 100
                    soc = min(max_soc, soc)

            if discharge_value > 0 and discharge_value > net_value:
                # Discharge is profitable
                price_rank = sum(1 for p in prices if p < price_czk) / n if n > 0 else 0.5
                if price_rank > 0.7 and discharge_possible > 0:
                    action = "discharge"
                    net_value = discharge_value
                    soc -= (discharge_kwh_per_block / battery_capacity_kwh) * 100
                    soc = max(effective_min_soc, soc)

            # Apply solar charging in hold/charge modes
            if action != "discharge" and net_solar > 0:
                solar_charge_kwh = min(
                    net_solar,
                    (battery_capacity_kwh * max_soc / 100 - battery_capacity_kwh * soc / 100)
                )
                if solar_charge_kwh > 0:
                    soc += (solar_charge_kwh * efficiency / battery_capacity_kwh) * 100
                    soc = min(max_soc, soc)

            # Apply consumption draw from battery in hold mode
            if action == "hold" and net_solar < 0:
                draw_kwh = min(
                    abs(net_solar),
                    (battery_capacity_kwh * soc / 100 - min_battery_kwh)
                )
                if draw_kwh > 0:
                    soc -= (draw_kwh / battery_capacity_kwh) * 100
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
