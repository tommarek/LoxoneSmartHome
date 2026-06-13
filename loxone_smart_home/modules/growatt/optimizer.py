"""Greedy forward-simulation optimizer for battery charge/discharge scheduling.

Replaces rule-based scheduling with a block-by-block simulation that considers
price curve, solar forecast, consumption forecast, and battery constraints to
minimize total electricity cost.
"""

import bisect
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class BlockDecision:
    """Decision for a single 15-minute block."""
    timestamp: datetime
    action: str  # "charge", "discharge", "hold", "hold_idle", "sell_production"
    price_czk: float  # Spot price CZK/kWh
    distribution_czk: float  # Distribution tariff CZK/kWh
    solar_kwh: float  # Expected solar production this block
    consumption_kwh: float  # Expected consumption this block
    soc_before: float  # SOC % before action
    soc_after: float  # SOC % after action
    net_value: float  # Value of this action (positive = saves money)


def compute_rate_ceiling(hardware_max_kw: float, max_power_kw: float) -> int:
    """Adaptive powerRate% ceiling for the C-rate cap: the operational max power
    (max_power_kw) as a fraction of the hardware max (hardware_max_kw, the
    powerRate=100% reference). Single source of truth so the controller and the
    dashboard endpoints can't drift on this formula (the denominator MUST be the
    hardware max, or the actuated rate would mean a different physical power)."""
    hw = hardware_max_kw or 1.0
    return int(round(min(hardware_max_kw, max_power_kw) / hw * 100))


def compute_charge_power_rates(
    decisions: Sequence[BlockDecision],
    battery_capacity_kwh: float,
    charge_max_kw: float,
    min_power_rate: int = 25,
    action: str = "charge",
    max_power_rate: int = 100,
    efficiency: float = 0.85,
) -> Dict[datetime, int]:
    """Map each charge/discharge block → the inverter powerRate% to move the
    planned energy over its contiguous window, at the gentlest rate that fits it.

    The inverter runs a single powerRate (% of charge_max_kw) for the whole
    slot, so we size the rate to the window's *average* required power — total
    |ΔSOC| energy across the window / window duration. Long cheap/expensive
    window → gentle; short window that must move a lot → up to max_power_rate.
    Clamped to [min_power_rate, max_power_rate]; the max clamp enforces a C-rate
    ceiling (e.g. 0.5C). The rate is sized from the GRID-side power the powerRate
    actually governs (see _grid_kwh) — not raw battery ΔSOC — using `efficiency`
    to convert, so it's dimensionally consistent with the grid-side charge_max_kw
    and a co-served house load / solar co-charge doesn't skew it. Engine- and
    direction-agnostic: pass action="charge" for the charge cap or
    action="discharge" for the grid-first discharge cap (charge_max_kw is then
    the discharge max).
    """
    block_max_kwh = charge_max_kw * 0.25
    if block_max_kwh <= 0:
        return {}
    charge = sorted(
        (d for d in decisions if d.action == action), key=lambda d: d.timestamp
    )
    rates: Dict[datetime, int] = {}
    window: List[BlockDecision] = []

    eta = max(1e-3, efficiency)

    def _grid_kwh(d: BlockDecision) -> float:
        """GRID-side power the inverter's powerRate actually governs for this
        block — not the raw battery ΔSOC. charge_max_kw is a grid/AC-side figure,
        so the basis must be grid-side too. The powerRate caps TOTAL battery
        throughput (house self-consumption AND grid export both run at it — see
        milp_optimizer's discharge model, confirmed from telemetry):
        - charge: total charge INTO the battery = ΔSOC_gain / η (SOC rises at
          input×η, so input = ΔSOC/η). This sizes from TOTAL battery charge
          (AC grid + any DC solar banked the same block), symmetric with the
          telemetry-confirmed discharge case below — the powerRate caps battery
          throughput, not just the AC leg. Erring slightly fast here only fills
          a cheap window sooner, and the max_power_rate clamp keeps it within the
          C-rate ceiling; sizing from the grid leg alone would risk UNDER-filling
          the short cheap windows this feature exists to capture.
        - discharge: total battery OUTPUT = ΔSOC_drain × η (SOC drains at
          output/η). Do NOT subtract co-served house load — it counts too.
        """
        dsoc = abs(d.soc_after - d.soc_before) / 100.0 * battery_capacity_kwh
        if action == "discharge":
            return dsoc * eta
        return dsoc / eta

    def _flush(win: List[BlockDecision]) -> None:
        if not win:
            return
        energy = sum(_grid_kwh(d) for d in win)
        duration_h = len(win) * 0.25
        avg_kw = energy / duration_h if duration_h > 0 else charge_max_kw
        rate = int(math.ceil(avg_kw / charge_max_kw * 100))
        # Apply the floor first, then the ceiling — the C-rate ceiling
        # (max_power_rate) is authoritative, so a min_power_rate above it can
        # never push the actuated rate past the hardware/wear limit.
        rate = min(int(max_power_rate), max(int(min_power_rate), rate))
        for d in win:
            rates[d.timestamp] = rate

    for d in charge:
        if window and (d.timestamp - window[-1].timestamp) > timedelta(minutes=15):
            _flush(window)
            window = []
        window.append(d)
    _flush(window)
    return rates


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

# Hardware reality (SPH grid-first): pure solar export with the battery passive
# is only physical when the battery is near FULL — below this margin the inverter
# banks surplus solar instead. Mirrors milp_optimizer.SP_MIN_SOC_MARGIN_PCT so
# the greedy fallback's plan/SOC projection matches the hardware (and the
# controller's sell_production→battery_hold actuation remap) at mid-SOC.
SP_MIN_SOC_MARGIN_PCT = 2.0

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


def _resolve_local_tz(local_tz: Any) -> Any:
    """Timezone for bucketing InfluxDB's UTC record times into local keys.

    The base-load profile is consumed with LOCAL (hour, weekday/weekend)
    keys, so training records must be converted to the same wall clock
    before extracting hour/weekday. Callers that don't pass the
    controller's tz get Europe/Prague — the controller's own default —
    so the default path is correct without caller changes.
    """
    if local_tz is not None:
        return local_tz
    import zoneinfo
    return zoneinfo.ZoneInfo("Europe/Prague")


def _record_local_time(t: Any, local_tz: Any) -> Any:
    """Convert a tz-aware record timestamp to local time.

    Naive timestamps pass through unchanged (assumed already-local),
    mirroring the forecasters' ``_to_local`` convention.
    """
    if local_tz is not None and getattr(t, "tzinfo", None) is not None:
        return t.astimezone(local_tz)
    return t


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

    async def build_base_load_profile(
        self, influxdb_client: Any, solar_bucket: str, loxone_bucket: str,
        days: int = 90, local_tz: Any = None
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
            local_tz: Timezone for the profile's (hour, weekend) keys
                (None → Europe/Prague). The consumer queries the profile
                with LOCAL hours, so records must be bucketed the same way.

        Returns:
            BaseLoadProfile with 48 slots
        """
        try:
            self.logger.info("Building base load profile from historical data...")
            local_tz = _resolve_local_tz(local_tz)

            # Query heating relay activity: which hours had heating ON.
            # timeSrc:"_start" — the Flux default labels windows by their
            # STOP time, which would shift every hour key +1h.
            heating_query = f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating")
  |> filter(fn: (r) => r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false, timeSrc: "_start")
'''
            # Query EV charging: which hours had EV charging active
            ev_query = f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "ev" and r._field == "ev_charging")
  |> filter(fn: (r) => r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false, timeSrc: "_start")
'''
            heating_result = await influxdb_client.query(heating_query)
            ev_result = await influxdb_client.query(ev_query)

            # Build set of hours where heating OR EV charging was active.
            # Keys are LOCAL hours — must match the load-record keys below.
            high_load_hours: set = set()  # "YYYY-MM-DD-HH" keys
            if heating_result:
                for table in heating_result:
                    for record in table.records:
                        t = _record_local_time(record.get_time(), local_tz)
                        high_load_hours.add(t.strftime("%Y-%m-%d-%H"))
            heating_count = len(high_load_hours)

            if ev_result:
                for table in ev_result:
                    for record in table.records:
                        t = _record_local_time(record.get_time(), local_tz)
                        high_load_hours.add(t.strftime("%Y-%m-%d-%H"))
            ev_count = len(high_load_hours) - heating_count

            self.logger.debug(f"Found {heating_count} heating + {ev_count} EV hours to exclude")

            # Query house load
            load_query = f'''
from(bucket: "{solar_bucket}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> filter(fn: (r) => r._value > 0)
'''
            load_result = await influxdb_client.query(load_query)
            if not load_result:
                self.logger.warning("No load data for base load profile")
                return self._base_load_profile

            # Bin by LOCAL (hour, is_weekend), excluding heating hours.
            # Raw UTC hour/weekday would land the profile 1-2h late and
            # misclassify weekday/weekend near midnight.
            import statistics
            bins: Dict[Tuple[int, bool], List[float]] = {}
            total = 0
            excluded = 0

            for table in load_result:
                for record in table.records:
                    t = _record_local_time(record.get_time(), local_tz)
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
        self, influxdb_client: Any, solar_bucket: str, loxone_bucket: str,
        local_tz: Any = None
    ) -> None:
        """Update profile with yesterday's actual data using EMA (0.9 old + 0.1 new).

        Records are keyed by LOCAL (hour, weekend) — same convention as
        build_base_load_profile (local_tz None → Europe/Prague).
        """
        try:
            local_tz = _resolve_local_tz(local_tz)
            # Query yesterday's heating + EV hours (start-labeled windows —
            # the Flux default stop-labeling would shift hour keys +1h)
            heating_result = await influxdb_client.query(f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating" and r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false, timeSrc: "_start")
''')
            ev_result = await influxdb_client.query(f'''
from(bucket: "{loxone_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "ev" and r._field == "ev_charging" and r._value == 1)
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false, timeSrc: "_start")
''')
            high_load_hours: set = set()
            if heating_result:
                for table in heating_result:
                    for record in table.records:
                        t = _record_local_time(record.get_time(), local_tz)
                        high_load_hours.add(t.strftime("%Y-%m-%d-%H"))
            if ev_result:
                for table in ev_result:
                    for record in table.records:
                        t = _record_local_time(record.get_time(), local_tz)
                        high_load_hours.add(t.strftime("%Y-%m-%d-%H"))

            # Query yesterday's load
            load_result = await influxdb_client.query(f'''
from(bucket: "{solar_bucket}")
  |> range(start: -1d, stop: -0d)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "INVPowerToLocalLoad")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false, timeSrc: "_start")
  |> filter(fn: (r) => r._value > 0)
''')
            if not load_result:
                return

            for table in load_result:
                for record in table.records:
                    t = _record_local_time(record.get_time(), local_tz)
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

        # Per-leg loss factor: `efficiency` is ROUND-TRIP, so each physical
        # conversion leg (charge OR discharge) costs sqrt(efficiency) — the
        # same convention as the MILP SOC continuity and the greedy SOC
        # simulation, so the reserve floor and the SOC it constrains agree.
        leg_eta = max(1e-3, efficiency) ** 0.5

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
            # (drawn at the discharge-leg loss → /leg_eta) and only the surplus
            # charges it (at the charge-leg loss → *leg_eta). This avoids the old
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
                    gross_reserve_kwh += net / leg_eta
                else:
                    incoming_kwh += (-net) * leg_eta

                # A cheap-ish grid block before the recharge gives ONE top-up
                # opportunity. Only note it here — crediting a fresh full charge
                # block per cheap block (the old behaviour) over-credited
                # incoming on price-flat nights, zeroing the reserve and letting
                # discharge drain the battery with nothing left for base load.
                # bisect_left on the pre-sorted prices counts strictly-cheaper
                # blocks in O(log n) — identical to the old O(n) rescan but
                # without rescanning all prices for every (block_idx, j) pair.
                price_rank = bisect.bisect_left(sorted_prices, prices[j]) / n if n > 0 else 0.5
                if price_rank < 0.35:
                    has_cheap_topup = True

            if has_cheap_topup:
                incoming_kwh += kwh_per_block * leg_eta

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
        battery_amortisation_export_czk: Optional[float] = None,  # export-only wear
        export_price_min: Optional[float] = None,  # strict export floor (CZK/kWh)
        inverter_off_price: Optional[float] = None,  # strict PV-off price (CZK/kWh)
        sell_production_min_soc_margin: float = SP_MIN_SOC_MARGIN_PCT,
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

        # Clamp SOC at the TOP only. Live telemetry can report e.g. 100% while
        # max_soc is 90 (documented real condition); an unclamped value makes
        # battery_gap_kwh go negative. A BELOW-min_soc reading is PRESERVED:
        # clamping it up would credit phantom stored energy. The simulation is
        # safe with soc < min_soc — every battery draw floors its available
        # energy at max(0, battery_kwh - min_battery_kwh) and the discharge
        # gate rejects non-positive discharge_possible, so the plan simply
        # cannot spend battery until the SOC recovers above the floor. (The
        # MILP keeps the two-sided clamp for LP feasibility and logs instead.)
        current_soc = min(max_soc, current_soc)

        # Battery energy parameters. discharge_rate_kw is the ACTUAL grid-discharge
        # power at the configured discharge_power_rate (~2.5 kW at 25% on this
        # inverter), so it is used directly — symmetric with charge_rate_kw. Do
        # NOT multiply by discharge_power_pct again: that double-counted the power
        # rate and modelled discharge 4x too slow, so the projected SOC barely
        # dropped while the real battery fell ~2.5 kW.
        kwh_per_block = charge_rate_kw * 0.25  # 15 minutes
        discharge_kwh_per_block = discharge_rate_kw * 0.25

        # Per-leg loss factor: `efficiency` is ROUND-TRIP, so each physical
        # conversion leg (charge OR discharge) costs sqrt(efficiency). The SOC
        # simulation applies this per leg — same convention as the MILP's SOC
        # continuity (eta_chg * eta_dis == efficiency). Economic VALUE terms
        # that price a full grid→battery→load replacement cycle keep the
        # round-trip `efficiency` (e.g. recharge_cost) — those model two legs.
        leg_eta = max(1e-3, efficiency) ** 0.5

        # Wear cost charged on battery→GRID export. Defaults to the shared wear
        # cost; an explicit export override raises the hurdle for arbitrage
        # cycling only (battery→house self-consumption keeps the base cost). When
        # unset this equals battery_amortisation_czk, so output is unchanged.
        amort_export = (
            battery_amortisation_czk
            if battery_amortisation_export_czk is None
            else battery_amortisation_export_czk
        )

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

        # Terminal value of leftover SOC at horizon end — mirrors the MILP's
        # terminal valuation (milp_optimizer.py) exactly so the engines stay in
        # parity: retained energy is worth the median self-consumption saving
        # (spot + dist - amort), capped at the cheapest grid-charge break-even
        # over the FULL round-trip efficiency. The cap uses round-trip on
        # purpose (not the per-leg eta): the break-even for "charge cheap now,
        # use later" is the whole charge→store→discharge cycle. The 0.99
        # discount makes a CERTAIN present saving win ties against this
        # estimated future value. Without it, leftover SOC is worth 0 and the
        # plan dumps the battery into the last evening blocks before tomorrow's
        # DAM prices arrive.
        sc_values = sorted(
            prices[i] + distribution_func(blocks[i][0].hour)
            - battery_amortisation_czk
            for i in range(n)
        )
        terminal_value_per_kwh = (
            max(0.0, sc_values[n // 2]) if sc_values else 0.0
        )
        min_charge_cost = min(
            prices[i] + distribution_func(blocks[i][0].hour) for i in range(n)
        )
        terminal_value_per_kwh = max(0.0, min(
            terminal_value_per_kwh, min_charge_cost / max(1e-3, efficiency)
        ))
        terminal_value_per_kwh *= 0.99

        # Future best discharge value: for each block, what's the maximum
        # discharge profit achievable in remaining blocks? Used to avoid
        # spending battery on mediocre blocks when better ones are ahead.
        # NOT floored at the terminal value: entries here are NET arbitrage
        # profits (sell_rev - recharge_cost; the battery ends refilled), while
        # the terminal value is the GROSS worth of a retained kWh — mixing the
        # two conventions over-blocked genuinely profitable mid-horizon
        # arbitrage (a dv>0 discharge with a real cheap recharge ahead pockets
        # dv AND preserves the terminal energy — strictly better than
        # holding). End-of-horizon dumps are prevented directly in the
        # discharge decision instead: the GROSS sell value per delivered kWh
        # must beat terminal_value_per_kwh (selling must beat retaining).
        future_max_discharge_value = [0.0] * n
        running_max = float('-inf')
        for i in range(n - 1, -1, -1):
            dist_i = distribution_func(blocks[i][0].hour)
            # Export pays no distribution — only the sell fee (+ battery wear).
            sell_rev_i = prices[i] - sell_fee_czk - amort_export
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
        # Floored at the terminal value: even past the last in-horizon
        # consumption, retained energy is worth terminal_value_per_kwh (it
        # offsets tomorrow's imports), so late blocks never see a 0 hold value.
        future_sc_value = [0.0] * n
        # Strictly-future variant: best SC value AFTER block i (block i itself
        # EXCLUDED). future_sc_value[i] folds in block i before assignment, so
        # it cannot distinguish "the peak is now" from "the peak is later" —
        # the hold_idle remap needs that distinction (idling THROUGH the peak
        # the energy was retained for strands the battery forever on a rolling
        # horizon, which always contains some future peak).
        future_sc_value_excl = [0.0] * n
        future_sc_kwh = [0.0] * n  # Cumulative kWh needing battery from block i onward
        running_best_sc = terminal_value_per_kwh
        running_sc_kwh = 0.0
        for i in range(n - 1, -1, -1):
            ts_i = blocks[i][0]
            h_i = ts_i.hour
            solar_i = _forecast_value(solar_hourly, ts_i) / 4.0
            cons_i = _forecast_value(consumption_hourly, ts_i) / 4.0
            # Same base-load fallback _single_pass applies per block: the
            # VALUE model must see the same house load the SOC simulation
            # serves, or a sparse consumption forecast would zero out the
            # future-SC values (no "deficits") while the simulation still
            # drains the battery for base load — and the forward-looking
            # charge gate below would then refuse plainly profitable charging.
            if cons_i <= 0:
                cons_i = self._base_load_profile.get(
                    h_i, ts_i.weekday() >= 5
                ) / 4.0
            net_cons = max(0, cons_i - solar_i)
            future_sc_value_excl[i] = running_best_sc  # before folding in block i
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

        # Energy-aware retention need for the hold_idle remap: for block i,
        # the DELIVERED-side kWh of strictly-better future deficits — the sum
        # over j > i of deficit kWh at blocks whose per-delivered SC value
        # exceeds block i's. The remap may only idle block i while the usable
        # stored energy cannot cover that need; a quantity-blind remap (any
        # strictly-better block ⇒ idle) imported today's peak from grid even
        # when the battery held enough for today AND tomorrow, and on a
        # rolling horizon (always another, slightly-better peak ahead) it
        # pinned the battery at max SOC forever. O(n²) like the reserve
        # helper — n ≈ 128-200, negligible.
        block_sc_value = [0.0] * n
        block_deficit_kwh = [0.0] * n
        for i in range(n):
            ts_i = blocks[i][0]
            solar_i = _forecast_value(solar_hourly, ts_i) / 4.0
            cons_i = _forecast_value(consumption_hourly, ts_i) / 4.0
            # Same base-load fallback as future_sc_value above — the two
            # retention arrays must agree on what counts as a deficit.
            if cons_i <= 0:
                cons_i = self._base_load_profile.get(
                    ts_i.hour, ts_i.weekday() >= 5
                ) / 4.0
            block_deficit_kwh[i] = max(0.0, cons_i - solar_i)
            block_sc_value[i] = (
                prices[i] + distribution_func(ts_i.hour)
                - battery_amortisation_czk
            )
        strictly_better_sc_need = [0.0] * n
        for i in range(n):
            v_i = block_sc_value[i]
            need = 0.0
            for j in range(i + 1, n):
                if block_deficit_kwh[j] > 0 and block_sc_value[j] > v_i:
                    need += block_deficit_kwh[j]
            strictly_better_sc_need[i] = need

        # Pre-select charge blocks: pick cheapest N blocks where charging is
        # profitable vs buying from grid later. Don't charge at 1.28 CZK when
        # -2.0 blocks are available — better to sit at min SOC and buy from grid.
        #
        # The break-even compares the FULL import cost (spot + distribution)
        # against the FORWARD-LOOKING value of the charged energy: charging 1
        # grid kWh delivers `efficiency` kWh later, each worth the best future
        # deficit self-consumption value (future_sc_value, per-delivered-kWh,
        # amortisation-adjusted, terminal-floored). The previous gate valued a
        # charged kWh at median_import_cost * efficiency, which fails whenever
        # cheap blocks are the horizon MAJORITY (negative/windy days, cheap-NT
        # tariff bands): the median sits in the cheap mode, cheap*efficiency <
        # cheap, zero candidates — and the `p < 0` fallback was dead because
        # the actuation gate used the same algebra. Valuing against the best
        # future deficit keeps the iteration-1 property (distribution priced
        # on both sides — high-dist/low-spread days still don't cycle) AND is
        # wear-consistent: future_sc_value subtracts battery_amortisation_czk,
        # so cycling that is cash-positive but wear-negative no longer charges.
        #
        # Candidate pre-selection is POSITION-INDEPENDENT on purpose: it uses
        # the permissive bound future_sc_value[0] (best SC value anywhere in
        # the horizon); the per-block actuation gate in _single_pass
        # (import_cost < future_sc_value[i] * efficiency) does the real
        # position-aware vetting. The "or p < 0" retention keeps negative-spot
        # blocks in the pool even above the bound — negative FULL cost always
        # actuates (future_sc_value >= 0), positive full cost is re-vetted.
        import_costs = [
            prices[i] + distribution_func(blocks[i][0].hour) for i in range(n)
        ]
        # Best charged-kWh value anywhere in the horizon (CZK per GRID kWh).
        max_charge_value = (future_sc_value[0] if n > 0 else 0.0) * efficiency
        profitable_charge_blocks = [
            (blocks[i][0], import_costs[i])
            for i in range(n)
            if import_costs[i] < max_charge_value or prices[i] < 0
        ]
        profitable_charge_blocks.sort(key=lambda x: x[1])  # Cheapest full cost first

        # Calculate how much grid charging we actually need:
        # Total battery gap minus expected net solar charging (solar - consumption)
        battery_gap_kwh = battery_capacity_kwh * (max_soc - current_soc) / 100
        expected_solar_charge = 0.0
        for ts, _ in blocks:
            solar_h = _forecast_value(solar_hourly, ts) / 4.0  # kWh per 15-min block
            consumption_h = _forecast_value(consumption_hourly, ts) / 4.0
            net = solar_h - consumption_h
            if net > 0:
                # Single charge leg (solar→battery banking): per-leg eta, so
                # selection sizing matches what the SOC simulation will bank.
                expected_solar_charge += net * leg_eta
        grid_needed_kwh = max(0, battery_gap_kwh - expected_solar_charge)

        # Convert to blocks needed (min 4 to always grab the cheapest/negative
        # prices). Grid→battery is one charge leg → per-leg eta, matching the
        # simulation's per-block SOC gain.
        kwh_per_charge = kwh_per_block * leg_eta
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
            future_sc_value_excl, strictly_better_sc_need,
            sorted_prices, sell_fee_czk, battery_amortisation_czk,
            first_block_ts, sell_production_min_soc_margin, amort_export,
            terminal_value_per_kwh,
        )

        # Pass 2: refine charge block selection using actual SOC trajectory
        refined_set = self._refine_charge_blocks(
            blocks, charge_block_set, decisions, max_soc, max_charge_value
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
                future_sc_value_excl, strictly_better_sc_need,
                sorted_prices, sell_fee_czk, battery_amortisation_czk,
                first_block_ts, sell_production_min_soc_margin, amort_export,
                terminal_value_per_kwh,
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

            sell_now           = spot - fee                         # solar→grid revenue (no dist)
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

        # Single charge-leg loss for solar→battery banking credits (the
        # grid_replacement value below keeps round-trip `efficiency` — that
        # one prices a full grid→battery→load replacement cycle).
        leg_eta = max(1e-3, efficiency) ** 0.5

        max_battery_kwh = battery_capacity_kwh * max_soc / 100
        current_battery_kwh = battery_capacity_kwh * current_soc / 100
        battery_gap_remaining = max(0.0, max_battery_kwh - current_battery_kwh)

        # Per-block sell-now revenue and solar excess
        sell_now = [0.0] * n
        solar_excess = [0.0] * n
        for i, (ts, p) in enumerate(blocks):
            h = ts.hour
            # Export pays no distribution — only the sell fee.
            sell_now[i] = p - sell_fee_czk
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
            # Solar surplus banks into the battery over ONE charge leg.
            solar_only_refills = (
                future_solar_surplus[i + 1] * leg_eta >= battery_gap_remaining
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

        # Aggregate refill budget: cheap grid charging capacity + future solar.
        # Both are single charge legs (grid→battery / solar→battery), so the
        # banked energy is input × leg_eta — round-trip here under-stated the
        # budget by another sqrt(efficiency).
        cheap_charge_kwh = sum(
            kwh_per_block * leg_eta
            for ts, _ in blocks if ts in charge_block_set
        )
        refill_budget = cheap_charge_kwh + future_solar_surplus[0] * leg_eta

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
        max_charge_value: float,
    ) -> Set[datetime]:
        """Identify wasted charge blocks and replace with better candidates.

        A charge block is "wasted" if the battery was nearly full (from solar)
        and the charge action barely increased SOC. These wasted slots are
        replaced with the cheapest available hold blocks that had room to charge.

        Replacements MUST still pass the candidate gate (FULL import cost —
        spot + distribution — below `max_charge_value`, or negative spot) —
        otherwise the refine step could inject a hopeless grid charge just to
        "use up" a freed slot. `max_charge_value` is the PERMISSIVE bound from
        optimize() (best future SC value anywhere in the horizon × efficiency,
        i.e. future_sc_value[0] * efficiency) — same convention as the primary
        candidate pre-selection; the Pass-2 re-simulation's per-block
        actuation gate (import_cost < future_sc_value[i] * efficiency) does
        the position-aware vetting, so a replacement that lands after the
        last valuable deficit simply falls through to hold.

        Returns the original set if no improvement is possible.
        """
        # Find wasted charge blocks: SOC barely changed (< 1%)
        # Never consider negative-price blocks as wasted. SPOT (not full cost)
        # is intentional here: at negative spot the meter pays us per imported
        # kWh, so keeping the slot is at worst a no-op (the battery is full —
        # nothing imports) and at best free money if the SOC trajectory shifts
        # in the Pass-2 re-simulation; the per-block actuation gate re-vets
        # the block's full-cost profitability either way.
        wasted = {
            d.timestamp for d in decisions
            if d.action == "charge"
            and (d.soc_after - d.soc_before) < 1.0
            and d.price_czk >= 0
        }
        if not wasted:
            return original_charge_set

        # Find candidate replacement blocks: hold blocks with room to charge
        # AND that pass the candidate gate (FULL import cost below the
        # permissive forward-looking bound, or negative spot) — same
        # convention as the primary selection in optimize().
        candidates = sorted(
            [(d.timestamp, d.price_czk + d.distribution_czk) for d in decisions
             if d.action == "hold" and d.soc_before < max_soc - 5
             and (d.price_czk + d.distribution_czk < max_charge_value
                  or d.price_czk < 0)],
            key=lambda x: x[1]  # Cheapest full cost first
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
        future_sc_value_excl: List[float],
        strictly_better_sc_need: List[float],
        sorted_prices: List[float],
        sell_fee_czk: float,
        battery_amortisation_czk: float,
        first_block_ts: datetime,
        sell_production_min_soc_margin: float = SP_MIN_SOC_MARGIN_PCT,
        amort_export: Optional[float] = None,
        terminal_value_per_kwh: float = 0.0,
    ) -> Tuple[Set[datetime], Set[datetime], Set[datetime], List[BlockDecision]]:
        """Run a single forward simulation pass with given charge block set."""
        n = len(blocks)
        # Export-only wear cost (battery→grid); falls back to the shared cost so
        # an unset override leaves the self-consumption vs export trade unchanged.
        if amort_export is None:
            amort_export = battery_amortisation_czk
        # Per-leg loss factor: `efficiency` is ROUND-TRIP; the SOC simulation
        # applies sqrt(efficiency) per physical conversion leg (charge OR
        # discharge), matching the MILP's eta. Value terms that price a full
        # replacement round trip (recharge_cost, future_value) keep `efficiency`.
        leg_eta = max(1e-3, efficiency) ** 0.5
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
            # When a block has no consumption forecast, fall back to the learned
            # base load ONCE here so the action SCORING (net_solar below) and the
            # SOC SIMULATION further down use the SAME house load — otherwise
            # scoring would bank all solar as excess while the sim still drains
            # the battery for base load. Weekday/weekend keys off THIS block.
            if consumption <= 0:
                consumption = self._base_load_profile.get(
                    hour, timestamp.weekday() >= 5
                ) / 4.0

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
            # FORWARD-LOOKING full-cost economics: charging 1 grid kWh at
            # (spot + distribution) banks energy that delivers `efficiency`
            # kWh later (round trip), each worth the best REMAINING deficit
            # self-consumption value — future_sc_value[i], per delivered kWh,
            # already amortisation-adjusted and terminal-floored. So charging
            # is profitable iff import_cost < future_sc_value[i] * efficiency.
            # The previous benchmark (median import cost × efficiency) broke
            # on cheap-majority horizons: the median sat in the cheap mode and
            # cheap*efficiency < cheap vetoed EVERY charge, including ahead of
            # an expensive evening peak (greedy/MILP parity loss). Position-
            # aware: past the last valuable deficit, future_sc_value falls to
            # the terminal value, whose cap (min import cost / efficiency)
            # guarantees terminal*efficiency < every import cost — trailing
            # blocks can never charge "for the terminal value" alone.
            # WEAR NOTE: unlike the pre-iteration-2 gate, this one inherits
            # the amortisation subtraction inside future_sc_value, so a cycle
            # that is cash-positive but wear-negative (spread smaller than the
            # battery wear cost) is correctly refused.
            charge_cost = price_czk + dist
            future_value = (
                future_sc_value[i] if i < len(future_sc_value)
                else terminal_value_per_kwh
            ) * efficiency
            charge_possible = min(kwh_per_block, (max_battery_kwh - battery_kwh))
            charge_value = (future_value - charge_cost) if charge_possible > 0 else float('-inf')

            # --- DISCHARGE value ---
            # sell_revenue already accounts for amortisation. Hard floor: never
            # discharge to grid when the sale itself is unprofitable. Round-trip
            # arbitrage with hypothetical future recharge does not justify
            # destroying value now — every discharged kWh wears the battery.
            # Export pays no distribution — only the sell fee (+ battery wear).
            sell_revenue = price_czk - sell_fee_czk - amort_export
            future_recharge_hour = future_min_price_hour[i] if i < n else 0
            future_dist = distribution_func(future_recharge_hour)
            recharge_cost = (future_cheapest + future_dist) / efficiency
            discharge_profit = sell_revenue - recharge_cost
            discharge_possible = min(
                discharge_kwh_per_block,
                # Stored headroom → grid-delivered energy is ONE discharge leg.
                (battery_kwh - min_battery_kwh) * leg_eta
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
            retention_hold = False
            if usable_kwh > 0 and i < n:
                best_future_sc = future_sc_value[i]
                # Delivered → stored kWh is ONE discharge leg.
                sc_kwh_needed = future_sc_kwh[i] / leg_eta
                if best_future_sc > recharge_cost and usable_kwh <= sc_kwh_needed:
                    # Battery is fully needed for self-consumption — don't discharge
                    retention_value = best_future_sc - recharge_cost
                    if retention_value > hold_value:
                        hold_value = retention_value
                        retention_hold = True

            # Pick best action
            soc_before = soc
            action = "hold"
            net_value = hold_value

            # Actuation gate: a pre-selected charge block must ALSO have a
            # positive charge_value at simulation time. The candidate set can
            # contain blocks the value model rejects (the `p < 0` fallback on
            # deeply-negative days where high distribution still makes the
            # full import cost exceed the future value, e.g. spot -0.1 + dist
            # 2.5 vs a low median import); previously such blocks charged
            # anyway, locking in the loss the engine itself had computed.
            # Skipping is always safe: nothing in the plan REQUIRES charging
            # — the dynamic reserve (effective_min_socs) only floors how far
            # discharge/hold may DRAIN existing energy, it never mandates a
            # grid charge, so a skipped block simply falls through to
            # hold/discharge scoring and the deficit imports from grid.
            if (
                timestamp in charge_block_set
                and charge_possible > 0
                and charge_value > 0
            ):
                action = "charge"
                net_value = charge_value

            best_remaining = future_max_discharge_value[i] if i < n else 0
            is_worthwhile = (best_remaining <= 0) or (discharge_value >= best_remaining * 0.8)
            # Selling must beat RETAINING (gross per-delivered-kWh on both
            # sides). NOTE: with the current terminal cap (min charge cost /
            # round-trip eff x0.99) this gate is implied by
            # `discharge_value > 0` — recharge_cost >= min charge cost /
            # efficiency > terminal — so it is belt-and-braces, NOT the
            # active dump protection. It only binds if the terminal-value or
            # recharge-cost conventions ever drift apart. (The old terminal
            # FLOOR on future_max_discharge_value compared a gross retained
            # value against NET arbitrage profits and wrongly blocked
            # profitable mid-horizon discharges with a cheap recharge ahead.)
            sell_beats_retention = sell_revenue > terminal_value_per_kwh
            if (
                discharge_value > 0 and discharge_value > net_value
                and is_worthwhile and sell_beats_retention
            ):
                if discharge_possible > 0 and soc > effective_min_soc:
                    action = "discharge"
                    net_value = discharge_value

            # Sell-production wins over charge/hold but never displaces a
            # profitable discharge — discharge is the stronger variant: it
            # sells battery AND solar excess. Sell_production only sells
            # solar excess (battery passive). So when discharge fires, it
            # already covers what sell_production would do.
            # sell_production is actuated as grid-first @ stop_soc=live SOC (battery
            # passive), so the surplus exports at ANY SOC — no near-full gate.
            if timestamp in sell_production_set and action != "discharge":
                action = "sell_production"
                # Informational value: sell-now revenue × solar excess this block
                # (export pays no distribution — only the sell fee).
                solar_excess_now = max(0.0, solar - consumption)
                net_value = (price_czk - sell_fee_czk) * solar_excess_now

            # On this SPH a plain "hold" actuates as load_first, which DRAINS
            # the battery into any consumption deficit. When the hold's value
            # came from RETAINING energy for future self-consumption, that
            # drain is exactly what the plan wants to avoid — so mark deficit
            # blocks above the reserve as "hold_idle" (actuated as battery_hold:
            # battery passive, deficit served from grid), same action string the
            # MILP emits and the controller already maps to battery_hold.
            #
            # Only idle when retaining is STRICTLY better than spending now:
            # compare the best STRICTLY-future SC value (future_sc_value_excl,
            # which excludes block i) against THIS block's self-consumption
            # value — both per delivered kWh, amort-adjusted (price + dist -
            # amortisation), so the comparison is convention-consistent.
            # Gating on future_sc_value[i] instead would idle the battery
            # straight through its own peak (the inclusive array counts block
            # i as "future"), stranding the energy forever on a rolling
            # horizon that always has another peak ahead. With the strict
            # comparison, peak blocks (now == best) self-consume via plain
            # hold and only genuinely-cheaper blocks idle.
            #
            # AND only while the battery actually NEEDS all its energy for
            # those strictly-better future blocks (quantity gate): a value-
            # only comparison idled the current peak whenever ANY marginally
            # better block existed ahead, even with enough energy for both —
            # importing today's peak at full price and, on a rolling horizon
            # with drifting peaks, pinning the battery at max SOC forever.
            # strictly_better_sc_need is DELIVERED-side kWh; usable_kwh is
            # STORED-side — one discharge leg (/leg_eta) converts, applied
            # exactly once. Once usable energy exceeds the strictly-better
            # need, the surplus is spent NOW via plain hold (load_first).
            strictly_future_sc = (
                future_sc_value_excl[i]
                if i < len(future_sc_value_excl) else 0.0
            )
            strictly_better_need_kwh = (
                strictly_better_sc_need[i]
                if i < len(strictly_better_sc_need) else 0.0
            )
            # AND only when spending now cannot be profitably replaced by a
            # future recharge (value gate #2): when self-consuming this block
            # is worth MORE than the cheapest future refill costs, spend-now
            # + refill strictly dominates idling — regardless of how much the
            # strictly-better future "needs". Without this, the quantity gate
            # degenerates whenever future need >= usable energy (the common
            # winter regime: tomorrow's marginally-better peak "needs" all of
            # it, today idles, and on a rolling horizon with upward-drifting
            # peaks the battery pins at max SOC forever).
            if (
                action == "hold" and retention_hold
                and net_solar < 0 and soc > effective_min_soc
                and strictly_future_sc > self_consumption_value
                and self_consumption_value < recharge_cost
                and usable_kwh <= strictly_better_need_kwh / leg_eta
            ):
                action = "hold_idle"

            # === SOC SIMULATION ===
            # `consumption` already carries the base-load fallback applied above,
            # so the simulated house load matches the scored net_solar exactly.
            solar_available = solar
            house_load = consumption
            net_from_solar = solar_available - house_load

            if action == "charge":
                grid_charge = kwh_per_block
                soc += (grid_charge * leg_eta) / battery_capacity_kwh * 100
                if net_from_solar > 0:
                    # Remaining headroom must exclude the grid charge just added,
                    # else solar banking double-spends capacity and over-states
                    # soc_after before the clamp.
                    solar_to_batt = min(
                        net_from_solar * leg_eta,
                        max_battery_kwh - (battery_kwh + grid_charge * leg_eta),
                    )
                    soc += max(0.0, solar_to_batt) / battery_capacity_kwh * 100
                soc = min(max_soc, soc)
            elif action == "discharge":
                # `discharge_possible` is the GRID-DELIVERED energy and already
                # respects the per-block dynamic reserve (battery_kwh-min)*leg,
                # so it can't push SOC below effective_min_soc. Delivering that
                # to the grid drains battery_delivered/leg_eta of stored energy
                # (one discharge leg). Mirrors the MILP, which drains battery by
                # grid_export / eta with eta = sqrt(round-trip efficiency).
                battery_drain = discharge_possible / leg_eta
                if net_from_solar < 0:
                    # The house deficit is ALSO served from the battery. Both the
                    # grid-export drain and the deficit drain draw from the SAME
                    # reserve headroom (battery_kwh - min_battery_kwh), so cap the
                    # export drain against what's left after the deficit instead of
                    # letting each independently drain to the floor (which the SOC
                    # clamp would silently absorb — an energy-conservation leak).
                    deficit_drain = abs(net_from_solar) / leg_eta
                    avail = max(0.0, battery_kwh - min_battery_kwh)
                    battery_drain = min(
                        battery_drain, max(0.0, avail - deficit_drain)
                    ) + deficit_drain
                soc -= battery_drain / battery_capacity_kwh * 100
                soc = max(effective_min_soc, soc)
            elif action == "sell_production":
                # Battery-PASSIVE on this SPH: sell_production actuates as
                # grid-first with stop_soc pinned to the live SOC, so the
                # battery neither banks the surplus (it exports) nor serves a
                # consumption deficit (the grid does). SOC carries over
                # unchanged — matching the MILP (batt_to_load forbidden on
                # is_sp blocks) and the backtest harness, which both model SP
                # deficits as grid-served. Draining the battery here projected
                # SOC the hardware would never reach.
                pass
            elif action == "hold_idle":
                # Battery-passive preserve: actuated as battery_hold
                # (battery_first with stop_soc pinned to the live SOC), so the
                # battery neither discharges into the deficit (grid serves it)
                # nor grid-charges. SOC carries over unchanged. Only deficit
                # blocks are remapped to hold_idle, so there is no surplus
                # solar to bank here.
                pass
            else:  # hold
                if net_from_solar > 0:
                    solar_to_batt = min(net_from_solar * leg_eta,
                                        max_battery_kwh - battery_kwh)
                    if solar_to_batt > 0:
                        soc += solar_to_batt / battery_capacity_kwh * 100
                        soc = min(max_soc, soc)
                else:
                    # Pure self-consumption: serve the deficit from the battery
                    # down to the per-block DYNAMIC reserve (not the hardware
                    # floor) so holding preserves overnight energy. Battery-side
                    # draw is one lossy discharge leg, matching the MILP.
                    need = abs(net_from_solar) / leg_eta
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
        # Battery-hold (preserve): grid serves the house, battery idle. Both
        # engines emit this — the MILP for grid-serves-load blocks with usable
        # battery, the greedy for retention holds with a consumption deficit.
        hold_idle_blocks = [d for d in decisions if d.action == "hold_idle"]
        sell_production_blocks = [d for d in decisions if d.action == "sell_production"]

        return {
            "total_blocks": len(decisions),
            "sell_production_blocks": len(sell_production_blocks),
            "charge_blocks": len(charge_blocks),
            "discharge_blocks": len(discharge_blocks),
            "hold_blocks": len(hold_blocks),
            "hold_idle_blocks": len(hold_idle_blocks),
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
