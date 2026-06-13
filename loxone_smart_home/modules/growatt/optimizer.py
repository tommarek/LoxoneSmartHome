"""Shared battery-scheduling infrastructure.

Home to the pieces the MILP optimizer (`milp_optimizer.py`) and the controller
both rely on: the base-load profile (training + daily EMA update), the dynamic
per-block reserve-SOC floor, the `BlockDecision` type, inverter power-rate
sizing, decision summarisation, and the shared sell-production constants.
`BatteryOptimizer` exposes only this shared infrastructure; dispatch itself is
done by the MILP optimizer.
"""

import bisect
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

    # Per-leg efficiency (sqrt of round-trip), matching the MILP SOC continuity
    # and the reserve helper (_compute_reserve_soc_per_block). Using the FULL
    # round-trip value here would under-estimate grid-side discharge energy
    # (dsoc*0.85 vs dsoc*0.92) and size the discharge powerRate too low, risking
    # not fully draining a short high-price spike the adaptive rate exists for.
    eta = max(1e-3, efficiency) ** 0.5

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
# the plan/SOC projection matches the hardware (and the controller's
# sell_production→battery_hold actuation remap) at mid-SOC.
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
    """Shared scheduling infrastructure used by the MILP optimizer and the
    controller: base-load profile training/update, the dynamic per-block
    reserve-SOC floor (`_compute_reserve_soc_per_block`), and decision
    summarisation. The MILP holds one of these as `self._helper`.
    """

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
        # same convention as the MILP SOC continuity, so the reserve floor and
        # the SOC it constrains agree.
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
            recharge_j = next_recharge_idx[block_idx + 1] if block_idx + 1 < n else n
            window_end = recharge_j if recharge_j < n else n

            # Net energy the battery must hold to cover consumption until the
            # next recharge. Work in per-BLOCK units over the SAME
            # [block_idx+1, window_end) horizon for load, solar and grid credit
            # so they can't disagree. Per block, solar first offsets the
            # concurrent base load; only the deficit draws the battery (at the
            # discharge-leg loss → /leg_eta) and only the surplus charges it (at
            # the charge-leg loss → *leg_eta).
            #
            # Accumulate a RUNNING cumulative draw and take its PEAK, rather than
            # netting all surplus against all deficit. Netting is order-blind: a
            # solar surplus LATE in the window would otherwise cancel a base-load
            # deficit that occurs EARLIER, which is physically impossible if the
            # battery is already drained before the surplus arrives. The running-
            # peak respects time order — a surplus only helps deficits that come
            # AFTER it (it reduces `running`), never an earlier one (the peak is
            # already locked in).
            running_kwh = 0.0
            peak_reserve_kwh = 0.0
            has_cheap_topup = False
            for j in range(block_idx + 1, window_end):
                ts_j = blocks[j][0]
                base_block = self._base_load_profile.get(
                    ts_j.hour, ts_j.weekday() >= 5
                ) / 4.0
                solar_block = max(0.0, _forecast_value(solar_hourly, ts_j) / 4.0)
                net = base_block - solar_block
                if net > 0:
                    running_kwh += net / leg_eta
                else:
                    running_kwh -= (-net) * leg_eta
                peak_reserve_kwh = max(peak_reserve_kwh, running_kwh)

                # A cheap-ish grid block before the recharge gives ONE top-up
                # opportunity, noted here as a single flag. Crediting a fresh
                # full charge block per cheap block would over-credit incoming
                # energy on price-flat nights, zeroing the reserve and letting
                # discharge drain the battery with nothing left for base load.
                # bisect_left on the pre-sorted prices counts strictly-cheaper
                # blocks in O(log n) without rescanning all prices for every
                # (block_idx, j) pair.
                price_rank = bisect.bisect_left(sorted_prices, prices[j]) / n if n > 0 else 0.5
                if price_rank < 0.35:
                    has_cheap_topup = True

            reserve_kwh = peak_reserve_kwh
            if has_cheap_topup:
                reserve_kwh -= kwh_per_block * leg_eta
            reserve_kwh = max(0, reserve_kwh)
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

    def summarize(self, decisions: List[BlockDecision]) -> Dict:
        """Summarize optimizer decisions for logging."""
        if not decisions:
            return {}

        charge_blocks = [d for d in decisions if d.action == "charge"]
        discharge_blocks = [d for d in decisions if d.action == "discharge"]
        hold_blocks = [d for d in decisions if d.action == "hold"]
        # Battery-hold (preserve): grid serves the house, battery idle. The MILP
        # emits this for grid-serves-load blocks with usable battery.
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
