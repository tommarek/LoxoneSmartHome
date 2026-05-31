"""Deferrable load scheduling.

A deferrable load is a controllable appliance that:
- Has a known/estimated total energy requirement
- Can be run inside a permitted time window (e.g. 06:00-22:00)
- Should be scheduled inside the cheapest blocks within that window

EV charging is the prototypical example: "deliver N kWh to the car
between when I plug in and when I need to leave, at the cheapest hours."
Heat-pump hot-water boost, dishwasher delay-start, washing-machine
night-mode are all the same shape.

This module is a planning/scheduling subsystem. Actual on/off control
is delegated to whatever interface the device exposes (typically a
Loxone relay published over MQTT). We just compute when to enable it
and the integration layer fires the command.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class DeferrableLoad:
    """Specification of a controllable load that can be time-shifted."""

    name: str  # human-readable, e.g. "ev_charger", "dishwasher"
    energy_required_kwh: float  # total energy to deliver, in kWh
    power_kw: float  # nominal draw when active (single-power loads only)
    earliest_start: time  # not-before window edge (local time)
    latest_end: time  # must-finish-by edge (local time)
    interruptible: bool = True  # if False, must run contiguously
    mqtt_topic_on: Optional[str] = None  # topic to publish "on" command
    mqtt_topic_off: Optional[str] = None  # topic to publish "off" command
    # Optional payloads — if None, simple value=1/value=0 JSON is used.
    payload_on: Optional[Any] = None
    payload_off: Optional[Any] = None

    def required_blocks(self, block_minutes: int = 15) -> int:
        """Number of 15-min blocks at nominal power to deliver the required kWh."""
        if self.power_kw <= 0:
            return 0
        kwh_per_block = self.power_kw * (block_minutes / 60.0)
        # Round UP so we always deliver at least the requested energy.
        return max(1, int((self.energy_required_kwh / kwh_per_block) + 0.999))

    def in_window(self, t: time) -> bool:
        """True when local time `t` is inside the load's allowed window.

        Windows may cross midnight (e.g. an EV charged 22:00→06:00): when
        ``earliest_start > latest_end`` the window wraps, so a time is
        in-window if it's at/after the start OR before the end. Shared by the
        greedy scheduler and the MILP co-optimizer so both agree on membership.
        """
        if self.earliest_start <= self.latest_end:
            return self.earliest_start <= t < self.latest_end
        return t >= self.earliest_start or t < self.latest_end


@dataclass
class DeferrableLoadSchedule:
    """Output of the planner: which 15-min blocks each load should run in."""

    load_name: str
    blocks: List[Tuple[str, str]] = field(default_factory=list)  # ("HH:MM", "HH:MM")
    block_datetimes: List[datetime] = field(default_factory=list)
    expected_cost_czk: float = 0.0  # total cost to run during these blocks
    naive_cost_czk: float = 0.0  # cost if run as soon as possible (for comparison)
    requested_blocks: int = 0  # blocks needed to fully deliver energy_required
    scheduled_blocks: int = 0  # blocks actually scheduled (may be < requested)
    energy_shortfall_kwh: float = 0.0  # unmet energy when window too narrow

    @property
    def savings_czk(self) -> float:
        return self.naive_cost_czk - self.expected_cost_czk

    @property
    def fully_scheduled(self) -> bool:
        """True when the whole requested energy fit inside the window."""
        return self.scheduled_blocks >= self.requested_blocks > 0


class DeferrableLoadScheduler:
    """Pick the cheapest N blocks in a load's allowed window.

    Uses the SAME 15-min price grid the battery optimizer sees, so prices
    and units are consistent. Does NOT interact with the battery model
    directly — the load is treated as additional consumption during its
    scheduled blocks and the battery optimizer's consumption_hourly is
    augmented accordingly before optimization.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def schedule(
        self,
        load: DeferrableLoad,
        price_blocks: List[Tuple[datetime, float]],
        distribution_func: Callable[[int], float],
        sell_fee_czk: float = 0.0,
    ) -> DeferrableLoadSchedule:
        """Pick the cheapest blocks within the load's allowed window.

        Args:
            load: the deferrable load specification
            price_blocks: [(timestamp, spot_czk_per_kwh)] sorted ascending
            distribution_func: callable(hour) → distribution tariff CZK/kWh
            sell_fee_czk: not used here (loads buy, don't sell) — kept for
                future symmetry

        Returns:
            DeferrableLoadSchedule with the chosen 15-min block windows.
        """
        n_needed = load.required_blocks()
        requested = n_needed
        if n_needed <= 0:
            return DeferrableLoadSchedule(load_name=load.name)

        # Filter to blocks within the allowed local-time window (DeferrableLoad
        # handles midnight-wrapping windows, e.g. an EV charged 22:00→06:00).
        candidates: List[Tuple[int, datetime, float]] = []  # (idx, ts, cost_per_kwh)
        for idx, (ts, spot) in enumerate(price_blocks):
            if not load.in_window(ts.time()):
                continue
            # Cost per kWh delivered = spot + distribution. Sell fee NA.
            cost_per_kwh = spot + distribution_func(ts.hour)
            candidates.append((idx, ts, cost_per_kwh))

        if len(candidates) < n_needed:
            # Window too narrow to fit the load — schedule what we can,
            # cheapest first within the window, accept the partial run.
            self.logger.warning(
                f"Deferrable load {load.name!r}: needs {n_needed} blocks "
                f"but only {len(candidates)} available in window "
                f"{load.earliest_start}-{load.latest_end} — scheduling partial"
            )
            n_needed = len(candidates)
            if n_needed == 0:
                kwh_per_block = load.power_kw * 0.25
                return DeferrableLoadSchedule(
                    load_name=load.name,
                    requested_blocks=requested,
                    scheduled_blocks=0,
                    energy_shortfall_kwh=round(requested * kwh_per_block, 3),
                )

        if load.interruptible:
            # Cheapest N blocks anywhere in the window.
            candidates.sort(key=lambda x: x[2])
            chosen = candidates[:n_needed]
        else:
            # Cheapest CONTIGUOUS run of N blocks. Slide a window.
            best_total = float("inf")
            best_start = 0
            # candidates are sorted chronologically by construction
            # (price_blocks input is sorted)
            for i in range(len(candidates) - n_needed + 1):
                window = candidates[i : i + n_needed]
                # Contiguity check: blocks must be back-to-back 15 min
                ok = True
                for a, b in zip(window, window[1:]):
                    if (b[1] - a[1]).total_seconds() != 15 * 60:
                        ok = False
                        break
                if not ok:
                    continue
                total = sum(c[2] for c in window)
                if total < best_total:
                    best_total = total
                    best_start = i
            if best_total == float("inf"):
                self.logger.warning(
                    f"Deferrable load {load.name!r} non-interruptible but no "
                    f"contiguous {n_needed}-block run available — falling back to "
                    f"interruptible scheduling"
                )
                candidates.sort(key=lambda x: x[2])
                chosen = candidates[:n_needed]
            else:
                chosen = candidates[best_start : best_start + n_needed]

        # Compute cost and naive-cost (the "run as soon as possible" baseline).
        kwh_per_block = load.power_kw * 0.25
        chosen.sort(key=lambda x: x[1])
        expected = sum(c[2] for c in chosen) * kwh_per_block
        by_time = sorted(candidates, key=lambda x: x[1])
        if load.interruptible:
            # Naive = the earliest N blocks in the window.
            naive_blocks = by_time[:n_needed]
        else:
            # Naive baseline must match the chosen plan's contiguity: the
            # earliest contiguous N-block run (else savings_czk is mis-stated).
            naive_blocks = by_time[:n_needed]
            for i in range(len(by_time) - n_needed + 1):
                window = by_time[i : i + n_needed]
                if all(
                    (b[1] - a[1]).total_seconds() == 15 * 60
                    for a, b in zip(window, window[1:])
                ):
                    naive_blocks = window
                    break
        naive = sum(c[2] for c in naive_blocks) * kwh_per_block if naive_blocks else 0.0

        # Build schedule
        blocks: List[Tuple[str, str]] = []
        block_datetimes: List[datetime] = []
        for _, ts, _ in chosen:
            end = ts + timedelta(minutes=15)
            blocks.append((ts.strftime("%H:%M"), end.strftime("%H:%M")))
            block_datetimes.append(ts)

        shortfall = max(0, requested - len(chosen)) * kwh_per_block
        if shortfall > 0:
            self.logger.warning(
                f"Deferrable load {load.name!r}: scheduled {len(chosen)}/"
                f"{requested} blocks — {shortfall:.2f} kWh shortfall"
            )

        return DeferrableLoadSchedule(
            load_name=load.name,
            blocks=blocks,
            block_datetimes=block_datetimes,
            expected_cost_czk=round(expected, 2),
            naive_cost_czk=round(naive, 2),
            requested_blocks=requested,
            scheduled_blocks=len(chosen),
            energy_shortfall_kwh=round(shortfall, 3),
        )

    def schedule_all(
        self,
        loads: List[DeferrableLoad],
        price_blocks: List[Tuple[datetime, float]],
        distribution_func: Callable[[int], float],
    ) -> List[DeferrableLoadSchedule]:
        """Schedule every load independently (no shared-power constraint).

        Returns one schedule per load.
        """
        return [
            self.schedule(load, price_blocks, distribution_func)
            for load in loads
        ]

    def consumption_overlay(
        self,
        schedules: List[DeferrableLoadSchedule],
        loads_by_name: Dict[str, DeferrableLoad],
    ) -> Dict[Any, float]:
        """Build a forecast overlay the battery optimizer can ADD to its
        consumption forecast.

        IMPORTANT unit convention: both optimizer engines treat
        forecast values as FULL-HOUR figures and divide by 4 to get the
        per-15-min-block energy. So a single scheduled 15-min block carrying
        ``power*0.25`` kWh must be added as ``power*1.0`` on the exact block
        timestamp, so after the optimizer's ``/4`` the block sees the true
        ``power*0.25`` kWh without smearing the load across the rest of the
        hour. Legacy schedules without datetimes fall back to hour keys.
        """
        overlay: Dict[Any, float] = {}
        for sched in schedules:
            load = loads_by_name.get(sched.load_name)
            if not load:
                continue
            # Per scheduled 15-min block, contribute the hourly-rate energy so
            # the optimizer's /4 recovers power*0.25 kWh for that block. The
            # hourly rate that yields power*0.25 kWh after /4 is exactly power.
            hour_rate_per_block = load.power_kw
            if sched.block_datetimes:
                for ts in sched.block_datetimes:
                    overlay[ts] = overlay.get(ts, 0.0) + hour_rate_per_block
            else:
                for start_str, _ in sched.blocks:
                    hour = int(start_str.split(":")[0])
                    overlay[hour] = overlay.get(hour, 0.0) + hour_rate_per_block
        return overlay
