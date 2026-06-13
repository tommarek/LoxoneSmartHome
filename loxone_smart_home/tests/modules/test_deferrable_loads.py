"""Tests for the deferrable-load scheduler."""

from datetime import datetime, time, timedelta
from typing import List, Tuple

from modules.growatt.deferrable_loads import (
    DeferrableLoad,
    DeferrableLoadSchedule,
    DeferrableLoadScheduler,
    filter_to_current_window_instance,
)


def make_blocks(prices: List[float], start_hour: int = 0) -> List[Tuple[datetime, float]]:
    """Build 15-min (timestamp, price) blocks from a list of prices."""
    base = datetime(2025, 6, 1, start_hour, 0, 0)
    return [(base + timedelta(minutes=15 * i), p) for i, p in enumerate(prices)]


def flat_dist(_hour):
    return 0.0


def test_required_blocks_rounds_up():
    load = DeferrableLoad(
        name="ev", energy_required_kwh=5.0, power_kw=2.0,
        earliest_start=time(0, 0), latest_end=time(23, 59),
    )
    # 2 kW * 0.25 h = 0.5 kWh/block → 5 / 0.5 = 10 blocks exactly
    assert load.required_blocks() == 10
    # 5.1 kWh needs 11 blocks (round up)
    load.energy_required_kwh = 5.1
    assert load.required_blocks() == 11


def test_required_blocks_zero_power():
    load = DeferrableLoad(
        name="x", energy_required_kwh=5.0, power_kw=0.0,
        earliest_start=time(0, 0), latest_end=time(23, 59),
    )
    assert load.required_blocks() == 0


def test_interruptible_picks_cheapest_blocks_in_window():
    # 8 blocks (2 hours): cheap blocks are indices 2,3 (price 0.5)
    prices = [5, 5, 0.5, 0.5, 5, 5, 5, 5]
    blocks = make_blocks(prices, start_hour=0)
    load = DeferrableLoad(
        name="ev", energy_required_kwh=1.0, power_kw=2.0,  # needs 2 blocks
        earliest_start=time(0, 0), latest_end=time(2, 0),
        interruptible=True,
    )
    sched = DeferrableLoadScheduler().schedule(load, blocks, flat_dist)
    # Should pick the two cheapest (00:30, 00:45)
    assert sorted(sched.blocks) == [("00:30", "00:45"), ("00:45", "01:00")]


def test_window_excludes_out_of_window_blocks():
    prices = [0.1, 0.1, 5, 5, 5, 5, 5, 5]
    blocks = make_blocks(prices, start_hour=0)
    # Window starts at 01:00 — the cheap blocks at 00:00/00:15 are excluded.
    load = DeferrableLoad(
        name="ev", energy_required_kwh=0.5, power_kw=2.0,  # 1 block
        earliest_start=time(1, 0), latest_end=time(2, 0),
    )
    sched = DeferrableLoadScheduler().schedule(load, blocks, flat_dist)
    for start, _ in sched.blocks:
        assert start >= "01:00"


def test_non_interruptible_picks_contiguous_run():
    # Cheapest single block is isolated; cheapest contiguous pair is 4,5
    prices = [9, 0.1, 9, 9, 1, 1, 9, 9]
    blocks = make_blocks(prices, start_hour=0)
    load = DeferrableLoad(
        name="dishwasher", energy_required_kwh=1.0, power_kw=2.0,  # 2 blocks
        earliest_start=time(0, 0), latest_end=time(2, 0),
        interruptible=False,
    )
    sched = DeferrableLoadScheduler().schedule(load, blocks, flat_dist)
    assert sched.blocks == [("01:00", "01:15"), ("01:15", "01:30")]


def test_partial_window_schedules_what_fits():
    prices = [1.0, 1.0]
    blocks = make_blocks(prices, start_hour=0)
    load = DeferrableLoad(
        name="ev", energy_required_kwh=10.0, power_kw=2.0,  # wants 10 blocks
        earliest_start=time(0, 0), latest_end=time(1, 0),
    )
    sched = DeferrableLoadScheduler().schedule(load, blocks, flat_dist)
    # Only 2 blocks available → schedule both, no crash.
    assert len(sched.blocks) == 2


def test_consumption_overlay_skips_legacy_blocks_without_datetimes():
    # A schedule carrying `blocks` but no `block_datetimes` cannot be safely
    # mapped to the optimizer's per-block consumption grid: the only key
    # available is the integer hour, and the optimizer applies an hourly value
    # to EACH of the four 15-min blocks in that hour (then /4), which would
    # multiply a single hour's deferrable draw four-fold and over-charge the
    # battery from grid. All real producers populate block_datetimes, so this
    # case is skipped (with a warning) rather than double-counted.
    load = DeferrableLoad(
        name="ev", energy_required_kwh=2.0, power_kw=2.0,
        earliest_start=time(0, 0), latest_end=time(23, 59),
    )
    sched = DeferrableLoadSchedule(
        load_name="ev",
        blocks=[("10:00", "10:15"), ("10:15", "10:30"), ("11:00", "11:15")],
    )
    overlay = DeferrableLoadScheduler().consumption_overlay(
        [sched], {"ev": load}
    )
    # No datetimes → no overlay contribution (no integer-hour double-count).
    assert overlay == {}


def test_consumption_overlay_preserves_date_when_available():
    load = DeferrableLoad(
        name="ev", energy_required_kwh=1.0, power_kw=2.0,
        earliest_start=time(0, 0), latest_end=time(23, 59),
    )
    today = datetime(2025, 6, 1, 10, 0)
    tomorrow = datetime(2025, 6, 2, 10, 0)
    sched = DeferrableLoadSchedule(
        load_name="ev",
        blocks=[("10:00", "10:15"), ("10:00", "10:15")],
        block_datetimes=[today, tomorrow],
    )

    overlay = DeferrableLoadScheduler().consumption_overlay(
        [sched], {"ev": load}
    )

    assert overlay[today] == 2.0
    assert overlay[tomorrow] == 2.0
    assert (today.date(), 10) not in overlay
    assert (tomorrow.date(), 10) not in overlay
    assert 10 not in overlay


def test_overnight_window_crosses_midnight():
    # 22:00-06:00 EV window: build 32 blocks spanning 20:00..04:00 so the
    # window wraps midnight. Cheap blocks placed at 23:00 and 01:00.
    base = datetime(2025, 6, 1, 20, 0, 0)
    prices = []
    for i in range(32):  # 8 hours of 15-min blocks
        ts = base + timedelta(minutes=15 * i)
        # Cheap inside the window after 22:00 and before 06:00, pricey before.
        prices.append((ts, 0.2 if (ts.hour >= 22 or ts.hour < 6) else 9.0))
    load = DeferrableLoad(
        name="ev", energy_required_kwh=1.0, power_kw=2.0,  # 2 blocks
        earliest_start=time(22, 0), latest_end=time(6, 0),
        interruptible=True,
    )
    sched = DeferrableLoadScheduler().schedule(load, prices, flat_dist)
    assert len(sched.blocks) == 2
    # Every chosen block must be inside the wrap-around window.
    for start, _ in sched.blocks:
        h = int(start.split(":")[0])
        assert h >= 22 or h < 6


def test_shortfall_reported_when_window_too_narrow():
    prices = [(datetime(2025, 6, 1, 0, 0) + timedelta(minutes=15 * i), 1.0)
              for i in range(2)]
    load = DeferrableLoad(
        name="ev", energy_required_kwh=10.0, power_kw=2.0,  # wants 10 blocks
        earliest_start=time(0, 0), latest_end=time(1, 0),
    )
    sched = DeferrableLoadScheduler().schedule(load, prices, flat_dist)
    # 10 kWh / (2 kW * 0.25 h) = 20 blocks required; only 2 fit the 1h window.
    assert sched.requested_blocks == 20
    assert sched.scheduled_blocks == 2
    assert sched.energy_shortfall_kwh > 0  # 18 blocks * 0.5 kWh = 9.0
    assert sched.fully_scheduled is False


def test_fully_scheduled_true_when_window_fits():
    prices = [(datetime(2025, 6, 1, 0, 0) + timedelta(minutes=15 * i), 1.0)
              for i in range(8)]
    load = DeferrableLoad(
        name="ev", energy_required_kwh=1.0, power_kw=2.0,  # 2 blocks
        earliest_start=time(0, 0), latest_end=time(2, 0),
    )
    sched = DeferrableLoadScheduler().schedule(load, prices, flat_dist)
    assert sched.fully_scheduled is True
    assert sched.energy_shortfall_kwh == 0


def test_savings_is_naive_minus_expected():
    sched = DeferrableLoadSchedule(
        load_name="ev", naive_cost_czk=10.0, expected_cost_czk=6.0
    )
    assert sched.savings_czk == 4.0


# --- Multi-day horizon: only the FIRST window instance is schedulable -------


def make_horizon(start: datetime, hours: int) -> List[datetime]:
    """A contiguous 15-min block grid spanning `hours` hours."""
    return [start + timedelta(minutes=15 * i) for i in range(hours * 4)]


def overnight_load(**kw) -> DeferrableLoad:
    defaults = dict(
        name="ev", energy_required_kwh=1.0, power_kw=2.0,  # 2 blocks
        earliest_start=time(22, 0), latest_end=time(6, 0),
        interruptible=True,
    )
    defaults.update(kw)
    return DeferrableLoad(**defaults)


def daytime_load(**kw) -> DeferrableLoad:
    defaults = dict(
        name="boiler", energy_required_kwh=1.0, power_kw=2.0,  # 2 blocks
        earliest_start=time(6, 0), latest_end=time(22, 0),
        interruptible=True,
    )
    defaults.update(kw)
    return DeferrableLoad(**defaults)


def test_filter_keeps_only_first_overnight_instance():
    # 32h horizon from 20:00 Jun 1 → 04:00 Jun 3: contains TWO instances of a
    # 22:00→06:00 window (Jun 1 22:00→Jun 2 06:00 and Jun 2 22:00→horizon end).
    horizon = make_horizon(datetime(2025, 6, 1, 20, 0), 32)
    kept = filter_to_current_window_instance(horizon, overnight_load())
    assert kept, "first instance must yield candidates"
    assert kept[0] == datetime(2025, 6, 1, 22, 0)
    assert kept[-1] == datetime(2025, 6, 2, 5, 45)
    # Nothing from the second night.
    assert all(ts < datetime(2025, 6, 2, 6, 0) for ts in kept)


def test_filter_keeps_only_first_daytime_instance():
    # 32h horizon from 04:00 Jun 1 → 12:00 Jun 2: contains TWO instances of a
    # 06:00→22:00 window (Jun 1 06:00→22:00 and Jun 2 06:00→12:00).
    horizon = make_horizon(datetime(2025, 6, 1, 4, 0), 32)
    kept = filter_to_current_window_instance(horizon, daytime_load())
    assert kept[0] == datetime(2025, 6, 1, 6, 0)
    assert kept[-1] == datetime(2025, 6, 1, 21, 45)
    assert all(ts < datetime(2025, 6, 1, 22, 0) for ts in kept)


def test_filter_now_already_inside_window():
    # Horizon starts at 23:00, already inside the 22:00→06:00 window — the
    # current (partial) instance is used, not skipped.
    horizon = make_horizon(datetime(2025, 6, 1, 23, 0), 32)
    kept = filter_to_current_window_instance(horizon, overnight_load())
    assert kept[0] == datetime(2025, 6, 1, 23, 0)
    assert kept[-1] == datetime(2025, 6, 2, 5, 45)


def test_filter_now_inside_after_midnight_tail():
    # Horizon starts at 03:00 — inside the after-midnight tail of 22:00→06:00.
    # The current instance closes at 06:00 the SAME day.
    horizon = make_horizon(datetime(2025, 6, 2, 3, 0), 32)
    kept = filter_to_current_window_instance(horizon, overnight_load())
    assert kept[0] == datetime(2025, 6, 2, 3, 0)
    assert kept[-1] == datetime(2025, 6, 2, 5, 45)
    assert all(ts < datetime(2025, 6, 2, 6, 0) for ts in kept)


def test_filter_full_day_window_closes_at_2359():
    # 00:00→23:59 "full-day" window: the 23:45→23:59 gap contains no block
    # start, but the instance must still close at 23:59 (not leak into
    # tomorrow).
    load = daytime_load(earliest_start=time(0, 0), latest_end=time(23, 59))
    horizon = make_horizon(datetime(2025, 6, 1, 0, 0), 32)
    kept = filter_to_current_window_instance(horizon, load)
    assert kept[-1] == datetime(2025, 6, 1, 23, 45)  # 23:45 < 23:59 → in-window
    assert all(ts < datetime(2025, 6, 1, 23, 59) for ts in kept)


def test_filter_no_blocks_in_window():
    # Horizon entirely outside the window → empty.
    load = overnight_load()
    horizon = make_horizon(datetime(2025, 6, 1, 8, 0), 8)  # 08:00-16:00
    assert filter_to_current_window_instance(horizon, load) == []


def test_scheduler_does_not_place_blocks_in_second_overnight_instance():
    # 32h horizon with two 22:00→06:00 instances; the SECOND night is much
    # cheaper, tempting the scheduler past the current cycle's deadline.
    base = datetime(2025, 6, 1, 20, 0)
    blocks = []
    for i in range(32 * 4):
        ts = base + timedelta(minutes=15 * i)
        in_window = ts.hour >= 22 or ts.hour < 6
        second_night = ts >= datetime(2025, 6, 2, 12, 0)
        price = 0.1 if (in_window and second_night) else (2.0 if in_window else 9.0)
        blocks.append((ts, price))
    load = overnight_load(energy_required_kwh=2.0)  # 4 blocks
    sched = DeferrableLoadScheduler().schedule(load, blocks, flat_dist)
    deadline = datetime(2025, 6, 2, 6, 0)
    assert sched.scheduled_blocks == 4
    assert sched.block_datetimes, "schedule must carry datetimes"
    assert all(ts < deadline for ts in sched.block_datetimes)


def test_schedule_all_respects_first_daytime_instance():
    # Two 06:00→22:00 instances in a 32h horizon; tomorrow morning is cheaper.
    base = datetime(2025, 6, 1, 4, 0)
    blocks = []
    for i in range(32 * 4):
        ts = base + timedelta(minutes=15 * i)
        in_window = 6 <= ts.hour < 22
        tomorrow = ts.date() > base.date()
        price = 0.1 if (in_window and tomorrow) else (2.0 if in_window else 9.0)
        blocks.append((ts, price))
    load = daytime_load(energy_required_kwh=2.0)  # 4 blocks
    schedules = DeferrableLoadScheduler().schedule_all([load], blocks, flat_dist)
    assert len(schedules) == 1
    deadline = datetime(2025, 6, 1, 22, 0)
    assert schedules[0].scheduled_blocks == 4
    assert all(ts < deadline for ts in schedules[0].block_datetimes)
