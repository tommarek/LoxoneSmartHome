"""Scheduling strategies for Growatt battery optimization."""

import asyncio
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from modules.growatt_controller import GrowattController, Period

# Constants
EOD_DTTIME = dt_time(23, 59, 59)
EOD_HHMM = "23:59"


class GrowattScheduler:
    """Handles scheduling strategies for Growatt battery optimization."""

    def __init__(self, controller: "GrowattController"):
        """Initialize the scheduler with a reference to the controller.

        Args:
            controller: The GrowattController instance
        """
        self.controller = controller
        self.logger = logging.getLogger(__name__)

    async def schedule_summer_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule summer operations with dynamic price-based mode switching.

        In summer:
        - If NO low prices (<1 CZK) exist that day:
          * Use regular when price >= 1 CZK (export enabled)
          * Use regular_no_export when price < 1 CZK (export disabled)

        - If low prices exist that day:
          * Night (00:00 to sunrise): regular_no_export
          * After sunrise until first low price: sell_production (export all solar)
          * When price < 1 CZK: charge_from_solar (store solar, no export)
          * When price >= 1 CZK: regular (normal operation with export)

        Args:
            hourly_prices: Dict of (start, end) -> price in EUR/MWh
            eur_czk_rate: EUR to CZK exchange rate
        """
        # Import Period here to avoid circular import
        from modules.growatt_controller import Period

        # Convert threshold from CZK/kWh to EUR/MWh for comparison
        eur_czk_rate = max(eur_czk_rate, 1.0)
        threshold_eur_mwh = self.controller.config.summer_price_threshold * 1000 / eur_czk_rate

        # Check if there are any low prices during the day
        has_low_prices = any(
            price < threshold_eur_mwh
            for (_, _), price in hourly_prices.items()
        )

        if not has_low_prices:
            # Simple mode switching: regular with export when high, regular_no_export when low
            self.logger.info("Summer: No prices below 1 CZK/kWh - using simple price-based switching")

            for hour in range(24):
                hour_str = f"{hour:02d}:00"
                next_hour_str = f"{hour+1:02d}:00" if hour < 23 else "24:00"
                price = hourly_prices.get((hour_str, next_hour_str), 0)
                price_czk = price * eur_czk_rate / 1000

                # Determine mode based on price
                if price >= threshold_eur_mwh:
                    mode = "regular"
                    mode_desc = f"export @ {price_czk:.3f} CZK/kWh"
                else:
                    mode = "regular_no_export"
                    mode_desc = f"no export @ {price_czk:.3f} CZK/kWh"

                # Create period for each hour
                end_time = EOD_DTTIME if hour == 23 else dt_time(hour + 1, 0)
                self.controller._scheduled_periods.append(
                    Period(mode, dt_time(hour, 0), end_time)
                )

                # Schedule mode change
                task = self.controller._schedule_action(
                    hour_str, self.controller._apply_composite_mode, mode
                )
                self.controller._scheduled_tasks.append(task)

                self.logger.debug(f"Hour {hour_str}: {mode} ({mode_desc})")

            return  # Exit early for simple mode

        # Complex mode with morning export strategy (only when low prices exist)
        self.logger.info("Summer: Found low prices - using morning export + battery storage strategy")

        # Get sunrise time and round to hour boundary
        now = self.controller._get_local_now()
        days_ahead = 1 if now.time() >= dt_time(23, 45) else 0
        sunrise_time = self.controller._get_sunrise_time(days_ahead)

        # Round sunrise to next hour boundary
        sunrise_hour = sunrise_time.hour
        if sunrise_time.minute > 0:
            sunrise_hour = (sunrise_hour + 1) % 24
        sunrise_hour_time = dt_time(sunrise_hour, 0)
        sunrise_str = f"{sunrise_hour:02d}:00"

        self.logger.info(f"Sunrise: {sunrise_time.strftime('%H:%M')} -> rounded to {sunrise_str}")

        # 1. Schedule night period (00:00 to sunrise)
        if sunrise_hour > 0:
            self.logger.info(f"Scheduling regular_no_export from 00:00 to {sunrise_str} (night)")
            self.controller._scheduled_periods.append(
                Period("regular_no_export", dt_time(0, 0), sunrise_hour_time)
            )
            task = self.controller._schedule_action(
                "00:00", self.controller._apply_composite_mode, "regular_no_export"
            )
            self.controller._scheduled_tasks.append(task)

        # 2. Find first low-price hour after sunrise
        first_low_hour_after_sunrise = None
        for hour in range(sunrise_hour, 24):
            hour_str = f"{hour:02d}:00"
            next_hour_str = f"{hour+1:02d}:00" if hour < 23 else "24:00"
            price = hourly_prices.get((hour_str, next_hour_str), float('inf'))
            if price < threshold_eur_mwh:
                first_low_hour_after_sunrise = hour
                break

        # 3. Schedule sell_production from sunrise until first low price (if exists)
        if first_low_hour_after_sunrise is not None and first_low_hour_after_sunrise > sunrise_hour:
            sell_end_str = f"{first_low_hour_after_sunrise:02d}:00"
            self.logger.info(
                f"Scheduling sell_production from {sunrise_str} to {sell_end_str} "
                f"(export all solar before cheap hours)"
            )
            self.controller._scheduled_periods.append(
                Period(
                    "sell_production",
                    sunrise_hour_time,
                    dt_time(first_low_hour_after_sunrise, 0)
                )
            )
            task = self.controller._schedule_action(
                sunrise_str, self.controller._apply_composite_mode, "sell_production"
            )
            self.controller._scheduled_tasks.append(task)
            start_hour = first_low_hour_after_sunrise
        else:
            # No low prices found or they start immediately at sunrise
            start_hour = sunrise_hour

        # 4. Schedule hourly periods based on price threshold
        previous_mode = None
        period_start = None

        for hour in range(start_hour, 24):
            hour_str = f"{hour:02d}:00"
            next_hour_str = f"{hour+1:02d}:00" if hour < 23 else "24:00"

            # Get price for this hour
            price = hourly_prices.get((hour_str, next_hour_str), 0)
            price_czk = price * eur_czk_rate / 1000

            # Determine mode based on price
            if price < threshold_eur_mwh:
                # Low price: store solar, no export
                mode = "charge_from_solar"
                mode_desc = f"store solar @ {price_czk:.3f} CZK/kWh"
            else:
                # High price: normal operation with export
                mode = "regular"
                mode_desc = f"export @ {price_czk:.3f} CZK/kWh"

            self.logger.info(f"Hour {hour_str}: {mode} ({mode_desc})")

            # Handle mode changes or continuation
            if mode != previous_mode:
                # Close previous period if exists
                if previous_mode is not None and period_start is not None:
                    # End the previous period at current hour
                    self.controller._scheduled_periods.append(
                        Period(previous_mode, period_start, dt_time(hour, 0))
                    )

                # Start new period
                period_start = dt_time(hour, 0)

                # Schedule mode change
                task = self.controller._schedule_action(
                    hour_str, self.controller._apply_composite_mode, mode
                )
                self.controller._scheduled_tasks.append(task)

                previous_mode = mode

        # Close the last period
        if previous_mode is not None and period_start is not None:
            self.controller._scheduled_periods.append(
                Period(previous_mode, period_start, EOD_DTTIME)
            )

        # Log summary (moved outside the loop)
        low_hours = sum(
            1 for (start, _), price in hourly_prices.items()
            if price < threshold_eur_mwh and
            datetime.strptime(start, "%H:%M").time() >= sunrise_hour_time
        )
        high_hours = 24 - sunrise_hour - low_hours

        self.logger.info(
            f"Summer schedule: {low_hours} low-price hours (charge_from_solar), "
            f"{high_hours} high-price hours (regular/sell_production)"
        )

    async def schedule_winter_strategy(
        self, hourly_prices: Dict[Tuple[str, str], float], eur_czk_rate: float
    ) -> None:
        """Schedule winter battery strategy with AC charging during cheapest hours.

        Args:
            hourly_prices: Dict of (start, end) -> price in EUR/MWh
            eur_czk_rate: EUR to CZK exchange rate
        """
        # Analyze prices
        cheapest_individual_hours = set(
            (start, stop)
            for start, stop, _ in self.controller._find_n_cheapest_hours(
                hourly_prices, n=self.controller.config.individual_cheapest_hours
            )
        )

        quadrants = self.controller._categorize_prices_into_quadrants(hourly_prices)
        cheapest_quadrant_hours = set(
            (start, stop) for start, stop, _ in quadrants.get("Cheapest", [])
        )

        # Combine cheapest hours
        cheapest_individual_hours = cheapest_individual_hours.union(cheapest_quadrant_hours)

        cheapest_consecutive = set(
            (start, stop)
            for start, stop, _ in self.controller._find_cheapest_consecutive_hours(
                hourly_prices, self.controller.config.battery_charge_hours
            )
        )

        # Union all cheap hours
        all_cheap_hours = cheapest_individual_hours.union(cheapest_consecutive)

        # Calculate price statistics
        all_prices = list(hourly_prices.values())
        min_price = min(all_prices)
        max_price = max(all_prices)
        avg_price = sum(all_prices) / len(all_prices)

        # Calculate average prices for selected hours
        if cheapest_individual_hours:
            individual_avg = sum(hourly_prices[hour] for hour in cheapest_individual_hours) / len(
                cheapest_individual_hours
            )
        else:
            individual_avg = 0.0

        if cheapest_consecutive:
            consecutive_avg = sum(hourly_prices[hour] for hour in cheapest_consecutive) / len(
                cheapest_consecutive
            )
        else:
            consecutive_avg = 0.0

        # Use provided rate for consistency
        min_price_czk = min_price * eur_czk_rate / 1000
        max_price_czk = max_price * eur_czk_rate / 1000
        avg_price_czk = avg_price * eur_czk_rate / 1000
        individual_avg_czk = individual_avg * eur_czk_rate / 1000
        consecutive_avg_czk = consecutive_avg * eur_czk_rate / 1000

        self.logger.info(
            f"Winter mode price analysis: min={min_price:.2f} EUR/MWh "
            f"({min_price_czk:.3f} CZK/kWh), "
            f"max={max_price:.2f} EUR/MWh ({max_price_czk:.3f} CZK/kWh), "
            f"avg={avg_price:.2f} EUR/MWh ({avg_price_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Cheapest individual hours: {len(cheapest_individual_hours)} "
            f"(avg: {individual_avg:.2f} EUR/MWh = {individual_avg_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Cheapest consecutive hours: {len(cheapest_consecutive)} "
            f"(avg: {consecutive_avg:.2f} EUR/MWh = {consecutive_avg_czk:.3f} CZK/kWh)"
        )
        self.logger.info(
            f"Total cheap hours: {len(all_cheap_hours)}, "
            f"Export threshold: {self.controller.config.export_price_threshold:.2f} CZK/kWh"
        )

        # Schedule battery-first mode with AC charging for winter
        await self.schedule_battery_control(
            all_cheap_hours, cheapest_consecutive, hourly_prices, eur_czk_rate
        )

    async def schedule_battery_control(
        self,
        all_cheap_hours: set[Tuple[str, str]],
        cheapest_consecutive: set[Tuple[str, str]],
        hourly_prices: Dict[Tuple[str, str], float],
        eur_czk_rate: float,
    ) -> None:
        """Schedule battery control based on price analysis using composite modes.

        Args:
            all_cheap_hours: Set of all cheap hour tuples
            cheapest_consecutive: Set of cheapest consecutive hour tuples
            hourly_prices: Dict of (start, end) -> price in EUR/MWh
            eur_czk_rate: EUR to CZK exchange rate
        """
        # Import Period here to avoid circular import
        from modules.growatt_controller import Period

        # Schedule charge_from_grid during cheapest consecutive hours
        if cheapest_consecutive:
            charge_hours = [
                (start, stop, hourly_prices[(start, stop)]) for start, stop in cheapest_consecutive
            ]
            charge_groups = self.controller._group_contiguous_hours_simple(charge_hours)

            for start_time, stop_time in charge_groups:
                # Calculate price statistics for this charging period
                start_t = self.controller._parse_hhmm(start_time)
                stop_t = self.controller._parse_hhmm(stop_time)
                charge_prices = [
                    price
                    for start, stop, price in charge_hours
                    if self.controller._parse_hhmm(start) >= start_t and
                       self.controller._parse_hhmm(stop) <= stop_t
                ]
                if charge_prices:
                    avg_charge_price = sum(charge_prices) / len(charge_prices)
                    avg_charge_price_czk = avg_charge_price * eur_czk_rate / 1000

                    self.logger.info(
                        f"Scheduling CHARGE_FROM_GRID from {start_time} to {stop_time} "
                        f"(avg: {avg_charge_price:.2f} EUR/MWh = {avg_charge_price_czk:.2f} CZK/kWh)"
                    )

                # Schedule charge_from_grid composite mode
                self.controller._scheduled_periods.append(
                    Period("charge_from_grid",
                           self.controller._parse_hhmm(start_time),
                           self.controller._parse_hhmm(stop_time),
                           params={"stop_soc": 90})
                )
                task = self.controller._schedule_action(
                    start_time, self.controller._apply_composite_mode, "charge_from_grid",
                    {"stop_soc": 90}
                )
                self.controller._scheduled_tasks.append(task)

        # Determine high-price hours for discharge_to_grid
        threshold_eur_mwh = self.controller.config.export_price_threshold * 1000 / eur_czk_rate
        high_price_hours = [
            (start, stop, price)
            for (start, stop), price in hourly_prices.items()
            if price > threshold_eur_mwh * 1.5  # 50% above threshold for discharge
        ]

        if high_price_hours:
            discharge_groups = self.controller._group_contiguous_hours_simple(high_price_hours)
            for start_time, stop_time in discharge_groups:
                self.logger.info(f"Scheduling DISCHARGE_TO_GRID from {start_time} to {stop_time}")
                self.controller._scheduled_periods.append(
                    Period("discharge_to_grid",
                           self.controller._parse_hhmm(start_time),
                           self.controller._parse_hhmm(stop_time),
                           params={"stop_soc": 20, "power_rate": 100})
                )
                task = self.controller._schedule_action(
                    start_time, self.controller._apply_composite_mode, "discharge_to_grid",
                    {"stop_soc": 20, "power_rate": 100}
                )
                self.controller._scheduled_tasks.append(task)

        # Fill gaps with regular or regular_no_export based on price
        all_scheduled_times = set()
        for period in self.controller._scheduled_periods:
            # Add all hours covered by scheduled periods
            start_hour = period.start.hour
            end_hour = period.end.hour if period.end != dt_time(0, 0) else 24
            if start_hour <= end_hour:
                for h in range(start_hour, end_hour):
                    all_scheduled_times.add(f"{h:02d}:00")
            else:
                # Wraps midnight
                for h in range(start_hour, 24):
                    all_scheduled_times.add(f"{h:02d}:00")
                for h in range(0, end_hour):
                    all_scheduled_times.add(f"{h:02d}:00")

        # Schedule regular modes for unscheduled hours
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            if hour_str not in all_scheduled_times:
                next_hour = f"{hour+1:02d}:00" if hour < 23 else "24:00"
                price = hourly_prices.get((hour_str, next_hour), 0)

                # Use regular if price above threshold, otherwise regular_no_export
                if price >= threshold_eur_mwh:
                    mode = "regular"
                else:
                    mode = "regular_no_export"

                self.controller._scheduled_periods.append(
                    Period(mode, dt_time(hour, 0),
                          dt_time(hour + 1, 0) if hour < 23 else EOD_DTTIME)
                )
                task = self.controller._schedule_action(
                    hour_str, self.controller._apply_composite_mode, mode
                )
                self.controller._scheduled_tasks.append(task)