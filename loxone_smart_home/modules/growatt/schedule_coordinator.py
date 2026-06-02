"""Schedule coordination: price logging and schedule state management.

Extracted from GrowattController to reduce its complexity.
Handles schedule block tracking, price table formatting, and schedule logging.
"""

import logging
from datetime import date as date_type
from datetime import datetime
from typing import List, Optional, Set, Tuple

from .types import GrowattLogLevel


class ScheduleCoordinator:
    """Manages schedule state and price/schedule logging.

    Owns the schedule block sets (charging, pre-discharge, discharge)
    and provides formatting/logging methods that operate on them.
    """

    def __init__(self, logger: logging.Logger, log_level: GrowattLogLevel) -> None:
        self.logger = logger
        self._log_level = log_level

        # Schedule state (moved from controller)
        self.cheapest_charging_blocks_today: Set[Tuple[str, str]] = set()
        self.cheapest_charging_blocks_tomorrow: Set[Tuple[str, str]] = set()
        self.pre_discharge_blocks_today: Set[Tuple[str, str]] = set()
        self.pre_discharge_blocks_tomorrow: Set[Tuple[str, str]] = set()
        self.discharge_periods_today: Set[Tuple[str, str]] = set()
        self.discharge_periods_tomorrow: Set[Tuple[str, str]] = set()
        self.sell_production_blocks_today: Set[Tuple[str, str]] = set()
        self.sell_production_blocks_tomorrow: Set[Tuple[str, str]] = set()
        # Battery-hold blocks (preserve battery, serve house from grid). MILP-only.
        self.hold_blocks_today: Set[Tuple[str, str]] = set()
        self.hold_blocks_tomorrow: Set[Tuple[str, str]] = set()
        self.combined_charging_blocks: Set[Tuple[str, str]] = set()
        self.queued_tomorrow_charging: List[Tuple[datetime, datetime, float]] = []
        self.queued_tomorrow_pre_discharge: List[Tuple[datetime, datetime, float]] = []
        self.queued_tomorrow_discharge: List[Tuple[datetime, datetime, float]] = []
        self.defer_charging_to_tomorrow: bool = False
        self.tomorrow_cheaper_by: Optional[float] = None

    def _should_log(self, level: GrowattLogLevel) -> bool:
        """Check if we should log at given level."""
        return self._log_level >= level

    @staticmethod
    def group_consecutive_blocks(
        blocks: List[Tuple[datetime, datetime, float]],
    ) -> List[List[Tuple[datetime, datetime, float]]]:
        """Group consecutive datetime-based blocks into periods.

        A pure function: takes a list of (start_dt, end_dt, price) tuples
        and returns groups of consecutive blocks (where end == next start).

        Args:
            blocks: List of (start_dt, end_dt, price) tuples

        Returns:
            List of groups, where each group is a list of consecutive blocks
        """
        if not blocks:
            return []

        # Sort by start time
        sorted_blocks = sorted(blocks, key=lambda x: x[0])

        groups: List[List[Tuple[datetime, datetime, float]]] = []
        current_group = [sorted_blocks[0]]

        for i in range(1, len(sorted_blocks)):
            prev_end = current_group[-1][1]
            curr_start = sorted_blocks[i][0]

            # Check if consecutive (end of prev equals start of current)
            if prev_end == curr_start:
                current_group.append(sorted_blocks[i])
            else:
                # Gap found - save current group and start new one
                groups.append(current_group)
                current_group = [sorted_blocks[i]]

        # Add the last group
        groups.append(current_group)

        return groups

    def format_price_summary(
        self,
        blocks: List[Tuple[datetime, datetime, float]],
    ) -> str:
        """Format a compact price summary for DETAIL level logging.

        Prices in blocks are already CZK/kWh -- no conversion needed.

        Args:
            blocks: List of (start_dt, end_dt, price_czk) tuples

        Returns:
            Formatted summary string
        """
        if not blocks:
            return "No blocks"

        prices_czk = [p for _, _, p in blocks]
        min_price = min(prices_czk)
        max_price = max(prices_czk)
        avg_price = sum(prices_czk) / len(prices_czk)

        # Group consecutive blocks for compact display
        groups = self.group_consecutive_blocks(blocks)

        if len(groups) == 1:
            group = groups[0]
            start = group[0][0]
            end = group[-1][1]
            return (
                f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')} "
                f"({len(blocks)} blocks, avg {avg_price:.2f} CZK/kWh)"
            )
        else:
            return (
                f"{len(blocks)} blocks in {len(groups)} periods, "
                f"avg {avg_price:.2f} CZK/kWh (min {min_price:.2f}, max {max_price:.2f})"
            )

    def log_compact_schedule(
        self,
        charging_today: List[Tuple[datetime, datetime, float]],
        charging_tomorrow: List[Tuple[datetime, datetime, float]],
        discharge_today: List[Tuple[datetime, datetime, float]],
        discharge_tomorrow: List[Tuple[datetime, datetime, float]],
    ) -> None:
        """Log compact schedule summary for DETAIL level."""
        self.logger.info("\U0001f4ca Schedule Summary:")

        if charging_today:
            summary = self.format_price_summary(charging_today)
            self.logger.info(f"  \U0001f50b Charging today: {summary}")

        if charging_tomorrow:
            summary = self.format_price_summary(charging_tomorrow)
            self.logger.info(f"  \U0001f50b Charging tomorrow: {summary}")

        if discharge_today:
            summary = self.format_price_summary(discharge_today)
            self.logger.info(f"  \u26a1 Discharge today: {summary}")

        if discharge_tomorrow:
            summary = self.format_price_summary(discharge_tomorrow)
            self.logger.info(f"  \u26a1 Discharge tomorrow: {summary}")

    def log_cross_day_price_table(
        self,
        window: List[Tuple[datetime, datetime, float]],
        force_display: bool = False,
        *,
        now: Optional[datetime] = None,
        full_window: Optional[List[Tuple[datetime, datetime, float]]] = None,
        has_optimizer: bool = False,
    ) -> None:
        """Log comprehensive cross-day price table showing all available blocks.

        Displays a compact table with visual markers for each block's treatment:
        - \U0001f50b = Regular charging block
        - \U0001f50c = Pre-discharge charging block
        - \u26a1 = Discharge block
        - (blank) = No special action

        Args:
            window: List of (start_dt, end_dt, price_czk) covering available price window
            force_display: If True, always show table regardless of log level (for startup)
            now: Current local time (caller provides, avoids coupling to controller timezone)
            full_window: Full window including past blocks (for force_display case).
                         If force_display is True and this is provided, it replaces window.
            has_optimizer: Whether the optimizer is active (affects legend labels)
        """
        if not window:
            return

        # When force_display, use full window with past blocks if provided
        if force_display and full_window:
            window = full_window

        # Only show full table at VERBOSE level (unless forced for startup)
        if not force_display and not self._should_log(GrowattLogLevel.VERBOSE):
            # At DETAIL level, just show a summary
            if self._should_log(GrowattLogLevel.DETAIL):
                all_prices_czk = [p for _, _, p in window]
                min_price = min(all_prices_czk)
                max_price = max(all_prices_czk)
                avg_price = sum(all_prices_czk) / len(all_prices_czk)

                self.logger.info(
                    f"\U0001f4ca Price summary ({len(window)} blocks): "
                    f"Min={min_price:.2f} CZK/kWh, Max={max_price:.2f} CZK/kWh, "
                    f"Avg={avg_price:.2f} CZK/kWh"
                )
            return

        if now is None:
            # Fallback: derive today from the window data
            today = window[0][0].date()
        else:
            today = now.date()

        # Create lookup sets for fast classification
        charging_today = self.cheapest_charging_blocks_today
        charging_tomorrow = self.cheapest_charging_blocks_tomorrow
        pre_discharge_today = self.pre_discharge_blocks_today
        pre_discharge_tomorrow = self.pre_discharge_blocks_tomorrow
        discharge_today = self.discharge_periods_today
        discharge_tomorrow = self.discharge_periods_tomorrow

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("\U0001f4ca COMPREHENSIVE PRICE TABLE (entire available window)")
        self.logger.info("=" * 70)

        # Group blocks by date
        today_blocks = [(s, e, p) for s, e, p in window if s.date() == today]
        tomorrow_blocks = [(s, e, p) for s, e, p in window if s.date() > today]

        # Display today's prices (if any remaining)
        if today_blocks:
            self.log_price_table_for_date(
                today_blocks,
                today,
                charging_today,
                pre_discharge_today,
                discharge_today,
                "TODAY",
            )

        # Display tomorrow's prices (if available)
        if tomorrow_blocks:
            tomorrow_date = tomorrow_blocks[0][0].date()
            self.log_price_table_for_date(
                tomorrow_blocks,
                tomorrow_date,
                charging_tomorrow,
                pre_discharge_tomorrow,
                discharge_tomorrow,
                "TOMORROW",
            )

        # Display legend
        self.logger.info("")
        legend_items = []
        if has_optimizer:
            # Optimizer mode: simplified labels
            if charging_today or charging_tomorrow:
                legend_items.append("\U0001f50b=Charge (optimizer)")
            if discharge_today or discharge_tomorrow:
                legend_items.append("\u26a1=Discharge (optimizer)")
        else:
            if charging_today or charging_tomorrow:
                legend_items.append("\U0001f50b=Regular charge")
            if pre_discharge_today or pre_discharge_tomorrow:
                legend_items.append("\U0001f50c=Pre-discharge charge")
            if discharge_today or discharge_tomorrow:
                legend_items.append("\u26a1=Discharge")
        if legend_items:
            self.logger.info(f"Legend: {', '.join(legend_items)}")

        # Display summary statistics across entire window
        all_prices_czk = [p for _, _, p in window]
        min_price = min(all_prices_czk)
        max_price = max(all_prices_czk)
        avg_price = sum(all_prices_czk) / len(all_prices_czk)

        self.logger.info(
            f"Window summary: Min={min_price:.3f} CZK/kWh, Max={max_price:.3f} CZK/kWh, "
            f"Avg={avg_price:.3f} CZK/kWh"
        )
        self.logger.info("=" * 70)

    def log_price_table_for_date(
        self,
        blocks: List[Tuple[datetime, datetime, float]],
        date: date_type,
        charging_blocks: Set[Tuple[str, str]],
        pre_discharge_blocks: Set[Tuple[str, str]],
        discharge_blocks: Set[Tuple[str, str]],
        title: str,
    ) -> None:
        """Log price table for a single date in 4-column format.

        Args:
            blocks: List of (start_dt, end_dt, price_czk) for this date
            date: Date being displayed
            charging_blocks: Set of (start_str, end_str) for charging
            pre_discharge_blocks: Set of (start_str, end_str) for pre-discharge
            discharge_blocks: Set of (start_str, end_str) for discharge
            title: Title to display for this table
        """
        self.logger.info("")
        self.logger.info(f"--- {title} ({date.strftime('%Y-%m-%d')}) ---")
        self.logger.info("\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        self.logger.info("\u2502  Hour   \u2502  :00-:15 \u2502  :15-:30 \u2502  :30-:45 \u2502  :45-:00 \u2502")
        self.logger.info("\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")

        # Create a dict for fast lookup by time string
        block_dict = {}
        for start_dt, end_dt, price in blocks:
            start_str = start_dt.strftime("%H:%M")
            end_str = end_dt.strftime("%H:%M")
            block_dict[(start_str, end_str)] = price

        # Process by hour (4 blocks per hour)
        # Determine hour range from actual blocks
        if blocks:
            start_hour = blocks[0][0].hour
            end_hour = blocks[-1][0].hour

            for hour in range(start_hour, end_hour + 1):
                row_prices = []

                # Process 4 quarter-hour blocks
                for quarter in range(4):
                    minute = quarter * 15
                    next_minute = (quarter + 1) * 15

                    start_str = f"{hour:02d}:{minute:02d}"
                    if next_minute < 60:
                        end_str = f"{hour:02d}:{next_minute:02d}"
                    else:
                        end_str = f"{(hour + 1) % 24:02d}:00"

                    # Find matching block
                    block_key = (start_str, end_str)
                    if block_key in block_dict:
                        price_czk = block_dict[block_key]  # Already CZK/kWh

                        # Determine marker
                        if block_key in pre_discharge_blocks:
                            marker = "\U0001f50c"
                        elif block_key in charging_blocks:
                            marker = "\U0001f50b"
                        elif block_key in discharge_blocks:
                            marker = "\u26a1"
                        else:
                            marker = " "

                        row_prices.append(f"{price_czk:5.2f}{marker}")
                    else:
                        row_prices.append("   -   ")

                # Pad if we don't have all 4 blocks
                while len(row_prices) < 4:
                    row_prices.append("   -   ")

                self.logger.info(
                    f"\u2502 {hour:02d}:00   \u2502 {row_prices[0]} \u2502 {row_prices[1]} \u2502 "
                    f"{row_prices[2]} \u2502 {row_prices[3]} \u2502"
                )

        self.logger.info("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
