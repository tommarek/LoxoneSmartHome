"""
Thermal Preprocessing Visualization Module for PEMS v2.

This module creates detailed visualizations showing how thermal data is preprocessed
and how heating/decay cycles are identified and filtered. It helps understand why
cycles are accepted or rejected during the analysis.

Key Features:
- Shows raw data with preprocessing steps
- Highlights identified heating and decay periods
- Annotates rejection reasons for filtered cycles
- Creates multi-panel plots for comprehensive analysis
- Focuses on winter data for better heating cycle visibility
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set up matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ThermalPreprocessingVisualizer:
    """Creates detailed visualizations of thermal data preprocessing and cycle identification."""

    def __init__(self, output_dir: Path = None):
        """Initialize the visualizer with output directory."""
        self.logger = logging.getLogger(f"{__name__}.ThermalPreprocessingVisualizer")
        self.output_dir = output_dir or Path("analysis/reports/preprocessing_viz")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme for different elements
        self.colors = {
            "raw_temp": "#1f77b4",  # Blue for raw temperature
            "processed_temp": "#ff7f0e",  # Orange for processed temperature
            "outdoor_temp": "#2ca02c",  # Green for outdoor temperature
            "heating_on": "#d62728",  # Red for heating periods
            "heating_off": "#9467bd",  # Purple for non-heating periods
            "accepted_decay": "#8c564b",  # Brown for accepted decay cycles
            "rejected_decay": "#e377c2",  # Pink for rejected decay cycles
            "accepted_rise": "#7f7f7f",  # Gray for accepted heating cycles
            "rejected_rise": "#bcbd22",  # Yellow-green for rejected heating cycles
            "outlier": "#ff0000",  # Red for outliers
            "interpolated": "#00ff00",  # Green for interpolated values
        }

    def visualize_room_preprocessing(
        self,
        room_name: str,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        relay_data: Optional[pd.DataFrame] = None,
        identified_cycles: Optional[Dict[str, List[Dict]]] = None,
        weather_data: Optional[pd.DataFrame] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Path:
        """
        Create comprehensive visualization of preprocessing for a single room.

        Args:
            room_name: Name of the room being analyzed
            raw_data: Original temperature data before preprocessing
            processed_data: Temperature data after preprocessing
            relay_data: Heating relay state data (optional)
            identified_cycles: Dictionary with 'heating' and 'decay' cycle lists
            weather_data: Outdoor temperature data (optional)
            date_range: Optional tuple of (start_date, end_date) to focus on

        Returns:
            Path to saved visualization file
        """
        # Filter data to date range if specified
        if date_range:
            start_date, end_date = date_range
            mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
            raw_data = raw_data[mask]
            mask = (processed_data.index >= start_date) & (
                processed_data.index <= end_date
            )
            processed_data = processed_data[mask]
            if relay_data is not None:
                mask = (relay_data.index >= start_date) & (relay_data.index <= end_date)
                relay_data = relay_data[mask]
            if weather_data is not None:
                mask = (weather_data.index >= start_date) & (
                    weather_data.index <= end_date
                )
                weather_data = weather_data[mask]

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(5, 1, height_ratios=[2, 2, 1, 2, 1], hspace=0.3)

        # Plot 1: Raw vs Processed Temperature
        ax1 = fig.add_subplot(gs[0])
        self._plot_temperature_comparison(ax1, raw_data, processed_data, weather_data)
        ax1.set_title(
            f"{room_name} - Temperature Data Processing", fontsize=14, fontweight="bold"
        )

        # Plot 2: Heating States and Temperature
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_heating_states(ax2, processed_data, relay_data)
        ax2.set_title(
            "Heating States and Temperature Response", fontsize=14, fontweight="bold"
        )

        # Plot 3: Temperature Derivatives
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_temperature_derivatives(ax3, processed_data)
        ax3.set_title(
            "Temperature Rate of Change (dT/dt)", fontsize=14, fontweight="bold"
        )

        # Plot 4: Identified Cycles
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        self._plot_identified_cycles(ax4, processed_data, identified_cycles, relay_data)
        ax4.set_title(
            "Identified Heating and Decay Cycles", fontsize=14, fontweight="bold"
        )

        # Plot 5: Cycle Statistics
        ax5 = fig.add_subplot(gs[4])
        self._plot_cycle_statistics(ax5, identified_cycles)
        ax5.set_title(
            "Cycle Acceptance/Rejection Statistics", fontsize=14, fontweight="bold"
        )

        # Format x-axis for all date plots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add overall title
        fig.suptitle(
            f"Thermal Preprocessing Analysis - {room_name}\n"
            f'Period: {raw_data.index[0].strftime("%Y-%m-%d")} to {raw_data.index[-1].strftime("%Y-%m-%d")}',
            fontsize=16,
            fontweight="bold",
        )

        # Save figure
        filename = (
            f"preprocessing_{room_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved preprocessing visualization to {filepath}")
        return filepath

    def _plot_temperature_comparison(
        self,
        ax: plt.Axes,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        weather_data: Optional[pd.DataFrame] = None,
    ):
        """Plot raw vs processed temperature data."""
        # Plot raw temperature
        if "room_temp" in raw_data.columns:
            ax.plot(
                raw_data.index,
                raw_data["room_temp"],
                color=self.colors["raw_temp"],
                alpha=0.5,
                linewidth=1,
                label="Raw Temperature",
            )

        # Plot processed temperature
        if "room_temp" in processed_data.columns:
            ax.plot(
                processed_data.index,
                processed_data["room_temp"],
                color=self.colors["processed_temp"],
                linewidth=2,
                label="Processed Temperature",
            )

        # Plot outdoor temperature on secondary axis if available
        if weather_data is not None and "outdoor_temp" in weather_data.columns:
            ax2 = ax.twinx()
            ax2.plot(
                weather_data.index,
                weather_data["outdoor_temp"],
                color=self.colors["outdoor_temp"],
                linewidth=1.5,
                alpha=0.7,
                label="Outdoor Temperature",
            )
            ax2.set_ylabel(
                "Outdoor Temperature (°C)", color=self.colors["outdoor_temp"]
            )
            ax2.tick_params(axis="y", labelcolor=self.colors["outdoor_temp"])
            ax2.legend(loc="upper right")

        # Highlight outliers and interpolated regions
        self._highlight_preprocessing_changes(ax, raw_data, processed_data)

        ax.set_ylabel("Room Temperature (°C)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_heating_states(
        self,
        ax: plt.Axes,
        processed_data: pd.DataFrame,
        relay_data: Optional[pd.DataFrame] = None,
    ):
        """Plot temperature with heating state overlay."""
        # Plot temperature
        if "room_temp" in processed_data.columns:
            ax.plot(
                processed_data.index,
                processed_data["room_temp"],
                color=self.colors["processed_temp"],
                linewidth=2,
                label="Room Temperature",
            )

        # Overlay heating states if available
        if relay_data is not None and "state" in relay_data.columns:
            # Create heating state regions
            heating_on = relay_data["state"] > 0.5

            # Find continuous heating periods
            heating_changes = heating_on.diff().fillna(False)
            heating_starts = relay_data.index[heating_changes & heating_on]
            heating_ends = relay_data.index[heating_changes & ~heating_on]

            # Ensure we have matching starts and ends
            if len(heating_starts) > len(heating_ends):
                heating_ends = list(heating_ends) + [relay_data.index[-1]]

            # Shade heating periods
            for start, end in zip(heating_starts, heating_ends):
                ax.axvspan(
                    start,
                    end,
                    alpha=0.3,
                    color=self.colors["heating_on"],
                    label="Heating On" if start == heating_starts[0] else "",
                )

        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    def _plot_temperature_derivatives(self, ax: plt.Axes, processed_data: pd.DataFrame):
        """Plot temperature rate of change."""
        if "room_temp" in processed_data.columns:
            # Calculate derivative (°C/hour)
            temp_series = processed_data["room_temp"]
            dt_hours = temp_series.index.to_series().diff().dt.total_seconds() / 3600
            dT = temp_series.diff()
            dT_dt = dT / dt_hours

            # Apply smoothing to reduce noise
            dT_dt_smooth = dT_dt.rolling(window=7, center=True).mean()

            # Plot derivative
            ax.plot(
                processed_data.index,
                dT_dt_smooth,
                color="black",
                linewidth=1.5,
                label="dT/dt",
            )

            # Add zero line
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            # Shade positive (heating) and negative (cooling) regions
            ax.fill_between(
                processed_data.index,
                0,
                dT_dt_smooth,
                where=(dT_dt_smooth > 0),
                interpolate=True,
                alpha=0.3,
                color=self.colors["heating_on"],
                label="Temperature Rising",
            )
            ax.fill_between(
                processed_data.index,
                0,
                dT_dt_smooth,
                where=(dT_dt_smooth < 0),
                interpolate=True,
                alpha=0.3,
                color=self.colors["heating_off"],
                label="Temperature Falling",
            )

        ax.set_ylabel("Temperature Change Rate (°C/hour)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    def _plot_identified_cycles(
        self,
        ax: plt.Axes,
        processed_data: pd.DataFrame,
        identified_cycles: Optional[Dict[str, List[Dict]]] = None,
        relay_data: Optional[pd.DataFrame] = None,
    ):
        """Plot identified heating and decay cycles with acceptance status."""
        # Plot temperature as background
        if "room_temp" in processed_data.columns:
            ax.plot(
                processed_data.index,
                processed_data["room_temp"],
                color="gray",
                linewidth=1,
                alpha=0.5,
                label="Temperature",
            )

        if identified_cycles:
            # Plot heating cycles
            if "heating" in identified_cycles:
                for cycle in identified_cycles["heating"]:
                    color = (
                        self.colors["accepted_rise"]
                        if cycle.get("accepted", True)
                        else self.colors["rejected_rise"]
                    )
                    alpha = 0.6 if cycle.get("accepted", True) else 0.3

                    ax.axvspan(
                        cycle["start_time"], cycle["end_time"], alpha=alpha, color=color
                    )

                    # Add rejection reason if rejected
                    if not cycle.get("accepted", True):
                        mid_time = (
                            cycle["start_time"]
                            + (cycle["end_time"] - cycle["start_time"]) / 2
                        )
                        ax.text(
                            mid_time,
                            ax.get_ylim()[1] * 0.95,
                            cycle.get("rejection_reason", "Unknown"),
                            rotation=90,
                            ha="center",
                            va="top",
                            fontsize=8,
                        )

            # Plot decay cycles
            if "decay" in identified_cycles:
                for cycle in identified_cycles["decay"]:
                    color = (
                        self.colors["accepted_decay"]
                        if cycle.get("accepted", True)
                        else self.colors["rejected_decay"]
                    )
                    alpha = 0.6 if cycle.get("accepted", True) else 0.3

                    ax.axvspan(
                        cycle["start_time"], cycle["end_time"], alpha=alpha, color=color
                    )

                    # Add rejection reason if rejected
                    if not cycle.get("accepted", True):
                        mid_time = (
                            cycle["start_time"]
                            + (cycle["end_time"] - cycle["start_time"]) / 2
                        )
                        ax.text(
                            mid_time,
                            ax.get_ylim()[0] * 1.05,
                            cycle.get("rejection_reason", "Unknown"),
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

        # Create legend
        legend_elements = [
            mpatches.Patch(
                color=self.colors["accepted_rise"], alpha=0.6, label="Accepted Heating"
            ),
            mpatches.Patch(
                color=self.colors["rejected_rise"], alpha=0.3, label="Rejected Heating"
            ),
            mpatches.Patch(
                color=self.colors["accepted_decay"], alpha=0.6, label="Accepted Decay"
            ),
            mpatches.Patch(
                color=self.colors["rejected_decay"], alpha=0.3, label="Rejected Decay"
            ),
        ]
        ax.legend(handles=legend_elements, loc="best")

        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, alpha=0.3)

    def _plot_cycle_statistics(
        self, ax: plt.Axes, identified_cycles: Optional[Dict[str, List[Dict]]] = None
    ):
        """Plot statistics about accepted and rejected cycles."""
        if not identified_cycles:
            ax.text(
                0.5,
                0.5,
                "No cycle data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.axis("off")
            return

        # Collect statistics
        stats = {
            "Heating Cycles": {"Accepted": 0, "Rejected": 0},
            "Decay Cycles": {"Accepted": 0, "Rejected": 0},
        }

        rejection_reasons = {"heating": {}, "decay": {}}

        if "heating" in identified_cycles:
            for cycle in identified_cycles["heating"]:
                if cycle.get("accepted", True):
                    stats["Heating Cycles"]["Accepted"] += 1
                else:
                    stats["Heating Cycles"]["Rejected"] += 1
                    reason = cycle.get("rejection_reason", "Unknown")
                    rejection_reasons["heating"][reason] = (
                        rejection_reasons["heating"].get(reason, 0) + 1
                    )

        if "decay" in identified_cycles:
            for cycle in identified_cycles["decay"]:
                if cycle.get("accepted", True):
                    stats["Decay Cycles"]["Accepted"] += 1
                else:
                    stats["Decay Cycles"]["Rejected"] += 1
                    reason = cycle.get("rejection_reason", "Unknown")
                    rejection_reasons["decay"][reason] = (
                        rejection_reasons["decay"].get(reason, 0) + 1
                    )

        # Create bar chart
        categories = list(stats.keys())
        accepted = [stats[cat]["Accepted"] for cat in categories]
        rejected = [stats[cat]["Rejected"] for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, accepted, width, label="Accepted", color="green", alpha=0.7
        )
        bars2 = ax.bar(
            x + width / 2, rejected, width, label="Rejected", color="red", alpha=0.7
        )

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                    )

        # Add rejection reason breakdown
        y_offset = max(accepted + rejected) * 1.1
        text_lines = ["Rejection Reasons:"]
        for cycle_type in ["heating", "decay"]:
            if rejection_reasons[cycle_type]:
                text_lines.append(f"\n{cycle_type.capitalize()}:")
                for reason, count in sorted(
                    rejection_reasons[cycle_type].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    text_lines.append(f"  {reason}: {count}")

        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Number of Cycles")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    def _highlight_preprocessing_changes(
        self, ax: plt.Axes, raw_data: pd.DataFrame, processed_data: pd.DataFrame
    ):
        """Highlight regions where preprocessing made changes."""
        if (
            "room_temp" not in raw_data.columns
            or "room_temp" not in processed_data.columns
        ):
            return

        # Find outliers (where raw and processed differ significantly)
        threshold = 0.1  # °C difference threshold

        # Align data on common index
        common_index = raw_data.index.intersection(processed_data.index)
        if len(common_index) == 0:
            return

        raw_aligned = raw_data.loc[common_index, "room_temp"]
        processed_aligned = processed_data.loc[common_index, "room_temp"]

        # Find differences
        diff = abs(raw_aligned - processed_aligned)
        outlier_mask = diff > threshold

        # Plot outliers as scatter points
        if outlier_mask.any():
            outlier_times = common_index[outlier_mask]
            outlier_values = raw_aligned[outlier_mask]
            ax.scatter(
                outlier_times,
                outlier_values,
                color=self.colors["outlier"],
                s=50,
                alpha=0.7,
                marker="x",
                label="Outliers Removed",
            )

        # Find interpolated regions (NaN in raw, not NaN in processed)
        raw_nan = raw_aligned.isna()
        processed_not_nan = ~processed_aligned.isna()
        interpolated_mask = raw_nan & processed_not_nan

        if interpolated_mask.any():
            # Group consecutive interpolated points
            interpolated_groups = []
            current_group = []

            for i, (time, is_interpolated) in enumerate(
                zip(common_index, interpolated_mask)
            ):
                if is_interpolated:
                    current_group.append(time)
                elif current_group:
                    interpolated_groups.append((current_group[0], current_group[-1]))
                    current_group = []

            if current_group:
                interpolated_groups.append((current_group[0], current_group[-1]))

            # Shade interpolated regions
            for start, end in interpolated_groups:
                ax.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=self.colors["interpolated"],
                    label="Interpolated" if start == interpolated_groups[0][0] else "",
                )


def create_winter_analysis_summary(
    rooms_data: Dict[str, pd.DataFrame],
    analysis_results: Dict[str, Any],
    output_dir: Path = None,
) -> Path:
    """
    Create a summary visualization of winter thermal analysis results.

    Args:
        rooms_data: Dictionary of room temperature data
        analysis_results: Results from thermal analysis
        output_dir: Directory to save visualization

    Returns:
        Path to saved summary file
    """
    output_dir = output_dir or Path("analysis/reports/preprocessing_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    # Plot 1: Cycle acceptance rates by room
    ax = axes[0]
    room_names = []
    acceptance_rates = []

    for room_name, result in analysis_results.items():
        if "thermal_parameters" in result:
            room_names.append(room_name)
            total_cycles = result.get("total_cycles_analyzed", 0)
            accepted_cycles = result.get("accepted_cycles", 0)
            rate = (accepted_cycles / total_cycles * 100) if total_cycles > 0 else 0
            acceptance_rates.append(rate)

    bars = ax.bar(room_names, acceptance_rates, color="skyblue", edgecolor="navy")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Decay Cycle Acceptance Rates by Room")
    ax.set_xticklabels(room_names, rotation=45, ha="right")

    # Add value labels
    for bar, rate in zip(bars, acceptance_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )

    # Plot 2: Common rejection reasons
    ax = axes[1]
    all_reasons = {}
    for result in analysis_results.values():
        if "rejection_reasons" in result:
            for reason, count in result["rejection_reasons"].items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count

    if all_reasons:
        reasons = list(all_reasons.keys())
        counts = list(all_reasons.values())

        # Sort by count
        sorted_pairs = sorted(zip(reasons, counts), key=lambda x: x[1], reverse=True)
        reasons, counts = zip(*sorted_pairs)

        bars = ax.barh(reasons[:10], counts[:10], color="coral")  # Top 10 reasons
        ax.set_xlabel("Number of Rejections")
        ax.set_title("Top 10 Rejection Reasons Across All Rooms")

        # Add count labels
        for bar, count in zip(bars, counts[:10]):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="left",
                va="center",
            )

    # Plot 3: Time constant distribution
    ax = axes[2]
    all_time_constants = []
    for result in analysis_results.values():
        if (
            "thermal_parameters" in result
            and "time_constant_hours" in result["thermal_parameters"]
        ):
            tc = result["thermal_parameters"]["time_constant_hours"]
            if tc and not np.isnan(tc):
                all_time_constants.append(tc)

    if all_time_constants:
        ax.hist(all_time_constants, bins=20, color="lightgreen", edgecolor="darkgreen")
        ax.set_xlabel("Time Constant (hours)")
        ax.set_ylabel("Number of Rooms")
        ax.set_title("Distribution of Thermal Time Constants")
        ax.axvline(
            np.median(all_time_constants),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(all_time_constants):.1f}h",
        )
        ax.legend()

    # Plot 4: Summary statistics
    ax = axes[3]
    ax.axis("off")

    # Calculate summary statistics
    total_rooms = len(analysis_results)
    successful_rooms = sum(
        1
        for r in analysis_results.values()
        if "thermal_parameters" in r and r["thermal_parameters"]
    )
    total_cycles = sum(
        r.get("total_cycles_analyzed", 0) for r in analysis_results.values()
    )
    accepted_cycles = sum(
        r.get("accepted_cycles", 0) for r in analysis_results.values()
    )

    summary_text = f"""
    Winter Thermal Analysis Summary
    ==============================
    
    Total Rooms Analyzed: {total_rooms}
    Successful RC Model Fits: {successful_rooms} ({successful_rooms/total_rooms*100:.1f}%)
    
    Total Decay Cycles Found: {total_cycles}
    Accepted Cycles: {accepted_cycles} ({(accepted_cycles/total_cycles*100 if total_cycles > 0 else 0):.1f}%)
    
    Average Time Constant: {np.mean(all_time_constants):.1f} hours
    Median Time Constant: {np.median(all_time_constants):.1f} hours
    Range: {min(all_time_constants):.1f} - {max(all_time_constants):.1f} hours
    """

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle("Winter Thermal Analysis Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save figure
    filename = f"winter_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath
