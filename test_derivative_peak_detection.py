#!/usr/bin/env python3
"""
Test the new derivative-based peak detection method for thermal analysis.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Add the pems_v2 directory to Python path
sys.path.append("/home/tom/git/LoxoneSmartHome/pems_v2")

from analysis.analyzers.thermal_analysis import ThermalAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_synthetic_thermal_data():
    """Create synthetic thermal data to test peak detection."""

    # Create time index (5-minute intervals for 6 hours)
    time_index = pd.date_range("2024-12-01 12:00", periods=72, freq="5min")

    # Create synthetic temperature profile
    # 1. Heating phase (0-60 min): temperature rises from 20°C to 22°C
    # 2. Thermal inertia (60-90 min): temperature continues to rise to 22.5°C after heating stops
    # 3. Decay phase (90-360 min): exponential decay back towards 20°C

    temps = []
    heating_on = []

    for i, t in enumerate(time_index):
        minutes = i * 5

        if minutes < 60:  # Heating on
            # Linear rise during heating
            temp = 20.0 + (2.0 * minutes / 60)
            heating_on.append(1)
        elif minutes < 90:  # Thermal inertia after heating stops
            # Continued rise due to thermal mass
            temp = 22.0 + 0.5 * (1 - np.exp(-(minutes - 60) / 15))
            heating_on.append(0)
        else:  # Decay phase
            # Exponential decay
            peak_temp = 22.5
            baseline = 20.0
            tau = 120  # 2-hour time constant
            temp = baseline + (peak_temp - baseline) * np.exp(-(minutes - 90) / tau)
            heating_on.append(0)

        # Add some realistic noise
        temp += np.random.normal(0, 0.05)
        temps.append(temp)

    df = pd.DataFrame(
        {
            "room_temp": temps,
            "heating_on": heating_on,
            "outdoor_temp": 5.0
            + np.random.normal(0, 0.2, len(temps)),  # Winter outdoor temp
        },
        index=time_index,
    )

    return df


def test_peak_detection():
    """Test the derivative-based peak detection method."""

    # Create synthetic data
    df = create_synthetic_thermal_data()

    # Find when heating turns off
    heating_changes = df["heating_on"].diff()
    heating_off_times = df[heating_changes == -1].index

    if len(heating_off_times) == 0:
        logger.error("No heating off transitions found")
        return

    relay_off_time = heating_off_times[0]
    logger.info(f"Heating turned off at: {relay_off_time}")

    # Create a mock analyzer to test the method
    class MockAnalyzer:
        def __init__(self):
            self.logger = logger

        def _find_thermal_peak_after_heating(
            self, df, relay_off_time, temp_col="room_temp"
        ):
            """Copy of the new peak detection method for testing."""
            search_window_hours = 2.0
            search_start_time = relay_off_time
            search_end_time = relay_off_time + pd.Timedelta(hours=search_window_hours)

            search_end_time = min(search_end_time, df.index[-1])
            search_data = df.loc[search_start_time:search_end_time]

            if len(search_data) < 5:
                self.logger.debug(
                    "Insufficient data in post-heating window to find peak."
                )
                return None, None

            temps = search_data[temp_col].dropna()
            if len(temps) < 5:
                self.logger.debug(
                    "Not enough valid temperature points in post-heating window."
                )
                return None, None

            window_length = min(len(temps), 7)
            if window_length < 3 or window_length % 2 == 0:
                window_length = max(3, window_length - 1) if window_length > 3 else 3

            if len(temps) < window_length:
                peak_temp = temps.max()
                peak_time = temps.idxmax()
                self.logger.debug(
                    f"Falling back to simple max() for peak detection due to insufficient data ({len(temps)} points)."
                )
                return peak_time, peak_temp

            try:
                # Calculate 1st derivative of temperature
                dt_dt = savgol_filter(
                    temps,
                    window_length=window_length,
                    polyorder=2,
                    deriv=1,
                    delta=5.0 / 60.0,
                )
                dt_dt_series = pd.Series(dt_dt, index=temps.index)

                peak_candidates = dt_dt_series[dt_dt_series < 0]

                if not peak_candidates.empty:
                    first_negative_idx = dt_dt_series.index.get_loc(
                        peak_candidates.index[0]
                    )
                    if first_negative_idx > 0:
                        peak_time = dt_dt_series.index[first_negative_idx - 1]
                        peak_temp = temps.loc[peak_time]

                        temp_at_relay_off = temps.iloc[0]
                        if peak_temp < temp_at_relay_off:
                            self.logger.debug(
                                f"Derivative-based peak ({peak_temp:.2f}°C) is lower than temp at relay off ({temp_at_relay_off:.2f}°C). "
                                f"Reverting to max() in window."
                            )
                            peak_temp = temps.max()
                            peak_time = temps.idxmax()

                        self.logger.debug(
                            f"Derivative-based peak found at {peak_time} with temp {peak_temp:.2f}°C."
                        )
                        return peak_time, peak_temp
                    else:
                        return temps.index[0], temps.iloc[0]

            except Exception as e:
                self.logger.warning(
                    f"Peak detection with derivative failed: {e}. Falling back to max()."
                )

            peak_temp = temps.max()
            peak_time = temps.idxmax()
            self.logger.debug("Using simple max() for peak detection as a fallback.")
            return peak_time, peak_temp

    # Test the method
    analyzer = MockAnalyzer()
    peak_time, peak_temp = analyzer._find_thermal_peak_after_heating(df, relay_off_time)

    if peak_time is not None:
        logger.info(
            f"Peak detected at: {peak_time} with temperature: {peak_temp:.2f}°C"
        )
        lag_minutes = (peak_time - relay_off_time).total_seconds() / 60
        logger.info(f"Thermal lag: {lag_minutes:.1f} minutes")

        # Visualize the results
        plt.figure(figsize=(12, 6))

        # Plot temperature
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df["room_temp"], "b-", label="Room Temperature")
        plt.axvline(relay_off_time, color="r", linestyle="--", label="Heating Off")
        plt.axvline(peak_time, color="g", linestyle="--", label="Peak Temperature")
        plt.scatter([peak_time], [peak_temp], color="g", s=100, zorder=5)
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.title("Temperature Profile with Peak Detection")
        plt.grid(True, alpha=0.3)

        # Plot heating state
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df["heating_on"], "r-", linewidth=2)
        plt.ylabel("Heating On")
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("thermal_peak_detection_test.png", dpi=150)
        logger.info("Saved visualization to thermal_peak_detection_test.png")

    else:
        logger.error("Failed to detect peak")


def test_with_real_data():
    """Test with real data from the processed files."""

    # Load real data
    room_name = "obyvak"
    room_file = f"pems_v2/data/processed/rooms_{room_name}_processed.parquet"
    relay_file = f"pems_v2/data/processed/relay_states_{room_name}_processed.parquet"

    try:
        room_df = pd.read_parquet(room_file)
        relay_df = pd.read_parquet(relay_file)

        # Check date range
        logger.info(f"Room data range: {room_df.index.min()} to {room_df.index.max()}")
        logger.info(
            f"Relay data range: {relay_df.index.min()} to {relay_df.index.max()}"
        )

        # Find a recent date with heating cycles
        # Look for days with significant relay activity
        relay_daily = relay_df.resample("D").sum()
        active_days = relay_daily[
            relay_daily["value"] > 60
        ].index  # At least 5 hours of heating

        if len(active_days) > 0:
            test_date = active_days[-1]  # Use most recent active day
            logger.info(f"Testing with data from {test_date.date()}")

            # Extract day's data
            start_time = test_date.replace(hour=0, minute=0, second=0)
            end_time = start_time + pd.Timedelta(days=1)

            room_day = room_df.loc[start_time:end_time].copy()
            relay_day = relay_df.loc[start_time:end_time].copy()

            # Merge data
            room_day["heating_on"] = (
                relay_day["value"].reindex(room_day.index, method="nearest").fillna(0)
            )

            # Find heating off transitions
            heating_changes = room_day["heating_on"].diff()
            heating_off_times = room_day[heating_changes == -1].index

            if len(heating_off_times) > 0:
                logger.info(f"Found {len(heating_off_times)} heating off transitions")
                # Test with first transition
                test_derivative_peak_on_real_data(room_day, heating_off_times[0])
            else:
                logger.warning("No heating off transitions found in selected day")

    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        logger.info("Falling back to synthetic data test")


def test_derivative_peak_on_real_data(df, relay_off_time):
    """Test peak detection on real data segment."""

    # Rename columns to match expected format
    if "temperature" in df.columns:
        df["room_temp"] = df["temperature"]

    # Use the mock analyzer
    class MockAnalyzer:
        def __init__(self):
            self.logger = logger

        # Include the full method implementation here
        def _find_thermal_peak_after_heating(
            self, df, relay_off_time, temp_col="room_temp"
        ):
            """Copy of the new peak detection method."""
            search_window_hours = 2.0
            search_start_time = relay_off_time
            search_end_time = relay_off_time + pd.Timedelta(hours=search_window_hours)

            search_end_time = min(search_end_time, df.index[-1])
            search_data = df.loc[search_start_time:search_end_time]

            if len(search_data) < 5:
                self.logger.debug(
                    "Insufficient data in post-heating window to find peak."
                )
                return None, None

            temps = search_data[temp_col].dropna()
            if len(temps) < 5:
                self.logger.debug(
                    "Not enough valid temperature points in post-heating window."
                )
                return None, None

            window_length = min(len(temps), 7)
            if window_length < 3 or window_length % 2 == 0:
                window_length = max(3, window_length - 1) if window_length > 3 else 3

            try:
                dt_dt = savgol_filter(
                    temps,
                    window_length=window_length,
                    polyorder=2,
                    deriv=1,
                    delta=5.0 / 60.0,
                )
                dt_dt_series = pd.Series(dt_dt, index=temps.index)

                peak_candidates = dt_dt_series[dt_dt_series < 0]

                if not peak_candidates.empty:
                    first_negative_idx = dt_dt_series.index.get_loc(
                        peak_candidates.index[0]
                    )
                    if first_negative_idx > 0:
                        peak_time = dt_dt_series.index[first_negative_idx - 1]
                        peak_temp = temps.loc[peak_time]

                        self.logger.debug(
                            f"Derivative-based peak found at {peak_time} with temp {peak_temp:.2f}°C."
                        )
                        return peak_time, peak_temp

            except Exception as e:
                self.logger.warning(f"Peak detection failed: {e}")

            peak_temp = temps.max()
            peak_time = temps.idxmax()
            return peak_time, peak_temp

    analyzer = MockAnalyzer()
    peak_time, peak_temp = analyzer._find_thermal_peak_after_heating(df, relay_off_time)

    if peak_time:
        logger.info(f"Real data peak at {peak_time}, temp={peak_temp:.2f}°C")
        logger.info(
            f"Thermal lag: {(peak_time - relay_off_time).total_seconds()/60:.1f} minutes"
        )


if __name__ == "__main__":
    # Test with synthetic data first
    logger.info("=== Testing with Synthetic Data ===")
    test_peak_detection()

    # Then test with real data
    logger.info("\n=== Testing with Real Data ===")
    test_with_real_data()
