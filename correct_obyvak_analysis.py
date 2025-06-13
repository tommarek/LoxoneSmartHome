#!/usr/bin/env python3
"""
Correctly analyze obývak thermal cycles using proper relay timing.
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("correct_obyvak_analysis.log"),
    ],
)

logger = logging.getLogger(__name__)


def correct_obyvak_analysis():
    """Correctly analyze obývak thermal cycles using proper relay timing."""

    logger.info("=== CORRECTED OBÝVAK THERMAL ANALYSIS ===")

    # Load the processed data
    pems_data_dir = Path("pems_v2/data/processed")

    try:
        # Load obývak room data
        obyvak_file = pems_data_dir / "rooms_obyvak_processed.parquet"
        room_data = pd.read_parquet(obyvak_file)
        logger.info(f"Loaded obývak room data: {len(room_data)} records")
        logger.info(f"Time range: {room_data.index.min()} to {room_data.index.max()}")

        # Load outdoor temperature data
        outdoor_file = pems_data_dir / "outdoor_temp_processed.parquet"
        outdoor_data = pd.read_parquet(outdoor_file)
        logger.info(f"Loaded outdoor data: {len(outdoor_data)} records")

        # Load relay data - THIS IS THE KEY!
        relay_file = pems_data_dir / "relay_states_obyvak_processed.parquet"
        relay_data = pd.read_parquet(relay_file)
        logger.info(f"Loaded relay data: {len(relay_data)} records")

        # Show relay timing first
        logger.info("\n=== ACTUAL RELAY TIMING ===")
        heating_cycles = []
        current_cycle = None

        for i, (timestamp, row) in enumerate(relay_data.iterrows()):
            state = "ON" if row["relay_state"] == 1.0 else "OFF"
            logger.info(f"{i+1:2d}. {timestamp} - {state}")

            if row["relay_state"] == 1.0:  # Heating starts
                if current_cycle is None:
                    current_cycle = {"start": timestamp}
            elif row["relay_state"] == 0.0:  # Heating ends
                if current_cycle is not None:
                    current_cycle["end"] = timestamp
                    heating_cycles.append(current_cycle)
                    current_cycle = None

        logger.info(
            f"\n=== IDENTIFIED {len(heating_cycles)} COMPLETE HEATING CYCLES ==="
        )

        for i, cycle in enumerate(heating_cycles):
            cycle_start = cycle["start"]
            cycle_end = cycle["end"]
            duration_hours = (cycle_end - cycle_start).total_seconds() / 3600

            logger.info(f"\nCYCLE {i+1}: {cycle_start} to {cycle_end}")
            logger.info(f"Duration: {duration_hours:.1f} hours")

            # Get room temperature data during heating
            heating_mask = (room_data.index >= cycle_start) & (
                room_data.index <= cycle_end
            )
            heating_temp_data = room_data[heating_mask]

            # Get temperature data for analysis
            valid_heating = heating_temp_data.dropna(subset=["temperature"])

            if len(valid_heating) >= 2:
                temp_start = valid_heating["temperature"].iloc[0]
                temp_end = valid_heating["temperature"].iloc[-1]
                temp_rise = temp_end - temp_start
                logger.info(
                    f"Temperature rise: {temp_start:.1f}°C → {temp_end:.1f}°C (Δ{temp_rise:.2f}°C)"
                )
                logger.info(
                    f"Valid temperature data points during heating: {len(valid_heating)}"
                )
            else:
                logger.warning(
                    f"Insufficient temperature data during heating: {len(valid_heating)} points"
                )
                temp_end = None

            # Analyze decay period AFTER heating ends
            next_cycle_start = (
                heating_cycles[i + 1]["start"] if i + 1 < len(heating_cycles) else None
            )

            # Set decay end (next heating or max 8 hours - longer window)
            max_decay_hours = 8
            max_decay_end = cycle_end + pd.Timedelta(hours=max_decay_hours)

            if next_cycle_start and next_cycle_start < max_decay_end:
                decay_end = next_cycle_start
                decay_reason = "next heating cycle"
            else:
                decay_end = max_decay_end
                decay_reason = f"max {max_decay_hours} hours"

            logger.info(f"Decay period: {cycle_end} to {decay_end} ({decay_reason})")

            # Get decay phase data
            decay_mask = (room_data.index > cycle_end) & (room_data.index <= decay_end)
            decay_temp_data = room_data[decay_mask]

            # Merge with outdoor temperature for decay analysis
            decay_with_outdoor = decay_temp_data.join(
                outdoor_data["outdoor_temp"], how="left"
            )
            decay_with_outdoor["outdoor_temp"] = decay_with_outdoor[
                "outdoor_temp"
            ].ffill()

            # Get valid decay data
            valid_decay = decay_with_outdoor.dropna(
                subset=["temperature", "outdoor_temp"]
            )

            decay_duration_hours = (decay_end - cycle_end).total_seconds() / 3600
            logger.info(f"Decay duration: {decay_duration_hours:.1f} hours")
            logger.info(f"Total decay data points: {len(decay_temp_data)}")
            logger.info(f"Valid decay data points: {len(valid_decay)}")

            # Analyze decay rejection criteria
            rejection_reasons = []

            # 1. Minimum data points (5)
            if len(valid_decay) < 5:
                rejection_reasons.append(
                    f"insufficient_data ({len(valid_decay)} < 5 points)"
                )

            if len(valid_decay) >= 2 and temp_end is not None:
                # 2. Decay magnitude
                decay_start_temp = temp_end  # End of heating is start of decay
                decay_end_temp = valid_decay["temperature"].iloc[-1]
                decay_magnitude = decay_start_temp - decay_end_temp

                logger.info(
                    f"Decay magnitude: {decay_magnitude:.2f}°C ({decay_start_temp:.1f} → {decay_end_temp:.1f})"
                )

                # More relaxed threshold for winter conditions
                min_decay_magnitude = 0.3  # Reduced from 1.0°C
                if decay_magnitude < min_decay_magnitude:
                    rejection_reasons.append(
                        f"insufficient_decay_magnitude ({decay_magnitude:.2f}°C < {min_decay_magnitude}°C)"
                    )

                # 3. Exponential fit quality
                try:
                    from scipy.optimize import curve_fit

                    def exponential_decay(t, delta_T0, tau):
                        return delta_T0 * np.exp(-t / tau)

                    # Calculate ΔT (room - outdoor)
                    valid_decay_copy = valid_decay.copy()
                    valid_decay_copy["delta_T"] = (
                        valid_decay_copy["temperature"]
                        - valid_decay_copy["outdoor_temp"]
                    )

                    # Convert timestamps to hours from start
                    start_time = valid_decay_copy.index[0]
                    time_hours = [
                        (t - start_time).total_seconds() / 3600
                        for t in valid_decay_copy.index
                    ]
                    delta_T_values = valid_decay_copy["delta_T"].values

                    logger.info(
                        f"ΔT range: {delta_T_values.min():.1f} to {delta_T_values.max():.1f}°C"
                    )

                    # Initial guess
                    initial_delta_T = delta_T_values[0]
                    initial_tau = 24.0  # hours - more realistic for winter

                    # Fit with relaxed bounds
                    tau_min, tau_max = 1.0, 200.0  # Wider range
                    bounds = ([0, tau_min], [np.inf, tau_max])

                    popt, pcov = curve_fit(
                        exponential_decay,
                        time_hours,
                        delta_T_values,
                        p0=[initial_delta_T, initial_tau],
                        bounds=bounds,
                        maxfev=2000,
                    )

                    fitted_delta_T0, fitted_tau = popt

                    # Calculate R²
                    y_pred = exponential_decay(
                        np.array(time_hours), fitted_delta_T0, fitted_tau
                    )
                    ss_res = np.sum((delta_T_values - y_pred) ** 2)
                    ss_tot = np.sum((delta_T_values - np.mean(delta_T_values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    logger.info(
                        f"Exponential fit: τ = {fitted_tau:.1f}h, ΔT₀ = {fitted_delta_T0:.1f}°C, R² = {r_squared:.3f}"
                    )

                    # More relaxed R² threshold for winter data
                    min_r_squared = 0.2  # Reduced from 0.4
                    if r_squared < min_r_squared:
                        rejection_reasons.append(
                            f"poor_fit_quality (R² = {r_squared:.3f} < {min_r_squared})"
                        )

                except Exception as e:
                    rejection_reasons.append(f"fitting_failed ({str(e)})")

            # Summary
            if rejection_reasons:
                logger.warning(
                    f"Cycle {i+1}: ❌ REJECTED - {', '.join(rejection_reasons)}"
                )
            else:
                logger.info(f"Cycle {i+1}: ✅ WOULD BE ACCEPTED with relaxed criteria!")

        logger.info(f"\n=== SUMMARY ===")
        logger.info(f"Total complete heating cycles: {len(heating_cycles)}")
        logger.info(f"Analysis used relaxed winter criteria:")
        logger.info(f"  - Minimum decay magnitude: 0.3°C (was 1.0°C)")
        logger.info(f"  - Minimum R²: 0.2 (was 0.4)")
        logger.info(f"  - Extended decay window: 8 hours (was 6 hours)")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    correct_obyvak_analysis()
