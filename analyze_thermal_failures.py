#!/usr/bin/env python3
"""
Analyze why thermal decay analysis is failing for most cycles.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add the pems_v2 directory to Python path
sys.path.append("/home/tom/git/LoxoneSmartHome/pems_v2")

from analysis.analyzers.thermal_analysis import ThermalAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class MockThermalAnalyzer(ThermalAnalyzer):
    """Mock thermal analyzer for testing methods without full initialization."""

    def __init__(self):
        # Initialize with minimal required attributes
        self.settings = {
            "decay_analysis_hours": 8.0,
            "min_heating_duration_minutes": 5,
            "max_heating_duration_minutes": 2880,
        }
        self.system_settings = {}
        self.logger = logger

    def _get_room_power_rating_watts(self):
        # Return a default power rating
        return 1800  # watts


def analyze_thermal_failures():
    """Analyze why thermal decay analysis is failing for most cycles."""

    logger.info("=== Analyzing Thermal Analysis Failures ===")

    # Load satna_dole data as an example
    pems_data_dir = Path("pems_v2/data/processed")

    try:
        # Load room data
        room_file = pems_data_dir / "rooms_satna_dole_processed.parquet"
        room_data = pd.read_parquet(room_file)
        logger.info(f"Loaded satna_dole room data: {len(room_data)} records")

        # Load outdoor temperature data
        outdoor_file = pems_data_dir / "rooms_outside_processed.parquet"
        outdoor_data = pd.read_parquet(outdoor_file)
        logger.info(f"Loaded outdoor data: {len(outdoor_data)} records")

        # Load relay data
        relay_file = pems_data_dir / "relay_states_satna_dole_processed.parquet"
        relay_data = pd.read_parquet(relay_file)
        logger.info(f"Loaded relay data: {len(relay_data)} records")

        # Sample a week of data for detailed analysis
        sample_start = pd.Timestamp("2025-01-20", tz="UTC")
        sample_end = pd.Timestamp("2025-01-27", tz="UTC")

        # Filter to sample period
        date_mask = (room_data.index >= sample_start) & (room_data.index <= sample_end)
        room_sample = room_data[date_mask]

        date_mask = (outdoor_data.index >= sample_start) & (
            outdoor_data.index <= sample_end
        )
        outdoor_sample = outdoor_data[date_mask]

        date_mask = (relay_data.index >= sample_start) & (
            relay_data.index <= sample_end
        )
        relay_sample = relay_data[date_mask]

        logger.info(f"\nSample period analysis ({sample_start} to {sample_end}):")
        logger.info(f"Room: {len(room_sample)} records")
        logger.info(f"Outdoor: {len(outdoor_sample)} records")
        logger.info(f"Relay: {len(relay_sample)} records")

        # Prepare thermal analysis data
        thermal_data = room_sample.join(
            outdoor_sample["temperature"].rename("outdoor_temp"), how="left"
        )
        thermal_data["outdoor_temp"] = thermal_data["outdoor_temp"].ffill()

        # Add relay state as heating_on column
        relay_series = (
            relay_sample["relay_state"].reindex(thermal_data.index).ffill().fillna(0)
        )
        thermal_data["heating_on"] = (relay_series > 0.5).astype(int)

        # Initialize mock thermal analyzer
        analyzer = MockThermalAnalyzer()

        # Test heating cycle detection
        cycles = analyzer._detect_heating_cycles(thermal_data)
        logger.info(f"\nDetected {len(cycles)} heating cycles in sample period")

        # Analyze failure reasons
        failure_reasons = {}
        successful_decays = []

        for i, cycle in enumerate(cycles):
            logger.debug(f"\n--- Analyzing Cycle {i+1} ---")
            logger.debug(f"Start: {cycle['start_time']}")
            logger.debug(f"End: {cycle['end_time']}")
            logger.debug(f"Duration: {cycle['duration_minutes']/60:.1f} hours")
            logger.debug(f"Peak temp: {cycle['peak_temp']:.2f}°C")

            # Get cycle data
            cycle_data = thermal_data.loc[cycle["start_time"] : cycle["end_time"]]
            temp_rise = cycle["peak_temp"] - cycle["start_temp"]
            logger.debug(f"Temperature rise: {temp_rise:.2f}°C")

            # Check outdoor conditions
            outdoor_avg = cycle.get("outdoor_temp_avg", None)
            if outdoor_avg is not None:
                logger.debug(f"Outdoor temp avg: {outdoor_avg:.1f}°C")

            # Test decay analysis
            next_cycle_start = (
                cycles[i + 1]["start_time"] if i + 1 < len(cycles) else None
            )

            try:
                decay_result = analyzer._analyze_heating_cycle_decay(
                    thermal_data, cycle, next_cycle_start
                )

                if decay_result and decay_result.get("fit_valid", False):
                    successful_decays.append(
                        {
                            "cycle": i + 1,
                            "duration_hours": cycle["duration_minutes"] / 60,
                            "temp_rise": temp_rise,
                            "outdoor_temp": outdoor_avg,
                            "r_squared": decay_result.get("r_squared"),
                            "time_constant": decay_result.get("time_constant"),
                            "decay_magnitude": decay_result.get("decay_magnitude"),
                        }
                    )
                    logger.debug(f"✅ SUCCESSFUL DECAY")
                else:
                    reason = (
                        decay_result.get("reason", "unknown")
                        if decay_result
                        else "exception"
                    )
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    logger.debug(f"❌ FAILED: {reason}")

                    # Additional debugging for common failures
                    if reason == "insufficient_decay_magnitude":
                        logger.debug(
                            f"  Decay magnitude: {decay_result.get('decay_magnitude', 'N/A')}°C"
                        )
                    elif reason == "poor_fit_quality":
                        logger.debug(f"  R²: {decay_result.get('r_squared', 'N/A')}")

            except Exception as e:
                failure_reasons["exception"] = failure_reasons.get("exception", 0) + 1
                logger.debug(f"❌ EXCEPTION: {e}")

        # Summary statistics
        logger.info(f"\n=== FAILURE ANALYSIS SUMMARY ===")
        logger.info(f"Total cycles analyzed: {len(cycles)}")
        logger.info(
            f"Successful decays: {len(successful_decays)} ({len(successful_decays)/len(cycles)*100:.1f}%)"
        )

        logger.info(f"\nFailure reasons:")
        for reason, count in sorted(
            failure_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"  {reason}: {count} ({count/len(cycles)*100:.1f}%)")

        if successful_decays:
            logger.info(f"\nSuccessful decay characteristics:")
            df_success = pd.DataFrame(successful_decays)
            logger.info(
                f"  Duration: {df_success['duration_hours'].mean():.1f} ± {df_success['duration_hours'].std():.1f} hours"
            )
            logger.info(
                f"  Temp rise: {df_success['temp_rise'].mean():.2f} ± {df_success['temp_rise'].std():.2f}°C"
            )
            logger.info(
                f"  Decay magnitude: {df_success['decay_magnitude'].mean():.2f} ± {df_success['decay_magnitude'].std():.2f}°C"
            )
            logger.info(
                f"  R²: {df_success['r_squared'].mean():.3f} ± {df_success['r_squared'].std():.3f}"
            )

            if "outdoor_temp" in df_success.columns:
                outdoor_temps = df_success["outdoor_temp"].dropna()
                if len(outdoor_temps) > 0:
                    logger.info(
                        f"  Outdoor temp: {outdoor_temps.mean():.1f} ± {outdoor_temps.std():.1f}°C"
                    )

        # Analyze timing patterns
        if len(cycles) > 1:
            logger.info(f"\nTiming patterns:")
            durations = [c["duration_minutes"] / 60 for c in cycles]
            logger.info(
                f"  Heating duration: {np.mean(durations):.1f} ± {np.std(durations):.1f} hours"
            )
            logger.info(f"  Min duration: {np.min(durations):.1f} hours")
            logger.info(f"  Max duration: {np.max(durations):.1f} hours")

            # Inter-cycle gaps
            gaps = []
            for i in range(len(cycles) - 1):
                gap = (
                    cycles[i + 1]["start_time"] - cycles[i]["end_time"]
                ).total_seconds() / 3600
                gaps.append(gap)

            if gaps:
                logger.info(
                    f"  Inter-cycle gap: {np.mean(gaps):.1f} ± {np.std(gaps):.1f} hours"
                )
                logger.info(f"  Min gap: {np.min(gaps):.1f} hours")
                logger.info(f"  Max gap: {np.max(gaps):.1f} hours")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    analyze_thermal_failures()
