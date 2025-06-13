#!/usr/bin/env python3
"""
Test the fixed thermal analysis on obývak data for Jan 20-23, 2025.
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_fixed_thermal_analysis():
    """Test the fixed thermal analysis on obývak data."""

    logger.info("=== Testing Fixed Thermal Analysis ===")

    # Date range for testing
    start_date = pd.Timestamp("2025-01-20", tz="UTC")
    end_date = pd.Timestamp("2025-01-24", tz="UTC")

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Room: obývak")

    # Load the processed data files directly (like the working analysis does)
    pems_data_dir = Path("pems_v2/data/processed")

    try:
        # Load obývak room data
        obyvak_file = pems_data_dir / "rooms_obyvak_processed.parquet"
        room_data = pd.read_parquet(obyvak_file)
        logger.info(f"Loaded obývak room data: {len(room_data)} records")

        # Load outdoor temperature data
        outdoor_file = pems_data_dir / "rooms_outside_processed.parquet"
        outdoor_data = pd.read_parquet(outdoor_file)
        logger.info(f"Loaded outdoor data: {len(outdoor_data)} records")

        # Load relay data
        relay_file = pems_data_dir / "relay_states_obyvak_processed.parquet"
        relay_data = pd.read_parquet(relay_file)
        logger.info(f"Loaded relay data: {len(relay_data)} records")

        # Filter to date range
        date_mask = (room_data.index >= start_date) & (room_data.index <= end_date)
        room_filtered = room_data[date_mask]

        date_mask = (outdoor_data.index >= start_date) & (
            outdoor_data.index <= end_date
        )
        outdoor_filtered = outdoor_data[date_mask]

        date_mask = (relay_data.index >= start_date) & (relay_data.index <= end_date)
        relay_filtered = relay_data[date_mask]

        logger.info(f"Filtered to date range:")
        logger.info(f"Room: {len(room_filtered)} records")
        logger.info(f"Outdoor: {len(outdoor_filtered)} records")
        logger.info(f"Relay: {len(relay_filtered)} records")

        # Prepare thermal analysis data manually (similar to working analysis)
        # Start with room data and merge outdoor temp
        thermal_data = room_filtered.join(
            outdoor_filtered["temperature"].rename("outdoor_temp"), how="left"
        )
        thermal_data["outdoor_temp"] = thermal_data["outdoor_temp"].ffill()

        # Add relay state as heating_on column
        relay_series = (
            relay_filtered["relay_state"].reindex(thermal_data.index).ffill().fillna(0)
        )
        thermal_data["heating_on"] = (relay_series > 0.5).astype(int)

        logger.info(f"Prepared thermal analysis data: {len(thermal_data)} records")
        logger.info(f"Columns: {list(thermal_data.columns)}")

        # Check heating transitions
        heating_transitions = thermal_data["heating_on"].diff() != 0
        heating_changes = thermal_data[heating_transitions]

        logger.info(f"\nHeating transitions:")
        for i, (timestamp, row) in enumerate(heating_changes.iterrows()):
            state = "ON" if row["heating_on"] > 0.5 else "OFF"
            logger.info(
                f"  {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M:%S')} -> {state}"
            )

        # Initialize thermal analyzer
        analyzer = ThermalAnalyzer()

        # Test heating cycle detection
        cycles = analyzer._detect_heating_cycles(thermal_data)
        logger.info(f"\nDetected heating cycles: {len(cycles)}")

        if len(cycles) > 0:
            logger.info("\nCycle details:")
            for i, cycle in enumerate(cycles):
                logger.info(f"  Cycle {i+1}:")
                logger.info(f"    Start: {cycle['start_time']}")
                logger.info(f"    End: {cycle['end_time']}")
                logger.info(f"    Duration: {cycle['duration_minutes']/60:.1f} hours")
                logger.info(f"    Peak temp: {cycle['peak_temp']:.2f}°C")

                # Test decay analysis on this cycle
                next_cycle_start = (
                    cycles[i + 1]["start_time"] if i + 1 < len(cycles) else None
                )

                logger.info(f"\n  Testing decay analysis on Cycle {i+1}...")
                try:
                    decay_result = analyzer._analyze_heating_cycle_decay(
                        thermal_data, cycle, next_cycle_start
                    )

                    if decay_result and decay_result.get("fit_valid", False):
                        logger.info(f"    ✅ Decay analysis SUCCESSFUL!")
                        logger.info(
                            f"    R²: {decay_result.get('r_squared', 'N/A'):.3f}"
                        )
                        logger.info(
                            f"    Time constant: {decay_result.get('time_constant', 'N/A'):.1f}h"
                        )
                        logger.info(
                            f"    Decay magnitude: {decay_result.get('decay_magnitude', 'N/A'):.2f}°C"
                        )
                    else:
                        logger.warning(
                            f"    ❌ Decay analysis REJECTED: {decay_result.get('reason', 'unknown')}"
                        )

                except Exception as e:
                    logger.error(f"    ❌ Decay analysis FAILED: {e}")

        else:
            logger.warning("No heating cycles detected!")

            # Debug: Check why no cycles were detected
            logger.info("\nDebugging cycle detection:")
            logger.info(
                f"Unique heating_on values: {thermal_data['heating_on'].unique()}"
            )

            heating_diff = thermal_data["heating_on"].diff()
            start_indices = thermal_data.index[heating_diff == 1]
            end_indices = thermal_data.index[heating_diff == -1]

            logger.info(f"Heating starts found: {len(start_indices)}")
            logger.info(f"Heating ends found: {len(end_indices)}")

            if len(start_indices) > 0:
                logger.info(f"First few heating starts: {start_indices[:3].tolist()}")
            if len(end_indices) > 0:
                logger.info(f"First few heating ends: {end_indices[:3].tolist()}")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test_fixed_thermal_analysis()
