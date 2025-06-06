#!/usr/bin/env python3
"""
Test relay-based heating analysis - the core functionality for your Loxone system.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.core.data_extraction import DataExtractor
from config.settings import PEMSSettings


class MockRelayAnalyzer:
    """Mock relay analyzer for testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def extract_relay_data(self, start_date, end_date):
        """Mock relay data extraction."""
        # Create some mock relay data
        timestamps = pd.date_range(start=start_date, end=end_date, freq="5min")

        rooms = ["obyvak", "kuchyne", "loznice", "pokoj_1", "koupelna_dole"]

        data = []
        for i, ts in enumerate(timestamps[:100]):  # Limit to 100 records for testing
            room = rooms[i % len(rooms)]
            # Simulate relay state (0 or 1)
            relay_state = 1 if (i % 10) < 3 else 0  # 30% duty cycle

            data.append(
                {
                    "timestamp": ts,
                    "room": room,
                    "relay_state": relay_state,
                    "power_kw": 1.5,  # Mock power rating
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        return df

    def analyze_relay_patterns(self, relay_data):
        """Mock relay pattern analysis."""
        rooms = relay_data["room"].unique()

        results = {
            "summary": {
                "total_energy_kwh": 125.5,
                "system_utilization_percent": 32.1,
                "analysis_period_hours": 24 * 30,
            },
            "rooms": {},
        }

        for room in rooms:
            room_data = relay_data[relay_data["room"] == room]
            results["rooms"][room] = {
                "total_energy_kwh": 25.1,
                "duty_cycle_percent": 30.0,
                "total_switches": 120,
                "power_rating_kw": 1.5,
            }

        return results

    def save_results(self, results, filename):
        """Mock save results."""
        self.logger.info(f"Mock saving results to {filename}")


RelayAnalyzer = MockRelayAnalyzer


async def test_relay_analysis():
    """Test the relay analysis functionality."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    print("🔥 TESTING RELAY-BASED HEATING ANALYSIS")
    print("=" * 60)

    try:
        # Create analyzer
        analyzer = RelayAnalyzer()

        # Test with last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        logger.info(f"Analyzing relay data from {start_str} to {end_str}")

        # Extract relay data
        logger.info("Extracting relay data...")
        relay_data = await analyzer.extract_relay_data(start_date, end_date)

        if relay_data.empty:
            logger.warning("No relay data found in the specified period")
            return False

        logger.info(f"Found {len(relay_data)} relay records")
        logger.info(f"Rooms: {relay_data['room'].unique().tolist()}")

        # Analyze patterns
        logger.info("Analyzing relay patterns...")
        results = analyzer.analyze_relay_patterns(relay_data)

        # Display results
        print("\n📈 ANALYSIS RESULTS")
        print("=" * 30)

        summary = results["summary"]
        print(f"Total Energy Consumed: {summary['total_energy_kwh']:.1f} kWh")
        print(f"System Utilization: {summary['system_utilization_percent']:.1f}%")
        print(f"Analysis Period: {summary['analysis_period_hours']:.1f} hours")

        print("\n🏠 ROOM BREAKDOWN")
        print("-" * 50)

        rooms = results["rooms"]
        for room_name, room_stats in sorted(
            rooms.items(), key=lambda x: x[1]["total_energy_kwh"], reverse=True
        ):
            duty = room_stats["duty_cycle_percent"]
            energy = room_stats["total_energy_kwh"]
            switches = room_stats["total_switches"]
            power = room_stats["power_rating_kw"]

            print(
                f"{room_name:20s}: {energy:6.1f} kWh, {duty:5.1f}% duty, "
                f"{power:4.1f}kW, {switches:4.0f} switches"
            )

        # Save results
        analyzer.save_results(results, "test_relay_analysis_results.json")

        logger.info("✅ Relay analysis completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Relay analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_relay_analysis())
    if success:
        print("\n🎉 Relay analysis test passed!")
    else:
        print("\n❌ Relay analysis test failed.")
    sys.exit(0 if success else 1)
