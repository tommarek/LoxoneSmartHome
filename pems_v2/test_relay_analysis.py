#!/usr/bin/env python3
"""
Test the corrected relay-based heating analysis with proper understanding.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
from influxdb_client import InfluxDBClient

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))


class RelayAnalyzer:
    """Analyze relay-based heating patterns correctly."""

    def __init__(self):
        """Initialize the analyzer."""
        self.logger = logging.getLogger(f"{__name__}.RelayAnalyzer")

        # InfluxDB configuration
        self.url = "http://192.168.0.201:8086"
        self.token = "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A=="
        self.org = "loxone"
        self.bucket = "loxone"

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=30000)
        self.query_api = self.client.query_api()

        # Room heating power mapping (kW) - actual power when relay is ON
        self.heating_power = {
            "hosti": 2.02,
            "chodba_dole": 1.8,
            "chodba_nahore": 1.2,
            "koupelna_dole": 0.47,
            "koupelna_nahore": 0.62,
            "loznice": 1.2,
            "obyvak": 4.8,
            "pokoj_1": 1.2,
            "pokoj_2": 1.2,
            "pracovna": 0.82,
            "satna_dole": 0.82,
            "satna_nahore": 0.56,
            "spajz": 0.46,
            "technicka_mistnost": 0.82,
            "zadveri": 0.82,
            "zachod": 0.22,
        }

    def __del__(self):
        """Close InfluxDB client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()

    async def extract_relay_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract relay on/off states from InfluxDB."""
        self.logger.info(f"Extracting relay data from {start_date} to {end_date}")

        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating" and r.room != "kuchyne")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "room"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No relay data found")
            return pd.DataFrame()

        # Process relay data
        relay_data = []
        for table in tables:
            for record in table.records:
                room = record.values.get("room", "unknown")
                if room in self.heating_power:
                    relay_state = (
                        record.get_value()
                    )  # 0.0 = OFF, 1.0 = ON (or values in between for partial)

                    relay_data.append(
                        {
                            "timestamp": record.get_time(),
                            "room": room,
                            "relay_state": relay_state,
                            "power_rating_kw": self.heating_power[room],
                            "actual_power_w": relay_state
                            * self.heating_power[room]
                            * 1000,  # Watts when ON
                            "energy_kwh_15min": (
                                relay_state * self.heating_power[room] * 0.25
                            ),  # Energy per 15min interval
                        }
                    )

        if not relay_data:
            self.logger.warning("No relay data after processing")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(relay_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

        self.logger.info(f"Extracted {len(df)} relay state records")
        return df

    def analyze_relay_patterns(self, relay_df: pd.DataFrame) -> dict:
        """Analyze relay switching patterns and energy consumption."""
        if relay_df.empty:
            return {}

        results = {}

        # Overall statistics
        total_energy = relay_df["energy_kwh_15min"].sum()
        total_capacity = sum(self.heating_power.values())
        analysis_hours = len(relay_df) * 0.25  # 15-min intervals

        results["summary"] = {
            "total_energy_kwh": total_energy,
            "analysis_period_hours": analysis_hours,
            "total_system_capacity_kw": total_capacity,
            "average_power_demand_kw": relay_df["actual_power_w"].sum() / 1000,
            "system_utilization_percent": (
                (total_energy / (total_capacity * analysis_hours)) * 100
                if analysis_hours > 0
                else 0
            ),
        }

        # Per-room analysis
        results["rooms"] = {}

        for room in relay_df["room"].unique():
            room_data = relay_df[relay_df["room"] == room].copy()

            # Basic statistics
            duty_cycle = room_data["relay_state"].mean() * 100  # Percentage of time ON
            total_switches = (room_data["relay_state"].diff().abs() > 0.1).sum()  # State changes
            room_energy = room_data["energy_kwh_15min"].sum()

            # On/off period analysis
            on_periods = []
            off_periods = []
            current_state = None
            current_start = None

            for timestamp, row in room_data.iterrows():
                state = row["relay_state"] > 0.5  # Consider >0.5 as "on"

                if current_state is None:
                    current_state = state
                    current_start = timestamp
                elif current_state != state:
                    duration_minutes = (timestamp - current_start).total_seconds() / 60

                    if current_state:  # Was on, now off
                        on_periods.append(duration_minutes)
                    else:  # Was off, now on
                        off_periods.append(duration_minutes)

                    current_state = state
                    current_start = timestamp

            results["rooms"][room] = {
                "duty_cycle_percent": duty_cycle,
                "total_switches": total_switches,
                "total_energy_kwh": room_energy,
                "power_rating_kw": self.heating_power[room],
                "avg_on_minutes": sum(on_periods) / len(on_periods) if on_periods else 0,
                "avg_off_minutes": sum(off_periods) / len(off_periods) if off_periods else 0,
                "total_on_time_hours": sum(on_periods) / 60 if on_periods else 0,
                "energy_efficiency_ratio": (
                    room_energy / (self.heating_power[room] * analysis_hours)
                    if analysis_hours > 0
                    else 0
                ),
            }

        return results

    def print_analysis_report(self, analysis_results: dict):
        """Print a comprehensive relay analysis report."""
        if not analysis_results:
            self.logger.warning("No analysis results to report")
            return

        print("\n" + "=" * 80)
        print("RELAY-BASED HEATING ANALYSIS REPORT")
        print("=" * 80)

        # Summary
        summary = analysis_results["summary"]
        print(f"\nSYSTEM SUMMARY:")
        print(f"Total Energy Consumed: {summary['total_energy_kwh']:.1f} kWh")
        print(f"Analysis Period: {summary['analysis_period_hours']:.0f} hours")
        print(f"Total System Capacity: {summary['total_system_capacity_kw']:.1f} kW")
        print(f"System Utilization: {summary['system_utilization_percent']:.1f}%")

        # Room-by-room analysis
        print(f"\nROOM-BY-ROOM RELAY ANALYSIS:")
        print("-" * 80)
        print(
            f"{'Room':<20} {'Duty%':<8} {'Switches':<10} {'Energy(kWh)':<12} {'Rating(kW)':<12} {'Efficiency':<10}"
        )
        print("-" * 80)

        for room, stats in analysis_results["rooms"].items():
            print(
                f"{room:<20} {stats['duty_cycle_percent']:<8.1f} {stats['total_switches']:<10.0f} "
                f"{stats['total_energy_kwh']:<12.1f} {stats['power_rating_kw']:<12.1f} "
                f"{stats['energy_efficiency_ratio']:<10.3f}"
            )

        # Insights
        print(f"\nRELAY SYSTEM INSIGHTS:")
        print("-" * 40)

        rooms = analysis_results["rooms"]

        # Highest energy consumers
        top_consumers = sorted(rooms.items(), key=lambda x: x[1]["total_energy_kwh"], reverse=True)[
            :3
        ]
        print(f"Top Energy Consumers:")
        for room, stats in top_consumers:
            print(
                f"  {room}: {stats['total_energy_kwh']:.1f} kWh ({stats['duty_cycle_percent']:.1f}% duty cycle)"
            )

        # Most active relays (highest switching)
        active_relays = sorted(rooms.items(), key=lambda x: x[1]["total_switches"], reverse=True)[
            :3
        ]
        print(f"\nMost Active Relays:")
        for room, stats in active_relays:
            print(
                f"  {room}: {stats['total_switches']:.0f} switches (avg {stats['avg_on_minutes']:.1f}min on, {stats['avg_off_minutes']:.1f}min off)"
            )

        # Efficiency analysis
        high_efficiency = {k: v for k, v in rooms.items() if v["energy_efficiency_ratio"] > 0.1}
        print(f"\nHigh Demand Rooms (>10% of capacity utilized):")
        for room, stats in high_efficiency.items():
            print(f"  {room}: {stats['energy_efficiency_ratio']*100:.1f}% utilization")

        print("\nNOTE: Relay states (0=OFF, 1=ON) control fixed-power heating elements.")
        print("Energy = Relay_State × Power_Rating × Time")


async def test_relay_analysis():
    """Test relay analysis with proper understanding."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RELAY-BASED HEATING ANALYSIS TEST")
    logger.info("=" * 60)

    try:
        # Create analyzer
        analyzer = RelayAnalyzer()

        # Test with 30-day period for good relay pattern data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        logger.info(
            f"Analyzing relay data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Extract relay data
        relay_data = await analyzer.extract_relay_data(start_date, end_date)

        if relay_data.empty:
            logger.warning("No relay data found - heating system may not have been active")
            print("\n⚠️ No relay data found for the analysis period.")
            print("This could mean:")
            print("- Heating system was not active (summer period)")
            print("- Different measurement names in InfluxDB")
            print("- Relay data stored in different bucket")
            return False

        # Show sample relay data
        print(f"\n📊 RELAY DATA SAMPLE:")
        print(f"Total records: {len(relay_data)}")
        print(f"Rooms with relays: {len(relay_data['room'].unique())}")
        print(f"Date range: {relay_data.index.min()} to {relay_data.index.max()}")

        # Show relay state distribution
        print(f"\nRelay State Distribution:")
        for room in sorted(relay_data["room"].unique()):
            room_data = relay_data[relay_data["room"] == room]
            on_pct = room_data["relay_state"].mean() * 100
            total_energy = room_data["energy_kwh_15min"].sum()
            print(f"  {room:20s}: {on_pct:5.1f}% ON, {total_energy:6.1f} kWh")

        # Run analysis
        analysis_results = analyzer.analyze_relay_patterns(relay_data)

        # Print comprehensive report
        analyzer.print_analysis_report(analysis_results)

        # Save results
        relay_data.to_parquet("data/raw/relay_analysis_corrected.parquet")

        import json

        with open("analysis/results/relay_analysis_corrected.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)

        logger.info("\n✅ Relay analysis completed successfully!")
        logger.info("Files saved:")
        logger.info("- data/raw/relay_analysis_corrected.parquet")
        logger.info("- analysis/results/relay_analysis_corrected.json")

        return True

    except Exception as e:
        logger.error(f"Relay analysis failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_relay_analysis())
    print(f"\nRelay Analysis Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
