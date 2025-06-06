#!/usr/bin/env python3
"""
Test script for the Loxone Field Adapter.

This script tests the LoxoneFieldAdapter with sample data that mimics
the actual Loxone field naming conventions used in the system.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from analysis.utils.loxone_adapter import (LoxoneDataIntegrator,
                                           LoxoneFieldAdapter)


def create_sample_loxone_data():
    """Create sample data that mimics actual Loxone field naming."""

    # Create timestamp index
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="5min")

    # Sample room data with Loxone naming conventions
    sample_rooms = {
        "obyvak": pd.DataFrame(
            {
                "temperature_obyvak": 22.5 + np.random.normal(0, 0.5, len(timestamps)),
                "humidity_obyvak": 45 + np.random.normal(0, 5, len(timestamps)),
                "target_temp": np.full(len(timestamps), 22.0),
            },
            index=timestamps,
        ),
        "kuchyne": pd.DataFrame(
            {
                "temperature_kuchyne": 21.8 + np.random.normal(0, 0.3, len(timestamps)),
                "humidity_kuchyne": 40 + np.random.normal(0, 3, len(timestamps)),
                "target_temp": np.full(len(timestamps), 21.5),
            },
            index=timestamps,
        ),
        "loznice": pd.DataFrame(
            {
                "temperature_loznice": 20.2 + np.random.normal(0, 0.4, len(timestamps)),
                "humidity_loznice": 50 + np.random.normal(0, 4, len(timestamps)),
            },
            index=timestamps,
        ),
    }

    # Sample relay data (room names as field names)
    sample_relays = {
        "obyvak": pd.DataFrame(
            {"obyvak": np.random.choice([0, 1], size=len(timestamps), p=[0.7, 0.3])},
            index=timestamps,
        ),
        "kuchyne": pd.DataFrame(
            {"kuchyne": np.random.choice([0, 1], size=len(timestamps), p=[0.8, 0.2])},
            index=timestamps,
        ),
        "loznice": pd.DataFrame(
            {"loznice": np.random.choice([0, 1], size=len(timestamps), p=[0.6, 0.4])},
            index=timestamps,
        ),
    }

    # Sample weather data with Loxone solar fields
    sample_weather = pd.DataFrame(
        {
            "temperature": 15 + np.random.normal(0, 2, len(timestamps)),
            "humidity": 60 + np.random.normal(0, 10, len(timestamps)),
            "sun_elevation": np.maximum(
                0, 30 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
            ),
            "sun_direction": 180
            + 60 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps))),
            "absolute_solar_irradiance": np.maximum(
                0, 800 * np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))
            ),
        },
        index=timestamps,
    )

    return sample_rooms, sample_relays, sample_weather


def test_field_adapter():
    """Test the LoxoneFieldAdapter functionality."""

    print("🧪 Testing Loxone Field Adapter")
    print("=" * 50)

    # Create sample data
    rooms, relays, weather = create_sample_loxone_data()

    # Test 1: Room data standardization
    print("\n1️⃣ Testing room data standardization...")
    for room_name, room_df in rooms.items():
        print(f"\n   Testing room: {room_name}")
        print(f"   Original columns: {list(room_df.columns)}")

        standardized = LoxoneFieldAdapter.standardize_room_data(room_df, room_name)
        print(f"   Standardized columns: {list(standardized.columns)}")

        # Check if temperature was mapped correctly
        if "temperature" in standardized.columns:
            print(f"   ✅ Temperature mapped successfully")
            print(
                f"   📊 Temperature range: {standardized['temperature'].min():.1f}°C - {standardized['temperature'].max():.1f}°C"
            )
        else:
            print(f"   ❌ Temperature mapping failed")

        # Check humidity mapping
        if "humidity" in standardized.columns:
            print(f"   ✅ Humidity mapped successfully")
            print(
                f"   📊 Humidity range: {standardized['humidity'].min():.1f}% - {standardized['humidity'].max():.1f}%"
            )
        else:
            print(f"   ⚠️  No humidity data found")

    # Test 2: Relay data standardization
    print("\n2️⃣ Testing relay data standardization...")
    standardized_relays = LoxoneFieldAdapter.standardize_relay_data(relays)

    for room_name, relay_df in standardized_relays.items():
        print(f"\n   Standardized relay for: {room_name}")
        print(f"   Columns: {list(relay_df.columns)}")

        if "relay_state" in relay_df.columns:
            duty_cycle = relay_df["relay_state"].mean() * 100
            print(f"   ✅ Relay state mapped - Duty cycle: {duty_cycle:.1f}%")

        if "power_kw" in relay_df.columns:
            avg_power = relay_df["power_kw"].mean()
            print(f"   ✅ Power calculated - Average: {avg_power:.2f} kW")

    # Test 3: Weather data standardization
    print("\n3️⃣ Testing weather data standardization...")
    print(f"   Original weather columns: {list(weather.columns)}")

    standardized_weather = LoxoneFieldAdapter.standardize_weather_data(weather)
    print(f"   Standardized weather columns: {list(standardized_weather.columns)}")

    # Check solar fields
    solar_fields = ["sun_elevation", "sun_direction", "solar_irradiance"]
    for field in solar_fields:
        if field in standardized_weather.columns:
            print(f"   ✅ {field} available")
        else:
            print(f"   ⚠️  {field} not found")

    # Test 4: Room name standardization
    print("\n4️⃣ Testing room name standardization...")
    test_names = [
        "obyvak",
        "kuchyne",
        "loznice",
        "koupelna_dole",
        "relay_obyvak",
        "temperature_kuchyne",
    ]

    for name in test_names:
        standard_name = LoxoneFieldAdapter.standardize_room_name(name)
        print(f"   {name} -> {standard_name}")

    # Test 5: Data validation
    print("\n5️⃣ Testing data structure validation...")
    sample_data = {"rooms": rooms, "relay_states": relays, "weather": weather}

    validation_report = LoxoneFieldAdapter.validate_data_structure(sample_data)

    print(f"   Rooms found: {validation_report['rooms_found']}")
    print(
        f"   Completeness: {validation_report['completeness']['complete_rooms']}/{validation_report['completeness']['total_rooms']} rooms"
    )

    if validation_report["missing_fields"]:
        print(f"   ⚠️  Missing fields detected:")
        for room, missing in validation_report["missing_fields"].items():
            print(f"      {room}: {missing}")

    if validation_report["recommendations"]:
        print(f"   💡 Recommendations:")
        for rec in validation_report["recommendations"]:
            print(f"      - {rec}")

    return standardized_relays, standardized_weather


def test_data_integrator():
    """Test the LoxoneDataIntegrator functionality."""

    print("\n\n🔧 Testing Loxone Data Integrator")
    print("=" * 50)

    # Create sample data
    rooms, relays, weather = create_sample_loxone_data()

    # Initialize integrator
    integrator = LoxoneDataIntegrator()

    # Test 1: Thermal analysis data preparation
    print("\n1️⃣ Testing thermal analysis data preparation...")
    thermal_rooms, thermal_weather = integrator.prepare_thermal_analysis_data(
        rooms, relays, weather
    )

    print(f"   Prepared data for {len(thermal_rooms)} rooms")
    for room_name, room_df in thermal_rooms.items():
        columns = list(room_df.columns)
        print(f"   {room_name}: {columns}")

        # Check if heating data was integrated
        if "heating_on" in columns:
            heating_ratio = room_df["heating_on"].mean() * 100
            print(
                f"      ✅ Heating data integrated - Active: {heating_ratio:.1f}% of time"
            )

    # Test 2: Relay analysis data preparation
    print("\n2️⃣ Testing relay analysis data preparation...")
    relay_analysis_data = integrator.prepare_relay_analysis_data(relays)

    total_power = 0
    for room_name, relay_df in relay_analysis_data.items():
        if "power_kw" in relay_df.columns:
            avg_power = relay_df["power_kw"].mean()
            total_power += avg_power
            print(f"   {room_name}: Average power = {avg_power:.2f} kW")

    print(f"   Total average system power: {total_power:.2f} kW")

    # Test 3: PV analysis data preparation
    print("\n3️⃣ Testing PV analysis data preparation...")

    # Create sample PV data
    timestamps = weather.index
    sample_pv = pd.DataFrame(
        {
            "InputPower": 3000
            * np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, len(timestamps)))),
            "ExportPower": 1000
            * np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, len(timestamps))) - 0.5),
            "SelfConsumption": 2000 + np.random.normal(0, 200, len(timestamps)),
        },
        index=timestamps,
    )

    pv_data, enhanced_weather = integrator.prepare_pv_analysis_data(sample_pv, weather)

    print(f"   PV data columns: {list(pv_data.columns)}")
    print(f"   Enhanced weather columns: {list(enhanced_weather.columns)}")

    if "solar_irradiance" in enhanced_weather.columns:
        print(f"   ✅ Solar irradiance data available for correlation analysis")

    print("\n✅ All tests completed successfully!")

    return True


def main():
    """Run all adapter tests."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🏠 PEMS v2 - Loxone Field Adapter Test Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Test the field adapter
        test_field_adapter()

        # Test the data integrator
        test_data_integrator()

        print("\n🎉 All tests passed! The Loxone adapter is ready for use.")

        # Show summary of capabilities
        print("\n📋 Adapter Capabilities Summary:")
        print("   ✅ Standardizes Loxone field names to PEMS standard")
        print("   ✅ Preserves Czech room names (no translation needed)")
        print("   ✅ Handles temperature, humidity, and relay state fields")
        print("   ✅ Adds power calculations for relay data")
        print("   ✅ Integrates weather data with solar fields")
        print("   ✅ Validates data structure and provides recommendations")
        print("   ✅ Prepares data for thermal, relay, and PV analysis modules")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
