#!/usr/bin/env python3
"""
Test script for relaxed filtering parameters in thermal analysis.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def create_realistic_thermal_scenarios():
    """Create test scenarios that simulate real-world conditions."""
    scenarios = {}

    # Base scenario parameters
    start_time = datetime(2025, 1, 15, 0, 0)
    hours = 8  # 8-hour decay period
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(hours * 12)]

    # Scenario 1: Small signal room (like "zadveri")
    room_temps_small = []
    outdoor_temps_stable = []
    base_outdoor = -3.0
    base_room = 19.0

    for i, _ in enumerate(timestamps):
        # Small temperature decay (0.2°C total)
        decay_factor = np.exp(-i / (2 * 12))  # 2-hour time constant
        room_temp = base_room + 0.2 * decay_factor

        # Add realistic noise
        room_temp += np.random.normal(0, 0.05)  # Small noise

        # Stable outdoor temperature with slight drift
        outdoor_temp = base_outdoor + 0.4 * (
            i / len(timestamps)
        )  # 0.4°C drift over 8 hours
        outdoor_temp += np.random.normal(0, 0.1)  # Small variations

        room_temps_small.append(room_temp)
        outdoor_temps_stable.append(outdoor_temp)

    scenarios["small_signal_stable"] = pd.DataFrame(
        {
            "room_temp": room_temps_small,
            "outdoor_temp": outdoor_temps_stable,
            "heating_on": [0] * len(timestamps),
        },
        index=pd.DatetimeIndex(timestamps),
    )

    # Scenario 2: Moderate signal with realistic noise (20% increasing points)
    room_temps_noisy = []
    outdoor_temps_drift = []

    for i, _ in enumerate(timestamps):
        # Moderate decay (0.5°C total)
        decay_factor = np.exp(-i / (3 * 12))  # 3-hour time constant
        room_temp = base_room + 0.5 * decay_factor

        # Add realistic noise that creates ~20% increasing points
        if np.random.random() < 0.2:  # 20% chance of upward spike
            room_temp += np.random.uniform(0.05, 0.15)
        else:
            room_temp += np.random.normal(0, 0.08)

        # Outdoor with gradual drift (0.6°C total change)
        outdoor_temp = base_outdoor + 0.6 * (i / len(timestamps))
        outdoor_temp += np.random.normal(0, 0.15)

        room_temps_noisy.append(room_temp)
        outdoor_temps_drift.append(outdoor_temp)

    scenarios["moderate_signal_noisy"] = pd.DataFrame(
        {
            "room_temp": room_temps_noisy,
            "outdoor_temp": outdoor_temps_drift,
            "heating_on": [0] * len(timestamps),
        },
        index=pd.DatetimeIndex(timestamps),
    )

    # Scenario 3: Previously rejected case (should now pass)
    room_temps_marginal = []
    outdoor_temps_marginal = []

    for i, _ in enumerate(timestamps):
        # Small-medium decay (0.35°C total)
        decay_factor = np.exp(-i / (2.5 * 12))
        room_temp = base_room + 0.35 * decay_factor

        # Moderate noise creating ~18% increasing points
        if np.random.random() < 0.18:
            room_temp += np.random.uniform(0.08, 0.2)
        else:
            room_temp += np.random.normal(0, 0.1)

        # Outdoor with 0.65°C drift (just under new 0.75 threshold)
        outdoor_temp = base_outdoor + 0.65 * (i / len(timestamps))
        outdoor_temp += np.random.normal(0, 0.12)

        room_temps_marginal.append(room_temp)
        outdoor_temps_marginal.append(outdoor_temp)

    scenarios["marginal_case"] = pd.DataFrame(
        {
            "room_temp": room_temps_marginal,
            "outdoor_temp": outdoor_temps_marginal,
            "heating_on": [0] * len(timestamps),
        },
        index=pd.DatetimeIndex(timestamps),
    )

    return scenarios


def simulate_relaxed_filtering(data: pd.DataFrame, scenario_name: str) -> dict:
    """Simulate the filtering checks with relaxed parameters."""
    results = {}

    # 1. Outdoor temperature stability (NEW: 0.75°C threshold)
    outdoor_std = data["outdoor_temp"].std()
    outdoor_stable = outdoor_std <= 0.75

    results["outdoor_stability"] = {
        "passed": outdoor_stable,
        "std_dev": outdoor_std,
        "threshold": 0.75,
        "improvement": "Was 0.5°C, now 0.75°C - allows realistic drift",
    }

    # 2. Monotonicity check (NEW: 20% threshold)
    temp_diff = data["room_temp"] - data["outdoor_temp"]
    temp_changes = np.diff(temp_diff.values)
    if len(temp_changes) > 0:
        increasing_pct = (temp_changes > 0).sum() / len(temp_changes)
        monotonic_ok = increasing_pct <= 0.20
    else:
        increasing_pct = 0
        monotonic_ok = True

    results["monotonicity"] = {
        "passed": monotonic_ok,
        "increasing_pct": increasing_pct * 100,
        "threshold_pct": 20.0,
        "improvement": "Was 15%, now 20% - accounts for real-world noise",
    }

    # 3. Minimum decay magnitude (NEW: relaxed thresholds)
    initial_temp_diff = temp_diff.iloc[0]
    final_temp_diff = temp_diff.iloc[-1]
    decay_magnitude = initial_temp_diff - final_temp_diff

    # Simulate winter conditions (outdoor < 5°C)
    outdoor_avg = data["outdoor_temp"].mean()
    is_winter = outdoor_avg < 5.0

    if is_winter:
        min_magnitude = 0.2  # Was 0.3°C
        condition = "winter"
    else:
        min_magnitude = 0.3  # Was 0.5°C
        condition = "other seasons"

    magnitude_ok = decay_magnitude >= min_magnitude

    results["decay_magnitude"] = {
        "passed": magnitude_ok,
        "magnitude": decay_magnitude,
        "threshold": min_magnitude,
        "condition": condition,
        "improvement": f"Winter: 0.3→0.2°C, Other: 0.5→0.3°C - enables small signal analysis",
    }

    # Overall assessment
    all_passed = all(r["passed"] for r in results.values())
    results["overall"] = {"passed": all_passed, "suitable_for_analysis": all_passed}

    return results


def test_relaxed_filtering():
    """Test the relaxed filtering parameters."""
    print("RELAXED FILTERING PARAMETERS TEST")
    print("=" * 60)
    print()

    scenarios = create_realistic_thermal_scenarios()

    for scenario_name, data in scenarios.items():
        print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
        print("-" * 50)

        results = simulate_relaxed_filtering(data, scenario_name)

        # Show results
        for check_name, result in results.items():
            if check_name == "overall":
                continue

            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"  {check_name.replace('_', ' ').title()}: {status}")

            if "std_dev" in result:
                print(
                    f"    Std Dev: {result['std_dev']:.2f}°C (threshold: {result['threshold']:.2f}°C)"
                )
            elif "increasing_pct" in result:
                print(
                    f"    Increasing points: {result['increasing_pct']:.1f}% (threshold: {result['threshold_pct']:.1f}%)"
                )
            elif "magnitude" in result:
                print(
                    f"    Decay magnitude: {result['magnitude']:.2f}°C (threshold: {result['threshold']:.2f}°C)"
                )
                print(f"    Condition: {result['condition']}")

            print(f"    Improvement: {result['improvement']}")
            print()

        overall_status = "✓ ACCEPTED" if results["overall"]["passed"] else "✗ REJECTED"
        print(f"  Overall Result: {overall_status}")
        print()

    print("SUMMARY OF RELAXED FILTERING IMPROVEMENTS")
    print("=" * 60)
    print("🌡️  Outdoor Stability: 0.5°C → 0.75°C (allows realistic temperature drift)")
    print("📈 Monotonicity: 15% → 20% (accommodates real-world sensor noise)")
    print("📏 Decay Magnitude:")
    print("   • Winter: 0.3°C → 0.2°C (enables analysis of small winter signals)")
    print("   • Other seasons: 0.5°C → 0.3°C (captures smaller temperature changes)")
    print("   • Very short cycles: Even more relaxed thresholds")
    print()
    print("🎯 Expected Outcome:")
    print("   • More heating cycles will pass the filters")
    print("   • Better analysis of rooms with small temperature changes")
    print("   • Improved RC parameter estimation for all room types")
    print("   • Maintained data quality while increasing sample size")


if __name__ == "__main__":
    test_relaxed_filtering()
