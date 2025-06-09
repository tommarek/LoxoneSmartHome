#!/usr/bin/env python3
"""
Basic structure and import tests for PEMS v2.
Tests core functionality without requiring external dependencies.
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path (parent of pems_v2)
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all analysis modules can be imported."""
    print("🔬 Testing module imports...")

    # Import and validate all modules
    from pems_v2.analysis.analyzers.pattern_analysis import PVAnalyzer

    assert PVAnalyzer is not None
    print("✅ PVAnalyzer imported successfully")

    from pems_v2.analysis.analyzers.thermal_analysis import ThermalAnalyzer

    assert ThermalAnalyzer is not None
    print("✅ ThermalAnalyzer imported successfully")

    from pems_v2.analysis.analyzers.base_load_analysis import BaseLoadAnalyzer

    assert BaseLoadAnalyzer is not None
    print("✅ BaseLoadAnalyzer imported successfully")

    from pems_v2.analysis.core.data_preprocessing import DataValidator

    assert DataValidator is not None
    print("✅ DataValidator imported successfully")

    from pems_v2.analysis.core.data_extraction import DataExtractor

    assert DataExtractor is not None
    print("✅ DataExtractor imported successfully")


def test_dependencies():
    """Test that required dependencies are available."""
    print("\n📦 Testing dependencies...")

    # Test all required dependencies
    import pandas as pd

    assert pd is not None
    print("✅ pandas available")

    import numpy as np

    assert np is not None
    print("✅ numpy available")

    import scipy

    assert scipy is not None
    print("✅ scipy available")

    import sklearn

    assert sklearn is not None
    print("✅ scikit-learn available")


def test_directory_structure():
    """Test that required directories exist."""
    print("\n📁 Testing directory structure...")

    required_dirs = ["analysis", "config", "utils", "tests"]
    project_root = Path(__file__).parent.parent

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory '{dir_name}' not found"
        print(f"✅ {dir_name}/ exists")


def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    from pems_v2.config.settings import InfluxDBSettings, PEMSSettings

    assert PEMSSettings is not None
    print("✅ Settings class imported successfully")

    # Test InfluxDB settings directly with dummy token (no env loading)
    influx_settings = InfluxDBSettings(token="test_token")
    print("✅ InfluxDB settings created successfully")

    # Test that the settings classes have the expected structure
    assert hasattr(influx_settings, "token"), "InfluxDB token field missing"
    print("✅ InfluxDB token field available")


def main():
    """Run all basic tests."""
    print("🧪 PEMS v2 BASIC STRUCTURE TESTS")
    print("=" * 50)

    # Set up minimal logging
    logging.basicConfig(level=logging.ERROR)

    # Run tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config_loading),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All basic tests passed! The PEMS v2 structure is correct.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
