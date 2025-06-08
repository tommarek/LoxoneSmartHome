#!/usr/bin/env python3
"""
Basic structure and import tests for PEMS v2.
Tests core functionality without requiring external dependencies.
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test that all analysis modules can be imported."""
    print("🔬 Testing module imports...")

    try:
        from analysis.analyzers.pattern_analysis import PVAnalyzer

        assert PVAnalyzer is not None  # Use the import
        print("✅ PVAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ PVAnalyzer import failed: {e}")
        return False

    try:
        from analysis.analyzers.thermal_analysis import ThermalAnalyzer

        assert ThermalAnalyzer is not None  # Use the import
        print("✅ ThermalAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ ThermalAnalyzer import failed: {e}")
        return False

    try:
        from analysis.analyzers.base_load_analysis import BaseLoadAnalyzer

        assert BaseLoadAnalyzer is not None  # Use the import
        print("✅ BaseLoadAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ BaseLoadAnalyzer import failed: {e}")
        return False

    try:
        from analysis.core.data_preprocessing import DataValidator

        assert DataValidator is not None  # Use the import
        print("✅ DataValidator imported successfully")
    except Exception as e:
        print(f"❌ DataValidator import failed: {e}")
        return False

    try:
        from analysis.core.data_extraction import DataExtractor

        assert DataExtractor is not None  # Use the import
        print("✅ DataExtractor imported successfully")
    except Exception as e:
        print(f"❌ DataExtractor import failed: {e}")
        return False

    return True


def test_dependencies():
    """Test that required dependencies are available."""
    print("\n📦 Testing dependencies...")

    missing_deps = []

    try:
        import pandas as pd

        assert pd is not None  # Use the import
        print("✅ pandas available")
    except ImportError:
        print("❌ pandas not available")
        missing_deps.append("pandas")

    try:
        import numpy as np

        assert np is not None  # Use the import
        print("✅ numpy available")
    except ImportError:
        print("❌ numpy not available")
        missing_deps.append("numpy")

    try:
        import scipy

        assert scipy is not None  # Use the import
        print("✅ scipy available")
    except ImportError:
        print("❌ scipy not available")
        missing_deps.append("scipy")

    try:
        import sklearn

        assert sklearn is not None  # Use the import
        print("✅ scikit-learn available")
    except ImportError:
        print("❌ scikit-learn not available")
        missing_deps.append("scikit-learn")

    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install pandas numpy scipy scikit-learn")
        return False

    return True


def test_directory_structure():
    """Test that required directories exist."""
    print("\n📁 Testing directory structure...")

    required_dirs = ["analysis", "config", "utils", "tests"]

    missing_dirs = []
    project_root = Path(__file__).parent.parent

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            missing_dirs.append(dir_name)

    return len(missing_dirs) == 0


def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    try:
        from config.settings import InfluxDBSettings, PEMSSettings

        assert PEMSSettings is not None  # Use the import
        print("✅ Settings class imported successfully")

        # Test InfluxDB settings directly with dummy token (no env loading)
        influx_settings = InfluxDBSettings(token="test_token")
        print("✅ InfluxDB settings created successfully")

        # Test that the settings classes have the expected structure
        if hasattr(influx_settings, "token"):
            print("✅ InfluxDB token field available")
        else:
            print("❌ InfluxDB token field missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Settings loading failed: {e}")
        return False


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
