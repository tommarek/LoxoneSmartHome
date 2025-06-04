"""
Simple test script for PEMS v2 analysis functions.
Tests core logic without heavy dependencies.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all analysis modules can be imported."""
    print("🔬 Testing module imports...")
    
    try:
        from analysis.pattern_analysis import PVAnalyzer
        print("✅ PVAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ PVAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.thermal_analysis import ThermalAnalyzer
        print("✅ ThermalAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ ThermalAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.base_load_analysis import BaseLoadAnalyzer
        print("✅ BaseLoadAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ BaseLoadAnalyzer import failed: {e}")
        return False
    
    try:
        from analysis.data_preprocessing import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ DataPreprocessor import failed: {e}")
        return False
    
    try:
        from analysis.data_extraction import DataExtractor
        print("✅ DataExtractor imported successfully")
    except Exception as e:
        print(f"❌ DataExtractor import failed: {e}")
        return False
    
    return True

def test_class_initialization():
    """Test that classes can be initialized."""
    print("\n🏗️  Testing class initialization...")
    
    try:
        from analysis.pattern_analysis import PVAnalyzer
        analyzer = PVAnalyzer()
        print("✅ PVAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ PVAnalyzer initialization failed: {e}")
        return False
    
    try:
        from analysis.thermal_analysis import ThermalAnalyzer
        analyzer = ThermalAnalyzer()
        print("✅ ThermalAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ ThermalAnalyzer initialization failed: {e}")
        return False
    
    try:
        from analysis.base_load_analysis import BaseLoadAnalyzer
        analyzer = BaseLoadAnalyzer()
        print("✅ BaseLoadAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ BaseLoadAnalyzer initialization failed: {e}")
        return False
    
    try:
        from analysis.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor initialized successfully")
    except Exception as e:
        print(f"❌ DataPreprocessor initialization failed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that required dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    missing_deps = []
    
    try:
        import pandas as pd
        print("✅ pandas available")
    except ImportError:
        print("❌ pandas not available")
        missing_deps.append("pandas")
    
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        print("❌ numpy not available")
        missing_deps.append("numpy")
    
    try:
        import scipy
        print("✅ scipy available")
    except ImportError:
        print("❌ scipy not available")
        missing_deps.append("scipy")
    
    try:
        import sklearn
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
    
    required_dirs = [
        "analysis",
        "config", 
        "modules",
        "utils",
        "tests"
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            missing_dirs.append(dir_name)
    
    # Check analysis files
    analysis_files = [
        "analysis/data_extraction.py",
        "analysis/pattern_analysis.py", 
        "analysis/thermal_analysis.py",
        "analysis/base_load_analysis.py",
        "analysis/data_preprocessing.py",
        "analysis/run_analysis.py"
    ]
    
    for file_path in analysis_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            missing_dirs.append(file_path)
    
    return len(missing_dirs) == 0

def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from config.settings import Settings
        print("✅ Settings class imported successfully")
        
        # Try to create settings (will use defaults if no env vars)
        settings = Settings()
        print("✅ Settings initialized successfully")
        
        # Check that some basic attributes exist
        if hasattr(settings, 'influxdb'):
            print("✅ InfluxDB settings available")
        else:
            print("❌ InfluxDB settings missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Settings loading failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🧪 PEMS v2 SIMPLE TESTING")
    print("="*50)
    print("Testing core functionality without running analysis...")
    
    # Set up minimal logging
    logging.basicConfig(level=logging.ERROR)
    
    # Run tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure), 
        ("Module Imports", test_imports),
        ("Class Initialization", test_class_initialization),
        ("Configuration", test_config_loading)
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
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
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
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure settings in .env file")
        print("3. Run full test: python test_analysis.py")
        print("4. Run analysis: python analysis/run_analysis.py")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)