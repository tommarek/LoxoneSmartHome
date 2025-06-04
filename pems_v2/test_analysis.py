"""
Test script for PEMS v2 analysis functions.

Creates synthetic test data and runs all analysis modules
to verify functionality without requiring real InfluxDB data.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from analysis.pattern_analysis import PVAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from analysis.base_load_analysis import BaseLoadAnalyzer
from analysis.data_preprocessing import DataPreprocessor


class MockDataGenerator:
    """Generate synthetic test data that mimics real smart home data."""
    
    def __init__(self, days: int = 30):
        """Initialize with specified number of days of data."""
        self.days = days
        self.start_date = datetime.now() - timedelta(days=days)
        self.end_date = datetime.now()
        
        # Create 15-minute frequency index
        self.index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='15T'
        )
        
    def generate_pv_data(self) -> pd.DataFrame:
        """Generate realistic PV production data."""
        data = []
        
        for timestamp in self.index:
            hour = timestamp.hour
            day_of_year = timestamp.dayofyear
            
            # Solar elevation model (simplified)
            solar_elevation = max(0, 60 * np.sin((hour - 6) * np.pi / 12))
            
            # Seasonal variation
            seasonal_factor = 0.7 + 0.3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            
            # Random weather effects
            cloud_factor = 0.6 + 0.4 * np.random.random()
            
            # Calculate PV power (0-5000W system)
            if solar_elevation > 0:
                pv_power = min(5000, solar_elevation * seasonal_factor * cloud_factor * 80)
                pv_power = max(0, pv_power + np.random.normal(0, 100))
            else:
                pv_power = 0
                
            data.append({
                'timestamp': timestamp,
                'solar_power': pv_power,
                'solar_energy': pv_power * 0.25  # 15 minutes = 0.25 hours
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def generate_weather_data(self) -> pd.DataFrame:
        """Generate realistic weather data."""
        data = []
        
        base_temp = 15  # Base temperature
        
        for timestamp in self.index:
            hour = timestamp.hour
            day_of_year = timestamp.dayofyear
            
            # Daily temperature cycle
            daily_temp = 8 * np.sin((hour - 6) * np.pi / 12)
            
            # Seasonal temperature cycle
            seasonal_temp = 15 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            
            # Random variations
            random_temp = np.random.normal(0, 3)
            
            temperature = base_temp + daily_temp + seasonal_temp + random_temp
            
            # Other weather parameters
            humidity = max(20, min(100, 60 + np.random.normal(0, 15)))
            wind_speed = max(0, np.random.exponential(5))
            cloud_cover = max(0, min(100, np.random.beta(2, 2) * 100))
            
            data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'cloud_cover': cloud_cover
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def generate_room_data(self, room_name: str) -> pd.DataFrame:
        """Generate realistic room temperature and heating data."""
        data = []
        
        # Room-specific parameters
        room_params = {
            'living_room': {'base_temp': 22, 'variation': 2, 'heating_power': 1500},
            'bedroom': {'base_temp': 20, 'variation': 1.5, 'heating_power': 1000},
            'kitchen': {'base_temp': 21, 'variation': 3, 'heating_power': 800}
        }
        
        params = room_params.get(room_name, {'base_temp': 21, 'variation': 2, 'heating_power': 1200})
        
        current_temp = params['base_temp']
        heating_on = False
        
        for timestamp in self.index:
            hour = timestamp.hour
            
            # Outdoor temperature influence (simplified)
            outdoor_temp = 10 + 10 * np.sin((hour - 6) * np.pi / 12)
            
            # Heating control logic
            if current_temp < params['base_temp'] - 1:
                heating_on = True
            elif current_temp > params['base_temp'] + 0.5:
                heating_on = False
            
            # Temperature evolution
            if heating_on:
                # Heating rate: ~2°C/hour
                temp_change = 0.5 * 0.25  # 15 minutes
            else:
                # Cooling rate based on temperature difference
                temp_diff = current_temp - outdoor_temp
                temp_change = -temp_diff * 0.1 * 0.25  # Cooling
            
            current_temp += temp_change + np.random.normal(0, 0.1)
            
            # Ensure reasonable bounds
            current_temp = max(15, min(30, current_temp))
            
            data.append({
                'timestamp': timestamp,
                'temperature': current_temp,
                'setpoint': params['base_temp'],
                'heating_on': int(heating_on),
                'heating_power': params['heating_power'] if heating_on else 0
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def generate_consumption_data(self, pv_data: pd.DataFrame, room_data: dict) -> pd.DataFrame:
        """Generate realistic total consumption data."""
        data = []
        
        # Calculate total heating consumption
        total_heating = pd.Series(0, index=self.index)
        for room_df in room_data.values():
            if 'heating_power' in room_df.columns:
                heating_resampled = room_df['heating_power'].reindex(self.index, method='nearest')
                total_heating += heating_resampled.fillna(0)
        
        for timestamp in self.index:
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            # Base load patterns
            if weekday < 5:  # Weekday
                if 6 <= hour <= 8 or 17 <= hour <= 22:  # Peak hours
                    base_load = 800 + np.random.normal(0, 100)
                else:
                    base_load = 400 + np.random.normal(0, 50)
            else:  # Weekend
                if 8 <= hour <= 22:
                    base_load = 600 + np.random.normal(0, 80)
                else:
                    base_load = 300 + np.random.normal(0, 40)
            
            # Add heating consumption
            heating_power = total_heating.loc[timestamp] if timestamp in total_heating.index else 0
            
            # Total consumption
            total_consumption = base_load + heating_power
            
            # Grid import/export calculation
            pv_power = pv_data.loc[timestamp, 'solar_power'] if timestamp in pv_data.index else 0
            
            if total_consumption > pv_power:
                grid_import = total_consumption - pv_power
                grid_export = 0
            else:
                grid_import = 0
                grid_export = pv_power - total_consumption
            
            data.append({
                'timestamp': timestamp,
                'total_consumption': total_consumption,
                'grid_import': grid_import,
                'grid_export': grid_export,
                'consumption': total_consumption  # Alias for compatibility
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df


def test_pv_analysis():
    """Test PV production analysis."""
    print("\n" + "="*50)
    print("TESTING PV ANALYSIS")
    print("="*50)
    
    # Generate test data
    generator = MockDataGenerator(days=60)  # 2 months of data
    pv_data = generator.generate_pv_data()
    weather_data = generator.generate_weather_data()
    
    print(f"Generated PV data: {len(pv_data)} records")
    print(f"Generated weather data: {len(weather_data)} records")
    print(f"PV data range: {pv_data['solar_power'].min():.1f}W to {pv_data['solar_power'].max():.1f}W")
    
    # Run analysis
    analyzer = PVAnalyzer()
    
    try:
        results = analyzer.analyze_pv_production(pv_data, weather_data)
        
        print("\n✅ PV Analysis Results:")
        if 'basic_stats' in results:
            stats = results['basic_stats']
            print(f"  - Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
            print(f"  - Max power: {stats.get('max_power', 0):.1f} W")
            print(f"  - Capacity factor: {stats.get('capacity_factor', 0)*100:.1f}%")
        
        if 'weather_correlations' in results:
            corr_count = len(results['weather_correlations'].get('correlations', {}))
            print(f"  - Weather correlations calculated: {corr_count}")
        
        if 'anomalies' in results:
            anomaly_count = results['anomalies'].get('total_anomalies', 0)
            print(f"  - Anomalies detected: {anomaly_count}")
        
        print("✅ PV analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ PV analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thermal_analysis():
    """Test thermal dynamics analysis."""
    print("\n" + "="*50)
    print("TESTING THERMAL ANALYSIS")
    print("="*50)
    
    # Generate test data
    generator = MockDataGenerator(days=60)
    weather_data = generator.generate_weather_data()
    
    # Generate room data
    rooms = ['living_room', 'bedroom', 'kitchen']
    room_data = {}
    
    for room in rooms:
        room_data[room] = generator.generate_room_data(room)
        print(f"Generated {room} data: {len(room_data[room])} records")
    
    # Run analysis
    analyzer = ThermalAnalyzer()
    
    try:
        results = analyzer.analyze_room_dynamics(room_data, weather_data)
        
        print("\n✅ Thermal Analysis Results:")
        for room_name, room_results in results.items():
            if room_name == 'room_coupling':
                continue
                
            if isinstance(room_results, dict) and 'basic_stats' in room_results:
                stats = room_results['basic_stats']
                print(f"  {room_name}:")
                print(f"    - Mean temp: {stats.get('mean_temperature', 0):.1f}°C")
                print(f"    - Heating usage: {stats.get('heating_percentage', 0):.1f}%")
                
                if 'time_constant' in room_results:
                    tc_data = room_results['time_constant']
                    if isinstance(tc_data, dict) and 'time_constant_hours' in tc_data:
                        print(f"    - Time constant: {tc_data['time_constant_hours']:.1f} hours")
        
        print("✅ Thermal analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Thermal analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_load_analysis():
    """Test base load analysis."""
    print("\n" + "="*50)
    print("TESTING BASE LOAD ANALYSIS")
    print("="*50)
    
    # Generate test data
    generator = MockDataGenerator(days=60)
    pv_data = generator.generate_pv_data()
    
    # Generate room data
    rooms = ['living_room', 'bedroom', 'kitchen']
    room_data = {}
    for room in rooms:
        room_data[room] = generator.generate_room_data(room)
    
    # Generate consumption data
    consumption_data = generator.generate_consumption_data(pv_data, room_data)
    print(f"Generated consumption data: {len(consumption_data)} records")
    print(f"Consumption range: {consumption_data['total_consumption'].min():.1f}W to {consumption_data['total_consumption'].max():.1f}W")
    
    # Run analysis
    analyzer = BaseLoadAnalyzer()
    
    try:
        results = analyzer.analyze_base_load(
            consumption_data, 
            pv_data, 
            room_data
        )
        
        print("\n✅ Base Load Analysis Results:")
        if 'basic_stats' in results:
            stats = results['basic_stats']
            print(f"  - Mean base load: {stats.get('mean_base_load', 0):.1f} W")
            print(f"  - Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
            print(f"  - Base load %: {stats.get('base_load_percentage', 0):.1f}%")
            print(f"  - Peak hour: {stats.get('peak_hour', 'N/A')}")
        
        if 'time_patterns' in results:
            patterns = results['time_patterns']
            if 'weekday_vs_weekend' in patterns:
                weekend_change = patterns['weekday_vs_weekend'].get('weekend_increase', 0)
                print(f"  - Weekend increase: {weekend_change:.1f}%")
        
        if 'anomalies' in results:
            anomalies = results['anomalies']
            if 'statistical_anomalies' in anomalies:
                anomaly_count = anomalies['statistical_anomalies'].get('count', 0)
                print(f"  - Anomalies detected: {anomaly_count}")
        
        print("✅ Base load analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Base load analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preprocessing():
    """Test data preprocessing functions."""
    print("\n" + "="*50)
    print("TESTING DATA PREPROCESSING")
    print("="*50)
    
    # Generate test data with some issues
    generator = MockDataGenerator(days=7)  # 1 week
    pv_data = generator.generate_pv_data()
    
    # Introduce some data quality issues
    pv_data_dirty = pv_data.copy()
    
    # Add some missing values
    missing_indices = np.random.choice(pv_data_dirty.index, size=50, replace=False)
    pv_data_dirty.loc[missing_indices, 'solar_power'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(pv_data_dirty.index, size=10, replace=False)
    pv_data_dirty.loc[outlier_indices, 'solar_power'] = 50000  # Unrealistic values
    
    print(f"Original data: {len(pv_data)} records")
    print(f"Dirty data: {pv_data_dirty.isnull().sum().sum()} missing values, outliers added")
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    
    try:
        cleaned_data = preprocessor.process_dataset(pv_data_dirty, 'pv')
        
        print("\n✅ Data Preprocessing Results:")
        print(f"  - Cleaned data: {len(cleaned_data)} records")
        print(f"  - Missing values after cleaning: {cleaned_data.isnull().sum().sum()}")
        print(f"  - Max power after cleaning: {cleaned_data['solar_power'].max():.1f}W")
        
        # Check quality report
        if 'pv' in preprocessor.quality_report:
            report = preprocessor.quality_report['pv']
            print(f"  - Quality improvement: {report['original_missing_percentage']:.1f}% -> {report['clean_missing_percentage']:.1f}% missing")
        
        print("✅ Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🔬 PEMS v2 ANALYSIS TESTING")
    print("="*60)
    print("Testing analysis functions with synthetic data...")
    
    # Set up logging to reduce noise during testing
    logging.basicConfig(level=logging.WARNING)
    
    # Create test directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("analysis/results").mkdir(parents=True, exist_ok=True)
    
    # Run all tests
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("PV Analysis", test_pv_analysis),
        ("Thermal Analysis", test_thermal_analysis),
        ("Base Load Analysis", test_base_load_analysis)
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
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The PEMS v2 analysis pipeline is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)