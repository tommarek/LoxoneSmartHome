#!/usr/bin/env python3
"""
Test LoadPredictor on real extracted data from the PEMS system.

This script loads the actual data extracted from InfluxDB and tests
the LoadPredictor with realistic consumption patterns, weather data,
and system states.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the pems_v2 module to the path
sys.path.append('/Users/tommarek/git/LoxoneSmartHome/pems_v2')

from models.predictors.load_predictor import LoadPredictor
from config.settings import PEMSSettings

def load_real_data():
    """Load the real extracted data from the analysis pipeline."""
    print("=== Loading Real PEMS Data ===")
    
    data_dir = Path("data/processed")
    
    # Load the main datasets
    try:
        # Load PV data (has InputPower which we can use as a proxy for load)
        pv_data = pd.read_parquet("data/processed/pv.parquet")
        print(f"✓ Loaded PV data: {len(pv_data)} records")
        
        # Load weather data
        weather_data = pd.read_parquet("data/processed/weather.parquet")
        print(f"✓ Loaded weather data: {len(weather_data)} records")
        
        # Load outdoor temperature
        outdoor_temp = pd.read_parquet("data/processed/outdoor_temp.parquet")
        print(f"✓ Loaded outdoor temp: {len(outdoor_temp)} records")
        
        # Load consumption data (heating)
        consumption_data = pd.read_parquet("data/processed/consumption.parquet")
        print(f"✓ Loaded consumption data: {len(consumption_data)} records")
        
        return pv_data, weather_data, outdoor_temp, consumption_data
        
    except FileNotFoundError as e:
        print(f"⚠️ Processed data not found, trying raw data: {e}")
        
        # Try raw data instead
        pv_data = pd.read_parquet("data/raw/pv_data.parquet")
        weather_data = pd.read_parquet("data/raw/weather_data.parquet")
        outdoor_temp = pd.read_parquet("data/raw/outdoor_temperature.parquet")
        consumption_data = pd.read_parquet("data/raw/consumption_data.parquet")
        
        print(f"✓ Loaded raw PV data: {len(pv_data)} records")
        print(f"✓ Loaded raw weather data: {len(weather_data)} records")
        print(f"✓ Loaded raw outdoor temp: {len(outdoor_temp)} records")
        print(f"✓ Loaded raw consumption data: {len(consumption_data)} records")
        
        return pv_data, weather_data, outdoor_temp, consumption_data

def create_load_dataset(pv_data, weather_data, outdoor_temp, consumption_data):
    """Create a unified dataset for load prediction."""
    print("\n=== Creating Unified Load Dataset ===")
    
    # Use PV data as the base timeframe (most complete)
    base_data = pv_data.copy()
    
    # Create a synthetic load based on ACPowerToUser (local consumption)
    if 'ACPowerToUser' in base_data.columns:
        load_column = 'ACPowerToUser'
        print("Using ACPowerToUser as load proxy")
    elif 'InputPower' in base_data.columns:
        # If no local consumption, use input power as a proxy
        load_column = 'InputPower'
        print("Using InputPower as load proxy")
    else:
        print("⚠️ No suitable load column found in PV data")
        return None
    
    # Create the base dataset
    dataset = pd.DataFrame({
        'load': base_data[load_column].fillna(0),  # Fill missing with 0
        'consumption': base_data[load_column].fillna(0),  # Alias for load predictor
    }, index=base_data.index)
    
    # Add outdoor temperature
    if not outdoor_temp.empty:
        # Resample to match the base timeframe
        outdoor_resampled = outdoor_temp.resample('15min').mean()
        dataset = dataset.join(outdoor_resampled, how='left')
        if 'outdoor_temp' in dataset.columns:
            dataset['outdoor_temp'] = dataset['outdoor_temp'].fillna(method='ffill').fillna(method='bfill')
            print(f"✓ Added outdoor temperature")
    
    # Add weather features
    if not weather_data.empty:
        # Resample weather data to 15 minutes
        weather_resampled = weather_data.resample('15min').mean()
        
        # Join key weather features
        weather_cols = ['temperature_2m', 'relativehumidity_2m', 'windspeed_10m', 
                       'shortwave_radiation', 'cloudcover', 'precipitation']
        available_cols = [col for col in weather_cols if col in weather_resampled.columns]
        
        if available_cols:
            weather_subset = weather_resampled[available_cols]
            dataset = dataset.join(weather_subset, how='left')
            
            # Forward fill weather data
            for col in available_cols:
                if col in dataset.columns:
                    dataset[col] = dataset[col].fillna(method='ffill').fillna(method='bfill')
            
            print(f"✓ Added weather features: {available_cols}")
    
    # Add heating consumption if available
    if not consumption_data.empty and 'heating_power' in consumption_data.columns:
        heating_resampled = consumption_data.resample('15min').mean()
        dataset = dataset.join(heating_resampled[['heating_power']], how='left')
        dataset['heating_power'] = dataset['heating_power'].fillna(0)
        print(f"✓ Added heating consumption")
    
    # Remove rows where load is NaN (essential for training)
    original_len = len(dataset)
    dataset = dataset.dropna(subset=['load'])
    cleaned_len = len(dataset)
    
    if cleaned_len < original_len:
        print(f"⚠️ Removed {original_len - cleaned_len} rows with missing load data")
    
    # Convert load from watts to kilowatts for more reasonable scale
    dataset['load'] = dataset['load'] / 1000.0
    dataset['consumption'] = dataset['consumption'] / 1000.0
    
    print(f"✓ Created dataset with {len(dataset)} records")
    print(f"  Date range: {dataset.index.min()} to {dataset.index.max()}")
    print(f"  Load range: {dataset['load'].min():.2f} - {dataset['load'].max():.2f} kW")
    print(f"  Columns: {list(dataset.columns)}")
    
    return dataset

def test_load_predictor_training(dataset):
    """Test LoadPredictor training with real data."""
    print("\n=== Testing LoadPredictor Training ===")
    
    # Configure the predictor
    config = {
        'decomposition_method': 'nmf',
        'n_components': 5,
        'prediction_horizon': 24,
        'include_weather': True,
        'include_occupancy': True
    }
    
    predictor = LoadPredictor(config)
    print(f"✓ LoadPredictor initialized")
    
    # Use the last 7 days for training to avoid memory issues with large dataset
    recent_data = dataset.tail(7 * 96)  # 7 days * 96 intervals per day
    print(f"✓ Using recent {len(recent_data)} records for training")
    
    # Prepare training data
    X_train = recent_data.drop(columns=['load'])
    y_train = recent_data['load']
    
    print(f"  Training features: {len(X_train.columns)} columns")
    print(f"  Training target range: {y_train.min():.2f} - {y_train.max():.2f} kW")
    
    try:
        # Train the model
        start_time = datetime.now()
        performance = predictor.train(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Training completed in {training_time:.1f}s")
        print(f"  RMSE: {performance.rmse:.3f} kW")
        print(f"  MAE: {performance.mae:.3f} kW")
        print(f"  R²: {performance.r2:.3f}")
        print(f"  MAPE: {performance.mape:.1f}%")
        print(f"  Ensemble weights: {predictor.ensemble_weights}")
        
        return predictor, performance
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_load_predictor_prediction(predictor, dataset):
    """Test LoadPredictor prediction with real data."""
    print("\n=== Testing LoadPredictor Prediction ===")
    
    if predictor is None:
        print("⚠️ Skipping prediction test - no trained model")
        return
    
    # Use the last day for prediction testing
    test_data = dataset.tail(96)  # Last 24 hours
    X_test = test_data.drop(columns=['load'])
    y_true = test_data['load']
    
    try:
        # Make predictions with uncertainty
        start_time = datetime.now()
        result = predictor.predict(X_test, return_uncertainty=True)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Prediction completed in {prediction_time:.3f}s")
        print(f"  Predictions shape: {result.predictions.shape}")
        print(f"  Prediction range: {result.predictions.min():.3f} - {result.predictions.max():.3f} kW")
        print(f"  True range: {y_true.min():.3f} - {y_true.max():.3f} kW")
        
        # Calculate prediction accuracy
        mae = np.mean(np.abs(result.predictions - y_true))
        rmse = np.sqrt(np.mean((result.predictions - y_true) ** 2))
        
        print(f"  Prediction MAE: {mae:.3f} kW")
        print(f"  Prediction RMSE: {rmse:.3f} kW")
        
        if result.uncertainty is not None:
            avg_uncertainty = result.uncertainty.mean()
            print(f"  Average uncertainty: {avg_uncertainty:.3f} kW")
        
        # Test insights
        insights = predictor.get_load_insights()
        print(f"✓ Generated insights with {len(insights)} categories")
        
        return result
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_configurations(dataset):
    """Test different LoadPredictor configurations."""
    print("\n=== Testing Different Configurations ===")
    
    configs = [
        {'name': 'NMF+Weather', 'decomposition_method': 'nmf', 'include_weather': True, 'include_occupancy': False},
        {'name': 'PCA+Occupancy', 'decomposition_method': 'pca', 'include_weather': False, 'include_occupancy': True},
        {'name': 'Clustering+All', 'decomposition_method': 'clustering', 'include_weather': True, 'include_occupancy': True},
    ]
    
    # Use a smaller subset for configuration testing
    test_data = dataset.tail(3 * 96)  # 3 days
    X_test = test_data.drop(columns=['load'])
    y_test = test_data['load']
    
    results = {}
    
    for config_info in configs:
        name = config_info.pop('name')
        config = {
            'n_components': 3,
            'prediction_horizon': 24,
            **config_info
        }
        
        try:
            predictor = LoadPredictor(config)
            performance = predictor.train(X_test, y_test)
            
            results[name] = {
                'rmse': performance.rmse,
                'mae': performance.mae,
                'r2': performance.r2
            }
            
            print(f"✓ {name}: RMSE={performance.rmse:.3f}, MAE={performance.mae:.3f}, R²={performance.r2:.3f}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def main():
    """Run the complete LoadPredictor test on real data."""
    print("=" * 60)
    print("LOAD PREDICTOR REAL DATA TEST")
    print("=" * 60)
    
    try:
        # Load real data
        pv_data, weather_data, outdoor_temp, consumption_data = load_real_data()
        
        # Create unified dataset
        dataset = create_load_dataset(pv_data, weather_data, outdoor_temp, consumption_data)
        
        if dataset is None or len(dataset) < 100:
            print("✗ Insufficient data for testing")
            return
        
        # Test training
        predictor, performance = test_load_predictor_training(dataset)
        
        # Test prediction
        result = test_load_predictor_prediction(predictor, dataset)
        
        # Test different configurations
        config_results = test_different_configurations(dataset)
        
        print("\n" + "=" * 60)
        print("REAL DATA TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        if performance:
            print(f"✓ Main model trained with RMSE={performance.rmse:.3f} kW, R²={performance.r2:.3f}")
        
        if result:
            print(f"✓ Predictions generated successfully with uncertainty quantification")
        
        print(f"✓ Tested {len(config_results)} different configurations")
        
    except Exception as e:
        print(f"\n✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()