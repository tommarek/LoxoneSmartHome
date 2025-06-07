#!/usr/bin/env python3
"""
PEMS v2 Phase 2: Train ML Models with Real Data

This script trains all ML models using the historical data from the Loxone system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.predictors.pv_predictor import PVPredictor
from models.predictors.load_predictor import LoadPredictor
from models.predictors.thermal_predictor import ThermalPredictor

def main():
    """Train all ML models with real historical data."""
    print('🚀 PEMS v2 Phase 2: ML Model Training with Real Data')
    print('=' * 60)

    # Create models/saved directory if it doesn't exist
    os.makedirs('models/saved', exist_ok=True)

    # Load processed data
    print('📊 Loading processed datasets...')
    try:
        pv_data = pd.read_parquet('data/processed/pv_processed.parquet')
        weather_data = pd.read_parquet('data/processed/weather_processed.parquet') 
        consumption_data = pd.read_parquet('data/processed/consumption_processed.parquet')
        room_obyvak = pd.read_parquet('data/processed/rooms_obyvak_processed.parquet')
        outdoor_temp = pd.read_parquet('data/processed/outdoor_temp_processed.parquet')

        print(f'✅ PV data: {len(pv_data):,} records')
        print(f'✅ Weather data: {len(weather_data):,} records')
        print(f'✅ Consumption data: {len(consumption_data):,} records')
        print(f'✅ Room data: {len(room_obyvak):,} records')
        print(f'✅ Outdoor temp: {len(outdoor_temp):,} records')
    except Exception as e:
        print(f'❌ Failed to load data: {e}')
        return

    # Train PV Predictor
    print('\n🤖 Training PV Production Predictor...')
    train_pv_predictor(pv_data, weather_data)

    # Train Load Predictor
    print('\n🏠 Training Load Predictor...')
    train_load_predictor(consumption_data)

    # Train Thermal Predictor
    print('\n🌡️ Training Thermal Predictor...')
    train_thermal_predictor(room_obyvak, outdoor_temp)

    print('\n' + '=' * 60)
    print('🎉 ML Model Training Phase Complete!')
    print('📁 Check models/saved/ directory for trained models')


def train_pv_predictor(pv_data, weather_data):
    """Train the PV production predictor."""
    try:
        # Create configuration for PV predictor
        pv_config = {
            'model_name': 'pv_predictor',
            'pv_system': {
                'capacity_kw': 10.0,
                'panel_tilt': 30.0,
                'panel_azimuth': 180.0,
                'latitude': 49.49,
                'longitude': 14.43,
                'altitude': 300
            },
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        }
        
        pv_predictor = PVPredictor(config=pv_config)
        print('  ✅ PV Predictor initialized successfully')
        
        # Prepare training data
        merged_data = pd.merge(pv_data, weather_data, left_index=True, right_index=True, how='inner')
        print(f'  📈 Merged dataset: {len(merged_data):,} records')
        
        if len(merged_data) < 100:
            print('  ⚠️  Insufficient data for training')
            return
            
        # Prepare features and target
        # Use weather features as input
        feature_cols = [
            'temperature_2m', 'shortwave_radiation', 'direct_radiation', 
            'diffuse_radiation', 'cloudcover', 'windspeed_10m'
        ]
        available_features = [col for col in feature_cols if col in merged_data.columns]
        
        if len(available_features) == 0:
            print('  ⚠️  No weather features available')
            return
            
        # Target is PV input power
        if 'InputPower' not in merged_data.columns:
            print('  ⚠️  No InputPower column in PV data')
            return
            
        X = merged_data[available_features].dropna()
        y = merged_data.loc[X.index, 'InputPower']
        
        print(f'  📊 Features: {available_features}')
        print(f'  📊 Training samples: {len(X):,}')
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f'  📚 Training on {len(X_train):,} records')
        print(f'  🔬 Validating on {len(X_val):,} records')
        
        # Train the model
        metrics = pv_predictor.train(X_train, y_train, validation_data=(X_val, y_val))
        print(f'  ✅ PV Model trained successfully!')
        print(f'     MAE: {metrics.mae:.2f} W')
        print(f'     RMSE: {metrics.rmse:.2f} W')
        print(f'     R²: {metrics.r2:.3f}')
        
        # Save the trained model
        pv_predictor.save_model('models/saved/pv_predictor_trained.pkl')
        print(f'  💾 Model saved to models/saved/pv_predictor_trained.pkl')
        
    except Exception as e:
        print(f'  ❌ PV training failed: {e}')
        import traceback
        traceback.print_exc()


def train_load_predictor(consumption_data):
    """Train the load predictor."""
    try:
        # Create configuration for Load predictor
        load_config = {
            'model_name': 'load_predictor',
            'components': ['base_load', 'hvac', 'water_heating', 'appliances'],
            'model_params': {
                'n_estimators': 100,
                'max_depth': 8
            }
        }
        
        load_predictor = LoadPredictor(config=load_config)
        print('  ✅ Load Predictor initialized successfully')
        
        if 'heating_power' not in consumption_data.columns:
            print('  ⚠️  No heating_power column in consumption data')
            return
            
        # Create time-based features
        data_with_features = consumption_data.copy()
        data_with_features['hour'] = data_with_features.index.hour
        data_with_features['day_of_week'] = data_with_features.index.dayofweek
        data_with_features['month'] = data_with_features.index.month
        data_with_features['is_weekend'] = data_with_features.index.dayofweek.isin([5, 6]).astype(int)
        
        # Features and target
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend']
        X = data_with_features[feature_cols].dropna()
        y = data_with_features.loc[X.index, 'heating_power']
        
        print(f'  📊 Training samples: {len(X):,}')
        
        if len(X) < 100:
            print('  ⚠️  Insufficient data for training')
            return
            
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f'  📚 Training on {len(X_train):,} records')
        print(f'  🔬 Validating on {len(X_val):,} records')
        
        # Train the model
        metrics = load_predictor.train(X_train, y_train, validation_data=(X_val, y_val))
        print(f'  ✅ Load Model trained successfully!')
        print(f'     MAE: {metrics.mae:.2f} W')
        print(f'     RMSE: {metrics.rmse:.2f} W')
        print(f'     R²: {metrics.r2:.3f}')
        
        # Save the trained model
        load_predictor.save_model('models/saved/load_predictor_trained.pkl')
        print(f'  💾 Model saved to models/saved/load_predictor_trained.pkl')
        
    except Exception as e:
        print(f'  ❌ Load training failed: {e}')
        import traceback
        traceback.print_exc()


def train_thermal_predictor(room_data, outdoor_temp):
    """Train the thermal predictor."""
    try:
        # Create configuration for Thermal predictor
        thermal_config = {
            'model_name': 'thermal_predictor',
            'room_name': 'obyvak',
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6
            }
        }
        
        thermal_predictor = ThermalPredictor(config=thermal_config)
        print('  ✅ Thermal Predictor initialized successfully')
        
        # Merge room and outdoor temperature data
        merged_thermal = pd.merge(room_data, outdoor_temp, left_index=True, right_index=True, how='inner')
        print(f'  📈 Merged thermal dataset: {len(merged_thermal):,} records')
        
        if 'temperature' not in merged_thermal.columns:
            print('  ⚠️  No temperature column in room data')
            return
            
        # Create features: outdoor temp, time-based features
        data_with_features = merged_thermal.copy()
        data_with_features['hour'] = data_with_features.index.hour
        data_with_features['day_of_week'] = data_with_features.index.dayofweek
        
        # Temperature lag features
        data_with_features['temp_lag_1h'] = data_with_features['temperature'].shift(4)  # 1 hour ago (15min intervals)
        data_with_features['outdoor_temp_lag'] = data_with_features['outdoor_temperature'].shift(1) if 'outdoor_temperature' in data_with_features.columns else data_with_features['temperature_y'].shift(1)
        
        # Features and target (predict next temperature)
        feature_cols = ['hour', 'day_of_week', 'temp_lag_1h']
        if 'outdoor_temperature' in data_with_features.columns:
            feature_cols.append('outdoor_temperature')
        elif 'temperature_y' in data_with_features.columns:
            feature_cols.append('temperature_y')
            
        X = data_with_features[feature_cols].dropna()
        y = data_with_features.loc[X.index, 'temperature']
        
        print(f'  📊 Features: {feature_cols}')
        print(f'  📊 Training samples: {len(X):,}')
        
        if len(X) < 100:
            print('  ⚠️  Insufficient data for training')
            return
            
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f'  📚 Training on {len(X_train):,} records')
        print(f'  🔬 Validating on {len(X_val):,} records')
        
        # Train the model
        metrics = thermal_predictor.train(X_train, y_train, validation_data=(X_val, y_val))
        print(f'  ✅ Thermal Model trained successfully!')
        print(f'     MAE: {metrics.mae:.2f} °C')
        print(f'     RMSE: {metrics.rmse:.2f} °C')
        print(f'     R²: {metrics.r2:.3f}')
        
        # Save the trained model
        thermal_predictor.save_model('models/saved/thermal_predictor_trained.pkl')
        print(f'  💾 Model saved to models/saved/thermal_predictor_trained.pkl')
        
    except Exception as e:
        print(f'  ❌ Thermal training failed: {e}')
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()