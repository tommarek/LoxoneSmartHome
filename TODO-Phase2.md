Based on the completed Phase 1 analysis framework, I'll create a comprehensive Phase 2 TODO that builds upon the existing infrastructure to implement ML models, optimization, and control systems.

# Phase 2 TODO: ML Model Development & Optimization Implementation

## Overview
Phase 2 focuses on building predictive ML models using the features engineered in Phase 1, implementing optimization algorithms, and creating the control system that will replace the basic Growatt controller.

## Folder Structure & Implementation Plan

```
pems_v2/
├── models/                      # 🤖 ML Model Components
│   ├── __init__.py
│   ├── base.py                 # Abstract base classes for all predictors
│   ├── model_registry.py       # Model versioning and management
│   ├── predictors/             # Individual prediction models
│   │   ├── __init__.py
│   │   ├── pv_predictor.py
│   │   ├── thermal_predictor.py
│   │   ├── load_predictor.py
│   │   └── price_predictor.py
│   ├── ensemble/               # Ensemble methods
│   │   ├── __init__.py
│   │   └── weighted_ensemble.py
│   └── training/               # Training pipelines
│       ├── __init__.py
│       ├── trainer.py
│       └── hyperparameter_tuning.py
│
├── optimization/               # 🎯 Optimization Engine
│   ├── __init__.py
│   ├── problem_formulation.py
│   ├── constraints.py
│   ├── objectives.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── milp_solver.py
│   │   └── stochastic_solver.py
│   └── mpc/
│       ├── __init__.py
│       └── mpc_controller.py
│
├── control/                    # 🎮 Control System
│   ├── __init__.py
│   ├── energy_controller.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── loxone_interface.py
│   │   ├── battery_interface.py
│   │   └── ev_interface.py
│   └── safety/
│       ├── __init__.py
│       └── safety_checks.py
│
├── monitoring/                 # 📊 Performance Monitoring
│   ├── __init__.py
│   ├── metrics_collector.py
│   ├── model_monitor.py
│   └── dashboards/
│       ├── __init__.py
│       └── grafana_config.py
│
└── deployment/                 # 🚀 Deployment & Integration
    ├── __init__.py
    ├── service_manager.py
    ├── config_validator.py
    └── backup_controller.py
```

## Detailed Implementation Tasks

### 1. Models Module (`models/`)

#### 1.1 `models/base.py`
```python
"""
Abstract base classes for all predictors.

PURPOSE:
- Define common interface for all prediction models
- Implement model persistence (save/load)
- Provide performance tracking methods
- Handle feature preprocessing pipeline

DATA USED:
- Feature sets from analysis/analyzers/feature_engineering.py
- Historical predictions for performance tracking

INTEGRATION:
- All predictors inherit from BasePredictor
- Uses analysis results for feature definitions
- Integrates with model_registry for versioning
"""
```

**TODO:**
- [ ] Implement `BasePredictor` abstract class with methods:
  - `train(X, y)` - Train model on historical data
  - `predict(X)` - Make predictions
  - `predict_proba(X)` - Prediction with uncertainty
  - `save_model(path)` - Persist trained model
  - `load_model(path)` - Load trained model
  - `get_feature_importance()` - Feature importance scores
  - `evaluate(X, y)` - Model evaluation metrics
- [ ] Implement `ModelMetadata` class for tracking:
  - Model version
  - Training date
  - Feature list
  - Performance metrics
  - Training parameters

#### 1.2 `models/predictors/pv_predictor.py`
```python
"""
PV production prediction model.

PURPOSE:
- Predict PV power output for next 48 hours
- Provide uncertainty quantification (P10, P50, P90)
- Handle weather forecast integration
- Account for seasonal variations

DATA USED:
- Historical PV data from analysis results
- Weather forecast data (cloud cover, temperature, radiation)
- Seasonal patterns from pattern_analysis.py
- Feature set from feature_engineering.create_pv_features()

INTEGRATION:
- Uses weather API for real-time forecasts
- Integrates with optimization for production planning
- Updates predictions based on actual vs predicted analysis
"""
```

**TODO:**
- [ ] Implement `PVPredictor` class:
  - XGBoost as primary model
  - Physical model (PVLib) as baseline
  - Ensemble approach for robustness
- [ ] Weather feature extraction:
  - DNI, DHI, GHI calculation
  - Cloud cover impact modeling
  - Temperature derating effects
- [ ] Uncertainty quantification:
  - Quantile regression for P10, P50, P90
  - Prediction intervals based on weather uncertainty
- [ ] Real-time adaptation:
  - Online learning with recent errors
  - Bias correction based on last 7 days
  - Seasonal adjustment factors

#### 1.3 `models/predictors/thermal_predictor.py`
```python
"""
Room temperature prediction using RC models.

PURPOSE:
- Predict room temperatures for 48-hour horizon
- Model heating system response (binary relay control)
- Account for thermal inertia and weather effects
- Enable optimal heating scheduling

DATA USED:
- RC parameters from thermal_analysis.py
- Historical room temperature data
- Weather forecast (outdoor temperature, solar gains)
- Relay state history

INTEGRATION:
- Provides temperature constraints for optimization
- Uses relay control decisions from optimizer
- Updates RC parameters based on prediction errors
"""
```

**TODO:**
- [ ] Implement state-space thermal model:
  - Discrete-time formulation for 5-min steps
  - Room coupling effects
  - Solar gain estimation
- [ ] Kalman filter for state estimation:
  - Temperature state tracking
  - Parameter adaptation
  - Uncertainty propagation
- [ ] Multi-zone coordination:
  - Heat transfer between rooms
  - Shared heating capacity constraints
- [ ] Comfort prediction:
  - Occupancy-based setpoints
  - Comfort zone violations

#### 1.4 `models/predictors/load_predictor.py`
```python
"""
Base load prediction model.

PURPOSE:
- Predict non-controllable electricity consumption
- Capture daily, weekly, and seasonal patterns
- Provide hourly forecasts for optimization

DATA USED:
- Historical base load from base_load_analysis.py
- Calendar features (holidays, weekends)
- Weather data for weather-sensitive loads
- Anomaly detection results

INTEGRATION:
- Feeds into optimization as fixed load
- Updates based on recent consumption patterns
- Handles special events/anomalies
"""
```

**TODO:**
- [ ] Implement time series models:
  - LightGBM for pattern recognition
  - Separate models for weekday/weekend/holiday
  - Hourly predictions with confidence intervals
- [ ] Feature engineering:
  - Lag features (24h, 168h)
  - Rolling statistics
  - Holiday encoding
- [ ] Anomaly handling:
  - Detect and exclude outliers in training
  - Special event detection
  - Adaptive model updates

### 2. Optimization Module (`optimization/`)

#### 2.1 `optimization/problem_formulation.py`
```python
"""
Energy optimization problem formulation.

PURPOSE:
- Define MILP/MINLP optimization problem
- Set up decision variables and constraints
- Implement multi-objective optimization
- Handle 48-hour rolling horizon

DATA USED:
- Predictions from all ML models
- Current system state (battery SOC, temperatures)
- Electricity prices and export policies
- System constraints from config

INTEGRATION:
- Uses predictions from models module
- Feeds decisions to control module
- Re-optimizes based on MPC feedback
"""
```

**TODO:**
- [ ] Decision variables:
  - Binary heating decisions for 16 rooms × 48 hours
  - Continuous battery charge/discharge power
  - EV charging power profile
  - Grid import/export
- [ ] Objectives:
  - Minimize total electricity cost
  - Maximize self-consumption
  - Minimize peak demand (optional)
- [ ] Time discretization:
  - 1-hour steps for optimization
  - Aggregation from 5-min control intervals
- [ ] Warm start strategies:
  - Use previous solution
  - Heuristic initialization

#### 2.2 `optimization/constraints.py`
```python
"""
Optimization constraint builders.

PURPOSE:
- Build all system constraints for optimization
- Handle physical limitations and comfort requirements
- Implement safety constraints
- Manage grid interaction limits

DATA USED:
- System specifications (battery limits, room power)
- Comfort boundaries from config
- Grid connection limits
- Physical models from predictors

INTEGRATION:
- Called by problem_formulation
- Uses thermal models for temperature constraints
- Validates against safety checks
"""
```

**TODO:**
- [ ] Power balance constraints:
  - PV + grid = load + battery + export
  - Account for conversion losses
- [ ] Battery constraints:
  - SOC limits [10%, 90%]
  - Charge/discharge power limits
  - Prevent simultaneous charge/discharge
  - Degradation-aware cycling limits
- [ ] Thermal comfort constraints:
  - Room temperature bounds
  - Heating power limits (binary × rated power)
  - Minimum off-time between cycles
- [ ] Grid constraints:
  - Maximum import/export limits
  - Export only when price > threshold
  - Power factor requirements

#### 2.3 `optimization/mpc/mpc_controller.py`
```python
"""
Model Predictive Control implementation.

PURPOSE:
- Implement receding horizon control
- Handle disturbances and model updates
- Coordinate re-optimization triggers
- Manage computational resources

DATA USED:
- Current system state from monitoring
- Updated predictions from models
- Optimization results from solver
- Actual vs planned deviations

INTEGRATION:
- Main loop in energy_controller
- Triggers re-optimization
- Updates model predictions
- Feeds back to ML models
"""
```

**TODO:**
- [ ] MPC configuration:
  - 48-hour prediction horizon
  - 24-hour control horizon
  - 1-hour re-optimization frequency
- [ ] Disturbance handling:
  - Measure prediction errors
  - Update state estimates
  - Trigger immediate re-optimization if needed
- [ ] Move blocking:
  - Detailed control for next 6 hours
  - Aggregated decisions for 6-24 hours
  - Rough plan for 24-48 hours
- [ ] Computational management:
  - Time limits for optimization
  - Fallback to simpler models if needed

### 3. Control Module (`control/`)

#### 3.1 `control/energy_controller.py`
```python
"""
Main energy management controller.

PURPOSE:
- Replace growatt_controller.py with advanced control
- Coordinate all subsystems (heating, battery, EV)
- Implement MPC main loop
- Handle fault conditions and fallbacks

DATA USED:
- Current state from InfluxDB
- Optimization results from MPC
- System status from interfaces
- Safety limits from config

INTEGRATION:
- Main entry point for control system
- Replaces existing main.py controller
- Interfaces with all hardware systems
- Logs all decisions to InfluxDB
"""
```

**TODO:**
- [ ] Initialization:
  - Load all ML models
  - Initialize optimization engine
  - Set up interface connections
  - Configure logging
- [ ] Main control loop (5-minute):
  - Read current state
  - Check for re-optimization triggers
  - Apply control decisions
  - Log metrics
- [ ] State management:
  - Track execution vs plan
  - Detect deviations
  - Handle manual overrides
- [ ] Fault handling:
  - Fallback to rule-based control
  - Safe mode operation
  - Alert generation

#### 3.2 `control/interfaces/loxone_interface.py`
```python
"""
Loxone system control interface.

PURPOSE:
- Send relay control commands via MQTT
- Read current relay states
- Handle Loxone-specific protocols
- Implement command queuing

DATA USED:
- Relay decisions from optimizer
- Current states from Loxone
- Room mapping from config
- Command acknowledgments

INTEGRATION:
- Called by energy_controller
- Uses existing MQTT infrastructure
- Maps to Loxone field names via adapter
- Handles async communication
"""
```

**TODO:**
- [ ] MQTT command structure:
  - Topic: `loxone/relay/{room}/set`
  - Payload: `{"state": 0/1, "until": timestamp}`
  - QoS level 2 for reliability
- [ ] State verification:
  - Read back commanded state
  - Retry on failure
  - Alert on persistent errors
- [ ] Batch commands:
  - Queue multiple relay changes
  - Coordinate switching order
  - Respect minimum switching intervals

### 4. Monitoring Module (`monitoring/`)

#### 4.1 `monitoring/metrics_collector.py`
```python
"""
System performance metrics collection.

PURPOSE:
- Track prediction accuracy
- Monitor optimization performance
- Calculate cost savings
- Generate performance reports

DATA USED:
- Predictions vs actuals from all models
- Optimization decisions and costs
- System state history
- Baseline comparisons

INTEGRATION:
- Stores metrics in InfluxDB
- Feeds Grafana dashboards
- Triggers model retraining
- Generates daily reports
"""
```

**TODO:**
- [ ] Prediction metrics:
  - RMSE for each predictor
  - Bias detection
  - Uncertainty calibration
- [ ] Optimization metrics:
  - Cost savings vs baseline
  - Self-consumption rate
  - Computation time
  - Constraint violations
- [ ] System metrics:
  - Uptime and reliability
  - Response times
  - Error rates
- [ ] Automated reporting:
  - Daily performance summary
  - Weekly trends
  - Monthly cost analysis

### 5. Training Pipeline Tasks

#### 5.1 Initial Model Training
**TODO:**
- [ ] Load Phase 1 analysis results
- [ ] Split data into train/validation/test sets
- [ ] Train PV predictor on 2 years of data
- [ ] Train thermal models for each room
- [ ] Train base load predictor
- [ ] Evaluate ensemble performance
- [ ] Save trained models with versioning

#### 5.2 Feature Pipeline
**TODO:**
- [ ] Implement real-time feature calculation
- [ ] Create feature store for online predictions
- [ ] Implement feature monitoring
- [ ] Handle missing data in production

## Implementation Timeline

### Week 1-2: Core ML Models
- [ ] Implement base model classes
- [ ] Develop PV predictor with weather integration
- [ ] Create thermal state-space models
- [ ] Build base load predictor

### Week 3: Optimization Engine
- [ ] Formulate MILP problem
- [ ] Implement constraint builders
- [ ] Integrate GEKKO/PuLP solver
- [ ] Test optimization convergence

### Week 4: Control System
- [ ] Develop main controller
- [ ] Implement Loxone interface
- [ ] Create battery/EV interfaces
- [ ] Add safety checks

### Week 5: Integration & Testing
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Fault scenario testing
- [ ] Documentation

### Week 6: Monitoring & Deployment
- [ ] Set up metrics collection
- [ ] Create Grafana dashboards
- [ ] Implement model monitoring
- [ ] Deploy in shadow mode

## Key Integration Points

1. **Feature Usage**: All ML models use features from `analysis/analyzers/feature_engineering.py`
2. **Data Flow**: InfluxDB → Feature Engineering → ML Models → Optimization → Control → InfluxDB
3. **State Feedback**: Control results feed back to improve predictions
4. **Configuration**: All parameters from `config/settings.py` and `config/energy_settings.py`
5. **Logging**: Unified logging through `utils/logging.py`

## Success Metrics

- PV prediction RMSE < 10% of capacity
- Temperature prediction MAE < 0.5°C
- Cost reduction > 20% vs baseline
- Self-consumption > 70%
- System uptime > 99.5%

## Risk Mitigation

1. **Fallback Control**: Keep rule-based controller as backup
2. **Shadow Mode**: Run parallel to existing system initially
3. **Gradual Rollout**: Start with one room, expand gradually
4. **Safety Limits**: Hard-coded limits in safety module
5. **Manual Override**: Always allow user intervention

This Phase 2 implementation builds directly on the Phase 1 analysis, using all the extracted features, patterns, and parameters to create a production-ready predictive energy management system.


this might help as well although not complete
I'll provide a comprehensive code review with detailed implementations and recommendations for the PEMS v2 project.

# PEMS v2 Comprehensive Code Review and Implementation Guide

## 1. Data Architecture Issues and Solutions

### Current Problem: Redundant Data Extraction

The current implementation has multiple methods extracting similar data:

```python
# CURRENT PROBLEMATIC CODE in data_extraction.py
async def extract_pv_data(self, start_date, end_date):
    # Extracts: ChargePower, DischargePower, SOC, BatteryTemperature
    ...

async def extract_battery_data(self, start_date, end_date):
    # ALSO extracts: ChargePower, DischargePower, SOC, BatteryVoltage
    ...

async def extract_energy_consumption(self, start_date, end_date):
    # Complex relay state calculations mixed with consumption
    ...
```

### Solution: Unified Data Extraction Architecture

```python
# analysis/core/unified_data_extractor.py
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import asyncio
from dataclasses import dataclass
from influxdb_client import InfluxDBClient

@dataclass
class EnergyDataset:
    """Structured container for all energy-related data."""
    production: pd.DataFrame
    consumption: pd.DataFrame
    storage: pd.DataFrame
    grid_flow: pd.DataFrame
    weather: pd.DataFrame
    prices: Optional[pd.DataFrame] = None
    relay_states: Optional[Dict[str, pd.DataFrame]] = None
    
    def validate(self) -> Dict[str, Any]:
        """Validate data completeness and quality."""
        validation_report = {
            'total_records': {},
            'missing_data': {},
            'time_coverage': {},
            'anomalies': {}
        }
        
        for name, df in self.__dict__.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                validation_report['total_records'][name] = len(df)
                validation_report['missing_data'][name] = df.isnull().sum().to_dict()
                validation_report['time_coverage'][name] = {
                    'start': df.index.min(),
                    'end': df.index.max(),
                    'gaps': self._find_time_gaps(df)
                }
        
        return validation_report
    
    def _find_time_gaps(self, df: pd.DataFrame, threshold_minutes: int = 60) -> List[Tuple[datetime, datetime]]:
        """Find significant gaps in time series data."""
        if len(df) < 2:
            return []
        
        time_diff = df.index.to_series().diff()
        gaps = time_diff[time_diff > pd.Timedelta(minutes=threshold_minutes)]
        
        return [(df.index[i-1], df.index[i]) for i in gaps.index[1:]]

class UnifiedDataExtractor:
    """Unified data extraction eliminating redundancy."""
    
    def __init__(self, client: InfluxDBClient, settings: dict):
        self.client = client
        self.settings = settings
        self.query_api = client.query_api()
        
    async def extract_complete_dataset(
        self, 
        start_date: datetime, 
        end_date: datetime,
        include_relay_states: bool = True
    ) -> EnergyDataset:
        """Extract all energy data in a single, efficient operation."""
        
        # Parallel extraction of all data types
        tasks = [
            self._extract_production_data(start_date, end_date),
            self._extract_consumption_data(start_date, end_date),
            self._extract_storage_data(start_date, end_date),
            self._extract_grid_flow_data(start_date, end_date),
            self._extract_weather_data(start_date, end_date),
            self._extract_price_data(start_date, end_date)
        ]
        
        if include_relay_states:
            tasks.append(self._extract_relay_states(start_date, end_date))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        dataset = EnergyDataset(
            production=results[0] if not isinstance(results[0], Exception) else pd.DataFrame(),
            consumption=results[1] if not isinstance(results[1], Exception) else pd.DataFrame(),
            storage=results[2] if not isinstance(results[2], Exception) else pd.DataFrame(),
            grid_flow=results[3] if not isinstance(results[3], Exception) else pd.DataFrame(),
            weather=results[4] if not isinstance(results[4], Exception) else pd.DataFrame(),
            prices=results[5] if not isinstance(results[5], Exception) else None,
            relay_states=results[6] if len(results) > 6 and not isinstance(results[6], Exception) else None
        )
        
        # Log any extraction errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Extraction task {i} failed: {result}")
        
        return dataset
    
    async def _extract_production_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract all production-related data (PV + Battery production side)."""
        query = f"""
        from(bucket: "{self.settings['bucket_solar']}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => 
              r["_field"] == "InputPower" or
              r["_field"] == "PV1InputPower" or
              r["_field"] == "PV2InputPower" or
              r["_field"] == "PV1Voltage" or
              r["_field"] == "PV2Voltage" or
              r["_field"] == "DischargePower" or
              r["_field"] == "TodayGenerateEnergy" or
              r["_field"] == "TotalGenerateEnergy"
          )
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        
        result = await self._execute_query(query)
        df = self._process_query_result(result)
        
        # Add calculated fields
        if not df.empty:
            df['total_pv_power'] = df.get('PV1InputPower', 0) + df.get('PV2InputPower', 0)
            df['pv_efficiency'] = self._calculate_pv_efficiency(df)
        
        return df
    
    async def _extract_consumption_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract all consumption data including calculated relay consumption."""
        # First get relay states
        relay_query = f"""
        from(bucket: "{self.settings['bucket_solar']}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" and r["tag1"] == "heating")
          |> aggregateWindow(every: 5m, fn: last, createEmpty: false)
          |> pivot(rowKey:["_time"], columnKey: ["room"], valueColumn: "_value")
        """
        
        relay_result = await self._execute_query(relay_query)
        relay_df = self._process_query_result(relay_result)
        
        # Calculate consumption from relay states
        consumption_df = pd.DataFrame(index=relay_df.index)
        
        from config.energy_settings import ROOM_CONFIG
        
        for room_name in relay_df.columns:
            if room_name in ROOM_CONFIG['rooms']:
                power_kw = ROOM_CONFIG['rooms'][room_name]['power_kw']
                consumption_df[f'{room_name}_consumption'] = relay_df[room_name] * power_kw * 1000  # W
        
        # Add total consumption
        consumption_df['heating_total'] = consumption_df.sum(axis=1)
        
        # Get other consumption data (base load, etc.)
        other_query = f"""
        from(bucket: "{self.settings['bucket_solar']}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => r["_field"] == "ACPowerToUser" or r["_field"] == "LocalLoadEnergyToday")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        
        other_result = await self._execute_query(other_query)
        other_df = self._process_query_result(other_result)
        
        # Merge all consumption data
        if not other_df.empty:
            consumption_df = consumption_df.join(other_df, how='outer')
        
        return consumption_df
    
    async def _extract_storage_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract battery storage data."""
        query = f"""
        from(bucket: "{self.settings['bucket_solar']}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => 
              r["_field"] == "SOC" or
              r["_field"] == "BatteryVoltage" or
              r["_field"] == "BatteryTemperature" or
              r["_field"] == "ChargePower" or
              r["_field"] == "DischargePower" or
              r["_field"] == "ChargeEnergyToday" or
              r["_field"] == "DischargeEnergyToday"
          )
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        
        result = await self._execute_query(query)
        df = self._process_query_result(result)
        
        # Add calculated fields
        if not df.empty:
            df['net_battery_power'] = df.get('ChargePower', 0) - df.get('DischargePower', 0)
            df['battery_energy_change'] = df['net_battery_power'] * (5/60) / 1000  # kWh
        
        return df
```

## 2. Missing Core Predictor Implementations

### PV Production Predictor

```python
# modules/predictors/pv_predictor.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from datetime import datetime, timedelta
import pvlib
import joblib

class PVPredictor:
    """Advanced PV production predictor with uncertainty quantification."""
    
    def __init__(self, system_config: Dict[str, Any]):
        self.config = system_config
        self.models = {
            'clear_sky': self._init_clearsky_model(),
            'ml_model': None,
            'ensemble_weights': {'clear_sky': 0.3, 'ml': 0.7}
        }
        self.feature_columns = None
        self.scaler = None
        
    def _init_clearsky_model(self) -> pvlib.pvsystem.PVSystem:
        """Initialize physical clear-sky model using pvlib."""
        location = pvlib.location.Location(
            latitude=self.config['latitude'],
            longitude=self.config['longitude'],
            tz='Europe/Prague',
            altitude=self.config.get('altitude', 300)
        )
        
        # Define PV system based on configuration
        system = pvlib.pvsystem.PVSystem(
            surface_tilt=self.config.get('tilt', 35),
            surface_azimuth=self.config.get('azimuth', 180),
            module_parameters=self.config.get('module_params', {
                'pdc0': 10000,  # 10kW system
                'gamma_pdc': -0.004  # Temperature coefficient
            }),
            inverter_parameters=self.config.get('inverter_params', {
                'pdc0': 10000,
                'eta_inv_nom': 0.96
            })
        )
        
        return {'location': location, 'system': system}
    
    def train(self, 
              historical_data: pd.DataFrame, 
              weather_data: pd.DataFrame,
              validation_split: float = 0.2) -> Dict[str, float]:
        """Train ML model on historical data."""
        
        # Feature engineering
        features_df = self._engineer_features(historical_data, weather_data)
        
        # Remove missing values
        features_df = features_df.dropna()
        
        # Split features and target
        target_col = 'pv_power'
        feature_cols = [col for col in features_df.columns if col != target_col]
        self.feature_columns = feature_cols
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train XGBoost model
        self.models['ml_model'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['ml_model'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Calculate validation metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = self.models['ml_model'].predict(X_val)
        
        metrics = {
            'mae': mean_absolute_error(y_val, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
            'r2': r2_score(y_val, predictions),
            'mape': np.mean(np.abs((y_val - predictions) / (y_val + 1))) * 100
        }
        
        # Feature importance
        if hasattr(self.models['ml_model'], 'feature_importances_'):
            feature_importance = dict(zip(
                feature_cols,
                self.models['ml_model'].feature_importances_
            ))
            metrics['feature_importance'] = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        
        return metrics
    
    def predict(self, 
                weather_forecast: pd.DataFrame,
                include_uncertainty: bool = True) -> pd.DataFrame:
        """Generate PV production forecast with uncertainty bands."""
        
        # Generate clear-sky baseline
        clearsky_pred = self._predict_clearsky(weather_forecast)
        
        predictions_df = pd.DataFrame(index=weather_forecast.index)
        
        if self.models['ml_model'] is not None:
            # Prepare features
            features = self._prepare_features(weather_forecast)
            features_scaled = self.scaler.transform(features[self.feature_columns])
            
            # ML predictions
            ml_pred = self.models['ml_model'].predict(features_scaled)
            
            # Ensemble prediction
            predictions_df['prediction'] = (
                self.models['ensemble_weights']['clear_sky'] * clearsky_pred +
                self.models['ensemble_weights']['ml'] * ml_pred
            )
            
            if include_uncertainty:
                # Generate prediction intervals using quantile regression
                predictions_df['p10'] = self._predict_quantile(features_scaled, 0.1)
                predictions_df['p90'] = self._predict_quantile(features_scaled, 0.9)
                
                # Add temporal uncertainty (increases with forecast horizon)
                horizon_hours = np.arange(len(predictions_df)) / 4  # 15-min intervals
                uncertainty_factor = 1 + 0.02 * horizon_hours  # 2% per hour
                
                predictions_df['p10'] *= (2 - uncertainty_factor)
                predictions_df['p90'] *= uncertainty_factor
        else:
            # Use only clear-sky model
            predictions_df['prediction'] = clearsky_pred
            
            if include_uncertainty:
                # Simple uncertainty based on cloud cover
                cloud_factor = weather_forecast.get('cloud_cover', 0) / 100
                predictions_df['p10'] = clearsky_pred * (1 - 0.5 * cloud_factor)
                predictions_df['p90'] = clearsky_pred
        
        # Post-processing
        predictions_df = self._postprocess_predictions(predictions_df)
        
        return predictions_df
    
    def _engineer_features(self, 
                          pv_data: pd.DataFrame, 
                          weather_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model training."""
        
        # Merge PV and weather data
        df = pv_data.join(weather_data, how='inner')
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Solar position features
        solar_position = pvlib.solarposition.get_solarposition(
            df.index,
            self.config['latitude'],
            self.config['longitude']
        )
        
        df['solar_elevation'] = solar_position['elevation']
        df['solar_azimuth'] = solar_position['azimuth']
        df['air_mass'] = pvlib.atmosphere.get_relative_airmass(
            solar_position['apparent_zenith']
        )
        
        # Weather features
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_effect'] = 1 - 0.004 * (df['temperature'] - 25)
        
        if 'wind_speed' in df.columns:
            df['wind_cooling'] = np.sqrt(df['wind_speed'])
        
        # Clear-sky radiation
        clearsky = self._calculate_clearsky_radiation(df.index)
        df['clearsky_ghi'] = clearsky['ghi']
        df['clearsky_dni'] = clearsky['dni']
        df['clearsky_dhi'] = clearsky['dhi']
        
        # Radiation ratios
        if 'ghi' in df.columns:
            df['clearness_index'] = df['ghi'] / (df['clearsky_ghi'] + 1)
        
        # Lag features
        df['pv_power_lag_1h'] = df['pv_power'].shift(4)  # 1 hour ago
        df['pv_power_lag_24h'] = df['pv_power'].shift(96)  # 24 hours ago
        
        # Rolling statistics
        df['pv_power_rolling_mean_1h'] = df['pv_power'].rolling(4).mean()
        df['pv_power_rolling_std_1h'] = df['pv_power'].rolling(4).std()
        
        return df
    
    def _predict_clearsky(self, weather_df: pd.DataFrame) -> np.ndarray:
        """Generate clear-sky PV production estimate."""
        
        location = self.models['clear_sky']['location']
        system = self.models['clear_sky']['system']
        
        # Calculate solar position
        solar_position = location.get_solarposition(weather_df.index)
        
        # Get clear-sky radiation
        clearsky = location.get_clearsky(weather_df.index)
        
        # Calculate POA irradiance
        poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
            system.surface_tilt,
            system.surface_azimuth,
            solar_position['apparent_zenith'],
            solar_position['azimuth'],
            clearsky['dni'],
            clearsky['ghi'],
            clearsky['dhi']
        )
        
        poa_global = pvlib.irradiance.get_total_irradiance(
            system.surface_tilt,
            system.surface_azimuth,
            solar_position['apparent_zenith'],
            solar_position['azimuth'],
            clearsky['dni'],
            clearsky['ghi'],
            clearsky['dhi']
        )
        
        # Temperature model (if available)
        if 'temperature' in weather_df.columns:
            cell_temp = pvlib.temperature.pvsyst_cell(
                poa_global['poa_global'],
                weather_df['temperature'],
                weather_df.get('wind_speed', 1)
            )
        else:
            cell_temp = 25  # Default
        
        # DC power
        dc_power = pvlib.pvsystem.pvwatts_dc(
            poa_global['poa_global'],
            cell_temp,
            self.config.get('pdc0', 10000),
            self.config.get('gamma_pdc', -0.004)
        )
        
        # AC power (including inverter efficiency)
        ac_power = pvlib.inverter.pvwatts(
            dc_power,
            self.config.get('pdc0', 10000),
            self.config.get('eta_inv_nom', 0.96)
        )
        
        # Apply cloud cover if available
        if 'cloud_cover' in weather_df.columns:
            cloud_factor = 1 - weather_df['cloud_cover'] / 100 * 0.8
            ac_power *= cloud_factor
        
        return np.maximum(ac_power, 0)
    
    def _predict_quantile(self, X: np.ndarray, quantile: float) -> np.ndarray:
        """Predict specific quantile using quantile regression."""
        # For now, use simple uncertainty estimation
        # In production, train separate quantile regression models
        base_pred = self.models['ml_model'].predict(X)
        
        if quantile < 0.5:
            return base_pred * (0.5 + quantile)
        else:
            return base_pred * (0.5 + quantile)
    
    def _postprocess_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Post-process predictions for physical constraints."""
        
        # Ensure non-negative
        for col in predictions.columns:
            predictions[col] = np.maximum(predictions[col], 0)
        
        # Cap at system maximum
        max_power = self.config.get('pdc0', 10000) * self.config.get('eta_inv_nom', 0.96)
        for col in predictions.columns:
            predictions[col] = np.minimum(predictions[col], max_power)
        
        # Set to zero during night (no sun elevation data)
        night_mask = predictions.index.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
        predictions.loc[night_mask] = 0
        
        # Smooth predictions to avoid unrealistic jumps
        for col in predictions.columns:
            predictions[col] = predictions[col].rolling(3, center=True, min_periods=1).mean()
        
        return predictions
    
    def update_online(self, 
                     recent_data: pd.DataFrame,
                     weather_data: pd.DataFrame) -> None:
        """Update model with recent prediction errors."""
        
        # Calculate recent errors
        features = self._engineer_features(recent_data, weather_data)
        X = features[self.feature_columns]
        y_true = recent_data['pv_power']
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.models['ml_model'].predict(X_scaled)
        
        errors = y_true - y_pred
        
        # Update ensemble weights based on recent performance
        clearsky_pred = self._predict_clearsky(weather_data)
        clearsky_errors = y_true - clearsky_pred
        
        ml_mae = np.mean(np.abs(errors))
        clearsky_mae = np.mean(np.abs(clearsky_errors))
        
        # Adjust weights inversely proportional to error
        total_error = ml_mae + clearsky_mae
        if total_error > 0:
            self.models['ensemble_weights']['ml'] = 1 - ml_mae / total_error
            self.models['ensemble_weights']['clear_sky'] = 1 - clearsky_mae / total_error
        
        # Bias correction
        bias = np.mean(errors)
        if abs(bias) > 50:  # Significant bias
            # Apply exponential smoothing to bias correction
            self._bias_correction = getattr(self, '_bias_correction', 0) * 0.9 + bias * 0.1
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        model_data = {
            'ml_model': self.models['ml_model'],
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'config': self.config,
            'ensemble_weights': self.models['ensemble_weights']
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.models['ml_model'] = model_data['ml_model']
        self.feature_columns = model_data['feature_columns']
        self.scaler = model_data['scaler']
        self.models['ensemble_weights'] = model_data['ensemble_weights']
```

## 3. Thermal Model Implementation

```python
# modules/predictors/thermal_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter
import json

class ThermalModel:
    """Multi-zone thermal model with inter-room heat transfer."""
    
    def __init__(self, building_config: Dict[str, Any]):
        self.config = building_config
        self.rooms = list(building_config['rooms'].keys())
        self.n_rooms = len(self.rooms)
        
        # Model parameters (to be identified)
        self.params = {
            'R': {},  # Thermal resistance [K/W]
            'C': {},  # Thermal capacitance [J/K]
            'R_coupling': {},  # Inter-room resistance [K/W]
            'solar_aperture': {},  # Solar gain coefficient
            'internal_gains': {}  # Base internal heat gains [W]
        }
        
        # Kalman filter for state estimation
        self.kf = self._init_kalman_filter()
        
        # Model state
        self.current_state = None
        
    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter for temperature estimation."""
        kf = KalmanFilter(dim_x=self.n_rooms, dim_z=self.n_rooms)
        
        # State transition matrix (discrete time dynamics)
        # Will be updated based on identified parameters
        kf.F = np.eye(self.n_rooms)
        
        # Measurement matrix (direct temperature measurements)
        kf.H = np.eye(self.n_rooms)
        
        # Process noise
        kf.Q = np.eye(self.n_rooms) * 0.01  # Temperature variance
        
        # Measurement noise
        kf.R = np.eye(self.n_rooms) * 0.1  # Sensor variance
        
        # Initial state
        kf.x = np.ones(self.n_rooms) * 20  # 20°C initial guess
        
        # Initial covariance
        kf.P = np.eye(self.n_rooms) * 1.0
        
        return kf
    
    def identify_parameters(self,
                          historical_data: Dict[str, pd.DataFrame],
                          weather_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify thermal parameters from historical data."""
        
        # Prepare training data
        X, y = self._prepare_training_data(historical_data, weather_data)
        
        # Initial parameter guess
        x0 = self._get_initial_params()
        
        # Bounds for parameters
        bounds = self._get_param_bounds()
        
        # Optimize parameters
        result = minimize(
            fun=lambda x: self._objective_function(x, X, y),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        # Extract optimized parameters
        self._set_params_from_vector(result.x)
        
        # Validate model
        validation_metrics = self._validate_model(X, y)
        
        return {
            'parameters': self.params,
            'optimization_result': {
                'success': result.success,
                'final_cost': result.fun,
                'iterations': result.nit
            },
            'validation_metrics': validation_metrics
        }
    
    def _prepare_training_data(self,
                              historical_data: Dict[str, pd.DataFrame],
                              weather_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for parameter identification."""
        
        # Align all room data
        aligned_data = {}
        common_index = None
        
        for room, df in historical_data.items():
            if 'temperature' in df.columns and 'heating_on' in df.columns:
                room_data = df[['temperature', 'heating_on']].copy()
                
                if common_index is None:
                    common_index = room_data.index
                else:
                    common_index = common_index.intersection(room_data.index)
                
                aligned_data[room] = room_data
        
        # Create feature matrix
        features = pd.DataFrame(index=common_index)
        targets = pd.DataFrame(index=common_index)
        
        # Room temperatures and heating states
        for i, room in enumerate(self.rooms):
            if room in aligned_data:
                features[f'T_{room}'] = aligned_data[room]['temperature']
                features[f'P_{room}'] = aligned_data[room]['heating_on'] * \
                                       self.config['rooms'][room]['power_kw'] * 1000
                targets[f'T_next_{room}'] = aligned_data[room]['temperature'].shift(-1)
        
        # Weather features
        if 'temperature' in weather_data.columns:
            features['T_out'] = weather_data['temperature'].reindex(common_index)
        
        if 'solar_irradiance' in weather_data.columns:
            features['solar'] = weather_data['solar_irradiance'].reindex(common_index)
        
        # Remove NaN values
        valid_mask = features.notna().all(axis=1) & targets.notna().all(axis=1)
        
        return features[valid_mask], targets[valid_mask]
    
    def _objective_function(self, params: np.ndarray, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """Objective function for parameter optimization."""
        
        # Set parameters
        self._set_params_from_vector(params)
        
        # Predict temperatures
        predictions = self._predict_batch(X)
        
        # Calculate MSE
        mse = 0
        for room in self.rooms:
            if f'T_next_{room}' in y.columns and room in predictions:
                error = y[f'T_next_{room}'] - predictions[room]
                mse += np.mean(error ** 2)
        
        return mse / len(self.rooms)
    
    def _predict_batch(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict temperature evolution for batch of data."""
        
        predictions = {room: [] for room in self.rooms}
        dt = 5 * 60  # 5 minutes in seconds
        
        for idx in X.index:
            # Current state
            T_current = {room: X.loc[idx, f'T_{room}'] for room in self.rooms 
                        if f'T_{room}' in X.columns}
            
            # Inputs
            P_heating = {room: X.loc[idx, f'P_{room}'] for room in self.rooms 
                        if f'P_{room}' in X.columns}
            T_out = X.loc[idx, 'T_out'] if 'T_out' in X.columns else 20
            solar = X.loc[idx, 'solar'] if 'solar' in X.columns else 0
            
            # Predict next state
            T_next = self._state_transition(T_current, P_heating, T_out, solar, dt)
            
            for room in self.rooms:
                if room in T_next:
                    predictions[room].append(T_next[room])
        
        return {room: np.array(pred) for room, pred in predictions.items()}
    
    def _state_transition(self,
                         T_current: Dict[str, float],
                         P_heating: Dict[str, float],
                         T_out: float,
                         solar: float,
                         dt: float) -> Dict[str, float]:
        """Calculate next state using thermal dynamics."""
        
        T_next = {}
        
        for room in self.rooms:
            if room not in T_current:
                continue
            
            # Current temperature
            T = T_current[room]
            
            # Heat flows
            Q_heating = P_heating.get(room, 0)
            Q_outdoor = (T_out - T) / self.params['R'].get(room, 0.005)
            Q_solar = solar * self.params['solar_aperture'].get(room, 0.001)
            Q_internal = self.params['internal_gains'].get(room, 100)
            
            # Inter-room heat transfer
            Q_coupling = 0
            for other_room in self.rooms:
                if other_room != room and other_room in T_current:
                    coupling_key = f'{room}_{other_room}'
                    if coupling_key in self.params['R_coupling']:
                        R_coup = self.params['R_coupling'][coupling_key]
                        Q_coupling += (T_current[other_room] - T) / R_coup
            
            # Total heat flow
            Q_total = Q_heating + Q_outdoor + Q_solar + Q_internal + Q_coupling
            
            # Temperature change
            C = self.params['C'].get(room, 1e7)  # J/K
            dT = Q_total * dt / C
            
            T_next[room] = T + dT
        
        return T_next
    
    def predict(self,
                initial_state: Dict[str, float],
                control_sequence: pd.DataFrame,
                weather_forecast: pd.DataFrame,
                horizon_hours: int = 48) -> pd.DataFrame:
        """Predict temperature evolution given control sequence."""
        
        # Time steps
        dt = 5 * 60  # 5 minutes
        n_steps = horizon_hours * 12  # 5-minute intervals
        
        # Initialize predictions
        predictions = pd.DataFrame(
            index=pd.date_range(
                start=control_sequence.index[0],
                periods=n_steps,
                freq='5min'
            ),
            columns=[f'T_{room}' for room in self.rooms]
        )
        
        # Initial state
        current_state = initial_state.copy()
        self.kf.x = np.array([initial_state.get(room, 20) for room in self.rooms])
        
        # Simulate
        for i in range(n_steps):
            time_idx = predictions.index[i]
            
            # Get inputs
            P_heating = {}
            for room in self.rooms:
                if f'heating_{room}' in control_sequence.columns:
                    # Find nearest control time
                    control_idx = control_sequence.index.get_indexer([time_idx], method='nearest')[0]
                    heating_on = control_sequence.iloc[control_idx][f'heating_{room}']
                    P_heating[room] = heating_on * self.config['rooms'][room]['power_kw'] * 1000
                else:
                    P_heating[room] = 0
            
            # Weather inputs
            weather_idx = weather_forecast.index.get_indexer([time_idx], method='nearest')[0]
            T_out = weather_forecast.iloc[weather_idx].get('temperature', 10)
            solar = weather_forecast.iloc[weather_idx].get('solar_irradiance', 0)
            
            # State transition
            current_state = self._state_transition(
                current_state, P_heating, T_out, solar, dt
            )
            
            # Kalman filter update (if measurements available)
            self.kf.predict()
            
            # Store predictions
            for j, room in enumerate(self.rooms):
                predictions.loc[time_idx, f'T_{room}'] = current_state.get(room, 20)
                self.kf.x[j] = current_state.get(room, 20)
        
        return predictions
    
    def get_comfort_constraints(self, 
                               time_range: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
        """Get temperature comfort constraints for optimization."""
        
        constraints = {}
        
        for room in self.rooms:
            room_config = self.config['rooms'][room]
            
            # Create constraint DataFrame
            df = pd.DataFrame(index=time_range)
            
            # Default comfort range
            default_min = room_config.get('temp_min', 19)
            default_max = room_config.get('temp_max', 23)
            
            # Time-based constraints
            df['T_min'] = default_min
            df['T_max'] = default_max
            
            # Night setback
            night_hours = (time_range.hour >= 22) | (time_range.hour < 6)
            df.loc[night_hours, 'T_min'] = room_config.get('temp_min_night', 17)
            df.loc[night_hours, 'T_max'] = room_config.get('temp_max_night', 21)
            
            # Away mode
            weekday_away = (time_range.weekday < 5) & \
                          (time_range.hour >= 9) & \
                          (time_range.hour < 17)
            df.loc[weekday_away, 'T_min'] = room_config.get('temp_min_away', 16)
            df.loc[weekday_away, 'T_max'] = room_config.get('temp_max_away', 20)
            
            constraints[room] = df
        
        return constraints
    
    def _get_initial_params(self) -> np.ndarray:
        """Get initial parameter vector for optimization."""
        params = []
        
        # R values (thermal resistance)
        for room in self.rooms:
            params.append(0.005)  # K/W
        
        # C values (thermal capacitance)  
        for room in self.rooms:
            room_volume = self.config['rooms'][room].get('volume', 50)  # m³
            params.append(room_volume * 1.2 * 1005 * 3)  # J/K
        
        # R_coupling values (simplified: adjacent rooms only)
        n_couplings = self.n_rooms * (self.n_rooms - 1) // 2
        params.extend([0.01] * n_couplings)  # K/W
        
        # Solar aperture
        for room in self.rooms:
            params.append(0.001)  # m²
        
        # Internal gains
        for room in self.rooms:
            params.append(100)  # W
        
        return np.array(params)
    
    def _get_param_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        # R values: 0.001 to 0.05 K/W
        bounds.extend([(0.001, 0.05)] * self.n_rooms)
        
        # C values: 1e6 to 1e8 J/K
        bounds.extend([(1e6, 1e8)] * self.n_rooms)
        
        # R_coupling: 0.005 to 0.1 K/W
        n_couplings = self.n_rooms * (self.n_rooms - 1) // 2
        bounds.extend([(0.005, 0.1)] * n_couplings)
        
        # Solar aperture: 0 to 0.01
        bounds.extend([(0, 0.01)] * self.n_rooms)
        
        # Internal gains: 0 to 500 W
        bounds.extend([(0, 500)] * self.n_rooms)
        
        return bounds
    
    def _set_params_from_vector(self, params: np.ndarray):
        """Set model parameters from optimization vector."""
        idx = 0
        
        # R values
        for room in self.rooms:
            self.params['R'][room] = params[idx]
            idx += 1
        
        # C values
        for room in self.rooms:
            self.params['C'][room] = params[idx]
            idx += 1
        
        # R_coupling values
        for i, room1 in enumerate(self.rooms):
            for room2 in self.rooms[i+1:]:
                coupling_key = f'{room1}_{room2}'
                self.params['R_coupling'][coupling_key] = params[idx]
                self.params['R_coupling'][f'{room2}_{room1}'] = params[idx]  # Symmetric
                idx += 1
        
        # Solar aperture
        for room in self.rooms:
            self.params['solar_aperture'][room] = params[idx]
            idx += 1
        
        # Internal gains
        for room in self.rooms:
            self.params['internal_gains'][room] = params[idx]
            idx += 1
    
    def _validate_model(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Validate model performance."""
        predictions = self._predict_batch(X)
        
        metrics = {}
        
        for room in self.rooms:
            if f'T_next_{room}' in y.columns and room in predictions:
                y_true = y[f'T_next_{room}']
                y_pred = predictions[room]
                
                # Calculate metrics
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                # R² score
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                metrics[room] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
        
        # Average metrics
        avg_metrics = {
            'avg_mae': np.mean([m['mae'] for m in metrics.values()]),
            'avg_rmse': np.mean([m['rmse'] for m in metrics.values()]),
            'avg_r2': np.mean([m['r2'] for m in metrics.values()])
        }
        
        return {'room_metrics': metrics, 'average_metrics': avg_metrics}
    
    def save_model(self, path: str):
        """Save model parameters to file."""
        model_data = {
            'config': self.config,
            'params': self.params,
            'rooms': self.rooms
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, path: str):
        """Load model parameters from file."""
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        self.config = model_data['config']
        self.params = model_data['params']
        self.rooms = model_data['rooms']
        self.n_rooms = len(self.rooms)
        
        # Reinitialize Kalman filter
        self.kf = self._init_kalman_filter()
```

## 4. Optimization Engine Implementation

```python
# modules/optimization/optimizer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import logging

@dataclass
class OptimizationProblem:
    """Container for optimization problem definition."""
    horizon_hours: int
    time_step_minutes: int
    rooms: List[str]
    prices: pd.Series
    pv_forecast: pd.DataFrame
    load_forecast: pd.Series
    temperature_forecast: pd.Series
    initial_battery_soc: float
    initial_temperatures: Dict[str, float]
    comfort_constraints: Dict[str, pd.DataFrame]
    
    @property
    def n_steps(self) -> int:
        return self.horizon_hours * 60 // self.time_step_minutes
    
    @property
    def time_index(self) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.prices.index[0],
            periods=self.n_steps,
            freq=f'{self.time_step_minutes}min'
        )

class EnergyOptimizer:
    """Multi-objective energy optimization with hierarchical approach."""
    
    def __init__(self, system_config: Dict[str, Any]):
        self.config = system_config
        self.logger = logging.getLogger(__name__)
        
        # Component models
        self.thermal_model = None
        self.battery_model = None
        
        # Optimization settings
        self.settings = {
            'time_limit': 30,  # seconds
            'mip_gap': 0.01,  # 1% optimality gap
            'weights': {
                'cost': 0.7,
                'self_consumption': 0.2,
                'peak_shaving': 0.1
            }
        }
        
    def set_models(self, thermal_model, battery_model=None):
        """Set component models for optimization."""
        self.thermal_model = thermal_model
        self.battery_model = battery_model
    
    def optimize(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Solve the energy optimization problem."""
        
        try:
            # Create optimization model
            model = gp.Model("energy_optimization")
            
            # Set solver parameters
            model.setParam('TimeLimit', self.settings['time_limit'])
            model.setParam('MIPGap', self.settings['mip_gap'])
            model.setParam('OutputFlag', 0)  # Quiet mode
            
            # Create variables
            vars_dict = self._create_variables(model, problem)
            
            # Add constraints
            self._add_constraints(model, vars_dict, problem)
            
            # Set objective
            self._set_objective(model, vars_dict, problem)
            
            # Solve
            model.optimize()
            
            # Extract solution
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = self._extract_solution(vars_dict, problem)
                solution['solve_time'] = model.Runtime
                solution['optimality_gap'] = model.MIPGap
                solution['objective_value'] = model.ObjVal
                
                return solution
            else:
                self.logger.error(f"Optimization failed with status: {model.status}")
                return self._get_fallback_solution(problem)
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self._get_fallback_solution(problem)
    
    def _create_variables(self, model: gp.Model, problem: OptimizationProblem) -> Dict[str, Any]:
        """Create optimization variables."""
        
        n_steps = problem.n_steps
        vars_dict = {}
        
        # Heating decisions (binary)
        vars_dict['heating'] = {}
        for room in problem.rooms:
            vars_dict['heating'][room] = model.addVars(
                n_steps, vtype=GRB.BINARY, name=f'heating_{room}'
            )
        
        # Battery variables
        if self.config.get('battery'):
            battery_config = self.config['battery']
            
            # Battery power (continuous, can be negative for discharge)
            vars_dict['battery_power'] = model.addVars(
                n_steps, 
                lb=-battery_config['max_discharge_power'],
                ub=battery_config['max_charge_power'],
                name='battery_power'
            )
            
            # Battery SOC
            vars_dict['battery_soc'] = model.addVars(
                n_steps + 1,
                lb=battery_config['min_soc'],
                ub=battery_config['max_soc'],
                name='battery_soc'
            )
            
            # Separate charge/discharge for efficiency
            vars_dict['battery_charge'] = model.addVars(
                n_steps, lb=0, name='battery_charge'
            )
            vars_dict['battery_discharge'] = model.addVars(
                n_steps, lb=0, name='battery_discharge'
            )
        
        # Grid variables
        vars_dict['grid_import'] = model.addVars(
            n_steps, lb=0, ub=self.config['grid']['max_import'], name='grid_import'
        )
        vars_dict['grid_export'] = model.addVars(
            n_steps, lb=0, ub=self.config['grid']['max_export'], name='grid_export'
        )
        
        # Peak demand variable
        vars_dict['peak_demand'] = model.addVar(lb=0, name='peak_demand')
        
        # Temperature variables (if using linear thermal model)
        vars_dict['temperature'] = {}
        for room in problem.rooms:
            vars_dict['temperature'][room] = model.addVars(
                n_steps + 1, lb=10, ub=30, name=f'temperature_{room}'
            )
        
        return vars_dict
    
    def _add_constraints(self, model: gp.Model, vars_dict: Dict, problem: OptimizationProblem):
        """Add optimization constraints."""
        
        n_steps = problem.n_steps
        dt_hours = problem.time_step_minutes / 60
        
        # Power balance constraint
        for t in range(n_steps):
            # Calculate total heating load
            heating_load = gp.quicksum(
                vars_dict['heating'][room][t] * 
                self.config['rooms'][room]['power_kw'] * 1000
                for room in problem.rooms
            )
            
            # Base load
            base_load = problem.load_forecast.iloc[t]
            
            # PV production
            pv_power = problem.pv_forecast.iloc[t]['prediction']
            
            # Power balance: PV + grid_import + battery_discharge = load + grid_export + battery_charge
            if self.config.get('battery'):
                model.addConstr(
                    pv_power + vars_dict['grid_import'][t] + vars_dict['battery_discharge'][t] ==
                    base_load + heating_load + vars_dict['grid_export'][t] + vars_dict['battery_charge'][t],
                    name=f'power_balance_{t}'
                )
            else:
                model.addConstr(
                    pv_power + vars_dict['grid_import'][t] ==
                    base_load + heating_load + vars_dict['grid_export'][t],
                    name=f'power_balance_{t}'
                )
        
        # Battery constraints
        if self.config.get('battery'):
            battery_config = self.config['battery']
            capacity_kwh = battery_config['capacity_kwh']
            
            # Initial SOC
            model.addConstr(
                vars_dict['battery_soc'][0] == problem.initial_battery_soc,
                name='initial_soc'
            )
            
            # SOC dynamics
            for t in range(n_steps):
                # Link charge/discharge to battery power
                model.addConstr(
                    vars_dict['battery_power'][t] == 
                    vars_dict['battery_charge'][t] - vars_dict['battery_discharge'][t],
                    name=f'battery_power_link_{t}'
                )
                
                # SOC update with efficiency
                charge_eff = battery_config['charge_efficiency']
                discharge_eff = battery_config['discharge_efficiency']
                
                model.addConstr(
                    vars_dict['battery_soc'][t + 1] == vars_dict['battery_soc'][t] +
                    (vars_dict['battery_charge'][t] * charge_eff * dt_hours / (capacity_kwh * 1000)) -
                    (vars_dict['battery_discharge'][t] * dt_hours / (discharge_eff * capacity_kwh * 1000)),
                    name=f'soc_dynamics_{t}'
                )
                
                # Power limits based on SOC
                # Reduce power limits near SOC boundaries
                # This requires piecewise linear approximation in Gurobi
        
        # Temperature constraints
        for room in problem.rooms:
            # Initial temperature
            model.addConstr(
                vars_dict['temperature'][room][0] == problem.initial_temperatures[room],
                name=f'initial_temp_{room}'
            )
            
            # Temperature dynamics (simplified linear model)
            R = self.thermal_model.params['R'][room]
            C = self.thermal_model.params['C'][room]
            tau = R * C / 3600  # Time constant in hours
            
            for t in range(n_steps):
                # Discrete time dynamics
                T_out = problem.temperature_forecast.iloc[t]
                P_heat = vars_dict['heating'][room][t] * \
                        self.config['rooms'][room]['power_kw'] * 1000
                
                # Linear approximation of exponential dynamics
                alpha = np.exp(-dt_hours / tau)
                
                model.addConstr(
                    vars_dict['temperature'][room][t + 1] == 
                    alpha * vars_dict['temperature'][room][t] +
                    (1 - alpha) * (T_out + R * P_heat),
                    name=f'temp_dynamics_{room}_{t}'
                )
                
                # Comfort constraints
                T_min = problem.comfort_constraints[room].iloc[t]['T_min']
                T_max = problem.comfort_constraints[room].iloc[t]['T_max']
                
                model.addConstr(
                    vars_dict['temperature'][room][t] >= T_min,
                    name=f'temp_min_{room}_{t}'
                )
                model.addConstr(
                    vars_dict['temperature'][room][t] <= T_max,
                    name=f'temp_max_{room}_{t}'
                )
        
        # Peak demand constraint
        for t in range(n_steps):
            model.addConstr(
                vars_dict['grid_import'][t] <= vars_dict['peak_demand'],
                name=f'peak_demand_{t}'
            )
        
        # Minimum switching time for heating (avoid rapid cycling)
        min_on_time = 3  # 15-minute periods
        min_off_time = 2
        
        for room in problem.rooms:
            heating_var = vars_dict['heating'][room]
            
            # Minimum on time
            for t in range(n_steps - min_on_time + 1):
                # If heating turns on at time t, it must stay on for min_on_time
                model.addConstr(
                    heating_var[t + 1] - heating_var[t] <=
                    gp.quicksum(heating_var[t + k] for k in range(1, min_on_time + 1)) / min_on_time,
                    name=f'min_on_time_{room}_{t}'
                )
            
            # Minimum off time (similar logic)
        
        # Grid export constraints (price-based)
        if self.config.get('export_policy'):
            threshold = self.config['export_policy']['price_threshold']
            
            for t in range(n_steps):
                price = problem.prices.iloc[t]
                
                if price < threshold:
                    # No export when price is below threshold
                    model.addConstr(
                        vars_dict['grid_export'][t] == 0,
                        name=f'no_export_{t}'
                    )
    
    def _set_objective(self, model: gp.Model, vars_dict: Dict, problem: OptimizationProblem):
        """Set multi-objective function."""
        
        n_steps = problem.n_steps
        dt_hours = problem.time_step_minutes / 60
        weights = self.settings['weights']
        
        # Objective 1: Minimize energy cost
        energy_cost = gp.quicksum(
            (vars_dict['grid_import'][t] * problem.prices.iloc[t] -
             vars_dict['grid_export'][t] * problem.prices.iloc[t] * 0.9) *  # 90% of import price
            dt_hours / 1000  # Convert to kWh
            for t in range(n_steps)
        )
        
        # Objective 2: Maximize self-consumption
        self_consumption = gp.quicksum(
            problem.pv_forecast.iloc[t]['prediction'] - vars_dict['grid_export'][t]
            for t in range(n_steps)
        )
        
        # Objective 3: Minimize peak demand
        peak_cost = vars_dict['peak_demand'] * self.config['grid'].get('peak_charge', 100)
        
        # Combined objective (minimization)
        model.setObjective(
            weights['cost'] * energy_cost -
            weights['self_consumption'] * self_consumption / 1000 +  # Scale to similar magnitude
            weights['peak_shaving'] * peak_cost,
            GRB.MINIMIZE
        )
    
    def _extract_solution(self, vars_dict: Dict, problem: OptimizationProblem) -> Dict[str, Any]:
        """Extract solution from solved model."""
        
        solution = {
            'heating_schedule': {},
            'battery_schedule': None,
            'grid_schedule': None,
            'temperature_forecast': {},
            'costs': {},
            'metrics': {}
        }
        
        # Extract heating schedules
        for room in problem.rooms:
            schedule = pd.Series(
                [vars_dict['heating'][room][t].X for t in range(problem.n_steps)],
                index=problem.time_index
            )
            solution['heating_schedule'][room] = schedule.astype(int)
        
        # Extract battery schedule
        if self.config.get('battery'):
            solution['battery_schedule'] = pd.DataFrame({
                'power': [vars_dict['battery_power'][t].X for t in range(problem.n_steps)],
                'soc': [vars_dict['battery_soc'][t].X for t in range(problem.n_steps + 1)][:-1],
                'charge': [vars_dict['battery_charge'][t].X for t in range(problem.n_steps)],
                'discharge': [vars_dict['battery_discharge'][t].X for t in range(problem.n_steps)]
            }, index=problem.time_index)
        
        # Extract grid schedule
        solution['grid_schedule'] = pd.DataFrame({
            'import': [vars_dict['grid_import'][t].X for t in range(problem.n_steps)],
            'export': [vars_dict['grid_export'][t].X for t in range(problem.n_steps)]
        }, index=problem.time_index)
        
        # Extract temperature forecasts
        for room in problem.rooms:
            temps = pd.Series(
                [vars_dict['temperature'][room][t].X for t in range(problem.n_steps + 1)][:-1],
                index=problem.time_index
            )
            solution['temperature_forecast'][room] = temps
        
        # Calculate costs and metrics
        dt_hours = problem.time_step_minutes / 60
        
        # Energy costs
        import_cost = sum(
            solution['grid_schedule']['import'].iloc[t] * problem.prices.iloc[t] * dt_hours / 1000
            for t in range(problem.n_steps)
        )
        export_revenue = sum(
            solution['grid_schedule']['export'].iloc[t] * problem.prices.iloc[t] * 0.9 * dt_hours / 1000
            for t in range(problem.n_steps)
        )
        
        solution['costs'] = {
            'import_cost': import_cost,
            'export_revenue': export_revenue,
            'net_cost': import_cost - export_revenue,
            'peak_demand': vars_dict['peak_demand'].X
        }
        
        # Calculate metrics
        total_pv = problem.pv_forecast['prediction'].sum() * dt_hours / 1000
        total_export = solution['grid_schedule']['export'].sum() * dt_hours / 1000
        
        solution['metrics'] = {
            'self_consumption_rate': (total_pv - total_export) / total_pv if total_pv > 0 else 0,
            'peak_reduction':