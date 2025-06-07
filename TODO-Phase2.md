# PEMS v2 Phase 2: Advanced ML-Based Energy Management System

## Executive Summary

Phase 2 transforms the analysis framework from Phase 1 into a production-ready, intelligent energy management system. This phase implements **predictive ML models**, **multi-objective optimization**, and **model predictive control (MPC)** to replace the basic rule-based Growatt controller with an advanced system capable of:

- **20%+ cost reduction** through optimal energy scheduling
- **70%+ self-consumption** via predictive battery management
- **Sub-0.5°C temperature control** using thermal modeling
- **Automatic adaptation** through online learning
- **Fault-tolerant operation** with graceful degradation

### Key Innovations
1. **Hybrid ML/Physics Models**: Combines XGBoost ML with physical PVLib and RC thermal models
2. **Uncertainty Quantification**: P10/P50/P90 predictions for robust optimization
3. **Multi-Objective MILP**: Simultaneous cost, comfort, and self-consumption optimization
4. **Receding Horizon MPC**: 48-hour prediction, 24-hour control, 1-hour re-optimization
5. **Online Learning**: Continuous model adaptation based on prediction errors

## Strategic Implementation Approach

### Development Philosophy
- **Incremental Development**: Each component builds on proven Phase 1 foundations
- **Shadow Mode Deployment**: Run parallel to existing system for validation
- **Graceful Degradation**: Always maintain fallback to rule-based control
- **Comprehensive Testing**: 95%+ test coverage with real-world scenarios
- **Performance Monitoring**: Continuous validation of model accuracy and system performance

### Architecture Principles
- **Modular Design**: Independent, swappable components
- **Async-First**: All I/O operations use async/await patterns
- **Type Safety**: Full mypy compliance with strict checking
- **Resource Management**: Proper cleanup and connection pooling
- **Error Resilience**: Comprehensive exception handling and recovery

---

# DETAILED PHASE 2 IMPLEMENTATION PLAN

## 🔧 1. Data Infrastructure Overhaul

### Problem Statement
Current Phase 1 data extraction has redundancy and inefficiency:
- Multiple methods extract overlapping data
- No unified data validation
- Inconsistent error handling
- Missing data quality metrics

### Solution: Unified Data Pipeline

#### 1.1 Enhanced Data Extraction (`analysis/core/unified_data_extractor.py`)

**PURPOSE**: Single source of truth for all energy data with built-in validation, gap detection, and quality metrics.

**KEY FEATURES**:
- Parallel extraction of all data streams
- Automatic data validation and gap detection
- Quality scoring for ML model input
- Efficient InfluxDB query optimization
- Structured data containers with type safety

```python
# analysis/core/unified_data_extractor.py
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import asyncio
from dataclasses import dataclass
from influxdb_client import InfluxDBClient
import numpy as np
from scipy import stats

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

    def calculate_data_quality_score(self, dataset: EnergyDataset) -> Dict[str, float]:
        """Calculate comprehensive data quality metrics for ML readiness."""
        quality_scores = {}
        
        for component_name, df in dataset.__dict__.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Completeness score
                completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
                
                # Consistency score (check for outliers)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                consistency = 1.0
                if len(numeric_cols) > 0:
                    z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
                    outlier_rate = (z_scores > 3).sum().sum() / (len(df) * len(numeric_cols))
                    consistency = 1 - min(outlier_rate, 0.1) / 0.1
                
                # Temporal consistency
                if hasattr(df.index, 'freq') or len(df) > 1:
                    expected_freq = pd.infer_freq(df.index[:100]) or '5min'
                    resampled = df.resample(expected_freq).count().iloc[:, 0]
                    temporal_consistency = (resampled > 0).mean()
                else:
                    temporal_consistency = 1.0
                
                # Combined score
                quality_scores[component_name] = {
                    'overall': (completeness * 0.4 + consistency * 0.3 + temporal_consistency * 0.3),
                    'completeness': completeness,
                    'consistency': consistency, 
                    'temporal_consistency': temporal_consistency
                }
        
        return quality_scores
```

---

## 🤖 2. Advanced ML Model Implementation

### 2.1 Base Model Infrastructure (`models/base.py`)

**PURPOSE**: Unified interface for all predictive models with automatic versioning, performance tracking, and deployment management.

```python
# models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class ModelMetadata:
    """Comprehensive model metadata for versioning and tracking."""
    model_name: str
    version: str
    training_date: datetime
    features: List[str]
    target_variable: str
    performance_metrics: Dict[str, float]
    training_params: Dict[str, Any]
    data_hash: str
    model_type: str
    deployment_status: str = "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['training_date'] = self.training_date.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data['training_date'] = datetime.fromisoformat(data['training_date'])
        return cls(**data)

class BasePredictor(ABC):
    """Abstract base class for all PEMS predictive models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata: Optional[ModelMetadata] = None
        self.performance_history = []
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, float]:
        """Train the model on historical data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Make predictions on new data."""
        pass
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with uncertainty quantification."""
        # Default implementation - override in specific models
        predictions = self.predict(X)
        if isinstance(predictions, pd.Series):
            return pd.DataFrame({
                'prediction': predictions,
                'uncertainty': predictions * 0.1  # 10% default uncertainty
            })
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions['prediction']
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / (y + 1e-8))) * 100
        }
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'sample_size': len(y)
        })
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if model supports it."""
        if hasattr(self.model, 'feature_importances_') and self.feature_columns is not None:
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        return None
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save complete model state with metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'performance_history': self.performance_history
        }
        
        joblib.dump(model_data, path / 'model.pkl')
        
        # Save metadata
        if self.metadata:
            with open(path / 'metadata.json', 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load complete model state."""
        path = Path(path)
        
        # Load model artifacts
        model_data = joblib.load(path / 'model.pkl')
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.performance_history = model_data.get('performance_history', [])
        
        # Load metadata
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = ModelMetadata.from_dict(metadata_dict)
    
    def update_online(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Update model with new data (online learning)."""
        # Default: retrain periodically
        # Override in specific models for true online learning
        recent_performance = self.evaluate(X, y)
        
        # Trigger retraining if performance degrades
        if len(self.performance_history) > 1:
            previous_mae = self.performance_history[-2]['metrics']['mae']
            current_mae = recent_performance['mae']
            
            if current_mae > previous_mae * 1.2:  # 20% degradation
                self._trigger_retraining_alert()
    
    def _trigger_retraining_alert(self) -> None:
        """Alert system that model needs retraining."""
        # Implementation would send alert via logging/monitoring system
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Model {self.__class__.__name__} requires retraining due to performance degradation")
    
    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data for versioning."""
        combined_data = pd.concat([X, y], axis=1)
        data_string = combined_data.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
```

### 2.2 PV Production Predictor (`models/predictors/pv_predictor.py`)

**PURPOSE**: Hybrid physical-ML model for accurate PV production forecasting with uncertainty quantification.

**TECHNICAL APPROACH**:
- **Base Model**: PVLib physical model for clear-sky baseline
- **ML Enhancement**: XGBoost for cloud/weather pattern learning
- **Ensemble**: Weighted combination based on weather conditions
- **Uncertainty**: Quantile regression + temporal uncertainty propagation
- **Features**: 50+ engineered features including solar geometry, weather, lags

**EXPECTED PERFORMANCE**: RMSE < 10% of system capacity, R² > 0.85

```python
# models/predictors/pv_predictor.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from datetime import datetime, timedelta
import pvlib
import joblib
from .base import BasePredictor

class PVPredictor(BasePredictor):
    """Advanced PV production predictor with uncertainty quantification."""
    
    def __init__(self, system_config: Dict[str, Any]):
        super().__init__(system_config)
        self.models = {
            'clear_sky': self._init_clearsky_model(),
            'ml_model': None,
            'ensemble_weights': {'clear_sky': 0.3, 'ml': 0.7}
        }
        
    def _init_clearsky_model(self) -> Dict[str, Any]:
        """Initialize physical clear-sky model using pvlib."""
        location = pvlib.location.Location(
            latitude=self.config['latitude'],
            longitude=self.config['longitude'],
            tz='Europe/Prague',
            altitude=self.config.get('altitude', 300)
        )
        
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
        
        # Train quantile models for uncertainty
        self.train_quantile_models(features_df[feature_cols], y)
        
        # Calculate validation metrics
        predictions = self.models['ml_model'].predict(X_val)
        
        metrics = {
            'mae': mean_absolute_error(y_val, predictions),
            'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
            'r2': r2_score(y_val, predictions),
            'mape': np.mean(np.abs((y_val - predictions) / (y_val + 1))) * 100
        }
        
        return metrics
    
    def predict(self, 
                weather_forecast: pd.DataFrame,
                include_uncertainty: bool = True) -> pd.DataFrame:
        """Generate PV production forecast with uncertainty bands."""
        
        # Validate weather forecast
        validation_result = self.validate_weather_forecast(weather_forecast)
        if validation_result['status'] == 'invalid':
            raise ValueError(f"Invalid weather forecast: {validation_result['errors']}")
        
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
    
    def train_quantile_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train separate quantile regression models for uncertainty."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.quantile_models = {}
        
        for quantile in [0.1, 0.9]:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
            
            X_scaled = self.scaler.transform(X[self.feature_columns])
            model.fit(X_scaled, y)
            self.quantile_models[quantile] = model
    
    def _predict_quantile(self, X: np.ndarray, quantile: float) -> np.ndarray:
        """Predict specific quantile using quantile regression."""
        if hasattr(self, 'quantile_models') and quantile in self.quantile_models:
            return self.quantile_models[quantile].predict(X)
        else:
            # Fallback to simple uncertainty estimation
            base_pred = self.models['ml_model'].predict(X)
            if quantile < 0.5:
                return base_pred * (0.5 + quantile)
            else:
                return base_pred * (0.5 + quantile)
    
    def validate_weather_forecast(self, weather_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate weather forecast data quality."""
        validation_results = {
            'status': 'valid',
            'warnings': [],
            'errors': []
        }
        
        required_columns = ['temperature', 'cloud_cover']
        missing_cols = [col for col in required_columns if col not in weather_df.columns]
        
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['status'] = 'invalid'
        
        # Check for reasonable value ranges
        if 'temperature' in weather_df.columns:
            temp_range = weather_df['temperature'].quantile([0.01, 0.99])
            if temp_range[0.01] < -20 or temp_range[0.99] > 50:
                validation_results['warnings'].append("Temperature values outside expected range (-20°C to 50°C)")
        
        if 'cloud_cover' in weather_df.columns:
            if weather_df['cloud_cover'].min() < 0 or weather_df['cloud_cover'].max() > 100:
                validation_results['errors'].append("Cloud cover must be between 0 and 100")
                validation_results['status'] = 'invalid'
        
        return validation_results
```

---

## ⚡ 3. Advanced Optimization Engine

### 3.1 Multi-Objective MILP Formulation (`optimization/optimizer.py`)

**PURPOSE**: Sophisticated energy optimization balancing cost, comfort, and self-consumption with real-world constraints.

**TECHNICAL APPROACH**:
- **Mixed-Integer Linear Programming**: Binary heating decisions + continuous power flows
- **Multi-Objective**: Weighted combination of cost, self-consumption, peak demand
- **Receding Horizon**: 48h prediction, 24h control, 1h re-optimization
- **Robust Optimization**: Uncertainty-aware decisions using prediction intervals
- **Constraint Handling**: Physical limits, comfort zones, grid limits, cycling restrictions

**SOLVER**: Gurobi (commercial) with PuLP fallback (open-source)

**EXPECTED PERFORMANCE**: Solutions within 30 seconds, <1% optimality gap

```python
# optimization/optimizer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import logging
import copy

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
    
    def optimize_with_uncertainty(self, 
                                 problem: OptimizationProblem,
                                 uncertainty_factor: float = 0.1) -> Dict[str, Any]:
        """Robust optimization considering forecast uncertainty."""
        # Create pessimistic scenarios
        scenarios = self._create_uncertainty_scenarios(problem, uncertainty_factor)
        
        # Solve for multiple scenarios and find robust solution
        solutions = []
        for scenario in scenarios:
            try:
                solution = self.optimize(scenario)
                if not solution.get('fallback_mode', False):
                    solutions.append(solution)
            except Exception as e:
                self.logger.warning(f"Scenario optimization failed: {e}")
        
        if not solutions:
            return self._get_fallback_solution(problem)
        
        # Select most robust solution (best worst-case performance)
        return self._select_robust_solution(solutions, problem)
    
    def _create_uncertainty_scenarios(self, 
                                    problem: OptimizationProblem, 
                                    uncertainty_factor: float) -> List[OptimizationProblem]:
        """Create multiple scenarios with perturbed forecasts."""
        scenarios = []
        
        # Base scenario (original)
        scenarios.append(problem)
        
        # Pessimistic PV scenario
        pessimistic_problem = copy.deepcopy(problem)
        pessimistic_problem.pv_forecast['prediction'] *= (1 - uncertainty_factor)
        scenarios.append(pessimistic_problem)
        
        # High load scenario
        high_load_problem = copy.deepcopy(problem)
        high_load_problem.load_forecast *= (1 + uncertainty_factor)
        scenarios.append(high_load_problem)
        
        # High price scenario
        high_price_problem = copy.deepcopy(problem)
        high_price_problem.prices *= (1 + uncertainty_factor)
        scenarios.append(high_price_problem)
        
        return scenarios
    
    def _get_fallback_solution(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """Generate simple rule-based fallback solution."""
        # Simple heating schedule based on time-of-use
        heating_schedule = {}
        for room in problem.rooms:
            # Heat during cheap hours (typically night)
            cheap_hours = problem.prices < problem.prices.quantile(0.3)
            schedule = pd.Series(
                cheap_hours.astype(int),
                index=problem.time_index[:len(cheap_hours)]
            )
            heating_schedule[room] = schedule
        
        # Simple battery schedule - charge when PV > load
        grid_schedule = pd.DataFrame(index=problem.time_index)
        pv_power = problem.pv_forecast['prediction']
        base_load = problem.load_forecast
        
        net_power = pv_power - base_load
        grid_schedule['export'] = np.maximum(net_power, 0)
        grid_schedule['import'] = np.maximum(-net_power, 0)
        
        return {
            'heating_schedule': heating_schedule,
            'battery_schedule': None,
            'grid_schedule': grid_schedule,
            'temperature_forecast': {},
            'costs': {'net_cost': 0, 'peak_demand': 0},
            'metrics': {'self_consumption_rate': 0.5},
            'fallback_mode': True
        }
```

---

## 🎮 4. Control System Implementation

### 4.1 Main Energy Controller (`control/energy_controller.py`)

**PURPOSE**: Central orchestrator replacing the basic Growatt controller with advanced MPC-based energy management.

**KEY FEATURES**:
- Async operation with proper resource management
- Fallback to rule-based control on ML/optimization failures
- Comprehensive state tracking and logging
- Manual override capabilities
- Safety limit enforcement

```python
# control/energy_controller.py
import asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

@dataclass
class SystemState:
    """Current system state snapshot."""
    timestamp: datetime
    battery_soc: float
    room_temperatures: Dict[str, float]
    pv_power: float
    grid_power: float
    heating_states: Dict[str, bool]
    
class EnergyController:
    """Advanced energy management controller replacing basic Growatt controller."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 models: Dict[str, Any],
                 optimizer,
                 interfaces: Dict[str, Any]):
        self.config = config
        self.models = models
        self.optimizer = optimizer
        self.interfaces = interfaces
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_state: Optional[SystemState] = None
        self.control_mode = "mpc"  # mpc, rule_based, manual, safety
        self.manual_overrides = {}
        
        # Performance tracking
        self.control_history = []
        self.mode_switches = []
        
        # Safety limits
        self.safety_limits = {
            'max_temperature': 25.0,
            'min_temperature': 16.0,
            'max_battery_power': 5000,  # W
            'max_heating_power': 20000  # W total
        }
    
    async def start(self) -> None:
        """Start the energy controller."""
        self.logger.info("Starting Advanced Energy Controller")
        
        # Initialize all components
        await self._initialize_interfaces()
        await self._load_models()
        
        # Start main control loop
        self.control_task = asyncio.create_task(self._main_control_loop())
        
        self.logger.info("Energy controller started successfully")
    
    async def _main_control_loop(self) -> None:
        """Main control loop - runs every 5 minutes."""
        while True:
            try:
                # Update system state
                await self._update_system_state()
                
                # Check safety conditions
                safety_status = self._check_safety_conditions()
                
                if not safety_status['safe']:
                    await self._enter_safety_mode(safety_status['violations'])
                    continue
                
                # Execute control logic based on current mode
                if self.control_mode == "mpc":
                    await self._execute_mpc_control()
                elif self.control_mode == "rule_based":
                    await self._execute_rule_based_control()
                elif self.control_mode == "manual":
                    await self._execute_manual_control()
                elif self.control_mode == "safety":
                    await self._execute_safety_control()
                
                # Log control actions
                await self._log_control_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                await self._enter_fallback_mode()
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'current_state': self.current_state,
            'control_mode': self.control_mode,
            'manual_overrides': self.manual_overrides,
            'recent_actions': self.control_history[-5:],
            'mode_switches': self.mode_switches[-10:],
            'safety_status': self._check_safety_conditions()
        }
```

---

## 📊 5. Performance Monitoring & Validation

### 5.1 Comprehensive Metrics Collection (`monitoring/metrics_collector.py`)

**PURPOSE**: Track all aspects of system performance for continuous improvement and validation.

```python
# monitoring/metrics_collector.py
import asyncio
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Container for system performance metrics."""
    timestamp: datetime
    
    # Prediction accuracy metrics
    pv_prediction_mae: float
    pv_prediction_rmse: float
    thermal_prediction_mae: float
    load_prediction_mae: float
    
    # Optimization metrics
    optimization_solve_time: float
    cost_savings_vs_baseline: float
    self_consumption_rate: float
    peak_reduction: float
    
    # Control system metrics
    control_loop_uptime: float
    mode_switch_frequency: float
    safety_violations: int
    
    # Energy metrics
    total_energy_cost: float
    pv_generation: float
    battery_cycles: float
    heating_efficiency: float
    
class MetricsCollector:
    """Comprehensive system performance monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.daily_metrics: List[PerformanceMetrics] = []
        self.hourly_predictions_vs_actuals = []
        self.cost_tracking = []
        
        # Baseline performance for comparison
        self.baseline_metrics = {
            'daily_cost': 150,  # CZK
            'self_consumption': 0.45,
            'peak_demand': 8000  # W
        }
    
    def calculate_cost_savings(self, 
                             actual_costs: pd.DataFrame,
                             baseline_period: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate cost savings vs baseline period."""
        # Get baseline costs
        baseline_start, baseline_end = baseline_period
        baseline_data = actual_costs[
            (actual_costs.index >= baseline_start) & 
            (actual_costs.index <= baseline_end)
        ]
        
        if len(baseline_data) == 0:
            return {'error': 'No baseline data available'}
        
        baseline_daily_avg = baseline_data['total_cost'].resample('D').sum().mean()
        
        # Get recent costs (last 30 days)
        recent_data = actual_costs[actual_costs.index >= datetime.now() - timedelta(days=30)]
        
        if len(recent_data) == 0:
            return {'error': 'No recent data available'}
        
        recent_daily_avg = recent_data['total_cost'].resample('D').sum().mean()
        
        # Calculate savings
        absolute_savings = baseline_daily_avg - recent_daily_avg
        percentage_savings = (absolute_savings / baseline_daily_avg) * 100 if baseline_daily_avg > 0 else 0
        
        return {
            'baseline_daily_cost': baseline_daily_avg,
            'recent_daily_cost': recent_daily_avg,
            'daily_savings_czk': absolute_savings,
            'percentage_savings': percentage_savings,
            'monthly_savings_czk': absolute_savings * 30,
            'annual_savings_czk': absolute_savings * 365
        }
    
    def generate_performance_report(self, 
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'report_period': {'start': start_date, 'end': end_date},
            'summary': {},
            'detailed_metrics': {},
            'recommendations': []
        }
        
        # Filter metrics for report period
        period_metrics = [
            m for m in self.daily_metrics 
            if start_date <= m.timestamp <= end_date
        ]
        
        if not period_metrics:
            report['summary'] = {'error': 'No data available for specified period'}
            return report
        
        # Calculate summary statistics
        report['summary'] = {
            'average_pv_prediction_mae': np.mean([m.pv_prediction_mae for m in period_metrics]),
            'average_thermal_prediction_mae': np.mean([m.thermal_prediction_mae for m in period_metrics]),
            'average_optimization_time': np.mean([m.optimization_solve_time for m in period_metrics]),
            'total_cost_savings': sum(m.cost_savings_vs_baseline for m in period_metrics),
            'average_self_consumption': np.mean([m.self_consumption_rate for m in period_metrics]),
            'system_uptime': np.mean([m.control_loop_uptime for m in period_metrics]),
            'total_safety_violations': sum(m.safety_violations for m in period_metrics)
        }
        
        # Generate recommendations
        avg_pv_mae = report['summary']['average_pv_prediction_mae']
        if avg_pv_mae > 500:  # 500W threshold
            report['recommendations'].append(
                f"PV prediction accuracy needs improvement (MAE: {avg_pv_mae:.0f}W). Consider retraining model."
            )
        
        avg_thermal_mae = report['summary']['average_thermal_prediction_mae']
        if avg_thermal_mae > 0.5:  # 0.5°C threshold
            report['recommendations'].append(
                f"Thermal model accuracy below target (MAE: {avg_thermal_mae:.2f}°C). Check RC parameters."
            )
        
        return report
```

---

## 🚀 6. Implementation Timeline & Phases

### Phase 2A: Foundation (Weeks 1-2)
**Target**: Establish robust ML infrastructure

#### Week 1: Core Infrastructure
- ✅ **Day 1-2**: Implement `models/base.py` with full metadata and versioning
- ✅ **Day 3-4**: Enhanced data extraction with quality scoring
- ✅ **Day 5**: Initial PV predictor with basic XGBoost

#### Week 2: Advanced Predictors  
- ✅ **Day 1-2**: Complete PV predictor with PVLib integration
- ✅ **Day 3-4**: Thermal model with RC parameter identification
- ✅ **Day 5**: Load predictor with time series features

**Success Criteria**:
- PV predictor RMSE < 15% (target: 10%)
- Thermal model MAE < 1.0°C (target: 0.5°C)
- All models deployable with versioning

### Phase 2B: Optimization Engine (Week 3)
**Target**: Production-ready optimization with MPC

#### Detailed Tasks:
- ✅ **Day 1-2**: MILP problem formulation with Gurobi
- ✅ **Day 3**: Constraint builders and safety limits
- ✅ **Day 4**: Multi-objective optimization with uncertainty
- ✅ **Day 5**: MPC controller with receding horizon

**Success Criteria**:
- Optimization solves within 30 seconds
- Handles 16 rooms × 48 hours problem size
- Robust to forecast uncertainty

### Phase 2C: Control Integration (Week 4)
**Target**: Replace Growatt controller with advanced system

#### Detailed Tasks:
- ✅ **Day 1-2**: Energy controller with mode switching
- ✅ **Day 3**: Loxone interface with MQTT commands
- ✅ **Day 4**: Safety systems and manual overrides
- ✅ **Day 5**: Integration testing with shadow mode

**Success Criteria**:
- Controller runs continuously without crashes
- Smooth fallback to rule-based control
- All safety systems functional

### Phase 2D: Validation & Deployment (Weeks 5-6)
**Target**: Proven system ready for production

#### Week 5: Testing & Validation
- ✅ **Day 1-2**: End-to-end system testing
- ✅ **Day 3**: Performance benchmarking vs Phase 1
- ✅ **Day 4**: Load testing and fault scenarios
- ✅ **Day 5**: Documentation and monitoring setup

#### Week 6: Production Deployment
- ✅ **Day 1-2**: Shadow mode deployment (parallel operation)
- ✅ **Day 3**: Gradual rollout (1 room, then 5, then all)
- ✅ **Day 4**: Performance monitoring and tuning
- ✅ **Day 5**: Final validation and handover

**Success Criteria**:
- Cost reduction > 15% (target: 20%)
- Self-consumption > 65% (target: 70%)
- System uptime > 99%
- All Phase 2 success metrics achieved

---

## 📈 Success Metrics & Validation

### Technical Performance Targets

| Metric | Phase 1 Baseline | Phase 2 Target | Measurement Method |
|--------|------------------|----------------|---------------------|
| PV Prediction RMSE | N/A | < 10% of capacity | Daily MAE vs actual production |
| Thermal Prediction MAE | N/A | < 0.5°C | Hourly temperature comparison |
| Load Prediction RMSE | N/A | < 15% | Daily consumption comparison |
| Optimization Solve Time | N/A | < 30 seconds | Average across all optimizations |
| System Uptime | 95% | > 99.5% | Controller availability monitoring |
| Cost Reduction | 0% (baseline) | > 20% | Monthly cost comparison |
| Self-Consumption Rate | 45% | > 70% | PV generation vs grid export |
| Peak Demand Reduction | 0% | > 15% | Monthly peak comparison |

### Economic Impact Targets

- **Daily Savings**: 30+ CZK/day (900+ CZK/month)
- **Annual Savings**: 10,000+ CZK/year
- **ROI Period**: < 2 years for development investment
- **Peak Demand Savings**: 1,500+ CZK/month during winter

### Validation Methodology

1. **A/B Testing**: Run new system in shadow mode parallel to existing controller
2. **Historical Backtesting**: Validate predictions against 2+ years of historical data
3. **Monte Carlo Analysis**: Test robustness across 1000+ scenario simulations
4. **Gradual Rollout**: Start with 1 room, expand to 5, then full system
5. **Continuous Monitoring**: Real-time performance tracking with automated alerts

---

## 🛡️ Risk Mitigation & Contingency Plans

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| ML models fail to achieve accuracy targets | Medium | High | Hybrid approach with physics models, ensemble methods |
| Optimization solver licensing/performance issues | Low | Medium | PuLP open-source fallback, problem size reduction |
| Integration issues with existing Loxone system | Medium | High | Comprehensive testing, gradual rollout, manual override |
| Weather forecast API failures | Medium | Medium | Multiple forecast sources, persistence forecasting |
| InfluxDB data quality issues | High | Medium | Data validation, gap filling, quality scoring |

### Fallback Strategies

1. **Rule-Based Controller**: Always maintain working rule-based system as backup
2. **Manual Override**: Web interface for immediate human intervention
3. **Safety Mode**: Minimal heating operation if all automated systems fail
4. **Gradual Rollback**: Ability to quickly revert to previous system version
5. **Expert Support**: Escalation procedures for complex issues

---

## 🔧 Development Infrastructure

### Required Tools & Libraries

```python
# Core ML/Optimization
xgboost>=1.7.0
lightgbm>=3.3.0
scikit-learn>=1.2.0
pvlib>=0.9.0
gurobipy>=10.0.0  # Commercial license required
pulp>=2.7.0      # Open source fallback
scipy>=1.9.0
filterpy>=1.4.5

# Data Processing
pandas>=1.5.0
numpy>=1.23.0
aiohttp>=3.8.0
aiofiles>=22.1.0

# Infrastructure  
aioredis>=2.0.1
aiomqtt>=1.1.0
asyncpg>=0.27.0
celery>=5.2.0

# Monitoring
prometheus-client>=0.15.0
grafana-api>=1.0.3
structlog>=22.2.0

# Testing
pytest>=7.2.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
hypothesis>=6.65.0
```

### Development Environment Setup

```bash
# Set up Phase 2 development environment
cd pems_v2
python -m venv venv_phase2
source venv_phase2/bin/activate

# Install dependencies
pip install -r requirements-phase2.txt

# Development tools
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Initialize testing database
pytest tests/ --setup-only

# Run full test suite
make test-phase2
```

### Code Quality Standards

- **Type Safety**: 100% mypy compliance with strict mode
- **Test Coverage**: Minimum 90% code coverage
- **Documentation**: Comprehensive docstrings for all public APIs
- **Linting**: Black, isort, flake8 with 100-character line length
- **Security**: Bandit security scanning
- **Performance**: Memory usage < 500MB, startup time < 30 seconds

---

## 📚 Integration with Phase 1

### Data Flow Integration

```
Phase 1 Analysis Results
    ↓
Phase 2 Feature Engineering
    ↓
ML Model Training
    ↓
Real-time Prediction
    ↓
Optimization Engine
    ↓
Control Execution
    ↓
Performance Monitoring
    ↓
Model Retraining (back to Phase 1)
```

### Reused Components

- **Data Extraction**: Enhanced version of Phase 1 extractors
- **InfluxDB Interface**: Extended with real-time capabilities
- **Configuration Management**: Unified config system
- **Logging Infrastructure**: Extended with structured logging
- **Testing Framework**: Built upon Phase 1 test patterns

### New Dependencies

- **Gurobi Optimizer**: Commercial MILP solver (requires license)
- **PVLib**: Solar physics calculations
- **FilterPy**: Kalman filtering implementation  
- **Celery**: Distributed task processing
- **Redis**: Caching and message broker
- **Prometheus**: Metrics collection
- **Grafana**: Performance dashboards

---

This comprehensive Phase 2 plan transforms the PEMS v2 system from analysis tool to production-ready intelligent energy management system. The implementation follows proven software engineering practices with robust testing, monitoring, and fallback mechanisms to ensure reliable operation in a critical home automation environment.