# TODO: Predictive Energy Management System (PEMS) Implementation

## Overview
Implement a comprehensive predictive energy management system that replaces the current Growatt controller with an ML-based optimizer for heating, battery, and EV charging control.

# TODO.md - PEMS v2 Phase 1 Completion Tasks

## Overview
This document provides detailed implementation instructions for completing Phase 1 of the PEMS v2 project. Each task includes specific requirements, code examples, and expected outputs.

---

## 1. Create Visualization Module 🎨

### Task: Implement `analysis/visualization.py`

**File Location**: `pems_v2/analysis/visualization.py`

**Implementation Requirements**:

```python
"""
Visualization utilities for PEMS v2 data analysis.
Creates interactive dashboards and static plots for analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

class AnalysisVisualizer:
    """Create visualizations for analysis results."""
    
    def __init__(self, output_dir: str = "analysis/figures"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_pv_analysis_dashboard(self, 
                                   pv_data: pd.DataFrame, 
                                   analysis_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive PV analysis dashboard.
        
        Required subplots:
        1. Daily PV production profile with confidence bands
        2. Seasonal patterns (box plots by month)
        3. Weather correlation scatter plots
        4. Performance ratio time series
        5. Battery cycling patterns
        6. Self-consumption analysis
        """
        # Implementation here...
        
    def plot_thermal_analysis(self,
                             room_name: str,
                             room_data: pd.DataFrame,
                             thermal_params: Dict[str, Any],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create thermal analysis visualization for a room.
        
        Required plots:
        1. Temperature and heating state over time
        2. Heat-up and cool-down curves with fitted models
        3. Temperature distribution histogram
        4. Thermal parameters summary table
        """
        # Implementation here...
        
    def plot_relay_patterns(self,
                           relay_data: pd.DataFrame,
                           analysis_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create relay pattern visualization.
        
        Required visualizations:
        1. Relay state timeline for each room
        2. Duty cycle comparison bar chart
        3. Energy consumption by room (stacked area)
        4. Switching frequency heatmap
        """
        # Implementation here...
        
    def plot_base_load_analysis(self,
                               base_load_data: pd.DataFrame,
                               analysis_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create base load analysis dashboard.
        
        Required plots:
        1. Daily load profiles (weekday vs weekend)
        2. Load duration curve
        3. Clustering results visualization
        4. Anomaly detection results
        """
        # Implementation here...
        
    def create_analysis_summary_report(self,
                                     all_results: Dict[str, Any],
                                     save_path: str = "analysis_summary.html") -> None:
        """
        Create comprehensive HTML report with all analysis results.
        Include navigation, interactive plots, and key findings.
        """
        # Implementation here...
```

**Expected Output**:
- Interactive Plotly dashboards saved as HTML
- Static matplotlib figures saved as PNG
- Combined HTML report with all visualizations

---

## 2. Complete Pattern Analysis Methods 📊

### Task: Implement missing methods in `analysis/pattern_analysis.py`

**Methods to implement**:

```python
def _analyze_temperature_efficiency(self, merged_data: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze PV efficiency vs temperature relationship.
    
    Steps:
    1. Filter for meaningful production periods (PV > 100W)
    2. Bin temperature data into 5°C ranges
    3. Calculate average efficiency for each bin
    4. Fit polynomial model to efficiency curve
    5. Find optimal temperature for maximum efficiency
    
    Returns:
        {
            'optimal_temp': 25.0,  # °C
            'temp_coefficient': -0.004,  # efficiency loss per °C
            'r_squared': 0.85,
            'efficiency_curve': {...}  # temp -> efficiency mapping
        }
    """
    # Implementation here...

def _detect_production_anomalies(self, pv_data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in PV production (snow, shading, faults).
    
    Methods to use:
    1. Compare actual vs clear-sky model
    2. Use isolation forest for multivariate anomaly detection
    3. Pattern matching for specific anomaly types:
       - Snow: Sudden drop to near-zero during daylight
       - Partial shading: Reduced production in specific hours
       - Inverter issues: Clipping or unusual patterns
    
    Returns DataFrame with columns:
    - timestamp
    - anomaly_type
    - severity (0-1)
    - confidence (0-1)
    """
    # Implementation here...

def _extract_seasonal_profiles(self, pv_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Extract detailed seasonal production profiles.
    
    For each season:
    1. Calculate hourly percentiles (10, 25, 50, 75, 90)
    2. Identify typical clear-sky days
    3. Calculate variability metrics
    4. Create uncertainty bands
    
    Returns:
        {
            'winter': DataFrame with hourly statistics,
            'spring': DataFrame with hourly statistics,
            'summer': DataFrame with hourly statistics,
            'autumn': DataFrame with hourly statistics
        }
    """
    # Implementation here...
```

---

## 3. Complete Thermal Analysis Methods 🌡️

### Task: Fix incomplete methods in `analysis/thermal_analysis.py`

**Methods to complete**:

```python
def _estimate_rc_parameters(self,
                           heatup_params: Dict,
                           cooldown_params: Dict,
                           room_df: pd.DataFrame,
                           weather_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate thermal resistance and capacitance.
    
    Use energy balance equation:
    C * dT/dt = (T_out - T_in)/R + P_heating - P_losses
    
    Steps:
    1. Use cooldown periods to estimate R (no heating)
    2. Use heatup periods to estimate C (known heating power)
    3. Validate with cross-validation on different periods
    4. Account for solar gains and internal gains
    
    Returns:
        {
            'R': 0.005,  # Thermal resistance [K/W]
            'C': 5e6,    # Thermal capacitance [J/K]
            'tau': 25200,  # Time constant [seconds]
            'UA': 200,   # Heat loss coefficient [W/K]
            'confidence': 0.92  # Model fit confidence
        }
    """
    # Implementation here...

def _analyze_room_coupling(self, room_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Analyze heat transfer between adjacent rooms.
    
    Steps:
    1. Calculate cross-correlation between room temperatures
    2. Identify lead/lag relationships
    3. Estimate inter-room heat transfer coefficients
    4. Build room adjacency matrix
    5. Validate with physical layout if available
    
    Returns:
        {
            'coupling_matrix': pd.DataFrame,  # Room-to-room heat transfer
            'strongest_couples': [...],  # List of strongly coupled room pairs
            'heat_flow_directions': {...},  # Dominant heat flow patterns
            'validation_metrics': {...}
        }
    """
    # Implementation here...
```

---

## 4. Create Jupyter Notebooks 📓

### Task: Implement analysis notebooks in `analysis/notebooks/`

**Required Notebooks**:

### 4.1 `01_data_exploration.ipynb`

```python
"""
# Data Exploration Notebook

## Objectives:
1. Load and inspect all data sources
2. Check data quality and completeness
3. Identify missing data patterns
4. Visualize data distributions
5. Document data issues and cleaning needs

## Sections:
1. Data Loading
2. Data Quality Assessment
3. Missing Data Analysis
4. Statistical Summaries
5. Time Series Visualization
6. Correlation Analysis
7. Recommendations
"""
```

### 4.2 `02_pv_production_analysis.ipynb`

```python
"""
# PV Production Analysis

## Objectives:
1. Analyze daily and seasonal production patterns
2. Correlate with weather conditions
3. Calculate system efficiency metrics
4. Identify anomalies and degradation
5. Create production forecasting features

## Key Visualizations:
- Daily production curves by season
- Clear sky vs actual production
- Temperature efficiency analysis
- Weather correlation heatmaps
- Anomaly detection results
"""
```

### 4.3 `03_heating_patterns.ipynb`

```python
"""
# Heating Relay Pattern Analysis

## Objectives:
1. Analyze relay switching patterns
2. Calculate room-specific duty cycles
3. Identify heating schedules
4. Optimize switching frequency
5. Calculate energy consumption by zone

## Key Analyses:
- Duty cycle statistics by room
- Switching frequency optimization
- Peak demand analysis
- Zone coordination opportunities
- Cost optimization potential
"""
```

**Each notebook should include**:
- Markdown documentation
- Interactive widgets for date selection
- Exportable visualizations
- Summary statistics tables
- Actionable insights section

---

## 5. Fix Data Preprocessing Integration 🔧

### Task: Integrate preprocessing into main pipeline

**File**: `analysis/run_analysis.py`

**Add to `_preprocess_all_data()` method**:

```python
def _preprocess_all_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced preprocessing with full integration."""
    
    processed = {}
    
    # Initialize components
    validator = DataValidator()
    outlier_detector = OutlierDetector()
    gap_filler = GapFiller()
    
    # Process each dataset
    for data_type, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # 1. Validate
            validation_result = validator.validate_data(df, data_type)
            self.logger.info(f"{data_type} validation: {validation_result}")
            
            # 2. Detect outliers
            outliers = outlier_detector.detect_statistical_outliers(df)
            df_clean = df[~outliers]
            
            # 3. Fill gaps
            df_filled = gap_filler.fill_gaps(df_clean, method='adaptive')
            
            # 4. Type-specific processing
            df_processed = self.preprocessor.process_dataset(df_filled, data_type)
            
            processed[data_type] = df_processed
            
    return processed
```

---

## 6. Fix Code Issues 🐛

### Task 6.1: Fix `main.py` imports and placeholders

```python
# Replace placeholder imports with actual implementations
from modules.predictors.pv_predictor import PVPredictor
from modules.predictors.load_predictor import LoadPredictor
from modules.predictors.thermal_model import ThermalModel
from modules.optimization.optimizer import EnergyOptimizer
from utils.async_influxdb_client import AsyncInfluxDBClient
from utils.async_mqtt_client import AsyncMQTTClient
```

### Task 6.2: Implement empty module files

**Create base classes in each module directory**:

`modules/predictors/pv_predictor.py`:
```python
"""PV production predictor using ML models."""
class PVPredictor:
    def __init__(self, config):
        self.config = config
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict PV production for given features."""
        # TODO: Implement in Phase 2
        raise NotImplementedError("PV prediction will be implemented in Phase 2")
```

### Task 6.3: Fix hardcoded values

**In `pattern_analysis.py`**, replace:
```python
# OLD
latitude=49.4949522  # Default to Prague coordinates

# NEW - Get from settings
latitude = self.settings.location.latitude if hasattr(self, 'settings') else 49.4949522
```

---

## 7. Complete Documentation 📚

### Task: Add comprehensive docstrings and create API documentation

**Required Documentation**:

1. **Method Docstrings** - Add to all incomplete methods:
```python
def method_name(self, param1: type, param2: type) -> return_type:
    """
    Brief description of what the method does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
        
    Example:
        >>> analyzer = ThermalAnalyzer()
        >>> result = analyzer.method_name(data, config)
    """
```

2. **Data Schema Documentation** - Create `docs/data_schema.md`:
```markdown
# PEMS v2 Data Schema

## InfluxDB Measurements

### Relay Data
- Measurement: `relay`
- Tags: `room`, `tag1=heating`
- Fields: `value` (0 or 1)
- Frequency: 5-minute intervals

### Temperature Data
- Measurement: `temperature`
- Tags: `room`
- Fields: `value` (°C)
- Frequency: 5-minute intervals

[Continue for all data types...]
```

3. **Feature Dictionary** - Create `docs/feature_dictionary.md`:
```markdown
# Feature Dictionary

## PV Features
| Feature Name | Description | Unit | Range |
|-------------|-------------|------|-------|
| pv_power | Instantaneous PV power | W | 0-10000 |
| sun_elevation | Solar elevation angle | degrees | 0-90 |
[Continue for all features...]
```

---

## 8. Testing Requirements 🧪

### Task: Create integration tests

**File**: `tests/test_analysis_pipeline.py`

```python
"""Integration tests for complete analysis pipeline."""

import pytest
import asyncio
from datetime import datetime, timedelta
from analysis.run_analysis import AnalysisPipeline

@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete analysis pipeline with sample data."""
    # Test implementation...
    
@pytest.mark.asyncio
async def test_visualization_generation():
    """Test that all visualizations are created correctly."""
    # Test implementation...
```

---

## Execution Order 🎯

1. **Day 1-2**: Complete visualization module
2. **Day 3-4**: Fix pattern and thermal analysis methods
3. **Day 5-7**: Create all Jupyter notebooks
4. **Day 8**: Fix preprocessing integration and code issues
5. **Day 9**: Complete documentation
6. **Day 10**: Testing and final integration

## Success Criteria ✅

- [ ] All visualization methods implemented and tested
- [ ] Pattern analysis methods complete with proper algorithms
- [ ] Thermal analysis provides accurate RC parameters
- [ ] All 7 Jupyter notebooks created and documented
- [ ] Preprocessing fully integrated into pipeline
- [ ] No placeholder code or TODO comments remain
- [ ] Documentation complete for all modules
- [ ] Integration tests pass
- [ ] Analysis pipeline runs end-to-end without errors
- [ ] HTML report generated with all visualizations

## Notes for Implementation 💡

1. Use existing data in `data/raw/` for testing
2. Refer to relay analysis implementation as a pattern
3. Ensure all timestamps are timezone-aware
4. Use logging extensively for debugging
5. Create sample outputs in `analysis/results/`
6. Test with both small (1 day) and large (2 year) datasets
7. Optimize for performance on 2-year datasets
8. Follow existing code style and patterns


## Phase 2: ML Model Development (Week 2)

### 2.1 Create Model Infrastructure

#### TODO: Implement model base classes
```python
# models/base.py
"""
Abstract base classes for all predictors:
- BasePredictor with common methods:
  - save_model()
  - load_model()
  - train()
  - predict()
  - evaluate()
  - get_feature_importance()
- Implement model versioning
- Add model performance tracking
"""
```

### 2.2 PV Production Predictor

#### TODO: Implement PV production model
```python
# modules/predictors/pv_predictor.py
"""
PV Production Predictor using ensemble approach:
1. Feature extraction from weather forecast:
   - dni, dhi, ghi (solar radiation components)
   - cloud_cover, cloud_cover_low/mid/high
   - temperature, wind_speed
   - precipitation (snow detection)
   - air_mass calculation
2. Implement models:
   - XGBoost as primary model
   - Physical model as baseline (PVLib python)
   - Ensemble with weighted average
3. Add uncertainty quantification:
   - Quantile regression (P10, P50, P90)
   - Prediction intervals
4. Real-time adaptation:
   - Online learning with recent errors
   - Bias correction based on last 7 days
"""
```

### 2.3 Thermal Models

#### TODO: Implement room thermal model
```python
# modules/predictors/thermal_model.py
"""
Room-specific thermal models:
1. State-space representation:
   - State: room temperature
   - Inputs: heating power, outdoor temp, solar gains
   - Output: predicted temperature
2. Parameter identification:
   - Thermal resistance (R)
   - Thermal capacitance (C)
   - Solar aperture coefficient
3. Implement as discrete-time model:
   T[k+1] = a*T[k] + b1*P_heat[k] + b2*T_out[k] + b3*Solar[k]
4. Kalman filter for state estimation
5. Multi-zone interactions (heat transfer between rooms)
"""
```

### 2.4 Load Predictor

#### TODO: Implement base load predictor
```python
# modules/predictors/load_predictor.py
"""
Base load prediction:
1. Features:
   - Hour of day (one-hot encoded)
   - Day of week
   - Month/season
   - Holiday indicator
   - Lagged values (t-24h, t-168h)
2. Model: LightGBM with custom objective
3. Separate models for:
   - Working days
   - Weekends
   - Holidays
4. Post-processing:
   - Ensure non-negative
   - Apply smoothing
   - Add random walk for uncertainty
"""
```

## Phase 3: Optimization Engine (Week 3)

### 3.1 Optimization Framework

#### TODO: Implement optimization core
```python
# modules/optimization/optimizer.py
"""
Multi-objective optimization using hierarchical approach:

1. Problem formulation:
   - Time horizon: 48 hours (rolling)
   - Time step: 1 hour
   - Decision variables:
     * heating[room][hour] ∈ {0,1}
     * battery_power[hour] ∈ [-5, 5] kW
     * ev_charge[hour] ∈ [0, 11] kW
     * grid_export[hour] ≥ 0 kW

2. Objectives:
   - min: Σ(grid_import[t] * price[t])
   - max: Σ(pv_self_consumed[t])
   - min: max(grid_import[t]) (peak shaving)

3. Constraints:
   - Power balance: PV + grid_import = load + battery + export
   - Battery: SOC ∈ [0.1, 0.9], power limits
   - Temperature: T_min[room] ≤ T[room][t] ≤ T_max[room]
   - EV: SOC_final ≥ target_SOC
   
4. Solution approach:
   - Use GEKKO for MINLP with APOPT solver
   - Warm start from previous solution
   - Time limit: 30 seconds
"""
```

#### TODO: Implement constraint builders
```python
# modules/optimization/constraints.py
"""
Constraint builders for optimization:

1. Power balance constraints:
   - Account for losses (battery efficiency, inverter)
   - Maximum grid import/export limits
   - Prevent simultaneous charge/discharge

2. Thermal constraints:
   - Use thermal model predictions
   - Implement as soft constraints with penalties
   - Add hysteresis to prevent oscillations

3. Battery constraints:
   - Implement degradation-aware limits
   - Power-dependent efficiency curves
   - Temperature derating

4. EV constraints:
   - Departure time hard constraint
   - Charging curve (CC-CV profile)
   - V2G capabilities (if available)
"""
```

### 3.2 Advanced Optimization Features

#### TODO: Implement scenario-based optimization
```python
# modules/optimization/stochastic_optimizer.py
"""
Stochastic optimization for uncertainty:

1. Generate scenarios:
   - PV production: P10, P50, P90
   - Price variations: ±20%
   - Base load uncertainty: ±10%

2. Implement two-stage stochastic programming:
   - First stage: heating schedule (slow dynamics)
   - Second stage: battery/EV (fast dynamics)

3. Use scenario tree with 3x3x3 = 27 scenarios
4. Minimize expected cost + CVaR for risk
"""
```

#### TODO: Implement MPC controller
```python
# modules/optimization/mpc_controller.py
"""
Model Predictive Control implementation:

1. Receding horizon: 48 hours
2. Control horizon: 24 hours
3. Re-optimize every hour
4. Disturbance handling:
   - Measure actual vs predicted
   - Update predictions with bias correction
   - Feedback to ML models

5. Implement move blocking:
   - Detailed control for next 6 hours
   - Aggregated for 6-24 hours
   - Rough plan for 24-48 hours
"""
```

## Phase 4: Control Implementation (Week 4)

### 4.1 Main Controller

#### TODO: Implement main energy controller
```python
# modules/energy_controller.py
"""
Main controller replacing growatt_controller.py:

1. Initialization:
   - Load all ML models
   - Initialize optimization engine
   - Set up state tracking
   - Configure fail-safe modes

2. Main control loop (every 5 minutes):
   - Read current state from InfluxDB
   - Check if re-optimization needed
   - Apply immediate control actions
   - Log decisions and metrics

3. Optimization triggers:
   - Every hour (scheduled)
   - Significant PV deviation (>20%)
   - Price changes
   - Manual override

4. State management:
   - Track execution vs plan
   - Maintain control history
   - Handle connection losses
"""
```

### 4.2 Control Interfaces

#### TODO: Implement Loxone control interface
```python
# modules/control/loxone_interface.py
"""
Loxone control implementation:

1. Heating control:
   - MQTT topic: loxone/heating/{room}/set
   - Payload: {"state": "on/off", "until": timestamp}
   - Implement queuing for reliability

2. Read current states:
   - Subscribe to: loxone/heating/{room}/status
   - Parse heating power consumption
   - Detect manual overrides

3. Error handling:
   - Retry failed commands
   - Verify state changes
   - Alert on communication failures
"""
```

#### TODO: Implement battery control interface
```python
# modules/control/battery_interface.py
"""
Battery control via Growatt API:

1. Control modes:
   - CHARGE_FROM_GRID: set power and duration
   - DISCHARGE_TO_GRID: set power limit
   - SELF_CONSUMPTION: default mode
   - BACKUP_RESERVE: set SOC reserve

2. Commands via MQTT:
   - Topic: growatt/battery/set_mode
   - Include schedule: [{"time": "HH:MM", "mode": "...", "power": X}]

3. Safety features:
   - Respect SOC limits
   - Temperature-based derating
   - Grid code compliance
"""
```

#### TODO: Implement EV charging interface
```python
# modules/control/ev_interface.py
"""
EV charger control:

1. Detect EV connection:
   - Monitor charger status
   - Read current SOC if available
   - Estimate from historical data

2. Implement charging strategies:
   - SOLAR_ONLY: match PV production
   - CHEAP_HOURS: use price valleys
   - ASAP: maximum rate
   - SMART: optimization result

3. OCPP or Modbus interface:
   - Set charging current (6-32A)
   - Start/stop charging
   - Read energy meter
"""
```

## Phase 5: Integration & Testing (Week 5)

### 5.1 Integration Tasks

#### TODO: Update configuration system
```python
# config/energy_settings.py
"""
Extend settings.py with energy management config:

1. Room configurations:
   - Thermal zones mapping
   - Comfort temperatures (day/night/away)
   - Heating power ratings
   - Priority levels

2. Optimization parameters:
   - Objective weights
   - Constraint penalties
   - Time horizons
   - Solver settings

3. ML model paths:
   - Model versions
   - Update frequencies
   - Fallback options

4. Control limits:
   - Max switching frequency
   - Deadbands/hysteresis
   - Emergency overrides
"""
```

#### TODO: Update main.py
```python
"""
Replace GrowattController with EnergyController:

1. Remove old growatt_controller import
2. Add new energy_controller
3. Update module initialization
4. Add new health checks
5. Implement graceful degradation
"""
```

### 5.2 Testing Strategy

#### TODO: Implement simulation environment
```python
# tests/simulation/house_simulator.py
"""
House simulation for testing:

1. Thermal simulation:
   - Use identified RC models
   - Add random disturbances
   - Simulate 24h scenarios

2. PV simulation:
   - Use historical data
   - Add forecast errors
   - Test edge cases (snow, clouds)

3. Price scenarios:
   - Historical prices
   - Extreme events
   - Negative prices

4. Validate against 1 month of real data
"""
```

#### TODO: Implement integration tests
```python
# tests/integration/test_energy_system.py
"""
End-to-end testing:

1. Test optimization convergence
2. Verify constraint satisfaction
3. Check control action execution
4. Measure computation time
5. Test failure modes:
   - Lost MQTT connection
   - Invalid predictions
   - Infeasible problems
"""
```

## Phase 6: Monitoring & Validation (Week 6)

### 6.1 Metrics & Monitoring

#### TODO: Implement performance metrics
```python
# modules/monitoring/metrics.py
"""
Track system performance:

1. Prediction accuracy:
   - RMSE for PV, load, temperatures
   - Forecast skill score
   - Bias detection

2. Optimization metrics:
   - Cost savings vs baseline
   - Self-consumption rate
   - Grid independence factor
   - Comfort violations

3. Control performance:
   - Setpoint tracking error
   - Switching frequency
   - Response time

4. Store in InfluxDB with dashboards
"""
```

#### TODO: Create Grafana dashboards
```yaml
# grafana/dashboards/energy_optimization.json
"""
Dashboards to create:

1. Overview Dashboard:
   - Current optimization plan (Gantt chart)
   - Cost savings today/week/month
   - Self-consumption gauge
   - Grid dependency graph

2. Prediction Dashboard:
   - PV forecast vs actual
   - Load prediction accuracy
   - Temperature predictions per room
   - Uncertainty bands

3. Control Dashboard:
   - Heating schedules per room
   - Battery SOC and power
   - EV charging timeline
   - Manual override indicators

4. Analytics Dashboard:
   - Historical performance
   - Model accuracy trends
   - Optimization solve times
   - Error analysis
"""
```

### 6.2 Continuous Improvement

#### TODO: Implement online learning
```python
# modules/learning/online_updater.py
"""
Continuous model improvement:

1. Collect prediction errors daily
2. Retrain models weekly:
   - Use last 3 months of data
   - Validate on last week
   - A/B test new vs old model

3. Auto-tune optimization weights:
   - Track user overrides
   - Adjust comfort boundaries
   - Learn preferences

4. Implement drift detection
"""
```

## Phase 7: Documentation & Deployment

### 7.1 Documentation

#### TODO: Create user documentation
```markdown
# docs/ENERGY_MANAGEMENT.md
"""
Document:
1. System overview and architecture
2. Configuration guide
3. Troubleshooting guide
4. Performance tuning
5. API reference
"""
```

#### TODO: Create development guide
```markdown
# docs/DEVELOPMENT.md
"""
Document:
1. Model training pipeline
2. Adding new rooms/devices
3. Optimization tuning
4. Testing procedures
5. Debugging tools
"""
```

### 7.2 Deployment

#### TODO: Create deployment scripts
```bash
# scripts/deploy_energy_system.sh
"""
1. Backup current system
2. Run integration tests
3. Deploy new code
4. Verify system health
5. Rollback procedure
"""
```

#### TODO: Implement feature flags
```python
# config/feature_flags.py
"""
Gradual rollout:
1. SIMULATION_MODE: test without control
2. ADVISORY_MODE: suggest but don't control
3. PARTIAL_CONTROL: control battery only
4. FULL_CONTROL: all systems active
"""
```

## Implementation Priority & Dependencies

### Critical Path:
1. **Week 1**: Data extraction and analysis (required for all ML models)
2. **Week 2**: PV predictor and thermal models (required for optimization)
3. **Week 3**: Basic optimization engine (core functionality)
4. **Week 4**: Control interfaces and main controller
5. **Week 5**: Integration and testing
6. **Week 6**: Monitoring and validation

### Parallel Work Streams:
- UI/Dashboard development (can start week 3)
- Documentation (ongoing)
- Advanced features (stochastic optimization, online learning) can be added later

## Success Criteria

1. **Prediction Accuracy**:
   - PV: RMSE < 10% of capacity
   - Temperature: MAE < 0.5°C
   - Load: MAPE < 15%

2. **Optimization Performance**:
   - Solve time < 30 seconds
   - Cost reduction > 20% vs baseline
   - Self-consumption > 70%

3. **System Reliability**:
   - Uptime > 99.5%
   - Failover time < 1 minute
   - No comfort violations

## Risk Mitigation

1. **Technical Risks**:
   - Optimization may not converge → Use warm starts, relaxations
   - Predictions inaccurate → Implement ensemble methods, safety margins
   - Communication failures → Queue commands, implement retries

2. **Operational Risks**:
   - User discomfort → Conservative comfort bounds, easy overrides
   - Equipment damage → Hard limits in controllers
   - Grid code violations → Implement safety checks

This TODO provides a comprehensive roadmap for implementing the predictive energy management system. Each task includes enough detail for implementation while maintaining flexibility for specific design decisions during development.