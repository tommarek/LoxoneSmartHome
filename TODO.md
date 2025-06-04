# TODO: Predictive Energy Management System (PEMS) Implementation

## Overview
Implement a comprehensive predictive energy management system that replaces the current Growatt controller with an ML-based optimizer for heating, battery, and EV charging control.

## Phase 1: Data Analysis & Feature Engineering (COMPLETED ✅)

### 1.1 Completed Tasks ✅
- ✅ **Data Extraction Module**: Complete implementation with all energy data sources
  - Solar/PV data with proper field mapping
  - Room temperature and heating status data  
  - Weather data including sun elevation calculations
  - Energy consumption breakdown by category (heating, appliances, etc.)
  - Battery charge/discharge data extraction
  - EV charging data extraction (when available)
  - Energy price data integration
- ✅ **Data Validation & Quality**: Comprehensive validation pipeline
  - Data completeness validation across all required sources
  - Quality thresholds and gap detection
  - Missing data reporting and recommendations
- ✅ **Room Configuration**: Centralized energy configuration system
  - Actual room power ratings from heating analysis (18.12 kW total)
  - Room zones and thermal characteristics
  - System parameters (battery, PV, grid limits)
- ✅ **Feature Engineering**: Complete ML feature pipeline
  - PV prediction features (weather correlations, solar geometry)
  - Thermal features for each room (temperature dynamics, heating patterns)
  - Energy management features (consumption patterns, grid interactions)
  - Feature scaling and importance calculation
- ✅ **Code Quality**: Professional development standards
  - Makefile with comprehensive linting and testing targets
  - Black code formatting and isort import organization  
  - Virtual environment setup for dependency management
  - Comprehensive test suite (Phase 1: 5/5 tests passing)

### 1.2 Remaining Quality Tasks for Phase 1
- 🔧 **Fix remaining linting issues**: 134 total issues to resolve
  - Remove 45 unused imports (F401)
  - Fix 34 long lines (E501) - wrap to 100 characters
  - Address 12 complex functions (C901) - consider refactoring
  - Add type annotations for strict mypy compliance (180 errors)
- 🔧 **Update Makefile type-check target**: Fix path to check current directory instead of non-existent pems_v2/

## Technologies & Libraries Required

### Core Dependencies
```bash
# Add to requirements.txt
scikit-learn==1.3.2          # For ML models
xgboost==2.0.3               # For gradient boosting models
lightgbm==4.1.0              # Alternative gradient boosting
tensorflow==2.15.0           # For deep learning models (optional)
cvxpy==1.4.1                 # For convex optimization
pyomo==6.7.0                 # For mixed-integer optimization
gekko==1.0.6                 # For dynamic optimization
statsmodels==0.14.1          # For time series analysis
prophet==1.1.5               # For time series forecasting
pandas==2.1.4                # Already included
numpy==1.26.2                # Already included
joblib==1.3.2                # For model persistence
optuna==3.5.0                # For hyperparameter tuning
```

### Development Dependencies
```bash
# Add to requirements-dev.txt
jupyter==1.0.0               # For data analysis notebooks
matplotlib==3.8.2            # For visualization during development
seaborn==0.13.0             # For statistical plots
plotly==5.18.0              # For interactive plots
```

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