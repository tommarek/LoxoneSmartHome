# TODO-Phase3.md: PEMS v2 Phase 3 Implementation Plan

## 🚨 **CRITICAL PHASE 2 REMAINING ISSUES (5% completion)**

### **ISSUE 1: ML Model Validation Indexing Error** ⚠️ **HIGH PRIORITY**

**Problem Description**:
The validation script fails with a pandas indexing error when trying to access ML model predictions:
```
❌ ML models validation failed: "None of [Index([12, 1, 2], dtype='int64', name='timestamp')] are in the [index]"
```

**Root Cause Analysis**:
1. **Index Mismatch**: The ML model is returning predictions with integer indices `[12, 1, 2]` instead of datetime indices
2. **DataFrame Join Issue**: The validation script expects datetime-indexed DataFrames for time series alignment
3. **Model Output Format**: The predictions DataFrame has wrong index type/format for time series operations

**Detailed Fix Required**:
```python
# File: validate_complete_system.py or models/predictors/*.py
# Current problematic code likely looks like:
predictions = model.predict(features)  # Returns int-indexed DataFrame
aligned_data = original_data.loc[predictions.index]  # FAILS - index mismatch

# SOLUTION 1: Fix in validation script
async def validate_ml_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validate ML model predictions with proper time alignment."""
    
    # Ensure consistent datetime indexing
    for model_name, model in [("pv", self.pv_predictor), ("load", self.load_predictor)]:
        # Create proper time index for predictions
        start_time = data['pv'].index.min()
        prediction_times = pd.date_range(
            start=start_time, 
            periods=len(predictions), 
            freq='5min'
        )
        
        # Reset prediction index to datetime
        predictions.index = prediction_times
        
        # Now safe to align with original data
        aligned_predictions = predictions.reindex(data['pv'].index, method='nearest')

# SOLUTION 2: Fix in model predict methods
class PVPredictor:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure predictions maintain proper datetime index."""
        predictions = self.model.predict(features.values)
        
        # Maintain original datetime index from features
        result_df = pd.DataFrame(predictions, 
                               index=features.index,  # Keep datetime index
                               columns=['pv_power_prediction'])
        return result_df
```

**Implementation Steps**:
1. **Identify exact location** of the indexing error in validation script
2. **Fix predict methods** in all ML models to maintain datetime indices
3. **Add index validation** in DataPreprocessor to ensure consistent indexing
4. **Test with various date ranges** to ensure robustness

---

### **ISSUE 2: Optimization Infeasibility Problem** ⚠️ **HIGH PRIORITY**

**Problem Description**:
Optimization solver returns `infeasible_inaccurate` status, causing optimization to fail:
```
⚠️ Optimization failed: Optimization failed: infeasible_inaccurate
```

**Root Cause Analysis**:
1. **Over-constrained Problem**: Comfort constraints may be too strict for available heating power
2. **Numerical Issues**: Constraint tolerances too tight causing numerical infeasibility
3. **Data Issues**: Invalid input data (e.g., negative prices, extreme temperatures)
4. **Solver Configuration**: ECOS_BB solver parameters may need tuning

**Detailed Fix Required**:
```python
# File: modules/optimization/optimizer.py

class EnergyOptimizer:
    def _create_optimization_problem(self, problem_data: Dict) -> cp.Problem:
        """Create optimization problem with robust constraint handling."""
        
        # SOLUTION 1: Add constraint relaxation
        # Soft constraints for comfort with penalty terms
        comfort_violations = {}
        for room in self.rooms:
            # Allow small comfort violations with high penalty
            violation_penalty = 1000  # €/°C violation
            
            comfort_violations[f'{room}_low'] = cp.Variable(self.horizon, nonneg=True)
            comfort_violations[f'{room}_high'] = cp.Variable(self.horizon, nonneg=True)
            
            # Soft comfort constraints
            constraints.extend([
                temp_vars[room][t] >= comfort_bounds[room]['min'][t] - comfort_violations[f'{room}_low'][t],
                temp_vars[room][t] <= comfort_bounds[room]['max'][t] + comfort_violations[f'{room}_high'][t]
            ])
            
            # Add violation penalties to objective
            objective += violation_penalty * cp.sum(comfort_violations[f'{room}_low'])
            objective += violation_penalty * cp.sum(comfort_violations[f'{room}_high'])
        
        # SOLUTION 2: Improve solver configuration
        def solve_with_fallback(problem: cp.Problem) -> None:
            """Solve with multiple solver configurations."""
            
            # Try primary solver (ECOS_BB) with relaxed tolerances
            try:
                problem.solve(
                    solver=cp.ECOS_BB,
                    verbose=False,
                    mi_max_iters=1000,
                    feastol=1e-6,      # Relaxed from 1e-8
                    abstol=1e-6,       # Relaxed from 1e-8
                    reltol=1e-6,       # Add relative tolerance
                    max_iters=2000     # More iterations
                )
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return
            except Exception as e:
                logger.warning(f"ECOS_BB failed: {e}")
            
            # Fallback 1: SCIP (if available)
            try:
                if cp.SCIP in cp.installed_solvers():
                    problem.solve(solver=cp.SCIP, verbose=False)
                    if problem.status == cp.OPTIMAL:
                        return
            except Exception as e:
                logger.warning(f"SCIP fallback failed: {e}")
            
            # Fallback 2: Continuous relaxation with ECOS
            logger.warning("Using continuous relaxation fallback")
            self._solve_continuous_relaxation(problem)
        
        # SOLUTION 3: Input validation and preprocessing
        def validate_problem_data(self, data: Dict) -> Dict:
            """Validate and clean optimization input data."""
            
            # Price validation
            if 'prices' in data:
                prices = data['prices']
                # Cap extreme prices
                prices = np.clip(prices, -100, 500)  # €/MWh bounds
                # Fill any NaN values
                prices = prices.fillna(method='ffill').fillna(50)  # Default 50 €/MWh
                data['prices'] = prices
            
            # Temperature validation
            if 'outdoor_temp' in data:
                temp = data['outdoor_temp']
                # Reasonable temperature bounds
                temp = np.clip(temp, -30, 50)  # °C bounds
                data['outdoor_temp'] = temp
            
            # PV forecast validation
            if 'pv_forecast' in data:
                pv = data['pv_forecast']
                # Non-negative and reasonable bounds
                pv = np.clip(pv, 0, 12000)  # Max 12kW system
                data['pv_forecast'] = pv
            
            return data

        # SOLUTION 4: Add feasibility diagnostics
        def diagnose_infeasibility(self, problem: cp.Problem) -> str:
            """Diagnose why optimization problem is infeasible."""
            
            if problem.status == cp.INFEASIBLE:
                # Check individual constraint groups
                diagnostics = []
                
                # Test power balance constraints
                if self._test_power_balance_feasibility():
                    diagnostics.append("✓ Power balance constraints feasible")
                else:
                    diagnostics.append("❌ Power balance constraints infeasible")
                
                # Test comfort constraints
                if self._test_comfort_constraints_feasibility():
                    diagnostics.append("✓ Comfort constraints feasible")
                else:
                    diagnostics.append("❌ Comfort constraints infeasible")
                
                # Test battery constraints
                if self._test_battery_constraints_feasibility():
                    diagnostics.append("✓ Battery constraints feasible")
                else:
                    diagnostics.append("❌ Battery constraints infeasible")
                
                return "\n".join(diagnostics)
            
            return "Problem status: " + problem.status
```

**Implementation Steps**:
1. **Add constraint relaxation** with penalty terms for comfort violations
2. **Improve solver configuration** with fallback options and relaxed tolerances
3. **Add input data validation** to prevent extreme values causing infeasibility
4. **Implement feasibility diagnostics** to identify which constraints are problematic
5. **Test with various scenarios** including extreme weather and price conditions

---

### **ISSUE 3: Production MQTT Integration Testing** 🔧 **MEDIUM PRIORITY**

**Problem Description**:
Control interfaces exist but need comprehensive testing with actual Loxone hardware to ensure reliable real-world operation.

**Required Implementation**:
```python
# File: tests/integration/test_loxone_integration.py

class LoxoneIntegrationTest:
    """Comprehensive Loxone hardware integration testing."""
    
    async def test_heating_control_reliability(self):
        """Test heating relay control with actual hardware."""
        
        # Test relay switching
        for room in ["kuchyne", "obyvak", "loznice"]:
            # Turn on heating
            await self.heating_controller.set_heating_relay(room, True)
            await asyncio.sleep(2)  # Allow hardware response time
            
            # Verify state change via MQTT feedback
            actual_state = await self.heating_controller.get_relay_state(room)
            assert actual_state == True, f"Heating relay {room} failed to turn on"
            
            # Turn off heating
            await self.heating_controller.set_heating_relay(room, False)
            await asyncio.sleep(2)
            
            actual_state = await self.heating_controller.get_relay_state(room)
            assert actual_state == False, f"Heating relay {room} failed to turn off"
    
    async def test_mqtt_resilience(self):
        """Test MQTT connection resilience and recovery."""
        
        # Simulate connection loss
        await self.mqtt_client.disconnect()
        
        # Attempt control operation (should queue)
        await self.heating_controller.set_heating_relay("kuchyne", True)
        
        # Reconnect and verify queued command executes
        await self.mqtt_client.reconnect()
        await asyncio.sleep(5)
        
        actual_state = await self.heating_controller.get_relay_state("kuchyne")
        assert actual_state == True, "Queued command failed to execute after reconnection"
```

---

## 🎯 **PROJECT STATUS OVERVIEW**

### ✅ **Completed Phases**
- **Phase 1**: Data Analysis & Feature Engineering [100% COMPLETE]
- **Phase 2**: ML Model Development & Optimization [95% COMPLETE - Production Ready]
  - ✅ All ML models implemented (PV, Load, Thermal)
  - ✅ Optimization engine with ECOS_BB solver
  - ✅ Control interfaces for heating, battery, inverter
  - ✅ Comprehensive validation framework
  - ❌ **2 critical bugs** preventing 100% completion (detailed above)

### 🚀 **Phase 3**: Production Deployment & System Integration

**Objective**: Transform PEMS v2 from a development system into a fully operational, production-ready energy management solution with real-time control, monitoring, and continuous optimization.

---

## **PHASE 3 SCOPE & PRIORITIES**

### **🏗️ INFRASTRUCTURE & DEPLOYMENT (Critical Path)**

#### **3.1 Complete Control System Integration**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 5-7 days

**Task**: Implement control interfaces for all currently available controllable loads
- **Controllable Systems**: Heating Relays, Temperature Setpoints, Battery Charging, Inverter Mode, Grid Export
- **Integration**: Full production MQTT/API implementation for each system
- **Testing**: Validate with actual hardware control
- **Safety**: Implement emergency shutdown and fail-safe modes
- **Note**: AC/HVAC will be added later in 2025 when hardware is available

**3.1.1 Heating Control & Temperature Setpoints**
```python
# modules/control/heating_controller.py
class HeatingController:
    """Control Loxone heating relays and temperature setpoints."""
    
    async def set_heating_relay(self, room: str, state: bool):
        """Control heating relay via MQTT."""
        topic = f"loxone/heating/{room}/set"
        payload = {"state": "on" if state else "off"}
        await self.mqtt_client.publish(topic, payload)
    
    async def set_room_temperature_setpoint(self, room: str, temperature: float):
        """Set target temperature for room thermostat."""
        topic = f"loxone/temperature/{room}/setpoint"
        payload = {
            "target_temperature": temperature,
            "unit": "celsius"
        }
        await self.mqtt_client.publish(topic, payload)
    
    async def get_room_temperature(self, room: str) -> float:
        """Get current room temperature."""
        topic = f"loxone/temperature/{room}/current"
        # Subscribe and return current temperature
```

**3.1.2 Unified Battery/Inverter Control**
```python
# modules/control/battery_inverter_controller.py
class BatteryInverterController:
    """Unified control for Growatt battery and inverter system.
    
    Note: Battery and inverter are controlled as single integrated unit.
    The inverter cannot be disconnected from grid - only operation modes can be changed.
    """
    
    async def set_system_mode(self, mode: str):
        """Set integrated battery/inverter system operation mode.
        
        Available Modes:
        - 'load_first': PV > Load > Battery > Grid (standard self-consumption)
        - 'battery_first': PV > Battery > Load > Grid (prioritize battery charging)  
        - 'time_of_use': Scheduled charging/discharging based on time periods
        - 'backup_reserve': Maintain minimum SOC for backup power
        
        Note: No 'grid_disconnect' mode - inverter always remains grid-tied
        """
        valid_modes = ['load_first', 'battery_first', 'time_of_use', 'backup_reserve']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
        
        command = {
            "action": "set_system_mode",
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_growatt_command(command)
    
    async def charge_from_grid(self, target_soc: float = 100.0, power_limit_kw: float = None):
        """Command battery to charge from grid to target SOC.
        
        Args:
            target_soc: Target state of charge (10-100%)
            power_limit_kw: Maximum charging power (optional)
        """
        # Validate SOC range
        target_soc = max(10, min(100, target_soc))
        
        command = {
            "action": "charge_from_grid",
            "target_soc": target_soc,
            "power_limit_kw": power_limit_kw or self.max_charge_power_kw,
            "priority": "grid_charging"
        }
        await self._send_growatt_command(command)
    
    async def set_discharge_schedule(self, start_hour: int, end_hour: int, 
                                   min_soc: float = 20.0):
        """Schedule battery discharge during specific hours.
        
        Args:
            start_hour: Hour to start discharging (0-23)
            end_hour: Hour to stop discharging (0-23)  
            min_soc: Minimum SOC to maintain (10-90%)
        """
        command = {
            "action": "set_discharge_schedule",
            "start_hour": start_hour,
            "end_hour": end_hour,
            "min_soc": max(10, min(90, min_soc)),
            "enabled": True
        }
        await self._send_growatt_command(command)
    
    async def set_export_control(self, enabled: bool, power_limit_kw: float = None):
        """Enable/disable grid export with optional power limit.
        
        Args:
            enabled: Whether to allow grid export
            power_limit_kw: Maximum export power (optional)
        """
        command = {
            "action": "configure_export", 
            "enabled": enabled,
            "power_limit_kw": power_limit_kw or self.max_export_power_kw
        }
        await self._send_growatt_command(command)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current battery and inverter status."""
        status = await self._query_growatt_status()
        return {
            "battery_soc": status.get("soc", 0),
            "battery_power": status.get("battery_power", 0),  # +charge, -discharge
            "inverter_mode": status.get("system_mode", "unknown"),
            "grid_export_enabled": status.get("export_enabled", False),
            "pv_power": status.get("pv_input_power", 0),
            "load_power": status.get("load_power", 0),
            "grid_power": status.get("grid_power", 0),  # +import, -export
            "system_online": status.get("online", False)
        }
    
    async def emergency_stop(self):
        """Emergency stop - set to safe backup mode."""
        await self.set_system_mode('backup_reserve')
        await self.set_export_control(enabled=False)
        self.logger.warning("Emergency stop activated - system in backup reserve mode")
```

**3.1.3 Unified Control Interface**
```python
# modules/control/unified_controller.py
class UnifiedEnergyController:
    """Unified interface for all controllable systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.heating = HeatingController(config['heating'])
        self.battery_inverter = BatteryInverterController(config['battery_inverter'])
        # Note: No separate inverter controller - integrated with battery
        
    async def execute_optimization_plan(self, plan: OptimizationResult):
        """Execute complete optimization plan across all systems."""
        # 1. Set battery/inverter system mode based on time and prices
        # 2. Schedule battery charging during cheap hours
        # 3. Manage heating relays and temperature setpoints  
        # 4. Control grid export based on price signals
        
        for action in plan.control_actions:
            if action.system == 'battery_inverter':
                if action.command == 'set_mode':
                    await self.battery_inverter.set_system_mode(action.mode)
                elif action.command == 'charge_from_grid':
                    await self.battery_inverter.charge_from_grid(
                        target_soc=action.target_soc,
                        power_limit_kw=action.power_limit
                    )
                elif action.command == 'set_export':
                    await self.battery_inverter.set_export_control(
                        enabled=action.enabled,
                        power_limit_kw=action.power_limit
                    )
            elif action.system == 'heating':
                await self._execute_heating_action(action)
```

**Deliverables**:
- ✅ Heating relay control via MQTT
- ✅ Temperature setpoint control for each room
- ✅ Unified battery/inverter control (single integrated system)
- ✅ Realistic system modes (load_first, battery_first, time_of_use, backup_reserve)
- ✅ Grid charging control with scheduling capabilities
- ✅ Grid export control (enable/disable with power limits)
- ✅ Unified control interface for optimization integration
- ✅ Emergency stop functionality (safe backup mode)
- ✅ Comprehensive system status monitoring

---

#### **3.2 Enhanced Optimization for Limited Control Variables**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 3-4 days

**Task**: Optimize control strategy for available actuators
- **File**: `modules/optimization/optimizer.py`
- **Focus**: Optimize heating, battery/inverter system modes, grid export
- **Enhancement**: Multi-stage optimization with realistic system constraints

```python
# Enhanced optimization with realistic control variables
class EnergyOptimizer:
    def _define_control_variables(self, n_steps):
        """Define control variables for available systems."""
        
        # Binary variables for heating control
        heating_vars = {room: cp.Variable(n_steps, boolean=True) 
                       for room in self.rooms}
        # Temperature setpoint variables (continuous)
        temp_setpoints = {room: cp.Variable(n_steps) 
                         for room in self.rooms}
        
        # Unified battery/inverter system mode selection (one-hot encoding)
        mode_load_first = cp.Variable(n_steps, boolean=True)      # Standard self-consumption
        mode_battery_first = cp.Variable(n_steps, boolean=True)   # Prioritize battery charging
        mode_time_of_use = cp.Variable(n_steps, boolean=True)     # Scheduled operation
        mode_backup_reserve = cp.Variable(n_steps, boolean=True)  # Emergency backup mode
        
        # Battery/inverter control variables
        battery_charge_grid = cp.Variable(n_steps, boolean=True)  # Grid charging enabled?
        battery_target_soc = cp.Variable(n_steps)                 # Target state of charge
        
        # Grid export control (always grid-tied, no disconnect option)
        grid_export_enabled = cp.Variable(n_steps, boolean=True)
        grid_export_limit = cp.Variable(n_steps)                  # Export power limit
        
        return {
            'heating': heating_vars,
            'temp_setpoints': temp_setpoints,
            'system_modes': {
                'load_first': mode_load_first,
                'battery_first': mode_battery_first, 
                'time_of_use': mode_time_of_use,
                'backup_reserve': mode_backup_reserve
            },
            'battery_charge_grid': battery_charge_grid,
            'battery_target_soc': battery_target_soc,
            'grid_export_enabled': grid_export_enabled,
            'grid_export_limit': grid_export_limit
        }
    
    def _add_control_constraints(self, variables, n_steps):
        """Add constraints specific to available controls."""
        constraints = []
        
        # System mode: exactly one mode active at each time step
        for t in range(n_steps):
            mode_sum = (variables['system_modes']['load_first'][t] +
                       variables['system_modes']['battery_first'][t] +
                       variables['system_modes']['time_of_use'][t] +
                       variables['system_modes']['backup_reserve'][t])
            constraints.append(mode_sum == 1)
        
        # Battery SOC constraints (10-100%)
        for t in range(n_steps):
            constraints.append(variables['battery_target_soc'][t] >= 10)
            constraints.append(variables['battery_target_soc'][t] <= 100)
        
        # Grid charging only during low price periods
        for t in range(n_steps):
            if self.price_forecast[t] > self.grid_charge_threshold:
                constraints.append(variables['battery_charge_grid'][t] == 0)
        
        # Grid export power limits (0-15kW max export)
        for t in range(n_steps):
            constraints.append(variables['grid_export_limit'][t] >= 0)
            constraints.append(variables['grid_export_limit'][t] <= 15)  # kW
            
            # Enable export during high price periods
            if self.price_forecast[t] > self.export_threshold:
                constraints.append(variables['grid_export_enabled'][t] == 1)
        
        # Backup reserve mode constraints - maintain minimum 50% SOC
        for t in range(n_steps):
            constraints.append(
                variables['battery_target_soc'][t] >= 
                50 * variables['system_modes']['backup_reserve'][t]
            )
        
        return constraints
```

**Optimization Strategy**:
```python
def optimize_energy_system(self):
    """Multi-stage optimization for realistic control variables."""
    
    # Stage 1: Optimize battery/inverter system mode schedule
    # - Use 'load_first' for standard self-consumption during normal times
    # - Use 'battery_first' during expensive hours to prioritize battery charging from PV
    # - Use 'time_of_use' for scheduled grid charging during cheapest hours
    # - Use 'backup_reserve' during grid instability or emergency conditions
    
    # Stage 2: Optimize grid charging schedule
    # - Identify cheapest 4-hour windows for grid charging
    # - Schedule charging to reach 100% SOC before expensive periods
    # - Respect battery SOC constraints (10-100%)
    
    # Stage 3: Optimize heating operation
    # - Pre-heat during cheap hours using stored battery energy
    # - Adjust temperature setpoints based on electricity prices
    # - Maintain comfort constraints (±2°C from target)
    
    # Stage 4: Grid export optimization
    # - Enable export during high feed-in tariffs (always grid-tied)
    # - Set appropriate export power limits based on local consumption
    # - Disable export during negative prices to avoid costs
```

**Deliverables**:
- ✅ Optimized control for heating relays and temperature setpoints
- ✅ Multi-stage optimization with realistic system mode selection
- ✅ Price-aware battery charging from grid with SOC targeting
- ✅ Unified battery/inverter system mode optimization
- ✅ Grid export optimization with power limit control (always grid-tied)
- ✅ Integrated control scheduling across all available systems

---

#### **3.3 Model Persistence & Versioning System**
**Priority**: 🟡 **HIGH** | **Estimated**: 2-3 days

**Task**: Complete production model management system
- **Files**: `models/base.py`, `models/predictors/*.py`
- **Current Status**: Framework exists, needs production integration
- **Enhancement**: Add automatic model updates and A/B testing

```python
# Enhanced model persistence system
class ModelRegistry:
    def auto_update_models(self):
        """Automatically update models based on recent performance."""
        for model_name in self.models:
            performance = self.get_recent_performance(model_name)
            if performance.degradation > 0.15:  # 15% performance drop
                self.trigger_retraining(model_name)
    
    def deploy_with_ab_testing(self, model_name: str, new_version: str):
        """Deploy new model with gradual rollout."""
        # Implementation here...
```

**Deliverables**:
- ✅ Automatic model retraining triggers
- ✅ A/B testing framework for model updates
- ✅ Performance monitoring and degradation detection
- ✅ Rollback capabilities for failed deployments

---

#### **3.4 Control Strategy Implementation Examples**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 2-3 days

**Task**: Implement specific control strategies for available actuators
- **Purpose**: Define clear control logic for optimization results
- **Integration**: Bridge between optimization decisions and hardware control

**Example Control Strategies**:

```python
# modules/control/control_strategies.py
class ControlStrategyExecutor:
    """Execute optimized control strategies for all available systems."""
    
    async def execute_daily_battery_inverter_strategy(self, price_forecast: pd.Series):
        """Execute unified battery/inverter system control based on price signals."""
        
        # Find cheapest 4-hour window for grid charging
        rolling_avg = price_forecast.rolling(window=4).mean()
        cheapest_start = rolling_avg.idxmin()
        
        # Schedule unified system operation
        system_schedule = []
        for hour in range(24):
            if cheapest_start <= hour < cheapest_start + 4:
                # Grid charging during cheapest hours
                system_schedule.append({
                    'hour': hour,
                    'action': 'set_system_mode',
                    'mode': 'time_of_use',  # Scheduled operation mode
                    'enable_grid_charge': True,
                    'target_soc': 100.0,
                    'reason': 'cheap_grid_charging'
                })
            elif price_forecast[hour] > self.high_price_threshold:
                # Prioritize battery usage during expensive hours
                system_schedule.append({
                    'hour': hour,
                    'action': 'set_system_mode',
                    'mode': 'battery_first',  # Battery priority during expensive hours
                    'enable_grid_charge': False,
                    'reason': 'expensive_grid_prices'
                })
            else:
                # Normal self-consumption mode
                system_schedule.append({
                    'hour': hour,
                    'action': 'set_system_mode', 
                    'mode': 'load_first',  # Standard self-consumption
                    'enable_grid_charge': False,
                    'reason': 'normal_operation'
                })
        
        return system_schedule
    
    async def execute_heating_optimization(self, price_forecast: pd.Series, 
                                          weather_forecast: pd.DataFrame):
        """Optimize heating and temperature setpoints based on prices."""
        
        heating_schedule = []
        for room in self.rooms:
            # Pre-heat during cheap morning hours (e.g., 4-6 AM)
            if price_forecast[4:6].mean() < self.cheap_price_threshold:
                heating_schedule.append({
                    'room': room,
                    'time': '04:00',
                    'action': 'preheat',
                    'target_temp': 22.0,
                    'duration_hours': 2
                })
            
            # Lower setpoint during expensive hours
            expensive_hours = price_forecast[price_forecast > self.high_price_threshold].index
            for hour in expensive_hours:
                heating_schedule.append({
                    'room': room,
                    'time': f'{hour}:00',
                    'action': 'reduce_setpoint',
                    'target_temp': 19.0,  # Lower but still comfortable
                    'duration_hours': 1
                })
        
        return heating_schedule
    
    async def execute_grid_export_strategy(self, price_forecast: pd.Series,
                                         pv_forecast: pd.Series):
        """Control grid export based on economics (always grid-tied system)."""
        
        export_schedule = []
        for hour in range(24):
            # Enable export if feed-in price is good
            if price_forecast[hour] > self.export_price_threshold:
                export_schedule.append({
                    'hour': hour,
                    'export_enabled': True,
                    'power_limit_kw': 15.0,  # Full system capacity
                    'reason': 'high_feed_in_tariff'
                })
            # Disable export during negative prices
            elif price_forecast[hour] < 0:
                export_schedule.append({
                    'hour': hour,
                    'export_enabled': False,
                    'power_limit_kw': 0.0,
                    'reason': 'negative_prices'
                })
            # Enable limited export if excess PV and reasonable price
            elif pv_forecast[hour] > self.base_load_forecast[hour] + 2000:
                excess_power = pv_forecast[hour] - self.base_load_forecast[hour]
                export_schedule.append({
                    'hour': hour,
                    'export_enabled': True,
                    'power_limit_kw': min(excess_power / 1000, 15.0),  # Limit to excess
                    'reason': 'excess_pv_production'
                })
            else:
                export_schedule.append({
                    'hour': hour,
                    'export_enabled': False,
                    'power_limit_kw': 0.0,
                    'reason': 'maximize_self_consumption'
                })
        
        return export_schedule
```

**Integrated Daily Control Flow**:
```python
async def execute_daily_optimization(self):
    """Complete daily optimization and control execution."""
    
    # 1. Get forecasts
    price_forecast = await self.get_price_forecast()
    pv_forecast = await self.get_pv_forecast()
    weather_forecast = await self.get_weather_forecast()
    
    # 2. Run optimization
    optimization_result = await self.optimizer.optimize(
        price_forecast, pv_forecast, weather_forecast
    )
    
    # 3. Execute control strategies
    # Unified battery/inverter system control
    system_strategy = await self.execute_daily_battery_inverter_strategy(price_forecast)
    for action in system_strategy:
        await self.schedule_battery_inverter_action(action)
    
    # Heating optimization
    heating_strategy = await self.execute_heating_optimization(
        price_forecast, weather_forecast
    )
    for action in heating_strategy:
        await self.schedule_heating_action(action)
    
    # Grid export control
    export_strategy = await self.execute_grid_export_strategy(
        price_forecast, pv_forecast
    )
    for action in export_strategy:
        await self.schedule_export_action(action)
    
    # 4. Monitor execution and adjust as needed
    await self.monitor_and_adjust()
```

**Deliverables**:
- ✅ Unified battery/inverter system control strategies 
- ✅ Realistic system mode scheduling (load_first, battery_first, time_of_use, backup_reserve)
- ✅ Heating pre-conditioning and setpoint optimization
- ✅ Dynamic grid export control with power limits (always grid-tied)
- ✅ Integrated control flow for daily optimization
- ✅ Real-time adjustment capabilities
- ✅ Clear strategy documentation and examples

---

### **🌞 DUAL-ARRAY PHOTOVOLTAIC SYSTEM ENHANCEMENT (High Priority)**

#### **3.4 Dual-Array PV Configuration Implementation**
**Priority**: 🟡 **HIGH** | **Estimated**: 3-4 days

**Task**: Update PV prediction and analysis modules to properly handle dual-array configuration with different orientations
- **Current Limitation**: Single-array modeling with default 180° azimuth and 30° tilt
- **New Configuration**: Two arrays with 143° SE and 233° SW azimuth, both at 35° tilt
- **Impact**: Improved prediction accuracy, better string balance analysis, orientation-aware optimization

**Physical Array Configuration**:
```yaml
Array Configuration:
  Array 1 (PV1 String):
    azimuth: 143°  # Southeast facing
    tilt: 35°      # Roof slope
    capacity: ~7.5 kW  # Estimated based on total 15kW system
    
  Array 2 (PV2 String):  
    azimuth: 233°  # Southwest facing  
    tilt: 35°      # Roof slope
    capacity: ~7.5 kW  # Estimated based on total 15kW system
    
  Benefits:
    - Extended daily production window (SE peaks morning, SW peaks afternoon)
    - Better performance in variable weather conditions
    - Reduced peak power stress on single inverter input
    - More consistent daily energy yield
```

**3.4.1 PV Predictor Model Enhancement**
**File**: `pems_v2/models/predictors/pv_predictor.py`

```python
# Enhanced dual-array PV prediction model
class DualArrayPVPredictor(BasePredictor):
    """Enhanced PV predictor supporting dual-array configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Dual-array physical configuration
        self.array_configs = {
            'pv1': {
                'azimuth': 143.0,    # Southeast facing
                'tilt': 35.0,        # Roof slope
                'capacity_kw': 7.5,  # Half of total system
                'string_id': 'PV1'   # Maps to InfluxDB field
            },
            'pv2': {
                'azimuth': 233.0,    # Southwest facing  
                'tilt': 35.0,        # Roof slope
                'capacity_kw': 7.5,  # Half of total system
                'string_id': 'PV2'   # Maps to InfluxDB field
            }
        }
        
        # System location (Prague)
        self.latitude = 49.2
        self.longitude = 16.6
        
    def _calculate_array_specific_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate PVLib features for each array separately."""
        import pvlib
        
        features = weather_data.copy()
        
        # Create PVLib location
        location = pvlib.location.Location(
            latitude=self.latitude,
            longitude=self.longitude,
            tz='Europe/Prague'
        )
        
        # Calculate solar position
        solar_position = location.get_solarposition(weather_data.index)
        
        # Calculate array-specific irradiance for each orientation
        for array_id, config in self.array_configs.items():
            # Calculate plane-of-array irradiance for this specific orientation
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=config['tilt'],
                surface_azimuth=config['azimuth'], 
                solar_zenith=solar_position['zenith'],
                solar_azimuth=solar_position['azimuth'],
                dni=weather_data.get('dni', weather_data.get('shortwave_radiation', 0)),
                ghi=weather_data.get('ghi', weather_data.get('shortwave_radiation', 0)),
                dhi=weather_data.get('dhi', weather_data.get('shortwave_radiation', 0) * 0.1)
            )
            
            # Add array-specific features
            features[f'{array_id}_poa_irradiance'] = poa_irradiance['poa_global']
            features[f'{array_id}_incidence_angle'] = pvlib.irradiance.aoi(
                surface_tilt=config['tilt'],
                surface_azimuth=config['azimuth'],
                solar_zenith=solar_position['zenith'], 
                solar_azimuth=solar_position['azimuth']
            )
            
            # Calculate array-specific cell temperature
            cell_temp = pvlib.temperature.faiman(
                poa_global=poa_irradiance['poa_global'],
                temp_air=weather_data.get('temperature_2m', 20),
                wind_speed=weather_data.get('windspeed_10m', 0)
            )
            features[f'{array_id}_cell_temperature'] = cell_temp
            
            # Calculate theoretical DC power for this array
            features[f'{array_id}_theoretical_dc'] = self._calculate_dc_power(
                poa_irradiance['poa_global'], 
                cell_temp, 
                config['capacity_kw']
            )
        
        # Calculate combined system features
        features['total_theoretical_dc'] = (
            features['pv1_theoretical_dc'] + features['pv2_theoretical_dc']
        )
        
        # Array balance feature (how much each array contributes)
        total_irradiance = features['pv1_poa_irradiance'] + features['pv2_poa_irradiance']
        features['array_balance'] = np.where(
            total_irradiance > 0,
            features['pv1_poa_irradiance'] / total_irradiance,
            0.5  # Default balanced when no sun
        )
        
        return features
        
    def _calculate_dc_power(self, poa_irradiance: pd.Series, cell_temp: pd.Series, 
                           capacity_kw: float) -> pd.Series:
        """Calculate DC power output for array using PVLib."""
        # Standard Test Conditions (STC) parameters
        temp_coeff = -0.004  # %/°C typical for crystalline silicon
        stc_temp = 25.0      # °C
        stc_irradiance = 1000 # W/m²
        
        # Temperature derating
        temp_derating = 1 + temp_coeff * (cell_temp - stc_temp)
        
        # Irradiance scaling
        irradiance_factor = poa_irradiance / stc_irradiance
        
        # DC power calculation with efficiency losses
        dc_power = capacity_kw * irradiance_factor * temp_derating
        
        # Apply realistic system losses
        system_losses = 0.85  # 15% losses (soiling, wiring, inverter, etc.)
        
        return np.maximum(0, dc_power * system_losses)
    
    def predict(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Enhanced prediction with array-specific outputs."""
        
        # Calculate array-specific PVLib features
        enhanced_features = self._calculate_array_specific_features(features)
        
        # Get base predictions using ML model
        base_predictions = super().predict(enhanced_features)
        
        # Add array-specific predictions
        predictions = base_predictions.copy()
        
        # Predict individual array outputs if historical data supports it
        for array_id in ['pv1', 'pv2']:
            array_features = self._extract_array_features(enhanced_features, array_id)
            predictions[f'{array_id}_power'] = self._predict_array_power(array_features)
        
        # Validate that sum of arrays matches total prediction
        predicted_total = predictions['pv1_power'] + predictions['pv2_power']
        # Apply scaling factor to maintain consistency
        if predicted_total.sum() > 0:
            scale_factor = base_predictions['power'].sum() / predicted_total.sum()
            predictions['pv1_power'] *= scale_factor
            predictions['pv2_power'] *= scale_factor
        
        return predictions
```

**3.4.2 Configuration Enhancement**
**File**: `pems_v2/config/energy_settings.py`

```python
# Enhanced PV array configuration
PV_ARRAY_CONFIGURATION = {
    'total_capacity_kw': 15.0,
    'arrays': {
        'pv1': {
            'name': 'Southeast Array',
            'azimuth': 143.0,           # Degrees from North (Southeast)
            'tilt': 35.0,               # Degrees from horizontal  
            'capacity_kw': 7.5,         # Nominal DC capacity
            'string_id': 'PV1',         # InfluxDB field mapping
            'technology': 'crystalline_silicon',
            'mounting': 'roof_mounted',
            'orientation_description': 'Morning optimized - peaks 9-11 AM'
        },
        'pv2': {
            'name': 'Southwest Array', 
            'azimuth': 233.0,           # Degrees from North (Southwest)
            'tilt': 35.0,               # Degrees from horizontal
            'capacity_kw': 7.5,         # Nominal DC capacity  
            'string_id': 'PV2',         # InfluxDB field mapping
            'technology': 'crystalline_silicon',
            'mounting': 'roof_mounted',
            'orientation_description': 'Afternoon optimized - peaks 1-3 PM'
        }
    },
    'system_location': {
        'latitude': 49.2,            # Prague coordinates
        'longitude': 16.6,
        'timezone': 'Europe/Prague',
        'elevation_m': 200           # Approximate elevation
    },
    'performance_parameters': {
        'system_losses': 0.15,       # 15% total system losses
        'inverter_efficiency': 0.96, # 96% inverter efficiency
        'dc_ac_ratio': 1.2,          # DC to AC sizing ratio
        'temperature_coefficient': -0.004  # %/°C power loss
    }
}

def get_array_configuration(array_id: str) -> Dict[str, Any]:
    """Get configuration for specific PV array."""
    if array_id not in PV_ARRAY_CONFIGURATION['arrays']:
        raise ValueError(f"Unknown array ID: {array_id}")
    return PV_ARRAY_CONFIGURATION['arrays'][array_id]

def get_optimal_production_windows() -> Dict[str, Dict[str, int]]:
    """Get expected optimal production windows for each array."""
    return {
        'pv1': {  # Southeast array - morning peak
            'peak_start_hour': 9,
            'peak_end_hour': 11, 
            'production_start_hour': 6,
            'production_end_hour': 15
        },
        'pv2': {  # Southwest array - afternoon peak  
            'peak_start_hour': 13,
            'peak_end_hour': 15,
            'production_start_hour': 10, 
            'production_end_hour': 19
        }
    }
```

**3.4.3 Analysis Module Enhancement**
**File**: `pems_v2/analysis/analyzers/pattern_analysis.py`

```python
def analyze_dual_array_performance(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance patterns for dual-array system."""
    
    analysis = {
        'array_performance': {},
        'system_balance': {},
        'optimization_opportunities': {}
    }
    
    # Individual array analysis
    for array_id in ['pv1', 'pv2']:
        power_field = f'{array_id.upper()}InputPower'
        if power_field in pv_data.columns:
            array_data = pv_data[power_field]
            config = get_array_configuration(array_id)
            
            analysis['array_performance'][array_id] = {
                'daily_peak_hour': array_data.groupby(array_data.index.hour).mean().idxmax(),
                'capacity_factor': array_data.mean() / (config['capacity_kw'] * 1000),
                'production_hours_per_day': (array_data > 100).groupby(array_data.index.date).sum().mean(),
                'weather_sensitivity': self._calculate_weather_correlation(array_data),
                'expected_peak_hours': get_optimal_production_windows()[array_id],
                'performance_vs_expected': self._compare_to_theoretical(array_data, array_id)
            }
    
    # System balance analysis
    if 'PV1InputPower' in pv_data.columns and 'PV2InputPower' in pv_data.columns:
        pv1_power = pv_data['PV1InputPower']
        pv2_power = pv_data['PV2InputPower'] 
        total_power = pv1_power + pv2_power
        
        # Calculate balance ratio (ideally should vary throughout day)
        balance_ratio = np.where(total_power > 100, pv1_power / total_power, np.nan)
        
        analysis['system_balance'] = {
            'daily_balance_pattern': pd.Series(balance_ratio, index=pv_data.index).groupby(
                pv_data.index.hour
            ).mean().to_dict(),
            'morning_dominance': balance_ratio[pv_data.index.hour.isin([8, 9, 10])].mean(),
            'afternoon_dominance': 1 - balance_ratio[pv_data.index.hour.isin([14, 15, 16])].mean(),
            'ideal_morning_ratio': 0.6,  # SE array should dominate morning
            'ideal_afternoon_ratio': 0.4, # SW array should dominate afternoon
            'balance_health_score': self._calculate_balance_health_score(balance_ratio)
        }
    
    return analysis

def _calculate_balance_health_score(self, balance_ratio: np.ndarray) -> float:
    """Calculate health score for array balance (0-1, higher is better)."""
    # Good balance means arrays complement each other throughout day
    # Morning hours (6-12): PV1 should dominate (ratio > 0.5)
    # Afternoon hours (12-18): PV2 should dominate (ratio < 0.5) 
    
    morning_mask = pd.Series(balance_ratio).index.hour < 12
    afternoon_mask = pd.Series(balance_ratio).index.hour >= 12
    
    morning_score = (balance_ratio[morning_mask] > 0.5).mean() if morning_mask.any() else 0
    afternoon_score = (balance_ratio[afternoon_mask] < 0.5).mean() if afternoon_mask.any() else 0
    
    return (morning_score + afternoon_score) / 2
```

**3.4.4 Database Schema Enhancement**
**File**: `pems_v2/analysis/db_data_docs/SOLAR_BUCKET.md`

```markdown
# Enhanced Solar Data Schema - Dual Array Support

## Array Configuration Metadata

### Array Specifications
- **PV1 (Southeast Array)**: 143° azimuth, 35° tilt, ~7.5kW capacity
- **PV2 (Southwest Array)**: 233° azimuth, 35° tilt, ~7.5kW capacity  
- **Total System**: 15kW DC capacity, roof-mounted crystalline silicon

## String-Specific Analysis Fields

### Individual Array Performance
- `PV1InputPower`: DC power from Southeast array (W)
- `PV2InputPower`: DC power from Southwest array (W) 
- `PV1Voltage`: String voltage for Southeast array (V)
- `PV2Voltage`: String voltage for Southwest array (V)

### Calculated Dual-Array Metrics
- `array_balance_ratio`: PV1/(PV1+PV2) production ratio
- `morning_production_dominance`: SE array advantage in AM hours
- `afternoon_production_dominance`: SW array advantage in PM hours
- `daily_production_window_hours`: Total productive hours across both arrays
- `complementary_production_score`: How well arrays complement each other
```

**3.4.5 Testing Enhancement**
**File**: `pems_v2/tests/test_pv_predictor.py`

```python
def test_dual_array_prediction_accuracy(self):
    """Test prediction accuracy for dual-array configuration."""
    # Create test data with realistic dual-array patterns
    test_data = self._create_dual_array_test_data()
    
    predictor = DualArrayPVPredictor(self.config)
    predictions = predictor.predict(test_data)
    
    # Verify array-specific predictions
    assert 'pv1_power' in predictions
    assert 'pv2_power' in predictions
    
    # Check that PV1 (SE) peaks in morning
    morning_hours = [9, 10, 11]
    pv1_morning_avg = predictions['pv1_power'][morning_hours].mean()
    pv2_morning_avg = predictions['pv2_power'][morning_hours].mean()
    assert pv1_morning_avg > pv2_morning_avg
    
    # Check that PV2 (SW) peaks in afternoon  
    afternoon_hours = [13, 14, 15]
    pv1_afternoon_avg = predictions['pv1_power'][afternoon_hours].mean()
    pv2_afternoon_avg = predictions['pv2_power'][afternoon_hours].mean()
    assert pv2_afternoon_avg > pv1_afternoon_avg

def test_array_configuration_validation(self):
    """Test that array configurations are properly validated."""
    config = get_array_configuration('pv1')
    assert config['azimuth'] == 143.0
    assert config['tilt'] == 35.0
    
    config = get_array_configuration('pv2') 
    assert config['azimuth'] == 233.0
    assert config['tilt'] == 35.0
```

**Implementation Priority & Dependencies**:
```yaml
Phase 3.4 Implementation Order:
  1. Update energy_settings.py with dual-array configuration (1 day)
  2. Enhance PV predictor model with PVLib dual-array support (2 days) 
  3. Update pattern analysis for array-specific performance (1 day)
  4. Add comprehensive testing for dual-array scenarios (0.5 days)
  5. Update documentation and database schema (0.5 days)

Dependencies:
  - pvlib-python library for solar calculations
  - Enhanced weather data (DNI, GHI, DHI) for accurate irradiance modeling
  - Historical PV1/PV2 string data for model training and validation

Benefits:
  - Improved prediction accuracy (estimated 15-20% RMSE reduction)
  - Better understanding of production patterns throughout day
  - Array-specific performance monitoring and fault detection
  - Optimization opportunities for time-of-use scenarios
```

**Deliverables**:
- ✅ Dual-array PV configuration system with 143°/233° azimuth support
- ✅ Enhanced PVLib-based modeling for realistic irradiance calculations  
- ✅ Array-specific performance analysis and monitoring
- ✅ Improved prediction accuracy for both morning and afternoon production
- ✅ String balance analysis considering different optimal production times
- ✅ Updated database schema and documentation for dual-array system

---

### **📊 MONITORING & OBSERVABILITY (Essential)**

#### **3.5 Real-time Monitoring Dashboard**
**Priority**: 🟡 **HIGH** | **Estimated**: 4-5 days

**Task**: Create comprehensive Grafana dashboards for operational monitoring
- **Location**: `grafana/dashboards/`
- **Integration**: InfluxDB metrics storage
- **Features**: Real-time control status, performance tracking, alerts

**Dashboard Structure**:
```yaml
# grafana/dashboards/pems_v2_overview.json
1. System Overview:
   - Current control states (Heating, AC, Battery mode, Export status)
   - Real-time cost savings (today/week/month)
   - Self-consumption rate gauge
   - Grid import/export power flows
   - Current electricity price indicator
   
2. Battery & Inverter Control:
   - Battery SOC gauge with charging status
   - Current inverter mode (load/battery/grid first)
   - Grid charging schedule timeline
   - Power flow diagram (PV -> Load/Battery/Grid)
   - Daily charge/discharge cycles chart

3. Heating Control:
   - Room heating relay states (ON/OFF grid)
   - Current vs target temperature setpoints by room
   - Pre-heating schedule timeline
   - Energy consumption by heating system
   - Temperature setpoint adjustments based on price
   - Comfort violation alerts

4. Grid Export Control:
   - Export enabled/disabled status
   - Current feed-in price vs threshold
   - Daily export revenue tracking
   - Export decision reasoning log

5. System Performance:
   - Control execution success rate
   - Strategy performance metrics
   - Price arbitrage opportunities captured
   - System response times
```

**Deliverables**:
- ✅ 4 comprehensive Grafana dashboards
- ✅ Real-time performance metrics collection
- ✅ Automated alerting for system issues
- ✅ Historical performance trend analysis

---

#### **3.6 Performance Metrics & Analytics**
**Priority**: 🟡 **HIGH** | **Estimated**: 3-4 days

**Task**: Implement comprehensive performance tracking for all control systems
- **File**: `modules/monitoring/metrics.py` (new)
- **Integration**: Store metrics in InfluxDB for dashboard consumption
- **Features**: Control-specific metrics, cost savings tracking, performance analysis

```python
# modules/monitoring/metrics.py
class PerformanceTracker:
    """Track and analyze PEMS v2 system performance."""
    
    def track_control_performance(self, control_type: str, action: Dict, result: bool):
        """Track individual control action performance."""
        metrics = {
            'control_type': control_type,  # 'heating', 'ac', 'battery', 'inverter', 'export'
            'action': action['action'],
            'success': result,
            'timestamp': datetime.now(),
            'response_time_ms': action.get('response_time', 0)
        }
        self._store_metrics('control_performance', metrics)
    
    def track_battery_performance(self, battery_state: Dict):
        """Track battery control effectiveness."""
        metrics = {
            'soc': battery_state['soc'],
            'power': battery_state['power'],
            'mode': battery_state['mode'],
            'grid_charge_active': battery_state.get('grid_charging', False),
            'energy_from_grid_kwh': battery_state.get('grid_energy', 0),
            'cost_of_grid_charge': battery_state.get('grid_cost', 0)
        }
        self._store_metrics('battery_performance', metrics)
    
    def track_cost_savings(self, baseline_cost: float, optimized_cost: float, 
                          control_actions: List[Dict]):
        """Track cost savings from optimization."""
        savings = baseline_cost - optimized_cost
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        metrics = {
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'savings': savings,
            'savings_percent': savings_percent,
            'battery_arbitrage': self._calculate_battery_arbitrage(control_actions),
            'heating_shifting_savings': self._calculate_heating_savings(control_actions),
            'export_revenue': self._calculate_export_revenue(control_actions)
        }
        self._store_metrics('cost_savings', metrics)
    
    def track_strategy_effectiveness(self, strategy_type: str, expected_savings: float,
                                   actual_savings: float):
        """Track how well control strategies perform vs expectations."""
        effectiveness = (actual_savings / expected_savings) * 100 if expected_savings > 0 else 0
        
        metrics = {
            'strategy': strategy_type,
            'expected_savings': expected_savings,
            'actual_savings': actual_savings,
            'effectiveness_percent': effectiveness,
            'timestamp': datetime.now()
        }
        self._store_metrics('strategy_effectiveness', metrics)
```

**Deliverables**:
- ✅ Automated performance tracking for all system components
- ✅ Daily/weekly performance reports
- ✅ Trend analysis and degradation detection
- ✅ Cost savings tracking and ROI calculation

---

### **🛡️ PRODUCTION READINESS & RELIABILITY (Critical)**

#### **3.6 Production Configuration & Security**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 2-3 days

**Task**: Enhance configuration system for production deployment
- **Files**: `config/settings.py`, `config/energy_settings.py`
- **Security**: Secure credential management, environment separation
- **Reliability**: Configuration validation, fallback values

```python
# Enhanced production configuration
class ProductionSettings(PEMSSettings):
    """Production-specific settings with enhanced security and validation."""
    
    # Security enhancements
    encryption_key: SecretStr = Field(..., description="Data encryption key")
    api_rate_limits: Dict[str, int] = Field(default_factory=dict)
    allowed_hosts: List[str] = Field(default_factory=list)
    
    # Production reliability
    max_memory_usage_mb: int = Field(default=2048)
    max_cpu_usage_percent: int = Field(default=80)
    health_check_interval: int = Field(default=60)
    
    # Environment separation
    deployment_environment: str = Field(default="production")
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")
```

**Deliverables**:
- ✅ Secure credential management system
- ✅ Environment-specific configurations (dev/staging/prod)
- ✅ Configuration validation and error handling
- ✅ Resource usage monitoring and limits

---

#### **3.7 Error Handling & Recovery Systems**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 3-4 days

**Task**: Implement comprehensive error handling and recovery mechanisms
- **Scope**: All system components
- **Features**: Graceful degradation, automatic recovery, manual fallbacks
- **Testing**: Failure mode testing and validation

```python
# modules/reliability/error_handler.py
class SystemErrorHandler:
    """Centralized error handling and recovery system."""
    
    def handle_optimization_failure(self, error: Exception, context: Dict):
        """Handle optimization engine failures with graceful degradation."""
        if isinstance(error, InfeasibleProblemError):
            # Relax constraints and retry
            return self._retry_with_relaxed_constraints(context)
        elif isinstance(error, SolverTimeoutError):
            # Use last successful solution
            return self._use_cached_solution(context)
        else:
            # Fall back to rule-based control
            return self._emergency_fallback_control(context)
    
    def handle_mqtt_disconnection(self, controller: HeatingController):
        """Handle MQTT connection failures."""
        # Implement reconnection logic with exponential backoff
        # Store commands in queue for replay when reconnected
        # Alert operators about communication issues
```

**Deliverables**:
- ✅ Comprehensive error handling for all system components
- ✅ Automatic recovery mechanisms with intelligent fallbacks
- ✅ Manual override capabilities for emergency situations
- ✅ Detailed error logging and alerting system

---

### **🔬 TESTING & VALIDATION (Essential)**

#### **3.8 Integration Testing Suite**
**Priority**: 🟡 **HIGH** | **Estimated**: 3-4 days

**Task**: Create comprehensive integration tests for production deployment
- **File**: `tests/integration/test_production_system.py` (new)
- **Scope**: End-to-end system testing with real components
- **Automation**: CI/CD integration for deployment validation

```python
# tests/integration/test_production_system.py
class TestProductionSystem:
    """Integration tests for complete PEMS v2 system."""
    
    @pytest.mark.integration
    async def test_end_to_end_optimization_cycle(self):
        """Test complete optimization cycle from data to control."""
        # 1. Extract real data from InfluxDB
        # 2. Run ML predictions
        # 3. Execute optimization
        # 4. Validate control commands
        # 5. Check safety constraints
        # 6. Verify monitoring data
    
    @pytest.mark.integration
    async def test_failure_recovery_scenarios(self):
        """Test system behavior under failure conditions."""
        scenarios = [
            'mqtt_disconnection',
            'influxdb_unavailable', 
            'optimization_timeout',
            'invalid_predictions',
            'hardware_faults'
        ]
        for scenario in scenarios:
            await self._test_failure_scenario(scenario)
```

**Deliverables**:
- ✅ Comprehensive integration test suite (80%+ coverage)
- ✅ Automated failure scenario testing
- ✅ Performance benchmarking tests
- ✅ CI/CD pipeline integration

---

#### **3.9 Production Validation & Commissioning**
**Priority**: 🔴 **CRITICAL** | **Estimated**: 5-7 days

**Task**: Validate system performance with real Loxone hardware
- **Approach**: Staged deployment with increasing control authority
- **Validation**: Comparative analysis against current Growatt controller
- **Safety**: Comprehensive safety testing and emergency procedures

**Validation Stages**:
```yaml
Stage 1 - Simulation Mode (2 days):
  - Run PEMS v2 alongside existing system
  - Generate recommendations without actual control
  - Validate predictions against real outcomes
  - Collect baseline performance data

Stage 2 - Limited Control (2 days):
  - Control heating in 1-2 test rooms only
  - Monitor temperature tracking performance
  - Validate safety limits and emergency stops
  - Test manual override capabilities

Stage 3 - Full System Control (3 days):
  - Transfer full heating control to PEMS v2
  - Run side-by-side comparison with Growatt controller
  - Measure cost savings and comfort improvements
  - Validate long-term stability and reliability
```

**Deliverables**:
- ✅ Validated system performance with real hardware
- ✅ Documented performance improvements vs baseline
- ✅ Commissioning procedures and safety validation
- ✅ User training materials and operational procedures

---

### **📚 DOCUMENTATION & KNOWLEDGE TRANSFER (Important)**

#### **3.10 Production Documentation**
**Priority**: 🟢 **MEDIUM** | **Estimated**: 2-3 days

**Task**: Create comprehensive production documentation
- **Audience**: System operators, maintenance staff, future developers
- **Formats**: Technical docs, user guides, troubleshooting manuals

**Documentation Structure**:
```markdown
docs/production/
├── OPERATIONS_MANUAL.md          # Day-to-day operations guide
├── TROUBLESHOOTING_GUIDE.md      # Common issues and solutions  
├── PERFORMANCE_TUNING.md         # Optimization parameter tuning
├── EMERGENCY_PROCEDURES.md       # Safety and emergency responses
├── API_REFERENCE.md              # Complete API documentation
├── MONITORING_GUIDE.md           # Dashboard usage and alerts
└── MAINTENANCE_PROCEDURES.md     # Routine maintenance tasks
```

**Deliverables**:
- ✅ Complete production documentation suite
- ✅ User training materials and videos
- ✅ API reference with examples
- ✅ Troubleshooting guides with decision trees

---

#### **3.11 Knowledge Transfer & Training**
**Priority**: 🟢 **MEDIUM** | **Estimated**: 2-3 days

**Task**: Transfer knowledge and train operators for production system
- **Training**: System operation, monitoring, troubleshooting
- **Documentation**: Record institutional knowledge and best practices
- **Support**: Establish support procedures and escalation paths

**Training Program**:
```yaml
Technical Training (1 day):
  - System architecture overview
  - Configuration management
  - Monitoring dashboard usage
  - Basic troubleshooting procedures

Operations Training (1 day):
  - Daily operation procedures
  - Performance monitoring
  - Emergency response protocols
  - Manual override procedures

Advanced Training (Optional):
  - Model retraining procedures
  - Performance tuning techniques
  - Advanced troubleshooting
  - System extension guidelines
```

**Deliverables**:
- ✅ Comprehensive training program with materials
- ✅ Standard operating procedures (SOPs)
- ✅ Support escalation procedures
- ✅ Knowledge base with FAQs and solutions

---

## **🎯 IMPLEMENTATION TIMELINE**

### **Week 1: Core Infrastructure**
- **Days 1-2**: Production MQTT integration and testing
- **Days 3-4**: Mixed-integer optimization enhancement
- **Day 5**: Model persistence system completion

### **Week 2: Monitoring & Reliability**  
- **Days 1-3**: Real-time monitoring dashboard creation
- **Days 4-5**: Performance metrics and analytics system

### **Week 3: Production Readiness**
- **Days 1-2**: Production configuration and security
- **Days 3-5**: Error handling and recovery systems

### **Week 4: Testing & Validation**
- **Days 1-2**: Integration testing suite development
- **Days 3-7**: Production validation and commissioning

### **Week 5: Documentation & Deployment**
- **Days 1-3**: Production documentation creation
- **Days 4-5**: Knowledge transfer and training
- **Weekend**: Final deployment and go-live

---

## **📋 SUCCESS CRITERIA**

### **Technical Performance**
- ✅ **System Reliability**: >99.5% uptime with automatic recovery
- ✅ **Control Execution**: <5 second response time for all control commands
- ✅ **Optimization Performance**: <30s solve time for 48h horizon  
- ✅ **Prediction Accuracy**: PV <10% RMSE, Temperature <0.5°C MAE

### **Control System Performance**
- ✅ **Battery Control**: Successful grid charging during lowest 20% price periods
- ✅ **Inverter Mode**: Automatic mode switching based on price/solar conditions
- ✅ **Heating Control**: Pre-heating achieves target temps before peak prices
- ✅ **Temperature Setpoints**: Dynamic adjustment saves >10% on heating costs
- ✅ **Grid Export**: Dynamic control captures >90% of profitable export opportunities

### **Economic Performance**
- ✅ **Cost Reduction**: >20% reduction in electricity costs vs baseline
- ✅ **Battery Arbitrage**: >€2/day profit from price arbitrage
- ✅ **Export Revenue**: Maximize feed-in tariff revenue when profitable
- ✅ **Peak Shaving**: Reduce peak demand charges by >30%

### **Operational Performance**
- ✅ **Comfort Maintenance**: <1% of time outside comfort bounds
- ✅ **Safety Compliance**: Zero equipment damage or safety violations
- ✅ **Manual Override**: Immediate response to user interventions
- ✅ **System Visibility**: Real-time status of all controllable systems

### **Integration Success**
- ✅ **Loxone Integration**: Seamless MQTT communication with all devices
- ✅ **Growatt Integration**: Reliable battery and inverter control
- ✅ **Monitoring Integration**: All metrics visible in Grafana dashboards
- ✅ **Fallback Operation**: Graceful degradation to manual control if needed

---

## **⚠️ RISK MITIGATION**

### **Technical Risks**
1. **MQTT Communication Failures**
   - *Mitigation*: Robust retry logic, command queuing, offline fallback modes
   
2. **Optimization Convergence Issues**
   - *Mitigation*: Multiple solver strategies, constraint relaxation, cached solutions
   
3. **Hardware Integration Problems**
   - *Mitigation*: Extensive testing in simulation mode, gradual rollout approach

### **Operational Risks**  
1. **User Comfort Disruption**
   - *Mitigation*: Conservative temperature bounds, immediate manual overrides
   
2. **System Complexity**
   - *Mitigation*: Comprehensive documentation, training programs, support procedures
   
3. **Performance Degradation**
   - *Mitigation*: Continuous monitoring, automated alerts, rollback capabilities

---

## **📊 DELIVERABLES SUMMARY**

### **Code Deliverables**
- ✅ Enhanced heating controller with production MQTT integration
- ✅ Robust mixed-integer optimization engine  
- ✅ Comprehensive monitoring and metrics system
- ✅ Production-grade error handling and recovery
- ✅ Complete integration testing suite

### **Infrastructure Deliverables**  
- ✅ 4 comprehensive Grafana monitoring dashboards
- ✅ Automated performance tracking and alerting
- ✅ Production configuration management system
- ✅ Secure credential and environment management

### **Documentation Deliverables**
- ✅ Complete production operations manual
- ✅ Comprehensive troubleshooting guides
- ✅ API reference and development documentation
- ✅ Training materials and knowledge transfer

### **Validation Deliverables**
- ✅ Production system validation report
- ✅ Performance comparison vs baseline system
- ✅ Safety certification and compliance documentation
- ✅ Commissioning procedures and checklists

---

## **🚀 POST-PHASE 3 ROADMAP**

### **Phase 4: Advanced Features (Future)**
- **Stochastic Optimization**: Handle uncertainty with scenario-based planning
- **Online Learning**: Continuous model improvement based on actual performance
- **Advanced Control**: EV charging optimization, demand response integration
- **Predictive Maintenance**: Equipment health monitoring and maintenance scheduling

### **Phase 5: System Expansion (Future)**
- **Multi-building Support**: Scale to multiple properties
- **Grid Services**: Provide grid balancing and ancillary services
- **Energy Trading**: Automated energy market participation
- **IoT Integration**: Support for additional smart home devices

---

**🎯 PHASE 3 OBJECTIVE**: Transform PEMS v2 from a sophisticated development system into a reliable, production-ready energy management solution that delivers measurable value while maintaining the highest standards of safety, reliability, and user satisfaction.

**📈 EXPECTED OUTCOMES**: 
- 20%+ cost reduction in energy bills
- 99.5%+ system reliability and uptime  
- Seamless integration with existing Loxone infrastructure
- Comprehensive monitoring and operational visibility
- Foundation for future advanced energy management features