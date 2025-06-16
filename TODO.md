# TODO: Implementation Tasks

## ✅ COMPLETED: Winter Analysis Enhancement (2025-06-15)

**DONE:** Enhanced winter analysis to use all historical data since September 2022
- ✅ Modified `get_available_winter_months()` to query from 2022-09-01 instead of last 3 years
- ✅ Enhanced `run_winter_thermal_analysis()` to support analyzing ALL available winters
- ✅ Added `--all-winters` command-line flag for comprehensive multi-winter analysis
- ✅ Updated help documentation with examples for new winter analysis options
- ✅ System now discovers all 3 winter seasons: 2022/2023, 2023/2024, 2024/2025
- ✅ Enables analysis of 882 days of winter data for maximum thermal modeling accuracy

**Files Modified:**
- `pems_v2/analysis/core/unified_data_extractor.py`
- `pems_v2/analysis/run_analysis.py`

**Commit:** `f0a555d` - feat: enhance winter analysis to use all historical data since September 2022

---

## ✅ COMPLETED: Heating Cycle Analysis Implementation

**SURPRISING DISCOVERY:** The heating cycle analysis described in the original TODO is already fully implemented in the codebase!

**What exists:**
- ✅ `_detect_heating_cycles()` - Detects individual heating on/off events with proper filtering
- ✅ `_analyze_heating_cycle_decay()` - Analyzes post-heating exponential decay with advanced data quality checks
- ✅ `_analyze_heating_cycle_rise()` - Calculates thermal capacitance from heating rise rate
- ✅ `_estimate_rc_decoupled()` - Primary RC estimation using heating cycles as controlled experiments
- ✅ Advanced filtering: nighttime cycle prioritization, monotonic decay checks, outdoor temp stability
- ✅ Confidence scoring based on cycle count, R² quality, and data cleanliness
- ✅ Physical plausibility checks with proper bounds (R: 0.008-0.5 K/W, C: 2-100 MJ/K, τ: 3-350h)

**No implementation needed - focus shifts to debugging and integration!**

---

## 🔥 High Priority: Debug RC Estimation with Production Data

**Problem:** Despite having sophisticated heating cycle analysis, some rooms still produce unrealistic RC parameters in production.

**Root Causes to Investigate:**
1. Data quality issues (sensor noise, stuck values, missing data)
2. Incorrect power ratings in `system_config.json` for specific rooms
3. Multi-zone thermal coupling not properly accounted for
4. Solar gains contaminating heating cycle analysis
5. Underfloor heating systems with very long time constants

### Task 1: Run Diagnostic Analysis on Problem Rooms
**Script:** Create `pems_v2/analysis/debug_rc_estimation.py`

**Requirements:**
```python
#!/usr/bin/env python3
"""Debug RC estimation issues for specific rooms."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from analyzers.thermal_analysis import ThermalAnalyzer
from core.unified_data_extractor import UnifiedDataExtractor
from config.system_config import ROOM_CONFIG

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rc_debug.log'),
        logging.StreamHandler()
    ]
)

async def debug_room_rc(room_name: str, start_date: str, end_date: str):
    """Run detailed RC analysis for a specific room."""
    
    # Extract data
    extractor = UnifiedDataExtractor()
    room_data = await extractor.extract_room_temperatures(start_date, end_date, [room_name])
    weather_data = await extractor.extract_weather_data(start_date, end_date)
    relay_data = await extractor.extract_relay_states(start_date, end_date, [room_name])
    
    # Check data quality
    print(f"\n=== Data Quality Check for {room_name} ===")
    print(f"Temperature records: {len(room_data)}")
    print(f"Missing values: {room_data[room_name].isna().sum()}")
    print(f"Temperature range: {room_data[room_name].min():.1f} - {room_data[room_name].max():.1f}°C")
    
    # Check for stuck sensors
    consecutive_same = (room_data[room_name].diff() == 0).rolling(12).sum().max()
    print(f"Max consecutive same values: {consecutive_same} (>12 indicates stuck sensor)")
    
    # Check power rating
    power_w = ROOM_CONFIG.get(room_name, {}).get('power_w', 0)
    print(f"Configured power: {power_w}W")
    
    # Run thermal analysis with debug logging
    analyzer = ThermalAnalyzer()
    results = analyzer.analyze_room(room_name, room_data, weather_data, relay_data)
    
    # Output detailed results
    print(f"\n=== RC Estimation Results ===")
    print(f"Method used: {results.get('rc_params', {}).get('method', 'unknown')}")
    print(f"R value: {results.get('rc_params', {}).get('R', 0):.4f} K/W")
    print(f"C value: {results.get('rc_params', {}).get('C', 0)/1e6:.1f} MJ/K")
    print(f"Time constant: {results.get('rc_params', {}).get('time_constant', 0):.1f} hours")
    print(f"Confidence: {results.get('rc_params', {}).get('confidence', 0):.2f}")
    print(f"Physically valid: {results.get('rc_params', {}).get('physically_valid', False)}")
    
    # Analyze heating cycles
    cycles = results.get('rc_params', {}).get('heating_cycles', [])
    print(f"\n=== Heating Cycle Analysis ===")
    print(f"Total cycles found: {len(cycles)}")
    print(f"Valid decay analyses: {results.get('rc_params', {}).get('successful_decays', 0)}")
    print(f"Valid rise analyses: {results.get('rc_params', {}).get('successful_rises', 0)}")
    
    return results

# Problem rooms to debug
PROBLEM_ROOMS = ['chodba_dole', 'koupelna_dole', 'posilovna']

if __name__ == "__main__":
    # Analyze last winter's data
    for room in PROBLEM_ROOMS:
        asyncio.run(debug_room_rc(room, "2024-12-01", "2025-03-01"))
```

### Task 2: Verify and Update Room Power Ratings
**File:** `pems_v2/config/system_config.json`

**Action Items:**
1. Cross-reference actual heating element specifications
2. Verify underfloor vs radiator heating types
3. Update power ratings based on electrical measurements
4. Add heating system type field for each room

### Task 3: Implement Enhanced Data Quality Filters
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`

**Enhancements to `_detect_heating_cycles()`:**
```python
def _validate_cycle_data_quality(self, df: pd.DataFrame, cycle: Dict) -> Tuple[bool, str]:
    """Enhanced data quality validation for heating cycles."""
    
    # Extract cycle data
    cycle_data = df.loc[cycle['start_time']:cycle['end_time']]
    
    # Check 1: Sensor stuck detection
    temp_changes = cycle_data['room_temp'].diff().abs()
    if (temp_changes < 0.01).sum() > len(cycle_data) * 0.5:
        return False, "Stuck sensor detected"
    
    # Check 2: Unrealistic temperature jumps
    max_jump = temp_changes.max()
    if max_jump > 2.0:  # More than 2°C in 5 minutes
        return False, f"Unrealistic temperature jump: {max_jump:.1f}°C"
    
    # Check 3: Heating power validation
    temp_rise = cycle['peak_temp'] - cycle['start_temp']
    duration_hours = cycle['duration_minutes'] / 60
    implied_power = (temp_rise * self.typical_capacitance) / (duration_hours * 3600)
    
    if implied_power > cycle['power_w'] * 2:  # More than 2x configured power
        return False, f"Implied power {implied_power:.0f}W exceeds 2x configured {cycle['power_w']}W"
    
    # Check 4: Multi-zone interference
    # (This would need access to other room data)
    
    return True, "Valid"
```

---

## 🚀 High Priority: PEMS v2 Service Integration

**Current State:** All PEMS v2 components are implemented and tested, but not integrated into the main service.

### Task 1: Create PEMS Controller Module
**File:** `loxone_smart_home/modules/pems_controller.py`

```python
#!/usr/bin/env python3
"""PEMS v2 Controller - Real-time predictive energy management."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# PEMS v2 imports (with proper path setup)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'pems_v2'))

from models.predictors import PVPredictor, ThermalPredictor, LoadPredictor
from modules.optimization import EnergyOptimizer
from modules.control import UnifiedController
from config.settings import PEMSSettings


class OptimizationResult(BaseModel):
    """Result from optimization cycle."""
    timestamp: datetime
    horizon_hours: int
    heating_schedule: Dict[str, List[float]]
    battery_schedule: List[float]
    grid_schedule: List[float]
    predicted_cost: float
    predicted_comfort_violations: float
    solve_time_ms: float


class PEMSController:
    """Main PEMS v2 controller for real-time operation."""
    
    def __init__(
        self,
        mqtt_client,
        influxdb_client,
        settings: PEMSSettings
    ):
        self.mqtt_client = mqtt_client
        self.influxdb_client = influxdb_client
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.pv_predictor = PVPredictor(model_path=settings.pv_model_path)
        self.thermal_predictor = ThermalPredictor(model_path=settings.thermal_model_path)
        self.load_predictor = LoadPredictor(model_path=settings.load_model_path)
        self.optimizer = EnergyOptimizer(settings=settings.optimization)
        self.controller = UnifiedController(
            mqtt_client=mqtt_client,
            settings=settings.control
        )
        
        # State tracking
        self.last_optimization: Optional[OptimizationResult] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_running = False
        
    async def initialize(self) -> None:
        """Initialize PEMS controller and load models."""
        self.logger.info("Initializing PEMS v2 controller...")
        
        # Load ML models
        try:
            await self.pv_predictor.load_model()
            await self.thermal_predictor.load_model()
            await self.load_predictor.load_model()
            self.logger.info("All prediction models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
            
        # Initialize controller
        await self.controller.initialize()
        
        # Subscribe to MQTT topics
        await self._setup_mqtt_subscriptions()
        
        self.logger.info("PEMS v2 controller initialized")
        
    async def start(self) -> None:
        """Start the optimization loop."""
        self.is_running = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info(f"Started optimization loop (interval: {self.settings.optimization_interval_minutes} min)")
        
    async def stop(self) -> None:
        """Stop the optimization loop."""
        self.is_running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        await self.controller.emergency_stop()
        self.logger.info("PEMS v2 controller stopped")
        
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.is_running:
            try:
                # Run optimization cycle
                result = await self._run_optimization_cycle()
                
                if result:
                    # Apply control schedule
                    await self._apply_control_schedule(result)
                    
                    # Publish status
                    await self._publish_status(result)
                    
                    # Store result
                    self.last_optimization = result
                    
                # Wait for next cycle
                await asyncio.sleep(self.settings.optimization_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Optimization cycle failed: {e}", exc_info=True)
                # Continue running even if one cycle fails
                await asyncio.sleep(60)  # Short delay before retry
                
    async def _run_optimization_cycle(self) -> Optional[OptimizationResult]:
        """Run a single optimization cycle."""
        start_time = datetime.now()
        self.logger.info("Starting optimization cycle...")
        
        try:
            # Get current state from InfluxDB
            current_state = await self._get_current_state()
            
            # Generate predictions
            pv_forecast = await self.pv_predictor.predict(
                horizon_hours=self.settings.prediction_horizon_hours,
                current_state=current_state
            )
            
            thermal_forecast = await self.thermal_predictor.predict(
                horizon_hours=self.settings.prediction_horizon_hours,
                current_state=current_state
            )
            
            load_forecast = await self.load_predictor.predict(
                horizon_hours=self.settings.prediction_horizon_hours,
                current_state=current_state
            )
            
            # Get electricity prices
            prices = await self._get_electricity_prices()
            
            # Run optimization
            optimization_result = await self.optimizer.optimize(
                pv_forecast=pv_forecast,
                load_forecast=load_forecast,
                thermal_state=current_state['thermal'],
                thermal_model=self.thermal_predictor,
                prices=prices,
                battery_soc=current_state['battery_soc']
            )
            
            # Calculate solve time
            solve_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = OptimizationResult(
                timestamp=start_time,
                horizon_hours=self.settings.prediction_horizon_hours,
                heating_schedule=optimization_result['heating_schedule'],
                battery_schedule=optimization_result['battery_power'],
                grid_schedule=optimization_result['grid_power'],
                predicted_cost=optimization_result['total_cost'],
                predicted_comfort_violations=optimization_result['comfort_penalty'],
                solve_time_ms=solve_time_ms
            )
            
            self.logger.info(
                f"Optimization completed in {solve_time_ms:.0f}ms. "
                f"Predicted cost: {result.predicted_cost:.2f} CZK, "
                f"Comfort violations: {result.predicted_comfort_violations:.1f}°C·h"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            return None
            
    async def _get_current_state(self) -> Dict[str, Any]:
        """Get current system state from InfluxDB."""
        # Query current temperatures, battery SoC, power flows
        # This is a simplified version - real implementation would query all needed data
        
        query = '''
        from(bucket: "loxone")
          |> range(start: -10m)
          |> filter(fn: (r) => r._measurement == "temperature" or 
                              r._measurement == "battery" or
                              r._measurement == "power")
          |> last()
        '''
        
        # Execute query and process results
        # (Implementation depends on InfluxDB client details)
        
        return {
            'thermal': {
                'room_temperatures': {},  # Room -> temperature mapping
                'outdoor_temp': 0.0,
                'setpoints': {}
            },
            'battery_soc': 0.5,  # 50% SoC
            'current_load': 2000,  # 2kW
            'pv_power': 0  # Nighttime
        }
        
    async def _get_electricity_prices(self) -> np.ndarray:
        """Get electricity prices for optimization horizon."""
        # Query from InfluxDB or external API
        # Return hourly prices in CZK/kWh
        return np.ones(self.settings.prediction_horizon_hours) * 3.0  # Placeholder
        
    async def _apply_control_schedule(self, result: OptimizationResult) -> None:
        """Apply the optimized control schedule."""
        self.logger.info("Applying optimized control schedule...")
        
        # Apply first timestep immediately
        schedule = {
            'heating': {
                room: schedule[0] 
                for room, schedule in result.heating_schedule.items()
            },
            'battery_power': result.battery_schedule[0],
            'mode': 'NORMAL'
        }
        
        success = await self.controller.execute_schedule(schedule)
        
        if not success:
            self.logger.error("Failed to apply control schedule")
            
    async def _publish_status(self, result: OptimizationResult) -> None:
        """Publish optimization status via MQTT."""
        status = {
            'timestamp': result.timestamp.isoformat(),
            'horizon_hours': result.horizon_hours,
            'predicted_cost': result.predicted_cost,
            'comfort_violations': result.predicted_comfort_violations,
            'solve_time_ms': result.solve_time_ms,
            'next_update': (
                result.timestamp + 
                timedelta(minutes=self.settings.optimization_interval_minutes)
            ).isoformat()
        }
        
        await self.mqtt_client.publish(
            'pems/status/optimization',
            json.dumps(status)
        )
        
    async def _setup_mqtt_subscriptions(self) -> None:
        """Setup MQTT subscriptions for manual control."""
        topics = [
            ('pems/control/+/override', self._handle_override),
            ('pems/control/mode', self._handle_mode_change),
            ('pems/control/reoptimize', self._handle_reoptimize)
        ]
        
        for topic, handler in topics:
            await self.mqtt_client.subscribe(topic, handler)
            
    async def _handle_override(self, topic: str, payload: str) -> None:
        """Handle manual override commands."""
        # Extract room from topic
        parts = topic.split('/')
        if len(parts) >= 3:
            room = parts[2]
            
            try:
                data = json.loads(payload)
                if 'temperature' in data:
                    await self.controller.set_room_temperature(
                        room, 
                        data['temperature'],
                        duration_minutes=data.get('duration', 60)
                    )
                elif 'heating' in data:
                    await self.controller.set_room_heating(
                        room,
                        data['heating'],
                        duration_minutes=data.get('duration', 60)
                    )
            except Exception as e:
                self.logger.error(f"Failed to process override: {e}")
                
    async def _handle_mode_change(self, topic: str, payload: str) -> None:
        """Handle system mode changes."""
        try:
            mode = payload.upper()
            if mode in ['NORMAL', 'ECONOMY', 'COMFORT', 'AWAY']:
                self.settings.system_mode = mode
                self.logger.info(f"System mode changed to: {mode}")
                # Trigger immediate reoptimization
                await self._handle_reoptimize(topic, payload)
        except Exception as e:
            self.logger.error(f"Failed to change mode: {e}")
            
    async def _handle_reoptimize(self, topic: str, payload: str) -> None:
        """Handle manual reoptimization request."""
        self.logger.info("Manual reoptimization requested")
        result = await self._run_optimization_cycle()
        if result:
            await self._apply_control_schedule(result)
            await self._publish_status(result)
```

### Task 2: Update Main Service Integration
**File:** `loxone_smart_home/main.py`

**Add to imports:**
```python
from modules.pems_controller import PEMSController
```

**Add to `__init__`:**
```python
self.pems_controller: Optional[PEMSController] = None
```

**Add to `setup_modules`:**
```python
# PEMS v2 Controller
if self.settings.pems_enabled:
    self.logger.info("Setting up PEMS v2 controller...")
    self.pems_controller = PEMSController(
        mqtt_client=self.mqtt_client,
        influxdb_client=self.influxdb_client,
        settings=self.settings.pems
    )
    await self.pems_controller.initialize()
    self.modules.append(
        asyncio.create_task(self.run_pems_controller())
    )
```

**Add run method:**
```python
async def run_pems_controller(self) -> None:
    """Run PEMS v2 controller."""
    if not self.pems_controller:
        return
        
    try:
        await self.pems_controller.start()
        await self.shutdown_event.wait()
    except asyncio.CancelledError:
        self.logger.info("PEMS controller cancelled")
    except Exception as e:
        self.logger.error(f"PEMS controller error: {e}", exc_info=True)
    finally:
        await self.pems_controller.stop()
```

### Task 3: Update Settings Configuration
**File:** `loxone_smart_home/config/settings.py`

**Add PEMS settings:**
```python
class PEMSSettings(BaseModel):
    """PEMS v2 configuration."""
    
    enabled: bool = Field(default=False, description="Enable PEMS v2 controller")
    
    # Model paths
    pv_model_path: str = Field(
        default="pems_v2/models/saved/pv_predictor.pkl",
        description="Path to PV prediction model"
    )
    thermal_model_path: str = Field(
        default="pems_v2/models/saved/thermal_predictor.pkl",
        description="Path to thermal prediction model"
    )
    load_model_path: str = Field(
        default="pems_v2/models/saved/load_predictor.pkl",
        description="Path to load prediction model"
    )
    
    # Optimization settings
    optimization_interval_minutes: int = Field(
        default=30,
        description="Interval between optimization cycles"
    )
    prediction_horizon_hours: int = Field(
        default=24,
        description="Prediction horizon for optimization"
    )
    
    # Control settings
    control: Dict[str, Any] = Field(
        default_factory=dict,
        description="Control subsystem settings"
    )
    
    # Optimization parameters
    optimization: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optimization algorithm settings"
    )
    
    # System mode
    system_mode: str = Field(
        default="NORMAL",
        description="System operating mode"
    )
```

**Add to main Settings class:**
```python
pems: PEMSSettings = Field(
    default_factory=PEMSSettings,
    description="PEMS v2 settings"
)
```

---

## 🔧 Medium Priority: Production Deployment

### Task 1: Create Docker Configuration for PEMS v2
**File:** `docker-compose.yml`

**Add PEMS v2 volumes:**
```yaml
services:
  loxone-smart-home:
    volumes:
      - ./pems_v2:/app/pems_v2
      - ./pems_v2/models/saved:/app/pems_v2/models/saved
      - ./pems_v2/config:/app/pems_v2/config
```

### Task 2: Create Systemd Service
**File:** `/etc/systemd/system/loxone-smart-home.service`

```ini
[Unit]
Description=Loxone Smart Home with PEMS v2
After=network.target influxdb.service mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/LoxoneSmartHome
Environment="PYTHONPATH=/home/pi/LoxoneSmartHome:/home/pi/LoxoneSmartHome/pems_v2"
ExecStart=/home/pi/LoxoneSmartHome/venv/bin/python -m loxone_smart_home.main
Restart=always
RestartSec=10

# Performance tuning for Raspberry Pi
CPUQuota=80%
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

### Task 3: Performance Monitoring
**File:** `loxone_smart_home/modules/pems_controller.py`

**Add performance metrics:**
```python
async def _publish_performance_metrics(self) -> None:
    """Publish performance metrics to InfluxDB."""
    metrics = {
        'optimization_time_ms': self.last_optimization.solve_time_ms,
        'prediction_accuracy': await self._calculate_prediction_accuracy(),
        'control_success_rate': await self.controller.get_success_rate(),
        'memory_usage_mb': self._get_memory_usage(),
        'cpu_usage_percent': self._get_cpu_usage()
    }
    
    await self.influxdb_client.write(
        bucket='metrics',
        measurement='pems_performance',
        fields=metrics,
        tags={'service': 'pems_v2'}
    )
```

---

## 📋 Medium Priority: MQTT Command Interface

### Task 1: Define MQTT Command Structure
**Documentation:** `docs/PEMS_MQTT_INTERFACE.md`

```markdown
# PEMS v2 MQTT Interface

## Control Topics

### Manual Overrides
- `pems/control/{room}/override`
  ```json
  {
    "temperature": 21.0,  // Target temperature in °C
    "duration": 60        // Duration in minutes
  }
  ```

- `pems/control/{room}/heating`
  ```json
  {
    "enabled": true,     // Heating on/off
    "duration": 30       // Duration in minutes
  }
  ```

### System Control
- `pems/control/mode`
  - Payload: `NORMAL`, `ECONOMY`, `COMFORT`, `AWAY`

- `pems/control/reoptimize`
  - Payload: Any string triggers immediate reoptimization

### Battery Control
- `pems/control/battery/mode`
  - Payload: `AUTO`, `CHARGE`, `DISCHARGE`, `IDLE`

- `pems/control/battery/power`
  ```json
  {
    "power_kw": 3.0,     // Positive = charge, negative = discharge
    "duration": 60       // Duration in minutes
  }
  ```

## Status Topics

### Optimization Status
- `pems/status/optimization`
  ```json
  {
    "timestamp": "2025-06-16T10:30:00Z",
    "horizon_hours": 24,
    "predicted_cost": 156.50,
    "comfort_violations": 0.0,
    "solve_time_ms": 850,
    "next_update": "2025-06-16T11:00:00Z"
  }
  ```

### Predictions
- `pems/status/predictions/pv`
- `pems/status/predictions/load`
- `pems/status/predictions/thermal/{room}`

### Current Schedule
- `pems/status/schedule/heating`
- `pems/status/schedule/battery`
- `pems/status/schedule/grid`
```

### Task 2: Implement Loxone UDP Interface
**File:** `loxone_smart_home/modules/pems_controller.py`

**Add method:**
```python
async def _send_to_loxone(self, data: Dict[str, Any]) -> None:
    """Send PEMS data to Loxone via UDP."""
    # Format: key1=value1;key2=value2;...
    message = ";".join([f"{k}={v}" for k, v in data.items()])
    
    # Send to Loxone UDP port
    await self.udp_sender.send(message, ('loxone.local', 4000))
```

---

## 🌍 Medium Priority: Timezone and Logging Improvements

### Task 1: Update Logging Configuration
**File:** `loxone_smart_home/utils/logging.py`

**Already implemented:** `TimezoneAwareFormatter` with Europe/Prague timezone

### Task 2: Add Service Prefixes
**File:** `loxone_smart_home/main.py`

**Update logger names:**
```python
# In each module's __init__:
self.logger = logging.getLogger(f"PEMS.{self.__class__.__name__}")
```

---

## 📊 Low Priority: Performance Optimizations

### Task 1: Database Query Optimization
**File:** `pems_v2/analysis/core/unified_data_extractor.py`

**Add chunked queries:**
```python
async def extract_large_date_range(
    self,
    start_date: str,
    end_date: str,
    chunk_days: int = 30
) -> pd.DataFrame:
    """Extract data in chunks for large date ranges."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        
        chunk_data = await self.extract_data(
            current.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        )
        chunks.append(chunk_data)
        
        current = chunk_end
        
    return pd.concat(chunks)
```

### Task 2: Model Loading Optimization
**File:** `pems_v2/models/base.py`

**Add lazy loading:**
```python
@property
def model(self):
    """Lazy load model on first access."""
    if self._model is None:
        self._model = self._load_model()
    return self._model
```

---

## 📚 Low Priority: Documentation

### Task 1: Create PRESENTATION.md
**File:** `pems_v2/PRESENTATION.md`

Create comprehensive technical documentation covering:
- System architecture
- Mathematical models
- API documentation
- Performance benchmarks
- Deployment guide

### Task 2: Update README.md
**File:** `README.md`

Add section on PEMS v2:
- Features and capabilities
- Quick start guide
- Configuration options
- MQTT interface reference

---

## 🎯 Implementation Timeline

**Week 1 (High Priority):**
- Day 1-2: Debug RC estimation with production data
- Day 3-4: Create PEMS controller module
- Day 5: Integrate with main service

**Week 2 (Medium Priority):**
- Day 1-2: MQTT command interface
- Day 3: Production deployment setup
- Day 4-5: Testing and debugging

**Week 3 (Low Priority):**
- Documentation updates
- Performance optimizations
- Long-term monitoring setup

---

## 🔍 Testing Strategy

### Unit Tests
- Test each PEMS component in isolation
- Mock external dependencies (MQTT, InfluxDB)
- Verify optimization algorithms

### Integration Tests
- Test full optimization cycle
- Verify control command execution
- Test failover scenarios

### System Tests
- Run with historical data replay
- Verify energy savings
- Monitor system stability

---

## 📝 Success Criteria

1. **RC Estimation:** All rooms produce physically plausible parameters
2. **Integration:** PEMS runs continuously without crashes
3. **Performance:** Optimization completes in <2 seconds
4. **Accuracy:** Predictions within 20% of actual values
5. **Control:** Smooth heating control without oscillations
6. **Cost:** Measurable reduction in energy costs