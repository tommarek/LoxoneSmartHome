#!/usr/bin/env python3
"""
PEMS v2 Controller - Real-time predictive energy management.

This module integrates all PEMS v2 components into the main Loxone Smart Home service:
- ML predictors (PV, thermal, load)
- Optimization engine (MPC)
- Control systems (heating, battery, inverter)
- MQTT interface for Loxone integration

Key Features:
- Periodic optimization cycles (15-30 minute intervals)
- Real-time state monitoring from InfluxDB
- MQTT control interface for manual overrides
- Performance monitoring and health checks
- Graceful degradation for edge cases
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Add PEMS v2 to path for imports
pems_root = Path(__file__).parent.parent.parent / 'pems_v2'
sys.path.insert(0, str(pems_root))

try:
    from models.predictors.pv_predictor import PVPredictor  
    from models.predictors.thermal_predictor import ThermalPredictor
    from models.predictors.load_predictor import LoadPredictor
    from modules.optimization.optimizer import EnergyOptimizer
    from modules.control.unified_controller import UnifiedController
except ImportError as e:
    # Graceful degradation if PEMS v2 is not available
    logging.getLogger(__name__).error(f"PEMS v2 imports failed: {e}")
    # Define placeholder classes to allow service to start
    
    class PVPredictor:
        def __init__(self, *args, **kwargs):
            pass
        async def predict(self, *args, **kwargs):
            return np.zeros(24)
    
    class ThermalPredictor:
        def __init__(self, *args, **kwargs):
            pass
        async def predict(self, *args, **kwargs):
            return {}
    
    class LoadPredictor:
        def __init__(self, *args, **kwargs):
            pass
        async def predict(self, *args, **kwargs):
            return np.ones(24) * 2.0  # 2kW baseline load
    
    class EnergyOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        async def optimize(self, *args, **kwargs):
            return {'status': 'pems_unavailable'}
    
    class UnifiedController:
        def __init__(self, *args, **kwargs):
            pass
        async def initialize(self):
            pass
        async def execute_schedule(self, *args, **kwargs):
            return False
        async def emergency_stop(self):
            pass


class OptimizationResult(BaseModel):
    """Result from optimization cycle."""
    timestamp: datetime
    horizon_hours: int
    heating_schedule: Dict[str, List[float]] = Field(default_factory=dict)
    battery_schedule: List[float] = Field(default_factory=list)
    grid_schedule: List[float] = Field(default_factory=list)
    predicted_cost: float = 0.0
    predicted_comfort_violations: float = 0.0
    solve_time_ms: float = 0.0
    status: str = "success"


class PEMSControllerSettings(BaseModel):
    """PEMS v2 controller configuration."""
    
    enabled: bool = Field(default=False, description="Enable PEMS v2 controller")
    
    # Optimization intervals
    optimization_interval_minutes: int = Field(
        default=30, description="Interval between optimization cycles"
    )
    prediction_horizon_hours: int = Field(
        default=24, description="Prediction horizon for optimization"
    )
    
    # Model paths (relative to PEMS v2 directory)
    pv_model_path: str = Field(
        default="models/saved/pv_predictor.pkl",
        description="Path to PV prediction model"
    )
    thermal_model_path: str = Field(
        default="models/saved/thermal_predictor.pkl", 
        description="Path to thermal prediction model"
    )
    load_model_path: str = Field(
        default="models/saved/load_predictor.pkl",
        description="Path to load prediction model" 
    )
    
    # Control settings
    simulation_mode: bool = Field(
        default=True, description="Run in simulation mode (no actual control)"
    )
    safety_checks: bool = Field(
        default=True, description="Enable safety constraint checking"
    )
    emergency_stop_temperature: float = Field(
        default=15.0, description="Emergency stop if room temp below this"
    )
    
    # System mode
    system_mode: str = Field(
        default="NORMAL", description="System operating mode"
    )
    
    # Performance monitoring
    max_solve_time_seconds: float = Field(
        default=2.0, description="Maximum allowed optimization time"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance metrics collection"
    )


class PEMSController:
    """Main PEMS v2 controller for real-time operation."""
    
    def __init__(
        self, 
        mqtt_client,
        influxdb_client, 
        settings
    ):
        """Initialize PEMS controller."""
        self.mqtt_client = mqtt_client
        self.influxdb_client = influxdb_client
        self.settings = settings
        self.logger = logging.getLogger(f"PEMS.{self.__class__.__name__}")
        
        # Initialize components (with graceful degradation)
        self._initialize_components()
        
        # State tracking
        self.last_optimization: Optional[OptimizationResult] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.health_status = {
            'predictors': 'unknown',
            'optimizer': 'unknown', 
            'controller': 'unknown',
            'last_optimization': None,
            'error_count': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'optimization_count': 0,
            'success_count': 0,
            'avg_solve_time_ms': 0,
            'last_error': None
        }
    
    def _initialize_components(self) -> None:
        """Initialize PEMS v2 components with error handling."""
        try:
            # Create PEMS settings from our settings
            pems_settings = PEMSSettings(
                optimization_interval_minutes=self.settings.optimization_interval_minutes,
                prediction_horizon_hours=self.settings.prediction_horizon_hours,
                simulation_mode=self.settings.simulation_mode
            )
            
            # Initialize predictors
            self.pv_predictor = PVPredictor(
                model_path=self.settings.pv_model_path
            )
            self.thermal_predictor = ThermalPredictor(
                model_path=self.settings.thermal_model_path
            )
            self.load_predictor = LoadPredictor(
                model_path=self.settings.load_model_path
            )
            
            # Initialize optimizer  
            self.optimizer = EnergyOptimizer(settings=pems_settings)
            
            # Initialize controller
            self.controller = UnifiedController(
                mqtt_client=self.mqtt_client,
                simulation_mode=self.settings.simulation_mode
            )
            
            self.logger.info("PEMS v2 components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PEMS components: {e}")
            # Components will be placeholder classes that do nothing
    
    async def initialize(self) -> None:
        """Initialize PEMS controller and load models."""
        if not self.settings.enabled:
            self.logger.info("PEMS v2 controller disabled in settings")
            return
            
        self.logger.info("Initializing PEMS v2 controller...")
        
        try:
            # Load ML models
            await self._load_models()
            
            # Initialize controller
            await self.controller.initialize()
            
            # Subscribe to MQTT topics
            await self._setup_mqtt_subscriptions()
            
            # Initial health check
            await self._health_check()
            
            self.logger.info("PEMS v2 controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"PEMS controller initialization failed: {e}")
            # Mark as degraded but continue running
            self.health_status['predictors'] = 'degraded'
            self.health_status['optimizer'] = 'degraded'
            self.health_status['controller'] = 'degraded'
    
    async def _load_models(self) -> None:
        """Load ML prediction models."""
        try:
            self.logger.info("Loading ML prediction models...")
            
            # Load models concurrently
            load_tasks = [
                self._load_model_safe(self.pv_predictor, "PV"),
                self._load_model_safe(self.thermal_predictor, "Thermal"),
                self._load_model_safe(self.load_predictor, "Load")
            ]
            
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Check results
            successful = sum(1 for r in results if r is True)
            self.logger.info(f"Loaded {successful}/3 prediction models successfully")
            
            if successful >= 2:
                self.health_status['predictors'] = 'healthy'
            elif successful >= 1:
                self.health_status['predictors'] = 'degraded'
            else:
                self.health_status['predictors'] = 'failed'
                
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            self.health_status['predictors'] = 'failed'
    
    async def _load_model_safe(self, predictor, name: str) -> bool:
        """Safely load a single model."""
        try:
            if hasattr(predictor, 'load_model'):
                await predictor.load_model()
            self.logger.debug(f"{name} predictor loaded")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load {name} predictor: {e}")
            return False
    
    async def start(self) -> None:
        """Start the optimization loop."""
        if not self.settings.enabled:
            self.logger.info("PEMS controller not started - disabled in settings")
            return
            
        self.is_running = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info(
            f"Started PEMS optimization loop "
            f"(interval: {self.settings.optimization_interval_minutes} min, "
            f"horizon: {self.settings.prediction_horizon_hours}h, "
            f"simulation: {self.settings.simulation_mode})"
        )
    
    async def stop(self) -> None:
        """Stop the optimization loop."""
        self.is_running = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        # Emergency stop controller
        try:
            await self.controller.emergency_stop()
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
        
        self.logger.info("PEMS v2 controller stopped")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self.is_running:
            try:
                # Run optimization cycle
                start_time = datetime.now()
                result = await self._run_optimization_cycle()
                cycle_time = (datetime.now() - start_time).total_seconds()
                
                # Update performance metrics
                self.performance_metrics['optimization_count'] += 1
                
                if result and result.status == 'success':
                    # Apply control schedule
                    await self._apply_control_schedule(result)
                    
                    # Publish status
                    await self._publish_status(result)
                    
                    # Store result
                    self.last_optimization = result
                    self.performance_metrics['success_count'] += 1
                    
                    # Update solve time average
                    old_avg = self.performance_metrics['avg_solve_time_ms']
                    count = self.performance_metrics['success_count']
                    self.performance_metrics['avg_solve_time_ms'] = (
                        (old_avg * (count - 1) + result.solve_time_ms) / count
                    )
                else:
                    self.health_status['error_count'] += 1
                    self.performance_metrics['last_error'] = datetime.now()
                
                # Performance monitoring
                if self.settings.enable_performance_monitoring:
                    await self._publish_performance_metrics()
                
                # Health check every 10 cycles
                if self.performance_metrics['optimization_count'] % 10 == 0:
                    await self._health_check()
                
                # Log performance
                self.logger.debug(
                    f"Optimization cycle completed in {cycle_time:.1f}s "
                    f"(solve: {result.solve_time_ms:.0f}ms)" if result else 
                    f"Optimization cycle failed in {cycle_time:.1f}s"
                )
                
                # Wait for next cycle
                await asyncio.sleep(self.settings.optimization_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization cycle error: {e}", exc_info=True)
                self.health_status['error_count'] += 1
                # Short delay before retry
                await asyncio.sleep(60)
    
    async def _run_optimization_cycle(self) -> Optional[OptimizationResult]:
        """Run a single optimization cycle."""
        start_time = datetime.now()
        
        try:
            self.logger.debug("Starting optimization cycle...")
            
            # Get current system state
            current_state = await self._get_current_state()
            
            # Generate predictions
            predictions = await self._generate_predictions(current_state)
            
            # Get electricity prices
            prices = await self._get_electricity_prices()
            
            # Run optimization
            optimization_result = await self._run_optimization(
                predictions, current_state, prices
            )
            
            # Calculate solve time
            solve_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            if optimization_result and optimization_result.get('status') == 'optimal':
                result = OptimizationResult(
                    timestamp=start_time,
                    horizon_hours=self.settings.prediction_horizon_hours,
                    heating_schedule=optimization_result.get('heating_schedule', {}),
                    battery_schedule=optimization_result.get('battery_power', []),
                    grid_schedule=optimization_result.get('grid_power', []),
                    predicted_cost=optimization_result.get('total_cost', 0.0),
                    predicted_comfort_violations=optimization_result.get('comfort_penalty', 0.0),
                    solve_time_ms=solve_time_ms,
                    status="success"
                )
                
                self.logger.info(
                    f"Optimization completed: cost={result.predicted_cost:.2f} CZK, "
                    f"comfort_violations={result.predicted_comfort_violations:.1f}°C·h, "
                    f"solve_time={solve_time_ms:.0f}ms"
                )
                
                return result
            else:
                self.logger.warning("Optimization failed or returned suboptimal result")
                return OptimizationResult(
                    timestamp=start_time,
                    horizon_hours=self.settings.prediction_horizon_hours,
                    solve_time_ms=solve_time_ms,
                    status="failed"
                )
                
        except Exception as e:
            self.logger.error(f"Optimization cycle failed: {e}")
            solve_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return OptimizationResult(
                timestamp=start_time,
                horizon_hours=self.settings.prediction_horizon_hours,
                solve_time_ms=solve_time_ms,
                status="error"
            )
    
    async def _get_current_state(self) -> Dict[str, Any]:
        """Get current system state from InfluxDB."""
        try:
            # This would query real InfluxDB data in production
            # For now, return mock state
            
            current_state = {
                'thermal': {
                    'room_temperatures': {
                        'obyvak': 21.5,
                        'kuchyne': 20.8,
                        'loznice': 19.2,
                        'pracovna': 21.0
                    },
                    'outdoor_temp': 5.0,
                    'setpoints': {
                        'obyvak': 22.0,
                        'kuchyne': 22.0,
                        'loznice': 20.0,
                        'pracovna': 21.0
                    }
                },
                'battery_soc': 0.65,  # 65% charge
                'current_load': 2500,  # 2.5kW
                'pv_power': 0,  # Nighttime
                'timestamp': datetime.now()
            }
            
            return current_state
            
        except Exception as e:
            self.logger.error(f"Failed to get current state: {e}")
            # Return safe defaults
            return {
                'thermal': {'room_temperatures': {}, 'outdoor_temp': 5.0, 'setpoints': {}},
                'battery_soc': 0.5,
                'current_load': 2000,
                'pv_power': 0,
                'timestamp': datetime.now()
            }
    
    async def _generate_predictions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forecasts from all predictors."""
        try:
            # Generate predictions concurrently
            prediction_tasks = [
                self._predict_safe(
                    self.pv_predictor, "PV", 
                    horizon_hours=self.settings.prediction_horizon_hours,
                    current_state=current_state
                ),
                self._predict_safe(
                    self.thermal_predictor, "Thermal",
                    horizon_hours=self.settings.prediction_horizon_hours, 
                    current_state=current_state
                ),
                self._predict_safe(
                    self.load_predictor, "Load",
                    horizon_hours=self.settings.prediction_horizon_hours,
                    current_state=current_state
                )
            ]
            
            pv_forecast, thermal_forecast, load_forecast = await asyncio.gather(
                *prediction_tasks, return_exceptions=True
            )
            
            return {
                'pv_forecast': pv_forecast if not isinstance(pv_forecast, Exception) else np.zeros(self.settings.prediction_horizon_hours),
                'thermal_forecast': thermal_forecast if not isinstance(thermal_forecast, Exception) else {},
                'load_forecast': load_forecast if not isinstance(load_forecast, Exception) else np.ones(self.settings.prediction_horizon_hours) * 2.0
            }
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            # Return safe defaults
            hours = self.settings.prediction_horizon_hours
            return {
                'pv_forecast': np.zeros(hours),
                'thermal_forecast': {},
                'load_forecast': np.ones(hours) * 2.0
            }
    
    async def _predict_safe(self, predictor, name: str, **kwargs) -> Any:
        """Safely run a predictor."""
        try:
            if hasattr(predictor, 'predict'):
                result = await predictor.predict(**kwargs)
                self.logger.debug(f"{name} prediction completed")
                return result
            else:
                self.logger.warning(f"{name} predictor not available")
                # Return appropriate default based on predictor type
                hours = kwargs.get('horizon_hours', 24)
                if name == 'PV':
                    return np.zeros(hours)
                elif name == 'Load':
                    return np.ones(hours) * 2.0
                else:  # Thermal
                    return {}
        except Exception as e:
            self.logger.warning(f"{name} prediction failed: {e}")
            # Return defaults
            hours = kwargs.get('horizon_hours', 24)
            if name == 'PV':
                return np.zeros(hours)
            elif name == 'Load':
                return np.ones(hours) * 2.0
            else:
                return {}
    
    async def _get_electricity_prices(self) -> np.ndarray:
        """Get electricity prices for optimization horizon."""
        try:
            # This would query real price data in production
            # For now, return typical Czech prices with day/night variation
            
            hours = self.settings.prediction_horizon_hours
            current_hour = datetime.now().hour
            
            # Simple day/night pricing (CZK/kWh)
            prices = []
            for h in range(hours):
                hour = (current_hour + h) % 24
                if 22 <= hour or hour <= 6:  # Night tariff
                    prices.append(2.5)
                else:  # Day tariff
                    prices.append(4.0)
            
            return np.array(prices)
            
        except Exception as e:
            self.logger.error(f"Failed to get electricity prices: {e}")
            # Return default flat rate
            return np.ones(self.settings.prediction_horizon_hours) * 3.5
    
    async def _run_optimization(
        self, predictions: Dict[str, Any], current_state: Dict[str, Any], prices: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Run the optimization engine."""
        try:
            if hasattr(self.optimizer, 'optimize'):
                result = await self.optimizer.optimize(
                    pv_forecast=predictions['pv_forecast'],
                    load_forecast=predictions['load_forecast'],
                    thermal_state=current_state['thermal'],
                    thermal_model=self.thermal_predictor,
                    prices=prices,
                    battery_soc=current_state['battery_soc']
                )
                return result
            else:
                self.logger.warning("Optimizer not available")
                return {'status': 'unavailable'}
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _apply_control_schedule(self, result: OptimizationResult) -> None:
        """Apply the optimized control schedule."""
        if self.settings.simulation_mode:
            self.logger.debug("Simulation mode - control schedule not applied")
            return
        
        try:
            self.logger.info("Applying optimized control schedule...")
            
            # Apply first timestep immediately
            schedule = {
                'heating': {
                    room: schedule_list[0] if schedule_list else 0
                    for room, schedule_list in result.heating_schedule.items()
                },
                'battery_power': result.battery_schedule[0] if result.battery_schedule else 0,
                'mode': self.settings.system_mode
            }
            
            # Safety checks
            if self.settings.safety_checks:
                if not await self._safety_check(schedule):
                    self.logger.error("Safety check failed - control schedule not applied")
                    return
            
            # Execute via unified controller
            success = await self.controller.execute_schedule(schedule)
            
            if success:
                self.logger.info("Control schedule applied successfully")
                self.health_status['controller'] = 'healthy'
            else:
                self.logger.error("Failed to apply control schedule")
                self.health_status['controller'] = 'degraded'
                
        except Exception as e:
            self.logger.error(f"Control schedule application failed: {e}")
            self.health_status['controller'] = 'failed'
    
    async def _safety_check(self, schedule: Dict[str, Any]) -> bool:
        """Perform safety checks on control schedule."""
        try:
            # Check room temperatures
            current_state = await self._get_current_state()
            room_temps = current_state.get('thermal', {}).get('room_temperatures', {})
            
            for room, temp in room_temps.items():
                if temp < self.settings.emergency_stop_temperature:
                    self.logger.error(
                        f"Emergency stop: {room} temperature {temp:.1f}°C below threshold "
                        f"{self.settings.emergency_stop_temperature}°C"
                    )
                    return False
            
            # Check battery schedule (prevent damage)
            battery_power = schedule.get('battery_power', 0)
            if abs(battery_power) > 10000:  # 10kW limit
                self.logger.error(f"Battery power {battery_power}W exceeds safety limit")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    async def _publish_status(self, result: OptimizationResult) -> None:
        """Publish optimization status via MQTT."""
        try:
            status = {
                'timestamp': result.timestamp.isoformat(),
                'horizon_hours': result.horizon_hours,
                'predicted_cost': result.predicted_cost,
                'comfort_violations': result.predicted_comfort_violations,
                'solve_time_ms': result.solve_time_ms,
                'status': result.status,
                'simulation_mode': self.settings.simulation_mode,
                'next_update': (
                    result.timestamp + 
                    timedelta(minutes=self.settings.optimization_interval_minutes)
                ).isoformat()
            }
            
            await self.mqtt_client.publish(
                'pems/status/optimization',
                json.dumps(status)
            )
            
            # Publish detailed schedules
            if result.heating_schedule:
                await self.mqtt_client.publish(
                    'pems/status/schedule/heating',
                    json.dumps(result.heating_schedule)
                )
            
            if result.battery_schedule:
                await self.mqtt_client.publish(
                    'pems/status/schedule/battery',
                    json.dumps(result.battery_schedule)
                )
                
        except Exception as e:
            self.logger.error(f"Status publishing failed: {e}")
    
    async def _publish_performance_metrics(self) -> None:
        """Publish performance metrics to InfluxDB."""
        try:
            metrics = {
                'optimization_count': self.performance_metrics['optimization_count'],
                'success_count': self.performance_metrics['success_count'],
                'success_rate': (
                    self.performance_metrics['success_count'] / 
                    max(1, self.performance_metrics['optimization_count'])
                ),
                'avg_solve_time_ms': self.performance_metrics['avg_solve_time_ms'],
                'error_count': self.health_status['error_count'],
                'predictors_status': self.health_status['predictors'],
                'optimizer_status': self.health_status['optimizer'],
                'controller_status': self.health_status['controller']
            }
            
            # This would write to InfluxDB in production
            self.logger.debug(f"Performance metrics: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Performance metrics publishing failed: {e}")
    
    async def _health_check(self) -> None:
        """Perform health check on all components."""
        try:
            # Check predictors
            if hasattr(self.pv_predictor, 'health_check'):
                predictor_health = await self.pv_predictor.health_check()
                self.health_status['predictors'] = predictor_health
            
            # Check optimizer
            if hasattr(self.optimizer, 'health_check'):
                optimizer_health = await self.optimizer.health_check()
                self.health_status['optimizer'] = optimizer_health
            
            # Check controller
            if hasattr(self.controller, 'health_check'):
                controller_health = await self.controller.health_check()
                self.health_status['controller'] = controller_health
            
            # Update last optimization time
            if self.last_optimization:
                self.health_status['last_optimization'] = self.last_optimization.timestamp
            
            # Log health status
            overall_health = "healthy"
            if any(status == 'failed' for status in [
                self.health_status['predictors'],
                self.health_status['optimizer'], 
                self.health_status['controller']
            ]):
                overall_health = "degraded"
            
            self.logger.debug(f"Health check: {overall_health} - {self.health_status}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    async def _setup_mqtt_subscriptions(self) -> None:
        """Setup MQTT subscriptions for manual control."""
        try:
            topics = [
                ('pems/control/+/override', self._handle_room_override),
                ('pems/control/mode', self._handle_mode_change),
                ('pems/control/reoptimize', self._handle_reoptimize),
                ('pems/control/battery/+', self._handle_battery_control),
                ('pems/control/emergency_stop', self._handle_emergency_stop)
            ]
            
            for topic, handler in topics:
                await self.mqtt_client.subscribe(topic, handler)
                
            self.logger.info("MQTT control topics subscribed")
            
        except Exception as e:
            self.logger.error(f"MQTT subscription setup failed: {e}")
    
    async def _handle_room_override(self, topic: str, payload: str) -> None:
        """Handle manual room control override."""
        try:
            # Extract room from topic: pems/control/{room}/override
            parts = topic.split('/')
            if len(parts) >= 3:
                room = parts[2]
                
                data = json.loads(payload)
                
                if 'temperature' in data:
                    success = await self.controller.set_room_temperature(
                        room,
                        data['temperature'],
                        duration_minutes=data.get('duration', 60)
                    )
                    self.logger.info(
                        f"Manual temperature override: {room} -> {data['temperature']}°C "
                        f"{'✅' if success else '❌'}"
                    )
                
                elif 'heating' in data:
                    success = await self.controller.set_room_heating(
                        room,
                        data['heating'],
                        duration_minutes=data.get('duration', 60)
                    )
                    self.logger.info(
                        f"Manual heating override: {room} -> {'ON' if data['heating'] else 'OFF'} "
                        f"{'✅' if success else '❌'}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Room override handling failed: {e}")
    
    async def _handle_mode_change(self, topic: str, payload: str) -> None:
        """Handle system mode changes."""
        try:
            mode = payload.upper().strip()
            valid_modes = ['NORMAL', 'ECONOMY', 'COMFORT', 'AWAY', 'EMERGENCY']
            
            if mode in valid_modes:
                old_mode = self.settings.system_mode
                self.settings.system_mode = mode
                self.logger.info(f"System mode changed: {old_mode} -> {mode}")
                
                # Trigger immediate reoptimization for mode changes
                await self._handle_reoptimize(topic, payload)
            else:
                self.logger.warning(f"Invalid system mode: {mode}")
                
        except Exception as e:
            self.logger.error(f"Mode change handling failed: {e}")
    
    async def _handle_reoptimize(self, topic: str, payload: str) -> None:
        """Handle manual reoptimization request."""
        try:
            self.logger.info("Manual reoptimization requested")
            
            # Run optimization cycle immediately
            result = await self._run_optimization_cycle()
            
            if result and result.status == 'success':
                await self._apply_control_schedule(result)
                await self._publish_status(result)
                self.logger.info("Manual reoptimization completed successfully")
            else:
                self.logger.error("Manual reoptimization failed")
                
        except Exception as e:
            self.logger.error(f"Manual reoptimization failed: {e}")
    
    async def _handle_battery_control(self, topic: str, payload: str) -> None:
        """Handle battery control commands."""
        try:
            # Extract command from topic: pems/control/battery/{command}
            parts = topic.split('/')
            if len(parts) >= 4:
                command = parts[3]
                
                if command == 'mode':
                    # Battery mode change
                    mode = payload.upper().strip()
                    valid_modes = ['AUTO', 'CHARGE', 'DISCHARGE', 'IDLE']
                    
                    if mode in valid_modes:
                        # This would be forwarded to the battery controller
                        self.logger.info(f"Battery mode change: {mode}")
                        
                elif command == 'power':
                    # Direct power control
                    data = json.loads(payload)
                    power_kw = data.get('power_kw', 0)
                    duration = data.get('duration', 60)
                    
                    # Safety check
                    if abs(power_kw) <= 10:  # 10kW limit
                        self.logger.info(f"Battery power control: {power_kw}kW for {duration}min")
                        # This would be forwarded to the battery controller
                    else:
                        self.logger.error(f"Battery power {power_kw}kW exceeds safety limit")
                        
        except Exception as e:
            self.logger.error(f"Battery control handling failed: {e}")
    
    async def _handle_emergency_stop(self, topic: str, payload: str) -> None:
        """Handle emergency stop command."""
        try:
            self.logger.warning("Emergency stop requested via MQTT")
            
            # Stop optimization loop
            self.is_running = False
            
            # Emergency stop all controllers
            await self.controller.emergency_stop()
            
            # Publish status
            await self.mqtt_client.publish(
                'pems/status/emergency',
                json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'emergency_stop',
                    'reason': 'mqtt_command'
                })
            )
            
            self.logger.warning("Emergency stop completed")
            
        except Exception as e:
            self.logger.error(f"Emergency stop handling failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current PEMS controller status."""
        return {
            'enabled': self.settings.enabled,
            'running': self.is_running,
            'simulation_mode': self.settings.simulation_mode,
            'system_mode': self.settings.system_mode,
            'health_status': self.health_status,
            'performance_metrics': self.performance_metrics,
            'last_optimization': (
                self.last_optimization.timestamp.isoformat() 
                if self.last_optimization else None
            )
        }