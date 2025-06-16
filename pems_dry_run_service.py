#!/usr/bin/env python3
"""
PEMS v2 Continuous Dry Run Service

This service runs PEMS v2 optimization cycles continuously in simulation mode,
collecting behavioral data, performance metrics, and operational logs for
analysis and validation. Designed to run as a Docker Compose service.

Features:
1. Continuous optimization cycles (configurable interval)
2. Behavioral data collection and storage
3. Performance metrics tracking
4. Structured logging with rotation
5. Health monitoring and status reporting
6. Data export for analysis
7. Configuration via environment variables

Data Collection:
- Optimization results and convergence metrics
- Control decisions and mode selections
- Performance timing and resource usage
- Error rates and failure patterns
- System state evolution over time
"""

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project roots to path
project_root = Path(__file__).parent
pems_root = project_root / "pems_v2"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(pems_root))

# Data storage directory
# Use local paths when not in container
if os.path.exists("/data"):
    default_data_dir = "/data/pems_dry_run"
    default_log_dir = "/logs/pems_dry_run"
else:
    default_data_dir = "./data/pems_dry_run"
    default_log_dir = "./logs/pems_dry_run"

DATA_DIR = Path(os.getenv("PEMS_DATA_DIR", default_data_dir))
LOG_DIR = Path(os.getenv("PEMS_LOG_DIR", default_log_dir))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OptimizationCycleData:
    """Data structure for optimization cycle results."""

    timestamp: str
    cycle_id: int
    success: bool
    solve_time_seconds: float
    objective_value: float
    message: str

    # System state
    battery_soc: float
    outdoor_temp: float
    room_temperatures: Dict[str, float]
    current_load: float
    pv_power: float

    # Predictions
    pv_forecast_peak: float
    load_forecast_avg: float
    price_forecast_min: float
    price_forecast_max: float
    price_forecast_avg: float

    # Control decisions
    heating_decisions: Dict[str, Dict[str, Any]]
    growatt_decisions: Dict[str, Dict[str, Any]]

    # Performance metrics
    total_cycle_time_ms: float
    prediction_time_ms: float
    optimization_time_ms: float
    control_time_ms: float

    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ServiceMetrics:
    """Service-level metrics and statistics."""

    start_time: str
    uptime_hours: float
    total_cycles: int
    successful_cycles: int
    failed_cycles: int
    success_rate: float
    avg_solve_time_seconds: float
    avg_cycle_time_ms: float
    last_cycle_time: str
    errors_last_hour: int
    memory_usage_mb: float
    cpu_usage_percent: float


class MockMQTTClient:
    """Enhanced mock MQTT client with data collection."""

    def __init__(self, data_collector):
        self.published_messages = []
        self.data_collector = data_collector
        self.logger = logging.getLogger("MockMQTT")

    async def publish(self, topic: str, payload: str):
        """Simulate MQTT publish and collect data."""
        message = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "payload": payload,
        }
        self.published_messages.append(message)

        # Log command for monitoring
        self.logger.debug(f"📤 {topic} -> {payload}")

        # Collect command data
        self.data_collector.add_mqtt_command(message)

    async def subscribe(self, topic: str, callback):
        """Mock subscribe."""
        self.logger.debug(f"📥 SUBSCRIBE: {topic}")

    def get_recent_commands(self, minutes: int = 60) -> List[Dict]:
        """Get commands from last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            msg
            for msg in self.published_messages
            if datetime.fromisoformat(msg["timestamp"]) > cutoff
        ]


class MockInfluxDBClient:
    """Enhanced mock InfluxDB client with realistic data simulation."""

    def __init__(self, data_collector):
        self.queries = []
        self.data_collector = data_collector
        self.logger = logging.getLogger("MockInfluxDB")
        self.simulation_state = self._initialize_simulation_state()

    def _initialize_simulation_state(self):
        """Initialize realistic simulation state that evolves over time."""
        return {
            "battery_soc": 0.45,
            "outdoor_temp": 5.0,
            "room_temperatures": {
                "obyvak": 21.2,
                "kuchyne": 20.8,
                "loznice": 19.5,
                "pracovna": 20.9,
                "hosti": 19.2,
            },
            "current_load": 2800,
            "pv_power": 0,
            "last_heating_state": {},
            "weather_pattern_offset": 0,
        }

    def _evolve_simulation_state(self):
        """Evolve simulation state to create realistic behavioral patterns."""
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # Battery SOC evolution (simplified discharge/charge pattern)
        if 6 <= hour <= 18:  # Day
            self.simulation_state["battery_soc"] = max(
                0.1, self.simulation_state["battery_soc"] - 0.01
            )  # Slow discharge
        else:  # Night
            self.simulation_state["battery_soc"] = min(
                0.9, self.simulation_state["battery_soc"] + 0.02
            )  # Charging

        # Outdoor temperature (daily cycle + weather variation)
        base_temp = 5.0 + 3.0 * np.sin(2 * np.pi * (hour + minute / 60 - 6) / 24)
        weather_variation = 2.0 * np.sin(
            self.simulation_state["weather_pattern_offset"]
        )
        self.simulation_state["outdoor_temp"] = base_temp + weather_variation
        self.simulation_state["weather_pattern_offset"] += 0.1

        # Room temperatures (respond to heating and outdoor temp)
        outdoor_temp = self.simulation_state["outdoor_temp"]
        for room in self.simulation_state["room_temperatures"]:
            current_temp = self.simulation_state["room_temperatures"][room]
            heating_on = self.simulation_state["last_heating_state"].get(room, False)

            # Simple thermal response
            if heating_on:
                target_temp = 22.0
                self.simulation_state["room_temperatures"][
                    room
                ] = current_temp + 0.1 * (target_temp - current_temp)
            else:
                # Natural cooling toward outdoor temp
                self.simulation_state["room_temperatures"][
                    room
                ] = current_temp + 0.02 * (outdoor_temp - current_temp)

        # Load variation (household pattern)
        base_load = 1500
        if 6 <= hour <= 9:  # Morning peak
            load_multiplier = 1.8
        elif 17 <= hour <= 21:  # Evening peak
            load_multiplier = 2.2
        elif 22 <= hour or hour <= 6:  # Night
            load_multiplier = 0.8
        else:  # Day
            load_multiplier = 1.2

        self.simulation_state["current_load"] = base_load * load_multiplier

        # PV power (seasonal winter pattern)
        if 8 <= hour <= 16:  # Daylight hours
            pv_peak_hour = 12
            pv_factor = max(0, np.cos(np.pi * (hour - pv_peak_hour) / 8))
            self.simulation_state["pv_power"] = 1500 * pv_factor  # Low winter PV
        else:
            self.simulation_state["pv_power"] = 0

    async def query(self, query: str) -> Dict[str, Any]:
        """Simulate InfluxDB query with evolving realistic data."""
        self.queries.append(query)
        self.logger.debug(f"🔍 QUERY: {query[:50]}...")

        # Evolve simulation state
        self._evolve_simulation_state()

        # Return current state
        current_state = {
            "thermal": {
                "room_temperatures": self.simulation_state["room_temperatures"].copy(),
                "outdoor_temp": self.simulation_state["outdoor_temp"],
                "setpoints": {
                    "obyvak": 22.0,
                    "kuchyne": 22.0,
                    "loznice": 20.0,
                    "pracovna": 21.0,
                    "hosti": 19.0,
                },
            },
            "battery_soc": self.simulation_state["battery_soc"],
            "current_load": self.simulation_state["current_load"],
            "pv_power": self.simulation_state["pv_power"],
            "timestamp": datetime.now(),
        }

        # Collect state data
        self.data_collector.add_system_state(current_state)

        return current_state


class DataCollector:
    """Collects and stores behavioral data from PEMS operations."""

    def __init__(self):
        self.logger = logging.getLogger("DataCollector")
        self.cycle_data: List[OptimizationCycleData] = []
        self.mqtt_commands: List[Dict] = []
        self.system_states: List[Dict] = []
        self.error_log: List[Dict] = []

        # File paths for data storage
        self.cycle_data_file = DATA_DIR / "optimization_cycles.jsonl"
        self.mqtt_commands_file = DATA_DIR / "mqtt_commands.jsonl"
        self.system_states_file = DATA_DIR / "system_states.jsonl"
        self.metrics_file = DATA_DIR / "service_metrics.json"

    def add_cycle_data(self, data: OptimizationCycleData):
        """Add optimization cycle data."""
        self.cycle_data.append(data)

        # Append to file (JSONL format for streaming)
        with open(self.cycle_data_file, "a") as f:
            f.write(json.dumps(asdict(data)) + "\n")

        # Keep memory usage reasonable
        if len(self.cycle_data) > 1000:
            self.cycle_data = self.cycle_data[-500:]  # Keep last 500

    def add_mqtt_command(self, command: Dict):
        """Add MQTT command data."""
        self.mqtt_commands.append(command)

        with open(self.mqtt_commands_file, "a") as f:
            f.write(json.dumps(command) + "\n")

        if len(self.mqtt_commands) > 1000:
            self.mqtt_commands = self.mqtt_commands[-500:]

    def add_system_state(self, state: Dict):
        """Add system state data."""
        # Convert datetime objects to strings for JSON serialization
        serializable_state = self._make_serializable(state)
        self.system_states.append(serializable_state)

        with open(self.system_states_file, "a") as f:
            f.write(json.dumps(serializable_state) + "\n")

        if len(self.system_states) > 1000:
            self.system_states = self.system_states[-500:]

    def add_error(self, error_type: str, error_message: str, context: Dict = None):
        """Add error information."""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context or {},
        }
        self.error_log.append(error_data)

        error_file = DATA_DIR / "errors.jsonl"
        with open(error_file, "a") as f:
            f.write(json.dumps(error_data) + "\n")

    def get_service_metrics(self, start_time: datetime) -> ServiceMetrics:
        """Calculate service metrics."""
        uptime = (datetime.now() - start_time).total_seconds() / 3600

        successful = sum(1 for cycle in self.cycle_data if cycle.success)
        total = len(self.cycle_data)

        avg_solve_time = (
            np.mean([cycle.solve_time_seconds for cycle in self.cycle_data])
            if self.cycle_data
            else 0
        )
        avg_cycle_time = (
            np.mean([cycle.total_cycle_time_ms for cycle in self.cycle_data])
            if self.cycle_data
            else 0
        )

        # Count errors in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        errors_last_hour = sum(
            1
            for error in self.error_log
            if datetime.fromisoformat(error["timestamp"]) > one_hour_ago
        )

        # Mock system resource usage
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        return ServiceMetrics(
            start_time=start_time.isoformat(),
            uptime_hours=uptime,
            total_cycles=total,
            successful_cycles=successful,
            failed_cycles=total - successful,
            success_rate=successful / total if total > 0 else 0,
            avg_solve_time_seconds=avg_solve_time,
            avg_cycle_time_ms=avg_cycle_time,
            last_cycle_time=self.cycle_data[-1].timestamp if self.cycle_data else "",
            errors_last_hour=errors_last_hour,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
        )

    def save_metrics(self, metrics: ServiceMetrics):
        """Save service metrics to file."""
        with open(self.metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


class PEMSDryRunService:
    """Main PEMS v2 continuous dry run service."""

    def __init__(self):
        self.logger = logging.getLogger("PEMSDryRunService")
        self.start_time = datetime.now()
        self.cycle_count = 0
        self.running = False

        # Configuration from environment
        self.cycle_interval_minutes = int(
            os.getenv("PEMS_CYCLE_INTERVAL_MINUTES", "15")
        )
        self.optimization_horizon_hours = int(os.getenv("PEMS_HORIZON_HOURS", "6"))
        self.max_solve_time_seconds = int(os.getenv("PEMS_MAX_SOLVE_TIME", "30"))
        self.data_retention_hours = int(
            os.getenv("PEMS_DATA_RETENTION_HOURS", "168")
        )  # 1 week

        # Initialize components
        self.data_collector = DataCollector()
        self.mqtt_client = MockMQTTClient(self.data_collector)
        self.influxdb_client = MockInfluxDBClient(self.data_collector)

        # Import PEMS components
        self._import_pems_components()

        self.logger.info(f"PEMS Dry Run Service initialized")
        self.logger.info(f"Cycle interval: {self.cycle_interval_minutes} minutes")
        self.logger.info(
            f"Optimization horizon: {self.optimization_horizon_hours} hours"
        )
        self.logger.info(f"Data directory: {DATA_DIR}")
        self.logger.info(f"Log directory: {LOG_DIR}")

    def _import_pems_components(self):
        """Import PEMS components with graceful fallback."""
        try:
            from modules.control.growatt_controller import GrowattController
            from modules.optimization.optimizer import (
                EnergyOptimizer, create_optimization_problem)

            self.EnergyOptimizer = EnergyOptimizer
            self.create_optimization_problem = create_optimization_problem
            self.GrowattController = GrowattController

            self.logger.info("✅ PEMS v2 components imported successfully")

        except ImportError as e:
            self.logger.error(f"❌ Failed to import PEMS components: {e}")
            # Use mock components
            self._create_mock_components()

    def _create_mock_components(self):
        """Create mock components for testing when PEMS is unavailable."""
        self.logger.warning("⚠️ Using mock PEMS components")

        class MockOptimizer:
            def __init__(self, config):
                self.config = config

            def optimize(self, problem):
                # Simulate optimization delay
                import time

                time.sleep(np.random.uniform(0.5, 2.0))

                # Create mock result
                from dataclasses import dataclass

                @dataclass
                class MockResult:
                    success: bool = True
                    objective_value: float = 100.0 + np.random.normal(0, 20)
                    solve_time_seconds: float = np.random.uniform(0.5, 2.0)
                    message: str = "Mock optimization completed"
                    heating_schedule: dict = None
                    battery_first_schedule: object = None
                    ac_charge_schedule: object = None
                    export_schedule: object = None
                    cost_breakdown: dict = None

                # Create mock schedules
                import pandas as pd

                n_steps = 24
                time_index = pd.date_range(
                    start=datetime.now(), periods=n_steps, freq="15min"
                )

                return MockResult(
                    heating_schedule={
                        "obyvak": pd.Series([1, 0] * 12, index=time_index)
                    },
                    battery_first_schedule=pd.Series([1] * n_steps, index=time_index),
                    ac_charge_schedule=pd.Series([0] * n_steps, index=time_index),
                    export_schedule=pd.Series([0] * n_steps, index=time_index),
                    cost_breakdown={"total_cost": 150.0},
                )

        self.EnergyOptimizer = MockOptimizer

        def create_optimization_problem(*args, **kwargs):
            return None

        self.create_optimization_problem = create_optimization_problem

    async def run_service(self):
        """Run the continuous dry run service."""
        self.running = True
        self.logger.info("🚀 Starting PEMS v2 Continuous Dry Run Service")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            while self.running:
                cycle_start = datetime.now()
                self.cycle_count += 1

                self.logger.info(f"🔄 Starting optimization cycle #{self.cycle_count}")

                try:
                    # Run optimization cycle
                    cycle_data = await self._run_optimization_cycle()
                    cycle_data.cycle_id = self.cycle_count

                    # Store cycle data
                    self.data_collector.add_cycle_data(cycle_data)

                    # Log cycle summary
                    status = "✅ SUCCESS" if cycle_data.success else "❌ FAILED"
                    self.logger.info(
                        f"  {status} Cycle #{self.cycle_count}: "
                        f"solve={cycle_data.solve_time_seconds:.1f}s, "
                        f"total={cycle_data.total_cycle_time_ms:.0f}ms, "
                        f"obj={cycle_data.objective_value:.1f}"
                    )

                except Exception as e:
                    self.logger.error(f"  ❌ Cycle #{self.cycle_count} failed: {e}")
                    self.data_collector.add_error(
                        "optimization_cycle",
                        str(e),
                        {
                            "cycle_id": self.cycle_count,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                # Update and save service metrics
                if self.cycle_count % 10 == 0:  # Every 10 cycles
                    metrics = self.data_collector.get_service_metrics(self.start_time)
                    self.data_collector.save_metrics(metrics)

                    self.logger.info(
                        f"📊 Service metrics: "
                        f"uptime={metrics.uptime_hours:.1f}h, "
                        f"success_rate={metrics.success_rate:.1%}, "
                        f"avg_solve={metrics.avg_solve_time_seconds:.1f}s"
                    )

                # Data cleanup
                if self.cycle_count % 100 == 0:  # Every 100 cycles
                    await self._cleanup_old_data()

                # Wait for next cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.cycle_interval_minutes * 60 - cycle_duration)

                if sleep_time > 0:
                    self.logger.debug(f"⏸️ Waiting {sleep_time:.1f}s until next cycle")
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(
                        f"⚠️ Cycle took longer than interval ({cycle_duration:.1f}s)"
                    )

        except asyncio.CancelledError:
            self.logger.info("Service cancelled")
        except Exception as e:
            self.logger.error(f"Service error: {e}", exc_info=True)
        finally:
            await self._cleanup_and_shutdown()

    async def _run_optimization_cycle(self) -> OptimizationCycleData:
        """Run a single optimization cycle and collect data."""
        cycle_start = datetime.now()

        # 1. Get current system state
        state_start = datetime.now()
        current_state = await self.influxdb_client.query("SELECT * FROM system_state")
        state_time = (datetime.now() - state_start).total_seconds() * 1000

        # 2. Generate predictions
        pred_start = datetime.now()
        predictions = self._generate_realistic_predictions(current_state)
        pred_time = (datetime.now() - pred_start).total_seconds() * 1000

        # 3. Run optimization
        opt_start = datetime.now()

        try:
            # Create optimization problem
            problem = self.create_optimization_problem(
                start_time=datetime.now(),
                horizon_hours=self.optimization_horizon_hours,
                pv_forecast=predictions["pv_forecast"],
                load_forecast=predictions["load_forecast"],
                price_forecast=predictions["price_forecast"],
                weather_forecast=predictions["weather_forecast"],
                initial_battery_soc=current_state["battery_soc"],
                initial_temperatures=current_state["thermal"]["room_temperatures"],
            )

            # Run optimizer
            optimizer_config = {
                "rooms": {"obyvak": 2.0, "kuchyne": 1.5, "loznice": 1.8},
                "solver_timeout": self.max_solve_time_seconds,
                "mip_gap": 0.05,
            }

            optimizer = self.EnergyOptimizer(optimizer_config)
            result = optimizer.optimize(problem)

            opt_time = (datetime.now() - opt_start).total_seconds() * 1000

            # 4. Test control execution
            control_start = datetime.now()

            growatt_controller = self.GrowattController(
                mqtt_client=self.mqtt_client, settings={"simulation_mode": True}
            )
            await growatt_controller.initialize()

            # Execute Growatt commands
            if hasattr(result, "battery_first_schedule"):
                growatt_results = (
                    await growatt_controller.execute_optimization_schedule(
                        result.battery_first_schedule,
                        result.ac_charge_schedule,
                        result.export_schedule,
                    )
                )
            else:
                growatt_results = {
                    "battery_first": True,
                    "ac_charge": True,
                    "export": True,
                }

            control_time = (datetime.now() - control_start).total_seconds() * 1000

            # Create cycle data
            total_time = (datetime.now() - cycle_start).total_seconds() * 1000

            return OptimizationCycleData(
                timestamp=cycle_start.isoformat(),
                cycle_id=0,  # Will be set by caller
                success=result.success,
                solve_time_seconds=result.solve_time_seconds,
                objective_value=result.objective_value,
                message=result.message,
                # System state
                battery_soc=current_state["battery_soc"],
                outdoor_temp=current_state["thermal"]["outdoor_temp"],
                room_temperatures=current_state["thermal"]["room_temperatures"],
                current_load=current_state["current_load"],
                pv_power=current_state["pv_power"],
                # Predictions
                pv_forecast_peak=predictions["pv_forecast"].max(),
                load_forecast_avg=predictions["load_forecast"].mean(),
                price_forecast_min=predictions["price_forecast"].min(),
                price_forecast_max=predictions["price_forecast"].max(),
                price_forecast_avg=predictions["price_forecast"].mean(),
                # Control decisions
                heating_decisions=self._summarize_heating_decisions(
                    result.heating_schedule
                )
                if hasattr(result, "heating_schedule")
                else {},
                growatt_decisions=self._summarize_growatt_decisions(result)
                if hasattr(result, "battery_first_schedule")
                else {},
                # Performance
                total_cycle_time_ms=total_time,
                prediction_time_ms=pred_time,
                optimization_time_ms=opt_time,
                control_time_ms=control_time,
            )

        except Exception as e:
            opt_time = (datetime.now() - opt_start).total_seconds() * 1000
            total_time = (datetime.now() - cycle_start).total_seconds() * 1000

            return OptimizationCycleData(
                timestamp=cycle_start.isoformat(),
                cycle_id=0,
                success=False,
                solve_time_seconds=0,
                objective_value=float("inf"),
                message=f"Error: {str(e)}",
                # System state
                battery_soc=current_state.get("battery_soc", 0),
                outdoor_temp=current_state.get("thermal", {}).get("outdoor_temp", 0),
                room_temperatures=current_state.get("thermal", {}).get(
                    "room_temperatures", {}
                ),
                current_load=current_state.get("current_load", 0),
                pv_power=current_state.get("pv_power", 0),
                # Predictions (empty)
                pv_forecast_peak=0,
                load_forecast_avg=0,
                price_forecast_min=0,
                price_forecast_max=0,
                price_forecast_avg=0,
                # Control decisions (empty)
                heating_decisions={},
                growatt_decisions={},
                # Performance
                total_cycle_time_ms=total_time,
                prediction_time_ms=pred_time,
                optimization_time_ms=opt_time,
                control_time_ms=0,
                # Error info
                error_type=type(e).__name__,
                error_message=str(e),
            )

    def _generate_realistic_predictions(self, current_state: Dict) -> Dict:
        """Generate realistic predictions that evolve over time."""
        n_steps = self.optimization_horizon_hours * 4
        time_index = pd.date_range(start=datetime.now(), periods=n_steps, freq="15min")

        # Get current hour for realistic patterns
        current_hour = datetime.now().hour
        hours = np.array([(current_hour + i * 0.25) % 24 for i in range(n_steps)])

        # PV forecast (winter pattern with weather variation)
        pv_base = np.maximum(0, 1500 * np.sin(np.pi * (hours - 8) / 10))
        weather_factor = np.random.uniform(0.3, 1.0)  # Cloud variation
        pv_forecast = pd.Series(pv_base * weather_factor, index=time_index)

        # Load forecast (realistic household pattern)
        load_base = np.where(
            (hours >= 6) & (hours <= 9),
            2500,  # Morning peak
            np.where(
                (hours >= 17) & (hours <= 21),
                3200,  # Evening peak
                np.where((hours >= 22) | (hours <= 6), 1200, 1800),  # Night  # Day
            ),
        )
        load_noise = np.random.normal(0, 200, n_steps)  # Random variation
        load_forecast = pd.Series(load_base + load_noise, index=time_index)

        # Price forecast (Czech electricity market pattern)
        base_prices = np.where(
            (hours >= 6) & (hours <= 22), 4.2, 2.8  # Day rate  # Night rate
        )
        price_variation = np.random.normal(0, 0.8, n_steps)  # Market volatility
        price_forecast = pd.Series(
            np.maximum(1.0, base_prices + price_variation), index=time_index
        )

        # Weather forecast
        outdoor_temp = current_state["thermal"]["outdoor_temp"]
        temp_variation = np.random.normal(0, 1, n_steps).cumsum() * 0.1
        weather_forecast = pd.DataFrame(
            {
                "temperature_2m": outdoor_temp + temp_variation,
                "cloudcover": np.random.uniform(30, 90, n_steps),
                "humidity": np.random.uniform(60, 95, n_steps),
            },
            index=time_index,
        )

        return {
            "pv_forecast": pv_forecast,
            "load_forecast": load_forecast,
            "price_forecast": price_forecast,
            "weather_forecast": weather_forecast,
        }

    def _summarize_heating_decisions(self, heating_schedule) -> Dict:
        """Summarize heating schedule decisions."""
        if not heating_schedule:
            return {}

        summary = {}
        for room, schedule in heating_schedule.items():
            if hasattr(schedule, "sum") and len(schedule) > 0:
                on_periods = schedule.sum()
                total_periods = len(schedule)
                summary[room] = {
                    "current_state": bool(schedule.iloc[0]),
                    "on_periods": int(on_periods),
                    "total_periods": total_periods,
                    "duty_cycle": float(on_periods / total_periods)
                    if total_periods > 0
                    else 0,
                }
        return summary

    def _summarize_growatt_decisions(self, result) -> Dict:
        """Summarize Growatt control decisions."""
        summary = {}

        for mode_name in [
            "battery_first_schedule",
            "ac_charge_schedule",
            "export_schedule",
        ]:
            if hasattr(result, mode_name):
                schedule = getattr(result, mode_name)
                if hasattr(schedule, "sum") and len(schedule) > 0:
                    on_periods = schedule.sum()
                    total_periods = len(schedule)
                    summary[mode_name] = {
                        "current_state": bool(schedule.iloc[0]),
                        "on_periods": int(on_periods),
                        "total_periods": total_periods,
                        "duty_cycle": float(on_periods / total_periods)
                        if total_periods > 0
                        else 0,
                    }

        return summary

    async def _cleanup_old_data(self):
        """Clean up old data files to manage disk usage."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)

            # This is a simplified cleanup - in production you'd implement
            # proper data rotation and archival
            self.logger.info(
                f"🧹 Data cleanup: retaining last {self.data_retention_hours} hours"
            )

        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")

    async def _cleanup_and_shutdown(self):
        """Cleanup and shutdown service gracefully."""
        self.logger.info("🛑 Shutting down PEMS Dry Run Service")

        # Save final metrics
        final_metrics = self.data_collector.get_service_metrics(self.start_time)
        self.data_collector.save_metrics(final_metrics)

        self.logger.info(f"📊 Final service metrics:")
        self.logger.info(f"  Total cycles: {final_metrics.total_cycles}")
        self.logger.info(f"  Success rate: {final_metrics.success_rate:.1%}")
        self.logger.info(f"  Uptime: {final_metrics.uptime_hours:.1f} hours")
        self.logger.info(
            f"  Avg solve time: {final_metrics.avg_solve_time_seconds:.1f}s"
        )

        self.logger.info("✅ Service shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False


def setup_logging():
    """Setup structured logging with rotation."""

    # Configure log format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    from logging.handlers import RotatingFileHandler

    log_file = LOG_DIR / "pems_dry_run.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cvxpy").setLevel(logging.WARNING)


async def main():
    """Main entry point for the service."""
    setup_logging()

    logger = logging.getLogger("main")
    logger.info("Starting PEMS v2 Continuous Dry Run Service")

    service = PEMSDryRunService()
    await service.run_service()


if __name__ == "__main__":
    asyncio.run(main())
