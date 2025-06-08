"""
Configuration settings for PEMS v2 (Personal Energy Management System).

This module provides the complete configuration infrastructure for the PEMS v2 system,
using Pydantic for validation and type safety. All settings can be configured via
environment variables with appropriate prefixes.

Key Features:
- Environment variable configuration with .env file support
- Type validation and automatic conversion
- Secure secret handling for tokens and passwords
- Modular settings organization by subsystem
- Default values for all optional settings
- Configuration inheritance and composition

Architecture:
- Each subsystem has its own settings class for clear separation
- PEMSSettings acts as the main configuration container
- All settings classes inherit from BaseSettings for env var support
- Secret values use SecretStr for secure handling
- Nested configuration allows for complex system setup

Usage:
    settings = PEMSSettings()  # Loads from environment and .env file
    db_url = settings.influxdb.url
    mqtt_broker = settings.mqtt.broker
"""

from typing import Dict, List, Optional

from pydantic import Field
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfluxDBSettings(BaseSettings):
    """
    InfluxDB time-series database configuration.
    
    InfluxDB serves as the primary data store for all PEMS v2 time-series data
    including sensor readings, predictions, and optimization results. The system
    uses multiple buckets to organize different data types for efficient querying
    and retention policies.
    
    Environment Variables:
        INFLUXDB_URL: Database server URL (default: http://localhost:8086)
        INFLUXDB_TOKEN: Authentication token (required)
        INFLUXDB_ORG: Organization name (default: loxone)
        INFLUXDB_BUCKET_*: Various bucket names for data organization
    
    Buckets:
        - loxone: Historical sensor data from Loxone system
        - weather_forecast: Weather forecast and actual weather data
        - predictions: ML model predictions (PV, load, thermal)
        - optimization: Optimization results and control decisions
    """

    url: str = Field(default="http://localhost:8086", description="InfluxDB server URL")
    token: SecretStr = Field(..., description="Authentication token for InfluxDB access")
    org: str = Field(default="loxone", description="InfluxDB organization name")
    
    # Data bucket configuration for different data types
    bucket_historical: str = Field(default="loxone", description="Historical sensor data bucket")
    bucket_loxone: str = Field(default="loxone", description="Live Loxone system data bucket")
    bucket_weather: str = Field(default="weather_forecast", description="Weather forecast data bucket")
    bucket_solar: str = Field(default="loxone", description="Solar/PV production data bucket")
    bucket_predictions: str = Field(default="predictions", description="ML model predictions bucket")
    bucket_optimization: str = Field(default="optimization", description="Optimization results bucket")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INFLUXDB_", extra="ignore"
    )


class MQTTSettings(BaseSettings):
    """
    MQTT message broker configuration.
    
    MQTT is used for real-time communication between PEMS components and the
    Loxone system. The system publishes optimization decisions and subscribes
    to live sensor data updates for immediate response to changing conditions.
    
    Environment Variables:
        MQTT_BROKER: Broker hostname (default: localhost)
        MQTT_PORT: Broker port (default: 1883)
        MQTT_USERNAME: Authentication username (optional)
        MQTT_PASSWORD: Authentication password (optional)
        MQTT_CLIENT_ID: Unique client identifier (default: pems_v2)
    """

    broker: str = Field(default="localhost", description="MQTT broker hostname")
    port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(default=None, description="MQTT authentication username")
    password: Optional[SecretStr] = Field(default=None, description="MQTT authentication password")
    client_id: str = Field(default="pems_v2", description="Unique MQTT client identifier")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MQTT_", extra="ignore"
    )


class PVPredictionSettings(BaseSettings):
    """
    Photovoltaic production prediction configuration.
    
    Configures the ML model for predicting solar PV energy production based on
    weather forecasts. The predictor uses historical production data combined
    with weather parameters to forecast energy generation for optimization.
    
    Environment Variables:
        PV_MODEL_PATH: Path to trained model file (default: models/pv_predictor.pkl)
        PV_UPDATE_INTERVAL: Prediction update frequency in seconds (default: 3600)
        PV_HORIZON_HOURS: Prediction horizon in hours (default: 48)
        PV_CONFIDENCE_LEVELS: Confidence intervals for uncertainty quantification
    """

    model_path: str = Field(default="models/pv_predictor.pkl", description="Path to trained PV prediction model")
    update_interval: int = Field(default=3600, description="Prediction update interval in seconds")
    horizon_hours: int = Field(default=48, description="Prediction horizon in hours")
    confidence_levels: List[float] = Field(
        default=[0.1, 0.5, 0.9], 
        description="Confidence levels for uncertainty quantification"
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PV_", extra="ignore")


class ThermalSettings(BaseSettings):
    """
    Thermal comfort and heating system configuration.
    
    Manages thermal comfort parameters and room-specific heating settings.
    The thermal model predicts heating requirements based on outdoor conditions,
    thermal mass, and occupancy patterns to maintain comfort while minimizing
    energy consumption.
    
    Environment Variables:
        THERMAL_DEFAULT_SETPOINT_DAY: Daytime temperature setpoint in °C (default: 21.0)
        THERMAL_DEFAULT_SETPOINT_NIGHT: Nighttime temperature setpoint in °C (default: 19.0)
        THERMAL_COMFORT_BAND: Acceptable temperature deviation in ±°C (default: 0.5)
    """

    rooms: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Room-specific thermal parameters"
    )
    default_setpoint_day: float = Field(
        default=21.0, 
        description="Default daytime temperature setpoint in °C"
    )
    default_setpoint_night: float = Field(
        default=19.0, 
        description="Default nighttime temperature setpoint in °C"
    )
    comfort_band: float = Field(
        default=0.5, 
        description="Acceptable temperature deviation in ±°C"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="THERMAL_", extra="ignore"
    )


class OptimizationSettings(BaseSettings):
    """
    Optimization engine configuration for energy management.
    
    Controls the mathematical optimization solver that determines optimal
    energy allocation, battery scheduling, and load shifting decisions.
    The optimizer balances multiple objectives including cost minimization,
    self-consumption maximization, and thermal comfort maintenance.
    
    Environment Variables:
        OPT_HORIZON_HOURS: Optimization horizon in hours (default: 48)
        OPT_CONTROL_HOURS: Control horizon in hours (default: 24)
        OPT_TIME_STEP_MINUTES: Time discretization in minutes (default: 60)
        OPT_SOLVER: Mathematical solver to use (default: APOPT)
        OPT_MAX_SOLVE_TIME: Maximum solver time in seconds (default: 30)
        OPT_*_WEIGHT: Objective function weights for multi-objective optimization
    """

    horizon_hours: int = Field(default=48, description="Optimization horizon in hours")
    control_hours: int = Field(default=24, description="Control horizon in hours")
    time_step_minutes: int = Field(default=60, description="Time discretization in minutes")
    solver: str = Field(default="APOPT", description="Mathematical solver (APOPT, IPOPT, etc.)")
    max_solve_time: int = Field(default=30, description="Maximum solver time in seconds")

    # Multi-objective optimization weights
    cost_weight: float = Field(default=1.0, description="Energy cost minimization weight")
    self_consumption_weight: float = Field(default=0.3, description="PV self-consumption maximization weight")
    peak_shaving_weight: float = Field(default=0.1, description="Peak demand reduction weight")
    comfort_weight: float = Field(default=0.5, description="Thermal comfort maintenance weight")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OPT_", extra="ignore"
    )


class BatterySettings(BaseSettings):
    """
    Battery energy storage system configuration.
    
    Defines the technical specifications and operational constraints of the
    battery system. These parameters are critical for accurate optimization
    modeling and safe battery operation within manufacturer limits.
    
    Environment Variables:
        BATTERY_CAPACITY_KWH: Total battery capacity in kWh (default: 10.0)
        BATTERY_MAX_POWER_KW: Maximum charge/discharge power in kW (default: 5.0)
        BATTERY_EFFICIENCY: Round-trip efficiency (0.0-1.0) (default: 0.95)
        BATTERY_MIN_SOC: Minimum state of charge (0.0-1.0) (default: 0.1)
        BATTERY_MAX_SOC: Maximum state of charge (0.0-1.0) (default: 0.9)
    """

    capacity_kwh: float = Field(default=10.0, description="Total battery capacity in kWh")
    max_power_kw: float = Field(default=5.0, description="Maximum charge/discharge power in kW")
    efficiency: float = Field(default=0.95, description="Round-trip efficiency (0.0-1.0)")
    min_soc: float = Field(default=0.1, description="Minimum state of charge (0.0-1.0)")
    max_soc: float = Field(default=0.9, description="Maximum state of charge (0.0-1.0)")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="BATTERY_", extra="ignore"
    )


class EVSettings(BaseSettings):
    """
    Electric vehicle charging system configuration.
    
    Manages EV charging parameters for smart charging optimization.
    The system schedules EV charging during low-cost or high-solar periods
    while ensuring the vehicle is ready by departure time.
    
    Environment Variables:
        EV_MAX_POWER_KW: Maximum charging power in kW (default: 11.0)
        EV_BATTERY_CAPACITY_KWH: EV battery capacity in kWh (default: 60.0)
        EV_DEFAULT_TARGET_SOC: Default target charge level (0.0-1.0) (default: 0.8)
        EV_DEPARTURE_TIME: Default departure time in HH:MM format (default: 07:00)
    """

    max_power_kw: float = Field(default=11.0, description="Maximum EV charging power in kW")
    battery_capacity_kwh: float = Field(default=60.0, description="EV battery capacity in kWh")
    default_target_soc: float = Field(default=0.8, description="Default target charge level (0.0-1.0)")
    departure_time: str = Field(default="07:00", description="Default departure time in HH:MM format")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EV_", extra="ignore")


class PEMSSettings(BaseSettings):
    """
    Main PEMS v2 system configuration container.
    
    Aggregates all subsystem configurations into a single, coherent settings
    object. Provides global system-level settings and operating modes that
    affect the entire PEMS operation.
    
    Operating Modes:
        - simulation_mode: Run without actual hardware control (testing/development)
        - advisory_mode: Generate recommendations without automatic control
    
    Update Intervals:
        - optimization_interval: How often to run optimization (default: 1 hour)
        - control_interval: How often to update control signals (default: 5 minutes)
    
    Environment Variables:
        PEMS_SIMULATION_MODE: Enable simulation mode (default: False)
        PEMS_ADVISORY_MODE: Enable advisory-only mode (default: False)
        PEMS_OPTIMIZATION_INTERVAL: Optimization frequency in seconds (default: 3600)
        PEMS_CONTROL_INTERVAL: Control update frequency in seconds (default: 300)
    
    Usage:
        settings = PEMSSettings()
        if settings.simulation_mode:
            logger.info("Running in simulation mode")
        
        # Access subsystem settings
        db_token = settings.influxdb.token.get_secret_value()
        mqtt_broker = settings.mqtt.broker
    """

    # System-wide feature flags for different operating modes
    simulation_mode: bool = Field(
        default=False, 
        description="Run in simulation mode without hardware control"
    )
    advisory_mode: bool = Field(
        default=False, 
        description="Generate recommendations without automatic control"
    )

    # System-wide timing configuration
    optimization_interval: int = Field(
        default=3600, 
        description="Optimization run frequency in seconds"
    )
    control_interval: int = Field(
        default=300, 
        description="Control signal update frequency in seconds"
    )

    # Subsystem configurations - each subsystem manages its own settings
    influxdb: InfluxDBSettings = Field(
        default_factory=InfluxDBSettings,
        description="InfluxDB database configuration"
    )
    mqtt: MQTTSettings = Field(
        default_factory=MQTTSettings,
        description="MQTT messaging configuration"
    )
    pv_prediction: PVPredictionSettings = Field(
        default_factory=PVPredictionSettings,
        description="PV production prediction configuration"
    )
    thermal: ThermalSettings = Field(
        default_factory=ThermalSettings,
        description="Thermal comfort and heating configuration"
    )
    optimization: OptimizationSettings = Field(
        default_factory=OptimizationSettings,
        description="Optimization engine configuration"
    )
    battery: BatterySettings = Field(
        default_factory=BatterySettings,
        description="Battery storage system configuration"
    )
    ev: EVSettings = Field(
        default_factory=EVSettings,
        description="Electric vehicle charging configuration"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="PEMS_", extra="ignore"
    )
