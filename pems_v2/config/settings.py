"""
Configuration settings for PEMS v2 (Personal Energy Management System).

This module provides a tiered configuration system that combines JSON-based
system configuration with environment variable overrides for secrets and
deployment-specific settings.

Architecture:
- system_config.json: Non-sensitive system configuration (models, thermal settings, etc.)
- Environment variables: Secrets, server URLs, and deployment-specific overrides
- Pydantic validation: Type safety and automatic validation for all settings

Key Features:
- JSON-based configuration for maintainability
- Environment variable overrides for secrets and deployment settings
- Type validation and automatic conversion
- Secure secret handling for tokens and passwords
- Per-room thermal settings and power ratings
- Model-specific configurations centralized in JSON

Usage:
    settings = PEMSSettings()  # Loads from JSON + environment variables
    db_url = settings.influxdb.url
    mqtt_broker = settings.mqtt.broker
    room_setpoint = settings.thermal_settings.room_setpoints["obyvak"]["day"]
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfluxDBSettings(BaseSettings):
    """
    InfluxDB time-series database configuration.

    Combines JSON configuration with environment variable overrides
    for connection details and authentication.
    """

    url: str = Field(default="http://localhost:8086", description="InfluxDB server URL")
    token: SecretStr = Field(
        ..., description="Authentication token for InfluxDB access"
    )
    org: str = Field(default="loxone", description="InfluxDB organization name")

    # Data bucket configuration for different data types
    bucket_historical: str = Field(
        default="loxone", description="Historical sensor data bucket"
    )
    bucket_loxone: str = Field(
        default="loxone", description="Live Loxone system data bucket"
    )
    bucket_weather: str = Field(
        default="weather_forecast", description="Weather forecast data bucket"
    )
    bucket_solar: str = Field(
        default="loxone", description="Solar/PV production data bucket"
    )
    bucket_predictions: str = Field(
        default="predictions", description="ML model predictions bucket"
    )
    bucket_optimization: str = Field(
        default="optimization", description="Optimization results bucket"
    )
    bucket_prices: str = Field(
        default="ote_prices", description="Energy price data bucket"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INFLUXDB_", extra="ignore"
    )


class MQTTSettings(BaseSettings):
    """
    MQTT message broker configuration.

    Connection details loaded from environment variables for security.
    """

    broker: str = Field(default="localhost", description="MQTT broker hostname")
    port: int = Field(default=1883, description="MQTT broker port")
    username: Optional[str] = Field(
        default=None, description="MQTT authentication username"
    )
    password: Optional[SecretStr] = Field(
        default=None, description="MQTT authentication password"
    )
    client_id: str = Field(
        default="pems_v2", description="Unique MQTT client identifier"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MQTT_", extra="ignore"
    )

    @model_validator(mode="after")
    def validate_mqtt_parameters(self):
        """Validate MQTT connection parameters."""
        if not (1 <= self.port <= 65535):
            raise ValueError(f"MQTT port ({self.port}) must be between 1 and 65535.")

        if not self.broker or not self.broker.strip():
            raise ValueError("MQTT broker hostname cannot be empty.")

        if not self.client_id or not self.client_id.strip():
            raise ValueError("MQTT client ID cannot be empty.")

        return self


class SystemSettings(BaseModel):
    """System-wide operational settings loaded from JSON configuration."""

    simulation_mode: bool = Field(default=False, description="Run in simulation mode")
    advisory_mode: bool = Field(
        default=False, description="Generate recommendations only"
    )
    optimization_interval_seconds: int = Field(
        default=3600, description="Optimization frequency"
    )
    control_interval_seconds: int = Field(
        default=300, description="Control update frequency"
    )

    @model_validator(mode="after")
    def validate_intervals(self):
        """Validate system timing intervals."""
        if self.control_interval_seconds <= 0:
            raise ValueError(
                f"Control interval ({self.control_interval_seconds}s) must be positive."
            )

        if self.optimization_interval_seconds <= 0:
            raise ValueError(
                f"Optimization interval ({self.optimization_interval_seconds}s) must be positive."
            )

        if self.control_interval_seconds > self.optimization_interval_seconds:
            raise ValueError(
                f"Control interval ({self.control_interval_seconds}s) cannot be longer than "
                f"optimization interval ({self.optimization_interval_seconds}s)."
            )

        if self.control_interval_seconds < 60:
            raise ValueError(
                f"Control interval ({self.control_interval_seconds}s) should be at least 60 seconds."
            )

        return self


class PVModelSettings(BaseModel):
    """PV prediction model configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(description="Path to trained PV prediction model")
    update_interval_seconds: int = Field(description="Prediction update interval")
    horizon_hours: int = Field(description="Prediction horizon in hours")
    confidence_levels: List[float] = Field(
        description="Confidence levels for uncertainty quantification"
    )

    @model_validator(mode="after")
    def validate_prediction_parameters(self):
        """Validate PV prediction parameters."""
        if self.update_interval_seconds < 300:
            raise ValueError(
                f"PV update interval ({self.update_interval_seconds}s) must be at least 300 seconds."
            )

        if not (1 <= self.horizon_hours <= 168):
            raise ValueError(
                f"PV prediction horizon ({self.horizon_hours}h) must be between 1 and 168 hours."
            )

        for level in self.confidence_levels:
            if not (0.0 < level < 1.0):
                raise ValueError(
                    f"Confidence level ({level}) must be between 0.0 and 1.0."
                )

        if self.confidence_levels != sorted(self.confidence_levels):
            raise ValueError(
                f"Confidence levels {self.confidence_levels} must be in ascending order."
            )

        return self


class LoadModelSettings(BaseModel):
    """Load prediction model configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(description="Path to trained load prediction model")
    horizon_hours: int = Field(description="Prediction horizon in hours")


class ThermalModelSettings(BaseModel):
    """Thermal prediction model configuration."""

    model_config = ConfigDict(protected_namespaces=())

    model_path: str = Field(description="Path to trained thermal prediction model")


class ModelSettings(BaseModel):
    """Combined ML model configurations."""

    pv: PVModelSettings
    load: LoadModelSettings
    thermal: ThermalModelSettings


class RoomSetpoint(BaseModel):
    """Day/night temperature setpoints for a room."""

    day: float = Field(description="Daytime temperature setpoint in °C")
    night: float = Field(description="Nighttime temperature setpoint in °C")

    @model_validator(mode="after")
    def validate_setpoints(self):
        """Validate temperature setpoints are reasonable."""
        if not (10.0 <= self.night <= 30.0):
            raise ValueError(
                f"Night setpoint ({self.night}°C) must be between 10°C and 30°C."
            )

        if not (15.0 <= self.day <= 35.0):
            raise ValueError(
                f"Day setpoint ({self.day}°C) must be between 15°C and 35°C."
            )

        if self.night > self.day:
            raise ValueError(
                f"Night setpoint ({self.night}°C) cannot be higher than day setpoint ({self.day}°C)."
            )

        return self


class ThermalSettings(BaseModel):
    """Thermal comfort and room-specific configuration."""

    comfort_band_celsius: float = Field(
        description="Acceptable temperature deviation in ±°C"
    )
    room_setpoints: Dict[str, RoomSetpoint] = Field(
        description="Per-room temperature setpoints"
    )

    @model_validator(mode="after")
    def validate_comfort_band(self):
        """Validate comfort band is reasonable."""
        if not (0.1 <= self.comfort_band_celsius <= 5.0):
            raise ValueError(
                f"Comfort band ({self.comfort_band_celsius}°C) must be between 0.1°C and 5.0°C."
            )
        return self

    def get_target_temp(self, room_name: str, hour: int) -> float:
        """Get target temperature for a room at a specific hour."""
        setpoints = self.room_setpoints.get(room_name, self.room_setpoints["default"])

        if 6 <= hour < 22:  # Daytime
            return setpoints.day
        else:  # Nighttime
            return setpoints.night


class OptimizationWeights(BaseModel):
    """Multi-objective optimization weights."""

    cost: float = Field(description="Energy cost minimization weight")
    self_consumption: float = Field(
        description="PV self-consumption maximization weight"
    )
    peak_shaving: float = Field(description="Peak demand reduction weight")
    comfort: float = Field(description="Thermal comfort maintenance weight")

    @model_validator(mode="after")
    def validate_weights(self):
        """Validate that optimization weights are non-negative."""
        weights = {
            "cost": self.cost,
            "self_consumption": self.self_consumption,
            "peak_shaving": self.peak_shaving,
            "comfort": self.comfort,
        }

        for weight_name, weight_value in weights.items():
            if weight_value < 0:
                raise ValueError(
                    f"{weight_name} ({weight_value}) must be non-negative."
                )

        if all(weight <= 0 for weight in weights.values()):
            raise ValueError("At least one optimization weight must be positive.")

        return self


class OptimizationSettings(BaseModel):
    """Optimization engine configuration."""

    horizon_hours: int = Field(description="Optimization horizon in hours")
    control_hours: int = Field(description="Control horizon in hours")
    time_step_minutes: int = Field(description="Time discretization in minutes")
    solver: str = Field(description="Mathematical solver (APOPT, IPOPT, etc.)")
    max_solve_time_seconds: int = Field(description="Maximum solver time in seconds")
    weights: OptimizationWeights = Field(
        description="Multi-objective optimization weights"
    )

    @model_validator(mode="after")
    def validate_horizons(self):
        """Validate that control horizon is not greater than optimization horizon."""
        if self.control_hours > self.horizon_hours:
            raise ValueError(
                f"Control horizon ({self.control_hours}h) cannot be greater than "
                f"optimization horizon ({self.horizon_hours}h)."
            )
        return self

    @model_validator(mode="after")
    def validate_time_step(self):
        """Validate that time step is reasonable for optimization."""
        if not (5 <= self.time_step_minutes <= 120):
            raise ValueError(
                f"Time step ({self.time_step_minutes} minutes) must be between 5 and 120 minutes."
            )
        return self


class BatterySettings(BaseModel):
    """Battery energy storage system configuration."""

    capacity_kwh: float = Field(description="Total battery capacity in kWh")
    max_power_kw: float = Field(description="Maximum charge/discharge power in kW")
    efficiency: float = Field(description="Round-trip efficiency (0.0-1.0)")
    min_soc: float = Field(description="Minimum state of charge (0.0-1.0)")
    max_soc: float = Field(description="Maximum state of charge (0.0-1.0)")

    @model_validator(mode="after")
    def validate_soc_range(self):
        """Validate that SOC range is logical and within bounds."""
        if not (0.0 <= self.min_soc < self.max_soc <= 1.0):
            raise ValueError(
                f"Invalid SOC range: min_soc ({self.min_soc}) must be less than "
                f"max_soc ({self.max_soc}) and both must be between 0.0 and 1.0."
            )
        return self

    @model_validator(mode="after")
    def validate_efficiency(self):
        """Validate that battery efficiency is within realistic bounds."""
        if not (0.5 <= self.efficiency <= 1.0):
            raise ValueError(
                f"Battery efficiency ({self.efficiency}) must be between 0.5 and 1.0."
            )
        return self

    @model_validator(mode="after")
    def validate_capacity_and_power(self):
        """Validate that capacity and power specifications are reasonable."""
        if self.capacity_kwh <= 0:
            raise ValueError(
                f"Battery capacity ({self.capacity_kwh} kWh) must be positive."
            )

        if self.max_power_kw <= 0:
            raise ValueError(
                f"Battery max power ({self.max_power_kw} kW) must be positive."
            )

        c_rate = self.max_power_kw / self.capacity_kwh
        if c_rate > 5.0:
            raise ValueError(
                f"Battery C-rate ({c_rate:.1f}C) is too high. "
                f"Power ({self.max_power_kw}kW) should not exceed 5x capacity ({self.capacity_kwh}kWh)."
            )

        return self


class EVSettings(BaseModel):
    """Electric vehicle charging system configuration."""

    max_power_kw: float = Field(description="Maximum EV charging power in kW")
    battery_capacity_kwh: float = Field(description="EV battery capacity in kWh")
    default_target_soc: float = Field(
        description="Default target charge level (0.0-1.0)"
    )
    departure_time: str = Field(description="Default departure time in HH:MM format")

    @model_validator(mode="after")
    def validate_ev_parameters(self):
        """Validate EV charging parameters."""
        if self.max_power_kw <= 0:
            raise ValueError(f"EV max power ({self.max_power_kw} kW) must be positive.")

        if self.battery_capacity_kwh <= 0:
            raise ValueError(
                f"EV battery capacity ({self.battery_capacity_kwh} kWh) must be positive."
            )

        if not (0.0 < self.default_target_soc <= 1.0):
            raise ValueError(
                f"EV target SOC ({self.default_target_soc}) must be between 0.0 and 1.0."
            )

        return self

    @model_validator(mode="after")
    def validate_departure_time(self):
        """Validate departure time format."""
        try:
            from datetime import datetime

            datetime.strptime(self.departure_time, "%H:%M")
        except ValueError:
            raise ValueError(
                f"Invalid departure time format '{self.departure_time}'. "
                f"Expected HH:MM format (e.g., '07:30')."
            )
        return self


class SystemConfig(BaseModel):
    """Complete system configuration loaded from JSON."""

    system: SystemSettings
    models: ModelSettings
    thermal_settings: ThermalSettings
    optimization: OptimizationSettings
    battery: BatterySettings
    ev: EVSettings
    room_power_ratings_kw: Dict[str, float] = Field(
        description="Room power ratings in kW"
    )


class PEMSSettings(BaseSettings):
    """
    Main PEMS v2 system configuration container.

    Combines JSON-based system configuration with environment variable
    overrides for secrets and deployment-specific settings.

    Configuration Loading Order:
    1. Load system_config.json for non-sensitive settings
    2. Override with environment variables for secrets/deployment settings
    3. Apply validation to ensure consistency

    Usage:
        settings = PEMSSettings()
        if settings.system.simulation_mode:
            logger.info("Running in simulation mode")

        # Access subsystem settings
        db_token = settings.influxdb.token.get_secret_value()
        mqtt_broker = settings.mqtt.broker
        room_temp = settings.thermal_settings.get_target_temp("obyvak", 14)
    """

    # Environment-specific settings (loaded from env vars)
    influxdb: InfluxDBSettings = Field(default_factory=InfluxDBSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)

    # System configuration (loaded from JSON, can be overridden by env vars)
    system: SystemSettings = Field(default_factory=SystemSettings)
    models: ModelSettings = Field(default=None)
    thermal_settings: ThermalSettings = Field(default=None)
    optimization: OptimizationSettings = Field(default=None)
    battery: BatterySettings = Field(default=None)
    ev: EVSettings = Field(default=None)
    room_power_ratings_kw: Dict[str, float] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="PEMS_", extra="ignore"
    )

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize PEMS settings with JSON configuration and environment overrides.

        Args:
            config_path: Optional path to system_config.json (defaults to config/system_config.json)
            **kwargs: Additional keyword arguments passed to BaseSettings
        """
        # Load JSON configuration first
        if config_path is None:
            # Default path relative to this module
            config_dir = Path(__file__).parent
            config_path = config_dir / "system_config.json"

        # Allow override via environment variable
        config_path = os.getenv("PEMS_CONFIG_PATH", config_path)

        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"System configuration file not found: {config_path}"
            )

        # Load and parse JSON configuration
        with open(config_path, "r") as f:
            json_config = json.load(f)

        # Parse JSON into typed configuration objects
        system_config = SystemConfig(**json_config)

        # Merge JSON config with any provided kwargs
        merged_config = {
            "system": system_config.system,
            "models": system_config.models,
            "thermal_settings": system_config.thermal_settings,
            "optimization": system_config.optimization,
            "battery": system_config.battery,
            "ev": system_config.ev,
            "room_power_ratings_kw": system_config.room_power_ratings_kw,
            **kwargs,
        }

        # Initialize with merged configuration
        super().__init__(**merged_config)

    def get_room_power(self, room_name: str) -> float:
        """Get power rating for a specific room in kW."""
        return self.room_power_ratings_kw.get(room_name, 0.0)

    @model_validator(mode="after")
    def validate_operational_modes(self):
        """Validate operational mode combinations."""
        # Both simulation and advisory mode can be enabled simultaneously
        return self


# Legacy compatibility aliases for gradual migration
PVPredictionSettings = PVModelSettings
ThermalSettings = ThermalSettings  # Keep same name
OptimizationSettings = OptimizationSettings  # Keep same name
BatterySettings = BatterySettings  # Keep same name
EVSettings = EVSettings  # Keep same name
