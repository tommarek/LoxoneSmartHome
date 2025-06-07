"""Configuration settings for PEMS v2."""

from typing import Dict, List, Optional

from pydantic import Field
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfluxDBSettings(BaseSettings):
    """InfluxDB configuration."""

    url: str = Field(default="http://localhost:8086")
    token: SecretStr = Field(...)
    org: str = Field(default="loxone")
    bucket_historical: str = Field(default="loxone")
    bucket_loxone: str = Field(default="loxone")
    bucket_weather: str = Field(default="weather_forecast")
    bucket_solar: str = Field(default="loxone")
    bucket_predictions: str = Field(default="predictions")
    bucket_optimization: str = Field(default="optimization")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="INFLUXDB_", extra="ignore")


class MQTTSettings(BaseSettings):
    """MQTT configuration."""

    broker: str = Field(default="localhost")
    port: int = Field(default=1883)
    username: Optional[str] = Field(default=None)
    password: Optional[SecretStr] = Field(default=None)
    client_id: str = Field(default="pems_v2")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MQTT_", extra="ignore")


class PVPredictionSettings(BaseSettings):
    """PV production prediction configuration."""

    model_path: str = Field(default="models/pv_predictor.pkl")
    update_interval: int = Field(default=3600)  # seconds
    horizon_hours: int = Field(default=48)
    confidence_levels: List[float] = Field(default=[0.1, 0.5, 0.9])

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PV_", extra="ignore")


class ThermalSettings(BaseSettings):
    """Thermal model configuration."""

    rooms: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    default_setpoint_day: float = Field(default=21.0)
    default_setpoint_night: float = Field(default=19.0)
    comfort_band: float = Field(default=0.5)  # ±°C

    model_config = SettingsConfigDict(env_file=".env", env_prefix="THERMAL_", extra="ignore")


class OptimizationSettings(BaseSettings):
    """Optimization engine configuration."""

    horizon_hours: int = Field(default=48)
    control_hours: int = Field(default=24)
    time_step_minutes: int = Field(default=60)
    solver: str = Field(default="APOPT")
    max_solve_time: int = Field(default=30)  # seconds

    # Objective weights
    cost_weight: float = Field(default=1.0)
    self_consumption_weight: float = Field(default=0.3)
    peak_shaving_weight: float = Field(default=0.1)
    comfort_weight: float = Field(default=0.5)

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPT_", extra="ignore")


class BatterySettings(BaseSettings):
    """Battery system configuration."""

    capacity_kwh: float = Field(default=10.0)
    max_power_kw: float = Field(default=5.0)
    efficiency: float = Field(default=0.95)
    min_soc: float = Field(default=0.1)
    max_soc: float = Field(default=0.9)

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BATTERY_", extra="ignore")


class EVSettings(BaseSettings):
    """EV charging configuration."""

    max_power_kw: float = Field(default=11.0)
    battery_capacity_kwh: float = Field(default=60.0)
    default_target_soc: float = Field(default=0.8)
    departure_time: str = Field(default="07:00")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EV_", extra="ignore")


class PEMSSettings(BaseSettings):
    """Main PEMS configuration."""

    # Feature flags
    simulation_mode: bool = Field(default=False)
    advisory_mode: bool = Field(default=False)

    # Update intervals
    optimization_interval: int = Field(default=3600)  # seconds
    control_interval: int = Field(default=300)  # seconds

    # Sub-configurations
    influxdb: InfluxDBSettings = Field(default_factory=InfluxDBSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    pv_prediction: PVPredictionSettings = Field(default_factory=PVPredictionSettings)
    thermal: ThermalSettings = Field(default_factory=ThermalSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    battery: BatterySettings = Field(default_factory=BatterySettings)
    ev: EVSettings = Field(default_factory=EVSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PEMS_",
        extra="ignore"
    )
