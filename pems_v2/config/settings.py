"""Configuration settings for PEMS v2."""

from typing import Dict, List, Optional

from pydantic import Field
from pydantic.types import SecretStr
from pydantic_settings import BaseSettings


class InfluxDBSettings(BaseSettings):
    """InfluxDB configuration."""

    url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    token: SecretStr = Field(..., env="INFLUXDB_TOKEN")
    org: str = Field(default="smart_home", env="INFLUXDB_ORG")
    bucket_historical: str = Field(default="smart_home", env="INFLUXDB_BUCKET_HISTORICAL")
    bucket_predictions: str = Field(default="predictions", env="INFLUXDB_BUCKET_PREDICTIONS")
    bucket_optimization: str = Field(default="optimization", env="INFLUXDB_BUCKET_OPTIMIZATION")

    class Config:
        env_prefix = "INFLUXDB_"


class MQTTSettings(BaseSettings):
    """MQTT configuration."""

    broker: str = Field(default="localhost", env="MQTT_BROKER")
    port: int = Field(default=1883, env="MQTT_PORT")
    username: Optional[str] = Field(default=None, env="MQTT_USERNAME")
    password: Optional[SecretStr] = Field(default=None, env="MQTT_PASSWORD")
    client_id: str = Field(default="pems_v2", env="MQTT_CLIENT_ID")

    class Config:
        env_prefix = "MQTT_"


class PVPredictionSettings(BaseSettings):
    """PV production prediction configuration."""

    model_path: str = Field(default="models/pv_predictor.pkl", env="PV_MODEL_PATH")
    update_interval: int = Field(default=3600, env="PV_UPDATE_INTERVAL")  # seconds
    horizon_hours: int = Field(default=48, env="PV_HORIZON_HOURS")
    confidence_levels: List[float] = Field(default=[0.1, 0.5, 0.9], env="PV_CONFIDENCE_LEVELS")

    class Config:
        env_prefix = "PV_"


class ThermalSettings(BaseSettings):
    """Thermal model configuration."""

    rooms: Dict[str, Dict[str, float]] = Field(default_factory=dict, env="THERMAL_ROOMS")
    default_setpoint_day: float = Field(default=21.0, env="THERMAL_DEFAULT_SETPOINT_DAY")
    default_setpoint_night: float = Field(default=19.0, env="THERMAL_DEFAULT_SETPOINT_NIGHT")
    comfort_band: float = Field(default=0.5, env="THERMAL_COMFORT_BAND")  # ±°C

    class Config:
        env_prefix = "THERMAL_"


class OptimizationSettings(BaseSettings):
    """Optimization engine configuration."""

    horizon_hours: int = Field(default=48, env="OPT_HORIZON_HOURS")
    control_hours: int = Field(default=24, env="OPT_CONTROL_HOURS")
    time_step_minutes: int = Field(default=60, env="OPT_TIME_STEP_MINUTES")
    solver: str = Field(default="APOPT", env="OPT_SOLVER")
    max_solve_time: int = Field(default=30, env="OPT_MAX_SOLVE_TIME")  # seconds

    # Objective weights
    cost_weight: float = Field(default=1.0, env="OPT_COST_WEIGHT")
    self_consumption_weight: float = Field(default=0.3, env="OPT_SELF_CONSUMPTION_WEIGHT")
    peak_shaving_weight: float = Field(default=0.1, env="OPT_PEAK_SHAVING_WEIGHT")
    comfort_weight: float = Field(default=0.5, env="OPT_COMFORT_WEIGHT")

    class Config:
        env_prefix = "OPT_"


class BatterySettings(BaseSettings):
    """Battery system configuration."""

    capacity_kwh: float = Field(default=10.0, env="BATTERY_CAPACITY_KWH")
    max_power_kw: float = Field(default=5.0, env="BATTERY_MAX_POWER_KW")
    efficiency: float = Field(default=0.95, env="BATTERY_EFFICIENCY")
    min_soc: float = Field(default=0.1, env="BATTERY_MIN_SOC")
    max_soc: float = Field(default=0.9, env="BATTERY_MAX_SOC")

    class Config:
        env_prefix = "BATTERY_"


class EVSettings(BaseSettings):
    """EV charging configuration."""

    max_power_kw: float = Field(default=11.0, env="EV_MAX_POWER_KW")
    battery_capacity_kwh: float = Field(default=60.0, env="EV_BATTERY_CAPACITY_KWH")
    default_target_soc: float = Field(default=0.8, env="EV_DEFAULT_TARGET_SOC")
    departure_time: str = Field(default="07:00", env="EV_DEPARTURE_TIME")

    class Config:
        env_prefix = "EV_"


class PEMSSettings(BaseSettings):
    """Main PEMS configuration."""

    # Feature flags
    simulation_mode: bool = Field(default=False, env="PEMS_SIMULATION_MODE")
    advisory_mode: bool = Field(default=False, env="PEMS_ADVISORY_MODE")

    # Update intervals
    optimization_interval: int = Field(default=3600, env="PEMS_OPTIMIZATION_INTERVAL")  # seconds
    control_interval: int = Field(default=300, env="PEMS_CONTROL_INTERVAL")  # seconds

    # Sub-configurations
    influxdb: InfluxDBSettings = Field(default_factory=InfluxDBSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    pv_prediction: PVPredictionSettings = Field(default_factory=PVPredictionSettings)
    thermal: ThermalSettings = Field(default_factory=ThermalSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    battery: BatterySettings = Field(default_factory=BatterySettings)
    ev: EVSettings = Field(default_factory=EVSettings)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "PEMS_"
