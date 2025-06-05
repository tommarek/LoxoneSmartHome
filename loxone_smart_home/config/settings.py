"""Configuration settings for Loxone Smart Home using Pydantic."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModuleConfig(BaseModel):
    """Configuration for which modules are enabled."""

    udp_listener_enabled: bool = True
    mqtt_bridge_enabled: bool = True
    weather_scraper_enabled: bool = True
    growatt_controller_enabled: bool = True


class MQTTConfig(BaseModel):
    """MQTT configuration."""

    broker: str = "mqtt"
    port: int = Field(default=1883, ge=1, le=65535)
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: str = "loxone-smart-home"

    # Topics
    topic_energy_solar: str = "energy/solar"
    topic_weather: str = "weather"
    topic_temperature_prefix: str = "teplomer/"
    topic_teslamate_prefix: str = "teslamate/"

    # Growatt control topics
    topic_growatt_cmd_prefix: str = "growatt/"


class InfluxDBConfig(BaseModel):
    """InfluxDB configuration."""

    url: str = "http://influxdb:8086"
    token: str = Field(min_length=1)
    org: str = "tmarek"

    # Buckets
    bucket_loxone: str = "loxone"
    bucket_solar: str = "solar"
    bucket_weather: str = "weather_forecast"

    # Write options
    batch_size: int = Field(default=5000, ge=1)
    flush_interval: int = Field(default=1000, ge=100)  # milliseconds


class UDPListenerConfig(BaseModel):
    """UDP Listener configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=2000, ge=1, le=65535)
    buffer_size: int = Field(default=1024, ge=128)

    # Data format: timestamp;measurement_name;value;room_name;measurement_type;tag1;tag2
    delimiter: str = ";"
    timezone: str = "Europe/Prague"


class LoxoneBridgeConfig(BaseModel):
    """Loxone Bridge configuration."""

    loxone_host: str
    loxone_udp_port: int = Field(default=4000, ge=1, le=65535)

    # Topics to bridge - can be comma-separated string or list
    bridge_topics: List[str] = Field(
        default_factory=lambda: [
            "energy/solar",
            "teplomer/TC",
        ]
    )

    @field_validator("bridge_topics", mode="before")
    @classmethod
    def parse_topics(cls, v: Any) -> List[str]:
        """Parse topics from comma-separated string or list."""
        if isinstance(v, str):
            return [topic.strip() for topic in v.split(",")]
        if isinstance(v, list):
            return v
        return []


class WeatherConfig(BaseModel):
    """Weather scraper configuration."""

    # Service selection
    weather_service: str = Field(default="openmeteo", pattern="^(openmeteo|aladin|openweathermap)$")

    # API endpoints
    openmeteo_url: str = "https://api.open-meteo.com/v1/forecast"
    aladin_url_base: str = "http://www.nts2.cz:443/meteo/aladin/"
    openweathermap_url: str = "https://api.openweathermap.org/data/2.5/onecall"

    # API keys
    openweathermap_api_key: Optional[str] = None

    # Location
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = "Europe/Prague"

    # Update intervals
    update_interval: int = Field(default=1800, ge=60)  # 30 minutes in seconds

    # Retry settings
    max_retries: int = Field(default=3, ge=1)
    retry_delay: int = Field(default=60, ge=1)  # seconds


class OTEConfig(BaseModel):
    """OTE price collector configuration."""

    # API endpoint
    base_url: str = "https://www.ote-cr.cz/cs/kratkodobe-trhy/elektrina/denni-trh/@@chart-data"
    time_resolution: str = "PT60M"  # 60-minute resolution

    # Update schedule (smart retry starting at 2 PM, then hourly until data found)
    first_check_hour: int = Field(default=14, ge=0, le=23)  # Start checking at 2 PM
    max_check_hour: int = Field(default=18, ge=0, le=23)  # Stop checking at 6 PM

    # Historical data
    load_historical_days: int = Field(default=1095, ge=1)  # 3 years = 1095 days

    # Rate limiting
    request_delay: float = Field(default=1.0, ge=0.1)  # Delay between requests in seconds
    error_delay: float = Field(default=5.0, ge=1.0)  # Delay after errors
    max_retries: int = Field(default=3, ge=1, le=10)  # Max retries per request

    # EUR to CZK conversion rate (approximate, could be made dynamic)
    eur_czk_rate: float = Field(default=25.0, gt=0)


class GrowattConfig(BaseModel):
    """Growatt controller configuration."""

    # API endpoints
    ote_dam_url: str = (
        "https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data"
    )

    # Control parameters
    battery_capacity: float = Field(default=10.0, gt=0)  # kWh
    max_charge_power: float = Field(default=3.0, gt=0)  # kW
    min_soc: float = Field(default=20.0, ge=0, le=100)  # %
    max_soc: float = Field(default=90.0, ge=0, le=100)  # %

    # Price thresholds and control parameters
    export_price_threshold: float = Field(default=1.0, gt=0)  # CZK/kWh
    battery_charge_hours: int = Field(default=2, ge=1, le=12)  # Consecutive hours for AC charging
    individual_cheapest_hours: int = Field(default=6, ge=1, le=24)  # Individual cheap hours

    # MQTT topics for Growatt control
    battery_first_topic: str = "energy/solar/command/batteryfirst/set/timeslot"
    ac_charge_topic: str = "energy/solar/command/batteryfirst/set/acchargeenabled"
    export_enable_topic: str = "energy/solar/command/export/enable"
    export_disable_topic: str = "energy/solar/command/export/disable"

    # Scheduling
    schedule_hour: int = Field(default=23, ge=0, le=23)  # Daily calculation hour
    schedule_minute: int = Field(default=59, ge=0, le=59)  # Daily calculation minute

    # Simulation mode
    simulation_mode: bool = False

    # Device IDs
    device_serial: Optional[str] = None

    # Retry settings
    max_retries: int = Field(default=3, ge=1)
    retry_delay: int = Field(default=60, ge=1)  # seconds

    @field_validator("max_soc")
    @classmethod
    def validate_soc_range(cls, v: float, info: Any) -> float:
        """Ensure max_soc > min_soc."""
        if "min_soc" in info.data and v <= info.data["min_soc"]:
            raise ValueError("max_soc must be greater than min_soc")
        return v


class Settings(BaseSettings):
    """Main settings class using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General settings
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_timezone: str = Field(default="Europe/Prague", description="Timezone for log timestamps")

    # Module configuration
    udp_listener_enabled: bool = True
    mqtt_bridge_enabled: bool = True
    weather_scraper_enabled: bool = True
    growatt_controller_enabled: bool = True
    ote_collector_enabled: bool = True

    # Service configurations
    mqtt_broker: str = "mqtt"
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None

    influxdb_url: str = Field(default="http://influxdb:8086", alias="INFLUXDB_HOST")
    influxdb_token: str
    influxdb_org: str = "tmarek"
    influxdb_bucket: str = Field(default="loxone", alias="INFLUXDB_BUCKET")

    udp_listener_host: str = "0.0.0.0"
    udp_listener_port: int = Field(default=2000, ge=1, le=65535)

    loxone_host: str = Field(default="192.168.101.34", alias="LOXONE_HOST")
    loxone_udp_port: int = Field(default=4000, ge=1, le=65535, alias="LOXONE_PORT")
    mqtt_topics: Optional[str] = Field(default=None, alias="MQTT_TOPICS")

    latitude: float = Field(default=49.00642, ge=-90, le=90)
    longitude: float = Field(default=14.51994, ge=-180, le=180)
    openweathermap_api_key: Optional[str] = None
    weather_service: str = Field(default="openmeteo", alias="USE_SERVICE")

    growatt_device_serial: Optional[str] = None
    growatt_simulation_mode: bool = False

    # OTE Price Collector settings (optional overrides)
    ote_request_delay: Optional[float] = Field(default=None, ge=0.1)
    ote_error_delay: Optional[float] = Field(default=None, ge=1.0)
    ote_max_retries: Optional[int] = Field(default=None, ge=1, le=10)
    ote_load_historical_days: Optional[int] = Field(default=None, ge=1)
    ote_first_check_hour: Optional[int] = Field(default=None, ge=0, le=23)
    ote_max_check_hour: Optional[int] = Field(default=None, ge=0, le=23)

    @property
    def modules(self) -> ModuleConfig:
        """Get module configuration."""
        return ModuleConfig(
            udp_listener_enabled=self.udp_listener_enabled,
            mqtt_bridge_enabled=self.mqtt_bridge_enabled,
            weather_scraper_enabled=self.weather_scraper_enabled,
            growatt_controller_enabled=self.growatt_controller_enabled,
        )

    @property
    def mqtt(self) -> MQTTConfig:
        """Get MQTT configuration."""
        return MQTTConfig(
            broker=self.mqtt_broker,
            port=self.mqtt_port,
            username=self.mqtt_username,
            password=self.mqtt_password,
        )

    @property
    def influxdb(self) -> InfluxDBConfig:
        """Get InfluxDB configuration."""
        return InfluxDBConfig(
            url=self.influxdb_url,
            token=self.influxdb_token,
            org=self.influxdb_org,
            bucket_loxone=self.influxdb_bucket,
        )

    @property
    def udp_listener(self) -> UDPListenerConfig:
        """Get UDP listener configuration."""
        return UDPListenerConfig(
            host=self.udp_listener_host,
            port=self.udp_listener_port,
        )

    @property
    def loxone_bridge(self) -> LoxoneBridgeConfig:
        """Get Loxone bridge configuration."""
        if self.mqtt_topics:
            return LoxoneBridgeConfig(
                loxone_host=self.loxone_host,
                loxone_udp_port=self.loxone_udp_port,
                bridge_topics=self.mqtt_topics,  # type: ignore[arg-type]
            )
        return LoxoneBridgeConfig(
            loxone_host=self.loxone_host,
            loxone_udp_port=self.loxone_udp_port,
        )

    @property
    def weather(self) -> WeatherConfig:
        """Get weather configuration."""
        return WeatherConfig(
            weather_service=self.weather_service,
            latitude=self.latitude,
            longitude=self.longitude,
            openweathermap_api_key=self.openweathermap_api_key,
        )

    @property
    def growatt(self) -> GrowattConfig:
        """Get Growatt configuration."""
        return GrowattConfig(
            device_serial=self.growatt_device_serial,
            simulation_mode=self.growatt_simulation_mode,
        )

    @property
    def ote(self) -> OTEConfig:
        """Get OTE configuration."""
        return OTEConfig(
            request_delay=self.ote_request_delay or 1.0,
            error_delay=self.ote_error_delay or 5.0,
            max_retries=self.ote_max_retries or 3,
            load_historical_days=self.ote_load_historical_days or 1095,
            first_check_hour=self.ote_first_check_hour or 14,
            max_check_hour=self.ote_max_check_hour or 18,
        )
