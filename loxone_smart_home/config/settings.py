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

    # MQTT publishing configuration
    mqtt_publish_enabled: bool = True
    mqtt_status_topic: str = "loxone/status"
    mqtt_publish_interval: int = Field(default=30, ge=1)  # seconds


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
    time_resolution: str = "PT15M"  # 15-minute resolution (96 blocks per day)

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


class GrowattConfig(BaseSettings):
    """Growatt controller configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GROWATT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API endpoints
    ote_dam_url: str = (
        "https://www.ote-cr.cz/en/short-term-markets/electricity/day-ahead-market/@@chart-data"
    )

    # Control parameters
    battery_capacity: float = Field(default=10.0, gt=0)  # kWh
    min_soc: float = Field(default=20.0, ge=0, le=100)  # %
    max_soc: float = Field(default=100.0, ge=0, le=100)  # %
    discharge_min_soc: float = Field(default=20.0, ge=0, le=100)  # % - Stop discharge at this SOC

    # Battery parameters
    battery_efficiency: float = Field(
        default=0.85, gt=0, le=1,
        description="Battery round-trip efficiency (15% loss)"
    )
    discharge_power_rate: int = Field(
        default=25, ge=10, le=100,
        description="Discharge power rate % (25% = gentle)"
    )
    battery_charge_rate_kw: float = Field(
        default=2.5, gt=0, le=50,
        description="Max battery charge power (kW). Used by the optimizer to "
                    "size per-block charge energy (kW * 0.25 kWh per 15-min block)."
    )
    battery_discharge_rate_kw: float = Field(
        default=2.5, gt=0, le=50,
        description="Actual battery discharge power (kW) at the configured "
                    "discharge_power_rate (~2.5 kW at 25% on this inverter). Used "
                    "directly by the optimizer to size per-block discharge energy "
                    "(kW * 0.25 kWh/block), symmetric with battery_charge_rate_kw. "
                    "Keep it consistent with discharge_power_rate if you change it."
    )
    # --- Adaptive charge rate (opt-in, default OFF) -----------------------
    # When enabled, the optimizer is allowed to charge faster than the gentle
    # battery_charge_rate_kw (up to battery_charge_max_kw) and the controller
    # sets the inverter powerRate% per charge window to the GENTLEST rate that
    # still fills the window in time. Default OFF keeps production unchanged
    # (fixed gentle ~25% charging).
    adaptive_charge_rate: bool = Field(
        default=False,
        description="Let the optimizer pick the inverter charge powerRate per "
                    "window (gentle unless a short cheap window needs full power)."
    )
    battery_charge_max_kw: float = Field(
        default=9.8, gt=0, le=50,
        description="True hardware max charge power (kW) at powerRate=100% "
                    "(~9.8 kW = 4x the 2.44 kW measured at 25%). Used as the MILP "
                    "charge cap AND the denominator for adaptive powerRate%, only "
                    "when adaptive_charge_rate is on."
    )
    min_charge_power_rate: int = Field(
        default=25, ge=10, le=100,
        description="Floor for the adaptive charge powerRate% so charging never "
                    "trickles below the gentle baseline."
    )
    # Operational power ceilings (kW) — cap the optimizer's adaptive charge/
    # discharge speed below the raw hardware max to limit battery C-rate/wear.
    # Default = hardware max (no extra cap). E.g. 5.3 kW ≈ 0.5C on a ~10.6 kWh
    # pack. The powerRate% is still measured against battery_*_max_kw, so this
    # caps both the plan (MILP rate) and the actuated powerRate.
    max_charge_power_kw: float = Field(
        default=9.8, gt=0, le=50,
        description="Ceiling on adaptive charge power (kW); set below "
                    "battery_charge_max_kw to limit C-rate. Only when "
                    "adaptive_charge_rate is on."
    )
    max_discharge_power_kw: float = Field(
        default=9.8, gt=0, le=50,
        description="Ceiling on adaptive discharge power (kW); set below "
                    "battery_discharge_max_kw to limit C-rate. Only when "
                    "adaptive_discharge_rate is on."
    )
    # Symmetric adaptive discharge rate (opt-in, default OFF). Lets the optimizer
    # discharge faster than the gentle ~2.5 kW to fully exploit a SHORT high-price
    # spike it would otherwise under-drain. grid_first already writes powerRate
    # (no guard), so only the MILP cap + per-window rate are needed.
    adaptive_discharge_rate: bool = Field(
        default=False,
        description="Let the optimizer pick the inverter discharge powerRate per "
                    "window (gentle unless a short expensive window needs full power)."
    )
    battery_discharge_max_kw: float = Field(
        default=9.8, gt=0, le=50,
        description="True hardware max discharge power (kW) at powerRate=100%. "
                    "Used as the MILP discharge cap and the denominator for the "
                    "adaptive discharge powerRate%, only when adaptive_discharge_rate "
                    "is on."
    )
    sell_production_min_soc_margin: float = Field(
        default=2.0, ge=0, le=20,
        description="SOC margin below max_soc within which sell_production may "
                    "actuate as true grid export. Below (max_soc - margin) the "
                    "battery isn't full, so grid-first would bank surplus solar "
                    "instead of exporting — so the controller actuates battery_hold "
                    "(charge from solar) and the MILP won't plan sell_production."
    )

    # Simple price thresholds (all in CZK/kWh for consistency)
    charge_price_max: float = Field(
        default=1.5, gt=0,
        description="Charge battery when price below this (CZK/kWh)"
    )
    export_price_min: float = Field(
        default=0.35, gt=0,
        description=(
            "STRICT export floor (CZK/kWh): never export to grid (solar or "
            "battery) when the spot price is below this — set to the export/"
            "transmission fee so export is only allowed when net revenue >= 0. "
            "Enforced both as a hardware gate and as a constraint the optimizer "
            "plans within."
        )
    )
    discharge_price_min: float = Field(
        default=5.0, gt=0,
        description="Discharge battery to grid when price above this (CZK/kWh)"
    )
    discharge_profit_margin: float = Field(
        default=4.0, ge=1.0,
        description="Required multiplier over cheapest hour (4.0 = sell at 4× cheapest price)"
    )

    # Scheduling - using 15-minute blocks (non-consecutive)
    battery_charge_blocks: int = Field(
        default=8, ge=1, le=96,
        description="Number of 15-minute blocks to charge battery (8 = 2 hours, non-consecutive)"
    )

    # Dynamic charge block count
    dynamic_charge_blocks: bool = Field(
        default=False,
        description=(
            "Auto-adjust charge block count based on price spread. Defaults OFF "
            "so the rule-based path keeps the fixed battery_charge_blocks count "
            "unless explicitly opted in (enabling it changes grid-charge volume)."
        )
    )
    min_charge_blocks: int = Field(
        default=4, ge=0, le=96,
        description="Minimum 15-min charge blocks (4 = 1 hour floor)"
    )
    max_charge_blocks: int = Field(
        default=16, ge=1, le=96,
        description="Maximum 15-min charge blocks (16 = 4 hours ceiling)"
    )

    # Pre-discharge charging parameters
    pre_discharge_charge_blocks: int = Field(
        default=8, ge=1, le=96,
        description="Number of 15-minute blocks to charge before discharge peaks (8 = 2 hours)"
    )
    pre_discharge_window_hours: int = Field(
        default=24, ge=2, le=24,
        description="Hours to look back for cheap blocks (default 24 = entire day before discharge)"
    )

    # Command control parameters
    command_delay: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Delay between inverter commands in seconds"
    )
    command_retry_count: int = Field(
        default=10, ge=1, le=20,
        description="Maximum number of retry attempts for failed commands"
    )
    command_retry_delay: float = Field(
        default=2.0, ge=0.5, le=30.0,
        description="Initial delay between command retries in seconds (exponential backoff)"
    )
    command_timeout: float = Field(
        default=10.0, ge=1.0, le=30.0,
        description="Timeout for waiting for command results in seconds"
    )
    clock_drift_buffer_minutes: int = Field(
        default=0, ge=0, le=10,
        description="Base buffer for clock drift (1 min minimum added for MQTT)"
    )

    # MQTT topics for Growatt control
    battery_first_topic: str = "energy/solar/command/batteryfirst/set/timeslot"
    ac_charge_topic: str = "energy/solar/command/batteryfirst/set/acchargeenabled"
    export_enable_topic: str = "energy/solar/command/export/enable"
    export_disable_topic: str = "energy/solar/command/export/disable"

    # Grid-first mode topics
    grid_first_topic: str = "energy/solar/command/gridfirst/set/timeslot"
    grid_first_stopsoc_topic: str = "energy/solar/command/gridfirst/set/stopsoc"
    grid_first_powerrate_topic: str = "energy/solar/command/gridfirst/set/powerrate"

    # Load-first mode topics
    load_first_stopsoc_topic: str = "energy/solar/command/loadfirst/set/stopsoc"

    # Inverter on/off via Modbus holding register 0 (OpenInverterGateway).
    # When the spot price drops below the threshold (with hysteresis), the
    # controller writes register 0 = 0 to power the inverter off; when the
    # price recovers above threshold + hysteresis, register 0 = 1 turns it
    # back on. Scheduled grid-charging blocks always force the inverter on.
    inverter_onoff_topic: str = "energy/solar/command/modbus/set"
    inverter_off_price_threshold_czk: float = Field(
        default=-2.0,
        description="Power inverter off when spot price (CZK/kWh) is below this. "
                    "Default -2.0 CZK = only deeply-negative hours where "
                    "exporting solar costs us AND importing direct from grid "
                    "earns money (grid balancing payment)."
    )
    inverter_off_price_hysteresis_czk: float = Field(
        default=0.1, ge=0,
        description="Hysteresis around threshold to avoid flapping (CZK/kWh)"
    )

    # Season detection parameters
    summer_temp_threshold: float = Field(default=15.0, ge=-20, le=40)  # °C
    temperature_avg_days: int = Field(default=3, ge=1, le=7)  # Days for temperature average
    summer_charge_price_max: float = Field(
        default=0.0, ge=-100, le=100,
        description="Max price (CZK/kWh) to charge from grid in summer. 0 = only free/negative prices"
    )

    # Solar forecast
    solar_forecast_enabled: bool = Field(
        default=False,
        description="Enable solar production forecast for smarter charging"
    )
    solar_arrays: str = Field(
        default='[{"name":"terasa","declination":35,"azimuth":226,"kwp":7.0},{"name":"ulice","declination":35,"azimuth":136,"kwp":6.5}]',
        description="JSON array of solar panel arrays [{name, declination, azimuth, kwp}]"
    )
    solar_forecast_confidence: float = Field(
        default=0.7, gt=0, le=1.0,
        description="Discount factor for forecast uncertainty (0.7 = use 70% of predicted)"
    )
    solar_forecast_update_hours: int = Field(
        default=6, ge=1, le=24,
        description="Hours between forecast.solar API updates"
    )
    solar_model_quantile: float = Field(
        default=0.5, ge=0.5, le=0.9,
        description=(
            "Per-bin quantile for the learned solar production model. 0.5 = "
            "median (default, unchanged). Curtailment thins out a bin's "
            "highest-production (sunniest) samples, biasing the median low, so "
            "a higher value (e.g. 0.7) recovers true PV potential. Used only "
            "for the final prediction model; curtailment detection stays at "
            "the median."
        ),
    )

    # Consumption forecast
    consumption_forecast_enabled: bool = Field(
        default=False,
        description="Enable temperature-aware consumption forecasting from historical data"
    )
    consumption_forecast_engine: str = Field(
        default="binned",
        pattern="^(binned|ml)$",
        description=(
            "Which consumption forecaster to use: 'binned' (temperature-binned "
            "median, default) or 'ml' (skforecast autoregressive). Falls back "
            "to 'binned' if ML training fails or skforecast isn't installed."
        ),
    )
    ml_consumption_quantile: float = Field(
        default=0.5, ge=0.5, le=0.95,
        description=(
            "Quantile the ML consumption forecaster emits. 0.5 = median "
            "(expected load). A higher value (e.g. 0.75) biases the forecast "
            "upward so the optimizer keeps a larger reserve against "
            "underestimating demand. Only used by the 'ml' engine."
        ),
    )

    # Solcast PV forecast (optional, replaces forecast.solar when configured).
    # Free tier: 10 API requests/day per rooftop site.
    solcast_api_key: str = Field(
        default="",
        description="Solcast API key (https://solcast.com). Empty disables Solcast."
    )
    solcast_rooftop_id: str = Field(
        default="",
        description="Solcast rooftop site UUID(s). For a home with multiple roof "
                    "orientations, list one site per array, comma-separated "
                    "(e.g. 'sw-uuid,se-uuid'); they are summed into total "
                    "production. The free 10/day call budget is shared across sites."
    )
    solcast_quantile: str = Field(
        default="p50",
        pattern="^(p10|p50|p90)$",
        description=(
            "Which Solcast PV estimate to use: 'p50' (median, default), "
            "'p10' (conservative/cloudy-biased — safer reserve sizing) or "
            "'p90' (optimistic). Solcast returns all three per interval."
        ),
    )

    # Deferrable loads — controllable appliances that can be time-shifted to
    # the cheapest hours within a permitted window. Schema (JSON array):
    # [{
    #   "name": "ev_charger",
    #   "energy_required_kwh": 25,
    #   "power_kw": 11,
    #   "earliest_start": "22:00",   # HH:MM local
    #   "latest_end": "06:00",
    #   "interruptible": true,
    #   "mqtt_topic_on": "loxone/ev/charge/on",
    #   "mqtt_topic_off": "loxone/ev/charge/off"
    # }]
    deferrable_loads_json: str = Field(
        default="[]",
        description=(
            "JSON array of deferrable load specs (see module docstring). "
            "earliest_start may be later than latest_end for overnight windows "
            "(e.g. 22:00-06:00). WARNING: only list loads NOT already present in "
            "the consumption history (INVPowerToLocalLoad) — adding a load the "
            "consumption forecast already learned double-counts it and makes the "
            "battery over-charge from grid."
        )
    )

    # Optimizer (MILP). When disabled the controller holds (no scheduling);
    # the decision_engine's safety gates still apply. The MILP is the only
    # engine — if PuLP is missing or a solve fails it uses a minimal safe
    # fallback (reuse last plan, else hold + charge cheapest blocks).
    optimizer_enabled: bool = Field(
        default=False,
        description=(
            "Enable the MILP battery optimizer. When off, the controller holds "
            "(no charge/discharge scheduling); decision_engine safety gates "
            "still apply."
        ),
    )
    milp_switch_penalty_czk: float = Field(
        default=0.05, ge=0,
        description=(
            "MILP only: small cost (CZK) charged when a block's grid-facing "
            "mode differs from the previously-solved plan, to damp schedule "
            "churn on noisy price/forecast updates. 0 disables. Kept well "
            "below real price spreads so it never overrides a genuine "
            "arbitrage opportunity."
        ),
    )

    # Sell economics
    sell_fee_czk: float = Field(
        default=0.35, ge=0,
        description="Fixed fee per kWh when selling to grid (CZK/kWh). For this "
                    "tariff (FVE buyback = OTE spot − 350 Kč/MWh) it is 0.35 and "
                    "is the ONLY deduction on export — no distribution is charged."
    )
    battery_amortisation_czk: float = Field(
        default=2.0, ge=0,
        description="Battery wear cost per kWh discharged (CZK/kWh)"
    )
    battery_amortisation_export_czk: Optional[float] = Field(
        default=None, ge=0,
        description="OPTIONAL extra-conservative wear cost per kWh applied ONLY to "
                    "battery→grid EXPORT (arbitrage), not to battery→house self-"
                    "consumption. Physically wear is identical, but exporting is a "
                    "thinner, riskier bet (sell fee + price-spread speculation), so "
                    "a higher value here raises the hurdle for cycling the battery "
                    "purely to sell. None (default) = use battery_amortisation_czk "
                    "for export too (single shared wear cost, original behaviour)."
    )

    # Distribution tariff (Czech D57d high/low) — per-kWh IMPORT surcharge.
    # Includes systémové služby (0.164 CZK/kWh, charged on every kWh) on top of
    # the VT/NT distribution rate: VT 0.755+0.164=0.919, NT 0.116+0.164=0.281.
    # Applies to IMPORT only; export pays no distribution (see sell_fee_czk).
    distribution_tariff_high: float = Field(
        default=0.919, ge=0,
        description="High tariff (VT) import distribution+system cost (CZK/kWh)"
    )
    distribution_tariff_low: float = Field(
        default=0.281, ge=0,
        description="Low tariff (NT) import distribution+system cost (CZK/kWh)"
    )
    low_tariff_hours: str = Field(
        default="0-10,11-12,13-14,15-17,18-24",
        description="Low tariff (NT) hour ranges (comma-separated, e.g. '0-10,11-12,13-14,15-17,18-24' for D57d)"
    )

    # Currency conversion
    eur_czk_rate: float = Field(default=25.0, gt=0)  # EUR to CZK exchange rate

    # High-load protection — prevent battery discharge while a big load (EV
    # charging or heating) is running, so stored energy isn't drained into it.
    # Heating is detected from the loxone/status MQTT cache (relay tag1=heating);
    # EV charging is read from InfluxDB (teslamate `ev` measurement) since it
    # never reaches that cache.
    high_load_protection_enabled: bool = Field(
        default=True,
        description="Block battery discharge while EV charging or heating is "
                    "active (decision-engine high_load_protected mode)."
    )
    ev_charging_power_threshold_w: int = Field(
        default=100, ge=0,
        description="EV is considered 'charging' when ev_charging=1 or "
                    "ev_charging_power exceeds this many watts."
    )
    ev_high_load_poll_seconds: int = Field(
        default=60, ge=15, le=600,
        description="How often (seconds) to poll InfluxDB for EV charging state."
    )

    # Simulation mode
    simulation_mode: bool = False

    # Logging configuration
    log_level: str = Field(
        default="DETAIL",
        description="Logging level for Growatt controller (SUMMARY, DETAIL, VERBOSE, DEBUG)"
    )

    # Price fetching settings
    price_fetch_hour: int = Field(
        default=14, ge=0, le=23,
        description="Hour of day to start fetching next day's DAM prices (14 = 2 PM)"
    )
    price_fetch_retry_initial_delay: int = Field(
        default=5, ge=1, le=60,
        description="Initial retry delay in minutes for price fetching (exponential backoff)"
    )
    price_fetch_retry_max_delay: int = Field(
        default=60, ge=1, le=120,
        description="Maximum retry delay in minutes for price fetching"
    )
    price_fetch_retry_max_attempts: int = Field(
        default=20, ge=1, le=50,
        description="Maximum retry attempts for price fetching before giving up"
    )
    defer_to_tomorrow_threshold: float = Field(
        default=15.0, ge=0.0, le=100.0,
        description=(
            "Percentage threshold for deferring charging to tomorrow "
            "(e.g., 15.0 = defer if tomorrow is 15%+ cheaper)"
        )
    )

    # Retry settings (generic, kept for backward compatibility)
    max_retries: int = Field(default=3, ge=1)
    retry_delay: int = Field(default=60, ge=1)  # seconds

    @field_validator("max_soc")
    @classmethod
    def validate_soc_range(cls, v: float, info: Any) -> float:
        """Ensure max_soc > min_soc."""
        if "min_soc" in info.data and v <= info.data["min_soc"]:
            raise ValueError("max_soc must be greater than min_soc")
        return v

    @field_validator("deferrable_loads_json")
    @classmethod
    def validate_deferrable_loads_json(cls, v: str) -> str:
        """Fail fast on malformed deferrable-loads JSON.

        The controller parses this best-effort at __init__ (swallowing errors
        into an empty list), so a typo would silently schedule no loads. Catch
        gross JSON/shape errors here at startup instead.
        """
        import json
        from datetime import time as _time
        try:
            parsed = json.loads(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"deferrable_loads_json is not valid JSON: {e}")
        if not isinstance(parsed, list):
            raise ValueError("deferrable_loads_json must be a JSON array")
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"deferrable_loads_json entry {i} must be an object"
                )
            if not isinstance(entry.get("name"), str) or not entry["name"]:
                raise ValueError(
                    f"deferrable_loads_json entry {i} needs a non-empty string 'name'"
                )
            name = entry["name"]
            for key in ("energy_required_kwh", "power_kw"):
                if key not in entry:
                    raise ValueError(
                        f"deferrable_loads_json entry {name!r} is missing required '{key}'"
                    )
                try:
                    num = float(entry[key])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"deferrable_loads_json entry {name!r}: '{key}' must be a number"
                    )
                # power_kw drives blocks_needed (energy / power) and energy_required_kwh
                # must be deliverable — both must be strictly positive, else scheduling
                # would divide by zero / never complete. Fail fast at startup.
                if num <= 0:
                    raise ValueError(
                        f"deferrable_loads_json entry {name!r}: '{key}' must be > 0 "
                        f"(got {entry[key]!r})"
                    )
            # Optional time windows default to a full day in the controller, but
            # if present they must be parseable by time.fromisoformat (HH:MM).
            window_times = {}
            for key in ("earliest_start", "latest_end"):
                if key in entry:
                    try:
                        window_times[key] = _time.fromisoformat(entry[key])
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"deferrable_loads_json entry {name!r}: '{key}' must be "
                            f"an HH:MM time string (got {entry[key]!r})"
                        )
            # A zero-width window (earliest_start == latest_end) passes time
            # parsing but in_window() is then always False — the load would
            # silently never be scheduled. Reject it loudly instead. Compare
            # AFTER applying the controller's defaults for missing keys
            # (00:00 / 23:59), so a single-key entry like
            # {"latest_end": "00:00"} is caught too.
            eff_start = window_times.get(
                "earliest_start", _time.fromisoformat("00:00")
            )
            eff_end = window_times.get(
                "latest_end", _time.fromisoformat("23:59")
            )
            # Width in minutes, mod 24h for midnight-wrapping windows. A
            # window narrower than one 15-min block can contain no block
            # start — same silent never-schedules failure as zero width.
            start_min = eff_start.hour * 60 + eff_start.minute
            end_min = eff_end.hour * 60 + eff_end.minute
            width_min = (end_min - start_min) % (24 * 60)
            if width_min < 15:
                raise ValueError(
                    f"deferrable_loads_json entry {name!r}: the effective "
                    f"window ({eff_start.strftime('%H:%M')} → "
                    f"{eff_end.strftime('%H:%M')}, after applying defaults "
                    f"for missing keys) is zero-width or narrower than one "
                    f"15-minute block — the load would (almost) never be "
                    f"scheduled. For a ~full-day window use earliest_start "
                    f"'00:00' and latest_end '23:59'."
                )
            # A load we can switch ON but not OFF would be energised at its
            # window start and then left running forever (the controller's stop
            # path has no command to send). Reject it here so the whole feature
            # fails fast at startup instead of the controller swallowing the
            # error and silently disabling ALL deferrable loads.
            if entry.get("mqtt_topic_on") and not entry.get("mqtt_topic_off"):
                raise ValueError(
                    f"deferrable_loads_json entry {name!r}: has 'mqtt_topic_on' "
                    f"but no 'mqtt_topic_off' — a load that cannot be turned off "
                    f"must not be controlled"
                )
        return v

    @field_validator("solar_arrays")
    @classmethod
    def validate_solar_arrays(cls, v: str) -> str:
        """Fail fast on malformed solar-arrays JSON.

        SolarForecast.from_config parses this at startup; a typo would
        otherwise surface deep inside model building only when
        solar_forecast_enabled is true. Catch gross JSON/shape errors here.
        """
        import json
        try:
            parsed = json.loads(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"solar_arrays is not valid JSON: {e}")
        if not isinstance(parsed, list):
            raise ValueError("solar_arrays must be a JSON array")
        # An empty array is valid: it just means "no arrays configured" (solar
        # forecast disabled or using a non-array source). The gated consumer
        # decides what to do; don't hard-fail startup here.
        if not parsed:
            return v
        for a in parsed:
            if not isinstance(a, dict) or not {
                "name", "declination", "azimuth", "kwp"
            } <= set(a):
                raise ValueError(
                    "each solar_arrays entry needs name/declination/azimuth/kwp"
                )
        return v

    @field_validator("low_tariff_hours")
    @classmethod
    def validate_low_tariff_hours(cls, v: str) -> str:
        """Validate the comma-separated 'start-end' low-tariff hour ranges.

        Consumed by the controller's cost math whenever it runs (not gated by
        a feature flag), so a malformed value should fail at startup rather
        than raise mid-decision.
        """
        n_ranges = 0
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                start_s, end_s = part.split("-")
                start, end = int(start_s), int(end_s)
            except ValueError:
                raise ValueError(
                    f"low_tariff_hours range {part!r} must be 'start-end' integers"
                )
            if not (0 <= start <= 24 and 0 <= end <= 24 and start < end):
                raise ValueError(
                    f"low_tariff_hours range {part!r} must satisfy 0<=start<end<=24"
                )
            n_ranges += 1
        # Require at least one range. An all-empty value parses as "no NT hours",
        # which silently bills EVERY import hour at the high VT tariff — a quiet
        # economic mis-config. Reject it (e.g. when a user clears the field in the
        # settings UI) instead of letting it validate.
        if n_ranges == 0:
            raise ValueError(
                "low_tariff_hours must contain at least one 'start-end' range"
            )
        return v

    @field_validator("max_charge_blocks")
    @classmethod
    def validate_charge_block_range(cls, v: int, info: Any) -> int:
        """Ensure max_charge_blocks >= min_charge_blocks.

        Otherwise calculate_dynamic_block_count's `max(min_blocks, count)` floor
        would silently exceed the intended ceiling, charging more blocks than
        max_charge_blocks.
        """
        if "min_charge_blocks" in info.data and v < info.data["min_charge_blocks"]:
            raise ValueError("max_charge_blocks must be >= min_charge_blocks")
        return v

    @field_validator("battery_discharge_max_kw")
    @classmethod
    def validate_discharge_rate_consistency(cls, v: float, info: Any) -> float:
        """Keep battery_discharge_rate_kw consistent with discharge_power_rate.

        battery_discharge_rate_kw (the kW the OPTIMIZER uses to size per-block
        discharge energy) and discharge_power_rate (the % the inverter is
        ACTUALLY actuated at, against this hardware max) feed two independent
        paths. Both are live-editable in the settings UI, so editing one without
        the other silently mis-sizes the MILP discharge plan vs. real actuation.
        The implied discharge power is battery_discharge_max_kw * rate/100; flag
        a gross divergence (>50%) so a desync fails fast instead of degrading the
        plan. (The default 2.5 kW vs 9.8 kW * 25% = 2.45 kW is well within band.)
        Validated here, on the later-defined field, so the other two are present.
        """
        rate_pct = info.data.get("discharge_power_rate")
        rate_kw = info.data.get("battery_discharge_rate_kw")
        if rate_pct and rate_kw and v > 0:
            implied_kw = v * rate_pct / 100.0
            if implied_kw > 0 and not (0.5 <= rate_kw / implied_kw <= 1.5):
                raise ValueError(
                    f"battery_discharge_rate_kw ({rate_kw} kW) is inconsistent "
                    f"with discharge_power_rate ({rate_pct}%) of "
                    f"battery_discharge_max_kw ({v} kW) = {implied_kw:.2f} kW "
                    "implied. Set battery_discharge_rate_kw near the implied kW."
                )
        return v


class WebServiceConfig(BaseSettings):
    """Configuration for the web monitoring service."""

    enabled: bool = Field(default=False, description="Enable web service")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8080, ge=1, le=65535, description="Port to listen on")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins for web service"
    )
    enable_auth: bool = Field(default=False, description="Enable API key authentication")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")

    model_config = SettingsConfigDict(
        env_prefix="WEB_SERVICE_",
        env_nested_delimiter="__"
    )


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
    # Unique per-process broker client id. Must differ between containers that
    # share a broker (e.g. the controller vs the data-ingest process) — an MQTT
    # broker evicts an existing connection when a new one reuses the same id.
    mqtt_client_id: str = "loxone-smart-home"

    influxdb_url: str = Field(default="http://influxdb:8086", alias="INFLUXDB_HOST")
    influxdb_token: str
    influxdb_org: str = "tmarek"
    influxdb_bucket: str = Field(default="loxone", alias="INFLUXDB_BUCKET")

    udp_listener_host: str = "0.0.0.0"
    udp_listener_port: int = Field(default=2000, ge=1, le=65535)

    loxone_host: str = Field(default="192.168.0.200", alias="LOXONE_HOST")
    loxone_udp_port: int = Field(default=4000, ge=1, le=65535, alias="LOXONE_PORT")
    mqtt_topics: Optional[str] = Field(default=None, alias="MQTT_TOPICS")

    latitude: float = Field(default=49.00642, ge=-90, le=90)
    longitude: float = Field(default=14.51994, ge=-180, le=180)
    openweathermap_api_key: Optional[str] = None
    weather_service: str = Field(default="openmeteo", alias="USE_SERVICE")

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
            client_id=self.mqtt_client_id,
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
        # GrowattConfig will load from GROWATT_* environment variables automatically
        return GrowattConfig()

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

    @property
    def web_service(self) -> WebServiceConfig:
        """Get web service configuration."""
        # WebServiceConfig will load from WEB_SERVICE_* environment variables automatically
        return WebServiceConfig()
