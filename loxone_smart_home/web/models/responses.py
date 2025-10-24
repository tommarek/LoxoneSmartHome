"""Pydantic models for API responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Energy models
class EnergyCurrentResponse(BaseModel):
    """Current energy flow response."""

    timestamp: datetime
    solar_power: float = Field(description="Solar power in Watts")
    grid_power: float = Field(description="Grid power in Watts (negative = export)")
    home_power: float = Field(description="Home consumption in Watts")
    battery_power: float = Field(description="Battery power in Watts (positive = charging)")
    battery_soc: float = Field(description="Battery state of charge percentage")


class EnergyDataPoint(BaseModel):
    """Single energy data point."""

    timestamp: datetime
    production: float
    consumption: float
    grid_import: float
    grid_export: float


class EnergyHistoryResponse(BaseModel):
    """Energy history response."""

    resolution: str
    data: List[EnergyDataPoint]


class BatteryStatusResponse(BaseModel):
    """Battery status response."""

    soc: float = Field(description="State of charge percentage")
    power: float = Field(description="Current power (positive = charging)")
    voltage: float
    current: float
    temperature: float
    status: str = Field(description="Battery status (charging, discharging, idle)")
    health: float = Field(description="Battery health percentage")


class ProductionStats(BaseModel):
    """Production statistics."""

    total: float = Field(description="Total production in kWh")
    peak: float = Field(description="Peak production in kW")
    average: float = Field(description="Average production in kW")


class ConsumptionStats(BaseModel):
    """Consumption statistics."""

    total: float
    peak: float
    average: float


class GridStats(BaseModel):
    """Grid statistics."""

    import_total: float = Field(description="Total import in kWh", alias="import")
    export: float = Field(description="Total export in kWh")
    net: float = Field(description="Net grid usage (negative = net export)")

    class Config:
        """Pydantic config."""

        populate_by_name = True


class SavingsStats(BaseModel):
    """Savings statistics."""

    amount: float = Field(description="Savings amount in CZK")
    co2_avoided: float = Field(description="CO2 avoided in kg")


class EnergyStatisticsResponse(BaseModel):
    """Energy statistics response."""

    period: str
    production: ProductionStats
    consumption: ConsumptionStats
    grid: GridStats
    self_sufficiency: float = Field(description="Self-sufficiency percentage")
    self_consumption: float = Field(description="Self-consumption percentage")
    savings: SavingsStats


# Price models
class PriceBlock(BaseModel):
    """Single price block."""

    timestamp: datetime
    price_eur_mwh: float
    price_czk_kwh: float
    level: str = Field(description="Price level (low, medium, high)")


class NextPriceChange(BaseModel):
    """Next price change information."""

    time: datetime
    price_czk_kwh: float
    direction: str = Field(description="Price direction (up, down, same)")


class CurrentPriceResponse(BaseModel):
    """Current price response."""

    timestamp: datetime
    block: str = Field(description="Current time block (e.g., '14:00-14:15')")
    price_eur_mwh: float
    price_czk_kwh: float
    level: str
    next_change: NextPriceChange


class PriceSummary(BaseModel):
    """Price summary statistics."""

    min_price: float
    max_price: float
    avg_price: float


class PriceForecastResponse(BaseModel):
    """Price forecast response."""

    forecast_hours: int
    blocks: List[PriceBlock]
    summary: PriceSummary


# Weather models
class WeatherCurrentResponse(BaseModel):
    """Current weather response."""

    timestamp: datetime
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    wind_speed: float
    wind_direction: int
    cloud_cover: int
    visibility: int
    precipitation: float
    description: str
    icon: str
    sunrise: str
    sunset: str
    uv_index: float


class HourlyForecast(BaseModel):
    """Hourly weather forecast."""

    timestamp: datetime
    temperature: float
    humidity: int
    cloud_cover: int
    precipitation: float
    wind_speed: float
    description: str


class DailyForecast(BaseModel):
    """Daily weather forecast."""

    date: str
    temperature_min: float
    temperature_max: float
    humidity_avg: float
    precipitation_total: float
    description: str


class WeatherForecastResponse(BaseModel):
    """Weather forecast response."""

    forecast_hours: int
    hourly: List[HourlyForecast]
    daily: List[DailyForecast]


# Analytics models
class EfficiencyMetrics(BaseModel):
    """Efficiency metrics."""

    round_trip: Optional[float] = None
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None
    energy_loss_kwh: Optional[float] = None
    capacity_factor: Optional[float] = None
    performance_ratio: Optional[float] = None
    specific_yield: Optional[float] = None
    availability: Optional[float] = None
    self_sufficiency: Optional[float] = None
    self_consumption: Optional[float] = None
    grid_independence: Optional[float] = None
    overall_efficiency: Optional[float] = None


class Recommendation(BaseModel):
    """Optimization recommendation."""

    priority: str = Field(description="Priority level (high, medium, low)")
    category: str
    title: str
    description: str
    potential_savings: float
    implementation: str


# Schedule models
class ScheduleBlock(BaseModel):
    """Single 15-minute schedule block."""

    time: str = Field(description="Time block (e.g., '00:00-00:15')")
    price_czk_kwh: float = Field(description="Price in CZK/kWh")
    mode: str = Field(description="Mode: charge, pre_discharge, discharge, normal")
    icon: str = Field(description="Icon: 🔋=charge, 🔌=pre-discharge, ⚡=discharge, -=normal")


class ScheduleHour(BaseModel):
    """Hour with 4 15-minute blocks."""

    hour: int = Field(description="Hour (0-23)")
    blocks: List[ScheduleBlock] = Field(description="4 blocks per hour")


class ScheduleDay(BaseModel):
    """Day schedule."""

    date: str = Field(description="Date (YYYY-MM-DD)")
    label: str = Field(description="Label: TODAY, TOMORROW, etc.")
    hours: List[ScheduleHour]


class ScheduleLegend(BaseModel):
    """Legend for schedule icons."""

    icon: str
    label: str
    color: Optional[str] = None


class ScheduleTableResponse(BaseModel):
    """Schedule table response."""

    days: List[ScheduleDay]
    legend: List[ScheduleLegend]
    summary: Dict[str, Any] = Field(description="Summary statistics")
