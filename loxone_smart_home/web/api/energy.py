"""Energy API endpoints."""

from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Query, Request

from ..models.responses import (
    EnergyCurrentResponse,
    EnergyHistoryResponse,
    BatteryStatusResponse,
    EnergyStatisticsResponse
)


router = APIRouter()


@router.get("/current", response_model=EnergyCurrentResponse)
async def get_current_energy(request: Request) -> Dict[str, Any]:
    """Get current energy flow data."""
    web_service = request.app.state.web_service

    # Get cached current data
    current_data = await web_service.cache.get("energy:current")

    if not current_data:
        # Fetch from InfluxDB if not cached
        query = '''
        from(bucket: "solar")
          |> range(start: -5m)
          |> filter(fn: (r) => r["_measurement"] == "power_flow")
          |> last()
        '''

        result = await web_service.influxdb_client.query(query)
        # Process result and cache it
        current_data = _process_power_flow(result)
        await web_service.cache.set("energy:current", current_data, ttl=10)

    return current_data


@router.get("/history", response_model=EnergyHistoryResponse)
async def get_energy_history(
    request: Request,
    start: datetime = Query(default=None, description="Start time"),
    end: datetime = Query(default=None, description="End time"),
    resolution: str = Query(default="15m", description="Time resolution (15m, 1h, 1d)")
) -> Dict[str, Any]:
    """Get historical energy data."""
    web_service = request.app.state.web_service

    # Default time range
    if not end:
        end = datetime.now()
    if not start:
        start = end - timedelta(days=1)

    # Build cache key
    cache_key = f"energy:history:{start.isoformat()}:{end.isoformat()}:{resolution}"

    # Check cache
    cached_data = await web_service.cache.get(cache_key)
    if cached_data:
        return cached_data

    # Determine aggregation window based on resolution
    window_map = {
        "15m": "15m",
        "1h": "1h",
        "1d": "1d"
    }
    window = window_map.get(resolution, "15m")

    # Query InfluxDB
    query = f'''
    from(bucket: "solar")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "power_flow")
      |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
    '''

    result = await web_service.influxdb_client.query(query)

    # Process and cache result
    history_data = _process_energy_history(result, resolution)
    await web_service.cache.set(cache_key, history_data, ttl=300)  # 5 min cache

    return history_data


@router.get("/battery/status", response_model=BatteryStatusResponse)
async def get_battery_status(request: Request) -> Dict[str, Any]:
    """Get current battery status."""
    web_service = request.app.state.web_service

    # Get cached battery data
    battery_data = await web_service.cache.get("battery:status")

    if not battery_data:
        # Query latest battery data
        query = '''
        from(bucket: "solar")
          |> range(start: -5m)
          |> filter(fn: (r) => r["_measurement"] == "battery")
          |> last()
        '''

        result = await web_service.influxdb_client.query(query)
        battery_data = _process_battery_status(result)
        await web_service.cache.set("battery:status", battery_data, ttl=10)

    return battery_data


@router.get("/statistics", response_model=EnergyStatisticsResponse)
async def get_energy_statistics(
    request: Request,
    period: str = Query(default="day", description="Statistics period (day, week, month, year)")
) -> Dict[str, Any]:
    """Get aggregated energy statistics."""
    web_service = request.app.state.web_service

    # Calculate time range based on period
    end = datetime.now()
    period_map = {
        "day": timedelta(days=1),
        "week": timedelta(days=7),
        "month": timedelta(days=30),
        "year": timedelta(days=365)
    }
    start = end - period_map.get(period, timedelta(days=1))

    # Build cache key
    cache_key = f"energy:stats:{period}:{end.date()}"

    # Check cache
    cached_stats = await web_service.cache.get(cache_key)
    if cached_stats:
        return cached_stats

    # Query multiple metrics
    stats = {}

    # Total production
    production_query = f'''
    from(bucket: "solar")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "energy" and r["_field"] == "production")
      |> sum()
    '''

    # Total consumption
    consumption_query = f'''
    from(bucket: "solar")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "energy" and r["_field"] == "consumption")
      |> sum()
    '''

    # Grid import/export
    grid_query = f'''
    from(bucket: "solar")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "energy" and
                 (r["_field"] == "grid_import" or r["_field"] == "grid_export"))
      |> sum()
    '''

    # Execute queries
    production = await web_service.influxdb_client.query(production_query)
    consumption = await web_service.influxdb_client.query(consumption_query)
    grid = await web_service.influxdb_client.query(grid_query)

    # Process results
    stats = _process_statistics(production, consumption, grid, period)

    # Cache for 1 hour
    await web_service.cache.set(cache_key, stats, ttl=3600)

    return stats


def _process_power_flow(result: Any) -> Dict[str, Any]:
    """Process power flow query result."""
    # TODO: Implement actual processing based on InfluxDB result format
    return {
        "timestamp": datetime.now().isoformat(),
        "solar_power": 2500,  # Watts
        "grid_power": -500,   # Negative = export
        "home_power": 2000,
        "battery_power": 0,   # Positive = charging
        "battery_soc": 75    # Percentage
    }


def _process_energy_history(result: Any, resolution: str) -> Dict[str, Any]:
    """Process energy history query result."""
    # TODO: Implement actual processing
    return {
        "resolution": resolution,
        "data": [
            {
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "production": 2000 + i * 100,
                "consumption": 1800 + i * 50,
                "grid_import": max(0, 1800 + i * 50 - 2000 - i * 100),
                "grid_export": max(0, 2000 + i * 100 - 1800 - i * 50)
            }
            for i in range(24)
        ]
    }


def _process_battery_status(result: Any) -> Dict[str, Any]:
    """Process battery status query result."""
    # TODO: Implement actual processing
    return {
        "soc": 75,  # State of charge percentage
        "power": 500,  # Current power (positive = charging)
        "voltage": 52.5,
        "current": 9.5,
        "temperature": 25.5,
        "status": "charging",
        "health": 98  # Battery health percentage
    }


def _process_statistics(
    production: Any, consumption: Any, grid: Any, period: str
) -> Dict[str, Any]:
    """Process statistics from multiple queries."""
    # TODO: Implement actual processing
    return {
        "period": period,
        "production": {
            "total": 150.5,  # kWh
            "peak": 3.5,     # kW
            "average": 1.8   # kW
        },
        "consumption": {
            "total": 120.3,
            "peak": 4.2,
            "average": 1.5
        },
        "grid": {
            "import": 20.1,
            "export": 50.3,
            "net": -30.2  # Negative = net export
        },
        "self_sufficiency": 83.3,  # Percentage
        "self_consumption": 66.7,  # Percentage
        "savings": {
            "amount": 450.50,  # CZK
            "co2_avoided": 75.25  # kg
        }
    }
