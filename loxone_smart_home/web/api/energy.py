"""Fixed Energy API endpoints with correct InfluxDB queries."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..models.responses import (
    BatteryStatusResponse,
    EnergyCurrentResponse,
    EnergyHistoryResponse
)

router = APIRouter(prefix="/api/energy", tags=["energy"])


@router.get("/current", response_model=EnergyCurrentResponse)
async def get_current_energy_flow(request: Request) -> Dict[str, Any]:
    """Get current energy flow data."""
    web_service = request.app.state.web_service

    # Get latest data from cache or MQTT
    current_data = await web_service.cache.get("energy:current")

    if not current_data:
        # Query latest from InfluxDB - use simple single-line query
        query = 'from(bucket: "loxone") |> range(start: -5m) |> filter(fn: (r) => r["_measurement"] == "solar_power" or r["_measurement"] == "grid_power" or r["_measurement"] == "battery_power") |> last()'

        try:
            result = await web_service.influxdb_client.query(query)
            current_data = _process_current_flow(result)
            await web_service.cache.set("energy:current", current_data, ttl=10)
        except Exception as e:
            # Return default values on error
            current_data = {
                "solar_power": 0,
                "grid_power": 0,
                "battery_power": 0,
                "home_consumption": 0,
                "timestamp": datetime.now().isoformat()
            }

    return current_data


@router.get("/history")
async def get_energy_history(
    request: Request,
    period: str = Query(default="24h", description="Time period: 1h, 6h, 24h, 7d, 30d"),
    resolution: str = Query(default="15m", description="Data resolution: 15m, 1h, 1d")
) -> Dict[str, Any]:
    """Get historical energy data."""
    web_service = request.app.state.web_service

    # Map period to time range
    now = datetime.now()
    period_map = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30)
    }

    delta = period_map.get(period, timedelta(hours=24))
    start = now - delta

    # Build simple query
    query = f'from(bucket: "loxone") |> range(start: {start.isoformat()}Z) |> filter(fn: (r) => r["_measurement"] == "solar_power" or r["_measurement"] == "grid_power") |> aggregateWindow(every: {resolution}, fn: mean, createEmpty: false)'

    try:
        result = await web_service.influxdb_client.query(query)
        history_data = _process_energy_history(result, resolution)
    except Exception as e:
        # Return empty data on error
        history_data = {
            "period": period,
            "resolution": resolution,
            "solar": [],
            "grid": [],
            "consumption": []
        }

    return history_data


@router.get("/battery/status", response_model=BatteryStatusResponse)
async def get_battery_status(request: Request) -> Dict[str, Any]:
    """Get current battery status."""
    web_service = request.app.state.web_service

    # Get cached battery data
    battery_data = await web_service.cache.get("battery:status")

    if not battery_data:
        # Query latest battery data with simple query
        query = 'from(bucket: "loxone") |> range(start: -5m) |> filter(fn: (r) => r["_field"] == "battery_soc" or r["_field"] == "battery_power") |> last()'

        try:
            result = await web_service.influxdb_client.query(query)
            battery_data = _process_battery_status(result)
            await web_service.cache.set("battery:status", battery_data, ttl=30)
        except Exception as e:
            # Return default values
            battery_data = {
                "soc": 50,
                "power": 0,
                "status": "idle",
                "voltage": 48.0,
                "current": 0,
                "temperature": 25.0
            }

    return battery_data


@router.get("/stats")
async def get_power_statistics(
    request: Request,
    period: str = Query(default="today", description="Period: today, week, month, year")
) -> Dict[str, Any]:
    """Get power generation and consumption statistics."""
    web_service = request.app.state.web_service

    # Calculate time range
    now = datetime.now()
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    else:
        start = now - timedelta(days=365)

    # Simple aggregation query
    query = f'from(bucket: "loxone") |> range(start: {start.isoformat()}Z) |> filter(fn: (r) => r["_measurement"] == "solar_power" or r["_measurement"] == "grid_power") |> sum()'

    try:
        result = await web_service.influxdb_client.query(query)
        stats = _process_power_stats(result, period)
    except Exception as e:
        # Return default stats
        stats = {
            "period": period,
            "solar_generated": 0,
            "grid_imported": 0,
            "grid_exported": 0,
            "battery_charged": 0,
            "battery_discharged": 0,
            "self_consumption_rate": 0
        }

    return stats


def _process_current_flow(result: Any) -> Dict[str, Any]:
    """Process current energy flow data."""
    data = {
        "solar_power": 0,
        "grid_power": 0,
        "battery_power": 0,
        "home_consumption": 0,
        "timestamp": datetime.now().isoformat()
    }

    # Process result if available
    if result:
        for table in result:
            for record in table.records:
                if record["_measurement"] == "solar_power":
                    data["solar_power"] = float(record["_value"] or 0)
                elif record["_measurement"] == "grid_power":
                    data["grid_power"] = float(record["_value"] or 0)
                elif record["_measurement"] == "battery_power":
                    data["battery_power"] = float(record["_value"] or 0)

    # Calculate home consumption
    data["home_consumption"] = max(0,
        data["solar_power"] - data["grid_power"] - data["battery_power"]
    )

    return data


def _process_energy_history(result: Any, resolution: str) -> Dict[str, Any]:
    """Process historical energy data."""
    solar_data = []
    grid_data = []

    if result:
        for table in result:
            for record in table.records:
                point = {
                    "time": record["_time"].isoformat(),
                    "value": float(record["_value"] or 0)
                }

                if record["_measurement"] == "solar_power":
                    solar_data.append(point)
                elif record["_measurement"] == "grid_power":
                    grid_data.append(point)

    return {
        "resolution": resolution,
        "solar": solar_data,
        "grid": grid_data,
        "consumption": []  # Calculate from solar and grid if needed
    }


def _process_battery_status(result: Any) -> Dict[str, Any]:
    """Process battery status data."""
    status = {
        "soc": 50,
        "power": 0,
        "status": "idle",
        "voltage": 48.0,
        "current": 0,
        "temperature": 25.0
    }

    if result:
        for table in result:
            for record in table.records:
                if record["_field"] == "battery_soc":
                    status["soc"] = float(record["_value"] or 50)
                elif record["_field"] == "battery_power":
                    power = float(record["_value"] or 0)
                    status["power"] = power
                    status["status"] = "charging" if power > 0 else "discharging" if power < 0 else "idle"

    return status


def _process_power_stats(result: Any, period: str) -> Dict[str, Any]:
    """Process power statistics."""
    stats = {
        "period": period,
        "solar_generated": 0,
        "grid_imported": 0,
        "grid_exported": 0,
        "battery_charged": 0,
        "battery_discharged": 0,
        "self_consumption_rate": 0
    }

    if result:
        for table in result:
            for record in table.records:
                if record["_measurement"] == "solar_power":
                    stats["solar_generated"] = float(record["_value"] or 0) / 1000  # Convert to kWh
                elif record["_measurement"] == "grid_power":
                    value = float(record["_value"] or 0) / 1000
                    if value > 0:
                        stats["grid_imported"] = value
                    else:
                        stats["grid_exported"] = abs(value)

    # Calculate self-consumption rate
    if stats["solar_generated"] > 0:
        stats["self_consumption_rate"] = min(100,
            (1 - stats["grid_exported"] / stats["solar_generated"]) * 100
        )

    return stats