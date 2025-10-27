"""Fixed Energy API endpoints with correct InfluxDB queries."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
import pytz

from utils.schedule_calculator import calculate_optimal_schedule, determine_block_mode
from modules.growatt.price_analyzer import PriceAnalyzer

logger = logging.getLogger(__name__)

# Prague timezone for local time display
PRAGUE_TZ = ZoneInfo("Europe/Prague")
PRAGUE_TZ_PYTZ = pytz.timezone("Europe/Prague")

from ..models.responses import (
    BatteryStatusResponse,
    EnergyCurrentResponse,
    EnergyHistoryResponse,
    ScheduleTableResponse
)

router = APIRouter()


@router.get("/current", response_model=EnergyCurrentResponse)
async def get_current_energy_flow(request: Request) -> Dict[str, Any]:
    """Get current energy flow data from Growatt telemetry."""
    web_service = request.app.state.web_service

    # Get latest data from cache
    current_data = await web_service.cache.get("energy:current")

    if not current_data:
        try:
            # Query latest solar data from Growatt telemetry
            query = '''from(bucket: "solar")
  |> range(start: -10m)
  |> filter(fn: (r) => r["_measurement"] == "solar")
  |> last()'''
            result = await web_service.influxdb_client.query(query)
            current_data = _process_current_flow(result)
        except Exception as e:
            # Return demo data if query fails
            current_data = {
                "solar_power": 2500,  # Demo: 2.5kW solar generation
                "grid_power": -500,   # Demo: 500W export to grid (negative = export)
                "battery_power": 1000,  # Demo: 1kW charging battery
                "home_power": 1000,    # Demo: 1kW home consumption
                "battery_soc": 75,     # Demo: 75% battery charge
                "timestamp": datetime.now().isoformat()
            }
        # Cache for 5 seconds
        await web_service.cache.set("energy:current", current_data, ttl=5)

    return current_data


@router.get("/history")
async def get_energy_history(
    request: Request,
    period: str = Query(default="24h", description="Time period: 1h, 6h, 24h, 7d, 30d"),
    resolution: str = Query(default="15m", description="Data resolution: 15m, 1h, 1d")
) -> Dict[str, Any]:
    """Get historical energy data from Growatt telemetry."""
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

    # Build query for solar bucket with Growatt telemetry - filter specific fields
    query = f'''from(bucket: "solar")
  |> range(start: {start.isoformat()}Z)
  |> filter(fn: (r) => r["_measurement"] == "solar")
  |> filter(fn: (r) => r["_field"] == "InputPower" or r["_field"] == "ACPowerToGrid" or r["_field"] == "ACPowerToUser")
  |> aggregateWindow(every: {resolution}, fn: mean, createEmpty: false)'''

    try:
        result = await web_service.influxdb_client.query(query)
        history_data = _process_energy_history(result, resolution)

        # Set the correct period
        history_data["period"] = period

        # If no data returned, fall back to demo data
        if not history_data.get("data") or len(history_data.get("data", [])) == 0:
            raise Exception("No data returned from query")

    except Exception as e:
        # Return demo data on error or empty result
        # Generate demo data points for the last 24 hours
        data_points = []
        for i in range(24):
            timestamp = now - timedelta(hours=24-i)
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "production": 2.5 + (i % 6) * 0.5 if 6 <= i <= 18 else 0,  # Solar only during day
                "consumption": 1.5 + (i % 4) * 0.3,
                "grid_import": 0.5 if i < 6 or i > 18 else 0,
                "grid_export": 1.0 if 10 <= i <= 16 else 0
            })

        history_data = {
            "period": period,
            "resolution": resolution,
            "data": data_points
        }

    return history_data


@router.get("/battery/status", response_model=BatteryStatusResponse)
async def get_battery_status(request: Request) -> Dict[str, Any]:
    """Get current battery status from Growatt telemetry."""
    web_service = request.app.state.web_service

    # Get cached battery data
    battery_data = await web_service.cache.get("battery:status")

    if not battery_data:
        try:
            # Query latest battery status from Growatt telemetry
            query = '''from(bucket: "solar")
  |> range(start: -10m)
  |> filter(fn: (r) => r["_measurement"] == "solar")
  |> last()'''
            result = await web_service.influxdb_client.query(query)
            battery_data = _process_battery_status(result)
        except Exception as e:
            # Return demo battery data if query fails
            battery_data = {
                "soc": 75,
                "power": 1000,
                "status": "charging",
                "voltage": 52.0,
                "current": 20.0,
                "temperature": 28.0,
                "health": 95
            }
        await web_service.cache.set("battery:status", battery_data, ttl=30)

    return battery_data


@router.get("/statistics")
async def get_power_statistics(
    request: Request,
    period: str = Query(default="today", description="Period: today, week, month, year")
) -> Dict[str, Any]:
    """Get power generation and consumption statistics from Growatt telemetry."""
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

    # Query Growatt telemetry with aggregation
    query = f'''from(bucket: "solar")
  |> range(start: {start.isoformat()}Z)
  |> filter(fn: (r) => r["_measurement"] == "solar")
  |> sum()'''

    try:
        result = await web_service.influxdb_client.query(query)
        stats = _process_power_stats(result, period)
    except Exception as e:
        # Return demo stats in the format expected by the frontend
        stats = {
            "period": period,
            "production": {
                "total": 24.5,  # kWh
                "peak": 3.8,
                "average": 2.1
            },
            "consumption": {
                "total": 18.3,
                "peak": 4.2,
                "average": 1.8
            },
            "grid": {
                "import": 3.2,  # Named 'import' but accessed via 'grid.import'
                "export": 9.4,
                "net": -6.2
            },
            "self_sufficiency": 82,
            "self_consumption": 75,
            "savings": {
                "amount": 145.50,  # CZK
                "co2_avoided": 12.3  # kg
            }
        }

    return stats


def _process_current_flow(result: Any) -> Dict[str, Any]:
    """Process current energy flow data from Growatt telemetry."""
    data = {
        "solar_power": 0,
        "grid_power": 0,
        "battery_power": 0,
        "home_power": 0,
        "battery_soc": 0,
        "timestamp": datetime.now().isoformat()
    }

    # Temporary storage for charge/discharge values
    charge_power = 0.0
    discharge_power = 0.0

    # Process Growatt telemetry fields
    if result:
        for table in result:
            for record in table.records:
                field_name = record["_field"]
                value = float(record["_value"] or 0)

                if field_name == "InputPower":
                    # Solar generation in watts
                    data["solar_power"] = value
                elif field_name == "ACPowerToGrid":
                    # Positive = export to grid, negative = import from grid
                    data["grid_power"] = value
                elif field_name == "ACPowerToUser":
                    # Power delivered to home in watts
                    data["home_power"] = value
                elif field_name == "ChargePower":
                    # Battery charging power
                    charge_power = value
                elif field_name == "DischargePower":
                    # Battery discharging power
                    discharge_power = value
                elif field_name == "SOC":
                    # Battery state of charge (%)
                    data["battery_soc"] = value

    # Calculate net battery power: positive = charging, negative = discharging
    data["battery_power"] = charge_power - discharge_power

    return data


def _process_energy_history(result: Any, resolution: str) -> Dict[str, Any]:
    """Process historical energy data from Growatt telemetry into chart format."""
    # Collect data by timestamp
    data_by_time = {}

    if result:
        for table in result:
            for record in table.records:
                timestamp = record["_time"].isoformat()
                field_name = record["_field"]
                value = float(record["_value"] or 0) / 1000.0  # Convert W to kW

                if timestamp not in data_by_time:
                    data_by_time[timestamp] = {
                        "timestamp": timestamp,
                        "production": 0,
                        "consumption": 0,
                        "grid_import": 0,
                        "grid_export": 0
                    }

                if field_name == "InputPower":
                    data_by_time[timestamp]["production"] = value
                elif field_name == "ACPowerToUser":
                    data_by_time[timestamp]["consumption"] = value
                elif field_name == "ACPowerToGrid":
                    # Positive = export, negative = import
                    if value > 0:
                        data_by_time[timestamp]["grid_export"] = value
                    else:
                        data_by_time[timestamp]["grid_import"] = abs(value)

    # Convert to sorted list
    data_points = sorted(data_by_time.values(), key=lambda x: x["timestamp"])

    return {
        "period": "custom",  # Will be overwritten by caller
        "resolution": resolution,
        "data": data_points,
        # Also include separate arrays for compatibility
        "solar": [{"time": p["timestamp"], "value": p["production"]} for p in data_points],
        "grid": [{"time": p["timestamp"], "value": p.get("grid_export", 0) - p.get("grid_import", 0)} for p in data_points],
        "home": [{"time": p["timestamp"], "value": p["consumption"]} for p in data_points]
    }


def _process_battery_status(result: Any) -> Dict[str, Any]:
    """Process battery status data from Growatt telemetry."""
    status = {
        "soc": 50,
        "power": 0,
        "status": "idle",
        "voltage": 48.0,
        "current": 0,
        "temperature": 25.0,
        "health": 95
    }

    charge_power = 0.0
    discharge_power = 0.0

    # Process Growatt telemetry fields
    if result:
        for table in result:
            for record in table.records:
                field_name = record["_field"]
                value = float(record["_value"] or 0)

                if field_name == "SOC":
                    status["soc"] = value
                elif field_name == "ChargePower":
                    charge_power = value
                elif field_name == "DischargePower":
                    discharge_power = value
                elif field_name == "BatteryVoltage":
                    status["voltage"] = value
                elif field_name == "BatteryTemperature":
                    status["temperature"] = value

    # Calculate net battery power and status
    net_power = charge_power - discharge_power
    status["power"] = net_power
    status["status"] = "charging" if net_power > 0 else "discharging" if net_power < 0 else "idle"

    return status


def _process_power_stats(result: Any, period: str) -> Dict[str, Any]:
    """Process power statistics from Growatt telemetry."""
    stats = {
        "period": period,
        "production": {
            "total": 0,
            "peak": 0,
            "average": 0
        },
        "consumption": {
            "total": 0,
            "peak": 0,
            "average": 0
        },
        "grid": {
            "import": 0,
            "export": 0,
            "net": 0
        },
        "self_sufficiency": 0,
        "self_consumption": 0,
        "savings": {
            "amount": 0,
            "co2_avoided": 0
        }
    }

    input_power_sum = 0.0
    ac_power_to_user_sum = 0.0
    ac_power_to_grid_sum = 0.0
    count = 0

    # Aggregate Growatt telemetry fields
    if result:
        for table in result:
            for record in table.records:
                field_name = record["_field"]
                value = float(record["_value"] or 0)

                if field_name == "InputPower":
                    input_power_sum += value
                elif field_name == "ACPowerToUser":
                    ac_power_to_user_sum += value
                elif field_name == "ACPowerToGrid":
                    ac_power_to_grid_sum += value
                count += 1

    # Convert from Wh to kWh (divide by 1000)
    stats["production"]["total"] = input_power_sum / 1000.0
    stats["consumption"]["total"] = ac_power_to_user_sum / 1000.0

    # Grid power: positive = export, negative = import
    if ac_power_to_grid_sum > 0:
        stats["grid"]["export"] = ac_power_to_grid_sum / 1000.0
    else:
        stats["grid"]["import"] = abs(ac_power_to_grid_sum) / 1000.0

    stats["grid"]["net"] = ac_power_to_grid_sum / 1000.0

    # Calculate averages if we have data
    if count > 0:
        stats["production"]["average"] = stats["production"]["total"] / count
        stats["consumption"]["average"] = stats["consumption"]["total"] / count
        stats["production"]["peak"] = stats["production"]["total"]  # Simplified peak

    # Calculate self-consumption metrics
    if stats["production"]["total"] > 0:
        stats["self_consumption"] = min(100,
            (stats["consumption"]["total"] / stats["production"]["total"]) * 100
        )
        stats["self_sufficiency"] = max(0,
            (stats["consumption"]["total"] - stats["grid"]["import"]) / stats["consumption"]["total"] * 100
        ) if stats["consumption"]["total"] > 0 else 0

    # Estimate savings (CZK per kWh * export)
    czk_per_kwh = 3.5  # Approximate electricity price
    stats["savings"]["amount"] = stats["grid"]["export"] * czk_per_kwh

    return stats


@router.get("/schedule", response_model=ScheduleTableResponse)
async def get_energy_schedule(request: Request) -> Dict[str, Any]:
    """Get energy schedule table showing prices and mode decisions for each 15-min block.

    Returns a table that is an EXACT COPY of what's shown in Growatt controller logs:
    - Prices for each 15-minute block
    - Mode indicators (🔋=charge, 🔌=pre-discharge, ⚡=discharge)
    - Today and tomorrow schedules

    Uses the Growatt controller's current state directly - no recalculation, no duplicate
    API calls. This ensures 100% consistency between logs and web UI.
    """
    web_service = request.app.state.web_service

    # Check cache (very short TTL so updates show immediately)
    schedule = await web_service.cache.get("energy:schedule")
    if schedule:
        return schedule

    try:
        # Get schedule data from InfluxDB (written by Growatt controller)
        # Look back 24 hours to ensure we find the schedule
        query = '''from(bucket: "solar")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "growatt_schedule")
  |> filter(fn: (r) => r["_field"] == "schedule_json")
  |> last()'''

        result = await web_service.influxdb_client.query(query)
        schedule_data = None

        logger.debug(f"Querying InfluxDB for schedule data, result: {result}")

        # Parse InfluxDB result
        if result:
            for table in result:
                for record in table.records:
                    if record["_field"] == "schedule_json":
                        import json
                        schedule_json = json.loads(record["_value"])

                        # Convert back to the format expected by formatter
                        # Convert string keys back to tuples
                        today_prices = {}
                        for k, v in schedule_json.get("today_prices", {}).items():
                            parts = k.split('-')
                            today_prices[(parts[0], parts[1])] = v

                        tomorrow_prices = {}
                        for k, v in schedule_json.get("tomorrow_prices", {}).items():
                            parts = k.split('-')
                            tomorrow_prices[(parts[0], parts[1])] = v

                        # Convert lists back to sets
                        schedule_data = {
                            "today_prices": today_prices,
                            "tomorrow_prices": tomorrow_prices,
                            "today_date": schedule_json.get("today_date"),
                            "tomorrow_date": schedule_json.get("tomorrow_date"),
                            "charging_blocks_today": {tuple(item) for item in schedule_json.get("charging_blocks_today", [])},
                            "charging_blocks_tomorrow": {tuple(item) for item in schedule_json.get("charging_blocks_tomorrow", [])},
                            "pre_discharge_blocks_today": {tuple(item) for item in schedule_json.get("pre_discharge_blocks_today", [])},
                            "pre_discharge_blocks_tomorrow": {tuple(item) for item in schedule_json.get("pre_discharge_blocks_tomorrow", [])},
                            "discharge_periods_today": {tuple(item) for item in schedule_json.get("discharge_periods_today", [])},
                            "discharge_periods_tomorrow": {tuple(item) for item in schedule_json.get("discharge_periods_tomorrow", [])},
                            "eur_czk_rate": schedule_json.get("eur_czk_rate", 25.0),
                            "prices_fetched": schedule_json.get("prices_fetched", False),
                            "next_day_prices_fetched": schedule_json.get("next_day_prices_fetched", False),
                        }
                        break

        if not schedule_data:
            logger.warning(
                f"No schedule data found in InfluxDB. "
                f"Query returned {len(result) if result else 0} tables. "
                f"Check if Growatt controller is writing schedule data."
            )
            raise Exception("Schedule data not available")

        # Convert to web API format
        now = datetime.now(PRAGUE_TZ)
        schedule = _format_schedule_from_controller(schedule_data, now)

        logger.info(
            f"Fetched schedule from InfluxDB: "
            f"{len(schedule_data.get('today_prices', {}))} today blocks, "
            f"{len(schedule_data.get('tomorrow_prices', {}))} tomorrow blocks"
        )

    except Exception as e:
        # Log error and return demo schedule
        logger.error(f"Failed to get schedule from InfluxDB: {e}", exc_info=True)
        now_fallback = datetime.now(PRAGUE_TZ)
        schedule = _generate_demo_schedule(now_fallback)

    # Cache for 30 seconds (short TTL so updates show quickly)
    await web_service.cache.set("energy:schedule", schedule, ttl=30)

    return schedule


def _format_schedule_from_controller(
    schedule_data: Dict[str, Any],
    now: datetime
) -> Dict[str, Any]:
    """Format schedule data from Growatt controller for web API response.

    This uses the EXACT data from the controller's state - the same data shown in logs.

    Args:
        schedule_data: Data from controller's get_schedule_table_data()
        now: Current time

    Returns:
        Formatted schedule for web API response
    """
    today = now.date()
    tomorrow = today + timedelta(days=1)

    # Extract data from controller state
    today_prices = schedule_data.get("today_prices", {})
    tomorrow_prices = schedule_data.get("tomorrow_prices", {})
    eur_czk_rate = schedule_data.get("eur_czk_rate", 25.0)

    # Get schedule blocks (these are sets of (start, end) tuples)
    charge_today = schedule_data.get("charging_blocks_today", set())
    charge_tomorrow = schedule_data.get("charging_blocks_tomorrow", set())
    pre_discharge_today = schedule_data.get("pre_discharge_blocks_today", set())
    pre_discharge_tomorrow = schedule_data.get("pre_discharge_blocks_tomorrow", set())
    discharge_today = schedule_data.get("discharge_periods_today", set())
    discharge_tomorrow = schedule_data.get("discharge_periods_tomorrow", set())

    days = []

    # Format today's schedule
    if today_prices:
        today_schedule = _format_day_schedule_from_controller(
            today_prices,
            today,
            "TODAY",
            charge_today,
            pre_discharge_today,
            discharge_today,
            eur_czk_rate,
            now
        )
        days.append(today_schedule)

    # Format tomorrow's schedule
    if tomorrow_prices:
        tomorrow_schedule = _format_day_schedule_from_controller(
            tomorrow_prices,
            tomorrow,
            "TOMORROW",
            charge_tomorrow,
            pre_discharge_tomorrow,
            discharge_tomorrow,
            eur_czk_rate,
            now
        )
        days.append(tomorrow_schedule)

    # Build legend
    legend = [
        {"icon": "🔋", "label": "Regular charge", "color": "#4CAF50"},
        {"icon": "🔌", "label": "Pre-discharge charge", "color": "#FF9800"},
        {"icon": "⚡", "label": "Discharge", "color": "#F44336"},
        {"icon": "-", "label": "Normal", "color": "#9E9E9E"}
    ]

    # Summary statistics
    total_charge = len(charge_today) + len(charge_tomorrow)
    total_pre_discharge = len(pre_discharge_today) + len(pre_discharge_tomorrow)
    total_discharge = len(discharge_today) + len(discharge_tomorrow)

    summary = {
        "charge_blocks": total_charge,
        "pre_discharge_blocks": total_pre_discharge,
        "discharge_blocks": total_discharge,
        "prices_fetched": schedule_data.get("prices_fetched", False),
        "next_day_prices_fetched": schedule_data.get("next_day_prices_fetched", False)
    }

    return {
        "days": days,
        "legend": legend,
        "summary": summary
    }


def _format_day_schedule_from_controller(
    prices: Dict[Tuple[str, str], float],
    date: Any,
    label: str,
    charge_blocks: set,
    pre_discharge_blocks: set,
    discharge_blocks: set,
    eur_czk_rate: float,
    now: datetime
) -> Dict[str, Any]:
    """Format a single day's schedule from controller data.

    Args:
        prices: Dict of (start, end) -> price_eur
        date: Date object for this day
        label: Label for the day (e.g., "TODAY", "TOMORROW")
        charge_blocks: Set of (start, end) tuples for charging
        pre_discharge_blocks: Set of (start, end) tuples for pre-discharge charging
        discharge_blocks: Set of (start, end) tuples for discharge
        eur_czk_rate: EUR to CZK conversion rate
        now: Current datetime

    Returns:
        Formatted day schedule dict
    """
    hours_dict: Dict[int, List[Dict[str, Any]]] = {}

    # Sort price blocks by time
    sorted_blocks = sorted(prices.items(), key=lambda x: x[0][0])

    for (start_str, end_str), price_eur in sorted_blocks:
        # Parse time "HH:MM"
        hour = int(start_str.split(':')[0])
        minute = int(start_str.split(':')[1])

        # Show ALL prices (no filtering of past blocks) - same as controller logs

        # Determine mode (exact same logic as controller logs)
        if (start_str, end_str) in pre_discharge_blocks:
            mode, icon = "pre-discharge", "🔌"
        elif (start_str, end_str) in charge_blocks:
            mode, icon = "charge", "🔋"
        elif (start_str, end_str) in discharge_blocks:
            mode, icon = "discharge", "⚡"
        else:
            mode, icon = "normal", "-"

        # Convert price EUR/MWh to CZK/kWh (same as controller)
        price_czk_kwh = price_eur * eur_czk_rate / 1000

        # Format time block
        time_block = f"{start_str}-{end_str}"

        block_data = {
            "time": time_block,
            "price_czk_kwh": round(price_czk_kwh, 3),
            "mode": mode,
            "icon": icon
        }

        if hour not in hours_dict:
            hours_dict[hour] = []

        hours_dict[hour].append(block_data)

    # Convert to list of hours
    hours = []
    for hour in sorted(hours_dict.keys()):
        hours.append({
            "hour": hour,
            "blocks": hours_dict[hour]
        })

    return {
        "date": date.isoformat(),
        "label": label,
        "hours": hours
    }


def _process_schedule_from_ote(
    today_prices_dict: Dict[Tuple[str, str], float],
    tomorrow_prices_dict: Dict[Tuple[str, str], float],
    today: Any,
    tomorrow: Any,
    now: datetime,
    web_service: Any
) -> Dict[str, Any]:
    """Process OTE price data into schedule table format.

    Uses the SAME conversion and logic as Growatt controller:
    - Converts EUR/MWh to CZK/kWh
    - Uses same EUR_CZK_RATE from settings
    - Uses shared scheduling logic from utils.schedule_calculator
    """
    # Get EUR to CZK rate from settings
    eur_czk_rate = web_service.settings.ote.eur_czk_rate  # Should be 25.0

    # Convert OTE price format to (datetime, price_czk) format
    all_blocks = []

    # Process today's prices
    for (start_time, end_time), price_eur_mwh in today_prices_dict.items():
        # Parse time string "HH:MM"
        hour, minute = map(int, start_time.split(':'))
        # Create datetime in Prague timezone
        dt = PRAGUE_TZ_PYTZ.localize(datetime(today.year, today.month, today.day, hour, minute))
        # Convert EUR/MWh to CZK/kWh (same as Growatt controller)
        price_czk_kwh = price_eur_mwh * eur_czk_rate / 1000
        all_blocks.append((dt, price_czk_kwh))

    # Process tomorrow's prices
    for (start_time, end_time), price_eur_mwh in tomorrow_prices_dict.items():
        # Parse time string "HH:MM"
        hour, minute = map(int, start_time.split(':'))
        # Create datetime in Prague timezone
        dt = PRAGUE_TZ_PYTZ.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute))
        # Convert EUR/MWh to CZK/kWh
        price_czk_kwh = price_eur_mwh * eur_czk_rate / 1000
        all_blocks.append((dt, price_czk_kwh))

    logger.info(f"Converted {len(all_blocks)} total blocks from OTE format to schedule format")

    if not all_blocks:
        return {"days": [], "legend": [], "summary": {}}

    # Use shared scheduling logic (same as Growatt controller)
    CHARGE_BLOCKS_COUNT = getattr(web_service.settings.growatt, 'battery_charge_blocks', 8)
    DISCHARGE_THRESHOLD_CZK = getattr(web_service.settings.growatt, 'discharge_price_min', 3.0)

    charge_times, discharge_times, charge_threshold, discharge_threshold = calculate_optimal_schedule(
        all_blocks,
        charge_blocks_count=CHARGE_BLOCKS_COUNT,
        discharge_threshold_czk=DISCHARGE_THRESHOLD_CZK
    )

    logger.info(
        f"Schedule logic: {len(charge_times)} charge blocks (threshold: {charge_threshold:.3f} CZK/kWh), "
        f"{len(discharge_times)} discharge blocks (threshold: {discharge_threshold:.1f} CZK/kWh)"
    )

    # Separate blocks by day and format schedule
    today_blocks = [(t, p) for t, p in all_blocks if t.date() == today]
    tomorrow_blocks = [(t, p) for t, p in all_blocks if t.date() == tomorrow]

    days = []

    # Process today
    if today_blocks:
        today_schedule = _format_day_schedule_with_modes(
            today_blocks, today, "TODAY (remaining)",
            charge_times, discharge_times, now
        )
        days.append(today_schedule)

    # Process tomorrow
    if tomorrow_blocks:
        tomorrow_schedule = _format_day_schedule_with_modes(
            tomorrow_blocks, tomorrow, "TOMORROW",
            charge_times, discharge_times, now
        )
        days.append(tomorrow_schedule)

    # Build legend
    legend = [
        {"icon": "🔋", "label": "Regular charge", "color": "#4CAF50"},
        {"icon": "🔌", "label": "Pre-discharge charge", "color": "#FF9800"},
        {"icon": "⚡", "label": "Discharge", "color": "#F44336"},
        {"icon": "-", "label": "Normal", "color": "#9E9E9E"}
    ]

    # Summary statistics
    summary = {
        "charge_blocks": len(charge_times),
        "discharge_blocks": len(discharge_times),
        "charge_threshold": round(charge_threshold, 3),
        "discharge_threshold": discharge_threshold
    }

    return {
        "days": days,
        "legend": legend,
        "summary": summary
    }


def _process_schedule_table(result: Any, now: datetime) -> Dict[str, Any]:
    """Process InfluxDB result into schedule table format.

    DEPRECATED: This function is no longer used. Web API now fetches directly from OTE API
    using PriceAnalyzer to match Growatt controller's data source.

    Uses shared scheduling logic from utils.schedule_calculator that matches
    the Growatt controller's optimization algorithm.
    """
    today = now.date()
    tomorrow = today + timedelta(days=1)

    # Collect all price blocks with timestamps
    all_blocks = []

    if result:
        for table in result:
            for record in table.records:
                time_dt_utc = record["_time"]
                # Convert UTC timestamp to Prague local time
                time_dt_local = time_dt_utc.astimezone(PRAGUE_TZ)
                price_czk = float(record["_value"] or 0)
                all_blocks.append((time_dt_local, price_czk))

    logger.info(f"Collected {len(all_blocks)} total blocks from InfluxDB")

    if not all_blocks:
        return {"days": [], "legend": [], "summary": {}}

    # Use shared scheduling logic (reused by Growatt controller)
    CHARGE_BLOCKS_COUNT = 8  # Default battery_charge_blocks from settings
    DISCHARGE_THRESHOLD_CZK = 3.0  # Default discharge_price_min from settings

    charge_times, discharge_times, charge_threshold, discharge_threshold = calculate_optimal_schedule(
        all_blocks,
        charge_blocks_count=CHARGE_BLOCKS_COUNT,
        discharge_threshold_czk=DISCHARGE_THRESHOLD_CZK
    )

    logger.info(
        f"Schedule logic: {len(charge_times)} charge blocks (threshold: {charge_threshold:.3f} CZK/kWh), "
        f"{len(discharge_times)} discharge blocks (threshold: {discharge_threshold:.1f} CZK/kWh)"
    )

    # === STEP 3: Separate blocks by day and format schedule ===
    today_blocks = [(t, p) for t, p in all_blocks if t.date() == today]
    tomorrow_blocks = [(t, p) for t, p in all_blocks if t.date() == tomorrow]

    days = []

    # Process today
    if today_blocks:
        today_schedule = _format_day_schedule_with_modes(
            today_blocks, today, "TODAY (remaining)",
            charge_times, discharge_times, now
        )
        days.append(today_schedule)

    # Process tomorrow
    if tomorrow_blocks:
        tomorrow_schedule = _format_day_schedule_with_modes(
            tomorrow_blocks, tomorrow, "TOMORROW",
            charge_times, discharge_times, now
        )
        days.append(tomorrow_schedule)

    # Build legend
    legend = [
        {"icon": "🔋", "label": "Regular charge", "color": "#4CAF50"},
        {"icon": "🔌", "label": "Pre-discharge charge", "color": "#FF9800"},
        {"icon": "⚡", "label": "Discharge", "color": "#F44336"},
        {"icon": "-", "label": "Normal", "color": "#9E9E9E"}
    ]

    # Summary statistics
    summary = {
        "charge_blocks": len(charge_times),
        "discharge_blocks": len(discharge_times),
        "charge_threshold": round(charge_threshold, 3),
        "discharge_threshold": discharge_threshold
    }

    return {
        "days": days,
        "legend": legend,
        "summary": summary
    }


def _format_day_schedule_with_modes(
    blocks: List[Tuple[datetime, float]],
    date: Any,
    label: str,
    charge_block_times: set,
    discharge_block_times: set,
    now: datetime
) -> Dict[str, Any]:
    """Format blocks into day schedule structure using Growatt controller logic."""
    hours_dict: Dict[int, List[Dict[str, Any]]] = {}

    for time, price in blocks:
        # Skip past blocks for "today"
        if label.startswith("TODAY") and time < now:
            continue

        hour = time.hour
        minute = time.minute

        # Determine mode using shared utility function
        mode, icon = determine_block_mode(time, price, charge_block_times, discharge_block_times)

        # Format time block
        time_str = f"{hour:02d}:{minute:02d}"
        next_minute = (minute + 15) % 60
        next_hour = hour if next_minute > minute else (hour + 1) % 24
        time_block = f"{time_str}-{next_hour:02d}:{next_minute:02d}"

        block_data = {
            "time": time_block,
            "price_czk_kwh": round(price, 3),
            "mode": mode,
            "icon": icon
        }

        if hour not in hours_dict:
            hours_dict[hour] = []

        hours_dict[hour].append(block_data)

    # Convert to list of hours
    hours = []
    for hour in sorted(hours_dict.keys()):
        hours.append({
            "hour": hour,
            "blocks": hours_dict[hour]
        })

    return {
        "date": date.isoformat(),
        "label": label,
        "hours": hours
    }


def _generate_demo_schedule(now: datetime) -> Dict[str, Any]:
    """Generate demo schedule data."""
    today = now.date()

    demo_prices = [0.58, 0.61, 0.75, 1.12, 1.45, 1.89,  # 00-05: night
                   2.14, 2.35, 2.68, 3.12, 3.45, 3.68,  # 06-11: morning
                   3.89, 4.12, 4.35, 4.58, 4.84, 4.35,  # 12-17: peak
                   3.89, 3.12, 2.35, 1.68, 1.12, 0.75]  # 18-23: evening

    charge_threshold = 1.5
    discharge_threshold = 4.0

    hours = []
    for hour in range(24):
        blocks = []
        base_price = demo_prices[hour]

        for quarter in range(4):
            minute = quarter * 15
            next_minute = (minute + 15) % 60
            next_hour = hour if next_minute > minute else (hour + 1) % 24

            price = base_price + (quarter * 0.05)  # Slight variation per quarter

            if price <= charge_threshold:
                mode, icon = "charge", "🔋"
            elif price >= discharge_threshold:
                mode, icon = "discharge", "⚡"
            else:
                mode, icon = "normal", "-"

            blocks.append({
                "time": f"{hour:02d}:{minute:02d}-{next_hour:02d}:{next_minute:02d}",
                "price_czk_kwh": round(price, 3),
                "mode": mode,
                "icon": icon
            })

        hours.append({"hour": hour, "blocks": blocks})

    return {
        "days": [{
            "date": today.isoformat(),
            "label": "TODAY (demo data)",
            "hours": hours
        }],
        "legend": [
            {"icon": "🔋", "label": "Regular charge", "color": "#4CAF50"},
            {"icon": "⚡", "label": "Discharge", "color": "#F44336"},
            {"icon": "-", "label": "Normal", "color": "#9E9E9E"}
        ],
        "summary": {
            "charge_blocks": 16,
            "discharge_blocks": 12,
            "charge_threshold": charge_threshold,
            "discharge_threshold": discharge_threshold
        }
    }