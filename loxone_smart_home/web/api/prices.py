"""Electricity price API endpoints."""

from datetime import datetime, timedelta
from typing import Any, Dict

from fastapi import APIRouter, Query, Request


router = APIRouter()


@router.get("/current")
async def get_current_price(request: Request) -> Dict[str, Any]:
    """Get current electricity price."""
    web_service = request.app.state.web_service

    # Get cached current price
    current_price = await web_service.cache.get("prices:current")

    if not current_price:
        # Get current 15-minute block
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute // 15 * 15

        # Query current price from InfluxDB
        query = f'''
        from(bucket: "electricity_prices")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "dam_prices")
          |> filter(fn: (r) => r["hour"] == {current_hour} and r["minute"] == {current_minute})
          |> last()
        '''

        result = await web_service.influxdb_client.query(query)
        current_price = _process_current_price(result, now)
        await web_service.cache.set("prices:current", current_price, ttl=60)

    return current_price


@router.get("/forecast")
async def get_price_forecast(
    request: Request,
    hours: int = Query(default=48, ge=1, le=72, description="Forecast hours")
) -> Dict[str, Any]:
    """Get electricity price forecast."""
    web_service = request.app.state.web_service

    # Build cache key
    cache_key = f"prices:forecast:{hours}"

    # Check cache
    cached_forecast = await web_service.cache.get(cache_key)
    if cached_forecast:
        return cached_forecast

    # Query price forecast
    now = datetime.now()
    end_time = now + timedelta(hours=hours)

    query = f'''
    from(bucket: "electricity_prices")
      |> range(start: {now.isoformat()}, stop: {end_time.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "dam_prices")
      |> sort(columns: ["_time"])
    '''

    result = await web_service.influxdb_client.query(query)
    forecast = _process_price_forecast(result, hours)

    # Cache for 5 minutes
    await web_service.cache.set(cache_key, forecast, ttl=300)

    return forecast


@router.get("/schedule")
async def get_optimized_schedule(request: Request) -> Dict[str, Any]:
    """Get optimized battery charging/discharging schedule."""
    web_service = request.app.state.web_service

    # Get cached schedule
    schedule = await web_service.cache.get("prices:schedule")

    if not schedule:
        # Get price forecast
        forecast_data = await get_price_forecast(request, hours=48)

        # Calculate optimal schedule
        schedule = _calculate_optimal_schedule(forecast_data)

        # Cache for 15 minutes
        await web_service.cache.set("prices:schedule", schedule, ttl=900)

    return schedule


@router.get("/savings")
async def get_savings_summary(
    request: Request,
    period: str = Query(default="day", description="Period (day, week, month)")
) -> Dict[str, Any]:
    """Get savings summary from price optimization."""
    web_service = request.app.state.web_service

    # Build cache key
    cache_key = f"prices:savings:{period}"

    # Check cache
    cached_savings = await web_service.cache.get(cache_key)
    if cached_savings:
        return cached_savings

    # Calculate time range
    end = datetime.now()
    period_map = {
        "day": timedelta(days=1),
        "week": timedelta(days=7),
        "month": timedelta(days=30)
    }
    start = end - period_map.get(period, timedelta(days=1))

    # Query historical prices and actual charging times
    prices_query = f'''
    from(bucket: "electricity_prices")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "dam_prices")
    '''

    charging_query = f'''
    from(bucket: "solar")
      |> range(start: {start.isoformat()}, stop: {end.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "battery" and r["_field"] == "charging")
      |> filter(fn: (r) => r["_value"] > 0)
    '''

    # Execute queries
    prices = await web_service.influxdb_client.query(prices_query)
    charging = await web_service.influxdb_client.query(charging_query)

    # Calculate savings
    savings = _calculate_savings(prices, charging, period)

    # Cache for 1 hour
    await web_service.cache.set(cache_key, savings, ttl=3600)

    return savings


@router.get("/comparison")
async def get_price_comparison(request: Request) -> Dict[str, Any]:
    """Compare today's and tomorrow's prices."""
    web_service = request.app.state.web_service

    # Get cached comparison
    comparison = await web_service.cache.get("prices:comparison")

    if not comparison:
        now = datetime.now()
        today = now.date()
        tomorrow = today + timedelta(days=1)

        # Query today's prices
        today_query = f'''
        from(bucket: "electricity_prices")
          |> range(start: {today.isoformat()}T00:00:00Z, stop: {today.isoformat()}T23:59:59Z)
          |> filter(fn: (r) => r["_measurement"] == "dam_prices")
        '''

        # Query tomorrow's prices
        tomorrow_query = f'''
        from(bucket: "electricity_prices")
          |> range(start: {tomorrow.isoformat()}T00:00:00Z, stop: {tomorrow.isoformat()}T23:59:59Z)
          |> filter(fn: (r) => r["_measurement"] == "dam_prices")
        '''

        today_prices = await web_service.influxdb_client.query(today_query)
        tomorrow_prices = await web_service.influxdb_client.query(tomorrow_query)

        comparison = _compare_prices(today_prices, tomorrow_prices)

        # Cache for 15 minutes
        await web_service.cache.set("prices:comparison", comparison, ttl=900)

    return comparison


def _process_current_price(result: Any, now: datetime) -> Dict[str, Any]:
    """Process current price query result."""
    # TODO: Implement actual processing
    current_minute = now.minute // 15 * 15
    next_minute = (current_minute + 15) % 60
    next_hour = now.hour if next_minute > current_minute else (now.hour + 1) % 24

    return {
        "timestamp": now.isoformat(),
        "block": f"{now.hour:02d}:{current_minute:02d}-{next_hour:02d}:{next_minute:02d}",
        "price_eur_mwh": 85.50,
        "price_czk_kwh": 2.14,
        "level": "medium",  # low, medium, high
        "next_change": {
            "time": (now + timedelta(minutes=15 - now.minute % 15)).isoformat(),
            "price_czk_kwh": 2.35,
            "direction": "up"
        }
    }


def _process_price_forecast(result: Any, hours: int) -> Dict[str, Any]:
    """Process price forecast query result."""
    # TODO: Implement actual processing
    now = datetime.now()
    blocks = []

    for i in range(hours * 4):  # 4 blocks per hour
        block_time = now + timedelta(minutes=15 * i)
        price = 80 + (i % 8) * 5  # Simulated price variation

        blocks.append({
            "timestamp": block_time.isoformat(),
            "price_eur_mwh": price,
            "price_czk_kwh": price * 25 / 1000,
            "level": _get_price_level(price)
        })

    return {
        "forecast_hours": hours,
        "blocks": blocks,
        "summary": {
            "min_price": min(b["price_czk_kwh"] for b in blocks),
            "max_price": max(b["price_czk_kwh"] for b in blocks),
            "avg_price": sum(b["price_czk_kwh"] for b in blocks) / len(blocks)
        }
    }


def _calculate_optimal_schedule(forecast: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimal charging/discharging schedule."""
    # TODO: Implement actual optimization logic
    blocks = forecast["blocks"]

    # Find cheapest blocks for charging
    sorted_blocks = sorted(blocks, key=lambda x: x["price_czk_kwh"])
    charging_blocks = sorted_blocks[:8]  # 2 hours of charging

    # Find most expensive blocks for discharging
    expensive_blocks = sorted(blocks, key=lambda x: x["price_czk_kwh"], reverse=True)
    discharge_blocks = [b for b in expensive_blocks[:12] if b["price_czk_kwh"] > 3.0]

    return {
        "charging": {
            "blocks": charging_blocks,
            "total_hours": len(charging_blocks) / 4,
            "avg_price": sum(b["price_czk_kwh"] for b in charging_blocks) / len(charging_blocks)
        },
        "discharging": {
            "blocks": discharge_blocks,
            "total_hours": len(discharge_blocks) / 4,
            "avg_price": (
                sum(b["price_czk_kwh"] for b in discharge_blocks) / len(discharge_blocks)
                if discharge_blocks else 0
            )
        },
        "estimated_savings": {
            "daily": 125.50,  # CZK
            "monthly": 3765.00
        }
    }


def _calculate_savings(prices: Any, charging: Any, period: str) -> Dict[str, Any]:
    """Calculate actual savings from optimized charging."""
    # TODO: Implement actual calculation
    return {
        "period": period,
        "actual_cost": 450.25,  # CZK
        "baseline_cost": 675.50,  # Without optimization
        "savings": 225.25,
        "savings_percentage": 33.4,
        "charging_sessions": 12,
        "avg_charging_price": 1.85,  # CZK/kWh
        "avg_market_price": 2.45,
        "best_session": {
            "timestamp": datetime.now().isoformat(),
            "price": 1.25,
            "energy": 8.5,  # kWh
            "savings": 10.20
        }
    }


def _compare_prices(today_prices: Any, tomorrow_prices: Any) -> Dict[str, Any]:
    """Compare today and tomorrow prices."""
    # TODO: Implement actual comparison
    return {
        "today": {
            "min": 1.85,
            "max": 3.45,
            "avg": 2.35,
            "cheapest_hours": ["02:00-04:00", "14:00-15:00"]
        },
        "tomorrow": {
            "min": 1.65,
            "max": 3.25,
            "avg": 2.15,
            "cheapest_hours": ["01:00-03:00", "13:00-14:00"],
            "available": True
        },
        "recommendation": {
            "defer_charging": True,
            "reason": "Tomorrow prices are 8.5% cheaper on average",
            "optimal_blocks": ["01:00-01:15", "01:15-01:30", "01:30-01:45"]
        }
    }


def _get_price_level(price_eur_mwh: float) -> str:
    """Determine price level."""
    if price_eur_mwh < 60:
        return "low"
    elif price_eur_mwh < 100:
        return "medium"
    else:
        return "high"
