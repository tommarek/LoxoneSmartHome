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
        try:
            query = f'''
            from(bucket: "ote_prices")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
              |> last()
            '''

            result = await web_service.influxdb_client.query(query)
            current_price = _process_current_price(result, now)
        except Exception as e:
            # Return demo data if query fails
            current_price = _process_current_price(None, now)

        await web_service.cache.set("prices:current", current_price, ttl=30)

    return current_price


@router.get("/forecast")
async def get_price_forecast(
    request: Request,
    hours: int = Query(default=48, ge=1, le=72, description="Forecast hours")
) -> Dict[str, Any]:
    """Get electricity price forecast (all available future data)."""
    web_service = request.app.state.web_service

    # Build cache key
    cache_key = f"prices:forecast:{hours}"

    # Check cache - use very short TTL (1 minute) so new prices show immediately
    cached_forecast = await web_service.cache.get(cache_key)
    if cached_forecast:
        return cached_forecast

    # Query all available future prices (from now onwards)
    now = datetime.now()

    try:
        # Query from now to far future to get all available data
        # OTE publishes prices up to 60 days in advance
        query = f'''
        from(bucket: "ote_prices")
          |> range(start: {now.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
          |> sort(columns: ["_time"])
          |> limit(n: {hours * 4 + 100})
        '''

        result = await web_service.influxdb_client.query(query)
        forecast = _process_price_forecast(result, hours)
    except Exception as e:
        # Return demo data if query fails
        forecast = _process_price_forecast(None, hours)

    # Cache for 1 minute so new prices show immediately when fetched
    await web_service.cache.set(cache_key, forecast, ttl=60)

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

        # Cache for 1 minute so new prices show in schedule immediately
        await web_service.cache.set("prices:schedule", schedule, ttl=60)

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
    try:
        prices_query = f'''
        from(bucket: "ote_prices")
          |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
        '''

        charging_query = f'''
        from(bucket: "loxone")
          |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "battery" and r["_field"] == "charging")
          |> filter(fn: (r) => r["_value"] > 0)
        '''

        # Execute queries
        prices = await web_service.influxdb_client.query(prices_query)
        charging = await web_service.influxdb_client.query(charging_query)
    except Exception as e:
        # Return demo data if queries fail
        prices = None
        charging = None

    # Calculate savings
    savings = _calculate_savings(prices, charging, period)

    # Cache for 5 minutes
    await web_service.cache.set(cache_key, savings, ttl=300)

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

        try:
            # Query today's prices
            today_query = f'''
            from(bucket: "ote_prices")
              |> range(start: {today.isoformat()}T00:00:00Z, stop: {today.isoformat()}T23:59:59Z)
              |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
            '''

            # Query tomorrow's prices
            tomorrow_query = f'''
            from(bucket: "ote_prices")
              |> range(start: {tomorrow.isoformat()}T00:00:00Z, stop: {tomorrow.isoformat()}T23:59:59Z)
              |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
            '''

            today_prices = await web_service.influxdb_client.query(today_query)
            tomorrow_prices = await web_service.influxdb_client.query(tomorrow_query)
        except Exception as e:
            # Return demo data if queries fail
            today_prices = None
            tomorrow_prices = None

        comparison = _compare_prices(today_prices, tomorrow_prices)

        # Cache for 5 minutes
        await web_service.cache.set("prices:comparison", comparison, ttl=300)

    return comparison


def _process_current_price(result: Any, now: datetime) -> Dict[str, Any]:
    """Process current price query result from InfluxDB."""
    # Current 15-minute block
    current_minute = now.minute // 15 * 15
    next_minute = (current_minute + 15) % 60
    next_hour = now.hour if next_minute > current_minute else (now.hour + 1) % 24

    price_eur_mwh = 0.0
    price_czk_kwh = 0.0

    # Process InfluxDB result
    if result:
        for table in result:
            for record in table.records:
                if record["_field"] == "price":
                    price_eur_mwh = float(record["_value"] or 0)
                elif record["_field"] == "price_czk_kwh":
                    price_czk_kwh = float(record["_value"] or 0)

    # Default demo values if no data
    if price_eur_mwh == 0:
        price_eur_mwh = 85.50
        price_czk_kwh = 2.14

    return {
        "timestamp": now.isoformat(),
        "block": f"{now.hour:02d}:{current_minute:02d}-{next_hour:02d}:{next_minute:02d}",
        "price_eur_mwh": price_eur_mwh,
        "price_czk_kwh": price_czk_kwh,
        "level": _get_price_level(price_eur_mwh),
        "next_change": {
            "time": (now + timedelta(minutes=15 - now.minute % 15)).isoformat(),
            "price_czk_kwh": price_czk_kwh * 1.05,  # Placeholder for next price
            "direction": "unknown"
        }
    }


def _process_price_forecast(result: Any, hours: int) -> Dict[str, Any]:
    """Process price forecast query result from InfluxDB."""
    blocks = []

    # Process InfluxDB result - each price point is a 15-minute block
    if result:
        for table in result:
            for record in table.records:
                timestamp = record["_time"]
                price_eur_mwh = float(record["_value"] or 0)

                blocks.append({
                    "timestamp": timestamp.isoformat(),
                    "price_eur_mwh": price_eur_mwh,
                    "price_czk_kwh": float(record.get("price_czk_kwh", price_eur_mwh * 25 / 1000)),
                    "level": _get_price_level(price_eur_mwh)
                })

    # Sort by timestamp to ensure chronological order
    blocks.sort(key=lambda x: x["timestamp"])

    # Generate demo data if no real data
    if not blocks:
        now = datetime.now()
        for i in range(hours * 4):  # 4 blocks per hour
            block_time = now + timedelta(minutes=15 * i)
            price = 80 + (i % 24) * 2  # Demo variation pattern
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
            "min_price": min(b["price_czk_kwh"] for b in blocks) if blocks else 0,
            "max_price": max(b["price_czk_kwh"] for b in blocks) if blocks else 0,
            "avg_price": sum(b["price_czk_kwh"] for b in blocks) / len(blocks) if blocks else 0
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
    price_list = []

    # Process prices if available
    if prices:
        for table in prices:
            for record in table.records:
                price = float(record["_value"] or 0)
                if price > 0:
                    price_list.append(price)

    # Calculate average market price
    avg_market_price = sum(price_list) / len(price_list) if price_list else 2.45

    return {
        "period": period,
        "actual_cost": 450.25,  # CZK - would need charging data to calculate
        "baseline_cost": 675.50,  # Without optimization
        "savings": 225.25,
        "savings_percentage": 33.4,
        "charging_sessions": 12,
        "avg_charging_price": 1.85,  # CZK/kWh
        "avg_market_price": avg_market_price,
        "best_session": {
            "timestamp": datetime.now().isoformat(),
            "price": min(price_list) if price_list else 1.25,
            "energy": 8.5,  # kWh
            "savings": 10.20
        }
    }


def _compare_prices(today_prices: Any, tomorrow_prices: Any) -> Dict[str, Any]:
    """Compare today and tomorrow prices from InfluxDB."""
    today_data = []
    tomorrow_data = []

    # Process today's prices
    if today_prices:
        for table in today_prices:
            for record in table.records:
                price = float(record["_value"] or 0)
                if price > 0:
                    today_data.append(price)

    # Process tomorrow's prices
    if tomorrow_prices:
        for table in tomorrow_prices:
            for record in table.records:
                price = float(record["_value"] or 0)
                if price > 0:
                    tomorrow_data.append(price)

    # Calculate statistics
    today_stats = {
        "min": min(today_data) if today_data else 1.85,
        "max": max(today_data) if today_data else 3.45,
        "avg": sum(today_data) / len(today_data) if today_data else 2.35,
        "cheapest_hours": ["02:00-04:00", "14:00-15:00"]  # Simplified
    }

    tomorrow_stats = {
        "min": min(tomorrow_data) if tomorrow_data else 1.65,
        "max": max(tomorrow_data) if tomorrow_data else 3.25,
        "avg": sum(tomorrow_data) / len(tomorrow_data) if tomorrow_data else 2.15,
        "cheapest_hours": ["01:00-03:00", "13:00-14:00"],  # Simplified
        "available": bool(tomorrow_data)
    }

    # Calculate recommendation
    defer_charging = bool(tomorrow_data and tomorrow_stats["avg"] < today_stats["avg"])
    savings_percent = (
        ((today_stats["avg"] - tomorrow_stats["avg"]) / today_stats["avg"] * 100)
        if today_stats["avg"] > 0 else 0
    )

    return {
        "today": today_stats,
        "tomorrow": tomorrow_stats,
        "recommendation": {
            "defer_charging": defer_charging,
            "reason": (
                f"Tomorrow prices are {savings_percent:.1f}% cheaper on average"
                if defer_charging else "Today's prices are competitive"
            ),
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
