"""REST API for Growatt controller status and control."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from modules.growatt_controller import GrowattController


async def _query_inverter_mode(
    controller: GrowattController, command: str
) -> Optional[Dict[str, Any]]:
    """Query inverter for mode status via MQTT.

    Args:
        controller: The GrowattController instance
        command: The command to send (e.g., "batteryfirst/get",
                  "gridfirst/get")

    Returns:
        Response data or None if failed
    """
    try:
        if not controller.mqtt_client:
            return None

        # Set up request/response pattern
        request_topic = f"energy/solar/command/{command}"
        response_topic = "energy/solar/result"
        correlation_id = f"{command}-{uuid.uuid4().hex[:8]}"

        # Create future for response
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[Dict[str, Any]] = loop.create_future()

        async def response_handler(_topic: str, payload: Any) -> None:
            try:
                if isinstance(payload, bytes):
                    payload = payload.decode()
                data = json.loads(payload)

                # Check if this is our response
                if data.get("command") == command:
                    if not response_future.done():
                        response_future.set_result(data)
            except Exception:
                pass

        # Subscribe and request
        await controller.mqtt_client.subscribe(
            response_topic, response_handler
        )

        try:
            await controller.mqtt_client.publish(
                request_topic,
                json.dumps({"correlationId": correlation_id})
            )

            # Wait for response with short timeout
            result = await asyncio.wait_for(response_future, timeout=2.0)
            return result.get("value") if result.get("success") else None

        except asyncio.TimeoutError:
            return None
        finally:
            await controller.mqtt_client.unsubscribe(response_topic)

    except Exception:
        return None


def create_growatt_api(
    app: web.Application,
    controller: Optional[GrowattController] = None
) -> None:
    """Create and register Growatt API routes.

    Args:
        app: The aiohttp web application
        controller: The GrowattController instance (can be set later via
                   app['growatt_controller'])
    """
    if controller:
        app['growatt_controller'] = controller

    # Register routes
    app.router.add_get('/api/growatt/status', get_status)
    app.router.add_get('/api/growatt/schedule', get_schedule)
    app.router.add_get('/api/growatt/prices', get_prices)
    app.router.add_post('/api/growatt/mode', set_mode)
    app.router.add_post('/api/growatt/sync-time', sync_time)
    app.router.add_get('/api/growatt/config', get_config)

    # Dashboard routes
    app.router.add_get('/inverter', serve_dashboard)
    app.router.add_get('/inverter/', serve_dashboard)
    app.router.add_get('/inverter/status', get_dashboard_status)


async def get_status(request: web.Request) -> web.Response:
    """Get current Growatt controller status with real-time inverter data.

    Returns:
        JSON with current mode, periods, season mode, and live inverter data.
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        # Get current state
        now = controller._get_local_now()
        now_t = now.time()

        # Find active periods
        active_periods = []
        for period in controller._scheduled_periods:
            if period.contains_time(now_t):
                active_periods.append({
                    "kind": period.kind,
                    "start": period.start.strftime("%H:%M"),
                    "end": period.end.strftime("%H:%M"),
                    "params": period.params
                })

        # Determine primary mode
        active_modes = {p["kind"] for p in active_periods}
        primary_mode = (
            controller._select_primary_mode(active_modes)
            if active_modes else "load_first"
        )

        # Try to get real-time inverter data
        inverter_data = {}
        if not controller._optional_config.get("simulation_mode", False):
            try:
                # Get inverter time for drift calculation
                inverter_time = await controller._get_inverter_time()
                if inverter_time:
                    drift = (now - inverter_time).total_seconds()
                    inverter_data["time"] = inverter_time.isoformat()
                    inverter_data["time_drift_seconds"] = drift

                # Query battery-first mode status
                bf_status = await _query_inverter_mode(
                    controller, "batteryfirst/get"
                )
                if bf_status:
                    inverter_data["battery_first_status"] = bf_status

                # Query grid-first mode status
                gf_status = await _query_inverter_mode(
                    controller, "gridfirst/get"
                )
                if gf_status:
                    inverter_data["grid_first_status"] = gf_status

                # Query active power rate
                power_rate = await _query_inverter_mode(
                    controller, "power/get/activerate"
                )
                if power_rate:
                    inverter_data["active_power_rate"] = power_rate

            except Exception as e:
                inverter_data["error"] = f"Failed to query inverter: {str(e)}"

        status = {
            "running": controller._running,
            "current_mode": primary_mode,
            "active_periods": active_periods,
            "season_mode": controller._season_mode,
            "season_mode_updated": (
                controller._season_mode_updated.isoformat()
                if controller._season_mode_updated else None
            ),
            "ac_enabled": controller._ac_enabled,
            "export_enabled": controller._export_enabled,
            "current_time": now.isoformat(),
            "simulation_mode": controller._optional_config.get(
                "simulation_mode", False
            ),
            "inverter_data": inverter_data
        }

        return web.json_response(status)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def get_schedule(request: web.Request) -> web.Response:
    """Get current schedule.

    Returns:
        JSON with scheduled periods and their parameters.
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        # Convert periods to JSON-serializable format
        schedule = []
        for period in controller._scheduled_periods:
            schedule.append({
                "kind": period.kind,
                "start": period.start.strftime("%H:%M"),
                "end": period.end.strftime("%H:%M"),
                "params": period.params or {}
            })

        # Sort by start time
        schedule.sort(key=lambda x: x["start"])

        # Find currently active period
        now_t = controller._get_local_now().time()
        active = None
        for item in schedule:
            start = datetime.strptime(item["start"], "%H:%M").time()
            end = datetime.strptime(item["end"], "%H:%M").time()

            # Handle midnight wrap
            if start <= end:
                if start <= now_t <= end:
                    active = item
                    break
            else:
                if now_t >= start or now_t <= end:
                    active = item
                    break

        return web.json_response({
            "schedule": schedule,
            "active_now": active,
            "total_periods": len(schedule)
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def get_prices(request: web.Request) -> web.Response:
    """Get energy prices information.

    Returns:
        JSON with current and upcoming energy prices.
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        # This would need access to price data stored in the controller
        # For now, return a placeholder
        return web.json_response({
            "message": "Price data endpoint - to be implemented",
            "eur_czk_rate": controller._eur_czk_rate,
            "eur_czk_rate_updated": (
                controller._eur_czk_rate_updated.isoformat()
                if controller._eur_czk_rate_updated else None
            )
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def set_mode(request: web.Request) -> web.Response:
    """Set inverter mode manually.

    Expected JSON payload:
    {
        "mode": "battery_first" | "grid_first" | "load_first",
        "start": "HH:MM",
        "stop": "HH:MM",
        "params": {
            "stop_soc": 90,
            "power_rate": 100
        }
    }
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        data = await request.json()
        mode = data.get("mode")

        if mode not in ["battery_first", "grid_first", "load_first"]:
            return web.json_response({"error": "Invalid mode"}, status=400)

        # Apply the mode
        if mode == "battery_first":
            start = data.get("start", "00:00")
            stop = data.get("stop", "23:59")
            params = data.get("params", {})
            await controller._set_battery_first(
                start, stop,
                stop_soc=params.get("stop_soc", 90),
                power_rate=params.get("power_rate", 100)
            )
        elif mode == "grid_first":
            start = data.get("start", "00:00")
            stop = data.get("stop", "23:59")
            params = data.get("params", {})
            await controller._set_grid_first(
                start, stop,
                stop_soc=params.get("stop_soc", 20),
                power_rate=params.get("power_rate", 10)
            )
        else:
            await controller._set_load_first()

        return web.json_response({
            "success": True,
            "mode": mode,
            "message": f"Mode set to {mode}"
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def sync_time(request: web.Request) -> web.Response:
    """Sync inverter time with server time.

    Expected JSON payload:
    {
        "force": true  # Force sync even if drift is small
    }
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        data = await request.json() if request.body_exists else {}
        force = data.get("force", False)

        # Get current inverter time
        inverter_time = await controller._get_inverter_time()
        if not inverter_time:
            return web.json_response({
                "success": False,
                "message": "Could not read inverter time"
            }, status=500)

        # Calculate drift
        server_time = controller._get_local_now()
        drift = (server_time - inverter_time).total_seconds()

        # Sync if needed
        synced = False
        if force or abs(drift) > 30:  # Sync if drift > 30 seconds
            synced = await controller._sync_inverter_time()

        return web.json_response({
            "success": synced,
            "inverter_time": inverter_time.isoformat(),
            "server_time": server_time.isoformat(),
            "drift_seconds": drift,
            "synced": synced
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def get_config(request: web.Request) -> web.Response:
    """Get current configuration.

    Returns:
        JSON with current configuration parameters.
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        config = {
            "battery_capacity": getattr(
                controller.config, "battery_capacity", None
            ),
            "max_charge_power": getattr(
                controller.config, "max_charge_power", None
            ),
            "min_soc": getattr(controller.config, "min_soc", None),
            "max_soc": getattr(controller.config, "max_soc", None),
            "export_price_threshold": (
                controller.config.export_price_threshold
            ),
            "battery_charge_hours": (
                controller.config.battery_charge_hours
            ),
            "individual_cheapest_hours": (
                controller.config.individual_cheapest_hours
            ),
            "summer_temp_threshold": controller._optional_config.get(
                "summer_temp_threshold", 15.0
            ),
            "summer_price_threshold": (
                controller.config.summer_price_threshold
            ),
            "temperature_avg_days": controller._optional_config.get(
                "temperature_avg_days", 3
            ),
            "eur_czk_rate": controller._optional_config.get(
                "eur_czk_rate", 25.0
            ),
            "simulation_mode": controller._optional_config.get(
                "simulation_mode", False
            ),
            "device_serial": controller.config.device_serial,
            "schedule_hour": controller.config.schedule_hour,
            "schedule_minute": controller.config.schedule_minute
        }

        return web.json_response(config)

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def serve_dashboard(request: web.Request) -> web.Response:
    """Serve the inverter dashboard HTML."""
    template_path = Path(__file__).parent.parent.parent / "templates" / "inverter_dashboard.html"

    if not template_path.exists():
        return web.Response(
            text="Dashboard template not found",
            status=404
        )

    with open(template_path, 'r') as f:
        html_content = f.read()

    return web.Response(
        text=html_content,
        content_type='text/html'
    )


async def _query_inverter_realtime(
    controller: GrowattController
) -> Dict[str, Any]:
    """Query real-time inverter data via MQTT.

    Returns comprehensive inverter status including power flows,
    battery state, and system parameters.
    """
    data = {}

    try:
        if not controller.mqtt_client:
            return data

        # List of data points to query
        queries = [
            ("power/get/solar", "solar_power"),
            ("power/get/battery", "battery_power"),
            ("power/get/grid", "grid_power"),
            ("power/get/load", "load_power"),
            ("battery/get/soc", "battery_soc"),
            ("battery/get/voltage", "battery_voltage"),
            ("battery/get/current", "battery_current"),
            ("battery/get/temperature", "battery_temp"),
            ("system/get/status", "system_status"),
            ("stats/get/daily/production", "daily_production"),
            ("stats/get/daily/consumption", "daily_consumption"),
            ("stats/get/daily/import", "daily_import"),
            ("stats/get/daily/export", "daily_export"),
        ]

        # Query each data point
        for command, key in queries:
            result = await _query_inverter_mode(controller, command)
            if result is not None:
                data[key] = result

    except Exception as e:
        data["error"] = str(e)

    return data


async def get_dashboard_status(request: web.Request) -> web.Response:
    """Get comprehensive inverter status for dashboard display.

    Returns all data needed for the web dashboard including:
    - Current operating mode and schedule
    - Real-time power flows and battery state
    - Energy prices and optimization status
    - Daily statistics and performance metrics
    """
    controller: GrowattController = request.app.get('growatt_controller')
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"},
            status=503
        )

    try:
        now = controller._get_local_now()
        now_t = now.time()

        # Get real-time inverter data
        realtime_data = await _query_inverter_realtime(controller)

        # Get inverter time and calculate offset
        inverter_time = await controller._get_inverter_time()
        time_offset = None
        if inverter_time:
            time_offset = (now - inverter_time).total_seconds() / 60  # in minutes

        # Find active periods and current mode
        active_periods = []
        for period in controller._scheduled_periods:
            if period.contains_time(now_t):
                active_periods.append({
                    "kind": period.kind,
                    "start": period.start.strftime("%H:%M"),
                    "end": period.end.strftime("%H:%M"),
                    "params": period.params
                })

        # Determine primary mode
        active_modes = {p["kind"] for p in active_periods}
        current_mode = (
            controller._select_primary_mode(active_modes)
            if active_modes else "load_first"
        )

        # Format mode for display
        mode_display = {
            "battery_first": "Battery First",
            "grid_first": "Grid First",
            "load_first": "Load First",
            "ac_charge": "AC Charge",
            "export": "Export Mode"
        }.get(current_mode, current_mode)

        # Get today's schedule
        schedule = []
        for period in sorted(controller._scheduled_periods,
                             key=lambda p: p.start):
            mode_name = {
                "battery_first": "Battery First",
                "grid_first": "Grid First",
                "load_first": "Load First",
                "ac_charge": "AC Charge",
                "export": "Export"
            }.get(period.kind, period.kind)

            schedule.append({
                "time": period.start.strftime("%H:%M"),
                "mode": mode_name
            })

        # Get price data if available
        price_data = {}
        if hasattr(controller, '_current_prices'):
            prices = controller._current_prices
            if prices:
                # Current hour price
                current_hour = now.hour
                if current_hour < len(prices):
                    price_data["current_price"] = prices[current_hour]

                # Calculate min/max/avg
                price_data["min_price_today"] = min(prices)
                price_data["max_price_today"] = max(prices)
                price_data["avg_price_today"] = sum(prices) / len(prices)

                # Create hourly price chart data
                hourly_prices = []
                for hour, price in enumerate(prices):
                    is_cheapest = hour in getattr(controller, '_cheapest_hours', [])
                    is_expensive = price > controller.config.export_price_threshold

                    hourly_prices.append({
                        "hour": hour,
                        "price": price,
                        "is_cheapest": is_cheapest,
                        "is_expensive": is_expensive
                    })
                price_data["hourly_prices"] = hourly_prices

        # Build comprehensive status response
        status = {
            "connected": controller._running,
            "current_mode": mode_display,
            "inverter_time": inverter_time.isoformat() if inverter_time else None,
            "time_offset_minutes": round(time_offset, 1) if time_offset else None,
            "simulation_mode": controller._optional_config.get("simulation_mode", False),

            # Power flow data
            "solar_power": realtime_data.get("solar_power", 0),
            "battery_power": realtime_data.get("battery_power", 0),
            "grid_power": realtime_data.get("grid_power", 0),
            "load_power": realtime_data.get("load_power", 0),

            # Battery status
            "battery_soc": realtime_data.get("battery_soc", 0),
            "battery_voltage": realtime_data.get("battery_voltage", 0),
            "battery_current": realtime_data.get("battery_current", 0),
            "battery_temp": realtime_data.get("battery_temp", 0),

            # Energy prices
            "current_price": price_data.get("current_price", 0),
            "avg_price_today": price_data.get("avg_price_today", 0),
            "min_price_today": price_data.get("min_price_today", 0),
            "max_price_today": price_data.get("max_price_today", 0),
            "export_threshold": controller.config.export_price_threshold,
            "hourly_prices": price_data.get("hourly_prices", []),

            # Schedule
            "schedule": schedule,

            # Daily statistics
            "daily_production": realtime_data.get("daily_production", 0),
            "daily_consumption": realtime_data.get("daily_consumption", 0),
            "daily_import": realtime_data.get("daily_import", 0),
            "daily_export": realtime_data.get("daily_export", 0),

            # System info
            "season_mode": controller._season_mode,
            "ac_enabled": controller._ac_enabled,
            "export_enabled": controller._export_enabled,
        }

        return web.json_response(status)

    except Exception as e:
        return web.json_response(
            {"error": str(e)},
            status=500
        )
