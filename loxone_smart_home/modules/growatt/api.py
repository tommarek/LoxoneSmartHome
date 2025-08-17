"""REST API for Growatt controller status and control."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from modules.growatt_controller import GrowattController

# Global cache for telemetry data
_telemetry_cache: Dict[str, Any] = {}
_telemetry_last_update: Optional[datetime] = None
_telemetry_subscription_active = False

# Global cache for command responses
_command_responses: Dict[str, asyncio.Future] = {}
_result_subscription_active = False


async def _setup_result_subscription(controller: GrowattController) -> None:
    """Setup subscription to energy/solar/result topic for command responses.

    This subscribes once and handles all command responses, avoiding
    repeated subscribe/unsubscribe operations.
    """
    global _result_subscription_active

    if _result_subscription_active or not controller.mqtt_client:
        return

    async def result_handler(_topic: str, payload: Any) -> None:
        """Handle incoming command responses from energy/solar/result topic."""
        try:
            if isinstance(payload, bytes):
                payload = payload.decode()
            data = json.loads(payload) if isinstance(payload, str) else payload

            # Get the command from the response
            command = data.get("command")
            if command and command in _command_responses:
                # Find the future waiting for this response
                future = _command_responses.get(command)
                if future and not future.done():
                    future.set_result(data)

        except Exception:
            # Silently ignore parse errors to avoid log spam
            pass

    try:
        await controller.mqtt_client.subscribe("energy/solar/result", result_handler)
        _result_subscription_active = True
    except Exception:
        # Silently fail if subscription fails
        pass


async def _setup_telemetry_subscription(controller: GrowattController) -> None:
    """Setup subscription to energy/solar topic for telemetry data.

    This subscribes once and caches all telemetry data that the inverter
    publishes every 5 seconds, avoiding the need for individual queries.
    """
    global _telemetry_subscription_active

    if _telemetry_subscription_active or not controller.mqtt_client:
        return

    async def telemetry_handler(_topic: str, payload: Any) -> None:
        """Handle incoming telemetry data from energy/solar topic."""
        global _telemetry_cache, _telemetry_last_update
        try:
            if isinstance(payload, bytes):
                payload = payload.decode()
            data = json.loads(payload) if isinstance(payload, str) else payload

            # Update cache with new telemetry data
            _telemetry_cache = data
            _telemetry_last_update = datetime.now()

        except Exception:
            # Silently ignore parse errors to avoid log spam
            pass

    try:
        await controller.mqtt_client.subscribe("energy/solar", telemetry_handler)
        _telemetry_subscription_active = True
    except Exception:
        # Silently fail if subscription fails
        pass


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

        # Ensure result subscription is active
        await _setup_result_subscription(controller)

        # Set up request/response pattern
        request_topic = f"energy/solar/command/{command}"
        correlation_id = f"{command}-{uuid.uuid4().hex[:8]}"

        # Create future for response
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[Dict[str, Any]] = loop.create_future()

        # Register the future for this command
        _command_responses[command] = response_future

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
            # Clean up the future from the response map
            _command_responses.pop(command, None)

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
        # Setup persistent subscriptions on startup
        asyncio.create_task(_setup_telemetry_subscription(controller))
        asyncio.create_task(_setup_result_subscription(controller))

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
    """Get real-time inverter data from cached telemetry.

    Returns comprehensive inverter status from the cached telemetry data
    that is automatically published by the inverter every 5 seconds.
    """
    # Ensure telemetry subscription is active
    await _setup_telemetry_subscription(controller)

    # Check if cache is fresh (less than 10 seconds old)
    if _telemetry_last_update and (datetime.now() - _telemetry_last_update) < timedelta(seconds=10):
        # Map telemetry fields to our expected format
        # Based on actual MQTT telemetry data structure
        # Calculate battery power (negative = discharging, positive = charging)
        charge_pwr = _telemetry_cache.get("ChargePower", 0)
        discharge_pwr = _telemetry_cache.get("DischargePower", 0)
        battery_power = charge_pwr - discharge_pwr

        # Calculate grid power (ACPowerToGrid - ACPowerToUser)
        to_grid = _telemetry_cache.get("ACPowerToGrid", 0)
        to_user = _telemetry_cache.get("ACPowerToUser", 0)
        grid_power = to_grid - to_user

        # Calculate battery current from power and voltage
        batt_voltage = _telemetry_cache.get("BatteryVoltage", 1)
        batt_current = battery_power / batt_voltage if batt_voltage > 0 else 0

        data = {
            "solar_power": _telemetry_cache.get("InputPower", 0),
            "battery_power": battery_power,
            "grid_power": grid_power,
            "load_power": _telemetry_cache.get("INVPowerToLocalLoad", 0),
            "battery_soc": _telemetry_cache.get("SOC", 0),
            "battery_voltage": batt_voltage,
            "battery_current": batt_current,
            "battery_temp": _telemetry_cache.get("BatteryTemperature", 0),
            "system_status": _telemetry_cache.get("InverterStatus", 0),
            "daily_production": _telemetry_cache.get("TodayGenerateEnergy", 0),
            "daily_consumption": _telemetry_cache.get("LocalLoadEnergyToday", 0),
            "daily_import": _telemetry_cache.get("ACChargeEnergyToday", 0),
            "daily_export": _telemetry_cache.get("EnergyToGridToday", 0),
            # Additional useful fields from telemetry
            "pv1_power": _telemetry_cache.get("PV1InputPower", 0),
            "pv2_power": _telemetry_cache.get("PV2InputPower", 0),
            "pv1_voltage": _telemetry_cache.get("PV1Voltage", 0),
            "pv2_voltage": _telemetry_cache.get("PV2Voltage", 0),
            "grid_voltage": _telemetry_cache.get("L1ThreePhaseGridVoltage", 0),
            "grid_frequency": _telemetry_cache.get("GridFrequency", 0),
            "inverter_temp": _telemetry_cache.get("InverterTemperature", 0),
            "charge_power": _telemetry_cache.get("ChargePower", 0),
            "discharge_power": _telemetry_cache.get("DischargePower", 0),
            "battery_state": _telemetry_cache.get("BatteryState", 0),
            "active_power_rate": _telemetry_cache.get("ActivePowerRate", 100),
        }
    else:
        # Cache is stale or doesn't exist, return empty data
        # Dashboard will show zeros until telemetry data arrives
        data = {
            "solar_power": 0,
            "battery_power": 0,
            "grid_power": 0,
            "load_power": 0,
            "battery_soc": 0,
            "battery_voltage": 0,
            "battery_current": 0,
            "battery_temp": 0,
            "system_status": "Waiting for data...",
            "daily_production": 0,
            "daily_consumption": 0,
            "daily_import": 0,
            "daily_export": 0,
        }

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
        # TODO: Implement price data caching in controller
        price_data = {}

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
