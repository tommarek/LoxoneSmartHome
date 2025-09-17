"""REST API for Growatt controller status and control."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from modules.growatt_controller import GrowattController

# Define AppKey for proper aiohttp usage
GROWATT_CONTROLLER_KEY = web.AppKey("growatt_controller", GrowattController)

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


async def _query_inverter_slots(controller: GrowattController) -> Dict[str, Any]:
    """Query the inverter for current battery-first and grid-first slot configurations.
    
    Returns:
        Dict containing raw battery_first and grid_first responses from inverter
    """
    if not controller.mqtt_client:
        return {}
    
    # Ensure result subscription is active
    await _setup_result_subscription(controller)
    
    result = {}
    
    # Query battery-first status
    try:
        # Create a future to wait for the response
        bf_future: asyncio.Future = asyncio.Future()
        _command_responses["batteryfirst/get"] = bf_future
        
        # Send the query with correct topic and empty JSON
        await controller.mqtt_client.publish("energy/solar/command/batteryfirst/get", "{}")
        
        # Wait for response with timeout
        bf_response = await asyncio.wait_for(bf_future, timeout=2.0)
        # Return the raw response, whatever it is
        result["battery_first_raw"] = bf_response
    except asyncio.TimeoutError:
        controller.logger.debug("Timeout querying battery-first status")
        result["battery_first_raw"] = {"error": "timeout"}
    except Exception as e:
        controller.logger.debug(f"Error querying battery-first status: {e}")
        result["battery_first_raw"] = {"error": str(e)}
    finally:
        # Clean up the future
        _command_responses.pop("batteryfirst/get", None)
    
    # Query grid-first status
    try:
        # Create a future to wait for the response
        gf_future: asyncio.Future = asyncio.Future()
        _command_responses["gridfirst/get"] = gf_future
        
        # Send the query with correct topic and empty JSON
        await controller.mqtt_client.publish("energy/solar/command/gridfirst/get", "{}")
        
        # Wait for response with timeout
        gf_response = await asyncio.wait_for(gf_future, timeout=2.0)
        # Return the raw response, whatever it is
        result["grid_first_raw"] = gf_response
    except asyncio.TimeoutError:
        controller.logger.debug("Timeout querying grid-first status")
        result["grid_first_raw"] = {"error": "timeout"}
    except Exception as e:
        controller.logger.debug(f"Error querying grid-first status: {e}")
        result["grid_first_raw"] = {"error": str(e)}
    finally:
        # Clean up the future
        _command_responses.pop("gridfirst/get", None)
    
    return result


def create_growatt_api(
    app: web.Application,
    controller: Optional[GrowattController] = None
) -> None:
    """Create and register Growatt API routes.

    Args:
        app: The aiohttp web application
        controller: The GrowattController instance (can be set later via
                   app[GROWATT_CONTROLLER_KEY])
    """
    if controller:
        app[GROWATT_CONTROLLER_KEY] = controller
        # Setup persistent subscriptions on startup
        # Note: These need to be awaited before first use, but we create tasks here
        # The actual subscription will happen when _query_inverter_slots calls _setup_result_subscription
        asyncio.create_task(_setup_telemetry_subscription(controller))

    # Register routes
    app.router.add_get('/api/growatt/status', get_status)
    app.router.add_get('/api/growatt/schedule', get_schedule)
    app.router.add_get('/api/growatt/prices', get_prices)
    app.router.add_post('/api/growatt/mode', set_mode)
    app.router.add_post('/api/growatt/sync-time', sync_time)
    app.router.add_get('/api/growatt/config', get_config)
    
    # Manual mode control routes
    app.router.add_post('/api/growatt/manual-mode', manual_mode_set)
    app.router.add_delete('/api/growatt/manual-mode', manual_mode_clear)
    app.router.add_get('/api/growatt/manual-mode', manual_mode_status)

    # Dashboard routes
    app.router.add_get('/inverter', serve_dashboard)
    app.router.add_get('/inverter/', serve_dashboard)
    app.router.add_get('/inverter/status', get_dashboard_status)


async def get_status(request: web.Request) -> web.Response:
    """Get current Growatt controller status with real-time inverter data.

    Returns:
        JSON with current mode, periods, season mode, and live inverter data.
    """
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
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

        # Get inverter mode data from actual device
        inverter_data = {}
        # Check if optional_config exists (it should be set during init)
        if hasattr(controller, '_optional_config') and not controller._optional_config.get("simulation_mode", False):
            # Use cached telemetry data if available
            if _telemetry_cache:
                # Active power rate is available in telemetry
                inverter_data["active_power_rate"] = _telemetry_cache.get("ActivePowerRate", 100)

            # Query actual slot configurations from the inverter
            slot_data = await _query_inverter_slots(controller)
            
            # Just pass through the raw responses
            inverter_data["battery_first_data"] = slot_data.get("battery_first_raw", {})
            inverter_data["grid_first_data"] = slot_data.get("grid_first_raw", {})

        # Get manual override status
        manual_override = controller.get_manual_override_status()
        
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
            ) if hasattr(controller, '_optional_config') else False,
            "inverter_data": inverter_data,
            "manual_override": manual_override
        }

        return web.json_response(status)

    except Exception as e:
        import traceback
        controller.logger.error(f"Error in get_status: {e}", exc_info=True)
        return web.json_response({"error": str(e), "traceback": traceback.format_exc()}, status=500)


async def get_schedule(request: web.Request) -> web.Response:
    """Get current schedule.

    Returns:
        JSON with scheduled periods and their parameters.
    """
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
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
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        # Get current time and price for current hour
        now = controller._get_local_now()
        current_hour = now.strftime("%H:00")
        
        # Convert EUR/MWh prices to CZK/kWh for display
        eur_czk_rate = controller._eur_czk_rate or 25.0
        
        # Find current price
        current_price_eur = None
        for (start, _), price in controller._current_prices.items():
            if start == current_hour:
                current_price_eur = price
                break
        
        # Calculate statistics
        prices_list = list(controller._current_prices.values())
        if prices_list:
            avg_price_eur = sum(prices_list) / len(prices_list)
            min_price_eur = min(prices_list)
            max_price_eur = max(prices_list)
        else:
            avg_price_eur = min_price_eur = max_price_eur = None
        
        # Convert to CZK/kWh (EUR/MWh * rate / 1000)
        def to_czk_kwh(eur_mwh):
            if eur_mwh is None:
                return None
            return round(eur_mwh * eur_czk_rate / 1000, 2)
        
        # Build hourly prices list for chart
        hourly_data = []
        for (start, end), price_eur in sorted(controller._current_prices.items()):
            hourly_data.append({
                "hour": start,
                "end": end,
                "price_eur_mwh": price_eur,
                "price_czk_kwh": to_czk_kwh(price_eur)
            })
        
        return web.json_response({
            "current_price_czk_kwh": to_czk_kwh(current_price_eur),
            "average_today_czk_kwh": to_czk_kwh(avg_price_eur),
            "min_price_czk_kwh": to_czk_kwh(min_price_eur),
            "max_price_czk_kwh": to_czk_kwh(max_price_eur),
            "export_threshold_czk_kwh": controller.config.export_price_threshold,
            "eur_czk_rate": eur_czk_rate,
            "eur_czk_rate_updated": (
                controller._eur_czk_rate_updated.isoformat()
                if controller._eur_czk_rate_updated else None
            ),
            "prices_date": controller._prices_date,
            "prices_updated": (
                controller._prices_updated.isoformat()
                if controller._prices_updated else None
            ),
            "hourly_prices": hourly_data,
            "has_data": bool(controller._current_prices)
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def set_mode(request: web.Request) -> web.Response:
    """Set inverter mode manually.

    Expected JSON payload:
    {
        "mode": "regular" | "sell_production" | "regular_no_export" | "charge_from_grid" | "discharge_to_grid",
        "params": {
            "stop_soc": 90,     // optional - for charge_from_grid or discharge_to_grid
            "power_rate": 100   // optional - for discharge_to_grid
        }
    }
    """
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )

    try:
        data = await request.json()
        mode = data.get("mode")

        valid_modes = ["regular", "sell_production", "regular_no_export", "charge_from_grid", "discharge_to_grid"]
        if mode not in valid_modes:
            return web.json_response({"error": f"Invalid mode. Must be one of {valid_modes}"}, status=400)

        # Apply the composite mode
        params = data.get("params", {})
        await controller._apply_composite_mode(mode, params)

        return web.json_response({
            "success": True,
            "mode": mode,
            "message": f"Mode set to {mode}",
            "params": params
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def manual_mode_set(request: web.Request) -> web.Response:
    """Set manual mode override.

    Expected JSON payload:
    {
        "mode": "regular" | "sell_production" | "regular_no_export" | "charge_from_grid" | "discharge_to_grid",
        "duration": {
            "type": "immediate" | "end_of_day" | "duration_hours" | "until_time",
            "value": null | 4 | "18:00"  // depends on type
        },
        "params": {
            "stop_soc": 90,     // optional - for charge_from_grid or discharge_to_grid
            "power_rate": 100   // optional - for discharge_to_grid
        },
        "source": "dashboard"  // optional, defaults to "api"
    }
    """
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )
    
    try:
        data = await request.json()
        
        # Extract parameters
        mode = data.get("mode")
        if not mode:
            return web.json_response({"error": "mode is required"}, status=400)
        
        duration = data.get("duration", {})
        duration_type = duration.get("type", "immediate")
        duration_value = duration.get("value")
        
        params = data.get("params")
        source = data.get("source", "api")
        
        # Set manual override
        result = await controller.set_manual_override(
            mode=mode,
            duration_type=duration_type,
            duration_value=duration_value,
            params=params,
            source=source
        )
        
        return web.json_response(result)
        
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def manual_mode_clear(request: web.Request) -> web.Response:
    """Clear manual mode override and return to automatic control."""
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )
    
    try:
        result = await controller.clear_manual_override()
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def manual_mode_status(request: web.Request) -> web.Response:
    """Get current manual mode override status."""
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
    if not controller:
        return web.json_response(
            {"error": "Controller not initialized"}, status=503
        )
    
    try:
        status = controller.get_manual_override_status()
        return web.json_response(status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def sync_time(request: web.Request) -> web.Response:
    """Sync inverter time with server time.

    Expected JSON payload:
    {
        "force": true  # Force sync even if drift is small
    }
    """
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
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
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
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
            ) if hasattr(controller, '_optional_config') else 15.0,
            "summer_price_threshold": (
                controller.config.summer_price_threshold
            ),
            "temperature_avg_days": controller._optional_config.get(
                "temperature_avg_days", 3
            ) if hasattr(controller, '_optional_config') else 3,
            "eur_czk_rate": controller._optional_config.get(
                "eur_czk_rate", 25.0
            ) if hasattr(controller, '_optional_config') else 25.0,
            "simulation_mode": controller._optional_config.get(
                "simulation_mode", False
            ) if hasattr(controller, '_optional_config') else False,
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
    controller: Optional[GrowattController] = request.app.get(GROWATT_CONTROLLER_KEY)
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

        # Build comprehensive status response
        status = {
            "connected": controller._running,
            "current_mode": mode_display,
            "simulation_mode": controller._optional_config.get("simulation_mode", False) if hasattr(controller, '_optional_config') else False,

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

            # Configuration
            "export_threshold": controller.config.export_price_threshold,

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
