"""Growatt Battery Controller Monitoring Dashboard.

Self-contained web server on port 5555 with:
- Real-time status display (mode, SOC, prices, solar, decisions)
- Historical log streaming via SSE
- Manual override controls
- Responsive design (mobile + desktop)
"""

import asyncio
import json
import logging
import collections
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiohttp import web


# Circular buffer for recent log messages
_log_buffer: collections.deque = collections.deque(maxlen=500)
_sse_clients: List[asyncio.Queue] = []


class DashboardLogHandler(logging.Handler):
    """Captures GROWATT log messages for the dashboard.

    Uses LogRecord ID to deduplicate — each log event has a unique record
    object, but propagation causes the same record to hit multiple handlers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen_ids: collections.OrderedDict = collections.OrderedDict()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Use record creation time + message as unique ID
            # (record.created is a float timestamp, unique per log call)
            record_id = (record.created, record.getMessage())
            if record_id in self._seen_ids:
                return
            self._seen_ids[record_id] = True
            # Keep only last 1000 IDs to prevent memory leak
            while len(self._seen_ids) > 1000:
                self._seen_ids.popitem(last=False)

            msg = self.format(record)
            now = datetime.now().strftime("%H:%M:%S")

            entry = {
                "time": now,
                "level": record.levelname,
                "message": msg,
            }
            _log_buffer.append(entry)
            # Notify SSE clients
            for q in _sse_clients:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass


def _get_controller(request: web.Request):
    return request.app.get("controller")


def _get_live_telemetry() -> Dict[str, Any]:
    """Get live inverter telemetry from the Growatt API cache."""
    try:
        from modules.growatt.api import _telemetry_cache
        if _telemetry_cache:
            return dict(_telemetry_cache)
    except Exception:
        pass
    return {}


def _get_live_soc() -> Optional[float]:
    """Get live battery SOC from the Growatt API telemetry cache."""
    t = _get_live_telemetry()
    return t.get("SOC") if t else None


# --- API Endpoints ---

async def api_live(request: web.Request) -> web.Response:
    """Get real-time power flow data from inverter telemetry."""
    telemetry = _get_live_telemetry()
    ctrl = _get_controller(request)

    solar_w = telemetry.get("InputPower", 0) or 0
    battery_w = telemetry.get("ChargePower", 0) or 0
    discharge_w = telemetry.get("DischargePower", 0) or 0
    grid_export_w = telemetry.get("ACPowerToGrid", 0) or 0
    grid_import_w = 0  # Derived: if load > solar + battery discharge
    load_w = telemetry.get("INVPowerToLocalLoad", 0) or 0
    soc = telemetry.get("SOC", 0) or 0

    # Energy totals today (kWh)
    solar_today = telemetry.get("TodayGenerateEnergy", 0) or 0
    export_today = telemetry.get("EnergyToGridToday", 0) or 0
    import_today = telemetry.get("EnergyToUserToday", 0) or 0
    load_today = telemetry.get("LocalLoadEnergyToday", 0) or 0
    charge_today = telemetry.get("ChargeEnergyToday", 0) or 0
    discharge_today = telemetry.get("DischargeEnergyToday", 0) or 0

    # Self-consumption rate
    self_consumed = solar_today - export_today if solar_today > export_today else 0
    self_consumption_rate = (self_consumed / solar_today * 100) if solar_today > 0 else 0

    # Current price info
    rate = 25.0
    if ctrl:
        rate = ctrl._eur_czk_rate or 25.0

    # Estimated savings today (very rough: self-consumed * avg price avoided)
    now = None
    avg_price = 0
    if ctrl:
        now = ctrl._get_local_now()
        if ctrl._current_prices:
            prices = list(ctrl._current_prices.values())
            avg_price = sum(prices) / len(prices) if prices else 0  # Already CZK/kWh

    return web.json_response({
        "power": {
            "solar_w": round(solar_w),
            "battery_charge_w": round(battery_w),
            "battery_discharge_w": round(discharge_w),
            "grid_export_w": round(grid_export_w),
            "load_w": round(load_w),
            "soc": round(soc),
        },
        "energy_today": {
            "solar_kwh": round(solar_today, 1),
            "export_kwh": round(export_today, 1),
            "import_kwh": round(import_today, 1),
            "load_kwh": round(load_today, 1),
            "charge_kwh": round(charge_today, 1),
            "discharge_kwh": round(discharge_today, 1),
            "self_consumed_kwh": round(self_consumed, 1),
            "self_consumption_pct": round(self_consumption_rate, 1),
        },
        "has_data": bool(telemetry),
    })

async def api_status(request: web.Request) -> web.Response:
    """Get comprehensive system status."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)

    now = ctrl._get_local_now()
    rate = ctrl._eur_czk_rate or 25.0

    # Current price (already CZK/kWh in _current_prices)
    current_price_czk = 0.0
    current_block = None
    if ctrl._current_prices:
        block_min = (now.minute // 15) * 15
        block_start = now.replace(minute=block_min, second=0, microsecond=0)
        from datetime import timedelta
        block_end = block_start + timedelta(minutes=15)
        start_str = block_start.strftime("%H:%M")
        if block_end.date() != block_start.date():
            end_str = "24:00"
        else:
            end_str = block_end.strftime("%H:%M")
        current_block = (start_str, end_str)
        current_price_czk = ctrl._current_prices.get(current_block, 0.0)

    # Inverter state
    inv = ctrl._current_inverter_state
    inverter_state = None
    if inv:
        inverter_state = {
            "mode": inv.inverter_mode,
            "stop_soc": inv.stop_soc,
            "power_rate": inv.power_rate,
            "ac_charge": inv.ac_charge_enabled,
            "export": inv.export_enabled,
            "time_start": inv.time_start,
            "time_stop": inv.time_stop,
            "source": inv.source,
        }

    # Solar forecast
    solar = None
    if ctrl._solar_forecast:
        today_kwh = ctrl._solar_forecast.get_expected_production_kwh(now.date())
        from datetime import timedelta as td
        tomorrow_kwh = ctrl._solar_forecast.get_expected_production_kwh(
            now.date() + td(days=1)
        )
        # Determine forecast source
        today_str = now.date().strftime("%Y-%m-%d")
        today_fc = ctrl._solar_forecast._consensus.get(today_str)
        source = today_fc.source if today_fc else "none"
        has_model = source.startswith("model") if source else False

        solar = {
            "today_kwh": round(today_kwh, 1),
            "tomorrow_kwh": round(tomorrow_kwh, 1),
            "confidence": 1.0 if has_model else ctrl._solar_forecast.confidence,
            "source": source,
            "has_model": has_model,
            "model_trained": ctrl._solar_forecast._production_model is not None,
            "model_bins": len(ctrl._solar_forecast._production_model.median_2d) if ctrl._solar_forecast._production_model else 0,
            "arrays": [
                {"name": a.name, "kwp": a.kwp, "azimuth": a.azimuth}
                for a in ctrl._solar_forecast.arrays
            ],
        }

    # Charging schedule
    schedule = {
        "charging_blocks": len(ctrl._combined_charging_blocks),
        "charging_today": sorted(list(ctrl._cheapest_charging_blocks_today)),
        "pre_discharge_today": sorted(list(getattr(ctrl, '_pre_discharge_blocks_today', set()))),
    }

    # Manual override
    override = ctrl.get_manual_override_status() if hasattr(ctrl, 'get_manual_override_status') else {"active": False}

    # Season
    season = getattr(ctrl, '_season_mode', 'unknown')

    # Last evaluation
    last_eval = getattr(ctrl, '_last_evaluation_reason', 'N/A')

    # Decision explanation
    decision = {}
    if hasattr(ctrl, '_decision_engine') and ctrl._decision_engine:
        try:
            decision = ctrl._decision_engine.explain_decision()
        except Exception:
            decision = {"reason": "unavailable"}

    # Optimizer summary
    optimizer_info = None
    charge_today: set = set()
    discharge_today: set = set()
    sell_production_today: set = set()
    if hasattr(ctrl, '_optimizer') and ctrl._optimizer:
        discharge_today = getattr(ctrl, '_discharge_periods_today', set())
        charge_today = getattr(ctrl, '_combined_charging_blocks', set())
        sell_production_today = getattr(ctrl, '_sell_production_blocks_today', set())
        profile = getattr(ctrl._optimizer, '_base_load_profile', None)
        profile_summary = profile.summary() if profile else "Not built"
        reserve_info = getattr(ctrl._optimizer, '_last_reserve_info', {})
        optimizer_info = {
            "enabled": True,
            "charge_blocks_today": len(charge_today),
            "discharge_blocks_today": len(discharge_today),
            "sell_production_blocks_today": len(sell_production_today),
            "charge_blocks": sorted(list(charge_today)),
            "discharge_blocks": sorted(list(discharge_today)),
            "sell_production_blocks": sorted(list(sell_production_today)),
            "base_load_profile": profile_summary,
            "reserve": reserve_info,
        }

    # Consumption forecast
    consumption_info = None
    if hasattr(ctrl, '_consumption_forecast') and ctrl._consumption_forecast and ctrl._consumption_forecast.model:
        m = ctrl._consumption_forecast.model
        # Predict today's consumption at current temperature
        try:
            from modules.growatt.api import _telemetry_cache
            temp = _telemetry_cache.get("InverterTemperature", 10)  # rough proxy
        except Exception:
            temp = 10
        daily = ctrl._consumption_forecast.predict_daily(temp, now.date())
        consumption_info = {
            "predicted_today_kwh": round(daily, 1),
            "model_bins": len(m.medians),
            "model_data_points": m.data_points,
        }

    # Next scheduled action
    next_action = None
    if optimizer_info and (charge_today or discharge_today):
        current_block_str = f"{now.hour:02d}:{(now.minute // 15) * 15:02d}"
        all_blocks = [(b, "charge") for b in sorted(charge_today)] + [(b, "discharge") for b in sorted(discharge_today)]
        for (start, end), action_type in sorted(all_blocks):
            if start > current_block_str:
                next_action = {"time": f"{start}-{end}", "action": action_type}
                break

    return web.json_response({
        "timestamp": now.isoformat(),
        "mode": ctrl._current_mode,
        "battery_soc": _get_live_soc() or ctrl._battery_soc,
        "current_price": {
            "czk_kwh": round(current_price_czk, 2),
            "block": current_block,
        },
        "high_loads_active": ctrl._high_loads_active,
        "high_loads": {
            "active": ctrl._high_loads_active,
            "ev_charging": ctrl._high_load_details.get("ev_charging", False),
            "ev_power": ctrl._high_load_details.get("ev_power", 0),
            "heating_active": ctrl._high_load_details.get("heating_active", False),
            "heating_relays": ctrl._high_load_details.get("heating_relays", []),
        },
        "season": season,
        "inverter": inverter_state,
        "solar_forecast": solar,
        "schedule": schedule,
        "manual_override": override,
        "last_evaluation": last_eval,
        "decision": decision,
        "optimizer": optimizer_info,
        "consumption": consumption_info,
        "next_action": next_action,
        "simulation_mode": getattr(ctrl.config, 'simulation_mode', False),
    })


async def api_prices(request: web.Request) -> web.Response:
    """Get today's 15-min price blocks."""
    ctrl = _get_controller(request)
    if not ctrl or not ctrl._current_prices:
        return web.json_response({"prices": []})

    rate = ctrl._eur_czk_rate or 25.0
    sell_fee = getattr(ctrl.config, 'sell_fee_czk', 0.5)
    batt_amort = getattr(ctrl.config, 'battery_amortisation_czk', 2.0)
    batt_cap = getattr(ctrl.config, 'battery_capacity', 10.0)
    now = ctrl._get_local_now()
    block_min = (now.minute // 15) * 15
    block_start = now.replace(minute=block_min, second=0, microsecond=0)
    from datetime import timedelta
    block_end = block_start + timedelta(minutes=15)
    cur_start = block_start.strftime("%H:%M")
    cur_end = "24:00" if block_end.date() != block_start.date() else block_end.strftime("%H:%M")

    # Build SOC projection lookup from optimizer decisions
    soc_lookup: Dict[str, Dict] = {}  # "today/tomorrow:HH:MM" -> {soc, kwh, action}
    if hasattr(ctrl, '_optimizer') and ctrl._optimizer:
        decisions = getattr(ctrl._optimizer, '_last_decisions', [])
        first_date = decisions[0].timestamp.date() if decisions else now.date()
        for d in decisions:
            day = "tomorrow" if d.timestamp.date() != first_date else "today"
            key = f"{day}:{d.timestamp.strftime('%H:%M')}"
            soc_delta = d.soc_after - d.soc_before
            net_flow = batt_cap * soc_delta / 100
            soc_lookup[key] = {
                "soc": round(d.soc_after, 1),
                "kwh": round(batt_cap * d.soc_after / 100, 1),
                "action": d.action,
                "solar_kwh": round(d.solar_kwh, 2),
                "consumption_kwh": round(d.consumption_kwh, 2),
                "net_flow_kwh": round(net_flow, 2),
            }

    pre_discharge = getattr(ctrl, '_pre_discharge_blocks_today', set())
    discharge = getattr(ctrl, '_discharge_periods_today', set())
    sell_production = getattr(ctrl, '_sell_production_blocks_today', set())

    prices = []
    for (start, end), price_czk in sorted(ctrl._current_prices.items()):
        is_current = (start == cur_start and end == cur_end)
        is_charging = (start, end) in ctrl._combined_charging_blocks
        is_pre_discharge = (start, end) in pre_discharge
        is_discharge = (start, end) in discharge
        is_sell_production = (start, end) in sell_production

        status = "normal"
        if is_charging and is_pre_discharge:
            status = "pre_discharge_charge"
        elif is_charging:
            status = "charging"
        elif is_discharge:
            status = "discharge"
        elif is_sell_production:
            status = "sell_production"

        czk = round(price_czk, 2)  # Already CZK/kWh
        proj = soc_lookup.get(f"today:{start}", {})
        # Sell economics: what you actually get per kWh sold
        from .decision_engine import GrowattDecisionEngine, PriceThresholds
        dist = GrowattDecisionEngine._get_distribution_tariff(
            int(start.split(":")[0]),
            PriceThresholds(
                charge_price_max=1.5, export_price_min=1.0, discharge_price_min=5.0,
                discharge_profit_margin=4.0, battery_efficiency=0.85,
                distribution_tariff_high=getattr(ctrl.config, 'distribution_tariff_high', 1.5),
                distribution_tariff_low=getattr(ctrl.config, 'distribution_tariff_low', 0.5),
                low_tariff_hours=getattr(ctrl.config, 'low_tariff_hours', '0-6,22-24'),
            )
        )
        net_sell = round(czk - dist - sell_fee - batt_amort, 2)

        prices.append({
            "start": start,
            "end": end,
            "day": "today",
            "czk_kwh": czk,
            "net_sell_czk": net_sell,
            "distribution_czk": round(dist, 2),
            "is_charging": is_charging,
            "is_pre_discharge": is_pre_discharge,
            "is_discharge": is_discharge,
            "is_sell_production": is_sell_production,
            "is_current": is_current,
            "status": status,
            "projected_soc": proj.get("soc"),
            "projected_kwh": proj.get("kwh"),
            "projected_action": proj.get("action"),
            "projected_solar": proj.get("solar_kwh"),
            "projected_consumption": proj.get("consumption_kwh"),
            "projected_net_flow": proj.get("net_flow_kwh"),
        })

    # Tomorrow's prices (if available)
    tomorrow_prices = []
    next_day = getattr(ctrl, '_next_day_prices', {})
    if next_day:
        charge_tmrw = getattr(ctrl, '_cheapest_charging_blocks_tomorrow', set())
        pre_dis_tmrw = getattr(ctrl, '_pre_discharge_blocks_tomorrow', set())
        dis_tmrw = getattr(ctrl, '_discharge_periods_tomorrow', set())
        sp_tmrw = getattr(ctrl, '_sell_production_blocks_tomorrow', set())

        for (start, end), price_czk_t in sorted(next_day.items()):
            is_charging = (start, end) in charge_tmrw
            is_pre_discharge = (start, end) in pre_dis_tmrw
            is_discharge = (start, end) in dis_tmrw
            is_sell_production = (start, end) in sp_tmrw

            status = "normal"
            if is_charging and is_pre_discharge:
                status = "pre_discharge_charge"
            elif is_charging:
                status = "charging"
            elif is_discharge:
                status = "discharge"
            elif is_sell_production:
                status = "sell_production"

            czk_t = round(price_czk_t, 2)  # Already CZK/kWh
            proj_t = soc_lookup.get(f"tomorrow:{start}", {})
            hour_t = int(start.split(":")[0])
            dist_t = GrowattDecisionEngine._get_distribution_tariff(
                hour_t,
                PriceThresholds(
                    charge_price_max=1.5, export_price_min=1.0, discharge_price_min=5.0,
                    discharge_profit_margin=4.0, battery_efficiency=0.85,
                    distribution_tariff_high=getattr(ctrl.config, 'distribution_tariff_high', 1.5),
                    distribution_tariff_low=getattr(ctrl.config, 'distribution_tariff_low', 0.5),
                    low_tariff_hours=getattr(ctrl.config, 'low_tariff_hours', '0-6,22-24'),
                )
            )
            net_sell_t = round(czk_t - dist_t - sell_fee - batt_amort, 2)

            tomorrow_prices.append({
                "start": start,
                "end": end,
                "day": "tomorrow",
                "czk_kwh": czk_t,
                "net_sell_czk": net_sell_t,
                "distribution_czk": round(dist_t, 2),
                "is_charging": is_charging,
                "is_pre_discharge": is_pre_discharge,
                "is_discharge": is_discharge,
                "is_sell_production": is_sell_production,
                "is_current": False,
                "projected_soc": proj_t.get("soc"),
                "projected_kwh": proj_t.get("kwh"),
                "projected_action": proj_t.get("action"),
                "projected_solar": proj_t.get("solar_kwh"),
                "projected_consumption": proj_t.get("consumption_kwh"),
                "projected_net_flow": proj_t.get("net_flow_kwh"),
                "status": status,
            })

    return web.json_response({
        "prices": prices,
        "tomorrow": tomorrow_prices,
        "has_tomorrow": len(tomorrow_prices) > 0,
    })


async def api_projection(request: web.Request) -> web.Response:
    """Get projected battery SOC timeline from optimizer decisions."""
    ctrl = _get_controller(request)
    if not ctrl or not hasattr(ctrl, '_optimizer') or not ctrl._optimizer:
        return web.json_response({"timeline": []})

    decisions = getattr(ctrl._optimizer, '_last_decisions', [])
    if not decisions:
        return web.json_response({"timeline": []})

    battery_capacity = getattr(ctrl.config, 'battery_capacity', 10.0)
    timeline = []
    for d in decisions:
        kwh = battery_capacity * d.soc_after / 100
        soc_delta = d.soc_after - d.soc_before
        net_flow_kwh = battery_capacity * soc_delta / 100
        timeline.append({
            "time": d.timestamp.strftime("%H:%M"),
            "day": "tomorrow" if hasattr(d.timestamp, 'date') and decisions[0].timestamp.date() != d.timestamp.date() else "today",
            "soc": round(d.soc_after, 1),
            "kwh": round(kwh, 1),
            "action": d.action,
            "price": round(d.price_czk, 2),
            "solar_kwh": round(d.solar_kwh, 2),
            "consumption_kwh": round(d.consumption_kwh, 2),
            "net_flow_kwh": round(net_flow_kwh, 2),
        })

    return web.json_response({"timeline": timeline})


async def api_logs(request: web.Request) -> web.Response:
    """Get recent log entries."""
    count = int(request.query.get("count", "100"))
    logs = list(_log_buffer)[-count:]
    return web.json_response({"logs": logs})


async def api_logs_stream(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events stream for real-time logs."""
    response = web.StreamResponse()
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["Access-Control-Allow-Origin"] = "*"
    await response.prepare(request)

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_clients.append(queue)

    try:
        while True:
            entry = await queue.get()
            data = json.dumps(entry)
            await response.write(f"data: {data}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        _sse_clients.remove(queue)

    return response


async def api_override_set(request: web.Request) -> web.Response:
    """Set manual override."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)

    try:
        body = await request.json()
        mode = body.get("mode", "regular")
        duration = body.get("duration_hours", 4)
        params = body.get("params", {})

        result = await ctrl.set_manual_override(
            mode=mode,
            duration_type="duration_hours",
            duration_value=duration,
            params=params,
            source="dashboard",
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def api_override_clear(request: web.Request) -> web.Response:
    """Clear manual override."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)

    result = await ctrl.clear_manual_override()
    return web.json_response(result)


async def dashboard_page(request: web.Request) -> web.Response:
    """Serve the main dashboard HTML page."""
    return web.Response(
        text=DASHBOARD_HTML, content_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# --- Dashboard Setup ---

def create_dashboard_app(controller=None) -> web.Application:
    """Create the dashboard web application."""
    app = web.Application()
    if controller:
        app["controller"] = controller

    app.router.add_get("/", dashboard_page)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/live", api_live)
    app.router.add_get("/api/prices", api_prices)
    app.router.add_get("/api/projection", api_projection)
    app.router.add_get("/api/logs", api_logs)
    app.router.add_get("/api/logs/stream", api_logs_stream)
    app.router.add_post("/api/override", api_override_set)
    app.router.add_delete("/api/override", api_override_clear)

    return app


_dashboard_handler_installed = False


async def start_dashboard(controller, port: int = 5555) -> None:
    """Start the dashboard web server."""
    global _dashboard_handler_installed
    logger = logging.getLogger(__name__)

    # Install log handler only once (prevents duplicates on restart)
    if not _dashboard_handler_installed:
        growatt_logger = logging.getLogger("modules.base.GrowattController")
        # Remove any existing DashboardLogHandlers first
        growatt_logger.handlers = [
            h for h in growatt_logger.handlers
            if not isinstance(h, DashboardLogHandler)
        ]
        handler = DashboardLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        growatt_logger.addHandler(handler)
        _dashboard_handler_installed = True

    app = create_dashboard_app(controller)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Dashboard listening on http://0.0.0.0:{port}")


# --- Embedded HTML Dashboard ---

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Growatt Battery Dashboard</title>
<style>
:root {
  --bg: #0f1117;
  --card: #1a1d27;
  --border: #2a2d3a;
  --text: #e4e6eb;
  --muted: #8b8fa3;
  --accent: #4f8cff;
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #eab308;
  --orange: #f97316;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}
.header {
  background: var(--card);
  border-bottom: 1px solid var(--border);
  padding: 12px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 { font-size: 16px; font-weight: 600; }
.header .status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green);
  display: inline-block;
  margin-right: 8px;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.container { max-width: 1200px; margin: 0 auto; padding: 12px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
}
.card h2 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 12px;
}
.stat { margin-bottom: 8px; }
.stat-label { font-size: 12px; color: var(--muted); }
.stat-value { font-size: 22px; font-weight: 700; }
.stat-value.small { font-size: 16px; }
.mode-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
}
.mode-regular { background: #1e3a5f; color: #60a5fa; }
.mode-charge_from_grid, .mode-battery_first_ac_charge { background: #164e2f; color: var(--green); }
.mode-discharge_to_grid { background: #4a1d1d; color: var(--red); }
.mode-high_load_protected { background: #4a3b1d; color: var(--yellow); }
.mode-sell_production { background: #3b1d4a; color: #c084fc; }

.price-chart {
  width: 100%;
  height: 180px;
  display: flex;
  align-items: flex-end;
  gap: 1px;
  margin-top: 8px;
}
.price-chart-label {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 2px;
  display: flex;
  justify-content: space-between;
}
.price-bar {
  flex: 1;
  min-width: 2px;
  border-radius: 2px 2px 0 0;
  position: relative;
  transition: opacity 0.2s;
}
.price-bar:hover { opacity: 0.8; cursor: pointer; }
.price-bar.charging { background: var(--green) !important; }
.price-bar.pre-discharge { background: #c084fc !important; }
.price-bar.discharge { background: var(--red) !important; }
.price-bar.sell-production { background: #f97316 !important; }
.price-bar.current { outline: 2px solid var(--accent); outline-offset: -1px; }
.price-bar.negative { background: #22c55e44; }
.price-bar.cheap { background: #4f8cff44; }
.price-bar.mid { background: #eab30844; }
.price-bar.expensive { background: #ef444444; }

.chart-tooltip {
  display: none;
  position: fixed;
  z-index: 200;
  background: #1e2130;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  line-height: 1.6;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  pointer-events: none;
  max-width: 260px;
}
.chart-tooltip .tt-time { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
.chart-tooltip .tt-price { font-size: 18px; font-weight: 700; }
.chart-tooltip .tt-price.neg { color: var(--green); }
.chart-tooltip .tt-price.pos { color: var(--text); }
.chart-tooltip .tt-price.high { color: var(--red); }
.chart-tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; }
.chart-tooltip .tt-label { color: var(--muted); }
.chart-tooltip .tt-status { margin-top: 4px; padding: 2px 8px; border-radius: 4px; display: inline-block; font-weight: 600; font-size: 11px; }
.chart-tooltip .tt-charging { background: #164e2f; color: var(--green); }
.chart-tooltip .tt-discharge { background: #4a1d1d; color: var(--red); }
.chart-tooltip .tt-pre-discharge { background: #2d1b4e; color: #c084fc; }
.chart-tooltip .tt-sell-production { background: #4a2a14; color: #f97316; }
.chart-tooltip .tt-current { background: #1e3a5f; color: var(--accent); }

.log-box {
  background: #0d0f14;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px;
  height: 300px;
  overflow-y: auto;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  line-height: 1.6;
}
.log-box::-webkit-scrollbar { width: 6px; }
.log-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.log-entry { padding: 1px 0; }
.log-time { color: var(--muted); margin-right: 6px; }
.log-warn { color: var(--yellow); }
.log-error { color: var(--red); }

.override-panel {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}
.btn {
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--text);
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.15s;
}
.btn:hover { background: var(--border); }
.btn:active { transform: scale(0.97); }
.btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn-danger { background: var(--red); border-color: var(--red); color: #fff; }
.btn-success { background: var(--green); border-color: var(--green); color: #fff; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

select, input {
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
}

.soc-bar {
  width: 100%;
  height: 24px;
  background: var(--bg);
  border-radius: 12px;
  overflow: hidden;
  margin-top: 4px;
}
.soc-fill {
  height: 100%;
  border-radius: 12px;
  transition: width 0.5s, background 0.5s;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 700;
  color: #fff;
}

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  margin-right: 4px;
}
.tag-on { background: #164e2f; color: var(--green); }
.tag-off { background: #2a1a1a; color: var(--red); }
.tag-warn { background: #4a3b1d; color: var(--yellow); }
.tag-info { background: #1e3a5f; color: #60a5fa; }

@media (max-width: 600px) {
  .grid { grid-template-columns: 1fr; }
  .stat-value { font-size: 18px; }
  .header h1 { font-size: 14px; }
}
</style>
</head>
<body>
<div class="header">
  <div>
    <span class="status-dot" id="statusDot"></span>
    <h1 style="display:inline">Growatt Battery Dashboard</h1>
  </div>
  <div style="font-size:12px;color:var(--muted)" id="lastUpdate">--</div>
</div>

<div class="container">
  <!-- Live Power Flow -->
  <div class="card" style="margin-bottom:12px">
    <h2>Live Power Flow</h2>
    <div id="powerFlowNoData" style="color:var(--muted);text-align:center;padding:20px;display:none">Waiting for inverter telemetry...</div>
    <div id="powerFlowGrid" style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;text-align:center">
      <div>
        <div style="font-size:24px">&#9728;&#65039;</div>
        <div class="stat-label">Solar</div>
        <div class="stat-value small" id="pwSolar" style="color:var(--yellow)">-- W</div>
      </div>
      <div>
        <div style="font-size:24px">&#128267;</div>
        <div class="stat-label">Battery <span id="pwSocBadge" style="font-size:11px;color:var(--muted)">--%</span></div>
        <div class="stat-value small" id="pwBattery" style="color:var(--green)">-- W</div>
        <div class="soc-bar" style="margin-top:4px"><div class="soc-fill" id="pwSocBar" style="width:50%">50%</div></div>
      </div>
      <div>
        <div style="font-size:24px">&#127968;</div>
        <div class="stat-label">Home Load</div>
        <div class="stat-value small" id="pwLoad" style="color:var(--accent)">-- W</div>
      </div>
      <div>
        <div style="font-size:24px">&#9889;</div>
        <div class="stat-label">Grid</div>
        <div class="stat-value small" id="pwGrid" style="color:var(--muted)">-- W</div>
      </div>
    </div>
  </div>

  <!-- Energy Today -->
  <div class="grid" style="grid-template-columns:repeat(auto-fit, minmax(120px, 1fr));margin-bottom:12px">
    <div class="card" style="text-align:center">
      <div class="stat-label">Solar Today</div>
      <div class="stat-value small" style="color:var(--yellow)" id="enSolar">-- kWh</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="stat-label">Self-Consumed</div>
      <div class="stat-value small" style="color:var(--green)" id="enSelfConsumed">-- kWh</div>
      <div class="stat-label" id="enSelfRate">--%</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="stat-label">Exported</div>
      <div class="stat-value small" style="color:var(--accent)" id="enExport">-- kWh</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="stat-label">Imported</div>
      <div class="stat-value small" style="color:var(--red)" id="enImport">-- kWh</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="stat-label">Home Load</div>
      <div class="stat-value small" id="enLoad">-- kWh</div>
    </div>
    <div class="card" style="text-align:center">
      <div class="stat-label">Batt Charged</div>
      <div class="stat-value small" style="color:var(--green)" id="enCharge">-- kWh</div>
    </div>
  </div>

  <div class="grid">
    <!-- Battery & Mode -->
    <div class="card">
      <h2>Battery & Mode</h2>
      <div class="stat">
        <div class="stat-label">Current Mode</div>
        <div id="currentMode"><span class="mode-badge mode-regular">--</span></div>
      </div>
      <div class="stat">
        <div class="stat-label">Battery SOC</div>
        <div class="soc-bar"><div class="soc-fill" id="socFill" style="width:50%">50%</div></div>
      </div>
      <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px" id="tags"></div>
    </div>

    <!-- Price -->
    <div class="card">
      <h2>Current Price</h2>
      <div class="stat">
        <div class="stat-value" id="priceCzk">--</div>
        <div class="stat-label">CZK/kWh</div>
      </div>
      <div class="stat" style="margin-top:8px">
        <div class="stat-label">Block: <span id="priceBlock">--</span></div>
      </div>
    </div>

    <!-- Solar Forecast -->
    <div class="card">
      <h2>Solar Forecast</h2>
      <div class="stat">
        <div class="stat-label">Today</div>
        <div class="stat-value small" id="solarToday">-- kWh</div>
      </div>
      <div class="stat">
        <div class="stat-label">Tomorrow</div>
        <div class="stat-value small" id="solarTomorrow">-- kWh</div>
      </div>
      <div class="stat-label" id="solarConf">Confidence: --</div>
    </div>

    <!-- Schedule -->
    <div class="card">
      <h2>Schedule</h2>
      <div class="stat">
        <div class="stat-label">Charging blocks today</div>
        <div class="stat-value small" id="chargeBlocks">--</div>
      </div>
      <div class="stat">
        <div class="stat-label">Evaluation trigger</div>
        <div style="font-size:13px;color:var(--muted)" id="lastEval">--</div>
      </div>
    </div>
  </div>

  <!-- Decision Reasoning -->
  <div class="grid" style="grid-template-columns: 1fr 1fr;">
    <div class="card">
      <h2>Decision Reasoning</h2>
      <div id="decisionReason" style="font-size:13px;line-height:1.8">
        <div class="stat-label">Why this mode?</div>
        <div id="decisionWhy" style="margin-bottom:8px">--</div>
        <div class="stat-label">Decision priority</div>
        <div id="decisionPriority" style="margin-bottom:8px">--</div>
        <div class="stat-label">Next scheduled action</div>
        <div id="nextAction" style="font-weight:600">--</div>
      </div>
    </div>
    <div class="card">
      <h2>Optimizer & Forecast</h2>
      <div id="optimizerInfo" style="font-size:13px;line-height:1.8">
        <div class="stat-label">Optimizer schedule</div>
        <div id="optimizerSummary" style="margin-bottom:8px">--</div>
        <div class="stat-label">Consumption forecast</div>
        <div id="consumptionInfo" style="margin-bottom:8px">--</div>
        <div class="stat-label">Solar model</div>
        <div id="solarModelInfo">--</div>
      </div>
    </div>
  </div>

  <!-- Price Chart -->
  <div class="card" style="margin-bottom:12px">
    <h2 id="priceChartTitle">Today's Prices &amp; Battery SOC</h2>
    <div class="price-chart-day" id="priceChartTodayWrap">
      <div class="price-chart-label" id="priceChartTodayLabel">Today</div>
      <div style="position:relative">
        <div class="price-chart" id="priceChartToday"></div>
        <svg id="socLineToday" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
      </div>
    </div>
    <div class="price-chart-day" id="priceChartTomorrowWrap" style="display:none;margin-top:12px">
      <div class="price-chart-label" id="priceChartTomorrowLabel">Tomorrow</div>
      <div style="position:relative">
        <div class="price-chart" id="priceChartTomorrow"></div>
        <svg id="socLineTomorrow" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
      </div>
    </div>
  </div>
  <div class="chart-tooltip" id="chartTooltip"></div>

  <!-- Override Controls -->
  <div class="card" style="margin-bottom:12px">
    <h2>Manual Override</h2>
    <div id="overrideStatus" style="margin-bottom:8px;font-size:13px"></div>
    <div class="override-panel">
      <select id="overrideMode">
        <option value="regular">Regular</option>
        <option value="charge_from_grid">Charge from Grid</option>
        <option value="discharge_to_grid">Discharge to Grid</option>
        <option value="high_load_protected">High Load Protected</option>
        <option value="sell_production">Sell Production</option>
      </select>
      <select id="overrideDuration">
        <option value="1">1 hour</option>
        <option value="2">2 hours</option>
        <option value="4" selected>4 hours</option>
        <option value="8">8 hours</option>
        <option value="12">12 hours</option>
        <option value="24">24 hours</option>
      </select>
      <button class="btn btn-primary" onclick="setOverride()">Set Override</button>
      <button class="btn btn-danger" onclick="clearOverride()">Reset to Auto</button>
    </div>
  </div>

  <!-- Logs -->
  <div class="card">
    <h2>Live Logs</h2>
    <div class="log-box" id="logBox"></div>
  </div>
</div>

<script>
let lastData = null;

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    lastData = data;
    updateUI(data);
    document.getElementById('statusDot').style.background = 'var(--green)';
  } catch (e) {
    document.getElementById('statusDot').style.background = 'var(--red)';
  }
}

function updateUI(d) {
  // Time
  const t = new Date(d.timestamp);
  document.getElementById('lastUpdate').textContent = t.toLocaleTimeString();

  // Mode
  const mode = d.mode || 'unknown';
  const modeClass = 'mode-' + mode.replace(/ /g, '_');
  document.getElementById('currentMode').innerHTML =
    '<span class="mode-badge ' + modeClass + '">' + mode.replace(/_/g, ' ') + '</span>';

  // SOC
  const soc = d.battery_soc || 0;
  const fill = document.getElementById('socFill');
  fill.style.width = soc + '%';
  fill.textContent = soc.toFixed(0) + '%';
  fill.style.background = soc > 60 ? 'var(--green)' : soc > 30 ? 'var(--yellow)' : 'var(--red)';

  // Tags
  const tags = [];
  if (d.simulation_mode) tags.push('<span class="tag tag-warn">SIM</span>');
  if (d.high_loads_active) {
    let hlParts = [];
    if (d.high_loads?.heating_relays?.length) hlParts.push('🔥 ' + d.high_loads.heating_relays.join(', '));
    if (d.high_loads?.ev_charging) hlParts.push('🚗 EV ' + (d.high_loads.ev_power/1000).toFixed(1) + ' kW');
    tags.push('<span class="tag tag-warn">' + (hlParts.length ? hlParts.join(' · ') : 'HIGH LOAD') + '</span>');
  }
  if (d.inverter) {
    tags.push(d.inverter.export ? '<span class="tag tag-on">EXP ON</span>' : '<span class="tag tag-off">EXP OFF</span>');
    tags.push(d.inverter.ac_charge ? '<span class="tag tag-on">AC CHG</span>' : '<span class="tag tag-off">AC CHG OFF</span>');
  }
  tags.push('<span class="tag tag-info">' + (d.season || '?') + '</span>');
  document.getElementById('tags').innerHTML = tags.join('');

  // Price
  const p = d.current_price || {};
  document.getElementById('priceCzk').textContent = (p.czk_kwh || 0).toFixed(2) + ' CZK/kWh';
  document.getElementById('priceBlock').textContent = p.block ? p.block[0] + ' - ' + p.block[1] : '--';
  const priceEl = document.getElementById('priceCzk');
  priceEl.style.color = (p.czk_kwh || 0) < 0 ? 'var(--green)' : (p.czk_kwh || 0) > 3 ? 'var(--red)' : 'var(--text)';

  // Solar
  if (d.solar_forecast) {
    document.getElementById('solarToday').textContent = d.solar_forecast.today_kwh + ' kWh';
    document.getElementById('solarTomorrow').textContent = d.solar_forecast.tomorrow_kwh + ' kWh';
    const sf = d.solar_forecast;
    let confText = '';
    if (sf.has_model) {
      confText = 'Source: Learned model (' + sf.model_bins + ' bins)';
    } else {
      confText = 'Confidence: ' + (sf.confidence * 100).toFixed(0) + '% (no model)';
    }
    document.getElementById('solarConf').textContent = confText;
  }

  // Schedule
  document.getElementById('chargeBlocks').textContent = (d.schedule?.charging_blocks || 0) + ' blocks';
  document.getElementById('lastEval').textContent = d.last_evaluation || '--';

  // Decision reasoning
  if (d.decision) {
    const dec = d.decision;
    document.getElementById('decisionWhy').innerHTML =
      '<span style="color:var(--accent);font-weight:600">' + (dec.reason || 'unknown') + '</span>';
    const prio = dec.priority;
    document.getElementById('decisionPriority').textContent =
      prio ? prio.name + ' (level ' + prio.level + ')' : '--';
  }

  // Next action
  if (d.next_action) {
    const na = d.next_action;
    const actionColor = na.action === 'charge' ? 'var(--green)' : 'var(--red)';
    document.getElementById('nextAction').innerHTML =
      '<span style="color:' + actionColor + '">' + na.action.toUpperCase() + '</span> at ' + na.time;
  } else {
    document.getElementById('nextAction').textContent = 'No upcoming actions today';
  }

  // Optimizer
  if (d.optimizer) {
    const opt = d.optimizer;
    const res = opt.reserve || {};
    let reserveHtml = '';
    if (res.next_recharge_ts) {
      const rechargeTime = new Date(res.next_recharge_ts).toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit'});
      const incoming = res.incoming_charge_kwh || 0;
      const gross = res.gross_reserve_kwh || res.net_reserve_kwh || 0;
      const net = res.net_reserve_kwh || 0;
      let reserveCalc = net + ' kWh';
      if (incoming > 0) {
        reserveCalc = gross + ' - ' + incoming + ' incoming = ' + net + ' kWh';
      }
      reserveHtml = '<br><span style="color:var(--accent)">Next recharge: ' + rechargeTime + '</span> (' + res.next_recharge_reason + ', ' + res.hours_until_recharge + 'h)' +
        '<br>Reserve: ' + reserveCalc + ' → min SOC ' + res.effective_min_soc + '%';
    } else {
      reserveHtml = '<br>No recharge found in schedule';
    }
    var sellProdCount = opt.sell_production_blocks_today || 0;
    var sellProdHtml = sellProdCount > 0
      ? ' + <span style="color:#c084fc">' + sellProdCount + ' solar-export</span>'
      : '';
    document.getElementById('optimizerSummary').innerHTML =
      '<span style="color:var(--green)">' + opt.charge_blocks_today + ' charge</span> + ' +
      '<span style="color:var(--red)">' + opt.discharge_blocks_today + ' discharge</span>' +
      sellProdHtml + ' blocks today' +
      '<span style="font-size:11px;color:var(--muted)">' + reserveHtml + '</span>' +
      '<br><span style="color:var(--muted);font-size:10px">' + (opt.base_load_profile || '') + '</span>';
  } else {
    document.getElementById('optimizerSummary').textContent = 'Optimizer disabled';
  }

  // Consumption forecast
  if (d.consumption) {
    document.getElementById('consumptionInfo').textContent =
      'Predicted today: ' + d.consumption.predicted_today_kwh + ' kWh (' + d.consumption.model_bins + ' bins)';
  } else {
    document.getElementById('consumptionInfo').textContent = 'Not available';
  }

  // Solar model
  if (d.solar_forecast?.has_model) {
    document.getElementById('solarModelInfo').textContent =
      'Learned model: ' + d.solar_forecast.model_bins + ' bins (trained on real data)';
  } else {
    document.getElementById('solarModelInfo').textContent = 'API-based forecast';
  }

  // Override
  const ov = d.manual_override || {};
  const ovEl = document.getElementById('overrideStatus');
  if (ov.active) {
    ovEl.innerHTML = '<span class="tag tag-warn">OVERRIDE ACTIVE</span> Mode: ' +
      (ov.mode || '?') + (ov.end_time ? ' until ' + new Date(ov.end_time).toLocaleTimeString() : '');
  } else {
    ovEl.innerHTML = '<span class="tag tag-on">AUTO</span> Automatic control active';
  }
}

async function fetchPrices() {
  try {
    const [priceRes, projRes] = await Promise.all([
      fetch('/api/prices'),
      fetch('/api/projection'),
    ]);
    const data = await priceRes.json();
    const projData = await projRes.json();
    const todayPrices = data.prices || [];
    const tomorrowPrices = data.tomorrow || [];
    const allPrices = [...todayPrices, ...tomorrowPrices];
    priceData = allPrices;  // unified array for tooltip lookups across both charts

    document.getElementById('priceChartTitle').textContent =
      data.has_tomorrow ? 'Today + Tomorrow Prices & Battery SOC' : "Today's Prices & Battery SOC";

    // Shared vertical scale so both charts are visually comparable
    const allVals = allPrices.map(p => p.czk_kwh);
    const maxAbs = allVals.length
      ? Math.max(Math.abs(Math.min(...allVals)), Math.abs(Math.max(...allVals)), 1)
      : 1;

    const timeline = projData.timeline || [];
    renderPriceChart(todayPrices, 'priceChartToday', 0, maxAbs);
    renderSocLine(todayPrices, timeline, 'priceChartToday', 'socLineToday');

    const tomorrowWrap = document.getElementById('priceChartTomorrowWrap');
    if (data.has_tomorrow && tomorrowPrices.length) {
      tomorrowWrap.style.display = '';
      renderPriceChart(tomorrowPrices, 'priceChartTomorrow', todayPrices.length, maxAbs);
      renderSocLine(tomorrowPrices, timeline, 'priceChartTomorrow', 'socLineTomorrow');
    } else {
      tomorrowWrap.style.display = 'none';
    }
  } catch (e) { console.error('fetchPrices error:', e); }
}

function renderSocLine(prices, timeline, chartId, svgId) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  if (!timeline.length || !prices.length) { svg.innerHTML = ''; return; }

  const chart = document.getElementById(chartId);
  if (!chart) { svg.innerHTML = ''; return; }
  const w = chart.offsetWidth;
  const h = chart.offsetHeight;
  if (!w || !h) return;

  // Build SOC lookup: "HH:MM|day" -> soc%
  const socMap = {};
  timeline.forEach(t => { socMap[t.time + '|' + t.day] = t.soc; });

  // Build points for the polyline. Track which price-bar index produced each
  // point so we can place start/middle/end SOC% labels on actual rendered bars.
  const points = [];
  const labelMeta = [];  // parallel array: { soc: number } per point
  const barWidth = w / prices.length;
  prices.forEach((p, i) => {
    const key = p.start + '|' + p.day;
    const soc = socMap[key];
    if (soc != null) {
      const x = i * barWidth + barWidth / 2;
      const y = h - (soc / 100 * h);  // SOC 100% = top, 0% = bottom
      points.push(x.toFixed(1) + ',' + y.toFixed(1));
      labelMeta.push({ soc: soc });
    }
  });

  if (points.length < 2) { svg.innerHTML = ''; return; }

  // Draw SOC line + percentage labels at key points
  let svgContent = '<polyline points="' + points.join(' ') + '" ' +
    'fill="none" stroke="#f97316" stroke-width="2" stroke-opacity="0.8" ' +
    'stroke-linejoin="round" stroke-linecap="round"/>';

  // SOC% labels at start, middle, end of THIS chart's points
  const labelIndices = [0, Math.floor(points.length / 2), points.length - 1];
  labelIndices.forEach(idx => {
    const meta = labelMeta[idx];
    if (!meta) return;
    const coords = points[idx].split(',');
    const x = parseFloat(coords[0]);
    const y = parseFloat(coords[1]) - 6;
    svgContent += '<text x="' + x + '" y="' + Math.max(12, y) + '" ' +
      'font-size="10" fill="#f97316" text-anchor="middle" font-weight="600">' +
      meta.soc + '%</text>';
  });

  svg.innerHTML = svgContent;
}

let priceData = [];

function renderPriceChart(prices, mountId, idxOffset, maxAbs) {
  const chart = document.getElementById(mountId);
  if (!chart) return;
  if (!prices.length) { chart.innerHTML = '<div style="color:var(--muted)">No price data</div>'; return; }

  // Fall back to per-chart scale if caller didn't pass one
  if (!maxAbs) {
    const vals = prices.map(p => p.czk_kwh);
    maxAbs = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)), 1);
  }

  let html = '';
  prices.forEach((p, i) => {
    const v = p.czk_kwh;
    const h = Math.abs(v) / maxAbs * 100;
    const isNeg = v < 0;
    let cls = v < 0 ? 'negative' : v < 1.5 ? 'cheap' : v < 3 ? 'mid' : 'expensive';
    if (p.status === 'pre_discharge_charge') cls = 'pre-discharge';
    else if (p.is_charging) cls = 'charging';
    else if (p.is_discharge) cls = 'discharge';
    else if (p.is_sell_production) cls = 'sell-production';
    if (p.is_current) cls += ' current';

    const globalIdx = i + (idxOffset || 0);
    const tmrwOpacity = p.day === 'tomorrow' ? 'opacity:0.6;' : '';
    if (isNeg) {
      html += '<div class="price-bar ' + cls + '" data-idx="' + globalIdx + '" style="height:' + h + '%;align-self:flex-start;opacity:0.5;' + tmrwOpacity + '"></div>';
    } else {
      html += '<div class="price-bar ' + cls + '" data-idx="' + globalIdx + '" style="height:' + Math.max(h, 3) + '%;' + tmrwOpacity + '"></div>';
    }
  });
  chart.innerHTML = html;

  // Attach tooltip events for THIS chart's bars
  const tooltip = document.getElementById('chartTooltip');
  chart.querySelectorAll('.price-bar').forEach(bar => {
    bar.addEventListener('mouseenter', (e) => showTooltip(e, bar, tooltip));
    bar.addEventListener('mousemove', (e) => moveTooltip(e, tooltip));
    bar.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    bar.addEventListener('touchstart', (e) => { e.preventDefault(); showTooltip(e.touches[0], bar, tooltip); });
  });
}

// Hide the tooltip on any touch outside a price bar (attached once globally)
document.addEventListener('touchstart', (e) => {
  if (!e.target.classList.contains('price-bar')) {
    const tt = document.getElementById('chartTooltip');
    if (tt) tt.style.display = 'none';
  }
});

function showTooltip(e, bar, tooltip) {
  const idx = parseInt(bar.dataset.idx);
  const p = priceData[idx];
  if (!p) return;

  const v = p.czk_kwh;
  const priceClass = v < 0 ? 'neg' : v > 3 ? 'high' : 'pos';

  let statusHtml = '';
  if (p.is_current) statusHtml += '<span class="tt-status tt-current">NOW</span> ';
  if (p.status === 'charging') statusHtml += '<span class="tt-status tt-charging">CHARGING</span>';
  else if (p.status === 'pre_discharge_charge') statusHtml += '<span class="tt-status tt-pre-discharge">PRE-DISCHARGE CHG</span>';
  else if (p.status === 'discharge') statusHtml += '<span class="tt-status tt-discharge">DISCHARGE</span>';
  else if (p.status === 'sell_production') statusHtml += '<span class="tt-status tt-sell-production">SELL PRODUCTION</span>';

  // Calculate actual price rank (1 = cheapest) within same day
  const sameDayPrices = priceData.filter(pp => pp.day === p.day);
  const sortedPrices = [...sameDayPrices].sort((a, b) => a.czk_kwh - b.czk_kwh);
  const priceRank = sortedPrices.findIndex(sp => sp.start === p.start && sp.end === p.end) + 1;
  const percentile = Math.round(priceRank / sameDayPrices.length * 100);
  const rankLabel = percentile <= 25 ? 'Cheapest quarter' : percentile <= 50 ? 'Below average' : percentile <= 75 ? 'Above average' : 'Most expensive quarter';
  const dayLabel = p.day === 'tomorrow' ? ' (Tomorrow)' : '';

  // Sell economics
  const netSell = p.net_sell_czk || 0;
  const sellColor = netSell > 0 ? 'var(--green)' : 'var(--red)';

  // Projected battery + energy flow
  let projHtml = '';
  if (p.projected_soc != null) {
    const projAction = p.projected_action || 'hold';
    const projColor = projAction === 'charge' ? 'var(--green)' : projAction === 'discharge' ? 'var(--red)' : 'var(--muted)';
    const nf = p.projected_net_flow || 0;
    const nfSign = nf >= 0 ? '+' : '';
    const nfColor = nf > 0 ? 'var(--green)' : nf < 0 ? 'var(--red)' : 'var(--muted)';
    const solar = p.projected_solar || 0;
    const cons = p.projected_consumption || 0;
    projHtml = '<div style="margin-top:4px;padding-top:4px;border-top:1px solid var(--border)">' +
      '<div class="tt-row"><span class="tt-label">Battery</span><span style="color:' + projColor + '">' + p.projected_soc + '% (' + p.projected_kwh + ' kWh) ' + projAction + '</span></div>' +
      '<div class="tt-row"><span class="tt-label">Net flow</span><span style="color:' + nfColor + '">' + nfSign + nf.toFixed(2) + ' kWh</span></div>' +
      '<div class="tt-row"><span class="tt-label">Solar / Load</span><span>' + solar.toFixed(2) + ' / ' + cons.toFixed(2) + ' kWh</span></div>' +
      '</div>';
  }

  tooltip.innerHTML =
    '<div class="tt-time">' + p.start + ' - ' + p.end + dayLabel + '</div>' +
    '<div class="tt-price ' + priceClass + '">' + v.toFixed(2) + ' CZK/kWh</div>' +
    '<div class="tt-row"><span class="tt-label">Net sell</span><span style="color:' + sellColor + '">' + netSell.toFixed(2) + ' CZK/kWh</span></div>' +
    '<div class="tt-row"><span class="tt-label" style="font-size:10px">spot ' + v.toFixed(2) + ' - dist ' + (p.distribution_czk||0).toFixed(2) + ' - fee 0.50 - amort 2.00</span></div>' +
    '<div class="tt-row"><span class="tt-label">Rank</span><span>#' + priceRank + '/' + sameDayPrices.length + ' (' + rankLabel + ')</span></div>' +
    projHtml +
    (statusHtml ? '<div style="margin-top:4px">' + statusHtml + '</div>' : '');

  tooltip.style.display = 'block';
  moveTooltip(e, tooltip);
}

function moveTooltip(e, tooltip) {
  const x = (e.clientX || e.pageX) + 12;
  const y = (e.clientY || e.pageY) - 10;
  const rect = tooltip.getBoundingClientRect();
  const maxX = window.innerWidth - 270;
  const maxY = window.innerHeight - rect.height - 10;
  tooltip.style.left = Math.min(x, maxX) + 'px';
  tooltip.style.top = Math.min(y, maxY) + 'px';
}

// Logs
async function fetchLogs() {
  try {
    const res = await fetch('/api/logs?count=50');
    const data = await res.json();
    const box = document.getElementById('logBox');
    box.innerHTML = (data.logs || []).map(formatLog).join('');
    box.scrollTop = box.scrollHeight;
  } catch (e) {}
}

function formatLog(entry) {
  const cls = entry.level === 'WARNING' ? 'log-warn' : entry.level === 'ERROR' ? 'log-error' : '';
  return '<div class="log-entry ' + cls + '"><span class="log-time">' + entry.time + '</span>' + escapeHtml(entry.message) + '</div>';
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// SSE for live logs — single connection with dedup
let _sseConnection = null;
const _seenLogIds = new Set();

function connectSSE() {
  // Close existing connection to prevent duplicates
  if (_sseConnection) {
    _sseConnection.close();
    _sseConnection = null;
  }

  const es = new EventSource('/api/logs/stream');
  _sseConnection = es;

  es.onmessage = function(e) {
    const entry = JSON.parse(e.data);
    // Deduplicate by time+message
    const logId = entry.time + '|' + entry.message;
    if (_seenLogIds.has(logId)) return;
    _seenLogIds.add(logId);
    // Keep set bounded
    if (_seenLogIds.size > 500) {
      const first = _seenLogIds.values().next().value;
      _seenLogIds.delete(first);
    }

    const box = document.getElementById('logBox');
    box.innerHTML += formatLog(entry);
    while (box.children.length > 200) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  };
  es.onerror = function() {
    es.close();
    _sseConnection = null;
    setTimeout(connectSSE, 5000);
  };
}

// Override controls
async function setOverride() {
  const mode = document.getElementById('overrideMode').value;
  const hours = parseInt(document.getElementById('overrideDuration').value);
  try {
    const res = await fetch('/api/override', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode, duration_hours: hours}),
    });
    const data = await res.json();
    if (data.error) alert('Error: ' + data.error);
    else fetchStatus();
  } catch (e) { alert('Failed: ' + e.message); }
}

async function clearOverride() {
  try {
    await fetch('/api/override', {method: 'DELETE'});
    fetchStatus();
  } catch (e) { alert('Failed: ' + e.message); }
}

// Init
// Live power flow
async function fetchLive() {
  try {
    const res = await fetch('/api/live');
    const d = await res.json();
    if (!d.has_data) {
      document.getElementById('powerFlowNoData').style.display = 'block';
      document.getElementById('powerFlowGrid').style.opacity = '0.3';
      return;
    }
    document.getElementById('powerFlowNoData').style.display = 'none';
    document.getElementById('powerFlowGrid').style.opacity = '1';

    const pw = d.power;
    document.getElementById('pwSolar').textContent = pw.solar_w + ' W';
    document.getElementById('pwLoad').textContent = pw.load_w + ' W';

    // Battery: show charge or discharge
    if (pw.battery_charge_w > 0) {
      document.getElementById('pwBattery').textContent = '+' + pw.battery_charge_w + ' W';
      document.getElementById('pwBattery').style.color = 'var(--green)';
    } else if (pw.battery_discharge_w > 0) {
      document.getElementById('pwBattery').textContent = '-' + pw.battery_discharge_w + ' W';
      document.getElementById('pwBattery').style.color = 'var(--red)';
    } else {
      document.getElementById('pwBattery').textContent = '0 W';
      document.getElementById('pwBattery').style.color = 'var(--muted)';
    }

    // SOC bar
    const soc = pw.soc;
    const socFill = document.getElementById('pwSocBar');
    socFill.style.width = soc + '%';
    socFill.textContent = soc + '%';
    socFill.style.background = soc > 60 ? 'var(--green)' : soc > 30 ? 'var(--yellow)' : 'var(--red)';
    document.getElementById('pwSocBadge').textContent = soc + '%';

    // Grid: export or import
    if (pw.grid_export_w > 0) {
      document.getElementById('pwGrid').textContent = pw.grid_export_w + ' W';
      document.getElementById('pwGrid').style.color = 'var(--green)';
    } else {
      document.getElementById('pwGrid').textContent = '0 W';
      document.getElementById('pwGrid').style.color = 'var(--muted)';
    }

    // Energy today
    const en = d.energy_today;
    document.getElementById('enSolar').textContent = en.solar_kwh + ' kWh';
    document.getElementById('enSelfConsumed').textContent = en.self_consumed_kwh + ' kWh';
    document.getElementById('enSelfRate').textContent = en.self_consumption_pct + '% self-consumed';
    document.getElementById('enExport').textContent = en.export_kwh + ' kWh';
    document.getElementById('enImport').textContent = en.import_kwh + ' kWh';
    document.getElementById('enLoad').textContent = en.load_kwh + ' kWh';
    document.getElementById('enCharge').textContent = en.charge_kwh + ' kWh';
  } catch (e) { console.error('fetchLive error:', e); }
}

fetchStatus();
fetchLive();
fetchPrices();
fetchLogs();
connectSSE();
setInterval(fetchStatus, 5000);
setInterval(fetchLive, 3000);
setInterval(fetchPrices, 60000);
</script>
</body>
</html>
"""
