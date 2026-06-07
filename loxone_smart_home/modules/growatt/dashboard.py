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
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiohttp import web


# Circular buffer for recent log messages
_log_buffer: collections.deque = collections.deque(maxlen=500)
# Each SSE client is (event_loop, queue): emit() can fire from a worker thread
# (asyncio.to_thread(optimize) logs off the loop), and asyncio.Queue is not
# thread-safe, so notifications are marshalled back onto the client's own loop.
_sse_clients: List[tuple] = []


def _safe_put(q: "asyncio.Queue", entry: dict) -> None:
    """Best-effort enqueue, dropping the entry if the client's queue is full."""
    try:
        q.put_nowait(entry)
    except asyncio.QueueFull:
        pass


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
            # Notify SSE clients. Iterate a copy: emit() can fire from the worker
            # thread used by asyncio.to_thread(optimize), while api_logs_stream
            # mutates _sse_clients on the event loop — iterating the live list
            # could raise mid-iteration. asyncio.Queue is not thread-safe, so
            # always hand the put to the client's own loop via call_soon_threadsafe
            # (safe to call from the loop thread too).
            for loop, q in list(_sse_clients):
                try:
                    loop.call_soon_threadsafe(_safe_put, q, entry)
                except RuntimeError:
                    # Loop is closed/stopped — client is going away; skip it.
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

    solar_w = telemetry.get("InputPower", 0) or 0
    battery_w = telemetry.get("ChargePower", 0) or 0
    discharge_w = telemetry.get("DischargePower", 0) or 0
    grid_export_w = telemetry.get("ACPowerToGrid", 0) or 0
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

    # Distribution tariff (import surcharge) for the current block, so the UI can
    # show the REAL all-in price you pay to buy = spot + distribution.
    current_distribution_czk = 0.0
    if current_block:
        from .decision_engine import GrowattDecisionEngine, PriceThresholds
        _th = PriceThresholds(
            charge_price_max=1.5, export_price_min=1.0, discharge_price_min=5.0,
            discharge_profit_margin=4.0, battery_efficiency=0.85,
            distribution_tariff_high=ctrl.config.distribution_tariff_high,
            distribution_tariff_low=ctrl.config.distribution_tariff_low,
            low_tariff_hours=getattr(
                ctrl.config, "low_tariff_hours", "0-10,11-12,13-14,15-17,18-24"
            ),
        )
        current_distribution_czk = GrowattDecisionEngine._get_distribution_tariff(
            int(current_block[0].split(":")[0]), _th
        )

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
            "on": bool(inv.inverter_on),
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

        # Solcast (optional weather-based source). Surface its status, the
        # per-day totals it contributed to the consensus, and the shared
        # free-tier API budget used so far today.
        solcast_client = getattr(ctrl, "_solcast_forecast", None)
        if solcast_client is not None and getattr(solcast_client, "enabled", False):
            ingested = getattr(ctrl._solar_forecast, "_solcast_forecast", {}) or {}
            tomorrow_str = (now.date() + td(days=1)).strftime("%Y-%m-%d")

            def _sc_total(dstr):
                fc = ingested.get(dstr)
                return round(fc.total_kwh, 1) if fc else None

            solar["solcast"] = {
                "enabled": True,
                "sites": len(getattr(solcast_client, "rooftop_ids", []) or []),
                "quantile": getattr(solcast_client, "quantile", "p50"),
                "today_kwh": _sc_total(today_str),
                "tomorrow_kwh": _sc_total(tomorrow_str),
                "requests_today": getattr(solcast_client, "_req_count", 0),
                "daily_budget": getattr(solcast_client, "_max_requests_per_day", 9),
            }
        else:
            solar["solcast"] = {"enabled": False}

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
        # Which engine is actually loaded (reflects real state, not just config:
        # MILP falls back to greedy if PuLP is missing).
        engine = "milp" if type(ctrl._optimizer).__name__.startswith("MILP") else "greedy"
        optimizer_info = {
            "enabled": True,
            "engine": engine,
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
        "battery_soc": (
            _get_live_soc() if _get_live_soc() is not None else ctrl._battery_soc
        ),
        "current_price": {
            "czk_kwh": round(current_price_czk, 2),
            "distribution_czk": round(current_distribution_czk, 2),
            "buy_czk": round(current_price_czk + current_distribution_czk, 2),
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


def _build_price_rows(
    ctrl, price_items, day, *, charging, pre_discharge, discharge,
    sell_production, soc_lookup, sell_fee, batt_amort, inv_off_threshold,
    cur_block=None, hold=frozenset(),
):
    """Build the per-block price rows for one day.

    The today and tomorrow price tables are identical except for the source
    price dict, the per-action block-sets, the soc_lookup day prefix, and
    whether ``is_current`` can be True — so both call this single builder.
    """
    from .decision_engine import GrowattDecisionEngine, PriceThresholds
    thresholds = PriceThresholds(
        charge_price_max=1.5, export_price_min=1.0, discharge_price_min=5.0,
        discharge_profit_margin=4.0, battery_efficiency=0.85,
        distribution_tariff_high=ctrl.config.distribution_tariff_high,
        distribution_tariff_low=ctrl.config.distribution_tariff_low,
        low_tariff_hours=getattr(
            ctrl.config, 'low_tariff_hours', '0-10,11-12,13-14,15-17,18-24'
        ),
    )
    rows = []
    for (start, end), price_czk in sorted(price_items):
        is_current = cur_block is not None and (start, end) == cur_block
        is_charging = (start, end) in charging
        is_pre_discharge = (start, end) in pre_discharge
        is_discharge = (start, end) in discharge
        is_sell_production = (start, end) in sell_production
        is_hold = (start, end) in hold
        is_inverter_off = (price_czk < inv_off_threshold) and not is_charging

        status = "normal"
        if is_charging and is_pre_discharge:
            status = "pre_discharge_charge"
        elif is_charging:
            status = "charging"
        elif is_discharge:
            status = "discharge"
        elif is_sell_production:
            status = "sell_production"
        elif is_hold:
            status = "battery_hold"
        elif is_inverter_off:
            status = "inverter_off"

        czk = round(price_czk, 2)  # Already CZK/kWh
        proj = soc_lookup.get(f"{day}:{start}", {})
        # Distribution (import surcharge) — still shown per block for reference.
        dist = GrowattDecisionEngine._get_distribution_tariff(
            int(start.split(":")[0]), thresholds
        )
        # Sell economics: what you actually net per kWh sold from the battery.
        # Export pays NO distribution — only the sell fee (+ battery wear).
        net_sell = round(czk - sell_fee - batt_amort, 2)

        rows.append({
            "start": start,
            "end": end,
            "day": day,
            "czk_kwh": czk,
            "buy_czk": round(czk + dist, 2),  # real all-in import price
            "net_sell_czk": net_sell,
            "distribution_czk": round(dist, 2),
            "is_charging": is_charging,
            "is_pre_discharge": is_pre_discharge,
            "is_discharge": is_discharge,
            "is_sell_production": is_sell_production,
            "is_hold": is_hold,
            "is_inverter_off": is_inverter_off,
            "is_current": is_current,
            "status": status,
            "projected_soc": proj.get("soc"),
            "projected_kwh": proj.get("kwh"),
            "projected_action": proj.get("action"),
            "projected_solar": proj.get("solar_kwh"),
            "projected_consumption": proj.get("consumption_kwh"),
            "projected_net_flow": proj.get("net_flow_kwh"),
            "projected_power_rate": proj.get("power_rate"),
        })
    return rows


async def api_prices(request: web.Request) -> web.Response:
    """Get today's 15-min price blocks."""
    ctrl = _get_controller(request)
    if not ctrl or not ctrl._current_prices:
        return web.json_response({"prices": []})

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
        # Planned inverter powerRate% per charge/discharge block (the speed the
        # optimizer would run the slot at). Computed against the hardware max so
        # it reads honestly even with adaptive off (gentle ≈ floor everywhere).
        from modules.growatt.optimizer import compute_charge_power_rates
        _cmax = getattr(ctrl.config, 'battery_charge_max_kw', 9.8)
        _dmax = getattr(ctrl.config, 'battery_discharge_max_kw', 9.8)
        from modules.growatt.optimizer import compute_rate_ceiling
        _chg_cap = compute_rate_ceiling(_cmax, getattr(ctrl.config, 'max_charge_power_kw', _cmax))
        _dis_cap = compute_rate_ceiling(_dmax, getattr(ctrl.config, 'max_discharge_power_kw', _dmax))
        _engine_used = getattr(getattr(ctrl, '_optimizer', None), '_last_engine', 'greedy')
        _eff = getattr(ctrl.config, 'battery_efficiency', 0.85)
        _leg_eta = _eff ** 0.5 if _engine_used == 'milp' else _eff
        charge_rates = compute_charge_power_rates(
            decisions, batt_cap, _cmax,
            int(getattr(ctrl.config, 'min_charge_power_rate', 25)),
            action="charge", max_power_rate=_chg_cap,
            efficiency=_leg_eta,
        )
        disch_rates = compute_charge_power_rates(
            decisions, batt_cap, _dmax,
            int(getattr(ctrl.config, 'discharge_power_rate', 25)),
            action="discharge", max_power_rate=_dis_cap,
            efficiency=_leg_eta,
        )
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
                "power_rate": charge_rates.get(d.timestamp) or disch_rates.get(d.timestamp),
            }

    pre_discharge = getattr(ctrl, '_pre_discharge_blocks_today', set())
    discharge = getattr(ctrl, '_discharge_periods_today', set())
    sell_production = getattr(ctrl, '_sell_production_blocks_today', set())
    # Threshold gate: inverter would be off below (threshold - hysteresis),
    # except in scheduled-charge blocks where it's forced on.
    inv_off_threshold = (
        getattr(ctrl.config, 'inverter_off_price_threshold_czk', -2.0)
        - getattr(ctrl.config, 'inverter_off_price_hysteresis_czk', 0.1)
    )

    prices = _build_price_rows(
        ctrl, ctrl._current_prices.items(), "today",
        charging=ctrl._combined_charging_blocks,
        pre_discharge=pre_discharge, discharge=discharge,
        sell_production=sell_production, soc_lookup=soc_lookup,
        sell_fee=sell_fee, batt_amort=batt_amort,
        inv_off_threshold=inv_off_threshold,
        cur_block=(cur_start, cur_end),
        hold=getattr(ctrl, '_hold_blocks_today', set()),
    )

    # Tomorrow's prices (if available)
    tomorrow_prices = []
    next_day = getattr(ctrl, '_next_day_prices', {})
    if next_day:
        tomorrow_prices = _build_price_rows(
            ctrl, next_day.items(), "tomorrow",
            charging=getattr(ctrl, '_cheapest_charging_blocks_tomorrow', set()),
            pre_discharge=getattr(ctrl, '_pre_discharge_blocks_tomorrow', set()),
            discharge=getattr(ctrl, '_discharge_periods_tomorrow', set()),
            sell_production=getattr(ctrl, '_sell_production_blocks_tomorrow', set()),
            soc_lookup=soc_lookup, sell_fee=sell_fee, batt_amort=batt_amort,
            inv_off_threshold=inv_off_threshold,
            hold=getattr(ctrl, '_hold_blocks_tomorrow', set()),
            cur_block=None,  # never "current" for a future day
        )

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
    from modules.growatt.optimizer import compute_charge_power_rates
    _cmax = getattr(ctrl.config, 'battery_charge_max_kw', 9.8)
    _dmax = getattr(ctrl.config, 'battery_discharge_max_kw', 9.8)
    from modules.growatt.optimizer import compute_rate_ceiling
    _chg_cap = compute_rate_ceiling(_cmax, getattr(ctrl.config, 'max_charge_power_kw', _cmax))
    _dis_cap = compute_rate_ceiling(_dmax, getattr(ctrl.config, 'max_discharge_power_kw', _dmax))
    _engine_used = getattr(getattr(ctrl, '_optimizer', None), '_last_engine', 'greedy')
    _eff = getattr(ctrl.config, 'battery_efficiency', 0.85)
    _leg_eta = _eff ** 0.5 if _engine_used == 'milp' else _eff
    charge_rates = compute_charge_power_rates(
        decisions, battery_capacity, _cmax,
        int(getattr(ctrl.config, 'min_charge_power_rate', 25)),
        action="charge", max_power_rate=_chg_cap, efficiency=_leg_eta,
    )
    disch_rates = compute_charge_power_rates(
        decisions, battery_capacity, _dmax,
        int(getattr(ctrl.config, 'discharge_power_rate', 25)),
        action="discharge", max_power_rate=_dis_cap, efficiency=_leg_eta,
    )
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
            "power_rate": charge_rates.get(d.timestamp) or disch_rates.get(d.timestamp),
        })

    return web.json_response({"timeline": timeline})


async def _today_local_actuals(
    ctrl, *, field, every, agg_fn, time_fmt, value_fn, result_key, cache_attr, ttl,
):
    """Shared builder for the today-actuals endpoints (solar / SOC / consumption).

    Each endpoint queries one InfluxDB ``solar``-measurement field, buckets the
    rows by LOCAL time so they line up with the optimizer's local-time grid, and
    caches the payload per controller. They differ only in the field, aggregate
    window/function, time format, value transform, result key, cache attribute
    and TTL — all passed in here — so the query/timezone/cache logic lives once.
    """
    if not ctrl or not getattr(ctrl, 'influxdb_client', None):
        return {result_key: {}}

    cache = getattr(ctrl, cache_attr, None)
    if cache:
        ts, payload = cache
        if (datetime.now() - ts).total_seconds() < ttl:
            return payload

    try:
        bucket = ctrl.settings.influxdb.bucket_solar
        q = f'''
from(bucket: "{bucket}")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "solar" and r._field == "{field}")
  |> aggregateWindow(every: {every}, fn: {agg_fn}, createEmpty: false)
'''
        r = await ctrl.influxdb_client.query(q)
        out: Dict[str, float] = {}
        # InfluxDB returns UTC-aware timestamps; bucket by LOCAL time so the
        # actuals line up with the optimizer's local-time grid (the tooltip
        # compares them directly). Fall back to naive/UTC without zoneinfo.
        try:
            from zoneinfo import ZoneInfo
            local_tz = ZoneInfo("Europe/Prague")
        except Exception:
            local_tz = None
        local_now = datetime.now(local_tz) if local_tz else datetime.now()
        today = local_now.date()
        for table in r:
            for record in table.records:
                t = record.get_time()
                if local_tz is not None and getattr(t, "tzinfo", None) is not None:
                    t = t.astimezone(local_tz)
                if t.date() != today:
                    continue
                val = record.get_value()
                if isinstance(val, (int, float)):
                    out[t.strftime(time_fmt)] = value_fn(float(val))
        payload = {result_key: out}
        setattr(ctrl, cache_attr, (datetime.now(), payload))
        return payload
    except Exception as e:
        return {result_key: {}, "error": str(e)}


async def api_solar_actuals(request: web.Request) -> web.Response:
    """Today's actual hourly solar production (for forecast-vs-actual overlay).

    Cached ~5 min per controller to keep dashboard refreshes cheap.
    """
    payload = await _today_local_actuals(
        _get_controller(request),
        field="InputPower", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: round(w / 1000.0, 3),  # W → kWh
        result_key="hourly", cache_attr="_solar_actuals_cache", ttl=300,
    )
    return web.json_response(payload)


async def api_soc_actuals(request: web.Request) -> web.Response:
    """Today's actual battery SOC per 15-min block (for the real-vs-projected
    SOC overlay). Lets the chart show what the battery ACTUALLY did for hours
    that have already elapsed, instead of only the optimizer's projection.

    `last` (not mean) per window: SOC is a level, so the value at the end of
    each block is what we want to plot, matching the projection grid. Cached
    ~90 s so the elapsed-hours line stays fresh without hammering InfluxDB.
    """
    payload = await _today_local_actuals(
        _get_controller(request),
        field="SOC", every="15m", agg_fn="last", time_fmt="%H:%M",
        value_fn=lambda s: round(s, 1),  # percent
        result_key="blocks", cache_attr="_soc_actuals_cache", ttl=90,
    )
    return web.json_response(payload)


async def api_consumption_actuals(request: web.Request) -> web.Response:
    """Today's actual hourly household consumption (kWh) for the forecast-vs-
    actual overlay on the chart. Mirrors api_solar_actuals but for load."""
    payload = await _today_local_actuals(
        _get_controller(request),
        field="INVPowerToLocalLoad", every="1h", agg_fn="mean", time_fmt="%H:00",
        value_fn=lambda w: round(w / 1000.0, 3),  # W → kWh
        result_key="hourly", cache_attr="_consumption_actuals_cache", ttl=300,
    )
    return web.json_response(payload)


def _avg_block_price(ctrl, blocks) -> Optional[float]:
    """Average spot price (CZK/kWh) over a set of (start,end) blocks today."""
    if not blocks or not ctrl._current_prices:
        return None
    vals = [ctrl._current_prices[b] for b in blocks if b in ctrl._current_prices]
    return sum(vals) / len(vals) if vals else None


async def api_insights(request: web.Request) -> web.Response:
    """Aggregated 'more info' panel: economics, forecast accuracy, active
    engines, command health, deferrable loads, and data freshness. Everything
    here is derived from data the controller already holds — no new state."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)
    now = ctrl._get_local_now()
    out: Dict[str, Any] = {"timestamp": now.isoformat()}

    # ---- Active engines (real runtime state, not just config) ----
    opt = getattr(ctrl, "_optimizer", None)
    engine = None
    if opt is not None:
        engine = "milp" if type(opt).__name__.startswith("MILP") else "greedy"
    opt_cfg = getattr(ctrl.config, "optimizer_engine", "greedy")
    ml = getattr(ctrl, "_ml_consumption_forecast", None)
    ml_active = bool(ml and getattr(ml, "is_trained", False))
    cons_cfg = getattr(ctrl.config, "consumption_forecast_engine", "binned")
    out["engines"] = {
        "optimizer": engine or "rule-based",
        "optimizer_configured": opt_cfg,
        "optimizer_fellback": bool(engine and engine != opt_cfg),
        "consumption": "ml" if ml_active else "binned",
        "consumption_configured": cons_cfg,
        "consumption_fellback": bool(cons_cfg == "ml" and not ml_active),
    }

    # ---- Solcast free-tier budget ----
    sc = getattr(ctrl, "_solcast_forecast", None)
    if sc is not None and getattr(sc, "enabled", False):
        out["solcast"] = {
            "enabled": True,
            "requests_today": getattr(sc, "_req_count", 0),
            "daily_budget": getattr(sc, "_max_requests_per_day", 9),
        }
    else:
        out["solcast"] = {"enabled": False}

    # ---- Forecast accuracy (solar: actual-so-far vs forecast-for-same-hours) ----
    sf = getattr(ctrl, "_solar_forecast", None)
    accuracy: Dict[str, Any] = {
        "solar_confidence": round(getattr(sf, "confidence", 0.0), 2) if sf else None,
        "calibration_ratio": getattr(ctrl, "_last_solar_calibration_ratio", None),
    }
    try:
        if sf and getattr(ctrl, "influxdb_client", None):
            # Actual produced so far today (sum of elapsed hourly InputPower).
            actuals = await api_solar_actuals(request)
            import json as _json
            act = _json.loads(actuals.body.decode()).get("hourly", {})
            elapsed = {h: v for h, v in act.items() if int(h[:2]) <= now.hour}
            actual_kwh = round(sum(elapsed.values()), 1)
            today_str = now.date().strftime("%Y-%m-%d")
            fc = sf._consensus.get(today_str)
            fc_elapsed = None
            if fc and fc.hourly:
                fc_elapsed = round(
                    sum(v for h, v in fc.hourly.items() if int(h) <= now.hour), 1
                )
            accuracy["solar_actual_kwh_so_far"] = actual_kwh
            accuracy["solar_forecast_kwh_so_far"] = fc_elapsed
            if fc_elapsed:
                accuracy["solar_error_pct"] = round(
                    (actual_kwh - fc_elapsed) / fc_elapsed * 100, 1
                )
    except Exception as e:
        accuracy["error"] = str(e)
    out["accuracy"] = accuracy

    # ---- Command health ----
    out["commands"] = {
        "sent": getattr(ctrl, "_commands_sent_count", 0),
        "skipped": getattr(ctrl, "_commands_skipped_count", 0),
        "last_results": {
            k: {"success": bool(v.get("success")), "message": v.get("message", "")}
            for k, v in (getattr(ctrl, "_last_command_results", {}) or {}).items()
        },
    }

    # ---- Deferrable loads ----
    defs = getattr(ctrl, "_deferrable_loads", []) or []
    scheds = {
        getattr(s, "load_name", None): s
        for s in (getattr(ctrl, "_deferrable_schedules", []) or [])
    }
    out["deferrable"] = [
        {
            "name": l.name,
            "energy_kwh": getattr(l, "energy_required_kwh", None),
            "power_kw": getattr(l, "power_kw", None),
            "blocks": len(getattr(scheds.get(l.name), "blocks", []) or []),
            "savings_czk": (
                round(getattr(scheds[l.name], "savings_czk", 0.0), 2)
                if l.name in scheds else None
            ),
        }
        for l in defs
    ]

    # ---- Economics (today). plan_value is exact; the rest are estimates from
    #      daily energy totals × representative prices and are flagged as such. ----
    econ: Dict[str, Any] = {"estimated": True}
    decisions = getattr(opt, "_last_decisions", []) if opt else []
    if decisions:
        econ["plan_value_czk"] = round(
            sum((getattr(d, "net_value", 0) or 0) for d in decisions), 1
        )
    tel = _get_live_telemetry()
    export_kwh = tel.get("EnergyToGridToday", 0) or 0
    import_kwh = tel.get("EnergyToUserToday", 0) or 0
    charge_kwh = tel.get("ChargeEnergyToday", 0) or 0
    discharge_kwh = tel.get("DischargeEnergyToday", 0) or 0
    prices = list(ctrl._current_prices.values()) if ctrl._current_prices else []
    avg_price = sum(prices) / len(prices) if prices else None
    sell_fee = getattr(ctrl.config, "sell_fee_czk", 0.5)
    dist_hi = getattr(ctrl.config, "distribution_tariff_high", 1.5)
    avg_charge_price = _avg_block_price(ctrl, getattr(ctrl, "_combined_charging_blocks", set()))
    avg_disch_price = _avg_block_price(ctrl, getattr(ctrl, "_discharge_periods_today", set()))
    if avg_price is not None:
        econ["export_revenue_czk"] = round(export_kwh * max(0.0, avg_price - sell_fee), 1)
        econ["import_cost_czk"] = round(import_kwh * (avg_price + dist_hi), 1)
    if avg_charge_price is not None and avg_disch_price is not None:
        econ["arbitrage_czk"] = round(
            discharge_kwh * avg_disch_price - charge_kwh * avg_charge_price, 1
        )
    econ["export_kwh"] = round(export_kwh, 1)
    econ["import_kwh"] = round(import_kwh, 1)
    out["economics"] = econ

    # ---- Data freshness ----
    def _age(ts):
        try:
            return round((now - ts).total_seconds()) if ts else None
        except Exception:
            return None
    out["freshness"] = {
        "prices_age_s": _age(getattr(ctrl, "_prices_updated", None)),
        "has_tomorrow_prices": bool(getattr(ctrl, "_next_day_prices", {})),
    }

    return web.json_response(out)


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
    # Register with this request's running loop so the (cross-thread) log handler
    # can marshal puts back onto it safely.
    client = (asyncio.get_running_loop(), queue)
    _sse_clients.append(client)

    try:
        while True:
            entry = await queue.get()
            data = json.dumps(entry)
            await response.write(f"data: {data}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        if client in _sse_clients:
            _sse_clients.remove(client)

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


async def api_reapply(request: web.Request) -> web.Response:
    """Force a re-send of the current decided state to the inverter (manual
    retry after a failed control command)."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)
    result = await ctrl.reapply_current_state()
    return web.json_response(result)


async def api_settings_get(request: web.Request) -> web.Response:
    """Return the grouped editable-settings schema with current values."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)
    try:
        return web.json_response({"groups": ctrl.get_settings_schema()})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def api_settings_post(request: web.Request) -> web.Response:
    """Validate, persist and hot-apply edited settings."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"success": False, "message": "Invalid JSON body"},
                                 status=400)
    updates = body.get("updates", body) if isinstance(body, dict) else None
    if not isinstance(updates, dict) or not updates:
        return web.json_response(
            {"success": False, "message": "Expected a non-empty object of settings"},
            status=400,
        )
    result = await ctrl.apply_setting_overrides(updates)
    status = 200 if result.get("success") else 400
    return web.json_response(result, status=status)


async def api_restart(request: web.Request) -> web.Response:
    """Restart the controller process (applies restart-only settings)."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)
    result = await ctrl.request_restart("dashboard request")
    return web.json_response(result)


async def settings_page(request: web.Request) -> web.Response:
    """Serve the settings editor page."""
    return web.Response(
        text=SETTINGS_HTML, content_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


async def dashboard_page(request: web.Request) -> web.Response:
    """Serve the main dashboard HTML page."""
    return web.Response(
        text=DASHBOARD_HTML, content_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


async def manifest_page(request: web.Request) -> web.Response:
    """PWA web manifest (makes the dashboard installable)."""
    return web.Response(
        text=MANIFEST_JSON, content_type="application/manifest+json",
        headers={"Cache-Control": "max-age=3600"},
    )


async def sw_page(request: web.Request) -> web.Response:
    """Service worker — required for installability. No-cache so updates apply."""
    return web.Response(
        text=SW_JS, content_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


async def icon_svg_page(request: web.Request) -> web.Response:
    """Maskable app icon (Chrome/Android, favicon)."""
    return web.Response(
        text=ICON_SVG, content_type="image/svg+xml",
        headers={"Cache-Control": "max-age=86400"},
    )


def _png_response(data: bytes) -> web.Response:
    if not data:
        return web.Response(status=404, text="icon not generated")
    return web.Response(
        body=data, content_type="image/png",
        headers={"Cache-Control": "max-age=86400"},
    )


async def icon_180_page(request: web.Request) -> web.Response:
    """Apple touch icon (iOS Add-to-Home-Screen)."""
    return _png_response(_ICON_180)


async def icon_192_page(request: web.Request) -> web.Response:
    return _png_response(_ICON_192)


async def icon_512_page(request: web.Request) -> web.Response:
    return _png_response(_ICON_512)


# --- Dashboard Setup ---

def _add_api_routes(app: "web.Application") -> None:
    """Register the controller-backed JSON + control endpoints.

    These read live in-process controller state (and the in-process telemetry
    cache), so they MUST run in the same process as the controller — i.e. the
    main service, never the standalone web container.
    """
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/live", api_live)
    app.router.add_get("/api/prices", api_prices)
    app.router.add_get("/api/projection", api_projection)
    app.router.add_get("/api/solar_actuals", api_solar_actuals)
    app.router.add_get("/api/soc_actuals", api_soc_actuals)
    app.router.add_get("/api/consumption_actuals", api_consumption_actuals)
    app.router.add_get("/api/insights", api_insights)
    app.router.add_get("/api/logs", api_logs)
    app.router.add_get("/api/logs/stream", api_logs_stream)
    app.router.add_post("/api/override", api_override_set)
    app.router.add_delete("/api/override", api_override_clear)
    app.router.add_post("/api/reapply", api_reapply)
    app.router.add_get("/api/settings", api_settings_get)
    app.router.add_post("/api/settings", api_settings_post)
    app.router.add_post("/api/restart", api_restart)


def _add_page_routes(app: "web.Application") -> None:
    """Register the static page + PWA-asset routes (no controller needed).

    These serve only embedded HTML/JS/icons, so they can run in a standalone
    web container that is restarted freely without touching the controller.
    """
    app.router.add_get("/", dashboard_page)
    app.router.add_get("/settings", settings_page)
    app.router.add_get("/manifest.webmanifest", manifest_page)
    app.router.add_get("/sw.js", sw_page)
    app.router.add_get("/icon.svg", icon_svg_page)
    app.router.add_get("/icon-180.png", icon_180_page)
    app.router.add_get("/icon-192.png", icon_192_page)
    app.router.add_get("/icon-512.png", icon_512_page)


def create_dashboard_app(controller=None) -> web.Application:
    """Create the COMBINED dashboard app (pages + API in one process).

    Used for single-process / dev runs (and back-compat). Production splits
    these via create_api_app() + create_pages_app().
    """
    app = web.Application()
    if controller:
        app["controller"] = controller
    _add_page_routes(app)
    _add_api_routes(app)
    return app


def create_api_app(controller) -> web.Application:
    """API-only app for the main (controller) container."""
    app = web.Application()
    app["controller"] = controller
    _add_api_routes(app)
    return app


def create_pages_app(api_upstream: str) -> web.Application:
    """Pages-only app for the standalone web container.

    Serves the embedded HTML/assets and reverse-proxies everything under
    ``/api/`` to the main container's API app (``api_upstream``), so the
    browser sees a single origin (no CORS) and the controller-backed data
    still comes from the live service.
    """
    app = web.Application()
    app["api_upstream"] = api_upstream.rstrip("/")
    _add_page_routes(app)
    # Catch-all proxy for the JSON + control API (incl. the SSE log stream).
    app.router.add_route("*", "/api/{tail:.*}", _proxy_to_api)
    return app


# Hop-by-hop headers that must not be forwarded across the proxy (RFC 7230),
# plus content framing headers we let aiohttp recompute for the new response.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}


async def _proxy_to_api(request: "web.Request") -> "web.StreamResponse":
    """Stream-proxy a request to the main container's API app.

    Streams the upstream response body chunk-by-chunk so Server-Sent Events
    (``/api/logs/stream``) keep flowing instead of buffering forever.
    """
    import aiohttp

    upstream = request.app["api_upstream"]
    url = upstream + request.rel_url.raw_path_qs
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
    }
    body = await request.read() if request.body_exists else None
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=None)
    session = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)
    try:
        up = await session.request(
            request.method, url, headers=fwd_headers, data=body,
            allow_redirects=False,
        )
    except aiohttp.ClientError as e:
        await session.close()
        return web.json_response(
            {"error": f"API upstream unreachable: {e}"}, status=502
        )
    try:
        resp = web.StreamResponse(status=up.status)
        for k, v in up.headers.items():
            if k.lower() not in _HOP_BY_HOP:
                resp.headers[k] = v
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp
    finally:
        up.release()
        await session.close()


_dashboard_handler_installed = False


def _install_log_handler() -> None:
    """Attach the SSE log-capture handler to the controller logger once.

    Must run in the MAIN (controller) process — that's where the log records
    originate and where the SSE buffer the API serves lives.
    """
    global _dashboard_handler_installed
    if _dashboard_handler_installed:
        return
    growatt_logger = logging.getLogger("modules.base.GrowattController")
    growatt_logger.handlers = [
        h for h in growatt_logger.handlers
        if not isinstance(h, DashboardLogHandler)
    ]
    handler = DashboardLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    growatt_logger.addHandler(handler)
    _dashboard_handler_installed = True


async def _serve_app(app: "web.Application", port: int, what: str) -> "web.AppRunner":
    """Bind an aiohttp app on 0.0.0.0:port and return its runner."""
    logger = logging.getLogger(__name__)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"{what} listening on http://0.0.0.0:{port}")
    return runner


async def start_dashboard(controller, port: int = 5555) -> "web.AppRunner":
    """Start the COMBINED dashboard (pages + API) in one process.

    Back-compat / single-process dev entry. Production uses
    start_api_dashboard() (main container) + start_pages_dashboard() (web
    container) instead. Returns the AppRunner so the caller can
    ``await runner.cleanup()`` on shutdown.
    """
    _install_log_handler()
    return await _serve_app(create_dashboard_app(controller), port, "Dashboard")


async def start_api_dashboard(controller, port: int = 5556) -> "web.AppRunner":
    """Start the controller-backed API app (main container)."""
    _install_log_handler()
    return await _serve_app(create_api_app(controller), port, "Dashboard API")


async def start_pages_dashboard(api_upstream: str, port: int = 5555) -> "web.AppRunner":
    """Start the standalone pages app + API proxy (web container)."""
    return await _serve_app(create_pages_app(api_upstream), port, "Dashboard pages")


# --- Embedded Settings Editor ---

SETTINGS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Growatt Settings</title>
<style>
  :root{
    --bg:#0f1216; --card:#1a1f27; --card2:#222834; --text:#e6e9ef; --muted:#8a93a6;
    --accent:#4ea1ff; --green:#3ecf8e; --amber:#f5b942; --red:#ff6b6b; --border:#2a313d;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:15px}
  a{color:var(--accent)}
  .appbar{position:sticky;top:0;z-index:10;background:var(--card);
    border-bottom:1px solid var(--border);padding:12px 16px;
    display:flex;align-items:center;justify-content:space-between;gap:12px}
  .appbar h1{font-size:17px;margin:0;font-weight:600}
  .appbar .sub{font-size:12px;color:var(--muted)}
  .back{color:var(--accent);text-decoration:none;font-size:14px}
  .container{max-width:780px;margin:0 auto;padding:16px;padding-bottom:120px}
  .group{background:var(--card);border:1px solid var(--border);border-radius:12px;
    margin-bottom:16px;overflow:hidden}
  .group > h2{margin:0;padding:14px 16px;font-size:15px;background:var(--card2);
    border-bottom:1px solid var(--border)}
  .group .gdesc{padding:8px 16px 0;font-size:12.5px;color:var(--muted)}
  .field{display:flex;align-items:center;gap:12px;padding:12px 16px;
    border-top:1px solid var(--border);flex-wrap:wrap}
  .field:first-of-type{border-top:none}
  .field .meta{flex:1 1 230px;min-width:200px}
  .field .lbl{font-weight:500;display:flex;align-items:center;gap:8px}
  .field .help{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.4}
  .field .ctl{flex:0 0 auto;display:flex;align-items:center;gap:8px}
  .field input[type=number],.field input[type=text],.field select{
    background:var(--bg);color:var(--text);border:1px solid var(--border);
    border-radius:8px;padding:8px 10px;font-size:14px;width:130px;text-align:right}
  .field input[type=text]{width:170px;text-align:left}
  .field select{width:130px;text-align:left}
  .field .unit{color:var(--muted);font-size:12.5px;min-width:48px}
  .field.dirty{background:rgba(78,161,255,.07)}
  .field.dirty input,.field.dirty select{border-color:var(--accent)}
  .badge{font-size:10px;font-weight:700;letter-spacing:.04em;padding:2px 7px;
    border-radius:999px;text-transform:uppercase}
  .badge.restart{background:rgba(245,185,66,.16);color:var(--amber)}
  .switch{position:relative;width:46px;height:26px}
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:var(--card2);border:1px solid var(--border);
    border-radius:999px;cursor:pointer;transition:.15s}
  .slider:before{content:"";position:absolute;height:18px;width:18px;left:3px;top:3px;
    background:var(--muted);border-radius:50%;transition:.15s}
  .switch input:checked + .slider{background:rgba(62,207,142,.25);border-color:var(--green)}
  .switch input:checked + .slider:before{transform:translateX(20px);background:var(--green)}
  .savebar{position:fixed;left:0;right:0;bottom:0;background:var(--card);
    border-top:1px solid var(--border);padding:12px 16px;display:flex;
    align-items:center;gap:12px;justify-content:space-between}
  .savebar .info{font-size:13px;color:var(--muted)}
  .btn{border:none;border-radius:10px;padding:11px 20px;font-size:15px;font-weight:600;
    cursor:pointer}
  .btn.primary{background:var(--accent);color:#05121f}
  .btn.primary:disabled{opacity:.4;cursor:not-allowed}
  .btn.ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
  .toast{position:fixed;left:50%;bottom:78px;transform:translateX(-50%);
    background:var(--card2);border:1px solid var(--border);border-radius:10px;
    padding:12px 16px;max-width:90%;font-size:13.5px;box-shadow:0 6px 24px rgba(0,0,0,.4);
    opacity:0;pointer-events:none;transition:opacity .2s}
  .toast.show{opacity:1}
  .toast.ok{border-color:var(--green)}
  .toast.err{border-color:var(--red)}
  .loading{padding:40px;text-align:center;color:var(--muted)}
  #restartBtn.pending{border-color:var(--amber);color:var(--amber);
    animation:pulse 1.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
  .overlay{position:fixed;inset:0;background:rgba(7,9,12,.82);display:none;
    align-items:center;justify-content:center;z-index:50}
  .overlay.show{display:flex}
  .overlay-box{background:var(--card);border:1px solid var(--border);
    border-radius:14px;padding:28px 32px;text-align:center;max-width:340px}
  .overlay-sub{font-size:12.5px;color:var(--muted);margin-top:8px}
  .spinner{width:34px;height:34px;margin:0 auto 14px;border-radius:50%;
    border:3px solid var(--border);border-top-color:var(--accent);
    animation:spin .9s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header class="appbar">
  <div>
    <h1>&#9881;&#65039; Settings</h1>
    <div class="sub">Live edits to the battery controller</div>
  </div>
  <a class="back" href="/">&larr; Dashboard</a>
</header>
<div class="container" id="root"><div class="loading">Loading settings&hellip;</div></div>

<div class="savebar">
  <span class="info" id="saveInfo">No changes</span>
  <div style="display:flex;gap:10px">
    <button class="btn ghost" id="restartBtn" title="Restart the controller to apply restart-only settings (e.g. forecast engine)">&#8635; Restart</button>
    <button class="btn ghost" id="resetBtn">Reset</button>
    <button class="btn primary" id="saveBtn" disabled>Save changes</button>
  </div>
</div>
<div class="toast" id="toast"></div>
<div class="overlay" id="overlay">
  <div class="overlay-box">
    <div class="spinner"></div>
    <div id="overlayMsg">Restarting controller&hellip;</div>
    <div class="overlay-sub" id="overlaySub">Applying restart-only settings. This takes ~30&ndash;40s.</div>
  </div>
</div>

<script>
const root = document.getElementById('root');
const saveBtn = document.getElementById('saveBtn');
const resetBtn = document.getElementById('resetBtn');
const restartBtn = document.getElementById('restartBtn');
const saveInfo = document.getElementById('saveInfo');
let SCHEMA = [];
const original = {};   // name -> original value
const inputs = {};     // name -> input element

function fmt(v){ return (v===null||v===undefined) ? '' : v; }

function showToast(msg, kind){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + (kind||'');
  setTimeout(()=>{ t.className = 'toast ' + (kind||''); }, kind==='err'?6000:4200);
}

function dirtyFields(){
  const out = {};
  for (const name in inputs){
    const el = inputs[name];
    let val;
    if (el.type === 'checkbox') val = el.checked;
    else val = el.value;
    // Normalise for comparison
    const orig = original[name];
    let changed;
    if (el.type === 'checkbox') changed = (val !== !!orig);
    else if (el.dataset.type === 'str') changed = (String(val) !== String(fmt(orig)));
    else changed = (val.trim() !== String(fmt(orig)).trim());
    if (changed) out[name] = val;
  }
  return out;
}

function refreshDirty(){
  const d = dirtyFields();
  const n = Object.keys(d).length;
  saveBtn.disabled = (n === 0);
  saveInfo.textContent = n === 0 ? 'No changes' : (n + ' change' + (n>1?'s':''));
  for (const name in inputs){
    const fieldEl = inputs[name].closest('.field');
    if (name in d) fieldEl.classList.add('dirty');
    else fieldEl.classList.remove('dirty');
  }
}

function makeControl(f){
  let el;
  if (f.type === 'bool'){
    const wrap = document.createElement('label');
    wrap.className = 'switch';
    el = document.createElement('input');
    el.type = 'checkbox';
    el.checked = !!f.value;
    const sl = document.createElement('span'); sl.className = 'slider';
    wrap.appendChild(el); wrap.appendChild(sl);
    el.dataset.type = f.type;
    el.addEventListener('change', refreshDirty);
    inputs[f.name] = el;
    return wrap;
  }
  if (f.choices){
    el = document.createElement('select');
    f.choices.forEach(c => {
      const o = document.createElement('option');
      o.value = c; o.textContent = c;
      if (String(c) === String(f.value)) o.selected = true;
      el.appendChild(o);
    });
  } else if (f.type === 'str'){
    el = document.createElement('input'); el.type = 'text';
    el.value = fmt(f.value);
  } else {
    el = document.createElement('input'); el.type = 'number';
    el.value = fmt(f.value);
    el.step = (f.type === 'int') ? '1' : 'any';
    if (f.min !== null && f.min !== undefined) el.min = f.min;
    if (f.max !== null && f.max !== undefined) el.max = f.max;
    if (f.name.indexOf('export_czk') >= 0) el.placeholder = '(shared)';
  }
  el.dataset.type = f.type;
  el.addEventListener('input', refreshDirty);
  el.addEventListener('change', refreshDirty);
  inputs[f.name] = el;
  return el;
}

function render(){
  root.innerHTML = '';
  SCHEMA.forEach(group => {
    const g = document.createElement('div'); g.className = 'group';
    const h = document.createElement('h2'); h.textContent = group.title; g.appendChild(h);
    if (group.description){
      const d = document.createElement('div'); d.className = 'gdesc'; d.textContent = group.description;
      g.appendChild(d);
    }
    group.fields.forEach(f => {
      original[f.name] = f.value;
      const row = document.createElement('div'); row.className = 'field';
      const meta = document.createElement('div'); meta.className = 'meta';
      const lbl = document.createElement('div'); lbl.className = 'lbl';
      lbl.appendChild(document.createTextNode(f.label));
      if (!f.hot){
        const b = document.createElement('span'); b.className='badge restart';
        b.textContent='restart'; b.title='Takes effect after a container restart';
        lbl.appendChild(b);
      }
      meta.appendChild(lbl);
      if (f.help){
        const hp = document.createElement('div'); hp.className='help'; hp.textContent=f.help;
        meta.appendChild(hp);
      }
      const ctl = document.createElement('div'); ctl.className = 'ctl';
      ctl.appendChild(makeControl(f));
      if (f.unit){ const u=document.createElement('span'); u.className='unit'; u.textContent=f.unit; ctl.appendChild(u); }
      row.appendChild(meta); row.appendChild(ctl);
      g.appendChild(row);
    });
    root.appendChild(g);
  });
  refreshDirty();
}

async function load(){
  try{
    const r = await fetch('/api/settings');
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Failed to load');
    SCHEMA = j.groups || [];
    render();
  }catch(e){
    root.innerHTML = '<div class="loading">Failed to load settings: ' + e.message + '</div>';
  }
}

resetBtn.addEventListener('click', () => { load(); });

saveBtn.addEventListener('click', async () => {
  const updates = dirtyFields();
  if (!Object.keys(updates).length) return;
  saveBtn.disabled = true; saveInfo.textContent = 'Saving…';
  try{
    const r = await fetch('/api/settings', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({updates})
    });
    const j = await r.json();
    if (!r.ok || !j.success) throw new Error(j.message || 'Save failed');
    showToast(j.message || 'Saved', 'ok');
    // A restart-only field changed -> draw attention to the Restart button.
    if (j.restart_required && j.restart_required.length){
      restartBtn.classList.add('pending');
    }
    await load();   // re-pull authoritative values
  }catch(e){
    showToast(e.message, 'err');
    refreshDirty();
  }
});

const overlay = document.getElementById('overlay');
const overlayMsg = document.getElementById('overlayMsg');
const overlaySub = document.getElementById('overlaySub');
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function pollUntilBack(){
  // Wait out the shutdown, then poll /api/status until the controller answers.
  await sleep(6000);
  for (let i = 0; i < 40; i++){   // ~2 min budget
    try{
      const r = await fetch('/api/status', {cache:'no-store'});
      if (r.ok){ return true; }
    }catch(_){ /* main still down / proxy 502 */ }
    overlaySub.textContent = 'Waiting for controller to come back… (' + ((i+1)*3) + 's)';
    await sleep(3000);
  }
  return false;
}

restartBtn.addEventListener('click', async () => {
  if (!confirm('Restart the controller now? It will be unavailable for ~30–40s while it reloads (model rebuild + restart-only settings).')) return;
  overlay.classList.add('show');
  overlayMsg.textContent = 'Restarting controller…';
  overlaySub.textContent = 'Applying restart-only settings. This takes ~30–40s.';
  try{
    await fetch('/api/restart', {method:'POST'});
  }catch(_){ /* the process may drop the connection as it exits — expected */ }
  const back = await pollUntilBack();
  if (back){
    overlayMsg.textContent = 'Controller is back ✓';
    restartBtn.classList.remove('pending');
    await load();
    await sleep(600);
    overlay.classList.remove('show');
    showToast('Controller restarted — settings applied', 'ok');
  }else{
    overlayMsg.textContent = 'Still waiting…';
    overlaySub.textContent = 'The controller has not answered yet. It may still be rebuilding models — reload the page in a moment.';
  }
});

load();
</script>
</body>
</html>"""


# --- Embedded HTML Dashboard ---

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Growatt Battery Dashboard</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#0f1117">
<link rel="icon" type="image/svg+xml" href="/icon.svg">
<link rel="apple-touch-icon" href="/icon-180.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Growatt">
<meta name="application-name" content="Growatt">
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
  font-size: 15px;
  line-height: 1.5;
  -webkit-text-size-adjust: 100%;
}
/* ===== App bar (always-visible live header) ===== */
.appbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: linear-gradient(180deg, #1c2030 0%, var(--card) 100%);
  border-bottom: 1px solid var(--border);
  padding: calc(8px + env(safe-area-inset-top)) 12px 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.35);
}
.appbar-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.brand { display: flex; align-items: center; gap: 8px; min-width: 0; }
.brand-name { font-size: 15px; font-weight: 700; letter-spacing: 0.2px; }
.inv-state {
  font-size: 10px; font-weight: 700; letter-spacing: 0.4px;
  padding: 2px 7px; border-radius: 999px; text-transform: uppercase;
}
.appbar-meta { display: flex; align-items: center; gap: 10px; }
.live-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
  color: var(--green); background: rgba(34,197,94,0.12);
  padding: 3px 8px; border-radius: 999px; transition: opacity .3s, color .3s, background .3s;
}
.live-pill.stale { color: var(--muted); background: rgba(139,143,163,0.12); }
.live-pill .live-dot {
  width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  animation: pulse 1.6s infinite;
}
.status-dot {
  width: 11px; height: 11px; border-radius: 50%;
  background: var(--muted); flex: 0 0 auto;
  box-shadow: 0 0 0 0 rgba(0,0,0,0);
  transition: background .3s, box-shadow .3s;
}
.status-dot.ok { background: var(--green); box-shadow: 0 0 8px rgba(34,197,94,0.7); animation: pulse 2.2s infinite; }
.status-dot.off { background: var(--red); box-shadow: 0 0 8px rgba(239,68,68,0.7); }
.status-dot.stale { background: var(--muted); }
.refresh-btn {
  background: none; border: none; color: var(--muted); cursor: pointer;
  font-size: 16px; padding: 2px 4px; line-height: 1;
  -webkit-tap-highlight-color: transparent;
}
.refresh-btn:active { color: var(--accent); }
.refresh-btn.spinning { animation: spin .8s linear infinite; color: var(--accent); }
@keyframes spin { to { transform: rotate(360deg); } }
.last-update { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }

/* ===== Live stat strip (the always-on glanceable summary) ===== */
.statstrip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
}
.statstrip.stale { opacity: 0.45; transition: opacity .3s; }
.ss-item {
  display: flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 7px 8px;
  min-width: 0;
}
.ss-ico { font-size: 17px; line-height: 1; flex: 0 0 auto; }
.ss-body { min-width: 0; }
.ss-val {
  font-size: 15px; font-weight: 800; line-height: 1.1;
  white-space: nowrap; font-variant-numeric: tabular-nums;
  transition: color .25s;
}
.ss-val.flash { animation: ssflash .6s ease-out; }
@keyframes ssflash { 0% { background: rgba(79,140,255,0.25); } 100% { background: transparent; } }
.ss-lbl { font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 380px) {
  .ss-item { flex-direction: column; align-items: flex-start; gap: 1px; padding: 6px; }
  .ss-ico { font-size: 14px; }
  .ss-val { font-size: 14px; }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

/* ===== Pull-to-refresh ===== */
.ptr {
  position: fixed; top: 0; left: 0; right: 0;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  height: 0; overflow: hidden; opacity: 0;
  color: var(--muted); font-size: 12px; font-weight: 600;
  z-index: 200; pointer-events: none;
  padding-top: env(safe-area-inset-top);
}
.ptr .ptr-spin {
  width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--border); border-top-color: var(--accent);
  transition: transform .1s;
}
.ptr.ready .ptr-spin { border-top-color: var(--green); }
.ptr.refreshing .ptr-spin { animation: spin .7s linear infinite; }
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 12px;
  /* Clear the fixed bottom tab bar on mobile (removed at >=900px). */
  padding-bottom: calc(72px + env(safe-area-inset-bottom));
}
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

/* Horizontal-scroll viewport for the price chart on narrow screens. The bars
   AND the SVG overlays live inside .chart-content so they scroll together and
   stay aligned. On wide screens the content fits and there is no scroll. */
.chart-scroll {
  overflow-x: auto;
  overflow-y: visible;
  -webkit-overflow-scrolling: touch;
  /* Let the browser own horizontal panning here (so dragging a bar scrolls the
     chart) while vertical drags still scroll the page. */
  touch-action: pan-x;
  overscroll-behavior-x: contain;
  scrollbar-width: thin;
  padding-bottom: 6px;
}
.chart-scroll::-webkit-scrollbar { height: 8px; }
.chart-scroll::-webkit-scrollbar-track { background: var(--bg); border-radius: 4px; }
.chart-scroll::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; opacity: 0.7; }
.chart-content { position: relative; }
/* High-load protection banner (home tab) */
.hl-banner {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 12px; padding: 11px 14px;
  border-radius: 10px;
  background: rgba(245, 185, 66, 0.14);
  border: 1px solid var(--amber);
  color: var(--text);
  font-size: 14px;
}
.hl-banner .hl-ico { font-size: 20px; line-height: 1; }
.hl-banner b { color: var(--amber); }
.hl-banner .hl-sub { color: var(--muted); font-size: 12.5px; }
.price-chart {
  height: 180px;
  display: flex;
  align-items: flex-end;
  gap: 1px;
  margin-top: 8px;
  position: relative;
}
/* 0 CZK/kWh reference line; bars below it are negative (you're paid). */
.zero-line {
  position: absolute;
  left: 0; right: 0;
  height: 0;
  border-top: 1px dashed var(--muted);
  opacity: 0.55;
  pointer-events: none;
  z-index: 1;
}
/* "Now" divider drawn in the chart between elapsed and upcoming blocks. */
.now-divider {
  position: absolute;
  top: 0; bottom: 0;
  width: 0;
  border-left: 2px dashed var(--accent);
  opacity: 0.7;
  pointer-events: none;
}
.now-divider::after {
  content: "NOW";
  position: absolute;
  top: -2px; left: 3px;
  font-size: 9px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.5px;
}
.soc-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  margin: 6px 0 2px;
  font-size: 11px;
  color: var(--muted);
}
.soc-legend.legend-actions { margin-top: 8px; }
.soc-legend-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.soc-legend-item svg { overflow: visible; flex: 0 0 auto; }
.legend-swatch {
  width: 12px; height: 12px; border-radius: 3px; flex: 0 0 auto;
  display: inline-block;
}
.legend-swatch.striped {
  background: repeating-linear-gradient(45deg,#555 0,#555 3px,#2a2a2a 3px,#2a2a2a 6px);
}
.alert-banner {
  background: #4a2a14;
  color: #fbbf24;
  border: 1px solid #92400e;
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 6px;
  font-size: 13px;
  font-weight: 600;
}
.price-chart-label {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 2px;
  display: flex;
  justify-content: space-between;
}
/* Each bar is a FULL-HEIGHT transparent column anchored on a 0 line at the
   bottom. Inside it: a distribution-fee segment sitting on 0, the spot price
   stacked on top of the fee (positive prices), and — for NEGATIVE spot — a
   segment hung from the TOP of the chart pointing down. The per-bar colour is
   carried in --bar-color so the segments can read it. */
.price-bar {
  flex: 0 0 var(--bar-w, 2px);
  width: var(--bar-w, 2px);
  position: relative;
  height: 100%;
  background: transparent;
  transition: opacity 0.2s;
}
.price-bar:hover { opacity: 0.8; cursor: pointer; }
.price-bar .seg-spot, .price-bar .seg-neg, .price-bar .seg-fee {
  position: absolute; left: 0; right: 0; pointer-events: none;
}
.price-bar .seg-fee {                /* distribution fee (bottom/top set inline) */
  background: rgba(148, 163, 184, 0.45);
}
.price-bar .seg-spot {               /* spot price, stacked above the fee */
  background: var(--bar-color, #4f8cff44);
  border-radius: 2px 2px 0 0;
}
.price-bar .seg-neg {                /* negative spot, hung from the top */
  top: 0;
  background: var(--bar-color, #22c55e);
  border-radius: 0 0 2px 2px;
}
.price-bar.charging      { --bar-color: var(--green); }
.price-bar.pre-discharge { --bar-color: #c084fc; }
.price-bar.discharge     { --bar-color: var(--red); }
.price-bar.sell-production { --bar-color: #f97316; }
.price-bar.battery-hold  { --bar-color: #0ea5e9; }
.price-bar.negative      { --bar-color: #22c55e; }
.price-bar.cheap         { --bar-color: #4f8cff88; }
.price-bar.mid           { --bar-color: #eab308aa; }
.price-bar.expensive     { --bar-color: #ef4444aa; }
.price-bar.inverter-off .seg-spot, .price-bar.inverter-off .seg-neg {
  background: repeating-linear-gradient(45deg, #555 0, #555 4px, #2a2a2a 4px, #2a2a2a 8px);
}
.price-bar.current { outline: 2px solid var(--accent); outline-offset: -1px; border-radius: 2px; }

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
  max-width: min(260px, 92vw);
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
.chart-tooltip .tt-battery-hold { background: #0c2a3a; color: #0ea5e9; }
.chart-tooltip .tt-inverter-off { background: #2a2a2a; color: #aaa; }
.chart-tooltip .tt-current { background: #1e3a5f; color: var(--accent); }

.log-box {
  background: #0d0f14;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px;
  height: min(50vh, 300px);
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
.override-panel > * { flex: 1 1 100%; }
.btn {
  min-height: 44px;
  padding: 10px 16px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--text);
  cursor: pointer;
  font-size: 15px;
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
  min-height: 44px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font-size: 15px;
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

/* ===== App tabs ===== */
.tab-page { display: none; }
.tab-page.active { display: block; }
.tabbar {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  z-index: 150;
  display: flex;
  background: var(--card);
  border-top: 1px solid var(--border);
  padding-bottom: env(safe-area-inset-bottom);
}
.tab-btn {
  flex: 1;
  min-width: 44px;
  min-height: 56px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  background: none;
  border: none;
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: color 0.15s;
}
.tab-btn .tab-ico { font-size: 20px; line-height: 1; }
.tab-btn.active { color: var(--accent); }
.tab-btn:active { transform: scale(0.94); }

/* ===== Grid hooks (replace per-section inline grid-template-columns) ===== */
.power-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.two-col { display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 12px; }

/* ===== Mobile-first breakpoints ===== */
@media (min-width: 600px) {
  .power-grid { grid-template-columns: repeat(4, 1fr); }
  .stat-value { font-size: 24px; }
}
@media (min-width: 900px) {
  .two-col { grid-template-columns: 1fr 1fr; }
  /* Desktop: the tab bar becomes a static top strip directly under the (sticky)
     header — placed before .container in the DOM — instead of a fixed bottom bar. */
  .container { padding-bottom: 12px; }
  .tabbar {
    position: static;
    bottom: auto;
    border-top: none;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
    justify-content: center;
    gap: 8px;
  }
  .tab-btn {
    flex: 0 1 auto;
    flex-direction: row;
    gap: 8px;
    min-height: 48px;
    padding: 0 18px;
    font-size: 13px;
  }
  .tab-btn .tab-ico { font-size: 16px; }
  .tab-btn.active { border-bottom: 2px solid var(--accent); }
}

/* ===== Polish: card depth, tab transitions, smooth scrolling ===== */
html { scroll-behavior: smooth; }
.card { transition: border-color .2s ease, box-shadow .2s ease; }
@media (hover: hover) and (min-width: 900px) {
  .card:hover { border-color: #353a4d; box-shadow: 0 6px 20px rgba(0,0,0,0.35); }
}
.tab-page.active { animation: tabin .28s cubic-bezier(.22,.61,.36,1); }
@keyframes tabin { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.btn, button { transition: transform .1s ease, filter .15s ease; }
.btn:active { transform: scale(0.97); }
.soc-fill { transition: width .5s cubic-bezier(.22,.61,.36,1), background .3s; }

/* Compact Home "Today" chart: 96 bars fit the width (flex), no scroll. */
.home-chart { position: relative; height: 134px; margin-top: 10px; }
.home-chart .price-chart { height: 134px; margin-top: 0; gap: 0; }
.home-chart .price-bar { flex: 1 1 0; min-width: 0; width: auto; }
.home-chart-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
.home-now-line { position: absolute; top: 0; bottom: 0; width: 0; border-left: 2px dashed var(--accent); opacity: .6; pointer-events: none; }
.home-legend { display: flex; flex-wrap: wrap; gap: 3px 11px; margin-top: 9px; font-size: 10.5px; color: var(--muted); }
.home-legend .lg { display: inline-flex; align-items: center; gap: 4px; }
.home-legend .lg i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.home-legend .lg.line i { width: 13px; height: 0; border-top: 2px solid; border-radius: 0; }

/* Today's money grid */
.money-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
.money-item { text-align: center; background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 9px; padding: 9px 4px; }
.money-val { font-size: 16px; font-weight: 800; white-space: nowrap; font-variant-numeric: tabular-nums; }
.money-lbl { font-size: 10px; color: var(--muted); margin-top: 2px; }
@media (max-width: 430px) { .money-grid { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<div class="ptr" id="ptr"><div class="ptr-spin"></div><span class="ptr-label">Pull to refresh</span></div>

<header class="appbar">
  <div class="appbar-top">
    <div class="brand">
      <span class="status-dot stale" id="statusDot" title="Inverter status"></span>
      <span class="brand-name">Growatt</span>
      <span id="invPowerBadge" class="inv-state" style="display:none"></span>
    </div>
    <div class="appbar-meta">
      <span class="live-pill" id="livePill"><span class="live-dot"></span>LIVE</span>
      <span class="last-update" id="lastUpdate">--</span>
      <a class="refresh-btn" href="/settings" title="Settings" aria-label="Settings" style="text-decoration:none">&#9881;&#65039;</a>
      <button class="refresh-btn" id="refreshBtn" title="Refresh" aria-label="Refresh">&#8635;</button>
    </div>
  </div>
  <div class="statstrip" id="statStrip">
    <div class="ss-item">
      <span class="ss-ico">&#9728;&#65039;</span>
      <div class="ss-body"><div class="ss-val" id="ssSolar" style="color:var(--yellow)">--</div><div class="ss-lbl">Solar</div></div>
    </div>
    <div class="ss-item">
      <span class="ss-ico">&#127968;</span>
      <div class="ss-body"><div class="ss-val" id="ssLoad">--</div><div class="ss-lbl">Home load</div></div>
    </div>
    <div class="ss-item">
      <span class="ss-ico">&#128267;</span>
      <div class="ss-body"><div class="ss-val" id="ssBatt">--</div><div class="ss-lbl" id="ssBattLbl">Battery</div></div>
    </div>
    <div class="ss-item">
      <span class="ss-ico" id="ssGridIco">&#9889;</span>
      <div class="ss-body"><div class="ss-val" id="ssGrid">--</div><div class="ss-lbl" id="ssGridLbl">Grid</div></div>
    </div>
  </div>
</header>

<nav class="tabbar" id="tabbar">
  <button class="tab-btn active" data-tab="home"><span class="tab-ico">&#9889;</span><span>Home</span></button>
  <button class="tab-btn" data-tab="chart"><span class="tab-ico">&#128202;</span><span>Prices</span></button>
  <button class="tab-btn" data-tab="insights"><span class="tab-ico">&#128161;</span><span>Insights</span></button>
  <button class="tab-btn" data-tab="control"><span class="tab-ico">&#9881;&#65039;</span><span>Control</span></button>
</nav>

<div class="container">
  <section class="tab-page active" id="tab-home">
  <!-- High-load protection banner: shown only while a big load (EV/heating) is
       active and the battery is being held off discharge. -->
  <div id="highLoadBanner" class="hl-banner" style="display:none"></div>
  <!-- Today overview: compact full-day price + SOC, no horizontal scroll -->
  <div class="card" style="margin-bottom:12px">
    <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap">
      <h2 style="margin:0">Today</h2>
      <div id="homeNow" style="font-size:12px;color:var(--muted)">--</div>
    </div>
    <div id="homeChartNoData" style="color:var(--muted);text-align:center;padding:24px;display:none">Loading…</div>
    <div class="home-chart" id="homeChartWrap">
      <div class="price-chart" id="homeChartBars"></div>
      <svg id="homeConsLine" class="home-chart-svg"></svg>
      <svg id="homeSolarLine" class="home-chart-svg"></svg>
      <svg id="homeChartSoc" class="home-chart-svg"></svg>
      <div class="home-now-line" id="homeNowLine"></div>
    </div>
    <div class="home-legend" id="homeLegend"></div>
  </div>

  <!-- Today's money -->
  <div class="card" style="margin-bottom:12px">
    <div style="display:flex;justify-content:space-between;align-items:baseline">
      <h2 style="margin:0">&#128176; Today&#39;s money</h2>
      <span id="moneyNet" style="font-size:21px;font-weight:800">--</span>
    </div>
    <div class="money-grid">
      <div class="money-item"><div class="money-val" id="moneyEarned" style="color:var(--green)">--</div><div class="money-lbl">Earned · export</div></div>
      <div class="money-item"><div class="money-val" id="moneySpent" style="color:var(--red)">--</div><div class="money-lbl">Spent · import</div></div>
      <div class="money-item"><div class="money-val" id="moneyArb" style="color:var(--accent)">--</div><div class="money-lbl">Arbitrage</div></div>
      <div class="money-item"><div class="money-val" id="moneyPlan">--</div><div class="money-lbl">Optimizer value</div></div>
    </div>
    <div class="stat-label" style="margin-top:8px;font-size:11px;opacity:.65" id="moneyNote">Net = earned − spent · estimated from today&#39;s totals × prices</div>
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
        <div class="stat-label">CZK/kWh <span style="color:var(--muted)">spot</span></div>
      </div>
      <div class="stat" style="margin-top:6px">
        <div class="stat-label" id="priceBuy" style="font-size:13px">
          + <span id="priceDist">--</span> tariff =
          <b style="color:var(--text)" id="priceAllIn">--</b> CZK/kWh to buy
        </div>
      </div>
      <div class="stat" style="margin-top:8px">
        <div class="stat-label">Block: <span id="priceBlock">--</span></div>
      </div>
    </div>

    <!-- Solar Forecast -->
    <div class="card">
      <h2>Solar Forecast</h2>
      <div style="display:flex;gap:18px">
        <div style="flex:1">
          <div class="stat-label">Today</div>
          <div class="stat-value" id="solarToday" style="color:var(--yellow)">-- kWh</div>
        </div>
        <div style="flex:1">
          <div class="stat-label">Tomorrow</div>
          <div class="stat-value" id="solarTomorrow">-- kWh</div>
        </div>
      </div>
      <div class="stat-label" id="solcastInfo" style="margin-top:8px"></div>
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

  </section>

  <section class="tab-page" id="tab-insights">
  <!-- Decision Reasoning -->
  <div class="two-col">
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
      <h2>Optimizer & Forecast <span id="engineBadge" style="font-size:11px;padding:2px 7px;border-radius:6px;vertical-align:middle"></span></h2>
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

  <!-- Alerts strip -->
  <div id="alertsStrip" style="margin-bottom:12px"></div>

  <!-- Insights -->
  <div class="grid" style="grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));margin-bottom:12px">
    <div class="card">
      <h2>💰 Economics (today)</h2>
      <div style="font-size:13px;line-height:1.9">
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Optimizer plan value</span><span id="econPlan" style="font-weight:700">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Export revenue (est.)</span><span id="econExport">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Import cost (est.)</span><span id="econImport">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Battery arbitrage (est.)</span><span id="econArbitrage">--</span></div>
        <div class="stat-label" style="margin-top:6px;font-size:11px;opacity:.7" id="econNote">estimates from daily totals × prices</div>
      </div>
    </div>
    <div class="card">
      <h2>🎯 Forecast Accuracy</h2>
      <div style="font-size:13px;line-height:1.9">
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Solar so far (actual/forecast)</span><span id="accSolar">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Solar error</span><span id="accSolarErr">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Intraday calibration</span><span id="accCalib">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Solar confidence</span><span id="accConf">--</span></div>
      </div>
    </div>
    <div class="card">
      <h2>⚙️ Engines &amp; Sources</h2>
      <div style="font-size:13px;line-height:1.9">
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Optimizer</span><span id="engOpt">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Consumption</span><span id="engCons">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Solcast budget</span><span id="engSolcast">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Last command</span><span id="engCmd">--</span></div>
        <div class="stat" style="display:flex;justify-content:space-between"><span class="stat-label">Prices updated</span><span id="engFresh">--</span></div>
        <div style="margin-top:10px">
          <button id="reapplyBtn" onclick="reapplyState()" style="width:100%;padding:8px;border:1px solid var(--border);border-radius:6px;background:#1e3a5f;color:var(--accent);font-weight:600;cursor:pointer">🔄 Re-apply current state</button>
          <div id="reapplyResult" style="font-size:12px;margin-top:6px;min-height:14px"></div>
        </div>
      </div>
    </div>
    <div class="card" id="deferrableCard" style="display:none">
      <h2>🔁 Deferrable Loads</h2>
      <div id="deferrableList" style="font-size:13px;line-height:1.8">--</div>
    </div>
  </div>

  </section>

  <section class="tab-page" id="tab-chart">
  <!-- Price Chart -->
  <div class="card" style="margin-bottom:12px">
    <h2 id="priceChartTitle">Today's Prices &amp; Battery SOC</h2>
    <div style="font-size:11px;color:var(--muted);margin-bottom:2px">
      &#128202; Drag sideways to scroll &middot; tap a bar for details &middot;
      <b style="color:var(--accent)">NOW</b> line splits done vs planned
    </div>
    <!-- Overlay lines: one swatch per metric; solid = actual, dashed = forecast -->
    <div class="soc-legend">
      <span class="soc-legend-item"><svg width="22" height="8"><line x1="0" y1="3" x2="11" y2="3" stroke="#f97316" stroke-width="2.5"/><line x1="11" y1="3" x2="22" y2="3" stroke="#f97316" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="3 2"/></svg> SOC %</span>
      <span class="soc-legend-item"><svg width="22" height="8"><line x1="0" y1="3" x2="11" y2="3" stroke="#fde047" stroke-width="2.5"/><line x1="11" y1="3" x2="22" y2="3" stroke="#fde047" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="3 2"/></svg> Solar</span>
      <span class="soc-legend-item"><svg width="22" height="8"><line x1="0" y1="3" x2="11" y2="3" stroke="#38bdf8" stroke-width="2.5"/><line x1="11" y1="3" x2="22" y2="3" stroke="#38bdf8" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="3 2"/></svg> Load</span>
      <span class="soc-legend-item" style="opacity:.85">&#9472; actual &middot; &#9476; forecast</span>
    </div>
    <!-- Bar action colours (what the battery is doing each 15-min block) -->
    <div class="soc-legend legend-actions">
      <span class="soc-legend-item"><span class="legend-swatch" style="background:var(--green)"></span> Charge</span>
      <span class="soc-legend-item"><span class="legend-swatch" style="background:var(--red)"></span> Discharge</span>
      <span class="soc-legend-item"><span class="legend-swatch" style="background:#f97316"></span> Sell solar</span>
      <span class="soc-legend-item"><span class="legend-swatch" style="background:#0ea5e9"></span> Battery hold</span>
      <span class="soc-legend-item"><span class="legend-swatch" style="background:#c084fc"></span> Pre-discharge</span>
      <span class="soc-legend-item"><span class="legend-swatch striped"></span> Inverter off</span>
    </div>
    <div class="price-chart-day" id="priceChartTodayWrap">
      <div class="price-chart-label" id="priceChartTodayLabel">Today</div>
      <div class="chart-scroll"><div class="chart-content">
        <div class="price-chart" id="priceChartToday"></div>
        <svg id="consLineToday" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
        <svg id="solarLineToday" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
        <svg id="socLineToday" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
      </div></div>
    </div>
    <div class="price-chart-day" id="priceChartTomorrowWrap" style="display:none;margin-top:12px">
      <div class="price-chart-label" id="priceChartTomorrowLabel">Tomorrow</div>
      <div class="chart-scroll"><div class="chart-content">
        <div class="price-chart" id="priceChartTomorrow"></div>
        <svg id="consLineTomorrow" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
        <svg id="solarLineTomorrow" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
        <svg id="socLineTomorrow" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;overflow:visible"></svg>
      </div></div>
    </div>
  </div>
  <div class="chart-tooltip" id="chartTooltip"></div>
  </section>

  <section class="tab-page" id="tab-control">
  <!-- Override Controls -->
  <div class="card" style="margin-bottom:12px">
    <h2>Manual Override</h2>
    <div id="overrideStatus" style="margin-bottom:8px;font-size:13px"></div>
    <div class="override-panel">
      <select id="overrideMode">
        <option value="regular">Regular</option>
        <option value="charge_from_grid">Charge from Grid</option>
        <option value="discharge_to_grid">Discharge to Grid</option>
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
  </section>
</div>

<script>
let lastData = null;

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    lastData = data;
    updateUI(data);
    markFresh();
  } catch (e) {
    setDot('stale');
    document.getElementById('livePill').classList.add('stale');
  }
}

function updateUI(d) {
  // Time
  const t = new Date(d.timestamp);
  document.getElementById('lastUpdate').textContent = t.toLocaleTimeString();

  // Inverter status → header dot (green=on, red=off, gray=unknown) + small pill
  const invBadge = document.getElementById('invPowerBadge');
  if (d.inverter && typeof d.inverter.on === 'boolean') {
    if (d.inverter.on) {
      setDot('ok');
      invBadge.style.display = 'inline-block';
      invBadge.style.background = 'rgba(34,197,94,0.15)';
      invBadge.style.color = 'var(--green)';
      invBadge.textContent = 'ON';
    } else {
      setDot('off');
      invBadge.style.display = 'inline-block';
      invBadge.style.background = 'rgba(239,68,68,0.15)';
      invBadge.style.color = 'var(--red)';
      invBadge.textContent = 'OFF';
    }
  } else {
    setDot('stale');
    invBadge.style.display = 'none';
  }

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

  // High-load protection banner (prominent, top of home tab) + tag.
  // ev_power from the controller is already in kW.
  const hlBanner = document.getElementById('highLoadBanner');
  let hlParts = [];
  if (d.high_loads_active) {
    if (d.high_loads?.ev_charging) hlParts.push('🚗 EV charging ' + (d.high_loads.ev_power || 0).toFixed(1) + ' kW');
    if (d.high_loads?.heating_relays?.length) hlParts.push('🔥 heating (' + d.high_loads.heating_relays.join(', ') + ')');
  }
  if (hlBanner) {
    if (d.high_loads_active) {
      const protectedNow = d.mode === 'high_load_protected';
      hlBanner.innerHTML =
        '<span class="hl-ico">⚡</span>' +
        '<div><b>High load — ' + (hlParts.join(' · ') || 'active') + '</b>' +
        '<div class="hl-sub">' + (protectedNow
          ? 'Battery protected — no discharge while this load runs'
          : 'High load detected') + '</div></div>';
      hlBanner.style.display = 'flex';
    } else {
      hlBanner.style.display = 'none';
    }
  }

  // Tags
  const tags = [];
  if (d.simulation_mode) tags.push('<span class="tag tag-warn">SIM</span>');
  if (d.high_loads_active) {
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
  // Real all-in import price = spot + distribution tariff (what you actually pay).
  const dist = (p.distribution_czk != null) ? p.distribution_czk : 0;
  const buy = (p.buy_czk != null) ? p.buy_czk : (p.czk_kwh || 0) + dist;
  document.getElementById('priceDist').textContent = dist.toFixed(2);
  document.getElementById('priceAllIn').textContent = buy.toFixed(2);

  // Solar
  if (d.solar_forecast) {
    document.getElementById('solarToday').textContent = d.solar_forecast.today_kwh + ' kWh';
    document.getElementById('solarTomorrow').textContent = d.solar_forecast.tomorrow_kwh + ' kWh';
    const sf = d.solar_forecast;

    // Solcast (weather-based source) status + API budget.
    const sc = sf.solcast;
    const scEl = document.getElementById('solcastInfo');
    if (scEl) {
      if (sc && sc.enabled) {
        const t = (sc.today_kwh != null ? sc.today_kwh : '--');
        const tm = (sc.tomorrow_kwh != null ? sc.tomorrow_kwh : '--');
        scEl.innerHTML =
          '☀️ Solcast (' + sc.sites + ' site' + (sc.sites === 1 ? '' : 's') +
          ', ' + sc.quantile + '): today ' + t + ' kWh, tomorrow ' + tm + ' kWh' +
          ' <span style="color:var(--muted)">· ' + sc.requests_today + '/' +
          sc.daily_budget + ' API calls today</span>';
      } else {
        scEl.innerHTML = '<span style="color:var(--muted)">Solcast: not configured</span>';
      }
    }
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
    // Engine badge (reflects the live engine, incl. MILP→greedy fallback).
    const engEl = document.getElementById('engineBadge');
    if (engEl) {
      const eng = (opt.engine || 'greedy');
      engEl.textContent = eng.toUpperCase();
      engEl.style.background = eng === 'milp' ? 'rgba(74,222,128,0.18)' : 'rgba(148,163,184,0.18)';
      engEl.style.color = eng === 'milp' ? 'var(--green)' : 'var(--muted)';
      engEl.title = eng === 'milp'
        ? 'MILP global optimiser (PuLP/CBC)'
        : 'Greedy forward-simulation';
    }
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

// ===== Compact Home "Today" overview chart =====
function homeBarClass(p) {
  const v = p.czk_kwh;
  let cls = v < 0 ? 'negative' : v < 1.5 ? 'cheap' : v < 3 ? 'mid' : 'expensive';
  if (p.status === 'pre_discharge_charge') cls = 'pre-discharge';
  else if (p.is_charging) cls = 'charging';
  else if (p.is_discharge) cls = 'discharge';
  else if (p.is_sell_production) cls = 'sell-production';
  else if (p.is_hold) cls = 'battery-hold';
  else if (p.is_inverter_off) cls = 'inverter-off';
  return cls;
}
// Vertical scale for a price chart on a ZERO line: how far the all-in price
// reaches ABOVE 0 (cost, incl. the fee floor) and BELOW 0 (net paid). The 0
// line is then placed maxNeg/range up from the bottom so negatives drop below.
function priceScale(prices) {
  let maxPos = 0.001, maxNeg = 0;
  prices.forEach(p => {
    const spot = p.czk_kwh, fee = p.distribution_czk || 0;
    const allin = (p.buy_czk != null) ? p.buy_czk : (spot + fee);
    maxPos = Math.max(maxPos, fee, allin > 0 ? allin : 0);  // fee floor always shown
    if (allin < 0) maxNeg = Math.max(maxNeg, -allin);
  });
  const range = maxPos + maxNeg;
  return { maxPos, maxNeg, range, zero: maxNeg / range * 100 };
}

// Inner segments of one full-height bar on the zero line at `sc.zero`% up:
//   - fee  : distribution fee, ALWAYS from the 0 line up to +fee (the floor you
//            pay, same level as positive bars)
//   - spot : positive net stacks above the fee -> top = all-in (cost)
//   - neg  : negative all-in drops BELOW the 0 line -> bottom = all-in (the real
//            value you're paid). fee stays above 0, net drops below: the total
//            span (fee top -> net bottom) == the gross negative spot.
function barInner(p, sc) {
  const spot = p.czk_kwh, fee = p.distribution_czk || 0;
  const allin = (p.buy_czk != null) ? p.buy_czk : (spot + fee);
  const range = sc.range, zero = sc.zero;
  let s = '';
  if (allin >= 0) {
    const feeH = Math.min(fee, allin) / range * 100;       // fee floor (capped at all-in)
    if (feeH > 0.2) s += '<div class="seg-fee" style="bottom:' + zero.toFixed(1) + '%;height:' + feeH.toFixed(1) + '%"></div>';
    const spotH = Math.max(0, allin - fee) / range * 100;  // net above the fee
    if (spotH > 0.2) s += '<div class="seg-spot" style="bottom:' + (zero + feeH).toFixed(1) + '%;height:' + spotH.toFixed(1) + '%"></div>';
  } else {
    const feeH = fee / range * 100;                        // full fee floor above 0
    if (feeH > 0.2) s += '<div class="seg-fee" style="bottom:' + zero.toFixed(1) + '%;height:' + feeH.toFixed(1) + '%"></div>';
    const negH = Math.min(zero, (-allin) / range * 100);   // net paid, below 0
    s += '<div class="seg-neg" style="bottom:' + (zero - negH).toFixed(1) + '%;height:' + Math.max(negH, 0.6).toFixed(1) + '%"></div>';
  }
  return s;
}

function zeroLineHtml(sc) {
  return sc.maxNeg > 0 ? '<div class="zero-line" style="bottom:' + sc.zero.toFixed(1) + '%"></div>' : '';
}
function renderHomeChart(prices, timeline) {
  const bars = document.getElementById('homeChartBars');
  const wrap = document.getElementById('homeChartWrap');
  const nodata = document.getElementById('homeChartNoData');
  if (!bars) return;
  if (!prices.length) { if (nodata) nodata.style.display = 'block'; if (wrap) wrap.style.display = 'none'; return; }
  if (nodata) nodata.style.display = 'none'; if (wrap) wrap.style.display = '';
  // Full-height bars on a 0 line: fee floor, positive net above, negative below.
  // data-idx maps to priceData (today is first there) so the home bars reuse the
  // same hover tooltip as the Prices tab; hover hits the whole column.
  const sc = priceScale(prices);
  let html = zeroLineHtml(sc);
  prices.forEach((p, i) => {
    let cls = homeBarClass(p); if (p.is_current) cls += ' current';
    html += '<div class="price-bar ' + cls + '" data-idx="' + i + '">' + barInner(p, sc) + '</div>';
  });
  bars.innerHTML = html;
  wireBarTooltips(bars);

  // Overlay lines, reusing the big chart's renderers so the Home chart shows the
  // SAME solar + draw (consumption) + SOC, each as actual (solid, elapsed) +
  // forecast (dashed, upcoming). Pin contentW to the fit width so the index-based
  // x-positions land on the flex-filled bars.
  bars.dataset.contentW = bars.offsetWidth || (wrap && wrap.offsetWidth) || 0;
  const eMax = energyScaleMax(prices, timeline, true);
  renderEnergyLine(prices, timeline, 'homeChartBars', 'homeConsLine', true, 'consumption_kwh', consActuals, '#38bdf8', eMax);
  renderSolarLine(prices, timeline, 'homeChartBars', 'homeSolarLine', true, eMax);
  renderSocLine(prices, timeline, 'homeChartBars', 'homeChartSoc', true);

  const n = prices.length, curIdx = prices.findIndex(p => p.is_current);
  const nowLine = document.getElementById('homeNowLine');
  if (nowLine) {
    if (curIdx >= 0) { nowLine.style.display = ''; nowLine.style.left = ((curIdx + 0.5) / n * 100) + '%'; }
    else nowLine.style.display = 'none';
  }

  const cur = prices[curIdx] || prices[prices.length - 1];
  const homeNow = document.getElementById('homeNow');
  if (cur && homeNow) {
    const act = (cur.projected_action || cur.status || 'hold').replace(/_/g, ' ');
    const soc = cur.projected_soc != null ? Math.round(cur.projected_soc) + '%' : '--';
    homeNow.innerHTML =
      '<b style="color:var(--text)">' + cur.czk_kwh.toFixed(2) + ' CZK</b> &middot; ' +
      '<span style="color:var(--green)">&#128267; ' + soc + '</span> &middot; ' + act;
  }

  const present = new Set(prices.map(homeBarClass));
  const items = [
    ['charging', '#22c55e', 'Charge'], ['discharge', '#ef4444', 'Discharge'],
    ['sell-production', '#f97316', 'Sell'], ['battery-hold', '#0ea5e9', 'Hold'],
    ['inverter-off', '#666', 'Off'],
  ];
  let lg = items.filter(it => present.has(it[0]))
    .map(it => '<span class="lg"><i style="background:' + it[1] + '"></i>' + it[2] + '</span>').join('');
  lg += '<span class="lg line"><i style="border-color:#22c55e"></i>SOC</span>'
      + '<span class="lg line"><i style="border-color:#fde047"></i>Solar</span>'
      + '<span class="lg line"><i style="border-color:#38bdf8"></i>Draw</span>'
      + '<span class="lg" style="opacity:.7">— now · ·· later</span>';
  const legEl = document.getElementById('homeLegend');
  if (legEl) legEl.innerHTML = lg;
}

// ===== Today's money summary =====
function renderHomeMoney(ec) {
  ec = ec || {};
  const earned = ec.export_revenue_czk, spent = ec.import_cost_czk;
  const net = (earned != null && spent != null) ? (earned - spent) : null;
  const netEl = document.getElementById('moneyNet');
  if (netEl) {
    if (net == null) { netEl.textContent = '--'; netEl.style.color = 'var(--muted)'; }
    else { netEl.textContent = (net >= 0 ? '+' : '') + net.toFixed(1) + ' CZK'; netEl.style.color = net >= 0 ? 'var(--green)' : 'var(--red)'; }
  }
  const set = (id, txt) => { const e = document.getElementById(id); if (e) e.textContent = txt; };
  set('moneyEarned', earned == null ? '--' : '+' + earned.toFixed(1));
  set('moneySpent', spent == null ? '--' : '-' + spent.toFixed(1));
  const arb = ec.arbitrage_czk;
  set('moneyArb', arb == null ? '--' : (arb >= 0 ? '+' : '') + arb.toFixed(1));
  const plan = ec.plan_value_czk, planEl = document.getElementById('moneyPlan');
  if (planEl) {
    planEl.textContent = plan == null ? '--' : (plan >= 0 ? '+' : '') + plan.toFixed(1);
    planEl.style.color = (plan || 0) >= 0 ? 'var(--green)' : 'var(--red)';
  }
}

async function fetchPrices() {
  try {
    const [priceRes, projRes, actualsRes, socRes, consRes] = await Promise.all([
      fetch('/api/prices'),
      fetch('/api/projection'),
      fetch('/api/solar_actuals'),
      fetch('/api/soc_actuals'),
      fetch('/api/consumption_actuals'),
    ]);
    const data = await priceRes.json();
    const projData = await projRes.json();
    try {
      const actualsData = await actualsRes.json();
      solarActuals = (actualsData && actualsData.hourly) || {};
    } catch (e) { solarActuals = {}; }
    try {
      const socData = await socRes.json();
      socActuals = (socData && socData.blocks) || {};
    } catch (e) { socActuals = {}; }
    try {
      const consData = await consRes.json();
      consActuals = (consData && consData.hourly) || {};
    } catch (e) { consActuals = {}; }
    const todayPrices = data.prices || [];
    const tomorrowPrices = data.tomorrow || [];
    const allPrices = [...todayPrices, ...tomorrowPrices];
    priceData = allPrices;  // unified array for tooltip lookups across both charts

    document.getElementById('priceChartTitle').textContent =
      data.has_tomorrow ? 'Today + Tomorrow Prices & Battery SOC' : "Today's Prices & Battery SOC";

    // Shared zero-line scale so today + tomorrow charts are comparable
    // (same 0-line position and units across both days).
    const maxAbs = priceScale(allPrices);

    const timeline = projData.timeline || [];
    renderHomeChart(todayPrices, timeline);  // compact Home overview
    const eMaxToday = energyScaleMax(todayPrices, timeline, true);
    renderPriceChart(todayPrices, 'priceChartToday', 0, maxAbs);
    renderEnergyLine(todayPrices, timeline, 'priceChartToday', 'consLineToday', true, 'consumption_kwh', consActuals, '#38bdf8', eMaxToday);
    renderSolarLine(todayPrices, timeline, 'priceChartToday', 'solarLineToday', true, eMaxToday);
    renderSocLine(todayPrices, timeline, 'priceChartToday', 'socLineToday', true);

    const tomorrowWrap = document.getElementById('priceChartTomorrowWrap');
    if (data.has_tomorrow && tomorrowPrices.length) {
      tomorrowWrap.style.display = '';
      const eMaxTomorrow = energyScaleMax(tomorrowPrices, timeline, false);
      renderPriceChart(tomorrowPrices, 'priceChartTomorrow', todayPrices.length, maxAbs);
      renderEnergyLine(tomorrowPrices, timeline, 'priceChartTomorrow', 'consLineTomorrow', false, 'consumption_kwh', consActuals, '#38bdf8', eMaxTomorrow);
      renderSolarLine(tomorrowPrices, timeline, 'priceChartTomorrow', 'solarLineTomorrow', false, eMaxTomorrow);
      renderSocLine(tomorrowPrices, timeline, 'priceChartTomorrow', 'socLineTomorrow', false);
    } else {
      tomorrowWrap.style.display = 'none';
    }
  } catch (e) { console.error('fetchPrices error:', e); }
}

function renderSocLine(prices, timeline, chartId, svgId, isToday) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  if (!timeline.length || !prices.length) { svg.innerHTML = ''; return; }

  const chart = document.getElementById(chartId);
  if (!chart) { svg.innerHTML = ''; return; }
  // Use the chart's scroll-content width (set by renderPriceChart) so the line
  // spans the full scrollable width and stays aligned with the bars.
  const w = parseFloat(chart.dataset.contentW) || chart.offsetWidth;
  const h = chart.offsetHeight;
  if (!w || !h) return;
  svg.setAttribute('width', w);

  // Build projected-SOC lookup: "HH:MM|day" -> soc%
  const socMap = {};
  timeline.forEach(t => { socMap[t.time + '|' + t.day] = t.soc; });

  // Index of the current (in-progress) block; everything before it has already
  // elapsed and gets the REAL measured SOC, everything from it onward stays the
  // optimizer's projection. -1 when no bar is current (e.g. tomorrow chart).
  const curIdx = isToday ? prices.findIndex(p => p.is_current) : -1;
  const barWidth = w / prices.length;
  const toXY = (i, soc) => {
    const x = i * barWidth + barWidth / 2;
    const y = h - (soc / 100 * h);  // SOC 100% = top, 0% = bottom
    return x.toFixed(1) + ',' + y.toFixed(1);
  };

  // Pass 1 — real (elapsed) points: actual measured SOC for past/current blocks.
  const realPts = [];
  let nowLabel = null;  // { x, y, soc } at the live edge
  prices.forEach((p, i) => {
    const isPastOrNow = isToday && curIdx !== -1 && i <= curIdx;
    const actual = isToday ? socActuals[p.start] : undefined;
    if (isPastOrNow && actual != null) {
      const pt = toXY(i, actual);
      realPts.push(pt);
      if (i === curIdx) {
        const c = pt.split(',');
        nowLabel = { x: parseFloat(c[0]), y: parseFloat(c[1]), soc: actual };
      }
    }
  });

  // Pass 2 — projected (future) points. When we have a real segment they start
  // at the current block (so the solid line owns the past); otherwise — no
  // telemetry yet today, or the tomorrow chart — the projection spans the whole
  // day so the chart never goes blank (preserves the original behaviour).
  const hasReal = realPts.length >= 2;
  const projPts = [];
  prices.forEach((p, i) => {
    const proj = socMap[p.start + '|' + p.day];
    if (proj == null) return;
    const inFuture = !hasReal || curIdx === -1 || i >= curIdx;
    if (inFuture) projPts.push(toXY(i, proj));
  });

  // Keep the line continuous: the dashed (projected) segment should start where
  // the solid (real) segment ends.
  if (realPts.length >= 1 && projPts.length >= 1) {
    // Ensure the dashed segment starts where the solid one ends.
    if (projPts[0] !== realPts[realPts.length - 1]) {
      projPts.unshift(realPts[realPts.length - 1]);
    }
  }

  if (realPts.length < 2 && projPts.length < 2) { svg.innerHTML = ''; return; }

  let svgContent = '';
  // Projected (future) — dashed, translucent.
  if (projPts.length >= 2) {
    svgContent += '<polyline points="' + projPts.join(' ') + '" ' +
      'fill="none" stroke="#f97316" stroke-width="2" stroke-opacity="0.5" ' +
      'stroke-dasharray="4 3" stroke-linejoin="round" stroke-linecap="round"/>';
  }
  // Real (elapsed) — solid, brighter, drawn on top.
  if (realPts.length >= 2) {
    svgContent += '<polyline points="' + realPts.join(' ') + '" ' +
      'fill="none" stroke="#f97316" stroke-width="2.5" stroke-opacity="0.95" ' +
      'stroke-linejoin="round" stroke-linecap="round"/>';
  }

  // SOC% labels: start + end of whatever line is shown, plus a highlighted
  // "now" marker at the real/projected boundary.
  const labelAt = (pt, soc, weight) => {
    const c = pt.split(',');
    const x = parseFloat(c[0]);
    const y = Math.max(12, parseFloat(c[1]) - 6);
    return '<text x="' + x + '" y="' + y + '" font-size="10" fill="#f97316" ' +
      'text-anchor="middle" font-weight="' + weight + '">' + Math.round(soc) + '%</text>';
  };
  const allPts = realPts.concat(projPts);
  if (allPts.length) {
    const firstSoc = isToday && socActuals[prices[0].start] != null
      ? socActuals[prices[0].start] : (socMap[prices[0].start + '|' + prices[0].day]);
    const lastProj = projPts.length ? projPts[projPts.length - 1] : realPts[realPts.length - 1];
    const lastSoc = socMap[prices[prices.length - 1].start + '|' + prices[prices.length - 1].day];
    if (firstSoc != null) svgContent += labelAt(allPts[0], firstSoc, '600');
    if (lastSoc != null) svgContent += labelAt(lastProj, lastSoc, '600');
  }
  if (nowLabel) {
    svgContent += '<circle cx="' + nowLabel.x + '" cy="' + nowLabel.y + '" r="3" ' +
      'fill="#f97316"/>';
    svgContent += '<text x="' + nowLabel.x + '" y="' + Math.max(12, nowLabel.y - 9) + '" ' +
      'font-size="10" fill="#f97316" text-anchor="middle" font-weight="700">' +
      Math.round(nowLabel.soc) + '%</text>';
  }

  svg.innerHTML = svgContent;
}

// Solar production line: actual (solid) for elapsed hours, projected (dashed)
// for the rest. Both aggregated to HOURLY kWh so the units match — projected
// solar in the timeline is per-15-min block, actual solar is per hour.
// Generic energy line (kWh/hour): actual (solid) for elapsed hours, projected
// (dashed) for the rest — same split logic as the SOC line. `field` is the
// projection timeline key (e.g. 'solar_kwh' / 'consumption_kwh'); `actualsMap`
// is the "HH:00" -> kWh actuals; `color` the line colour. Both sources are
// aggregated to hourly so units match (projection is per-15-min block).
// Shared kWh scale for a day's solar + consumption lines, so the two are drawn
// against ONE ruler and their relative heights are meaningful (solar above
// consumption == real surplus). Without this each line auto-scaled to its own
// max and equal kWh values landed at wildly different heights.
function energyScaleMax(prices, timeline, isToday) {
  let mx = 0.5;
  if (!prices.length) return mx;
  const dayOf = prices[0].day;
  ['solar_kwh', 'consumption_kwh'].forEach(field => {
    const projHour = {};
    timeline.forEach(t => {
      if (t.day !== dayOf) return;
      const hr = parseInt(t.time.split(':')[0], 10);
      projHour[hr] = (projHour[hr] || 0) + (t[field] || 0);
    });
    Object.values(projHour).forEach(v => { if (v > mx) mx = v; });
  });
  if (isToday) {
    [consActuals, solarActuals].forEach(m => {
      if (m) Object.values(m).forEach(v => { if (v > mx) mx = v; });
    });
  }
  return mx;
}

function renderEnergyLine(prices, timeline, chartId, svgId, isToday, field, actualsMap, color, sharedMax) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  if (!timeline.length || !prices.length) { svg.innerHTML = ''; return; }

  const chart = document.getElementById(chartId);
  if (!chart) { svg.innerHTML = ''; return; }
  // Use the chart's scroll-content width (set by renderPriceChart) so the line
  // spans the full scrollable width and stays aligned with the bars.
  const w = parseFloat(chart.dataset.contentW) || chart.offsetWidth;
  const h = chart.offsetHeight;
  if (!w || !h) return;
  svg.setAttribute('width', w);

  const dayOf = prices[0].day;  // every bar in a given chart shares one day

  // Projected hourly value (kWh): sum the four 15-min blocks in each hour.
  const projHour = {};
  timeline.forEach(t => {
    if (t.day !== dayOf) return;
    const hr = parseInt(t.time.split(':')[0], 10);
    projHour[hr] = (projHour[hr] || 0) + (t[field] || 0);
  });
  // Actual hourly value (today only; keyed "HH:00").
  const actHour = {};
  if (isToday && actualsMap) {
    Object.keys(actualsMap).forEach(k => {
      actHour[parseInt(k.split(':')[0], 10)] = actualsMap[k];
    });
  }

  // Map hour -> centre x of that hour's price bars.
  const barWidth = w / prices.length;
  const hourBars = {};
  prices.forEach((p, i) => {
    const hr = parseInt(p.start.split(':')[0], 10);
    (hourBars[hr] = hourBars[hr] || []).push(i);
  });
  const centerX = hr => {
    const idxs = hourBars[hr];
    if (!idxs) return null;
    const mid = idxs.reduce((a, b) => a + b, 0) / idxs.length;
    return mid * barWidth + barWidth / 2;
  };

  // Current hour boundary (today only).
  let curHr = -1;
  if (isToday) {
    const cur = prices.find(p => p.is_current);
    if (cur) curHr = parseInt(cur.start.split(':')[0], 10);
  }

  // Sit energy lines in the lower band so they read under the SOC line.
  // Prefer the shared scale (so solar + consumption share one ruler); fall back
  // to self-scaling only when no shared max was supplied.
  let maxKwh = 0.5;
  if (sharedMax && sharedMax > maxKwh) {
    maxKwh = sharedMax;
  } else {
    Object.values(projHour).forEach(v => { if (v > maxKwh) maxKwh = v; });
    Object.values(actHour).forEach(v => { if (v > maxKwh) maxKwh = v; });
  }
  const yOf = v => h - (v / maxKwh) * h * 0.55;

  const hours = Array.from(new Set(
    Object.keys(projHour).concat(Object.keys(actHour)).map(Number)
  )).sort((a, b) => a - b);

  // Pass 1 — actual (elapsed) hours.
  const realPts = [];
  hours.forEach(hr => {
    const x = centerX(hr);
    if (x == null) return;
    const isPastOrNow = isToday && curHr !== -1 && hr <= curHr;
    if (isPastOrNow && actHour[hr] != null) {
      realPts.push(x.toFixed(1) + ',' + yOf(actHour[hr]).toFixed(1));
    }
  });
  // Pass 2 — projected hours (future, or full span when no actuals/tomorrow).
  const hasReal = realPts.length >= 2;
  const projPts = [];
  hours.forEach(hr => {
    const x = centerX(hr);
    if (x == null || projHour[hr] == null) return;
    const inFuture = !hasReal || curHr === -1 || hr >= curHr;
    if (inFuture) projPts.push(x.toFixed(1) + ',' + yOf(projHour[hr]).toFixed(1));
  });
  if (realPts.length >= 1 && projPts.length >= 1 &&
      projPts[0] !== realPts[realPts.length - 1]) {
    projPts.unshift(realPts[realPts.length - 1]);
  }

  if (realPts.length < 2 && projPts.length < 2) { svg.innerHTML = ''; return; }

  let s = '';
  if (projPts.length >= 2) {
    s += '<polyline points="' + projPts.join(' ') + '" fill="none" ' +
      'stroke="' + color + '" stroke-width="2" stroke-opacity="0.45" ' +
      'stroke-dasharray="4 3" stroke-linejoin="round" stroke-linecap="round"/>';
  }
  if (realPts.length >= 2) {
    s += '<polyline points="' + realPts.join(' ') + '" fill="none" ' +
      'stroke="' + color + '" stroke-width="2.5" stroke-opacity="0.9" ' +
      'stroke-linejoin="round" stroke-linecap="round"/>';
  }
  svg.innerHTML = s;
}

function renderSolarLine(prices, timeline, chartId, svgId, isToday, sharedMax) {
  renderEnergyLine(prices, timeline, chartId, svgId, isToday,
    'solar_kwh', solarActuals, '#fde047', sharedMax);
}

let priceData = [];
let solarActuals = {};  // "HH:00" -> actual solar kWh (today only)
let socActuals = {};    // "HH:MM" -> actual battery SOC% (today, elapsed blocks)
let consActuals = {};   // "HH:00" -> actual household load kWh (today only)

function fmtCzk(v) {
  return (v == null) ? '--' : (v >= 0 ? '+' : '') + v.toFixed(1) + ' CZK';
}

async function fetchInsights() {
  let d;
  try { d = await (await fetch('/api/insights')).json(); }
  catch (e) { return; }
  if (!d || d.error) return;

  // --- Economics ---
  const ec = d.economics || {};
  renderHomeMoney(ec);  // compact Home money summary
  const planEl = document.getElementById('econPlan');
  if (planEl) {
    planEl.textContent = fmtCzk(ec.plan_value_czk);
    planEl.style.color = (ec.plan_value_czk || 0) >= 0 ? 'var(--green)' : 'var(--red)';
  }
  setText('econExport', fmtCzk(ec.export_revenue_czk));
  setText('econImport', ec.import_cost_czk == null ? '--' : '-' + ec.import_cost_czk.toFixed(1) + ' CZK');
  setText('econArbitrage', fmtCzk(ec.arbitrage_czk));

  // --- Forecast accuracy ---
  const ac = d.accuracy || {};
  const a = ac.solar_actual_kwh_so_far, f = ac.solar_forecast_kwh_so_far;
  setText('accSolar', (a == null ? '--' : a + ' kWh') + ' / ' + (f == null ? '--' : f + ' kWh'));
  const errEl = document.getElementById('accSolarErr');
  if (errEl) {
    if (ac.solar_error_pct == null) { errEl.textContent = '--'; }
    else {
      errEl.textContent = (ac.solar_error_pct > 0 ? '+' : '') + ac.solar_error_pct + '%';
      errEl.style.color = Math.abs(ac.solar_error_pct) <= 15 ? 'var(--green)'
        : Math.abs(ac.solar_error_pct) <= 35 ? '#f59e0b' : 'var(--red)';
    }
  }
  setText('accCalib', ac.calibration_ratio == null ? '--' : '×' + Number(ac.calibration_ratio).toFixed(2));
  setText('accConf', ac.solar_confidence == null ? '--' : Math.round(ac.solar_confidence * 100) + '%');

  // --- Engines & sources ---
  const en = d.engines || {};
  const optTxt = (en.optimizer || '--').toUpperCase() +
    (en.optimizer_fellback ? ' (fell back from ' + en.optimizer_configured + ')' : '');
  setText('engOpt', optTxt);
  const consTxt = (en.consumption || '--').toUpperCase() +
    (en.consumption_fellback ? ' (fell back from ml)' : '');
  setText('engCons', consTxt);
  const sc = d.solcast || {};
  setText('engSolcast', sc.enabled ? (sc.requests_today + '/' + sc.daily_budget + ' reqs') : 'off');
  // last command (worst result wins for visibility)
  const cmds = (d.commands && d.commands.last_results) || {};
  const keys = Object.keys(cmds);
  const cmdEl = document.getElementById('engCmd');
  if (cmdEl) {
    if (!keys.length) { cmdEl.textContent = '--'; }
    else {
      const failed = keys.filter(k => !cmds[k].success);
      if (failed.length) { cmdEl.textContent = '⚠️ ' + failed[0] + ' failed'; cmdEl.style.color = 'var(--red)'; }
      else { cmdEl.textContent = '✅ ' + (d.commands.sent || 0) + ' ok'; cmdEl.style.color = 'var(--green)'; }
    }
  }
  const fr = d.freshness || {};
  setText('engFresh', fr.prices_age_s == null ? '--'
    : (fr.prices_age_s < 90 ? 'just now' : Math.round(fr.prices_age_s / 60) + ' min ago')
      + (fr.has_tomorrow_prices ? ' · +tomorrow' : ''));

  // --- Deferrable loads ---
  const card = document.getElementById('deferrableCard');
  const list = document.getElementById('deferrableList');
  const defs = d.deferrable || [];
  if (card && list) {
    if (!defs.length) { card.style.display = 'none'; }
    else {
      card.style.display = '';
      list.innerHTML = defs.map(l =>
        '<div class="stat" style="display:flex;justify-content:space-between">' +
        '<span>' + l.name + ' <span class="stat-label">(' + (l.energy_kwh ?? '?') + ' kWh)</span></span>' +
        '<span>' + (l.blocks ? l.blocks + ' blk' + (l.savings_czk != null ? ' · ' + fmtCzk(l.savings_czk) : '') : 'not scheduled') + '</span></div>'
      ).join('');
    }
  }

  // --- Alerts strip ---
  const alerts = [];
  if (en.optimizer_fellback) alerts.push('Optimizer fell back to ' + en.optimizer + ' (PuLP/solver issue)');
  if (en.consumption_fellback) alerts.push('ML consumption unavailable — using binned model');
  if (Object.keys(cmds).some(k => !cmds[k].success)) alerts.push('A control command failed — check logs');
  if (fr.prices_age_s != null && fr.prices_age_s > 7200) alerts.push('Prices are stale (' + Math.round(fr.prices_age_s / 3600) + 'h old)');
  if (ac.solar_error_pct != null && Math.abs(ac.solar_error_pct) > 40) alerts.push('Solar forecast off by ' + ac.solar_error_pct + '% today');
  const strip = document.getElementById('alertsStrip');
  if (strip) {
    strip.innerHTML = alerts.length
      ? alerts.map(a => '<div class="alert-banner">⚠️ ' + a + '</div>').join('')
      : '';
  }
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

async function reapplyState() {
  const btn = document.getElementById('reapplyBtn');
  const res = document.getElementById('reapplyResult');
  if (!btn) return;
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = '⏳ Re-applying…';
  if (res) { res.textContent = ''; res.style.color = 'var(--muted)'; }
  try {
    const r = await fetch('/api/reapply', { method: 'POST' });
    const d = await r.json();
    if (res) {
      if (d.success) {
        res.style.color = 'var(--green)';
        res.textContent = '✅ ' + (d.message || 'Re-applied') +
          (d.mode ? ' (mode: ' + d.mode + ')' : '');
      } else {
        res.style.color = 'var(--red)';
        res.textContent = '⚠️ ' + (d.message || 'Re-apply failed');
      }
    }
    fetchInsights();  // refresh command-health display
    fetchStatus();
  } catch (e) {
    if (res) { res.style.color = 'var(--red)'; res.textContent = '⚠️ ' + e; }
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

function renderPriceChart(prices, mountId, idxOffset, maxAbs) {
  const chart = document.getElementById(mountId);
  if (!chart) return;
  if (!prices.length) { chart.innerHTML = '<div style="color:var(--muted)">No price data</div>'; return; }

  // Single scale across both day-charts (zero line, neg below it).
  if (!maxAbs) maxAbs = priceScale(prices);

  let html = zeroLineHtml(maxAbs);
  prices.forEach((p, i) => {
    const v = p.czk_kwh;
    let cls = v < 0 ? 'negative' : v < 1.5 ? 'cheap' : v < 3 ? 'mid' : 'expensive';
    if (p.status === 'pre_discharge_charge') cls = 'pre-discharge';
    else if (p.is_charging) cls = 'charging';
    else if (p.is_discharge) cls = 'discharge';
    else if (p.is_sell_production) cls = 'sell-production';
    else if (p.is_hold) cls = 'battery-hold';
    else if (p.is_inverter_off) cls = 'inverter-off';
    if (p.is_current) cls += ' current';

    const globalIdx = i + (idxOffset || 0);
    const tmrwOpacity = p.day === 'tomorrow' ? 'opacity:0.6;' : '';
    html += '<div class="price-bar ' + cls + '" data-idx="' + globalIdx + '" style="' + tmrwOpacity + '">' + barInner(p, maxAbs) + '</div>';
  });
  chart.innerHTML = html;

  // Size the chart for horizontal scroll: each bar is at least MIN_BAR_W px.
  // On a wide screen the bars fit the viewport (no scroll); on a phone the
  // content overflows the .chart-scroll card and scrolls sideways. The overlay
  // renderers read the SAME width via dataset.contentW so the SVG SOC/solar/load
  // lines stay pinned to the bars at any scroll offset.
  const MIN_BAR_W = 14, GAP = 1;
  const scrollEl = chart.closest('.chart-scroll');
  const viewW = (scrollEl ? scrollEl.clientWidth : chart.offsetWidth) || 0;
  const n = prices.length;
  const barW = Math.max(MIN_BAR_W, (viewW - (n - 1) * GAP) / n);
  const contentW = Math.round(n * barW + (n - 1) * GAP);
  chart.style.setProperty('--bar-w', barW + 'px');
  chart.style.width = contentW + 'px';
  const content = chart.parentElement; // .chart-content
  if (content && content.classList.contains('chart-content')) content.style.width = contentW + 'px';
  chart.dataset.contentW = contentW;

  // "NOW" divider between elapsed and upcoming blocks, so it's obvious at a
  // glance what already happened vs what's planned. Auto-centre it once.
  const curIdx = prices.findIndex(p => p.is_current);
  if (content && content.classList.contains('chart-content')) {
    const oldDiv = content.querySelector('.now-divider');
    if (oldDiv) oldDiv.remove();
    if (curIdx >= 0) {
      const div = document.createElement('div');
      div.className = 'now-divider';
      div.style.left = (curIdx * (barW + GAP)) + 'px';
      content.appendChild(div);
      if (scrollEl && contentW > viewW && scrollEl.clientWidth > 0
          && chart.dataset.autoscrolled !== '1') {
        scrollEl.scrollLeft = Math.max(0, curIdx * (barW + GAP) - viewW * 0.4);
        chart.dataset.autoscrolled = '1';
      }
    }
  }

  wireBarTooltips(chart);
}

// Attach tooltip events to every .price-bar in a container. Each bar is a
// FULL-HEIGHT column (transparent, segments have pointer-events:none), so the
// hover/tap target is the whole column, not just the coloured part. On touch we
// DON'T preventDefault (so dragging still scrolls): a near-stationary touch is a
// tap → show the tooltip; a drag scrolls and hides any open tooltip.
function wireBarTooltips(container) {
  const tooltip = document.getElementById('chartTooltip');
  if (!container || !tooltip) return;
  container.querySelectorAll('.price-bar').forEach(bar => {
    bar.addEventListener('mouseenter', (e) => showTooltip(e, bar, tooltip));
    bar.addEventListener('mousemove', (e) => moveTooltip(e, tooltip));
    bar.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    let tx = 0, ty = 0, moved = false;
    bar.addEventListener('touchstart', (e) => {
      const t = e.touches[0]; tx = t.clientX; ty = t.clientY; moved = false;
    }, { passive: true });
    bar.addEventListener('touchmove', (e) => {
      const t = e.touches[0];
      if (Math.abs(t.clientX - tx) > 8 || Math.abs(t.clientY - ty) > 8) {
        moved = true;
        tooltip.style.display = 'none';
      }
    }, { passive: true });
    bar.addEventListener('touchend', () => {
      if (!moved) showTooltip({ clientX: tx, clientY: ty }, bar, tooltip);
    });
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
  else if (p.status === 'battery_hold') statusHtml += '<span class="tt-status tt-battery-hold">BATTERY HOLD</span>';
  else if (p.status === 'inverter_off') statusHtml += '<span class="tt-status tt-inverter-off">INVERTER OFF</span>';

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
    // Planned inverter powerRate for charge/discharge blocks (e.g. "@ 75%").
    const pr = p.projected_power_rate;
    const rateStr = (pr != null && (projAction === 'charge' || projAction === 'discharge'))
      ? ' <span style="color:var(--accent)">@ ' + pr + '%</span>' : '';
    const nf = p.projected_net_flow || 0;
    const nfSign = nf >= 0 ? '+' : '';
    const nfColor = nf > 0 ? 'var(--green)' : nf < 0 ? 'var(--red)' : 'var(--muted)';
    const solar = p.projected_solar || 0;
    const cons = p.projected_consumption || 0;
    // Forecast-vs-actual solar overlay (today only; actuals keyed by hour).
    let solarCell = solar.toFixed(2);
    if (p.day !== 'tomorrow' && p.start) {
      const hourKey = p.start.slice(0, 2) + ':00';
      if (Object.prototype.hasOwnProperty.call(solarActuals, hourKey)) {
        const act = solarActuals[hourKey];
        const driftColor = act >= solar ? 'var(--green)' : 'var(--red)';
        solarCell = solar.toFixed(2) + ' <span style="color:' + driftColor + '">(act ' + act.toFixed(2) + ')</span>';
      }
    }
    projHtml = '<div style="margin-top:4px;padding-top:4px;border-top:1px solid var(--border)">' +
      '<div class="tt-row"><span class="tt-label">Battery</span><span style="color:' + projColor + '">' + p.projected_soc + '% (' + p.projected_kwh + ' kWh) ' + projAction + rateStr + '</span></div>' +
      '<div class="tt-row"><span class="tt-label">Net flow</span><span style="color:' + nfColor + '">' + nfSign + nf.toFixed(2) + ' kWh</span></div>' +
      '<div class="tt-row"><span class="tt-label">Solar / Load</span><span>' + solarCell + ' / ' + cons.toFixed(2) + ' kWh</span></div>' +
      '</div>';
  }

  tooltip.innerHTML =
    '<div class="tt-time">' + p.start + ' - ' + p.end + dayLabel + '</div>' +
    '<div class="tt-price ' + priceClass + '">' + v.toFixed(2) + ' CZK/kWh <span style="font-size:10px;color:var(--muted)">spot</span></div>' +
    '<div class="tt-row"><span class="tt-label">Buy (incl. tariff)</span><span style="color:var(--text)"><b>' + (p.buy_czk != null ? p.buy_czk : v + (p.distribution_czk||0)).toFixed(2) + '</b> CZK/kWh</span></div>' +
    '<div class="tt-row"><span class="tt-label" style="font-size:10px">spot ' + v.toFixed(2) + ' + dist ' + (p.distribution_czk||0).toFixed(2) + ' (' + ((p.distribution_czk||0) > 0.5 ? 'VT high' : 'NT low') + ')</span></div>' +
    '<div class="tt-row"><span class="tt-label">Net sell</span><span style="color:' + sellColor + '">' + netSell.toFixed(2) + ' CZK/kWh</span></div>' +
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
// ===== Header live-strip + status helpers =====
let _ssPrev = {};
function fmtW(w) {
  w = Math.round(w || 0);
  const a = Math.abs(w);
  if (a >= 1000) return (w / 1000).toFixed(a >= 10000 ? 0 : 1) + ' kW';
  return w + ' W';
}
function setSS(id, text, color) {
  const el = document.getElementById(id);
  if (!el) return;
  if (_ssPrev[id] !== text) {
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    _ssPrev[id] = text;
  }
  el.textContent = text;
  if (color) el.style.color = color;
}
function renderHeaderStrip(pw) {
  // Solar
  setSS('ssSolar', pw.solar_w > 0 ? fmtW(pw.solar_w) : '0 W',
        pw.solar_w > 30 ? 'var(--yellow)' : 'var(--muted)');
  // Home load
  setSS('ssLoad', fmtW(pw.load_w), 'var(--text)');
  // Battery: SOC% as the headline, charge/discharge in the label
  const soc = pw.soc;
  setSS('ssBatt', soc + '%',
        soc > 60 ? 'var(--green)' : soc > 30 ? 'var(--yellow)' : 'var(--red)');
  let bl = 'idle';
  if (pw.battery_charge_w > 30) bl = '▲ ' + fmtW(pw.battery_charge_w);
  else if (pw.battery_discharge_w > 30) bl = '▼ ' + fmtW(pw.battery_discharge_w);
  document.getElementById('ssBattLbl').textContent = bl;
  // Grid: export (sell, green ▲) vs import (buy, red ▼)
  let gv = '0 W', gc = 'var(--muted)', gl = 'Grid';
  if (pw.grid_export_w > 50) {
    gv = '▲ ' + fmtW(pw.grid_export_w); gc = 'var(--green)'; gl = 'Selling';
  } else {
    const imp = Math.max(0, pw.load_w + pw.battery_charge_w - pw.solar_w - pw.battery_discharge_w);
    if (imp > 50) { gv = '▼ ' + fmtW(imp); gc = 'var(--red)'; gl = 'Buying'; }
  }
  setSS('ssGrid', gv, gc);
  document.getElementById('ssGridLbl').textContent = gl;
}
let _lastOkTs = 0;
function setDot(state) { // 'ok' | 'off' | 'stale'
  const dot = document.getElementById('statusDot');
  if (dot) dot.className = 'status-dot ' + state;
}
function markFresh() {
  _lastOkTs = Date.now();
  document.getElementById('livePill').classList.remove('stale');
  document.getElementById('statStrip').classList.remove('stale');
}
function freshnessWatchdog() {
  if (Date.now() - _lastOkTs > 20000) {
    document.getElementById('livePill').classList.add('stale');
    document.getElementById('statStrip').classList.add('stale');
    setDot('stale');
  }
}
setInterval(freshnessWatchdog, 5000);

// Live power flow
async function fetchLive() {
  try {
    const res = await fetch('/api/live');
    const d = await res.json();
    if (!d.has_data) {
      document.getElementById('statStrip').classList.add('stale');
      return;
    }
    // Live power flow now lives in the always-visible header strip.
    renderHeaderStrip(d.power);
    markFresh();

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

// ===== App tab navigation =====
function showTab(name) {
  document.querySelectorAll('.tab-page').forEach(function(p) {
    p.classList.toggle('active', p.id === 'tab-' + name);
  });
  document.querySelectorAll('.tab-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  try { localStorage.setItem('gw_tab', name); } catch (e) {}
  if (name === 'chart' || name === 'home') {
    // Charts + their SVG overlays size from offsetWidth, which is 0 while a tab
    // is hidden. Re-render once layout has settled so they aren't blank.
    requestAnimationFrame(function() { try { fetchPrices(); } catch (e) {} });
  }
  window.scrollTo(0, 0);
}
document.querySelectorAll('.tab-btn').forEach(function(b) {
  b.addEventListener('click', function() { showTab(b.dataset.tab); });
});
showTab((function() {
  try { return localStorage.getItem('gw_tab') || 'home'; } catch (e) { return 'home'; }
})());

// ===== PWA service worker =====
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/sw.js').catch(function() {});
  });
}

// ===== Refresh (button + pull-to-refresh) =====
let _refreshing = false;
async function refreshAll() {
  if (_refreshing) return;
  _refreshing = true;
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('spinning');
  try {
    await Promise.all([
      fetchStatus(), fetchLive(),
      fetchPrices().catch(function(){}), fetchInsights().catch(function(){}),
    ]);
  } finally {
    setTimeout(function(){ btn.classList.remove('spinning'); }, 500);
    _refreshing = false;
  }
}
document.getElementById('refreshBtn').addEventListener('click', refreshAll);

// Pull-to-refresh: only when scrolled to the very top. Indicator-only (doesn't
// fight native scroll); release past the threshold triggers a full refresh.
(function() {
  const ptr = document.getElementById('ptr');
  const label = ptr.querySelector('.ptr-label');
  const spin = ptr.querySelector('.ptr-spin');
  const THRESH = 70;
  let startY = 0, pulling = false, dist = 0;
  window.addEventListener('touchstart', function(e) {
    if (window.scrollY <= 0 && !_refreshing) { startY = e.touches[0].clientY; pulling = true; }
  }, { passive: true });
  window.addEventListener('touchmove', function(e) {
    if (!pulling) return;
    dist = e.touches[0].clientY - startY;
    if (dist <= 0 || window.scrollY > 0) { reset(); return; }
    const pull = Math.min(dist * 0.5, 90);
    ptr.style.height = pull + 'px';
    ptr.style.opacity = Math.min(pull / THRESH, 1);
    spin.style.transform = 'rotate(' + (pull * 3) + 'deg)';
    const ready = pull >= THRESH;
    ptr.classList.toggle('ready', ready);
    label.textContent = ready ? 'Release to refresh' : 'Pull to refresh';
  }, { passive: true });
  window.addEventListener('touchend', async function() {
    if (!pulling) return;
    pulling = false;
    if (ptr.classList.contains('ready')) {
      ptr.classList.add('refreshing');
      ptr.style.height = '48px'; ptr.style.opacity = '1';
      label.textContent = 'Refreshing…';
      await refreshAll();
    }
    reset();
  }, { passive: true });
  function reset() {
    pulling = false; dist = 0;
    ptr.classList.remove('ready', 'refreshing');
    ptr.style.height = ''; ptr.style.opacity = '';
    spin.style.transform = '';
    label.textContent = 'Pull to refresh';
  }
})();

fetchStatus();
fetchLive();
fetchPrices();
fetchInsights();
fetchLogs();
connectSSE();
setInterval(fetchStatus, 5000);
setInterval(fetchLive, 3000);
setInterval(fetchPrices, 60000);
setInterval(fetchInsights, 30000);
</script>
</body>
</html>
"""


# --- PWA assets (manifest, service worker, icons) ---

MANIFEST_JSON = json.dumps({
    "name": "Growatt Battery Dashboard",
    "short_name": "Growatt",
    "description": "Solar battery optimization dashboard",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#0f1117",
    "theme_color": "#0f1117",
    "icons": [
        {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml",
         "purpose": "any maskable"},
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png",
         "purpose": "any"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
})

# Minimal service worker: just enough to be installable. Pass-through fetch — we
# deliberately DO NOT cache anything (the dashboard is all live data: /api/* and
# the SSE log stream must never be served stale).
SW_JS = """\
self.addEventListener('install', function(e) { self.skipWaiting(); });
self.addEventListener('activate', function(e) { e.waitUntil(self.clients.claim()); });
self.addEventListener('fetch', function(e) { /* network pass-through, no cache */ });
"""

# Maskable app icon: battery glyph on the dashboard's dark background, with a
# safe-zone margin so the maskable crop never clips it.
ICON_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <rect width="512" height="512" rx="96" fill="#0f1117"/>
  <rect x="150" y="150" width="180" height="240" rx="22" fill="none"
        stroke="#4f8cff" stroke-width="20"/>
  <rect x="214" y="120" width="52" height="34" rx="10" fill="#4f8cff"/>
  <rect x="174" y="300" width="132" height="66" rx="8" fill="#22c55e"/>
  <rect x="174" y="232" width="132" height="60" rx="8" fill="#22c55e" opacity="0.55"/>
  <path d="M286 168 L214 286 L256 286 L238 372 L322 240 L274 240 Z"
        fill="#eab308" stroke="#0f1117" stroke-width="8" stroke-linejoin="round"/>
</svg>
"""

_STATIC_DIR = pathlib.Path(__file__).parent / "static"


def _read_icon(name: str) -> bytes:
    """Load a committed PNG icon; empty bytes if missing (route 404s gracefully)."""
    try:
        return (_STATIC_DIR / name).read_bytes()
    except OSError:
        return b""


_ICON_180 = _read_icon("icon-180.png")
_ICON_192 = _read_icon("icon-192.png")
_ICON_512 = _read_icon("icon-512.png")
