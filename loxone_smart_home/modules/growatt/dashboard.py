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

    Deduplicates messages that arrive within the same second (caused by
    multiple log handlers/formatters in the logging chain).
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_msg: str = ""
        self._last_time: str = ""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            now = datetime.now().strftime("%H:%M:%S")

            # Deduplicate: skip if same message in the same second
            if msg == self._last_msg and now == self._last_time:
                return
            self._last_msg = msg
            self._last_time = now

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


def _get_live_soc() -> Optional[float]:
    """Get live battery SOC from the Growatt API telemetry cache."""
    try:
        from modules.growatt.api import _telemetry_cache
        if _telemetry_cache:
            return _telemetry_cache.get("SOC")
    except Exception:
        pass
    return None


# --- API Endpoints ---

async def api_status(request: web.Request) -> web.Response:
    """Get comprehensive system status."""
    ctrl = _get_controller(request)
    if not ctrl:
        return web.json_response({"error": "Controller not ready"}, status=503)

    now = ctrl._get_local_now()
    rate = ctrl._eur_czk_rate or 25.0

    # Current price
    current_price_eur = 0.0
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
        current_price_eur = ctrl._current_prices.get(current_block, 0.0)

    current_price_czk = current_price_eur * rate / 1000

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
        solar = {
            "today_kwh": round(today_kwh, 1),
            "tomorrow_kwh": round(tomorrow_kwh, 1),
            "confidence": ctrl._solar_forecast.confidence,
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

    return web.json_response({
        "timestamp": now.isoformat(),
        "mode": ctrl._current_mode,
        "battery_soc": _get_live_soc() or ctrl._battery_soc,
        "current_price": {
            "eur_mwh": round(current_price_eur, 2),
            "czk_kwh": round(current_price_czk, 2),
            "block": current_block,
        },
        "high_loads_active": ctrl._high_loads_active,
        "season": season,
        "inverter": inverter_state,
        "solar_forecast": solar,
        "schedule": schedule,
        "manual_override": override,
        "last_evaluation": last_eval,
        "simulation_mode": getattr(ctrl.config, 'simulation_mode', False),
    })


async def api_prices(request: web.Request) -> web.Response:
    """Get today's 15-min price blocks."""
    ctrl = _get_controller(request)
    if not ctrl or not ctrl._current_prices:
        return web.json_response({"prices": []})

    rate = ctrl._eur_czk_rate or 25.0
    now = ctrl._get_local_now()
    block_min = (now.minute // 15) * 15
    block_start = now.replace(minute=block_min, second=0, microsecond=0)
    from datetime import timedelta
    block_end = block_start + timedelta(minutes=15)
    cur_start = block_start.strftime("%H:%M")
    cur_end = "24:00" if block_end.date() != block_start.date() else block_end.strftime("%H:%M")

    pre_discharge = getattr(ctrl, '_pre_discharge_blocks_today', set())
    discharge = getattr(ctrl, '_discharge_periods_today', set())

    prices = []
    for (start, end), price_eur in sorted(ctrl._current_prices.items()):
        is_current = (start == cur_start and end == cur_end)
        is_charging = (start, end) in ctrl._combined_charging_blocks
        is_pre_discharge = (start, end) in pre_discharge
        is_discharge = (start, end) in discharge

        status = "normal"
        if is_charging and is_pre_discharge:
            status = "pre_discharge_charge"
        elif is_charging:
            status = "charging"
        elif is_discharge:
            status = "discharge"

        prices.append({
            "start": start,
            "end": end,
            "eur_mwh": round(price_eur, 2),
            "czk_kwh": round(price_eur * rate / 1000, 2),
            "is_charging": is_charging,
            "is_pre_discharge": is_pre_discharge,
            "is_discharge": is_discharge,
            "is_current": is_current,
            "status": status,
        })
    return web.json_response({"prices": prices})


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
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


# --- Dashboard Setup ---

def create_dashboard_app(controller=None) -> web.Application:
    """Create the dashboard web application."""
    app = web.Application()
    if controller:
        app["controller"] = controller

    app.router.add_get("/", dashboard_page)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/prices", api_prices)
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
  height: 120px;
  display: flex;
  align-items: flex-end;
  gap: 1px;
  margin-top: 8px;
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
        <div class="stat-label">CZK/kWh (<span id="priceEur">--</span> EUR/MWh)</div>
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
        <div class="stat-label">Evaluation reason</div>
        <div class="stat-value small" style="font-size:13px;color:var(--muted)" id="lastEval">--</div>
      </div>
    </div>
  </div>

  <!-- Price Chart -->
  <div class="card" style="margin-bottom:12px">
    <h2>Today's Prices (15-min blocks)</h2>
    <div class="price-chart" id="priceChart"></div>
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
  if (d.high_loads_active) tags.push('<span class="tag tag-warn">HIGH LOAD</span>');
  if (d.inverter) {
    tags.push(d.inverter.export ? '<span class="tag tag-on">EXP ON</span>' : '<span class="tag tag-off">EXP OFF</span>');
    tags.push(d.inverter.ac_charge ? '<span class="tag tag-on">AC CHG</span>' : '<span class="tag tag-off">AC CHG OFF</span>');
  }
  tags.push('<span class="tag tag-info">' + (d.season || '?') + '</span>');
  document.getElementById('tags').innerHTML = tags.join('');

  // Price
  const p = d.current_price || {};
  document.getElementById('priceCzk').textContent = (p.czk_kwh || 0).toFixed(2) + ' CZK/kWh';
  document.getElementById('priceEur').textContent = (p.eur_mwh || 0).toFixed(1);
  document.getElementById('priceBlock').textContent = p.block ? p.block[0] + ' - ' + p.block[1] : '--';
  const priceEl = document.getElementById('priceCzk');
  priceEl.style.color = (p.czk_kwh || 0) < 0 ? 'var(--green)' : (p.czk_kwh || 0) > 3 ? 'var(--red)' : 'var(--text)';

  // Solar
  if (d.solar_forecast) {
    document.getElementById('solarToday').textContent = d.solar_forecast.today_kwh + ' kWh';
    document.getElementById('solarTomorrow').textContent = d.solar_forecast.tomorrow_kwh + ' kWh';
    document.getElementById('solarConf').textContent =
      'Confidence: ' + (d.solar_forecast.confidence * 100).toFixed(0) + '%';
  }

  // Schedule
  document.getElementById('chargeBlocks').textContent = (d.schedule?.charging_blocks || 0) + ' blocks';
  document.getElementById('lastEval').textContent = d.last_evaluation || '--';

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
    const res = await fetch('/api/prices');
    const data = await res.json();
    renderPriceChart(data.prices || []);
  } catch (e) {}
}

let priceData = [];

function renderPriceChart(prices) {
  priceData = prices;
  const chart = document.getElementById('priceChart');
  if (!prices.length) { chart.innerHTML = '<div style="color:var(--muted)">No price data</div>'; return; }

  const vals = prices.map(p => p.czk_kwh);
  const maxAbs = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)), 1);

  let html = '';
  prices.forEach((p, i) => {
    const v = p.czk_kwh;
    const h = Math.abs(v) / maxAbs * 100;
    const isNeg = v < 0;
    let cls = v < 0 ? 'negative' : v < 1.5 ? 'cheap' : v < 3 ? 'mid' : 'expensive';
    if (p.status === 'pre_discharge_charge') cls = 'pre-discharge';
    else if (p.is_charging) cls = 'charging';
    else if (p.is_discharge) cls = 'discharge';
    if (p.is_current) cls += ' current';

    if (isNeg) {
      html += '<div class="price-bar ' + cls + '" data-idx="' + i + '" style="height:' + h + '%;align-self:flex-start;opacity:0.7"></div>';
    } else {
      html += '<div class="price-bar ' + cls + '" data-idx="' + i + '" style="height:' + Math.max(h, 3) + '%"></div>';
    }
  });
  chart.innerHTML = html;

  // Attach tooltip events
  const tooltip = document.getElementById('chartTooltip');
  chart.querySelectorAll('.price-bar').forEach(bar => {
    bar.addEventListener('mouseenter', (e) => showTooltip(e, bar, tooltip));
    bar.addEventListener('mousemove', (e) => moveTooltip(e, tooltip));
    bar.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    bar.addEventListener('touchstart', (e) => { e.preventDefault(); showTooltip(e.touches[0], bar, tooltip); });
  });
  document.addEventListener('touchstart', (e) => {
    if (!e.target.classList.contains('price-bar')) tooltip.style.display = 'none';
  });
}

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

  // Calculate actual price rank (1 = cheapest)
  const sortedPrices = [...priceData].sort((a, b) => a.czk_kwh - b.czk_kwh);
  const priceRank = sortedPrices.findIndex(sp => sp.start === p.start && sp.end === p.end) + 1;
  const percentile = Math.round(priceRank / priceData.length * 100);
  const rankLabel = percentile <= 25 ? 'Cheapest quarter' : percentile <= 50 ? 'Below average' : percentile <= 75 ? 'Above average' : 'Most expensive quarter';

  tooltip.innerHTML =
    '<div class="tt-time">' + p.start + ' - ' + p.end + '</div>' +
    '<div class="tt-price ' + priceClass + '">' + v.toFixed(2) + ' CZK/kWh</div>' +
    '<div class="tt-row"><span class="tt-label">EUR/MWh</span><span>' + p.eur_mwh.toFixed(1) + '</span></div>' +
    '<div class="tt-row"><span class="tt-label">Price rank</span><span>#' + priceRank + ' of ' + priceData.length + ' (' + rankLabel + ')</span></div>' +
    (statusHtml ? '<div style="margin-top:6px">' + statusHtml + '</div>' : '');

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

// SSE for live logs
function connectSSE() {
  const es = new EventSource('/api/logs/stream');
  es.onmessage = function(e) {
    const entry = JSON.parse(e.data);
    const box = document.getElementById('logBox');
    box.innerHTML += formatLog(entry);
    // Keep last 200 entries in DOM
    while (box.children.length > 200) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  };
  es.onerror = function() {
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
fetchStatus();
fetchPrices();
fetchLogs();
connectSSE();
setInterval(fetchStatus, 5000);
setInterval(fetchPrices, 60000);
</script>
</body>
</html>
"""
