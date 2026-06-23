# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a consolidated Python service that combines multiple smart home automation services into a single async application. It integrates Loxone home automation with Growatt solar systems, weather data, and energy price optimization.

## Remote Server & Deployment

**IMPORTANT**: The service runs on a Synology NAS, not on the local development machine. Develop locally, but always **deploy and verify on the server before committing** — local can't fully run the service (encrypted `.env`, heavy deps like PuLP/skforecast, live InfluxDB/MQTT).

| | |
|---|---|
| **Server** | `ssh -p 2222 tom@192.168.0.201` (SSH key passphrase required) |
| **Repo on server** | `/volume1/homes/tom/git/loxone-db-grafana/` |
| **Containers** | **THREE**, all built from `./loxone_smart_home/` (same Dockerfile/context, **separate image tags**): `loxone_ingest` (the data-saving process — UDP listener + weather + OTE + MQTT bridge; owns the `2000/udp` publish), `loxone_smart_home` (the Growatt controller + the API on internal port 5556), and `loxone_web` (the public dashboard pages on 5555, proxies `/api/*` to `loxone_smart_home:5556`). Ingest and controller run the **same default entry point** (`main.py`) and are differentiated only by per-container `*_ENABLED` + `MQTT_CLIENT_ID` `environment:` overrides. See "Web/controller split" below. |
| **Network** | `caddy_net` (external) — shares `influxdb`, `mosquitto`, Grafana with the `selfhosted` stack. The compose `environment:` block overrides `INFLUXDB_HOST=http://influxdb:8086` and `MQTT_BROKER=mosquitto` to reach them by container name. |
| **Local repo** | `/Users/tom/git/LoxoneSmartHome` |
| **selfhosted repo** | `/Users/tom/git/selfhosted` (local) / `/volume1/homes/tom/git/selfhosted` (server) — holds the Caddy config; the energy upstream points at `loxone_web:5555`. |

**Two SSH rules** (both mandatory or commands fail):
- Always use a **login shell** — `ssh -p 2222 tom@192.168.0.201 bash -l -c '<cmd>'` or a `bash -l` heredoc. Docker is not on `PATH` otherwise.
- **No SCP** — the Synology has no SCP subsystem. Transfer files with **tar over ssh** (preserves directory structure).
- For commands with awkward quoting, use a `bash -l << 'ENDSSH'` heredoc (single-quoted marker = no local expansion) or base64-encode the script and `base64 -d` on the server.

### Deploy a change (the standard loop)

```bash
# 1. Copy ONLY changed files (run from loxone_smart_home/; paths are relative to it)
cd /Users/tom/git/LoxoneSmartHome/loxone_smart_home
tar cf - \
  modules/growatt/milp_optimizer.py \
  modules/growatt/deferrable_loads.py \
  modules/growatt_controller.py \
| ssh -p 2222 tom@192.168.0.201 \
  "cd /volume1/homes/tom/git/loxone-db-grafana/loxone_smart_home && tar xf - -v"

# 2. Rebuild + restart. ALL THREE containers build from the same context but have
#    SEPARATE image tags, so rebuild whichever the change affects (see the
#    "which container" rule below). When in doubt, build all three.
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
cd /volume1/homes/tom/git/loxone-db-grafana
docker-compose build loxone_smart_home loxone_web
docker-compose up -d loxone_smart_home loxone_web
ENDSSH

# 3. Verify — tail logs (there is no healthcheck, so inspect Status, not Health)
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
docker ps --format '{{.Names}}: {{.Status}}' | grep loxone
docker logs loxone_smart_home --tail 50 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
ENDSSH
```

**Which container to rebuild** (the file decides):
- Data-saving modules (`modules/udp_listener.py`, `modules/weather_scraper.py`, `modules/ote_price_collector.py`, `modules/mqtt_bridge.py`) → **`loxone_ingest`**. Restarting it does NOT touch the controller (no model rebuild / re-actuation). The Loxone `2000/udp` listener lives here.
- Controller / Growatt async modules / the `/api/*` JSON+control handlers (`modules/growatt_controller.py`, `modules/growatt/*.py`, the `api_*` functions in `dashboard.py`) → **`loxone_smart_home`**. Recreating it restarts the controller (~30–40s: model rebuild, inverter re-actuation).
- Shared code (`config/*`, `main.py`, `utils/*`, `requirements.txt`) affects BOTH `loxone_ingest` and `loxone_smart_home` (same image/entry point) — rebuild both.
- The dashboard **pages/HTML/JS** (`DASHBOARD_HTML`/`SETTINGS_HTML` and page handlers in `dashboard.py`, `run_web_apps.py`) → **`loxone_web`** only (restart is cheap; controller untouched). This is the whole point of the split — iterate on UI without bouncing the controller.
- `modules/growatt/dashboard.py` contains BOTH the API handlers (main) and the page strings (web), so a change there usually means **rebuild both** `loxone_smart_home` + `loxone_web`.

Notes:
- After recreating `loxone_smart_home`, the new container gets a fresh IP; the proxy in `loxone_web` resolves it by service name each request, so no action needed there. But `caddy` caches DNS — if you change the Caddy upstream, **restart the `caddy` container** (`sed -i` on a bind-mounted file changes the inode and the container keeps the old one — see git history / memory).
- Strip ANSI colors from logs with `sed 's/\x1b\[[0-9;]*m//g'`.
- The `.profile` warning (`/opt/etc/profile: No such file or directory`) is harmless — ignore it.
- **InfluxDB queries from the server**: the auto-mode classifier blocks inlining the token literal. Run a `python3 -c`/heredoc **inside a container** and read `os.environ["INFLUXDB_TOKEN"]` (org `loxone`); the container has no `curl`. Flux quoting in a single-quoted `-c '...'` is fiddly — use `q("""from(bucket:\"solar\") ...""")` triple-quoted form (proven to work).

### Data/controller/web split, entry points & ports

Three containers, separated so each tier can be restarted independently:
- **data ingestion** keeps filling InfluxDB even while the controller is bounced or swapped,
- the **controller** can be replaced with a different one in the future without touching ingestion, and
- the **UI** can be iterated without restarting the controller (which would retrain models + re-actuate the inverter).

- **`loxone_ingest`** (image default CMD `run_integrated.py` → `main.py`, but with `GROWATT_CONTROLLER_ENABLED=false`): the data-saving modules only — UDP listener (Loxone → `loxone` bucket, owns `2000/udp`), weather scraper (→ `weather_forecast`), OTE price collector (→ `ote_prices`), and the MQTT→Loxone bridge. Writes nothing controller-derived. A future replacement controller reuses this unchanged. **Must have a distinct `MQTT_CLIENT_ID`** (`loxone-ingest`) — sharing the controller's id would make the two broker connections evict each other in a reconnect loop. All controller↔ingest coupling is via the shared mosquitto broker + InfluxDB (e.g. ingest publishes `loxone/status`; the controller's high-load detection subscribes), so they work fine in separate processes.
- **`loxone_smart_home`** (same image default CMD, with the data modules disabled via `UDP_LISTENER_ENABLED=false`/`WEATHER_SCRAPER_ENABLED=false`/`OTE_COLLECTOR_ENABLED=false`/`MQTT_BRIDGE_ENABLED=false` and `MQTT_CLIENT_ID=loxone-controller`): the controller + Growatt async modules, and the **controller-backed API app on internal port 5556** (`create_api_app` / `start_api_dashboard`). 5556 is `expose`d, NOT published. (`run_integrated.py` also *can* launch the `web/` FastAPI on 8080 when `WEB_SERVICE_ENABLED=true`, but it's **off in prod**.)
- **`loxone_web`** (compose `command: ["python","run_web_apps.py"]`): serves the dashboard **pages** on **5555** (`create_pages_app`) and **reverse-proxies `/api/*`** (incl. the SSE log stream, via the streaming `_proxy_to_api`) to `DASHBOARD_API_UPSTREAM` (default `http://loxone_smart_home:5556`). Stateless — settings POSTs proxy to main, which persists.

Single-process dev: `create_dashboard_app` / `start_dashboard` still serve pages + API in one process. `_add_api_routes` / `_add_page_routes` are the shared route groups.

Ports: `2000/udp` (Loxone UDP listener), **`5555`** (public dashboard pages, served by `loxone_web`), `5556` (internal API on `loxone_smart_home`). API routes include `/api/status /api/live /api/insights /api/economics /api/prices /api/projection /api/settings(GET/POST) /api/restart /api/override /api/reapply /api/logs/stream …`. `/api/economics` (and the `economics` block of `/api/insights`, shared via `_build_economics`) reports today's money from meters/actuals: meter-accurate import cost / export revenue (`_today_grid_economics`, no longer silently drops blocks on a price-lookup miss — counts them in `grid_dropped_*`), real battery arbitrage with grid-vs-solar charge attribution (`_today_battery_arbitrage`), and **saved-today vs a no-battery baseline** (`_today_plan_savings`) replacing the old `plan_value_czk = sum(net_value)` (which mixed CZK and CZK/kWh units). Legacy keys `plan_value_czk`/`arbitrage_czk` are kept `null` during the frontend migration. The `selfhosted` Caddy reverse-proxies `energy.markovi.online` → **`loxone_web:5555`** (`{$DOMAIN_ENERGY}` block), behind Caddy forward-auth + **Pocket ID (OIDC SSO)** — so the public page is already authenticated; there is no app-level auth.

### Settings editor, restart-from-UI, runtime config overrides

- **`/settings`** page (gear icon in the appbar) edits a curated subset of `GrowattConfig` live. Backed by `config/settings_overrides.py` (`EDITABLE_GROUPS` registry → group/label/unit + `hot` vs restart flag; validates via Pydantic incl. cross-field rules; persists to `config_overrides.json`).
- Layering: `.env` defaults → `config_overrides.json` → live `GrowattConfig` (mutated in place at controller `__init__`, and on each POST). **hot** fields apply on the next eval tick + a forced re-eval; **restart** fields (engine/forecaster toggles, log level) need a restart.
- Persistence path: `/app/config_state/config_overrides.json`, backed by the **named docker volume `loxone_config_state`** on `loxone_smart_home` (overridable via `CONFIG_OVERRIDES_PATH` env, used in tests).
- **Restart-from-UI**: `POST /api/restart` → `controller.request_restart()` sends the process `SIGTERM` → graceful shutdown → Docker `restart: unless-stopped` recreates the container, re-reading the overrides. The settings page shows a "Restart" button that polls `/api/status` until it's back.

## Testing

### Locally (fast, default)

Run from the `loxone_smart_home/` directory (not the repo root):

```bash
cd loxone_smart_home

python3 -m pytest tests/modules/ -v                       # main module suite
python3 -m pytest tests/modules/test_milp_optimizer.py -v  # one file
python3 -m pytest tests/ -v                                # everything
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_*` needs no decorator.

**Expected local failures / skips:**
- `tests/web/test_app.py` and `tests/modules/test_growatt_controller.py` try to load the git-crypt-encrypted `.env` → fail locally. Run them in-container (below).
- The EMHASS modules degrade gracefully when their optional deps are missing: `test_milp_optimizer.py` needs **PuLP**, `test_ml_consumption_forecast.py` needs **skforecast + scikit-learn**, `test_solcast_forecast.py` needs **aiohttp**. Install from `requirements.txt` locally, or run the full suite in-container where the real deps live.

### On the server, inside the container (full fidelity)

Use this when a test needs the encrypted `.env`, the heavy ML/LP deps, or live InfluxDB/MQTT. Deploy the changed files first (deploy loop above), then exec into the running container:

```bash
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
docker exec -w /app loxone_smart_home python3 -m pytest tests/ -q 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g'
ENDSSH
```

The image pins the real deps (pulp 3.x, scikit-learn 1.x, skforecast 0.2x), so in-container is the source of truth for the optimizer/forecaster tests.

## Architecture

The application uses:
- **asyncio** for concurrent operations with thread-safe resource management
- **Pydantic** for configuration validation
- **Async shared clients** with connection pooling and retry logic
- **Modular design** with base classes in `modules/base.py`
- **Background task management** for connection monitoring and data buffering

## Key Implementation Details

### UDP Listener
- Receives data from Loxone on port 2000
- Data format: `timestamp;measurement_name;value;room_name;measurement_type;tag1;tag2`
- Converts timestamps from Prague timezone to UTC
- Normalizes measurement names (lowercase, spaces to underscores)
- Uses measurement_type as InfluxDB measurement name, measurement_name as field name

### MQTT-Loxone Bridge
- Subscribes to configurable MQTT topics
- Converts JSON messages to semicolon-separated key=value format
- Forwards messages to Loxone via UDP on port 4000
- Supports comma-separated topic lists via MQTT_TOPICS environment variable
- Example: `{"power": 2500, "voltage": 240}` becomes `power=2500;voltage=240`

### Weather Scraper
- Supports three weather services: OpenMeteo, Aladin, OpenWeatherMap
- Fetches weather forecasts and air quality data
- OpenMeteo: Includes comprehensive weather parameters and air quality (PM10, PM2.5, UV index)
- Aladin: Czech meteorological service with local forecasts
- OpenWeatherMap: Requires API key, provides current weather and hourly forecasts
- Publishes consolidated data to MQTT topic `weather`
- Stores weather data in InfluxDB bucket `weather_forecast`
- Periodic updates with configurable interval (default: 30 minutes)
- Standardized data format using HourlyData named tuple

### Growatt Controller (`modules/growatt_controller.py`)
- Energy price-based battery management with DAM market integration
- Timezone-aware scheduling (Europe/Prague) for automated control
- Orchestrates solar forecast, consumption forecast, and battery optimizer
- Simulation mode for testing without actual hardware control

### Solar Forecast (`modules/growatt/solar_forecast.py`)
- **Learned model** (`SolarProductionModel`): trained on 730 days of historical production + weather data
  - 3-level bin hierarchy: 3D (radiation, cloud, altitude) → 2D (radiation, cloud) → interpolation → global median
  - Two-pass training: build rough model, filter curtailed hours (SOC=100% + low load), rebuild
  - IQR outlier removal, physics cap at nameplate capacity
  - `hit_level_counts` dict tracks which bin level is used for observability
- **forecast.solar API**: per-array estimates, rate-limit handling with InfluxDB cache
- **OpenMeteo weather fallback**: GHI-based calculation with temperature derating (-0.4%/°C above 25°C cell temp)
- **Consensus**: combines all sources. Averages when they agree (<30% divergence); on divergence the trust is gated by the model's per-hour bin level (`hourly_levels`): well-supported `3d`/`2d` hits trust the model in BOTH directions, `interpolate`/`global` hits use the non-model average. A model zero only vetoes other sources when it's a genuine `below_horizon` zero
- **Calibration**: auto-tunes confidence by comparing past forecasts vs actuals. Persists consensus to `solar_forecast_history` measurement in InfluxDB so calibration survives restarts
- **Reliability check**: seasonal thresholds (April-Sept: 0.3 kWh/kWp, Oct-March: 0.1 kWh/kWp)

### Consumption Forecast (`modules/growatt/consumption_forecast.py`)
- Temperature-aware model: (temp_bucket, hour, weekday/weekend) → median kWh
- 30 temperature buckets (-20 to 40°C, 2°C steps), trained on 365 days
- IQR outlier removal, weekly rebuild; bins need ≥4 samples (sparser bins fall back to opposite day-type → adjacent buckets → hourly)
- Failed builds back off 1h (`needs_rebuild()` won't re-fire every 60s tick); same for the ML engine
- Tomorrow is predicted with OpenMeteo forecast temps stashed by `_update_solar_forecast` (`_temps_for_date`), falling back to last-24h actuals

### Battery Optimizer — MILP only (`modules/growatt/milp_optimizer.py`)
Battery dispatch is **MILP-only**: a PuLP/CBC mixed-integer linear program over the 15-min blocks (`MILPBatteryOptimizer`), run off the event loop via `asyncio.to_thread`. The greedy engine and the rule-based scheduler were removed (see git history); `optimizer.py`'s `BatteryOptimizer` is now just **shared infrastructure** the MILP + controller reuse: the base-load profile (48 slots, hour × weekday/weekend, excludes heating + EV), the dynamic per-block reserve-SOC floor (`_compute_reserve_soc_per_block`), `BlockDecision`, `summarize`, and inverter power-rate sizing (`compute_charge_power_rates`/`compute_rate_ceiling`).
- **Safe fallback**: when PuLP is missing or a solve is infeasible/times out, `optimize()` returns a minimal safe plan (`_safe_fallback`): reuse the last good plan if it still covers the horizon, else hold every block and grid-charge only the cheapest blocks up to the reserve floor — never discharge, never export. The `decision_engine` safety gates (battery protection, export, inverter-off) still apply on top.
- `optimizer_enabled=False` means hold (no scheduling); the safety gates still apply. There is no `optimizer_engine` choice anymore.

### EMHASS-inspired features (opt-in, all default OFF)

Added on the `feature/battery-optimization-v2` branch. Each is gated behind a config flag and **defaults to existing behaviour**, so production dispatch is unchanged until explicitly enabled. All degrade gracefully (log + fall back) when their optional dependency is missing.

| Feature | Module | Enable via | Falls back to |
|---|---|---|---|
| ML consumption forecast | `growatt/ml_consumption_forecast.py` | `CONSUMPTION_FORECAST_ENGINE=ml` | binned model (if skforecast missing / training declines) |
| Solcast PV forecast | `growatt/solcast_forecast.py` | `SOLCAST_API_KEY` + `SOLCAST_ROOFTOP_ID` | forecast.solar + model consensus |
| Deferrable loads | `growatt/deferrable_loads.py` | `DEFERRABLE_LOADS_JSON` (JSON array) | none scheduled (empty default) |

(The MILP optimizer used to be an opt-in `OPTIMIZER_ENGINE=milp` feature alongside a greedy engine; it is now the **only** dispatch engine — see the Battery Optimizer section above.)

Notes:
- `consumption_forecast_engine` is pattern-validated in `config/settings.py` (`^(binned|ml)$`) so a typo fails fast at startup instead of silently degrading.
- **MILP** uses an explicit per-block solar/grid/battery/load energy-flow model. Its reserve is emergent from the objective (its `RESERVE_SHORTFALL_FLOOR` is deliberately 0.0 after a live incident — see the comment in `milp_optimizer.py`; don't make it nonzero). Per-leg sqrt efficiency convention (`leg_eta = round_trip ** 0.5` on each conversion); terminal SOC valued with capped economics. The shared reserve helper (`BatteryOptimizer._compute_reserve_soc_per_block`) uses the same convention.
- **Deferrable loads** are added to the optimizer's `consumption_hourly` as an overlay so the battery plans around them. **Only list loads NOT already present in the consumption history** (`INVPowerToLocalLoad`) — otherwise the forecast already learned them and you double-count, over-charging from grid.
- **Solcast** free tier is 10 req/day per rooftop; the client throttles to ≤9/day with a UTC-monotonic interval guard. Quota state is persisted to `solcast_quota.json` next to `config_overrides.json` (survives restarts); any received HTTP response counts, and timeouts count too (the request may have reached Solcast's ledger) — only provably-unsent DNS/refused connect errors are refunded.
- **Deferrable windows** are clamped to the FIRST window instance in the price horizon (`filter_to_current_window_instance`) — the ~32h horizon would otherwise match tomorrow's window too and schedule energy past the current cycle's deadline.
- **Flux gotcha**: every hour/day-keyed `aggregateWindow` query feeding a MODEL or hour-keyed lookup MUST pass `timeSrc: "_start"` — the Flux default labels windows by `_stop`, which silently shifts all hour keys +1 (this bug affected every model pipeline until June 2026). The dashboard's display-only chart queries (`dashboard.py`) still use the stop-label default; treat any new hour-keyed consumer as needing `_start`.
- **Optimizer economics are per-block** (not daily-average): both engines build `import_cost[i] = prices[i] + distribution(hour)` and `export = prices[i] - sell_fee` per 15-min block, so negative prices and VT/NT tariff bands are priced correctly. (The dashboard's "Economics (today)" card is a *separate* reporting estimate — it was meter-accurate-ified in `_today_grid_economics`, but never fed dispatch.)

### High-load protection (EV / heating) — `growatt_controller.py` + `decision_engine.py`

Stops the battery DISCHARGING while a big load runs, so stored energy isn't drained into it. Gate: Priority-3 `high_load_protected` decision node + `and not ctx.high_loads_active` guards on every discharge/sell/hold node. Driven by `self._high_loads_active`. Config group "High-load protection" (`high_load_protection_enabled`, `ev_charging_power_threshold_w`, `ev_high_load_poll_seconds`).

Two detection sources (different, both fixed this branch):
- **Heating**: Loxone relays via the `loxone/status` MQTT cache (`_detect_high_loads_from_status`, `tag1=heating`). Event-driven.
- **EV**: teslamate → InfluxDB only (`ev` measurement: `ev_charging` 0/1, `ev_charging_power` in **kW**, `ev_connected`). Polled by `_high_load_poll_loop` (`_query_ev_charging_from_influx`), merged with heating by `_recompute_high_load_state`. **GOTCHA**: `ev_charging` is written ON CHANGE so it ages out of the 30-min window mid-charge; `ev_charging_power` is **kW** but the threshold is **W**, so detection must use `(power_kw*1000) > threshold_w`. Dashboard `ev_power` is kW (don't /1000).

### SPH inverter mode quirk (critical — `decision_engine.py` MODE_DEFINITIONS + `_build_desired_state`)

On this Growatt SPH, confirmed from telemetry:
- `battery_first` CHARGES the battery up to `stop_soc` **from the grid**, even with `ac_charge=False` (the flag does NOT gate grid-charge).
- `load_first` DISCHARGES the battery to cover any load deficit — `stop_soc` is NOT a discharge floor there.
- `grid_first` maintains `stop_soc` (charges up to it, discharges down to it).

So the "battery passive, no discharge AND no grid-charge" modes — **`sell_production`, `battery_hold`, `high_load_protected`** — must use **`battery_first` with `stop_soc` pinned to the LIVE SOC** (the pin list in `_build_desired_state`). `load_first`/`max_soc` do NOT hold the battery. You cannot bank surplus solar during a hold without also grid-charging, so a hold exports surplus instead. Don't "fix" these back to load_first or stop_soc=max_soc.

Also added: `battery_amortisation_export_czk` (optional, default None=use shared wear cost) — applies an export-only wear penalty to the grid-export term in both engines, leaving self-consumption at the base cost.

### Async Resource Management
- **AsyncMQTTClient**: Thread-safe MQTT operations with automatic reconnection
  - Publish queue for reliable message delivery
  - Background tasks for connection monitoring and message processing
  - Exponential backoff retry logic with configurable timeouts
  - Concurrent callback execution with proper error handling
- **AsyncInfluxDBClient**: Connection pooling with batch processing
  - 5-connection pool for optimal database performance
  - Write buffer (5000 points) with 1-second flush intervals
  - Automatic batching by bucket for efficient writes
  - Retry logic with exponential backoff for failed writes
  - Background flush loop with graceful shutdown handling

### Configuration
- All settings use Pydantic models in `config/settings.py`
- Environment variables are loaded from `.env` file
- Each module can be enabled/disabled via environment variables
- Field validation with proper error messages
- Support for comma-separated lists (e.g., MQTT topics)

## Test Conventions

- Tests use pytest with asyncio support (`asyncio: mode=auto` in pyproject.toml)
- Mock objects for external dependencies (MQTT, InfluxDB)
- Test files mirror the source structure in `tests/`
- Use `AsyncMock` for async methods, `MagicMock` for sync methods

## InfluxDB Buckets

- **solar**: Inverter telemetry (`solar` measurement: `SOC`, `ChargePower`, `DischargePower`, `ACPowerToUser` [grid import W], `ACPowerToGrid` [export W], `INVPowerToLocalLoad`, `PV1/2InputPower`, cumulative `EnergyToUserToday`/`EnergyToGridToday`/`Charge`/`Discharge`…), forecast cache (`solar_forecast_cache`), consensus history (`solar_forecast_history`)
- **weather_forecast**: Weather data from OpenMeteo/Aladin/OWM (`weather_forecast` measurement, `type` tag: `hour`/`day`)
- **loxone**: Loxone Miniserver data: `relay` (`tag1=heating`/`shading`), `ev` (`ev_charging`, `ev_charging_power` kW, `ev_connected` — from teslamate), `temperature`, `target_temp`, humidity, etc.
- **ote_prices**: OTE day-ahead spot prices (`electricity_prices` measurement, fields `price` [EUR/MWh], `price_czk_kwh`), written with future timestamps for the next day. Written by `ote_price_collector`; the controller fetches DAM separately via `price_analyzer.fetch_dam_energy_prices` (negative prices are kept, never clamped).