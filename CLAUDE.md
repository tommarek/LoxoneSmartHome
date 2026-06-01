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
| **Container** | `loxone_smart_home` (built from `./loxone_smart_home/` via docker-compose) |
| **Network** | `caddy_net` (external) — shares `influxdb`, `mosquitto`, Grafana with the `selfhosted` stack. The compose `environment:` block overrides `INFLUXDB_HOST=http://influxdb:8086` and `MQTT_BROKER=mosquitto` to reach them by container name. |
| **Local repo** | `/Users/tom/git/LoxoneSmartHome` |

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

# 2. Rebuild + restart (heredoc, from the REPO ROOT on the server)
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
cd /volume1/homes/tom/git/loxone-db-grafana
docker-compose build loxone_smart_home
docker-compose up -d loxone_smart_home
ENDSSH

# 3. Verify — tail logs and check health
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
docker logs loxone_smart_home --tail 50 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
docker inspect loxone_smart_home --format='{{.State.Health.Status}}'
ENDSSH
```

- If you only changed Python (no new deps), `docker-compose up -d` alone re-runs from the rebuilt layer — but always `build` when in doubt. New deps in `requirements.txt` **require** a rebuild.
- Strip ANSI colors from logs with `sed 's/\x1b\[[0-9;]*m//g'`.
- The `.profile` warning (`/opt/etc/profile: No such file or directory`) is harmless — ignore it.

### Entry points & ports

The Docker image runs `run_integrated.py`, which launches **both** the async main app and the FastAPI web service in one container.

| Script | What it runs | Use for |
|---|---|---|
| `run_integrated.py` | main app **+** web service (the container default) | production |
| `main.py` | async modules only (UDP, MQTT, weather, Growatt) | headless debugging |
| `run_web_service.py` | FastAPI web/dashboard only (needs `WEB_SERVICE_ENABLED=true`) | dashboard/API work |

Ports: `2000/udp` (Loxone UDP listener), `8080` (web service + dashboard + JSON API, e.g. `/api/status`, `/api/live`, `/api/solar_actuals`). The `selfhosted` Caddy reverse-proxies `8080` to `energy.markovi.online`.

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
- **Consensus**: combines all sources. Model is primary; averages with others when they agree (<30% divergence), trusts model when it predicts higher, uses non-model average when model underpredicts
- **Calibration**: auto-tunes confidence by comparing past forecasts vs actuals. Persists consensus to `solar_forecast_history` measurement in InfluxDB so calibration survives restarts
- **Reliability check**: seasonal thresholds (April-Sept: 0.3 kWh/kWp, Oct-March: 0.1 kWh/kWp)

### Consumption Forecast (`modules/growatt/consumption_forecast.py`)
- Temperature-aware model: (temp_bucket, hour, weekday/weekend) → median kWh
- 30 temperature buckets (-20 to 40°C, 2°C steps), trained on 365 days
- IQR outlier removal, weekly rebuild with daily EMA updates

### Battery Optimizer (`modules/growatt/optimizer.py`)
- Greedy forward-simulation over 15-minute blocks
- Scores charge/discharge/hold per block considering price, solar, consumption, battery state
- Dynamic reserve calculation: sums base load until next recharge opportunity
- Base load profile: 48 slots (hour × weekday/weekend), excludes heating + EV hours

### EMHASS-inspired features (opt-in, all default OFF)

Added on the `feature/battery-optimization-v2` branch. Each is gated behind a config flag and **defaults to existing behaviour**, so production dispatch is unchanged until explicitly enabled. All degrade gracefully (log + fall back) when their optional dependency is missing.

| Feature | Module | Enable via | Falls back to |
|---|---|---|---|
| MILP optimizer | `growatt/milp_optimizer.py` | `OPTIMIZER_ENGINE=milp` | greedy (if PuLP missing / infeasible / timeout) |
| ML consumption forecast | `growatt/ml_consumption_forecast.py` | `CONSUMPTION_FORECAST_ENGINE=ml` | binned model (if skforecast missing / training declines) |
| Solcast PV forecast | `growatt/solcast_forecast.py` | `SOLCAST_API_KEY` + `SOLCAST_ROOFTOP_ID` | forecast.solar + model consensus |
| Deferrable loads | `growatt/deferrable_loads.py` | `DEFERRABLE_LOADS_JSON` (JSON array) | none scheduled (empty default) |

Notes:
- `optimizer_engine` / `consumption_forecast_engine` are pattern-validated in `config/settings.py` (`^(greedy|milp)$`, `^(binned|ml)$`) so a typo fails fast at startup instead of silently degrading.
- **MILP** is a drop-in for the greedy engine (identical signature, same 4-tuple output) using an explicit per-block solar/grid/battery/load energy-flow model; runs off the event loop via `asyncio.to_thread`. Reuses the greedy helper's reserve-SOC computation so both engines protect the same overnight energy.
- **Deferrable loads** are added to the optimizer's `consumption_hourly` as an overlay so the battery plans around them. **Only list loads NOT already present in the consumption history** (`INVPowerToLocalLoad`) — otherwise the forecast already learned them and you double-count, over-charging from grid.
- **Solcast** free tier is 10 req/day per rooftop; the client throttles to ≤9/day with a UTC-monotonic interval guard and counts every attempt (success or failure).

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

- **solar**: Inverter telemetry (`solar` measurement), forecast cache (`solar_forecast_cache`), consensus history (`solar_forecast_history`)
- **weather_forecast**: Weather data from OpenMeteo/Aladin/OWM (`weather_forecast` measurement, `type` tag: `hour`/`day`)
- **loxone**: Loxone Miniserver data (relays, temperatures, humidity, etc.)