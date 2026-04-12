# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a consolidated Python service that combines multiple smart home automation services into a single async application. It integrates Loxone home automation with Growatt solar systems, weather data, and energy price optimization.

## Remote Server & Deployment

**IMPORTANT**: The service runs on a Synology NAS, not on the local development machine.

- **Server**: `ssh -p 2222 tom@192.168.0.201`
- **Repo on server**: `/volume1/homes/tom/git/loxone-db-grafana/`
- **Container**: `loxone_smart_home` (built from `./loxone_smart_home/` via docker-compose)
- **No SCP** — the Synology doesn't have SCP subsystem. Transfer files via tar over ssh.

**Deploy workflow** — always deploy and verify before committing:

```bash
# 1. Copy changed files (from loxone_smart_home/ directory, paths relative to it)
cd /Users/tom/git/LoxoneSmartHome/loxone_smart_home
tar cf - \
  modules/growatt/optimizer.py \
  modules/growatt/decision_engine.py \
  modules/growatt_controller.py \
| ssh -p 2222 tom@192.168.0.201 \
  "cd /volume1/homes/tom/git/loxone-db-grafana/loxone_smart_home && tar xf - -v"

# 2. Rebuild and restart (use heredoc for multi-command)
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
cd /volume1/homes/tom/git/loxone-db-grafana
docker-compose build loxone_smart_home
docker-compose up -d loxone_smart_home
ENDSSH

# 3. Check logs
ssh -p 2222 tom@192.168.0.201 bash -l << 'ENDSSH'
docker logs loxone_smart_home --tail 50 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
ENDSSH
```

Notes:
- Always use `bash -l` for SSH — docker isn't on PATH without a login shell.
- The `.profile` warning (`/opt/etc/profile: No such file or directory`) is harmless — ignore it.
- Use `sed 's/\x1b\[[0-9;]*m//g'` to strip ANSI color codes from log output.
- The `tar` approach preserves directory structure; list only the files that changed.

## Testing

Run tests from the `loxone_smart_home/` directory (not repo root):

```bash
cd loxone_smart_home

# Run all module tests (the main test suite)
python3 -m pytest tests/modules/ -v

# Run a specific test file
python3 -m pytest tests/modules/test_solar_forecast.py -v
```

Note: `tests/web/test_app.py` and `tests/modules/test_growatt_controller.py` will fail locally because they try to load `.env` which is git-crypt encrypted. This is expected — run those tests on the server inside the container if needed.

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