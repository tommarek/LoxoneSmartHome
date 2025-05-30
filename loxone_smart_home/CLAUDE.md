# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a consolidated Python service that combines multiple smart home automation services into a single async application. It integrates Loxone home automation with Growatt solar systems, weather data, and energy price optimization.

## Development Commands

Before making any commits, run these commands to ensure code quality:

```bash
# Run all tests
make test

# Run linting (flake8 with 100 char line limit)
make lint

# Run type checking (mypy in strict mode)
make type-check

# Format code automatically
make format
```

## Architecture

The application uses:
- **asyncio** for concurrent operations
- **Pydantic** for configuration validation
- **Shared clients** for MQTT and InfluxDB connections
- **Modular design** with base classes in `modules/base.py`

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

### Configuration
- All settings use Pydantic models in `config/settings.py`
- Environment variables are loaded from `.env` file
- Each module can be enabled/disabled via environment variables
- Field validation with proper error messages
- Support for comma-separated lists (e.g., MQTT topics)

## Testing

- Tests use pytest with asyncio support
- Mock objects for external dependencies (MQTT, InfluxDB)
- Test files mirror the source structure in `tests/`
- Use `AsyncMock` for async methods, `MagicMock` for sync methods
- Comprehensive test coverage: 60 tests covering all modules
- Type safety: All tests pass strict mypy checking
- Mock best practices: Use `# type: ignore[attr-defined]` for test-specific mock attributes

## Module Status

### Completed ✅
- **UDP Listener**: Fully migrated from Rust with exact behavior matching
- **MQTT Bridge**: Complete implementation with 11 comprehensive test cases
- **Weather Scraper**: Full implementation with three weather APIs and 11 test cases
- **Type Safety**: All 23 source files pass strict mypy checking
- **Code Quality**: 100% linting compliance (flake8 with 100-char limit)
- **Test Coverage**: 60 tests covering all implemented modules

### In Progress 🚧
- **Growatt Controller**: Basic structure created, needs energy price scraping and control logic

## Error Handling

- All modules include proper null checks for optional dependencies
- Configuration validation with descriptive error messages
- Graceful degradation when services are unavailable
- Proper exception logging with context information