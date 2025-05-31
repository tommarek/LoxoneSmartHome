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

### Growatt Controller
- Energy price-based battery management with DAM market integration
- Timezone-aware scheduling (Europe/Prague) for automated control
- Battery-first mode scheduling during cheapest electricity hours
- Export control during high-price periods above configurable threshold
- Startup state synchronization to apply correct mode on service restart
- Enhanced logging with price analysis (EUR/MWh to CZK/kWh conversion)
- Contiguous hour grouping for optimal battery charging/discharging periods
- Simulation mode for testing without actual hardware control

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

## Testing

- Tests use pytest with asyncio support
- Mock objects for external dependencies (MQTT, InfluxDB)
- Test files mirror the source structure in `tests/`
- Use `AsyncMock` for async methods, `MagicMock` for sync methods
- Comprehensive test coverage: 76 tests covering all modules
- Type safety: All tests pass strict mypy checking
- Mock best practices: Use `# type: ignore[attr-defined]` for test-specific mock attributes
- Clean async test execution with proper resource cleanup

## Module Status

### Completed ✅
- **UDP Listener**: Fully migrated from Rust with exact behavior matching
- **MQTT Bridge**: Complete implementation with 11 comprehensive test cases
- **Weather Scraper**: Full implementation with three weather APIs and 11 test cases
- **Growatt Controller**: Complete energy price-based battery management system
- **Async Resource Management**: Thread-safe MQTT and InfluxDB clients with connection pooling
- **Type Safety**: All source files pass strict mypy checking
- **Code Quality**: 100% linting compliance (flake8 with 100-char limit)
- **Test Coverage**: 76 tests covering all implemented modules with clean async execution

### Next Phase 🎯
- **Logging Improvements**: Local timezone timestamps and service-specific prefixes
- **Loxone Control Integration**: MQTT command structure for manual override controls

## Error Handling

- All modules include proper null checks for optional dependencies
- Configuration validation with descriptive error messages
- Graceful degradation when services are unavailable
- Proper exception logging with context information
- Robust async task cleanup with graceful shutdown
- Connection retry logic with exponential backoff
- Background task management with proper cancellation handling

## Performance Features

- **Connection Pooling**: 5-connection InfluxDB pool for optimal throughput
- **Batch Processing**: 5000-point write buffer with 1-second flush intervals
- **Concurrent Operations**: Thread-safe access to shared resources across modules
- **Resource Optimization**: Automatic reconnection and health monitoring
- **Memory Management**: Proper cleanup of async tasks and connections