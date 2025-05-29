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

### Configuration
- All settings use Pydantic models in `config/settings.py`
- Environment variables are loaded from `.env` file
- Each module can be enabled/disabled via environment variables

## Testing

- Tests use pytest with asyncio support
- Mock objects for external dependencies (MQTT, InfluxDB)
- Test files mirror the source structure in `tests/`
- Use `AsyncMock` for async methods, `MagicMock` for sync methods