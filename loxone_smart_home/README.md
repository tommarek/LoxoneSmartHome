# Loxone Smart Home - Consolidated Python Service

A unified Python service that consolidates multiple smart home automation services into a single async application.

## Features

- **UDP Listener**: Receives data from Loxone and stores in InfluxDB
- **MQTT Bridge**: Forwards MQTT messages to Loxone via UDP
- **Weather Scraper**: Fetches weather data from multiple sources
- **Growatt Controller**: Manages solar battery based on energy prices

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for InfluxDB, MQTT broker, etc.)

### Development Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   cd loxone_smart_home
   ```

2. Set up the development environment:
   ```bash
   make install-dev
   ```

3. Copy the example environment file and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Configuration

Configure the application using environment variables in `.env`:

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `*_ENABLED`: Enable/disable individual modules
- `MQTT_*`: MQTT broker configuration
- `INFLUXDB_*`: InfluxDB configuration
- `LOXONE_*`: Loxone system configuration
- See `.env.example` for all available options

## Development

### Code Quality Tools

- **Type checking**: `make type-check`
- **Linting**: `make lint`
- **Formatting**: `make format`
- **All checks**: `make check`

### Testing

Run tests:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

### Pre-commit Hooks

Pre-commit hooks are configured to run:
- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- mypy (type checking)
- Bandit (security checks)

## Running

### Local Development

```bash
make run
```

### Docker

```bash
docker-compose up
```

## Project Structure

```
loxone_smart_home/
├── config/          # Configuration and settings
├── modules/         # Service modules
├── utils/           # Shared utilities
├── tests/           # Test suite
├── main.py          # Application entry point
├── requirements.txt # Production dependencies
└── requirements-dev.txt # Development dependencies
```

## Architecture

The application uses:
- **Pydantic** for configuration validation
- **asyncio** for concurrent operations
- **Shared clients** for MQTT and InfluxDB connections
- **Modular design** for easy extension and maintenance

## Contributing

1. Install development dependencies: `make install-dev`
2. Make your changes
3. Run tests and checks: `make check test`
4. Format code: `make format`
5. Submit a pull request
