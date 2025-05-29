# TODO - Loxone Smart Home Consolidation

## Completed ✅

- [x] Create new branch for Python consolidation
- [x] Create directory structure for consolidated Python app
- [x] Set up requirements.txt with all dependencies
- [x] Create main.py with async architecture
- [x] Set up virtual environment
- [x] Create requirements-dev.txt
- [x] Create test structure and initial tests
- [x] Fix all flake8 linting issues
- [x] Fix most mypy type checking issues

## In Progress 🚧

### Code Migration
- [x] Migrate UDP listener (loxone_to_db) from Rust to Python ✅
  - Implemented exact parsing logic from Rust code
  - Matches field names, defaults, and behavior
  - Handles timezone conversion (Prague to UTC)
  - Normalizes measurement names (lowercase, spaces to underscores)
  - All tests passing

- [ ] Integrate MQTT-Loxone bridge functionality
  - Current: Basic structure created
  - TODO: Test UDP forwarding to Loxone

- [ ] Integrate weather scraper module
  - Current: Basic structure created
  - TODO: Implement OpenMeteo, Aladin, OpenWeatherMap API calls
  - TODO: Implement data transformation and storage

- [ ] Integrate Growatt controller module
  - Current: Basic structure created
  - TODO: Implement energy price scraping from OTE
  - TODO: Implement battery control logic
  - TODO: Implement scheduling and optimization algorithms

### Type Safety
- [ ] Fix remaining mypy type errors in tests
  - AsyncMock type assignments
  - Add proper type stubs for test fixtures
  - Fix Settings constructor calls with missing influxdb_token

### Deployment
- [ ] Create Dockerfile for consolidated app
- [ ] Update docker-compose.yml to use new consolidated service
- [ ] Remove old service definitions from docker-compose.yml

## Future Enhancements 🔮

### Performance
- [ ] Add connection pooling for database writes
- [ ] Implement batch processing for UDP messages
- [ ] Add metrics/monitoring with Prometheus

### Features
- [ ] Add web API for configuration management
- [ ] Add health check endpoints
- [ ] Implement graceful reload for configuration changes
- [ ] Add support for multiple Loxone systems

### Testing
- [ ] Add integration tests with real MQTT broker
- [ ] Add integration tests with real InfluxDB
- [ ] Add performance/load tests
- [ ] Achieve >90% test coverage

### Documentation
- [ ] Create comprehensive API documentation
- [ ] Add deployment guide
- [ ] Create configuration examples
- [ ] Add troubleshooting guide

## Known Issues 🐛

1. **Type checking**: Some mypy errors remain in test files due to AsyncMock typing
2. **Weather APIs**: OpenMeteo library not available, need to implement direct API calls
3. **Energy price scraping**: Web scraping logic needs to be tested with actual OTE website

## Dependencies to Consider 📦

- Consider using `httpx` instead of `requests` for async HTTP calls
- Consider `aiofiles` for async file operations if needed
- Consider `structlog` for structured logging
- Consider `tenacity` for retry logic
