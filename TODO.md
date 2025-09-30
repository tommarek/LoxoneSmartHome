# TODO - LoxoneSmartHome

## Smart Battery Optimization - Remaining Tasks

### Performance Monitoring
- [ ] Add metrics collection for battery cycling patterns
- [ ] Track actual vs predicted profit from discharge operations
- [ ] Monitor command success rate after retry improvements
- [ ] Analyze clock drift patterns over time

### Future Enhancements

#### Advanced Battery Management
- [ ] Implement weather-based charging predictions (solar forecast integration)
- [ ] Add machine learning for consumption pattern prediction
- [ ] Dynamic profit margin adjustment based on historical success
- [ ] Multi-day optimization for extreme price events

#### System Resilience
- [ ] Add health check endpoint for monitoring
- [ ] Implement circuit breaker pattern for failing commands
- [ ] Add persistent state storage for recovery after restarts
- [ ] Create backup configuration profiles

#### User Experience
- [ ] Add Grafana dashboard templates for battery optimization metrics
- [ ] Create mobile-friendly control interface
- [ ] Implement notification system for exceptional events (high profit opportunities)
- [ ] Add simulation mode UI for testing strategies

### Code Quality Improvements
- [ ] Increase test coverage to >90%
- [ ] Add integration tests for complete charge/discharge cycles
- [ ] Performance profiling of decision engine
- [ ] Documentation of optimization strategies

### Configuration Enhancements
- [ ] Add per-season configuration profiles
- [ ] Support multiple battery systems
- [ ] Time-of-use tariff configuration wizard
- [ ] Export configuration validation tool

## Completed in Current Branch 
- Fixed linting issues in decision_engine.py
- Cleaned up configuration duplicates
- Enhanced MQTT retry mechanism (10 retries, 10s timeout)
- Reduced retry logging verbosity
- Added configurable clock drift buffer
- Implemented transaction-like command execution with rollback
- Fixed mypy module path conflicts
- Adjusted discharge parameters for realistic operation
- Fixed type errors in GrowattController API (state tracking)
- Fixed type errors in weather_scraper (params type hints)
- Updated all mode_manager tests for new retry behavior
- Fixed retry count logging (shows actual retries, not attempts)

## Notes
- Current discharge margin (2.5�) is more realistic than previous (4.0�)
- Retry mechanism now uses exponential backoff capped at 30s
- Clock drift buffer is configurable (2-10 minutes)
- Rollback mechanism ensures consistent inverter state on failures