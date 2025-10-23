# Loxone Smart Home Web Service

A comprehensive web monitoring and analytics service for the Loxone Smart Home system.

## Features

### Real-time Monitoring
- ⚡ Live energy flow visualization
- 🔋 Battery status and state of charge
- 💰 Current electricity prices
- 🌤️ Weather conditions
- 📊 System performance metrics

### Analytics & Insights
- 📈 Energy production and consumption charts
- 💹 Price forecasting and optimization
- 🎯 Optimal charging/discharging schedules
- 📊 Usage pattern analysis
- 💡 System optimization recommendations

### Data Export
- CSV export for analysis
- PDF report generation
- Historical data access

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `.env` file (if not already configured)

## Running the Service

### Standalone Mode
Run the web service independently:
```bash
python run_web_service.py
```

The service will be available at `http://localhost:8080`

### Integrated Mode
The web service can also run as part of the main Loxone Smart Home application.

## API Documentation

When running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## API Endpoints

### Energy
- `GET /api/energy/current` - Current power flow
- `GET /api/energy/history` - Historical energy data
- `GET /api/energy/battery/status` - Battery status
- `GET /api/energy/statistics` - Aggregated statistics

### Prices
- `GET /api/prices/current` - Current electricity price
- `GET /api/prices/forecast` - Price forecast (48h)
- `GET /api/prices/schedule` - Optimal charging schedule
- `GET /api/prices/savings` - Savings summary

### Weather
- `GET /api/weather/current` - Current weather
- `GET /api/weather/forecast` - Weather forecast
- `GET /api/weather/solar-forecast` - Solar production forecast

### Analytics
- `GET /api/analytics/efficiency` - System efficiency metrics
- `GET /api/analytics/patterns` - Usage pattern analysis
- `GET /api/analytics/recommendations` - Optimization recommendations
- `GET /api/analytics/performance` - Component performance

### WebSocket
- `WS /ws/live` - Real-time data stream

## Dashboard Features

### Main View
- Real-time power flow diagram
- Current electricity price with next change notification
- Weather conditions and forecast
- System status overview

### Charts
- 24-hour energy production/consumption
- 48-hour price forecast with colored indicators
- Optimal charging/discharging schedule

### Statistics
- Daily/weekly/monthly aggregations
- Self-sufficiency and self-consumption rates
- Cost savings and CO₂ avoidance

## Configuration

### Web Service Settings

```python
# In config/settings.py or via environment variables
WEB_ENABLED=true
WEB_PORT=8080
WEB_HOST=0.0.0.0

# Cache settings
WEB_CACHE_TTL_CURRENT=10  # seconds
WEB_CACHE_TTL_HISTORICAL=300  # 5 minutes
WEB_CACHE_TTL_ANALYTICS=3600  # 1 hour

# WebSocket settings
WEB_WEBSOCKET_INTERVAL=5  # seconds

# Feature flags
WEB_ENABLE_API_DOCS=true
WEB_ENABLE_METRICS=false
WEB_ENABLE_EXPORT=true
```

## Development

### Project Structure
```
web/
├── api/           # API endpoints
├── models/        # Pydantic models
├── services/      # Business logic
├── static/        # CSS, JS, images
├── templates/     # HTML templates
└── app.py         # FastAPI application
```

### Adding New Endpoints
1. Create endpoint in appropriate `api/` module
2. Add response model in `models/responses.py`
3. Update dashboard JavaScript if needed

### Testing
```bash
# Run tests
pytest tests/web/

# Test specific endpoint
curl http://localhost:8080/api/energy/current
```

## Performance Optimization

### Caching Strategy
- Current data: 10-second TTL
- Historical data: 5-minute TTL
- Analytics: 1-hour TTL

### Data Aggregation
- 15-minute aggregation for real-time views
- Hourly rollups for daily charts
- Daily summaries for monthly views

### WebSocket Optimization
- Broadcast only changed data
- Configurable update intervals
- Automatic reconnection

## Troubleshooting

### Connection Issues
- Check if the service is running: `curl http://localhost:8080/health`
- Verify InfluxDB connection
- Check MQTT broker connectivity

### Missing Data
- Ensure data collection modules are running
- Check InfluxDB for data presence
- Verify correct bucket names in configuration

### WebSocket Disconnections
- Check browser console for errors
- Verify WebSocket URL in dashboard.js
- Check for proxy/firewall issues

## Future Enhancements

- [ ] User authentication and multi-user support
- [ ] Mobile app integration
- [ ] Advanced alerting system
- [ ] Machine learning predictions
- [ ] Integration with other smart home platforms
- [ ] Historical data comparison tools
- [ ] Energy consumption forecasting
- [ ] Automated report scheduling