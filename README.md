# LoxoneSmartHome

Consolidated smart home automation system that integrates Loxone Miniserver with solar energy management, weather forecasting, and comprehensive data visualization.

## Overview

This project consolidates multiple services into a single, efficient Python application that:
- Collects data from Loxone smart home system
- Manages Growatt solar battery based on electricity prices
- Provides weather forecasting and monitoring
- Visualizes all data through Grafana dashboards

## Architecture

### Core Service: `loxone_smart_home`

A single consolidated Python service that replaces the previous separate containers:

- **UDP Listener Module** (formerly loxone_to_db)
  - Receives sensor data from Loxone on UDP port 2000
  - Stores data in InfluxDB with proper timestamps and tags
  
- **MQTT-Loxone Bridge Module** (formerly mqtt-loxone-bridge)
  - Forwards MQTT messages to Loxone via UDP
  - Configurable topic filtering
  
- **Weather Scraper Module** (formerly weather_scraper)
  - Fetches forecasts from OpenMeteo, Aladin, and OpenWeatherMap
  - Publishes to MQTT and stores in InfluxDB
  
- **Growatt Controller Module** (formerly growatt_controller)
  - Fetches day-ahead electricity prices
  - Optimizes battery charging schedule
  - Controls export based on price thresholds

### Supporting Services

- **Grafana** - Data visualization and dashboards
- **InfluxDB** - Time series database for all metrics
- **Mosquitto** - MQTT broker for device communication
- **Telegraf** - Additional metrics collection
- **TeslaMate** - Tesla vehicle tracking (optional)

## Quick Start

1. **Setup Environment**
   ```bash
   cd loxone_smart_home
   cp .env.production .env
   # Edit .env if needed
   ```

2. **Deploy**
   ```bash
   docker-compose build loxone_smart_home
   docker-compose up -d
   ```

3. **Access**
   - Grafana: http://localhost:3000 (admin/adminadmin)
   - InfluxDB: http://localhost:8086

## Configuration

Key environment variables:
- `LOXONE_HOST` - IP address of your Loxone Miniserver
- `INFLUXDB_TOKEN` - Authentication token for InfluxDB
- `MQTT_TOPICS` - Comma-separated list of topics to forward to Loxone
- `UDP_LISTENER_PORT` - Port for receiving Loxone data (default: 2000)
- `GROWATT_EXPORT_PRICE_THRESHOLD` - Price above which to enable grid export

See [.env.example](loxone_smart_home/.env.example) for all options.

## Data Flow

```
Loxone Miniserver → UDP:2000 → loxone_smart_home → InfluxDB → Grafana
                                        ↓
MQTT Broker ← ← ← ← ← ← ← ← ← ← ← ← ← ↓
     ↓                                 ↓
Growatt Inverter              Weather APIs
```

## Status

✅ **Project Completed**: All services have been successfully consolidated into a single Python application with comprehensive testing, timezone support, and enhanced logging.

## Documentation

- [Deployment Guide](loxone_smart_home/DEPLOYMENT.md) - Detailed deployment instructions
- [Quick Start](QUICKSTART.md) - Get up and running quickly
- [Development Guide](loxone_smart_home/README.md) - For developers
- [TODO List](TODO.md) - Planned features and improvements

## License

This project is licensed under the MIT License.