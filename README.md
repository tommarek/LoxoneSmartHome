# LoxoneSmartHome

Advanced smart home automation system that integrates Loxone Miniserver with predictive energy management, AI-powered optimization, and comprehensive control systems.

## Project Structure

This repository contains two applications:
1. **`loxone_smart_home`** - Production data collection system (v1) ✅ Complete
2. **`pems_v2`** - Advanced Predictive Energy Management System (v2) ✅ **Production Ready**

## Overview

This project provides enterprise-grade smart home energy management with two integrated systems:

### 🏠 **v1 System** - Data Collection & Monitoring
- Collects real-time data from Loxone smart home system (17 rooms, 70+ sensors)
- Manages Growatt solar battery based on electricity prices
- Provides weather forecasting and monitoring
- Visualizes all data through Grafana dashboards

### 🤖 **v2 PEMS** - AI-Powered Energy Management
- **Advanced ML Predictors**: Solar production, thermal dynamics, load forecasting
- **Real-Time Optimization**: <1 second decision-making for 6-hour horizons
- **Intelligent Control**: 17-room heating, battery, and inverter management
- **Multi-Strategy Operation**: Economic, Comfort, Environmental, Balanced modes
- **Enterprise Features**: 76+ tests, 90%+ data quality, error recovery

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

### Running the Current System (v1)

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

### Running PEMS v2 (Production Ready)

1. **Setup PEMS Environment**
   ```bash
   cd pems_v2
   make setup                 # Set up virtual environment and dependencies
   source venv/bin/activate   # Activate virtual environment (REQUIRED)
   ```

2. **Validate System Health**
   ```bash
   make test-basic           # Test system structure (30s)
   make test-extraction      # Test database connectivity (2min)
   make test                 # Full test suite with 76+ tests (3min)
   ```

3. **Run Analysis & Control**
   ```bash
   # Complete 2-year analysis (generates ROI reports, recommendations)
   python analysis/run_analysis.py
   
   # Live system validation (tests all components working together)
   python validate_complete_system.py
   
   # Interactive control demonstration
   python examples/control_system_demo.py
   ```

4. **Both Systems Together**
   ```bash
   # Start all services including both v1 and v2
   docker-compose up -d
   docker-compose -f docker-compose.pems.yml up -d
   ```

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

✅ **v1 System Complete**: Production data collection with comprehensive testing, timezone support, and enhanced logging  
🚀 **v2 PEMS Production Ready**: Enterprise-grade AI energy management system with 96% documentation coverage

### Key Achievements
- **🏠 17-Room Control**: Individual heating management with 18.12kW total capacity
- **⚡ Real-Time Performance**: 50,000+ records/second, <1 second optimization
- **🤖 Advanced AI**: Physics+ML hybrid models with uncertainty quantification
- **🔧 Enterprise Quality**: 76+ tests, error recovery, graceful degradation
- **📊 Complete Analysis**: 2-year ROI analysis with implementation roadmaps

## Documentation

### Production Documentation
- **[PRESENTATION.md](PRESENTATION.md)** - 🎯 **Complete Technical Documentation** (96% coverage, 52/54 files)
- **[pems_v2/README_ANALYSIS.md](pems_v2/README_ANALYSIS.md)** - Complete user manual and analysis guide
- **[pems_v2/validate_complete_system.py](pems_v2/validate_complete_system.py)** - Production readiness validation

### Setup & Operations
- [Deployment Guide](loxone_smart_home/DEPLOYMENT.md) - v1 system deployment
- [Quick Start](QUICKSTART.md) - Get up and running quickly
- [Development Guide](loxone_smart_home/README.md) - v1 system development
- [CLAUDE.md](CLAUDE.md) - Development guidelines and system architecture

## License

This project is licensed under the MIT License.