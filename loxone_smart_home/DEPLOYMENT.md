# Deployment Guide - Loxone Smart Home

This guide explains how to deploy the consolidated Loxone Smart Home service.

## Prerequisites

- Docker and Docker Compose installed
- Access to InfluxDB with a valid authentication token
- MQTT broker (Mosquitto) running
- Loxone Miniserver accessible on the network

## Deployment Steps

### 1. Environment Configuration

Copy the example environment file and configure it:

```bash
cd loxone_smart_home
cp .env.example .env
```

Edit `.env` and set:
- `INFLUXDB_TOKEN` - Your InfluxDB authentication token
- `LOXONE_HOST` - IP address of your Loxone Miniserver
- `MQTT_TOPICS` - Comma-separated list of MQTT topics to forward
- `OPENWEATHERMAP_API_KEY` - If using OpenWeatherMap service
- Adjust other settings as needed

### 2. Build the Docker Image

```bash
docker build -t loxone_smart_home ./loxone_smart_home
```

### 3. Deploy with Docker Compose

Use the new consolidated docker-compose file:

```bash
# Stop old services
docker-compose down

# Start with new configuration
docker-compose -f docker-compose.new.yml up -d
```

### 4. Verify Deployment

Check that all services are running:

```bash
docker-compose -f docker-compose.new.yml ps
```

Check logs:

```bash
# All services
docker-compose -f docker-compose.new.yml logs -f

# Just the consolidated service
docker-compose -f docker-compose.new.yml logs -f loxone_smart_home
```

### 5. Health Checks

The service includes a health check that verifies the UDP listener is running:

```bash
docker inspect loxone_smart_home --format='{{.State.Health.Status}}'
```

## Configuration

### Module Control

Enable/disable individual modules via environment variables:
- `UDP_LISTENER_ENABLED` - Receives data from Loxone
- `MQTT_BRIDGE_ENABLED` - Forwards MQTT to Loxone
- `WEATHER_SCRAPER_ENABLED` - Fetches weather data
- `GROWATT_CONTROLLER_ENABLED` - Manages solar battery

### Service-Specific Settings

#### UDP Listener
- `UDP_LISTENER_PORT` - Port for receiving Loxone data (default: 2000)

#### MQTT Bridge
- `MQTT_TOPICS` - Topics to forward (comma-separated)
- `LOXONE_PORT` - UDP port on Loxone (default: 4000)

#### Weather Scraper
- `USE_SERVICE` - Weather service: openmeteo, aladin, or openweathermap
- `LATITUDE/LONGITUDE` - Location coordinates

#### Growatt Controller
- `GROWATT_SIMULATION_MODE` - Test without sending commands
- `GROWATT_EXPORT_PRICE_THRESHOLD` - Price to enable export (CZK/kWh)
- `GROWATT_BATTERY_CHARGE_HOURS` - Consecutive hours for AC charging
- `GROWATT_INDIVIDUAL_CHEAPEST_HOURS` - Individual cheap hours to use

## Migration from Old Services

The consolidated service replaces these individual services:
- `loxone_to_db` → UDP Listener module
- `mqtt-loxone-bridge` → MQTT Bridge module  
- `weather_scraper` → Weather Scraper module
- `growatt_controller` → Growatt Controller module

To migrate:

1. Export any necessary data from old services
2. Stop old services: `docker-compose stop loxone_to_db mqtt-loxone-bridge weather_scraper growatt_controller`
3. Deploy the consolidated service
4. Remove old containers: `docker-compose rm loxone_to_db mqtt-loxone-bridge weather_scraper growatt_controller`

## Troubleshooting

### Service won't start
- Check logs: `docker-compose -f docker-compose.new.yml logs loxone_smart_home`
- Verify environment variables in `.env`
- Ensure InfluxDB token is valid

### UDP data not received
- Check firewall rules for port 2000/udp
- Verify Loxone is sending to correct IP/port
- Use `tcpdump` to monitor UDP traffic

### MQTT connection issues
- Verify MQTT broker is accessible
- Check MQTT credentials if authentication is enabled
- Monitor MQTT logs

### Weather data missing
- Check API keys for weather services
- Verify internet connectivity from container
- Check service-specific rate limits

## Performance Tuning

### InfluxDB Batching
Adjust in code if needed:
- `batch_size` - Number of points before flush (default: 5000)
- `flush_interval` - Max time between flushes in ms (default: 1000)

### Logging
Reduce log verbosity for production:
```bash
LOG_LEVEL=WARNING
```

### Resource Limits
Add to docker-compose.yml if needed:
```yaml
loxone_smart_home:
  ...
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 512M
      reservations:
        cpus: '0.5'
        memory: 256M
```

## Monitoring

### Prometheus Metrics (Future)
The service is prepared for Prometheus integration. Metrics endpoint will be available at `:8080/metrics` when implemented.

### InfluxDB Dashboards
Import Grafana dashboards from the `grafana/` directory to visualize:
- Loxone sensor data
- Weather forecasts
- Solar production and battery status
- Energy prices and optimization

## Backup

Important data to backup:
- `.env` file with configuration
- InfluxDB data directory
- Grafana dashboards and settings

## Security Considerations

1. The service runs as non-root user (uid 1000)
2. Application code is mounted read-only
3. Only necessary ports are exposed
4. Consider using Docker secrets for sensitive data
5. Enable MQTT authentication in production
6. Use HTTPS for external access to Grafana

## Updates

To update the service:

```bash
# Pull latest code
git pull

# Rebuild image
docker build -t loxone_smart_home ./loxone_smart_home

# Restart service
docker-compose -f docker-compose.new.yml up -d loxone_smart_home
```