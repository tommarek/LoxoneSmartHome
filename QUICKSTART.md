# Quick Start - Loxone Smart Home Deployment

## Prerequisites
- Docker and Docker Compose installed
- Existing InfluxDB with authentication token

## Deployment Steps

### 1. Setup Environment
```bash
cd loxone_smart_home
cp .env.production .env
# Edit .env if needed (but production defaults should work)
```

### 2. Build and Deploy
```bash
# Build the consolidated service
docker-compose build loxone_smart_home

# Start the service
docker-compose up -d loxone_smart_home
```

### 3. Verify
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f loxone_smart_home
```

## What's Running

The consolidated `loxone_smart_home` service replaces these old services:
- ✅ **UDP Listener** (port 2000) - Receives Loxone data → InfluxDB
- ✅ **MQTT Bridge** - Forwards MQTT topics → Loxone UDP port 4000
- ✅ **Weather Scraper** - OpenMeteo/Aladin/OpenWeatherMap → MQTT & InfluxDB
- ✅ **Growatt Controller** - Energy price optimization → MQTT commands

## Configuration

All settings are in environment variables. Key ones:
- `LOXONE_HOST=192.168.0.200` - Your Loxone IP
- `MQTT_TOPICS=energy/solar,teplomer/TC,teplomer/RH,teslamate/cars/1/+` - Topics to forward
- `INFLUXDB_TOKEN` - Already set in .env.production

## Monitoring

- **Grafana**: http://localhost:3000 (admin/adminadmin)
- **InfluxDB**: http://localhost:8086
- **Service Health**: `docker inspect loxone_smart_home --format='{{.State.Health.Status}}'`

## Troubleshooting

If data isn't flowing:
1. Check logs: `docker-compose logs loxone_smart_home`
2. Verify UDP traffic: `docker exec loxone_smart_home netstat -ulnp`
3. Test MQTT: `docker exec mosquitto mosquitto_sub -t '#' -v`