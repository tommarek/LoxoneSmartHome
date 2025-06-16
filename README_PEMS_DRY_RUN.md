# PEMS v2 Continuous Dry Run Service

This service runs PEMS v2 optimization cycles continuously in simulation mode, collecting behavioral data and performance metrics for analysis and validation.

## 🚀 Quick Start

```bash
# Start the service
./scripts/start_pems_dry_run.sh

# View live logs
docker-compose -f docker-compose.pems-dry-run.yml logs -f pems-dry-run

# Analyze collected data
./scripts/start_pems_dry_run.sh analyze

# Stop the service
./scripts/start_pems_dry_run.sh stop
```

## 📊 What It Does

The PEMS dry run service:

1. **Runs Optimization Cycles**: Executes PEMS v2 optimization every 15 minutes (configurable)
2. **Simulates System State**: Evolves realistic system conditions (temperature, battery, load)
3. **Tests Control Logic**: Validates Growatt and heating control decisions
4. **Collects Behavioral Data**: Records optimization results, control decisions, and performance metrics
5. **Monitors Performance**: Tracks success rates, solve times, and system health

## 📁 Data Collection

The service collects data in structured JSONL format:

### Data Files

- **`optimization_cycles.jsonl`**: Complete optimization cycle results
  - Solve times, objective values, success/failure status
  - System state snapshots (battery SOC, temperatures, load)
  - Control decisions (heating on/off, Growatt modes)
  - Performance metrics and error information

- **`mqtt_commands.jsonl`**: All MQTT commands generated
  - Command topics and payloads
  - Timestamps for command frequency analysis
  - Control action validation

- **`system_states.jsonl`**: System state evolution
  - Battery SOC evolution patterns
  - Temperature dynamics and responses
  - Load patterns and variations

- **`service_metrics.json`**: Service-level metrics
  - Uptime, cycle counts, success rates
  - Performance statistics and resource usage

- **`errors.jsonl`**: Error logs and failure patterns

### Example Data Structure

```json
// optimization_cycles.jsonl
{
  "timestamp": "2024-01-15T14:30:00",
  "cycle_id": 123,
  "success": true,
  "solve_time_seconds": 1.2,
  "objective_value": 245.6,
  "battery_soc": 0.65,
  "room_temperatures": {"obyvak": 21.2, "kuchyne": 20.8},
  "heating_decisions": {"obyvak": {"current_state": true, "duty_cycle": 0.75}},
  "growatt_decisions": {"battery_first_schedule": {"current_state": true}},
  "total_cycle_time_ms": 1850
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PEMS_CYCLE_INTERVAL_MINUTES` | 15 | Time between optimization cycles |
| `PEMS_HORIZON_HOURS` | 6 | Optimization horizon length |
| `PEMS_MAX_SOLVE_TIME` | 30 | Maximum solver time (seconds) |
| `PEMS_DATA_RETENTION_HOURS` | 168 | Data retention period (1 week) |

### Customization

Copy and edit the environment file:
```bash
cp .env.pems-dry-run .env
# Edit .env with your preferred settings
```

## 📈 Data Analysis

### Automated Analysis

```bash
# Run complete analysis
python scripts/analyze_pems_data.py --data-dir ./data/pems_dry_run --output-dir ./analysis

# The analysis generates:
# - Performance metrics and trends
# - Control pattern analysis  
# - System behavior insights
# - Error analysis and recommendations
# - Comprehensive markdown report
```

### Analysis Outputs

- **`optimization_performance.png`**: Performance charts and trends
- **`system_behavior.png`**: System state evolution plots
- **`pems_analysis_report.md`**: Comprehensive analysis report

### Key Metrics Tracked

**Optimization Performance:**
- Success rate over time
- Solve time distribution and trends
- Objective value evolution
- Hourly performance patterns

**Control Behavior:**
- Heating duty cycles by room
- Growatt mode selection patterns
- MQTT command frequencies
- Control decision correlations

**System Dynamics:**
- Battery SOC evolution
- Temperature response patterns
- Load variation analysis
- Weather impact assessment

## 🔍 Monitoring & Health Checks

### Service Health

The service includes built-in health monitoring:

```bash
# Check service status
docker-compose -f docker-compose.pems-dry-run.yml ps

# View health check logs
docker inspect pems-dry-run-service | jq '.[0].State.Health'
```

### Performance Monitoring

Monitor key performance indicators:

- **Success Rate**: Should be >80% for healthy operation
- **Solve Time**: Average <10s, P95 <30s recommended
- **Memory Usage**: Tracked automatically with alerts
- **Error Rate**: Logged and analyzed for patterns

### Log Analysis

```bash
# Real-time logs
docker-compose -f docker-compose.pems-dry-run.yml logs -f pems-dry-run

# Search for errors
docker-compose -f docker-compose.pems-dry-run.yml logs pems-dry-run | grep ERROR

# Performance summary
grep "SUCCESS\|FAILED" logs/pems_dry_run/pems_dry_run.log | tail -20
```

## 🛠️ Troubleshooting

### Common Issues

**Optimization Failures**
```bash
# Check for constraint infeasibility
grep "infeasible" logs/pems_dry_run/pems_dry_run.log

# Analyze error patterns
python scripts/analyze_pems_data.py --data-dir ./data/pems_dry_run --output-dir ./debug
```

**High Solve Times**
- Reduce `PEMS_HORIZON_HOURS` to 3-4 hours
- Increase `PEMS_MIP_GAP` to 0.1 for faster solving
- Check system resource availability

**Memory Issues**
- Monitor with `docker stats pems-dry-run-service`
- Increase memory limit in docker-compose.yml
- Reduce data retention period

### Debug Mode

Enable detailed logging:
```bash
# Set in .env file
PEMS_DEBUG_MODE=true
PEMS_VERBOSE_LOGGING=true

# Restart service
./scripts/start_pems_dry_run.sh restart
```

## 📊 Optional Services

### Grafana Dashboard

Visualize data in real-time:
```bash
# Start Grafana
docker-compose -f docker-compose.pems-dry-run.yml up -d pems-dashboard

# Access at http://localhost:3000
# Login: admin/admin
```

### Log Aggregation (Seq)

Centralized log viewing:
```bash
# Start Seq
docker-compose -f docker-compose.pems-dry-run.yml up -d pems-logs

# Access at http://localhost:5341
```

## 🎯 Use Cases

### Performance Validation

**Before Production Deployment:**
- Run for 24-48 hours to validate stability
- Analyze success rates and solve times
- Verify control logic correctness
- Identify potential failure modes

### Algorithm Development

**Testing New Features:**
- Compare optimization algorithms
- Validate new constraint formulations
- Analyze control decision quality
- Performance impact assessment

### System Monitoring

**Continuous Operation:**
- Long-term behavior analysis
- Seasonal pattern identification
- Performance degradation detection
- Capacity planning insights

## 🔧 Development

### Adding Custom Metrics

Extend data collection by modifying `PEMSDryRunService`:

```python
# Add custom metrics in _run_optimization_cycle()
custom_metric = calculate_custom_metric(result)
cycle_data.custom_metric = custom_metric
```

### Custom Analysis

Create analysis plugins in `scripts/analyze_pems_data.py`:

```python
def analyze_custom_behavior(self) -> Dict:
    """Add your custom analysis here."""
    # Analyze collected data
    return analysis_results
```

## 📋 Maintenance

### Data Management

```bash
# Check data usage
du -sh data/pems_dry_run/

# Archive old data
tar -czf pems_data_$(date +%Y%m%d).tar.gz data/pems_dry_run/

# Clean up old files (automatic via data retention setting)
```

### Service Updates

```bash
# Update service code
./scripts/start_pems_dry_run.sh stop
git pull
./scripts/start_pems_dry_run.sh start
```

## 🚀 Production Readiness

The dry run service validates PEMS v2 for production deployment:

✅ **Optimization Stability**: Consistent convergence and performance  
✅ **Control Logic**: Correct Growatt and heating decisions  
✅ **Error Handling**: Graceful failure recovery  
✅ **Resource Usage**: Efficient memory and CPU utilization  
✅ **Data Quality**: Complete and consistent data collection  

When dry run shows consistent >90% success rate with acceptable performance, PEMS v2 is ready for production deployment with real hardware control.