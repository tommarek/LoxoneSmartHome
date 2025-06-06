# PEMS v2 Analysis Module - Usage Guide

This document explains how to use the corrected PEMS v2 analysis module that properly understands your Loxone relay-based heating system.

## 🔧 System Understanding

Your heating system works with **binary relay control**:
- **Relays**: ON (1) or OFF (0) states only
- **Power**: Each room has fixed power rating when relay is ON
- **Energy**: `Energy = Relay_State × Power_Rating × Time_Duration`
- **Duty Cycle**: Percentage of time relay is ON (indicates heating demand)

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have Python 3.13+ and virtual environment
make setup
```

### Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make clean-analysis` | Remove all analysis outputs and data files |
| `make run-relay-analysis` | Run corrected relay heating analysis (30 days) |
| `make run-analysis` | Run full 2-year thermal analysis pipeline |
| `make test-extraction` | Test data extraction from InfluxDB |

### Basic Usage

1. **Clean previous analysis results:**
   ```bash
   make clean-analysis
   ```

2. **Run relay analysis (recommended):**
   ```bash
   make run-relay-analysis
   ```

3. **Run full thermal analysis:**
   ```bash
   make run-analysis
   ```

## 📊 Analysis Outputs

### Generated Files

After running analysis, you'll find:

```
data/raw/
├── relay_analysis_corrected.parquet    # Raw relay state data
├── room_*_2year.parquet                # Individual room temperature data
└── outdoor_weather_2year.parquet       # Weather data

analysis/results/
├── relay_analysis_corrected.json       # Relay analysis results
└── thermal_analysis_2year.json         # Thermal dynamics results

analysis/reports/
├── corrected_relay_analysis_2year.txt  # Comprehensive relay report
└── 2year_analysis_report.txt           # Full thermal analysis report
```

### Key Metrics Explained

**Duty Cycle**: Percentage of time relay is ON
- 50% = Relay on half the time
- Higher % = More heating demand

**Energy Consumption**: Actual kWh used
- Calculated as: `Relay_ON_Time × Room_Power_Rating`
- Example: 50% duty × 2.0kW × 24h = 24 kWh/day

**Switching Frequency**: Number of ON→OFF or OFF→ON changes
- High switching may indicate control tuning needed
- Typical: 50-200 switches per week per room

## 🏠 Your System Configuration

**Room Power Ratings (when relay is ON):**
```
obyvak: 4.8kW          (highest power - living room)
hosti: 2.02kW           (guest room)  
chodba_dole: 1.8kW      (downstairs hallway)
chodba_nahore: 1.2kW    (upstairs hallway)
loznice: 1.2kW          (bedroom)
pokoj_1: 1.2kW          (room 1)
pokoj_2: 1.2kW          (room 2)
pracovna: 0.82kW        (office)
satna_dole: 0.82kW      (downstairs wardrobe)
zadveri: 0.82kW         (entryway)
technicka_mistnost: 0.82kW (technical room)
koupelna_nahore: 0.62kW (upstairs bathroom)
satna_nahore: 0.56kW    (upstairs wardrobe)
spajz: 0.46kW           (pantry)
koupelna_dole: 0.47kW   (downstairs bathroom)
zachod: 0.22kW          (toilet - lowest power)
```

**Total System Capacity**: 19.0kW

## 📈 Understanding Your Results

### Typical Analysis Results (2-year data):

- **Total Energy**: ~3,339 kWh over 2 years (1,670 kWh/year)
- **System Utilization**: ~4% (very efficient!)
- **Top Consumers**: obyvak (1,400 kWh), chodba_dole (568 kWh)
- **Duty Cycles**: Most rooms 48-53% (well balanced)

### What the Numbers Mean:

**Low System Utilization (4%)**: 
- ✅ Excellent efficiency
- ✅ Good temperature control
- ✅ No energy waste

**Balanced Duty Cycles (50%)**:
- ✅ Optimal comfort vs efficiency
- ✅ Proper thermostat settings
- ✅ Good thermal characteristics

**Room Variations**:
- Higher duty = more heat loss or higher setpoint
- Lower duty = better insulation or lower usage

## 🔍 Optimization Insights

### From Your Analysis Results:

1. **obyvak** uses most energy (large room, high power)
2. **zadveri** has highest switching (2,518 switches) - may need tuning
3. **technicka_mistnost** has highest duty cycle (53.1%) - check insulation
4. **spajz** barely heats (2.1 kWh, 2 switches) - very efficient

### Recommended Actions:

1. **Reduce switching frequency** in high-activity rooms
2. **Coordinate relay timing** to reduce peak demand
3. **Implement time-of-use scheduling** for cost optimization
4. **Group similar rooms** for coordinated operation

## 🛠️ Customization

### Modify Analysis Period

Edit the date range in analysis scripts:
```python
# For 30-day analysis
start_date = end_date - timedelta(days=30)

# For 1-year analysis  
start_date = end_date - timedelta(days=365)

# For specific period
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
```

### Add New Room

Update the power mapping in analysis scripts:
```python
ROOM_POWER = {
    # ... existing rooms ...
    "new_room": 1.5,  # kW rating when relay is ON
}
```

### Change InfluxDB Connection

Update `.env` file:
```bash
INFLUXDB_URL=http://your-server:8086
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=your-org
INFLUXDB_bucket_solar=your-bucket
```

## 📚 Analysis Components

### 1. Relay Analysis (`test_relay_analysis.py`)
- Binary relay state analysis
- Duty cycle calculations  
- Switching frequency analysis
- Energy consumption from relay states

### 2. Thermal Analysis (`run_2year_analysis.py`)
- Room temperature patterns
- Thermal dynamics modeling
- Heat loss characteristics
- Temperature stability analysis

### 3. Corrected Reports (`corrected_analysis_report.py`)
- Comprehensive system overview
- Room-by-room performance
- Optimization recommendations
- System efficiency metrics

## ❓ Troubleshooting

### No Relay Data Found
- Check if heating season (relays more active in winter)
- Verify measurement names in InfluxDB match "relay" 
- Confirm tag "heating" exists on relay measurements
- Check room names match your system

### Analysis Errors
- Ensure virtual environment is activated: `. venv/bin/activate`
- Check InfluxDB connection in `.env`
- Verify all dependencies installed: `make setup`

### Empty Results
- Relay systems may be inactive in summer
- Try longer date ranges for more data
- Check InfluxDB bucket and measurement names

## 🎯 Next Steps

1. **Run regular analysis** to track system performance
2. **Compare seasonal patterns** (winter vs summer)
3. **Implement optimizations** based on insights
4. **Monitor efficiency improvements** over time

---

*Generated for PEMS v2 - Relay-Based Heating Analysis*