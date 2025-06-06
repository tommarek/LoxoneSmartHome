# PEMS v2 Analysis Framework - Complete Usage Guide

This document provides a comprehensive guide to the PEMS v2 analysis framework, which has been completely restructured in Phase 1B to provide a modular, extensible analysis pipeline for your Loxone smart home system.

## 🏗️ Framework Architecture

### System Understanding

Your Loxone heating system works with **binary relay control**:
- **Relays**: ON (1) or OFF (0) states only
- **Power**: Each room has fixed power rating when relay is ON
- **Energy**: `Energy = Relay_State × Power_Rating × Time_Duration`
- **Duty Cycle**: Percentage of time relay is ON (indicates heating demand)

## 🚀 Quick Start

### Prerequisites
```bash
# Set up development environment (first time only)
make setup

# Activate virtual environment (required for all operations)
source venv/bin/activate
```

### Available Analysis Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `python analysis/run_analysis.py` | Full comprehensive analysis (2 years) | Complete system analysis |
| `python -c "from analysis.run_analysis import run_quick_analysis; run_quick_analysis(30)"` | Quick analysis (30 days) | Recent performance check |
| `python -c "from analysis.run_analysis import run_pv_only_analysis; run_pv_only_analysis()"` | PV production analysis only | Solar system performance |
| `python -c "from analysis.run_analysis import run_thermal_only_analysis; run_thermal_only_analysis()"` | Thermal analysis only | Heating system analysis |
| `python analysis/daily_analysis.py` | Daily analysis workflow | Automated daily reports |

### Test Commands

| Command | Description |
|---------|-------------|
| `make test-basic` | Basic structure and import tests |
| `make test-extraction` | Data extraction from InfluxDB |
| `make test-relay` | Relay analysis functionality |
| `make test` | Full test suite with coverage |
| `make lint` | Code formatting and quality check |

## 📊 Comprehensive Analysis Pipeline

### 1. ComprehensiveAnalyzer (Main Orchestrator)

The `ComprehensiveAnalyzer` class coordinates the entire analysis pipeline:

```python
from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer
from config.settings import PEMSSettings

# Initialize comprehensive analyzer
settings = PEMSSettings()
analyzer = ComprehensiveAnalyzer(settings)

# Run complete analysis
results = await analyzer.run_comprehensive_analysis(
    start_date=datetime(2023, 1, 1),
    end_date=datetime.now(),
    analysis_types={
        "pv": True,               # PV production analysis
        "thermal": True,          # Thermal dynamics analysis  
        "base_load": True,        # Base load analysis
        "relay_patterns": True,   # Relay pattern analysis
        "weather_correlation": True  # Weather correlation analysis
    }
)
```

### 2. Analysis Modules

#### PV Production Analysis (`analyzers/pattern_analysis.py`)
- **Solar production patterns** with weather correlation
- **Seasonal analysis** and capacity factor calculation
- **Loxone solar field integration** (sun_elevation, sun_direction, solar_irradiance)
- **Export policy optimization** and curtailment analysis
- **ML model development** for production forecasting

#### Thermal Dynamics Analysis (`analyzers/thermal_analysis.py`)
- **Room-by-room thermal modeling** with RC parameter estimation
- **Heat-up and cool-down rate analysis** for each room
- **Thermal time constants** and comfort analysis
- **Weather correlation** with outdoor temperature
- **Relay integration** for heating period detection
- **Room coupling analysis** for thermal interactions

#### Base Load Analysis (`analyzers/base_load_analysis.py`)
- **Baseline consumption patterns** excluding heating and PV
- **Time-of-day and seasonal patterns** analysis
- **Load factor calculations** and efficiency metrics
- **Peak demand identification** and load shifting opportunities
- **Economic analysis** with time-of-use pricing

#### Relay Pattern Analysis (`analyzers/pattern_analysis.py`)
- **Switching frequency analysis** per room
- **Duty cycle calculations** and energy consumption
- **Peak demand coordination** opportunities
- **Load distribution analysis** across rooms
- **Economic optimization** with energy pricing

### 3. Data Processing Pipeline

#### Data Extraction (`core/data_extraction.py`)
```python
# Extract data from multiple sources
data = {
    "pv": await extractor.extract_pv_data(start_date, end_date),
    "rooms": await extractor.extract_room_temperatures(start_date, end_date),
    "weather": await extractor.extract_weather_data(start_date, end_date),
    "consumption": await extractor.extract_energy_consumption(start_date, end_date),
    "relay_states": await extractor.extract_relay_states(start_date, end_date),
    "battery": await extractor.extract_battery_data(start_date, end_date),
    "ev": await extractor.extract_ev_data(start_date, end_date),
    "prices": await extractor.extract_energy_prices(start_date, end_date)
}
```

#### Data Preprocessing (`core/data_preprocessing.py`)
```python
# Clean and standardize data
preprocessor = DataPreprocessor()
processed_data = {}

for data_type, raw_data in data.items():
    processed_data[data_type] = preprocessor.process_dataset(raw_data, data_type)
```

#### Loxone Field Standardization (`utils/loxone_adapter.py`)
```python
# Standardize Loxone field names
from analysis.utils.loxone_adapter import LoxoneFieldAdapter

# Room data standardization
standardized_room = LoxoneFieldAdapter.standardize_room_data(room_df, "obyvak")
# temperature_obyvak → temperature
# humidity_obyvak → humidity  
# obyvak → relay_state

# Weather data standardization  
standardized_weather = LoxoneFieldAdapter.standardize_weather_data(weather_df)
# sun_elevation → sun_elevation
# absolute_solar_irradiance → solar_irradiance
```

### 4. Report Generation (`reports/report_generator.py`)

#### Comprehensive Summary Reports
```python
from analysis.reports.report_generator import ReportGenerator

generator = ReportGenerator()

# Text summary report
summary = generator.create_comprehensive_summary(analysis_results)

# HTML report with visualizations
html_report = generator.create_html_report(analysis_results, processed_data)

# Data quality assessment
quality_report = generator.create_data_quality_report(quality_data)
```

#### Generated Reports
- **Text summaries** with key metrics and recommendations
- **HTML reports** with interactive visualizations  
- **Data quality assessments** with gap analysis
- **Automated recommendations** based on analysis results

## 🏠 Your System Configuration

### Room Power Ratings (when relay is ON)
```python
# From config/energy_settings.py
ROOM_CONFIG = {
    "rooms": {
        "obyvak": {"power_kw": 4.8},           # Living room (highest power)
        "hosti": {"power_kw": 2.02},           # Guest room  
        "chodba_dole": {"power_kw": 1.8},      # Downstairs hallway
        "chodba_nahore": {"power_kw": 1.2},    # Upstairs hallway
        "loznice": {"power_kw": 1.2},          # Bedroom
        "pokoj_1": {"power_kw": 1.2},          # Room 1
        "pokoj_2": {"power_kw": 1.2},          # Room 2
        "pracovna": {"power_kw": 0.82},        # Office
        "satna_dole": {"power_kw": 0.82},      # Downstairs wardrobe
        "zadveri": {"power_kw": 0.82},         # Entryway
        "technicka_mistnost": {"power_kw": 0.82}, # Technical room
        "koupelna_nahore": {"power_kw": 0.62},    # Upstairs bathroom
        "satna_nahore": {"power_kw": 0.56},       # Upstairs wardrobe
        "spajz": {"power_kw": 0.46},              # Pantry
        "koupelna_dole": {"power_kw": 0.47},      # Downstairs bathroom
        "zachod": {"power_kw": 0.22}              # Toilet (lowest power)
    }
}

# Total System Capacity: 18.12 kW
```

### Loxone Field Mappings

The system automatically maps Loxone field names:

| Data Type | Loxone Field | Standard Field | Example |
|-----------|--------------|----------------|---------|
| Temperature | `temperature_obyvak` | `temperature` | 22.5°C |
| Humidity | `humidity_obyvak` | `humidity` | 45% |
| Relay State | `obyvak` | `relay_state` | 1 (ON) |
| Solar Elevation | `sun_elevation` | `sun_elevation` | 45° |
| Solar Irradiance | `absolute_solar_irradiance` | `solar_irradiance` | 800 W/m² |

## 📈 Analysis Output Structure

### Generated Files

After running analysis, you'll find organized outputs:

```
data/
├── raw/                           # Raw data from InfluxDB
│   ├── pv_data.parquet
│   ├── room_{room_name}.parquet
│   ├── weather_data.parquet
│   ├── relay_{room_name}.parquet
│   └── ...
├── processed/                     # Cleaned and standardized data
│   ├── pv_processed.parquet
│   ├── rooms_{room_name}_processed.parquet
│   └── ...
└── features/                      # ML-ready features

analysis/
├── results/                       # Analysis results (JSON)
│   └── comprehensive_analysis_results.json
├── reports/                       # Generated reports
│   ├── comprehensive_analysis_summary.txt
│   ├── comprehensive_analysis_report.html
│   ├── data_quality_report.json
│   └── daily/                     # Daily analysis reports
│       └── daily_summary_YYYYMMDD.txt
└── figures/                       # Generated visualizations
```

### Key Analysis Results Structure

```json
{
  "pv_analysis": {
    "basic_stats": {
      "total_energy_kwh": 12500.5,
      "max_power": 8500.0,
      "capacity_factor": 0.18,
      "peak_months": [5, 6, 7, 8]
    },
    "weather_correlations": {
      "strongest_positive": ["solar_irradiance", {"correlation": 0.95}]
    },
    "seasonal_patterns": { "best_month": "June", "worst_month": "December" }
  },
  "thermal_analysis": {
    "obyvak": {
      "basic_stats": {
        "mean_temperature": 22.1,
        "temperature_range": 3.2,
        "heating_percentage": 45.2
      },
      "time_constant": { "time_constant_hours": 8.5 },
      "rc_parameters": {
        "recommended_parameters": {
          "R": 0.15, "C": 8000, "time_constant": 8.5
        }
      }
    }
  },
  "base_load_analysis": {
    "basic_stats": {
      "mean_base_load": 1250.0,
      "total_energy_kwh": 3650.0,
      "base_load_percentage": 35.2
    }
  },
  "relay_analysis": {
    "obyvak": {
      "daily_cycles": { "mean_cycles_per_day": 12.5 },
      "efficiency": { "heating_efficiency": 87.3 }
    }
  }
}
```

## 🔄 Daily Analysis Workflow

### Automated Daily Analysis

```python
from analysis.daily_analysis import DailyAnalysisWorkflow
from config.settings import PEMSSettings

# Set up daily workflow
settings = PEMSSettings()
workflow = DailyAnalysisWorkflow(settings)

# Run daily analysis for yesterday
results = await workflow.run_daily_analysis()

# Run weekly analysis
weekly_results = await workflow.run_weekly_analysis()
```

### Daily Analysis Features

- **Incremental processing** for recent data (7-day window)
- **Daily performance metrics** extraction
- **Trend detection** and anomaly identification  
- **Automated summary generation** with alerts
- **Weekly aggregation** and comparison

### Daily Report Example

```text
PEMS v2 Daily Analysis Summary - 2024-12-06
============================================================

PV Production:
  • Daily Energy: 15.3 kWh
  • Peak Power: 6,200 W
  • Capacity Factor: 18.5%

Thermal Performance:
  • Total Heating Hours: 8.2
  • Rooms Analyzed: 16
  • Average Efficiency: 89.2%

Base Load:
  • Daily Consumption: 35.7 kWh
  • Peak Demand: 2,100 W
  • Load Factor: 0.712

Trend Analysis:
  • PV Trend: increasing
  • Thermal Trend: stable
  • Base Load Trend: stable

No alerts detected.
============================================================
```

## 🎛️ Advanced Usage

### Custom Analysis Configuration

```python
# Run analysis with specific configuration
from analysis.run_analysis import run_analysis
from datetime import datetime, timedelta

# Custom date range
start_date = datetime(2024, 6, 1)
end_date = datetime(2024, 8, 31)

# Custom analysis types
analysis_types = {
    "pv": True,               # Enable PV analysis
    "thermal": True,          # Enable thermal analysis
    "base_load": False,       # Skip base load analysis
    "relay_patterns": True,   # Enable relay analysis
    "weather_correlation": False  # Skip weather correlation
}

results = await run_analysis(start_date, end_date, analysis_types)
```

### Individual Module Usage

```python
# Use individual analyzers
from analysis.analyzers.thermal_analysis import ThermalAnalyzer
from analysis.analyzers.pattern_analysis import PVAnalyzer

# Thermal analysis for specific rooms
thermal_analyzer = ThermalAnalyzer()
thermal_results = await thermal_analyzer.analyze_room_dynamics(
    room_data, weather_data, relay_data
)

# PV production analysis
pv_analyzer = PVAnalyzer()
pv_results = await pv_analyzer.analyze_pv_production(pv_data, weather_data)
```

### Data Integration with Loxone

```python
from analysis.utils.loxone_adapter import LoxoneDataIntegrator

integrator = LoxoneDataIntegrator()

# Prepare data for thermal analysis
thermal_rooms, thermal_weather = integrator.prepare_thermal_analysis_data(
    rooms_data, relay_data, weather_data
)

# Prepare data for PV analysis  
pv_data, enhanced_weather = integrator.prepare_pv_analysis_data(
    pv_data, weather_data
)
```

## 🔍 Understanding Your Results

### Key Performance Indicators

#### System Efficiency Metrics
- **System Utilization**: Percentage of total capacity used
- **Duty Cycles**: Percentage of time each room's relay is ON
- **Energy Distribution**: kWh consumption by room
- **Switching Frequency**: Number of relay operations per day

#### Thermal Performance
- **Time Constants**: How quickly rooms heat up/cool down (hours)
- **RC Parameters**: Thermal resistance (°C/W) and capacitance (Wh/°C)
- **Heat-up/Cool-down Rates**: Temperature change per hour
- **Comfort Scores**: Temperature stability and control quality

#### PV Performance
- **Capacity Factor**: Actual vs. theoretical production
- **Weather Correlations**: Impact of solar irradiance and temperature
- **Seasonal Patterns**: Monthly and daily production variations
- **Export Efficiency**: Self-consumption vs. grid export

### Optimization Insights

#### Automated Recommendations
The system generates specific recommendations based on analysis:

1. **High heating usage rooms** → Check insulation
2. **High switching frequency** → Adjust thermostat deadband
3. **Low PV capacity factor** → Check for shading/maintenance
4. **Peak demand coordination** → Stagger relay operations
5. **Load shifting opportunities** → Use time-of-use pricing

#### Example Analysis Results (2-year data)
- **Total Energy**: ~3,339 kWh over 2 years (1,670 kWh/year)
- **System Utilization**: ~4% (very efficient!)
- **Top Consumers**: obyvak (1,400 kWh), chodba_dole (568 kWh)
- **Duty Cycles**: Most rooms 48-53% (well balanced)

## 🛠️ Customization and Extension

### Adding New Analysis Modules

1. Create new analyzer in `analysis/analyzers/`:
```python
class NewAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NewAnalyzer")
    
    async def analyze_new_feature(self, data):
        # Your analysis logic here
        return analysis_results
```

2. Integrate in `ComprehensiveAnalyzer`:
```python
# Add to _run_comprehensive_analyses method
if analysis_types.get("new_feature", True):
    self.analysis_results["new_analysis"] = await self.new_analyzer.analyze_new_feature(data)
```

### Extending Report Generation

```python
# Add new report section in ReportGenerator
def _generate_new_summary(self, new_results):
    lines = ["NEW ANALYSIS", "-" * 20]
    # Add your report formatting
    return lines
```

### Custom Data Sources

Extend the data extraction for new data sources:
```python
# Add to DataExtractor
async def extract_new_data_source(self, start_date, end_date):
    # Your data extraction logic
    return data
```

## 📚 Testing and Quality Assurance

### Available Tests

| Test Type | Command | Coverage |
|-----------|---------|----------|
| Basic Structure | `make test-basic` | Module imports, directory structure |
| Data Extraction | `make test-extraction` | InfluxDB connectivity, data retrieval |
| Relay Analysis | `make test-relay` | Relay pattern analysis functionality |
| Full Suite | `make test` | Complete test coverage with pytest |

### Code Quality

- **Linting**: `make lint` - Black formatting, isort imports, flake8 checks
- **Type Safety**: All modules pass strict mypy checking
- **Test Coverage**: 76 tests covering all implemented modules
- **Documentation**: Comprehensive docstrings and type hints

### Continuous Integration

```bash
# Before committing changes
make lint        # Format code and check quality
make test        # Run full test suite
```

## ❓ Troubleshooting

### Common Issues

#### Import Errors After Restructure
- **Cause**: Old import paths after Phase 1B restructure
- **Solution**: Use new import paths:
  ```python
  # Old (Phase 1A)
  from analysis.pattern_analysis import PVAnalyzer
  
  # New (Phase 1B)
  from analysis.analyzers.pattern_analysis import PVAnalyzer
  ```

#### No Data Found
- **Cause**: Incorrect date range or missing data in InfluxDB
- **Solution**: 
  ```python
  # Check available data range
  python -c "from analysis.core.data_extraction import DataExtractor; import asyncio; asyncio.run(DataExtractor().check_data_availability())"
  ```

#### Analysis Errors
- **Environment**: Ensure virtual environment is activated: `source venv/bin/activate`
- **Dependencies**: Check all dependencies: `make setup`
- **Configuration**: Verify `.env` file with InfluxDB settings
- **Permissions**: Check InfluxDB token permissions for all buckets

#### Empty Results
- **Seasonal**: Heating systems inactive in summer - try winter months
- **Date Range**: Extend analysis period for more data
- **Data Sources**: Verify InfluxDB bucket and measurement names

### Debug Mode

Enable detailed logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
results = await run_analysis(start_date, end_date)
```

## 🎯 Next Steps and Roadmap

### Phase 2: ML Model Development (Planned)

1. **PV Production Forecasting**
   - Weather-based production prediction
   - Seasonal model adjustment
   - Real-time forecast updates

2. **Thermal Predictive Models**
   - Room temperature prediction
   - Heating demand forecasting
   - Optimal scheduling algorithms

3. **Load Prediction Models**
   - Base load forecasting
   - Peak demand prediction
   - Grid interaction optimization

4. **Energy Management System**
   - Automated control strategies
   - Cost optimization algorithms
   - Grid services integration

### Immediate Recommendations

1. **Regular Analysis**: Run monthly analysis to track system performance
2. **Seasonal Comparison**: Compare winter vs summer patterns
3. **Optimization Implementation**: Apply recommendations from analysis results
4. **Performance Monitoring**: Track efficiency improvements over time
5. **Data Quality**: Monitor and maintain high-quality data collection

### Integration Opportunities

- **Home Assistant**: Real-time data visualization
- **Grafana**: Custom dashboards and alerting
- **Energy Trading**: Dynamic pricing optimization
- **Smart Grid**: Demand response participation

---

*PEMS v2 Analysis Framework - Phase 1B Complete*  
*Professional, modular analysis pipeline for Loxone smart home systems*