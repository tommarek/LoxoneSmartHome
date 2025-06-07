# Predictive Energy Management System (PEMS) v2

PEMS v2 is an advanced machine learning-based energy management system that optimizes heating, battery storage, and EV charging to minimize energy costs while maintaining comfort.

## Overview

PEMS replaces the simple rule-based Growatt controller with a sophisticated predictive optimization system that:

- **Predicts** PV production, energy consumption, and thermal dynamics using ML models
- **Optimizes** energy flows across a 48-hour horizon considering electricity prices
- **Controls** heating systems, battery storage, and EV charging in real-time
- **Learns** from historical data to continuously improve predictions

## Phase 1 Status: ✅ COMPLETE

Phase 1 data analysis and feature engineering is now complete! All core functionality has been implemented:

### Completed Components:
- ✅ **Data Extraction**: Full async InfluxDB integration with all energy sources
- ✅ **PV Analysis**: Production patterns, export policy detection, curtailment analysis
- ✅ **Thermal Analysis**: RC parameter estimation for 16 relay-controlled rooms
- ✅ **Relay Analysis**: Pattern recognition, coordination opportunities, optimization potential
- ✅ **Weather Correlation**: Comprehensive weather-energy correlation analysis
- ✅ **Feature Engineering**: 100+ engineered features for ML models including relay and price features
- ✅ **Visualization**: Interactive Plotly dashboards for all analysis types
- ✅ **Data Quality**: Validation and completeness checking throughout pipeline

### Analysis Capabilities:
- **Complete 2-year analysis** in 15-20 minutes with 10+ detailed reports
- **Interactive Jupyter notebooks** for detailed exploration (moved to `../pems_v2_analysis/`)
- **Energy optimization roadmap** with cost-benefit analysis and implementation phases
- **Executive summaries** with ROI calculations and strategic recommendations

## Key Features

- **ML-based Forecasting**: XGBoost and ensemble models for accurate predictions
- **Multi-objective Optimization**: Minimize costs while maximizing self-consumption
- **Model Predictive Control**: Rolling horizon optimization with real-time adaptation
- **Stochastic Optimization**: Handle uncertainty in predictions and prices
- **Thermal Modeling**: Physics-based models for accurate room temperature predictions
- **Relay Coordination**: Optimize 16 binary-controlled heating relays (18.12 kW total)
- **Export Policy Aware**: Handles conditional PV export based on price thresholds

## Architecture

```
pems_v2/                   # 📁 PEMS v2 Analysis Framework
├── analysis/              # 📊 Complete analysis pipeline [PHASE 1 COMPLETE]
│   ├── core/              # Core data processing
│   │   ├── data_extraction.py     # Async InfluxDB extraction
│   │   ├── data_preprocessing.py  # Data cleaning & standardization
│   │   └── visualization.py       # Advanced visualizations
│   ├── analyzers/         # Specialized analysis modules
│   │   ├── pattern_analysis.py    # PV & relay patterns
│   │   ├── thermal_analysis.py    # RC thermal modeling
│   │   ├── base_load_analysis.py  # Load forecasting
│   │   └── feature_engineering.py # ML feature generation
│   ├── pipelines/         # End-to-end analysis pipelines
│   │   └── comprehensive_analysis.py
│   ├── reports/           # Report generation
│   ├── utils/             # Analysis utilities
│   │   └── loxone_adapter.py      # Loxone field mapping
│   └── run_analysis.py    # 🚀 Main analysis entry point
├── config/               # ⚙️ Configuration management
│   ├── settings.py       # Main settings with Pydantic
│   └── energy_settings.py # Room power configurations
├── utils/                # 🔧 Shared utilities  
│   └── logging.py        # Logging configuration
└── tests/                # 🧪 Comprehensive test suite
    ├── test_basic_structure.py
    ├── test_data_extraction.py
    ├── test_relay_analysis.py
    └── test_new_extractors.py
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the system by setting environment variables or creating a `.env` file

3. Run complete energy analysis:
   ```bash
   cd pems_v2
   python analysis/run_analysis.py
   ```

## 📊 Running Analysis

For detailed instructions on running the complete 2-year analysis and getting all reports, see [README_ANALYSIS.md](README_ANALYSIS.md).

**Quick commands:**
- **Full 2-year analysis**: `python analysis/run_analysis.py` 
- **Interactive notebooks**: Open `../pems_v2_analysis/` folder
- **Test system**: `make test-basic && make test-extraction`

## Requirements

- Python 3.11+
- InfluxDB for time series data
- MQTT broker for device communication
- Access to weather forecast APIs
- Loxone home automation system

## Development

**Always use the virtual environment and Makefile for all development:**

```bash
# Set up development environment (first time only)
make setup

# Activate virtual environment (required for all development)
source venv/bin/activate

# Run tests
make test

# Code quality (runs black, isort, flake8)
make lint
```

See the main project documentation for detailed development guidelines.