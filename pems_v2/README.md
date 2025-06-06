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

### Analysis Notebooks:
- `02_pv_production_analysis.ipynb`: PV system analysis with export policy impact
- `06_weather_correlation.ipynb`: Weather-energy correlations and patterns
- `07_feature_selection.ipynb`: Advanced feature engineering for ML models

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
pems_v2/
├── analysis/              # Data extraction and pattern analysis [PHASE 1 COMPLETE]
│   ├── data_extraction.py # Async InfluxDB data extraction
│   ├── pattern_analysis.py # PV export policy detection
│   ├── thermal_analysis.py # RC parameter estimation (16 rooms)
│   ├── feature_engineering.py # Relay & price feature generation
│   ├── visualization.py   # Comprehensive plotting functions
│   └── notebooks/         # Jupyter analysis notebooks
├── models/               # ML model implementations [PHASE 2]
├── modules/              # Core functionality
│   ├── predictors/       # PV, load, and thermal predictors [PHASE 2]
│   ├── optimization/     # Optimization engine and constraints [PHASE 2]
│   └── control/          # Device control interfaces [PHASE 3]
├── config/               # Configuration management
├── utils/                # Shared utilities
└── tests/                # Test suite
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the system by setting environment variables or creating a `.env` file

3. Run the main controller:
   ```bash
   python main.py
   ```

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