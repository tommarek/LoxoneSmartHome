# Predictive Energy Management System (PEMS) v2

PEMS v2 is an advanced machine learning-based energy management system that optimizes heating, battery storage, and EV charging to minimize energy costs while maintaining comfort.

## Overview

PEMS replaces the simple rule-based Growatt controller with a sophisticated predictive optimization system that:

- **Predicts** PV production, energy consumption, and thermal dynamics using ML models
- **Optimizes** energy flows across a 48-hour horizon considering electricity prices
- **Controls** heating systems, battery storage, and EV charging in real-time
- **Learns** from historical data to continuously improve predictions

## Key Features

- **ML-based Forecasting**: XGBoost and ensemble models for accurate predictions
- **Multi-objective Optimization**: Minimize costs while maximizing self-consumption
- **Model Predictive Control**: Rolling horizon optimization with real-time adaptation
- **Stochastic Optimization**: Handle uncertainty in predictions and prices
- **Thermal Modeling**: Physics-based models for accurate room temperature predictions

## Architecture

```
pems_v2/
├── analysis/          # Data extraction and pattern analysis
├── models/            # ML model implementations
├── modules/           # Core functionality
│   ├── predictors/    # PV, load, and thermal predictors
│   ├── optimization/  # Optimization engine and constraints
│   └── control/       # Device control interfaces
├── config/            # Configuration management
├── utils/             # Shared utilities
└── tests/             # Test suite
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