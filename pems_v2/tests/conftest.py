"""
Pytest configuration and fixtures for PEMS v2 tests.

This module provides shared test fixtures and configuration for the PEMS v2 test suite,
including mock configuration files and test-safe environments.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture
def mock_system_config(tmp_path) -> Path:
    """
    Create a temporary system_config.json file for testing.

    This fixture creates a minimal, valid configuration file that can be used
    in tests without requiring the actual system configuration file. It includes
    all required fields with test-safe values.

    Args:
        tmp_path: pytest temporary path fixture

    Returns:
        Path to the temporary system_config.json file
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "system_config.json"

    # Create a minimal, valid config for testing
    test_config = {
        "system": {
            "simulation_mode": True,  # Always simulate in tests
            "advisory_mode": False,
            "optimization_interval_seconds": 1800,  # 30 minutes for faster testing
            "control_interval_seconds": 300,
        },
        "models": {
            "pv": {
                "model_path": "models/test_pv_predictor.pkl",
                "update_interval_seconds": 300,  # Minimum allowed value
                "horizon_hours": 24,  # Shorter for faster tests
                "confidence_levels": [0.1, 0.5, 0.9],
            },
            "load": {
                "model_path": "models/test_load_predictor.pkl",
                "horizon_hours": 24,
            },
            "thermal": {"model_path": "models/test_thermal_predictor.pkl"},
        },
        "thermal_settings": {
            "comfort_band_celsius": 1.0,  # Wider band for testing
            "room_setpoints": {
                "test_room": {"day": 20.0, "night": 18.0},
                "obyvak": {"day": 21.0, "night": 19.0},
                "default": {"day": 20.0, "night": 18.0},
            },
        },
        "optimization": {
            "horizon_hours": 24,  # Shorter for faster testing
            "control_hours": 12,
            "time_step_minutes": 60,
            "solver": "APOPT",
            "max_solve_time_seconds": 10,  # Shorter for tests
            "weights": {
                "cost": 1.0,
                "self_consumption": 0.5,
                "peak_shaving": 0.0,  # Disabled for simpler test cases
                "comfort": 0.5,
            },
        },
        "battery": {
            "capacity_kwh": 5.0,  # Small test battery
            "max_power_kw": 2.5,  # C-rate = 0.5, well within limits
            "efficiency": 0.9,
            "min_soc": 0.2,
            "max_soc": 0.8,
        },
        "ev": {
            "max_power_kw": 7.4,  # Single-phase charger
            "battery_capacity_kwh": 40.0,  # Smaller EV for testing
            "default_target_soc": 0.7,
            "departure_time": "08:00",
        },
        "room_power_ratings_kw": {
            "test_room": 1.0,
            "obyvak": 2.0,
            "kuchyne": 1.5,
            "loznice": 1.0,
            "default": 1.0,
        },
    }

    with open(config_file, "w") as f:
        json.dump(test_config, f, indent=2)

    return config_file


@pytest.fixture
def mock_env_config(monkeypatch, tmp_path) -> Dict[str, str]:
    """
    Set up test environment variables for PEMS configuration.

    This fixture provides test-safe environment variables that can be used
    for testing without requiring actual database credentials or server connections.

    Args:
        monkeypatch: pytest monkeypatch fixture
        tmp_path: pytest temporary path fixture

    Returns:
        Dictionary of environment variables set for the test
    """
    env_vars = {
        "INFLUXDB_URL": "http://test-influxdb:8086",
        "INFLUXDB_TOKEN": "test_token_12345",
        "INFLUXDB_ORG": "test_org",
        "INFLUXDB_BUCKET_HISTORICAL": "test_loxone",
        "INFLUXDB_BUCKET_LOXONE": "test_loxone",
        "INFLUXDB_BUCKET_WEATHER": "test_weather_forecast",
        "INFLUXDB_BUCKET_SOLAR": "test_loxone",
        "INFLUXDB_BUCKET_PREDICTIONS": "test_predictions",
        "INFLUXDB_BUCKET_OPTIMIZATION": "test_optimization",
        "INFLUXDB_BUCKET_PRICES": "test_ote_prices",
        "MQTT_BROKER": "test-mqtt",
        "MQTT_PORT": "1883",
        "MQTT_CLIENT_ID": "pems_v2_test",
    }

    # Set environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def pems_test_settings(mock_system_config, mock_env_config, monkeypatch):
    """
    Create a complete PEMSSettings instance for testing.

    This fixture combines the mock system configuration with test environment
    variables to create a fully functional PEMSSettings instance that can be
    used in tests.

    Args:
        mock_system_config: Path to mock system_config.json
        mock_env_config: Test environment variables
        monkeypatch: pytest monkeypatch fixture

    Returns:
        PEMSSettings instance configured for testing
    """
    # Point the settings loader to our test config
    monkeypatch.setenv("PEMS_CONFIG_PATH", str(mock_system_config))

    # Import here to ensure environment variables are set
    from pems_v2.config.settings import PEMSSettings

    return PEMSSettings()


@pytest.fixture
def isolated_test_env(tmp_path, monkeypatch):
    """
    Create an isolated test environment with temporary directories.

    This fixture sets up temporary directories for data, models, and logs
    to ensure tests don't interfere with actual system files.

    Args:
        tmp_path: pytest temporary path fixture
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Dictionary with paths to temporary directories
    """
    # Create temporary directories
    test_dirs = {
        "data_dir": tmp_path / "data",
        "models_dir": tmp_path / "models",
        "logs_dir": tmp_path / "logs",
        "config_dir": tmp_path / "config",
    }

    for dir_path in test_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Set environment variables to point to test directories
    monkeypatch.setenv("PEMS_DATA_DIR", str(test_dirs["data_dir"]))
    monkeypatch.setenv("PEMS_MODELS_DIR", str(test_dirs["models_dir"]))
    monkeypatch.setenv("PEMS_LOGS_DIR", str(test_dirs["logs_dir"]))

    return test_dirs


@pytest.fixture(scope="session")
def test_data_samples():
    """
    Provide sample test data for use across multiple tests.

    This session-scoped fixture provides commonly used test data samples
    that can be reused across different test modules.

    Returns:
        Dictionary containing sample data structures
    """
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd

    # Create sample time series data
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(24)]

    sample_data = {
        "pv_data": pd.DataFrame(
            {
                "timestamp": timestamps,
                "power_w": np.random.normal(2000, 500, 24),
                "voltage_v": np.random.normal(240, 10, 24),
            }
        ).set_index("timestamp"),
        "weather_data": pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_2m": np.random.normal(15, 5, 24),
                "cloudcover": np.random.uniform(0, 100, 24),
                "shortwave_radiation": np.random.exponential(200, 24),
            }
        ).set_index("timestamp"),
        "room_data": {
            "test_room": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "temperature": np.random.normal(21, 1, 24),
                    "setpoint": np.full(24, 21.0),
                    "relay_state": np.random.choice([0, 1], 24),
                }
            ).set_index("timestamp")
        },
    }

    return sample_data
