"""
Test the new configuration system with JSON-based settings.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import (
    PEMSSettings, 
    SystemSettings, 
    BatterySettings, 
    ThermalSettings,
    RoomSetpoint
)


def test_pems_settings_with_mock_config(pems_test_settings):
    """Test that PEMSSettings loads correctly with mock configuration."""
    
    settings = pems_test_settings
    
    # Verify settings loaded correctly
    assert settings is not None
    assert settings.system.simulation_mode is True  # Test environment
    assert settings.battery.capacity_kwh == 5.0  # Test battery size
    
    # Test room power lookup
    assert settings.get_room_power("test_room") == 1.0
    assert settings.get_room_power("nonexistent") == 0.0
    
    # Test thermal settings
    temp = settings.thermal_settings.get_target_temp("test_room", 14)  # Daytime
    assert temp == 20.0
    
    temp = settings.thermal_settings.get_target_temp("test_room", 2)  # Nighttime
    assert temp == 18.0


def test_room_setpoint_validation():
    """Test RoomSetpoint validation."""
    
    # Valid setpoint
    setpoint = RoomSetpoint(day=21.0, night=19.0)
    assert setpoint.day == 21.0
    assert setpoint.night == 19.0
    
    # Invalid: night > day
    with pytest.raises(ValidationError):
        RoomSetpoint(day=18.0, night=22.0)
    
    # Invalid: temperature out of range
    with pytest.raises(ValidationError):
        RoomSetpoint(day=50.0, night=19.0)


def test_battery_settings_validation():
    """Test BatterySettings validation."""
    
    # Valid settings
    battery = BatterySettings(
        capacity_kwh=10.0,
        max_power_kw=5.0,
        efficiency=0.95,
        min_soc=0.1,
        max_soc=0.9
    )
    assert battery.capacity_kwh == 10.0
    
    # Invalid: min_soc >= max_soc
    with pytest.raises(ValidationError):
        BatterySettings(
            capacity_kwh=10.0,
            max_power_kw=5.0,
            efficiency=0.95,
            min_soc=0.9,
            max_soc=0.1
        )
    
    # Invalid: efficiency > 1.0
    with pytest.raises(ValidationError):
        BatterySettings(
            capacity_kwh=10.0,
            max_power_kw=5.0,
            efficiency=1.5,  # >100%
            min_soc=0.1,
            max_soc=0.9
        )
    
    # Invalid: C-rate too high
    with pytest.raises(ValidationError):
        BatterySettings(
            capacity_kwh=1.0,
            max_power_kw=10.0,  # 10C rate
            efficiency=0.95,
            min_soc=0.1,
            max_soc=0.9
        )


def test_system_settings_validation():
    """Test SystemSettings validation."""
    
    # Valid settings
    system = SystemSettings(
        simulation_mode=True,
        advisory_mode=False,
        optimization_interval_seconds=3600,
        control_interval_seconds=300
    )
    assert system.optimization_interval_seconds == 3600
    
    # Invalid: control > optimization interval
    with pytest.raises(ValidationError):
        SystemSettings(
            simulation_mode=True,
            advisory_mode=False,
            optimization_interval_seconds=300,
            control_interval_seconds=3600  # Longer than optimization
        )
    
    # Invalid: control interval too short
    with pytest.raises(ValidationError):
        SystemSettings(
            simulation_mode=True,
            advisory_mode=False,
            optimization_interval_seconds=3600,
            control_interval_seconds=30  # Less than 60 seconds
        )


def test_environment_variable_override(mock_system_config, monkeypatch):
    """Test that environment variables can override JSON settings."""
    
    # Set environment override
    monkeypatch.setenv("PEMS_CONFIG_PATH", str(mock_system_config))
    monkeypatch.setenv("PEMS_SYSTEM__SIMULATION_MODE", "false")
    monkeypatch.setenv("PEMS_BATTERY__CAPACITY_KWH", "15.0")
    
    # Load settings
    settings = PEMSSettings()
    
    # Verify overrides took effect
    assert settings.system.simulation_mode is False  # Overridden from True
    assert settings.battery.capacity_kwh == 15.0  # Overridden from 5.0


def test_thermal_settings_get_target_temp():
    """Test ThermalSettings.get_target_temp method."""
    
    # Create test thermal settings
    from config.settings import ThermalSettings, RoomSetpoint
    
    thermal_settings = ThermalSettings(
        comfort_band_celsius=0.5,
        room_setpoints={
            "test_room": RoomSetpoint(day=22.0, night=18.0),
            "default": RoomSetpoint(day=21.0, night=19.0)
        }
    )
    
    # Test daytime temperature (6-22)
    assert thermal_settings.get_target_temp("test_room", 14) == 22.0
    assert thermal_settings.get_target_temp("unknown_room", 14) == 21.0  # Uses default
    
    # Test nighttime temperature (22-6)
    assert thermal_settings.get_target_temp("test_room", 2) == 18.0
    assert thermal_settings.get_target_temp("unknown_room", 2) == 19.0  # Uses default


def test_config_file_not_found():
    """Test behavior when config file is missing."""
    
    with pytest.raises(FileNotFoundError):
        PEMSSettings(config_path="/nonexistent/path/config.json")


def test_invalid_json_config(tmp_path):
    """Test behavior with invalid JSON configuration."""
    
    # Create invalid JSON file
    config_file = tmp_path / "invalid_config.json"
    with open(config_file, 'w') as f:
        f.write('{"invalid": json}')  # Invalid JSON
    
    with pytest.raises(json.JSONDecodeError):
        PEMSSettings(config_path=str(config_file))


def test_missing_required_fields(tmp_path):
    """Test behavior with missing required configuration fields."""
    
    # Create config missing required fields
    config_file = tmp_path / "incomplete_config.json"
    incomplete_config = {
        "system": {
            "simulation_mode": True
            # Missing required fields
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(incomplete_config, f)
    
    with pytest.raises(ValidationError):
        PEMSSettings(config_path=str(config_file))