Objective: Centralize Model and Thermal Configurations
The primary goal is to move all non-sensitive, model-related configurations and the newly required per-room thermal comfort settings from environment variables (.env) and hardcoded Python dictionaries into a single, structured JSON file: config/system_config.json. This will dramatically improve maintainability.

Phase 1: Establish the Central Configuration File
This phase is foundational. We will create the single source of truth for all system configurations that are not secret or environment-specific.

1. Create the New Configuration File:

Action: In the pems_v2/config/ directory, create a new file named system_config.json.

2. Define and Populate the JSON Structure:

Action: Copy the complete structure below into config/system_config.json. This file now contains all settings previously found in .env.example (except for secrets and server URLs) and adds the new per-room temperature settings.

{
  "system": {
    "simulation_mode": false,
    "advisory_mode": false,
    "optimization_interval_seconds": 3600,
    "control_interval_seconds": 300
  },
  "models": {
    "pv": {
      "model_path": "models/pv_predictor.pkl",
      "update_interval_seconds": 3600,
      "horizon_hours": 48,
      "confidence_levels": [0.1, 0.5, 0.9]
    },
    "load": {
      "model_path": "models/load_predictor.pkl",
      "horizon_hours": 24
    },
    "thermal": {
      "model_path": "models/thermal_predictor.pkl"
    }
  },
  "thermal_settings": {
    "comfort_band_celsius": 0.5,
    "room_setpoints": {
      "obyvak":    { "day": 21.5, "night": 20.0 },
      "kuchyne":   { "day": 21.0, "night": 19.5 },
      "loznice":   { "day": 20.5, "night": 19.0 },
      "pracovna":  { "day": 22.0, "night": 19.0 },
      "hosti":     { "day": 20.0, "night": 18.0 },
      "chodba_dole": { "day": 20.0, "night": 18.5 },
      "chodba_nahore": { "day": 20.0, "night": 18.5 },
      "koupelna_dole": { "day": 22.0, "night": 20.0 },
      "koupelna_nahore": { "day": 22.0, "night": 20.0 },
      "default":   { "day": 21.0, "night": 19.0 }
    }
  },
  "optimization": {
    "horizon_hours": 48,
    "control_hours": 24,
    "time_step_minutes": 60,
    "solver": "APOPT",
    "max_solve_time_seconds": 30,
    "weights": {
      "cost": 1.0,
      "self_consumption": 0.3,
      "peak_shaving": 0.1,
      "comfort": 0.5
    }
  },
  "battery": {
    "capacity_kwh": 10.0,
    "max_power_kw": 5.0,
    "efficiency": 0.95,
    "min_soc": 0.1,
    "max_soc": 0.9
  },
  "ev": {
    "max_power_kw": 11.0,
    "battery_capacity_kwh": 60.0,
    "default_target_soc": 0.8,
    "departure_time": "07:00"
  },
  "room_power_ratings_kw": {
    "obyvak": 4.8,
    "hosti": 2.02,
    "chodba_dole": 1.8,
    "chodba_nahore": 1.2,
    "loznice": 1.2,
    "pokoj_1": 1.2,
    "pokoj_2": 1.2,
    "pracovna": 0.82,
    "satna_dole": 0.82,
    "zadveri": 0.82,
    "technicka_mistnost": 0.82,
    "koupelna_nahore": 0.62,
    "satna_nahore": 0.56,
    "spajz": 0.46,
    "koupelna_dole": 0.47,
    "zachod": 0.22,
    "kuchyne": 1.8
  }
}

Rationale: We've also moved ROOM_CONFIG from energy_settings.py into this JSON as room_power_ratings_kw. This fully centralizes the physical properties of the house.

Phase 2: Refactor Pydantic Settings and Configuration Files
Now we'll update the Python code to read from our new central configuration file.

1. Refactor pems_v2/config/settings.py:

Action: Replace the entire content of this file with the code below. This new structure mirrors the JSON file, uses modern Pydantic features to load the JSON, and still allows for overrides via environment variables.

I will provide the refactored code for pems_v2/config/settings.py in a separate code block for clarity.

2. Refactor pems_v2/config/energy_settings.py:

Action: This file is now redundant. Its contents (ROOM_CONFIG, get_room_power) have been moved into the JSON file and the new settings.py.

Action: Delete the file pems_v2/config/energy_settings.py. We will create a new helper function later or access the power ratings directly from the settings object where needed.

Phase 3: Update Application Code to Use New Settings
This is the most detailed phase, involving changes across the application. The main principle is to pass the PEMSSettings object to any class that needs configuration, rather than passing a generic dictionary.

1. Refactor Data Extraction (analysis/core/data_extraction.py):

Problem: The DataExtractor currently hardcodes the ote_prices bucket name.

Action: Modify the extract_energy_prices method.

Before: from(bucket: "ote_prices")

After: from(bucket: "{self.settings.influxdb.bucket_prices}")
(Note: You'll need to add bucket_prices: str to InfluxDBSettings in settings.py as planned).

2. Refactor Model Predictors (models/predictors/*.py):

Problem: Predictors are initialized with a generic config dict.

Action: Change the __init__ signature and internal logic.

File: models/predictors/pv_predictor.py

Before: def __init__(self, config: Dict[str, Any]):

After: def __init__(self, pv_settings: PVPredictionSettings):

Logic Change: Replace self.config.get("capacity_kw") with pv_settings.capacity_kw.

Repeat this pattern for LoadPredictor and ThermalPredictor, passing them LoadModelSettings and ThermalModelSettings respectively.

3. Refactor Thermal Analysis (analysis/analyzers/thermal_analysis.py):

Problem: Needs to use the new per-room temperature setpoints.

Action: The ThermalAnalyzer's methods that deal with comfort or setpoints must now access settings.thermal_settings.room_setpoints.

Logic Example:

# Inside a method in ThermalAnalyzer
def get_target_temp(self, room_name: str, hour: int) -> float:
    setpoints = self.settings.thermal_settings.room_setpoints
    room_specific = setpoints.get(room_name, setpoints["default"])

    if 6 <= hour < 22: # Daytime
        return room_specific["day"]
    else: # Nighttime
        return room_specific["night"]

4. Refactor Control Modules (modules/control/*.py):

Problem: Controllers are hardcoding logic or using old config structures.

Action (HeatingController): Update any strategy logic (e.g., in _create_comfort_heating_schedule) to use the new get_target_temp logic shown above. It must be initialized with the main PEMSSettings object to have access to this.

Action (BatteryController): Its __init__ should take battery_settings: BatterySettings.

Before: self.capacity_kwh = config.get("battery", {}).get("capacity_kwh", 10.0)

After: self.capacity_kwh = battery_settings.capacity_kwh

5. Refactor Main Analysis Runner (analysis/run_analysis.py):

Problem: The script instantiates analyzers and passes them generic configs.

Action:

The run_analysis function should be the single point of PEMSSettings instantiation.

settings = PEMSSettings()

Pass the specific sub-settings to each component.

Example: analyzer = ComprehensiveAnalyzer(settings)

Inside ComprehensiveAnalyzer, when it creates ThermalAnalyzer, it will pass settings to it.

Phase 4: Update Documentation & Supporting Files
1. Update .env.example:

Action: This file will be significantly cleaned up. Remove all variables that were moved to system_config.json.

Result: The new .env.example should only contain secrets and server addresses.

Before (.env.example):

INFLUXDB_URL=...
INFLUXDB_TOKEN=...
MQTT_BROKER=...
PV_MODEL_PATH=...
THERMAL_DEFAULT_SETPOINT_DAY=...
OPT_COST_WEIGHT=...
BATTERY_CAPACITY_KWH=...
# ... and many more

After (.env.example):

# Environment-specific settings and secrets for PEMS v2

# InfluxDB Connection (REQUIRED)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_secret_token_here
INFLUXDB_ORG=your_org

# MQTT Connection (REQUIRED)
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=

2. Update README.md:

Action: Add a new section called "System Configuration" right after "Prerequisites".

Content Snippet for README.md:

Configuration
PEMS v2 uses a tiered configuration system:

config/system_config.json: This is the primary file for configuring the physical properties of your home, model parameters, and optimization settings. You should edit this file first. It includes per-room heating setpoints, battery capacity, PV model paths, etc.

.env file: This file is for secrets and environment-specific server addresses. Create it by copying .env.example. It should contain your INFLUXDB_TOKEN, INFLUXDB_URL, and MQTT_BROKER address.

Environment Variables: Any setting can be overridden by setting an environment variable (e.g., PEMS_SYSTEM__SIMULATION_MODE=true).

3. Update Tests:

Problem: Tests that rely on the old settings structure will fail.

Strategy: Use pytest fixtures and monkeypatch to create a temporary system_config.json for each test run. This ensures tests are isolated and don't depend on a real file.

Example Test Fixture (tests/conftest.py):

import json
import pytest

@pytest.fixture
def mock_system_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "system_config.json"

    # Create a minimal, valid config for testing
    test_config = {"system": {"simulation_mode": True}, ...} 

    with open(config_file, 'w') as f:
        json.dump(test_config, f)

    return config_file

# In your test function:
def test_something(mock_system_config, monkeypatch):
    # Temporarily point the settings loader to our test config
    monkeypatch.setenv("PEMS_CONFIG_PATH", str(mock_system_config))
    from pems_v2.config.settings import PEMSSettings
    settings = PEMSSettings()
    assert settings.system.simulation_mode is True

This detailed plan provides a clear, actionable roadmap. The next logical step is to generate the refactored code for pems_v2/config/settings.py.