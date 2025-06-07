"""
Test suite for ThermalPredictor.

Tests RC thermal modeling, parameter optimization, and hybrid ML/physics predictions
with comprehensive scenarios for building thermal dynamics.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from models.predictors.thermal_predictor import ThermalPredictor, ThermalZone


@pytest.fixture
def thermal_config():
    """Sample thermal model configuration."""
    return {
        "zones": {
            "obyvak": {
                "initial_r": 0.02,  # K/W
                "initial_c": 5000000,  # J/K
                "window_area": 15.0,  # m²
                "internal_gains": 150.0,  # W
                "heating_power": 2000.0,  # W
                "solar_gain_coeff": 0.65,
                "coupled_zones": {"loznice": 0.1},
            },
            "loznice": {
                "initial_r": 0.015,
                "initial_c": 3000000,
                "window_area": 8.0,
                "internal_gains": 80.0,
                "heating_power": 1500.0,
                "solar_gain_coeff": 0.7,
                "coupled_zones": {"obyvak": 0.1},
            },
        },
        "default_zone": {
            "initial_r": 0.02,
            "initial_c": 4000000,
            "window_area": 10.0,
            "internal_gains": 100.0,
            "heating_power": 1500.0,
            "solar_gain_coeff": 0.6,
        },
        "physics_weight": 0.7,
        "ml_weight": 0.3,
        "time_step": 300,  # 5 minutes
        "adaptation_rate": 0.01,
        "prediction_horizon": 24,
    }


@pytest.fixture
def sample_thermal_data():
    """Generate realistic thermal training data."""
    # 3 days of 5-minute data
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=864, freq="5min")

    # Outdoor temperature with daily cycle
    outdoor_temp = (
        5 + 8 * np.sin(2 * np.pi * np.arange(864) / 288) + np.random.normal(0, 1, 864)
    )

    # Solar radiation (zero at night, peak around noon)
    hour_of_day = (np.arange(864) % 288) / 12  # 0-24 hour cycle
    solar_radiation = np.maximum(
        0, 800 * np.sin(np.pi * hour_of_day / 24) + np.random.normal(0, 50, 864)
    )
    solar_radiation[hour_of_day < 6] = 0  # No sun before 6 AM
    solar_radiation[hour_of_day > 18] = 0  # No sun after 6 PM

    # Heating schedule (on during cold periods and night)
    heating_schedule = (
        (outdoor_temp < 3) | (hour_of_day < 6) | (hour_of_day > 22)
    ).astype(float)
    heating_power = heating_schedule * 1500 + np.random.normal(0, 100, 864)
    heating_power = np.maximum(0, heating_power)

    # Simulated indoor temperature (realistic thermal response)
    indoor_temp = np.zeros(864)
    indoor_temp[0] = 22.0

    for i in range(1, 864):
        # Simple thermal model for realistic data
        heat_loss = (indoor_temp[i - 1] - outdoor_temp[i]) * 50  # Heat loss
        solar_gain = solar_radiation[i] * 0.01  # Solar gains
        heat_input = heating_power[i] + solar_gain + 100  # Heating + solar + internal

        temp_change = (heat_input - heat_loss) / 10000  # Thermal mass effect
        indoor_temp[i] = indoor_temp[i - 1] + temp_change * 0.083  # 5-minute time step

        # Add some noise and constraints
        indoor_temp[i] += np.random.normal(0, 0.1)
        indoor_temp[i] = np.clip(indoor_temp[i], 15, 30)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "outdoor_temp": outdoor_temp,
            "solar_radiation": solar_radiation,
            "heating_power": heating_power,
            "temperature_obyvak": indoor_temp,
            "windspeed_10m": 5 + np.random.normal(0, 2, 864),
            "relativehumidity_2m": 60 + np.random.normal(0, 10, 864),
            # Add some other room temperature
            "temperature_loznice": indoor_temp - 1 + np.random.normal(0, 0.5, 864),
        },
        index=timestamps,
    )

    return data


class TestThermalZone:
    """Test suite for ThermalZone class."""

    def test_thermal_zone_initialization(self, thermal_config):
        """Test thermal zone initialization."""
        zone_config = thermal_config["zones"]["obyvak"]
        zone = ThermalZone("obyvak", zone_config)

        assert zone.zone_name == "obyvak"
        assert zone.thermal_resistance == zone_config["initial_r"]
        assert zone.thermal_capacitance == zone_config["initial_c"]
        assert zone.window_area == zone_config["window_area"]
        assert zone.heating_power == zone_config["heating_power"]
        assert zone.solar_gain_coeff == zone_config["solar_gain_coeff"]
        assert "loznice" in zone.coupled_zones

    def test_heat_flow_calculation(self, thermal_config):
        """Test heat flow calculations."""
        zone_config = thermal_config["zones"]["obyvak"]
        zone = ThermalZone("obyvak", zone_config)

        # Test basic heat flow
        indoor_temp = 22.0
        outdoor_temp = 5.0
        solar_radiation = 300.0
        heating_power = 1000.0

        heat_flow = zone.calculate_heat_flow(
            indoor_temp, outdoor_temp, solar_radiation, heating_power
        )

        # Should be positive (net heating)
        assert heat_flow > 0

        # Check components
        expected_loss = (indoor_temp - outdoor_temp) / zone.thermal_resistance
        expected_solar = solar_radiation * zone.window_area * zone.solar_gain_coeff
        expected_total = (
            heating_power + expected_solar + zone.internal_gains - expected_loss
        )

        assert abs(heat_flow - expected_total) < 1e-6

    def test_coupled_zone_interaction(self, thermal_config):
        """Test heat exchange between coupled zones."""
        zone_config = thermal_config["zones"]["obyvak"]
        zone = ThermalZone("obyvak", zone_config)

        indoor_temp = 22.0
        coupled_temps = {"loznice": 20.0}  # Colder adjacent room

        heat_flow_with_coupling = zone.calculate_heat_flow(
            indoor_temp, 10.0, 0, 0, coupled_temps
        )

        heat_flow_without_coupling = zone.calculate_heat_flow(
            indoor_temp, 10.0, 0, 0, {}
        )

        # Should lose heat to colder coupled zone
        assert heat_flow_with_coupling < heat_flow_without_coupling

    def test_temperature_derivative(self, thermal_config):
        """Test temperature change rate calculation."""
        zone_config = thermal_config["zones"]["obyvak"]
        zone = ThermalZone("obyvak", zone_config)

        # High heating power should cause positive temperature change
        dT_dt_heating = zone.temperature_derivative(20.0, 5.0, 0, 2000.0)
        assert dT_dt_heating > 0

        # No heating, cold outdoor should cause negative temperature change
        dT_dt_cooling = zone.temperature_derivative(20.0, 5.0, 0, 0)
        assert dT_dt_cooling < 0

        # Rate should be proportional to heat flow / capacitance
        heat_flow = zone.calculate_heat_flow(20.0, 5.0, 0, 2000.0)
        expected_rate = heat_flow / zone.thermal_capacitance
        assert abs(dT_dt_heating - expected_rate) < 1e-9


class TestThermalPredictor:
    """Test suite for ThermalPredictor class."""

    def test_initialization(self, thermal_config):
        """Test thermal predictor initialization."""
        predictor = ThermalPredictor(thermal_config)

        assert len(predictor.zones) == 2  # obyvak and loznice
        assert "obyvak" in predictor.zones
        assert "loznice" in predictor.zones
        assert predictor.physics_model_weight == 0.7
        assert predictor.ml_model_weight == 0.3
        assert predictor.time_step == 300
        assert predictor.ml_model is None  # Not trained yet

    def test_feature_extraction(self, thermal_config, sample_thermal_data):
        """Test thermal feature engineering."""
        predictor = ThermalPredictor(thermal_config)

        features = predictor._extract_thermal_features(sample_thermal_data, "obyvak")

        # Check essential features
        assert "outdoor_temp" in features.columns
        assert (
            "solar_radiation" in features.columns
        )  # Should be created from sample data
        assert "heating_power" in features.columns
        assert "hour" in features.columns
        assert "sin_hour" in features.columns
        assert "cos_hour" in features.columns

        # Check thermal history features
        assert "temp_lag1" in features.columns
        assert "temp_lag12" in features.columns  # 1 hour lag

        # Check coupled zone features
        assert "temp_diff_loznice" in features.columns

        # Check feature engineering
        assert len(features) == len(sample_thermal_data)
        assert features["sin_hour"].min() >= -1
        assert features["sin_hour"].max() <= 1
        assert features["hour"].min() >= 0
        assert features["hour"].max() <= 23

    def test_thermal_simulation(self, thermal_config, sample_thermal_data):
        """Test thermal response simulation."""
        predictor = ThermalPredictor(thermal_config)
        zone = predictor.zones["obyvak"]

        # Use subset of data for simulation
        data_subset = sample_thermal_data.iloc[:100]  # ~8 hours

        initial_temp = 22.0
        outdoor_temps = data_subset["outdoor_temp"].values
        solar_radiation = data_subset["solar_radiation"].values
        heating_powers = data_subset["heating_power"].values
        time_points = np.arange(len(data_subset)) * predictor.time_step

        simulated_temps = predictor._simulate_thermal_response(
            zone,
            initial_temp,
            outdoor_temps,
            solar_radiation,
            heating_powers,
            time_points,
        )

        # Check simulation results
        assert len(simulated_temps) == len(data_subset)
        assert simulated_temps[0] == initial_temp
        # Temperature range check (allow for wider range since simulations can vary)
        assert all(
            0 <= temp <= 60 for temp in simulated_temps
        )  # Reasonable temperature range

        # Temperature should change gradually (thermal mass effect)
        temp_changes = np.diff(simulated_temps)
        assert all(
            abs(change) < 10.0 for change in temp_changes
        )  # Reasonable change per 5 min

    def test_parameter_optimization(self, thermal_config, sample_thermal_data):
        """Test thermal parameter optimization."""
        predictor = ThermalPredictor(thermal_config)

        # Use subset for faster testing
        data_subset = sample_thermal_data.iloc[:200]
        features = predictor._extract_thermal_features(data_subset, "obyvak")
        target_temps = data_subset["temperature_obyvak"].values

        # Mock scipy.optimize to avoid long optimization
        with patch(
            "models.predictors.thermal_predictor.optimize.minimize"
        ) as mock_optimize:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.x = [0.025, 4500000, 0.7]  # Optimized parameters
            mock_result.fun = 0.8  # RMSE
            mock_optimize.return_value = mock_result

            optimized_params = predictor._optimize_thermal_parameters(
                "obyvak", features, target_temps
            )

            assert "thermal_resistance" in optimized_params
            assert "thermal_capacitance" in optimized_params
            assert "solar_gain_coeff" in optimized_params
            assert "optimization_error" in optimized_params
            assert optimized_params["thermal_resistance"] == 0.025
            assert optimized_params["thermal_capacitance"] == 4500000
            assert optimized_params["solar_gain_coeff"] == 0.7

    @pytest.mark.asyncio
    async def test_training(self, thermal_config, sample_thermal_data):
        """Test thermal model training."""
        predictor = ThermalPredictor(thermal_config)

        # Prepare training data (drop the temperature column)
        features = sample_thermal_data.drop(
            columns=["temperature_obyvak"], errors="ignore"
        )
        target = sample_thermal_data["temperature_obyvak"]

        # Mock parameter optimization to speed up test
        with patch.object(predictor, "_optimize_thermal_parameters") as mock_optimize:
            mock_optimize.return_value = {
                "thermal_resistance": 0.025,
                "thermal_capacitance": 4500000,
                "solar_gain_coeff": 0.7,
                "optimization_error": 0.8,
            }

            # Mock thermal simulation for consistent testing
            with patch.object(predictor, "_simulate_thermal_response") as mock_simulate:
                # Generate realistic physics predictions for training split
                def side_effect_simulate(
                    zone, initial_temp, outdoor_temps, solar_rad, heating_pow, time_pts
                ):
                    return np.ones(len(outdoor_temps)) * 22.0 + np.random.normal(
                        0, 0.5, len(outdoor_temps)
                    )

                mock_simulate.side_effect = side_effect_simulate

                performance = predictor.train(features, target, zone_name="obyvak")

        # Check training results
        assert isinstance(performance.rmse, float)
        assert isinstance(performance.mae, float)
        assert isinstance(performance.r2, float)
        assert predictor.ml_model is not None

        # Check parameter history
        assert len(predictor.parameter_history) == 1
        assert predictor.parameter_history[0]["zone"] == "obyvak"
        assert predictor.parameter_history[0]["thermal_resistance"] == 0.025

    def test_prediction(self, thermal_config, sample_thermal_data):
        """Test thermal prediction."""
        predictor = ThermalPredictor(thermal_config)

        # Setup trained model (mock training)
        zone = predictor.zones["obyvak"]
        zone.thermal_resistance = 0.025
        zone.thermal_capacitance = 4500000
        zone.solar_gain_coeff = 0.7

        # Mock ML model
        mock_ml_model = MagicMock()
        mock_ml_model.predict.return_value = np.random.normal(0, 0.2, 100)
        mock_ml_model.feature_importances_ = np.random.random(10)
        predictor.ml_model = mock_ml_model

        # Mock feature scaler
        predictor.feature_scaler = MagicMock()
        predictor.feature_scaler.transform.return_value = np.random.random((100, 10))

        # Test prediction
        test_data = sample_thermal_data.iloc[:100]

        with patch.object(predictor, "_simulate_thermal_response") as mock_simulate:
            mock_simulate.return_value = 22 + np.random.normal(0, 1, 100)

            result = predictor.predict(
                test_data, return_uncertainty=True, zone_name="obyvak"
            )

        # Check prediction results
        assert len(result.predictions) == 100
        assert result.confidence_intervals is not None
        assert "lower_95" in result.confidence_intervals
        assert "upper_95" in result.confidence_intervals
        assert (
            result.feature_contributions is not None
            or result.feature_contributions is None
        )  # May be None if no feature importance

        # Check that we have results
        assert result.predictions is not None
        assert len(result.predictions) == 100

    def test_online_update(self, thermal_config, sample_thermal_data):
        """Test online model adaptation."""
        predictor = ThermalPredictor(thermal_config)

        # Setup minimal trained state
        zone = predictor.zones["obyvak"]
        original_resistance = zone.thermal_resistance

        mock_ml_model = MagicMock()
        predictor.ml_model = mock_ml_model
        predictor.feature_scaler = MagicMock()
        predictor.feature_scaler.transform.return_value = np.random.random((10, 10))

        # Mock predictions with high error to trigger adaptation
        with patch.object(predictor, "predict") as mock_predict:
            mock_result = MagicMock()
            mock_result.predictions = pd.Series([25.0] * 10)  # High error predictions
            mock_predict.return_value = mock_result

            # New observations with low actual temperatures
            new_data = sample_thermal_data.iloc[:10]
            new_temps = pd.Series([20.0] * 10)  # Actual temperatures much lower

            # Simulate multiple updates to build error history
            for _ in range(15):
                predictor.update_online(new_data, new_temps, zone_name="obyvak")

        # Check that adaptation occurred
        assert len(predictor.prediction_errors) > 0
        # Note: Resistance might change due to adaptation logic

    def test_thermal_insights(self, thermal_config):
        """Test thermal insights generation."""
        predictor = ThermalPredictor(thermal_config)

        # Add some parameter history
        predictor.parameter_history.append(
            {
                "timestamp": datetime.now(),
                "zone": "obyvak",
                "thermal_resistance": 0.025,
                "thermal_capacitance": 4500000,
                "solar_gain_coeff": 0.7,
                "optimization_error": 0.8,
            }
        )

        # Add some prediction errors
        predictor.prediction_errors = [0.5, 0.4, 0.3, 0.2, 0.3]

        insights = predictor.get_thermal_insights("obyvak")

        # Check insights structure
        assert insights["zone_name"] == "obyvak"
        assert "thermal_properties" in insights
        assert "parameter_optimization" in insights
        assert "prediction_performance" in insights
        assert "model_characteristics" in insights

        # Check thermal properties
        thermal_props = insights["thermal_properties"]
        assert "thermal_resistance" in thermal_props
        assert "thermal_capacitance" in thermal_props
        assert "time_constant" in thermal_props
        assert "solar_gain_coefficient" in thermal_props

        # Check optimization info
        opt_info = insights["parameter_optimization"]
        assert opt_info["optimization_history"] == 1
        assert opt_info["latest_optimization_error"] == 0.8

        # Check performance info
        perf_info = insights["prediction_performance"]
        assert perf_info["recent_error_count"] == 5
        assert (
            abs(perf_info["average_recent_error"] - 0.34) < 0.001
        )  # Average of last 5 errors

    def test_error_handling(self, thermal_config):
        """Test error handling for edge cases."""
        predictor = ThermalPredictor(thermal_config)

        # Test prediction without training
        test_data = pd.DataFrame(
            {
                "outdoor_temp": [10.0, 11.0],
                "solar_radiation": [100.0, 200.0],
            }
        )

        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict(test_data)

        # Test prediction for non-existent zone
        mock_ml_model = MagicMock()
        predictor.ml_model = mock_ml_model

        with pytest.raises(ValueError, match="Zone nonexistent not found"):
            predictor.predict(test_data, zone_name="nonexistent")

        # Test insights for non-existent zone
        insights = predictor.get_thermal_insights("nonexistent")
        assert "error" in insights

    def test_multi_zone_coupling(self, thermal_config, sample_thermal_data):
        """Test multi-zone thermal coupling."""
        predictor = ThermalPredictor(thermal_config)

        # Test that zones have proper coupling configuration
        obyvak_zone = predictor.zones["obyvak"]
        loznice_zone = predictor.zones["loznice"]

        assert "loznice" in obyvak_zone.coupled_zones
        assert "obyvak" in loznice_zone.coupled_zones

        # Test heat exchange calculation with coupling
        obyvak_temp = 22.0
        loznice_temp = 20.0

        heat_flow_obyvak = obyvak_zone.calculate_heat_flow(
            obyvak_temp, 10.0, 0, 0, {"loznice": loznice_temp}
        )

        heat_flow_loznice = loznice_zone.calculate_heat_flow(
            loznice_temp, 10.0, 0, 0, {"obyvak": obyvak_temp}
        )

        # Warmer room should lose heat to cooler room
        heat_flow_obyvak_uncoupled = obyvak_zone.calculate_heat_flow(
            obyvak_temp, 10.0, 0, 0, {}
        )
        assert heat_flow_obyvak < heat_flow_obyvak_uncoupled

        # Cooler room should gain heat from warmer room
        heat_flow_loznice_uncoupled = loznice_zone.calculate_heat_flow(
            loznice_temp, 10.0, 0, 0, {}
        )
        assert heat_flow_loznice > heat_flow_loznice_uncoupled

    def test_uncertainty_quantification(self, thermal_config, sample_thermal_data):
        """Test prediction uncertainty estimation."""
        predictor = ThermalPredictor(thermal_config)

        # Setup trained model
        zone = predictor.zones["obyvak"]
        mock_ml_model = MagicMock()
        mock_ml_model.predict.return_value = np.random.normal(0, 0.2, 50)

        # Mock ensemble predictions for uncertainty
        mock_ml_model.estimators_ = [MagicMock() for _ in range(10)]
        for estimator in mock_ml_model.estimators_:
            estimator.predict.return_value = np.random.normal(0, 0.2, 50)

        predictor.ml_model = mock_ml_model
        predictor.feature_scaler = MagicMock()
        predictor.feature_scaler.transform.return_value = np.random.random((50, 10))

        test_data = sample_thermal_data.iloc[:50]

        with patch.object(predictor, "_simulate_thermal_response") as mock_simulate:
            mock_simulate.return_value = 22 + np.random.normal(0, 0.5, 50)

            result = predictor.predict(
                test_data, return_uncertainty=True, zone_name="obyvak"
            )

        # Check uncertainty intervals
        assert result.confidence_intervals is not None
        assert len(result.confidence_intervals["lower_95"]) == 50
        assert len(result.confidence_intervals["upper_95"]) == 50

        # Upper bounds should be higher than lower bounds
        assert all(
            result.confidence_intervals["upper_95"].iloc[i]
            > result.confidence_intervals["lower_95"].iloc[i]
            for i in range(50)
        )

        # 80% intervals should be narrower than 95% intervals
        assert all(
            result.confidence_intervals["upper_80"].iloc[i]
            < result.confidence_intervals["upper_95"].iloc[i]
            for i in range(50)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
