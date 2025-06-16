"""
Thermal Predictor for PEMS v2.

Advanced room temperature prediction using RC thermal models combined with ML.
Implements physical building thermal dynamics with data-driven enhancements.

Key Features:
- RC thermal circuit modeling (R-C networks)
- Heat transfer coefficient estimation
- Solar gain and internal heat modeling
- Adaptive model parameters with online learning
- Uncertainty quantification for control applications
- Multi-room thermal coupling
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from pems_v2.config.settings import ThermalModelSettings

from ..base import BasePredictor, PerformanceMetrics, PredictionResult


class ThermalZone:
    """
    Represents a single thermal zone with RC circuit parameters.
    """

    def __init__(self, zone_name: str, config: Dict[str, Any]):
        """Initialize thermal zone."""
        self.zone_name = zone_name
        self.config = config

        # RC thermal parameters (to be estimated)
        self.thermal_resistance = config.get("initial_r", 0.01)  # K/W
        self.thermal_capacitance = config.get("initial_c", 3600000)  # J/K
        self.window_area = config.get("window_area", 10.0)  # m²
        self.internal_gains = config.get("internal_gains", 100.0)  # W
        self.heating_power = config.get("heating_power", 1500.0)  # W

        # Solar gain coefficient
        self.solar_gain_coeff = config.get("solar_gain_coeff", 0.7)

        # Coupling to other zones
        self.coupled_zones: Dict[str, float] = config.get("coupled_zones", {})

    def calculate_heat_flow(
        self,
        indoor_temp: float,
        outdoor_temp: float,
        solar_radiation: float,
        heating_power: float,
        coupled_temps: Dict[str, float] = None,
    ) -> float:
        """
        Calculate total heat flow into the zone.

        Args:
            indoor_temp: Indoor temperature (°C)
            outdoor_temp: Outdoor temperature (°C)
            solar_radiation: Solar radiation (W/m²)
            heating_power: Active heating power (W)
            coupled_temps: Temperatures of coupled zones

        Returns:
            Total heat flow rate (W)
        """
        if coupled_temps is None:
            coupled_temps = {}

        # Heat loss to outdoor through envelope
        envelope_loss = (indoor_temp - outdoor_temp) / self.thermal_resistance

        # Solar gains through windows
        solar_gains = solar_radiation * self.window_area * self.solar_gain_coeff

        # Heat exchange with coupled zones
        zone_exchange = 0.0
        for zone_name, coupling_coeff in self.coupled_zones.items():
            if zone_name in coupled_temps:
                zone_exchange += coupling_coeff * (
                    coupled_temps[zone_name] - indoor_temp
                )

        # Total heat flow (positive = heating)
        total_heat_flow = (
            heating_power
            + solar_gains
            + self.internal_gains
            - envelope_loss
            + zone_exchange
        )

        return total_heat_flow

    def temperature_derivative(
        self,
        indoor_temp: float,
        outdoor_temp: float,
        solar_radiation: float,
        heating_power: float,
        coupled_temps: Dict[str, float] = None,
    ) -> float:
        """
        Calculate temperature change rate (dT/dt).

        Returns:
            Temperature derivative (K/s)
        """
        heat_flow = self.calculate_heat_flow(
            indoor_temp, outdoor_temp, solar_radiation, heating_power, coupled_temps
        )
        return heat_flow / self.thermal_capacitance


class ThermalPredictor(BasePredictor):
    """
    Advanced thermal predictor using RC models + ML enhancement.

    Combines physical thermal modeling with machine learning for accurate
    temperature prediction and heating demand forecasting.
    """

    def __init__(
        self,
        thermal_settings: ThermalModelSettings,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize thermal predictor with configuration.

        Args:
            thermal_settings: Thermal model configuration from system settings
            config: Optional additional configuration for model-specific parameters
        """
        # Merge settings with additional config
        merged_config = {
            "model_path": thermal_settings.model_path,
        }
        if config:
            merged_config.update(config)

        super().__init__(merged_config)
        self.thermal_settings = thermal_settings
        self.logger = logging.getLogger(f"{__name__}.ThermalPredictor")

        # Model configuration
        self.zones: Dict[str, ThermalZone] = {}
        self.ml_model: Optional[RandomForestRegressor] = None
        self.physics_model = None
        self.thermal_parameters = None
        self.zone_capacitance = None
        self.zone_resistance = None
        self.physics_model_weight = config.get("physics_weight", 0.6)
        self.ml_model_weight = config.get("ml_weight", 0.4)
        self.prediction_horizon = config.get("prediction_horizon", 24)  # hours

        # Feature preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Model parameters
        self.time_step = config.get("time_step", 300)  # seconds (5 minutes)
        self.adaptation_rate = config.get("adaptation_rate", 0.01)

        # Initialize thermal zones from config
        self._initialize_thermal_zones(config.get("zones", {}))

        # Performance tracking
        self.parameter_history: List[Dict[str, Any]] = []
        self.prediction_errors: List[float] = []

    def _initialize_thermal_zones(
        self, zones_config: Dict[str, Dict[str, Any]]
    ) -> None:
        """Initialize thermal zones from configuration."""
        for zone_name, zone_config in zones_config.items():
            self.zones[zone_name] = ThermalZone(zone_name, zone_config)
            self.logger.info(f"Initialized thermal zone: {zone_name}")

    def _extract_thermal_features(
        self, data: pd.DataFrame, zone_name: str
    ) -> pd.DataFrame:
        """
        Extract and engineer features for thermal modeling.

        Args:
            data: Input data with weather and system state
            zone_name: Target zone name

        Returns:
            Engineered feature DataFrame
        """
        features = pd.DataFrame(index=data.index)

        # Weather features
        if "outdoor_temp" in data.columns:
            features["outdoor_temp"] = data["outdoor_temp"]
            features["outdoor_temp_lag1h"] = data["outdoor_temp"].shift(12)  # 1h lag
            features["outdoor_temp_trend"] = data["outdoor_temp"].diff(12)  # 1h trend
        elif "current_temperature" in data.columns:
            features["outdoor_temp"] = data["current_temperature"]
            features["outdoor_temp_lag1h"] = data["current_temperature"].shift(12)
            features["outdoor_temp_trend"] = data["current_temperature"].diff(12)

        # Solar radiation features
        if "shortwave_radiation" in data.columns:
            features["solar_radiation"] = data["shortwave_radiation"]
            features["solar_radiation_cumulative"] = (
                data["shortwave_radiation"].rolling(window=12).sum()
            )
        elif "absolute_solar_irradiance" in data.columns:
            features["solar_radiation"] = data["absolute_solar_irradiance"]
            features["solar_radiation_cumulative"] = (
                data["absolute_solar_irradiance"].rolling(window=12).sum()
            )
        elif "solar_radiation" in data.columns:
            features["solar_radiation"] = data["solar_radiation"]
            features["solar_radiation_cumulative"] = (
                data["solar_radiation"].rolling(window=12).sum()
            )

        # Wind and humidity (affect heat transfer)
        if "windspeed_10m" in data.columns:
            features["wind_speed"] = data["windspeed_10m"]
        if "relativehumidity_2m" in data.columns:
            features["humidity"] = data["relativehumidity_2m"]
        elif "relative_humidity" in data.columns:
            features["humidity"] = data["relative_humidity"]

        # Heating state
        heating_col = (
            f"{zone_name}_heating"
            if f"{zone_name}_heating" in data.columns
            else "heating_power"
        )
        if heating_col in data.columns:
            features["heating_power"] = data[heating_col]
            features["heating_power_lag"] = data[heating_col].shift(1)
            # Heating duration (consecutive heating periods)
            features["heating_duration"] = (
                data[heating_col]
                .groupby((data[heating_col] == 0).cumsum())
                .cumsum()
                .where(data[heating_col] > 0, 0)
            )

        # Time features (daily and seasonal patterns)
        features["hour"] = data.index.hour
        features["day_of_week"] = data.index.dayofweek
        features["month"] = data.index.month
        features["sin_hour"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["cos_hour"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["sin_day"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["cos_day"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        # Temperature history (for thermal mass effect)
        temp_col = None
        if f"temperature_{zone_name}" in data.columns:
            temp_col = f"temperature_{zone_name}"
        elif "temperature" in data.columns:
            temp_col = "temperature"

        if temp_col is not None:
            features["temp_lag1"] = data[temp_col].shift(1)
            features["temp_lag2"] = data[temp_col].shift(2)
            features["temp_lag12"] = data[temp_col].shift(12)  # 1 hour
            features["temp_change_rate"] = data[temp_col].diff()

        # Coupled zones (thermal exchange with adjacent rooms)
        for other_zone in self.zones.keys():
            if other_zone != zone_name:
                other_temp_col = f"temperature_{other_zone}"
                if other_temp_col in data.columns and temp_col is not None:
                    features[f"temp_diff_{other_zone}"] = (
                        data[temp_col] - data[other_temp_col]
                    )

        # Fill missing values
        features = features.ffill().bfill()

        return features

    def _simulate_thermal_response(
        self,
        zone: ThermalZone,
        initial_temp: float,
        outdoor_temps: np.ndarray,
        solar_radiation: np.ndarray,
        heating_powers: np.ndarray,
        time_points: np.ndarray,
        coupled_temps: Dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulate thermal response using RC model.

        Args:
            zone: Thermal zone object
            initial_temp: Initial indoor temperature
            outdoor_temps: Outdoor temperature time series
            solar_radiation: Solar radiation time series
            heating_powers: Heating power time series
            time_points: Time points for simulation
            coupled_temps: Coupled zone temperatures

        Returns:
            Simulated indoor temperatures
        """
        if coupled_temps is None:
            coupled_temps = {}

        def thermal_ode(
            temp, t, zone, outdoor_temps, solar_radiation, heating_powers, coupled_temps
        ):
            """ODE for thermal dynamics."""
            # Interpolate inputs at current time
            idx = min(int(t / self.time_step), len(outdoor_temps) - 1)

            outdoor_temp = outdoor_temps[idx]
            solar_rad = solar_radiation[idx] if len(solar_radiation) > idx else 0
            heating_power = heating_powers[idx] if len(heating_powers) > idx else 0

            coupled_temp_values = {}
            for zone_name, temps in coupled_temps.items():
                coupled_temp_values[zone_name] = (
                    temps[idx] if len(temps) > idx else temp
                )

            return zone.temperature_derivative(
                temp, outdoor_temp, solar_rad, heating_power, coupled_temp_values
            )

        # Solve ODE
        temperatures = odeint(
            thermal_ode,
            initial_temp,
            time_points,
            args=(zone, outdoor_temps, solar_radiation, heating_powers, coupled_temps),
        )

        return temperatures.flatten()

    def _optimize_thermal_parameters(
        self,
        zone_name: str,
        training_data: pd.DataFrame,
        target_temps: np.ndarray,
    ) -> Dict[str, float]:
        """
        Optimize thermal parameters using measured data.

        Args:
            zone_name: Zone name to optimize
            training_data: Training features
            target_temps: Measured temperatures

        Returns:
            Optimized parameters
        """
        zone = self.zones[zone_name]
        initial_temp = target_temps[0]

        # Extract inputs for simulation
        outdoor_temps = training_data["outdoor_temp"].values
        solar_radiation = training_data.get(
            "solar_radiation", pd.Series(0, index=training_data.index)
        ).values
        heating_powers = training_data.get(
            "heating_power", pd.Series(0, index=training_data.index)
        ).values
        time_points = np.arange(len(target_temps)) * self.time_step

        def objective_function(params):
            """Objective function for parameter optimization."""
            # Update zone parameters
            zone.thermal_resistance = max(params[0], 0.001)  # Avoid division by zero
            zone.thermal_capacitance = max(params[1], 1000)  # Minimum capacitance
            zone.solar_gain_coeff = np.clip(params[2], 0, 1)  # Solar gain coefficient

            try:
                # Simulate with current parameters
                simulated_temps = self._simulate_thermal_response(
                    zone,
                    initial_temp,
                    outdoor_temps,
                    solar_radiation,
                    heating_powers,
                    time_points,
                )

                # Calculate RMSE
                rmse = np.sqrt(np.mean((simulated_temps - target_temps) ** 2))
                return rmse
            except Exception:
                return 1000  # Large error for invalid parameters

        # Initial guess
        x0 = [zone.thermal_resistance, zone.thermal_capacitance, zone.solar_gain_coeff]

        # Bounds for parameters
        bounds = [
            (0.001, 0.1),  # Thermal resistance (K/W)
            (1000, 10000000),  # Thermal capacitance (J/K)
            (0, 1),  # Solar gain coefficient
        ]

        # Optimize
        result = optimize.minimize(
            objective_function,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )

        if result.success:
            optimized_params = {
                "thermal_resistance": result.x[0],
                "thermal_capacitance": result.x[1],
                "solar_gain_coeff": result.x[2],
                "optimization_error": result.fun,
            }
            self.logger.info(
                f"Optimized parameters for {zone_name}: R={result.x[0]:.4f}, "
                f"C={result.x[1]:.0f}, Solar={result.x[2]:.3f}, RMSE={result.fun:.3f}"
            )
        else:
            optimized_params = {
                "thermal_resistance": zone.thermal_resistance,
                "thermal_capacitance": zone.thermal_capacitance,
                "solar_gain_coeff": zone.solar_gain_coeff,
                "optimization_error": float("inf"),
            }
            self.logger.warning(f"Parameter optimization failed for {zone_name}")

        return optimized_params

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        zone_name: str = "default",
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Train thermal model using hybrid physics + ML approach.

        Args:
            X: Feature matrix with weather and system data
            y: Target temperatures
            validation_data: Optional validation data
            zone_name: Zone name for multi-zone modeling
            **kwargs: Additional training parameters

        Returns:
            Performance metrics
        """
        self.logger.info(f"Training thermal model for zone: {zone_name}")
        start_time = datetime.now()

        # Ensure zone exists
        if zone_name not in self.zones:
            self.zones[zone_name] = ThermalZone(
                zone_name, self.config.get("default_zone", {})
            )

        # Extract thermal features
        features = self._extract_thermal_features(X, zone_name)
        self.logger.info(f"Extracted {len(features.columns)} thermal features")

        # Split into training and validation if not provided
        if validation_data is None:
            split_idx = int(len(features) * 0.8)
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = features, self._extract_thermal_features(
                validation_data[0], zone_name
            )
            y_train, y_val = y, validation_data[1]

        # 1. Optimize physics model parameters
        self.logger.info("Optimizing thermal physics parameters...")
        optimized_params = self._optimize_thermal_parameters(
            zone_name, X_train, y_train.values
        )

        # Update zone with optimized parameters
        zone = self.zones[zone_name]
        zone.thermal_resistance = optimized_params["thermal_resistance"]
        zone.thermal_capacitance = optimized_params["thermal_capacitance"]
        zone.solar_gain_coeff = optimized_params["solar_gain_coeff"]

        # 2. Generate physics-based predictions
        initial_temp = y_train.iloc[0]
        outdoor_temps = X_train["outdoor_temp"].values
        solar_radiation = X_train.get(
            "solar_radiation", pd.Series(0, index=X_train.index)
        ).values
        heating_powers = X_train.get(
            "heating_power", pd.Series(0, index=X_train.index)
        ).values
        time_points = np.arange(len(y_train)) * self.time_step

        physics_predictions = self._simulate_thermal_response(
            zone,
            initial_temp,
            outdoor_temps,
            solar_radiation,
            heating_powers,
            time_points,
        )

        # 3. Train ML model on residuals (physics prediction errors)
        residuals = y_train.values - physics_predictions

        # Prepare ML features (remove temperature lags to avoid leakage)
        ml_feature_cols = [col for col in features.columns if "temp_lag" not in col]
        ml_features = features[ml_feature_cols]

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train[ml_features.columns])

        # Train ML model on residuals
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )

        self.ml_model.fit(X_train_scaled, residuals)

        # 4. Validate combined model
        val_physics_predictions = self._simulate_thermal_response(
            zone,
            y_val.iloc[0],
            X_val["outdoor_temp"].values,
            X_val.get("solar_radiation", pd.Series(0, index=X_val.index)).values,
            X_val.get("heating_power", pd.Series(0, index=X_val.index)).values,
            np.arange(len(y_val)) * self.time_step,
        )

        X_val_scaled = self.feature_scaler.transform(X_val[ml_features.columns])
        val_ml_residuals = self.ml_model.predict(X_val_scaled)

        # Combined predictions
        val_predictions = (
            self.physics_model_weight * val_physics_predictions
            + self.ml_model_weight * (val_physics_predictions + val_ml_residuals)
        )

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        mae = mean_absolute_error(y_val, val_predictions)
        r2 = r2_score(y_val, val_predictions)

        # Cross-validation score
        cv_scores = cross_val_score(
            self.ml_model,
            X_train_scaled,
            residuals,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        training_time = (datetime.now() - start_time).total_seconds()

        # Store parameter history
        self.parameter_history.append(
            {
                "timestamp": datetime.now(),
                "zone": zone_name,
                "thermal_resistance": zone.thermal_resistance,
                "thermal_capacitance": zone.thermal_capacitance,
                "solar_gain_coeff": zone.solar_gain_coeff,
                "optimization_error": optimized_params["optimization_error"],
                "ml_model_features": len(ml_features.columns),
            }
        )

        performance = PerformanceMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=0.0,  # Calculate if needed
            bias=np.mean(val_predictions - y_val),
            std_residual=np.std(val_predictions - y_val),
            max_error=np.max(np.abs(val_predictions - y_val)),
            explained_variance=r2,
        )

        self.logger.info(
            f"Thermal model training completed for {zone_name}: "
            f"RMSE={rmse:.3f}°C, MAE={mae:.3f}°C, R²={r2:.3f}, "
            f"Training time={training_time:.1f}s"
        )

        # Set the trained model for persistence
        self.model = {
            "physics_model": getattr(self, "physics_model", None),
            "ml_model": self.ml_model,
            "feature_scaler": self.feature_scaler,
            "thermal_parameters": getattr(self, "thermal_parameters", None),
            "zone_capacitance": getattr(self, "zone_capacitance", None),
            "zone_resistance": getattr(self, "zone_resistance", None),
        }

        return performance

    def predict(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = False,
        zone_name: str = "default",
        **kwargs,
    ) -> PredictionResult:
        """
        Predict room temperatures using hybrid thermal model.

        Args:
            X: Feature matrix with weather and system data
            return_uncertainty: Whether to return prediction uncertainty
            zone_name: Zone name for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Prediction results with temperatures and uncertainty
        """
        if self.ml_model is None:
            raise ValueError("Model must be trained before making predictions")

        if zone_name not in self.zones:
            raise ValueError(f"Zone {zone_name} not found in trained zones")

        # Extract features
        features = self._extract_thermal_features(X, zone_name)

        # Get initial temperature (use last known or reasonable default)
        initial_temp = kwargs.get("initial_temp", 20.0)
        if "temp_lag1" in features.columns:
            initial_temp = (
                features["temp_lag1"].iloc[0]
                if not pd.isna(features["temp_lag1"].iloc[0])
                else initial_temp
            )

        zone = self.zones[zone_name]

        # 1. Physics-based predictions
        outdoor_temps = features["outdoor_temp"].values
        solar_radiation = features.get(
            "solar_radiation", pd.Series(0, index=features.index)
        ).values
        heating_powers = features.get(
            "heating_power", pd.Series(0, index=features.index)
        ).values
        time_points = np.arange(len(features)) * self.time_step

        physics_predictions = self._simulate_thermal_response(
            zone,
            initial_temp,
            outdoor_temps,
            solar_radiation,
            heating_powers,
            time_points,
        )

        # 2. ML-based residual predictions
        ml_feature_cols = [col for col in features.columns if "temp_lag" not in col]
        ml_features = features[ml_feature_cols]
        X_scaled = self.feature_scaler.transform(ml_features)
        ml_residuals = self.ml_model.predict(X_scaled)

        # 3. Combined predictions
        predictions = (
            self.physics_model_weight * physics_predictions
            + self.ml_model_weight * (physics_predictions + ml_residuals)
        )

        # 4. Uncertainty estimation
        uncertainty = None
        if return_uncertainty:
            # Use ensemble variance for uncertainty
            n_bootstrap = 10
            bootstrap_predictions = []

            for _ in range(n_bootstrap):
                # Add small noise to physics parameters
                noisy_zone = ThermalZone(zone.zone_name, zone.config)
                noisy_zone.thermal_resistance = (
                    zone.thermal_resistance * np.random.normal(1, 0.05)
                )
                noisy_zone.thermal_capacitance = (
                    zone.thermal_capacitance * np.random.normal(1, 0.05)
                )
                noisy_zone.solar_gain_coeff = np.clip(
                    zone.solar_gain_coeff * np.random.normal(1, 0.05), 0, 1
                )

                noisy_physics = self._simulate_thermal_response(
                    noisy_zone,
                    initial_temp,
                    outdoor_temps,
                    solar_radiation,
                    heating_powers,
                    time_points,
                )

                # Add ML prediction variance
                if hasattr(self.ml_model, "estimators_"):
                    tree_predictions = np.array(
                        [tree.predict(X_scaled) for tree in self.ml_model.estimators_]
                    )
                    ml_uncertainty = np.std(tree_predictions, axis=0)
                    noisy_ml = ml_residuals + np.random.normal(0, ml_uncertainty)
                else:
                    noisy_ml = ml_residuals + np.random.normal(
                        0, 0.1, len(ml_residuals)
                    )

                bootstrap_pred = (
                    self.physics_model_weight * noisy_physics
                    + self.ml_model_weight * (noisy_physics + noisy_ml)
                )
                bootstrap_predictions.append(bootstrap_pred)

            bootstrap_predictions = np.array(bootstrap_predictions)
            uncertainty = np.std(bootstrap_predictions, axis=0)

        # Create prediction result
        result = PredictionResult(
            predictions=pd.Series(predictions, index=X.index),
            uncertainty=pd.Series(uncertainty, index=X.index)
            if uncertainty is not None
            else None,
            confidence_intervals={
                "lower_95": pd.Series(predictions - 1.96 * uncertainty, index=X.index),
                "upper_95": pd.Series(predictions + 1.96 * uncertainty, index=X.index),
                "lower_80": pd.Series(predictions - 1.28 * uncertainty, index=X.index),
                "upper_80": pd.Series(predictions + 1.28 * uncertainty, index=X.index),
            }
            if uncertainty is not None
            else None,
            feature_contributions=pd.DataFrame(
                {
                    col: [importance] * len(X)
                    for col, importance in zip(
                        ml_features.columns, self.ml_model.feature_importances_
                    )
                },
                index=X.index,
            )
            if hasattr(self.ml_model, "feature_importances_")
            else None,
        )

        return result

    def update_online(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        zone_name: str = "default",
        **kwargs,
    ) -> None:
        """
        Update model parameters with new observations (online learning).

        Args:
            X_new: New feature observations
            y_new: New temperature observations
            zone_name: Zone name to update
            **kwargs: Additional update parameters
        """
        if zone_name not in self.zones or self.ml_model is None:
            self.logger.warning(f"Cannot update: zone {zone_name} not trained")
            return

        # Calculate recent prediction error
        recent_prediction = self.predict(X_new, zone_name=zone_name)
        error = np.mean(np.abs(recent_prediction.predictions - y_new))
        self.prediction_errors.append(error)

        # Adapt thermal parameters based on recent errors
        if len(self.prediction_errors) > 10:
            recent_errors = self.prediction_errors[-10:]
            error_trend = np.mean(recent_errors[-5:]) - np.mean(recent_errors[:5])

            if error_trend > 0.1:  # Increasing error trend
                zone = self.zones[zone_name]
                # Slight adjustment to thermal resistance
                zone.thermal_resistance *= 1 + self.adaptation_rate * error_trend
                self.logger.info(
                    f"Adapted thermal resistance for {zone_name}: {zone.thermal_resistance:.4f}"
                )

        # Keep error history manageable
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-50:]

    def get_thermal_insights(self, zone_name: str = "default") -> Dict[str, Any]:
        """
        Get thermal insights and diagnostics for a zone.

        Args:
            zone_name: Zone name for insights

        Returns:
            Dictionary of thermal insights
        """
        if zone_name not in self.zones:
            return {"error": f"Zone {zone_name} not found"}

        zone = self.zones[zone_name]

        insights = {
            "zone_name": zone_name,
            "thermal_properties": {
                "thermal_resistance": zone.thermal_resistance,
                "thermal_capacitance": zone.thermal_capacitance,
                "time_constant": zone.thermal_resistance
                * zone.thermal_capacitance
                / 3600,  # hours
                "solar_gain_coefficient": zone.solar_gain_coeff,
                "heating_power": zone.heating_power,
                "window_area": zone.window_area,
            },
            "parameter_optimization": {
                "optimization_history": len(self.parameter_history),
                "latest_optimization_error": self.parameter_history[-1].get(
                    "optimization_error"
                )
                if self.parameter_history
                else None,
            },
            "prediction_performance": {
                "recent_error_count": len(self.prediction_errors),
                "average_recent_error": np.mean(self.prediction_errors[-10:])
                if self.prediction_errors
                else None,
                "error_trend": "improving"
                if len(self.prediction_errors) > 5
                and np.mean(self.prediction_errors[-3:])
                < np.mean(self.prediction_errors[-6:-3])
                else "stable",
            },
            "model_characteristics": {
                "physics_model_weight": self.physics_model_weight,
                "ml_model_weight": self.ml_model_weight,
                "time_step_seconds": self.time_step,
                "adaptation_rate": self.adaptation_rate,
                "coupled_zones": list(zone.coupled_zones.keys()),
            },
        }

        return insights
