"""
PV Production Predictor for PEMS v2.

Hybrid ML/Physics model for solar photovoltaic production forecasting with:
- XGBoost ML model for pattern recognition
- PVLib physical modeling for baseline predictions
- Weather integration with multiple data sources
- Uncertainty quantification (P10/P50/P90 predictions)
- Online learning for continuous improvement
- Seasonal and weather-dependent adjustments
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pvlib
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

from config.settings import PVModelSettings
from ..base import (BasePredictor, ModelMetadata, PerformanceMetrics,
                    PredictionResult)


class PVPredictor(BasePredictor):
    """
    Hybrid ML/Physics PV production predictor.

    Combines machine learning pattern recognition with physical solar modeling
    for accurate and interpretable photovoltaic power forecasting.
    """

    def __init__(self, pv_settings: PVModelSettings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PV predictor.

        Args:
            pv_settings: PV model configuration from system settings
            config: Optional additional configuration for system-specific parameters
        """
        # Merge settings with additional config
        merged_config = {
            "model_path": pv_settings.model_path,
            "update_interval": pv_settings.update_interval_seconds,
            "horizon_hours": pv_settings.horizon_hours,
            "quantiles": pv_settings.confidence_levels,
        }
        if config:
            merged_config.update(config)
        
        super().__init__(merged_config)
        self.pv_settings = pv_settings

        # PV system configuration
        self.system_config = config.get("pv_system", {})
        self.capacity_kw = self.system_config.get("capacity_kw", 10.0)
        self.panel_tilt = self.system_config.get("panel_tilt", 30.0)
        self.panel_azimuth = self.system_config.get(
            "panel_azimuth", 180.0
        )  # South-facing
        self.location_lat = self.system_config.get("latitude", 49.2)  # Prague
        self.location_lon = self.system_config.get("longitude", 16.6)

        # Model components
        self.ml_model = None
        self.quantile_models = {}  # For uncertainty quantification
        self.physical_model = None

        # Feature engineering
        self.weather_features = [
            "temperature_2m",
            "cloudcover",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "precipitation",
            "windspeed_10m",
            "relativehumidity_2m",
        ]

        self.temporal_features = [
            "hour",
            "day_of_year",
            "month",
            "weekday",
            "is_weekend",
            "season",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]

        # Ensemble weights
        self.ml_weight = config.get("ml_weight", 0.7)
        self.physics_weight = config.get("physics_weight", 0.3)

        # Training configuration
        self.quantiles = pv_settings.confidence_levels  # P10, P50, P90 from settings
        self.lookback_hours = merged_config.get("lookback_hours", 24)
        self.forecast_horizon = pv_settings.horizon_hours

        # Setup PVLib location
        self.location = pvlib.location.Location(
            latitude=self.location_lat, longitude=self.location_lon, tz="Europe/Prague"
        )

        # Initialize models
        self.ml_model = None
        self.physics_model = None
        self.q10_model = None
        self.q90_model = None
        self.feature_scaler = None

        self.logger.info(f"Initialized PV predictor for {self.capacity_kw}kW system")

    def create_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive weather features for ML model.

        Args:
            weather_df: Weather forecast data

        Returns:
            Enhanced weather features
        """
        features = weather_df.copy()

        # Solar position calculations
        solar_position = self.location.get_solarposition(weather_df.index)
        features["sun_elevation"] = solar_position["elevation"]
        features["sun_azimuth"] = solar_position["azimuth"]
        features["air_mass"] = pvlib.atmosphere.get_relative_airmass(
            solar_position["apparent_zenith"]
        )

        # Clear sky irradiance for reference
        clear_sky = self.location.get_clearsky(weather_df.index)
        features["clear_sky_ghi"] = clear_sky["ghi"]
        features["clear_sky_dni"] = clear_sky["dni"]
        features["clear_sky_dhi"] = clear_sky["dhi"]

        # Weather-based features
        if "cloudcover" in features.columns:
            features["clear_sky_ratio"] = (100 - features["cloudcover"]) / 100
            features["cloud_impact"] = (
                features["cloudcover"] / 100 * features["clear_sky_ghi"]
            )

        if "temperature_2m" in features.columns:
            # Temperature effects on PV efficiency
            features["temp_efficiency"] = 1 - 0.004 * (
                features["temperature_2m"] - 25
            )  # 0.4%/°C loss

        if "windspeed_10m" in features.columns:
            # Wind cooling effect
            features["wind_cooling"] = np.log1p(features["windspeed_10m"])

        # Irradiance features
        irradiance_cols = [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
        ]
        for col in irradiance_cols:
            if col in features.columns:
                features[f"{col}_normalized"] = features[col] / (
                    features["clear_sky_ghi"] + 1e-6
                )

        # Weather stability features
        if len(features) > 1:
            for col in ["temperature_2m", "cloudcover", "shortwave_radiation"]:
                if col in features.columns:
                    features[f"{col}_change"] = features[col].diff().fillna(0)
                    features[f"{col}_rolling_mean"] = (
                        features[col]
                        .rolling(window=3, center=True)
                        .mean()
                        .fillna(features[col])
                    )

        return features

    def create_temporal_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create temporal features from datetime index.

        Args:
            index: Datetime index

        Returns:
            Temporal features DataFrame
        """
        features = pd.DataFrame(index=index)

        # Basic time features
        features["hour"] = index.hour
        features["day_of_year"] = index.dayofyear
        features["month"] = index.month
        features["weekday"] = index.weekday
        features["is_weekend"] = (index.weekday >= 5).astype(int)

        # Season encoding
        features["season"] = ((index.month % 12 + 3) // 3).map(
            {1: 0, 2: 1, 3: 2, 4: 3}  # Winter  # Spring  # Summer  # Autumn
        )

        # Cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365.25)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365.25)

        # Solar-specific time features
        features["daylight_hours"] = self._calculate_daylight_hours(index)
        features["solar_noon_distance"] = np.abs(features["hour"] - 12)

        return features

    def _calculate_daylight_hours(self, index: pd.DatetimeIndex) -> pd.Series:
        """Calculate daylight hours for each day."""
        daylight_hours = []

        for date in index.date:
            day_start = pd.Timestamp(date)
            day_end = day_start + pd.Timedelta(days=1)

            # Calculate sunrise and sunset
            solar_position = self.location.get_solarposition(
                pd.date_range(day_start, day_end, freq="1min")
            )

            # Find daylight period (sun elevation > 0)
            daylight_mask = solar_position["elevation"] > 0
            daylight_minutes = daylight_mask.sum()
            daylight_hours.append(daylight_minutes / 60.0)

        return pd.Series(daylight_hours, index=index)

    def create_lag_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.

        Args:
            data: Input data with target column
            target_col: Name of target column

        Returns:
            Data with lag features
        """
        features = data.copy()

        # Create lag features for target variable
        lag_periods = [1, 2, 3, 6, 12, 24]  # 5min, 10min, 15min, 30min, 1h, 2h

        for lag in lag_periods:
            features[f"{target_col}_lag_{lag}"] = features[target_col].shift(lag)

        # Rolling statistics
        for window in [6, 12, 24]:  # 30min, 1h, 2h windows
            features[f"{target_col}_rolling_mean_{window}"] = (
                features[target_col].rolling(window=window).mean()
            )
            features[f"{target_col}_rolling_std_{window}"] = (
                features[target_col].rolling(window=window).std()
            )

        # Previous day same time
        features[f"{target_col}_prev_day"] = features[target_col].shift(
            288
        )  # 24h * 12 (5min intervals)

        # Previous week same time
        features[f"{target_col}_prev_week"] = features[target_col].shift(288 * 7)

        return features

    def calculate_physical_model_baseline(self, weather_df: pd.DataFrame) -> pd.Series:
        """
        Calculate PV production using PVLib physical model.

        Args:
            weather_df: Weather data with irradiance and temperature

        Returns:
            Predicted PV power output
        """
        try:
            # Get solar position
            solar_position = self.location.get_solarposition(weather_df.index)

            # Calculate POA irradiance
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                surface_tilt=self.panel_tilt,
                surface_azimuth=self.panel_azimuth,
                dni=weather_df.get("direct_radiation", 0),
                ghi=weather_df.get("shortwave_radiation", 0),
                dhi=weather_df.get("diffuse_radiation", 0),
                solar_zenith=solar_position["apparent_zenith"],
                solar_azimuth=solar_position["azimuth"],
            )

            # Temperature modeling
            cell_temperature = pvlib.temperature.faiman(
                poa_global=poa_irradiance["poa_global"],
                temp_air=weather_df.get("temperature_2m", 20),
                wind_speed=weather_df.get("windspeed_10m", 1),
            )

            # Simple PV model (5-parameter model would be more accurate)
            # Using basic efficiency calculation
            reference_irradiance = 1000  # W/m²
            reference_temperature = 25  # °C
            temp_coefficient = -0.004  # %/°C

            # Irradiance effect
            irradiance_ratio = poa_irradiance["poa_global"] / reference_irradiance

            # Temperature effect
            temp_effect = 1 + temp_coefficient * (
                cell_temperature - reference_temperature
            )

            # Calculate power output
            pv_power = self.capacity_kw * irradiance_ratio * temp_effect

            # Clip to physical limits
            pv_power = np.clip(pv_power, 0, self.capacity_kw)

            return pd.Series(pv_power, index=weather_df.index, name="pv_power_physical")

        except Exception as e:
            self.logger.error(f"Physical model calculation failed: {e}")
            # Return zeros as fallback
            return pd.Series(0, index=weather_df.index, name="pv_power_physical")

    def prepare_training_data(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with comprehensive feature engineering.

        Args:
            pv_data: Historical PV production data
            weather_data: Historical weather data

        Returns:
            Feature matrix and target series
        """
        # Align data by index
        aligned_data = pv_data.join(weather_data, how="inner")

        # Create target variable (assume 'InputPower' is the target)
        target_col = "InputPower"
        if target_col not in aligned_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in PV data")

        # Convert W to kW for easier handling
        aligned_data[target_col] = aligned_data[target_col] / 1000.0

        # Create comprehensive features
        self.logger.info("Creating weather features...")
        weather_features = self.create_weather_features(
            aligned_data[self.weather_features]
        )

        self.logger.info("Creating temporal features...")
        temporal_features = self.create_temporal_features(aligned_data.index)

        self.logger.info("Creating lag features...")
        lag_features = self.create_lag_features(aligned_data[[target_col]], target_col)

        # Physical model baseline
        self.logger.info("Calculating physical model baseline...")
        physical_baseline = self.calculate_physical_model_baseline(
            aligned_data[self.weather_features]
        )

        # Combine all features
        feature_columns = []

        # Weather features
        for col in weather_features.columns:
            if col not in [target_col]:
                feature_columns.append(col)

        # Temporal features
        feature_columns.extend(temporal_features.columns)

        # Lag features (exclude original target)
        lag_cols = [col for col in lag_features.columns if col != target_col]
        feature_columns.extend(lag_cols)

        # Physical model as feature
        feature_columns.append("physical_baseline")

        # Create final feature matrix
        X = pd.DataFrame(index=aligned_data.index)

        # Add weather features
        for col in weather_features.columns:
            if col != target_col:
                X[col] = weather_features[col]

        # Add temporal features
        for col in temporal_features.columns:
            X[col] = temporal_features[col]

        # Add lag features
        for col in lag_cols:
            X[col] = lag_features[col]

        # Add physical baseline
        X["physical_baseline"] = physical_baseline

        # Target variable
        y = aligned_data[target_col]

        # Remove rows with NaN values (from lag features)
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]

        self.logger.info(
            f"Prepared training data: {len(X)} samples, {len(X.columns)} features"
        )

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Train the hybrid PV prediction model.

        Args:
            X: Feature matrix
            y: Target variable (PV power in kW)
            validation_data: Optional validation set
            **kwargs: Additional training parameters

        Returns:
            Performance metrics on validation set
        """
        training_start = datetime.now()

        # Fit preprocessing
        self.fit_preprocessing(X)
        X_processed = self.prepare_features(X)

        # Split validation data if not provided
        if validation_data is None:
            split_idx = int(len(X_processed) * (1 - self.validation_split))
            X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X_processed, y
            X_val, y_val = self.prepare_features(validation_data[0]), validation_data[1]

        # Train main ML model (XGBoost)
        self.logger.info("Training XGBoost model...")

        xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "n_estimators": kwargs.get("n_estimators", 500),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        self.ml_model = xgb.XGBRegressor(**xgb_params)
        # Check if validation data is provided for early stopping
        if validation_data is not None:
            X_val_fit, y_val_fit = validation_data
            self.ml_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val_fit, y_val_fit)],
                verbose=False,
            )
        else:
            self.ml_model.fit(X_train, y_train, verbose=False)

        # Train quantile models for uncertainty
        self.logger.info("Training quantile models...")

        for quantile in self.quantiles:
            self.logger.info(f"Training quantile {quantile} model...")

            quantile_model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1,
            )

            quantile_model.fit(X_train, y_train, verbose=False)
            self.quantile_models[f"q{int(quantile*100)}"] = quantile_model

        # Calculate training metrics
        training_duration = (datetime.now() - training_start).total_seconds()

        # Evaluate on validation set
        performance = self.evaluate(X_val, y_val)

        # Create metadata
        data_hash = self.calculate_data_hash(X, y)
        version = self.generate_version(data_hash)

        self.metadata = ModelMetadata(
            model_name="PVPredictor",
            version=version,
            training_date=training_start,
            features=list(X.columns),
            target_variable="pv_power_kw",
            performance_metrics=performance.to_dict(),
            training_params=xgb_params,
            data_hash=data_hash,
            model_type="XGBoost_PVLib_Hybrid",
            training_samples=len(X_train),
            validation_samples=len(X_val),
            training_duration_seconds=training_duration,
        )

        # Store feature importance
        self.metadata.feature_importance = self.get_feature_importance()

        self.logger.info(f"Training completed in {training_duration:.1f}s")
        self.logger.info(
            f"Validation RMSE: {performance.rmse:.3f} kW, R²: {performance.r2:.3f}"
        )

        # Set the trained model for persistence
        self.model = {
            "ml_model": self.ml_model,
            "physics_model": getattr(self, "physics_model", None),
            "q10_model": getattr(self, "q10_model", None),
            "q90_model": getattr(self, "q90_model", None),
            "feature_scaler": self.feature_scaler,
            "system_config": self.system_config,
        }

        return performance

    def predict(
        self, X: pd.DataFrame, return_uncertainty: bool = False, **kwargs
    ) -> PredictionResult:
        """
        Make PV production predictions.

        Args:
            X: Feature matrix
            return_uncertainty: Whether to include uncertainty estimates
            **kwargs: Additional prediction parameters

        Returns:
            Structured prediction result with optional uncertainty
        """
        if self.ml_model is None:
            raise ValueError("Model not trained yet")

        # Prepare features
        X_processed = self.prepare_features(X)

        # ML model prediction
        ml_prediction = self.ml_model.predict(X_processed)

        # Physical model prediction (if weather features available)
        physical_prediction = None
        if any(col in X.columns for col in self.weather_features):
            try:
                physical_prediction = self.calculate_physical_model_baseline(X)
            except Exception as e:
                self.logger.warning(f"Physical model prediction failed: {e}")
                physical_prediction = np.zeros(len(X))

        # Ensemble prediction
        if physical_prediction is not None:
            ensemble_prediction = (
                self.ml_weight * ml_prediction
                + self.physics_weight * physical_prediction
            )
        else:
            ensemble_prediction = ml_prediction

        # Clip to physical limits
        ensemble_prediction = np.clip(ensemble_prediction, 0, self.capacity_kw)

        # Convert back to watts for consistency with input data
        predictions = pd.Series(
            ensemble_prediction * 1000,  # kW to W
            index=X.index,
            name="pv_power_predicted",
        )

        result = PredictionResult(
            predictions=predictions,
            model_version=self.metadata.version if self.metadata else "unknown",
        )

        # Add uncertainty quantification if requested
        if return_uncertainty and self.quantile_models:
            uncertainty_predictions = {}
            confidence_intervals = {}

            for quantile_name, quantile_model in self.quantile_models.items():
                quantile_pred = quantile_model.predict(X_processed)
                quantile_pred = (
                    np.clip(quantile_pred, 0, self.capacity_kw) * 1000
                )  # kW to W

                uncertainty_predictions[quantile_name] = pd.Series(
                    quantile_pred, index=X.index, name=f"pv_power_{quantile_name}"
                )

            # Calculate prediction intervals
            if "q10" in uncertainty_predictions and "q90" in uncertainty_predictions:
                confidence_intervals["80%"] = pd.DataFrame(
                    {
                        "lower": uncertainty_predictions["q10"],
                        "upper": uncertainty_predictions["q90"],
                    }
                )

            if "q25" in uncertainty_predictions and "q75" in uncertainty_predictions:
                confidence_intervals["50%"] = pd.DataFrame(
                    {
                        "lower": uncertainty_predictions["q25"],
                        "upper": uncertainty_predictions["q75"],
                    }
                )

            # Simple uncertainty estimate (standard deviation of quantiles)
            if len(uncertainty_predictions) >= 3:
                quantile_array = np.array(
                    [pred.values for pred in uncertainty_predictions.values()]
                )
                uncertainty_std = np.std(quantile_array, axis=0)
                result.uncertainty = pd.Series(
                    uncertainty_std, index=X.index, name="prediction_uncertainty"
                )

            result.confidence_intervals = confidence_intervals

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the ML model."""
        if self.ml_model is None or self.feature_columns is None:
            return None

        importance = self.ml_model.feature_importances_
        importance_dict = dict(zip(self.feature_columns, importance))

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def update_online(
        self, X_new: pd.DataFrame, y_new: pd.Series, learning_rate: float = 0.01
    ) -> bool:
        """
        Update model with new observations (online learning).

        Args:
            X_new: New feature data
            y_new: New target data (in kW)
            learning_rate: Learning rate for update

        Returns:
            True if update was successful
        """
        if not self.enable_online_learning:
            return False

        try:
            # Convert target to kW if needed
            if y_new.max() > 100:  # Likely in watts
                y_new = y_new / 1000.0

            # Prepare features
            X_processed = self.prepare_features(X_new)

            # Simple online learning: retrain on recent data window
            # In production, this could be more sophisticated (e.g., incremental learning)

            # Evaluate prediction error for adaptation
            current_pred = self.predict(X_new)
            prediction_error = np.mean(
                np.abs(y_new.values * 1000 - current_pred.predictions.values)
            )

            # If error is high, adapt model weights
            error_threshold = self.capacity_kw * 100  # 10% of capacity in watts

            if prediction_error > error_threshold:
                # Increase physical model weight if ML model is performing poorly
                adaptation_factor = min(prediction_error / error_threshold, 2.0)
                self.physics_weight = min(0.5, self.physics_weight * adaptation_factor)
                self.ml_weight = 1.0 - self.physics_weight

                self.logger.info(
                    f"Adapted model weights: ML={self.ml_weight:.2f}, "
                    f"Physics={self.physics_weight:.2f} (error: {prediction_error:.1f}W)"
                )

            return True

        except Exception as e:
            self.logger.error(f"Online update failed: {e}")
            return False
