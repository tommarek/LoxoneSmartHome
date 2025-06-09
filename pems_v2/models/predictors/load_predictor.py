"""
Load Predictor for PEMS v2.

Advanced base electrical load forecasting using ensemble methods and pattern recognition.
Predicts non-controllable electricity consumption patterns including appliances, lighting,
and other base loads excluding heating/cooling and EV charging.

Key Features:
- Multi-scale temporal modeling (hourly, daily, weekly, seasonal)
- Appliance usage pattern recognition and decomposition
- Weather dependency modeling for temperature-sensitive loads
- Occupancy-based consumption patterns
- Holiday and event impact modeling
- Load disaggregation and component analysis
- Uncertainty quantification for optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config.settings import LoadModelSettings
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..base import BasePredictor, PerformanceMetrics, PredictionResult


class LoadComponent:
    """
    Represents a decomposed load component (e.g., base load, appliances, lighting).
    """

    def __init__(self, name: str, pattern_type: str = "variable"):
        """Initialize load component.

        Args:
            name: Component name (e.g., 'base_load', 'appliances', 'lighting')
            pattern_type: 'constant', 'periodic', 'variable', 'seasonal'
        """
        self.name = name
        self.pattern_type = pattern_type
        self.model = None
        self.scaler = StandardScaler()
        self.contribution = 0.0  # Fraction of total load
        self.seasonal_pattern = None
        self.daily_pattern = None

    def extract_pattern(
        self, data: pd.Series, timestamps: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Extract characteristic patterns from the component data."""
        patterns = {}

        # Daily pattern (24 hour cycle)
        hourly_avg = data.groupby(timestamps.hour).mean()
        patterns["daily_pattern"] = hourly_avg

        # Weekly pattern
        weekly_avg = data.groupby(timestamps.dayofweek).mean()
        patterns["weekly_pattern"] = weekly_avg

        # Seasonal pattern (monthly)
        monthly_avg = data.groupby(timestamps.month).mean()
        patterns["seasonal_pattern"] = monthly_avg

        # Statistical characteristics
        patterns["statistics"] = {
            "mean": data.mean(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "variability": data.std() / data.mean() if data.mean() > 0 else 0,
        }

        return patterns


class LoadPredictor(BasePredictor):
    """
    Advanced load predictor for base electrical consumption forecasting.

    Uses ensemble methods with load decomposition and pattern recognition
    to predict non-controllable electricity consumption.
    """

    def __init__(
        self, load_settings: LoadModelSettings, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize load predictor with configuration.

        Args:
            load_settings: Load model configuration from system settings
            config: Optional additional configuration for model-specific parameters
        """
        # Merge settings with additional config
        merged_config = {
            "model_path": load_settings.model_path,
            "horizon_hours": load_settings.horizon_hours,
        }
        if config:
            merged_config.update(config)

        super().__init__(merged_config)
        self.load_settings = load_settings
        self.logger = logging.getLogger(f"{__name__}.LoadPredictor")

        # Model configuration
        self.decomposition_method = config.get(
            "decomposition_method", "nmf"
        )  # 'nmf', 'pca', 'clustering'
        self.n_components = config.get("n_components", 5)
        self.prediction_horizon = load_settings.horizon_hours  # hours from settings
        self.include_weather = config.get("include_weather", True)
        self.include_occupancy = config.get("include_occupancy", True)

        # Models for different components
        self.component_models: Dict[str, Any] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.load_components: Dict[str, LoadComponent] = {}

        # Main ensemble models
        self.base_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.pattern_model = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        self.trend_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

        # Preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.pattern_scaler = StandardScaler()
        self.trend_scaler = StandardScaler()

        # Load decomposition
        self.decomposer = None
        self.baseline_load = None

        # Performance tracking
        self.component_contributions: Dict[str, float] = {}
        self.prediction_accuracy: Dict[str, float] = {}

    def _extract_load_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for load prediction.

        Args:
            data: Input DataFrame with timestamps as index

        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame(index=data.index)

        # Temporal features
        features["hour"] = data.index.hour
        features["day_of_week"] = data.index.dayofweek
        features["day_of_year"] = data.index.dayofyear
        features["month"] = data.index.month
        features["week_of_year"] = data.index.isocalendar().week.astype(
            int
        )  # Convert to int

        # Cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        # Weekend/weekday indicator
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Holiday indicator (simplified - major holidays)
        features["is_holiday"] = self._detect_holidays(data.index)

        # Time-based patterns
        features["is_business_hours"] = (
            (features["hour"] >= 8)
            & (features["hour"] <= 18)
            & (features["day_of_week"] < 5)
        ).astype(int)
        features["is_evening"] = (
            (features["hour"] >= 18) & (features["hour"] <= 23)
        ).astype(int)
        features["is_night"] = (
            (features["hour"] >= 0) & (features["hour"] <= 6)
        ).astype(int)

        # Weather features (if available)
        if self.include_weather:
            weather_features = self._extract_weather_features(data)
            features = pd.concat([features, weather_features], axis=1)

        # Occupancy proxy features
        if self.include_occupancy:
            occupancy_features = self._extract_occupancy_features(data, features)
            features = pd.concat([features, occupancy_features], axis=1)

        # Load lag features (autoregressive)
        if "load" in data.columns or "consumption" in data.columns:
            load_col = "load" if "load" in data.columns else "consumption"
            features["load_lag_1h"] = data[load_col].shift(
                4
            )  # 1 hour lag (15min intervals)
            features["load_lag_24h"] = data[load_col].shift(96)  # 24 hour lag
            features["load_lag_7d"] = data[load_col].shift(672)  # 7 day lag
            features["load_rolling_mean_24h"] = data[load_col].rolling(window=96).mean()
            features["load_rolling_std_24h"] = data[load_col].rolling(window=96).std()

        # Remove any remaining NaN values
        features = features.ffill().bfill().fillna(0)

        return features

    def _detect_holidays(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Detect major holidays (simplified implementation).

        Args:
            timestamps: DatetimeIndex

        Returns:
            Boolean series indicating holidays
        """
        holidays = pd.Series(False, index=timestamps)

        for year in timestamps.year.unique():
            # New Year's Day
            holidays.loc[f"{year}-01-01"] = True
            # Christmas
            holidays.loc[f"{year}-12-25"] = True
            holidays.loc[f"{year}-12-26"] = True
            # Czech holidays
            holidays.loc[f"{year}-05-01"] = True  # Labour Day
            holidays.loc[f"{year}-05-08"] = True  # Victory Day
            holidays.loc[f"{year}-07-05"] = True  # Cyril and Methodius
            holidays.loc[f"{year}-07-06"] = True  # Jan Hus
            holidays.loc[f"{year}-09-28"] = True  # Czech Statehood
            holidays.loc[f"{year}-10-28"] = True  # Independence Day
            holidays.loc[f"{year}-11-17"] = True  # Freedom Day

        return holidays.fillna(False)

    def _extract_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract weather-related features affecting electricity consumption.

        Args:
            data: Input DataFrame

        Returns:
            Weather features DataFrame
        """
        weather_features = pd.DataFrame(index=data.index)

        # Temperature features (affects appliance usage)
        if "outdoor_temp" in data.columns:
            weather_features["outdoor_temp"] = data["outdoor_temp"]
            weather_features["temp_deviation"] = (
                data["outdoor_temp"] - data["outdoor_temp"].rolling(window=672).mean()
            )  # Weekly average
            # Cooling/heating degree hours
            weather_features["cooling_degree_hours"] = np.maximum(
                data["outdoor_temp"] - 24, 0
            )
            weather_features["heating_degree_hours"] = np.maximum(
                18 - data["outdoor_temp"], 0
            )
        elif "current_temperature" in data.columns:
            weather_features["outdoor_temp"] = data["current_temperature"]
            weather_features["temp_deviation"] = (
                data["current_temperature"]
                - data["current_temperature"].rolling(window=672).mean()
            )
            weather_features["cooling_degree_hours"] = np.maximum(
                data["current_temperature"] - 24, 0
            )
            weather_features["heating_degree_hours"] = np.maximum(
                18 - data["current_temperature"], 0
            )

        # Humidity (affects comfort and appliance usage)
        if "relativehumidity_2m" in data.columns:
            weather_features["humidity"] = data["relativehumidity_2m"]
            weather_features["humidity_discomfort"] = np.maximum(
                data["relativehumidity_2m"] - 60, 0
            )  # Above comfort zone
        elif "relative_humidity" in data.columns:
            weather_features["humidity"] = data["relative_humidity"]
            weather_features["humidity_discomfort"] = np.maximum(
                data["relative_humidity"] - 60, 0
            )

        # Wind (affects building heat loss)
        if "windspeed_10m" in data.columns:
            weather_features["wind_speed"] = data["windspeed_10m"]

        # Solar radiation (affects cooling loads and natural lighting)
        if "shortwave_radiation" in data.columns:
            weather_features["solar_radiation"] = data["shortwave_radiation"]
            weather_features["solar_heating_effect"] = (
                data["shortwave_radiation"] * 0.1
            )  # Simplified solar gain
        elif "solar_radiation" in data.columns:
            weather_features["solar_radiation"] = data["solar_radiation"]
            weather_features["solar_heating_effect"] = data["solar_radiation"] * 0.1

        return weather_features.fillna(0)

    def _extract_occupancy_features(
        self, data: pd.DataFrame, time_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract occupancy proxy features from consumption patterns.

        Args:
            data: Input DataFrame
            time_features: Temporal features DataFrame

        Returns:
            Occupancy proxy features
        """
        occupancy_features = pd.DataFrame(index=data.index)

        # Occupancy probability based on time patterns
        occupancy_features["occupancy_prob"] = (
            0.9 * time_features["is_evening"]
            + 0.8 * time_features["is_night"]  # High probability in evening
            + 0.7  # Medium-high at night
            * (
                time_features["is_weekend"]
                & (time_features["hour"] >= 8)
                & (time_features["hour"] <= 20)
            )
            + 0.3  # Weekend daytime
            * time_features["is_business_hours"]  # Lower during business hours
        ).clip(0, 1)

        # Working from home indicator (higher consumption during business hours)
        occupancy_features["wfh_indicator"] = (
            time_features["is_business_hours"] & ~time_features["is_weekend"]
        ).astype(float)

        # Sleep hours indicator
        occupancy_features["sleep_hours"] = (
            (time_features["hour"] >= 22) | (time_features["hour"] <= 6)
        ).astype(float)

        return occupancy_features

    def _decompose_load(
        self, load_data: pd.Series, features: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Decompose total load into components using specified method.

        Args:
            load_data: Time series of load data
            features: Feature DataFrame

        Returns:
            Dictionary of load components
        """
        self.logger.info(f"Decomposing load using {self.decomposition_method} method")

        # Prepare data matrix (load with temporal features)
        load_matrix = np.column_stack(
            [
                load_data.values.reshape(-1, 1),
                features[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].values,
            ]
        )

        # Remove any NaN values
        valid_mask = ~np.isnan(load_matrix).any(axis=1)
        load_matrix_clean = load_matrix[valid_mask]

        if self.decomposition_method == "nmf":
            # Non-negative Matrix Factorization
            self.decomposer = NMF(
                n_components=self.n_components, random_state=42, max_iter=500
            )
            components = self.decomposer.fit_transform(np.abs(load_matrix_clean))
            component_names = [f"nmf_component_{i}" for i in range(self.n_components)]

        elif self.decomposition_method == "pca":
            # Principal Component Analysis
            self.decomposer = PCA(n_components=self.n_components, random_state=42)
            components = self.decomposer.fit_transform(load_matrix_clean)
            component_names = [f"pca_component_{i}" for i in range(self.n_components)]

        elif self.decomposition_method == "clustering":
            # K-means clustering based decomposition
            self.decomposer = KMeans(
                n_clusters=self.n_components, random_state=42, n_init=10
            )
            cluster_labels = self.decomposer.fit_predict(load_matrix_clean)

            # Create components based on cluster assignments
            components = np.zeros((len(load_matrix_clean), self.n_components))
            for i in range(self.n_components):
                mask = cluster_labels == i
                components[mask, i] = load_data.values[valid_mask][mask]

            component_names = [
                f"cluster_component_{i}" for i in range(self.n_components)
            ]

        # Map components back to original timeline
        decomposed_components = {}
        for i, name in enumerate(component_names):
            full_component = pd.Series(0.0, index=load_data.index)
            full_component.iloc[valid_mask] = components[:, i]
            decomposed_components[name] = full_component

        # Calculate baseline load (minimum consistent load)
        self.baseline_load = (
            load_data.rolling(window=672).quantile(0.1).median()
        )  # 10th percentile over week

        # Add interpretable components
        decomposed_components["baseline"] = pd.Series(
            self.baseline_load, index=load_data.index
        )
        decomposed_components["variable"] = load_data - self.baseline_load

        self.logger.info(
            f"Load decomposition complete. Baseline load: {self.baseline_load:.2f} kW"
        )

        return decomposed_components

    def _analyze_load_patterns(
        self, load_data: pd.Series, features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze load patterns and characteristics.

        Args:
            load_data: Load time series
            features: Feature DataFrame

        Returns:
            Pattern analysis results
        """
        analysis = {}

        # Basic statistics
        analysis["statistics"] = {
            "mean_load": load_data.mean(),
            "std_load": load_data.std(),
            "min_load": load_data.min(),
            "max_load": load_data.max(),
            "load_factor": load_data.mean() / load_data.max()
            if load_data.max() > 0
            else 0,
        }

        # Daily patterns
        daily_pattern = load_data.groupby(load_data.index.hour).agg(["mean", "std"])
        analysis["daily_pattern"] = {
            "peak_hour": daily_pattern["mean"].idxmax(),
            "min_hour": daily_pattern["mean"].idxmin(),
            "peak_load": daily_pattern["mean"].max(),
            "min_load": daily_pattern["mean"].min(),
            "daily_variation": daily_pattern["mean"].std(),
        }

        # Weekly patterns
        weekly_pattern = load_data.groupby(load_data.index.dayofweek).mean()
        analysis["weekly_pattern"] = {
            "weekday_avg": weekly_pattern[:5].mean(),
            "weekend_avg": weekly_pattern[5:].mean(),
            "weekday_weekend_ratio": weekly_pattern[:5].mean()
            / weekly_pattern[5:].mean()
            if weekly_pattern[5:].mean() > 0
            else 1,
        }

        # Frequency analysis
        if len(load_data) > 96:  # At least 24 hours of data
            fft_result = fft(load_data.values)
            frequencies = fftfreq(len(load_data), d=0.25)  # 15-minute intervals

            # Find dominant frequencies
            power_spectrum = np.abs(fft_result) ** 2
            dominant_freq_idx = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies

            analysis["frequency_analysis"] = {
                "dominant_frequencies": frequencies[dominant_freq_idx].tolist(),
                "power_spectrum_peak": power_spectrum.max(),
                "spectral_centroid": np.sum(frequencies * power_spectrum)
                / np.sum(power_spectrum),
            }

        # Seasonal patterns (if enough data)
        if len(load_data) > 672:  # At least a week of data
            seasonal_pattern = load_data.groupby(load_data.index.month).mean()
            analysis["seasonal_pattern"] = {
                "summer_avg": seasonal_pattern[6:9].mean(),  # Jun-Aug
                "winter_avg": seasonal_pattern[[12, 1, 2]].mean(),  # Dec-Feb
                "seasonal_variation": seasonal_pattern.std(),
            }

        return analysis

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Train load prediction model using ensemble approach with decomposition.

        Args:
            X: Feature matrix with system and weather data
            y: Target load consumption values
            validation_data: Optional validation data
            **kwargs: Additional training parameters

        Returns:
            Performance metrics
        """
        self.logger.info("Training load prediction model")
        start_time = datetime.now()

        # Extract comprehensive features
        features = self._extract_load_features(X)
        self.logger.info(f"Extracted {len(features.columns)} load prediction features")

        # Decompose load into components
        decomposed_load = self._decompose_load(y, features)

        # Analyze load patterns
        load_analysis = self._analyze_load_patterns(y, features)
        self.logger.info(
            f"Load analysis: Mean={load_analysis['statistics']['mean_load']:.2f} kW, "
            f"Peak hour={load_analysis['daily_pattern']['peak_hour']}"
        )

        # Split data for training/validation
        if validation_data is None:
            split_idx = int(len(features) * 0.8)
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = features, self._extract_load_features(validation_data[0])
            y_train, y_val = y, validation_data[1]

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        # Scale target
        y_train_scaled = self.target_scaler.fit_transform(
            y_train.values.reshape(-1, 1)
        ).flatten()

        # Train ensemble models
        self.logger.info("Training ensemble models...")

        # 1. Base model (main predictor)
        self.base_model.fit(X_train_scaled, y_train_scaled)
        base_pred = self.base_model.predict(X_val_scaled)
        base_pred = self.target_scaler.inverse_transform(
            base_pred.reshape(-1, 1)
        ).flatten()

        # 2. Pattern model (captures periodic patterns)
        pattern_features = X_train[
            [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "is_weekend",
                "is_holiday",
            ]
        ]
        pattern_features_scaled = self.pattern_scaler.fit_transform(pattern_features)
        self.pattern_model.fit(pattern_features_scaled, y_train_scaled)

        pattern_val_features = X_val[
            [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "is_weekend",
                "is_holiday",
            ]
        ]
        pattern_val_scaled = self.pattern_scaler.transform(pattern_val_features)
        pattern_pred = self.pattern_model.predict(pattern_val_scaled)
        pattern_pred = self.target_scaler.inverse_transform(
            pattern_pred.reshape(-1, 1)
        ).flatten()

        # 3. Trend model (captures long-term trends)
        if "load_lag_24h" in X_train.columns:
            trend_features = X_train[
                ["load_lag_24h", "load_rolling_mean_24h", "day_of_year", "month"]
            ].fillna(0)
            trend_features_scaled = self.trend_scaler.fit_transform(trend_features)
            self.trend_model.fit(trend_features_scaled, y_train_scaled)

            trend_val_features = X_val[
                ["load_lag_24h", "load_rolling_mean_24h", "day_of_year", "month"]
            ].fillna(0)
            trend_val_scaled = self.trend_scaler.transform(trend_val_features)
            trend_pred = self.trend_model.predict(trend_val_scaled)
            trend_pred = self.target_scaler.inverse_transform(
                trend_pred.reshape(-1, 1)
            ).flatten()
        else:
            trend_pred = np.zeros_like(base_pred)

        # Ensemble weighting (simple weighted average)
        base_error = mean_absolute_error(y_val, base_pred)
        pattern_error = mean_absolute_error(y_val, pattern_pred)
        trend_error = (
            mean_absolute_error(y_val, trend_pred)
            if not np.all(trend_pred == 0)
            else float("inf")
        )

        # Calculate weights (inverse of error)
        total_inv_error = (
            1 / base_error
            + 1 / pattern_error
            + (1 / trend_error if trend_error != float("inf") else 0)
        )
        self.ensemble_weights = {
            "base": (1 / base_error) / total_inv_error,
            "pattern": (1 / pattern_error) / total_inv_error,
            "trend": (1 / trend_error) / total_inv_error
            if trend_error != float("inf")
            else 0,
        }

        # Combined prediction
        ensemble_pred = (
            self.ensemble_weights["base"] * base_pred
            + self.ensemble_weights["pattern"] * pattern_pred
            + self.ensemble_weights["trend"] * trend_pred
        )

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        mae = mean_absolute_error(y_val, ensemble_pred)
        r2 = r2_score(y_val, ensemble_pred)

        # Calculate MAPE
        mape = (
            np.mean(np.abs((y_val - ensemble_pred) / y_val)) * 100
            if y_val.min() > 0
            else float("inf")
        )

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(
            self.base_model,
            X_train_scaled,
            y_train_scaled,
            cv=tscv,
            scoring="neg_mean_squared_error",
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        training_time = (datetime.now() - start_time).total_seconds()

        # Store component contributions
        self.component_contributions = {
            "baseline_load": self.baseline_load / y.mean() if y.mean() > 0 else 0,
            "variable_load": 1 - (self.baseline_load / y.mean()) if y.mean() > 0 else 1,
        }

        performance = PerformanceMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            bias=np.mean(ensemble_pred - y_val),
            std_residual=np.std(ensemble_pred - y_val),
            max_error=np.max(np.abs(ensemble_pred - y_val)),
            explained_variance=r2,
        )

        self.logger.info(
            f"Load model training completed: RMSE={rmse:.3f} kW, MAE={mae:.3f} kW, "
            f"R²={r2:.3f}, Training time={training_time:.1f}s"
        )
        self.logger.info(
            f"Ensemble weights: Base={self.ensemble_weights['base']:.3f}, "
            f"Pattern={self.ensemble_weights['pattern']:.3f}, "
            f"Trend={self.ensemble_weights['trend']:.3f}"
        )

        # Set the trained model for persistence
        self.model = {
            "base_model": self.base_model,
            "pattern_model": self.pattern_model,
            "trend_model": self.trend_model,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "pattern_scaler": self.pattern_scaler,
            "trend_scaler": self.trend_scaler,
            "ensemble_weights": self.ensemble_weights,
            "baseline_load": self.baseline_load,
            "component_contributions": self.component_contributions,
        }

        return performance

    def predict(
        self, X: pd.DataFrame, return_uncertainty: bool = False, **kwargs
    ) -> PredictionResult:
        """
        Predict load consumption using trained ensemble model.

        Args:
            X: Feature matrix
            return_uncertainty: Whether to return prediction uncertainty
            **kwargs: Additional prediction parameters

        Returns:
            Prediction results with load forecasts
        """
        if self.base_model is None or not hasattr(self.feature_scaler, "scale_"):
            raise ValueError("Model must be trained before making predictions")

        # Extract features
        features = self._extract_load_features(X)

        # Scale features
        X_scaled = self.feature_scaler.transform(features)

        # Generate predictions from ensemble
        base_pred_scaled = self.base_model.predict(X_scaled)
        base_pred = self.target_scaler.inverse_transform(
            base_pred_scaled.reshape(-1, 1)
        ).flatten()

        # Pattern prediction
        pattern_features = features[
            [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "is_weekend",
                "is_holiday",
            ]
        ]
        pattern_features_scaled = self.pattern_scaler.transform(pattern_features)
        pattern_pred_scaled = self.pattern_model.predict(pattern_features_scaled)
        pattern_pred = self.target_scaler.inverse_transform(
            pattern_pred_scaled.reshape(-1, 1)
        ).flatten()

        # Trend prediction
        if "load_lag_24h" in features.columns:
            trend_features = features[
                ["load_lag_24h", "load_rolling_mean_24h", "day_of_year", "month"]
            ].fillna(0)
            trend_features_scaled = self.trend_scaler.transform(trend_features)
            trend_pred_scaled = self.trend_model.predict(trend_features_scaled)
            trend_pred = self.target_scaler.inverse_transform(
                trend_pred_scaled.reshape(-1, 1)
            ).flatten()
        else:
            trend_pred = np.zeros_like(base_pred)

        # Ensemble prediction
        predictions = (
            self.ensemble_weights["base"] * base_pred
            + self.ensemble_weights["pattern"] * pattern_pred
            + self.ensemble_weights["trend"] * trend_pred
        )

        # Uncertainty estimation
        uncertainty = None
        if return_uncertainty:
            # Calculate prediction variance from ensemble disagreement
            ensemble_preds = np.column_stack([base_pred, pattern_pred, trend_pred])
            uncertainty = np.std(ensemble_preds, axis=1)

            # Add model uncertainty (based on training residuals)
            if hasattr(self.base_model, "predict"):
                # Use bootstrap sampling for additional uncertainty
                n_bootstrap = 20
                bootstrap_preds = []

                for _ in range(n_bootstrap):
                    # Add noise to features
                    noise_scale = 0.02  # 2% noise
                    X_noisy = X_scaled + np.random.normal(
                        0, noise_scale, X_scaled.shape
                    )

                    # Predict with noisy features
                    pred_noisy = self.base_model.predict(X_noisy)
                    pred_noisy = self.target_scaler.inverse_transform(
                        pred_noisy.reshape(-1, 1)
                    ).flatten()
                    bootstrap_preds.append(pred_noisy)

                bootstrap_uncertainty = np.std(bootstrap_preds, axis=0)
                uncertainty = np.sqrt(uncertainty**2 + bootstrap_uncertainty**2)

        # Create result
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
                    "base_model": [self.ensemble_weights["base"]] * len(X),
                    "pattern_model": [self.ensemble_weights["pattern"]] * len(X),
                    "trend_model": [self.ensemble_weights["trend"]] * len(X),
                },
                index=X.index,
            )
            if hasattr(self.base_model, "feature_importances_")
            else None,
        )

        return result

    def update_online(self, X_new: pd.DataFrame, y_new: pd.Series, **kwargs) -> None:
        """
        Update model with new observations (online learning).

        Args:
            X_new: New feature observations
            y_new: New load observations
            **kwargs: Additional update parameters
        """
        if self.base_model is None or not hasattr(self.feature_scaler, "scale_"):
            self.logger.warning("Cannot update: model not trained")
            return

        # Extract features from new data
        features_new = self._extract_load_features(X_new)

        # Update baseline load estimate
        if len(y_new) > 0:
            new_baseline = np.percentile(y_new, 10)  # 10th percentile
            if self.baseline_load is not None:
                # Exponential moving average update
                alpha = 0.1  # Learning rate
                self.baseline_load = (
                    1 - alpha
                ) * self.baseline_load + alpha * new_baseline
            else:
                self.baseline_load = new_baseline

        # Update ensemble weights based on recent performance
        if len(y_new) > 10:  # Minimum samples for weight update
            # Get recent predictions
            recent_pred = self.predict(X_new)

            # Calculate recent errors
            recent_error = mean_absolute_error(y_new, recent_pred.predictions)

            # Adjust weights if performance degrades
            if recent_error > self.prediction_accuracy.get("last_mae", 0) * 1.2:
                # Increase pattern model weight (more stable)
                self.ensemble_weights["pattern"] = min(
                    0.8, self.ensemble_weights["pattern"] * 1.1
                )
                self.ensemble_weights["base"] = max(
                    0.1, self.ensemble_weights["base"] * 0.9
                )

                # Renormalize weights
                total_weight = sum(self.ensemble_weights.values())
                for key in self.ensemble_weights:
                    self.ensemble_weights[key] /= total_weight

                self.logger.info(
                    f"Updated ensemble weights due to performance change. Recent MAE: {recent_error:.3f}"
                )

            # Store performance for next update
            self.prediction_accuracy["last_mae"] = recent_error

    def get_load_insights(self) -> Dict[str, Any]:
        """
        Get load prediction insights and diagnostics.

        Returns:
            Dictionary of load insights
        """
        if self.base_model is None or not hasattr(self.feature_scaler, "scale_"):
            return {"error": "Model not trained"}

        insights = {
            "model_configuration": {
                "decomposition_method": self.decomposition_method,
                "n_components": self.n_components,
                "prediction_horizon": self.prediction_horizon,
                "include_weather": self.include_weather,
                "include_occupancy": self.include_occupancy,
            },
            "load_characteristics": {
                "baseline_load_kw": self.baseline_load,
                "baseline_contribution": self.component_contributions.get(
                    "baseline_load", 0
                ),
                "variable_contribution": self.component_contributions.get(
                    "variable_load", 0
                ),
            },
            "ensemble_performance": {
                "base_model_weight": self.ensemble_weights.get("base", 0),
                "pattern_model_weight": self.ensemble_weights.get("pattern", 0),
                "trend_model_weight": self.ensemble_weights.get("trend", 0),
                "last_mae": self.prediction_accuracy.get("last_mae", None),
            },
            "feature_importance": {
                "temporal_features": ["hour", "day_of_week", "month"],
                "weather_features": ["outdoor_temp", "humidity"]
                if self.include_weather
                else [],
                "occupancy_features": ["occupancy_prob", "wfh_indicator"]
                if self.include_occupancy
                else [],
                "lag_features": ["load_lag_1h", "load_lag_24h", "load_lag_7d"],
            },
            "model_complexity": {
                "base_model_trees": getattr(self.base_model, "n_estimators", 0),
                "pattern_model_trees": getattr(self.pattern_model, "n_estimators", 0),
                "total_features": len(self.feature_scaler.feature_names_in_)
                if hasattr(self.feature_scaler, "feature_names_in_")
                else 0,
            },
        }

        return insights
