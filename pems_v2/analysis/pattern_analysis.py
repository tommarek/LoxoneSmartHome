"""
PV Production Pattern Analysis for PEMS v2.

Analyzes PV production patterns:
1. Correlate with weather data (solar radiation, cloud cover, temperature)
2. Identify seasonal patterns using STL decomposition
3. Create feature engineering pipeline
4. Test models for production forecasting
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


class PVAnalyzer:
    """Analyze PV production patterns and correlations."""

    def __init__(self):
        """Initialize the PV analyzer."""
        self.logger = logging.getLogger(f"{__name__}.PVAnalyzer")

    def analyze_pv_production(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run comprehensive PV production analysis.

        Args:
            pv_data: PV production data with power/energy columns
            weather_data: Weather data with temperature, cloud cover, etc.

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting PV production analysis")

        if pv_data.empty:
            self.logger.warning("No PV data provided")
            return {}

        results = {}

        # Merge PV and weather data
        merged_data = self._merge_pv_weather_data(pv_data, weather_data)

        if merged_data.empty:
            self.logger.warning("No merged PV-weather data available")
            return {}

        # Basic statistics
        results["basic_stats"] = self._calculate_basic_statistics(merged_data)

        # Weather correlations
        results["weather_correlations"] = self._analyze_weather_correlations(merged_data)

        # Seasonal decomposition
        results["seasonal_patterns"] = self._analyze_seasonal_patterns(merged_data)

        # Clear sky analysis
        results["clear_sky_analysis"] = self._analyze_clear_sky_conditions(merged_data)

        # Efficiency analysis
        results["efficiency_analysis"] = self._analyze_efficiency_patterns(merged_data)

        # Anomaly detection
        results["anomalies"] = self._detect_anomalies(merged_data)

        # Feature importance
        results["feature_importance"] = self._analyze_feature_importance(merged_data)

        # Performance predictions
        results["prediction_performance"] = self._evaluate_prediction_models(merged_data)

        self.logger.info("PV production analysis completed")
        return results

    def _merge_pv_weather_data(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge PV and weather data on timestamp."""
        # Find the main PV power column
        pv_power_cols = [col for col in pv_data.columns if "power" in col.lower()]
        if not pv_power_cols:
            self.logger.warning("No PV power columns found")
            return pd.DataFrame()

        # Use the first power column as main PV data
        main_power_col = pv_power_cols[0]

        # Prepare PV data
        pv_clean = pv_data[[main_power_col]].copy()
        pv_clean.columns = ["pv_power"]

        # Add time features
        pv_clean["hour"] = pv_clean.index.hour
        pv_clean["day_of_year"] = pv_clean.index.dayofyear
        pv_clean["month"] = pv_clean.index.month
        pv_clean["weekday"] = pv_clean.index.weekday

        # Calculate solar position features (simplified)
        pv_clean["solar_elevation"] = self._calculate_solar_elevation(
            pv_clean.index, latitude=49.4949522  # Default to Prague coordinates
        )

        # Merge with weather data
        if not weather_data.empty:
            # Resample weather data to match PV data frequency
            weather_resampled = weather_data.resample("15min").interpolate(method="linear")
            merged = pv_clean.join(weather_resampled, how="inner")
        else:
            merged = pv_clean
            self.logger.warning("No weather data available for merge")

        # Remove negative PV values (nighttime)
        merged = merged[merged["pv_power"] >= 0]

        return merged.dropna()

    def _calculate_solar_elevation(
        self, timestamps: pd.DatetimeIndex, latitude: float
    ) -> pd.Series:
        """Calculate approximate solar elevation angle."""
        # Simplified solar elevation calculation
        # This is a rough approximation - for production use, consider pyephem or similar

        day_of_year = timestamps.dayofyear
        hour = timestamps.hour + timestamps.minute / 60.0

        # Solar declination (approximate)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        hour_angle = 15 * (hour - 12)

        # Solar elevation (approximate)
        lat_rad = np.radians(latitude)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)

        elevation = np.degrees(
            np.arcsin(
                np.sin(lat_rad) * np.sin(decl_rad)
                + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad)
            )
        )

        # Set negative elevations to 0 (sun below horizon)
        elevation = np.maximum(elevation, 0)

        return pd.Series(elevation, index=timestamps)

    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic PV production statistics."""
        pv_power = data["pv_power"]

        # Filter for daylight hours (solar elevation > 0)
        if "solar_elevation" in data.columns:
            daylight_data = data[data["solar_elevation"] > 0]["pv_power"]
        else:
            daylight_data = data[(data.index.hour >= 6) & (data.index.hour <= 18)]["pv_power"]

        stats = {
            "total_records": len(data),
            "daylight_records": len(daylight_data),
            "max_power": pv_power.max(),
            "mean_power": pv_power.mean(),
            "mean_daylight_power": daylight_data.mean() if len(daylight_data) > 0 else 0,
            "capacity_factor": pv_power.mean() / pv_power.max() if pv_power.max() > 0 else 0,
            "daylight_capacity_factor": (
                daylight_data.mean() / pv_power.max()
                if pv_power.max() > 0 and len(daylight_data) > 0
                else 0
            ),
            "total_energy_kwh": (pv_power.sum() * 0.25) / 1000,  # 15-min intervals to kWh
            "peak_months": pv_power.groupby(pv_power.index.month).mean().nlargest(3).index.tolist(),
            "low_months": pv_power.groupby(pv_power.index.month).mean().nsmallest(3).index.tolist(),
        }

        return stats

    def _analyze_weather_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between PV production and weather variables."""
        weather_cols = [
            col
            for col in data.columns
            if col not in ["pv_power", "hour", "day_of_year", "month", "weekday"]
        ]

        if not weather_cols:
            return {"warning": "No weather columns available for correlation analysis"}

        correlations = {}

        # Calculate correlations for each weather variable
        for col in weather_cols:
            if col in data.columns and data[col].notna().sum() > 100:  # Minimum data points
                corr = data["pv_power"].corr(data[col])
                correlations[col] = {
                    "correlation": corr,
                    "p_value": (
                        stats.pearsonr(data["pv_power"].dropna(), data[col].dropna())[1]
                        if len(data[col].dropna()) > 10
                        else 1.0
                    ),
                    "data_points": data[col].notna().sum(),
                }

        # Sort by absolute correlation strength
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)
        )

        # Identify strongest positive and negative correlations
        strongest_positive = (
            max(correlations.items(), key=lambda x: x[1]["correlation"]) if correlations else None
        )
        strongest_negative = (
            min(correlations.items(), key=lambda x: x[1]["correlation"]) if correlations else None
        )

        return {
            "correlations": sorted_correlations,
            "strongest_positive": strongest_positive,
            "strongest_negative": strongest_negative,
            "significant_correlations": {
                k: v
                for k, v in correlations.items()
                if abs(v["correlation"]) > 0.3 and v["p_value"] < 0.05
            },
        }

    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns using STL decomposition."""
        if len(data) < 365 * 24 * 4:  # Less than 1 year of 15-min data
            return {"warning": "Insufficient data for seasonal analysis (need at least 1 year)"}

        # Resample to daily data for seasonal analysis
        daily_data = data["pv_power"].resample("D").mean()

        # STL decomposition
        try:
            stl = STL(daily_data.dropna(), seasonal=365)  # Annual seasonality
            result = stl.fit()

            seasonal_patterns = {
                "trend": result.trend.to_dict(),
                "seasonal": result.seasonal.to_dict(),
                "residual": result.resid.to_dict(),
                "seasonal_strength": 1
                - (result.resid.var() / (result.seasonal + result.resid).var()),
                "trend_strength": 1 - (result.resid.var() / (result.trend + result.resid).var()),
            }

            # Monthly profiles
            monthly_profiles = (
                data.groupby([data.index.month, data.index.hour])["pv_power"]
                .mean()
                .unstack(level=0)
            )
            monthly_profiles.columns = [f"month_{i}" for i in monthly_profiles.columns]

            # Seasonal profiles (simplified)
            seasons = {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11],
            }

            seasonal_profiles = {}
            for season, months in seasons.items():
                season_data = data[data.index.month.isin(months)]
                if not season_data.empty:
                    seasonal_profiles[season] = season_data.groupby(season_data.index.hour)[
                        "pv_power"
                    ].mean()

            return {
                "decomposition": seasonal_patterns,
                "monthly_profiles": monthly_profiles.to_dict(),
                "seasonal_profiles": seasonal_profiles,
                "peak_season": (
                    max(
                        seasons.keys(),
                        key=lambda s: data[data.index.month.isin(seasons[s])]["pv_power"].mean(),
                    )
                    if data["pv_power"].sum() > 0
                    else None
                ),
            }

        except Exception as e:
            self.logger.warning(f"STL decomposition failed: {e}")
            return {"warning": f"STL decomposition failed: {str(e)}"}

    def _analyze_clear_sky_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze PV performance under clear sky conditions."""
        if "cloud_cover" not in data.columns:
            return {"warning": "No cloud cover data available for clear sky analysis"}

        # Define clear sky conditions (low cloud cover)
        clear_sky_mask = data["cloud_cover"] < 20  # Less than 20% cloud cover
        clear_sky_data = data[clear_sky_mask]

        if clear_sky_data.empty:
            return {"warning": "No clear sky conditions found in data"}

        # Calculate clear sky index (actual vs expected for clear conditions)
        clear_sky_max = clear_sky_data.groupby(
            [clear_sky_data.index.month, clear_sky_data.index.hour]
        )["pv_power"].quantile(0.95)

        analysis = {
            "clear_sky_records": len(clear_sky_data),
            "clear_sky_percentage": len(clear_sky_data) / len(data) * 100,
            "clear_sky_mean_power": clear_sky_data["pv_power"].mean(),
            "clear_sky_max_power": clear_sky_data["pv_power"].max(),
            "cloudy_sky_mean_power": data[~clear_sky_mask]["pv_power"].mean(),
            "clear_sky_advantage": (
                (clear_sky_data["pv_power"].mean() / data[~clear_sky_mask]["pv_power"].mean())
                if data[~clear_sky_mask]["pv_power"].mean() > 0
                else float("inf")
            ),
        }

        return analysis

    def _analyze_efficiency_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze PV efficiency patterns."""
        if "temperature" not in data.columns or "solar_elevation" not in data.columns:
            return {
                "warning": "Insufficient data for efficiency analysis (need temperature and solar elevation)"
            }

        # Filter for meaningful solar conditions
        meaningful_data = data[(data["solar_elevation"] > 10) & (data["pv_power"] > 0)]

        if meaningful_data.empty:
            return {"warning": "No meaningful solar data for efficiency analysis"}

        # Temperature coefficient analysis
        temp_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
        temp_efficiency = {}

        for low, high in temp_ranges:
            temp_mask = (meaningful_data["temperature"] >= low) & (
                meaningful_data["temperature"] < high
            )
            temp_data = meaningful_data[temp_mask]
            if not temp_data.empty:
                temp_efficiency[f"{low}-{high}C"] = {
                    "mean_power": temp_data["pv_power"].mean(),
                    "max_power": temp_data["pv_power"].max(),
                    "records": len(temp_data),
                }

        # Performance ratio calculation (simplified)
        # This would typically require irradiance data
        if len(meaningful_data) > 100:
            performance_ratio = meaningful_data["pv_power"] / meaningful_data["solar_elevation"]
            performance_stats = {
                "mean_performance_ratio": performance_ratio.mean(),
                "performance_ratio_std": performance_ratio.std(),
                "low_performance_threshold": performance_ratio.quantile(0.1),
                "high_performance_threshold": performance_ratio.quantile(0.9),
            }
        else:
            performance_stats = {"warning": "Insufficient data for performance ratio calculation"}

        return {
            "temperature_efficiency": temp_efficiency,
            "performance_statistics": performance_stats,
            "optimal_temperature_range": (
                min(temp_efficiency.items(), key=lambda x: x[1]["mean_power"])[0]
                if temp_efficiency
                else None
            ),
        }

    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in PV production."""
        if len(data) < 100:
            return {"warning": "Insufficient data for anomaly detection"}

        # Filter daylight hours for anomaly detection
        if "solar_elevation" in data.columns:
            daylight_data = data[data["solar_elevation"] > 0]
        else:
            daylight_data = data[(data.index.hour >= 6) & (data.index.hour <= 18)]

        if daylight_data.empty:
            return {"warning": "No daylight data for anomaly detection"}

        # Statistical anomaly detection using IQR
        Q1 = daylight_data["pv_power"].quantile(0.25)
        Q3 = daylight_data["pv_power"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = daylight_data[
            (daylight_data["pv_power"] < lower_bound) | (daylight_data["pv_power"] > upper_bound)
        ]

        # Zero production during daylight (potential system issues)
        zero_production = daylight_data[daylight_data["pv_power"] == 0]

        return {
            "total_anomalies": len(anomalies),
            "anomaly_percentage": len(anomalies) / len(daylight_data) * 100,
            "zero_production_events": len(zero_production),
            "anomaly_dates": anomalies.index.date.tolist()[:20],  # First 20 anomaly dates
            "largest_anomaly": {
                "timestamp": anomalies["pv_power"].idxmax() if not anomalies.empty else None,
                "value": anomalies["pv_power"].max() if not anomalies.empty else None,
            },
        }

    def _analyze_feature_importance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance for PV prediction."""
        feature_cols = [col for col in data.columns if col != "pv_power"]

        if not feature_cols or len(data) < 100:
            return {"warning": "Insufficient features or data for importance analysis"}

        # Prepare features
        X = data[feature_cols].select_dtypes(include=[np.number])
        y = data["pv_power"]

        # Remove any remaining NaN values
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 50:
            return {"warning": "Insufficient clean data for feature importance"}

        # Random Forest for feature importance
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_clean, y_clean)

            feature_importance = dict(zip(X_clean.columns, rf.feature_importances_))
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "feature_importance": sorted_importance,
                "top_5_features": list(sorted_importance.keys())[:5],
                "model_score": rf.score(X_clean, y_clean),
                "features_analyzed": list(X_clean.columns),
            }

        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")
            return {"warning": f"Feature importance analysis failed: {str(e)}"}

    def _evaluate_prediction_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate different prediction models for PV production."""
        feature_cols = [col for col in data.columns if col != "pv_power"]

        if not feature_cols or len(data) < 200:
            return {"warning": "Insufficient data for model evaluation (need at least 200 records)"}

        # Prepare data
        X = data[feature_cols].select_dtypes(include=[np.number])
        y = data["pv_power"]

        # Remove NaN values
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            return {"warning": "Insufficient clean data for model evaluation"}

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Models to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}

        for name, model in models.items():
            scores = {"rmse": [], "mae": [], "r2": []}

            try:
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    scores["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    scores["mae"].append(mean_absolute_error(y_test, y_pred))
                    scores["r2"].append(r2_score(y_test, y_pred))

                results[name] = {
                    "mean_rmse": np.mean(scores["rmse"]),
                    "std_rmse": np.std(scores["rmse"]),
                    "mean_mae": np.mean(scores["mae"]),
                    "std_mae": np.std(scores["mae"]),
                    "mean_r2": np.mean(scores["r2"]),
                    "std_r2": np.std(scores["r2"]),
                }

            except Exception as e:
                self.logger.warning(f"Model {name} evaluation failed: {e}")
                results[name] = {"error": str(e)}

        # Best model selection
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_model = min(valid_models.items(), key=lambda x: x[1]["mean_rmse"])
            results["best_model"] = best_model[0]
            results["best_performance"] = best_model[1]

        return results
