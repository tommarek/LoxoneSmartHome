"""
PV Production Pattern Analysis for PEMS v2.

Analyzes PV production patterns with Loxone integration:
1. Correlate with weather data (solar radiation, cloud cover, temperature)
2. Identify seasonal patterns using STL decomposition
3. Create feature engineering pipeline
4. Test models for production forecasting
5. Handle Loxone field naming conventions
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
# Import Loxone adapter
from analysis.utils.loxone_adapter import (LoxoneDataIntegrator,
                                           LoxoneFieldAdapter)
from scipy import stats
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
        self.loxone_adapter = LoxoneFieldAdapter()

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
        self.logger.info("Starting PV production analysis...")

        # --- Input Validation Logging ---
        if pv_data.empty:
            self.logger.error("PV data is empty. Aborting PV analysis.")
            return {"error": "Empty PV data"}
        if weather_data.empty:
            self.logger.warning(
                "Weather data is empty. Weather correlation analysis will be limited."
            )

        self.logger.debug(
            f"PV data shape: {pv_data.shape}, columns: {pv_data.columns.tolist()}"
        )
        self.logger.debug(
            f"Weather data shape: {weather_data.shape}, columns: {weather_data.columns.tolist()}"
        )

        results = {}

        # Merge PV and weather data
        merged_data = self._merge_pv_weather_data(pv_data, weather_data)

        if merged_data.empty:
            self.logger.warning("No merged PV-weather data available")
            return {}

        # Basic statistics
        results["basic_stats"] = self._calculate_basic_statistics(merged_data)

        # Weather correlations
        results["weather_correlations"] = self._analyze_weather_correlations(
            merged_data
        )

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
        results["prediction_performance"] = self._evaluate_prediction_models(
            merged_data
        )

        # Export policy detection (if price data available in weather_data)
        if not weather_data.empty and "price_czk_kwh" in weather_data.columns:
            self.logger.info("Analyzing export policy periods...")
            results["export_policy"] = self._identify_export_periods(
                pv_data, weather_data
            )

            # Analyze pre/post export periods separately
            export_periods = results["export_policy"]
            if "policy_change_date" in export_periods:
                policy_date = export_periods["policy_change_date"]
                results["pre_export_analysis"] = self._analyze_self_consumption_period(
                    pv_data, policy_date
                )
                results[
                    "post_export_analysis"
                ] = self._analyze_conditional_export_period(
                    pv_data, weather_data, policy_date
                )
                results[
                    "optimization_potential"
                ] = self._calculate_optimization_potential(
                    pv_data, weather_data, policy_date
                )

        self.logger.info("PV production analysis completed")
        return results

    def _get_loxone_field(
        self, df: pd.DataFrame, field_type: str, room_name: str = None
    ) -> Optional[str]:
        """
        Find Loxone field name for a given type.

        Args:
            df: DataFrame to search in
            field_type: Type of field ('temperature', 'humidity', 'relay', etc.)
            room_name: Optional room name for room-specific fields

        Returns:
            Column name if found, None otherwise
        """
        return self.loxone_adapter._find_field(
            df.columns.tolist(), field_type, room_name
        )

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
            # Standardize weather data to include Loxone solar fields
            standardized_weather = LoxoneFieldAdapter.standardize_weather_data(
                weather_data
            )

            # Resample weather data to match PV data frequency
            weather_resampled = standardized_weather.resample("15min").interpolate(
                method="linear"
            )
            merged = pv_clean.join(weather_resampled, how="inner")

            # Log available solar fields from Loxone
            solar_fields = ["sun_elevation", "sun_direction", "solar_irradiance"]
            available_solar = [f for f in solar_fields if f in merged.columns]
            if available_solar:
                self.logger.info(
                    f"Enhanced weather data with Loxone solar fields: {available_solar}"
                )
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
            daylight_data = data[(data.index.hour >= 6) & (data.index.hour <= 18)][
                "pv_power"
            ]

        stats = {
            "total_records": len(data),
            "daylight_records": len(daylight_data),
            "max_power": pv_power.max(),
            "mean_power": pv_power.mean(),
            "mean_daylight_power": daylight_data.mean()
            if len(daylight_data) > 0
            else 0,
            "capacity_factor": pv_power.mean() / pv_power.max()
            if pv_power.max() > 0
            else 0,
            "daylight_capacity_factor": (
                daylight_data.mean() / pv_power.max()
                if pv_power.max() > 0 and len(daylight_data) > 0
                else 0
            ),
            "total_energy_kwh": (pv_power.sum() * 0.25)
            / 1000,  # 15-min intervals to kWh
            "peak_months": pv_power.groupby(pv_power.index.month)
            .mean()
            .nlargest(3)
            .index.tolist(),
            "low_months": pv_power.groupby(pv_power.index.month)
            .mean()
            .nsmallest(3)
            .index.tolist(),
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
            if (
                col in data.columns and data[col].notna().sum() > 100
            ):  # Minimum data points
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
            sorted(
                correlations.items(),
                key=lambda x: abs(x[1]["correlation"]),
                reverse=True,
            )
        )

        # Identify strongest positive and negative correlations
        strongest_positive = (
            max(correlations.items(), key=lambda x: x[1]["correlation"])
            if correlations
            else None
        )
        strongest_negative = (
            min(correlations.items(), key=lambda x: x[1]["correlation"])
            if correlations
            else None
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
            return {
                "warning": "Insufficient data for seasonal analysis (need at least 1 year)"
            }

        # Resample to daily data for seasonal analysis
        daily_data = data["pv_power"].resample("D").mean().dropna()

        # Check if we have enough non-null daily data points
        if (
            len(daily_data) < 730
        ):  # At least 2 years of daily data for reliable seasonal patterns
            return {
                "warning": f"Insufficient daily data points for seasonal analysis (have {len(daily_data)}, need at least 730)"
            }

        # STL decomposition
        try:
            # Use a smaller seasonal period if we don't have enough data for annual seasonality
            seasonal_period = min(365, len(daily_data) // 3)  # At least 3 cycles
            if seasonal_period < 7:  # If less than a week, try weekly pattern
                seasonal_period = 7

            stl = STL(daily_data, seasonal=seasonal_period)
            result = stl.fit()

            seasonal_patterns = {
                "trend": result.trend.to_dict(),
                "seasonal": result.seasonal.to_dict(),
                "residual": result.resid.to_dict(),
                "seasonal_strength": 1
                - (result.resid.var() / (result.seasonal + result.resid).var()),
                "trend_strength": 1
                - (result.resid.var() / (result.trend + result.resid).var()),
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
                    seasonal_profiles[season] = season_data.groupby(
                        season_data.index.hour
                    )["pv_power"].mean()

            return {
                "decomposition": seasonal_patterns,
                "monthly_profiles": monthly_profiles.to_dict(),
                "seasonal_profiles": seasonal_profiles,
                "peak_season": (
                    max(
                        seasons.keys(),
                        key=lambda s: data[data.index.month.isin(seasons[s])][
                            "pv_power"
                        ].mean(),
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
        # clear_sky_max is calculated but not used in current implementation
        # clear_sky_max = clear_sky_data.groupby(
        #     [clear_sky_data.index.month, clear_sky_data.index.hour]
        # )["pv_power"].quantile(0.95)

        analysis = {
            "clear_sky_records": len(clear_sky_data),
            "clear_sky_percentage": len(clear_sky_data) / len(data) * 100,
            "clear_sky_mean_power": clear_sky_data["pv_power"].mean(),
            "clear_sky_max_power": clear_sky_data["pv_power"].max(),
            "cloudy_sky_mean_power": data[~clear_sky_mask]["pv_power"].mean(),
            "clear_sky_advantage": (
                (
                    clear_sky_data["pv_power"].mean()
                    / data[~clear_sky_mask]["pv_power"].mean()
                )
                if data[~clear_sky_mask]["pv_power"].mean() > 0
                else float("inf")
            ),
        }

        return analysis

    def _analyze_efficiency_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze PV efficiency patterns."""
        if "temperature" not in data.columns or "solar_elevation" not in data.columns:
            return {
                "warning": (
                    "Insufficient data for efficiency analysis "
                    "(need temperature and solar elevation)"
                )
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
            performance_ratio = (
                meaningful_data["pv_power"] / meaningful_data["solar_elevation"]
            )
            performance_stats = {
                "mean_performance_ratio": performance_ratio.mean(),
                "performance_ratio_std": performance_ratio.std(),
                "low_performance_threshold": performance_ratio.quantile(0.1),
                "high_performance_threshold": performance_ratio.quantile(0.9),
            }
        else:
            performance_stats = {
                "warning": "Insufficient data for performance ratio calculation"
            }

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
            (daylight_data["pv_power"] < lower_bound)
            | (daylight_data["pv_power"] > upper_bound)
        ]

        # Zero production during daylight (potential system issues)
        zero_production = daylight_data[daylight_data["pv_power"] == 0]

        return {
            "total_anomalies": len(anomalies),
            "anomaly_percentage": len(anomalies) / len(daylight_data) * 100,
            "zero_production_events": len(zero_production),
            "anomaly_dates": anomalies.index.date.tolist()[
                :20
            ],  # First 20 anomaly dates
            "largest_anomaly": {
                "timestamp": anomalies["pv_power"].idxmax()
                if not anomalies.empty
                else None,
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
            return {
                "warning": "Insufficient data for model evaluation (need at least 200 records)"
            }

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
            "Random Forest": RandomForestRegressor(
                n_estimators=50, random_state=42, n_jobs=-1
            ),
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

    def _identify_export_periods(
        self, pv_data: pd.DataFrame, price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect when PV export policy changed from disabled to price-based.

        Analyzes export patterns to identify the transition date.
        """
        if "ExportPower" not in pv_data.columns:
            return {"warning": "No export power data available"}

        # Calculate daily export amounts
        daily_export = pv_data.resample("D")["ExportPower"].sum()

        # Find the first significant export day (policy change)
        export_threshold = daily_export.quantile(0.1)  # 10th percentile as threshold
        first_export_dates = daily_export[daily_export > export_threshold].index

        if len(first_export_dates) == 0:
            return {"warning": "No significant export periods found"}

        policy_change_date = first_export_dates[0]

        # Analyze export pattern consistency
        post_change_data = pv_data[pv_data.index >= policy_change_date]
        export_days = (post_change_data.resample("D")["ExportPower"].sum() > 0).sum()
        total_days = len(post_change_data.resample("D").first())
        export_consistency = export_days / total_days if total_days > 0 else 0

        return {
            "policy_change_date": policy_change_date,
            "export_consistency": export_consistency,
            "days_analyzed": total_days,
            "export_days": export_days,
            "detection_method": "first_significant_export",
        }

    def _analyze_self_consumption_period(
        self, pv_data: pd.DataFrame, export_enabled_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Analyze forced self-consumption period when export was disabled.

        Focus on curtailment losses and self-consumption optimization.
        """
        # Extract pre-export period
        pre_export = pv_data[pv_data.index < export_enabled_date]

        if pre_export.empty:
            return {"warning": "No pre-export period data available"}

        # Calculate self-consumption metrics
        if "SelfConsumption" in pre_export.columns:
            total_production = pre_export["InputPower"].sum()
            total_self_consumption = pre_export["SelfConsumption"].sum()
            self_consumption_ratio = (
                total_self_consumption / total_production if total_production > 0 else 0
            )

            # Estimate curtailment (production that couldn't be used or exported)
            curtailment = total_production - total_self_consumption
            curtailment_ratio = (
                curtailment / total_production if total_production > 0 else 0
            )
        else:
            # Estimate self-consumption if not directly available
            # Assume all production was self-consumed (worst case for curtailment analysis)
            total_production = pre_export["InputPower"].sum()
            estimated_consumption = total_production  # Conservative estimate
            curtailment = 0  # Unknown without consumption data
            self_consumption_ratio = 1.0
            curtailment_ratio = 0.0

        # Analyze peak production vs consumption patterns
        daily_production = pre_export.resample("D")["InputPower"].sum()

        # Identify high-production days (potential curtailment candidates)
        high_production_threshold = daily_production.quantile(0.8)
        high_production_days = daily_production[
            daily_production > high_production_threshold
        ]

        # Calculate seasonal patterns
        monthly_production = pre_export.resample("ME")["InputPower"].sum()
        peak_production_months = monthly_production.nlargest(3).index.month.tolist()

        return {
            "period_duration_days": len(pre_export.resample("D").first()),
            "total_production_kwh": total_production * 0.25 / 1000,  # Convert to kWh
            "self_consumption_ratio": self_consumption_ratio,
            "estimated_curtailment_kwh": curtailment * 0.25 / 1000,
            "curtailment_ratio": curtailment_ratio,
            "high_production_days": len(high_production_days),
            "peak_production_months": peak_production_months,
            "avg_daily_production": daily_production.mean() * 0.25 / 1000,
            "max_daily_production": daily_production.max() * 0.25 / 1000,
        }

    def _analyze_conditional_export_period(
        self,
        pv_data: pd.DataFrame,
        price_data: pd.DataFrame,
        export_enabled_date: pd.Timestamp,
    ) -> Dict[str, Any]:
        """
        Analyze price-based export effectiveness in the post-policy period.
        """
        # Extract post-export period
        post_export = pv_data[pv_data.index >= export_enabled_date]

        if post_export.empty:
            return {"warning": "No post-export period data available"}

        # Merge with price data
        if price_data.empty:
            return {"warning": "No price data available for export analysis"}

        # Align price data with PV data
        aligned_data = post_export.merge(
            price_data, left_index=True, right_index=True, how="inner"
        )

        if aligned_data.empty:
            return {"warning": "No aligned price-PV data for analysis"}

        # Analyze export behavior vs price
        export_threshold = 100  # CZK/MWh - adjust based on your system

        # Categorize by price levels
        high_price_mask = aligned_data["price_czk_kwh"] >= export_threshold
        low_price_mask = aligned_data["price_czk_kwh"] < export_threshold

        high_price_data = aligned_data[high_price_mask]
        low_price_data = aligned_data[low_price_mask]

        # Export statistics by price category
        export_stats = {
            "total_periods": len(aligned_data),
            "high_price_periods": len(high_price_data),
            "low_price_periods": len(low_price_data),
            "export_threshold_czk_mwh": export_threshold,
        }

        if "ExportPower" in aligned_data.columns:
            export_stats.update(
                {
                    "total_export_kwh": aligned_data["ExportPower"].sum() * 0.25 / 1000,
                    "high_price_export_kwh": high_price_data["ExportPower"].sum()
                    * 0.25
                    / 1000
                    if not high_price_data.empty
                    else 0,
                    "low_price_export_kwh": low_price_data["ExportPower"].sum()
                    * 0.25
                    / 1000
                    if not low_price_data.empty
                    else 0,
                    "export_when_high_price_ratio": (
                        high_price_data["ExportPower"] > 0
                    ).mean()
                    if not high_price_data.empty
                    else 0,
                    "export_when_low_price_ratio": (
                        low_price_data["ExportPower"] > 0
                    ).mean()
                    if not low_price_data.empty
                    else 0,
                }
            )

        # Price correlation analysis
        if "ExportPower" in aligned_data.columns:
            price_export_correlation = aligned_data["price_czk_kwh"].corr(
                aligned_data["ExportPower"]
            )
            export_stats["price_export_correlation"] = price_export_correlation

        # Calculate revenue from exports
        if "ExportPower" in aligned_data.columns:
            # Estimate revenue (simplified calculation)
            aligned_data["export_revenue"] = (
                aligned_data["ExportPower"]
                * aligned_data["price_czk_kwh"]
                * 0.25
                / 1000
                / 1000
            )  # CZK
            total_revenue = aligned_data["export_revenue"].sum()
            export_stats["estimated_total_revenue_czk"] = total_revenue

            # Revenue efficiency (revenue per kWh exported)
            total_export_kwh = aligned_data["ExportPower"].sum() * 0.25 / 1000
            if total_export_kwh > 0:
                export_stats["revenue_per_kwh_czk"] = total_revenue / total_export_kwh

        return export_stats

    def _calculate_optimization_potential(
        self,
        pv_data: pd.DataFrame,
        price_data: pd.DataFrame,
        export_enabled_date: pd.Timestamp,
    ) -> Dict[str, Any]:
        """
        Quantify lost opportunities and improvement potential.
        """
        pre_export_analysis = self._analyze_self_consumption_period(
            pv_data, export_enabled_date
        )
        post_export_analysis = self._analyze_conditional_export_period(
            pv_data, price_data, export_enabled_date
        )

        if "warning" in pre_export_analysis or "warning" in post_export_analysis:
            return {
                "warning": "Cannot calculate optimization potential due to insufficient data"
            }

        # Calculate potential revenue from curtailed energy
        curtailed_kwh = pre_export_analysis.get("estimated_curtailment_kwh", 0)

        # Estimate average market price during pre-export period
        pre_export_period = pv_data[pv_data.index < export_enabled_date]
        if not price_data.empty and not pre_export_period.empty:
            aligned_pre = price_data.loc[
                pre_export_period.index[0] : pre_export_period.index[-1]
            ]
            avg_price_pre = (
                aligned_pre["price_czk_kwh"].mean() if not aligned_pre.empty else 2.0
            )  # CZK/kWh fallback
        else:
            avg_price_pre = 2.0  # CZK/kWh fallback

        lost_revenue_curtailment = (
            curtailed_kwh * avg_price_pre / 1000
        )  # Convert to CZK

        # Calculate improvement opportunities
        optimization_potential = {
            "lost_revenue_curtailment_czk": lost_revenue_curtailment,
            "curtailed_energy_kwh": curtailed_kwh,
            "pre_export_period_days": pre_export_analysis.get(
                "period_duration_days", 0
            ),
            "current_export_revenue_czk": post_export_analysis.get(
                "estimated_total_revenue_czk", 0
            ),
            "price_threshold_optimization": export_enabled_date,
        }

        # Storage optimization potential
        if curtailed_kwh > 0:
            # Estimate benefit of battery storage to capture curtailed energy
            storage_efficiency = 0.85  # Round-trip efficiency
            usable_storage = curtailed_kwh * storage_efficiency
            storage_value = usable_storage * avg_price_pre / 1000
            optimization_potential["storage_value_potential_czk"] = storage_value

        # Load shifting potential
        total_production = pre_export_analysis.get("total_production_kwh", 0)
        if total_production > 0:
            # Estimate value of better load management
            load_shift_potential = (
                total_production * 0.1 * avg_price_pre / 1000
            )  # 10% improvement assumption
            optimization_potential["load_shift_potential_czk"] = load_shift_potential

        return optimization_potential


class RelayPatternAnalyzer:
    """Enhanced pattern analysis specifically for relay-controlled heating systems."""

    def __init__(self):
        """Initialize the relay pattern analyzer."""
        self.logger = logging.getLogger(f"{__name__}.RelayPatternAnalyzer")
        self.loxone_integrator = LoxoneDataIntegrator()

    def analyze_relay_patterns(
        self,
        relay_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame = None,
        price_data: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive relay pattern analysis for optimization opportunities.

        Args:
            relay_data: Dictionary with room relay data
            weather_data: Weather conditions for correlation analysis
            price_data: Electricity prices for cost optimization

        Returns:
            Dictionary with pattern analysis results
        """
        self.logger.info("Starting relay pattern analysis...")

        # --- Input Validation Logging ---
        if relay_data is None:
            self.logger.error("Relay data is None. Aborting relay pattern analysis.")
            return {"error": "No relay data provided"}

        if isinstance(relay_data, pd.DataFrame) and relay_data.empty:
            self.logger.error(
                "Relay data DataFrame is empty. Aborting relay pattern analysis."
            )
            return {"error": "No relay data provided"}

        if isinstance(relay_data, dict) and not relay_data:
            self.logger.error(
                "Relay data dictionary is empty. Aborting relay pattern analysis."
            )
            return {"error": "No relay data provided"}

        if weather_data is None or weather_data.empty:
            self.logger.warning(
                "Weather data is empty. Weather correlation analysis will be limited."
            )
        if price_data is None or price_data.empty:
            self.logger.warning(
                "Price data is empty. Economic optimization analysis will be limited."
            )

        if isinstance(relay_data, dict):
            self.logger.debug(f"Relay data rooms: {list(relay_data.keys())}")
            for room, df in relay_data.items():
                if isinstance(df, pd.DataFrame):
                    self.logger.debug(f"Room {room} data shape: {df.shape}")
        elif isinstance(relay_data, pd.DataFrame):
            self.logger.debug(
                f"Relay data shape: {relay_data.shape}, columns: {relay_data.columns.tolist()}"
            )

        if weather_data is not None and not weather_data.empty:
            self.logger.debug(
                f"Weather data shape: {weather_data.shape}, columns: {weather_data.columns.tolist()}"
            )
        if price_data is not None and not price_data.empty:
            self.logger.debug(
                f"Price data shape: {price_data.shape}, columns: {price_data.columns.tolist()}"
            )

        # Standardize relay data using Loxone adapter
        self.logger.info("Standardizing relay data for analysis")
        standardized_relay_data = self.loxone_integrator.prepare_relay_analysis_data(
            relay_data
        )

        results = {}

        # Peak demand analysis
        results["peak_demand"] = self._analyze_peak_demand_patterns(
            standardized_relay_data
        )

        # Relay coordination opportunities
        results["coordination"] = self._analyze_relay_coordination(
            standardized_relay_data
        )

        # Switching pattern analysis
        results["switching_patterns"] = self._analyze_switching_patterns(
            standardized_relay_data
        )

        # Weather correlation analysis
        if weather_data is not None and not weather_data.empty:
            results["weather_correlation"] = self._analyze_weather_correlation(
                standardized_relay_data, weather_data
            )

        # Economic optimization opportunities
        if price_data is not None and not price_data.empty:
            results["economic_optimization"] = self._analyze_economic_patterns(
                standardized_relay_data, price_data
            )

        # Load distribution analysis
        results["load_distribution"] = self._analyze_load_distribution(
            standardized_relay_data
        )

        return results

    def _analyze_peak_demand_patterns(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze peak demand patterns and reduction opportunities."""
        # Calculate total system load
        total_load = self._calculate_total_system_load(relay_data)

        if total_load.empty:
            return {"warning": "No valid relay data for peak analysis"}

        # Identify peak events (top 5% of load)
        peak_threshold = total_load.quantile(0.95)
        peak_events = total_load[total_load >= peak_threshold]

        # Analyze peak characteristics
        peak_analysis = {
            "peak_threshold_kw": peak_threshold / 1000,
            "peak_events_count": len(peak_events),
            "max_peak_kw": total_load.max() / 1000,
            "avg_peak_kw": peak_events.mean() / 1000,
            "peak_frequency": len(peak_events) / len(total_load) * 100,
        }

        # Peak timing analysis
        peak_hours = peak_events.index.hour.value_counts()
        peak_analysis["peak_hours_distribution"] = peak_hours.to_dict()
        peak_analysis["most_common_peak_hour"] = peak_hours.index[0]

        # Weekday vs weekend patterns
        peak_weekdays = peak_events[peak_events.index.weekday < 5]
        peak_weekends = peak_events[peak_events.index.weekday >= 5]

        peak_analysis["weekday_peaks"] = len(peak_weekdays)
        peak_analysis["weekend_peaks"] = len(peak_weekends)
        peak_analysis["weekday_peak_ratio"] = (
            len(peak_weekdays) / len(peak_events) * 100
        )

        # Room contribution to peaks
        room_contributions = {}
        for room_name, room_df in relay_data.items():
            if not room_df.empty:
                # Find relay state column
                relay_col = self._find_relay_column(room_df)
                if relay_col:
                    room_states = room_df[relay_col]
                    # Calculate contribution during peak events
                    peak_indices = peak_events.index.intersection(room_states.index)
                    if len(peak_indices) > 0:
                        contribution = (
                            (room_states.loc[peak_indices] > 0).sum()
                            / len(peak_indices)
                            * 100
                        )
                        room_contributions[room_name] = contribution

        peak_analysis["room_contributions"] = room_contributions

        # Peak reduction potential
        # Estimate reduction from preventing simultaneous operation
        simultaneous_count = self._calculate_simultaneous_operations(relay_data)
        avg_simultaneous_at_peak = simultaneous_count.loc[peak_events.index].mean()

        peak_analysis["avg_simultaneous_relays_at_peak"] = avg_simultaneous_at_peak
        peak_analysis["estimated_reduction_potential_kw"] = (
            avg_simultaneous_at_peak * 2.0 * 0.2
        )  # 20% reduction

        return peak_analysis

    def _analyze_relay_coordination(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze opportunities for relay coordination."""
        # Create correlation matrix
        relay_states_df = self._create_relay_states_dataframe(relay_data)

        if relay_states_df.empty:
            return {"warning": "Insufficient data for coordination analysis"}

        correlation_matrix = relay_states_df.corr()

        # Find high correlation pairs (coordination candidates)
        coordination_opportunities = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > 0.6:  # High correlation threshold
                    room1 = correlation_matrix.columns[i]
                    room2 = correlation_matrix.columns[j]

                    # Calculate potential savings
                    simultaneous_hours = (
                        (relay_states_df[room1] > 0) & (relay_states_df[room2] > 0)
                    ).sum()
                    total_hours = len(relay_states_df)
                    simultaneity_rate = simultaneous_hours / total_hours * 100

                    coordination_opportunities.append(
                        {
                            "room_pair": f"{room1} + {room2}",
                            "correlation": corr_value,
                            "simultaneity_rate_percent": simultaneity_rate,
                            "coordination_potential": "HIGH"
                            if simultaneity_rate > 30
                            else "MEDIUM",
                        }
                    )

        # Sort by correlation strength
        coordination_opportunities.sort(key=lambda x: x["correlation"], reverse=True)

        # Calculate system-wide coordination metrics
        total_rooms = len(relay_states_df.columns)
        avg_correlation = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()

        coordination_analysis = {
            "total_coordination_opportunities": len(coordination_opportunities),
            "high_priority_pairs": [
                opp
                for opp in coordination_opportunities
                if opp["coordination_potential"] == "HIGH"
            ],
            "average_room_correlation": avg_correlation,
            "coordination_opportunities": coordination_opportunities[:10],  # Top 10
            "system_diversity_score": 1
            - avg_correlation,  # Higher = more diverse operation
        }

        return coordination_analysis

    def _analyze_economic_patterns(
        self, relay_data: Dict[str, pd.DataFrame], price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze economic optimization opportunities based on electricity prices."""
        # Calculate total system load
        total_load = self._calculate_total_system_load(relay_data)

        if total_load.empty or price_data.empty:
            return {"warning": "Insufficient data for economic analysis"}

        # Align load and price data
        aligned_data = pd.merge_asof(
            total_load.to_frame("load_w"),
            price_data,
            left_index=True,
            right_index=True,
            direction="nearest",
        )

        if len(aligned_data) < 100:
            return {"warning": "Insufficient aligned price-load data"}

        # Calculate current operating costs
        aligned_data["cost"] = (
            aligned_data["load_w"] * aligned_data["price_czk_kwh"] / 1000000
        )  # Convert to CZK

        # Analyze price-load correlation
        price_load_correlation = aligned_data["load_w"].corr(
            aligned_data["price_czk_kwh"]
        )

        # Price-based operation analysis
        price_quartiles = aligned_data["price_czk_kwh"].quantile([0.25, 0.5, 0.75])

        load_by_price_quartile = {
            "Q1_lowest_prices": aligned_data[
                aligned_data["price_czk_kwh"] <= price_quartiles[0.25]
            ]["load_w"].mean(),
            "Q2_low_prices": aligned_data[
                (aligned_data["price_czk_kwh"] > price_quartiles[0.25])
                & (aligned_data["price_czk_kwh"] <= price_quartiles[0.5])
            ]["load_w"].mean(),
            "Q3_high_prices": aligned_data[
                (aligned_data["price_czk_kwh"] > price_quartiles[0.5])
                & (aligned_data["price_czk_kwh"] <= price_quartiles[0.75])
            ]["load_w"].mean(),
            "Q4_highest_prices": aligned_data[
                aligned_data["price_czk_kwh"] > price_quartiles[0.75]
            ]["load_w"].mean(),
        }

        # Calculate current costs
        total_energy_kwh = (
            aligned_data["load_w"].sum() * (5 / 60) / 1000
        )  # 5-minute intervals to kWh
        total_cost_czk = aligned_data["cost"].sum() * (5 / 60)  # 5-minute intervals
        avg_price_czk_kwh = (
            total_cost_czk / total_energy_kwh if total_energy_kwh > 0 else 0
        )

        # Optimization potential calculation
        # Simulate shifting 30% of high-price load to low-price periods
        high_price_mask = aligned_data["price_czk_kwh"] > price_quartiles[0.75]
        low_price_mask = aligned_data["price_czk_kwh"] <= price_quartiles[0.25]

        high_price_load = aligned_data.loc[high_price_mask, "load_w"].sum()
        shifted_load = high_price_load * 0.3

        # Calculate savings from load shifting
        avg_high_price = aligned_data.loc[high_price_mask, "price_czk_kwh"].mean()
        avg_low_price = aligned_data.loc[low_price_mask, "price_czk_kwh"].mean()

        potential_savings_czk = (
            shifted_load * (avg_high_price - avg_low_price) * (5 / 60) / 1000000
        )  # Convert to CZK

        # Annualize the savings
        days_analyzed = (aligned_data.index[-1] - aligned_data.index[0]).days
        annual_savings_czk = (
            potential_savings_czk * (365 / days_analyzed) if days_analyzed > 0 else 0
        )

        economic_analysis = {
            "current_operation": {
                "total_energy_kwh": total_energy_kwh,
                "total_cost_czk": total_cost_czk,
                "avg_price_czk_kwh": avg_price_czk_kwh,
                "price_load_correlation": price_load_correlation,
            },
            "load_by_price_quartile": load_by_price_quartile,
            "optimization_potential": {
                "load_shifting_savings_czk_annual": annual_savings_czk,
                "savings_percentage": (
                    annual_savings_czk / (total_cost_czk * 365 / days_analyzed) * 100
                )
                if total_cost_czk > 0 and days_analyzed > 0
                else 0,
                "optimal_operation_hours": aligned_data.loc[low_price_mask]
                .index.hour.value_counts()
                .head(5)
                .to_dict(),
            },
            "recommendations": {
                "load_shifting_potential": "HIGH"
                if price_load_correlation > 0.2
                else "MEDIUM",
                "target_shift_percentage": 30,
                "focus_hours": list(
                    aligned_data.loc[low_price_mask]
                    .index.hour.value_counts()
                    .head(3)
                    .index
                ),
            },
        }

        return economic_analysis

    def _analyze_switching_patterns(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze relay switching patterns."""
        if not relay_data:
            return {"warning": "No relay data for switching analysis"}

        switching_analysis = {}

        for room_name, relay_df in relay_data.items():
            if relay_df.empty or "relay_state" not in relay_df.columns:
                continue

            relay_states = relay_df["relay_state"]

            # Calculate switching events
            switches = relay_states.diff().abs()
            total_switches = switches.sum()

            # Calculate switching frequency (switches per day)
            days = (relay_states.index[-1] - relay_states.index[0]).days
            switch_frequency = total_switches / max(days, 1)

            # Analyze switching patterns by hour
            hourly_switches = switches.groupby(switches.index.hour).sum()

            switching_analysis[room_name] = {
                "total_switches": total_switches,
                "switches_per_day": switch_frequency,
                "peak_switching_hour": hourly_switches.idxmax()
                if not hourly_switches.empty
                else None,
                "hourly_pattern": hourly_switches.to_dict(),
            }

        return switching_analysis

    def _analyze_weather_correlation(
        self, relay_data: Dict[str, pd.DataFrame], weather_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze correlation between relay usage and weather."""
        if weather_data.empty:
            return {"warning": "No weather data for correlation analysis"}

        correlations = {}

        # Calculate total heating power
        total_heating = pd.Series(0, index=weather_data.index)
        for room_name, relay_df in relay_data.items():
            if not relay_df.empty and "power_kw" in relay_df.columns:
                room_power = (
                    relay_df["power_kw"]
                    .reindex(weather_data.index, method="nearest")
                    .fillna(0)
                )
                total_heating += room_power

        if total_heating.sum() > 0 and "temperature" in weather_data.columns:
            # Calculate correlation with outdoor temperature
            temp_correlation = total_heating.corr(weather_data["temperature"])
            correlations["temperature_correlation"] = temp_correlation

            # Analyze heating response by temperature bins
            temp_bins = pd.cut(weather_data["temperature"], bins=10)
            heating_by_temp = total_heating.groupby(temp_bins).mean()
            correlations["heating_response_by_temp"] = heating_by_temp.to_dict()

        return correlations

    def _analyze_load_distribution(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze load distribution across rooms."""
        if not relay_data:
            return {"warning": "No relay data for load distribution analysis"}

        distribution = {}
        total_energy = 0

        for room_name, relay_df in relay_data.items():
            if relay_df.empty or "power_kw" not in relay_df.columns:
                continue

            # Calculate energy consumption (assuming 5-minute intervals)
            room_energy = (relay_df["power_kw"] * (5 / 60)).sum()  # kWh
            distribution[room_name] = room_energy
            total_energy += room_energy

        # Calculate percentages
        if total_energy > 0:
            for room in distribution:
                distribution[room] = {
                    "energy_kwh": distribution[room],
                    "percentage": (distribution[room] / total_energy) * 100,
                }

        return {
            "room_distribution": distribution,
            "total_energy_kwh": total_energy,
            "most_consuming_room": max(
                distribution.keys(), key=lambda x: distribution[x]["energy_kwh"]
            )
            if distribution
            else None,
        }

    # Helper methods
    def _calculate_total_system_load(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Calculate total system power load from all relays."""
        total_load = None

        # Power ratings are now handled by LoxoneFieldAdapter

        for room_name, room_df in relay_data.items():
            if room_df.empty:
                continue

            relay_col = self._find_relay_column(room_df)
            if not relay_col:
                continue

            # Get power rating for this room
            standard_room_name = LoxoneFieldAdapter.standardize_room_name(room_name)
            power_rating = (
                LoxoneFieldAdapter._get_room_power_rating(standard_room_name) * 1000
            )  # Convert kW to W

            # Calculate power consumption
            room_power = room_df[relay_col] * power_rating

            if total_load is None:
                total_load = room_power
            else:
                total_load = total_load.add(room_power, fill_value=0)

        return total_load if total_load is not None else pd.Series()

    def _find_relay_column(self, room_df: pd.DataFrame) -> str:
        """Find the relay state column in room data."""
        for col in room_df.columns:
            if any(
                keyword in col.lower()
                for keyword in ["relay", "state", "heating", "on"]
            ):
                return col
        return None

    def _create_relay_states_dataframe(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create a unified DataFrame with all room relay states."""
        relay_states = {}

        for room_name, room_df in relay_data.items():
            if room_df.empty:
                continue

            relay_col = self._find_relay_column(room_df)
            if relay_col:
                relay_states[room_name] = room_df[relay_col]

        if relay_states:
            return pd.DataFrame(relay_states)
        else:
            return pd.DataFrame()

    def _calculate_simultaneous_operations(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Calculate number of simultaneous relay operations."""
        relay_states_df = self._create_relay_states_dataframe(relay_data)

        if relay_states_df.empty:
            return pd.Series()

        return (relay_states_df > 0).sum(axis=1)

    def _analyze_temperature_efficiency(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze PV temperature efficiency patterns."""
        if pv_data.empty or weather_data.empty:
            return {"warning": "Insufficient data for temperature efficiency analysis"}

        # Find power and temperature columns
        power_col = None
        for col in pv_data.columns:
            if "power" in col.lower() and "export" not in col.lower():
                power_col = col
                break

        if power_col is None:
            return {"warning": "No power column found in PV data"}

        if "temperature" not in weather_data.columns:
            return {"warning": "No temperature column found in weather data"}

        # Align PV and weather data
        aligned_data = pd.merge_asof(
            pv_data[[power_col]].rename(columns={power_col: "pv_power"}),
            weather_data[["temperature"]],
            left_index=True,
            right_index=True,
            direction="nearest",
        )

        # Filter for daylight hours (PV production > 0)
        daylight_data = aligned_data[aligned_data["pv_power"] > 0]

        if len(daylight_data) < 100:
            return {"warning": "Insufficient daylight data for analysis"}

        # Temperature efficiency analysis
        temp_bins = pd.cut(
            daylight_data["temperature"],
            bins=8,
            labels=[
                "Very Cold (<0°C)",
                "Cold (0-5°C)",
                "Cool (5-10°C)",
                "Mild (10-15°C)",
                "Warm (15-20°C)",
                "Hot (20-25°C)",
                "Very Hot (25-30°C)",
                "Extreme (>30°C)",
            ],
        )

        efficiency_by_temp = (
            daylight_data.groupby(temp_bins)
            .agg({"pv_power": ["mean", "max", "count"]})
            .round(2)
        )

        # Calculate temperature coefficient
        # PV efficiency typically decreases by ~0.4-0.5% per °C above 25°C
        correlation = daylight_data["pv_power"].corr(daylight_data["temperature"])

        # Optimal temperature range
        temp_power_mean = daylight_data.groupby(
            pd.cut(daylight_data["temperature"], bins=20)
        )["pv_power"].mean()
        optimal_temp_range = temp_power_mean.idxmax()

        # Seasonal efficiency patterns
        seasonal_efficiency = {}
        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        }

        for season, months in seasons.items():
            season_data = daylight_data[daylight_data.index.month.isin(months)]
            if not season_data.empty:
                seasonal_efficiency[season] = {
                    "avg_power": season_data["pv_power"].mean(),
                    "avg_temperature": season_data["temperature"].mean(),
                    "efficiency_ratio": season_data["pv_power"].mean()
                    / daylight_data["pv_power"].mean(),
                }

        # Performance degradation analysis
        high_temp_threshold = 25  # °C
        high_temp_data = daylight_data[
            daylight_data["temperature"] > high_temp_threshold
        ]
        normal_temp_data = daylight_data[
            daylight_data["temperature"] <= high_temp_threshold
        ]

        degradation_analysis = {}
        if not high_temp_data.empty and not normal_temp_data.empty:
            degradation_analysis = {
                "high_temp_avg_power": high_temp_data["pv_power"].mean(),
                "normal_temp_avg_power": normal_temp_data["pv_power"].mean(),
                "degradation_percentage": (
                    (
                        normal_temp_data["pv_power"].mean()
                        - high_temp_data["pv_power"].mean()
                    )
                    / normal_temp_data["pv_power"].mean()
                    * 100
                ),
                "high_temp_hours": len(high_temp_data),
                "total_daylight_hours": len(daylight_data),
            }

        return {
            "temperature_correlation": correlation,
            "efficiency_by_temperature": efficiency_by_temp.to_dict(),
            "optimal_temperature_range": str(optimal_temp_range),
            "seasonal_efficiency": seasonal_efficiency,
            "degradation_analysis": degradation_analysis,
            "analysis_summary": {
                "total_daylight_hours": len(daylight_data),
                "temperature_range": f"{daylight_data['temperature'].min():.1f}°C to {daylight_data['temperature'].max():.1f}°C",
                "avg_daylight_power": daylight_data["pv_power"].mean(),
                "temperature_sensitivity": "HIGH"
                if abs(correlation) > 0.3
                else "MEDIUM"
                if abs(correlation) > 0.1
                else "LOW",
            },
        }

    def _detect_production_anomalies(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Detect anomalies in PV production patterns."""
        if pv_data.empty:
            return {"warning": "No PV data provided for anomaly detection"}

        # Find power column
        power_col = None
        for col in pv_data.columns:
            if "power" in col.lower() and "export" not in col.lower():
                power_col = col
                break

        if power_col is None:
            return {"warning": "No power column found in PV data"}

        pv_power = pv_data[power_col]

        # Filter daylight hours only
        daylight_mask = (pv_power.index.hour >= 6) & (pv_power.index.hour <= 18)
        daylight_power = pv_power[daylight_mask]

        if len(daylight_power) < 100:
            return {"warning": "Insufficient daylight data for anomaly detection"}

        # 1. Statistical anomaly detection
        Q1 = daylight_power.quantile(0.25)
        Q3 = daylight_power.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        statistical_anomalies = daylight_power[
            (daylight_power < lower_bound) | (daylight_power > upper_bound)
        ]

        # 2. Zero production during daylight
        zero_production = daylight_power[daylight_power == 0]

        # 3. Sudden drops (>50% from previous reading)
        power_changes = daylight_power.pct_change()
        sudden_drops = daylight_power[power_changes < -0.5]

        # 4. Expected vs actual based on time patterns
        expected_pattern = daylight_power.groupby(
            [daylight_power.index.month, daylight_power.index.hour]
        ).median()

        anomaly_threshold = 0.3  # 30% deviation from expected
        time_based_anomalies = []

        for timestamp, power in daylight_power.items():
            month, hour = timestamp.month, timestamp.hour
            if (month, hour) in expected_pattern.index:
                expected = expected_pattern.loc[(month, hour)]
                if expected > 0:
                    deviation = abs(power - expected) / expected
                    if deviation > anomaly_threshold:
                        time_based_anomalies.append(
                            {
                                "timestamp": timestamp,
                                "actual": power,
                                "expected": expected,
                                "deviation": deviation,
                            }
                        )

        # 5. Weather-based anomaly detection (if weather data available)
        weather_anomalies = {}
        if weather_data is not None and not weather_data.empty:
            if "cloud_cover" in weather_data.columns:
                # Align data
                aligned_data = pd.merge_asof(
                    daylight_power.to_frame("power"),
                    weather_data[["cloud_cover"]],
                    left_index=True,
                    right_index=True,
                    direction="nearest",
                )

                # High production with high cloud cover = anomaly
                high_power_cloudy = aligned_data[
                    (aligned_data["power"] > aligned_data["power"].quantile(0.8))
                    & (aligned_data["cloud_cover"] > 70)
                ]

                # Low production with clear skies = anomaly
                low_power_clear = aligned_data[
                    (aligned_data["power"] < aligned_data["power"].quantile(0.2))
                    & (aligned_data["cloud_cover"] < 30)
                ]

                weather_anomalies = {
                    "high_power_cloudy_periods": len(high_power_cloudy),
                    "low_power_clear_periods": len(low_power_clear),
                    "weather_based_anomalies": high_power_cloudy.index.tolist()
                    + low_power_clear.index.tolist(),
                }

        # Anomaly classification
        anomaly_types = {
            "statistical_outliers": {
                "count": len(statistical_anomalies),
                "percentage": len(statistical_anomalies) / len(daylight_power) * 100,
                "dates": statistical_anomalies.index.date.tolist()[:20],  # First 20
            },
            "zero_production": {
                "count": len(zero_production),
                "percentage": len(zero_production) / len(daylight_power) * 100,
                "dates": zero_production.index.date.tolist()[:20],
            },
            "sudden_drops": {
                "count": len(sudden_drops),
                "percentage": len(sudden_drops) / len(daylight_power) * 100,
                "dates": sudden_drops.index.date.tolist()[:20],
            },
            "time_pattern_anomalies": {
                "count": len(time_based_anomalies),
                "percentage": len(time_based_anomalies) / len(daylight_power) * 100,
                "examples": time_based_anomalies[:10],  # First 10 examples
            },
        }

        if weather_anomalies:
            anomaly_types["weather_anomalies"] = weather_anomalies

        # Overall anomaly assessment
        total_anomalies = (
            len(statistical_anomalies)
            + len(zero_production)
            + len(sudden_drops)
            + len(time_based_anomalies)
        )

        anomaly_severity = (
            "HIGH"
            if total_anomalies > len(daylight_power) * 0.05
            else "MEDIUM"
            if total_anomalies > len(daylight_power) * 0.02
            else "LOW"
        )

        return {
            "anomaly_types": anomaly_types,
            "total_anomalies": total_anomalies,
            "total_daylight_records": len(daylight_power),
            "overall_anomaly_rate": total_anomalies / len(daylight_power) * 100,
            "anomaly_severity": anomaly_severity,
            "recommendations": self._generate_anomaly_recommendations(
                anomaly_types, anomaly_severity
            ),
        }

    def _extract_seasonal_profiles(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract seasonal production profiles for forecasting."""
        if pv_data.empty:
            return {"warning": "No PV data provided for seasonal analysis"}

        # Find power column
        power_col = None
        for col in pv_data.columns:
            if "power" in col.lower() and "export" not in col.lower():
                power_col = col
                break

        if power_col is None:
            return {"warning": "No power column found in PV data"}

        pv_power = pv_data[power_col]

        # Need at least 6 months of data for meaningful seasonal analysis
        if len(pv_power) < 180 * 24 * 4:  # 180 days * 24 hours * 4 (15-min intervals)
            return {
                "warning": "Insufficient data for seasonal analysis (need at least 6 months)"
            }

        # Define seasons
        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        }

        seasonal_profiles = {}

        for season_name, months in seasons.items():
            season_data = pv_power[pv_power.index.month.isin(months)]

            if not season_data.empty:
                # Daily profiles (hourly averages)
                hourly_profile = (
                    season_data.groupby(season_data.index.hour)
                    .agg(
                        {
                            "mean": "mean",
                            "std": "std",
                            "min": "min",
                            "max": "max",
                            "count": "count",
                        }
                    )
                    .round(2)
                )

                # Weekly profiles
                weekly_profile = season_data.groupby(season_data.index.weekday).mean()

                # Monthly breakdown within season
                monthly_profile = (
                    season_data.groupby(season_data.index.month)
                    .agg({"mean": "mean", "sum": "sum", "count": "count"})
                    .round(2)
                )

                # Peak production analysis
                peak_hour = hourly_profile["mean"].idxmax()
                peak_power = hourly_profile["mean"].max()

                # Production duration analysis
                productive_hours = (hourly_profile["mean"] > 0).sum()

                # Variability analysis
                cv_hourly = (
                    hourly_profile["std"] / hourly_profile["mean"]
                )  # Coefficient of variation
                avg_variability = cv_hourly.mean()

                seasonal_profiles[season_name] = {
                    "hourly_profile": hourly_profile.to_dict(),
                    "weekly_profile": weekly_profile.to_dict(),
                    "monthly_profile": monthly_profile.to_dict(),
                    "statistics": {
                        "total_production_kwh": season_data.sum()
                        * 0.25
                        / 1000,  # Convert to kWh
                        "avg_daily_production_kwh": season_data.resample("D")
                        .sum()
                        .mean()
                        * 0.25
                        / 1000,
                        "peak_hour": peak_hour,
                        "peak_power_kw": peak_power / 1000,
                        "productive_hours_per_day": productive_hours,
                        "avg_variability": avg_variability,
                        "data_points": len(season_data),
                    },
                }

        # Year-over-year comparison (if multiple years available)
        yearly_comparison = {}
        years = pv_power.index.year.unique()

        if len(years) > 1:
            for year in years:
                year_data = pv_power[pv_power.index.year == year]
                if not year_data.empty:
                    yearly_comparison[year] = {
                        "total_production_kwh": year_data.sum() * 0.25 / 1000,
                        "avg_daily_production_kwh": year_data.resample("D").sum().mean()
                        * 0.25
                        / 1000,
                        "peak_power_kw": year_data.max() / 1000,
                        "capacity_factor": year_data.mean() / year_data.max()
                        if year_data.max() > 0
                        else 0,
                        "data_completeness": len(year_data)
                        / (365 * 24 * 4)
                        * 100,  # Assuming 15-min intervals
                    }

        # Trend analysis
        if len(years) > 1:
            yearly_totals = [
                yearly_comparison[year]["total_production_kwh"]
                for year in sorted(years)
            ]
            trend_analysis = {
                "yearly_totals": yearly_totals,
                "average_annual_growth": np.mean(np.diff(yearly_totals))
                if len(yearly_totals) > 1
                else 0,
                "trend_direction": "INCREASING"
                if yearly_totals[-1] > yearly_totals[0]
                else "DECREASING",
            }
        else:
            trend_analysis = {"warning": "Insufficient years for trend analysis"}

        # Generate forecasting templates
        forecasting_templates = {}
        for season_name, profile in seasonal_profiles.items():
            hourly_template = profile["hourly_profile"]["mean"]
            forecasting_templates[season_name] = {
                "hourly_template": hourly_template,
                "scaling_factors": {
                    "monthly": profile["monthly_profile"]["mean"],
                    "weekday_weekend_ratio": (
                        profile["weekly_profile"][list(range(5))].mean()
                        / profile["weekly_profile"][  # Weekdays
                            list(range(5, 7))
                        ].mean()  # Weekends
                        if profile["weekly_profile"][list(range(5, 7))].mean() > 0
                        else 1
                    ),
                },
            }

        return {
            "seasonal_profiles": seasonal_profiles,
            "yearly_comparison": yearly_comparison,
            "trend_analysis": trend_analysis,
            "forecasting_templates": forecasting_templates,
            "analysis_metadata": {
                "total_data_points": len(pv_power),
                "analysis_period": f"{pv_power.index.min().date()} to {pv_power.index.max().date()}",
                "years_analyzed": len(years),
                "seasons_with_data": len(seasonal_profiles),
            },
        }

    def _generate_anomaly_recommendations(
        self, anomaly_types: Dict[str, Any], severity: str
    ) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []

        # Zero production recommendations
        zero_count = anomaly_types.get("zero_production", {}).get("count", 0)
        if zero_count > 10:
            recommendations.append(
                f"Investigate {zero_count} periods of zero production during daylight hours. "
                "Check for inverter issues, shading, or maintenance activities."
            )

        # Sudden drops recommendations
        drops_count = anomaly_types.get("sudden_drops", {}).get("count", 0)
        if drops_count > 5:
            recommendations.append(
                f"Investigate {drops_count} sudden power drops (>50%). "
                "This could indicate equipment malfunctions or grid issues."
            )

        # Weather anomaly recommendations
        if "weather_anomalies" in anomaly_types:
            weather_count = anomaly_types["weather_anomalies"].get(
                "high_power_cloudy_periods", 0
            ) + anomaly_types["weather_anomalies"].get("low_power_clear_periods", 0)
            if weather_count > 0:
                recommendations.append(
                    f"Review {weather_count} weather-inconsistent production periods. "
                    "Verify weather data accuracy or check for local shading effects."
                )

        # General recommendations based on severity
        if severity == "HIGH":
            recommendations.append(
                "HIGH anomaly rate detected. Consider comprehensive system inspection "
                "and performance monitoring setup."
            )
        elif severity == "MEDIUM":
            recommendations.append(
                "Moderate anomaly rate. Regular monitoring and periodic maintenance recommended."
            )

        return recommendations
