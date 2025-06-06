"""
Base Load Analysis for PEMS v2.

Analyzes non-controllable load patterns:
1. Separate base load from total consumption
2. Identify patterns (weekday vs weekend, seasonal, time-of-day)
3. Model selection for prediction
4. Anomaly detection and clustering
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


class BaseLoadAnalyzer:
    """Analyze base load consumption patterns."""

    def __init__(self):
        """Initialize the base load analyzer."""
        self.logger = logging.getLogger(f"{__name__}.BaseLoadAnalyzer")

    def analyze_base_load(
        self,
        consumption_data: pd.DataFrame,
        pv_data: pd.DataFrame,
        room_data: Dict[str, pd.DataFrame],
        ev_data: Optional[pd.DataFrame] = None,
        battery_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Analyze base load patterns.

        Args:
            consumption_data: Total energy consumption data
            pv_data: PV production data for self-consumption calculation
            room_data: Room temperature data for heating estimation
            ev_data: EV charging data (optional)
            battery_data: Battery charge/discharge data (optional)

        Returns:
            Dictionary with base load analysis results
        """
        self.logger.info("Starting base load analysis")

        if consumption_data.empty:
            self.logger.warning("No consumption data provided")
            return {}

        results = {}

        # Calculate base load by subtracting controllable loads
        base_load = self._calculate_base_load(
            consumption_data, pv_data, room_data, ev_data, battery_data
        )

        if base_load.empty:
            self.logger.warning("Could not calculate base load")
            return {"error": "Could not calculate base load"}

        # Basic statistics
        results["basic_stats"] = self._calculate_base_load_stats(base_load)

        # Time-based patterns
        results["time_patterns"] = self._analyze_time_patterns(base_load)

        # Seasonal analysis
        results["seasonal_analysis"] = self._analyze_seasonal_patterns(base_load)

        # Load profiling and clustering
        results["load_profiles"] = self._analyze_load_profiles(base_load)

        # Anomaly detection
        results["anomalies"] = self._detect_load_anomalies(base_load)

        # Prediction model evaluation
        results["prediction_models"] = self._evaluate_prediction_models(base_load)

        # Energy efficiency analysis
        results["efficiency_analysis"] = self._analyze_energy_efficiency(base_load)

        # Special events detection
        results["special_events"] = self._detect_special_events(base_load)

        self.logger.info("Base load analysis completed")
        return results

    def _calculate_base_load(
        self,
        consumption_data: pd.DataFrame,
        pv_data: pd.DataFrame,
        room_data: Dict[str, pd.DataFrame],
        ev_data: Optional[pd.DataFrame],
        battery_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Calculate base load by subtracting controllable loads."""

        # Find total consumption column
        consumption_cols = [
            col
            for col in consumption_data.columns
            if any(
                keyword in col.lower()
                for keyword in ["consumption", "total", "load", "grid_import"]
            )
        ]

        if not consumption_cols:
            self.logger.warning("No consumption columns found")
            return pd.DataFrame()

        total_consumption = consumption_data[consumption_cols[0]].copy()

        # Start with total consumption
        base_load_data = pd.DataFrame(index=total_consumption.index)
        base_load_data["total_consumption"] = total_consumption
        base_load_data["base_load"] = total_consumption.copy()

        # Subtract PV self-consumption
        if not pv_data.empty:
            pv_power_cols = [col for col in pv_data.columns if "power" in col.lower()]
            if pv_power_cols:
                pv_power = pv_data[pv_power_cols[0]]
                # Resample to match consumption data frequency
                pv_resampled = pv_power.resample(
                    total_consumption.index.freq or "15T"
                ).mean()

                # Self-consumption = min(PV_production, total_consumption)
                pv_aligned = pv_resampled.reindex(
                    base_load_data.index, method="nearest"
                )
                pv_self_consumption = np.minimum(
                    pv_aligned.fillna(0), base_load_data["total_consumption"]
                )

                base_load_data["pv_self_consumption"] = pv_self_consumption
                base_load_data["base_load"] -= pv_self_consumption

        # Subtract heating consumption
        heating_consumption = self._estimate_heating_consumption(
            room_data, base_load_data.index
        )
        if heating_consumption is not None:
            base_load_data["heating_consumption"] = heating_consumption
            base_load_data["base_load"] -= heating_consumption

        # Subtract EV charging
        if ev_data is not None and not ev_data.empty:
            ev_cols = [
                col
                for col in ev_data.columns
                if any(keyword in col.lower() for keyword in ["power", "charge"])
            ]
            if ev_cols:
                ev_power = ev_data[ev_cols[0]]
                ev_resampled = ev_power.resample(
                    total_consumption.index.freq or "15T"
                ).mean()
                ev_aligned = ev_resampled.reindex(
                    base_load_data.index, method="nearest"
                )

                base_load_data["ev_consumption"] = ev_aligned.fillna(0)
                base_load_data["base_load"] -= ev_aligned.fillna(0)

        # Subtract battery consumption (charging losses)
        if battery_data is not None and not battery_data.empty:
            battery_cols = [
                col for col in battery_data.columns if "power" in col.lower()
            ]
            if battery_cols:
                battery_power = battery_data[battery_cols[0]]
                battery_resampled = battery_power.resample(
                    total_consumption.index.freq or "15T"
                ).mean()
                battery_aligned = battery_resampled.reindex(
                    base_load_data.index, method="nearest"
                )

                # Only subtract when battery is charging (positive power)
                battery_charging = np.maximum(battery_aligned.fillna(0), 0)
                base_load_data["battery_consumption"] = battery_charging
                base_load_data["base_load"] -= battery_charging

        # Ensure base load is non-negative
        base_load_data["base_load"] = np.maximum(base_load_data["base_load"], 0)

        # Add time features
        base_load_data["hour"] = base_load_data.index.hour
        base_load_data["weekday"] = base_load_data.index.weekday
        base_load_data["month"] = base_load_data.index.month
        base_load_data["is_weekend"] = base_load_data["weekday"].isin([5, 6])

        return base_load_data

    def _estimate_heating_consumption(
        self, room_data: Dict[str, pd.DataFrame], target_index: pd.DatetimeIndex
    ) -> Optional[pd.Series]:
        """Estimate heating consumption from room heating status."""
        if not room_data:
            return None

        total_heating = None

        for room_name, room_df in room_data.items():
            if room_df.empty:
                continue

            # Look for heating status column
            heating_cols = [
                col
                for col in room_df.columns
                if any(
                    keyword in col.lower() for keyword in ["heating", "heat", "state"]
                )
            ]

            if heating_cols:
                heating_status = room_df[heating_cols[0]]

                # Resample to target frequency
                heating_resampled = heating_status.resample(
                    target_index.freq or "15T"
                ).mean()
                heating_aligned = heating_resampled.reindex(
                    target_index, method="nearest"
                )

                # Assume 1kW per room when heating is on (adjust based on your system)
                heating_power = heating_aligned.fillna(0) * 1000  # 1kW per room

                if total_heating is None:
                    total_heating = heating_power
                else:
                    total_heating += heating_power

        return total_heating

    def _calculate_base_load_stats(
        self, base_load_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate basic base load statistics."""
        base_load = base_load_data["base_load"]

        stats = {
            "total_records": len(base_load_data),
            "mean_base_load": base_load.mean(),
            "median_base_load": base_load.median(),
            "min_base_load": base_load.min(),
            "max_base_load": base_load.max(),
            "std_base_load": base_load.std(),
            "total_energy_kwh": (base_load.sum() * 0.25)
            / 1000,  # 15-min intervals to kWh
            "base_load_factor": base_load.mean() / base_load.max()
            if base_load.max() > 0
            else 0,
        }

        # Percentage of total consumption
        if "total_consumption" in base_load_data.columns:
            total_consumption = base_load_data["total_consumption"]
            stats["base_load_percentage"] = (
                (base_load.sum() / total_consumption.sum() * 100)
                if total_consumption.sum() > 0
                else 0
            )

            # Breakdown of other components
            components = [
                "pv_self_consumption",
                "heating_consumption",
                "ev_consumption",
                "battery_consumption",
            ]
            for component in components:
                if component in base_load_data.columns:
                    component_sum = base_load_data[component].sum()
                    stats[f"{component}_percentage"] = (
                        (component_sum / total_consumption.sum() * 100)
                        if total_consumption.sum() > 0
                        else 0
                    )

        # Time-based statistics
        hourly_avg = base_load.groupby(base_load.index.hour).mean()
        stats.update(
            {
                "peak_hour": hourly_avg.idxmax(),
                "min_hour": hourly_avg.idxmin(),
                "peak_load": hourly_avg.max(),
                "minimum_load": hourly_avg.min(),
                "peak_to_minimum_ratio": (
                    hourly_avg.max() / hourly_avg.min()
                    if hourly_avg.min() > 0
                    else float("inf")
                ),
            }
        )

        return stats

    def _analyze_time_patterns(self, base_load_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based patterns in base load."""
        base_load = base_load_data["base_load"]

        patterns = {}

        # Hourly patterns
        hourly_profile = base_load.groupby(base_load.index.hour).agg(
            ["mean", "std", "min", "max"]
        )
        patterns["hourly_profile"] = hourly_profile.to_dict()

        # Weekday vs weekend patterns
        weekday_load = base_load[~base_load_data["is_weekend"]]
        weekend_load = base_load[base_load_data["is_weekend"]]

        patterns["weekday_vs_weekend"] = {
            "weekday_mean": weekday_load.mean(),
            "weekend_mean": weekend_load.mean(),
            "weekend_increase": (
                (weekend_load.mean() / weekday_load.mean() - 1) * 100
                if weekday_load.mean() > 0
                else 0
            ),
            "weekday_peak_hour": weekday_load.groupby(weekday_load.index.hour)
            .mean()
            .idxmax(),
            "weekend_peak_hour": weekend_load.groupby(weekend_load.index.hour)
            .mean()
            .idxmax(),
        }

        # Day of week patterns
        daily_profile = base_load.groupby(base_load.index.weekday).agg(["mean", "std"])
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        daily_profile.index = day_names
        patterns["daily_profile"] = daily_profile.to_dict()

        # Monthly patterns
        monthly_profile = base_load.groupby(base_load.index.month).agg(
            ["mean", "std", "sum"]
        )
        patterns["monthly_profile"] = monthly_profile.to_dict()

        # Identify peak and off-peak periods
        hourly_mean = base_load.groupby(base_load.index.hour).mean()
        mean_load = hourly_mean.mean()

        peak_hours = hourly_mean[hourly_mean > mean_load * 1.2].index.tolist()
        off_peak_hours = hourly_mean[hourly_mean < mean_load * 0.8].index.tolist()

        patterns["peak_off_peak"] = {
            "peak_hours": peak_hours,
            "off_peak_hours": off_peak_hours,
            "peak_load_avg": hourly_mean[peak_hours].mean() if peak_hours else None,
            "off_peak_load_avg": hourly_mean[off_peak_hours].mean()
            if off_peak_hours
            else None,
        }

        return patterns

    def _analyze_seasonal_patterns(
        self, base_load_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns using decomposition."""
        base_load = base_load_data["base_load"]

        if len(base_load) < 365 * 24 * 4:  # Less than 1 year of 15-min data
            return {
                "warning": "Insufficient data for seasonal analysis (need at least 1 year)"
            }

        try:
            # Resample to daily data for seasonal analysis
            daily_load = base_load.resample("D").mean()

            # STL decomposition
            stl = STL(daily_load.dropna(), seasonal=365)  # Annual seasonality
            result = stl.fit()

            seasonal_analysis = {
                "seasonal_strength": 1
                - (result.resid.var() / (result.seasonal + result.resid).var()),
                "trend_strength": 1
                - (result.resid.var() / (result.trend + result.resid).var()),
                "has_strong_seasonal": 1
                - (result.resid.var() / (result.seasonal + result.resid).var())
                > 0.6,
                "has_strong_trend": 1
                - (result.resid.var() / (result.trend + result.resid).var())
                > 0.6,
            }

            # Seasonal profiles
            seasons = {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11],
            }

            seasonal_profiles = {}
            for season, months in seasons.items():
                season_data = base_load[base_load.index.month.isin(months)]
                if not season_data.empty:
                    seasonal_profiles[season] = {
                        "mean_load": season_data.mean(),
                        "peak_hour": season_data.groupby(season_data.index.hour)
                        .mean()
                        .idxmax(),
                        "peak_load": season_data.groupby(season_data.index.hour)
                        .mean()
                        .max(),
                        "min_load": season_data.groupby(season_data.index.hour)
                        .mean()
                        .min(),
                    }

            seasonal_analysis["seasonal_profiles"] = seasonal_profiles

            # Find highest and lowest consumption seasons
            if seasonal_profiles:
                highest_season = max(
                    seasonal_profiles.keys(),
                    key=lambda s: seasonal_profiles[s]["mean_load"],
                )
                lowest_season = min(
                    seasonal_profiles.keys(),
                    key=lambda s: seasonal_profiles[s]["mean_load"],
                )

                seasonal_analysis.update(
                    {
                        "highest_consumption_season": highest_season,
                        "lowest_consumption_season": lowest_season,
                        "seasonal_variation_percentage": (
                            (
                                seasonal_profiles[highest_season]["mean_load"]
                                / seasonal_profiles[lowest_season]["mean_load"]
                                - 1
                            )
                            * 100
                            if seasonal_profiles[lowest_season]["mean_load"] > 0
                            else 0
                        ),
                    }
                )

            return seasonal_analysis

        except Exception as e:
            self.logger.warning(f"Seasonal analysis failed: {e}")
            return {"warning": f"Seasonal analysis failed: {str(e)}"}

    def _analyze_load_profiles(self, base_load_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and cluster load profiles."""
        base_load = base_load_data["base_load"]

        # Create daily load profiles
        daily_profiles = []
        dates = []

        for date in base_load.index.date:
            day_data = base_load[base_load.index.date == date]
            if len(day_data) >= 80:  # At least 80 data points (20 hours of 15-min data)
                # Resample to hourly and interpolate missing values
                hourly_data = day_data.resample("H").mean().interpolate()
                if len(hourly_data) == 24:
                    daily_profiles.append(hourly_data.values)
                    dates.append(date)

        if len(daily_profiles) < 30:  # Need at least 30 days
            return {
                "warning": (
                    "Insufficient data for load profile analysis "
                    "(need at least 30 complete days)"
                )
            }

        profiles_array = np.array(daily_profiles)

        # Normalize profiles for clustering
        scaler = StandardScaler()
        profiles_normalized = scaler.fit_transform(profiles_array)

        # K-means clustering to identify different load patterns
        optimal_k = self._find_optimal_clusters(profiles_normalized, max_k=8)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(profiles_normalized)

        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_profiles = profiles_array[cluster_mask]
            cluster_dates = [dates[i] for i in range(len(dates)) if cluster_mask[i]]

            # Calculate cluster characteristics
            mean_profile = cluster_profiles.mean(axis=0)

            cluster_analysis[f"cluster_{cluster_id}"] = {
                "days_count": len(cluster_profiles),
                "percentage": len(cluster_profiles) / len(daily_profiles) * 100,
                "mean_daily_consumption": cluster_profiles.sum(axis=1).mean(),
                "peak_hour": mean_profile.argmax(),
                "min_hour": mean_profile.argmin(),
                "peak_to_average_ratio": mean_profile.max() / mean_profile.mean(),
                "load_factor": mean_profile.mean() / mean_profile.max(),
                "sample_dates": cluster_dates[:5],  # First 5 dates as examples
            }

        # PCA for dimensionality reduction and visualization
        pca = PCA(n_components=2)
        # profiles_pca = pca.fit_transform(profiles_normalized)
        pca.fit(profiles_normalized)

        return {
            "clustering": {
                "optimal_clusters": optimal_k,
                "cluster_analysis": cluster_analysis,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            "profile_statistics": {
                "total_profiles": len(daily_profiles),
                "mean_daily_peak": profiles_array.max(axis=1).mean(),
                "mean_daily_minimum": profiles_array.min(axis=1).mean(),
                "mean_daily_consumption": profiles_array.sum(axis=1).mean(),
            },
        }

    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        k_range = range(
            2, min(max_k + 1, len(data) // 5)
        )  # Ensure reasonable cluster sizes

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        if len(inertias) < 2:
            return 3  # Default to 3 clusters

        # Find elbow using second derivative
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)

        if len(delta_deltas) > 0:
            elbow_idx = (
                np.argmax(delta_deltas) + 2
            )  # +2 because of double diff and 0-indexing
            return k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
        else:
            return 3  # Default

    def _detect_load_anomalies(self, base_load_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in base load patterns."""
        base_load = base_load_data["base_load"]

        if len(base_load) < 100:
            return {"warning": "Insufficient data for anomaly detection"}

        # Statistical anomaly detection using IQR
        Q1 = base_load.quantile(0.25)
        Q3 = base_load.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        statistical_anomalies = base_load[
            (base_load < lower_bound) | (base_load > upper_bound)
        ]

        # Isolation Forest for more sophisticated anomaly detection
        if len(base_load) > 200:
            try:
                # Prepare features for anomaly detection
                features = base_load_data[
                    ["base_load", "hour", "weekday", "month"]
                ].copy()
                features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
                features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
                features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7)
                features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7)

                # Remove original categorical features
                features = features.drop(["hour", "weekday", "month"], axis=1)
                features_clean = features.dropna()

                if len(features_clean) > 100:
                    iso_forest = IsolationForest(
                        contamination=0.05, random_state=42, n_estimators=100
                    )
                    anomaly_labels = iso_forest.fit_predict(features_clean)

                    ml_anomalies = features_clean[anomaly_labels == -1]["base_load"]
                else:
                    ml_anomalies = pd.Series(dtype=float)

            except Exception as e:
                self.logger.warning(f"ML anomaly detection failed: {e}")
                ml_anomalies = pd.Series(dtype=float)
        else:
            ml_anomalies = pd.Series(dtype=float)

        # Zero or very low consumption anomalies
        very_low_threshold = base_load.quantile(0.05)
        low_consumption_anomalies = base_load[base_load < very_low_threshold]

        # Very high consumption anomalies
        very_high_threshold = base_load.quantile(0.95)
        high_consumption_anomalies = base_load[base_load > very_high_threshold]

        return {
            "statistical_anomalies": {
                "count": len(statistical_anomalies),
                "percentage": len(statistical_anomalies) / len(base_load) * 100,
                "sample_dates": statistical_anomalies.index[:10].tolist(),
            },
            "ml_anomalies": {
                "count": len(ml_anomalies),
                "percentage": len(ml_anomalies) / len(base_load) * 100
                if len(base_load) > 0
                else 0,
                "sample_dates": ml_anomalies.index[:10].tolist()
                if not ml_anomalies.empty
                else [],
            },
            "low_consumption_events": {
                "count": len(low_consumption_anomalies),
                "threshold": very_low_threshold,
                "sample_dates": low_consumption_anomalies.index[:10].tolist(),
            },
            "high_consumption_events": {
                "count": len(high_consumption_anomalies),
                "threshold": very_high_threshold,
                "sample_dates": high_consumption_anomalies.index[:10].tolist(),
            },
        }

    def _evaluate_prediction_models(
        self, base_load_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate different prediction models for base load."""
        base_load = base_load_data["base_load"]

        if len(base_load) < 200:
            return {
                "warning": "Insufficient data for model evaluation (need at least 200 records)"
            }

        # Prepare features
        features = base_load_data[["hour", "weekday", "month"]].copy()

        # Add cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7)
        features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        # Add lag features
        for lag in [1, 24, 168]:  # 1 step, 1 day, 1 week (assuming 15-min data)
            features[f"load_lag_{lag}"] = base_load.shift(lag)

        # Add rolling statistics
        features["load_rolling_mean_24"] = base_load.rolling(
            window=24, min_periods=1
        ).mean()
        features["load_rolling_std_24"] = base_load.rolling(
            window=24, min_periods=1
        ).std()

        # Remove original categorical features
        features = features.drop(["hour", "weekday", "month"], axis=1)

        # Remove NaN values
        mask = features.notna().all(axis=1) & base_load.notna()
        X_clean = features[mask]
        y_clean = base_load[mask]

        if len(X_clean) < 100:
            return {"warning": "Insufficient clean data for model evaluation"}

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Models to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=10, n_jobs=-1
            ),
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}

        for name, model in models.items():
            scores = {"rmse": [], "mae": [], "r2": [], "mape": []}

            try:
                for train_idx, test_idx in tscv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Ensure non-negative predictions
                    y_pred = np.maximum(y_pred, 0)

                    scores["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    scores["mae"].append(mean_absolute_error(y_test, y_pred))
                    scores["r2"].append(r2_score(y_test, y_pred))

                    # MAPE calculation with handling for zero values
                    mape = (
                        np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1)))
                        * 100
                    )
                    scores["mape"].append(mape)

                results[name] = {
                    "mean_rmse": np.mean(scores["rmse"]),
                    "std_rmse": np.std(scores["rmse"]),
                    "mean_mae": np.mean(scores["mae"]),
                    "std_mae": np.std(scores["mae"]),
                    "mean_r2": np.mean(scores["r2"]),
                    "std_r2": np.std(scores["r2"]),
                    "mean_mape": np.mean(scores["mape"]),
                    "std_mape": np.std(scores["mape"]),
                }

                # Feature importance for Random Forest
                if name == "Random Forest":
                    model.fit(X_scaled, y_clean)
                    feature_importance = dict(
                        zip(X_clean.columns, model.feature_importances_)
                    )
                    sorted_importance = dict(
                        sorted(
                            feature_importance.items(), key=lambda x: x[1], reverse=True
                        )
                    )
                    results[name]["feature_importance"] = sorted_importance

            except Exception as e:
                self.logger.warning(f"Model {name} evaluation failed: {e}")
                results[name] = {"error": str(e)}

        # Best model selection
        valid_models = {k: v for k, v in results.items() if "error" not in v}
        if valid_models:
            best_model = min(valid_models.items(), key=lambda x: x[1]["mean_mae"])
            results["best_model"] = {
                "name": best_model[0],
                "performance": best_model[1],
            }

        # Simple persistence model for comparison
        persistence_mae = mean_absolute_error(y_clean.iloc[1:], y_clean.iloc[:-1])
        results["persistence_baseline"] = {
            "mae": persistence_mae,
            "description": "Previous value as prediction",
        }

        return results

    def _analyze_energy_efficiency(
        self, base_load_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze energy efficiency patterns."""
        base_load = base_load_data["base_load"]

        # Calculate efficiency metrics
        hourly_consumption = base_load.resample("H").sum()
        daily_consumption = base_load.resample("D").sum()
        monthly_consumption = base_load.resample("M").sum()

        efficiency_metrics = {
            "mean_hourly_consumption": hourly_consumption.mean(),
            "mean_daily_consumption": daily_consumption.mean(),
            "mean_monthly_consumption": monthly_consumption.mean(),
            "consumption_variability": {
                "hourly_cv": (
                    hourly_consumption.std() / hourly_consumption.mean()
                    if hourly_consumption.mean() > 0
                    else 0
                ),
                "daily_cv": (
                    daily_consumption.std() / daily_consumption.mean()
                    if daily_consumption.mean() > 0
                    else 0
                ),
                "monthly_cv": (
                    monthly_consumption.std() / monthly_consumption.mean()
                    if monthly_consumption.mean() > 0
                    else 0
                ),
            },
        }

        # Identify most and least efficient periods
        if len(daily_consumption) > 30:
            most_efficient_days = daily_consumption.nsmallest(10)
            least_efficient_days = daily_consumption.nlargest(10)

            efficiency_metrics.update(
                {
                    "most_efficient_days": {
                        "mean_consumption": most_efficient_days.mean(),
                        "sample_dates": most_efficient_days.index.strftime(
                            "%Y-%m-%d"
                        ).tolist(),
                    },
                    "least_efficient_days": {
                        "mean_consumption": least_efficient_days.mean(),
                        "sample_dates": least_efficient_days.index.strftime(
                            "%Y-%m-%d"
                        ).tolist(),
                    },
                    "efficiency_ratio": (
                        least_efficient_days.mean() / most_efficient_days.mean()
                        if most_efficient_days.mean() > 0
                        else 0
                    ),
                }
            )

        # Nighttime efficiency (low consumption periods)
        night_hours = [0, 1, 2, 3, 4, 5, 23]
        night_consumption = base_load[base_load_data["hour"].isin(night_hours)]

        if not night_consumption.empty:
            efficiency_metrics["nighttime_efficiency"] = {
                "mean_night_consumption": night_consumption.mean(),
                "min_night_consumption": night_consumption.min(),
                "night_load_factor": (
                    night_consumption.mean() / base_load.mean()
                    if base_load.mean() > 0
                    else 0
                ),
            }

        return efficiency_metrics

    def _detect_special_events(self, base_load_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect special events that cause unusual consumption patterns."""
        base_load = base_load_data["base_load"]

        if len(base_load) < 100:
            return {"warning": "Insufficient data for special event detection"}

        # Calculate rolling statistics for anomaly detection
        window_size = min(168, len(base_load) // 10)  # 1 week or 10% of data
        rolling_mean = base_load.rolling(window=window_size, center=True).mean()
        rolling_std = base_load.rolling(window=window_size, center=True).std()

        # Define special events as significant deviations
        z_scores = (base_load - rolling_mean) / rolling_std

        # Different types of special events
        high_consumption_events = base_load[z_scores > 3]  # Very high consumption
        low_consumption_events = base_load[z_scores < -3]  # Very low consumption

        # Consecutive high consumption (parties, holidays, etc.)
        consecutive_high = []
        current_streak = []

        for idx, z_score in z_scores.items():
            if z_score > 2:  # Moderately high threshold for consecutive detection
                current_streak.append(idx)
            else:
                if len(current_streak) >= 6:  # At least 1.5 hours of high consumption
                    consecutive_high.append(
                        {
                            "start": current_streak[0],
                            "end": current_streak[-1],
                            "duration_hours": len(current_streak) * 0.25,
                            "avg_consumption": base_load[current_streak].mean(),
                        }
                    )
                current_streak = []

        # Handle case where streak continues to end of data
        if len(current_streak) >= 6:
            consecutive_high.append(
                {
                    "start": current_streak[0],
                    "end": current_streak[-1],
                    "duration_hours": len(current_streak) * 0.25,
                    "avg_consumption": base_load[current_streak].mean(),
                }
            )

        # Vacation/away periods (extended low consumption)
        consecutive_low = []
        current_low_streak = []

        for idx, z_score in z_scores.items():
            if z_score < -1.5:  # Lower threshold for low consumption
                current_low_streak.append(idx)
            else:
                if (
                    len(current_low_streak) >= 48
                ):  # At least 12 hours of low consumption
                    consecutive_low.append(
                        {
                            "start": current_low_streak[0],
                            "end": current_low_streak[-1],
                            "duration_hours": len(current_low_streak) * 0.25,
                            "avg_consumption": base_load[current_low_streak].mean(),
                        }
                    )
                current_low_streak = []

        # Handle case where low streak continues to end of data
        if len(current_low_streak) >= 48:
            consecutive_low.append(
                {
                    "start": current_low_streak[0],
                    "end": current_low_streak[-1],
                    "duration_hours": len(current_low_streak) * 0.25,
                    "avg_consumption": base_load[current_low_streak].mean(),
                }
            )

        return {
            "high_consumption_events": {
                "count": len(high_consumption_events),
                "sample_dates": high_consumption_events.index[:10].tolist(),
                "avg_consumption": (
                    high_consumption_events.mean()
                    if not high_consumption_events.empty
                    else None
                ),
            },
            "low_consumption_events": {
                "count": len(low_consumption_events),
                "sample_dates": low_consumption_events.index[:10].tolist(),
                "avg_consumption": (
                    low_consumption_events.mean()
                    if not low_consumption_events.empty
                    else None
                ),
            },
            "consecutive_high_consumption": {
                "events": len(consecutive_high),
                "details": consecutive_high[:5],  # First 5 events
            },
            "consecutive_low_consumption": {
                "events": len(consecutive_low),
                "details": consecutive_low[:5],  # First 5 events
            },
        }
