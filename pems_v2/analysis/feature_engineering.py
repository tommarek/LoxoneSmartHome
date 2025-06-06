"""
Feature engineering module for PEMS v2.

Creates features for ML models from analyzed data:
- PV production features (weather, time, lag features)
- Thermal features (temperature, heating patterns)
- Energy consumption features
- Grid interaction features
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from config.energy_settings import ROOM_CONFIG
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Create features for ML models from analyzed data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineer")
        self.scalers: Dict[str, StandardScaler] = {}

    def create_pv_features(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create PV prediction features.

        Args:
            pv_data: PV production data with InputPower, etc.
            weather_data: Weather data with sun_elevation, temperature, etc.

        Returns:
            DataFrame with engineered features for PV prediction
        """
        self.logger.info("Creating PV prediction features")

        # Align datasets by timestamp
        if pv_data.empty or weather_data.empty:
            self.logger.warning("Empty input data for PV features")
            return pd.DataFrame()

        # Resample to common frequency (15 minutes)
        pv_resampled = pv_data.resample("15min").mean()
        weather_resampled = weather_data.resample("15min").mean()

        # Combine datasets
        features = pd.DataFrame(index=pv_resampled.index)

        # Target variable
        if "InputPower" in pv_resampled.columns:
            features["target_pv_power"] = pv_resampled["InputPower"]

        # Time-based features
        features["hour"] = features.index.hour
        features["day_of_year"] = features.index.dayofyear
        features["month"] = features.index.month
        features["weekend"] = (features.index.weekday >= 5).astype(int)

        # Cyclical encoding for time features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)

        # Weather features
        if not weather_resampled.empty:
            # Align weather data to feature timestamps
            weather_aligned = weather_resampled.reindex(
                features.index, method="nearest"
            )

            if "sun_elevation" in weather_aligned.columns:
                features["sun_elevation"] = weather_aligned["sun_elevation"]
                features["sun_elevation_squared"] = features["sun_elevation"] ** 2
                features["is_daytime"] = (features["sun_elevation"] > 0).astype(int)

                # Clear sky solar radiation model
                features["clear_sky_radiation"] = features["sun_elevation"].apply(
                    lambda x: max(0, 1000 * math.sin(math.radians(x))) if x > 0 else 0
                )

            if "temperature" in weather_aligned.columns:
                features["temperature"] = weather_aligned["temperature"]
                # Temperature effects on PV efficiency
                features["temp_efficiency"] = 1 - 0.004 * (
                    features["temperature"] - 25
                )  # -0.4%/°C

            if "humidity" in weather_aligned.columns:
                features["humidity"] = weather_aligned["humidity"]

            if "wind_speed" in weather_aligned.columns:
                features["wind_speed"] = weather_aligned["wind_speed"]
                # Wind cooling effect on panels
                features["wind_cooling"] = np.sqrt(features["wind_speed"].fillna(0))

            if "cloud_cover" in weather_aligned.columns:
                features["cloud_cover"] = weather_aligned["cloud_cover"]
                features["cloud_factor"] = 1 - features["cloud_cover"] / 100

        # Lag features from PV data
        if "InputPower" in pv_resampled.columns:
            # Previous hour, day, week
            features["pv_lag_1h"] = features["target_pv_power"].shift(
                4
            )  # 1 hour ago (4 * 15min)
            features["pv_lag_3h"] = features["target_pv_power"].shift(12)  # 3 hours ago
            features["pv_lag_24h"] = features["target_pv_power"].shift(
                96
            )  # 24 hours ago
            features["pv_lag_7d"] = features["target_pv_power"].shift(
                96 * 7
            )  # 7 days ago

            # Rolling statistics
            features["pv_rolling_mean_1h"] = (
                features["target_pv_power"].rolling(4).mean()
            )
            features["pv_rolling_mean_3h"] = (
                features["target_pv_power"].rolling(12).mean()
            )
            features["pv_rolling_std_1h"] = features["target_pv_power"].rolling(4).std()

            # Daily patterns
            features["pv_daily_max"] = (
                features["target_pv_power"]
                .groupby(features.index.date)
                .transform("max")
            )
            features["pv_daily_mean"] = (
                features["target_pv_power"]
                .groupby(features.index.date)
                .transform("mean")
            )

        # Clear sky index (actual vs theoretical)
        if (
            "clear_sky_radiation" in features.columns
            and "target_pv_power" in features.columns
        ):
            features["clear_sky_index"] = np.where(
                features["clear_sky_radiation"] > 0,
                features["target_pv_power"]
                / (
                    features["clear_sky_radiation"] * 0.2
                ),  # Assuming 20% panel efficiency
                0,
            )

        # Remove rows with all NaN (before data starts)
        features = features.dropna(how="all")

        self.logger.info(
            f"Created {len(features.columns)} PV features for {len(features)} time points"
        )
        return features

    def create_thermal_features(
        self,
        room_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame,
        pv_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create thermal prediction features.

        Args:
            room_data: Dict of room temperature DataFrames
            weather_data: Weather data
            pv_data: PV production data for self-consumption

        Returns:
            DataFrame with thermal features
        """
        self.logger.info("Creating thermal prediction features")

        if not room_data:
            self.logger.warning("No room data available for thermal features")
            return pd.DataFrame()

        # Find common time range
        all_indices = []
        for room_df in room_data.values():
            if not room_df.empty:
                all_indices.append(room_df.index)

        if not all_indices:
            return pd.DataFrame()

        # Create common time index
        start_time = max([idx.min() for idx in all_indices])
        end_time = min([idx.max() for idx in all_indices])
        common_index = pd.date_range(start=start_time, end=end_time, freq="15min")

        features = pd.DataFrame(index=common_index)

        # Time features
        features["hour"] = features.index.hour
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month
        features["is_weekend"] = (features.index.weekday >= 5).astype(int)

        # Working hours (higher activity, heat gains)
        features["is_working_hours"] = (
            (features["hour"] >= 8) & (features["hour"] <= 18)
        ).astype(int)
        features["is_evening"] = (
            (features["hour"] >= 18) & (features["hour"] <= 22)
        ).astype(int)
        features["is_night"] = (
            (features["hour"] >= 22) | (features["hour"] <= 6)
        ).astype(int)

        # Weather features
        if not weather_data.empty:
            weather_aligned = weather_data.reindex(features.index, method="nearest")

            if "temperature" in weather_aligned.columns:
                features["outdoor_temp"] = weather_aligned["temperature"]
                # Heating degree days (base 18°C)
                features["heating_degree"] = np.maximum(
                    0, 18 - features["outdoor_temp"]
                )

            if "wind_speed" in weather_aligned.columns:
                features["wind_speed"] = weather_aligned["wind_speed"]
                # Wind chill effect
                features["wind_chill"] = (
                    features.get("outdoor_temp", 0) - features["wind_speed"] * 0.5
                )

            if "humidity" in weather_aligned.columns:
                features["humidity"] = weather_aligned["humidity"]

        # Room-specific features
        for room_name, room_df in room_data.items():
            if room_df.empty:
                continue

            room_aligned = room_df.reindex(features.index, method="nearest")
            room_power = ROOM_CONFIG["rooms"].get(room_name, {}).get("power_kw", 1.0)

            # Temperature features
            if "temperature" in room_aligned.columns:
                temp_col = f"{room_name}_temp"
                features[temp_col] = room_aligned["temperature"]

                # Temperature differences
                if "outdoor_temp" in features.columns:
                    features[f"{room_name}_temp_diff"] = (
                        features[temp_col] - features["outdoor_temp"]
                    )

                # Temperature trends
                features[f"{room_name}_temp_trend_1h"] = features[temp_col].diff(
                    4
                )  # 1 hour change
                features[f"{room_name}_temp_rolling_std"] = (
                    features[temp_col].rolling(12).std()
                )  # 3h std

                # Lag features
                features[f"{room_name}_temp_lag_1h"] = features[temp_col].shift(4)
                features[f"{room_name}_temp_lag_24h"] = features[temp_col].shift(96)

            # Heating status features
            if "heating_on" in room_aligned.columns or "state" in room_aligned.columns:
                heating_col = (
                    "heating_on" if "heating_on" in room_aligned.columns else "state"
                )
                heating_feature = f"{room_name}_heating"
                features[heating_feature] = room_aligned[heating_col].fillna(0)

                # Heating energy consumption
                features[f"{room_name}_heating_power"] = (
                    features[heating_feature] * room_power * 1000
                )  # W

                # Heating patterns
                features[f"{room_name}_heating_hours_today"] = (
                    features[heating_feature].groupby(features.index.date).cumsum()
                    * 0.25
                )

                # Heating cycles (on/off transitions)
                features[f"{room_name}_heating_cycles"] = (
                    features[heating_feature].diff().abs()
                )

            # Setpoint features if available
            if "setpoint" in room_aligned.columns:
                setpoint_col = f"{room_name}_setpoint"
                features[setpoint_col] = room_aligned["setpoint"]

                if temp_col in features.columns:
                    # Temperature error (actual - setpoint)
                    features[f"{room_name}_temp_error"] = (
                        features[temp_col] - features[setpoint_col]
                    )

        # Aggregate features across all rooms
        temp_cols = [
            col
            for col in features.columns
            if col.endswith("_temp") and not col.endswith("_temp_diff")
        ]
        heating_cols = [
            col for col in features.columns if col.endswith("_heating_power")
        ]

        if temp_cols:
            features["avg_indoor_temp"] = features[temp_cols].mean(axis=1)
            features["min_indoor_temp"] = features[temp_cols].min(axis=1)
            features["max_indoor_temp"] = features[temp_cols].max(axis=1)
            features["temp_variance"] = features[temp_cols].var(axis=1)

        if heating_cols:
            features["total_heating_power"] = features[heating_cols].sum(axis=1)
            features["heating_power_ratio"] = features["total_heating_power"] / (
                sum(
                    ROOM_CONFIG["rooms"][room]["power_kw"]
                    for room in ROOM_CONFIG["rooms"]
                )
                * 1000
            )  # Fraction of total capacity

        # Solar gains from PV production
        if not pv_data.empty and "InputPower" in pv_data.columns:
            pv_aligned = pv_data.reindex(features.index, method="nearest")
            features["solar_gains"] = (
                pv_aligned["InputPower"] * 0.1
            )  # Assume 10% becomes indoor heat gain

        # Thermal inertia features (building response)
        if "avg_indoor_temp" in features.columns and "outdoor_temp" in features.columns:
            # Temperature response to weather
            features["thermal_response"] = (
                features["avg_indoor_temp"].rolling(24).mean()
                - features["outdoor_temp"].rolling(24).mean()
            )

        features = features.dropna(how="all")

        self.logger.info(
            f"Created {len(features.columns)} thermal features for {len(features)} time points"
        )
        return features

    def create_energy_features(
        self,
        consumption_data: pd.DataFrame,
        pv_data: pd.DataFrame,
        battery_data: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create energy management features.

        Args:
            consumption_data: Total consumption data
            pv_data: PV production data
            battery_data: Battery state data
            price_data: Energy price data (optional)

        Returns:
            DataFrame with energy features
        """
        self.logger.info("Creating energy management features")

        # Find common time range
        datasets = [
            df for df in [consumption_data, pv_data, battery_data] if not df.empty
        ]
        if not datasets:
            return pd.DataFrame()

        start_time = max([df.index.min() for df in datasets])
        end_time = min([df.index.max() for df in datasets])
        common_index = pd.date_range(start=start_time, end=end_time, freq="15min")

        features = pd.DataFrame(index=common_index)

        # Time features
        features["hour"] = features.index.hour
        features["day_of_week"] = features.index.dayofweek
        features["is_weekend"] = (features.index.weekday >= 5).astype(int)

        # Peak/off-peak periods
        features["is_peak_hours"] = (
            (features["hour"] >= 17) & (features["hour"] <= 20)
        ).astype(int)
        features["is_night_hours"] = (
            (features["hour"] >= 22) | (features["hour"] <= 6)
        ).astype(int)

        # Consumption features
        if not consumption_data.empty:
            consumption_aligned = consumption_data.reindex(
                features.index, method="nearest"
            )

            if "total_consumption" in consumption_aligned.columns:
                features["consumption"] = consumption_aligned["total_consumption"]

                # Consumption patterns
                features["consumption_lag_1h"] = features["consumption"].shift(4)
                features["consumption_lag_24h"] = features["consumption"].shift(96)
                features["consumption_rolling_mean_3h"] = (
                    features["consumption"].rolling(12).mean()
                )
                features["consumption_daily_max"] = (
                    features["consumption"]
                    .groupby(features.index.date)
                    .transform("max")
                )

                # Base load estimation (minimum daily consumption)
                features["base_load"] = (
                    features["consumption"]
                    .groupby(features.index.date)
                    .transform("min")
                )

        # PV production features
        if not pv_data.empty:
            pv_aligned = pv_data.reindex(features.index, method="nearest")

            if "InputPower" in pv_aligned.columns:
                features["pv_production"] = pv_aligned["InputPower"]

                # Self-consumption ratio
                if "consumption" in features.columns:
                    features["self_consumption"] = np.minimum(
                        features["pv_production"], features["consumption"]
                    )
                    features["self_consumption_ratio"] = np.where(
                        features["pv_production"] > 0,
                        features["self_consumption"] / features["pv_production"],
                        0,
                    )

                # Grid interaction
                if "ACPowerToGrid" in pv_aligned.columns:
                    features["grid_export"] = pv_aligned["ACPowerToGrid"].fillna(0)

                if "consumption" in features.columns:
                    features["grid_import"] = np.maximum(
                        0, features["consumption"] - features.get("self_consumption", 0)
                    )

                    # Net grid power (positive = import, negative = export)
                    features["net_grid_power"] = features.get(
                        "grid_import", 0
                    ) - features.get("grid_export", 0)

        # Battery features
        if not battery_data.empty:
            battery_aligned = battery_data.reindex(features.index, method="nearest")

            if "SOC" in battery_aligned.columns:
                features["battery_soc"] = battery_aligned["SOC"]
                features["battery_soc_change"] = features["battery_soc"].diff()

                # Battery utilization
                features["battery_available_capacity"] = (
                    100 - features["battery_soc"]
                ) / 100
                features["battery_usable_energy"] = features["battery_soc"] / 100

            if "net_battery_power" in battery_aligned.columns:
                features["battery_power"] = battery_aligned["net_battery_power"]
                features["battery_charging"] = (features["battery_power"] > 0).astype(
                    int
                )
                features["battery_discharging"] = (
                    features["battery_power"] < 0
                ).astype(int)

        # Price features
        if price_data is not None and not price_data.empty:
            price_aligned = price_data.reindex(features.index, method="nearest")

            if "price_czk_kwh" in price_aligned.columns:
                features["electricity_price"] = price_aligned["price_czk_kwh"]

                # Price quantiles for optimization decisions
                features["price_quantile"] = (
                    features["electricity_price"]
                    .rolling(96 * 7)
                    .rank(pct=True)  # 7 days
                )

                # High/low price periods
                features["is_high_price"] = (features["price_quantile"] > 0.8).astype(
                    int
                )
                features["is_low_price"] = (features["price_quantile"] < 0.2).astype(
                    int
                )

        # Energy balance features
        if all(col in features.columns for col in ["pv_production", "consumption"]):
            features["energy_balance"] = (
                features["pv_production"] - features["consumption"]
            )
            features["energy_surplus"] = np.maximum(0, features["energy_balance"])
            features["energy_deficit"] = np.maximum(0, -features["energy_balance"])

        features = features.dropna(how="all")

        self.logger.info(
            f"Created {len(features.columns)} energy features for {len(features)} time points"
        )
        return features

    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            features: Input features DataFrame
            fit: Whether to fit the scaler (True for training, False for prediction)

        Returns:
            Scaled features DataFrame
        """
        if features.empty:
            return features

        # Identify numerical columns (exclude binary indicators)
        numerical_cols = []
        for col in features.columns:
            if features[col].dtype in ["float64", "int64"] and not col.startswith(
                "is_"
            ):
                # Check if column is not binary (0/1)
                unique_vals = features[col].dropna().unique()
                if len(unique_vals) > 2 or not all(
                    val in [0, 1] for val in unique_vals
                ):
                    numerical_cols.append(col)

        if not numerical_cols:
            return features

        scaled_features = features.copy()

        for col in numerical_cols:
            scaler_key = f"feature_scaler_{col}"

            if fit:
                # Fit new scaler
                self.scalers[scaler_key] = StandardScaler()
                scaled_values = self.scalers[scaler_key].fit_transform(
                    features[[col]].fillna(features[col].mean())
                )
            else:
                # Use existing scaler
                if scaler_key not in self.scalers:
                    self.logger.warning(
                        f"No fitted scaler found for {col}, skipping scaling"
                    )
                    continue
                scaled_values = self.scalers[scaler_key].transform(
                    features[[col]].fillna(features[col].mean())
                )

            scaled_features[col] = scaled_values.flatten()

        self.logger.info(f"Scaled {len(numerical_cols)} numerical features")
        return scaled_features

    def create_relay_features(
        self, relay_states: Dict[str, pd.DataFrame], weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create relay pattern features for heating optimization.

        Args:
            relay_states: Dict of room relay state DataFrames
            weather_data: Weather data for correlation analysis

        Returns:
            DataFrame with relay pattern features
        """
        self.logger.info("Creating relay pattern features")

        if not relay_states:
            self.logger.warning("No relay states available for feature creation")
            return pd.DataFrame()

        # Find common time range
        all_indices = []
        for relay_df in relay_states.values():
            if not relay_df.empty:
                all_indices.append(relay_df.index)

        if not all_indices:
            return pd.DataFrame()

        start_time = max([idx.min() for idx in all_indices])
        end_time = min([idx.max() for idx in all_indices])
        common_index = pd.date_range(start=start_time, end=end_time, freq="15min")

        features = pd.DataFrame(index=common_index)

        # Time features
        features["hour"] = features.index.hour
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month
        features["is_weekend"] = (features.index.weekday >= 5).astype(int)

        # Weather features for correlation
        if not weather_data.empty:
            weather_aligned = weather_data.reindex(features.index, method="nearest")

            if "temperature" in weather_aligned.columns:
                features["outdoor_temp"] = weather_aligned["temperature"]
                features["heating_degree_hours"] = np.maximum(
                    0, 18 - features["outdoor_temp"]
                )

        # Individual room relay features
        total_heating_demand = pd.Series(0, index=common_index)
        active_relays = pd.Series(0, index=common_index)

        for room_name, relay_df in relay_states.items():
            if relay_df.empty:
                continue

            relay_aligned = relay_df.reindex(features.index, method="nearest")
            room_power = ROOM_CONFIG["rooms"].get(room_name, {}).get("power_kw", 1.0)

            # Individual room relay state
            relay_col = f"{room_name}_relay"
            features[relay_col] = relay_aligned.iloc[:, 0].fillna(
                0
            )  # First column is relay state

            # Room heating power
            features[f"{room_name}_heating_power"] = (
                features[relay_col] * room_power * 1000
            )

            # Relay cycling patterns
            features[f"{room_name}_relay_cycles"] = features[relay_col].diff().abs()

            # Cumulative heating time
            features[f"{room_name}_heating_time_today"] = (
                features[relay_col].groupby(features.index.date).cumsum() * 0.25
            )  # Hours

            # Lag features for room heating patterns
            features[f"{room_name}_relay_lag_1h"] = features[relay_col].shift(4)
            features[f"{room_name}_relay_lag_3h"] = features[relay_col].shift(12)

            # Add to totals
            total_heating_demand += features[relay_col] * room_power
            active_relays += features[relay_col]

        # Aggregate features
        features["total_heating_demand"] = total_heating_demand
        features["active_relay_count"] = active_relays
        features["heating_load_factor"] = total_heating_demand / sum(
            ROOM_CONFIG["rooms"][room]["power_kw"] for room in ROOM_CONFIG["rooms"]
        )

        # Heating patterns
        features["heating_demand_high"] = (features["active_relay_count"] > 8).astype(
            int
        )
        features["heating_demand_medium"] = (
            (features["active_relay_count"] > 4) & (features["active_relay_count"] <= 8)
        ).astype(int)
        features["heating_demand_low"] = (
            (features["active_relay_count"] > 0) & (features["active_relay_count"] <= 4)
        ).astype(int)

        # Rolling statistics for heating patterns
        features["heating_demand_rolling_mean_1h"] = (
            features["total_heating_demand"].rolling(4).mean()
        )
        features["heating_demand_rolling_mean_3h"] = (
            features["total_heating_demand"].rolling(12).mean()
        )
        features["heating_demand_rolling_std_1h"] = (
            features["total_heating_demand"].rolling(4).std()
        )

        # Daily heating patterns
        features["heating_hours_today"] = (
            features["total_heating_demand"].groupby(features.index.date).cumsum()
            * 0.25
        )
        features["max_heating_today"] = (
            features["total_heating_demand"]
            .groupby(features.index.date)
            .transform("max")
        )

        # Temperature correlation features
        if "outdoor_temp" in features.columns:
            # Expected vs actual heating based on temperature
            temp_bins = pd.cut(
                features["outdoor_temp"], bins=[-50, -5, 0, 5, 10, 15, 20, 50]
            )
            for bin_name, group in features.groupby(temp_bins):
                if len(group) > 0:
                    bin_str = f"temp_bin_{str(bin_name).replace('(', '').replace(']', '').replace(', ', '_to_')}"
                    features[f"heating_in_{bin_str}"] = features[
                        "total_heating_demand"
                    ].where(features.index.isin(group.index), 0)

        # Heating efficiency features
        features["heating_starts"] = (
            (features["active_relay_count"] > 0)
            & (features["active_relay_count"].shift(1) == 0)
        ).astype(int)

        features["heating_stops"] = (
            (features["active_relay_count"] == 0)
            & (features["active_relay_count"].shift(1) > 0)
        ).astype(int)

        features = features.dropna(how="all")

        self.logger.info(
            f"Created {len(features.columns)} relay features for {len(features)} time points"
        )
        return features

    def create_price_features(
        self, price_data: pd.DataFrame, consumption_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create energy price features for optimization decisions.

        Args:
            price_data: Energy price data with timestamps
            consumption_data: Energy consumption data

        Returns:
            DataFrame with price-based features
        """
        self.logger.info("Creating energy price features")

        if price_data is None or price_data.empty:
            self.logger.warning("No price data available for feature creation")
            return pd.DataFrame()

        # Resample price data to common frequency
        price_resampled = price_data.resample("15min").mean()
        features = pd.DataFrame(index=price_resampled.index)

        # Basic price features
        if "price" in price_resampled.columns:
            features["electricity_price"] = price_resampled["price"]
        elif "price_czk_kwh" in price_resampled.columns:
            features["electricity_price"] = price_resampled["price_czk_kwh"]
        else:
            # Use first numeric column as price
            numeric_cols = price_resampled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features["electricity_price"] = price_resampled[numeric_cols[0]]
            else:
                self.logger.error("No numeric price column found")
                return pd.DataFrame()

        # Time features
        features["hour"] = features.index.hour
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month
        features["is_weekend"] = (features.index.weekday >= 5).astype(int)

        # Price statistics and patterns
        features["price_lag_1h"] = features["electricity_price"].shift(4)
        features["price_lag_24h"] = features["electricity_price"].shift(96)

        # Rolling price statistics
        features["price_rolling_mean_24h"] = (
            features["electricity_price"].rolling(96).mean()
        )
        features["price_rolling_std_24h"] = (
            features["electricity_price"].rolling(96).std()
        )
        features["price_rolling_min_24h"] = (
            features["electricity_price"].rolling(96).min()
        )
        features["price_rolling_max_24h"] = (
            features["electricity_price"].rolling(96).max()
        )

        # Price quantiles and ranks
        features["price_quantile_daily"] = (
            features["electricity_price"].groupby(features.index.date).rank(pct=True)
        )
        features["price_quantile_weekly"] = (
            features["electricity_price"].rolling(96 * 7).rank(pct=True)
        )

        # Price categories
        daily_median = (
            features["electricity_price"].groupby(features.index.date).median()
        )
        features["daily_median_price"] = features.index.to_series().dt.date.map(
            daily_median
        )

        features["is_price_very_high"] = (
            features["price_quantile_daily"] > 0.9
        ).astype(int)
        features["is_price_high"] = (features["price_quantile_daily"] > 0.7).astype(int)
        features["is_price_low"] = (features["price_quantile_daily"] < 0.3).astype(int)
        features["is_price_very_low"] = (features["price_quantile_daily"] < 0.1).astype(
            int
        )

        # Price relative to historical averages
        features["price_vs_daily_avg"] = (
            features["electricity_price"] / features["price_rolling_mean_24h"]
        )

        # Price volatility
        features["price_volatility_6h"] = (
            features["electricity_price"].rolling(24).std()
            / features["electricity_price"].rolling(24).mean()
        )

        # Future price predictions (next few hours)
        features["price_next_1h"] = features["electricity_price"].shift(-4)
        features["price_next_3h"] = features["electricity_price"].shift(-12)
        features["price_next_6h"] = features["electricity_price"].shift(-24)

        # Price trends
        features["price_trend_1h"] = features["electricity_price"].diff(4)
        features["price_trend_3h"] = features["electricity_price"].diff(12)

        # Optimal timing features
        # Find cheapest/most expensive hours in next 24h
        for lookahead_hours in [6, 12, 24]:
            lookahead_periods = lookahead_hours * 4  # 15-min periods

            # Cheapest hour in next X hours
            features[f"is_cheapest_next_{lookahead_hours}h"] = (
                features["electricity_price"]
                .rolling(lookahead_periods, center=True)
                .rank()
                == 1
            ).astype(int)

            # Most expensive hour in next X hours
            features[f"is_most_expensive_next_{lookahead_hours}h"] = (
                features["electricity_price"]
                .rolling(lookahead_periods, center=True)
                .rank(ascending=False)
                == 1
            ).astype(int)

        # Energy cost features
        if not consumption_data.empty:
            consumption_aligned = consumption_data.reindex(
                features.index, method="nearest"
            )

            if len(consumption_aligned.columns) > 0:
                consumption_col = consumption_aligned.columns[0]
                features["consumption"] = consumption_aligned[consumption_col]

                # Energy costs
                features["energy_cost_per_15min"] = (
                    features["electricity_price"]
                    * features["consumption"]
                    * 0.25
                    / 1000
                )  # CZK per 15 minutes

                # Cumulative daily costs
                features["energy_cost_today"] = (
                    features["energy_cost_per_15min"]
                    .groupby(features.index.date)
                    .cumsum()
                )

                # Cost savings potential
                min_daily_price = (
                    features["electricity_price"].groupby(features.index.date).min()
                )
                features["min_price_today"] = features.index.to_series().dt.date.map(
                    min_daily_price
                )

                features["potential_savings_per_15min"] = (
                    (features["electricity_price"] - features["min_price_today"])
                    * features["consumption"]
                    * 0.25
                    / 1000
                )

        # Market price indicators
        features["is_peak_price_period"] = (
            (features["hour"] >= 17)
            & (features["hour"] <= 20)
            & (features["day_of_week"] < 5)  # Weekday evening peak
        ).astype(int)

        features["is_off_peak_period"] = (
            (features["hour"] >= 22) | (features["hour"] <= 6)
        ).astype(int)

        # Seasonal price patterns
        features["price_month_avg"] = (
            features["electricity_price"]
            .groupby(features.index.month)
            .transform("mean")
        )
        features["price_vs_month_avg"] = (
            features["electricity_price"] / features["price_month_avg"]
        )

        features = features.dropna(how="all")

        self.logger.info(
            f"Created {len(features.columns)} price features for {len(features)} time points"
        )
        return features
