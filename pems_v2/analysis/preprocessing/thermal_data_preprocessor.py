"""
Thermal Data Preprocessing Module for PEMS v2.

Separates data preprocessing logic from thermal analysis.
Handles data cleaning, filtering, merging, and preparation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

try:
    from analysis.utils.loxone_adapter import LoxoneDataIntegrator, LoxoneFieldAdapter
except ImportError:
    # Fallback for testing
    LoxoneDataIntegrator = None
    LoxoneFieldAdapter = None


class ThermalDataPreprocessor:
    """Handles all data preprocessing for thermal analysis."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings.get("thermal_analysis", {})
        self.logger = logging.getLogger(f"{__name__}.ThermalDataPreprocessor")
        if LoxoneDataIntegrator:
            self.integrator = LoxoneDataIntegrator()
        else:
            self.integrator = None

    def prepare_thermal_analysis_data(
        self,
        room_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame,
        relay_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Main preprocessing pipeline for thermal analysis.
        
        Args:
            room_data: Raw room temperature data
            weather_data: Raw weather data
            relay_data: Optional relay state data
            
        Returns:
            Tuple of (processed_room_data, processed_weather_data)
        """
        self.logger.info("Starting thermal data preprocessing pipeline...")

        # Validate inputs
        if not room_data:
            self.logger.error("Room data is empty")
            return {}, pd.DataFrame()

        # Prepare data using Loxone integrator if relay data is provided
        if relay_data is not None and self.integrator is not None:
            self.logger.info("Preparing thermal analysis data with relay integration")
            self._log_relay_data_info(relay_data)
            
            # Store relay data in integrator for adaptive analysis
            self.integrator._processed_relay_data = relay_data
            standardized_rooms, standardized_weather = (
                self.integrator.prepare_thermal_analysis_data(
                    room_data, relay_data, weather_data
                )
            )
        else:
            # Standardize room data without relay integration
            self.logger.info("Standardizing room data without relay integration")
            standardized_rooms = self._standardize_room_data(room_data)
            standardized_weather = self._standardize_weather_data(weather_data)

        # Filter out external environment data
        processed_rooms = self._filter_interior_rooms(standardized_rooms)
        
        # Apply data cleaning and merging for each room
        for room_name, room_df in processed_rooms.items():
            self.logger.debug(f"Processing room {room_name} with {len(room_df)} records")
            self.logger.debug(f"Room {room_name} columns: {list(room_df.columns)}")
            processed_room = self._merge_room_weather_data(room_df, standardized_weather)
            if not processed_room.empty:
                processed_rooms[room_name] = processed_room
                self.logger.debug(f"Room {room_name} processed successfully with columns: {list(processed_room.columns)}")
            else:
                self.logger.warning(f"Room {room_name} processing resulted in empty DataFrame")
                # Keep original data if preprocessing failed
                processed_rooms[room_name] = room_df

        self.logger.info(
            f"Preprocessing completed for {len(processed_rooms)} rooms"
        )
        return processed_rooms, standardized_weather

    def _standardize_room_data(
        self, room_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Standardize room data format."""
        standardized_rooms = {}
        for room_name, room_df in room_data.items():
            if LoxoneFieldAdapter:
                standardized_rooms[room_name] = LoxoneFieldAdapter.standardize_room_data(
                    room_df, room_name
                )
            else:
                # Simple fallback standardization
                standardized_rooms[room_name] = room_df.copy()
        return standardized_rooms

    def _standardize_weather_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize weather data format."""
        if LoxoneFieldAdapter:
            return LoxoneFieldAdapter.standardize_weather_data(weather_data)
        else:
            # Simple fallback
            return weather_data.copy()

    def _filter_interior_rooms(
        self, room_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Filter out external environment data (not actual rooms)."""
        external_environments = [
            "outside",
            "outdoor", 
            "external",
            "environment",
            "weather",
        ]
        interior_rooms = {
            room_name: room_df
            for room_name, room_df in room_data.items()
            if not any(env in room_name.lower() for env in external_environments)
        }

        self.logger.info(
            f"Filtered to {len(interior_rooms)} interior rooms "
            f"(excluded external environment data)"
        )
        return interior_rooms

    def _merge_room_weather_data(
        self, room_df: pd.DataFrame, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge room temperature data with weather data and apply cleaning."""
        # Ensure we have temperature column
        temp_col = self._find_temperature_column(room_df)
        if temp_col is None:
            self.logger.warning("No temperature column found in room data - returning empty DataFrame")
            return pd.DataFrame()

        # Prepare room data
        room_clean = room_df.copy()
        room_clean.rename(columns={temp_col: "room_temp"}, inplace=True)

        # Apply data cleaning
        room_clean = self._clean_temperature_data(room_clean)

        # Add heating status
        room_clean = self._add_heating_status(room_clean, room_df)

        # Add setpoint if available
        if "setpoint" in room_df.columns:
            room_clean["setpoint"] = room_df["setpoint"]

        # Merge with weather data
        if not weather_data.empty:
            merged_data = self._merge_with_weather(room_clean, weather_data)
        else:
            merged_data = room_clean
            self.logger.warning("No weather data available for merging")

        # Calculate temperature difference if outdoor temperature is available
        if "outdoor_temp" in merged_data.columns and not merged_data["outdoor_temp"].isna().all():
            merged_data["temp_diff"] = merged_data["room_temp"] - merged_data["outdoor_temp"]
            self.logger.debug("Created temp_diff column from room and outdoor temperature")
        else:
            # Create temp_diff with NaN values when outdoor temp is not available
            merged_data["temp_diff"] = pd.Series(index=merged_data.index, dtype=float)
            if "outdoor_temp" not in merged_data.columns:
                merged_data["outdoor_temp"] = pd.Series(index=merged_data.index, dtype=float)
            self.logger.debug("Created temp_diff column with NaN values (no outdoor temperature)")

        # Add time features for analysis
        merged_data["hour"] = merged_data.index.hour
        merged_data["weekday"] = merged_data.index.weekday

        return merged_data

    def _find_temperature_column(self, room_df: pd.DataFrame) -> Optional[str]:
        """Find the temperature column in room data."""
        self.logger.debug(f"Available columns in room data: {list(room_df.columns)}")
        
        # Common temperature column names
        temp_candidates = [
            "temperature", "value", "temp", "room_temp", 
            "Temperature", "Value", "Temp", "room_temperature",
            "teplota", "teplomer"  # Czech names
        ]
        
        for col in temp_candidates:
            if col in room_df.columns:
                self.logger.debug(f"Found temperature column: {col}")
                return col
                
        # If no exact match, look for columns containing temperature-related keywords
        for col in room_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["temp", "teplota", "temperature"]):
                self.logger.debug(f"Found temperature column by keyword match: {col}")
                return col
                
        self.logger.warning(f"No temperature column found in: {list(room_df.columns)}")
        return None

    def _add_heating_status(
        self, room_clean: pd.DataFrame, original_room_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add heating status to room data."""
        heating_cols = [
            col
            for col in original_room_df.columns
            if "heating" in col.lower()
            or "heat" in col.lower()
            or col.lower() == "relay_state"
        ]

        if heating_cols:
            heating_col = heating_cols[0]
            self.logger.debug(f"Found heating column '{heating_col}' in room data")

            if heating_col == "relay_state":
                room_clean["heating_on"] = (original_room_df[heating_col] > 0).astype(int)
                relay_on_count = room_clean["heating_on"].sum()
                self.logger.debug(
                    f"Relay states - ON intervals: {relay_on_count}, "
                    f"Total points: {len(room_clean)}"
                )
            else:
                room_clean["heating_on"] = original_room_df[heating_col]
        elif "heating_on" in original_room_df.columns:
            self.logger.debug("Using existing 'heating_on' column")
            room_clean["heating_on"] = original_room_df["heating_on"]
        else:
            self.logger.info("No heating column found, using inference")
            room_clean["heating_on"] = self._infer_heating_status(
                room_clean["room_temp"]
            )

        # Log heating statistics
        if "heating_on" in room_clean.columns:
            heating_stats = room_clean["heating_on"].value_counts()
            heating_pct = (room_clean["heating_on"] == 1).mean() * 100
            self.logger.debug(
                f"Heating statistics: {heating_stats.to_dict()}, "
                f"{heating_pct:.1f}% heating time"
            )

        return room_clean

    def _clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data cleaning with rolling median outlier removal.
        
        Removes spurious spikes from sensor readings that can ruin
        derivative calculations (dT/dt) and statistical fits.
        """
        if "room_temp" not in df.columns:
            return df

        df_clean = df.copy()

        # Apply rolling median filter to remove outliers
        window_size = 5  # 25 minutes for 5-minute data
        temp_median = (
            df_clean["room_temp"]
            .rolling(window=window_size, center=True, min_periods=1)
            .median()
        )

        # Calculate deviation from median
        temp_deviation = abs(df_clean["room_temp"] - temp_median)

        # Define outlier threshold
        outlier_threshold = 2.0  # degrees Celsius
        outlier_mask = temp_deviation > outlier_threshold

        # Replace outliers with NaN first, then apply smart interpolation
        df_clean.loc[outlier_mask, "room_temp"] = np.nan

        # Detect stuck sensor periods
        stuck_sensor_mask = self._detect_stuck_sensor(df_clean["room_temp"])
        df_clean.loc[stuck_sensor_mask, "room_temp"] = np.nan

        # Apply smart interpolation: only fill gaps of up to 3 consecutive NaNs
        df_clean["room_temp"] = df_clean["room_temp"].interpolate(
            method="time", limit=3, limit_direction="both"
        )

        # Log cleaning results
        self._log_cleaning_results(outlier_mask, df_clean["room_temp"])

        # Clean outdoor temperature if present
        if "outdoor_temp" in df_clean.columns and not df_clean["outdoor_temp"].isna().all():
            df_clean = self._clean_outdoor_temperature(df_clean, window_size, outlier_threshold)

        # Apply Savitzky-Golay filter for noise reduction
        df_clean = self._apply_savgol_smoothing(df_clean)

        return df_clean

    def _detect_stuck_sensor(self, temp_series: pd.Series) -> pd.Series:
        """
        Detect periods where sensor is stuck (constant values for >240 minutes).
        """
        window_size = 48  # 240 minutes for 5-minute data
        rolling_std = temp_series.rolling(
            window=window_size, center=True, min_periods=1
        ).std()

        stuck_mask = (rolling_std == 0) & temp_series.notna()

        stuck_count = stuck_mask.sum()
        if stuck_count > 0:
            stuck_pct = (stuck_count / len(temp_series)) * 100
            self.logger.info(
                f"Detected {stuck_count} stuck sensor readings "
                f"({stuck_pct:.1f}% of data) - marking as invalid"
            )

        return stuck_mask

    def _apply_savgol_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Savitzky-Golay filter for noise reduction while preserving signal shape.
        """
        df_smooth = df.copy()
        window_length = 7  # Must be odd
        polyorder = 2

        # Apply smoothing to room temperature
        if "room_temp" in df_smooth.columns and not df_smooth["room_temp"].isna().all():
            df_smooth = self._smooth_temperature_column(
                df_smooth, "room_temp", window_length, polyorder
            )

        # Apply smoothing to outdoor temperature
        if "outdoor_temp" in df_smooth.columns and not df_smooth["outdoor_temp"].isna().all():
            df_smooth = self._smooth_temperature_column(
                df_smooth, "outdoor_temp", window_length, polyorder
            )

        return df_smooth

    def _smooth_temperature_column(
        self, df: pd.DataFrame, column: str, window_length: int, polyorder: int
    ) -> pd.DataFrame:
        """Apply Savitzky-Golay smoothing to a specific temperature column."""
        valid_mask = df[column].notna()
        valid_count = valid_mask.sum()

        if valid_count >= window_length:
            temp_values = df[column].copy()
            temp_filled = temp_values.ffill().bfill()

            if len(temp_filled.dropna()) >= window_length:
                temp_smooth = savgol_filter(temp_filled, window_length, polyorder)
                df[column] = temp_smooth
                df.loc[~valid_mask, column] = np.nan

                self.logger.debug(
                    f"Applied Savitzky-Golay smoothing to {column} "
                    f"(window={window_length}, poly={polyorder})"
                )
        else:
            self.logger.debug(
                f"Insufficient data for {column} smoothing "
                f"({valid_count} < {window_length} required)"
            )

        return df

    def _clean_outdoor_temperature(
        self, df: pd.DataFrame, window_size: int, outlier_threshold: float
    ) -> pd.DataFrame:
        """Clean outdoor temperature data."""
        outdoor_median = (
            df["outdoor_temp"]
            .rolling(window=window_size, center=True, min_periods=1)
            .median()
        )

        outdoor_deviation = abs(df["outdoor_temp"] - outdoor_median)
        outdoor_outlier_mask = outdoor_deviation > outlier_threshold

        df.loc[outdoor_outlier_mask, "outdoor_temp"] = np.nan
        df["outdoor_temp"] = df["outdoor_temp"].interpolate(
            method="time", limit=3, limit_direction="both"
        )

        outdoor_outliers_removed = outdoor_outlier_mask.sum()
        outdoor_remaining_nans = df["outdoor_temp"].isna().sum()
        if outdoor_outliers_removed > 0:
            self.logger.debug(
                f"Cleaned {outdoor_outliers_removed} outdoor temperature outliers, "
                f"{outdoor_remaining_nans} gaps too large to interpolate"
            )

        return df

    def _merge_with_weather(
        self, room_df: pd.DataFrame, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge room data with weather data."""
        self.logger.info(
            f"Weather data available with columns: {list(weather_data.columns)}"
        )

        # Find outdoor temperature column
        outdoor_temp_col = self._find_outdoor_temperature_column(weather_data)
        if outdoor_temp_col is None:
            self.logger.warning("No outdoor temperature column found in weather data")
            return room_df

        # Merge weather data
        try:
            weather_subset = weather_data[[outdoor_temp_col]].rename(
                columns={outdoor_temp_col: "outdoor_temp"}
            )

            # Ensure timezone consistency
            room_df, weather_subset = self._ensure_timezone_consistency(
                room_df, weather_subset
            )

            # Resample outdoor temp to match room data frequency
            resampled_weather = weather_subset.reindex(
                room_df.index, method="pad", limit=1
            )

            merged_df = room_df.join(resampled_weather)

            # Validate merge results
            valid_outdoor_temps = merged_df["outdoor_temp"].notna().sum()
            if valid_outdoor_temps == 0:
                self.logger.error(
                    "Merge resulted in zero valid outdoor temperature points"
                )
                return room_df

            self.logger.info(
                f"Successfully merged room and outdoor temperature data. "
                f"Outdoor temp range: {merged_df['outdoor_temp'].min():.1f}°C to "
                f"{merged_df['outdoor_temp'].max():.1f}°C"
            )
            return merged_df

        except Exception as e:
            self.logger.error(f"Error during weather data merge: {e}", exc_info=True)
            return room_df

    def _find_outdoor_temperature_column(self, weather_data: pd.DataFrame) -> Optional[str]:
        """Find outdoor temperature column in weather data."""
        for col in ["outdoor_temp", "temperature_2m", "temperature", "temp"]:
            if col in weather_data.columns:
                return col
        return None

    def _ensure_timezone_consistency(
        self, room_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ensure both dataframes have consistent timezone."""
        if room_df.index.tz is None:
            room_df.index = room_df.index.tz_localize("UTC")
        if weather_df.index.tz is None:
            weather_df.index = weather_df.index.tz_localize("UTC")
        return room_df, weather_df

    def _infer_heating_status(self, temperature_series: pd.Series) -> pd.Series:
        """Infer heating status from temperature changes."""
        # Simple inference: heating when temperature is rising significantly
        temp_diff = temperature_series.diff()
        heating_threshold = 0.1  # degrees per period
        return (temp_diff > heating_threshold).astype(int)

    def _log_relay_data_info(self, relay_data: Dict[str, pd.DataFrame]) -> None:
        """Log information about relay data."""
        self.logger.info(f"Relay data provided for rooms: {list(relay_data.keys())}")
        for room, relay_df in relay_data.items():
            if not relay_df.empty:
                self.logger.info(f"Relay data for {room}: {relay_df.shape[0]} records")
            else:
                self.logger.info(f"Empty relay data for {room}")

    def _log_cleaning_results(
        self, outlier_mask: pd.Series, cleaned_temp: pd.Series
    ) -> None:
        """Log the results of data cleaning."""
        outliers_removed = outlier_mask.sum()
        remaining_nans = cleaned_temp.isna().sum()
        
        if outliers_removed > 0:
            total_points = len(outlier_mask)
            self.logger.info(
                f"Cleaned {outliers_removed} temperature outliers "
                f"({outliers_removed/total_points*100:.1f}% of data), "
                f"{remaining_nans} gaps too large to interpolate"
            )