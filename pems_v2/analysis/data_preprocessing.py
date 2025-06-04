"""
Data preprocessing utilities for PEMS v2.

Provides data cleaning, validation, and preprocessing functions
for preparing data for analysis and ML model training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DataValidator:
    """Validate data quality and integrity."""

    def __init__(self):
        """Initialize the data validator."""
        self.logger = logging.getLogger(f"{__name__}.DataValidator")

    def validate_pv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate PV production data."""
        validation_results = {"valid": True, "warnings": [], "errors": []}

        if df.empty:
            validation_results["errors"].append("PV data is empty")
            validation_results["valid"] = False
            return validation_results

        # Check for power columns
        power_cols = [col for col in df.columns if "power" in col.lower()]
        if not power_cols:
            validation_results["errors"].append("No power columns found in PV data")
            validation_results["valid"] = False

        # Check for reasonable power values
        for col in power_cols:
            if col in df.columns:
                max_power = df[col].max()
                if max_power > 50000:  # 50kW seems high for residential
                    validation_results["warnings"].append(
                        f"Very high maximum power in {col}: {max_power}W"
                    )

                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validation_results["warnings"].append(
                        f"Found {negative_count} negative power values in {col}"
                    )

        # Check for data gaps
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            large_gaps = time_diff[time_diff > pd.Timedelta(hours=2)]
            if len(large_gaps) > 0:
                validation_results["warnings"].append(
                    f"Found {len(large_gaps)} time gaps larger than 2 hours"
                )

        return validation_results

    def validate_temperature_data(self, df: pd.DataFrame, room_name: str) -> Dict[str, Any]:
        """Validate room temperature data."""
        validation_results = {"valid": True, "warnings": [], "errors": []}

        if df.empty:
            validation_results["errors"].append(f"Temperature data for {room_name} is empty")
            validation_results["valid"] = False
            return validation_results

        # Check for temperature columns
        temp_cols = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["temp", "temperature"])
        ]
        if not temp_cols:
            validation_results["errors"].append(f"No temperature columns found for {room_name}")
            validation_results["valid"] = False

        # Check for reasonable temperature ranges
        for col in temp_cols:
            if col in df.columns:
                min_temp = df[col].min()
                max_temp = df[col].max()

                if min_temp < -20 or max_temp > 50:
                    validation_results["warnings"].append(
                        f"Temperature range for {room_name}.{col} seems unusual: {min_temp:.1f}°C to {max_temp:.1f}°C"
                    )

                # Check for constant temperatures (sensor issues)
                if df[col].nunique() < 5:
                    validation_results["warnings"].append(
                        f"Very few unique temperature values in {room_name}.{col}"
                    )

        return validation_results


class OutlierDetector:
    """Detect and handle outliers in time series data."""

    def __init__(self):
        """Initialize the outlier detector."""
        self.logger = logging.getLogger(f"{__name__}.OutlierDetector")

    def detect_statistical_outliers(self, series: pd.Series, method: str = "iqr") -> pd.Series:
        """Detect outliers using statistical methods."""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            return pd.Series(z_scores > 3, index=series.index).reindex(
                series.index, fill_value=False
            )

        elif method == "modified_zscore":
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > 3.5

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def detect_contextual_outliers(
        self, series: pd.Series, context_window: str = "24H"
    ) -> pd.Series:
        """Detect outliers based on local context (rolling window)."""
        rolling_mean = series.rolling(window=context_window, center=True).mean()
        rolling_std = series.rolling(window=context_window, center=True).std()

        # Points that are more than 3 standard deviations from local mean
        outliers = np.abs(series - rolling_mean) > (3 * rolling_std)
        return outliers.fillna(False)


class GapFiller:
    """Fill gaps in time series data."""

    def __init__(self):
        """Initialize the gap filler."""
        self.logger = logging.getLogger(f"{__name__}.GapFiller")

    def fill_gaps(
        self, df: pd.DataFrame, method: str = "adaptive", max_gap_hours: int = 6
    ) -> pd.DataFrame:
        """Fill gaps in time series data using adaptive methods."""
        df_filled = df.copy()

        for column in df_filled.columns:
            if df_filled[column].dtype in ["float64", "int64"]:
                if method == "adaptive":
                    df_filled[column] = self._adaptive_fill(df_filled[column], max_gap_hours)
                elif method == "interpolate":
                    df_filled[column] = self._interpolate_fill(df_filled[column], max_gap_hours)
                elif method == "seasonal":
                    df_filled[column] = self._seasonal_fill(df_filled[column], max_gap_hours)

        return df_filled

    def _adaptive_fill(self, series: pd.Series, max_gap_hours: int) -> pd.Series:
        """Adaptive gap filling based on data characteristics."""
        filled_series = series.copy()

        # Identify gaps
        is_missing = series.isnull()

        if not is_missing.any():
            return filled_series

        # Group consecutive missing values
        missing_groups = (is_missing != is_missing.shift()).cumsum()

        for group_id in missing_groups[is_missing].unique():
            gap_mask = (missing_groups == group_id) & is_missing
            gap_size = gap_mask.sum()

            # Convert gap size to hours (assuming index frequency)
            if hasattr(series.index, "freq") and series.index.freq:
                freq_minutes = pd.Timedelta(series.index.freq).total_seconds() / 60
                gap_hours = gap_size * freq_minutes / 60
            else:
                # Estimate frequency from first few non-null intervals
                intervals = series.index.to_series().diff().dropna()
                if len(intervals) > 0:
                    avg_interval = intervals.median()
                    gap_hours = gap_size * avg_interval.total_seconds() / 3600
                else:
                    gap_hours = gap_size * 0.25  # Assume 15-minute intervals

            if gap_hours <= max_gap_hours:
                gap_start = gap_mask.idxmax()
                gap_end = gap_mask[::-1].idxmax()

                # Choose filling method based on gap characteristics
                if gap_hours <= 1:  # Short gaps: linear interpolation
                    filled_series.loc[gap_mask] = series.loc[gap_start:gap_end].interpolate(
                        method="linear"
                    )
                elif gap_hours <= 3:  # Medium gaps: spline interpolation
                    filled_series.loc[gap_mask] = series.loc[gap_start:gap_end].interpolate(
                        method="spline", order=2
                    )
                else:  # Longer gaps: seasonal decomposition approach
                    filled_series.loc[gap_mask] = self._seasonal_fill_gap(series, gap_mask)

        return filled_series

    def _interpolate_fill(self, series: pd.Series, max_gap_hours: int) -> pd.Series:
        """Simple interpolation filling."""
        # Calculate maximum gap size in data points
        if hasattr(series.index, "freq") and series.index.freq:
            freq_minutes = pd.Timedelta(series.index.freq).total_seconds() / 60
            max_gap_points = int(max_gap_hours * 60 / freq_minutes)
        else:
            max_gap_points = max_gap_hours * 4  # Assume 15-minute intervals

        return series.interpolate(method="linear", limit=max_gap_points)

    def _seasonal_fill(self, series: pd.Series, max_gap_hours: int) -> pd.Series:
        """Fill gaps using seasonal patterns."""
        filled_series = series.copy()

        # For seasonal filling, we need at least a week of data
        if len(series) < 7 * 24 * 4:  # 7 days of 15-min data
            return self._interpolate_fill(series, max_gap_hours)

        # Calculate weekly seasonal pattern
        series_reindexed = series.dropna()
        if len(series_reindexed) < 100:
            return self._interpolate_fill(series, max_gap_hours)

        # Create weekly pattern
        weekday_hour_pattern = series_reindexed.groupby(
            [series_reindexed.index.weekday, series_reindexed.index.hour]
        ).median()

        # Fill missing values with seasonal pattern
        for idx in series[series.isnull()].index:
            weekday = idx.weekday
            hour = idx.hour

            if (weekday, hour) in weekday_hour_pattern.index:
                filled_series.loc[idx] = weekday_hour_pattern.loc[(weekday, hour)]

        return filled_series

    def _seasonal_fill_gap(self, series: pd.Series, gap_mask: pd.Series) -> pd.Series:
        """Fill a specific gap using seasonal patterns."""
        # Simple implementation: use same time from previous week
        filled_values = series[gap_mask].copy()

        for idx in gap_mask[gap_mask].index:
            # Look for same time previous week
            prev_week = idx - pd.Timedelta(days=7)

            if prev_week in series.index and pd.notna(series.loc[prev_week]):
                filled_values.loc[idx] = series.loc[prev_week]
            else:
                # Fallback to interpolation
                filled_values.loc[idx] = series.interpolate().loc[idx]

        return filled_values


class FeatureEngineer:
    """Create features for ML models."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineer")

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features_df = df.copy()

        # Basic time features
        features_df["hour"] = features_df.index.hour
        features_df["day_of_week"] = features_df.index.weekday
        features_df["day_of_year"] = features_df.index.dayofyear
        features_df["month"] = features_df.index.month
        features_df["quarter"] = features_df.index.quarter
        features_df["is_weekend"] = features_df["day_of_week"].isin([5, 6])

        # Cyclical encoding for better ML performance
        features_df["hour_sin"] = np.sin(2 * np.pi * features_df["hour"] / 24)
        features_df["hour_cos"] = np.cos(2 * np.pi * features_df["hour"] / 24)
        features_df["day_of_week_sin"] = np.sin(2 * np.pi * features_df["day_of_week"] / 7)
        features_df["day_of_week_cos"] = np.cos(2 * np.pi * features_df["day_of_week"] / 7)
        features_df["day_of_year_sin"] = np.sin(2 * np.pi * features_df["day_of_year"] / 365)
        features_df["day_of_year_cos"] = np.cos(2 * np.pi * features_df["day_of_year"] / 365)

        return features_df

    def create_lag_features(
        self, df: pd.DataFrame, target_col: str, lags: List[int]
    ) -> pd.DataFrame:
        """Create lag features for time series."""
        features_df = df.copy()

        for lag in lags:
            features_df[f"{target_col}_lag_{lag}"] = features_df[target_col].shift(lag)

        return features_df

    def create_rolling_features(
        self, df: pd.DataFrame, target_col: str, windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling window features."""
        features_df = df.copy()

        for window in windows:
            features_df[f"{target_col}_rolling_mean_{window}"] = (
                features_df[target_col].rolling(window=window, min_periods=1).mean()
            )
            features_df[f"{target_col}_rolling_std_{window}"] = (
                features_df[target_col].rolling(window=window, min_periods=1).std()
            )
            features_df[f"{target_col}_rolling_min_{window}"] = (
                features_df[target_col].rolling(window=window, min_periods=1).min()
            )
            features_df[f"{target_col}_rolling_max_{window}"] = (
                features_df[target_col].rolling(window=window, min_periods=1).max()
            )

        return features_df
