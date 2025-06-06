"""
Data preprocessing utilities for PEMS v2.

Provides data cleaning, validation, and preprocessing functions
for preparing data for analysis and ML model training.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
# Import Loxone adapter for field standardization
from analysis.utils.loxone_adapter import (LoxoneDataIntegrator,
                                           LoxoneFieldAdapter)
from config.energy_settings import ROOM_CONFIG
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

    def validate_temperature_data(
        self, df: pd.DataFrame, room_name: str
    ) -> Dict[str, Any]:
        """Validate room temperature data."""
        validation_results = {"valid": True, "warnings": [], "errors": []}

        if df.empty:
            validation_results["errors"].append(
                f"Temperature data for {room_name} is empty"
            )
            validation_results["valid"] = False
            return validation_results

        # Check for temperature columns
        temp_cols = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["temp", "temperature"])
        ]
        if not temp_cols:
            validation_results["errors"].append(
                f"No temperature columns found for {room_name}"
            )
            validation_results["valid"] = False

        # Check for reasonable temperature ranges
        for col in temp_cols:
            if col in df.columns:
                min_temp = df[col].min()
                max_temp = df[col].max()

                if min_temp < -20 or max_temp > 50:
                    validation_results["warnings"].append(
                        f"Temperature range for {room_name}.{col} seems unusual: "
                        f"{min_temp:.1f}°C to {max_temp:.1f}°C"
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

    def detect_statistical_outliers(
        self, series: pd.Series, method: str = "iqr"
    ) -> pd.Series:
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
                    df_filled[column] = self._adaptive_fill(
                        df_filled[column], max_gap_hours
                    )
                elif method == "interpolate":
                    df_filled[column] = self._interpolate_fill(
                        df_filled[column], max_gap_hours
                    )
                elif method == "seasonal":
                    df_filled[column] = self._seasonal_fill(
                        df_filled[column], max_gap_hours
                    )

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
                    filled_series.loc[gap_mask] = series.loc[
                        gap_start:gap_end
                    ].interpolate(method="linear")
                elif gap_hours <= 3:  # Medium gaps: spline interpolation
                    filled_series.loc[gap_mask] = series.loc[
                        gap_start:gap_end
                    ].interpolate(method="spline", order=2)
                else:  # Longer gaps: seasonal decomposition approach
                    filled_series.loc[gap_mask] = self._seasonal_fill_gap(
                        series, gap_mask
                    )

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
        features_df["day_of_week_sin"] = np.sin(
            2 * np.pi * features_df["day_of_week"] / 7
        )
        features_df["day_of_week_cos"] = np.cos(
            2 * np.pi * features_df["day_of_week"] / 7
        )
        features_df["day_of_year_sin"] = np.sin(
            2 * np.pi * features_df["day_of_year"] / 365
        )
        features_df["day_of_year_cos"] = np.cos(
            2 * np.pi * features_df["day_of_year"] / 365
        )

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


class RelayDataProcessor:
    """Enhanced data preprocessing specifically for relay-controlled heating systems."""

    def __init__(self):
        """Initialize the relay data processor."""
        self.logger = logging.getLogger(f"{__name__}.RelayDataProcessor")

        # Import room power ratings from configuration
        from config.energy_settings import ROOM_CONFIG

        self.room_power_ratings = {
            room_name: config["power_kw"]
            for room_name, config in ROOM_CONFIG["rooms"].items()
        }
        self.logger.info(
            f"Loaded power ratings for {len(self.room_power_ratings)} rooms from energy_settings.py"
        )

    def process_relay_data(self, relay_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process raw relay data for analysis.

        Args:
            relay_data: Dictionary with room names as keys, DataFrames with relay states as values

        Returns:
            Dictionary with processed relay data and statistics
        """
        if not relay_data:
            return {"error": "No relay data provided"}

        processed_data = {}
        total_power_series = None

        for room_name, room_df in relay_data.items():
            if room_df.empty:
                continue

            # Process individual room data
            room_processed = self._process_single_room_relay(room_name, room_df)
            processed_data[room_name] = room_processed

            # Accumulate total power consumption
            if "power_consumption" in room_processed:
                if total_power_series is None:
                    total_power_series = room_processed["power_consumption"].copy()
                else:
                    total_power_series = total_power_series.add(
                        room_processed["power_consumption"], fill_value=0
                    )

        # Add system-wide statistics
        if total_power_series is not None:
            processed_data["system_totals"] = self._calculate_system_statistics(
                total_power_series, relay_data
            )

        return processed_data

    def _process_single_room_relay(
        self, room_name: str, room_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Process relay data for a single room."""
        # Find relay state column
        relay_col = None
        for col in room_df.columns:
            if any(
                keyword in col.lower()
                for keyword in ["relay", "state", "heating", "on"]
            ):
                relay_col = col
                break

        if relay_col is None:
            return {"error": f"No relay state column found for {room_name}"}

        # Clean relay data
        relay_states = room_df[relay_col].copy()

        # Binary state validation and smoothing
        cleaned_states = self._validate_and_smooth_binary_states(relay_states)

        # Calculate power consumption
        power_rating = self.room_power_ratings.get(room_name, 2.0)  # Default 2kW
        power_consumption = cleaned_states * power_rating * 1000  # Convert to watts

        # Calculate switching statistics
        switching_stats = self._calculate_switching_statistics(cleaned_states)

        # Time-based features
        time_features = self._create_relay_time_features(cleaned_states)

        # Gap analysis
        gap_analysis = self._analyze_relay_gaps(cleaned_states)

        return {
            "cleaned_relay_states": cleaned_states,
            "power_consumption": power_consumption,
            "power_rating_kw": power_rating,
            "switching_statistics": switching_stats,
            "time_features": time_features,
            "gap_analysis": gap_analysis,
        }

    def _validate_and_smooth_binary_states(self, relay_states: pd.Series) -> pd.Series:
        """Validate and smooth binary relay states to remove excessive switching."""
        # Convert to binary (0/1)
        binary_states = (relay_states > 0.5).astype(int)

        # Remove rapid switching (less than 10 minutes between changes)
        smoothed_states = binary_states.copy()

        # Find state changes
        state_changes = binary_states.diff().abs() > 0
        change_times = state_changes[state_changes].index

        # Group rapid switches and use majority vote
        min_duration = pd.Timedelta(minutes=10)
        i = 0
        while i < len(change_times) - 1:
            current_change = change_times[i]
            next_change = change_times[i + 1]

            if next_change - current_change < min_duration:
                # Short duration - check if this is excessive switching
                period_end = current_change + min_duration
                period_mask = (binary_states.index >= current_change) & (
                    binary_states.index <= period_end
                )
                period_data = binary_states[period_mask]

                # Use majority vote for the period
                majority_state = (
                    period_data.mode().iloc[0]
                    if not period_data.empty
                    else binary_states.iloc[i]
                )
                smoothed_states.loc[period_mask] = majority_state

                # Skip processed changes
                while i < len(change_times) - 1 and change_times[i + 1] <= period_end:
                    i += 1

            i += 1

        return smoothed_states

    def _calculate_switching_statistics(
        self, relay_states: pd.Series
    ) -> Dict[str, Any]:
        """Calculate relay switching statistics."""
        # Count state changes
        state_changes = relay_states.diff().abs() > 0
        total_switches = state_changes.sum()

        # Calculate ON and OFF periods
        state_changes_idx = state_changes[state_changes].index.tolist()

        on_periods = []
        off_periods = []

        current_state = relay_states.iloc[0] if len(relay_states) > 0 else 0
        last_change = relay_states.index[0] if len(relay_states) > 0 else None

        for change_time in state_changes_idx + [relay_states.index[-1]]:
            if last_change is not None:
                duration = change_time - last_change
                duration_minutes = duration.total_seconds() / 60

                if current_state == 1:
                    on_periods.append(duration_minutes)
                else:
                    off_periods.append(duration_minutes)

            if change_time in state_changes_idx:
                current_state = 1 - current_state  # Switch state
            last_change = change_time

        # Calculate statistics
        stats = {
            "total_switches": total_switches,
            "switches_per_day": total_switches / (len(relay_states) / (24 * 4))
            if len(relay_states) > 0
            else 0,
            "on_time_percentage": relay_states.mean() * 100,
            "avg_on_duration_minutes": np.mean(on_periods) if on_periods else 0,
            "avg_off_duration_minutes": np.mean(off_periods) if off_periods else 0,
            "max_on_duration_minutes": max(on_periods) if on_periods else 0,
            "max_off_duration_minutes": max(off_periods) if off_periods else 0,
            "on_periods_count": len(on_periods),
            "off_periods_count": len(off_periods),
        }

        return stats

    def _create_relay_time_features(self, relay_states: pd.Series) -> pd.DataFrame:
        """Create time-based features for relay analysis."""
        features_df = pd.DataFrame(index=relay_states.index)

        # Basic time features
        features_df["hour"] = relay_states.index.hour
        features_df["weekday"] = relay_states.index.weekday
        features_df["is_weekend"] = features_df["weekday"].isin([5, 6])
        features_df["is_night"] = features_df["hour"].isin(range(22, 24)) | features_df[
            "hour"
        ].isin(range(0, 6))
        features_df["is_peak_hours"] = features_df["hour"].isin(
            range(17, 21)
        )  # Evening peak

        # Relay-specific features
        features_df["relay_state"] = relay_states
        features_df["state_duration"] = self._calculate_state_durations(relay_states)
        features_df["time_since_last_change"] = self._calculate_time_since_change(
            relay_states
        )

        return features_df

    def _calculate_state_durations(self, relay_states: pd.Series) -> pd.Series:
        """Calculate how long the relay has been in current state."""
        durations = pd.Series(index=relay_states.index, dtype=float)

        current_state = None
        state_start = None

        for timestamp, state in relay_states.items():
            if state != current_state:
                current_state = state
                state_start = timestamp
                durations.loc[timestamp] = 0
            else:
                if state_start is not None:
                    duration = timestamp - state_start
                    durations.loc[timestamp] = duration.total_seconds() / 60  # Minutes

        return durations.fillna(0)

    def _calculate_time_since_change(self, relay_states: pd.Series) -> pd.Series:
        """Calculate time since last state change."""
        state_changes = relay_states.diff().abs() > 0
        change_times = state_changes[state_changes].index

        time_since_change = pd.Series(index=relay_states.index, dtype=float)

        for timestamp in relay_states.index:
            # Find most recent change before this timestamp
            recent_changes = change_times[change_times <= timestamp]
            if len(recent_changes) > 0:
                last_change = recent_changes[-1]
                time_since = timestamp - last_change
                time_since_change.loc[timestamp] = (
                    time_since.total_seconds() / 60
                )  # Minutes
            else:
                time_since_change.loc[timestamp] = 0

        return time_since_change.fillna(0)

    def _analyze_relay_gaps(self, relay_states: pd.Series) -> Dict[str, Any]:
        """Analyze gaps in relay data."""
        if len(relay_states) < 2:
            return {"warning": "Insufficient data for gap analysis"}

        # Calculate time intervals
        time_diffs = relay_states.index.to_series().diff()

        # Expected interval (mode of differences)
        expected_interval = (
            time_diffs.mode().iloc[0]
            if not time_diffs.empty
            else pd.Timedelta(minutes=15)
        )

        # Find gaps (intervals significantly larger than expected)
        gap_threshold = expected_interval * 3
        gaps = time_diffs[time_diffs > gap_threshold]

        gap_analysis = {
            "total_gaps": len(gaps),
            "total_gap_duration_hours": gaps.sum().total_seconds() / 3600
            if len(gaps) > 0
            else 0,
            "largest_gap_hours": gaps.max().total_seconds() / 3600
            if len(gaps) > 0
            else 0,
            "expected_interval_minutes": expected_interval.total_seconds() / 60,
            "data_completeness_percentage": (
                1 - gaps.sum() / (relay_states.index[-1] - relay_states.index[0])
            )
            * 100,
        }

        return gap_analysis

    def _calculate_system_statistics(
        self, total_power: pd.Series, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate system-wide relay statistics."""
        # Peak demand analysis
        daily_peak = total_power.resample("D").max()
        monthly_peak = total_power.resample("M").max()

        # Simultaneous operation analysis
        simultaneous_rooms = 0
        total_rooms = len(relay_data)

        # Count rooms operating simultaneously at peak times
        peak_times = total_power.nlargest(100).index  # Top 100 power consumption times

        if len(peak_times) > 0:
            # For simplification, estimate from total power and average room power
            avg_room_power = (
                np.mean(list(self.room_power_ratings.values())) * 1000
            )  # Watts
            simultaneous_rooms = total_power.loc[peak_times].mean() / avg_room_power

        # Load distribution analysis
        hourly_avg = total_power.groupby(total_power.index.hour).mean()
        peak_hour = hourly_avg.idxmax()
        off_peak_hour = hourly_avg.idxmin()

        system_stats = {
            "total_rooms": total_rooms,
            "total_installed_capacity_kw": sum(self.room_power_ratings.values()),
            "peak_demand_kw": total_power.max() / 1000,
            "average_demand_kw": total_power.mean() / 1000,
            "daily_peak_avg_kw": daily_peak.mean() / 1000,
            "monthly_peak_max_kw": monthly_peak.max() / 1000,
            "peak_hour": peak_hour,
            "off_peak_hour": off_peak_hour,
            "load_factor": total_power.mean() / total_power.max()
            if total_power.max() > 0
            else 0,
            "estimated_simultaneous_rooms_at_peak": min(
                simultaneous_rooms, total_rooms
            ),
            "diversity_factor": total_power.max()
            / (sum(self.room_power_ratings.values()) * 1000),
        }

        return system_stats


class PVDataProcessor:
    """Enhanced data preprocessing specifically for PV systems with export constraints."""

    def __init__(self):
        """Initialize the PV data processor."""
        self.logger = logging.getLogger(f"{__name__}.PVDataProcessor")

    def process_pv_data(
        self, pv_data: pd.DataFrame, price_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Process PV data considering export constraints and policy changes.

        Args:
            pv_data: PV production and consumption data
            price_data: Electricity price data for export analysis

        Returns:
            Dictionary with processed PV data and analysis
        """
        if pv_data.empty:
            return {"error": "No PV data provided"}

        processed_data = {}

        # Detect export policy periods
        export_analysis = self._detect_export_periods(pv_data)
        processed_data["export_periods"] = export_analysis

        # Process production data
        production_analysis = self._process_production_data(pv_data)
        processed_data["production_analysis"] = production_analysis

        # Self-consumption analysis
        self_consumption_analysis = self._analyze_self_consumption(pv_data)
        processed_data["self_consumption"] = self_consumption_analysis

        # Export behavior analysis (if export data available)
        if "ExportPower" in pv_data.columns:
            export_behavior = self._analyze_export_behavior(pv_data, price_data)
            processed_data["export_behavior"] = export_behavior

        # Curtailment estimation
        curtailment_analysis = self._estimate_curtailment(pv_data)
        processed_data["curtailment"] = curtailment_analysis

        return processed_data

    def _detect_export_periods(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect periods with different export policies."""
        if "ExportPower" not in pv_data.columns:
            return {"warning": "No export power data available for period detection"}

        # Calculate daily export amounts
        daily_export = pv_data.resample("D")["ExportPower"].sum()

        # Find first significant export day
        export_threshold = daily_export.quantile(0.05)  # 5th percentile as minimum
        first_export_day = daily_export[daily_export > export_threshold].index.min()

        if pd.isna(first_export_day):
            return {"warning": "No significant export periods found"}

        periods = {
            "pre_export_period": {
                "start": pv_data.index.min(),
                "end": first_export_day,
                "description": "Export disabled period",
            },
            "post_export_period": {
                "start": first_export_day,
                "end": pv_data.index.max(),
                "description": "Conditional export period",
            },
            "policy_change_date": first_export_day,
        }

        return periods

    def _process_production_data(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
        """Process PV production data with quality checks."""
        # Find production column
        prod_col = None
        for col in pv_data.columns:
            if any(
                keyword in col.lower() for keyword in ["input", "production", "power"]
            ):
                if "export" not in col.lower():  # Exclude export power
                    prod_col = col
                    break

        if prod_col is None:
            return {"error": "No production power column found"}

        production = pv_data[prod_col].copy()

        # Clean negative values (nighttime/errors)
        production_clean = production.clip(lower=0)

        # Calculate production statistics
        daily_production = (
            production_clean.resample("D").sum() * 0.25 / 1000
        )  # Convert to kWh
        monthly_production = daily_production.resample("M").sum()

        # Identify peak production periods
        peak_days = daily_production.nlargest(30)  # Top 30 production days
        low_days = daily_production[daily_production > 0].nsmallest(
            30
        )  # Bottom 30 non-zero days

        # Weather-adjusted production analysis
        production_by_hour = production_clean.groupby(
            production_clean.index.hour
        ).mean()
        peak_production_hour = production_by_hour.idxmax()

        analysis = {
            "total_production_kwh": daily_production.sum(),
            "daily_avg_kwh": daily_production.mean(),
            "peak_power_kw": production_clean.max() / 1000,
            "peak_production_hour": peak_production_hour,
            "peak_production_days": peak_days.index.tolist()[:10],  # Top 10 dates
            "low_production_days": low_days.index.tolist()[:10],  # Bottom 10 dates
            "monthly_production_kwh": monthly_production.to_dict(),
            "capacity_factor": production_clean.mean() / production_clean.max()
            if production_clean.max() > 0
            else 0,
            "production_variance": daily_production.std(),
            "zero_production_days": (daily_production == 0).sum(),
        }

        return analysis

    def _analyze_self_consumption(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze self-consumption patterns."""
        if "SelfConsumption" in pv_data.columns and "InputPower" in pv_data.columns:
            production = pv_data["InputPower"]
            self_consumption = pv_data["SelfConsumption"]

            # Calculate self-consumption ratio
            daily_production = production.resample("D").sum()
            daily_self_consumption = self_consumption.resample("D").sum()

            self_consumption_ratio = daily_self_consumption / daily_production
            self_consumption_ratio = self_consumption_ratio.fillna(0).clip(0, 1)

            # Analyze patterns
            hourly_ratio = (
                self_consumption.groupby(production.index.hour).sum()
                / production.groupby(production.index.hour).sum()
            ).fillna(0)

            analysis = {
                "overall_self_consumption_ratio": self_consumption_ratio.mean(),
                "daily_self_consumption_ratio": self_consumption_ratio.to_dict(),
                "hourly_self_consumption_ratio": hourly_ratio.to_dict(),
                "peak_self_consumption_hour": hourly_ratio.idxmax(),
                "min_self_consumption_hour": hourly_ratio.idxmin(),
                "self_consumption_variability": self_consumption_ratio.std(),
                "high_self_consumption_days": self_consumption_ratio.nlargest(
                    10
                ).index.tolist(),
                "low_self_consumption_days": self_consumption_ratio.nsmallest(
                    10
                ).index.tolist(),
            }

            return analysis
        else:
            return {"warning": "Insufficient data for self-consumption analysis"}

    def _analyze_export_behavior(
        self, pv_data: pd.DataFrame, price_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Analyze export behavior patterns."""
        export_power = pv_data["ExportPower"]

        # Basic export statistics
        daily_export = export_power.resample("D").sum() * 0.25 / 1000  # Convert to kWh
        export_days = (daily_export > 0).sum()
        total_export_kwh = daily_export.sum()

        # Export timing analysis
        hourly_export = export_power.groupby(export_power.index.hour).mean()
        peak_export_hour = hourly_export.idxmax()

        analysis = {
            "total_export_kwh": total_export_kwh,
            "export_days": export_days,
            "avg_daily_export_kwh": daily_export[daily_export > 0].mean()
            if export_days > 0
            else 0,
            "peak_export_hour": peak_export_hour,
            "export_frequency": export_days / len(daily_export)
            if len(daily_export) > 0
            else 0,
            "hourly_export_pattern": hourly_export.to_dict(),
        }

        # Price correlation analysis (if price data available)
        if price_data is not None and not price_data.empty:
            # Align export and price data
            aligned_data = pv_data[["ExportPower"]].merge(
                price_data, left_index=True, right_index=True, how="inner"
            )

            if not aligned_data.empty:
                price_correlation = aligned_data["ExportPower"].corr(
                    aligned_data["price"]
                )

                # Export behavior by price quartiles
                price_quartiles = aligned_data["price"].quantile([0.25, 0.5, 0.75])

                q1_export = aligned_data[
                    aligned_data["price"] <= price_quartiles[0.25]
                ]["ExportPower"].mean()
                q2_export = aligned_data[
                    (aligned_data["price"] > price_quartiles[0.25])
                    & (aligned_data["price"] <= price_quartiles[0.5])
                ]["ExportPower"].mean()
                q3_export = aligned_data[
                    (aligned_data["price"] > price_quartiles[0.5])
                    & (aligned_data["price"] <= price_quartiles[0.75])
                ]["ExportPower"].mean()
                q4_export = aligned_data[aligned_data["price"] > price_quartiles[0.75]][
                    "ExportPower"
                ].mean()

                analysis.update(
                    {
                        "price_correlation": price_correlation,
                        "export_by_price_quartile": {
                            "q1_low_price": q1_export,
                            "q2_med_low_price": q2_export,
                            "q3_med_high_price": q3_export,
                            "q4_high_price": q4_export,
                        },
                    }
                )

        return analysis

    def _estimate_curtailment(self, pv_data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate energy curtailment during no-export periods."""
        if "InputPower" not in pv_data.columns:
            return {
                "warning": "No production data available for curtailment estimation"
            }

        production = pv_data["InputPower"]

        # If we have both self-consumption and export data
        if "SelfConsumption" in pv_data.columns and "ExportPower" in pv_data.columns:
            self_consumption = pv_data["SelfConsumption"]
            export_power = pv_data["ExportPower"]

            # Curtailment = Production - Self-consumption - Export
            curtailment = production - self_consumption - export_power
            curtailment = curtailment.clip(lower=0)  # Can't be negative

        elif "SelfConsumption" in pv_data.columns:
            # No export data - assume all excess production was curtailed
            self_consumption = pv_data["SelfConsumption"]
            curtailment = (production - self_consumption).clip(lower=0)

        else:
            # No consumption data - estimate based on typical patterns
            # This is a rough estimation and should be improved with actual data
            return {"warning": "Insufficient data for accurate curtailment estimation"}

        # Calculate curtailment statistics
        daily_curtailment = (
            curtailment.resample("D").sum() * 0.25 / 1000
        )  # Convert to kWh
        total_curtailment = daily_curtailment.sum()

        # Curtailment as percentage of production
        daily_production = production.resample("D").sum() * 0.25 / 1000
        curtailment_ratio = daily_curtailment / daily_production
        curtailment_ratio = curtailment_ratio.fillna(0)

        analysis = {
            "total_curtailment_kwh": total_curtailment,
            "avg_daily_curtailment_kwh": daily_curtailment.mean(),
            "max_daily_curtailment_kwh": daily_curtailment.max(),
            "curtailment_ratio": curtailment_ratio.mean(),
            "high_curtailment_days": daily_curtailment.nlargest(10).index.tolist(),
            "curtailment_by_month": daily_curtailment.resample("M").sum().to_dict(),
        }

        return analysis


class DataPreprocessor:
    """Main data preprocessing coordinator."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(f"{__name__}.DataPreprocessor")
        self.quality_report = {}

        # Initialize sub-processors
        self.validator = DataValidator()
        self.outlier_detector = OutlierDetector()
        self.gap_filler = GapFiller()
        self.relay_processor = RelayDataProcessor()
        self.pv_processor = PVDataProcessor()
        self.loxone_processor = LoxoneDataProcessor()

    def process_dataset(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process and clean a dataset."""
        if df.empty:
            self.logger.warning(f"Empty dataset for {data_type}")
            return df

        self.logger.info(f"Processing {data_type} dataset with {len(df)} records")

        # Remove duplicates
        df_clean = df.loc[~df.index.duplicated(keep="first")]

        # Handle missing values based on data type
        if data_type == "pv":
            df_clean = self._clean_pv_data(df_clean)
        elif data_type.startswith("room") or "temperature" in data_type:
            df_clean = self._clean_temperature_data(df_clean)
        elif data_type == "weather":
            df_clean = self._clean_weather_data(df_clean)
        elif data_type.startswith("relay"):
            df_clean = self._clean_relay_data(df_clean)
        else:
            df_clean = self._general_cleaning(df_clean)

        # Generate quality report
        self.quality_report[data_type] = self._generate_quality_report(
            df, df_clean, data_type
        )

        self.logger.info(
            f"Processed {data_type}: {len(df_clean)} clean records from {len(df)} original"
        )
        return df_clean

    def _clean_pv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean PV production data."""
        df_clean = df.copy()

        # Set negative values to 0 (no negative production)
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].clip(lower=0)

        # Fill missing values during nighttime with 0
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        night_mask = df_clean.index.hour.isin(night_hours)

        for col in numeric_cols:
            df_clean.loc[night_mask, col] = df_clean.loc[night_mask, col].fillna(0)
            # Interpolate remaining missing values during daylight
            df_clean[col] = df_clean[col].interpolate(method="linear", limit=6)

        return df_clean

    def _clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean temperature data."""
        df_clean = df.copy()

        # Remove obviously wrong temperature values
        temp_cols = [col for col in df_clean.columns if "temp" in col.lower()]
        for col in temp_cols:
            df_clean[col] = df_clean[col].where(
                (df_clean[col] >= -10) & (df_clean[col] <= 50)
            )

        # Interpolate missing values
        for col in df_clean.select_dtypes(include=["number"]).columns:
            df_clean[col] = df_clean[col].interpolate(method="linear", limit=12)

        return df_clean

    def _clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather data."""
        df_clean = df.copy()

        # Forward fill then interpolate
        df_clean = df_clean.fillna(method="ffill").fillna(method="bfill")
        df_clean = df_clean.interpolate(method="linear", limit=6)

        return df_clean

    def _clean_relay_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean relay state data."""
        df_clean = df.copy()

        # Relay data should be 0 or 1
        for col in df_clean.select_dtypes(include=["number"]).columns:
            df_clean[col] = df_clean[col].round().clip(0, 1)

        return df_clean

    def _general_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """General data cleaning."""
        df_clean = df.copy()

        # Basic interpolation for numeric columns
        for col in df_clean.select_dtypes(include=["number"]).columns:
            df_clean[col] = df_clean[col].interpolate(method="linear", limit=6)

        return df_clean

    def _generate_quality_report(
        self, original_df: pd.DataFrame, clean_df: pd.DataFrame, data_type: str
    ) -> dict:
        """Generate data quality report."""
        if original_df.empty:
            return {
                "total_records": 0,
                "clean_records": 0,
                "date_range": (None, None),
                "original_missing_percentage": 100,
                "clean_missing_percentage": 100,
                "time_gaps": [],
                "cleaning_summary": "No data available",
            }

        # Calculate missing data percentage
        original_missing = original_df.isnull().sum().sum()
        original_total = original_df.size
        original_missing_pct = (
            (original_missing / original_total * 100) if original_total > 0 else 100
        )

        clean_missing = clean_df.isnull().sum().sum()
        clean_total = clean_df.size
        clean_missing_pct = (
            (clean_missing / clean_total * 100) if clean_total > 0 else 100
        )

        # Find significant time gaps (>2 hours)
        time_gaps = []
        if not clean_df.index.empty:
            time_diffs = clean_df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
            time_gaps = [
                (gap_time, gap_duration)
                for gap_time, gap_duration in large_gaps.items()
            ]

        return {
            "total_records": len(original_df),
            "clean_records": len(clean_df),
            "date_range": (
                (original_df.index.min(), original_df.index.max())
                if not original_df.index.empty
                else (None, None)
            ),
            "original_missing_percentage": round(original_missing_pct, 2),
            "clean_missing_percentage": round(clean_missing_pct, 2),
            "time_gaps": time_gaps[:10],  # First 10 gaps
            "cleaning_summary": (
                f"Cleaned {len(original_df)} -> {len(clean_df)} records, "
                f"missing data: {original_missing_pct:.1f}% -> {clean_missing_pct:.1f}%"
            ),
        }


class LoxoneDataProcessor:
    """Enhanced data preprocessing specifically for Loxone system integration."""

    def __init__(self):
        """Initialize the Loxone data processor."""
        self.logger = logging.getLogger(f"{__name__}.LoxoneDataProcessor")
        self.loxone_adapter = LoxoneFieldAdapter()

    def process_loxone_room_data(
        self, room_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process room data with Loxone field standardization.

        Args:
            room_data: Dictionary with room names as keys, raw DataFrames as values

        Returns:
            Dictionary with standardized room data
        """
        if not room_data:
            return {}

        processed_rooms = {}

        for room_name, room_df in room_data.items():
            if room_df.empty:
                continue

            try:
                # Standardize Loxone field names
                standardized_df = self.loxone_adapter.standardize_room_data(
                    room_df, room_name
                )

                if not standardized_df.empty:
                    processed_rooms[room_name] = standardized_df
                    self.logger.info(
                        f"Processed room {room_name}: {len(standardized_df)} records"
                    )
                else:
                    self.logger.warning(
                        f"No valid data after processing room {room_name}"
                    )

            except Exception as e:
                self.logger.error(f"Failed to process room {room_name}: {e}")

        return processed_rooms

    def process_loxone_relay_data(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process relay data with power calculations and state standardization.

        Args:
            relay_data: Dictionary with room relay DataFrames

        Returns:
            Dictionary with processed relay data including power calculations
        """
        if not relay_data:
            return {}

        processed_relays = {}

        for room_name, relay_df in relay_data.items():
            if relay_df.empty:
                continue

            try:
                processed_df = pd.DataFrame(index=relay_df.index)

                # Find relay state column (typically the room name)
                relay_col = None
                for col in relay_df.columns:
                    if room_name.lower() in col.lower() or any(
                        keyword in col.lower()
                        for keyword in ["state", "relay", "heating"]
                    ):
                        relay_col = col
                        break

                if relay_col is None:
                    # Use first column as fallback
                    relay_col = relay_df.columns[0]

                # Standardize relay state (0/1)
                processed_df["relay_state"] = (relay_df[relay_col] > 0.5).astype(int)

                # Add power calculations using room configuration
                standard_room_name = self.loxone_adapter.standardize_room_name(
                    room_name
                )
                power_rating = self.loxone_adapter._get_room_power_rating(
                    standard_room_name
                )

                processed_df["power_kw"] = processed_df["relay_state"] * power_rating
                processed_df["power_w"] = processed_df["power_kw"] * 1000

                processed_relays[room_name] = processed_df
                self.logger.info(
                    f"Processed relay {room_name}: {len(processed_df)} records, {power_rating} kW"
                )

            except Exception as e:
                self.logger.error(f"Failed to process relay {room_name}: {e}")

        return processed_relays

    def process_loxone_weather_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather data including Loxone solar fields.

        Args:
            weather_data: Raw weather DataFrame

        Returns:
            Standardized weather DataFrame
        """
        if weather_data.empty:
            return pd.DataFrame()

        try:
            # Use Loxone adapter to standardize weather fields
            standardized_weather = self.loxone_adapter.standardize_weather_data(
                weather_data
            )

            self.logger.info(
                f"Processed weather data: {len(standardized_weather)} records"
            )

            # Log available solar fields
            solar_fields = ["sun_elevation", "sun_direction", "solar_irradiance"]
            available_solar = [
                f for f in solar_fields if f in standardized_weather.columns
            ]
            if available_solar:
                self.logger.info(f"Available Loxone solar fields: {available_solar}")

            return standardized_weather

        except Exception as e:
            self.logger.error(f"Failed to process weather data: {e}")
            return pd.DataFrame()

    def validate_loxone_data_integration(
        self,
        room_data: Dict[str, pd.DataFrame],
        relay_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Validate that Loxone data integration is working correctly.

        Args:
            room_data: Processed room data
            relay_data: Processed relay data
            weather_data: Processed weather data

        Returns:
            Validation report dictionary
        """
        validation_report = {
            "rooms": {},
            "relays": {},
            "weather": {},
            "integration_status": "unknown",
        }

        # Validate room data
        for room_name, room_df in room_data.items():
            validation_report["rooms"][room_name] = {
                "records": len(room_df),
                "has_temperature": "temperature" in room_df.columns,
                "has_humidity": "humidity" in room_df.columns,
                "data_quality": room_df.isnull().sum().sum() / room_df.size * 100
                if not room_df.empty
                else 100,
            }

        # Validate relay data
        for room_name, relay_df in relay_data.items():
            validation_report["relays"][room_name] = {
                "records": len(relay_df),
                "has_relay_state": "relay_state" in relay_df.columns,
                "has_power_calc": "power_kw" in relay_df.columns,
                "duty_cycle": relay_df.get("relay_state", pd.Series([0])).mean() * 100,
            }

        # Validate weather data
        if not weather_data.empty:
            validation_report["weather"] = {
                "records": len(weather_data),
                "has_temperature": "temperature" in weather_data.columns,
                "has_solar_fields": any(
                    field in weather_data.columns
                    for field in ["sun_elevation", "sun_direction", "solar_irradiance"]
                ),
                "data_quality": weather_data.isnull().sum().sum()
                / weather_data.size
                * 100,
            }

        # Overall integration status
        rooms_ok = len(validation_report["rooms"]) > 0
        relays_ok = len(validation_report["relays"]) > 0
        weather_ok = len(validation_report["weather"]) > 0

        if rooms_ok and relays_ok and weather_ok:
            validation_report["integration_status"] = "excellent"
        elif rooms_ok and (relays_ok or weather_ok):
            validation_report["integration_status"] = "good"
        elif rooms_ok or relays_ok or weather_ok:
            validation_report["integration_status"] = "partial"
        else:
            validation_report["integration_status"] = "failed"

        self.logger.info(
            f"Loxone integration status: {validation_report['integration_status']}"
        )
        return validation_report
