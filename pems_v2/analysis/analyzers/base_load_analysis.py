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
        grid_data: pd.DataFrame,  # Changed from consumption_data
        pv_data: pd.DataFrame,
        room_data: Dict[str, pd.DataFrame],
        relay_data: Optional[Dict[str, pd.DataFrame]] = None,
        ev_data: Optional[pd.DataFrame] = None,
        battery_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Analyze base load patterns.

        Args:
            grid_data: Grid import/export data
            pv_data: PV production data
            room_data: Room temperature data for heating estimation
            relay_data: Room relay states for accurate heating calculation
            ev_data: EV charging data (optional)
            battery_data: Battery charge/discharge data (optional)

        Returns:
            Dictionary with base load analysis results
        """
        self.logger.info("Starting base load analysis...")

        # --- Input Validation Logging ---
        if grid_data.empty:
            self.logger.error("Grid data is empty. Aborting base load analysis.")
            return {"error": "Empty grid data"}
        if pv_data.empty:
            self.logger.warning(
                "PV data is empty. Base load calculation may be less accurate."
            )
        if not room_data or all(
            df.empty for df in room_data.values() if isinstance(df, pd.DataFrame)
        ):
            self.logger.warning(
                "Room data is empty. Controllable load cannot be determined."
            )

        self.logger.debug(
            f"Grid data shape: {grid_data.shape}, columns: {grid_data.columns.tolist()}"
        )
        self.logger.debug(
            f"PV data shape: {pv_data.shape}, columns: {pv_data.columns.tolist()}"
        )
        self.logger.debug(
            f"Room data rooms: {list(room_data.keys()) if room_data else 'None'}"
        )

        if relay_data:
            self.logger.debug(f"Relay data rooms: {list(relay_data.keys())}")
        if ev_data is not None and not ev_data.empty:
            self.logger.debug(
                f"EV data shape: {ev_data.shape}, columns: {ev_data.columns.tolist()}"
            )
        if battery_data is not None and not battery_data.empty:
            self.logger.debug(
                f"Battery data shape: {battery_data.shape}, columns: {battery_data.columns.tolist()}"
            )

        results = {}

        # Prepare heating data from room data and relay states
        heating_data = self._prepare_heating_data(room_data, relay_data)

        # Calculate base load using energy conservation approach
        base_load = self._calculate_base_load(
            grid_data, pv_data, heating_data, ev_data, battery_data
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
        grid_data: pd.DataFrame,  # Changed from consumption_data
        pv_data: pd.DataFrame,
        heating_data: pd.DataFrame,  # Changed from room_data
        ev_data: Optional[pd.DataFrame] = None,
        battery_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate base load using an energy conservation approach.
        """
        self.logger.info("Calculating base load with new energy conservation logic.")

        # --- 1. Ensure proper datetime index and align data ---
        # Convert index to datetime if needed
        if not isinstance(grid_data.index, pd.DatetimeIndex):
            if "timestamp" in grid_data.index.names or "timestamp" in grid_data.columns:
                grid_data = grid_data.reset_index().set_index("timestamp")
            else:
                self.logger.warning("Grid data does not have proper datetime index")
                return pd.DataFrame()

        if not isinstance(pv_data.index, pd.DatetimeIndex):
            if "timestamp" in pv_data.index.names or "timestamp" in pv_data.columns:
                pv_data = pv_data.reset_index().set_index("timestamp")
            else:
                self.logger.warning("PV data does not have proper datetime index")
                return pd.DataFrame()

        if (
            heating_data is not None
            and not heating_data.empty
            and not isinstance(heating_data.index, pd.DatetimeIndex)
        ):
            if (
                "timestamp" in heating_data.index.names
                or "timestamp" in heating_data.columns
            ):
                heating_data = heating_data.reset_index().set_index("timestamp")

        if (
            battery_data is not None
            and not battery_data.empty
            and not isinstance(battery_data.index, pd.DatetimeIndex)
        ):
            if (
                "timestamp" in battery_data.index.names
                or "timestamp" in battery_data.columns
            ):
                battery_data = battery_data.reset_index().set_index("timestamp")

        # Find a common time range across all essential data sources
        common_index = grid_data.index.intersection(pv_data.index)
        if battery_data is not None and not battery_data.empty:
            common_index = common_index.intersection(battery_data.index)
        if common_index.empty:
            self.logger.warning(
                "No common time index found between grid, PV, and battery data."
            )
            return pd.DataFrame()

        # Resample and align all data sources
        freq = "15min"  # Updated to use new pandas frequency string
        grid = grid_data.resample(freq).mean().reindex(common_index, method="nearest")
        pv = pv_data.resample(freq).mean().reindex(common_index, method="nearest")

        # Handle heating data
        if heating_data is not None and not heating_data.empty:
            heating = (
                heating_data.resample(freq)
                .sum()
                .reindex(common_index, method="nearest")
            )
        else:
            heating = pd.DataFrame(index=common_index)

        # --- 2. Extract individual energy flows (in Watts) ---
        # Use specific column names from your documentation
        grid_import = grid.get(
            "ACPowerToUser", pd.Series(0, index=common_index)
        ).fillna(0)
        grid_export = grid.get(
            "ACPowerToGrid", pd.Series(0, index=common_index)
        ).fillna(0)
        pv_production = pv.get("InputPower", pd.Series(0, index=common_index)).fillna(0)

        # If specific columns don't exist, fall back to generic column detection
        if grid_import.sum() == 0:
            grid_import_cols = [
                col
                for col in grid.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["grid_import", "import", "to_user", "consumption"]
                )
            ]
            if grid_import_cols:
                grid_import = grid[grid_import_cols[0]].fillna(0)

        if grid_export.sum() == 0:
            grid_export_cols = [
                col
                for col in grid.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["grid_export", "export", "to_grid"]
                )
            ]
            if grid_export_cols:
                grid_export = grid[grid_export_cols[0]].fillna(0)

        if pv_production.sum() == 0:
            pv_power_cols = [col for col in pv.columns if "power" in col.lower()]
            if pv_power_cols:
                pv_production = pv[pv_power_cols[0]].fillna(0)

        battery_charge = pd.Series(0, index=common_index)
        battery_discharge = pd.Series(0, index=common_index)

        if battery_data is not None and not battery_data.empty:
            battery = (
                battery_data.resample(freq)
                .mean()
                .reindex(common_index, method="nearest")
            )

            # Look for specific charge/discharge columns
            charge_cols = [
                col
                for col in battery.columns
                if "charge" in col.lower() and "power" in col.lower()
            ]
            discharge_cols = [
                col
                for col in battery.columns
                if "discharge" in col.lower() and "power" in col.lower()
            ]

            if charge_cols and discharge_cols:
                battery_charge = battery[charge_cols[0]].fillna(0)
                battery_discharge = battery[discharge_cols[0]].fillna(0)
            else:
                # Try generic ChargePower/DischargePower columns
                battery_charge = battery.get(
                    "ChargePower", pd.Series(0, index=common_index)
                ).fillna(0)
                battery_discharge = battery.get(
                    "DischargePower", pd.Series(0, index=common_index)
                ).fillna(0)

        heating_load = heating.get(
            "total_heating_power", pd.Series(0, index=common_index)
        ).fillna(0)

        # If total_heating_power doesn't exist, calculate it from room data
        if heating_load.sum() == 0 and not heating.empty:
            heating_load = self._estimate_heating_consumption_from_df(
                heating, common_index
            )

        # Handle optional EV data
        ev_charge = pd.Series(0, index=common_index)
        if ev_data is not None and not ev_data.empty:
            ev_resampled = (
                ev_data.resample(freq).mean().reindex(common_index, method="nearest")
            )
            ev_cols = [
                col
                for col in ev_resampled.columns
                if any(keyword in col.lower() for keyword in ["power", "charge", "ev"])
            ]
            if ev_cols:
                ev_charge = ev_resampled[ev_cols[0]].fillna(0)

        # --- 3. Calculate Total House Load using energy conservation ---
        # Total energy supplied to the house
        power_sources = pv_production + grid_import + battery_discharge

        # Energy directed away from the house loads
        power_sinks = grid_export + battery_charge

        # The remainder is the total power consumed by the house
        total_house_load = power_sources - power_sinks

        # --- 4. Calculate Base Load using statistical approach ---
        # Base load should be the minimum baseline consumption, not total minus other loads
        # Use a rolling minimum approach to identify the true baseline

        # First, try the energy conservation approach
        controllable_load = heating_load + ev_charge
        conservation_base_load = total_house_load - controllable_load
        conservation_base_load_clipped = conservation_base_load.clip(lower=0)

        # Second, use statistical approach: rolling quantile for robust baseline detection
        # A 24-hour window and a low quantile (5%) are robust settings for base load identification
        window_hours = 24
        window_size = (
            window_hours * 4
        )  # 24 hours * 4 (15-min intervals per hour) = 96 periods
        window_size = min(window_size, len(total_house_load))
        if window_size < 4:
            window_size = 4

        # Use 24-hour rolling window with 5% quantile for robust base load estimation
        statistical_base_load = total_house_load.rolling(
            window=window_size, center=True, min_periods=max(1, window_size // 4)
        ).quantile(0.05)

        # Primary method: Use statistical approach as more robust
        # Statistical method is more reliable as it doesn't depend on accurate controllable load estimates
        base_load_clipped = statistical_base_load.clip(lower=0)
        calculation_method = "statistical_minimum"

        # Secondary validation: Check conservation approach for comparison
        if (
            controllable_load.sum() > total_house_load.sum() * 0.1
        ):  # If controllable loads are > 10%
            conservation_diff = abs(
                conservation_base_load_clipped.mean() - statistical_base_load.mean()
            )
            statistical_std = statistical_base_load.std()

            # Log significant differences for analysis
            if conservation_diff > statistical_std:
                self.logger.warning(
                    f"Large difference between methods: conservation={conservation_base_load_clipped.mean():.1f}W, "
                    f"statistical={statistical_base_load.mean():.1f}W (diff={conservation_diff:.1f}W)"
                )

        # --- 5. Assemble the final DataFrame for analysis ---
        analysis_df = pd.DataFrame(index=common_index)
        analysis_df["total_house_load"] = total_house_load
        analysis_df["heating_load"] = heating_load
        analysis_df["ev_load"] = ev_charge
        analysis_df["controllable_load"] = controllable_load
        analysis_df["base_load"] = base_load_clipped  # This is the corrected base load

        # Add time features for subsequent analysis
        analysis_df["hour"] = analysis_df.index.hour
        analysis_df["weekday"] = analysis_df.index.weekday
        analysis_df["is_weekend"] = analysis_df["weekday"].isin([5, 6])
        analysis_df["month"] = analysis_df.index.month

        # Add metadata about calculation method
        analysis_df.attrs["calculation_method"] = calculation_method
        analysis_df.attrs["controllable_load_percentage"] = (
            controllable_load.sum() / total_house_load.sum() * 100
            if total_house_load.sum() > 0
            else 0
        )

        avg_base_load = base_load_clipped.mean()
        self.logger.info(
            f"Successfully calculated base load using {calculation_method} method. "
            f"Average statistical base load: {avg_base_load:.1f}W. "
            f"Average: {analysis_df['base_load'].mean():.2f}W, "
            f"Controllable load: {analysis_df.attrs['controllable_load_percentage']:.1f}%"
        )

        return analysis_df

    def _prepare_heating_data(
        self,
        room_data: Dict[str, pd.DataFrame],
        relay_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Prepare heating data from room data and relay states."""
        heating_data_dict = {}

        # Prioritize relay data if available (more accurate for heating states)
        if relay_data:
            self.logger.info(
                f"Using relay data for heating calculation from {len(relay_data)} rooms"
            )
            for room_name, relay_df in relay_data.items():
                if relay_df.empty:
                    continue

                # Look for heating relay columns
                heating_cols = [
                    col
                    for col in relay_df.columns
                    if any(
                        keyword in col.lower()
                        for keyword in ["heating", "heat", "state", "relay"]
                    )
                ]

                for col in heating_cols:
                    # Use room name as prefix for column name
                    new_col_name = f"{room_name}_{col}"
                    heating_data_dict[new_col_name] = relay_df[col]

        # Fallback to room data if relay data not available or insufficient
        elif room_data:
            self.logger.info(
                f"Using room temperature data for heating estimation from {len(room_data)} rooms"
            )
            for room_name, room_df in room_data.items():
                if room_df.empty:
                    continue

                # Look for heating status columns
                heating_cols = [
                    col
                    for col in room_df.columns
                    if any(
                        keyword in col.lower()
                        for keyword in ["heating", "heat", "state"]
                    )
                ]

                for col in heating_cols:
                    # Use room name as prefix for column name
                    new_col_name = f"{room_name}_{col}"
                    heating_data_dict[new_col_name] = room_df[col]

        if not heating_data_dict:
            self.logger.warning("No heating data found in either relay or room data")
            return pd.DataFrame()

        heating_df = pd.DataFrame(heating_data_dict)
        self.logger.info(
            f"Prepared heating data with {len(heating_df.columns)} heating columns"
        )
        return heating_df

    def _estimate_heating_consumption_from_df(
        self, heating_df: pd.DataFrame, target_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Estimate heating consumption from a heating DataFrame."""
        total_heating = pd.Series(0, index=target_index)

        # Look for heating-related columns
        heating_cols = [
            col
            for col in heating_df.columns
            if any(
                keyword in col.lower()
                for keyword in ["heating", "heat", "state", "power"]
            )
        ]

        for col in heating_cols:
            # If it's a power column, use directly
            if "power" in col.lower():
                total_heating += heating_df[col].fillna(0)
            else:
                # Assume it's a status column, estimate power based on room
                room_name = col.split("_")[0] if "_" in col else "unknown"
                try:
                    from config.settings import PEMSSettings

                    settings = PEMSSettings()
                    room_power_kw = settings.get_room_power(room_name)
                    heating_power = (
                        heating_df[col].fillna(0) * room_power_kw * 1000
                    )  # Convert kW to W
                    total_heating += heating_power
                except ImportError:
                    # Fallback to default power estimate
                    default_power_w = 1000  # 1kW default
                    heating_power = heating_df[col].fillna(0) * default_power_w
                    total_heating += heating_power

        return total_heating

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
                    target_index.freq or "15min"
                ).mean()
                heating_aligned = heating_resampled.reindex(
                    target_index, method="nearest"
                )

                # Use actual room power rating from configuration
                try:
                    from config.settings import PEMSSettings

                    settings = PEMSSettings()
                    room_power_kw = settings.get_room_power(room_name)
                except ImportError:
                    # Fallback to default room power
                    room_power_kw = 1.0  # 1 kW default
                heating_power = (
                    heating_aligned.fillna(0) * room_power_kw * 1000
                )  # Convert kW to W

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
            "calculation_method": base_load_data.attrs.get(
                "calculation_method", "unknown"
            ),
            "controllable_load_detected": base_load_data.attrs.get(
                "controllable_load_percentage", 0
            ),
        }

        # Percentage of total consumption
        if "total_house_load" in base_load_data.columns:
            total_house_load = base_load_data["total_house_load"]
            stats["base_load_percentage"] = (
                (base_load.sum() / total_house_load.sum() * 100)
                if total_house_load.sum() > 0
                else 0
            )

            # Breakdown of other components
            components = [
                "heating_load",
                "ev_load",
            ]
            for component in components:
                if component in base_load_data.columns:
                    component_sum = base_load_data[component].sum()
                    stats[f"{component}_percentage"] = (
                        (component_sum / total_house_load.sum() * 100)
                        if total_house_load.sum() > 0
                        else 0
                    )

            # Special handling for net battery impact
            if "net_battery_power" in base_load_data.columns:
                net_battery_sum = base_load_data["net_battery_power"].sum()
                stats["net_battery_impact_percentage"] = (
                    (net_battery_sum / total_house_load.sum() * 100)
                    if total_house_load.sum() > 0
                    else 0
                )
                stats["battery_charging_energy_kwh"] = (
                    base_load_data.get("battery_charge_power", pd.Series()).sum()
                    * 0.25
                    / 1000
                )
                stats["battery_discharging_energy_kwh"] = (
                    base_load_data.get("battery_discharge_power", pd.Series()).sum()
                    * 0.25
                    / 1000
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
            daily_load = base_load.resample("D").mean().dropna()

            # Check if we have enough non-null daily data points
            if (
                len(daily_load) < 730
            ):  # At least 2 years of daily data for reliable seasonal patterns
                return {
                    "warning": f"Insufficient daily data points for seasonal analysis (have {len(daily_load)}, need at least 730)"
                }

            # STL decomposition
            # Use a smaller seasonal period if we don't have enough data for annual seasonality
            seasonal_period = min(365, len(daily_load) // 3)  # At least 3 cycles
            if seasonal_period < 7:  # If less than a week, try weekly pattern
                seasonal_period = 7

            # Ensure no NaN values in the data
            daily_load_clean = daily_load.dropna()
            if len(daily_load_clean) < seasonal_period * 2:
                return {
                    "warning": f"Insufficient clean data for STL decomposition (have {len(daily_load_clean)}, need at least {seasonal_period * 2})"
                }

            stl = STL(
                daily_load_clean, seasonal=seasonal_period, period=seasonal_period
            )
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
                hourly_data = day_data.resample("h").mean().interpolate()
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
        hourly_consumption = base_load.resample("h").sum()
        daily_consumption = base_load.resample("D").sum()
        monthly_consumption = base_load.resample("ME").sum()

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
