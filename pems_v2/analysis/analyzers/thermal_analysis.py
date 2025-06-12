"""
Thermal Dynamics Analysis for PEMS v2.

Analyzes thermal dynamics per room with Loxone integration:
1. Calculate thermal parameters (heat-up rate, cool-down rate, time constant)
2. Use system identification (ARX model, state-space model)
3. Account for solar gains, internal gains, adjacent room heat transfer
4. Handle Loxone field naming conventions and data structure
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    from analysis.reports.report_generator import ReportGenerator
    from analysis.utils.loxone_adapter import (LoxoneDataIntegrator,
                                               LoxoneFieldAdapter)
except ImportError:
    # Fallback for testing
    LoxoneDataIntegrator = None
    ReportGenerator = None

    class LoxoneFieldAdapter:
        @staticmethod
        def standardize_room_data(room_df, room_name):
            return room_df

        @staticmethod
        def standardize_weather_data(weather_df):
            return weather_df


# DataLoader is a generic type, define it separately
try:
    from typing import Protocol

    class DataLoader(Protocol):
        def get_dataset(self, name: str):
            ...

except ImportError:
    DataLoader = None


# --- Constants for Physical Plausibility ---
# Realistic thermal resistance range for a room in K/W (increased min to avoid unphysical values)
R_MIN, R_MAX = 0.008, 0.5
# Realistic thermal capacitance range for a room in MJ/K (increased min for better stability)
C_MIN_MJ, C_MAX_MJ = 2.0, 100.0
# Realistic time constant range in hours - extended for winter conditions (increased min)
TAU_MIN, TAU_MAX = 3.0, 200.0


class ThermalAnalyzer:
    """Analyzes thermal dynamics of rooms based on temperature, weather, and heating data."""

    def __init__(
        self,
        data_loader: DataLoader,
        settings: Dict[str, Any],
        report_generator: ReportGenerator,
    ):
        self.data_loader = data_loader
        self.settings = settings.get("thermal_analysis", {})
        self.system_settings = settings
        self.report = report_generator
        self.logger = logging.getLogger(f"{__name__}.ThermalAnalyzer")
        if LoxoneDataIntegrator:
            self.integrator = LoxoneDataIntegrator()
        else:
            self.integrator = None
        # Load room power ratings from settings
        self.room_power_ratings_kw = self.system_settings.get(
            "room_power_ratings_kw", {}
        )
        self.room_configs = {
            room["name"]: room for room in self.system_settings.get("rooms", [])
        }

        # Configuration for sustained heating analysis
        self.min_heating_duration_hours = self.settings.get(
            "min_heating_duration_hours", 2.0
        )
        self.min_non_heating_duration_hours = self.settings.get(
            "min_non_heating_duration_hours", 3.0
        )
        self.decay_analysis_hours = self.settings.get(
            "decay_analysis_hours", 8.0
        )  # How long to analyze decay - extended for winter

    def get_target_temp(self, room_name: str, hour: int) -> float:
        """Get target temperature for a room at a specific hour.

        Args:
            room_name: Name of the room
            hour: Hour of day (0-23)

        Returns:
            Target temperature in °C
        """
        if self.settings and self.settings.thermal_settings:
            return self.settings.thermal_settings.get_target_temp(room_name, hour)
        else:
            # Fallback to default setpoints
            if 6 <= hour < 22:  # Daytime
                return 21.0
            else:  # Nighttime
                return 19.0

    def analyze_room_dynamics(
        self,
        room_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame,
        relay_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze thermal dynamics for all rooms with Loxone integration.

        Args:
            room_data: Dictionary of room DataFrames with Loxone temperature data
            weather_data: Weather data with outdoor temperature
            relay_data: Optional relay state data for heating period detection

        Returns:
            Dictionary with thermal analysis results for each room
        """
        self.logger.info(
            "Starting thermal dynamics analysis with Loxone integration..."
        )

        # --- Input Validation Logging ---
        if not room_data:
            self.logger.error("Room data is empty. Aborting thermal analysis.")
            return {"error": "Empty room data"}
        if weather_data.empty:
            self.logger.warning(
                "Weather data is empty. Outdoor temperature correlation will be limited."
            )

        self.logger.debug(f"Room data rooms: {list(room_data.keys())}")
        for room, df in room_data.items():
            if isinstance(df, pd.DataFrame):
                self.logger.debug(
                    f"Room {room} data shape: {df.shape}, columns: {df.columns.tolist()}"
                )

        if not weather_data.empty:
            self.logger.debug(
                f"Weather data shape: {weather_data.shape}, columns: {weather_data.columns.tolist()}"
            )
        if relay_data:
            self.logger.debug(f"Relay data rooms: {list(relay_data.keys())}")
        else:
            self.logger.debug("No relay data provided for heating period detection")

        # Prepare data using Loxone integrator if relay data is provided
        if relay_data is not None and self.integrator is not None:
            self.logger.info("Preparing thermal analysis data with relay integration")
            self.logger.info(
                f"Relay data provided for rooms: {list(relay_data.keys()) if relay_data else 'None'}"
            )
            for room, relay_df in relay_data.items():
                if not relay_df.empty:
                    self.logger.info(
                        f"Relay data for {room}: {relay_df.shape[0]} records"
                    )
                else:
                    self.logger.info(f"Empty relay data for {room}")
            # Store relay data in integrator for adaptive analysis
            self.integrator._processed_relay_data = relay_data
            (
                standardized_rooms,
                standardized_weather,
            ) = self.integrator.prepare_thermal_analysis_data(
                room_data, relay_data, weather_data
            )
        else:
            # Standardize room data without relay integration
            self.logger.info("Standardizing room data without relay integration")
            standardized_rooms = {}
            for room_name, room_df in room_data.items():
                standardized_rooms[
                    room_name
                ] = LoxoneFieldAdapter.standardize_room_data(room_df, room_name)
            standardized_weather = LoxoneFieldAdapter.standardize_weather_data(
                weather_data
            )

        results = {}

        # Filter out external environment data (not actual rooms)
        external_environments = [
            "outside",
            "outdoor",
            "external",
            "environment",
            "weather",
        ]
        interior_rooms = {
            room_name: room_df
            for room_name, room_df in standardized_rooms.items()
            if not any(env in room_name.lower() for env in external_environments)
        }

        self.logger.info(
            f"Filtered to {len(interior_rooms)} interior rooms (excluded external environment data)"
        )

        # Analyze each interior room individually using standardized data
        for room_name, room_df in interior_rooms.items():
            self.logger.info(f"Analyzing thermal dynamics for room: {room_name}")
            self.logger.info(f"Room {room_name} processed: {room_df.shape[0]} records")
            if "heating_on" not in room_df.columns:
                self.logger.info(f"Room {room_name} has no heating_on column")

            try:
                # Store room name for power calculations
                self._current_room_name = room_name
                room_results = self._analyze_single_room(
                    room_df, standardized_weather, room_name
                )
                results[room_name] = room_results
            except Exception as e:
                self.logger.error(f"Failed to analyze room {room_name}: {e}")
                results[room_name] = {"error": str(e)}

        # Analyze room coupling (heat transfer between rooms)
        if len(results) > 1:
            results["room_coupling"] = self._analyze_room_coupling(standardized_rooms)

        self.logger.info("Thermal dynamics analysis completed")
        return results

    def _prepare_thermal_data(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
        """Integrates room temperatures, relay states, and outdoor temperatures."""
        self.logger.info("Preparing thermal analysis data with relay integration")
        if self.integrator:
            # Get room data directly from data loader
            all_rooms_data = {}
            room_names = [
                "chodba_dole",
                "chodba_nahore",
                "hosti",
                "koupelna_dole",
                "koupelna_nahore",
                "kuchyne",
                "loznice",
                "obyvak",
                "pokoj_1",
                "pokoj_2",
                "pracovna",
                "satna_dole",
                "satna_nahore",
                "spajz",
                "technicka_mistnost",
                "zachod",
                "zadveri",
                "outside",
            ]
            for room in room_names:
                room_data = self.data_loader.get_dataset(f"room_{room}")
                if room_data is not None:
                    all_rooms_data[room] = room_data
        else:
            all_rooms_data = {}

        outdoor_temp = self.data_loader.get_dataset("outdoor_temp")
        weather_data = self.data_loader.get_dataset("weather")

        outdoor_temp_df = None
        if outdoor_temp is not None and not outdoor_temp.empty:
            self.logger.info(
                "Using outdoor temperature data from primary 'teplomer' sensor."
            )
            outdoor_temp_df = outdoor_temp.rename(columns={"value": "outdoor_temp"})
        elif weather_data is not None and not weather_data.empty:
            self.logger.info(
                "Falling back to weather service 'temperature_2m' for outdoor temp."
            )
            outdoor_temp_df = weather_data[["temperature_2m"]].rename(
                columns={"temperature_2m": "outdoor_temp"}
            )

        if outdoor_temp_df is None:
            self.logger.error(
                "No outdoor temperature data available. Thermal analysis will be limited."
            )

        return all_rooms_data, outdoor_temp_df

    def _merge_with_outdoor_temp(
        self, room_df: pd.DataFrame, outdoor_temp_df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Merges room data with outdoor temperature data, ensuring timezone consistency."""
        if outdoor_temp_df is None:
            self.logger.warning(
                "Cannot merge data: Outdoor temperature data is missing."
            )
            return (
                room_df  # Return as-is, downstream functions must handle missing data
            )

        try:
            # Ensure both dataframes are timezone-aware (UTC)
            if room_df.index.tz is None:
                room_df.index = room_df.index.tz_localize("UTC")
            if outdoor_temp_df.index.tz is None:
                outdoor_temp_df.index = outdoor_temp_df.index.tz_localize("UTC")

            # Resample outdoor temp to match room data frequency for a clean merge
            resampled_outdoor = outdoor_temp_df.reindex(
                room_df.index, method="pad", limit=1
            )

            merged_df = room_df.join(resampled_outdoor)

            # Check data quality post-merge
            valid_outdoor_temps = merged_df["outdoor_temp"].notna().sum()
            if valid_outdoor_temps == 0:
                self.logger.error(
                    "Merge resulted in zero valid outdoor temperature points. Check data alignment."
                )
                return None

            self.logger.info(
                f"Successfully merged room and outdoor temperature data. "
                f"Outdoor temp range: {merged_df['outdoor_temp'].min():.1f}°C to {merged_df['outdoor_temp'].max():.1f}°C"
            )
            return merged_df

        except Exception as e:
            self.logger.error(
                f"Error during merging of outdoor temperature data: {e}", exc_info=True
            )
            return None

    def _analyze_single_room(
        self, room_df: pd.DataFrame, weather_data: pd.DataFrame, room_name: str
    ) -> Dict[str, Any]:
        """Analyze thermal dynamics for a single room."""
        if room_df.empty:
            return {"error": "No room data available"}

        # Merge room and weather data
        merged_data = self._merge_room_weather_data(room_df, weather_data)

        if merged_data.empty:
            return {"error": "No merged room-weather data available"}

        results = {}

        # Basic thermal statistics
        results["basic_stats"] = self._calculate_basic_thermal_stats(
            merged_data, room_name
        )

        # Heat-up and cool-down analysis
        results["heatup_cooldown"] = self._analyze_heatup_cooldown(merged_data)

        # Thermal time constant identification
        results["time_constant"] = self._identify_time_constant(merged_data)

        # Heat loss coefficient
        results["heat_loss"] = self._calculate_heat_loss_coefficient(merged_data)

        # Solar gain analysis
        results["solar_gains"] = self._analyze_solar_gains(merged_data)

        # RC model parameters (enhanced for relay systems)
        # Use the improved RC parameter estimation
        power_rating_kw = self.room_power_ratings_kw.get(room_name, 0)
        if power_rating_kw == 0:
            self.logger.warning(
                f"No power rating for room {room_name}, using default 2000W"
            )
            power_rating_w = 2000
        else:
            power_rating_w = power_rating_kw * 1000  # Convert kW to W
        results["rc_parameters"] = self._estimate_rc_parameters(
            merged_data, power_rating_w
        )

        # ARX model identification
        results["arx_model"] = self._fit_arx_model(merged_data)

        # Setpoint tracking analysis
        results["setpoint_analysis"] = self._analyze_setpoint_tracking(merged_data)

        # Thermal comfort analysis
        results["comfort_analysis"] = self._analyze_thermal_comfort(merged_data)

        # Add sustained heating cycle analysis
        try:
            # Get relay data from the integrator if available
            relay_data = getattr(self.integrator, "_processed_relay_data", {})
            outdoor_temp = (
                weather_data.get("outdoor_temp", pd.Series())
                if isinstance(weather_data, pd.DataFrame)
                else pd.Series()
            )

            # Find matching relay states for this room
            relay_states = pd.Series(dtype=int)
            room_key = self._find_matching_relay_room(room_name, relay_data.keys())
            if room_key and room_key in relay_data:
                relay_df = relay_data[room_key]
                if "relay_state" in relay_df.columns:
                    relay_states = relay_df["relay_state"].reindex(
                        merged_data.index, method="nearest", fill_value=0
                    )

            if len(relay_states) > 0 and len(outdoor_temp) > 0:
                sustained_analysis = self.analyze_sustained_heating_cycles(
                    merged_data["room_temp"],
                    outdoor_temp.reindex(merged_data.index, method="nearest"),
                    relay_states,
                )
                results["sustained_heating_analysis"] = sustained_analysis

                # Also calculate real heating usage from relay data
                heating_usage = self._calculate_real_heating_usage(
                    room_name, relay_data
                )
                results["heating_usage"] = heating_usage

                # Replace misleading heating_percentage with actual heating usage
                if (
                    "basic_stats" in results
                    and heating_usage.get("heating_data_source") == "actual_relay"
                ):
                    results["basic_stats"]["heating_percentage"] = heating_usage.get(
                        "actual_heating_usage_pct", 0
                    )
                    results["basic_stats"]["heating_data_source"] = heating_usage.get(
                        "heating_data_source", "inference"
                    )
            else:
                results["sustained_heating_analysis"] = {
                    "warning": "No relay data available for sustained heating analysis"
                }
                results["heating_usage"] = {
                    "heating_data_source": "no_relay_data",
                    "actual_heating_usage_pct": 0,
                }

        except Exception as e:
            self.logger.warning(
                f"Sustained heating analysis failed for {room_name}: {e}"
            )
            results["sustained_heating_analysis"] = {"error": str(e)}

        return results

    def _merge_room_weather_data(
        self, room_df: pd.DataFrame, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge room temperature data with weather data."""
        # Ensure we have temperature column
        temp_col = None
        for col in ["temperature", "value", "temp"]:
            if col in room_df.columns:
                temp_col = col
                break

        if temp_col is None:
            self.logger.warning("No temperature column found in room data")
            return pd.DataFrame()

        # Prepare room data - copy the entire DataFrame first
        room_clean = room_df.copy()
        # Rename temperature column to standard name
        room_clean.rename(columns={temp_col: "room_temp"}, inplace=True)

        # Apply upstream data cleaning with rolling median outlier removal
        room_clean = self._clean_temperature_data(room_clean)

        # Add heating status if available
        heating_cols = [
            col
            for col in room_df.columns
            if "heating" in col.lower()
            or "heat" in col.lower()
            or col.lower() == "relay_state"
        ]

        if heating_cols:
            # Use the first heating column found
            heating_col = heating_cols[0]
            self.logger.debug(f"Found heating column '{heating_col}' in room data")

            if heating_col == "relay_state":
                # Convert relay state to heating_on (binary)
                room_clean["heating_on"] = (room_df[heating_col] > 0).astype(int)
                # Debug relay state timing
                relay_on_count = room_clean["heating_on"].sum()
                self.logger.debug(
                    f"Relay states - ON intervals: {relay_on_count}, Total points: {len(room_clean)}"
                )
            else:
                room_clean["heating_on"] = room_df[heating_col]
        elif "heating_on" in room_df.columns:
            # Direct heating_on column (already standardized)
            self.logger.debug("Using existing 'heating_on' column")
            room_clean["heating_on"] = room_df["heating_on"]
        else:
            # Infer heating from temperature changes
            self.logger.info("No heating column found, using inference")
            room_clean["heating_on"] = self._infer_heating_status(
                room_clean["room_temp"]
            )

        # Log heating statistics for debugging
        if "heating_on" in room_clean.columns:
            heating_stats = room_clean["heating_on"].value_counts()
            heating_pct = (room_clean["heating_on"] == 1).mean() * 100
            self.logger.debug(
                f"Heating statistics: {heating_stats.to_dict()}, {heating_pct:.1f}% heating time"
            )

        # Add setpoint if available
        if "setpoint" in room_df.columns:
            room_clean["setpoint"] = room_df["setpoint"]

        # Merge with weather/outdoor temperature data
        if not weather_data.empty:
            self.logger.info(
                f"Weather data available with columns: {list(weather_data.columns)}"
            )
            self.logger.info(f"Weather data shape: {weather_data.shape}")
            self.logger.info(f"Weather data index type: {type(weather_data.index)}")
            self.logger.info(f"Room data index type: {type(room_clean.index)}")

            # Check for possible temperature column names (outdoor_temp from teplomer or temperature_2m from forecast)
            temp_column = None
            for col in ["outdoor_temp", "temperature_2m", "temperature", "temp"]:
                if col in weather_data.columns:
                    temp_column = col
                    break

            if temp_column:
                self.logger.info(f"Using temperature column: {temp_column}")

                # Ensure both dataframes have datetime index and are sorted
                if not isinstance(weather_data.index, pd.DatetimeIndex):
                    self.logger.warning(
                        "Weather data index is not DatetimeIndex, attempting conversion"
                    )
                    weather_data.index = pd.to_datetime(weather_data.index)

                # Ensure weather data index is sorted
                if not weather_data.index.is_monotonic_increasing:
                    self.logger.debug("Sorting non-monotonic weather data index")
                    weather_data = weather_data.sort_index()

                if not isinstance(room_clean.index, pd.DatetimeIndex):
                    self.logger.warning(
                        "Room data index is not DatetimeIndex, attempting conversion"
                    )
                    room_clean.index = pd.to_datetime(room_clean.index)

                # Ensure room data index is sorted
                if not room_clean.index.is_monotonic_increasing:
                    self.logger.debug("Sorting non-monotonic room data index")
                    room_clean = room_clean.sort_index()

                # Resample weather data to match room data frequency (5 minutes)
                try:
                    weather_resampled = (
                        weather_data[[temp_column]]
                        .resample("5min")
                        .interpolate(method="time", limit=3, limit_direction="both")
                    )
                    weather_resampled.columns = ["outdoor_temp"]

                    # Use outer join to see what data we have
                    merged = room_clean.join(weather_resampled, how="left")

                    # Check if we got any outdoor temperature data
                    outdoor_temp_count = merged["outdoor_temp"].notna().sum()
                    self.logger.info(
                        f"Merged data has {outdoor_temp_count} valid outdoor temperature records out of {len(merged)}"
                    )

                    if outdoor_temp_count > 0:
                        # Forward fill missing outdoor temperature values
                        merged["outdoor_temp"] = merged["outdoor_temp"].ffill().bfill()
                    else:
                        self.logger.warning(
                            "No outdoor temperature data was merged successfully"
                        )

                except Exception as e:
                    self.logger.error(f"Error resampling weather data: {e}")
                    merged = room_clean
            else:
                merged = room_clean
                self.logger.warning(
                    f"No temperature column found in weather data. Available columns: {list(weather_data.columns)}"
                )
        else:
            merged = room_clean
            self.logger.warning("Weather data is empty")

        # Calculate temperature difference if outdoor temperature is available
        if "outdoor_temp" in merged.columns:
            merged["temp_diff"] = merged["room_temp"] - merged["outdoor_temp"]
            self.logger.info(
                f"Successfully merged room and outdoor temperature data. Outdoor temp range: {merged['outdoor_temp'].min():.1f}°C to {merged['outdoor_temp'].max():.1f}°C"
            )
        else:
            self.logger.warning(
                "No outdoor temperature data available for thermal analysis"
            )

        # Add time features
        merged["hour"] = merged.index.hour
        merged["weekday"] = merged.index.weekday

        # Only drop rows where essential columns are NaN
        # Keep heating_on data even if outdoor_temp is missing
        return merged.dropna(subset=["room_temp", "heating_on"])

    def _infer_heating_status(self, temperature: pd.Series) -> pd.Series:
        """Deprecated: Use actual relay data instead of inference."""
        # Return zeros - we should use actual relay data
        return pd.Series(0, index=temperature.index)

    def _clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply upstream data cleaning with rolling median outlier removal.

        This removes spurious spikes from sensor readings that can ruin
        derivative calculations (dT/dt) and statistical fits.
        """
        if "room_temp" not in df.columns:
            return df

        df_clean = df.copy()

        # Apply rolling median filter to remove outliers
        # Window of 3-5 samples is sufficient for spike removal
        window_size = 5  # 25 minutes for 5-minute data

        # Use center=True to avoid time-shifting the data
        temp_median = (
            df_clean["room_temp"]
            .rolling(window=window_size, center=True, min_periods=1)
            .median()
        )

        # Calculate deviation from median
        temp_deviation = abs(df_clean["room_temp"] - temp_median)

        # Define outlier threshold (e.g., 2°C deviation from local median)
        outlier_threshold = 2.0  # degrees Celsius
        outlier_mask = temp_deviation > outlier_threshold

        # Replace outliers with NaN first, then apply smart interpolation
        df_clean.loc[outlier_mask, "room_temp"] = np.nan

        # Detect stuck sensor periods (constant values for >30 minutes)
        stuck_sensor_mask = self._detect_stuck_sensor(df_clean["room_temp"])
        df_clean.loc[stuck_sensor_mask, "room_temp"] = np.nan

        # Apply smart interpolation: only fill gaps of up to 3 consecutive NaNs (15 minutes)
        # This prevents creating artificial data over long periods
        df_clean["room_temp"] = df_clean["room_temp"].interpolate(
            method="time", limit=3, limit_direction="both"
        )

        # Count and log outliers removed
        outliers_removed = outlier_mask.sum()
        remaining_nans = df_clean["room_temp"].isna().sum()
        if outliers_removed > 0:
            self.logger.info(
                f"Cleaned {outliers_removed} temperature outliers "
                f"({outliers_removed/len(df)*100:.1f}% of data), "
                f"{remaining_nans} gaps too large to interpolate"
            )

            # Log statistics about outliers
            if outliers_removed > 0:
                max_deviation = temp_deviation[outlier_mask].max()
                self.logger.debug(
                    f"Max outlier deviation: {max_deviation:.2f}°C "
                    f"(threshold: {outlier_threshold}°C)"
                )

        # Also clean outdoor temperature if present
        if (
            "outdoor_temp" in df_clean.columns
            and not df_clean["outdoor_temp"].isna().all()
        ):
            outdoor_median = (
                df_clean["outdoor_temp"]
                .rolling(window=window_size, center=True, min_periods=1)
                .median()
            )

            outdoor_deviation = abs(df_clean["outdoor_temp"] - outdoor_median)
            outdoor_outlier_mask = outdoor_deviation > outlier_threshold

            # Replace outliers with NaN first, then apply smart interpolation
            df_clean.loc[outdoor_outlier_mask, "outdoor_temp"] = np.nan

            # Apply smart interpolation for outdoor temperature
            df_clean["outdoor_temp"] = df_clean["outdoor_temp"].interpolate(
                method="time", limit=3, limit_direction="both"
            )

            outdoor_outliers_removed = outdoor_outlier_mask.sum()
            outdoor_remaining_nans = df_clean["outdoor_temp"].isna().sum()
            if outdoor_outliers_removed > 0:
                self.logger.debug(
                    f"Cleaned {outdoor_outliers_removed} outdoor temperature outliers, "
                    f"{outdoor_remaining_nans} gaps too large to interpolate"
                )

        # Apply Savitzky-Golay filter for noise reduction while preserving signal shape
        # This is crucial for accurate derivative calculations (dT/dt)
        df_clean = self._apply_savgol_smoothing(df_clean)

        return df_clean

    def _apply_savgol_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Savitzky-Golay filter for noise reduction while preserving signal shape.

        This is superior to moving averages as it preserves the true shape of signals
        (like heating curve starts) while removing noise - essential for derivative calculations.
        """
        df_smooth = df.copy()

        # Parameters for Savitzky-Golay filter
        window_length = 7  # Must be odd, 5-9 is good for 5-minute data
        polyorder = 2  # Polynomial order 2-3 works well

        # Apply smoothing to room temperature if available and has sufficient data
        if "room_temp" in df_smooth.columns and not df_smooth["room_temp"].isna().all():
            valid_mask = df_smooth["room_temp"].notna()
            valid_count = valid_mask.sum()

            if valid_count >= window_length:
                # Create a temporary series for filtering (Savitzky-Golay needs no NaN values)
                temp_values = df_smooth["room_temp"].copy()

                # Handle NaN values by forward/backward filling for filtering
                temp_filled = temp_values.ffill().bfill()

                if len(temp_filled.dropna()) >= window_length:
                    # Apply Savitzky-Golay filter
                    temp_smooth = savgol_filter(temp_filled, window_length, polyorder)

                    # Restore NaN values where they originally existed
                    df_smooth["room_temp"] = temp_smooth
                    df_smooth.loc[~valid_mask, "room_temp"] = np.nan

                    self.logger.debug(
                        f"Applied Savitzky-Golay smoothing to room temperature "
                        f"(window={window_length}, poly={polyorder})"
                    )
            else:
                self.logger.debug(
                    f"Insufficient data for room temperature smoothing "
                    f"({valid_count} < {window_length} required)"
                )

        # Apply smoothing to outdoor temperature if available
        if (
            "outdoor_temp" in df_smooth.columns
            and not df_smooth["outdoor_temp"].isna().all()
        ):
            valid_mask = df_smooth["outdoor_temp"].notna()
            valid_count = valid_mask.sum()

            if valid_count >= window_length:
                temp_values = df_smooth["outdoor_temp"].copy()
                temp_filled = temp_values.ffill().bfill()

                if len(temp_filled.dropna()) >= window_length:
                    temp_smooth = savgol_filter(temp_filled, window_length, polyorder)
                    df_smooth["outdoor_temp"] = temp_smooth
                    df_smooth.loc[~valid_mask, "outdoor_temp"] = np.nan

                    self.logger.debug(
                        f"Applied Savitzky-Golay smoothing to outdoor temperature"
                    )
            else:
                self.logger.debug(
                    f"Insufficient data for outdoor temperature smoothing "
                    f"({valid_count} < {window_length} required)"
                )

        return df_smooth

    def _detect_stuck_sensor(self, temp_series: pd.Series) -> pd.Series:
        """
        Detect periods where sensor is stuck (constant values for >30 minutes).

        A stuck sensor reports the same value for extended periods, which is
        physically unrealistic for thermal systems and corrupts analysis.
        """
        # Calculate rolling standard deviation to find periods with no variation
        window_size = 6  # 30 minutes for 5-minute data
        rolling_std = temp_series.rolling(
            window=window_size, center=True, min_periods=1
        ).std()

        # Periods with zero standard deviation indicate stuck sensor
        stuck_mask = (rolling_std == 0) & temp_series.notna()

        # Count stuck periods
        stuck_count = stuck_mask.sum()
        if stuck_count > 0:
            stuck_pct = (stuck_count / len(temp_series)) * 100
            self.logger.info(
                f"Detected {stuck_count} stuck sensor readings "
                f"({stuck_pct:.1f}% of data) - marking as invalid"
            )

            # Additional check: look for abnormally long constant periods
            # Group consecutive stuck periods and log the longest ones
            stuck_groups = (stuck_mask != stuck_mask.shift()).cumsum()
            for group_id in stuck_groups[stuck_mask].unique():
                group_size = (stuck_groups == group_id).sum()
                if group_size >= window_size:  # Only log significant stuck periods
                    duration_minutes = group_size * 5  # 5-minute intervals
                    self.logger.debug(
                        f"Found stuck sensor period: {duration_minutes} minutes "
                        f"({group_size} consecutive readings)"
                    )

        return stuck_mask

    def _is_nighttime_cycle(
        self, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> bool:
        """
        Check if a decay cycle occurs primarily during nighttime (10 PM - 6 AM).

        Nighttime cycles are prioritized because they're free of solar gains,
        providing cleaner data for thermal resistance estimation.
        """
        # Calculate what percentage of the cycle occurs during nighttime hours
        total_duration = (end_time - start_time).total_seconds()
        nighttime_duration = 0

        # Check each hour within the cycle
        current_time = start_time
        while current_time < end_time:
            hour = current_time.hour

            # Nighttime is defined as 22:00 (10 PM) to 06:00 (6 AM)
            is_night_hour = hour >= 22 or hour < 6

            # Calculate duration of this hour segment within the cycle
            next_hour = current_time.replace(
                minute=0, second=0, microsecond=0
            ) + pd.Timedelta(hours=1)
            segment_end = min(next_hour, end_time)
            segment_duration = (segment_end - current_time).total_seconds()

            if is_night_hour:
                nighttime_duration += segment_duration

            current_time = segment_end

        # Consider it a nighttime cycle if >70% occurs during night hours
        nighttime_percentage = nighttime_duration / total_duration
        return nighttime_percentage > 0.7

    def _calculate_basic_thermal_stats(
        self, data: pd.DataFrame, room_name: str
    ) -> Dict[str, Any]:
        """Calculate basic thermal statistics."""
        room_temp = data["room_temp"]

        stats = {
            "room_name": room_name,
            "total_records": len(data),
            "mean_temperature": room_temp.mean(),
            "min_temperature": room_temp.min(),
            "max_temperature": room_temp.max(),
            "temperature_range": room_temp.max() - room_temp.min(),
            "temperature_std": room_temp.std(),
        }

        # Heating statistics
        if "heating_on" in data.columns:
            heating_data = data[data["heating_on"] == 1]
            stats.update(
                {
                    "heating_percentage": len(heating_data) / len(data) * 100,
                    "mean_temp_heating_on": (
                        heating_data["room_temp"].mean()
                        if not heating_data.empty
                        else None
                    ),
                    "mean_temp_heating_off": (
                        data[data["heating_on"] == 0]["room_temp"].mean()
                        if (data["heating_on"] == 0).any()
                        else None
                    ),
                }
            )

        # Outdoor temperature relationship with variance check
        if "outdoor_temp" in data.columns:
            if data["room_temp"].std() > 1e-10 and data["outdoor_temp"].std() > 1e-10:
                correlation = data["room_temp"].corr(data["outdoor_temp"])
            else:
                correlation = None
            stats.update(
                {
                    "outdoor_correlation": correlation,
                    "mean_temp_diff": data["temp_diff"].mean(),
                    "min_temp_diff": data["temp_diff"].min(),
                    "max_temp_diff": data["temp_diff"].max(),
                }
            )

        return stats

    def _analyze_heatup_cooldown(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze heat-up and cool-down rates."""
        if "heating_on" not in data.columns:
            return {"warning": "No heating status data available"}

        # Calculate temperature change rate
        data_copy = data.copy()
        data_copy["temp_change_rate"] = (
            data_copy["room_temp"].diff() * 12
        )  # Per hour (5-min intervals)

        # Heat-up analysis (heating on, temperature rising)
        heatup_mask = (data_copy["heating_on"] == 1) & (
            data_copy["temp_change_rate"] > 0
        )
        heatup_data = data_copy[heatup_mask]

        # Cool-down analysis (heating off, temperature falling)
        cooldown_mask = (data_copy["heating_on"] == 0) & (
            data_copy["temp_change_rate"] < 0
        )
        cooldown_data = data_copy[cooldown_mask]

        results = {
            "heatup_rate": {
                "mean_rate": (
                    heatup_data["temp_change_rate"].mean()
                    if not heatup_data.empty
                    else None
                ),
                "max_rate": (
                    heatup_data["temp_change_rate"].max()
                    if not heatup_data.empty
                    else None
                ),
                "std_rate": (
                    heatup_data["temp_change_rate"].std()
                    if not heatup_data.empty
                    else None
                ),
                "samples": len(heatup_data),
            },
            "cooldown_rate": {
                "mean_rate": (
                    abs(cooldown_data["temp_change_rate"].mean())
                    if not cooldown_data.empty
                    else None
                ),
                "max_rate": (
                    abs(cooldown_data["temp_change_rate"].min())
                    if not cooldown_data.empty
                    else None
                ),
                "std_rate": (
                    cooldown_data["temp_change_rate"].std()
                    if not cooldown_data.empty
                    else None
                ),
                "samples": len(cooldown_data),
            },
        }

        # Estimate heating power based on heat-up rate and outdoor temperature
        if not heatup_data.empty and "outdoor_temp" in data.columns:
            # Simple estimation: higher heat-up rate with lower outdoor temp suggests higher power
            temp_diff_heatup = heatup_data["temp_diff"]
            heatup_rate_values = heatup_data["temp_change_rate"]

            if len(temp_diff_heatup) > 10:
                # Linear regression to estimate power relationship
                slope, intercept, r_value, p_value, std_err = linregress(
                    temp_diff_heatup, heatup_rate_values
                )
                results["power_estimation"] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                }

        return results

    def _identify_time_constant(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify thermal time constant using exponential fitting."""
        if "heating_on" not in data.columns or len(data) < 100:
            return {"warning": "Insufficient data for time constant identification"}

        # Find heating events (transitions from off to on)
        heating_changes = data["heating_on"].diff()
        heating_starts = data[heating_changes == 1].index

        time_constants = []

        for start_time in heating_starts[:10]:  # Analyze first 10 events
            # Look for 2-4 hours after heating starts
            end_time = start_time + pd.Timedelta(hours=4)
            event_data = data.loc[start_time:end_time]

            if len(event_data) < 20:  # Need at least 20 data points (100 minutes)
                continue

            # Extract temperature evolution
            temp_evolution = event_data["room_temp"].values
            time_minutes = np.arange(len(temp_evolution)) * 5  # 5-minute intervals

            # Fit exponential model: T(t) = T_final + (T_initial - T_final) * exp(-t/tau)
            try:
                tau = self._fit_exponential_response(time_minutes, temp_evolution)
                if tau > 0 and tau < 10 * 60:  # Between 0 and 10 hours (in minutes)
                    time_constants.append(tau)
            except Exception:
                continue

        if time_constants:
            return {
                "time_constant_minutes": np.mean(time_constants),
                "time_constant_hours": np.mean(time_constants) / 60,
                "time_constant_std": np.std(time_constants),
                "valid_events": len(time_constants),
            }
        else:
            return {"warning": "Could not identify time constant from available data"}

    def _fit_exponential_response(
        self, time: np.ndarray, temperature: np.ndarray
    ) -> float:
        """Fit exponential response to temperature data."""
        if len(time) < 10:
            raise ValueError("Insufficient data points")

        # Define exponential model
        def exp_model(t, T_final, T_initial, tau):
            return T_final + (T_initial - T_final) * np.exp(-t / tau)

        # Initial parameter guess
        T_initial = temperature[0]
        T_final = temperature[-1]
        tau_guess = len(time) * 5 / 3  # Rough guess based on data length

        # Fit the model
        try:
            popt, _ = optimize.curve_fit(
                exp_model,
                time,
                temperature,
                p0=[T_final, T_initial, tau_guess],
                bounds=(
                    [T_initial - 5, T_initial - 5, 10],
                    [T_initial + 5, T_initial + 5, 600],
                ),
            )
            return popt[2]  # Return tau
        except Exception:
            raise ValueError("Exponential fitting failed")

    def _calculate_heat_loss_coefficient(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate heat loss coefficient (UA value)."""
        if "outdoor_temp" not in data.columns or "heating_on" not in data.columns:
            return {"warning": "Insufficient data for heat loss calculation"}

        # Use steady-state periods (heating off, temperature stable)
        stable_mask = data["heating_on"] == 0
        stable_data = data[stable_mask]

        if len(stable_data) < 50:
            return {"warning": "Insufficient stable periods for heat loss calculation"}

        # Calculate heat loss rate during stable periods
        stable_data = stable_data.copy()
        stable_data["temp_change_rate"] = (
            stable_data["room_temp"].diff() * 12
        )  # Per hour

        # Filter for actual cooling periods
        cooling_data = stable_data[
            stable_data["temp_change_rate"] < -0.05
        ]  # At least 0.05°C/hour cooling

        if len(cooling_data) < 20:
            return {"warning": "Insufficient cooling periods for analysis"}

        # Heat loss = UA * (T_indoor - T_outdoor)
        # Cooling rate = -Heat_loss / thermal_mass
        # So: cooling_rate = -UA * temp_diff / thermal_mass

        temp_diff = cooling_data["temp_diff"]
        cooling_rate = -cooling_data["temp_change_rate"]  # Make positive

        # Linear regression to find relationship
        if len(temp_diff) > 10:
            slope, intercept, r_value, p_value, std_err = linregress(
                temp_diff, cooling_rate
            )

            # UA / thermal_mass = slope
            # Assume typical thermal mass for room estimation
            estimated_thermal_mass = 10000  # Wh/°C (rough estimate for typical room)
            ua_estimate = slope * estimated_thermal_mass

            return {
                "ua_coefficient": ua_estimate,  # W/°C
                "base_heat_loss": intercept
                * estimated_thermal_mass,  # Base heat loss in W
                "r_squared": r_value**2,
                "p_value": p_value,
                "cooling_samples": len(cooling_data),
                "thermal_mass_assumed": estimated_thermal_mass,
            }

        return {"warning": "Could not calculate heat loss coefficient"}

    def _analyze_solar_gains(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze solar heat gains."""
        # Add solar gain proxy (hour of day and season)
        data_copy = data.copy()
        data_copy["solar_proxy"] = np.sin(2 * np.pi * data_copy["hour"] / 24) * np.sin(
            2 * np.pi * data_copy.index.dayofyear / 365
        )

        # Look for temperature increases during non-heating periods
        if "heating_on" in data.columns:
            no_heating_data = data_copy[data_copy["heating_on"] == 0]
        else:
            no_heating_data = data_copy

        if len(no_heating_data) < 50:
            return {"warning": "Insufficient non-heating data for solar analysis"}

        # Calculate temperature change during non-heating periods
        no_heating_data = no_heating_data.copy()
        no_heating_data["temp_change"] = no_heating_data["room_temp"].diff()

        # Analyze relationship between solar proxy and temperature change
        solar_warming = no_heating_data[no_heating_data["temp_change"] > 0]

        if len(solar_warming) < 20:
            return {"warning": "Insufficient solar warming periods found"}

        # Correlation analysis with variance check
        if (
            solar_warming["solar_proxy"].std() > 1e-10
            and solar_warming["temp_change"].std() > 1e-10
        ):
            solar_correlation = solar_warming["solar_proxy"].corr(
                solar_warming["temp_change"]
            )
        else:
            solar_correlation = None

        # Peak solar gain estimation
        peak_solar_hours = no_heating_data[
            (no_heating_data["hour"] >= 11) & (no_heating_data["hour"] <= 15)
        ]
        if not peak_solar_hours.empty:
            peak_warming_rate = peak_solar_hours["temp_change"].mean() * 12  # Per hour
        else:
            peak_warming_rate = None

        return {
            "solar_correlation": solar_correlation,
            "peak_warming_rate": peak_warming_rate,
            "solar_warming_events": len(solar_warming),
            "peak_solar_hours_data": len(peak_solar_hours),
        }

    def _fit_rc_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit RC thermal model to the data."""
        if len(data) < 100 or "outdoor_temp" not in data.columns:
            return {"warning": "Insufficient data for RC model fitting"}

        # RC model: C * dT/dt = (T_outdoor - T_indoor)/R + P_heating
        # Where C is thermal capacity, R is thermal resistance, P is heating power

        # Calculate temperature derivative
        data_copy = data.copy()
        dt = 5 / 60  # 5 minutes in hours
        data_copy["dT_dt"] = data_copy["room_temp"].diff() / dt

        # Prepare features
        temp_diff = (
            data_copy["outdoor_temp"] - data_copy["room_temp"]
        )  # Heat flow driving force

        if "heating_on" in data_copy.columns:
            heating_power = data_copy["heating_on"] * 1000  # Assume 1kW heating when on
        else:
            heating_power = pd.Series(0, index=data_copy.index)

        # Remove NaN values
        valid_mask = data_copy["dT_dt"].notna() & temp_diff.notna()
        dT_dt_clean = data_copy.loc[valid_mask, "dT_dt"]
        temp_diff_clean = temp_diff[valid_mask]
        heating_clean = heating_power[valid_mask]

        if len(dT_dt_clean) < 50:
            return {"warning": "Insufficient clean data for RC model"}

        # Multiple linear regression: C * dT_dt = temp_diff/R + P_heating
        # Rearrange: dT_dt = (1/RC) * temp_diff + (1/C) * P_heating

        X = np.column_stack([temp_diff_clean, heating_clean])
        y = dT_dt_clean

        try:
            # Use Ridge regression for better stability with noisy data
            from sklearn.linear_model import Ridge

            reg = Ridge(alpha=1.0).fit(X, y)

            # Extract parameters
            coeff_temp = reg.coef_[0]  # 1/(R*C)
            coeff_heating = reg.coef_[1]  # 1/C

            # Physical constraint checks
            if coeff_heating > 0 and coeff_temp > 0:
                C = 1 / coeff_heating  # Thermal capacity in Wh/°C
                R = 1 / (coeff_temp * C)  # Thermal resistance in °C/W

                # Physical sanity checks
                if C <= 0 or R <= 0:
                    return {
                        "warning": f"Invalid parameters: C={C:.3f}, R={R:.3f} (must be > 0)"
                    }

                if R > 10.0:  # Very high resistance
                    self.logger.warning(
                        f"Unusually high thermal resistance: {R:.3f} °C/W"
                    )
                elif R < 0.0001:  # Very low resistance
                    self.logger.warning(
                        f"Unusually low thermal resistance: {R:.6f} °C/W"
                    )

                if C > 10000:  # Very high capacitance
                    self.logger.warning(
                        f"Unusually high thermal capacitance: {C:.1f} Wh/°C"
                    )
                elif C < 10:  # Very low capacitance
                    self.logger.warning(
                        f"Unusually low thermal capacitance: {C:.3f} Wh/°C"
                    )

                # Model quality
                y_pred = reg.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                return {
                    "thermal_capacity": C,  # Wh/°C
                    "thermal_resistance": R,  # °C/W
                    "time_constant": R * C / 3600,  # hours
                    "r_squared": r2,
                    "rmse": rmse,
                    "model_intercept": reg.intercept_,
                    "physically_valid": True,
                }
            else:
                return {
                    "warning": f"Invalid coefficients: temp={coeff_temp:.6f}, heating={coeff_heating:.6f} (must be > 0)"
                }

        except Exception as e:
            return {"warning": f"RC model fitting failed: {str(e)}"}

    def _estimate_rc_parameters(
        self, df: pd.DataFrame, p_heat_w: float
    ) -> Dict[str, Any]:
        """
        Selects the best RC parameters from multiple estimation methods.
        This function orchestrates the different estimation strategies and chooses the most
        physically plausible and confident result.
        """
        self.logger.info("Starting enhanced RC parameter estimation for relay system")

        # Ensure required columns are present
        if "outdoor_temp" not in df.columns or "heating_on" not in df.columns:
            self.logger.warning(
                "Missing 'outdoor_temp' or 'heating_on' columns. Skipping RC estimation."
            )
            return self._get_default_rc_params()

        estimation_methods = {
            "decoupled": self._estimate_rc_decoupled,
            "state_space": self._estimate_rc_state_space,
        }

        results = {}
        for name, method_func in estimation_methods.items():
            try:
                params = method_func(df, p_heat_w)
                if params and params.get("physically_valid", False):
                    results[name] = params
            except Exception as e:
                self.logger.error(
                    f"Estimation method '{name}' failed: {e}", exc_info=True
                )

        if not results:
            self.logger.warning(
                "All RC parameter estimation methods failed to produce a valid result."
            )
            return self._get_default_rc_params()

        # Select the best result based on confidence score
        best_method = max(results, key=lambda k: results[k]["confidence"])
        final_params = results[best_method]

        self.logger.info(
            f"RC parameter estimation completed. Best method: '{best_method}'"
        )
        self.logger.info(
            f"Final Parameters: R={final_params['R']:.4f} K/W, "
            f"C={final_params['C']/1e6:.2f} MJ/K, τ={final_params['time_constant']:.1f}h"
        )

        return final_params

    def _estimate_rc_decoupled(
        self, df: pd.DataFrame, p_heat_w: float
    ) -> Optional[Dict[str, Any]]:
        """
        Estimates R and C using heating cycle analysis for robust thermal parameter estimation.
        This method analyzes individual heating cycles as controlled experiments.
        """
        # Try new heating cycle analysis first
        cycles = self._detect_heating_cycles(df)

        if len(cycles) < 3:
            self.logger.info(
                f"Only {len(cycles)} heating cycles found, falling back to simplified estimation"
            )
            return self._estimate_rc_simplified(df, p_heat_w)

        self.logger.info(f"Analyzing {len(cycles)} heating cycles for RC estimation")

        # Analyze each cycle
        decay_results = []
        rise_results = []

        for i, cycle in enumerate(cycles):
            # Find next heating cycle start time (if any)
            next_cycle_start = None
            if i + 1 < len(cycles):
                next_cycle_start = cycles[i + 1]["start_time"]

            # Analyze cooling decay with next cycle information
            decay = self._analyze_heating_cycle_decay(df, cycle, next_cycle_start)
            if decay["fit_valid"]:
                decay_results.append(decay)
                self.logger.debug(
                    f"Cycle {i+1} decay: τ={decay['time_constant_hours']:.1f}h, R²={decay['r_squared']:.3f}, "
                    f"duration={decay['decay_duration_hours']:.1f}h, end_reason={decay['decay_end_reason']}"
                )
            else:
                # Enhanced debugging for failed decay fits
                reason = decay.get("reason", "unknown")
                self.logger.debug(f"Cycle {i+1} decay analysis failed: {reason}")
                if i < 3:  # Show details for first 3 failures
                    extra_info = []
                    if "data_points" in decay:
                        extra_info.append(f"data_points={decay['data_points']}")
                    if "heating_intervals" in decay:
                        extra_info.append(
                            f"heating_resumed={decay['heating_intervals']}"
                        )
                    if "decay_duration_hours" in decay:
                        extra_info.append(
                            f"duration={decay['decay_duration_hours']:.1f}h"
                        )
                    if "decay_end_reason" in decay:
                        extra_info.append(f"end_reason={decay['decay_end_reason']}")
                    if extra_info:
                        self.logger.debug(f"  Details: {', '.join(extra_info)}")

                    # Show cycle timing info
                    if next_cycle_start:
                        gap_hours = (
                            next_cycle_start - cycle["end_time"]
                        ).total_seconds() / 3600
                        self.logger.debug(f"  Gap to next cycle: {gap_hours:.1f}h")
                    else:
                        self.logger.debug(f"  No next cycle (last cycle)")

                    self.logger.debug(
                        f"  Cycle: {cycle['end_time']} (end), temp rise: {cycle.get('peak_temp', 0) - cycle.get('start_temp', 0):.1f}°C"
                    )

            # Analyze heating rise
            rise = self._analyze_heating_cycle_rise(df, cycle)
            if rise["fit_valid"]:
                rise_results.append(rise)
                self.logger.debug(
                    f"Cycle {i+1} rise: C={rise['thermal_capacitance_j_per_k']/1e6:.1f}MJ/K, R²={rise['fit_r_squared']:.3f}"
                )
            else:
                self.logger.debug(
                    f"Cycle {i+1} rise analysis failed: {rise.get('reason', 'unknown')}"
                )

        # Check if we have enough valid results
        if len(decay_results) == 0 and len(rise_results) == 0:
            self.logger.warning(
                "No valid heating cycle analyses, falling back to simplified estimation"
            )
            return self._estimate_rc_simplified(df, p_heat_w)

        # Calculate robust statistics from successful analyses
        tau_values = []
        C_values = []

        if len(decay_results) > 0:
            tau_values = [r["time_constant_hours"] for r in decay_results]
            tau_median = np.median(tau_values)
            tau_std = np.std(tau_values)
        else:
            tau_median = None
            tau_std = 0

        if len(rise_results) > 0:
            C_values = [r["thermal_capacitance_j_per_k"] for r in rise_results]
            C_median = np.median(C_values)
            C_std = np.std(C_values)
        else:
            C_median = None
            C_std = 0

        # If we have both τ and C, calculate R
        if tau_median is not None and C_median is not None:
            R_calculated = (tau_median * 3600) / C_median  # Convert hours to seconds
            method_used = "heating_cycle_analysis"
        elif tau_median is not None:
            # Only have τ, estimate C from typical values
            C_median = 30e6  # 30 MJ/K in J/K
            R_calculated = (tau_median * 3600) / C_median
            method_used = "heating_cycle_decay_only"
            self.logger.info(
                "Using typical thermal capacitance with measured time constant"
            )
        elif C_median is not None:
            # Only have C, estimate τ from typical values
            tau_median = 12.0  # 12 hour typical time constant
            R_calculated = (tau_median * 3600) / C_median
            method_used = "heating_cycle_rise_only"
            self.logger.info(
                "Using typical time constant with measured thermal capacitance"
            )
        else:
            # Shouldn't reach here given the check above, but safety fallback
            return self._estimate_rc_simplified(df, p_heat_w)

        # Quality assessment with improved thresholds
        # Reduced from 10 cycles to 6 cycles for good confidence
        # Added R² quality bonus for better fits
        cycle_confidence = min(1.0, len(decay_results) / 6)

        # Enhanced quality assessment based on data filtering improvements
        if decay_results:
            # Base R² quality from successful fits
            avg_r_squared = np.mean([d.get("fit_r_squared", 0) for d in decay_results])
            r_squared_bonus = min(0.2, avg_r_squared * 0.3)  # Up to 20% bonus

            # Nighttime cycle bonus - prioritize cycles free of solar contamination
            nighttime_cycles = sum(
                1 for d in decay_results if d.get("is_nighttime", False)
            )
            nighttime_ratio = nighttime_cycles / len(decay_results)
            nighttime_bonus = min(0.15, nighttime_ratio * 0.15)  # Up to 15% bonus

            # Data quality bonus - reward cycles that passed advanced filtering
            # Count cycles that would have passed strict filtering (no stuck sensors, stable outdoor temp, monotonic)
            high_quality_cycles = 0
            for d in decay_results:
                # These cycles already passed all filters, so they're high quality
                if d.get("fit_valid", False):
                    high_quality_cycles += 1

            quality_ratio = high_quality_cycles / len(decay_results)
            quality_bonus = min(
                0.1, quality_ratio * 0.1
            )  # Up to 10% bonus for clean data

            self.logger.debug(
                f"Confidence factors: base={cycle_confidence:.2f}, R²={r_squared_bonus:.2f}, "
                f"nighttime={nighttime_bonus:.2f} ({nighttime_cycles}/{len(decay_results)} cycles), "
                f"quality={quality_bonus:.2f} ({high_quality_cycles}/{len(decay_results)} clean)"
            )
        else:
            r_squared_bonus = nighttime_bonus = quality_bonus = 0

        confidence = min(
            1.0, cycle_confidence + r_squared_bonus + nighttime_bonus + quality_bonus
        )

        # Physical validity checks
        physically_valid = (
            R_MIN < R_calculated < R_MAX
            and C_MIN_MJ * 1e6 < C_median < C_MAX_MJ * 1e6
            and TAU_MIN < tau_median < TAU_MAX
        )

        # Log detailed statistics
        self.logger.info("Heating cycle analysis complete:")
        self.logger.info(
            f"  Valid decay fits: {len(decay_results)}/{len(cycles)} ({len(decay_results)/len(cycles)*100:.1f}%)"
        )
        self.logger.info(
            f"  Valid rise fits: {len(rise_results)}/{len(cycles)} ({len(rise_results)/len(cycles)*100:.1f}%)"
        )

        if tau_values:
            self.logger.info(
                f"  Time constant: median={tau_median:.1f}h, std={tau_std:.1f}h, range={min(tau_values):.1f}-{max(tau_values):.1f}h"
            )
        if C_values:
            self.logger.info(
                f"  Thermal capacitance: median={C_median/1e6:.1f}MJ/K, std={C_std/1e6:.1f}MJ/K"
            )

        # Quality warnings
        if tau_median is not None and tau_std > tau_median * 0.5:
            self.logger.warning(
                f"High variability in time constants (std={tau_std:.1f}h, median={tau_median:.1f}h)"
            )

        if len(decay_results) < 5:
            self.logger.warning(
                f"Limited heating cycle data ({len(decay_results)} valid cycles) - results may be unreliable"
            )

        if not physically_valid:
            self.logger.warning(
                f"Heating cycle analysis produced unphysical results: R={R_calculated:.4f}, C={C_median/1e6:.2f}, τ={tau_median:.1f}h"
            )
            # Clamp to physical bounds
            R_calculated = np.clip(R_calculated, R_MIN, R_MAX)
            C_median = np.clip(C_median, C_MIN_MJ * 1e6, C_MAX_MJ * 1e6)
            tau_median = np.clip(tau_median, TAU_MIN, TAU_MAX)
            physically_valid = True
            confidence *= 0.8  # Reduced penalty: was 0.5, now 0.8

        return {
            "method": method_used,
            "confidence": confidence,
            "R": R_calculated,
            "C": C_median,
            "time_constant": tau_median,
            "physically_valid": physically_valid,
            "cycles_analyzed": len(cycles),
            "successful_decays": len(decay_results),
            "successful_rises": len(rise_results),
            "tau_std_dev": tau_std,
            "C_std_dev": C_std,
        }

    def _detect_heating_cycles(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect heating cycles from room temperature and heating state data.

        Args:
            df: DataFrame with 'heating_on', 'room_temp', and 'outdoor_temp' columns

        Returns:
            List of heating cycle dictionaries with cycle information
        """
        cycles = []

        if "heating_on" not in df.columns:
            self.logger.error("'heating_on' column not found in DataFrame")
            return cycles

        # Find heating state changes
        heating_diff = df["heating_on"].diff()
        start_indices = df.index[heating_diff == 1]  # Heating starts
        end_indices = df.index[heating_diff == -1]  # Heating ends

        self.logger.info(
            f"Found {len(start_indices)} heating starts and {len(end_indices)} heating ends"
        )

        # Debug: Log relay timing for troubleshooting
        if len(start_indices) > 0:
            self.logger.debug(f"First few heating starts: {start_indices[:5].tolist()}")
        if len(end_indices) > 0:
            self.logger.debug(f"First few heating ends: {end_indices[:5].tolist()}")

        # Handle edge cases: dataset starts/ends during heating
        if len(df) > 0 and df["heating_on"].iloc[0] == 1:
            # Dataset starts with heating on
            start_indices = start_indices.insert(0, df.index[0])

        if (
            len(df) > 0
            and df["heating_on"].iloc[-1] == 1
            and len(start_indices) > len(end_indices)
        ):
            # Dataset ends with heating on
            end_indices = end_indices.insert(len(end_indices), df.index[-1])

        # Pair start and end events
        for i in range(min(len(start_indices), len(end_indices))):
            start_time = start_indices[i]
            end_time = end_indices[i]

            # Skip if end comes before start (data issue)
            if end_time <= start_time:
                continue

            duration_minutes = (end_time - start_time).total_seconds() / 60.0

            # Filter by duration - relaxed minimum for winter short cycles
            if (
                duration_minutes < 5 or duration_minutes > 2880
            ):  # 5min to 48 hours (relaxed from 10min for winter conditions)
                self.logger.debug(
                    f"Cycle {start_time} to {end_time} rejected: duration {duration_minutes:.1f}min out of range"
                )
                continue

            # Get temperature data for this cycle
            cycle_data = df.loc[start_time:end_time]
            if len(cycle_data) < 2:
                continue

            # Filter out NaN temperature values for this cycle
            # Check both 'room_temp' and 'temperature' column names for compatibility
            temp_col = (
                "room_temp" if "room_temp" in cycle_data.columns else "temperature"
            )
            if temp_col not in cycle_data.columns:
                self.logger.debug(
                    f"Cycle {start_time} to {end_time} rejected: no temperature column found"
                )
                continue

            valid_temps = cycle_data[temp_col].dropna()
            if len(valid_temps) < 2:
                continue  # Need at least 2 valid temperature readings

            start_temp = valid_temps.iloc[0]
            peak_temp = valid_temps.max()

            # Filter by temperature rise - relaxed for winter conditions
            temp_rise = peak_temp - start_temp
            if (
                temp_rise < 0.1
            ):  # Relaxed from 0.5°C - even small rises are valuable in winter
                self.logger.debug(
                    f"Cycle {start_time} to {end_time} rejected: temp rise {temp_rise:.2f}°C too small"
                )
                continue

            # Check for valid outdoor temperature data (optional)
            outdoor_temp_avg = None
            if "outdoor_temp" in cycle_data.columns:
                outdoor_temps = cycle_data["outdoor_temp"].dropna()
                if len(outdoor_temps) > 0:
                    outdoor_temp_avg = outdoor_temps.mean()

            # Get power rating from room config
            power_w = self._get_room_power_rating_watts()

            cycle = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": duration_minutes,
                "start_temp": start_temp,
                "peak_temp": peak_temp,
                "outdoor_temp_avg": outdoor_temp_avg,
                "power_w": power_w,
            }
            cycles.append(cycle)

        self.logger.info(f"Detected {len(cycles)} valid heating cycles")
        return cycles

    def _analyze_heating_cycle_decay(
        self, df: pd.DataFrame, cycle: Dict, next_cycle_start: pd.Timestamp = None
    ) -> Dict:
        """
        Analyze the cooling decay phase after a heating cycle.

        Args:
            df: Full temperature DataFrame
            cycle: Heating cycle information from _detect_heating_cycles()
            next_cycle_start: Start time of the next heating cycle (if any)

        Returns:
            Dictionary with decay analysis results
        """
        # Find actual peak temperature time (accounting for thermal lag after relay OFF)
        relay_off_time = cycle["end_time"]
        peak_temp = cycle["peak_temp"]

        # Look for peak temperature time after relay stops (thermal lag effect)
        # Search in a window after relay stops to find when peak actually occurs
        search_window_hours = 2.0  # Allow up to 2 hours for thermal lag
        search_end = relay_off_time + pd.Timedelta(hours=search_window_hours)

        # Get data from relay OFF until search window end
        if len(df) > 0 and search_end <= df.index[-1]:
            post_relay_data = df.loc[relay_off_time:search_end]
        else:
            post_relay_data = df.loc[relay_off_time:]

        # Check temperature column name for compatibility
        temp_col = "room_temp" if "room_temp" in df.columns else "temperature"

        if not post_relay_data.empty and temp_col in post_relay_data.columns:
            # Find when the actual peak temperature occurs (could be after relay OFF)
            valid_temps = post_relay_data[temp_col].dropna()
            if len(valid_temps) > 0:
                actual_peak_temp = valid_temps.max()
                # If temperature continued rising after relay OFF, use that peak time
                if actual_peak_temp >= peak_temp:
                    peak_temp_idx = valid_temps.idxmax()
                    decay_start = peak_temp_idx
                    self.logger.debug(
                        f"Peak temperature {actual_peak_temp:.1f}°C found at {peak_temp_idx} (thermal lag after relay OFF at {relay_off_time})"
                    )
                else:
                    # Peak was during heating period, use relay OFF time
                    decay_start = relay_off_time
                    self.logger.debug(
                        f"Peak temperature {peak_temp:.1f}°C occurred during heating, starting decay from relay OFF at {relay_off_time}"
                    )
            else:
                decay_start = relay_off_time
        else:
            decay_start = relay_off_time

        # Get baseline temperature for analysis
        baseline_temp = cycle["start_temp"]
        baseline_tolerance = 0.3  # °C

        # Find end of decay period - use next heating cycle start if available
        if next_cycle_start is not None and next_cycle_start > decay_start:
            # Use the time until next heating cycle starts
            decay_end = next_cycle_start
            decay_end_reason = "next_heating_cycle"
        else:
            # Fallback: Find natural end point or use maximum period
            max_decay_hours = (
                8  # Extended for winter conditions to capture longer natural decay
            )

            decay_end = decay_start + pd.Timedelta(hours=max_decay_hours)
            if decay_end > df.index[-1]:
                decay_end = df.index[-1]
                decay_end_reason = "data_end"
            else:
                decay_end_reason = "max_time"

            # Check for natural return to baseline
            potential_decay_data = df.loc[decay_start:decay_end]
            baseline_reached = potential_decay_data[
                abs(potential_decay_data[temp_col] - baseline_temp)
                <= baseline_tolerance
            ]
            if len(baseline_reached) > 0:
                decay_end = baseline_reached.index[0]
                decay_end_reason = "baseline_reached"

        # Get decay period data and ensure heating is actually off
        decay_data = df.loc[decay_start:decay_end].copy()

        # Remove any data points where heating is still on (edge case handling)
        heating_off_mask = decay_data["heating_on"] == 0
        decay_data = decay_data[heating_off_mask]

        # Check if we have any data left after filtering
        if len(decay_data) == 0:
            return {
                "fit_valid": False,
                "reason": "no_decay_data",
                "note": "All data points during decay period have heating on",
            }

        # Adaptive minimum data points based on decay duration
        decay_duration_hours = (
            decay_data.index[-1] - decay_data.index[0]
        ).total_seconds() / 3600
        # For short decays, require fewer points (minimum 3 for curve fitting)
        min_data_points = 3 if decay_duration_hours < 1.0 else 5

        if len(decay_data) < min_data_points:  # Need minimum data points
            return {
                "fit_valid": False,
                "reason": "insufficient_data",
                "data_points": len(decay_data),
                "required_points": min_data_points,
                "decay_duration_hours": decay_duration_hours,
            }

        # Verify no heating during decay period (should be zero now)
        heating_during_decay = decay_data["heating_on"].sum()
        if heating_during_decay > 0:
            # This should not happen now that we filter out heating_on points
            self.logger.warning(
                f"Unexpected heating during decay: {heating_during_decay} intervals"
            )
            return {
                "fit_valid": False,
                "reason": "heating_resumed",
                "heating_intervals": heating_during_decay,
            }

        # Check outdoor temperature stability during decay period
        # Unstable outdoor temperature invalidates the simple exponential decay model
        if "outdoor_temp" in decay_data.columns:
            outdoor_temp_std = decay_data["outdoor_temp"].std()
            outdoor_stability_threshold = 0.5  # °C maximum standard deviation

            if outdoor_temp_std > outdoor_stability_threshold:
                return {
                    "fit_valid": False,
                    "reason": "unstable_outdoor_temp",
                    "outdoor_temp_std": outdoor_temp_std,
                    "threshold": outdoor_stability_threshold,
                    "decay_duration_hours": decay_duration_hours,
                }

        # Prepare data for exponential decay fitting
        # Use temperature difference: ΔT(t) = T_room(t) - T_outdoor(t)
        decay_data["temp_diff"] = decay_data[temp_col] - decay_data["outdoor_temp"]
        decay_data = decay_data.dropna(subset=["temp_diff"])

        if len(decay_data) < 5:
            return {"fit_valid": False, "reason": "insufficient_valid_data"}

        # Time array in hours from decay start
        time_hours = (decay_data.index - decay_start).total_seconds() / 3600.0
        temp_diff = decay_data["temp_diff"].values

        # Initial conditions
        initial_temp_diff = temp_diff[0]
        outdoor_temp_avg = decay_data["outdoor_temp"].mean()

        # Check for minimum decay magnitude (signal-to-noise ratio)
        final_temp_diff = temp_diff[-1]
        decay_magnitude = initial_temp_diff - final_temp_diff

        # Adaptive threshold based on outdoor temperature and cycle characteristics
        outdoor_temp_avg = decay_data["outdoor_temp"].mean()
        cycle_duration_hours = cycle.get("duration_minutes", 60) / 60.0

        # More relaxed thresholds for short cycles and winter conditions
        if outdoor_temp_avg < 5.0:  # Winter conditions (< 5°C)
            if cycle_duration_hours < 0.5:  # Very short cycles (< 30 min)
                min_decay_magnitude = 0.2  # °C - very relaxed for short winter cycles
            else:
                min_decay_magnitude = 0.3  # °C - relaxed for winter
        else:
            if cycle_duration_hours < 0.5:  # Very short cycles
                min_decay_magnitude = 0.3  # °C - relaxed for short cycles
            else:
                min_decay_magnitude = 0.5  # °C - moderate for other seasons

        if decay_magnitude < min_decay_magnitude:
            self.logger.debug(
                f"Decay magnitude {decay_magnitude:.2f}°C < {min_decay_magnitude:.2f}°C "
                f"(cycle duration: {cycle_duration_hours:.1f}h, outdoor: {outdoor_temp_avg:.1f}°C)"
            )
            return {
                "fit_valid": False,
                "reason": "insufficient_decay_magnitude",
                "decay_magnitude": decay_magnitude,
                "min_required": min_decay_magnitude,
                "outdoor_temp_avg": outdoor_temp_avg,
                "cycle_duration_hours": cycle_duration_hours,
            }

        # Check monotonicity of decay curve
        # A clean decay should be mostly monotonically decreasing
        temp_diff_changes = np.diff(temp_diff)
        increasing_points = (temp_diff_changes > 0).sum()
        total_changes = len(temp_diff_changes)

        if total_changes > 0:
            increasing_percentage = increasing_points / total_changes
            monotonicity_threshold = 0.15  # Allow up to 15% of points to increase

            if increasing_percentage > monotonicity_threshold:
                return {
                    "fit_valid": False,
                    "reason": "non_monotonic_decay",
                    "increasing_percentage": increasing_percentage * 100,
                    "threshold_percentage": monotonicity_threshold * 100,
                    "increasing_points": increasing_points,
                    "total_points": total_changes,
                }

        try:
            # Exponential decay model: ΔT(t) = ΔT_initial * exp(-t/τ)
            def decay_model(t, tau):
                return initial_temp_diff * np.exp(-t / tau)

            # Fit with bounds for time constant - extended iterations for better convergence
            bounds = ([TAU_MIN], [TAU_MAX])
            popt, _ = curve_fit(
                decay_model, time_hours, temp_diff, bounds=bounds, maxfev=2000
            )

            tau_fitted = popt[0]

            # Calculate R² for fit quality
            y_pred = decay_model(time_hours, tau_fitted)
            r_squared = r2_score(temp_diff, y_pred)

            # Validate fit quality - adaptive threshold based on conditions
            # Winter conditions typically have more noise due to frequent cycling
            outdoor_temp_avg = decay_data["outdoor_temp"].mean()
            if outdoor_temp_avg < 5.0:  # Winter conditions
                r_squared_threshold = 0.2  # Very relaxed for winter
            elif outdoor_temp_avg < 15.0:  # Shoulder seasons
                r_squared_threshold = 0.3  # Moderate
            else:  # Summer conditions
                r_squared_threshold = 0.4  # Standard

            fit_valid = r_squared > r_squared_threshold

            if fit_valid:
                self.logger.debug(
                    f"DECAY FIT SUCCESS: τ={tau_fitted:.1f}h, R²={r_squared:.3f}, magnitude={decay_magnitude:.1f}°C, threshold={r_squared_threshold:.1f} (outdoor={outdoor_temp_avg:.1f}°C)"
                )
            else:
                self.logger.debug(
                    f"DECAY FIT REJECTED: τ={tau_fitted:.1f}h, R²={r_squared:.3f} < {r_squared_threshold:.1f}, magnitude={decay_magnitude:.1f}°C (outdoor={outdoor_temp_avg:.1f}°C)"
                )

            decay_duration_hours = (decay_end - decay_start).total_seconds() / 3600.0

            # Check if this is a nighttime decay cycle (10 PM to 6 AM)
            # Nighttime cycles are prioritized as they're free of solar contamination
            is_nighttime = self._is_nighttime_cycle(decay_start, decay_end)

            # Apply nighttime quality bonus to R² for prioritization
            quality_score = r_squared
            if is_nighttime:
                quality_score += 0.1  # 10% bonus for nighttime cycles
                quality_score = min(1.0, quality_score)  # Cap at 1.0

            return {
                "time_constant_hours": tau_fitted,
                "r_squared": r_squared,
                "fit_r_squared": quality_score,  # Enhanced score for cycle prioritization
                "decay_start_temp": cycle["peak_temp"],
                "baseline_temp": baseline_temp,
                "outdoor_temp_avg": outdoor_temp_avg,
                "data_points": len(decay_data),
                "decay_duration_hours": decay_duration_hours,
                "decay_end_reason": decay_end_reason,
                "decay_magnitude": decay_magnitude,
                "fit_valid": fit_valid,
                "is_nighttime": is_nighttime,
            }

        except Exception as e:
            self.logger.debug(f"Decay fitting failed: {e}")
            return {"fit_valid": False, "reason": "fitting_failed"}

    def _analyze_heating_cycle_rise(self, df: pd.DataFrame, cycle: Dict) -> Dict:
        """
        Analyze the heating rise phase of a heating cycle.

        Args:
            df: Full temperature DataFrame
            cycle: Heating cycle information from _detect_heating_cycles()

        Returns:
            Dictionary with rise analysis results
        """
        # Extract heating period data
        heating_data = df.loc[cycle["start_time"] : cycle["end_time"]].copy()

        # Remove first 2 minutes (system lag)
        lag_time = pd.Timedelta(minutes=2)
        analysis_start = cycle["start_time"] + lag_time
        if analysis_start >= cycle["end_time"]:
            return {"fit_valid": False, "reason": "cycle_too_short"}

        steady_heating = heating_data.loc[analysis_start:]

        if len(steady_heating) < 3:
            return {"fit_valid": False, "reason": "insufficient_steady_data"}

        # Calculate heating rate using linear regression on first 5-10 minutes
        # This minimizes heat loss effects during initial heating
        initial_period_minutes = min(
            10, len(steady_heating) * 0.5
        )  # First 10 min or half of data
        initial_end = analysis_start + pd.Timedelta(minutes=initial_period_minutes)

        initial_data = steady_heating.loc[:initial_end]
        if len(initial_data) < 3:
            initial_data = steady_heating  # Use all data if too short

        # Time array in seconds from heating start
        time_seconds = (initial_data.index - analysis_start).total_seconds()
        temp_values = initial_data["room_temp"].values

        try:
            # Linear regression for heating rate
            slope, _, r_value, _, _ = linregress(time_seconds, temp_values)

            heating_rate_k_per_s = slope  # K/s
            r_squared = r_value**2

            # Calculate thermal capacitance: C = P_heat / (dT/dt)
            power_w = cycle["power_w"]
            if heating_rate_k_per_s > 0:
                thermal_capacitance_j_per_k = power_w / heating_rate_k_per_s
            else:
                return {"fit_valid": False, "reason": "negative_heating_rate"}

            # Validate fit quality and physical plausibility
            fit_valid = (
                r_squared > 0.7
                and heating_rate_k_per_s > 0
                and C_MIN_MJ * 1e6 < thermal_capacitance_j_per_k < C_MAX_MJ * 1e6
            )

            return {
                "thermal_capacitance_j_per_k": thermal_capacitance_j_per_k,
                "heating_rate_k_per_s": heating_rate_k_per_s,
                "corrected_heating_rate_k_per_s": heating_rate_k_per_s,  # No correction applied for simplicity
                "heat_loss_correction_applied": False,
                "fit_r_squared": r_squared,
                "fit_valid": fit_valid,
            }

        except Exception as e:
            self.logger.debug(f"Rise analysis failed: {e}")
            return {"fit_valid": False, "reason": "fitting_failed"}

    def _get_room_power_rating_watts(self) -> float:
        """Get room power rating in watts from configuration."""
        if hasattr(self, "_current_room_name") and self._current_room_name:
            power_kw = self.room_power_ratings_kw.get(self._current_room_name, 1.0)
            return power_kw * 1000.0  # Convert kW to W
        return 1000.0  # 1kW default fallback

    def _estimate_rc_simplified(
        self, df: pd.DataFrame, p_heat_w: float
    ) -> Optional[Dict[str, Any]]:
        """
        Simplified RC estimation for summer data with minimal thermal dynamics.
        Uses typical building physics values scaled by room characteristics.
        """
        self.logger.info("Using simplified RC estimation for summer data")

        # Analyze temperature variation to estimate building quality
        temp_std = df["room_temp"].std()

        # Estimate thermal properties based on temperature stability
        if temp_std < 1.0:
            # Very stable temperature -> well-insulated building
            R_estimate = 0.15  # K/W - good insulation
            C_estimate_mj = 40.0  # MJ/K - higher thermal mass
        elif temp_std < 2.0:
            # Moderate stability -> average building
            R_estimate = 0.10  # K/W - average insulation
            C_estimate_mj = 30.0  # MJ/K - typical thermal mass
        else:
            # High variation -> poorly insulated
            R_estimate = 0.05  # K/W - poor insulation
            C_estimate_mj = 20.0  # MJ/K - lower thermal mass

        tau_estimate = (R_estimate * C_estimate_mj * 1e6) / 3600  # hours

        # Ensure values are within physical bounds
        R_estimate = np.clip(R_estimate, R_MIN, R_MAX)
        C_estimate_mj = np.clip(C_estimate_mj, C_MIN_MJ, C_MAX_MJ)
        tau_estimate = np.clip(tau_estimate, TAU_MIN, TAU_MAX)

        self.logger.info(
            f"Simplified estimation: R={R_estimate:.3f} K/W, C={C_estimate_mj:.1f} MJ/K, τ={tau_estimate:.1f}h"
        )

        return {
            "method": "simplified",
            "confidence": 0.6,  # Moderate confidence for physics-based estimates
            "R": R_estimate,
            "C": C_estimate_mj * 1e6,  # Convert to J/K
            "time_constant": tau_estimate,
            "physically_valid": True,
            "note": "Estimated from building thermal characteristics (summer data)",
        }

    def _estimate_rc_state_space(
        self, df: pd.DataFrame, power_rating_w: float
    ) -> Optional[Dict[str, Any]]:
        """
        Estimates R and C using a constrained state-space model (ARMA).
        Currently disabled due to numerical instability issues.
        """
        self.logger.warning(
            "State-space estimation disabled due to convergence issues. "
            "Use decoupled method instead."
        )
        return None

        # TODO: Fix state-space estimation method
        # Issues to resolve:
        # 1. Maximum Likelihood convergence failures
        # 2. Negative resistance values from poor ARIMA fits
        # 3. Insufficient data validation (need 200+ points)
        # 4. Incorrect physical parameter conversion
        # 5. No convergence validation

        # Original code disabled - needs complete rewrite
        # df_resampled = df.resample("15min").mean().interpolate(method="linear").dropna()
        # if len(df_resampled) < 200:  # Increased minimum data requirement
        #     self.logger.warning("Not enough data for state-space modeling (need 200+ points).")
        #     return None

    def _get_default_rc_params(self) -> Dict[str, Any]:
        """Returns a default, invalid parameter set when estimation fails."""
        return {
            "method": "none",
            "confidence": 0,
            "R": np.nan,
            "C": np.nan,
            "time_constant": np.nan,
            "physically_valid": False,
        }

    def estimate_rc_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced RC parameter estimation specifically for relay-based heating systems.

        This method implements multiple approaches to estimate thermal resistance (R) and
        thermal capacitance (C) parameters for binary ON/OFF relay control systems typical
        in residential heating applications.

        Args:
            data: DataFrame with room temperature, heating state, and optionally outdoor temperature

        Returns:
            Dict containing RC parameters estimated using different methods with confidence metrics
        """
        if "heating_on" not in data.columns or len(data) < 200:
            return {
                "warning": "Insufficient data for RC parameter estimation (need heating_on column and >200 points)"
            }

        self.logger.info("Starting enhanced RC parameter estimation for relay system")
        results = {}

        # Ensure we have the expected column names for RC analysis
        # The data should already be prepared by _merge_room_weather_data
        expected_cols = ["room_temp", "heating_on"]
        missing_cols = [col for col in expected_cols if col not in data.columns]

        if missing_cols:
            return {
                "warning": f"Missing required columns for RC estimation: {missing_cols}"
            }

        # Step 1: Analyze cooldown periods to get the cooling factor (1/RC)
        cooldown_results = self._analyze_cooldown_periods(data)
        if "cooling_rate_factor" in cooldown_results:
            results["cooldown_analysis"] = cooldown_results
            self.logger.info(
                f"Cooldown analysis complete: 1/(RC) = {cooldown_results['cooling_rate_factor']:.4f} 1/h"
            )

        # Step 2: Analyze heatup periods to get thermal capacitance (C)
        heatup_results = self._analyze_heatup_periods(data)
        if (
            "thermal_capacitance_j_per_k" in heatup_results
            and heatup_results["thermal_capacitance_j_per_k"] is not None
        ):
            results["heatup_analysis"] = heatup_results
            self.logger.info(
                f"Heatup analysis complete: C = {heatup_results['thermal_capacitance_j_per_k']/1e6:.2f} MJ/K"
            )

        # Step 3: Solve for R and C if both analyses were successful
        if "cooldown_analysis" in results and "heatup_analysis" in results:
            solved_params = self._solve_for_rc_parameters(
                results["heatup_analysis"], results["cooldown_analysis"]
            )
            if solved_params:
                results["decoupled_estimation"] = solved_params
                self.logger.info(
                    f"Decoupled estimation: R={solved_params['R']:.4f} K/W, "
                    f"C={solved_params['C']/1e6:.2f} MJ/K, "
                    f"τ={solved_params['time_constant']:.1f}h"
                )

        # Step 4: Also run the legacy combined method for comparison (now identified as flawed)
        combined_results = self._combined_rc_estimation(data)
        if "R" in combined_results and "C" in combined_results:
            results["legacy_combined_estimation"] = combined_results
            time_const = combined_results.get("time_constant", 0)
            self.logger.info(
                f"Legacy combined analysis: R={combined_results['R']:.2f} °C/W, C={combined_results['C']:.0f} Wh/°C, τ={time_const:.1f}h"
            )

        # Step 5: Also run the more advanced state-space model as an alternative
        ss_results = self._relay_state_space_identification(data)
        if "thermal_parameters" in ss_results:
            results["state_space"] = ss_results
            self.logger.info("State-space identification complete")

        # Step 6: Select the best estimate
        best_estimate = self._select_best_rc_estimate(results)
        if best_estimate:
            results["recommended_parameters"] = best_estimate
            self.logger.info(
                f"Recommended parameters from '{best_estimate.get('method')}': {best_estimate}"
            )

        self.logger.info("RC parameter estimation completed")
        return results

    def _select_best_rc_estimate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best RC parameter estimate based on confidence metrics."""
        candidates = []

        # PRIORITY 1: Decoupled simultaneous estimation (logically sound)
        if "decoupled_estimation" in results:
            decoupled = results["decoupled_estimation"]
            # High base confidence for logically sound method
            cooldown_r2 = results.get("cooldown_analysis", {}).get("r_squared", 0)
            heating_events = results.get("heatup_analysis", {}).get(
                "heating_events_analyzed", 0
            )
            # Improved confidence calculation with lower thresholds
            confidence = 0.7 + (cooldown_r2 * 0.2) + min(heating_events / 15, 0.1)
            # Base confidence reduced from 0.8 to 0.7
            # R² bonus increased from 0.15 to 0.2
            # Heating events bonus: threshold reduced from 20 to 15, max increased to 0.1
            candidates.append(("decoupled", confidence, decoupled))

        # PRIORITY 2: State-space identification
        if "state_space" in results and "model_quality" in results["state_space"]:
            ss = results["state_space"]
            r2 = ss["model_quality"].get("r_squared", 0)
            candidates.append(
                ("state_space", r2 * 0.7, ss.get("thermal_parameters", {}))
            )

        # PRIORITY 3: Legacy combined estimation (known to be flawed)
        if "legacy_combined_estimation" in results:
            combined = results["legacy_combined_estimation"]
            confidence = (
                combined.get("confidence_score", 0) * 0.5
            )  # Penalize flawed method
            candidates.append(("legacy_combined", confidence, combined))

        # FALLBACK: Manual combination from individual analyses
        if "cooldown_analysis" in results and "heatup_analysis" in results:
            cooldown = results["cooldown_analysis"]
            heatup = results["heatup_analysis"]

            # For legacy compatibility, handle old field names
            if (
                "cooling_rate_factor" in cooldown
                and "thermal_capacitance_j_per_k" in heatup
            ):
                # Use the new decoupled method if not already processed
                if "decoupled_estimation" not in results:
                    solved_params = self._solve_for_rc_parameters(heatup, cooldown)
                    if solved_params:
                        confidence = (
                            0.6  # Lower than proper decoupled but higher than legacy
                        )
                        candidates.append(
                            ("fallback_decoupled", confidence, solved_params)
                        )

            elif "thermal_resistance" in cooldown and "thermal_capacitance" in heatup:
                # Legacy fields - create manual combination with low confidence
                R = cooldown["thermal_resistance"]
                C = heatup["thermal_capacitance"]
                confidence = (
                    0.4  # Improved confidence for manual combinations (was 0.3)
                )

                manual = {
                    "R": R,
                    "C": C,
                    "time_constant": R * C / 3600,
                    "method": "manual_combination",
                }
                candidates.append(("manual", confidence, manual))

        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return {"method": best[0], "confidence": best[1], **best[2]}

        return {}

    def _analyze_cooldown_periods(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cooldown periods when relay is OFF to estimate thermal resistance."""
        # Find relay OFF periods longer than 1 hour
        relay_off = data[data["heating_on"] == 0].copy()

        if len(relay_off) < 50:
            return {"warning": "Insufficient relay OFF periods"}

        # Calculate temperature decay during OFF periods
        relay_off["temp_change_rate"] = relay_off["room_temp"].diff() * 12  # Per hour

        # Only consider periods with actual cooling (negative rate)
        cooling_periods = relay_off[relay_off["temp_change_rate"] < -0.01]

        if len(cooling_periods) < 20:
            return {"warning": "Insufficient cooling periods found"}

        # For exponential decay: dT/dt = -(T_room - T_outdoor) / (R*C)
        # So: thermal_resistance R can be estimated from decay rate vs temp difference
        if "outdoor_temp" in data.columns:
            try:
                self.logger.info(
                    f"Outdoor temp data available. Cooling periods columns: {list(cooling_periods.columns)}"
                )

                # Check if cooling_periods already has outdoor_temp (it should, since it's a subset of data)
                if "outdoor_temp" not in cooling_periods.columns:
                    self.logger.error(
                        f"outdoor_temp column missing in cooling periods. Available columns: {list(cooling_periods.columns)}"
                    )
                    return {"warning": "outdoor_temp column missing in cooling periods"}

                temp_diff = (
                    cooling_periods["room_temp"] - cooling_periods["outdoor_temp"]
                )
                decay_rate = -cooling_periods["temp_change_rate"]  # Make positive
            except Exception as e:
                self.logger.error(f"Error processing outdoor temp data: {e}")
                return {
                    "warning": f"Error processing outdoor temperature data: {str(e)}"
                }

            # Linear regression: decay_rate = temp_diff / (R*C)
            # Instead of assuming C, return the cooling rate factor (1/RC)
            if len(temp_diff) > 10:
                slope, intercept, r_value, p_value, _ = linregress(
                    temp_diff, decay_rate
                )

                # The slope of this regression is our cooling rate factor, 1/(R*C)
                # The unit is 1/hours, as decay_rate is in °C/hour
                cooling_rate_factor = slope if slope > 0 else None

                if cooling_rate_factor is not None:
                    return {
                        "cooling_rate_factor": cooling_rate_factor,  # This is 1 / (R*C) in 1/h
                        "r_squared": r_value**2,
                        "p_value": p_value,
                        "cooling_samples": len(cooling_periods),
                        "method": "cooldown_regression",
                    }

        return {"warning": "Could not estimate thermal resistance from cooldown"}

    def _analyze_heatup_periods(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze heatup periods when relay is ON to estimate thermal capacitance."""
        # Find relay ON periods
        relay_on = data[data["heating_on"] == 1].copy()

        if len(relay_on) < 50:
            return {"warning": "Insufficient relay ON periods"}

        # Calculate temperature rise rate during ON periods
        relay_on["temp_change_rate"] = relay_on["room_temp"].diff() * 12  # Per hour

        # Only consider periods with actual heating (positive rate)
        heating_periods = relay_on[relay_on["temp_change_rate"] > 0.01]

        if len(heating_periods) < 20:
            return {"warning": "Insufficient heating periods found"}

        # For heating: C * dT/dt = P_heating - (T_room - T_outdoor)/R
        # Initial heating rate when temp difference is small gives: C * dT/dt ≈ P_heating

        # Find periods right after relay turns ON (first 30 minutes)
        relay_changes = data["heating_on"].diff()
        heating_starts = data[relay_changes == 1].index

        initial_heating_rates = []

        for start_time in heating_starts[:20]:  # Analyze first 20 events
            # Look at first 30 minutes after heating starts
            end_time = start_time + pd.Timedelta(minutes=30)
            initial_period = data.loc[start_time:end_time]

            if len(initial_period) >= 6:  # At least 30 minutes of data
                initial_rate = (
                    initial_period["room_temp"].diff().mean() * 12
                )  # Per hour
                if initial_rate > 0:
                    initial_heating_rates.append(initial_rate)

        if initial_heating_rates:
            mean_initial_rate = np.mean(initial_heating_rates)

            # Use actual room power rating from configuration
            room_name = getattr(self, "_current_room_name", "unknown")
            # Get room power from settings or fallback
            if self.settings:
                heating_power_w = (
                    self.settings.get_room_power(room_name) * 1000
                )  # Convert kW to W
            else:
                heating_power_w = (
                    LoxoneFieldAdapter._get_room_power_rating(room_name) * 1000
                )

            # The rate of temperature change (dT/dt) is approximately P_heating / C
            # So, C = P_heating / (dT/dt)
            # We will return the capacitance and the power used to estimate it.
            if mean_initial_rate > 0:
                thermal_capacitance_j_per_k = (
                    heating_power_w * 3600
                ) / mean_initial_rate
            else:
                thermal_capacitance_j_per_k = None

            return {
                "thermal_capacitance_j_per_k": thermal_capacitance_j_per_k,  # This is C in Joules/Kelvin
                "heating_power_w": heating_power_w,
                "mean_initial_heating_rate_c_per_hr": mean_initial_rate,
                "heating_events_analyzed": len(initial_heating_rates),
                "method": "initial_heating_response",
            }

        return {"warning": "Could not estimate thermal capacitance from heatup"}

    def _combined_rc_estimation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Combined RC estimation using both heating and cooling periods."""
        # Get separate estimates
        cooldown_results = self._analyze_cooldown_periods(data)
        heatup_results = self._analyze_heatup_periods(data)

        R = cooldown_results.get("thermal_resistance")
        C = heatup_results.get("thermal_capacitance")

        if R is not None and C is not None:
            time_constant = R * C / 3600  # Convert to hours

            # Confidence metrics
            cooldown_r2 = cooldown_results.get("r_squared", 0)
            heating_events = heatup_results.get("heating_events_analyzed", 0)

            confidence_score = (cooldown_r2 + min(heating_events / 10, 1)) / 2

            return {
                "R": R,  # °C/W
                "C": C,  # Wh/°C
                "time_constant": time_constant,  # hours
                "confidence_score": confidence_score,
                "method": "combined_relay_analysis",
            }

        return {"warning": "Could not combine RC estimates"}

    def _solve_for_rc_parameters(
        self, heatup_results: Dict[str, Any], cooldown_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Solves for R and C using decoupled results from heating and cooling phases.
        Includes physical constraints to prevent impossible values.
        """
        C_j_per_k = heatup_results.get("thermal_capacitance_j_per_k")
        cooling_factor = cooldown_results.get(
            "cooling_rate_factor"
        )  # This is 1 / (R * C) in 1/hours

        if C_j_per_k is None or cooling_factor is None:
            self.logger.warning(
                "Missing C or cooling factor, cannot solve for R and C."
            )
            return None

        # Physical constraint checks
        if C_j_per_k <= 0:
            self.logger.warning(
                f"Invalid thermal capacitance: {C_j_per_k} J/K (must be > 0)"
            )
            return None

        if cooling_factor <= 0:
            self.logger.warning(
                f"Invalid cooling factor: {cooling_factor} 1/h (must be > 0)"
            )
            return None

        # C is already calculated in Joules per Kelvin
        # cooling_factor is in 1/hours, so 1/cooling_factor is tau in hours.
        # tau_seconds = (1 / cooling_factor) * 3600
        tau_seconds = (1 / cooling_factor) * 3600

        # R = tau / C
        R_k_per_w = tau_seconds / C_j_per_k

        # Physical sanity checks for thermal resistance
        # Typical residential room: R should be between 0.001 and 1.0 K/W
        if R_k_per_w <= 0:
            self.logger.warning(
                f"Invalid thermal resistance: {R_k_per_w} K/W (must be > 0)"
            )
            return None
        elif R_k_per_w > 10.0:  # Very high resistance (over-insulated)
            self.logger.warning(
                f"Unusually high thermal resistance: {R_k_per_w:.3f} K/W (>10 K/W)"
            )
        elif R_k_per_w < 0.0001:  # Very low resistance (no insulation)
            self.logger.warning(
                f"Unusually low thermal resistance: {R_k_per_w:.6f} K/W (<0.0001 K/W)"
            )

        # Physical sanity checks for thermal capacitance
        # Typical residential room: C should be between 1e6 and 1e8 J/K
        if C_j_per_k > 1e9:  # Very high mass
            self.logger.warning(
                f"Unusually high thermal capacitance: {C_j_per_k/1e6:.1f} MJ/K (>1000 MJ/K)"
            )
        elif C_j_per_k < 1e5:  # Very low mass
            self.logger.warning(
                f"Unusually low thermal capacitance: {C_j_per_k/1e6:.3f} MJ/K (<0.1 MJ/K)"
            )

        # Physical sanity checks for time constant
        time_constant_hours = tau_seconds / 3600
        if time_constant_hours > 100:  # Very slow response
            self.logger.warning(
                f"Unusually long time constant: {time_constant_hours:.1f} hours (>100h)"
            )
        elif time_constant_hours < 0.1:  # Very fast response
            self.logger.warning(
                f"Unusually short time constant: {time_constant_hours:.3f} hours (<0.1h)"
            )

        return {
            "R": R_k_per_w,  # Units: K/W or °C/W
            "C": C_j_per_k,  # Units: J/K
            "time_constant": time_constant_hours,  # Units: hours
            "method": "decoupled_simultaneous_estimation",
            "physically_valid": True,
        }

    def _relay_state_space_identification(self, data: pd.DataFrame) -> Dict[str, Any]:
        """State-space identification specifically for relay-controlled systems."""
        if len(data) < 300:
            return {"warning": "Insufficient data for state-space identification"}

        # Discrete-time state-space model for relay control:
        # T[k+1] = a*T[k] + b*T_outdoor[k] + c*relay[k] + d
        # Where: a = exp(-dt/(R*C)), b = (1-a), c = P*R*(1-a), d = noise

        # Prepare data
        T_room = data["room_temp"].values[1:]  # T[k+1]
        T_room_prev = data["room_temp"].values[:-1]  # T[k]
        relay_state = data["heating_on"].values[:-1]  # relay[k]

        if "outdoor_temp" in data.columns:
            T_outdoor = data["outdoor_temp"].values[:-1]  # T_outdoor[k]
        else:
            T_outdoor = np.zeros_like(T_room_prev)

        # Create feature matrix
        X = np.column_stack(
            [T_room_prev, T_outdoor, relay_state, np.ones(len(T_room_prev))]
        )
        y = T_room

        try:
            # Fit linear model
            reg = LinearRegression().fit(X, y)
            a, b, c, d = reg.coef_

            # Extract physical parameters
            dt = 5 / 60  # 5 minutes in hours

            if 0 < a < 1:  # Stability check
                RC = -dt / np.log(a)  # Time constant in hours

                # Estimate individual R and C using additional constraints
                # Use heating power estimation from coefficient c
                if abs(b) > 1e-6:  # Have outdoor temperature influence
                    R_estimate = c / ((1 - a) * 2000)  # Assume 2kW heating power
                    C_estimate = RC / R_estimate
                else:
                    # Fall back to combined estimate
                    R_estimate = None
                    C_estimate = None

                # Model quality
                y_pred = reg.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                results = {
                    "thermal_parameters": {
                        "time_constant": RC,  # hours
                        "thermal_resistance": R_estimate,  # °C/W
                        "thermal_capacitance": C_estimate,  # Wh/°C
                    },
                    "state_space_coefficients": {
                        "a": a,  # Temperature persistence
                        "b": b,  # Outdoor influence
                        "c": c,  # Heating effect
                        "d": d,  # Bias term
                    },
                    "model_quality": {
                        "r_squared": r2,
                        "rmse": rmse,
                        "stable": 0 < a < 1,
                    },
                    "method": "discrete_state_space",
                }

                return results
            else:
                return {"warning": "Unstable state-space model identified"}

        except Exception as e:
            return {"warning": f"State-space identification failed: {str(e)}"}

    def _fit_arx_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit ARX (AutoRegressive with eXogenous inputs) model."""
        if len(data) < 200:
            return {
                "warning": "Insufficient data for ARX model (need at least 200 points)"
            }

        # Prepare data for ARX model
        # T[k] = a1*T[k-1] + a2*T[k-2] + b1*T_out[k-1] + b2*P_heat[k-1]

        room_temp = data["room_temp"].values

        # Create lagged variables
        T_lag1 = np.roll(room_temp, 1)
        T_lag2 = np.roll(room_temp, 2)

        if "outdoor_temp" in data.columns:
            outdoor_temp = data["outdoor_temp"].values
            T_out_lag1 = np.roll(outdoor_temp, 1)
        else:
            T_out_lag1 = np.zeros_like(room_temp)

        if "heating_on" in data.columns:
            heating = data["heating_on"].values * 1000  # Assume 1kW
            P_heat_lag1 = np.roll(heating, 1)
        else:
            P_heat_lag1 = np.zeros_like(room_temp)

        # Remove initial samples affected by rolling
        start_idx = 2
        y = room_temp[start_idx:]
        X = np.column_stack(
            [
                T_lag1[start_idx:],
                T_lag2[start_idx:],
                T_out_lag1[start_idx:],
                P_heat_lag1[start_idx:],
            ]
        )

        if len(y) < 50:
            return {"warning": "Insufficient data after creating lags"}

        try:
            # Fit ARX model
            reg = LinearRegression().fit(X, y)

            # Extract coefficients
            a1, a2, b1, b2 = reg.coef_

            # Model validation
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Stability check (characteristic equation roots should be inside unit circle)
            char_poly = [1, -a1, -a2]
            roots = np.roots(char_poly)
            stable = all(abs(root) < 1 for root in roots)

            return {
                "coefficients": {
                    "a1": a1,  # T[k-1] coefficient
                    "a2": a2,  # T[k-2] coefficient
                    "b1": b1,  # T_outdoor[k-1] coefficient
                    "b2": b2,  # P_heating[k-1] coefficient
                },
                "intercept": reg.intercept_,
                "r_squared": r2,
                "rmse": rmse,
                "stable": stable,
                "characteristic_roots": roots.tolist(),
            }

        except Exception as e:
            return {"warning": f"ARX model fitting failed: {str(e)}"}

    def _analyze_setpoint_tracking(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze setpoint tracking performance."""
        if "setpoint" not in data.columns:
            return {"warning": "No setpoint data available"}

        # Calculate tracking error
        tracking_error = data["room_temp"] - data["setpoint"]

        # Basic tracking statistics
        stats = {
            "mean_error": tracking_error.mean(),
            "rms_error": np.sqrt((tracking_error**2).mean()),
            "max_positive_error": tracking_error.max(),
            "max_negative_error": tracking_error.min(),
            "error_std": tracking_error.std(),
            "percentage_within_1C": (abs(tracking_error) <= 1.0).mean() * 100,
            "percentage_within_0.5C": (abs(tracking_error) <= 0.5).mean() * 100,
        }

        # Overshoot and undershoot analysis
        setpoint_changes = (
            data["setpoint"].diff().abs() > 0.5
        )  # Significant setpoint changes
        if setpoint_changes.any():
            change_periods = data[setpoint_changes]

            overshoots = []
            undershoots = []

            for change_time in change_periods.index[:10]:  # Analyze first 10 changes
                # Look at 2 hours after setpoint change
                end_time = change_time + pd.Timedelta(hours=2)
                period_data = data.loc[change_time:end_time]

                if len(period_data) > 1:
                    new_setpoint = period_data["setpoint"].iloc[0]
                    max_temp = period_data["room_temp"].max()
                    min_temp = period_data["room_temp"].min()

                    overshoot = max(0, max_temp - new_setpoint)
                    undershoot = max(0, new_setpoint - min_temp)

                    if overshoot > 0:
                        overshoots.append(overshoot)
                    if undershoot > 0:
                        undershoots.append(undershoot)

            stats.update(
                {
                    "mean_overshoot": np.mean(overshoots) if overshoots else 0,
                    "mean_undershoot": np.mean(undershoots) if undershoots else 0,
                    "overshoot_events": len(overshoots),
                    "undershoot_events": len(undershoots),
                }
            )

        return stats

    def _analyze_thermal_comfort(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze thermal comfort patterns."""
        room_temp = data["room_temp"]

        # Define comfort zones
        comfort_zones = {
            "too_cold": room_temp < 18,
            "cold": (room_temp >= 18) & (room_temp < 20),
            "comfortable": (room_temp >= 20) & (room_temp <= 24),
            "warm": (room_temp > 24) & (room_temp <= 26),
            "too_warm": room_temp > 26,
        }

        comfort_stats = {}
        for zone, mask in comfort_zones.items():
            comfort_stats[zone] = {
                "percentage": mask.mean() * 100,
                "hours": mask.sum() * 5 / 60,  # 5-minute intervals to hours
            }

        # Temperature stability (variation within periods)
        hourly_std = room_temp.resample("1h").std()
        daily_range = room_temp.resample("1D").agg(lambda x: x.max() - x.min())

        stability_stats = {
            "mean_hourly_std": hourly_std.mean(),
            "mean_daily_range": daily_range.mean(),
            "max_daily_range": daily_range.max(),
            "stable_hours_percentage": (hourly_std < 0.5).mean()
            * 100,  # Hours with <0.5°C variation
        }

        return {
            "comfort_zones": comfort_stats,
            "stability": stability_stats,
            "overall_comfort_score": comfort_stats["comfortable"]["percentage"],
        }

    def _analyze_room_coupling(
        self, room_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze thermal coupling between rooms."""
        if len(room_data) < 2:
            return {"warning": "Need at least 2 rooms for coupling analysis"}

        # Extract temperature data for all rooms
        room_temps = {}
        common_index = None

        for room_name, room_df in room_data.items():
            if room_df.empty:
                continue

            # Find temperature column
            temp_col = None
            for col in ["temperature", "value", "temp"]:
                if col in room_df.columns:
                    temp_col = col
                    break

            if temp_col is not None:
                room_temps[room_name] = room_df[temp_col]

                if common_index is None:
                    common_index = room_df.index
                else:
                    common_index = common_index.intersection(room_df.index)

        if len(room_temps) < 2 or common_index.empty:
            return {
                "warning": "Insufficient room temperature data for coupling analysis"
            }

        # Create correlation matrix
        temp_df = pd.DataFrame(
            {name: temp[common_index] for name, temp in room_temps.items()}
        )
        # Calculate room coupling matrix with variance check
        if temp_df.std().min() > 1e-10:  # All columns have sufficient variance
            correlation_matrix = temp_df.corr()
        else:
            correlation_matrix = pd.DataFrame(
                index=temp_df.columns, columns=temp_df.columns
            )
            correlation_matrix.loc[:, :] = np.nan
            self.logger.debug("Room coupling matrix skipped due to zero variance")

        # Heat transfer analysis
        coupling_results = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "room_pairs": {},
        }

        # Analyze each room pair
        for i, room1 in enumerate(temp_df.columns):
            for j, room2 in enumerate(temp_df.columns[i + 1 :], i + 1):
                # Calculate temperature difference and heat transfer potential
                temp_diff = temp_df[room1] - temp_df[room2]

                # Estimate heat transfer coefficient between rooms
                # This is simplified - in reality would need room dimensions and wall properties
                heat_transfer_stats = {
                    "correlation": correlation_matrix.loc[room1, room2],
                    "mean_temp_diff": temp_diff.mean(),
                    "max_temp_diff": temp_diff.abs().max(),
                    "temp_diff_std": temp_diff.std(),
                }

                coupling_results["room_pairs"][f"{room1}_{room2}"] = heat_transfer_stats

        # Identify most and least coupled room pairs
        correlations = [
            (pair, data["correlation"])
            for pair, data in coupling_results["room_pairs"].items()
        ]
        if correlations:
            most_coupled = max(correlations, key=lambda x: x[1])
            least_coupled = min(correlations, key=lambda x: x[1])

            coupling_results.update(
                {
                    "most_coupled_pair": most_coupled[0],
                    "highest_correlation": most_coupled[1],
                    "least_coupled_pair": least_coupled[0],
                    "lowest_correlation": least_coupled[1],
                }
            )

        return coupling_results

    def _calculate_thermal_capacitance_from_peak(
        self,
        cycle: Dict[str, Any],
        initial_temp: float,
        heating_end_temp: float,
        peak_temp: float,
        peak_time: pd.Timestamp,
        heating_start: pd.Timestamp,
        heating_end: pd.Timestamp,
        energy_input_kwh: float,
    ) -> Dict[str, Any]:
        """Calculate thermal capacitance considering heat losses and room coupling.

        During heating: Heat_input = Heat_stored + Heat_lost_outside + Heat_lost_to_rooms
        After heating: Heat_stored = Heat_lost_outside + Heat_lost_to_rooms

        The peak timing tells us when the net heat flow becomes zero.
        """

        if energy_input_kwh <= 0:
            return {"error": "No energy input data available"}

        heating_duration_hours = (heating_end - heating_start).total_seconds() / 3600
        peak_delay_hours = (peak_time - heating_end).total_seconds() / 3600

        # Phase 1: During heating (heat input vs losses)
        heating_phase_rise = heating_end_temp - initial_temp

        # Phase 2: After heating (thermal inertia vs losses)
        post_heating_rise = peak_temp - heating_end_temp

        # Estimate heat loss rates
        # Assume outdoor temperature and adjacent room temperatures are relatively constant
        # Heat loss rate is proportional to temperature difference

        # Average temperature during heating phase for heat loss calculations
        avg_temp_during_heating = (initial_temp + heating_end_temp) / 2

        # Heat input rate (kW)
        heat_input_rate_kw = energy_input_kwh / heating_duration_hours

        # During heating phase:
        # Net heat storage rate = Heat input rate - Heat loss rate
        # dT/dt = (P_in - P_loss) / C
        # where P_loss = (T_room - T_outside) / R_total

        # Temperature rise rate during heating
        heating_temp_rise_rate = heating_phase_rise / heating_duration_hours  # °C/hr

        # After heating stops:
        # dT/dt = -P_loss / C = -(T_room - T_outside) / (R_total * C)
        # The fact that temperature continues to rise means stored heat is redistributing

        analysis = {
            "heating_phase_analysis": {
                "temperature_rise": heating_phase_rise,
                "rise_rate_per_hour": heating_temp_rise_rate,
                "heat_input_rate_kw": heat_input_rate_kw,
                "net_heat_storage_efficiency": heating_temp_rise_rate
                / heat_input_rate_kw
                if heat_input_rate_kw > 0
                else None,
            },
            "post_heating_analysis": {
                "temperature_rise": post_heating_rise,
                "peak_delay_hours": peak_delay_hours,
                "thermal_inertia_indicator": post_heating_rise / heating_phase_rise
                if heating_phase_rise > 0
                else None,
            },
            "thermal_mass_indicators": {
                "total_temperature_rise": peak_temp - initial_temp,
                "peak_delay_ratio": peak_delay_hours / heating_duration_hours
                if heating_duration_hours > 0
                else None,
            },
        }

        # Enhanced thermal capacitance calculation considering heat flows
        if heating_temp_rise_rate > 0 and heat_input_rate_kw > 0:
            # Key insight: The peak delay tells us about thermal inertia
            # Longer delay = more thermal mass storing and redistributing heat

            # During heating: P_input = C * dT/dt + P_losses
            # Where P_losses = heat to outside + heat to adjacent rooms

            # The net heat storage rate during heating
            net_heat_storage_rate = heating_temp_rise_rate  # °C/hr

            # Estimate heat loss coefficient from post-heating behavior
            # After heating stops, the continued temperature rise indicates
            # heat redistribution from thermal mass

            if peak_delay_hours > 0 and post_heating_rise > 0:
                # Heat redistribution rate after heating stops
                redistribution_rate = post_heating_rise / peak_delay_hours  # °C/hr

                # This gives us insight into thermal mass vs heat losses
                thermal_mass_factor = peak_delay_hours * heating_duration_hours

                # Corrected thermal capacitance accounting for heat losses
                # The fact that temperature rises after heating indicates stored heat
                correction_factor = 1 + (post_heating_rise / heating_phase_rise)

                corrected_thermal_capacitance = (
                    energy_input_kwh * correction_factor
                ) / (peak_temp - initial_temp)

                analysis["estimated_parameters"] = {
                    "thermal_capacitance_kwh_per_k": corrected_thermal_capacitance,
                    "thermal_capacitance_mj_per_k": corrected_thermal_capacitance * 3.6,
                    "correction_factor": correction_factor,
                    "thermal_mass_factor": thermal_mass_factor,
                    "redistribution_rate_c_per_hour": redistribution_rate,
                    "confidence": "medium",
                    "note": "Accounts for thermal inertia and heat redistribution",
                }

                # Estimate effective thermal resistance during heating
                # During heating: Net power stored = Input power - Lost power
                # Lost power ≈ Temperature difference / Thermal resistance

                if avg_temp_during_heating > 0:
                    # This is a rough estimate - would need outdoor temp for accuracy
                    estimated_heat_loss_rate = heat_input_rate_kw - (
                        net_heat_storage_rate * corrected_thermal_capacitance
                    )

                    if estimated_heat_loss_rate > 0:
                        analysis["heat_loss_analysis"] = {
                            "estimated_heat_loss_rate_kw": estimated_heat_loss_rate,
                            "heat_loss_fraction": estimated_heat_loss_rate
                            / heat_input_rate_kw,
                            "storage_efficiency": (
                                heat_input_rate_kw - estimated_heat_loss_rate
                            )
                            / heat_input_rate_kw,
                            "note": "Heat loss estimates require outdoor and adjacent room temperatures for accuracy",
                        }
            else:
                # Fallback to simple calculation
                simple_capacitance = energy_input_kwh / (peak_temp - initial_temp)
                analysis["estimated_parameters"] = {
                    "thermal_capacitance_kwh_per_k": simple_capacitance,
                    "thermal_capacitance_mj_per_k": simple_capacitance * 3.6,
                    "confidence": "low",
                    "note": "Simple calculation - no post-heating rise detected",
                }

        return analysis

    def _estimate_heat_losses_during_cycle(
        self,
        room_temp_series: pd.Series,
        outdoor_temp_series: pd.Series,
        heating_start: pd.Timestamp,
        heating_end: pd.Timestamp,
    ) -> Dict[str, float]:
        """Estimate heat losses to outside and adjacent rooms during heating cycle."""

        # Get temperature data for the heating period
        heating_period = room_temp_series[heating_start:heating_end]
        outdoor_period = outdoor_temp_series[heating_start:heating_end]

        if len(heating_period) < 3 or len(outdoor_period) < 3:
            return {"error": "Insufficient temperature data"}

        # Calculate average temperature differences
        avg_room_temp = heating_period.mean()
        avg_outdoor_temp = outdoor_period.mean()
        avg_temp_diff_outside = avg_room_temp - avg_outdoor_temp

        # For adjacent rooms, we'd need their temperatures
        # For now, estimate based on the fact that adjacent rooms are typically
        # warmer than outside but cooler than the heated room

        estimated_losses = {
            "avg_temp_diff_to_outside": avg_temp_diff_outside,
            "note": "Heat loss estimation requires outdoor and adjacent room temperatures",
        }

        return estimated_losses

    def _find_matching_relay_room(self, room_name: str, relay_room_keys: list) -> str:
        """
        Find the matching relay key for a given room name.

        Args:
            room_name: Name of the room (e.g., 'obyvak')
            relay_room_keys: List of available relay data keys (e.g., ['relay_obyvak'])

        Returns:
            Matching relay key or None if not found
        """
        # Direct match first
        if room_name in relay_room_keys:
            return room_name

        # Try with 'relay_' prefix
        relay_key = f"relay_{room_name}"
        if relay_key in relay_room_keys:
            return relay_key

        # Try without 'room_' prefix if present
        if room_name.startswith("room_"):
            clean_room_name = room_name[5:]  # Remove 'room_' prefix
            if clean_room_name in relay_room_keys:
                return clean_room_name
            relay_key = f"relay_{clean_room_name}"
            if relay_key in relay_room_keys:
                return relay_key

        # Fuzzy matching for common variations
        for key in relay_room_keys:
            # Remove common prefixes for comparison
            clean_key = key.replace("relay_", "").replace("room_", "")
            clean_room = room_name.replace("relay_", "").replace("room_", "")

            if clean_key == clean_room:
                return key

        self.logger.debug(
            f"No matching relay found for room '{room_name}' in {relay_room_keys}"
        )
        return None

    def analyze_sustained_heating_cycles(
        self, room_temp: pd.Series, outdoor_temp: pd.Series, relay_states: pd.Series
    ) -> Dict[str, Any]:
        """Analyze sustained heating cycles followed by decay periods.

        This method looks for heating periods longer than min_heating_duration_hours,
        followed by non-heating periods longer than min_non_heating_duration_hours.
        It analyzes the temperature rise, peak finding, and decay characteristics.
        """

        if len(room_temp) < 100 or len(relay_states) < 100:
            return {"error": "Insufficient data for sustained heating analysis"}

        # Ensure data is aligned
        common_index = room_temp.index.intersection(relay_states.index).intersection(
            outdoor_temp.index
        )
        if len(common_index) < 100:
            return {"error": "Insufficient aligned data"}

        room_temp = room_temp.reindex(common_index)
        relay_states = relay_states.reindex(common_index)
        outdoor_temp = outdoor_temp.reindex(common_index)

        # Find sustained heating periods
        heating_cycles = self._find_sustained_heating_cycles(relay_states)

        if not heating_cycles:
            return {
                "warning": f"No sustained heating cycles found (min duration: {self.min_heating_duration_hours}h)"
            }

        cycle_analyses = []

        for cycle in heating_cycles:
            cycle_analysis = self._analyze_single_heating_cycle(
                room_temp, outdoor_temp, cycle
            )
            if cycle_analysis:
                cycle_analyses.append(cycle_analysis)

        if not cycle_analyses:
            return {"warning": "No valid heating cycles could be analyzed"}

        # Aggregate results across all cycles
        summary_stats = self._aggregate_cycle_statistics(cycle_analyses)

        return {
            "heating_cycles_analyzed": len(cycle_analyses),
            "sustained_cycles_found": len(heating_cycles),
            "cycle_details": cycle_analyses,
            "summary_statistics": summary_stats,
            "configuration": {
                "min_heating_duration_hours": self.min_heating_duration_hours,
                "min_non_heating_duration_hours": self.min_non_heating_duration_hours,
                "decay_analysis_hours": self.decay_analysis_hours,
            },
        }

    def _find_sustained_heating_cycles(
        self, relay_states: pd.Series
    ) -> List[Dict[str, Any]]:
        """Find periods of sustained heating followed by sustained non-heating."""

        cycles = []
        heating_start = None

        for i in range(1, len(relay_states)):
            current_state = relay_states.iloc[i]
            prev_state = relay_states.iloc[i - 1]
            current_time = relay_states.index[i]

            # Detect start of heating period
            if current_state == 1 and prev_state == 0:
                heating_start = current_time

            # Detect end of heating period
            elif current_state == 0 and prev_state == 1 and heating_start is not None:
                heating_end = current_time
                heating_duration_hours = (
                    heating_end - heating_start
                ).total_seconds() / 3600

                # Check if heating period is long enough
                if heating_duration_hours >= self.min_heating_duration_hours:
                    # Look for sustained non-heating period after this
                    non_heating_end = self._find_non_heating_period_end(
                        relay_states, heating_end
                    )

                    if non_heating_end:
                        non_heating_duration_hours = (
                            non_heating_end - heating_end
                        ).total_seconds() / 3600

                        if (
                            non_heating_duration_hours
                            >= self.min_non_heating_duration_hours
                        ):
                            cycles.append(
                                {
                                    "heating_start": heating_start,
                                    "heating_end": heating_end,
                                    "non_heating_end": non_heating_end,
                                    "heating_duration_hours": heating_duration_hours,
                                    "non_heating_duration_hours": non_heating_duration_hours,
                                }
                            )

                heating_start = None

        return cycles

    def _find_non_heating_period_end(
        self, relay_states: pd.Series, heating_end
    ) -> Optional[pd.Timestamp]:
        """Find the end of a sustained non-heating period."""

        # Look at data after heating ended
        after_heating = relay_states[relay_states.index > heating_end]

        last_time = heating_end

        for timestamp, state in after_heating.items():
            current_duration = (timestamp - heating_end).total_seconds() / 3600

            # If we hit another heating period before reaching minimum duration
            if state == 1 and current_duration < self.min_non_heating_duration_hours:
                return None

            # If we've reached minimum non-heating duration
            if current_duration >= self.min_non_heating_duration_hours:
                return timestamp

            last_time = timestamp

        # If we reached end of data and have enough non-heating time
        final_duration = (last_time - heating_end).total_seconds() / 3600
        if final_duration >= self.min_non_heating_duration_hours:
            return last_time

        return None

    def _analyze_single_heating_cycle(
        self,
        room_temp: pd.Series,
        outdoor_temp: pd.Series,
        cycle: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single sustained heating cycle."""

        heating_start = cycle["heating_start"]
        heating_end = cycle["heating_end"]
        non_heating_end = cycle["non_heating_end"]

        # Get temperature data for the cycle
        cycle_data = room_temp[heating_start:non_heating_end]

        if len(cycle_data) < 10:
            return None

        # Analyze heating phase
        heating_data = room_temp[heating_start:heating_end]
        if len(heating_data) < 5:
            return None

        initial_temp = heating_data.iloc[0]
        heating_end_temp = heating_data.iloc[-1]

        # Find temperature peak across the entire heating + non-heating period
        entire_cycle_data = room_temp[heating_start:non_heating_end]
        if len(entire_cycle_data) > 0:
            peak_temp = entire_cycle_data.max()
            peak_time = entire_cycle_data.idxmax()
            peak_delay_actual = (peak_time - heating_end).total_seconds() / 3600
        else:
            peak_temp = heating_end_temp
            peak_time = heating_end
            peak_delay_actual = 0

        # Analyze decay phase
        decay_start = peak_time
        decay_end = min(
            decay_start + pd.Timedelta(hours=self.decay_analysis_hours), non_heating_end
        )

        decay_analysis = self._analyze_temperature_decay(
            room_temp[decay_start:decay_end],
            outdoor_temp[decay_start:decay_end],
            peak_temp,
        )

        # Calculate energy input estimate (if room power rating available)
        energy_input_kwh = self._estimate_energy_input(cycle)

        # Calculate temperature rise metrics
        total_temp_rise = peak_temp - initial_temp
        heating_phase_rise = heating_end_temp - initial_temp
        post_heating_rise = peak_temp - heating_end_temp

        # Calculate thermal capacitance using peak timing and energy input
        thermal_capacitance_analysis = self._calculate_thermal_capacitance_from_peak(
            cycle,
            initial_temp,
            heating_end_temp,
            peak_temp,
            peak_time,
            heating_start,
            heating_end,
            energy_input_kwh,
        )

        return {
            "cycle_start": heating_start,
            "cycle_end": non_heating_end,
            "heating_duration_hours": cycle["heating_duration_hours"],
            "non_heating_duration_hours": cycle["non_heating_duration_hours"],
            "initial_temperature": initial_temp,
            "heating_end_temperature": heating_end_temp,
            "peak_temperature": peak_temp,
            "peak_time": peak_time,
            "peak_delay_hours": peak_delay_actual,
            "total_temperature_rise": total_temp_rise,
            "heating_phase_rise": heating_phase_rise,
            "post_heating_rise": post_heating_rise,
            "energy_input_kwh": energy_input_kwh,
            "thermal_efficiency": total_temp_rise / energy_input_kwh
            if energy_input_kwh > 0
            else None,
            "thermal_capacitance_analysis": thermal_capacitance_analysis,
            "decay_analysis": decay_analysis,
        }

    def _analyze_temperature_decay(
        self, decay_temp: pd.Series, outdoor_temp: pd.Series, peak_temp: float
    ) -> Dict[str, Any]:
        """Analyze temperature decay characteristics."""

        if len(decay_temp) < 5:
            return {"error": "Insufficient decay data"}

        # Calculate decay rate
        time_hours = [
            (t - decay_temp.index[0]).total_seconds() / 3600 for t in decay_temp.index
        ]
        temps = decay_temp.values

        if len(time_hours) < 3:
            return {"error": "Insufficient time points"}

        # Linear decay rate (simple)
        try:
            slope, _, r_value, _, _ = linregress(time_hours, temps)
            linear_decay_rate = -slope  # Make positive (temperature is dropping)
        except Exception:
            linear_decay_rate = None
            r_value = None

        # Exponential decay analysis
        exponential_params = self._fit_exponential_decay(time_hours, temps, peak_temp)

        # Calculate heat loss to outside coupling
        temp_diff_to_outside = decay_temp - outdoor_temp.reindex(
            decay_temp.index, method="nearest"
        )
        avg_temp_diff = temp_diff_to_outside.mean()

        # Room coupling (how much temperature tracks outdoor changes)
        outdoor_changes = outdoor_temp.diff()
        room_changes = decay_temp.diff()

        if len(outdoor_changes) > 5 and len(room_changes) > 5:
            # Align the series
            common_idx = outdoor_changes.index.intersection(room_changes.index)
            if len(common_idx) > 5:
                outdoor_aligned = outdoor_changes.reindex(common_idx).fillna(0)
                room_aligned = room_changes.reindex(common_idx).fillna(0)
                try:
                    # Check for sufficient variance to avoid divide by zero warnings
                    if outdoor_aligned.std() > 1e-10 and room_aligned.std() > 1e-10:
                        coupling_corr = np.corrcoef(outdoor_aligned, room_aligned)[0, 1]
                    else:
                        coupling_corr = None
                        self.logger.debug(
                            "Correlation skipped due to zero variance in temperature data"
                        )
                except Exception:
                    coupling_corr = None
            else:
                coupling_corr = None
        else:
            coupling_corr = None

        return {
            "decay_duration_hours": (
                decay_temp.index[-1] - decay_temp.index[0]
            ).total_seconds()
            / 3600,
            "temperature_drop": peak_temp - decay_temp.iloc[-1],
            "linear_decay_rate_per_hour": linear_decay_rate,
            "linear_fit_r_squared": r_value**2 if r_value else None,
            "exponential_decay": exponential_params,
            "average_temp_diff_to_outside": avg_temp_diff,
            "room_outdoor_coupling": coupling_corr,
            "initial_decay_temp": decay_temp.iloc[0],
            "final_decay_temp": decay_temp.iloc[-1],
        }

    def _fit_exponential_decay(
        self, time_hours: List[float], temps: List[float], _: float
    ) -> Dict[str, Any]:
        """Fit exponential decay model to temperature data."""

        try:
            # Exponential decay: T(t) = T_ambient + (T_peak - T_ambient) * exp(-t/tau)
            # Simplified: T(t) = A * exp(-t/tau) + C

            def exp_decay(t, A, tau, C):
                return A * np.exp(-np.array(t) / tau) + C

            # Improved initial guess for better numerical stability
            temp_range = max(temps) - min(temps)
            A_guess = max(0.1, abs(temps[0] - temps[-1]))  # Ensure positive amplitude
            tau_guess = np.clip(
                max(time_hours) / 3, TAU_MIN, TAU_MAX / 2
            )  # Constrain to reasonable range
            C_guess = (
                np.mean(temps[-3:]) if len(temps) >= 3 else temps[-1]
            )  # Use last few points for asymptote

            # Add bounds for physical plausibility and better numerical stability
            bounds = (
                [
                    0,
                    TAU_MIN,
                    min(temps) - 5,
                ],  # Lower bounds: positive amplitude, min time constant, reasonable asymptote
                [max(temps) - min(temps) + 5, TAU_MAX, max(temps) + 5],  # Upper bounds
            )

            popt, pcov = curve_fit(
                exp_decay,
                time_hours,
                temps,
                p0=[A_guess, tau_guess, C_guess],
                bounds=bounds,
                maxfev=2000,
                method="trf",  # Trust Region Reflective algorithm for better stability
            )

            # Check parameter errors from covariance matrix
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else None

            A_fit, tau_fit, C_fit = popt

            # Calculate R-squared
            y_pred = exp_decay(time_hours, *popt)
            ss_res = np.sum((temps - y_pred) ** 2)
            ss_tot = np.sum((temps - np.mean(temps)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else None

            return {
                "amplitude": A_fit,
                "time_constant_hours": tau_fit,
                "asymptote_temp": C_fit,
                "r_squared": r_squared,
                "fit_successful": True,
            }

        except Exception as e:
            return {"error": str(e), "fit_successful": False}

    def _estimate_energy_input(self, cycle: Dict[str, Any]) -> float:
        """Estimate energy input during heating cycle."""

        heating_duration_hours = cycle["heating_duration_hours"]

        # Try to get room power rating from current analysis context
        room_power_kw = 0
        if hasattr(self, "_current_room_name") and self._current_room_name:
            room_power_kw = self.room_power_ratings_kw.get(self._current_room_name, 0)

        if room_power_kw > 0:
            # Calculate actual energy input based on power rating
            energy_input_kwh = room_power_kw * heating_duration_hours
            return energy_input_kwh
        else:
            # Fallback: estimate based on typical residential heating power
            # Assume 2 kW per room as default (can be overridden in config)
            default_power_kw = 2.0
            return default_power_kw * heating_duration_hours

    def _aggregate_cycle_statistics(
        self, cycle_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate statistics across all analyzed cycles."""

        if not cycle_analyses:
            return {}

        # Extract numeric values for aggregation
        temp_rises = [
            c["total_temperature_rise"]
            for c in cycle_analyses
            if c.get("total_temperature_rise")
        ]
        heating_durations = [c["heating_duration_hours"] for c in cycle_analyses]
        peak_delays = [
            c["peak_delay_hours"] for c in cycle_analyses if c.get("peak_delay_hours")
        ]

        decay_rates = []
        time_constants = []
        coupling_values = []

        for cycle in cycle_analyses:
            decay = cycle.get("decay_analysis", {})
            if decay.get("linear_decay_rate_per_hour"):
                decay_rates.append(decay["linear_decay_rate_per_hour"])

            exp_decay = decay.get("exponential_decay", {})
            if exp_decay.get("time_constant_hours") and exp_decay.get("fit_successful"):
                time_constants.append(exp_decay["time_constant_hours"])

            if decay.get("room_outdoor_coupling") is not None:
                coupling_values.append(decay["room_outdoor_coupling"])

        def safe_stats(values):
            if not values:
                return {"mean": None, "std": None, "min": None, "max": None}
            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

        return {
            "temperature_rise_stats": safe_stats(temp_rises),
            "heating_duration_stats": safe_stats(heating_durations),
            "peak_delay_stats": safe_stats(
                peak_delays
            ),  # Keep for analysis of peak timing
            "decay_rate_stats": safe_stats(decay_rates),
            "time_constant_stats": safe_stats(time_constants),
            "coupling_stats": safe_stats(coupling_values),
        }

    def _calculate_real_heating_usage(self, room_name: str, relay_data: Dict) -> Dict:
        """
        Calculate real heating usage from relay data.

        Args:
            room_name: Name of the room
            relay_data: Dictionary of relay data from integrator

        Returns:
            Dictionary with heating usage statistics
        """
        try:
            # Find matching relay data for this room
            room_key = self._find_matching_relay_room(room_name, relay_data.keys())

            if not room_key or room_key not in relay_data:
                return {
                    "heating_data_source": "no_relay_data",
                    "actual_heating_usage_pct": 0,
                    "note": f"No relay data found for room {room_name}",
                }

            relay_df = relay_data[room_key]

            if "relay_state" not in relay_df.columns:
                return {
                    "heating_data_source": "no_relay_state",
                    "actual_heating_usage_pct": 0,
                    "note": f"No relay_state column in data for {room_name}",
                }

            # Calculate actual heating usage from relay states
            total_points = len(relay_df)
            heating_points = (relay_df["relay_state"] > 0).sum()

            if total_points == 0:
                return {
                    "heating_data_source": "empty_data",
                    "actual_heating_usage_pct": 0,
                    "note": f"Empty relay data for {room_name}",
                }

            heating_percentage = (heating_points / total_points) * 100

            return {
                "heating_data_source": "actual_relay",
                "actual_heating_usage_pct": heating_percentage,
                "total_data_points": total_points,
                "heating_data_points": heating_points,
                "note": f"Calculated from {total_points} relay state measurements",
            }

        except Exception as e:
            self.logger.warning(
                f"Error calculating real heating usage for {room_name}: {e}"
            )
            return {
                "heating_data_source": "calculation_error",
                "actual_heating_usage_pct": 0,
                "error": str(e),
            }

    def _find_matching_relay_room(self, room_name: str, available_keys: list) -> str:
        """
        Find matching relay data key for a room name.

        Args:
            room_name: Name of the room
            available_keys: List of available relay data keys

        Returns:
            Matching key or None
        """
        # Direct match
        if room_name in available_keys:
            return room_name

        # Try with 'relay_' prefix
        relay_key = f"relay_{room_name}"
        if relay_key in available_keys:
            return relay_key

        # Try variations (remove/add underscores, different cases)
        variations = [
            room_name.replace("_", ""),
            room_name.replace("_", "-"),
            room_name.lower(),
            room_name.upper(),
            f"relay_states_{room_name}",
            f"heating_{room_name}",
        ]

        for variation in variations:
            if variation in available_keys:
                return variation

        # Fuzzy matching - look for partial matches
        for key in available_keys:
            if room_name.lower() in key.lower() or key.lower() in room_name.lower():
                return key

        return None
