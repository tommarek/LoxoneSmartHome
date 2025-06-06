"""
Thermal Dynamics Analysis for PEMS v2.

Analyzes thermal dynamics per room:
1. Calculate thermal parameters (heat-up rate, cool-down rate, time constant)
2. Use system identification (ARX model, state-space model)
3. Account for solar gains, internal gains, adjacent room heat transfer
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from config.energy_settings import get_room_power


class ThermalAnalyzer:
    """Analyze thermal dynamics for each room."""

    def __init__(self):
        """Initialize the thermal analyzer."""
        self.logger = logging.getLogger(f"{__name__}.ThermalAnalyzer")

    def analyze_room_dynamics(
        self, room_data: Dict[str, pd.DataFrame], weather_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze thermal dynamics for all rooms.

        Args:
            room_data: Dictionary of room DataFrames with temperature data
            weather_data: Weather data with outdoor temperature

        Returns:
            Dictionary with thermal analysis results for each room
        """
        self.logger.info("Starting thermal dynamics analysis")

        if not room_data:
            self.logger.warning("No room data provided")
            return {}

        results = {}

        # Analyze each room individually
        for room_name, room_df in room_data.items():
            self.logger.info(f"Analyzing thermal dynamics for room: {room_name}")

            try:
                # Store room name for power calculations
                self._current_room_name = room_name
                room_results = self._analyze_single_room(
                    room_df, weather_data, room_name
                )
                results[room_name] = room_results
            except Exception as e:
                self.logger.error(f"Failed to analyze room {room_name}: {e}")
                results[room_name] = {"error": str(e)}

        # Analyze room coupling (heat transfer between rooms)
        if len(results) > 1:
            results["room_coupling"] = self._analyze_room_coupling(room_data)

        self.logger.info("Thermal dynamics analysis completed")
        return results

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
        results["rc_parameters"] = self.estimate_rc_parameters(merged_data)

        # ARX model identification
        results["arx_model"] = self._fit_arx_model(merged_data)

        # Setpoint tracking analysis
        results["setpoint_analysis"] = self._analyze_setpoint_tracking(merged_data)

        # Thermal comfort analysis
        results["comfort_analysis"] = self._analyze_thermal_comfort(merged_data)

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

        # Prepare room data
        room_clean = room_df[[temp_col]].copy()
        room_clean.columns = ["room_temp"]

        # Add heating status if available
        heating_cols = [
            col
            for col in room_df.columns
            if "heating" in col.lower() or "heat" in col.lower()
        ]
        if heating_cols:
            room_clean["heating_on"] = room_df[heating_cols[0]]
        else:
            # Infer heating from temperature changes
            room_clean["heating_on"] = self._infer_heating_status(
                room_clean["room_temp"]
            )

        # Add setpoint if available
        if "setpoint" in room_df.columns:
            room_clean["setpoint"] = room_df["setpoint"]

        # Merge with weather data
        if not weather_data.empty and "temperature" in weather_data.columns:
            weather_resampled = (
                weather_data[["temperature"]]
                .resample("5T")
                .interpolate(method="linear")
            )
            weather_resampled.columns = ["outdoor_temp"]
            merged = room_clean.join(weather_resampled, how="inner")
        else:
            merged = room_clean
            self.logger.warning("No outdoor temperature data available")

        # Calculate temperature difference
        if "outdoor_temp" in merged.columns:
            merged["temp_diff"] = merged["room_temp"] - merged["outdoor_temp"]

        # Add time features
        merged["hour"] = merged.index.hour
        merged["weekday"] = merged.index.weekday

        return merged.dropna()

    def _infer_heating_status(self, temperature: pd.Series) -> pd.Series:
        """Infer heating status from temperature changes."""
        # Simple heuristic: heating is on when temperature is rising significantly
        temp_change = (
            temperature.diff().rolling(window=3).mean()
        )  # 15-minute moving average
        heating_threshold = 0.1  # 0.1°C increase per 5 minutes indicates heating

        return (temp_change > heating_threshold).astype(int)

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

        # Outdoor temperature relationship
        if "outdoor_temp" in data.columns:
            correlation = data["room_temp"].corr(data["outdoor_temp"])
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

        # Correlation analysis
        solar_correlation = solar_warming["solar_proxy"].corr(
            solar_warming["temp_change"]
        )

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
            reg = LinearRegression().fit(X, y)

            # Extract parameters
            coeff_temp = reg.coef_[0]  # 1/(R*C)
            coeff_heating = reg.coef_[1]  # 1/C

            if coeff_heating > 0:
                C = 1 / coeff_heating  # Thermal capacity in Wh/°C
                R = 1 / (coeff_temp * C)  # Thermal resistance in °C/W

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
                }
            else:
                return {"warning": "Invalid heating coefficient in RC model"}

        except Exception as e:
            return {"warning": f"RC model fitting failed: {str(e)}"}

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

        # Method 1: Cooldown Analysis (Relay OFF periods)
        cooldown_results = self._analyze_cooldown_periods(data)
        if "thermal_resistance" in cooldown_results:
            results["cooldown_analysis"] = cooldown_results
            self.logger.info(
                f"Cooldown analysis complete: R = {cooldown_results['thermal_resistance']:.2f} °C/W"
            )

        # Method 2: Heatup Analysis (Relay ON periods)
        heatup_results = self._analyze_heatup_periods(data)
        if "thermal_capacitance" in heatup_results:
            results["heatup_analysis"] = heatup_results
            self.logger.info(
                f"Heatup analysis complete: C = {heatup_results['thermal_capacitance']:.0f} Wh/°C"
            )

        # Method 3: Combined RC estimation using both periods
        combined_results = self._combined_rc_estimation(data)
        if "R" in combined_results and "C" in combined_results:
            results["combined_estimation"] = combined_results
            time_const = combined_results.get("time_constant", 0)
            self.logger.info(
                f"Combined analysis: R={combined_results['R']:.2f} °C/W, C={combined_results['C']:.0f} Wh/°C, τ={time_const:.1f}h"
            )

        # Method 4: State-space identification for relay systems
        ss_results = self._relay_state_space_identification(data)
        if "thermal_parameters" in ss_results:
            results["state_space"] = ss_results
            self.logger.info("State-space identification complete")

        # Select best estimate based on confidence metrics
        best_estimate = self._select_best_rc_estimate(results)
        if best_estimate:
            results["recommended_parameters"] = best_estimate
            self.logger.info(f"Recommended parameters: {best_estimate}")

        self.logger.info("RC parameter estimation completed")
        return results

    def _select_best_rc_estimate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best RC parameter estimate based on confidence metrics."""
        candidates = []

        # Evaluate each method
        if "combined_estimation" in results:
            combined = results["combined_estimation"]
            confidence = combined.get("confidence_score", 0)
            candidates.append(("combined", confidence, combined))

        if "state_space" in results and "model_quality" in results["state_space"]:
            ss = results["state_space"]
            r2 = ss["model_quality"].get("r_squared", 0)
            candidates.append(("state_space", r2, ss.get("thermal_parameters", {})))

        if "cooldown_analysis" in results and "heatup_analysis" in results:
            # Create manual combination
            cooldown = results["cooldown_analysis"]
            heatup = results["heatup_analysis"]

            if "thermal_resistance" in cooldown and "thermal_capacitance" in heatup:
                R = cooldown["thermal_resistance"]
                C = heatup["thermal_capacitance"]
                confidence = (
                    cooldown.get("r_squared", 0) + 0.5
                ) / 2  # Lower confidence for manual

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
            cooling_periods = cooling_periods.merge(
                data[["outdoor_temp"]], left_index=True, right_index=True, how="inner"
            )

            temp_diff = cooling_periods["room_temp"] - cooling_periods["outdoor_temp"]
            decay_rate = -cooling_periods["temp_change_rate"]  # Make positive

            # Linear regression: decay_rate = temp_diff / (R*C)
            # Assuming typical C for room, estimate R
            if len(temp_diff) > 10:
                slope, intercept, r_value, p_value, _ = linregress(
                    temp_diff, decay_rate
                )

                # Estimate thermal capacitance (typical room values)
                estimated_C = 15000  # Wh/°C (conservative estimate for room)
                thermal_resistance = 1 / (slope * estimated_C) if slope > 0 else None

                return {
                    "thermal_resistance": thermal_resistance,  # °C/W
                    "assumed_capacitance": estimated_C,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "cooling_samples": len(cooling_periods),
                    "method": "cooldown_exponential_decay",
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

            # Estimate heating power and thermal capacitance
            # Use actual room power rating from configuration
            room_name = getattr(self, "_current_room_name", "unknown")
            estimated_power = get_room_power(room_name) * 1000  # Convert kW to W
            thermal_capacitance = (
                estimated_power / mean_initial_rate if mean_initial_rate > 0 else None
            )

            return {
                "thermal_capacitance": thermal_capacitance,  # Wh/°C
                "assumed_power": estimated_power,
                "mean_initial_heating_rate": mean_initial_rate,
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
        correlation_matrix = temp_df.corr()

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
