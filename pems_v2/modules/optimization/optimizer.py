"""
Multi-objective Energy Optimization Engine for PEMS v2.

Implements model predictive control with multi-objective optimization for:
- Cost minimization
- Self-consumption maximization  
- Peak shaving
- Comfort maintenance
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    success: bool
    objective_value: float
    heating_schedule: Dict[str, pd.Series]
    
    # Growatt control schedules (binary mode decisions)
    battery_first_schedule: pd.Series
    ac_charge_schedule: pd.Series
    export_schedule: pd.Series
    
    # Power flow schedules (continuous values)
    battery_schedule: pd.Series  # Actual battery power (W)
    grid_schedule: pd.Series     # Net grid power (W, positive = import)
    
    temperature_forecast: Dict[str, pd.Series]
    cost_breakdown: Dict[str, float]
    solve_time_seconds: float
    message: str = ""

    @property
    def solve_time(self) -> float:
        """Alias for solve_time_seconds for compatibility."""
        return self.solve_time_seconds


@dataclass
class OptimizationProblem:
    """Definition of the optimization problem."""

    horizon_hours: int
    time_step_minutes: int
    start_time: datetime

    # Forecasts
    pv_forecast: pd.Series
    load_forecast: pd.Series
    price_forecast: pd.Series
    weather_forecast: pd.DataFrame

    # System states
    initial_battery_soc: float
    initial_temperatures: Dict[str, float]

    # Constraints
    comfort_bounds: Dict[str, Tuple[float, float]]  # (min_temp, max_temp) per room
    battery_capacity_kwh: float = 10.0
    battery_max_power_kw: float = 5.0

    # Weights for multi-objective optimization
    cost_weight: float = 0.7
    comfort_weight: float = 0.2
    self_consumption_weight: float = 0.1


class EnergyOptimizer:
    """
    Multi-objective energy optimization engine.

    Uses model predictive control to optimize energy flows across:
    - PV production
    - Battery storage
    - Heating systems
    - Grid interaction
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the energy optimizer.

        Args:
            config: Configuration dictionary with system parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # System configuration
        self.rooms = config.get("rooms", {})
        self.battery_config = config.get("battery", {})
        self.pv_config = config.get("pv_system", {})

        # Optimization parameters
        self.solver_timeout = config.get("solver_timeout", 30)  # seconds
        self.mip_gap = config.get("mip_gap", 0.01)  # 1%

        # Thermal model parameters (simplified)
        self.thermal_params = self._load_thermal_parameters()

        self.logger.info(f"Energy optimizer initialized for {len(self.rooms)} rooms")

    def optimize(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve the energy optimization problem.

        Args:
            problem: Optimization problem definition

        Returns:
            Optimization result with schedules and metrics
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting optimization for {problem.horizon_hours}h horizon"
            )

            # Set up optimization variables
            n_steps = problem.horizon_hours * 60 // problem.time_step_minutes

            # Decision variables
            heating_vars = {}
            for room in self.rooms:
                heating_vars[room] = cp.Variable(
                    n_steps, boolean=True, name=f"heat_{room}"
                )

            # Growatt inverter control variables (discrete modes)
            battery_first_mode = cp.Variable(
                n_steps, boolean=True, name="battery_first_mode"
            )
            ac_charge_mode = cp.Variable(
                n_steps, boolean=True, name="ac_charge_mode"
            )
            export_enabled = cp.Variable(
                n_steps, boolean=True, name="export_enabled"
            )
            
            # Power flows (continuous variables derived from modes)
            battery_power = cp.Variable(n_steps, name="battery_power")  # Actual battery power
            grid_import = cp.Variable(n_steps, name="grid_import")
            grid_export = cp.Variable(n_steps, name="grid_export")

            # Temperature variables (simplified linear model)
            temp_vars = {}
            for room in self.rooms:
                temp_vars[room] = cp.Variable(n_steps + 1, name=f"temp_{room}")

            # Comfort violation variables (soft constraints)
            comfort_violations = {}
            violation_penalty = 1000  # €/°C violation
            for room in self.rooms:
                if room in problem.comfort_bounds:
                    comfort_violations[room] = {
                        "low": cp.Variable(
                            n_steps, nonneg=True, name=f"viol_low_{room}"
                        ),
                        "high": cp.Variable(
                            n_steps, nonneg=True, name=f"viol_high_{room}"
                        ),
                    }

            # Objective function with violation penalties
            objective = self._build_objective(
                problem, heating_vars, battery_power, grid_import, grid_export, 
                battery_first_mode, ac_charge_mode, export_enabled, n_steps
            )

            # Add comfort violation penalties to objective
            for room, violations in comfort_violations.items():
                objective += violation_penalty * cp.sum(violations["low"])
                objective += violation_penalty * cp.sum(violations["high"])

            # Constraints
            constraints = []

            # Power balance constraints
            constraints.extend(
                self._power_balance_constraints(
                    problem,
                    heating_vars,
                    battery_power,
                    grid_import,
                    grid_export,
                    n_steps,
                )
            )

            # Growatt inverter constraints
            constraints.extend(
                self._growatt_constraints(
                    problem, battery_first_mode, ac_charge_mode, export_enabled, 
                    battery_power, grid_import, grid_export, n_steps
                )
            )

            # Thermal constraints with soft violations
            constraints.extend(
                self._thermal_constraints(
                    problem, heating_vars, temp_vars, n_steps, comfort_violations
                )
            )

            # Grid constraints
            constraints.extend(
                self._grid_constraints(grid_import, grid_export, n_steps)
            )

            # Solve the optimization problem
            prob = cp.Problem(cp.Minimize(objective), constraints)

            # Use appropriate solver based on problem type
            has_binary_vars = any(
                getattr(var, "attributes", {}).get("boolean", False)
                for var in heating_vars.values()
            )

            self.logger.debug(f"Binary variables detected: {has_binary_vars}")

            if has_binary_vars:
                # Use ECOS_BB for mixed-integer problems
                solver_options = {
                    "verbose": False,
                    "mi_max_iters": 1000,
                    "feastol": 1e-6,  # Relaxed from 1e-8
                    "abstol": 1e-6,  # Relaxed from 1e-8
                    "reltol": 1e-6,  # Add relative tolerance
                }
                prob.solve(solver=cp.ECOS_BB, **solver_options)
            else:
                # Use ECOS for continuous problems
                solver_options = {
                    "verbose": False,
                    "max_iters": 2000,  # More iterations
                    "feastol": 1e-6,  # Relaxed from 1e-8
                    "abstol": 1e-6,  # Relaxed from 1e-8
                    "reltol": 1e-6,  # Add relative tolerance
                }
                prob.solve(solver=cp.ECOS, **solver_options)

            solve_time = (datetime.now() - start_time).total_seconds()

            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                result = self._extract_solution(
                    problem,
                    heating_vars,
                    battery_first_mode,
                    ac_charge_mode,
                    export_enabled,
                    battery_power,
                    grid_import,
                    grid_export,
                    temp_vars,
                    prob.value,
                    solve_time,
                )
                result.success = True
                result.message = f"Optimal solution found ({prob.status})"

            else:
                self.logger.warning(f"Optimization failed with status: {prob.status}")
                result = self._get_fallback_solution(problem, solve_time)
                result.success = False
                result.message = f"Optimization failed: {prob.status}"

            return result

        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            solve_time = (datetime.now() - start_time).total_seconds()
            result = self._get_fallback_solution(problem, solve_time)
            result.success = False
            result.message = f"Error: {str(e)}"
            return result

    def _build_objective(
        self, problem, heating_vars, battery_power, grid_import, grid_export,
        battery_first_mode, ac_charge_mode, export_enabled, n_steps
    ):
        """Build the multi-objective function."""

        dt_hours = problem.time_step_minutes / 60

        # 1. Energy cost objective
        energy_cost = 0
        for t in range(n_steps):
            if t < len(problem.price_forecast):
                price = problem.price_forecast.iloc[t]
                energy_cost += (
                    (grid_import[t] - 0.9 * grid_export[t]) * price * dt_hours
                )

        # 2. Self-consumption objective (maximize)
        self_consumption = 0
        for t in range(n_steps):
            if t < len(problem.pv_forecast):
                pv_power = problem.pv_forecast.iloc[t]
                self_consumption += cp.minimum(
                    pv_power, grid_import[t] + battery_power[t]
                )

        # 3. Comfort penalty (heating switching frequency)
        comfort_penalty = 0
        for room, heating_var in heating_vars.items():
            for t in range(n_steps - 1):
                comfort_penalty += cp.abs(heating_var[t + 1] - heating_var[t])

        # 4. Growatt mode switching penalties
        mode_switching_penalty = 0
        for t in range(n_steps - 1):
            # Penalize frequent mode switches (causes wear on relays)
            mode_switching_penalty += cp.abs(battery_first_mode[t + 1] - battery_first_mode[t])
            mode_switching_penalty += cp.abs(ac_charge_mode[t + 1] - ac_charge_mode[t])
            mode_switching_penalty += cp.abs(export_enabled[t + 1] - export_enabled[t])

        # 5. Price-based mode incentives
        mode_price_incentive = 0
        for t in range(n_steps):
            if t < len(problem.price_forecast):
                price = problem.price_forecast.iloc[t]
                # Incentivize AC charging when prices are low
                if price < 50:  # Low price threshold (EUR/MWh)
                    mode_price_incentive -= 10 * ac_charge_mode[t]  # Reward AC charging
                # Incentivize export when prices are high  
                if price > 100:  # High price threshold (EUR/MWh)
                    mode_price_incentive -= 20 * export_enabled[t]  # Reward export

        # Combine objectives
        objective = (
            problem.cost_weight * energy_cost
            - problem.self_consumption_weight * self_consumption * 0.1
            + problem.comfort_weight * comfort_penalty * 10  # Heating switching penalty
            + 0.05 * mode_switching_penalty  # Growatt mode switching penalty
            + mode_price_incentive  # Price-based mode incentives
        )

        return objective

    def _power_balance_constraints(
        self, problem, heating_vars, battery_power, grid_import, grid_export, n_steps
    ):
        """Power balance constraints: generation = consumption."""

        constraints = []
        dt_hours = problem.time_step_minutes / 60

        for t in range(n_steps):
            # Generation side
            if t < len(problem.pv_forecast):
                pv_power = problem.pv_forecast.iloc[t]
            else:
                pv_power = 0

            generation = (
                pv_power + grid_import[t] - battery_power[t]
            )  # Negative battery_power = discharge

            # Consumption side
            base_load = (
                problem.load_forecast.iloc[t]
                if t < len(problem.load_forecast)
                else 1000
            )

            heating_load = 0
            for room, heating_var in heating_vars.items():
                if room in self.rooms:
                    # Handle both dict and float room configurations
                    if isinstance(self.rooms[room], dict):
                        room_power = self.rooms[room].get("power_kw", 1.0) * 1000
                    else:
                        room_power = float(self.rooms[room]) * 1000  # Assume kW, convert to W
                    heating_load += heating_var[t] * room_power

            consumption = base_load + heating_load + grid_export[t]

            # Power balance
            constraints.append(generation == consumption)

        return constraints

    def _battery_constraints(self, problem, battery_power, n_steps):
        """Battery operation constraints."""

        constraints = []
        dt_hours = problem.time_step_minutes / 60
        capacity_wh = problem.battery_capacity_kwh * 1000
        max_power_w = problem.battery_max_power_kw * 1000

        # Power limits
        for t in range(n_steps):
            constraints.append(battery_power[t] >= -max_power_w)  # Max discharge
            constraints.append(battery_power[t] <= max_power_w)  # Max charge

        # SOC tracking (simplified - assume linear efficiency)
        soc = problem.initial_battery_soc
        for t in range(n_steps):
            # SOC update
            energy_change = battery_power[t] * dt_hours
            soc_change = energy_change / capacity_wh
            soc = soc + soc_change

            # SOC limits
            constraints.append(soc >= 0.1)  # 10% minimum
            constraints.append(soc <= 0.9)  # 90% maximum

        return constraints

    def _growatt_constraints(
        self, problem, battery_first_mode, ac_charge_mode, export_enabled,
        battery_power, grid_import, grid_export, n_steps
    ):
        """Growatt inverter operational constraints."""
        
        constraints = []
        dt_hours = problem.time_step_minutes / 60
        
        # Battery specifications (from Growatt system)
        battery_capacity_kwh = problem.battery_capacity_kwh
        max_charge_power_kw = 5.0  # Growatt AC charging limit
        max_discharge_power_kw = 5.0  # Growatt discharge limit
        
        # SOC tracking with Growatt-specific behavior
        soc = problem.initial_battery_soc
        for t in range(n_steps):
            # Battery power depends on operating modes
            # AC charge mode: force battery charging from grid
            # Battery first mode: prioritize battery usage over grid
            # Export enabled: allow battery discharge for export
            
            # AC charging constraint: when AC charge is on, battery charges at max rate
            ac_charge_power = ac_charge_mode[t] * max_charge_power_kw * 1000  # Convert to W
            
            # Battery first mode affects power flow but doesn't directly set battery power
            # Export mode allows discharge when price is high
            
            # Actual battery power constraints
            constraints.append(battery_power[t] >= -max_discharge_power_kw * 1000)  # Max discharge
            constraints.append(battery_power[t] <= max_charge_power_kw * 1000)     # Max charge
            
            # When AC charging is enabled, force battery to charge
            constraints.append(battery_power[t] >= ac_charge_power - (1 - ac_charge_mode[t]) * 1e6)
            
            # SOC evolution
            energy_change_wh = battery_power[t] * dt_hours
            soc_change = energy_change_wh / (battery_capacity_kwh * 1000)
            soc = soc + soc_change
            
            # SOC limits (Growatt protection limits)
            constraints.append(soc >= 0.1)  # 10% minimum
            constraints.append(soc <= 0.9)  # 90% maximum
        
        # Mode interaction constraints
        for t in range(n_steps):
            # Can't have AC charge and export simultaneously (safety)
            constraints.append(ac_charge_mode[t] + export_enabled[t] <= 1)
            
            # Battery first mode can coexist with AC charge (typical night charging)
            # No additional constraints needed for this combination
        
        # Grid export constraints
        max_export_kw = 10.0  # Growatt export limit
        for t in range(n_steps):
            # Export only when enabled
            constraints.append(grid_export[t] <= export_enabled[t] * max_export_kw * 1000)
            constraints.append(grid_export[t] >= 0)
            
            # Grid import constraints
            constraints.append(grid_import[t] >= 0)
            constraints.append(grid_import[t] <= 20000)  # 20kW grid connection limit
        
        return constraints

    def _thermal_constraints(
        self, problem, heating_vars, temp_vars, n_steps, comfort_violations=None
    ):
        """Thermal dynamics and comfort constraints with soft violations."""

        constraints = []
        dt_hours = problem.time_step_minutes / 60

        for room in self.rooms:
            if room not in temp_vars or room not in heating_vars:
                continue

            # Initial temperature
            initial_temp = problem.initial_temperatures.get(room, 20.0)
            constraints.append(temp_vars[room][0] == initial_temp)

            # Thermal dynamics (simplified first-order model)
            R = self.thermal_params.get(room, {}).get("R", 0.005)  # K/W
            C = self.thermal_params.get(room, {}).get("C", 1e7)  # J/K
            tau = R * C / 3600  # Time constant in hours

            # Handle both dict and float room configurations
            if isinstance(self.rooms[room], dict):
                room_power = self.rooms[room].get("power_kw", 1.0) * 1000  # W
            else:
                room_power = float(self.rooms[room]) * 1000  # Assume kW, convert to W

            for t in range(n_steps):
                # Outdoor temperature
                if (
                    t < len(problem.weather_forecast)
                    and "temperature_2m" in problem.weather_forecast.columns
                ):
                    T_out = problem.weather_forecast.iloc[t]["temperature_2m"]
                else:
                    T_out = 10.0  # Default

                # Temperature dynamics: T[t+1] = alpha * T[t] + (1-alpha) * (T_out + R * P_heat)
                alpha = np.exp(-dt_hours / tau) if tau > 0 else 0.9

                if t < n_steps - 1:  # Avoid index out of bounds
                    constraints.append(
                        temp_vars[room][t + 1]
                        == alpha * temp_vars[room][t]
                        + (1 - alpha) * (T_out + R * heating_vars[room][t] * room_power)
                    )

                # Comfort bounds with soft violations
                if room in problem.comfort_bounds:
                    T_min, T_max = problem.comfort_bounds[room]

                    if comfort_violations and room in comfort_violations:
                        # Soft constraints with violation variables
                        constraints.append(
                            temp_vars[room][t]
                            >= T_min - comfort_violations[room]["low"][t]
                        )
                        constraints.append(
                            temp_vars[room][t]
                            <= T_max + comfort_violations[room]["high"][t]
                        )
                    else:
                        # Hard constraints (original behavior)
                        constraints.append(temp_vars[room][t] >= T_min)
                        constraints.append(temp_vars[room][t] <= T_max)

        return constraints

    def _grid_constraints(self, grid_import, grid_export, n_steps):
        """Grid operation constraints."""

        constraints = []
        max_import = self.config.get("max_grid_import", 20000)  # W
        max_export = self.config.get("max_grid_export", 10000)  # W

        for t in range(n_steps):
            constraints.append(grid_import[t] >= 0)
            constraints.append(grid_import[t] <= max_import)
            constraints.append(grid_export[t] >= 0)
            constraints.append(grid_export[t] <= max_export)

        return constraints

    def _extract_solution(
        self,
        problem,
        heating_vars,
        battery_first_mode,
        ac_charge_mode,
        export_enabled,
        battery_power,
        grid_import,
        grid_export,
        temp_vars,
        objective_value,
        solve_time,
    ):
        """Extract solution from optimization variables."""

        n_steps = problem.horizon_hours * 60 // problem.time_step_minutes
        time_index = pd.date_range(
            start=problem.start_time,
            periods=n_steps,
            freq=f"{problem.time_step_minutes}min",
        )

        # Extract heating schedules
        heating_schedule = {}
        for room in self.rooms:
            if room in heating_vars:
                schedule = [
                    int(round(heating_vars[room][t].value)) for t in range(n_steps)
                ]
                heating_schedule[room] = pd.Series(schedule, index=time_index)

        # Extract Growatt control schedules
        battery_first_schedule = pd.Series(
            [int(round(battery_first_mode[t].value)) for t in range(n_steps)], index=time_index
        )
        ac_charge_schedule = pd.Series(
            [int(round(ac_charge_mode[t].value)) for t in range(n_steps)], index=time_index
        )
        export_schedule = pd.Series(
            [int(round(export_enabled[t].value)) for t in range(n_steps)], index=time_index
        )

        # Extract battery power schedule (actual power flow)
        battery_schedule = pd.Series(
            [battery_power[t].value for t in range(n_steps)], index=time_index
        )

        # Extract grid schedule (net import/export)
        grid_schedule = pd.Series(
            [grid_import[t].value - grid_export[t].value for t in range(n_steps)],
            index=time_index,
        )

        # Extract temperature forecasts
        temperature_forecast = {}
        for room in self.rooms:
            if room in temp_vars:
                temps = [temp_vars[room][t].value for t in range(n_steps)]
                temperature_forecast[room] = pd.Series(temps, index=time_index)

        # Calculate cost breakdown
        dt_hours = problem.time_step_minutes / 60
        energy_cost = sum(
            (grid_import[t].value - 0.9 * grid_export[t].value)
            * problem.price_forecast.iloc[t]
            * dt_hours
            for t in range(min(n_steps, len(problem.price_forecast)))
        )

        cost_breakdown = {"energy_cost": energy_cost, "total_cost": energy_cost}

        return OptimizationResult(
            success=True,
            objective_value=objective_value,
            heating_schedule=heating_schedule,
            battery_first_schedule=battery_first_schedule,
            ac_charge_schedule=ac_charge_schedule,
            export_schedule=export_schedule,
            battery_schedule=battery_schedule,
            grid_schedule=grid_schedule,
            temperature_forecast=temperature_forecast,
            cost_breakdown=cost_breakdown,
            solve_time_seconds=solve_time,
        )

    def _get_fallback_solution(self, problem, solve_time):
        """Generate a simple fallback solution if optimization fails."""

        n_steps = problem.horizon_hours * 60 // problem.time_step_minutes
        time_index = pd.date_range(
            start=problem.start_time,
            periods=n_steps,
            freq=f"{problem.time_step_minutes}min",
        )

        # Simple rule-based fallback
        heating_schedule = {}
        for room in self.rooms:
            # Heat if temperature is below setpoint
            initial_temp = problem.initial_temperatures.get(room, 20.0)
            setpoint = problem.comfort_bounds.get(room, (19.0, 23.0))[0] + 1.0
            heat_on = 1 if initial_temp < setpoint else 0
            heating_schedule[room] = pd.Series([heat_on] * n_steps, index=time_index)

        # Safe Growatt modes - no aggressive charging/exporting
        battery_first_schedule = pd.Series([0] * n_steps, index=time_index)  # Load-first mode
        ac_charge_schedule = pd.Series([0] * n_steps, index=time_index)      # No AC charging
        export_schedule = pd.Series([0] * n_steps, index=time_index)         # Export disabled
        
        # No battery activity
        battery_schedule = pd.Series([0.0] * n_steps, index=time_index)

        # Balance grid import
        grid_schedule = pd.Series([1000.0] * n_steps, index=time_index)  # 1kW base

        # Constant temperatures
        temperature_forecast = {}
        for room in self.rooms:
            temp = problem.initial_temperatures.get(room, 20.0)
            temperature_forecast[room] = pd.Series([temp] * n_steps, index=time_index)

        return OptimizationResult(
            success=False,
            objective_value=float("inf"),
            heating_schedule=heating_schedule,
            battery_first_schedule=battery_first_schedule,
            ac_charge_schedule=ac_charge_schedule,
            export_schedule=export_schedule,
            battery_schedule=battery_schedule,
            grid_schedule=grid_schedule,
            temperature_forecast=temperature_forecast,
            cost_breakdown={"energy_cost": 0.0, "total_cost": 0.0},
            solve_time_seconds=solve_time,
        )

    def _load_thermal_parameters(self):
        """Load thermal parameters for rooms (from previous analysis)."""

        # Default thermal parameters based on analysis results
        return {
            "obyvak": {"R": 0.003, "C": 8e6},  # Living room
            "kuchyne": {"R": 0.004, "C": 6e6},  # Kitchen
            "loznice": {"R": 0.005, "C": 5e6},  # Bedroom
            "pracovna": {"R": 0.006, "C": 4e6},  # Office
            "hosti": {"R": 0.007, "C": 4e6},  # Guest room
            "pokoj_1": {"R": 0.005, "C": 4e6},  # Room 1
            "pokoj_2": {"R": 0.005, "C": 4e6},  # Room 2
        }


def create_optimization_problem(
    start_time: datetime,
    horizon_hours: int = 24,
    pv_forecast: Optional[pd.Series] = None,
    load_forecast: Optional[pd.Series] = None,
    price_forecast: Optional[pd.Series] = None,
    weather_forecast: Optional[pd.DataFrame] = None,
    initial_battery_soc: float = 0.5,
    initial_temperatures: Optional[Dict[str, float]] = None,
) -> OptimizationProblem:
    """
    Create an optimization problem with sensible defaults.

    Args:
        start_time: Start time for optimization
        horizon_hours: Optimization horizon in hours
        pv_forecast: PV production forecast
        load_forecast: Load forecast
        price_forecast: Energy price forecast
        weather_forecast: Weather forecast
        initial_battery_soc: Initial battery state of charge
        initial_temperatures: Initial room temperatures

    Returns:
        Configured optimization problem
    """

    # Create time index
    time_index = pd.date_range(
        start=start_time, periods=horizon_hours * 4, freq="15min"  # 15-minute intervals
    )

    # Default forecasts if not provided
    if pv_forecast is None:
        # Simple sinusoidal PV forecast
        hours = np.array([(t.hour + t.minute / 60) for t in time_index])
        pv_forecast = pd.Series(
            np.maximum(0, 8000 * np.sin(np.pi * (hours - 6) / 12)), index=time_index
        )
    else:
        # Validate PV forecast
        pv_forecast = pd.Series(pv_forecast)
        pv_forecast = pv_forecast.clip(lower=0, upper=15000)  # Max 15kW system
        pv_forecast = pv_forecast.fillna(0)

    if load_forecast is None:
        # Constant base load
        load_forecast = pd.Series([1200.0] * len(time_index), index=time_index)

    if price_forecast is None:
        # Time-of-use pricing
        hours = np.array([t.hour for t in time_index])
        prices = np.where(
            (hours >= 7) & (hours <= 22),  # Day rate
            0.15,  # 15 cents/kWh
            0.08,  # 8 cents/kWh night rate
        )
        price_forecast = pd.Series(prices, index=time_index)
    else:
        # Validate price forecast (Czech market can see -10 to 40 CZK/kWh)
        price_forecast = pd.Series(price_forecast)
        price_forecast = price_forecast.clip(
            lower=-0.5, upper=2.0
        )  # -0.5 to 2.0 €/kWh (~-12 to 48 CZK/kWh at 24 CZK/EUR)
        price_forecast = price_forecast.fillna(
            0.15
        )  # Default 0.15 €/kWh (~3.6 CZK/kWh)

    if weather_forecast is None:
        # Simple weather forecast
        weather_forecast = pd.DataFrame(
            {
                "temperature_2m": [15.0] * len(time_index),
                "cloudcover": [50.0] * len(time_index),
            },
            index=time_index,
        )
    else:
        # Validate weather forecast
        if "temperature_2m" in weather_forecast.columns:
            weather_forecast["temperature_2m"] = weather_forecast[
                "temperature_2m"
            ].clip(lower=-30, upper=50)
            weather_forecast["temperature_2m"] = weather_forecast[
                "temperature_2m"
            ].fillna(15.0)

    if initial_temperatures is None:
        initial_temperatures = {
            "obyvak": 21.0,
            "kuchyne": 20.5,
            "loznice": 20.0,
            "pracovna": 19.5,
            "hosti": 19.0,
        }

    # Comfort bounds
    comfort_bounds = {room: (19.0, 23.0) for room in initial_temperatures}

    return OptimizationProblem(
        horizon_hours=horizon_hours,
        time_step_minutes=15,
        start_time=start_time,
        pv_forecast=pv_forecast,
        load_forecast=load_forecast,
        price_forecast=price_forecast,
        weather_forecast=weather_forecast,
        initial_battery_soc=initial_battery_soc,
        initial_temperatures=initial_temperatures,
        comfort_bounds=comfort_bounds,
    )
