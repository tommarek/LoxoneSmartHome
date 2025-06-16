# Mathematical Background for PEMS v2

This document provides the comprehensive theoretical foundations and mathematical models used in the Predictive Energy Management System (PEMS) v2.

## Table of Contents
1. [Thermal Dynamics](#thermal-dynamics)
   - [RC Model](#rc-model)
   - [Heating Cycle Analysis](#heating-cycle-analysis)
   - [Multi-Zone Coupling](#multi-zone-coupling)
   - [Parameter Estimation](#parameter-estimation)
2. [Solar PV Modeling](#solar-pv-modeling)
   - [Irradiance Models](#irradiance-models)
   - [Temperature Effects](#temperature-effects)
   - [Power Output Prediction](#power-output-prediction)
3. [Load Forecasting](#load-forecasting)
   - [Time Series Analysis](#time-series-analysis)
   - [Machine Learning Models](#machine-learning-models)
   - [Feature Engineering](#feature-engineering)
4. [Optimization Theory](#optimization-theory)
   - [Model Predictive Control](#model-predictive-control)
   - [Mixed-Integer Programming](#mixed-integer-programming)
   - [Multi-Objective Optimization](#multi-objective-optimization)
5. [Battery Management](#battery-management)
   - [State of Charge](#state-of-charge)
   - [Power Flow Constraints](#power-flow-constraints)
   - [Degradation Models](#degradation-models)
6. [Control Theory](#control-theory)
   - [State-Space Representation](#state-space-representation)
   - [Feedback Control](#feedback-control)
   - [Constraint Handling](#constraint-handling)

---

## Thermal Dynamics

### RC Model

The thermal behavior of a room is modeled using an electrical RC (Resistance-Capacitance) circuit analogy. This first-order model captures the essential dynamics while remaining computationally tractable for real-time control.

#### Fundamental Equation

The heat balance equation for a single room:

```
C × dT_room/dt = (T_outdoor - T_room)/R + P_heating + Q_solar + Q_internal
```

Where:
- **C**: Thermal capacitance (J/K) - ability to store thermal energy
- **R**: Thermal resistance (K/W) - resistance to heat flow
- **T_room**: Room temperature (°C)
- **T_outdoor**: Outdoor temperature (°C)
- **P_heating**: Heating power input (W)
- **Q_solar**: Solar heat gains (W)
- **Q_internal**: Internal heat gains from occupants and equipment (W)

#### Time Constant

The time constant τ = R × C characterizes the thermal response:

```
τ = R × C
```

Physical interpretation:
- **τ < 10 hours**: Poor insulation, rapid temperature changes
- **τ = 20-40 hours**: Average residential building
- **τ > 50 hours**: Excellent insulation, slow thermal response

#### Solution Forms

**Heating Phase** (constant power input):
```
T(t) = T_outdoor + R × P_heating × (1 - e^(-t/τ)) + (T_initial - T_outdoor) × e^(-t/τ)
```

**Cooling Phase** (no heating):
```
T(t) = T_outdoor + (T_initial - T_outdoor) × e^(-t/τ)
```

### Heating Cycle Analysis

PEMS v2 uses individual heating cycles as controlled experiments to estimate RC parameters accurately.

#### Cycle Detection

A heating cycle consists of:
1. **Heating Phase**: Relay turns ON, temperature rises
2. **Cooling Phase**: Relay turns OFF, temperature decays

Detection algorithm:
```python
heating_changes = df['heating_on'].diff()
heating_starts = df[heating_changes == 1].index
heating_ends = df[heating_changes == -1].index
```

Filtering criteria:
- Minimum duration: 10 minutes (avoid transients)
- Maximum duration: 4 hours (avoid multi-cycle events)
- Minimum temperature rise: 0.5°C (ensure measurable response)

#### Decay Analysis

Post-heating temperature decay follows exponential behavior:

```
T(t) = T_outdoor + (T_peak - T_outdoor) × e^(-t/τ)
```

Linearized form for regression:
```
ln[(T(t) - T_outdoor)/(T_peak - T_outdoor)] = -t/τ
```

Robust fitting procedure:
1. Extract decay period (heating OFF until return to baseline)
2. Remove outliers using Median Absolute Deviation
3. Apply weighted least squares (recent points weighted higher)
4. Validate with R² > 0.7 threshold

#### Rise Analysis

During heating, temperature rise rate reveals thermal capacitance:

```
dT/dt = P_heating/C - (T_room - T_outdoor)/(R × C)
```

At heating start (T_room ≈ T_outdoor):
```
(dT/dt)_initial ≈ P_heating/C
```

Therefore:
```
C = P_heating / (dT/dt)_initial
```

Implementation:
1. Extract first 10 minutes of heating (after 2-minute lag)
2. Linear regression on temperature vs. time
3. Slope gives dT/dt for capacitance calculation

### Multi-Zone Coupling

Real buildings exhibit thermal coupling between adjacent rooms, requiring a multi-zone model.

#### Coupled RC Network

For n rooms, the system becomes:

```
C_i × dT_i/dt = Σ_j[(T_j - T_i)/R_ij] + (T_out - T_i)/R_i,out + P_i + Q_i
```

Matrix form:
```
C × dT/dt = -G × T + B × u + d
```

Where:
- **C**: Diagonal capacitance matrix (n×n)
- **G**: Conductance matrix, G_ij = 1/R_ij
- **T**: Temperature vector (n×1)
- **B**: Input matrix for heating powers
- **u**: Control input vector (heating powers)
- **d**: Disturbance vector (outdoor temp influence + solar gains)

#### Conductance Matrix Structure

```
G = [
  g_11  -g_12  -g_13  ...
  -g_21  g_22  -g_23  ...
  -g_31  -g_32  g_33  ...
  ...
]
```

Where:
- Diagonal: g_ii = Σ_j(1/R_ij) + 1/R_i,out (total conductance)
- Off-diagonal: g_ij = -1/R_ij (coupling conductance)

#### Eigenvalue Analysis

System eigenvalues determine thermal modes:

```
det(G - λC) = 0
```

Time constants of thermal modes:
```
τ_k = 1/λ_k
```

Typical mode interpretation:
- **Fast modes** (τ < 5h): Individual room responses
- **Medium modes** (τ = 10-30h): Floor-level dynamics
- **Slow mode** (τ > 50h): Whole-building response

### Parameter Estimation

#### Physics-Based Constraints

RC parameters must satisfy physical bounds:

```python
R_MIN = 0.008  # K/W (very poor insulation)
R_MAX = 0.5    # K/W (excellent insulation)
C_MIN = 2e6    # J/K (light construction)
C_MAX = 100e6  # J/K (heavy construction)
TAU_MIN = 3    # hours (minimum time constant)
TAU_MAX = 350  # hours (maximum time constant)
```

#### Confidence Scoring

Parameter confidence based on:

```
confidence = w_1 × (n_cycles/n_target) + w_2 × avg_R² + w_3 × night_ratio + w_4 × quality_score
```

Where:
- **n_cycles**: Number of valid heating cycles analyzed
- **avg_R²**: Average goodness of fit
- **night_ratio**: Fraction of nighttime cycles (no solar interference)
- **quality_score**: Data quality metrics (no stuck sensors, stable outdoor temp)

Weights: w_1=0.4, w_2=0.3, w_3=0.15, w_4=0.15

---

## Solar PV Modeling

### Irradiance Models

#### Solar Position

Solar angles calculation:

```
δ = 23.45° × sin(360° × (284 + n)/365)  # Declination
h = 15° × (TST - 12)                     # Hour angle
α = arcsin(sin(δ)sin(φ) + cos(δ)cos(φ)cos(h))  # Elevation
```

Where:
- **δ**: Solar declination angle
- **n**: Day of year
- **h**: Hour angle
- **TST**: True Solar Time
- **φ**: Latitude
- **α**: Solar elevation angle

#### Clear Sky Irradiance

Direct normal irradiance:

```
I_DN = I_0 × τ_atm^(1/sin(α))
```

Where:
- **I_0**: Extraterrestrial radiation (1367 W/m²)
- **τ_atm**: Atmospheric transmittance (0.75 typical)

Global horizontal irradiance:

```
I_GHI = I_DN × sin(α) + I_diffuse
```

#### Cloud Cover Effects

Clearness index model:

```
k_t = I_measured / I_clear_sky
```

Diffuse fraction correlation (Erbs et al.):
```
k_d = {
  1.0 - 0.09×k_t                    for k_t ≤ 0.22
  0.951 - 0.1604×k_t + 4.388×k_t²   for 0.22 < k_t ≤ 0.8
  0.165                              for k_t > 0.8
}
```

### Temperature Effects

#### Cell Temperature Model

Energy balance on PV module:

```
T_cell = T_ambient + (NOCT - 20°C) × (I_POA/800) × (1 - η_ref/τα)
```

Where:
- **NOCT**: Nominal Operating Cell Temperature (45°C typical)
- **I_POA**: Plane of Array irradiance (W/m²)
- **η_ref**: Reference efficiency
- **τα**: Transmittance-absorptance product (0.9 typical)

#### Power Temperature Correction

```
P = P_STC × (I_POA/1000) × [1 + γ(T_cell - 25°C)]
```

Where:
- **P_STC**: Rated power at Standard Test Conditions
- **γ**: Power temperature coefficient (-0.4%/°C typical)

### Power Output Prediction

#### Physical Model

DC power output:

```
P_DC = η_module × A × I_POA × [1 + γ(T_cell - 25°C)]
```

AC power output (including inverter):

```
P_AC = P_DC × η_inverter(P_DC/P_rated)
```

Inverter efficiency curve:

```
η_inverter = (P_DC/P_rated) / (k_0 + k_1×(P_DC/P_rated) + k_2×(P_DC/P_rated)²)
```

#### Machine Learning Enhancement

Hybrid approach combining physics and ML:

```
P_predicted = w_physics × P_physical + w_ML × P_ML
```

Where typically w_physics = 0.6, w_ML = 0.4

ML features include:
- Weather forecast variables
- Historical patterns
- Seasonal indicators
- Cloud motion vectors

---

## Load Forecasting

### Time Series Analysis

#### Components Decomposition

Load time series decomposition:

```
L(t) = T(t) + S(t) + R(t)
```

Where:
- **T(t)**: Trend component
- **S(t)**: Seasonal component (daily, weekly patterns)
- **R(t)**: Residual (random) component

#### Fourier Analysis

Seasonal patterns captured via Fourier series:

```
S(t) = Σ_k[a_k × cos(2πkt/T) + b_k × sin(2πkt/T)]
```

Key periods:
- T = 24 hours (daily cycle)
- T = 168 hours (weekly cycle)
- T = 8760 hours (annual cycle)

### Machine Learning Models

#### Random Forest Regression

Ensemble prediction:

```
L_pred = (1/N) × Σ_i Tree_i(features)
```

Feature importance calculated via:
```
Importance_j = Σ_nodes (w × Δimpurity × I[feature_j used])
```

#### LSTM Architecture

State equations for load prediction:

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)        # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)        # Input gate
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)    # Candidate values
C_t = f_t × C_{t-1} + i_t × C̃_t           # Cell state
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)        # Output gate
h_t = o_t × tanh(C_t)                      # Hidden state
L_t = W_y × h_t + b_y                      # Output (load prediction)
```

### Feature Engineering

#### Temporal Features
- Hour of day (sine/cosine encoding)
- Day of week (one-hot encoding)
- Month of year
- Holiday indicators
- Business day flag

#### Weather Features
- Temperature (current and lagged)
- Temperature squared (for heating/cooling)
- Humidity
- Wind speed
- Solar radiation

#### Historical Features
- Lagged load values (t-1h, t-24h, t-168h)
- Rolling averages (6h, 24h)
- Same hour previous week
- Load differences (capturing trends)

---

## Optimization Theory

### Model Predictive Control

PEMS v2 implements receding horizon control with the following formulation:

#### Objective Function

Minimize total cost over prediction horizon H:

```
J = Σ_{t=0}^{H-1} [c_t × P_grid_t × Δt + ρ × V_comfort_t²]
```

Where:
- **c_t**: Time-of-use electricity price (CZK/kWh)
- **P_grid_t**: Grid power import (kW)
- **ρ**: Comfort violation penalty (1000 CZK/°C²)
- **V_comfort_t**: Temperature deviation from comfort bounds

#### System Dynamics

Discrete-time thermal dynamics:

```
T_{t+1} = A × T_t + B_u × u_t + B_d × d_t
```

Where:
- **A**: State transition matrix (discretized from continuous RC model)
- **B_u**: Control input matrix
- **B_d**: Disturbance input matrix
- **u_t**: Heating control vector (binary for each room)
- **d_t**: Disturbances (outdoor temp, solar gains)

Discretization using zero-order hold:

```
A = e^(-G×C^(-1)×Δt)
B_u = C^(-1) × G^(-1) × (I - A) × P_heat
```

### Mixed-Integer Programming

#### Binary Heating Controls

Room heating as binary variables:

```
u_i,t ∈ {0, 1}  ∀i ∈ rooms, ∀t ∈ horizon
```

Minimum on/off time constraints:

```
u_i,t - u_i,t-1 ≤ u_i,τ  ∀τ ∈ [t+1, t+t_min_on]    # Minimum on time
u_i,t-1 - u_i,t ≤ 1-u_i,τ  ∀τ ∈ [t+1, t+t_min_off] # Minimum off time
```

#### Linearization Techniques

Comfort violation (absolute value) linearization:

```
V_comfort_t ≥ T_t - T_max
V_comfort_t ≥ T_min - T_t
V_comfort_t ≥ 0
```

Battery power (bidirectional) decomposition:

```
P_battery_t = P_charge_t - P_discharge_t
P_charge_t ≥ 0
P_discharge_t ≥ 0
P_charge_t ≤ M × δ_charge_t        # M = large number
P_discharge_t ≤ M × (1-δ_charge_t)  # δ binary variable
```

### Multi-Objective Optimization

#### Weighted Sum Method

Combined objectives:

```
J_total = w_cost × J_cost + w_comfort × J_comfort + w_self × J_self_consumption
```

Where:
- **J_cost**: Energy cost objective
- **J_comfort**: Thermal comfort objective
- **J_self**: PV self-consumption objective

#### Pareto Optimization

ε-constraint method for Pareto front:

```
Minimize: J_cost
Subject to:
  J_comfort ≤ ε_comfort
  J_self ≥ ε_self
  All other constraints...
```

---

## Battery Management

### State of Charge

#### Dynamic Model

SOC evolution with efficiency:

```
SOC_{t+1} = SOC_t + (η_charge × P_charge_t - P_discharge_t/η_discharge) × Δt / E_capacity
```

Where:
- **η_charge**: Charging efficiency (0.95 typical)
- **η_discharge**: Discharging efficiency (0.95 typical)
- **E_capacity**: Battery capacity (kWh)

#### Constraints

Operational limits:

```
SOC_min ≤ SOC_t ≤ SOC_max           # Typically 0.1 ≤ SOC ≤ 0.9
P_charge_t ≤ P_charge_max            # Maximum charge rate
P_discharge_t ≤ P_discharge_max      # Maximum discharge rate
P_charge_t × P_discharge_t = 0       # Cannot charge and discharge simultaneously
```

### Power Flow Constraints

#### Energy Balance

System power balance at each timestep:

```
P_load_t = P_grid_t + P_pv_t + P_discharge_t - P_charge_t
```

#### Grid Constraints

Export limitations:

```
P_grid_t ≥ -P_export_max  # Negative = export
```

Power factor requirements:

```
|Q_t| ≤ P_t × tan(acos(PF_min))
```

### Degradation Models

#### Cycle Aging

Rainflow counting for equivalent cycles:

```
L_cycle = Σ_i (N_i × DOD_i^k)
```

Where:
- **N_i**: Number of cycles at depth i
- **DOD_i**: Depth of discharge
- **k**: Stress factor (2.0 typical)

Capacity fade:

```
C_fade_cycle = α × L_cycle^β
```

#### Calendar Aging

Time-based degradation:

```
C_fade_calendar = γ × (t/t_ref)^0.5 × exp((T-T_ref)/T_scale)
```

Total capacity:

```
C_remaining = C_nominal × (1 - C_fade_cycle - C_fade_calendar)
```

---

## Control Theory

### State-Space Representation

#### Continuous-Time System

Full state-space model:

```
dx/dt = A_c × x + B_c × u + E_c × w
y = C_c × x + D_c × u
```

Where:
- **x**: State vector [T_room_1, ..., T_room_n, SOC]ᵀ
- **u**: Control vector [P_heat_1, ..., P_heat_n, P_battery]ᵀ
- **w**: Disturbance vector [T_outdoor, I_solar, P_load]ᵀ
- **y**: Output vector (measured temperatures)

#### Discrete-Time Conversion

Using zero-order hold:

```
x_{k+1} = A_d × x_k + B_d × u_k + E_d × w_k
y_k = C_d × x_k + D_d × u_k
```

Where:
```
A_d = e^(A_c × Ts)
B_d = A_c^(-1) × (A_d - I) × B_c
E_d = A_c^(-1) × (A_d - I) × E_c
```

### Feedback Control

#### State Estimation

Kalman filter for unmeasured states:

```
x̂_{k+1|k} = A_d × x̂_{k|k} + B_d × u_k
P_{k+1|k} = A_d × P_{k|k} × A_d^T + Q

K_k = P_{k+1|k} × C_d^T × (C_d × P_{k+1|k} × C_d^T + R)^(-1)
x̂_{k+1|k+1} = x̂_{k+1|k} + K_k × (y_{k+1} - C_d × x̂_{k+1|k})
P_{k+1|k+1} = (I - K_k × C_d) × P_{k+1|k}
```

Where:
- **Q**: Process noise covariance
- **R**: Measurement noise covariance
- **K**: Kalman gain

#### Disturbance Rejection

Feedforward compensation:

```
u_ff = -B_d^(-1) × E_d × w_predicted
```

### Constraint Handling

#### Soft Constraints

Barrier function method:

```
J_barrier = J_original - μ × Σ_i log(g_i(x))
```

Where g_i(x) > 0 represents inequality constraints.

#### Constraint Tightening

Robust MPC with uncertainty:

```
T_min + δ ≤ T_t ≤ T_max - δ
```

Where δ accounts for prediction uncertainty.

---

## Implementation Considerations

### Numerical Stability

#### Condition Number Monitoring

For matrix operations:

```python
κ(A) = ||A|| × ||A^(-1)||
if κ(A) > 10^10:
    # Apply regularization
    A_reg = A + λ × I
```

#### Scaling

Variable normalization:

```
x_normalized = (x - x_mean) / x_std
u_normalized = u / u_max
```

### Computational Efficiency

#### Sparse Matrix Exploitation

Thermal coupling matrix is typically sparse:

```python
from scipy.sparse import csr_matrix
G_sparse = csr_matrix(G)
# Use sparse linear algebra routines
```

#### Warm Starting

For sequential optimizations:

```
x_0^(k+1) = [x_1^(k), x_2^(k), ..., x_{H-1}^(k), x_H^(k)]
```

Shift previous solution as initial guess.

### Real-Time Guarantees

#### Optimization Timeouts

Ensure bounded execution time:

```python
solver.options['max_iter'] = 1000
solver.options['max_cpu_time'] = 1.5  # seconds
```

#### Fallback Strategies

If optimization fails:
1. Use previous solution
2. Apply rule-based control
3. Maintain safety constraints

---

## Validation and Testing

### Model Validation

#### Cross-Validation

Time series cross-validation:

```
Train: [0, t_1] → Test: [t_1, t_2]
Train: [0, t_2] → Test: [t_2, t_3]
...
```

#### Residual Analysis

Check for model adequacy:

```
Residuals: e_t = y_measured_t - y_predicted_t
```

Tests:
- Normality: Shapiro-Wilk test
- Autocorrelation: Ljung-Box test
- Heteroscedasticity: Breusch-Pagan test

### Performance Metrics

#### Prediction Accuracy

Mean Absolute Percentage Error:

```
MAPE = (100/n) × Σ|y_actual - y_predicted|/|y_actual|
```

Normalized Root Mean Square Error:

```
NRMSE = RMSE / (y_max - y_min)
```

#### Control Performance

Integrated Absolute Error:

```
IAE = ∫|setpoint - actual| dt
```

Total Variation:

```
TV = Σ|u_{t+1} - u_t|
```

### Robustness Testing

#### Monte Carlo Simulation

Parameter uncertainty:

```python
for i in range(N_simulations):
    R_i = R_nominal × (1 + σ_R × randn())
    C_i = C_nominal × (1 + σ_C × randn())
    simulate_system(R_i, C_i)
```

#### Worst-Case Analysis

Robust optimization formulation:

```
min_u max_w J(u, w)
```

Where w represents bounded uncertainties.

---

## References

1. **Thermal Modeling**: Madsen, H., & Holst, J. (1995). "Estimation of continuous-time models for the heat dynamics of a building." Energy and Buildings.

2. **PV Modeling**: Duffie, J. A., & Beckman, W. A. (2013). "Solar Engineering of Thermal Processes."

3. **MPC Theory**: Rawlings, J. B., & Mayne, D. Q. (2009). "Model Predictive Control: Theory and Design."

4. **Battery Management**: Plett, G. L. (2015). "Battery Management Systems, Volume I: Battery Modeling."

5. **Load Forecasting**: Hong, T., & Fan, S. (2016). "Probabilistic electric load forecasting: A tutorial review." International Journal of Forecasting.

---

This mathematical background provides the theoretical foundation for all algorithms implemented in PEMS v2. The models balance physical accuracy with computational efficiency, enabling real-time optimization for residential energy management.