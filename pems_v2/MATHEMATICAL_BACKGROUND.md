# Mathematical Background for PEMS v2

This document provides the theoretical foundations and mathematical models used in the Predictive Energy Management System (PEMS) v2.

## Table of Contents
1. [Thermal Dynamics](#thermal-dynamics)
   - [RC Model](#rc-model)
   - [Time Constant](#time-constant)
   - [Heat Transfer Equations](#heat-transfer-equations)
2. [Solar PV Modeling](#solar-pv-modeling)
   - [Irradiance Models](#irradiance-models)
   - [Temperature Effects](#temperature-effects)
3. [Load Forecasting](#load-forecasting)
   - [Time Series Analysis](#time-series-analysis)
   - [Machine Learning Models](#machine-learning-models)
4. [Optimization Theory](#optimization-theory)
   - [Linear Programming](#linear-programming)
   - [Dynamic Programming](#dynamic-programming)
5. [Battery Management](#battery-management)
   - [State of Charge](#state-of-charge)
   - [Degradation Models](#degradation-models)

---

## Thermal Dynamics

### RC Model

The thermal behavior of a room is modeled using an electrical RC (Resistance-Capacitance) circuit analogy:

```
Q̇ = (T_outdoor - T_room) / R + P_heating
C × dT_room/dt = Q̇
```

Where:
- **Q̇**: Heat flow rate (W)
- **R**: Thermal resistance (K/W)
- **C**: Thermal capacitance (J/K)
- **T_room**: Room temperature (°C)
- **T_outdoor**: Outdoor temperature (°C)
- **P_heating**: Heating power input (W)

### Time Constant

The time constant (τ) is a fundamental parameter describing how quickly a room loses heat:

#### Definition
```
τ = R × C
```

The time constant represents the time it takes for the temperature difference to decay to **1/e (≈37%)** of its initial value.

#### Temperature Decay Equation
After heating stops, the room temperature follows an exponential decay:

```
ΔT(t) = ΔT₀ × e^(-t/τ)
```

Where:
- **ΔT(t)**: Temperature difference at time t (°C)
- **ΔT₀**: Initial temperature difference (°C)
- **τ**: Time constant (hours)
- **e**: Euler's number (≈2.718)

#### Physical Interpretation
- **Small τ (5-10 hours)**: Poor insulation, rapid heat loss
- **Medium τ (20-40 hours)**: Average insulation
- **Large τ (50-100 hours)**: Excellent insulation, slow heat loss

#### Practical Example
For a room with τ = 20 hours:
- After 20 hours: 37% of initial temperature difference remains
- After 40 hours: 14% remains (e^-2)
- After 60 hours: 5% remains (e^-3)

### Heat Transfer Equations

#### Conduction
Heat flow through walls, windows, and other building elements:

```
Q̇_cond = U × A × (T_out - T_in)
```

Where:
- **U**: Overall heat transfer coefficient (W/m²K)
- **A**: Surface area (m²)

#### Convection
Heat transfer due to air movement:

```
Q̇_conv = h × A × (T_surface - T_air)
```

Where:
- **h**: Convection coefficient (W/m²K)

#### Radiation
Solar gains through windows:

```
Q̇_solar = A_window × SHGC × I_solar × cos(θ)
```

Where:
- **SHGC**: Solar Heat Gain Coefficient
- **I_solar**: Solar irradiance (W/m²)
- **θ**: Angle of incidence

### Room Coupling

Real buildings have thermal coupling between adjacent rooms. The multi-zone RC model accounts for these interactions:

#### Multi-Zone RC Model

For a building with n rooms, the thermal network becomes:

```
C_i × dT_i/dt = Σ(T_j - T_i)/R_ij + (T_out - T_i)/R_i + P_heating_i + Q̇_solar_i
```

Where:
- **i, j**: Room indices
- **R_ij**: Thermal resistance between rooms i and j (K/W)
- **R_i**: Thermal resistance from room i to outside (K/W)

#### Coupling Matrix

The system can be written in matrix form:

```
C × dT/dt = -G × T + P + Q_external
```

Where:
- **C**: Diagonal capacitance matrix
- **G**: Conductance matrix (G_ij = 1/R_ij)
- **T**: Temperature vector
- **P**: Heating power vector
- **Q_external**: External heat gains

#### Steady-State Solution

At equilibrium (dT/dt = 0):

```
T_steady = G^(-1) × (P + Q_external)
```

#### Thermal Coupling Coefficient

The strength of coupling between rooms i and j:

```
k_ij = 1/(R_ij × √(C_i × C_j))
```

Values:
- **k > 0.1**: Strong coupling (open doors, thin walls)
- **0.01 < k < 0.1**: Moderate coupling (normal walls)
- **k < 0.01**: Weak coupling (insulated walls, different floors)

#### Dynamic Response

The coupled system has multiple time constants (eigenvalues of G×C^(-1)):

```
τ_mode = -1/λ_mode
```

- **Fast modes**: Individual room responses
- **Slow modes**: Whole-building thermal response

#### Identification Strategy

1. **Individual Analysis**: Estimate single-room RC parameters
2. **Cross-Correlation**: Identify coupled rooms from temperature correlations
3. **Multi-Zone Fitting**: Simultaneously fit coupled RC network
4. **Validation**: Compare predicted vs. actual inter-room heat flows

---

## Solar PV Modeling

### Irradiance Models

#### Clear Sky Model
The theoretical maximum solar irradiance:

```
I_clear = I_0 × (1 + 0.033 × cos(360×n/365)) × cos(θ_z)
```

Where:
- **I_0**: Solar constant (1367 W/m²)
- **n**: Day of year
- **θ_z**: Solar zenith angle

#### Diffuse Irradiance
For cloudy conditions:

```
I_diffuse = I_global × (1 - k_t)^1.5
```

Where:
- **k_t**: Clearness index (0-1)

### Temperature Effects

PV panel efficiency decreases with temperature:

```
P_actual = P_STC × [1 + α(T_cell - T_STC)]
```

Where:
- **P_STC**: Power at Standard Test Conditions
- **α**: Temperature coefficient (typically -0.4%/°C)
- **T_cell**: Cell temperature (°C)
- **T_STC**: Standard temperature (25°C)

Cell temperature estimation:

```
T_cell = T_ambient + (NOCT - 20)/800 × I_solar
```

Where:
- **NOCT**: Nominal Operating Cell Temperature (typically 45°C)

---

## Load Forecasting

### Time Series Analysis

#### ARIMA Model
AutoRegressive Integrated Moving Average:

```
y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + ε_t
```

Where:
- **y_t**: Load at time t
- **φᵢ**: Autoregressive parameters
- **θᵢ**: Moving average parameters
- **ε_t**: Error term

#### Fourier Series
For capturing daily and weekly patterns:

```
L(t) = a₀ + Σ[aₙcos(2πnt/T) + bₙsin(2πnt/T)]
```

Where:
- **T**: Period (24 hours or 168 hours)
- **aₙ, bₙ**: Fourier coefficients

### Machine Learning Models

#### LSTM Networks
Long Short-Term Memory networks capture long-term dependencies:

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t × C_{t-1} + i_t × C̃_t  # Cell state
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t × tanh(C_t)  # Hidden state
```

---

## Optimization Theory

### Linear Programming

The energy optimization problem:

```
Minimize: Σ(c_t × P_grid_t × Δt)

Subject to:
- P_load_t = P_grid_t + P_battery_t + P_pv_t
- P_battery_min ≤ P_battery_t ≤ P_battery_max
- SOC_min ≤ SOC_t ≤ SOC_max
- SOC_{t+1} = SOC_t + η × P_battery_t × Δt / E_capacity
```

Where:
- **c_t**: Electricity price at time t
- **P_grid_t**: Grid power
- **P_battery_t**: Battery power (positive = discharge)
- **η**: Battery efficiency

### Dynamic Programming

Bellman equation for optimal control:

```
V*(s_t) = min_a [c(s_t, a_t) + γ × V*(s_{t+1})]
```

Where:
- **V*(s_t)**: Optimal value function
- **s_t**: System state at time t
- **a_t**: Action (control decision)
- **γ**: Discount factor

---

## Battery Management

### State of Charge

#### Coulomb Counting
Basic SOC estimation:

```
SOC(t) = SOC(0) + (1/C_nominal) × ∫I(τ)dτ
```

Where:
- **C_nominal**: Nominal capacity (Ah)
- **I(τ)**: Current (positive for charging)

#### Voltage-Based SOC
Using Open Circuit Voltage (OCV):

```
SOC = f(V_oc)  # Lookup table or polynomial fit
```

### Degradation Models

#### Cycle Life
Battery capacity fade with cycling:

```
C_fade = k_1 × N^0.5 + k_2 × N
```

Where:
- **N**: Number of equivalent full cycles
- **k_1, k_2**: Degradation coefficients

#### Calendar Aging
Capacity loss over time:

```
C_loss = A × exp(-E_a / (R × T)) × t^z
```

Where:
- **E_a**: Activation energy
- **R**: Gas constant
- **T**: Temperature (K)
- **z**: Power law exponent (typically 0.5)

---

## Statistical Metrics

### Error Metrics

#### Mean Absolute Percentage Error (MAPE)
```
MAPE = (100/n) × Σ|y_actual - y_predicted|/y_actual
```

#### Root Mean Square Error (RMSE)
```
RMSE = √[(1/n) × Σ(y_actual - y_predicted)²]
```

### Goodness of Fit

#### R-squared (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```

Where:
- **SS_res**: Σ(y_actual - y_predicted)²
- **SS_tot**: Σ(y_actual - ȳ)²

---

## Implementation Notes

### Numerical Stability

1. **Exponential Decay**: For large time constants, use log-space calculations:
   ```python
   log_decay = -t / tau
   if log_decay > -20:  # Avoid underflow
       decay = np.exp(log_decay)
   else:
       decay = 0
   ```

2. **Matrix Operations**: Use condition number checks for RC estimation:
   ```python
   if np.linalg.cond(A) > 1e10:
       # Matrix is ill-conditioned, use regularization
       A_reg = A + lambda * np.eye(n)
   ```

### Convergence Criteria

For iterative algorithms:
```
|x_{k+1} - x_k| / |x_k| < ε_rel  OR  |x_{k+1} - x_k| < ε_abs
```

Typical values:
- **ε_rel**: 1e-3 (0.1% relative change)
- **ε_abs**: 1e-6 (absolute tolerance)

