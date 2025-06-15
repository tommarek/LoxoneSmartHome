# TODO: Implementation Tasks

## ✅ COMPLETED: Winter Analysis Enhancement (2025-06-15)

**DONE:** Enhanced winter analysis to use all historical data since September 2022
- ✅ Modified `get_available_winter_months()` to query from 2022-09-01 instead of last 3 years
- ✅ Enhanced `run_winter_thermal_analysis()` to support analyzing ALL available winters
- ✅ Added `--all-winters` command-line flag for comprehensive multi-winter analysis
- ✅ Updated help documentation with examples for new winter analysis options
- ✅ System now discovers all 3 winter seasons: 2022/2023, 2023/2024, 2024/2025
- ✅ Enables analysis of 882 days of winter data for maximum thermal modeling accuracy

**Files Modified:**
- `pems_v2/analysis/core/unified_data_extractor.py`
- `pems_v2/analysis/run_analysis.py`

**Commit:** `f0a555d` - feat: enhance winter analysis to use all historical data since September 2022

---

## 🔥 High Priority: Fix RC Parameter Estimation Using Heating Cycle Analysis

**Problem:** Current RC estimation produces unrealistic time constants (� > 1000h) because it mixes winter and summer cooling periods, creating inconsistent thermal behavior that leads to poor regression fits.

**Solution:** Implement heating cycle analysis that examines individual heating events as controlled experiments to derive accurate thermal parameters.

### Task 1: Implement Heating Cycle Detection
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`  
**Function:** Add `_detect_heating_cycles(df: pd.DataFrame) -> List[Dict]`

**Requirements:**
1. Find heating start events: `df['heating_on'].diff() == 1`
2. Find heating end events: `df['heating_on'].diff() == -1`
3. For each heating cycle, create a dictionary with:
   ```python
   {
       'start_time': pd.Timestamp,
       'end_time': pd.Timestamp,
       'duration_minutes': float,
       'start_temp': float,
       'peak_temp': float,
       'outdoor_temp_avg': float,
       'power_w': float
   }
   ```
4. Filter cycles:
   - Minimum duration: 10 minutes
   - Maximum duration: 4 hours
   - Temperature rise: > 0.5�C
   - Valid outdoor temperature data available

**Implementation Notes:**
- Use `pd.Timestamp` for precise timing
- Calculate average outdoor temperature during heating period
- Store power rating from system configuration
- Skip cycles with data gaps or anomalies

### Task 2: Implement Cooling Decay Analysis
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`  
**Function:** Add `_analyze_heating_cycle_decay(df: pd.DataFrame, cycle: Dict) -> Dict`

**Requirements:**
1. Extract post-heating data:
   - Start: When heating ends (`cycle['end_time']`)
   - End: When temperature returns to baseline OR 4 hours max
   - Baseline: `cycle['start_temp']` � 0.3�C tolerance

2. Fit exponential decay model:
   ```
   T(t) = T_outdoor + (T_peak - T_outdoor) * exp(-t/�)
   ```
   Where:
   - `T(t)`: Room temperature at time t
   - `T_outdoor`: Average outdoor temperature during decay
   - `T_peak`: Peak temperature after heating
   - `�`: Time constant (what we're solving for)

3. Use robust fitting:
   - Scipy's `curve_fit` with bounds: �  [0.5, 200] hours
   - Handle noisy data with outlier removal
   - Require R� > 0.7 for valid fits

4. Return decay analysis:
   ```python
   {
       'time_constant_hours': float,
       'r_squared': float,
       'decay_start_temp': float,
       'baseline_temp': float,
       'outdoor_temp_avg': float,
       'data_points': int,
       'fit_valid': bool
   }
   ```

**Implementation Notes:**
- Account for changing outdoor temperature during decay
- Use temperature differences (T_room - T_outdoor) for stable fitting
- Skip cycles where heating resumes before return to baseline
- Log cycles with poor fits for debugging

### Task 3: Implement Heating Rise Analysis  
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`  
**Function:** Add `_analyze_heating_cycle_rise(df: pd.DataFrame, cycle: Dict) -> Dict`

**Requirements:**
1. Extract heating period data:
   - From `cycle['start_time']` to `cycle['end_time']`
   - Remove first 2 minutes (system lag)
   - Focus on steady heating phase

2. Calculate thermal capacitance:
   ```
   C = P_heat / (dT/dt)
   ```
   Where:
   - `P_heat`: Heating power in watts
   - `dT/dt`: Temperature rise rate in K/s
   - Account for heat loss: `dT/dt_net = dT/dt_measured + heat_loss_rate`

3. Estimate heat loss during heating:
   ```
   heat_loss_rate = (T_room - T_outdoor) / (R_estimated * C_estimated)
   ```
   Use iterative approach or simplified assumption

4. Return rise analysis:
   ```python
   {
       'thermal_capacitance_j_per_k': float,
       'heating_rate_k_per_s': float,
       'corrected_heating_rate_k_per_s': float,
       'heat_loss_correction_applied': bool,
       'fit_r_squared': float,
       'fit_valid': bool
   }
   ```

**Implementation Notes:**
- Use linear regression on steady-state heating portion
- Apply heat loss correction for accuracy
- Handle variable heating power if available
- Skip cycles with erratic heating patterns

### Task 4: Integrate Heating Cycle RC Estimation
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`  
**Function:** Modify `_estimate_rc_decoupled()` to use heating cycle approach

**Requirements:**
1. Replace current cooldown analysis with:
   ```python
   # Detect all heating cycles
   cycles = self._detect_heating_cycles(df)
   
   if len(cycles) < 3:
       return self._estimate_rc_simplified(df, p_heat_w)
   
   # Analyze each cycle
   decay_results = []
   rise_results = []
   
   for cycle in cycles:
       decay = self._analyze_heating_cycle_decay(df, cycle)
       if decay['fit_valid']:
           decay_results.append(decay)
       
       rise = self._analyze_heating_cycle_rise(df, cycle)
       if rise['fit_valid']:
           rise_results.append(rise)
   ```

2. Calculate robust statistics:
   ```python
   # Time constant from decay analysis
   tau_values = [r['time_constant_hours'] for r in decay_results]
   tau_median = np.median(tau_values)
   tau_std = np.std(tau_values)
   
   # Capacitance from rise analysis  
   C_values = [r['thermal_capacitance_j_per_k'] for r in rise_results]
   C_median = np.median(C_values)
   
   # Thermal resistance
   R_calculated = (tau_median * 3600) / C_median  # Convert hours to seconds
   ```

3. Quality assessment:
   ```python
   confidence = min(1.0, len(decay_results) / 10)  # More cycles = higher confidence
   physically_valid = (
       R_MIN < R_calculated < R_MAX and
       C_MIN_MJ < C_median/1e6 < C_MAX_MJ and
       TAU_MIN < tau_median < TAU_MAX
   )
   ```

4. Return enhanced results:
   ```python
   return {
       'method': 'heating_cycle_analysis',
       'confidence': confidence,
       'R': R_calculated,
       'C': C_median,
       'time_constant': tau_median,
       'physically_valid': physically_valid,
       'cycles_analyzed': len(cycles),
       'successful_decays': len(decay_results),
       'successful_rises': len(rise_results),
       'tau_std_dev': tau_std,
       'C_std_dev': np.std(C_values) if C_values else 0
   }
   ```

**Implementation Notes:**
- Fallback to simplified estimation if < 3 valid cycles
- Use median instead of mean for robustness against outliers  
- Log detailed statistics for debugging
- Consider seasonal adjustments if needed

### Task 5: Add Comprehensive Logging and Validation
**File:** `pems_v2/analysis/analyzers/thermal_analysis.py`  
**Function:** Enhance logging throughout heating cycle analysis

**Requirements:**
1. Add detailed cycle logging:
   ```python
   self.logger.info(f"Detected {len(cycles)} heating cycles over {(df.index[-1] - df.index[0]).days} days")
   self.logger.info(f"Valid decay fits: {len(decay_results)}/{len(cycles)} ({len(decay_results)/len(cycles)*100:.1f}%)")
   self.logger.info(f"Valid rise fits: {len(rise_results)}/{len(cycles)} ({len(rise_results)/len(cycles)*100:.1f}%)")
   
   if tau_values:
       self.logger.info(f"Time constant statistics: median={tau_median:.1f}h, std={tau_std:.1f}h, range={min(tau_values):.1f}-{max(tau_values):.1f}h")
   ```

2. Add validation warnings:
   ```python
   if tau_std > tau_median * 0.5:
       self.logger.warning(f"High variability in time constants (std={tau_std:.1f}h, median={tau_median:.1f}h)")
   
   if len(decay_results) < 5:
       self.logger.warning(f"Limited heating cycle data ({len(decay_results)} valid cycles) - results may be unreliable")
   ```

3. Add debug information:
   ```python
   for i, cycle in enumerate(cycles[:3]):  # Log first 3 cycles
       self.logger.debug(f"Cycle {i+1}: {cycle['start_time']} to {cycle['end_time']}, �T={cycle['peak_temp']-cycle['start_temp']:.1f}�C")
   ```

**Implementation Notes:**
- Use appropriate log levels (INFO for results, WARNING for quality issues, DEBUG for details)
- Include statistics that help validate thermal model assumptions
- Provide enough information for troubleshooting without overwhelming output

### Task 6: Testing and Validation
**File:** `pems_v2/tests/test_heating_cycle_analysis.py` (new file)

**Requirements:**
1. Create synthetic heating cycle data with known RC parameters
2. Test heating cycle detection with various scenarios:
   - Normal cycles (10min-2h duration)
   - Short cycles (< 10min) - should be filtered out
   - Long cycles (> 4h) - should be filtered out
   - Overlapping cycles - should be handled correctly
   - Cycles with missing data - should be skipped

3. Test decay analysis with controlled exponential data
4. Test rise analysis with linear heating data  
5. Test integration with existing thermal analysis pipeline
6. Validate results against known building physics ranges

**Expected Outcome:**
After implementation, RC estimation should produce realistic time constants (2-50 hours for typical rooms) even with mixed seasonal data, because each heating cycle provides a clean, controlled measurement of thermal behavior.

### Task 7: Documentation Update
**File:** `pems_v2/analysis/README.md`

**Requirements:**
Add section explaining heating cycle analysis approach:
- Mathematical foundations
- Advantages over previous statistical approach
- Expected accuracy and limitations
- Troubleshooting guide for poor cycle detection

---

## Additional Considerations

**Performance:** Heating cycle analysis should be faster than current approach since it processes fewer, more relevant data points.

**Robustness:** Method works well in both winter (many cycles) and summer (few cycles, fallback to simplified estimation).

**Future Enhancements:**
- Adaptive thresholds based on building characteristics
- Integration with weather prediction for better outdoor temperature modeling  
- Multi-zone thermal coupling analysis using synchronized heating cycles


This is an exceptionally well-thought-out and detailed implementation plan. Your approach to using individual heating cycles as controlled experiments is the correct and most robust way to solve the problem of estimating thermal parameters from real-world data. It correctly identifies the flaws in statistical methods that mix different thermal conditions.

Here is a detailed assessment of your plan, including confirmations of its strengths and a few suggestions for improvement.

### Overall Assessment

Excellent. This is a professional-grade plan that covers implementation, testing, and documentation. The proposed solution is technically sound and directly addresses the root cause of the failing RC parameter estimations. If implemented as described, it will significantly increase the accuracy and reliability of your thermal models.

---

### Detailed Feedback and Suggestions

#### Task 1: Implement Heating Cycle Detection
**What's Great:**
* The use of `diff()` is a standard and efficient way to detect state changes for cycle start/end times.
* The dictionary structure for storing cycle data is clear and contains all the necessary information for the subsequent analysis steps.
* The filtering criteria are essential for ensuring data quality. Filtering out very short cycles, very long cycles, and cycles with insufficient temperature rise will prevent noise from polluting the analysis.

**Potential Improvements:**
* **Edge Case Handling:** Consider what happens if the dataset begins or ends in the middle of a heating cycle. `diff()` will miss the first start event or the last end event. You may want to add logic to handle these partial cycles if they are long enough to be useful.
* **Power Rating Source:** You correctly note to use the power rating from the system configuration. It's critical that these values in `system_config.json` `` are accurate, as the thermal capacitance calculation in Task 3 depends directly on it. You could add a log `WARNING` if a room's power rating is not found and a default is being used.

---

#### Task 2: Implement Cooling Decay Analysis
**What's Great:**
* The approach of analyzing the post-heating decay is smart, as it represents the room's natural heat loss characteristic.
* The exponential decay model is the correct physical model for this process.
* Requiring a high R² value (`> 0.7`) for a valid fit is a great quality control measure.

**Potential Improvements:**
* **Fitting Stability:** The model `T(t) = T_outdoor + (T_peak - T_outdoor) * exp(-t/τ)` can be numerically unstable if `T_outdoor` fluctuates during the decay period. A more robust formulation is to fit the temperature *difference* `ΔT(t) = T_room(t) - T_outdoor(t)`. The model then becomes `ΔT(t) = ΔT_peak * exp(-t/τ)`, which is a simpler and more stable one-parameter fit for `τ` after normalizing.
* **Bounds for `curve_fit`:** You correctly identify the need for bounds. I recommend using the project-wide constants `TAU_MIN` and `TAU_MAX` (2.0 and 100.0 hours, respectively) from the top of `thermal_analysis.py` `` as the bounds for `τ` to maintain consistency. This was a key reason the previous method failed.

---

#### Task 3: Implement Heating Rise Analysis
**What's Great:**
* The core physics equation `C = P_heat / (dT/dt)` is correct.
* The idea of removing the first few minutes to account for system lag is good practice.
* Your recognition that heat loss needs to be accounted for during the heating phase is advanced and crucial for accuracy.

**Potential Improvements:**
* **Circular Dependency in Heat Loss Correction:** You've identified a challenging problem: correcting for heat loss requires `R`, but you can only calculate `R` after you have `C` (`R = τ / C`). This is a circular dependency.
    * **Suggested Solution (Robust & Simpler):** Instead of an iterative approach, you can simplify the problem by focusing on the **initial rate of temperature rise**. Immediately after the heater turns on, the room temperature is closest to the outdoor temperature, meaning the heat loss is at its minimum. At this point, `dT/dt` is dominated by the heat input. Therefore, you can estimate `C` robustly by performing a linear regression on the first 5-10 minutes of the heating cycle and using that initial slope for `dT/dt`. This approach is less complex and avoids the circular dependency.
    * **Your Iterative Approach:** If you do choose the iterative path, the logic would be:
        1.  Calculate an initial `C_0` assuming zero heat loss.
        2.  Calculate an initial `R_0` using `R_0 = τ / C_0`.
        3.  In the next iteration, re-calculate the heating rate `dT/dt` for each cycle, this time correcting for heat loss using `R_0`.
        4.  Use the corrected `dT/dt` to calculate a more accurate `C_1`.
        5.  This can be repeated, but usually one iteration is sufficient.

I recommend the **simpler solution** of using the initial heating rate, as it's less prone to instability.

---

#### Task 4: Integrate Heating Cycle RC Estimation
**What's Great:**
* This integration logic is excellent. Detecting a minimum number of cycles before proceeding is a key robustness check.
* Using the **median** of the results from all valid cycles is the correct statistical approach to reject outliers and arrive at a stable, representative value for `τ` and `C`.
* The confidence score based on the number of valid cycles is a very smart heuristic for weighting the quality of the result.
* The final check for physical plausibility using `R_MIN`, `R_MAX`, etc., is the ultimate safety net.

**No improvements suggested here, this is a solid plan.**

---

#### Task 5 & 6: Logging and Testing
**What's Great:**
* Your logging and testing plans are comprehensive. Logging detailed statistics and warnings is crucial for debugging, and creating synthetic data with known parameters is the gold standard for validating a physics-based model.

**Potential Improvements:**
* **Testing:** In addition to synthetic data, I recommend creating a test case that uses a snippet of **real data** from your system. Choose one "well-behaved" room and one "problematic" room (e.g., `chodba_dole` from your logs) to ensure your new logic is robust enough to handle both clean and noisy real-world scenarios.

---

#### Task 7: Documentation
**What's Great:**
* The plan to update the documentation is essential for maintainability.

**Potential Improvements:**
* Consider adding a small plot or diagram to the `README.md` that visually explains a single heating-and-cooling cycle analysis. This can make the concept much easier to grasp for other developers.

---

### High-Level Strategic Recommendations

1.  **Parameter Persistence:** Once you have calculated the robust RC parameters for a room, they should be **saved**. You don't want to re-run this entire analysis every time the system starts. Consider saving the final parameters to a JSON or YAML file in the `config` directory. The `ThermalAnalyzer` can then load these parameters on startup and only trigger a re-analysis if a "recalibrate" flag is set or if model performance degrades over time.

2.  **Fallback Strategy:** What happens if a room has fewer than 3 valid heating cycles (e.g., in summer)? Your plan correctly falls back to a simplified estimation. As a further enhancement, you could create a "regional" fallback. For instance, if the `pokoj_1` model fails, its initial parameters could be based on the validated model from `pokoj_2`, since they are likely to have similar thermal properties.

This is an outstanding plan that demonstrates a deep understanding of the problem. My suggestions are primarily aimed at refining the implementation details and ensuring maximum robustness. This approach will lead to a far more accurate and reliable thermal model.

---

## 🚀 Future Enhancements (Lower Priority)

### Database Query Optimization
- **Issue:** Large date range queries (882 days for all-winters analysis) can timeout with current 30-second InfluxDB timeout
- **Solution:** Implement chunked data extraction or increase timeout for comprehensive analyses
- **Files:** `pems_v2/analysis/core/data_extraction.py`

### Advanced Winter Analysis Features  
- **Multi-Season Comparison:** Compare thermal performance across different winter seasons
- **Seasonal Trend Analysis:** Track how RC parameters change over multiple years
- **Weather Impact Assessment:** Correlate thermal performance with specific weather patterns
- **Energy Efficiency Tracking:** Monitor heating system performance improvements over time

### Real-Time Analysis Integration
- **Live Thermal Monitoring:** Use validated RC parameters for real-time room temperature prediction
- **Heating Optimization:** Implement predictive heating control based on accurate thermal models
- **Anomaly Detection:** Detect heating system failures or building envelope changes

### System Integration
- **MQTT Command Structure:** Implement control commands for manual heating overrides
- **Local Timezone Logging:** Update log timestamps to Europe/Prague timezone
- **Service-Specific Prefixes:** Add clear service identification in log messages

---

## 📊 Current System Status

### Thermal Analysis Capabilities ✅
- ✅ Comprehensive data extraction (PV, rooms, weather, relays, battery, EV, prices)
- ✅ Advanced thermal preprocessing with Savitzky-Golay filtering and outlier detection
- ✅ RC parameter estimation (needs improvement with heating cycle analysis)
- ✅ Room coupling analysis and thermal network visualization
- ✅ Professional HTML reports with interactive visualizations
- ✅ Winter-focused analysis with all historical data (Sep 2022 onwards)

### Data Coverage ✅
- ✅ 3 complete winter seasons: 2022/2023, 2023/2024, 2024/2025
- ✅ 882 days of winter thermal data available
- ✅ 17 rooms with temperature monitoring
- ✅ Relay heating state tracking
- ✅ Weather correlation data
- ✅ Energy price integration

### Code Quality ✅
- ✅ 76+ test cases with 96% coverage
- ✅ Strict type checking (mypy)
- ✅ Professional linting (black, isort, flake8)
- ✅ Comprehensive documentation
- ✅ Makefile-based development workflow

**Next Major Milestone:** Complete the heating cycle analysis implementation to achieve production-ready thermal modeling accuracy.