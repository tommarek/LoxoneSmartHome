# PEMS v2 Thermal Analysis - 5-Minute Resampling Update

## Overview

This update modifies the main thermal analysis components in PEMS v2 to use 5-minute resampling for all data, specifically optimized for underfloor heating systems with high thermal mass characteristics.

## Key Changes Made

### 1. Thermal Data Preprocessor (`preprocessing/thermal_data_preprocessor.py`)

**Added 5-minute resampling capability:**
- New `resample_interval = '5min'` property for consistent data intervals
- New `_resample_room_data()` method for temperature, relay states, and setpoints
- New `_resample_weather_data()` method for outdoor temperature and weather parameters
- Integrated resampling into the main preprocessing pipeline
- Updated all filtering and smoothing parameters for 5-minute data:
  - Rolling median window: 5 points = 25 minutes
  - Stuck sensor detection: 48 points = 240 minutes
  - Savitzky-Golay filter: 7 points = 35 minutes window

**Resampling strategies:**
- Temperature data: Mean aggregation with linear interpolation for gaps
- Relay/heating states: Forward fill to maintain state transitions
- Weather data: Mean aggregation with interpolation for smooth transitions

### 2. Thermal Analysis Module (`analyzers/thermal_analysis.py`)

**Updated for underfloor heating characteristics:**
- Extended minimum heating duration: 1.5 hours (was 2.0 hours)
- Reduced minimum non-heating duration: 2.0 hours (was 3.0 hours)
- Extended decay analysis period: 12.0 hours (was 8.0 hours)
- Extended thermal peak search window: 3.0 hours (was 2.0 hours)

**Improved decay cycle detection:**
- Minimum decay duration: 1.0 hour (was 0.5 hours) for underfloor heating
- Minimum data points: 12 points per hour for 5-minute data
- Relaxed outdoor temperature stability: 1.0°C (was 0.75°C)
- Updated derivative calculations for 5-minute intervals

**Enhanced thermal lag handling:**
- Extended search window for thermal peaks after heating stops
- Better handling of thermal mass effects (30-60 minute delays)
- Improved acceptance criteria for longer thermal response times

### 3. System Configuration (`config/system_config.json`)

**Added new thermal analysis configuration section:**
```json
"thermal_analysis": {
  "min_heating_duration_hours": 1.5,
  "min_non_heating_duration_hours": 2.0,
  "decay_analysis_hours": 12.0,
  "resample_interval": "5min",
  "underfloor_heating": true,
  "thermal_lag_hours": 1.0,
  "min_decay_duration_hours": 1.0
}
```

## Benefits for Underfloor Heating Systems

### 1. **Improved Thermal Mass Handling**
- 5-minute resampling captures thermal inertia without oversampling
- Extended analysis windows accommodate slower thermal response
- Better peak detection for systems with 30-60 minute thermal lag

### 2. **Enhanced Cycle Detection**
- Adjusted duration thresholds for realistic underfloor heating cycles
- Improved filtering criteria for thermal mass systems
- Better handling of long decay periods (up to 12 hours)

### 3. **More Accurate Parameter Estimation**
- Consistent data intervals improve exponential decay fitting
- Better signal-to-noise ratio with appropriate averaging
- Reduced computational overhead while maintaining accuracy

### 4. **Robust Data Quality**
- Maintained outlier detection and sensor validation
- Improved interpolation for missing data points
- Consistent time alignment across all data sources

## Technical Implementation Details

### Data Resampling Strategy
1. **Temperature Data**: Mean aggregation preserves thermal trends
2. **Relay States**: Forward fill maintains state transitions accurately
3. **Weather Data**: Linear interpolation for smooth outdoor conditions
4. **Gap Handling**: Maximum 3 consecutive missing points (15 minutes)

### Parameter Adjustments for 5-Minute Data
- **Window sizes**: All adjusted to maintain equivalent time periods
- **Smoothing filters**: Optimized for 5-minute noise characteristics
- **Acceptance criteria**: Relaxed for longer thermal response times
- **Data validation**: Updated point count requirements

### Quality Assurance
- Comprehensive logging of resampling operations
- Data point count validation before and after resampling
- Quality metrics maintained throughout the pipeline
- Backward compatibility with existing analysis workflows

## Usage

The updated system automatically applies 5-minute resampling when using the thermal data preprocessor:

```python
from preprocessing.thermal_data_preprocessor import ThermalDataPreprocessor

# Create preprocessor (automatically uses 5-minute resampling)
preprocessor = ThermalDataPreprocessor(settings)

# Process data - resampling applied automatically
processed_rooms, processed_weather = preprocessor.prepare_thermal_analysis_data(
    room_data, weather_data, relay_data
)
```

## Testing

A comprehensive test script (`test_5min_resampling.py`) validates:
- ✅ Resampling functionality for room, relay, and weather data
- ✅ Integration with existing preprocessing pipeline
- ✅ Thermal analyzer configuration for underfloor heating
- ✅ Data quality preservation through resampling

## Backward Compatibility

- All existing analysis workflows continue to work
- Configuration defaults maintain reasonable behavior
- Optional relay data handling preserved
- Existing file formats and outputs unchanged

## Performance Impact

- **Memory**: Reduced by ~80% due to fewer data points (5min vs 1min)
- **Processing**: Faster analysis with maintained accuracy
- **Storage**: Smaller intermediate datasets
- **Accuracy**: Maintained or improved for thermal mass systems

This update makes the PEMS v2 thermal analysis system specifically optimized for underfloor heating systems while maintaining compatibility with all existing functionality.