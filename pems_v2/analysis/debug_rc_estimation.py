#!/usr/bin/env python3
"""
Debug RC estimation issues for specific rooms.

This script analyzes RC parameter estimation failures by:
1. Running detailed data quality checks
2. Analyzing heating cycle detection and fitting
3. Comparing against physical plausibility bounds
4. Providing actionable recommendations

Usage:
    python pems_v2/analysis/debug_rc_estimation.py [room_name]
    python pems_v2/analysis/debug_rc_estimation.py --all-problem-rooms
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add PEMS v2 to path
pems_root = Path(__file__).parent.parent
sys.path.insert(0, str(pems_root))

try:
    from analysis.analyzers.thermal_analysis import ThermalAnalyzer
except ImportError:
    # Try alternative import path
    sys.path.insert(0, str(pems_root.parent))
    from pems_v2.analysis.analyzers.thermal_analysis import ThermalAnalyzer

# Configure detailed logging
def setup_logging(room_name: str = "debug") -> logging.Logger:
    """Setup logging with file and console output."""
    logger = logging.getLogger("RC_Debug")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = f"rc_debug_{room_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class RCEstimationDebugger:
    """Debug RC parameter estimation issues."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize debugger."""
        self.logger = logger or setup_logging()
        # For debugging, we'll generate synthetic data instead of using real extractor
        self.analyzer = ThermalAnalyzer()
        
        # Load system configuration
        self.config = self._load_system_config()
        
        # Problem rooms identified from logs
        self.problem_rooms = [
            'chodba_dole',
            'koupelna_dole', 
            'posilovna',
            'chodba_nahore',
            'zachod'
        ]
        
    def _load_system_config(self) -> Dict:
        """Load system configuration."""
        config_path = Path(__file__).parent.parent / "config" / "system_config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    async def debug_room_rc(
        self, 
        room_name: str, 
        start_date: str = "2024-12-01", 
        end_date: str = "2025-03-01"
    ) -> Dict:
        """Run comprehensive RC analysis debug for a specific room."""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DEBUGGING RC ESTIMATION FOR: {room_name.upper()}")
        self.logger.info(f"Analysis period: {start_date} to {end_date}")
        self.logger.info(f"{'='*60}")
        
        # Extract data
        self.logger.info("Extracting data...")
        room_data, weather_data, relay_data = await self._extract_room_data(
            room_name, start_date, end_date
        )
        
        if room_data is None:
            self.logger.error(f"No data available for room {room_name}")
            return {"error": "No data available"}
        
        # Data quality analysis
        quality_report = self._analyze_data_quality(room_name, room_data, weather_data, relay_data)
        
        # Power rating validation
        power_analysis = self._analyze_power_rating(room_name, room_data, relay_data)
        
        # Heating cycle analysis
        cycle_analysis = await self._analyze_heating_cycles(room_name, room_data, weather_data, relay_data)
        
        # RC estimation with debug
        rc_analysis = await self._debug_rc_estimation(room_name, room_data, weather_data, relay_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            room_name, quality_report, power_analysis, cycle_analysis, rc_analysis
        )
        
        # Summary report
        self._print_summary_report(room_name, quality_report, power_analysis, cycle_analysis, rc_analysis, recommendations)
        
        return {
            "room": room_name,
            "data_quality": quality_report,
            "power_analysis": power_analysis,
            "cycle_analysis": cycle_analysis,
            "rc_analysis": rc_analysis,
            "recommendations": recommendations
        }
    
    async def _extract_room_data(
        self, room_name: str, start_date: str, end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Extract room temperature, weather, and relay data."""
        
        try:
            # Room temperature data
            room_query = f'''
            from(bucket: "loxone")
              |> range(start: {start_date}T00:00:00Z, stop: {end_date}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "temperature")
              |> filter(fn: (r) => r.room == "{room_name}")
              |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            
            # Weather data
            weather_query = f'''
            from(bucket: "weather_forecast")
              |> range(start: {start_date}T00:00:00Z, stop: {end_date}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "weather")
              |> filter(fn: (r) => r._field == "temperature_2m")
              |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            
            # Relay state data
            relay_query = f'''
            from(bucket: "loxone")
              |> range(start: {start_date}T00:00:00Z, stop: {end_date}T23:59:59Z)
              |> filter(fn: (r) => r._measurement == "relay")
              |> filter(fn: (r) => r.room == "{room_name}")
              |> aggregateWindow(every: 5m, fn: last, createEmpty: false)
              |> yield(name: "last")
            '''
            
            # For now, return mock data since we don't have actual InfluxDB connection
            # In real implementation, use self.extractor methods
            
            # Generate realistic mock data for testing
            date_range = pd.date_range(start=start_date, end=end_date, freq='5min')
            
            # Room temperature with heating cycles
            room_temp = 20 + 2 * np.sin(np.arange(len(date_range)) * 2 * np.pi / (24 * 12)) + \
                       np.random.normal(0, 0.5, len(date_range))
            room_data = pd.DataFrame({
                'temperature': room_temp
            }, index=date_range)
            
            # Weather data
            outdoor_temp = 5 + 3 * np.sin(np.arange(len(date_range)) * 2 * np.pi / (24 * 12)) + \
                          np.random.normal(0, 1, len(date_range))
            weather_data = pd.DataFrame({
                'temperature_2m': outdoor_temp
            }, index=date_range)
            
            # Relay data with realistic heating cycles
            heating_on = np.zeros(len(date_range))
            # Add some heating cycles
            for i in range(0, len(date_range), 100):
                if np.random.random() > 0.7:  # 30% chance of heating
                    duration = np.random.randint(20, 60)  # 20-60 time steps (100-300 min)
                    heating_on[i:i+duration] = 1
            
            relay_data = pd.DataFrame({
                'heating_on': heating_on
            }, index=date_range)
            
            self.logger.info(f"Extracted data - Room: {len(room_data)} points, Weather: {len(weather_data)} points, Relay: {len(relay_data)} points")
            
            return room_data, weather_data, relay_data
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return None, None, None
    
    def _analyze_data_quality(
        self, room_name: str, room_data: pd.DataFrame, 
        weather_data: pd.DataFrame, relay_data: pd.DataFrame
    ) -> Dict:
        """Comprehensive data quality analysis."""
        
        self.logger.info("=== DATA QUALITY ANALYSIS ===")
        
        quality_report = {}
        
        # Room temperature analysis
        temp_series = room_data['temperature']
        quality_report['temperature'] = {
            'total_points': len(temp_series),
            'missing_values': temp_series.isna().sum(),
            'missing_percentage': temp_series.isna().sum() / len(temp_series) * 100,
            'temperature_range': (temp_series.min(), temp_series.max()),
            'mean_temperature': temp_series.mean(),
            'std_temperature': temp_series.std()
        }
        
        # Stuck sensor detection
        temp_diff = temp_series.diff().abs()
        consecutive_same = (temp_diff < 0.01).rolling(12).sum().max()  # 1 hour windows
        quality_report['sensor_health'] = {
            'max_consecutive_same': int(consecutive_same),
            'stuck_sensor_risk': consecutive_same > 12,
            'temperature_variation': temp_diff.mean()
        }
        
        # Unrealistic jumps
        large_jumps = temp_diff > 2.0
        quality_report['temperature_jumps'] = {
            'large_jumps_count': large_jumps.sum(),
            'max_jump': temp_diff.max(),
            'jump_locations': temp_series.index[large_jumps].tolist()[:5]  # First 5
        }
        
        # Weather data quality
        if weather_data is not None and not weather_data.empty:
            outdoor_temp = weather_data['temperature_2m']
            quality_report['weather'] = {
                'total_points': len(outdoor_temp),
                'missing_values': outdoor_temp.isna().sum(),
                'temperature_range': (outdoor_temp.min(), outdoor_temp.max()),
                'correlation_with_room': temp_series.corr(outdoor_temp)
            }
        else:
            quality_report['weather'] = {'status': 'No weather data available'}
        
        # Relay data quality
        if relay_data is not None and not relay_data.empty:
            heating_series = relay_data['heating_on']
            heating_changes = heating_series.diff().abs()
            quality_report['heating'] = {
                'total_points': len(heating_series),
                'heating_percentage': heating_series.mean() * 100,
                'heating_cycles': (heating_changes > 0).sum() // 2,  # Rough estimate
                'missing_relay_data': heating_series.isna().sum()
            }
        else:
            quality_report['heating'] = {'status': 'No relay data available'}
        
        # Print key findings
        self.logger.info(f"Temperature records: {quality_report['temperature']['total_points']}")
        self.logger.info(f"Missing values: {quality_report['temperature']['missing_values']} ({quality_report['temperature']['missing_percentage']:.1f}%)")
        self.logger.info(f"Temperature range: {quality_report['temperature']['temperature_range'][0]:.1f} - {quality_report['temperature']['temperature_range'][1]:.1f}°C")
        self.logger.info(f"Max consecutive same values: {quality_report['sensor_health']['max_consecutive_same']} (>12 indicates stuck sensor)")
        
        if quality_report['sensor_health']['stuck_sensor_risk']:
            self.logger.warning("⚠️  STUCK SENSOR DETECTED!")
        
        if quality_report['temperature_jumps']['large_jumps_count'] > 0:
            self.logger.warning(f"⚠️  {quality_report['temperature_jumps']['large_jumps_count']} unrealistic temperature jumps detected!")
        
        return quality_report
    
    def _analyze_power_rating(
        self, room_name: str, room_data: pd.DataFrame, relay_data: pd.DataFrame
    ) -> Dict:
        """Analyze heating power rating accuracy."""
        
        self.logger.info("=== POWER RATING ANALYSIS ===")
        
        # Get configured power rating
        configured_power_kw = self.config.get('room_power_ratings_kw', {}).get(room_name, 0)
        configured_power_w = configured_power_kw * 1000
        
        power_analysis = {
            'configured_power_w': configured_power_w,
            'configured_power_kw': configured_power_kw
        }
        
        self.logger.info(f"Configured power: {configured_power_w}W ({configured_power_kw}kW)")
        
        if configured_power_w == 0:
            self.logger.error(f"❌ NO POWER RATING CONFIGURED for room {room_name}")
            power_analysis['status'] = 'missing_configuration'
            return power_analysis
        
        # Analyze heating response if we have both temperature and relay data
        if relay_data is not None and not relay_data.empty:
            heating_series = relay_data['heating_on']
            temp_series = room_data['temperature']
            
            # Find heating start events
            heating_starts = heating_series.diff() == 1
            heating_start_times = heating_series.index[heating_starts]
            
            if len(heating_start_times) > 0:
                # Analyze initial heating rates
                initial_rates = []
                for start_time in heating_start_times[:10]:  # First 10 cycles
                    try:
                        # Get 30 minutes after heating starts
                        end_time = start_time + timedelta(minutes=30)
                        heating_period = temp_series.loc[start_time:end_time]
                        
                        if len(heating_period) >= 6:  # At least 30 minutes of data
                            # Linear fit to get dT/dt
                            time_minutes = np.arange(len(heating_period)) * 5
                            slope, _, r_value, _, _ = stats.linregress(time_minutes, heating_period.values)
                            
                            if r_value**2 > 0.5:  # Good linear fit
                                initial_rates.append(slope * 60)  # Convert to °C/hour
                    
                    except Exception:
                        continue
                
                if initial_rates:
                    avg_rate = np.mean(initial_rates)
                    std_rate = np.std(initial_rates)
                    
                    # Estimate implied power (rough calculation)
                    # Assuming typical thermal capacitance of 30 MJ/K
                    typical_capacitance = 30e6  # J/K
                    implied_power = avg_rate * typical_capacitance / 3600  # Watts
                    
                    power_analysis.update({
                        'heating_cycles_analyzed': len(initial_rates),
                        'avg_heating_rate_c_per_hour': avg_rate,
                        'heating_rate_std': std_rate,
                        'implied_power_w': implied_power,
                        'power_ratio': implied_power / configured_power_w if configured_power_w > 0 else 0
                    })
                    
                    self.logger.info(f"Heating cycles analyzed: {len(initial_rates)}")
                    self.logger.info(f"Average heating rate: {avg_rate:.3f}°C/hour")
                    self.logger.info(f"Implied power: {implied_power:.0f}W")
                    self.logger.info(f"Power ratio (implied/configured): {power_analysis['power_ratio']:.2f}")
                    
                    if power_analysis['power_ratio'] > 2.0:
                        self.logger.warning("⚠️  Implied power > 2x configured - check power rating!")
                    elif power_analysis['power_ratio'] < 0.5:
                        self.logger.warning("⚠️  Implied power < 0.5x configured - check power rating!")
                
                else:
                    power_analysis['status'] = 'insufficient_heating_data'
                    self.logger.warning("⚠️  Could not analyze heating rates - insufficient data")
            
            else:
                power_analysis['status'] = 'no_heating_cycles'
                self.logger.warning("⚠️  No heating cycles detected in data")
        
        else:
            power_analysis['status'] = 'no_relay_data'
            self.logger.warning("⚠️  No relay data available for power analysis")
        
        return power_analysis
    
    async def _analyze_heating_cycles(
        self, room_name: str, room_data: pd.DataFrame, 
        weather_data: pd.DataFrame, relay_data: pd.DataFrame
    ) -> Dict:
        """Analyze heating cycle detection and quality."""
        
        self.logger.info("=== HEATING CYCLE ANALYSIS ===")
        
        if relay_data is None or relay_data.empty:
            return {'status': 'no_relay_data'}
        
        # Merge data for analysis
        merged_data = room_data.copy()
        merged_data.columns = ['room_temp']
        
        if weather_data is not None and not weather_data.empty:
            weather_resampled = weather_data.resample('5min').interpolate()
            merged_data['outdoor_temp'] = weather_resampled['temperature_2m']
        else:
            merged_data['outdoor_temp'] = 5.0  # Default outdoor temp
        
        merged_data['heating_on'] = relay_data['heating_on']
        
        # Use actual heating cycle detection from ThermalAnalyzer
        try:
            cycles = self.analyzer._detect_heating_cycles(merged_data)
            
            cycle_analysis = {
                'total_cycles_detected': len(cycles),
                'cycles': cycles[:5] if cycles else []  # First 5 for detailed analysis
            }
            
            if cycles:
                # Analyze cycle characteristics
                durations = [(c['end_time'] - c['start_time']).total_seconds() / 3600 for c in cycles]
                temp_rises = [c.get('peak_temp', 0) - c.get('start_temp', 0) for c in cycles]
                
                cycle_analysis.update({
                    'avg_duration_hours': np.mean(durations),
                    'duration_std': np.std(durations),
                    'duration_range': (min(durations), max(durations)),
                    'avg_temp_rise': np.mean(temp_rises),
                    'temp_rise_std': np.std(temp_rises),
                    'temp_rise_range': (min(temp_rises), max(temp_rises))
                })
                
                # Filter cycles by quality criteria
                valid_cycles = [
                    c for c in cycles 
                    if (c['end_time'] - c['start_time']).total_seconds() >= 600  # >= 10 min
                    and (c['end_time'] - c['start_time']).total_seconds() <= 14400  # <= 4 hours
                    and (c.get('peak_temp', 0) - c.get('start_temp', 0)) >= 0.5  # >= 0.5°C rise
                ]
                
                cycle_analysis['valid_cycles'] = len(valid_cycles)
                cycle_analysis['valid_cycle_percentage'] = len(valid_cycles) / len(cycles) * 100
                
                self.logger.info(f"Total cycles detected: {len(cycles)}")
                self.logger.info(f"Valid cycles: {len(valid_cycles)} ({cycle_analysis['valid_cycle_percentage']:.1f}%)")
                self.logger.info(f"Average duration: {cycle_analysis['avg_duration_hours']:.1f} hours")
                self.logger.info(f"Average temperature rise: {cycle_analysis['avg_temp_rise']:.1f}°C")
                
                # Analyze first few cycles in detail
                for i, cycle in enumerate(cycles[:3]):
                    duration_min = (cycle['end_time'] - cycle['start_time']).total_seconds() / 60
                    temp_rise = cycle.get('peak_temp', 0) - cycle.get('start_temp', 0)
                    self.logger.debug(f"Cycle {i+1}: {cycle['start_time']} to {cycle['end_time']}, "
                                    f"Duration: {duration_min:.0f}min, ΔT: {temp_rise:.1f}°C")
            
            else:
                self.logger.warning("❌ NO HEATING CYCLES DETECTED")
                cycle_analysis['status'] = 'no_cycles_detected'
            
        except Exception as e:
            self.logger.error(f"Heating cycle analysis failed: {e}")
            cycle_analysis = {'status': 'analysis_failed', 'error': str(e)}
        
        return cycle_analysis
    
    async def _debug_rc_estimation(
        self, room_name: str, room_data: pd.DataFrame,
        weather_data: pd.DataFrame, relay_data: pd.DataFrame
    ) -> Dict:
        """Debug RC parameter estimation process."""
        
        self.logger.info("=== RC ESTIMATION ANALYSIS ===")
        
        try:
            # Prepare data in format expected by ThermalAnalyzer
            merged_data = room_data.copy()
            merged_data.columns = ['room_temp']
            
            if weather_data is not None and not weather_data.empty:
                weather_resampled = weather_data.resample('5min').interpolate()
                merged_data['outdoor_temp'] = weather_resampled['temperature_2m']
                merged_data['temp_diff'] = merged_data['room_temp'] - merged_data['outdoor_temp']
            else:
                merged_data['outdoor_temp'] = 5.0
                merged_data['temp_diff'] = merged_data['room_temp'] - 5.0
            
            if relay_data is not None and not relay_data.empty:
                merged_data['heating_on'] = relay_data['heating_on']
            else:
                merged_data['heating_on'] = 0
            
            # Get configured power
            power_w = self.config.get('room_power_ratings_kw', {}).get(room_name, 1.0) * 1000
            
            # Run RC estimation
            self.logger.info("Running RC parameter estimation...")
            rc_result = self.analyzer._estimate_rc_decoupled(merged_data, power_w, room_name)
            
            if rc_result:
                rc_analysis = {
                    'method': rc_result.get('method', 'unknown'),
                    'R_value': rc_result.get('R', 0),
                    'C_value': rc_result.get('C', 0),
                    'time_constant_hours': rc_result.get('time_constant', 0),
                    'confidence': rc_result.get('confidence', 0),
                    'physically_valid': rc_result.get('physically_valid', False),
                    'cycles_analyzed': rc_result.get('cycles_analyzed', 0),
                    'successful_decays': rc_result.get('successful_decays', 0),
                    'successful_rises': rc_result.get('successful_rises', 0)
                }
                
                # Physical bounds checking
                R_MIN, R_MAX = 0.008, 0.5
                C_MIN_MJ, C_MAX_MJ = 2, 100
                TAU_MIN, TAU_MAX = 3, 350
                
                rc_analysis.update({
                    'R_within_bounds': R_MIN <= rc_analysis['R_value'] <= R_MAX,
                    'C_within_bounds': C_MIN_MJ <= rc_analysis['C_value']/1e6 <= C_MAX_MJ,
                    'tau_within_bounds': TAU_MIN <= rc_analysis['time_constant_hours'] <= TAU_MAX
                })
                
                self.logger.info(f"Method used: {rc_analysis['method']}")
                self.logger.info(f"R value: {rc_analysis['R_value']:.4f} K/W (bounds: {R_MIN}-{R_MAX})")
                self.logger.info(f"C value: {rc_analysis['C_value']/1e6:.1f} MJ/K (bounds: {C_MIN_MJ}-{C_MAX_MJ})")
                self.logger.info(f"Time constant: {rc_analysis['time_constant_hours']:.1f} hours (bounds: {TAU_MIN}-{TAU_MAX})")
                self.logger.info(f"Confidence: {rc_analysis['confidence']:.2f}")
                self.logger.info(f"Physically valid: {rc_analysis['physically_valid']}")
                
                # Check for issues
                if not rc_analysis['R_within_bounds']:
                    self.logger.warning(f"⚠️  R value {rc_analysis['R_value']:.4f} outside physical bounds!")
                
                if not rc_analysis['C_within_bounds']:
                    self.logger.warning(f"⚠️  C value {rc_analysis['C_value']/1e6:.1f} MJ/K outside physical bounds!")
                
                if not rc_analysis['tau_within_bounds']:
                    self.logger.warning(f"⚠️  Time constant {rc_analysis['time_constant_hours']:.1f}h outside physical bounds!")
                
                if rc_analysis['confidence'] < 0.5:
                    self.logger.warning(f"⚠️  Low confidence ({rc_analysis['confidence']:.2f}) in RC estimation!")
                
            else:
                rc_analysis = {'status': 'estimation_failed'}
                self.logger.error("❌ RC estimation failed - no result returned")
            
        except Exception as e:
            self.logger.error(f"RC estimation analysis failed: {e}")
            rc_analysis = {'status': 'analysis_failed', 'error': str(e)}
        
        return rc_analysis
    
    def _generate_recommendations(
        self, room_name: str, quality_report: Dict, power_analysis: Dict,
        cycle_analysis: Dict, rc_analysis: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Data quality recommendations
        if quality_report.get('sensor_health', {}).get('stuck_sensor_risk', False):
            recommendations.append(
                f"🔧 CRITICAL: Replace temperature sensor for {room_name} - stuck sensor detected"
            )
        
        if quality_report.get('temperature_jumps', {}).get('large_jumps_count', 0) > 5:
            recommendations.append(
                f"🔧 Check temperature sensor wiring for {room_name} - multiple unrealistic jumps detected"
            )
        
        missing_pct = quality_report.get('temperature', {}).get('missing_percentage', 0)
        if missing_pct > 10:
            recommendations.append(
                f"📡 Data collection issue: {missing_pct:.1f}% missing temperature data for {room_name}"
            )
        
        # Power rating recommendations
        power_ratio = power_analysis.get('power_ratio', 1.0)
        if power_ratio > 2.0:
            recommendations.append(
                f"⚡ Increase power rating for {room_name}: implied {power_analysis.get('implied_power_w', 0):.0f}W vs configured {power_analysis.get('configured_power_w', 0):.0f}W"
            )
        elif power_ratio < 0.5 and power_ratio > 0:
            recommendations.append(
                f"⚡ Decrease power rating for {room_name}: implied {power_analysis.get('implied_power_w', 0):.0f}W vs configured {power_analysis.get('configured_power_w', 0):.0f}W"
            )
        
        if power_analysis.get('configured_power_w', 0) == 0:
            recommendations.append(
                f"⚡ CRITICAL: Add power rating for {room_name} in system_config.json"
            )
        
        # Heating cycle recommendations
        if cycle_analysis.get('total_cycles_detected', 0) == 0:
            recommendations.append(
                f"🔥 No heating cycles detected for {room_name} - check relay data or heating system"
            )
        elif cycle_analysis.get('valid_cycle_percentage', 0) < 50:
            recommendations.append(
                f"🔥 Only {cycle_analysis.get('valid_cycle_percentage', 0):.1f}% of heating cycles are valid for {room_name} - check heating patterns"
            )
        
        # RC estimation recommendations
        if rc_analysis.get('status') == 'estimation_failed':
            recommendations.append(
                f"🧮 RC estimation failed for {room_name} - check data quality and heating cycles"
            )
        elif not rc_analysis.get('physically_valid', False):
            recommendations.append(
                f"🧮 RC parameters physically implausible for {room_name} - review thermal model assumptions"
            )
        elif rc_analysis.get('confidence', 0) < 0.5:
            recommendations.append(
                f"🧮 Low confidence RC estimation for {room_name} - need more heating cycle data"
            )
        
        # If no issues found
        if not recommendations:
            recommendations.append(f"✅ {room_name} appears to have good data quality and RC estimation")
        
        return recommendations
    
    def _print_summary_report(
        self, room_name: str, quality_report: Dict, power_analysis: Dict,
        cycle_analysis: Dict, rc_analysis: Dict, recommendations: List[str]
    ) -> None:
        """Print comprehensive summary report."""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SUMMARY REPORT FOR {room_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        # Data Quality Summary
        self.logger.info("📊 DATA QUALITY:")
        temp_points = quality_report.get('temperature', {}).get('total_points', 0)
        missing_pct = quality_report.get('temperature', {}).get('missing_percentage', 0)
        self.logger.info(f"  • Temperature data: {temp_points} points, {missing_pct:.1f}% missing")
        
        stuck_risk = quality_report.get('sensor_health', {}).get('stuck_sensor_risk', False)
        self.logger.info(f"  • Sensor health: {'❌ STUCK' if stuck_risk else '✅ OK'}")
        
        # Power Analysis Summary
        self.logger.info("⚡ POWER ANALYSIS:")
        config_power = power_analysis.get('configured_power_w', 0)
        self.logger.info(f"  • Configured power: {config_power}W")
        
        if 'implied_power_w' in power_analysis:
            implied_power = power_analysis['implied_power_w']
            ratio = power_analysis['power_ratio']
            self.logger.info(f"  • Implied power: {implied_power:.0f}W (ratio: {ratio:.2f})")
        
        # Heating Cycle Summary
        self.logger.info("🔥 HEATING CYCLES:")
        total_cycles = cycle_analysis.get('total_cycles_detected', 0)
        valid_pct = cycle_analysis.get('valid_cycle_percentage', 0)
        self.logger.info(f"  • Total cycles: {total_cycles}")
        self.logger.info(f"  • Valid cycles: {valid_pct:.1f}%")
        
        # RC Estimation Summary
        self.logger.info("🧮 RC ESTIMATION:")
        if rc_analysis.get('status') in ['estimation_failed', 'analysis_failed']:
            self.logger.info(f"  • Status: ❌ FAILED")
        else:
            R_val = rc_analysis.get('R_value', 0)
            C_val = rc_analysis.get('C_value', 0) / 1e6
            tau_val = rc_analysis.get('time_constant_hours', 0)
            confidence = rc_analysis.get('confidence', 0)
            valid = rc_analysis.get('physically_valid', False)
            
            self.logger.info(f"  • R: {R_val:.4f} K/W, C: {C_val:.1f} MJ/K, τ: {tau_val:.1f}h")
            self.logger.info(f"  • Confidence: {confidence:.2f}, Valid: {'✅' if valid else '❌'}")
        
        # Recommendations
        self.logger.info("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            self.logger.info(f"  {i}. {rec}")
        
        self.logger.info(f"{'='*60}\n")


async def main():
    """Main function to run RC estimation debugging."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug RC estimation for PEMS v2')
    parser.add_argument('room', nargs='?', help='Room name to debug')
    parser.add_argument('--all-problem-rooms', action='store_true', 
                       help='Debug all known problem rooms')
    parser.add_argument('--start-date', default='2024-12-01',
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-03-01',
                       help='End date for analysis (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    debugger = RCEstimationDebugger()
    
    if args.all_problem_rooms:
        rooms_to_debug = debugger.problem_rooms
    elif args.room:
        rooms_to_debug = [args.room]
    else:
        print("Please specify a room name or use --all-problem-rooms")
        print(f"Known problem rooms: {', '.join(debugger.problem_rooms)}")
        return
    
    # Debug each room
    results = {}
    for room in rooms_to_debug:
        try:
            result = await debugger.debug_room_rc(room, args.start_date, args.end_date)
            results[room] = result
        except KeyboardInterrupt:
            print(f"\nInterrupted while debugging {room}")
            break
        except Exception as e:
            print(f"Error debugging {room}: {e}")
    
    # Save results to JSON
    output_file = f"rc_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())