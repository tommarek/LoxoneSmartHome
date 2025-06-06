# PEMS v2 Phase 1 Implementation Battle Plan

Based on the comprehensive TODO-Phase1.md, here's the strategic implementation plan:

## **Phase 1A: Core Infrastructure (Priority: HIGH)**

### 1. **Loxone Adaptation Layer** 🆕
Create the foundation for handling Loxone field naming:

**Tasks:**
- [x] Create `analysis/utils/loxone_adapter.py` with `LoxoneFieldAdapter` class
- [x] Implement field mapping for temperature, humidity, relay states
- [x] Add standardization methods for room data processing
- [x] Test adapter with actual Loxone field names

### 2. **Pattern Analysis Enhancement** ✅🟡
Update existing pattern_analysis.py for Loxone compatibility:

**Tasks:**
- [x] Add `_get_loxone_field()` method for dynamic field detection
- [x] Update PV export policy detection methods
- [x] Enhance relay coordination optimization algorithms
- [x] Test with Loxone field naming conventions

### 3. **Thermal Analysis Enhancement** ✅🟡
Update existing thermal_analysis.py to work with Loxone data:

**Tasks:**
- [x] Integrate `LoxoneFieldAdapter` in `_analyze_single_room()`
- [x] Update `_merge_room_weather_data()` for Loxone naming
- [x] Ensure RC parameter estimation works with dict-based room data
- [x] Test relay state integration for heating period detection

### 4. **Data Preprocessing Enhancement** 🟡
Add Loxone-specific processing to existing data_preprocessing.py:

**Tasks:**
- [x] Add `process_loxone_room_data()` method
- [x] Implement field mapping for temperature_{room_name} patterns
- [x] Add humidity and target_temp field handling
- [x] Test preprocessing with actual room data

## **Phase 1B: Analysis Pipeline (Priority: HIGH)**

### 5. **ComprehensiveAnalyzer Implementation** 🆕
Create the master orchestrator that uses existing analyzers:

**Tasks:**
- [x] Create `analysis/pipelines/comprehensive_analysis.py`
- [x] Implement `ComprehensiveAnalyzer` class integrating existing modules
- [x] Add data quality assessment using existing components
- [x] Build integrated insights generation from all analyzers
- [x] Add economic opportunity analysis pipeline

### 6. **Run Analysis Pipeline Update** 🟡
Update existing run_analysis.py for integrated workflow:

**Tasks:**
- [x] Replace generic pipeline with `ComprehensiveAnalyzer` integration
- [x] Update `_merge_room_weather_data()` for Loxone field handling
- [x] Add field detection for temperature_ patterns
- [x] Integrate with existing analyzers seamlessly

### 7. **Validation & Testing**
Ensure everything works with real data:

**Tasks:**
- [x] Test comprehensive pipeline with 30-day data subset
- [x] Validate Loxone field mapping with actual database data
- [x] Check thermal analysis with relay integration
- [x] Verify PV export detection algorithms work with enhanced data

## **Phase 1C: Supporting Infrastructure (Priority: MEDIUM)**

### 8. **File Structure Reorganization** 📋
Reorganize existing files according to recommended structure:

**Tasks:**
- [x] Create `analysis/core/` directory and move `data_extraction.py`, `visualization.py`
- [x] Create `analysis/analyzers/` directory and move analysis modules
- [x] Create `analysis/utils/` directory for `loxone_adapter.py`
- [x] Create `analysis/pipelines/` directory for comprehensive analysis
- [x] Update import statements in all modules

### 9. **Daily Analysis Integration** 🆕
Create daily analysis workflow using existing components:

**Tasks:**
- [x] Create `analysis/daily_analysis.py` using `ComprehensiveAnalyzer`
- [x] Implement routine for PV export, relay optimization, economic analysis
- [x] Add automated report generation using existing visualization
- [x] Test daily workflow with real data

### 10. **Report Generation Enhancement** 🆕
Extract reporting functionality from visualization.py:

**Tasks:**
- [x] Create `analysis/reports/report_generator.py`
- [x] Extract report generation methods from existing visualization
- [x] Add text and HTML report templates
- [x] Integrate with comprehensive analysis results

## **Files to Keep As-Is** ✅

The following existing files need **NO CHANGES** (keep exactly as they are):

### Core Components (Keep 100%)
- [x] `data_extraction.py` - ✅ Already updated and working well
- [x] `base_load_analysis.py` - ✅ Useful for understanding non-controllable loads  
- [x] `feature_engineering.py` - ✅ Essential for ML models in Phase 2
- [x] `visualization.py` - ✅ Comprehensive plotting functions, well-designed

## **Implementation Order & Dependencies**

```
Week 1: Loxone Adaptation Layer
├── analysis/utils/loxone_adapter.py (LoxoneFieldAdapter)
├── Update thermal_analysis.py with adapter integration
├── Update pattern_analysis.py with Loxone field detection
└── Test adapter with real Loxone field names

Week 2: Pipeline Integration
├── analysis/pipelines/comprehensive_analysis.py
├── Update run_analysis.py for integrated workflow
├── Update data_preprocessing.py with Loxone handlers
└── End-to-end pipeline testing with existing modules

Week 3: Structure & Organization
├── File structure reorganization (core/, analyzers/, utils/, pipelines/)
├── analysis/daily_analysis.py for routine workflows
├── analysis/reports/report_generator.py extraction
└── Update all import statements

Week 4: Validation & Testing
├── Full pipeline testing with 2-year data
├── Validate all existing analyzers work with Loxone data
├── Performance optimization and error handling
└── Documentation updates
```

## **Updated File Structure** 📋

```
pems_v2/analysis/
├── core/                          # Core components (no changes needed)
│   ├── data_extraction.py         # ✅ Keep as-is
│   ├── data_preprocessing.py      # 🟡 Add Loxone handlers
│   └── visualization.py           # ✅ Keep as-is
│
├── analyzers/                     # Analysis modules (minor updates)
│   ├── pattern_analysis.py        # 🟡 Add Loxone field detection
│   ├── thermal_analysis.py        # 🟡 Integrate LoxoneFieldAdapter
│   ├── base_load_analysis.py      # ✅ Keep as-is
│   └── feature_engineering.py     # ✅ Keep as-is
│
├── utils/                         # New utilities
│   └── loxone_adapter.py          # 🆕 New adapter for Loxone fields
│
├── pipelines/                     # Pipeline orchestration
│   ├── comprehensive_analysis.py  # 🆕 Master orchestrator
│   ├── run_analysis.py           # 🟡 Update for integration
│   └── daily_analysis.py         # 🆕 Daily workflow
│
└── reports/                       # Report generation
    └── report_generator.py        # 🆕 Extract from visualization.py
```

## **Key Technical Considerations**

1. **Data Structure Compatibility**: All analyzers must handle both DataFrame and dict inputs for rooms/relay_states
2. **Field Name Mapping**: Consistent translation from Loxone naming to standard analysis names
3. **Performance**: Efficient processing of 2-year data (730 days × 96 intervals/day = ~70k records per field)
4. **Error Handling**: Graceful degradation when optional data sources are missing
5. **Modularity**: Each analyzer should work independently for partial analysis runs

## **Success Metrics**

- [ ] Process 2-year historical data in <10 minutes
- [ ] Generate actionable insights for PV export optimization
- [ ] Identify 10-15% peak demand reduction potential through relay coordination
- [ ] Detect thermal parameters for all 16 rooms with >80% confidence
- [ ] Create interactive dashboards for system monitoring

---

# Original Implementation Details

Looking at your updated `data_extraction.py`, I can see you've made significant improvements to handle your Loxone system's data structure. Here's what you need to update and how to achieve the best possible analysis:

## 1. **Key Updates Needed**

### A. Update Analysis Scripts to Use New Data Structure

Your `extract_room_temperatures()` now returns room data organized differently. Update the analysis scripts:

```python
# In thermal_analysis.py and other analysis files
# OLD approach:
for room_name, room_df in room_data.items():
    if "temperature" in room_df.columns:
        # process temperature
        
# NEW approach should handle the Loxone field naming:
for room_name, room_df in room_data.items():
    # Temperature might be in columns like "temperature_obyvak"
    temp_cols = [col for col in room_df.columns if 'temperature' in col.lower()]
    if temp_cols:
        # Use the first temperature column found
        temp_col = temp_cols[0]
```

### B. Create Unified Analysis Pipeline

Based on your documentation, create a comprehensive analysis script:## 2. **Best Practices for Your Analysis**

```
"""
Comprehensive PEMS v2 Analysis Pipeline
Optimized for Loxone relay-based heating system with PV export constraints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from analysis.data_extraction import DataExtractor
from analysis.pattern_analysis import PVAnalyzer, RelayPatternAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from analysis.base_load_analysis import BaseLoadAnalyzer
from analysis.feature_engineering import FeatureEngineer
from analysis.visualization import AnalysisVisualizer
from config.settings import PEMSSettings

class ComprehensiveAnalyzer:
    """Enhanced analyzer for Loxone-based PEMS system."""
    
    def __init__(self, settings: PEMSSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.extractor = DataExtractor(settings)
        self.pv_analyzer = PVAnalyzer()
        self.relay_analyzer = RelayPatternAnalyzer()
        self.thermal_analyzer = ThermalAnalyzer()
        self.base_load_analyzer = BaseLoadAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = AnalysisVisualizer()
        
    async def run_comprehensive_analysis(
        self, 
        start_date: datetime, 
        end_date: datetime,
        analysis_types: list = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis with all available data sources.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            analysis_types: List of analysis to run (default: all)
        """
        if analysis_types is None:
            analysis_types = [
                'pv_export', 'relay_optimization', 'thermal_dynamics',
                'base_load', 'weather_correlation', 'economic_optimization'
            ]
            
        self.logger.info(f"Starting comprehensive analysis from {start_date} to {end_date}")
        
        # Step 1: Extract all data
        self.logger.info("=== Step 1: Data Extraction ===")
        data = await self._extract_all_data(start_date, end_date)
        
        # Step 2: Data quality assessment
        self.logger.info("=== Step 2: Data Quality Assessment ===")
        quality_report = self._assess_data_quality(data)
        
        # Step 3: Run selected analyses
        self.logger.info("=== Step 3: Running Analyses ===")
        results = {
            'data_quality': quality_report,
            'analysis_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            }
        }
        
        if 'pv_export' in analysis_types and not data['pv'].empty:
            results['pv_export_analysis'] = await self._analyze_pv_export_system(data)
            
        if 'relay_optimization' in analysis_types and data['relay_states']:
            results['relay_optimization'] = await self._analyze_relay_optimization(data)
            
        if 'thermal_dynamics' in analysis_types and data['rooms']:
            results['thermal_dynamics'] = await self._analyze_thermal_dynamics(data)
            
        if 'base_load' in analysis_types and not data['consumption'].empty:
            results['base_load_analysis'] = await self._analyze_base_load(data)
            
        if 'weather_correlation' in analysis_types:
            results['weather_correlation'] = await self._analyze_weather_correlations(data)
            
        if 'economic_optimization' in analysis_types and data['prices'] is not None:
            results['economic_optimization'] = await self._analyze_economic_opportunities(data)
        
        # Step 4: Generate integrated insights
        self.logger.info("=== Step 4: Generating Integrated Insights ===")
        results['integrated_insights'] = self._generate_integrated_insights(results)
        
        # Step 5: Create visualizations
        self.logger.info("=== Step 5: Creating Visualizations ===")
        self._create_comprehensive_visualizations(data, results)
        
        # Step 6: Generate report
        self.logger.info("=== Step 6: Generating Report ===")
        self._generate_comprehensive_report(results)
        
        return results
    
    async def _extract_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Extract all available data sources."""
        data = {}
        
        # Core data sources
        self.logger.info("Extracting PV data...")
        data['pv'] = await self.extractor.extract_pv_data(start_date, end_date)
        
        self.logger.info("Extracting room temperature data...")
        data['rooms'] = await self.extractor.extract_room_temperatures(start_date, end_date)
        
        self.logger.info("Extracting weather data...")
        data['weather'] = await self.extractor.extract_weather_data(start_date, end_date)
        
        self.logger.info("Extracting current weather from Loxone...")
        data['current_weather'] = await self.extractor.extract_current_weather(start_date, end_date)
        
        self.logger.info("Extracting consumption data...")
        data['consumption'] = await self.extractor.extract_energy_consumption(start_date, end_date)
        
        self.logger.info("Extracting relay states...")
        data['relay_states'] = await self.extractor.extract_relay_states(start_date, end_date)
        
        # Optional data sources
        try:
            self.logger.info("Extracting energy prices...")
            data['prices'] = await self.extractor.extract_energy_prices(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract prices: {e}")
            data['prices'] = None
            
        try:
            self.logger.info("Extracting battery data...")
            data['battery'] = await self.extractor.extract_battery_data(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract battery data: {e}")
            data['battery'] = pd.DataFrame()
            
        try:
            self.logger.info("Extracting EV data...")
            data['ev'] = await self.extractor.extract_ev_data(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract EV data: {e}")
            data['ev'] = pd.DataFrame()
            
        return data
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {}
        
        # Check each data source
        for source_name, source_data in data.items():
            if source_name == 'rooms' and isinstance(source_data, dict):
                # Special handling for room data
                room_quality = {}
                for room_name, room_df in source_data.items():
                    if isinstance(room_df, pd.DataFrame) and not room_df.empty:
                        room_quality[room_name] = {
                            'records': len(room_df),
                            'missing_pct': room_df.isnull().sum().sum() / room_df.size * 100,
                            'date_range': (room_df.index.min(), room_df.index.max()),
                            'columns': list(room_df.columns)
                        }
                quality_report[source_name] = room_quality
                
            elif source_name == 'relay_states' and isinstance(source_data, dict):
                # Special handling for relay states
                relay_quality = {}
                for room_name, relay_df in source_data.items():
                    if isinstance(relay_df, pd.DataFrame) and not relay_df.empty:
                        relay_quality[room_name] = {
                            'records': len(relay_df),
                            'duty_cycle': (relay_df.iloc[:, 0] > 0).mean() * 100,
                            'switches': relay_df.iloc[:, 0].diff().abs().sum()
                        }
                quality_report[source_name] = relay_quality
                
            elif isinstance(source_data, pd.DataFrame):
                if not source_data.empty:
                    quality_report[source_name] = {
                        'records': len(source_data),
                        'missing_pct': source_data.isnull().sum().sum() / source_data.size * 100,
                        'date_range': (source_data.index.min(), source_data.index.max()),
                        'columns': list(source_data.columns)
                    }
                else:
                    quality_report[source_name] = {'status': 'empty'}
            elif source_data is None:
                quality_report[source_name] = {'status': 'not_available'}
                
        return quality_report
    
    async def _analyze_pv_export_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PV system with export constraints."""
        self.logger.info("Analyzing PV export system...")
        
        # Detect export policy change
        export_analysis = self.pv_analyzer._identify_export_periods(
            data['pv'], data.get('prices', pd.DataFrame())
        )
        
        if 'policy_change_date' in export_analysis:
            policy_date = export_analysis['policy_change_date']
            
            # Analyze pre and post export periods
            pre_export = self.pv_analyzer._analyze_self_consumption_period(
                data['pv'], policy_date
            )
            post_export = self.pv_analyzer._analyze_conditional_export_period(
                data['pv'], data.get('prices', pd.DataFrame()), policy_date
            )
            
            # Calculate optimization potential
            optimization = self.pv_analyzer._calculate_optimization_potential(
                data['pv'], data.get('prices', pd.DataFrame()), policy_date
            )
            
            return {
                'export_policy': export_analysis,
                'pre_export_period': pre_export,
                'post_export_period': post_export,
                'optimization_potential': optimization,
                'recommendations': self._generate_pv_recommendations(
                    pre_export, post_export, optimization
                )
            }
        else:
            # Standard PV analysis if no export policy detected
            return self.pv_analyzer.analyze_pv_production(data['pv'], data['weather'])
    
    async def _analyze_relay_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relay system optimization opportunities."""
        self.logger.info("Analyzing relay optimization...")
        
        # Run comprehensive relay analysis
        relay_results = self.relay_analyzer.analyze_relay_patterns(
            data['relay_states'],
            data.get('weather'),
            data.get('prices')
        )
        
        # Add room-specific power consumption analysis
        room_consumption = {}
        for room_name, relay_df in data['relay_states'].items():
            if not relay_df.empty and 'power_kw' in relay_df.columns:
                room_consumption[room_name] = {
                    'total_energy_kwh': (relay_df['power_kw'] * 0.25).sum(),
                    'avg_power_kw': relay_df['power_kw'].mean(),
                    'peak_power_kw': relay_df['power_kw'].max(),
                    'duty_cycle_pct': (relay_df['relay_state'] > 0).mean() * 100
                }
        
        relay_results['room_consumption'] = room_consumption
        
        return relay_results
    
    async def _analyze_thermal_dynamics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal dynamics with Loxone data structure."""
        self.logger.info("Analyzing thermal dynamics...")
        
        # Process room data considering Loxone field naming
        processed_rooms = {}
        for room_name, room_df in data['rooms'].items():
            if room_df.empty:
                continue
                
            # Find temperature column (might be named like "temperature_obyvak")
            temp_cols = [col for col in room_df.columns if 'temperature' in col.lower()]
            if temp_cols:
                # Create standardized DataFrame
                processed_df = pd.DataFrame(index=room_df.index)
                processed_df['temperature'] = room_df[temp_cols[0]]
                
                # Add humidity if available
                humidity_cols = [col for col in room_df.columns if 'humidity' in col.lower()]
                if humidity_cols:
                    processed_df['humidity'] = room_df[humidity_cols[0]]
                
                # Add relay state if available
                if room_name in data['relay_states']:
                    relay_df = data['relay_states'][room_name]
                    if not relay_df.empty:
                        # Align relay data with temperature data
                        processed_df['heating_on'] = relay_df['relay_state'].reindex(
                            processed_df.index, method='nearest'
                        )
                
                processed_rooms[room_name] = processed_df
        
        # Run thermal analysis with processed data
        return self.thermal_analyzer.analyze_room_dynamics(
            processed_rooms, 
            data.get('weather', pd.DataFrame())
        )
    
    async def _analyze_base_load(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze base load patterns."""
        self.logger.info("Analyzing base load...")
        
        return self.base_load_analyzer.analyze_base_load(
            data['consumption'],
            data['pv'],
            data['rooms'],
            data.get('ev'),
            data.get('battery')
        )
    
    async def _analyze_weather_correlations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze weather correlations with energy patterns."""
        self.logger.info("Analyzing weather correlations...")
        
        correlations = {}
        
        # Merge weather data sources
        weather_combined = data['weather'].copy()
        if not data['current_weather'].empty:
            # Add Loxone weather data
            for col in ['sun_elevation', 'sun_direction', 'absolute_solar_irradiance']:
                if col in data['current_weather'].columns:
                    weather_combined[col] = data['current_weather'][col]
        
        # PV-weather correlation
        if not data['pv'].empty and not weather_combined.empty:
            pv_weather = pd.merge(
                data['pv'][['InputPower']], 
                weather_combined,
                left_index=True, 
                right_index=True, 
                how='inner'
            )
            
            if not pv_weather.empty:
                correlations['pv_weather'] = pv_weather.corr()['InputPower'].to_dict()
        
        # Heating-weather correlation
        if data['relay_states']:
            total_heating = pd.Series(0, index=weather_combined.index)
            for room_name, relay_df in data['relay_states'].items():
                if not relay_df.empty and 'power_w' in relay_df.columns:
                    room_power = relay_df['power_w'].reindex(
                        weather_combined.index, method='nearest'
                    ).fillna(0)
                    total_heating += room_power
            
            if total_heating.sum() > 0:
                heating_weather = pd.DataFrame({
                    'heating_power': total_heating,
                    'outdoor_temp': weather_combined.get('temperature', 0)
                }).dropna()
                
                if not heating_weather.empty:
                    correlations['heating_weather'] = {
                        'temperature_correlation': heating_weather.corr().iloc[0, 1],
                        'heating_response': heating_weather.groupby(
                            pd.cut(heating_weather['outdoor_temp'], bins=10)
                        )['heating_power'].mean().to_dict()
                    }
        
        return correlations
    
    async def _analyze_economic_opportunities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze economic optimization opportunities."""
        self.logger.info("Analyzing economic opportunities...")
        
        if data['prices'] is None or data['prices'].empty:
            return {'status': 'no_price_data'}
        
        opportunities = {}
        
        # Load shifting potential
        if not data['consumption'].empty:
            consumption_price = pd.merge(
                data['consumption'][['total_consumption']],
                data['prices'][['price_czk_kwh']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if not consumption_price.empty:
                # Calculate current costs
                consumption_price['cost'] = (
                    consumption_price['total_consumption'] * 
                    consumption_price['price_czk_kwh'] * 0.25 / 1000
                )
                
                # Identify high and low price periods
                price_quantiles = consumption_price['price_czk_kwh'].quantile([0.25, 0.75])
                
                high_price_consumption = consumption_price[
                    consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                ]['total_consumption'].sum()
                
                low_price_avg = consumption_price[
                    consumption_price['price_czk_kwh'] < price_quantiles[0.25]
                ]['price_czk_kwh'].mean()
                
                high_price_avg = consumption_price[
                    consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                ]['price_czk_kwh'].mean()
                
                # Calculate potential savings from load shifting
                shift_potential = high_price_consumption * 0.3  # Assume 30% can be shifted
                potential_savings = shift_potential * (high_price_avg - low_price_avg) * 0.25 / 1000
                
                opportunities['load_shifting'] = {
                    'shiftable_energy_kwh': shift_potential * 0.25 / 1000,
                    'potential_savings_czk': potential_savings,
                    'high_price_periods': consumption_price[
                        consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                    ].index.hour.value_counts().to_dict(),
                    'low_price_periods': consumption_price[
                        consumption_price['price_czk_kwh'] < price_quantiles[0.25]
                    ].index.hour.value_counts().to_dict()
                }
        
        # PV export optimization
        if not data['pv'].empty and 'ExportPower' in data['pv'].columns:
            export_price = pd.merge(
                data['pv'][['ExportPower']],
                data['prices'][['price_czk_kwh']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if not export_price.empty:
                # Calculate export revenue
                export_price['revenue'] = (
                    export_price['ExportPower'] * 
                    export_price['price_czk_kwh'] * 0.25 / 1000000
                )
                
                opportunities['export_optimization'] = {
                    'total_export_kwh': export_price['ExportPower'].sum() * 0.25 / 1000,
                    'total_revenue_czk': export_price['revenue'].sum(),
                    'avg_export_price': export_price[
                        export_price['ExportPower'] > 0
                    ]['price_czk_kwh'].mean() if (export_price['ExportPower'] > 0).any() else 0,
                    'missed_high_price_exports': export_price[
                        (export_price['price_czk_kwh'] > price_quantiles[0.75]) &
                        (export_price['ExportPower'] == 0)
                    ].shape[0]
                }
        
        return opportunities
    
    def _generate_integrated_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated insights from all analyses."""
        insights = {
            'key_findings': [],
            'optimization_priorities': [],
            'system_health': {},
            'recommended_actions': []
        }
        
        # System health metrics
        if 'data_quality' in results:
            total_records = sum(
                v.get('records', 0) for k, v in results['data_quality'].items()
                if isinstance(v, dict) and 'records' in v
            )
            insights['system_health']['data_completeness'] = min(total_records / 100000, 1.0) * 100
        
        # Key findings from each analysis
        if 'pv_export_analysis' in results:
            pv = results['pv_export_analysis']
            if 'optimization_potential' in pv:
                opt = pv['optimization_potential']
                if opt.get('curtailed_energy_kwh', 0) > 100:
                    insights['key_findings'].append({
                        'category': 'PV System',
                        'finding': f"Significant curtailment detected: {opt['curtailed_energy_kwh']:.0f} kWh",
                        'impact': 'HIGH'
                    })
        
        if 'relay_optimization' in results:
            relay = results['relay_optimization']
            if 'peak_demand' in relay:
                peak = relay['peak_demand']
                if peak.get('estimated_reduction_potential_kw', 0) > 2:
                    insights['optimization_priorities'].append({
                        'priority': 1,
                        'action': 'Implement relay coordination',
                        'potential_savings': f"{peak['estimated_reduction_potential_kw']:.1f} kW peak reduction"
                    })
        
        # Generate recommended actions based on findings
        insights['recommended_actions'] = self._generate_recommendations(results)
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        # PV recommendations
        if 'pv_export_analysis' in results:
            pv = results['pv_export_analysis']
            if 'pre_export_period' in pv:
                pre = pv['pre_export_period']
                if pre.get('curtailment_ratio', 0) > 0.1:
                    recommendations.append({
                        'category': 'PV System',
                        'priority': 'HIGH',
                        'action': 'Install battery storage',
                        'rationale': f"Capture {pre.get('estimated_curtailment_kwh', 0):.0f} kWh of curtailed energy",
                        'estimated_roi': '3-5 years'
                    })
        
        # Heating recommendations
        if 'relay_optimization' in results:
            relay = results['relay_optimization']
            if 'coordination' in relay:
                coord = relay['coordination']
                if coord.get('total_coordination_opportunities', 0) > 5:
                    recommendations.append({
                        'category': 'Heating System',
                        'priority': 'MEDIUM',
                        'action': 'Implement relay coordination algorithm',
                        'rationale': f"Reduce simultaneous operation of {coord['total_coordination_opportunities']} room pairs",
                        'estimated_savings': '10-15% reduction in peak demand'
                    })
        
        # Economic recommendations
        if 'economic_optimization' in results:
            econ = results['economic_optimization']
            if 'load_shifting' in econ:
                shift = econ['load_shifting']
                if shift.get('potential_savings_czk', 0) > 1000:
                    recommendations.append({
                        'category': 'Energy Management',
                        'priority': 'HIGH',
                        'action': 'Implement time-of-use load shifting',
                        'rationale': f"Shift {shift['shiftable_energy_kwh']:.0f} kWh to low-price periods",
                        'estimated_savings': f"{shift['potential_savings_czk']:.0f} CZK annually"
                    })
        
        return sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']])
    
    def _generate_pv_recommendations(self, pre_export, post_export, optimization):
        """Generate specific PV system recommendations."""
        recommendations = []"""
Comprehensive PEMS v2 Analysis Pipeline
Optimized for Loxone relay-based heating system with PV export constraints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

from analysis.data_extraction import DataExtractor
from analysis.pattern_analysis import PVAnalyzer, RelayPatternAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from analysis.base_load_analysis import BaseLoadAnalyzer
from analysis.feature_engineering import FeatureEngineer
from analysis.visualization import AnalysisVisualizer
from config.settings import PEMSSettings

class ComprehensiveAnalyzer:
    """Enhanced analyzer for Loxone-based PEMS system."""
    
    def __init__(self, settings: PEMSSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.extractor = DataExtractor(settings)
        self.pv_analyzer = PVAnalyzer()
        self.relay_analyzer = RelayPatternAnalyzer()
        self.thermal_analyzer = ThermalAnalyzer()
        self.base_load_analyzer = BaseLoadAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.visualizer = AnalysisVisualizer()
        
    async def run_comprehensive_analysis(
        self, 
        start_date: datetime, 
        end_date: datetime,
        analysis_types: list = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis with all available data sources.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            analysis_types: List of analysis to run (default: all)
        """
        if analysis_types is None:
            analysis_types = [
                'pv_export', 'relay_optimization', 'thermal_dynamics',
                'base_load', 'weather_correlation', 'economic_optimization'
            ]
            
        self.logger.info(f"Starting comprehensive analysis from {start_date} to {end_date}")
        
        # Step 1: Extract all data
        self.logger.info("=== Step 1: Data Extraction ===")
        data = await self._extract_all_data(start_date, end_date)
        
        # Step 2: Data quality assessment
        self.logger.info("=== Step 2: Data Quality Assessment ===")
        quality_report = self._assess_data_quality(data)
        
        # Step 3: Run selected analyses
        self.logger.info("=== Step 3: Running Analyses ===")
        results = {
            'data_quality': quality_report,
            'analysis_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            }
        }
        
        if 'pv_export' in analysis_types and not data['pv'].empty:
            results['pv_export_analysis'] = await self._analyze_pv_export_system(data)
            
        if 'relay_optimization' in analysis_types and data['relay_states']:
            results['relay_optimization'] = await self._analyze_relay_optimization(data)
            
        if 'thermal_dynamics' in analysis_types and data['rooms']:
            results['thermal_dynamics'] = await self._analyze_thermal_dynamics(data)
            
        if 'base_load' in analysis_types and not data['consumption'].empty:
            results['base_load_analysis'] = await self._analyze_base_load(data)
            
        if 'weather_correlation' in analysis_types:
            results['weather_correlation'] = await self._analyze_weather_correlations(data)
            
        if 'economic_optimization' in analysis_types and data['prices'] is not None:
            results['economic_optimization'] = await self._analyze_economic_opportunities(data)
        
        # Step 4: Generate integrated insights
        self.logger.info("=== Step 4: Generating Integrated Insights ===")
        results['integrated_insights'] = self._generate_integrated_insights(results)
        
        # Step 5: Create visualizations
        self.logger.info("=== Step 5: Creating Visualizations ===")
        self._create_comprehensive_visualizations(data, results)
        
        # Step 6: Generate report
        self.logger.info("=== Step 6: Generating Report ===")
        self._generate_comprehensive_report(results)
        
        return results
    
    async def _extract_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Extract all available data sources."""
        data = {}
        
        # Core data sources
        self.logger.info("Extracting PV data...")
        data['pv'] = await self.extractor.extract_pv_data(start_date, end_date)
        
        self.logger.info("Extracting room temperature data...")
        data['rooms'] = await self.extractor.extract_room_temperatures(start_date, end_date)
        
        self.logger.info("Extracting weather data...")
        data['weather'] = await self.extractor.extract_weather_data(start_date, end_date)
        
        self.logger.info("Extracting current weather from Loxone...")
        data['current_weather'] = await self.extractor.extract_current_weather(start_date, end_date)
        
        self.logger.info("Extracting consumption data...")
        data['consumption'] = await self.extractor.extract_energy_consumption(start_date, end_date)
        
        self.logger.info("Extracting relay states...")
        data['relay_states'] = await self.extractor.extract_relay_states(start_date, end_date)
        
        # Optional data sources
        try:
            self.logger.info("Extracting energy prices...")
            data['prices'] = await self.extractor.extract_energy_prices(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract prices: {e}")
            data['prices'] = None
            
        try:
            self.logger.info("Extracting battery data...")
            data['battery'] = await self.extractor.extract_battery_data(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract battery data: {e}")
            data['battery'] = pd.DataFrame()
            
        try:
            self.logger.info("Extracting EV data...")
            data['ev'] = await self.extractor.extract_ev_data(start_date, end_date)
        except Exception as e:
            self.logger.warning(f"Could not extract EV data: {e}")
            data['ev'] = pd.DataFrame()
            
        return data
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {}
        
        # Check each data source
        for source_name, source_data in data.items():
            if source_name == 'rooms' and isinstance(source_data, dict):
                # Special handling for room data
                room_quality = {}
                for room_name, room_df in source_data.items():
                    if isinstance(room_df, pd.DataFrame) and not room_df.empty:
                        room_quality[room_name] = {
                            'records': len(room_df),
                            'missing_pct': room_df.isnull().sum().sum() / room_df.size * 100,
                            'date_range': (room_df.index.min(), room_df.index.max()),
                            'columns': list(room_df.columns)
                        }
                quality_report[source_name] = room_quality
                
            elif source_name == 'relay_states' and isinstance(source_data, dict):
                # Special handling for relay states
                relay_quality = {}
                for room_name, relay_df in source_data.items():
                    if isinstance(relay_df, pd.DataFrame) and not relay_df.empty:
                        relay_quality[room_name] = {
                            'records': len(relay_df),
                            'duty_cycle': (relay_df.iloc[:, 0] > 0).mean() * 100,
                            'switches': relay_df.iloc[:, 0].diff().abs().sum()
                        }
                quality_report[source_name] = relay_quality
                
            elif isinstance(source_data, pd.DataFrame):
                if not source_data.empty:
                    quality_report[source_name] = {
                        'records': len(source_data),
                        'missing_pct': source_data.isnull().sum().sum() / source_data.size * 100,
                        'date_range': (source_data.index.min(), source_data.index.max()),
                        'columns': list(source_data.columns)
                    }
                else:
                    quality_report[source_name] = {'status': 'empty'}
            elif source_data is None:
                quality_report[source_name] = {'status': 'not_available'}
                
        return quality_report
    
    async def _analyze_pv_export_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PV system with export constraints."""
        self.logger.info("Analyzing PV export system...")
        
        # Detect export policy change
        export_analysis = self.pv_analyzer._identify_export_periods(
            data['pv'], data.get('prices', pd.DataFrame())
        )
        
        if 'policy_change_date' in export_analysis:
            policy_date = export_analysis['policy_change_date']
            
            # Analyze pre and post export periods
            pre_export = self.pv_analyzer._analyze_self_consumption_period(
                data['pv'], policy_date
            )
            post_export = self.pv_analyzer._analyze_conditional_export_period(
                data['pv'], data.get('prices', pd.DataFrame()), policy_date
            )
            
            # Calculate optimization potential
            optimization = self.pv_analyzer._calculate_optimization_potential(
                data['pv'], data.get('prices', pd.DataFrame()), policy_date
            )
            
            return {
                'export_policy': export_analysis,
                'pre_export_period': pre_export,
                'post_export_period': post_export,
                'optimization_potential': optimization,
                'recommendations': self._generate_pv_recommendations(
                    pre_export, post_export, optimization
                )
            }
        else:
            # Standard PV analysis if no export policy detected
            return self.pv_analyzer.analyze_pv_production(data['pv'], data['weather'])
    
    async def _analyze_relay_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relay system optimization opportunities."""
        self.logger.info("Analyzing relay optimization...")
        
        # Run comprehensive relay analysis
        relay_results = self.relay_analyzer.analyze_relay_patterns(
            data['relay_states'],
            data.get('weather'),
            data.get('prices')
        )
        
        # Add room-specific power consumption analysis
        room_consumption = {}
        for room_name, relay_df in data['relay_states'].items():
            if not relay_df.empty and 'power_kw' in relay_df.columns:
                room_consumption[room_name] = {
                    'total_energy_kwh': (relay_df['power_kw'] * 0.25).sum(),
                    'avg_power_kw': relay_df['power_kw'].mean(),
                    'peak_power_kw': relay_df['power_kw'].max(),
                    'duty_cycle_pct': (relay_df['relay_state'] > 0).mean() * 100
                }
        
        relay_results['room_consumption'] = room_consumption
        
        return relay_results
    
    async def _analyze_thermal_dynamics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal dynamics with Loxone data structure."""
        self.logger.info("Analyzing thermal dynamics...")
        
        # Process room data considering Loxone field naming
        processed_rooms = {}
        for room_name, room_df in data['rooms'].items():
            if room_df.empty:
                continue
                
            # Find temperature column (might be named like "temperature_obyvak")
            temp_cols = [col for col in room_df.columns if 'temperature' in col.lower()]
            if temp_cols:
                # Create standardized DataFrame
                processed_df = pd.DataFrame(index=room_df.index)
                processed_df['temperature'] = room_df[temp_cols[0]]
                
                # Add humidity if available
                humidity_cols = [col for col in room_df.columns if 'humidity' in col.lower()]
                if humidity_cols:
                    processed_df['humidity'] = room_df[humidity_cols[0]]
                
                # Add relay state if available
                if room_name in data['relay_states']:
                    relay_df = data['relay_states'][room_name]
                    if not relay_df.empty:
                        # Align relay data with temperature data
                        processed_df['heating_on'] = relay_df['relay_state'].reindex(
                            processed_df.index, method='nearest'
                        )
                
                processed_rooms[room_name] = processed_df
        
        # Run thermal analysis with processed data
        return self.thermal_analyzer.analyze_room_dynamics(
            processed_rooms, 
            data.get('weather', pd.DataFrame())
        )
    
    async def _analyze_base_load(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze base load patterns."""
        self.logger.info("Analyzing base load...")
        
        return self.base_load_analyzer.analyze_base_load(
            data['consumption'],
            data['pv'],
            data['rooms'],
            data.get('ev'),
            data.get('battery')
        )
    
    async def _analyze_weather_correlations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze weather correlations with energy patterns."""
        self.logger.info("Analyzing weather correlations...")
        
        correlations = {}
        
        # Merge weather data sources
        weather_combined = data['weather'].copy()
        if not data['current_weather'].empty:
            # Add Loxone weather data
            for col in ['sun_elevation', 'sun_direction', 'absolute_solar_irradiance']:
                if col in data['current_weather'].columns:
                    weather_combined[col] = data['current_weather'][col]
        
        # PV-weather correlation
        if not data['pv'].empty and not weather_combined.empty:
            pv_weather = pd.merge(
                data['pv'][['InputPower']], 
                weather_combined,
                left_index=True, 
                right_index=True, 
                how='inner'
            )
            
            if not pv_weather.empty:
                correlations['pv_weather'] = pv_weather.corr()['InputPower'].to_dict()
        
        # Heating-weather correlation
        if data['relay_states']:
            total_heating = pd.Series(0, index=weather_combined.index)
            for room_name, relay_df in data['relay_states'].items():
                if not relay_df.empty and 'power_w' in relay_df.columns:
                    room_power = relay_df['power_w'].reindex(
                        weather_combined.index, method='nearest'
                    ).fillna(0)
                    total_heating += room_power
            
            if total_heating.sum() > 0:
                heating_weather = pd.DataFrame({
                    'heating_power': total_heating,
                    'outdoor_temp': weather_combined.get('temperature', 0)
                }).dropna()
                
                if not heating_weather.empty:
                    correlations['heating_weather'] = {
                        'temperature_correlation': heating_weather.corr().iloc[0, 1],
                        'heating_response': heating_weather.groupby(
                            pd.cut(heating_weather['outdoor_temp'], bins=10)
                        )['heating_power'].mean().to_dict()
                    }
        
        return correlations
    
    async def _analyze_economic_opportunities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze economic optimization opportunities."""
        self.logger.info("Analyzing economic opportunities...")
        
        if data['prices'] is None or data['prices'].empty:
            return {'status': 'no_price_data'}
        
        opportunities = {}
        
        # Load shifting potential
        if not data['consumption'].empty:
            consumption_price = pd.merge(
                data['consumption'][['total_consumption']],
                data['prices'][['price_czk_kwh']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if not consumption_price.empty:
                # Calculate current costs
                consumption_price['cost'] = (
                    consumption_price['total_consumption'] * 
                    consumption_price['price_czk_kwh'] * 0.25 / 1000
                )
                
                # Identify high and low price periods
                price_quantiles = consumption_price['price_czk_kwh'].quantile([0.25, 0.75])
                
                high_price_consumption = consumption_price[
                    consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                ]['total_consumption'].sum()
                
                low_price_avg = consumption_price[
                    consumption_price['price_czk_kwh'] < price_quantiles[0.25]
                ]['price_czk_kwh'].mean()
                
                high_price_avg = consumption_price[
                    consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                ]['price_czk_kwh'].mean()
                
                # Calculate potential savings from load shifting
                shift_potential = high_price_consumption * 0.3  # Assume 30% can be shifted
                potential_savings = shift_potential * (high_price_avg - low_price_avg) * 0.25 / 1000
                
                opportunities['load_shifting'] = {
                    'shiftable_energy_kwh': shift_potential * 0.25 / 1000,
                    'potential_savings_czk': potential_savings,
                    'high_price_periods': consumption_price[
                        consumption_price['price_czk_kwh'] > price_quantiles[0.75]
                    ].index.hour.value_counts().to_dict(),
                    'low_price_periods': consumption_price[
                        consumption_price['price_czk_kwh'] < price_quantiles[0.25]
                    ].index.hour.value_counts().to_dict()
                }
        
        # PV export optimization
        if not data['pv'].empty and 'ExportPower' in data['pv'].columns:
            export_price = pd.merge(
                data['pv'][['ExportPower']],
                data['prices'][['price_czk_kwh']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if not export_price.empty:
                # Calculate export revenue
                export_price['revenue'] = (
                    export_price['ExportPower'] * 
                    export_price['price_czk_kwh'] * 0.25 / 1000000
                )
                
                opportunities['export_optimization'] = {
                    'total_export_kwh': export_price['ExportPower'].sum() * 0.25 / 1000,
                    'total_revenue_czk': export_price['revenue'].sum(),
                    'avg_export_price': export_price[
                        export_price['ExportPower'] > 0
                    ]['price_czk_kwh'].mean() if (export_price['ExportPower'] > 0).any() else 0,
                    'missed_high_price_exports': export_price[
                        (export_price['price_czk_kwh'] > price_quantiles[0.75]) &
                        (export_price['ExportPower'] == 0)
                    ].shape[0]
                }
        
        return opportunities
    
    def _generate_integrated_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated insights from all analyses."""
        insights = {
            'key_findings': [],
            'optimization_priorities': [],
            'system_health': {},
            'recommended_actions': []
        }
        
        # System health metrics
        if 'data_quality' in results:
            total_records = sum(
                v.get('records', 0) for k, v in results['data_quality'].items()
                if isinstance(v, dict) and 'records' in v
            )
            insights['system_health']['data_completeness'] = min(total_records / 100000, 1.0) * 100
        
        # Key findings from each analysis
        if 'pv_export_analysis' in results:
            pv = results['pv_export_analysis']
            if 'optimization_potential' in pv:
                opt = pv['optimization_potential']
                if opt.get('curtailed_energy_kwh', 0) > 100:
                    insights['key_findings'].append({
                        'category': 'PV System',
                        'finding': f"Significant curtailment detected: {opt['curtailed_energy_kwh']:.0f} kWh",
                        'impact': 'HIGH'
                    })
        
        if 'relay_optimization' in results:
            relay = results['relay_optimization']
            if 'peak_demand' in relay:
                peak = relay['peak_demand']
                if peak.get('estimated_reduction_potential_kw', 0) > 2:
                    insights['optimization_priorities'].append({
                        'priority': 1,
                        'action': 'Implement relay coordination',
                        'potential_savings': f"{peak['estimated_reduction_potential_kw']:.1f} kW peak reduction"
                    })
        
        # Generate recommended actions based on findings
        insights['recommended_actions'] = self._generate_recommendations(results)
        
        return insights
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate actionable recommendations."""
        recommendations = []
        
        # PV recommendations
        if 'pv_export_analysis' in results:
            pv = results['pv_export_analysis']
            if 'pre_export_period' in pv:
                pre = pv['pre_export_period']
                if pre.get('curtailment_ratio', 0) > 0.1:
                    recommendations.append({
                        'category': 'PV System',
                        'priority': 'HIGH',
                        'action': 'Install battery storage',
                        'rationale': f"Capture {pre.get('estimated_curtailment_kwh', 0):.0f} kWh of curtailed energy",
                        'estimated_roi': '3-5 years'
                    })
        
        # Heating recommendations
        if 'relay_optimization' in results:
            relay = results['relay_optimization']
            if 'coordination' in relay:
                coord = relay['coordination']
                if coord.get('total_coordination_opportunities', 0) > 5:
                    recommendations.append({
                        'category': 'Heating System',
                        'priority': 'MEDIUM',
                        'action': 'Implement relay coordination algorithm',
                        'rationale': f"Reduce simultaneous operation of {coord['total_coordination_opportunities']} room pairs",
                        'estimated_savings': '10-15% reduction in peak demand'
                    })
        
        # Economic recommendations
        if 'economic_optimization' in results:
            econ = results['economic_optimization']
            if 'load_shifting' in econ:
                shift = econ['load_shifting']
                if shift.get('potential_savings_czk', 0) > 1000:
                    recommendations.append({
                        'category': 'Energy Management',
                        'priority': 'HIGH',
                        'action': 'Implement time-of-use load shifting',
                        'rationale': f"Shift {shift['shiftable_energy_kwh']:.0f} kWh to low-price periods",
                        'estimated_savings': f"{shift['potential_savings_czk']:.0f} CZK annually"
                    })
        
        return sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']])
    
    def _generate_pv_recommendations(self, pre_export, post_export, optimization):
        """Generate specific PV system recommendations."""
        recommendations = []
        
        # Curtailment recommendations
        if pre_export.get('curtailment_ratio', 0) > 0.05:
            recommendations.append(
                f"High curtailment detected ({pre_export['curtailment_ratio']:.1%}). "
                f"Consider battery storage to capture {pre_export.get('estimated_curtailment_kwh', 0):.0f} kWh."
            )
        
        # Export optimization
        if post_export.get('export_when_low_price_ratio', 0) > 0.2:
            recommendations.append(
                f"Optimize export timing: {post_export['export_when_low_price_ratio']:.1%} "
                f"of exports occur during low prices. Implement price-aware export control."
            )
        
        # Storage value
        if optimization.get('storage_value_potential_czk', 0) > 10000:
            recommendations.append(
                f"Battery storage could provide {optimization['storage_value_potential_czk']:.0f} CZK "
                f"annual value from curtailment recovery."
            )
        
        return recommendations
    
    def _create_comprehensive_visualizations(self, data: Dict[str, Any], results: Dict[str, Any]):
        """Create all visualization dashboards."""
        output_dir = Path("analysis/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # PV Dashboard
        if not data['pv'].empty:
            self.visualizer.plot_pv_analysis_dashboard(
                data['pv'],
                results.get('pv_export_analysis', {}),
                data.get('prices'),
                results.get('pv_export_analysis', {}).get('export_policy', {}).get('policy_change_date'),
                save_path=output_dir / "pv_analysis_dashboard.html"
            )
        
        # Relay Dashboard
        if data['relay_states']:
            # Combine relay data into single DataFrame
            relay_combined = pd.DataFrame()
            for room_name, relay_df in data['relay_states'].items():
                if not relay_df.empty:
                    relay_combined[f"{room_name}_relay"] = relay_df['relay_state']
                    relay_combined[f"{room_name}_power"] = relay_df.get('power_kw', 0)
            
            if not relay_combined.empty:
                self.visualizer.plot_relay_patterns(
                    relay_combined,
                    results.get('relay_optimization', {}),
                    save_path=output_dir / "relay_patterns_dashboard.html"
                )
        
        # Thermal Dashboard
        if data['rooms'] and 'thermal_dynamics' in results:
            self.visualizer.plot_thermal_analysis(
                data['rooms'],
                results['thermal_dynamics'],
                save_path=output_dir / "thermal_analysis_dashboard.html"
            )
        
        # Economic Dashboard
        if 'economic_optimization' in results and results['economic_optimization'] != {'status': 'no_price_data'}:
            # Create economic summary visualization
            self.visualizer.create_analysis_summary_report(
                results,
                save_path=output_dir / "economic_summary.html"
            )
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive text and HTML reports."""
        report_dir = Path("analysis/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        report_lines = [
            "=" * 80,
            "PEMS V2 COMPREHENSIVE ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {results['analysis_period']['start']} to {results['analysis_period']['end']}",
            f"Duration: {results['analysis_period']['days']} days",
            "",
            "DATA QUALITY SUMMARY",
            "-" * 40,
        ]
        
        # Add data quality details
        for source, quality in results['data_quality'].items():
            if isinstance(quality, dict) and 'records' in quality:
                report_lines.append(f"{source}: {quality['records']:,} records ({quality.get('missing_pct', 0):.1f}% missing)")
            elif isinstance(quality, dict) and source in ['rooms', 'relay_states']:
                report_lines.append(f"{source}: {len(quality)} entities with data")
        
        # Add analysis results
        for analysis_type in ['pv_export_analysis', 'relay_optimization', 'thermal_dynamics', 
                             'base_load_analysis', 'economic_optimization']:
            if analysis_type in results:
                report_lines.extend(["", analysis_type.upper().replace('_', ' '), "-" * 40])
                report_lines.extend(self._format_analysis_results(analysis_type, results[analysis_type]))
        
        # Add integrated insights
        if 'integrated_insights' in results:
            insights = results['integrated_insights']
            report_lines.extend(["", "INTEGRATED INSIGHTS", "-" * 40])
            
            if 'key_findings' in insights:
                report_lines.append("Key Findings:")
                for finding in insights['key_findings']:
                    report_lines.append(f"  • [{finding['impact']}] {finding['finding']}")
            
            if 'recommended_actions' in insights:
                report_lines.append("\nRecommended Actions:")
                for i, action in enumerate(insights['recommended_actions'], 1):
                    report_lines.append(f"  {i}. [{action['priority']}] {action['action']}")
                    report_lines.append(f"     Rationale: {action['rationale']}")
                    if 'estimated_savings' in action:
                        report_lines.append(f"     Estimated Savings: {action['estimated_savings']}")
        
        report_lines.extend(["", "=" * 80, "End of Report"])
        
        # Save text report
        with open(report_dir / "comprehensive_analysis_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        
        # Generate HTML report
        html_report = self.visualizer.create_analysis_summary_report(
            results,
            save_path=report_dir / "comprehensive_analysis_report.html"
        )
        
        self.logger.info(f"Reports saved to {report_dir}")
    
    def _format_analysis_results(self, analysis_type: str, results: Dict[str, Any]) -> list:
        """Format analysis results for text report."""
        lines = []
        
        if analysis_type == 'pv_export_analysis' and 'optimization_potential' in results:
            opt = results['optimization_potential']
            lines.extend([
                f"Curtailed Energy: {opt.get('curtailed_energy_kwh', 0):.0f} kWh",
                f"Lost Revenue: {opt.get('lost_revenue_curtailment_czk', 0):.0f} CZK",
                f"Storage Value Potential: {opt.get('storage_value_potential_czk', 0):.0f} CZK"
            ])
        
        elif analysis_type == 'relay_optimization' and 'peak_demand' in results:
            peak = results['peak_demand']
            lines.extend([
                f"Peak Demand: {peak.get('max_peak_kw', 0):.1f} kW",
                f"Average Demand: {peak.get('avg_peak_kw', 0):.1f} kW",
                f"Reduction Potential: {peak.get('estimated_reduction_potential_kw', 0):.1f} kW"
            ])
        
        elif analysis_type == 'economic_optimization' and 'load_shifting' in results:
            shift = results['load_shifting']
            lines.extend([
                f"Shiftable Energy: {shift.get('shiftable_energy_kwh', 0):.0f} kWh",
                f"Potential Savings: {shift.get('potential_savings_czk', 0):.0f} CZK"
            ])
        
        return lines


async def main():
    """Run comprehensive analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load settings
    settings = PEMSSettings()
    
    # Create analyzer
    analyzer = ComprehensiveAnalyzer(settings)
    
    # Define analysis period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    # Run comprehensive analysis
    results = await analyzer.run_comprehensive_analysis(
        start_date, 
        end_date,
        analysis_types=['pv_export', 'relay_optimization', 'thermal_dynamics', 
                       'base_load', 'weather_correlation', 'economic_optimization']
    )
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to analysis/reports/")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
        
        # Curtailment recommendations
        if pre_export.get('curtailment_ratio', 0) > 0.05:
            recommendations.append(
                f"High curtailment detected ({pre_export['curtailment_ratio']:.1%}). "
                f"Consider battery storage to capture {pre_export.get('estimated_curtailment_kwh', 0):.0f} kWh."
            )
        
        # Export optimization
        if post_export.get('export_when_low_price_ratio', 0) > 0.2:
            recommendations.append(
                f"Optimize export timing: {post_export['export_when_low_price_ratio']:.1%} "
                f"of exports occur during low prices. Implement price-aware export control."
            )
        
        # Storage value
        if optimization.get('storage_value_potential_czk', 0) > 10000:
            recommendations.append(
                f"Battery storage could provide {optimization['storage_value_potential_czk']:.0f} CZK "
                f"annual value from curtailment recovery."
            )
        
        return recommendations
    
    def _create_comprehensive_visualizations(self, data: Dict[str, Any], results: Dict[str, Any]):
        """Create all visualization dashboards."""
        output_dir = Path("analysis/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # PV Dashboard
        if not data['pv'].empty:
            self.visualizer.plot_pv_analysis_dashboard(
                data['pv'],
                results.get('pv_export_analysis', {}),
                data.get('prices'),
                results.get('pv_export_analysis', {}).get('export_policy', {}).get('policy_change_date'),
                save_path=output_dir / "pv_analysis_dashboard.html"
            )
        
        # Relay Dashboard
        if data['relay_states']:
            # Combine relay data into single DataFrame
            relay_combined = pd.DataFrame()
            for room_name, relay_df in data['relay_states'].items():
                if not relay_df.empty:
                    relay_combined[f"{room_name}_relay"] = relay_df['relay_state']
                    relay_combined[f"{room_name}_power"] = relay_df.get('power_kw', 0)
            
            if not relay_combined.empty:
                self.visualizer.plot_relay_patterns(
                    relay_combined,
                    results.get('relay_optimization', {}),
                    save_path=output_dir / "relay_patterns_dashboard.html"
                )
        
        # Thermal Dashboard
        if data['rooms'] and 'thermal_dynamics' in results:
            self.visualizer.plot_thermal_analysis(
                data['rooms'],
                results['thermal_dynamics'],
                save_path=output_dir / "thermal_analysis_dashboard.html"
            )
        
        # Economic Dashboard
        if 'economic_optimization' in results and results['economic_optimization'] != {'status': 'no_price_data'}:
            # Create economic summary visualization
            self.visualizer.create_analysis_summary_report(
                results,
                save_path=output_dir / "economic_summary.html"
            )
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive text and HTML reports."""
        report_dir = Path("analysis/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        report_lines = [
            "=" * 80,
            "PEMS V2 COMPREHENSIVE ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {results['analysis_period']['start']} to {results['analysis_period']['end']}",
            f"Duration: {results['analysis_period']['days']} days",
            "",
            "DATA QUALITY SUMMARY",
            "-" * 40,
        ]
        
        # Add data quality details
        for source, quality in results['data_quality'].items():
            if isinstance(quality, dict) and 'records' in quality:
                report_lines.append(f"{source}: {quality['records']:,} records ({quality.get('missing_pct', 0):.1f}% missing)")
            elif isinstance(quality, dict) and source in ['rooms', 'relay_states']:
                report_lines.append(f"{source}: {len(quality)} entities with data")
        
        # Add analysis results
        for analysis_type in ['pv_export_analysis', 'relay_optimization', 'thermal_dynamics', 
                             'base_load_analysis', 'economic_optimization']:
            if analysis_type in results:
                report_lines.extend(["", analysis_type.upper().replace('_', ' '), "-" * 40])
                report_lines.extend(self._format_analysis_results(analysis_type, results[analysis_type]))
        
        # Add integrated insights
        if 'integrated_insights' in results:
            insights = results['integrated_insights']
            report_lines.extend(["", "INTEGRATED INSIGHTS", "-" * 40])
            
            if 'key_findings' in insights:
                report_lines.append("Key Findings:")
                for finding in insights['key_findings']:
                    report_lines.append(f"  • [{finding['impact']}] {finding['finding']}")
            
            if 'recommended_actions' in insights:
                report_lines.append("\nRecommended Actions:")
                for i, action in enumerate(insights['recommended_actions'], 1):
                    report_lines.append(f"  {i}. [{action['priority']}] {action['action']}")
                    report_lines.append(f"     Rationale: {action['rationale']}")
                    if 'estimated_savings' in action:
                        report_lines.append(f"     Estimated Savings: {action['estimated_savings']}")
        
        report_lines.extend(["", "=" * 80, "End of Report"])
        
        # Save text report
        with open(report_dir / "comprehensive_analysis_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        
        # Generate HTML report
        html_report = self.visualizer.create_analysis_summary_report(
            results,
            save_path=report_dir / "comprehensive_analysis_report.html"
        )
        
        self.logger.info(f"Reports saved to {report_dir}")
    
    def _format_analysis_results(self, analysis_type: str, results: Dict[str, Any]) -> list:
        """Format analysis results for text report."""
        lines = []
        
        if analysis_type == 'pv_export_analysis' and 'optimization_potential' in results:
            opt = results['optimization_potential']
            lines.extend([
                f"Curtailed Energy: {opt.get('curtailed_energy_kwh', 0):.0f} kWh",
                f"Lost Revenue: {opt.get('lost_revenue_curtailment_czk', 0):.0f} CZK",
                f"Storage Value Potential: {opt.get('storage_value_potential_czk', 0):.0f} CZK"
            ])
        
        elif analysis_type == 'relay_optimization' and 'peak_demand' in results:
            peak = results['peak_demand']
            lines.extend([
                f"Peak Demand: {peak.get('max_peak_kw', 0):.1f} kW",
                f"Average Demand: {peak.get('avg_peak_kw', 0):.1f} kW",
                f"Reduction Potential: {peak.get('estimated_reduction_potential_kw', 0):.1f} kW"
            ])
        
        elif analysis_type == 'economic_optimization' and 'load_shifting' in results:
            shift = results['load_shifting']
            lines.extend([
                f"Shiftable Energy: {shift.get('shiftable_energy_kwh', 0):.0f} kWh",
                f"Potential Savings: {shift.get('potential_savings_czk', 0):.0f} CZK"
            ])
        
        return lines


async def main():
    """Run comprehensive analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load settings
    settings = PEMSSettings()
    
    # Create analyzer
    analyzer = ComprehensiveAnalyzer(settings)
    
    # Define analysis period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    # Run comprehensive analysis
    results = await analyzer.run_comprehensive_analysis(
        start_date, 
        end_date,
        analysis_types=['pv_export', 'relay_optimization', 'thermal_dynamics', 
                       'base_load', 'weather_correlation', 'economic_optimization']
    )
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to analysis/reports/")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
```

### A. Data Pipeline Optimization

1. **Use the DB documentation files** you provided to ensure correct field names:
   - `LOXONE_BUCKET.md` shows exact field names for temperatures, relays, etc.
   - `SOLAR_BUCKET.md` shows PV system fields
   - `OTE_PRICES_BUCKET.md` for price data structure

2. **Handle Loxone naming conventions**:
   ```python
   # Temperature fields: temperature_obyvak, temperature_kuchyne, etc.
   # Relay fields: obyvak, kuchyne, etc. (with tag1='heating')
   # Humidity fields: humidity_obyvak, humidity_kuchyne, etc.
   ```

### B. Analysis Improvements

3. **Enhance PV Export Analysis**:
   - Track export events vs price thresholds
   - Calculate actual vs potential export revenue
   - Identify curtailment patterns before export was enabled

4. **Optimize Relay Analysis**:
   - Use actual room power ratings from your config
   - Analyze switching patterns to reduce wear
   - Identify coordination opportunities between rooms

5. **Improve Thermal Modeling**:
   - Use relay states as ground truth for heating periods
   - Correlate with outdoor temperature from Loxone
   - Account for solar gains using `sun_elevation` and `absolute_solar_irradiance`

### C. Key Analysis Workflows

6. **Daily Analysis Routine**:
   ```python
   # Quick daily check
   analyzer = ComprehensiveAnalyzer(settings)
   results = await analyzer.run_comprehensive_analysis(
       datetime.now() - timedelta(days=1),
       datetime.now(),
       analysis_types=['pv_export', 'relay_optimization']
   )
   ```

7. **Seasonal Comparison**:
   ```python
   # Compare winter vs summer patterns
   winter_results = await analyzer.run_comprehensive_analysis(
       datetime(2024, 1, 1),
       datetime(2024, 3, 31),
       analysis_types=['thermal_dynamics', 'relay_optimization']
   )
   
   summer_results = await analyzer.run_comprehensive_analysis(
       datetime(2024, 6, 1),
       datetime(2024, 8, 31),
       analysis_types=['pv_export', 'base_load']
   )
   ```

## 3. **Critical Updates for Your System**

### Update room power ratings everywhere to use `config/energy_settings.py`:

### Update analysis scripts to use correct field names:
```python
# In pattern_analysis.py, thermal_analysis.py, etc.
def process_loxone_data(self, room_df):
    # Map Loxone field names to standard names
    field_mapping = {
        'temperature': ['temperature_', 'temp_'],
        'humidity': ['humidity_'],
        'target_temp': ['target_temp'],
        'relay': ['relay_state', 'heating_on']
    }
    
    standardized_df = pd.DataFrame(index=room_df.index)
    for standard_name, patterns in field_mapping.items():
        for col in room_df.columns:
            if any(pattern in col.lower() for pattern in patterns):
                standardized_df[standard_name] = room_df[col]
                break
    
    return standardized_df
```

## 4. **Running the Enhanced Analysis**

1. **Create a new analysis script** using the comprehensive analyzer:
   ```bash
   python -m analysis.comprehensive_analysis
   ```

2. **Use the Makefile commands** you already have:
   ```bash
   make clean-analysis
   make run-relay-analysis
   ```

3. **Monitor the outputs** in:
   - `analysis/figures/` - Interactive dashboards
   - `analysis/reports/` - Detailed reports
   - `analysis/results/` - JSON data for further processing

The comprehensive analyzer I've created will:
- Properly handle your Loxone data structure
- Detect and analyze your PV export policy changes
- Optimize relay coordination for your 16-room system
- Generate actionable insights based on your specific setup
- Create visualizations tailored to your system configuration

Would you like me to create any specific analysis scripts or help you implement particular optimizations based on your system?


Looking at your existing analysis scripts, here's what I recommend keeping, modifying, or removing:

## 🟢 **KEEP & ENHANCE** (Core Components)

### 1. **`data_extraction.py`** ✅
- **Status**: Already updated and working well
- **Action**: Keep as-is, it's your data foundation

### 2. **`pattern_analysis.py`** ✅
- **Status**: Contains valuable PV and relay analysis logic
- **Action**: Keep but ensure it uses the Loxone field naming:
```python
# Add a method to handle Loxone field mapping
def _get_loxone_field(self, df: pd.DataFrame, field_type: str) -> Optional[str]:
    """Find Loxone field name for a given type."""
    if field_type == 'temperature':
        for col in df.columns:
            if 'temperature_' in col.lower():
                return col
    # etc...
```

### 3. **`thermal_analysis.py`** ✅
- **Status**: Good RC modeling, especially `estimate_rc_parameters()`
- **Action**: Keep but update the `_merge_room_weather_data()` to handle Loxone naming

### 4. **`base_load_analysis.py`** ✅
- **Status**: Useful for understanding non-controllable loads
- **Action**: Keep, it properly subtracts controllable loads

### 5. **`feature_engineering.py`** ✅
- **Status**: Essential for ML models in Phase 2
- **Action**: Keep, already handles relay features well

### 6. **`visualization.py`** ✅
- **Status**: Comprehensive plotting functions
- **Action**: Keep all methods, they're well-designed

## 🟡 **MODIFY** (Needs Updates)

### 1. **`run_analysis.py`**
- **Current**: Generic pipeline
- **Action**: Replace with the `ComprehensiveAnalyzer` I provided, or update to:
```python
class AnalysisPipeline:
    def _merge_room_weather_data(self, room_df, weather_data):
        # Update to handle Loxone field names
        temp_cols = [col for col in room_df.columns if 'temperature_' in col]
        if temp_cols:
            temp_col = temp_cols[0]
            # ... rest of logic
```

### 2. **`data_preprocessing.py`**
- **Current**: Generic preprocessing
- **Action**: Add Loxone-specific field handling:
```python
def process_loxone_room_data(self, room_df: pd.DataFrame, room_name: str) -> pd.DataFrame:
    """Process Loxone room data with correct field mapping."""
    processed = pd.DataFrame(index=room_df.index)
    
    # Map Loxone fields
    if f'temperature_{room_name}' in room_df.columns:
        processed['temperature'] = room_df[f'temperature_{room_name}']
    if f'humidity_{room_name}' in room_df.columns:
        processed['humidity'] = room_df[f'humidity_{room_name}']
    
    return processed
```

## 🔴 **CONSIDER REMOVING** (Redundant/Test Files)

None of your analysis files should be removed - they all serve specific purposes. However, you might want to:

1. **Consolidate test scripts** into the main analysis modules
2. **Archive old notebooks** that are superseded by the new analysis

## 📋 **Recommended File Structure**

```
pems_v2/analysis/
├── core/
│   ├── data_extraction.py      # ✅ Keep as-is
│   ├── data_preprocessing.py   # 🟡 Add Loxone handlers
│   └── visualization.py        # ✅ Keep as-is
│
├── analyzers/
│   ├── pattern_analysis.py     # ✅ Keep, minor updates
│   ├── thermal_analysis.py     # ✅ Keep, minor updates
│   ├── base_load_analysis.py   # ✅ Keep as-is
│   └── relay_analysis.py       # 🆕 Extract from pattern_analysis
│
├── ml/
│   └── feature_engineering.py  # ✅ Keep for Phase 2
│
├── pipelines/
│   ├── comprehensive_analysis.py # 🆕 New integrated analyzer
│   └── run_analysis.py          # 🟡 Update or replace
│
└── reports/
    └── report_generator.py      # 🆕 Extract from visualization.py
```

## 🚀 **Action Plan**

1. **First**, create a simple adapter module to handle Loxone field mapping:

```python
# analysis/utils/loxone_adapter.py
"""Adapter for Loxone data field naming conventions."""

class LoxoneFieldAdapter:
    """Maps Loxone field names to standard names."""
    
    @staticmethod
    def standardize_room_data(room_df: pd.DataFrame, room_name: str) -> pd.DataFrame:
        """Convert Loxone fields to standard names."""
        mapping = {
            'temperature': f'temperature_{room_name}',
            'humidity': f'humidity_{room_name}',
            'target_temp': 'target_temp',
            'relay_state': room_name  # Relay fields use room name directly
        }
        
        standardized = pd.DataFrame(index=room_df.index)
        for standard_name, loxone_name in mapping.items():
            if loxone_name in room_df.columns:
                standardized[standard_name] = room_df[loxone_name]
            # Also check for variations
            elif standard_name == 'temperature':
                temp_cols = [col for col in room_df.columns if 'temperature' in col.lower()]
                if temp_cols:
                    standardized[standard_name] = room_df[temp_cols[0]]
        
        return standardized
```

2. **Then**, update existing analyzers to use the adapter:

```python
# In thermal_analysis.py
from analysis.utils.loxone_adapter import LoxoneFieldAdapter

def _analyze_single_room(self, room_df: pd.DataFrame, weather_data: pd.DataFrame, room_name: str):
    # Standardize Loxone data
    room_df = LoxoneFieldAdapter.standardize_room_data(room_df, room_name)
    
    # Now use standard field names
    if 'temperature' in room_df.columns:
        # ... existing logic works
```

3. **Finally**, use the comprehensive analyzer for daily runs:

```python
# analysis/daily_analysis.py
from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer

async def run_daily_analysis():
    analyzer = ComprehensiveAnalyzer(settings)
    results = await analyzer.run_comprehensive_analysis(
        datetime.now() - timedelta(days=1),
        datetime.now(),
        analysis_types=['pv_export', 'relay_optimization', 'economic_optimization']
    )
    return results
```

## 📊 **Summary**

**Keep all your existing analysis modules** - they contain valuable logic. The main changes needed are:

1. Add Loxone field mapping/adaptation layer
2. Update data merging functions to handle Loxone naming
3. Use the ComprehensiveAnalyzer for integrated analysis
4. Keep all visualization and feature engineering as-is

Your existing code is well-structured; it just needs adaptation for the Loxone field naming conventions. The comprehensive analyzer I provided can orchestrate all your existing analyzers while handling the Loxone-specific details.


Looking at your existing analysis scripts, here's what I recommend keeping, modifying, or removing:

## 🟢 **KEEP & ENHANCE** (Core Components)

### 1. **`data_extraction.py`** ✅
- **Status**: Already updated and working well
- **Action**: Keep as-is, it's your data foundation

### 2. **`pattern_analysis.py`** ✅
- **Status**: Contains valuable PV and relay analysis logic
- **Action**: Keep but ensure it uses the Loxone field naming:
```python
# Add a method to handle Loxone field mapping
def _get_loxone_field(self, df: pd.DataFrame, field_type: str) -> Optional[str]:
    """Find Loxone field name for a given type."""
    if field_type == 'temperature':
        for col in df.columns:
            if 'temperature_' in col.lower():
                return col
    # etc...
```

### 3. **`thermal_analysis.py`** ✅
- **Status**: Good RC modeling, especially `estimate_rc_parameters()`
- **Action**: Keep but update the `_merge_room_weather_data()` to handle Loxone naming

### 4. **`base_load_analysis.py`** ✅
- **Status**: Useful for understanding non-controllable loads
- **Action**: Keep, it properly subtracts controllable loads

### 5. **`feature_engineering.py`** ✅
- **Status**: Essential for ML models in Phase 2
- **Action**: Keep, already handles relay features well

### 6. **`visualization.py`** ✅
- **Status**: Comprehensive plotting functions
- **Action**: Keep all methods, they're well-designed

## 🟡 **MODIFY** (Needs Updates)

### 1. **`run_analysis.py`**
- **Current**: Generic pipeline
- **Action**: Replace with the `ComprehensiveAnalyzer` I provided, or update to:
```python
class AnalysisPipeline:
    def _merge_room_weather_data(self, room_df, weather_data):
        # Update to handle Loxone field names
        temp_cols = [col for col in room_df.columns if 'temperature_' in col]
        if temp_cols:
            temp_col = temp_cols[0]
            # ... rest of logic
```

### 2. **`data_preprocessing.py`**
- **Current**: Generic preprocessing
- **Action**: Add Loxone-specific field handling:
```python
def process_loxone_room_data(self, room_df: pd.DataFrame, room_name: str) -> pd.DataFrame:
    """Process Loxone room data with correct field mapping."""
    processed = pd.DataFrame(index=room_df.index)
    
    # Map Loxone fields
    if f'temperature_{room_name}' in room_df.columns:
        processed['temperature'] = room_df[f'temperature_{room_name}']
    if f'humidity_{room_name}' in room_df.columns:
        processed['humidity'] = room_df[f'humidity_{room_name}']
    
    return processed
```

## 🔴 **CONSIDER REMOVING** (Redundant/Test Files)

None of your analysis files should be removed - they all serve specific purposes. However, you might want to:

1. **Consolidate test scripts** into the main analysis modules
2. **Archive old notebooks** that are superseded by the new analysis

## 📋 **Recommended File Structure**

```
pems_v2/analysis/
├── core/
│   ├── data_extraction.py      # ✅ Keep as-is
│   ├── data_preprocessing.py   # 🟡 Add Loxone handlers
│   └── visualization.py        # ✅ Keep as-is
│
├── analyzers/
│   ├── pattern_analysis.py     # ✅ Keep, minor updates
│   ├── thermal_analysis.py     # ✅ Keep, minor updates
│   ├── base_load_analysis.py   # ✅ Keep as-is
│   └── relay_analysis.py       # 🆕 Extract from pattern_analysis
│
├── ml/
│   └── feature_engineering.py  # ✅ Keep for Phase 2
│
├── pipelines/
│   ├── comprehensive_analysis.py # 🆕 New integrated analyzer
│   └── run_analysis.py          # 🟡 Update or replace
│
└── reports/
    └── report_generator.py      # 🆕 Extract from visualization.py
```

## 🚀 **Action Plan**

1. **First**, create a simple adapter module to handle Loxone field mapping:

```python
# analysis/utils/loxone_adapter.py
"""Adapter for Loxone data field naming conventions."""

class LoxoneFieldAdapter:
    """Maps Loxone field names to standard names."""
    
    @staticmethod
    def standardize_room_data(room_df: pd.DataFrame, room_name: str) -> pd.DataFrame:
        """Convert Loxone fields to standard names."""
        mapping = {
            'temperature': f'temperature_{room_name}',
            'humidity': f'humidity_{room_name}',
            'target_temp': 'target_temp',
            'relay_state': room_name  # Relay fields use room name directly
        }
        
        standardized = pd.DataFrame(index=room_df.index)
        for standard_name, loxone_name in mapping.items():
            if loxone_name in room_df.columns:
                standardized[standard_name] = room_df[loxone_name]
            # Also check for variations
            elif standard_name == 'temperature':
                temp_cols = [col for col in room_df.columns if 'temperature' in col.lower()]
                if temp_cols:
                    standardized[standard_name] = room_df[temp_cols[0]]
        
        return standardized
```

2. **Then**, update existing analyzers to use the adapter:

```python
# In thermal_analysis.py
from analysis.utils.loxone_adapter import LoxoneFieldAdapter

def _analyze_single_room(self, room_df: pd.DataFrame, weather_data: pd.DataFrame, room_name: str):
    # Standardize Loxone data
    room_df = LoxoneFieldAdapter.standardize_room_data(room_df, room_name)
    
    # Now use standard field names
    if 'temperature' in room_df.columns:
        # ... existing logic works
```

3. **Finally**, use the comprehensive analyzer for daily runs:

```python
# analysis/daily_analysis.py
from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer

async def run_daily_analysis():
    analyzer = ComprehensiveAnalyzer(settings)
    results = await analyzer.run_comprehensive_analysis(
        datetime.now() - timedelta(days=1),
        datetime.now(),
        analysis_types=['pv_export', 'relay_optimization', 'economic_optimization']
    )
    return results
```

## 📊 **Summary**

**Keep all your existing analysis modules** - they contain valuable logic. The main changes needed are:

1. Add Loxone field mapping/adaptation layer
2. Update data merging functions to handle Loxone naming
3. Use the ComprehensiveAnalyzer for integrated analysis
4. Keep all visualization and feature engineering as-is

Your existing code is well-structured; it just needs adaptation for the Loxone field naming conventions. The comprehensive analyzer I provided can orchestrate all your existing analyzers while handling the Loxone-specific details.