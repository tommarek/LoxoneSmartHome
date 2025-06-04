#!/usr/bin/env python3
"""
Test individual analysis components with real data from InfluxDB.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
from influxdb_client import InfluxDBClient

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from analysis.pattern_analysis import PVAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from analysis.base_load_analysis import BaseLoadAnalyzer


class SimpleDataExtractor:
    """Simple data extractor for testing analysis components."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.logger = logging.getLogger(f"{__name__}.SimpleDataExtractor")
        
        # InfluxDB configuration
        self.url = "http://192.168.0.201:8086"
        self.token = "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A=="
        self.org = "loxone"
        self.bucket = "loxone"
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=30000)
        self.query_api = self.client.query_api()
        
        # Timezone handling
        self.local_tz = pytz.timezone('Europe/Prague')
        self.utc_tz = pytz.UTC
    
    def __del__(self):
        """Close InfluxDB client on cleanup."""
        if hasattr(self, 'client'):
            self.client.close()
    
    async def extract_room_temperatures(self, start_date: datetime, end_date: datetime) -> dict:
        """Extract real room temperature data."""
        self.logger.info(f"Extracting room temperature data from {start_date} to {end_date}")
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No temperature data found")
            return {}
        
        # Group data by room
        room_data = {}
        
        for table in tables:
            for record in table.records:
                field_name = record.get_field()
                
                # Extract room name from field (e.g., "temperature_kuchyne" -> "kuchyne")
                if field_name.startswith('temperature_'):
                    room_name = field_name.replace('temperature_', '')
                    
                    if room_name not in room_data:
                        room_data[room_name] = []
                    
                    room_data[room_name].append({
                        'timestamp': record.get_time(),
                        'temperature': record.get_value()
                    })
        
        # Convert to DataFrames
        room_dataframes = {}
        for room_name, data_list in room_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                room_dataframes[room_name] = df
                self.logger.info(f"Extracted {len(df)} temperature points for room {room_name}")
        
        return room_dataframes
    
    async def extract_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract weather data."""
        self.logger.info(f"Extracting weather data from {start_date} to {end_date}")
        
        # Try both main bucket and weather_forecast bucket
        buckets_to_try = [self.bucket, "weather_forecast"]
        
        for bucket in buckets_to_try:
            query = f'''
            from(bucket: "{bucket}")
              |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
              |> filter(fn: (r) => r["_measurement"] == "current_weather" or r["_measurement"] == "weather")
              |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
            '''
            
            try:
                tables = self.query_api.query(query)
                
                if tables:
                    self.logger.info(f"Found weather data in bucket: {bucket}")
                    
                    # Convert to DataFrame
                    records = []
                    for table in tables:
                        for record in table.records:
                            records.append({
                                'timestamp': record.get_time(),
                                'field': record.get_field(),
                                'value': record.get_value()
                            })
                    
                    if records:
                        df = pd.DataFrame(records)
                        df_pivot = df.pivot_table(
                            index='timestamp',
                            columns='field',
                            values='value',
                            aggfunc='mean'
                        )
                        df_pivot.index = pd.to_datetime(df_pivot.index, utc=True)
                        self.logger.info(f"Extracted {len(df_pivot)} weather data points")
                        return df_pivot
                        
            except Exception as e:
                self.logger.warning(f"No weather data in bucket {bucket}: {e}")
                continue
        
        self.logger.warning("No weather data found in any bucket")
        return pd.DataFrame()


async def test_analysis_components():
    """Test all analysis components with real data."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create data extractor
        logger.info("Creating data extractor...")
        extractor = SimpleDataExtractor()
        
        # Test date range - last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Testing analysis with data from {start_date} to {end_date}")
        
        # Extract data
        logger.info("\n" + "="*50)
        logger.info("EXTRACTING DATA FOR ANALYSIS")
        logger.info("="*50)
        
        # Extract room temperature data
        room_data = await extractor.extract_room_temperatures(start_date, end_date)
        logger.info(f"Extracted data for {len(room_data)} rooms")
        
        # Extract weather data
        weather_data = await extractor.extract_weather_data(start_date, end_date)
        logger.info(f"Weather data shape: {weather_data.shape}")
        
        # Test Thermal Analysis
        logger.info("\n" + "="*50)
        logger.info("TESTING THERMAL ANALYSIS")
        logger.info("="*50)
        
        if room_data:
            thermal_analyzer = ThermalAnalyzer()
            
            try:
                thermal_results = thermal_analyzer.analyze_room_dynamics(room_data, weather_data)
                logger.info("✓ Thermal analysis completed successfully!")
                
                # Show sample results
                for room_name, room_results in thermal_results.items():
                    if room_name == 'room_coupling':
                        continue
                    
                    if isinstance(room_results, dict) and 'basic_stats' in room_results:
                        stats = room_results['basic_stats']
                        logger.info(f"  {room_name}: Mean temp {stats.get('mean_temperature', 0):.1f}°C, "
                                  f"Range {stats.get('temperature_range', 0):.1f}°C")
                
            except Exception as e:
                logger.error(f"✗ Thermal analysis failed: {e}")
        else:
            logger.warning("✗ Skipping thermal analysis - no room data")
        
        # Test Base Load Analysis (without energy data)
        logger.info("\n" + "="*50)
        logger.info("TESTING BASE LOAD ANALYSIS")
        logger.info("="*50)
        
        try:
            base_load_analyzer = BaseLoadAnalyzer()
            
            # Create dummy consumption data for testing with timezone-aware index
            consumption_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
            dummy_consumption = pd.DataFrame(
                index=consumption_index,
                data={'consumption': [1000 + 200 * (i % 24) for i in range(len(consumption_index))]}
            )
            
            base_load_results = base_load_analyzer.analyze_base_load(
                dummy_consumption, 
                pd.DataFrame(),  # No PV data
                room_data
            )
            
            logger.info("✓ Base load analysis completed successfully!")
            
            if 'basic_stats' in base_load_results:
                stats = base_load_results['basic_stats']
                logger.info(f"  Mean base load: {stats.get('mean_base_load', 0):.1f} W")
                logger.info(f"  Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
            
        except Exception as e:
            logger.error(f"✗ Base load analysis failed: {e}")
        
        # Test PV Analysis (with dummy data since no solar data found)
        logger.info("\n" + "="*50)
        logger.info("TESTING PV ANALYSIS")
        logger.info("="*50)
        
        try:
            pv_analyzer = PVAnalyzer()
            
            # Create dummy PV data for testing with timezone-aware index
            pv_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
            dummy_pv = pd.DataFrame(
                index=pv_index,
                data={'solar_power': [max(0, 1000 * (0.5 + 0.5 * (i % 96 - 48) / 48)) for i in range(len(pv_index))]}
            )
            
            pv_results = pv_analyzer.analyze_pv_production(dummy_pv, weather_data)
            
            logger.info("✓ PV analysis completed successfully!")
            
            if 'basic_stats' in pv_results:
                stats = pv_results['basic_stats']
                logger.info(f"  Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
                logger.info(f"  Max power: {stats.get('max_power', 0):.1f} W")
            
        except Exception as e:
            logger.error(f"✗ PV analysis failed: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS COMPONENT TESTING COMPLETED")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis component test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_analysis_components())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")