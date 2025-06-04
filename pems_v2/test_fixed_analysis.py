#!/usr/bin/env python3
"""
Test the fixed PEMS v2 analysis with actual Loxone data structure.
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

from analysis.data_extraction import DataExtractor
from analysis.pattern_analysis import PVAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from analysis.base_load_analysis import BaseLoadAnalyzer


class LoxoneDataExtractor:
    """Data extractor that uses actual Loxone data structure."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.logger = logging.getLogger(f"{__name__}.LoxoneDataExtractor")
        
        # InfluxDB configuration
        self.url = "http://192.168.0.201:8086"
        self.token = "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A=="
        self.org = "loxone"
        self.bucket_loxone = "loxone"
        self.bucket_solar = "solar"
        
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=30000)
        self.query_api = self.client.query_api()
        
        # Room heating power mapping from your example
        self.heating_power = {
            "hosti": 2.02, "chodba_dole": 1.8, "chodba_nahore": 1.2, "koupelna_dole": 0.47, 
            "koupelna_nahore": 0.62, "loznice": 1.2, "obyvak": 4.8, "pokoj_1": 1.2, 
            "pokoj_2": 1.2, "pracovna": 0.82, "satna_dole": 0.82, "satna_nahore": 0.56,
            "spajz": 0.46, "technicka_mistnost": 0.82, "zadveri": 0.82, "zachod": 0.22
        }
    
    def __del__(self):
        """Close InfluxDB client on cleanup."""
        if hasattr(self, 'client'):
            self.client.close()
    
    async def extract_room_temperatures(self, start_date: datetime, end_date: datetime) -> dict:
        """Extract room temperature data using actual Loxone structure."""
        self.logger.info(f"Extracting room temperature data from {start_date} to {end_date}")
        
        # Use the exact query structure from your example
        query = f'''
        from(bucket: "{self.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No temperature data found")
            return {}
        
        # Group data by room (extracted from field name)
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
        
        # Convert to DataFrames with proper timezone
        room_dataframes = {}
        for room_name, data_list in room_data.items():
            if data_list:
                df = pd.DataFrame(data_list)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df.set_index('timestamp', inplace=True)
                room_dataframes[room_name] = df
                self.logger.info(f"Extracted {len(df)} temperature points for room {room_name}")
        
        return room_dataframes
    
    async def extract_heating_consumption(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract heating consumption using relay data like in your example."""
        self.logger.info(f"Extracting heating consumption from {start_date} to {end_date}")
        
        # Use your exact relay query structure
        query = f'''
        from(bucket: "{self.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating" and r.room != "kuchyne")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "room"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No heating relay data found")
            return pd.DataFrame()
        
        # Process data similar to your example
        consumption_data = []
        for table in tables:
            for record in table.records:
                room = record.values.get('room', 'unknown')
                if room in self.heating_power:
                    # Calculate power consumption: relay_state * room_power_kW
                    power_consumption = record.get_value() * self.heating_power[room] * 1000  # Convert to W
                    
                    consumption_data.append({
                        'timestamp': record.get_time(),
                        'room': room,
                        'power_w': power_consumption
                    })
        
        if not consumption_data:
            self.logger.warning("No consumption data after processing")
            return pd.DataFrame()
        
        # Create DataFrame and aggregate by time
        df = pd.DataFrame(consumption_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Sum power across all rooms for total consumption
        total_consumption = df.groupby('timestamp')['power_w'].sum().reset_index()
        total_consumption.set_index('timestamp', inplace=True)
        total_consumption.columns = ['total_heating_consumption']
        
        self.logger.info(f"Extracted {len(total_consumption)} heating consumption points")
        return total_consumption
    
    async def extract_outdoor_weather(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Extract outdoor temperature and wind from solar bucket."""
        self.logger.info(f"Extracting outdoor weather from {start_date} to {end_date}")
        
        # Use your solar bucket query structure
        query = f'''
        from(bucket: "{self.bucket_solar}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "teplomer" and r["topic"] == "teplomer/TC")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> keep(columns: ["topic", "_time", "_value"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No outdoor weather data found in solar bucket")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'outdoor_temperature': record.get_value()
                })
        
        if records:
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
            self.logger.info(f"Extracted {len(df)} outdoor weather points")
            return df
        
        return pd.DataFrame()


async def test_fixed_analysis():
    """Test the fixed analysis with actual Loxone data structure."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create data extractor
        logger.info("Creating Loxone data extractor...")
        extractor = LoxoneDataExtractor()
        
        # Test date range - last 3 days for faster testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        logger.info(f"Testing analysis with data from {start_date} to {end_date}")
        
        # Extract real data
        logger.info("\n" + "="*50)
        logger.info("EXTRACTING REAL LOXONE DATA")
        logger.info("="*50)
        
        # Extract room temperature data
        room_data = await extractor.extract_room_temperatures(start_date, end_date)
        logger.info(f"Extracted temperature data for {len(room_data)} rooms")
        
        # Extract heating consumption data
        heating_data = await extractor.extract_heating_consumption(start_date, end_date)
        logger.info(f"Heating consumption data shape: {heating_data.shape}")
        
        # Extract outdoor weather data
        weather_data = await extractor.extract_outdoor_weather(start_date, end_date)
        logger.info(f"Outdoor weather data shape: {weather_data.shape}")
        
        # Test Thermal Analysis with real data
        logger.info("\n" + "="*50)
        logger.info("TESTING THERMAL ANALYSIS WITH REAL DATA")
        logger.info("="*50)
        
        if room_data:
            thermal_analyzer = ThermalAnalyzer()
            
            try:
                thermal_results = thermal_analyzer.analyze_room_dynamics(room_data, weather_data)
                logger.info("✓ Thermal analysis with real data completed successfully!")
                
                # Show sample results for a few rooms
                sample_rooms = list(thermal_results.keys())[:5]  # First 5 rooms
                for room_name in sample_rooms:
                    if room_name in thermal_results and room_name != 'room_coupling':
                        room_results = thermal_results[room_name]
                        if isinstance(room_results, dict) and 'basic_stats' in room_results:
                            stats = room_results['basic_stats']
                            logger.info(f"  {room_name}: Mean {stats.get('mean_temperature', 0):.1f}°C, "
                                      f"Range {stats.get('temperature_range', 0):.1f}°C")
                
            except Exception as e:
                logger.error(f"✗ Thermal analysis failed: {e}")
        else:
            logger.warning("✗ Skipping thermal analysis - no room data")
        
        # Test Base Load Analysis with real heating data
        logger.info("\n" + "="*50)
        logger.info("TESTING BASE LOAD ANALYSIS WITH REAL HEATING DATA")
        logger.info("="*50)
        
        if not heating_data.empty:
            try:
                base_load_analyzer = BaseLoadAnalyzer()
                
                base_load_results = base_load_analyzer.analyze_base_load(
                    heating_data,  # Use real heating consumption
                    pd.DataFrame(),  # No PV data yet
                    room_data
                )
                
                logger.info("✓ Base load analysis with real data completed successfully!")
                
                if 'basic_stats' in base_load_results:
                    stats = base_load_results['basic_stats']
                    logger.info(f"  Mean base load: {stats.get('mean_base_load', 0):.1f} W")
                    logger.info(f"  Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
                
            except Exception as e:
                logger.error(f"✗ Base load analysis failed: {e}")
        else:
            logger.warning("✗ Skipping base load analysis - no heating data")
        
        # Create some sample PV data for PV analysis test
        logger.info("\n" + "="*50)
        logger.info("TESTING PV ANALYSIS WITH SAMPLE DATA")
        logger.info("="*50)
        
        try:
            pv_analyzer = PVAnalyzer()
            
            # Create sample PV data with proper timezone
            pv_index = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
            sample_pv = pd.DataFrame(
                index=pv_index,
                data={'solar_power': [max(0, 2000 * abs(((i % 96) - 48) / 48)) for i in range(len(pv_index))]}
            )
            
            pv_results = pv_analyzer.analyze_pv_production(sample_pv, weather_data)
            
            logger.info("✓ PV analysis with timezone-fixed data completed successfully!")
            
            if 'basic_stats' in pv_results:
                stats = pv_results['basic_stats']
                logger.info(f"  Total energy: {stats.get('total_energy_kwh', 0):.1f} kWh")
                logger.info(f"  Max power: {stats.get('max_power', 0):.1f} W")
            
        except Exception as e:
            logger.error(f"✗ PV analysis failed: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("FIXED ANALYSIS TESTING COMPLETED")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_fixed_analysis())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")