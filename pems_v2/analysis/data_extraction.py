"""
Data extraction module for PEMS v2.

Extracts historical data from InfluxDB for analysis:
- Query 2 years of data for PV production, room temperatures, weather, etc.
- Save as parquet files for fast analysis
- Implement data quality checks
- Handle missing data interpolation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytz
from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable

from config.settings import PEMSSettings as Settings


class DataExtractor:
    """Extract and process historical data from InfluxDB."""

    def __init__(self, settings: Settings):
        """Initialize the data extractor."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.DataExtractor")
        
        # InfluxDB client
        self.client = InfluxDBClient(
            url=settings.influxdb.url,
            token=settings.influxdb.token.get_secret_value(),
            org=settings.influxdb.org,
            timeout=30000
        )
        self.query_api = self.client.query_api()
        
        # Timezone handling
        self.local_tz = pytz.timezone('Europe/Prague')
        self.utc_tz = pytz.UTC
        
        # Data output directory
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        """Close InfluxDB client on cleanup."""
        if hasattr(self, 'client'):
            self.client.close()

    async def extract_pv_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract PV production data from InfluxDB.
        
        Returns DataFrame with columns:
        - timestamp: datetime index
        - power_W: instantaneous power in watts
        - energy_Wh: energy production per interval
        """
        self.logger.info(f"Extracting PV data from {start_date} to {end_date}")
        
        # Query for PV production data from solar bucket (like the temperature example from "solar" bucket)
        query = f'''
        from(bucket: "solar")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "teplomer" or r["_measurement"] == "solar" or r["_measurement"] == "power")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field", "_measurement"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No PV data found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'measurement': record.get_measurement(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'tags': {k: v for k, v in record.values.items() if k.startswith('tag_') or k in ['room', 'location']}
                })
        
        if not records:
            self.logger.warning("No PV records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Pivot to get power values as columns
        df_pivot = df.pivot_table(
            index='timestamp', 
            columns=['measurement', 'field'], 
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Flatten column names
        df_pivot.columns = [f"{col[0]}_{col[1]}" if col[0] and col[1] else 'timestamp' for col in df_pivot.columns]
        
        # Ensure timestamp is datetime and localized
        df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        df_pivot.set_index('timestamp', inplace=True)
        
        # Calculate energy from power (15-minute intervals)
        power_cols = [col for col in df_pivot.columns if 'power' in col.lower()]
        for col in power_cols:
            energy_col = col.replace('power', 'energy')
            df_pivot[energy_col] = df_pivot[col] * 0.25  # 15 minutes = 0.25 hours
        
        self.logger.info(f"Extracted {len(df_pivot)} PV data points")
        return df_pivot

    async def extract_room_temperatures(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Extract room temperature data from InfluxDB.
        
        Returns dict of DataFrames by room name with columns:
        - timestamp: datetime index  
        - temperature: current temperature
        - setpoint: target temperature (if available)
        - heating_on: boolean heating status
        """
        self.logger.info(f"Extracting room temperature data from {start_date} to {end_date}")
        
        # Query for temperature data
        query = f'''
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature" or r["_measurement"] == "heating")
          |> filter(fn: (r) => r["_field"] == "value" or r["_field"] == "temperature" or r["_field"] == "setpoint" or r["_field"] == "state")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No temperature data found")
            return {}
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                room_name = record.values.get('room', 'unknown')
                if not room_name or room_name == 'unknown':
                    # Try to extract room from measurement name or tags
                    measurement = record.get_measurement()
                    for tag_key in ['location', 'zone', 'area']:
                        if tag_key in record.values:
                            room_name = record.values[tag_key]
                            break
                    else:
                        # Skip if no room identified
                        continue
                
                records.append({
                    'timestamp': record.get_time(),
                    'room': room_name,
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'measurement': record.get_measurement()
                })
        
        if not records:
            self.logger.warning("No temperature records found")
            return {}
        
        df = pd.DataFrame(records)
        
        # Group by room
        room_data = {}
        for room_name, room_df in df.groupby('room'):
            # Pivot to get fields as columns
            room_pivot = room_df.pivot_table(
                index='timestamp',
                columns='field', 
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Set timestamp as index
            room_pivot['timestamp'] = pd.to_datetime(room_pivot['timestamp'])
            room_pivot.set_index('timestamp', inplace=True)
            
            # Ensure required columns exist
            if 'temperature' not in room_pivot.columns and 'value' in room_pivot.columns:
                room_pivot['temperature'] = room_pivot['value']
            
            room_data[room_name] = room_pivot
            self.logger.info(f"Extracted {len(room_pivot)} temperature points for room {room_name}")
        
        return room_data

    async def extract_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract weather forecast data from InfluxDB.
        
        Returns DataFrame with columns:
        - timestamp: datetime index
        - temperature: outdoor temperature
        - humidity: relative humidity  
        - wind_speed: wind speed
        - cloud_cover: cloud coverage
        - solar_radiation: solar irradiance (if available)
        """
        self.logger.info(f"Extracting weather data from {start_date} to {end_date}")
        
        # Query for weather data - outdoor temperature and wind from solar bucket  
        query = f'''
        from(bucket: "solar")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "teplomer" and r["topic"] == "teplomer/TC")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> keep(columns: ["topic", "_time", "_value"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No weather data found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'source': record.values.get('source', 'unknown')
                })
        
        if not records:
            self.logger.warning("No weather records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Pivot to get weather parameters as columns
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='field',
            values='value', 
            aggfunc='mean'
        ).reset_index()
        
        # Set timestamp as index
        df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        df_pivot.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Extracted {len(df_pivot)} weather data points")
        return df_pivot

    async def extract_energy_prices(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Extract energy price data from InfluxDB.
        
        Returns DataFrame with columns:
        - timestamp: datetime index
        - price_eur_mwh: electricity price in EUR/MWh
        - price_czk_kwh: electricity price in CZK/kWh
        """
        self.logger.info(f"Extracting energy price data from {start_date} to {end_date}")
        
        # Query for energy price data  
        query = f'''
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "energy_prices" or r["_measurement"] == "electricity_prices")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No energy price data found")
            return None
        
        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value()
                })
        
        if not records:
            self.logger.warning("No energy price records found") 
            return None
        
        df = pd.DataFrame(records)
        
        # Pivot to get price fields as columns
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='field',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Set timestamp as index
        df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        df_pivot.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Extracted {len(df_pivot)} energy price points")
        return df_pivot

    async def extract_energy_consumption(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract total energy consumption data.
        
        Returns DataFrame with columns:
        - timestamp: datetime index
        - grid_import: power imported from grid
        - grid_export: power exported to grid
        - battery_power: battery charge/discharge power
        - total_consumption: total house consumption
        """
        self.logger.info(f"Extracting energy consumption data from {start_date} to {end_date}")
        
        # Query for energy consumption data - calculate from relay heating power like the example
        query = f'''
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" and r["tag1"] == "heating")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field", "room"])
        '''
        
        tables = self.query_api.query(query)
        
        if not tables:
            self.logger.warning("No energy consumption data found")
            return pd.DataFrame()
        
        # Convert to DataFrame similar to other methods
        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    'timestamp': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value()
                })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='field',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        df_pivot.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Extracted {len(df_pivot)} energy consumption points")
        return df_pivot

    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to parquet file for fast loading."""
        if df.empty:
            self.logger.warning(f"Not saving {filename} - DataFrame is empty")
            return
            
        filepath = self.data_dir / f"{filename}.parquet"
        df.to_parquet(filepath, compression='snappy')
        self.logger.info(f"Saved {len(df)} records to {filepath}")

    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from parquet file."""
        filepath = self.data_dir / f"{filename}.parquet"
        if not filepath.exists():
            self.logger.warning(f"File {filepath} does not exist")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        self.logger.info(f"Loaded {len(df)} records from {filepath}")
        return df

    def get_data_quality_report(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Generate data quality report for a dataset."""
        if df.empty:
            return {
                'data_type': data_type,
                'total_records': 0,
                'date_range': (None, None),
                'missing_percentage': 100,
                'time_gaps': [],
                'columns': []
            }
        
        # Calculate missing data percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 100
        
        # Find time gaps (>1 hour for continuous data)
        time_gaps = []
        if not df.index.empty:
            time_diffs = df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
            time_gaps = [(gap_time, gap_duration) for gap_time, gap_duration in large_gaps.items()]
        
        return {
            'data_type': data_type,
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()) if not df.index.empty else (None, None),
            'missing_percentage': round(missing_percentage, 2),
            'time_gaps': time_gaps[:10],  # Limit to first 10 gaps
            'columns': list(df.columns)
        }