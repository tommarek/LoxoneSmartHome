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

from config.energy_settings import (
    CONSUMPTION_CATEGORIES,
    DATA_QUALITY_THRESHOLDS,
    ROOM_CONFIG,
    get_room_power,
)
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
            timeout=30000,
        )
        self.query_api = self.client.query_api()

        # Timezone handling
        self.local_tz = pytz.timezone("Europe/Prague")
        self.utc_tz = pytz.UTC

        # Data output directory
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        """Close InfluxDB client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()

    async def extract_pv_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract PV production data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - InputPower: solar panel input power (W)
        - INVPowerToLocalLoad: inverter power to local load (W)
        - ACPowerToUser: AC power to user consumption (W)
        - ACPowerToGrid: AC power exported to grid (W)
        - ChargePower: battery charging power (W)
        - SOC: battery state of charge (%)
        """
        self.logger.info(f"Extracting PV data from {start_date} to {end_date}")

        # Query for actual solar fields from your system
        query = f"""
        from(bucket: "solar")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => r["_field"] == "InputPower" or 
                              r["_field"] == "INVPowerToLocalLoad" or 
                              r["_field"] == "ACPowerToUser" or 
                              r["_field"] == "ACPowerToGrid" or
                              r["_field"] == "ChargePower" or
                              r["_field"] == "SOC")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No PV data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No PV records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get solar fields as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        # Ensure timestamp is datetime and set as index
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        # Calculate derived metrics
        if "InputPower" in df_pivot.columns:
            # Calculate energy production from input power (15-minute intervals)
            df_pivot["solar_energy_kwh"] = (
                df_pivot["InputPower"] * 0.25 / 1000
            )  # Convert W*0.25h to kWh

        if "ACPowerToGrid" in df_pivot.columns and "ACPowerToUser" in df_pivot.columns:
            # Calculate total AC output
            df_pivot["total_ac_output"] = df_pivot["ACPowerToGrid"].fillna(0) + df_pivot[
                "ACPowerToUser"
            ].fillna(0)

        if "ChargePower" in df_pivot.columns:
            # Calculate battery energy change (15-minute intervals)
            df_pivot["battery_energy_kwh"] = (
                df_pivot["ChargePower"] * 0.25 / 1000
            )  # Convert W*0.25h to kWh

        self.logger.info(
            f"Extracted {len(df_pivot)} PV data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_room_temperatures(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
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
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature" or r["_measurement"] == "heating")
          |> filter(fn: (r) => r["_field"] == "value" or r["_field"] == "temperature" or r["_field"] == "setpoint" or r["_field"] == "state")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No temperature data found")
            return {}

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                room_name = record.values.get("room", "unknown")
                if not room_name or room_name == "unknown":
                    # Try to extract room from measurement name or tags
                    for tag_key in ["location", "zone", "area"]:
                        if tag_key in record.values:
                            room_name = record.values[tag_key]
                            break
                    else:
                        # Skip if no room identified
                        continue

                records.append(
                    {
                        "timestamp": record.get_time(),
                        "room": room_name,
                        "field": record.get_field(),
                        "value": record.get_value(),
                        "measurement": record.get_measurement(),
                    }
                )

        if not records:
            self.logger.warning("No temperature records found")
            return {}

        df = pd.DataFrame(records)

        # Group by room
        room_data = {}
        for room_name, room_df in df.groupby("room"):
            # Pivot to get fields as columns
            room_pivot = room_df.pivot_table(
                index="timestamp", columns="field", values="value", aggfunc="mean"
            ).reset_index()

            # Set timestamp as index
            room_pivot["timestamp"] = pd.to_datetime(room_pivot["timestamp"])
            room_pivot.set_index("timestamp", inplace=True)

            # Ensure required columns exist
            if "temperature" not in room_pivot.columns and "value" in room_pivot.columns:
                room_pivot["temperature"] = room_pivot["value"]

            room_data[room_name] = room_pivot
            self.logger.info(f"Extracted {len(room_pivot)} temperature points for room {room_name}")

        return room_data

    async def extract_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract weather data including sun elevation.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - sun_elevation: sun elevation angle (degrees)
        - temperature: outdoor temperature
        - humidity: relative humidity
        - wind_speed: wind speed (if available)
        - cloud_cover: cloud coverage (if available)
        """
        self.logger.info(f"Extracting weather data from {start_date} to {end_date}")

        # Query for sun elevation and weather data from loxone bucket
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "current_weather")
          |> filter(fn: (r) => r["_field"] == "sun_elevation" or 
                              r["_field"] == "temperature" or
                              r["_field"] == "humidity" or
                              r["_field"] == "wind_speed" or
                              r["_field"] == "cloud_cover")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No weather data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No weather records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get weather parameters as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        # Set timestamp as index
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        # Calculate derived solar metrics if sun_elevation is available
        if "sun_elevation" in df_pivot.columns:
            # Filter for positive sun elevation (daytime)
            df_pivot["is_daytime"] = df_pivot["sun_elevation"] > 0

            # Calculate theoretical solar radiation based on sun elevation
            # This is a simplified model: radiation ∝ sin(elevation)
            import math

            df_pivot["theoretical_solar_radiation"] = df_pivot["sun_elevation"].apply(
                lambda x: max(0, 1000 * math.sin(math.radians(x))) if x > 0 else 0
            )

        self.logger.info(
            f"Extracted {len(df_pivot)} weather data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_energy_prices(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Extract energy price data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - price_eur_mwh: electricity price in EUR/MWh
        - price_czk_kwh: electricity price in CZK/kWh
        """
        self.logger.info(f"Extracting energy price data from {start_date} to {end_date}")

        # Query for energy price data
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "energy_prices" or r["_measurement"] == "electricity_prices")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No energy price data found")
            return None

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No energy price records found")
            return None

        df = pd.DataFrame(records)

        # Pivot to get price fields as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        # Set timestamp as index
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        self.logger.info(f"Extracted {len(df_pivot)} energy price points")
        return df_pivot

    async def extract_energy_consumption(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
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

        # Query for ALL energy consumption data - not just heating
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" or 
                              r["_measurement"] == "power" or
                              r["_measurement"] == "energy")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field", "room", "tag1", "tag2"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No energy consumption data found")
            return pd.DataFrame()

        # Convert to DataFrame with more metadata
        records = []
        for table in tables:
            for record in table.records:
                room_name = record.values.get("room", "unknown")
                tag1 = record.values.get("tag1", "")
                tag2 = record.values.get("tag2", "")

                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                        "room": room_name,
                        "tag1": tag1,
                        "tag2": tag2,
                        "measurement": record.get_measurement(),
                    }
                )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Calculate total consumption by category
        consumption_data = {}

        # Group by consumption category
        for category, config in CONSUMPTION_CATEGORIES.items():
            category_df = df[
                (df["measurement"] == config["measurement"])
                & (df["tag1"].str.contains(category, na=False))
            ]

            if not category_df.empty:
                # For heating, multiply relay state by room power
                if category == "heating" and "room" in category_df.columns:
                    category_df = category_df.copy()
                    category_df["power_kw"] = category_df["room"].apply(get_room_power)
                    # Convert relay state (0/1) to actual power consumption
                    category_df["actual_power"] = (
                        category_df["value"] * category_df["power_kw"] * 1000
                    )  # Convert to W

                    # Group by timestamp and sum all rooms
                    category_consumption = category_df.groupby("timestamp")["actual_power"].sum()
                else:
                    # For other categories, use direct power values
                    category_consumption = category_df.groupby("timestamp")["value"].sum()

                consumption_data[f"{category}_power"] = category_consumption

        # Combine all consumption categories
        if consumption_data:
            df_consumption = pd.DataFrame(consumption_data)
            df_consumption.index = pd.to_datetime(df_consumption.index)

            # Calculate total consumption
            power_columns = [col for col in df_consumption.columns if col.endswith("_power")]
            df_consumption["total_consumption"] = df_consumption[power_columns].sum(axis=1)

            # Calculate energy consumption (15-minute intervals)
            for col in power_columns + ["total_consumption"]:
                energy_col = col.replace("_power", "_energy_kwh")
                df_consumption[energy_col] = (
                    df_consumption[col] * 0.25 / 1000
                )  # Convert W*0.25h to kWh
        else:
            df_consumption = pd.DataFrame()

        self.logger.info(
            f"Extracted {len(df_consumption)} energy consumption points with categories: {list(consumption_data.keys())}"
        )
        return df_consumption

    async def extract_battery_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract battery charge/discharge data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - ChargePower: battery charging power (W, positive = charging)
        - DischargePower: battery discharging power (W, positive = discharging)
        - SOC: battery state of charge (%)
        - BatteryVoltage: battery voltage (V)
        - BatteryCurrent: battery current (A)
        """
        self.logger.info(f"Extracting battery data from {start_date} to {end_date}")

        # Query for battery data from solar bucket
        query = f"""
        from(bucket: "solar")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => r["_field"] == "ChargePower" or 
                              r["_field"] == "DischargePower" or 
                              r["_field"] == "SOC" or
                              r["_field"] == "BatteryVoltage" or
                              r["_field"] == "BatteryCurrent")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No battery data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No battery records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get battery fields as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        # Set timestamp as index
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        # Calculate derived metrics
        if "ChargePower" in df_pivot.columns and "DischargePower" in df_pivot.columns:
            # Net battery power (positive = charging, negative = discharging)
            df_pivot["net_battery_power"] = df_pivot["ChargePower"].fillna(0) - df_pivot[
                "DischargePower"
            ].fillna(0)

            # Battery energy change (15-minute intervals)
            df_pivot["battery_energy_change_kwh"] = df_pivot["net_battery_power"] * 0.25 / 1000

        if "BatteryVoltage" in df_pivot.columns and "BatteryCurrent" in df_pivot.columns:
            # Calculate instantaneous battery power from V*I
            df_pivot["calculated_battery_power"] = (
                df_pivot["BatteryVoltage"] * df_pivot["BatteryCurrent"]
            )

        self.logger.info(
            f"Extracted {len(df_pivot)} battery data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_ev_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract EV charging data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - ev_power: EV charging power (W)
        - ev_energy: EV charging energy (kWh)
        - ev_connected: EV connection status
        """
        self.logger.info(f"Extracting EV data from {start_date} to {end_date}")

        # Query for EV data - adjust measurement/field names based on your system
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_historical}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "ev_charger" or 
                              r["_measurement"] == "wallbox" or
                              r["_measurement"] == "car_charging")
          |> filter(fn: (r) => r["_field"] == "power" or 
                              r["_field"] == "energy" or 
                              r["_field"] == "connected" or
                              r["_field"] == "charging_state")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No EV data found - EV charging may not be available")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No EV records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get EV fields as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        # Set timestamp as index
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        # Rename columns for consistency
        column_mapping = {
            "power": "ev_power",
            "energy": "ev_energy",
            "connected": "ev_connected",
            "charging_state": "ev_charging_state",
        }
        df_pivot.rename(columns=column_mapping, inplace=True)

        # Calculate EV energy consumption if power is available
        if "ev_power" in df_pivot.columns:
            df_pivot["ev_energy_kwh"] = df_pivot["ev_power"] * 0.25 / 1000  # 15min intervals

        self.logger.info(
            f"Extracted {len(df_pivot)} EV data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to parquet file for fast loading."""
        if df.empty:
            self.logger.warning(f"Not saving {filename} - DataFrame is empty")
            return

        filepath = self.data_dir / f"{filename}.parquet"
        df.to_parquet(filepath, compression="snappy")
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
                "data_type": data_type,
                "total_records": 0,
                "date_range": (None, None),
                "missing_percentage": 100,
                "time_gaps": [],
                "columns": [],
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
            "data_type": data_type,
            "total_records": len(df),
            "date_range": (df.index.min(), df.index.max()) if not df.index.empty else (None, None),
            "missing_percentage": round(missing_percentage, 2),
            "time_gaps": time_gaps[:10],  # Limit to first 10 gaps
            "columns": list(df.columns),
        }

    def validate_data_completeness(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate that we have all required data sources and check data quality.

        Args:
            data: Dictionary of DataFrames by data type

        Returns:
            Dictionary with validation results
        """
        required_sources = ["pv", "weather", "rooms", "consumption"]
        optional_sources = ["battery", "ev", "energy_prices"]

        validation_results = {
            "is_complete": True,
            "missing_required": [],
            "missing_optional": [],
            "data_quality": {},
            "recommendations": [],
        }

        # Check required sources
        for source in required_sources:
            if source not in data:
                validation_results["missing_required"].append(source)
                validation_results["is_complete"] = False
            elif hasattr(data[source], "empty") and data[source].empty:
                validation_results["missing_required"].append(source)
                validation_results["is_complete"] = False
            elif isinstance(data[source], dict) and len(data[source]) == 0:
                validation_results["missing_required"].append(source)
                validation_results["is_complete"] = False

        # Check optional sources
        for source in optional_sources:
            if source not in data:
                validation_results["missing_optional"].append(source)
            elif hasattr(data[source], "empty") and data[source].empty:
                validation_results["missing_optional"].append(source)
            elif isinstance(data[source], dict) and len(data[source]) == 0:
                validation_results["missing_optional"].append(source)

        # Validate data quality for available sources
        for source, df in data.items():
            # Handle rooms dict specially
            if source == "rooms" and isinstance(df, dict):
                for room_name, room_df in df.items():
                    if not room_df.empty:
                        quality_report = self.get_data_quality_report(
                            room_df, f"{source}_{room_name}"
                        )
                        validation_results["data_quality"][f"{source}_{room_name}"] = quality_report
            elif hasattr(df, "empty") and not df.empty:
                quality_report = self.get_data_quality_report(df, source)
                validation_results["data_quality"][source] = quality_report

                # Check against thresholds
                missing_pct = quality_report["missing_percentage"]
                if missing_pct > DATA_QUALITY_THRESHOLDS["max_missing_percentage"]:
                    validation_results["recommendations"].append(
                        f"{source}: High missing data ({missing_pct:.1f}%) - consider data interpolation"
                    )

                # Check for large time gaps
                if len(quality_report["time_gaps"]) > 0:
                    max_gap = max(
                        [gap[1].total_seconds() / 3600 for gap in quality_report["time_gaps"]]
                    )
                    if max_gap > DATA_QUALITY_THRESHOLDS["max_gap_hours"]:
                        validation_results["recommendations"].append(
                            f"{source}: Large time gaps found (max {max_gap:.1f}h) - check data collection"
                        )

        # Add feature-specific recommendations
        if "battery" in validation_results["missing_optional"]:
            validation_results["recommendations"].append(
                "Battery data missing - energy storage optimization will be limited"
            )

        if "ev" in validation_results["missing_optional"]:
            validation_results["recommendations"].append(
                "EV data missing - EV charging optimization not available"
            )

        return validation_results
