"""
Data extraction module for PEMS v2.

Extracts comprehensive historical data from InfluxDB for analysis:
- PV production data with string-level monitoring and battery status
- Room temperature, humidity, and target temperature data
- Weather forecast data with solar radiation parameters
- Current weather data with real-time solar position from Loxone
- Heating and shading relay states
- Energy prices from OTE market
- EV charging data (if available)
- Save as parquet files for fast analysis
- Implement data quality checks
- Handle missing data interpolation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytz
from config.energy_settings import (CONSUMPTION_CATEGORIES,
                                    DATA_QUALITY_THRESHOLDS, get_room_power)
from config.settings import PEMSSettings as Settings
from influxdb_client import InfluxDBClient


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

    async def extract_pv_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract comprehensive PV production data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - InputPower: total solar panel input power (W)
        - PV1InputPower: PV string 1 power (W)
        - PV2InputPower: PV string 2 power (W)
        - PV1Voltage: PV string 1 voltage (V)
        - PV2Voltage: PV string 2 voltage (V)
        - INVPowerToLocalLoad: inverter power to local load (W)
        - ACPowerToUser: AC power to user consumption (W)
        - ACPowerToGrid: AC power exported to grid (W)
        - ChargePower: battery charging power (W)
        - DischargePower: battery discharging power (W)
        - SOC: battery state of charge (%)
        - BatteryTemperature: battery temperature (°C)
        - InverterTemperature: inverter temperature (°C)
        - InverterStatus: inverter operational status
        - TodayGenerateEnergy: energy generated today (kWh)
        """
        self.logger.info(f"Extracting PV data from {start_date} to {end_date}")

        # Query for comprehensive solar fields from your system
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_solar}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "solar")
          |> filter(fn: (r) => r["_field"] == "InputPower" or
                              r["_field"] == "PV1InputPower" or
                              r["_field"] == "PV2InputPower" or
                              r["_field"] == "PV1Voltage" or
                              r["_field"] == "PV2Voltage" or
                              r["_field"] == "INVPowerToLocalLoad" or
                              r["_field"] == "ACPowerToUser" or
                              r["_field"] == "ACPowerToGrid" or
                              r["_field"] == "ChargePower" or
                              r["_field"] == "DischargePower" or
                              r["_field"] == "SOC" or
                              r["_field"] == "BatteryTemperature" or
                              r["_field"] == "InverterTemperature" or
                              r["_field"] == "InverterStatus" or
                              r["_field"] == "TodayGenerateEnergy" or
                              r["_field"] == "LocalLoadEnergyToday" or
                              r["_field"] == "EnergyToGridToday" or
                              r["_field"] == "EnergyToUserToday")
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
            df_pivot["total_ac_output"] = df_pivot["ACPowerToGrid"].fillna(
                0
            ) + df_pivot["ACPowerToUser"].fillna(0)

        if "ChargePower" in df_pivot.columns and "DischargePower" in df_pivot.columns:
            # Calculate net battery power
            df_pivot["net_battery_power"] = df_pivot["ChargePower"].fillna(
                0
            ) - df_pivot["DischargePower"].fillna(0)
            # Calculate battery energy change (15-minute intervals)
            df_pivot["battery_energy_kwh"] = (
                df_pivot["net_battery_power"] * 0.25 / 1000
            )  # Convert W*0.25h to kWh

        if "PV1InputPower" in df_pivot.columns and "PV2InputPower" in df_pivot.columns:
            # Calculate total PV power if individual strings available
            df_pivot["total_pv_power"] = df_pivot["PV1InputPower"].fillna(0) + df_pivot[
                "PV2InputPower"
            ].fillna(0)
            # Calculate string balance factor
            df_pivot["pv_string_balance"] = df_pivot["PV1InputPower"] / (
                df_pivot["total_pv_power"] + 1e-6
            )

        self.logger.info(
            f"Extracted {len(df_pivot)} PV data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_room_temperatures(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract room temperature and humidity data from InfluxDB.

        Returns dict of DataFrames by room name with columns:
        - timestamp: datetime index
        - temperature: current temperature
        - humidity: current humidity (%)
        - target_temp: target temperature (if available)
        - heating_on: boolean heating status
        """
        self.logger.info(
            f"Extracting room temperature and humidity data from {start_date} to {end_date}"
        )

        # Query for temperature and humidity data
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature" or
                      r["_measurement"] == "humidity" or
                      r["_measurement"] == "target_temp")
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

                # Extract room from field name if present
                field_name = record.get_field()
                measurement = record.get_measurement()

                # For temperature and humidity, extract room from field name
                if measurement in ["temperature", "humidity"] and "_" in field_name:
                    parts = field_name.split("_")
                    if len(parts) >= 2:
                        room_name = "_".join(
                            parts[1:]
                        )  # Everything after measurement type

                records.append(
                    {
                        "timestamp": record.get_time(),
                        "room": room_name,
                        "field": field_name,
                        "value": record.get_value(),
                        "measurement": measurement,
                    }
                )

        if not records:
            self.logger.warning("No temperature records found")
            return {}

        df = pd.DataFrame(records)

        # Group by room
        room_data = {}
        for room_name, room_df in df.groupby("room"):
            # Process temperature, humidity, and target temp fields
            # Create separate dataframes for each measurement type
            temp_data = room_df[room_df["measurement"] == "temperature"]
            humidity_data = room_df[room_df["measurement"] == "humidity"]
            target_data = room_df[room_df["measurement"] == "target_temp"]

            # Start with temperature data
            if not temp_data.empty:
                room_pivot = temp_data.set_index("timestamp")[["value"]].rename(
                    columns={"value": "temperature"}
                )
            else:
                continue  # Skip rooms without temperature data

            # Add humidity if available
            if not humidity_data.empty:
                humidity_pivot = humidity_data.set_index("timestamp")[["value"]].rename(
                    columns={"value": "humidity"}
                )
                room_pivot = room_pivot.join(humidity_pivot, how="outer")

            # Add target temperature if available
            if not target_data.empty:
                target_pivot = target_data.set_index("timestamp")[["value"]].rename(
                    columns={"value": "target_temp"}
                )
                room_pivot = room_pivot.join(target_pivot, how="outer")

            # Ensure timestamp index is datetime
            room_pivot.index = pd.to_datetime(room_pivot.index)

            room_data[room_name] = room_pivot
            self.logger.info(
                f"Extracted {len(room_pivot)} temperature points for room {room_name}"
            )

        return room_data

    async def extract_weather_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract weather forecast data from weather_forecast bucket.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - temperature_2m: air temperature at 2m height (°C)
        - relativehumidity_2m: relative humidity at 2m (%)
        - windspeed_10m: wind speed at 10m (km/h)
        - cloudcover: total cloud coverage (%)
        - precipitation: precipitation amount (mm)
        - shortwave_radiation: solar radiation (W/m²)
        - direct_radiation: direct solar radiation (W/m²)
        - diffuse_radiation: diffuse solar radiation (W/m²)
        - uv_index: UV radiation index
        - apparent_temperature: "feels like" temperature (°C)
        - surface_pressure: atmospheric pressure (hPa)

        Note: For actual solar position (sun_elevation, sun_direction),
        use extract_current_weather() which gets real-time data from Loxone.
        """
        self.logger.info(f"Extracting weather data from {start_date} to {end_date}")

        # Query for comprehensive weather forecast data
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_weather}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "weather_forecast")
          |> filter(fn: (r) => r["_field"] == "temperature_2m" or
                              r["_field"] == "relativehumidity_2m" or
                              r["_field"] == "windspeed_10m" or
                              r["_field"] == "cloudcover" or
                              r["_field"] == "cloudcover_low" or
                              r["_field"] == "cloudcover_mid" or
                              r["_field"] == "cloudcover_high" or
                              r["_field"] == "precipitation" or
                              r["_field"] == "shortwave_radiation" or
                              r["_field"] == "direct_radiation" or
                              r["_field"] == "diffuse_radiation" or
                              r["_field"] == "direct_normal_irradiance" or
                              r["_field"] == "terrestrial_radiation" or
                              r["_field"] == "uv_index" or
                              r["_field"] == "apparent_temperature" or
                              r["_field"] == "dewpoint_2m" or
                              r["_field"] == "surface_pressure")
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

        # Create DataFrame from weather records or generate time range if no data
        if records:
            df = pd.DataFrame(records)
            # Pivot to get weather parameters as columns
            df_pivot = df.pivot_table(
                index="timestamp", columns="field", values="value", aggfunc="mean"
            ).reset_index()
            df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
            df_pivot.set_index("timestamp", inplace=True)
        else:
            # If no weather data available, create time range for solar calculations
            self.logger.info(
                "No weather data found in database, creating time range for solar calculations"
            )
            time_range = pd.date_range(start=start_date, end=end_date, freq="15min")
            df_pivot = pd.DataFrame(index=time_range)

        # Solar position data should come from current_weather in loxone bucket
        # No need to calculate it separately

        self.logger.info(
            f"Extracted {len(df_pivot)} weather data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_outdoor_temperature_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract outdoor temperature data from solar bucket (teplomer sensor).

        Returns DataFrame with columns:
        - timestamp: datetime index
        - outdoor_temp: outdoor temperature (°C)
        """
        self.logger.info("Extracting outdoor temperature data from teplomer sensor...")

        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_solar}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "teplomer")
          |> filter(fn: (r) => r["topic"] == "teplomer/TC")
          |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
          |> keep(columns: ["topic", "_time", "_value"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No outdoor temperature data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "outdoor_temp": record.get_value(),
                    }
                )

        if not records:
            self.logger.warning("No outdoor temperature records found")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        self.logger.info(f"Extracted {len(df)} outdoor temperature records")
        return df

    async def extract_energy_prices(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Extract energy price data from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - price_czk_kwh: electricity price in CZK/kWh
        """
        self.logger.info(
            f"Extracting energy price data from {start_date} to {end_date}"
        )

        # Query for energy price data from the correct bucket
        query = f"""
        from(bucket: "ote_prices")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "electricity_prices")
          |> filter(fn: (r) => r["_field"] == "price_czk_kwh")
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
        Extract energy consumption data categorized by usage type.

        NOTE: This method serves a different purpose than extract_pv_data():
        - extract_pv_data(): Focuses on energy GENERATION (solar, battery storage)
        - extract_energy_consumption(): Focuses on energy CONSUMPTION by category

        This method extracts relay states and power measurements to calculate
        consumption for different categories (heating, lighting, etc.) as defined
        in CONSUMPTION_CATEGORIES. For heating, it multiplies relay states by
        room power ratings to get actual power consumption.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - {category}_power: power consumption for each category (W)
        - {category}_energy_kwh: energy consumption for each category
        - total_consumption: sum of all category powers (W)
        - total_consumption_energy_kwh: total energy consumption
        """
        self.logger.info(
            f"Extracting energy consumption data from {start_date} to {end_date}"
        )

        # Query specifically for heating relay data (only category we track)
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" and r["tag1"] == "heating")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field", "room"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No energy consumption data found")
            return pd.DataFrame()

        # Convert to DataFrame - simpler structure for heating only
        records = []
        for table in tables:
            for record in table.records:
                room_name = record.values.get("room", "unknown")

                records.append(
                    {
                        "timestamp": record.get_time(),
                        "value": record.get_value(),
                        "room": room_name,
                    }
                )

        if not records:
            self.logger.warning("No relay/power/energy records found in database")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        self.logger.info(f"Found {len(df)} heating relay records")

        # Calculate heating consumption
        consumption_data = {}

        if not df.empty:
            # Add room power ratings
            df = df.copy()
            df["power_kw"] = df["room"].apply(get_room_power)
            # Convert relay state (0/1) to actual power consumption
            df["actual_power"] = df["value"] * df["power_kw"] * 1000  # Convert to W

            # Group by timestamp and sum all rooms
            heating_consumption = df.groupby("timestamp")["actual_power"].sum()
            consumption_data["heating_power"] = heating_consumption
            self.logger.info(
                f"Calculated heating consumption for {len(heating_consumption)} time points"
            )

        # Combine all consumption categories
        if consumption_data:
            df_consumption = pd.DataFrame(consumption_data)
            df_consumption.index = pd.to_datetime(df_consumption.index)

            # Calculate total consumption
            power_columns = [
                col for col in df_consumption.columns if col.endswith("_power")
            ]
            df_consumption["total_consumption"] = df_consumption[power_columns].sum(
                axis=1
            )

            # Calculate energy consumption (15-minute intervals)
            for col in power_columns + ["total_consumption"]:
                energy_col = col.replace("_power", "_energy_kwh")
                df_consumption[energy_col] = (
                    df_consumption[col] * 0.25 / 1000
                )  # Convert W*0.25h to kWh
        else:
            df_consumption = pd.DataFrame()

        self.logger.info(
            f"Extracted {len(df_consumption)} energy consumption points "
            f"with categories: {list(consumption_data.keys())}"
        )
        return df_consumption

    async def extract_battery_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract battery charge/discharge data from InfluxDB.

        NOTE: This method has some overlap with extract_pv_data() as both query
        battery fields (ChargePower, DischargePower, SOC) from the same bucket.
        However, this method:
        1. Focuses solely on battery data (no PV/inverter fields)
        2. Includes additional battery fields: BatteryVoltage, BatteryCurrent
        3. Calculates battery power from V*I for validation

        Consider using extract_pv_data() if you need comprehensive energy data.
        Use this method if you only need battery-specific analysis.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - ChargePower: battery charging power (W, positive = charging)
        - DischargePower: battery discharging power (W, positive = discharging)
        - SOC: battery state of charge (%)
        - BatteryVoltage: battery voltage (V)
        - BatteryCurrent: battery current (A)
        - net_battery_power: calculated net power (positive = charging)
        - battery_energy_change_kwh: energy change per interval
        - calculated_battery_power: V*I validation
        """
        self.logger.info(f"Extracting battery data from {start_date} to {end_date}")

        # Query for battery data from loxone bucket
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_solar}")
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
            df_pivot["net_battery_power"] = df_pivot["ChargePower"].fillna(
                0
            ) - df_pivot["DischargePower"].fillna(0)

            # Battery energy change (15-minute intervals)
            df_pivot["battery_energy_change_kwh"] = (
                df_pivot["net_battery_power"] * 0.25 / 1000
            )

        if (
            "BatteryVoltage" in df_pivot.columns
            and "BatteryCurrent" in df_pivot.columns
        ):
            # Calculate instantaneous battery power from V*I
            df_pivot["calculated_battery_power"] = (
                df_pivot["BatteryVoltage"] * df_pivot["BatteryCurrent"]
            )

        self.logger.info(
            f"Extracted {len(df_pivot)} battery data points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_ev_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract EV charging data from InfluxDB.

        NOTE: EV charging in this system is just a load without specific charging
        timestamps or dedicated measurements. This method returns an empty DataFrame
        as a placeholder. To identify EV charging patterns:

        1. Look for characteristic load patterns in total consumption data:
           - Sudden increase of 3.7kW (single phase) or 11kW (three phase)
           - Sustained load for 2-8 hours typically during night
           - Regular daily/weekly patterns

        2. Use load disaggregation techniques in feature engineering phase
        3. Consider adding dedicated EV charger monitoring in the future

        Returns:
            Empty DataFrame as EV data is not separately tracked
        """
        self.logger.info(
            f"EV data extraction called for {start_date} to {end_date}, "
            "but EV charging is not separately tracked in the database"
        )

        # Return empty DataFrame with expected columns for compatibility
        return pd.DataFrame(columns=["ev_power", "ev_energy_kwh", "ev_connected"])

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

    def get_data_quality_report(
        self, df: pd.DataFrame, data_type: str
    ) -> Dict[str, Any]:
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
        missing_percentage = (
            (missing_cells / total_cells * 100) if total_cells > 0 else 100
        )

        # Find time gaps (>1 hour for continuous data)
        time_gaps = []
        if not df.index.empty:
            time_diffs = df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
            time_gaps = [
                (gap_time, gap_duration)
                for gap_time, gap_duration in large_gaps.items()
            ]

        return {
            "data_type": data_type,
            "total_records": len(df),
            "date_range": (df.index.min(), df.index.max())
            if not df.index.empty
            else (None, None),
            "missing_percentage": round(missing_percentage, 2),
            "time_gaps": time_gaps[:10],  # Limit to first 10 gaps
            "columns": list(df.columns),
        }

    def validate_data_completeness(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
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
                        validation_results["data_quality"][
                            f"{source}_{room_name}"
                        ] = quality_report
            elif hasattr(df, "empty") and not df.empty:
                quality_report = self.get_data_quality_report(df, source)
                validation_results["data_quality"][source] = quality_report

                # Check against thresholds
                missing_pct = quality_report["missing_percentage"]
                if missing_pct > DATA_QUALITY_THRESHOLDS["max_missing_percentage"]:
                    validation_results["recommendations"].append(
                        f"{source}: High missing data ({missing_pct:.1f}%) - "
                        "consider data interpolation"
                    )

                # Check for large time gaps
                if len(quality_report["time_gaps"]) > 0:
                    max_gap = max(
                        [
                            gap[1].total_seconds() / 3600
                            for gap in quality_report["time_gaps"]
                        ]
                    )
                    if max_gap > DATA_QUALITY_THRESHOLDS["max_gap_hours"]:
                        validation_results["recommendations"].append(
                            f"{source}: Large time gaps found (max {max_gap:.1f}h) - "
                            "check data collection"
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

    async def extract_relay_states(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract relay on/off states for all 16 rooms.

        Returns dictionary of DataFrames by room name with columns:
        - timestamp: datetime index
        - relay_state: 0/1 for OFF/ON
        - power_kw: calculated power consumption based on room rating
        """
        self.logger.info(f"Extracting relay states from {start_date} to {end_date}")

        # Query for relay data
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" and r["tag1"] == "heating")
          |> aggregateWindow(every: 5m, fn: last, createEmpty: false)
          |> keep(columns: ["_time", "_value", "room"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No relay data found")
            return {}

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                room_name = record.values.get("room", "unknown")
                if room_name and room_name != "unknown":
                    records.append(
                        {
                            "timestamp": record.get_time(),
                            "room": room_name,
                            "relay_state": record.get_value(),
                        }
                    )

        if not records:
            self.logger.warning("No relay records found")
            return {}

        df = pd.DataFrame(records)

        # Group by room
        room_relay_data = {}
        for room_name, room_df in df.groupby("room"):
            # Pivot and set timestamp as index
            room_pivot = room_df.set_index("timestamp")[["relay_state"]]
            room_pivot.index = pd.to_datetime(room_pivot.index)

            # Add calculated power consumption
            room_power_kw = get_room_power(room_name)
            room_pivot["power_kw"] = room_pivot["relay_state"] * room_power_kw
            room_pivot["power_w"] = room_pivot["power_kw"] * 1000

            room_relay_data[room_name] = room_pivot
            self.logger.info(
                f"Extracted {len(room_pivot)} relay states for room {room_name}"
            )

        return room_relay_data

    async def extract_current_weather(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract current weather data from Loxone system including solar position.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - absolute_solar_irradiance: solar irradiance (W/m²)
        - current_temperature: outdoor temperature (°C)
        - pressure: atmospheric pressure (hPa)
        - relative_humidity: humidity (%)
        - wind_direction: wind direction (degrees)
        - sun_direction: sun azimuth (degrees)
        - sun_elevation: sun elevation (degrees)
        - minutes_past_midnight: time of day
        - precipitation: current precipitation
        - brightness: ambient brightness
        - rain: rain measurement
        - wind_speed: wind speed
        - sunshine: sunshine measurement
        """
        self.logger.info(
            f"Extracting current weather data from {start_date} to {end_date}"
        )

        # Query for current weather data from Loxone
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "current_weather" or
                              r["_measurement"] == "brightness" or
                              r["_measurement"] == "rain" or
                              r["_measurement"] == "wind_speed" or
                              r["_measurement"] == "sunshine")
          |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field", "_measurement"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No current weather data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                measurement = record.get_measurement()
                field = record.get_field()

                # Map measurement to field name
                if measurement == "current_weather":
                    field_name = field
                else:
                    field_name = measurement

                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": field_name,
                        "value": record.get_value(),
                    }
                )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get weather parameters as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="mean"
        ).reset_index()

        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        self.logger.info(
            f"Extracted {len(df_pivot)} current weather points with fields: {list(df_pivot.columns)}"
        )
        return df_pivot

    async def extract_shading_relays(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract shading/blinds relay states from InfluxDB.

        Returns DataFrame with columns:
        - timestamp: datetime index
        - {room}_{position}: relay state for each blind (0/1)
        """
        self.logger.info(
            f"Extracting shading relay states from {start_date} to {end_date}"
        )

        # Query for shading relay data
        query = f"""
        from(bucket: "{self.settings.influxdb.bucket_loxone}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "relay" and r["tag1"] == "shading")
          |> aggregateWindow(every: 15m, fn: last, createEmpty: false)
          |> keep(columns: ["_time", "_value", "_field"])
        """

        tables = self.query_api.query(query)

        if not tables:
            self.logger.warning("No shading relay data found")
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for table in tables:
            for record in table.records:
                field_name = record.get_field()
                records.append(
                    {
                        "timestamp": record.get_time(),
                        "field": f"shading_{field_name}",
                        "value": record.get_value(),
                    }
                )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Pivot to get shading relays as columns
        df_pivot = df.pivot_table(
            index="timestamp", columns="field", values="value", aggfunc="last"
        ).reset_index()

        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
        df_pivot.set_index("timestamp", inplace=True)

        self.logger.info(
            f"Extracted {len(df_pivot)} shading relay points with {len(df_pivot.columns)} blinds"
        )
        return df_pivot
