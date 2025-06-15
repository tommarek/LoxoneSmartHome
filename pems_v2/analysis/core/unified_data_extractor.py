"""
Unified data extraction pipeline for PEMS v2.

Single source of truth for all energy data with built-in validation, gap detection, 
and quality metrics. Replaces redundant extraction methods with efficient parallel 
extraction and comprehensive data quality assessment.

Key Features:
- Parallel extraction of all data streams
- Automatic data validation and gap detection
- Quality scoring for ML model input
- Efficient InfluxDB query optimization
- Structured data containers with type safety
- Memory-efficient processing with async patterns
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi
from scipy import stats

from ...config.settings import PEMSSettings

# Data quality thresholds for ML readiness assessment
DATA_QUALITY_THRESHOLDS = {
    "completeness_min": 0.8,
    "consistency_min": 0.7,
    "temporal_consistency_min": 0.8,
    "reasonableness_min": 0.9,
}


def get_room_power(room_name: str) -> float:
    """Get power rating for a specific room in kW - legacy compatibility."""
    # Default room power ratings (this should ideally come from settings)
    room_power_ratings = {
        "obyvak": 2.4,
        "kuchyne": 2.4,
        "loznice": 1.2,
        "pokoj_1": 1.2,
        "pokoj_2": 1.2,
        "pracovna": 1.2,
        "hosti": 1.2,
        "koupelna_nahore": 0.8,
        "koupelna_dole": 0.8,
        "chodba_nahore": 0.4,
        "chodba_dole": 0.4,
        "satna_nahore": 0.4,
        "satna_dole": 0.4,
        "technicka_mistnost": 0.4,
        "spajz": 0.2,
        "zachod": 0.2,
        "zadveri": 0.4,
    }
    return room_power_ratings.get(room_name, 0.0)


@dataclass
class EnergyDataset:
    """Structured container for all energy system data."""

    # Core energy flows
    pv_production: pd.DataFrame = field(default_factory=pd.DataFrame)
    battery_storage: pd.DataFrame = field(default_factory=pd.DataFrame)
    grid_interaction: pd.DataFrame = field(default_factory=pd.DataFrame)
    consumption_heating: pd.DataFrame = field(default_factory=pd.DataFrame)
    consumption_base_load: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Environmental data
    weather_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    weather_current: pd.DataFrame = field(default_factory=pd.DataFrame)
    outdoor_temperature: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Building thermal data
    room_temperatures: Dict[str, pd.DataFrame] = field(default_factory=dict)
    heating_relay_states: Dict[str, pd.DataFrame] = field(default_factory=dict)
    shading_states: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Market data
    energy_prices: pd.DataFrame = field(default_factory=pd.DataFrame)

    # EV data (placeholder for future expansion)
    ev_charging: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    date_range: Tuple[datetime, datetime] = field(
        default_factory=lambda: (datetime.min, datetime.max)
    )
    quality_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class QueryDefinition:
    """Defines an InfluxDB query with metadata for parallel execution."""

    name: str
    bucket: str
    measurement: str
    fields: List[str]
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_window: str = "5m"
    aggregation_function: str = "mean"
    additional_filters: List[str] = field(default_factory=list)


class UnifiedDataExtractor:
    """Unified data extraction pipeline with parallel processing and quality assessment."""

    def __init__(self, settings: PEMSSettings):
        """Initialize the unified data extractor."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.UnifiedDataExtractor")

        # InfluxDB setup
        self.client = InfluxDBClient(
            url=settings.influxdb.url,
            token=settings.influxdb.token.get_secret_value(),
            org=settings.influxdb.org,
            timeout=60000,  # Increased timeout for parallel queries
        )
        self.query_api = self.client.query_api()

        # Timezone handling
        self.local_tz = pytz.timezone("Europe/Prague")
        self.utc_tz = pytz.UTC

        # Define all query configurations
        self._define_query_configurations()

    def __del__(self):
        """Clean up InfluxDB client."""
        if hasattr(self, "client"):
            self.client.close()

    def _define_query_configurations(self) -> None:
        """Define all InfluxDB query configurations for parallel execution."""
        self.query_configs = {
            "pv_production": QueryDefinition(
                name="pv_production",
                bucket=self.settings.influxdb.bucket_solar,
                measurement="solar",
                fields=[
                    "InputPower",
                    "PV1InputPower",
                    "PV2InputPower",
                    "PV1Voltage",
                    "PV2Voltage",
                    "TodayGenerateEnergy",
                    "INVPowerToLocalLoad",
                    "InverterTemperature",
                    "InverterStatus",
                ],
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "battery_storage": QueryDefinition(
                name="battery_storage",
                bucket=self.settings.influxdb.bucket_solar,
                measurement="solar",
                fields=[
                    "ChargePower",
                    "DischargePower",
                    "SOC",
                    "BatteryVoltage",
                    "BatteryTemperature",
                    "BatteryCurrent",
                ],
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "grid_interaction": QueryDefinition(
                name="grid_interaction",
                bucket=self.settings.influxdb.bucket_solar,
                measurement="solar",
                fields=[
                    "ACPowerToUser",
                    "ACPowerToGrid",
                    "EnergyToGridToday",
                    "EnergyToUserToday",
                    "LocalLoadEnergyToday",
                ],
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "heating_relays": QueryDefinition(
                name="heating_relays",
                bucket=self.settings.influxdb.bucket_loxone,
                measurement="relay",
                fields=["relay_state"],
                tags={"tag1": "heating"},
                aggregation_window="5m",
                aggregation_function="last",
            ),
            "room_temperatures": QueryDefinition(
                name="room_temperatures",
                bucket=self.settings.influxdb.bucket_loxone,
                measurement="temperature",
                fields=["temperature"],
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "room_humidity": QueryDefinition(
                name="room_humidity",
                bucket=self.settings.influxdb.bucket_loxone,
                measurement="humidity",
                fields=["humidity"],
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "weather_forecast": QueryDefinition(
                name="weather_forecast",
                bucket=self.settings.influxdb.bucket_weather,
                measurement="weather_forecast",
                fields=[
                    "temperature_2m",
                    "relativehumidity_2m",
                    "windspeed_10m",
                    "cloudcover",
                    "precipitation",
                    "shortwave_radiation",
                    "direct_radiation",
                    "diffuse_radiation",
                    "uv_index",
                    "apparent_temperature",
                    "surface_pressure",
                ],
                aggregation_window="15m",
                aggregation_function="mean",
            ),
            "weather_current": QueryDefinition(
                name="weather_current",
                bucket=self.settings.influxdb.bucket_loxone,
                measurement="current_weather",
                fields=[
                    "absolute_solar_irradiance",
                    "current_temperature",
                    "pressure",
                    "relative_humidity",
                    "wind_direction",
                    "sun_direction",
                    "sun_elevation",
                    "precipitation",
                    "brightness",
                ],
                aggregation_window="15m",
                aggregation_function="mean",
            ),
            "outdoor_temperature": QueryDefinition(
                name="outdoor_temperature",
                bucket=self.settings.influxdb.bucket_solar,
                measurement="teplomer",
                fields=["temperature"],
                tags={"topic": "teplomer/TC"},
                aggregation_window="5m",
                aggregation_function="mean",
            ),
            "energy_prices": QueryDefinition(
                name="energy_prices",
                bucket="ote_prices",
                measurement="electricity_prices",
                fields=["price_czk_kwh"],
                aggregation_window="1h",
                aggregation_function="mean",
            ),
            "shading_states": QueryDefinition(
                name="shading_states",
                bucket=self.settings.influxdb.bucket_loxone,
                measurement="relay",
                fields=["relay_state"],
                tags={"tag1": "shading"},
                aggregation_window="15m",
                aggregation_function="last",
            ),
        }

    async def extract_complete_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        include_quality_analysis: bool = True,
    ) -> EnergyDataset:
        """
        Extract complete energy dataset with parallel processing.

        Args:
            start_date: Start of extraction period
            end_date: End of extraction period
            include_quality_analysis: Whether to calculate quality scores

        Returns:
            Complete EnergyDataset with all available data
        """
        self.logger.info(
            f"Starting unified data extraction: {start_date} to {end_date}"
        )

        # Create dataset container
        dataset = EnergyDataset(
            extraction_timestamp=datetime.now(), date_range=(start_date, end_date)
        )

        # Execute all queries in parallel
        extraction_tasks = []
        for config_name, config in self.query_configs.items():
            task = asyncio.create_task(
                self._execute_single_extraction(config, start_date, end_date),
                name=config_name,
            )
            extraction_tasks.append(task)

        # Wait for all extractions to complete
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Process results and populate dataset
        for i, (config_name, result) in enumerate(
            zip(self.query_configs.keys(), results)
        ):
            if isinstance(result, Exception):
                self.logger.error(f"Extraction failed for {config_name}: {result}")
                continue

            data = result

            # Handle both DataFrame and dict results (room-based data)
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    self.logger.warning(f"No data extracted for {config_name}")
                    continue
            elif isinstance(data, dict):
                if not data:  # Empty dict
                    self.logger.warning(f"No data extracted for {config_name}")
                    continue
            else:
                self.logger.warning(
                    f"Unexpected data type for {config_name}: {type(data)}"
                )
                continue

            # Map results to dataset fields
            self._populate_dataset_field(dataset, config_name, data)

        # Post-process data (calculate derived metrics, merge related data)
        self._post_process_dataset(dataset)

        # Calculate data quality scores
        if include_quality_analysis:
            dataset.quality_scores = self.calculate_data_quality_score(dataset)

        # Log extraction summary
        self._log_extraction_summary(dataset)

        return dataset

    async def _execute_single_extraction(
        self, config: QueryDefinition, start_date: datetime, end_date: datetime
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Execute a single InfluxDB query based on configuration."""
        try:
            # Build Flux query
            query = self._build_flux_query(config, start_date, end_date)

            # Execute query
            tables = self.query_api.query(query)

            if not tables:
                return pd.DataFrame()

            # Convert to DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    record_data = {
                        "timestamp": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }

                    # Add tag values if present
                    for tag_key in config.tags.keys():
                        if tag_key in record.values:
                            record_data[tag_key] = record.values[tag_key]

                    # Add room information if available
                    if "room" in record.values:
                        record_data["room"] = record.values["room"]

                    records.append(record_data)

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)

            # Process DataFrame based on data type
            if "room" in df.columns:
                # Room-based data - return as multi-index or dict structure
                return self._process_room_data(df, config)
            else:
                # Time-series data - pivot and set index
                df_pivot = df.pivot_table(
                    index="timestamp", columns="field", values="value", aggfunc="mean"
                ).reset_index()

                df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"])
                df_pivot.set_index("timestamp", inplace=True)

                return df_pivot

        except Exception as e:
            self.logger.error(f"Query execution failed for {config.name}: {e}")
            return pd.DataFrame()

    def _build_flux_query(
        self, config: QueryDefinition, start_date: datetime, end_date: datetime
    ) -> str:
        """Build Flux query string from configuration."""
        # Handle timezone conversion properly
        # If input dates are naive, assume they are in Prague timezone
        if start_date.tzinfo is None:
            start_date = self.local_tz.localize(start_date)
        if end_date.tzinfo is None:
            end_date = self.local_tz.localize(end_date)

        # Convert to UTC for InfluxDB
        start_utc = start_date.astimezone(self.utc_tz)
        end_utc = end_date.astimezone(self.utc_tz)

        # Base query
        query_parts = [
            f'from(bucket: "{config.bucket}")',
            f"|> range(start: {start_utc.isoformat()}, stop: {end_utc.isoformat()})",
            f'|> filter(fn: (r) => r["_measurement"] == "{config.measurement}")',
        ]

        # Add field filters
        if config.fields:
            field_filters = " or ".join(
                [f'r["_field"] == "{field}"' for field in config.fields]
            )
            query_parts.append(f"|> filter(fn: (r) => {field_filters})")

        # Add tag filters
        for tag_key, tag_value in config.tags.items():
            query_parts.append(f'|> filter(fn: (r) => r["{tag_key}"] == "{tag_value}")')

        # Add additional filters
        for filter_expr in config.additional_filters:
            query_parts.append(f"|> filter(fn: (r) => {filter_expr})")

        # Add aggregation
        query_parts.append(
            f"|> aggregateWindow(every: {config.aggregation_window}, "
            f"fn: {config.aggregation_function}, createEmpty: false)"
        )

        # Keep relevant columns
        if "room" in [tag for tag in config.tags.keys()] or config.name in [
            "heating_relays",
            "room_temperatures",
            "room_humidity",
        ]:
            query_parts.append(
                '|> keep(columns: ["_time", "_value", "_field", "room"])'
            )
        else:
            query_parts.append('|> keep(columns: ["_time", "_value", "_field"])')

        return "\n  ".join(query_parts)

    def _process_room_data(
        self, df: pd.DataFrame, config: QueryDefinition
    ) -> Dict[str, pd.DataFrame]:
        """Process room-based data into dictionary of DataFrames by room."""
        room_data = {}

        if "room" not in df.columns:
            # Extract room from field name for temperature/humidity data
            if config.name in ["room_temperatures", "room_humidity"]:
                df = df.copy()
                df["room"] = df["field"].str.extract(r"(?:temperature|humidity)_(.+)$")[
                    0
                ]
                df = df.dropna(subset=["room"])

        for room_name, room_df in df.groupby("room"):
            if room_name and room_name != "unknown":
                # Create pivot table for the room
                room_pivot = room_df.pivot_table(
                    index="timestamp", columns="field", values="value", aggfunc="mean"
                ).reset_index()

                room_pivot["timestamp"] = pd.to_datetime(room_pivot["timestamp"])
                room_pivot.set_index("timestamp", inplace=True)

                # Add derived metrics for relay data
                if config.name == "heating_relays":
                    room_power_kw = get_room_power(room_name)
                    if "relay_state" in room_pivot.columns:
                        room_pivot["power_kw"] = (
                            room_pivot["relay_state"] * room_power_kw
                        )
                        room_pivot["power_w"] = room_pivot["power_kw"] * 1000

                room_data[room_name] = room_pivot

        return room_data

    def _populate_dataset_field(
        self, dataset: EnergyDataset, config_name: str, data: Union[pd.DataFrame, Dict]
    ) -> None:
        """Populate the appropriate dataset field with extracted data."""
        field_mapping = {
            "pv_production": "pv_production",
            "battery_storage": "battery_storage",
            "grid_interaction": "grid_interaction",
            "heating_relays": "heating_relay_states",
            "room_temperatures": "room_temperatures",
            "room_humidity": "room_temperatures",  # Merge with temperatures
            "weather_forecast": "weather_forecast",
            "weather_current": "weather_current",
            "outdoor_temperature": "outdoor_temperature",
            "energy_prices": "energy_prices",
            "shading_states": "shading_states",
        }

        field_name = field_mapping.get(config_name, config_name)

        if isinstance(data, dict):
            # Room-based data
            if field_name == "heating_relay_states":
                dataset.heating_relay_states = data
            elif config_name == "room_temperatures":
                dataset.room_temperatures.update(data)
            elif config_name == "room_humidity":
                # Merge humidity data with existing temperature data
                for room_name, humidity_df in data.items():
                    if room_name in dataset.room_temperatures:
                        dataset.room_temperatures[
                            room_name
                        ] = dataset.room_temperatures[room_name].join(
                            humidity_df, how="outer", rsuffix="_humidity"
                        )
                    else:
                        dataset.room_temperatures[room_name] = humidity_df
        else:
            # Time-series data
            setattr(dataset, field_name, data)

    def _post_process_dataset(self, dataset: EnergyDataset) -> None:
        """Calculate derived metrics and merge related data."""
        # Calculate PV production metrics
        if not dataset.pv_production.empty:
            df = dataset.pv_production

            # Total PV power if individual strings available
            if "PV1InputPower" in df.columns and "PV2InputPower" in df.columns:
                df["total_pv_power"] = df["PV1InputPower"].fillna(0) + df[
                    "PV2InputPower"
                ].fillna(0)
                df["pv_string_balance"] = df["PV1InputPower"] / (
                    df["total_pv_power"] + 1e-6
                )

            # Solar energy calculation (5-minute intervals)
            if "InputPower" in df.columns:
                df["solar_energy_kwh"] = df["InputPower"] * (5 / 60) / 1000

        # Calculate battery metrics
        if not dataset.battery_storage.empty:
            df = dataset.battery_storage

            # Net battery power
            if "ChargePower" in df.columns and "DischargePower" in df.columns:
                df["net_battery_power"] = df["ChargePower"].fillna(0) - df[
                    "DischargePower"
                ].fillna(0)
                df["battery_energy_change_kwh"] = (
                    df["net_battery_power"] * (5 / 60) / 1000
                )

            # Validate with voltage/current if available
            if "BatteryVoltage" in df.columns and "BatteryCurrent" in df.columns:
                df["calculated_battery_power"] = (
                    df["BatteryVoltage"] * df["BatteryCurrent"]
                )

        # Calculate heating consumption from relay states
        if dataset.heating_relay_states:
            heating_consumption = []

            for room_name, relay_df in dataset.heating_relay_states.items():
                if "power_w" in relay_df.columns:
                    heating_consumption.append(
                        relay_df["power_w"].rename(f"{room_name}_heating")
                    )

            if heating_consumption:
                consumption_df = pd.concat(heating_consumption, axis=1).fillna(0)
                consumption_df["total_heating_power"] = consumption_df.sum(axis=1)
                consumption_df["total_heating_energy_kwh"] = (
                    consumption_df["total_heating_power"] * (5 / 60) / 1000
                )
                dataset.consumption_heating = consumption_df

        # Merge grid interaction with total consumption to calculate base load
        if not dataset.grid_interaction.empty and not dataset.consumption_heating.empty:
            grid_df = dataset.grid_interaction
            heating_df = dataset.consumption_heating

            # Align time indices
            merged_df = grid_df.join(
                heating_df[["total_heating_power"]], how="outer"
            ).fillna(0)

            # Calculate base load (total consumption minus heating)
            if "ACPowerToUser" in merged_df.columns:
                merged_df["base_load_power"] = (
                    merged_df["ACPowerToUser"] - merged_df["total_heating_power"]
                )
                merged_df["base_load_power"] = merged_df["base_load_power"].clip(
                    lower=0
                )  # No negative base load
                merged_df["base_load_energy_kwh"] = (
                    merged_df["base_load_power"] * (5 / 60) / 1000
                )

                dataset.consumption_base_load = merged_df[
                    ["base_load_power", "base_load_energy_kwh"]
                ].copy()

    def calculate_data_quality_score(
        self, dataset: EnergyDataset
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive data quality metrics for ML readiness."""
        quality_scores = {}

        # Process all DataFrame fields
        for field_name in dataset.__dataclass_fields__.keys():
            field_value = getattr(dataset, field_name)

            if isinstance(field_value, pd.DataFrame) and not field_value.empty:
                quality_scores[field_name] = self._calculate_df_quality(
                    field_value, field_name
                )

            elif isinstance(field_value, dict):
                # Room-based data
                for room_name, room_df in field_value.items():
                    if isinstance(room_df, pd.DataFrame) and not room_df.empty:
                        quality_scores[
                            f"{field_name}_{room_name}"
                        ] = self._calculate_df_quality(
                            room_df, f"{field_name}_{room_name}"
                        )

        return quality_scores

    def _calculate_df_quality(
        self, df: pd.DataFrame, data_name: str
    ) -> Dict[str, float]:
        """Calculate quality metrics for a single DataFrame."""
        # Completeness score
        completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))

        # Consistency score (outlier detection)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency = 1.0
        if len(numeric_cols) > 0:
            z_scores = np.abs(
                stats.zscore(df[numeric_cols].fillna(0), nan_policy="omit")
            )
            outlier_rate = (z_scores > 3).sum().sum() / (len(df) * len(numeric_cols))
            consistency = 1 - min(outlier_rate, 0.1) / 0.1

        # Temporal consistency
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_freq = pd.Timedelta(minutes=5)  # Expected 5-minute frequency
                freq_consistency = (time_diffs == expected_freq).mean()
                # Allow some tolerance for minor timing variations
                freq_tolerance = (
                    (time_diffs >= expected_freq * 0.8)
                    & (time_diffs <= expected_freq * 1.2)
                ).mean()
                temporal_consistency = max(freq_consistency, freq_tolerance)
            else:
                temporal_consistency = 1.0
        else:
            temporal_consistency = 1.0

        # Value range reasonableness
        reasonableness = 1.0
        if len(numeric_cols) > 0:
            # Check for reasonable value ranges based on field names
            range_checks = []
            for col in numeric_cols:
                col_lower = col.lower()
                if "temperature" in col_lower:
                    # Temperature should be between -40°C and 60°C
                    reasonable = ((df[col] >= -40) & (df[col] <= 60)).mean()
                elif "power" in col_lower and "kw" not in col_lower:
                    # Power in watts should be positive and < 100kW for residential
                    reasonable = ((df[col] >= 0) & (df[col] <= 100000)).mean()
                elif "soc" in col_lower or "percentage" in col_lower:
                    # Percentages should be 0-100
                    reasonable = ((df[col] >= 0) & (df[col] <= 100)).mean()
                elif "voltage" in col_lower:
                    # Battery voltage typically 40-60V, grid 220-240V
                    reasonable = ((df[col] >= 0) & (df[col] <= 500)).mean()
                else:
                    # Generic positive value check
                    reasonable = (df[col] >= 0).mean()

                range_checks.append(reasonable)

            if range_checks:
                reasonableness = np.mean(range_checks)

        # Combined score
        overall_score = (
            completeness * 0.3
            + consistency * 0.25
            + temporal_consistency * 0.25
            + reasonableness * 0.2
        )

        return {
            "overall": overall_score,
            "completeness": completeness,
            "consistency": consistency,
            "temporal_consistency": temporal_consistency,
            "reasonableness": reasonableness,
        }

    def _log_extraction_summary(self, dataset: EnergyDataset) -> None:
        """Log comprehensive extraction summary."""
        total_records = 0
        data_summaries = []

        # Count records from all fields
        for field_name in dataset.__dataclass_fields__.keys():
            field_value = getattr(dataset, field_name)

            if isinstance(field_value, pd.DataFrame) and not field_value.empty:
                records = len(field_value)
                total_records += records
                quality = dataset.quality_scores.get(field_name, {}).get("overall", 0)
                data_summaries.append(
                    f"  {field_name}: {records} records (quality: {quality:.2f})"
                )

            elif isinstance(field_value, dict):
                room_count = len(
                    [
                        df
                        for df in field_value.values()
                        if isinstance(df, pd.DataFrame) and not df.empty
                    ]
                )
                room_records = sum(
                    len(df)
                    for df in field_value.values()
                    if isinstance(df, pd.DataFrame) and not df.empty
                )
                total_records += room_records
                if room_records > 0:
                    avg_quality = np.mean(
                        [
                            dataset.quality_scores.get(f"{field_name}_{room}", {}).get(
                                "overall", 0
                            )
                            for room in field_value.keys()
                        ]
                    )
                    data_summaries.append(
                        f"  {field_name}: {room_count} rooms, {room_records} total records "
                        f"(avg quality: {avg_quality:.2f})"
                    )

        self.logger.info(f"Extraction completed successfully:")
        self.logger.info(f"  Total records: {total_records}")
        self.logger.info(
            f"  Date range: {dataset.date_range[0]} to {dataset.date_range[1]}"
        )
        self.logger.info(f"  Data components:")
        for summary in data_summaries:
            self.logger.info(summary)

        # Log quality warnings
        low_quality_components = []
        for component, scores in dataset.quality_scores.items():
            if scores.get("overall", 1) < 0.8:
                low_quality_components.append(
                    f"{component} (quality: {scores['overall']:.2f})"
                )

        if low_quality_components:
            self.logger.warning(
                f"Low quality data detected in: {', '.join(low_quality_components)}"
            )

    async def get_available_data_ranges(self) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Query InfluxDB to find available data ranges for each measurement.

        Returns:
            Dictionary mapping measurement names to (start_date, end_date) tuples
        """
        self.logger.info("Querying available data ranges from InfluxDB")

        available_ranges = {}

        # Query each bucket for data availability
        buckets_to_check = [
            (
                self.settings.influxdb.bucket_loxone,
                ["temperature", "humidity", "relay"],
            ),
            (self.settings.influxdb.bucket_solar, ["solarAPI", "growatt", "teplomer"]),
            (self.settings.influxdb.bucket_weather, ["weather_forecast"]),
            ("ote_prices", ["electricity_prices"]),
        ]

        with InfluxDBClient(
            url=self.settings.influxdb.url,
            token=self.settings.influxdb.token.get_secret_value(),
            org=self.settings.influxdb.org,
        ) as client:
            query_api = client.query_api()

            for bucket, measurements in buckets_to_check:
                for measurement in measurements:
                    # Query to find min and max time for each measurement
                    query = f"""
                    from(bucket: "{bucket}")
                      |> range(start: -3y)
                      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                      |> group()
                      |> reduce(
                          fn: (r, accumulator) => ({{
                            minTime: if r._time < accumulator.minTime then r._time else accumulator.minTime,
                            maxTime: if r._time > accumulator.maxTime then r._time else accumulator.maxTime,
                            count: accumulator.count + 1
                          }}),
                          identity: {{minTime: 3000-01-01T00:00:00Z, maxTime: 1970-01-01T00:00:00Z, count: 0}}
                      )
                    """

                    try:
                        result = query_api.query(
                            org=self.settings.influxdb.org, query=query
                        )

                        if result and len(result) > 0 and len(result[0].records) > 0:
                            record = result[0].records[0]
                            if (
                                record.get_value()
                                and record.get_value().get("count", 0) > 0
                            ):
                                min_time = record.get_value().get("minTime")
                                max_time = record.get_value().get("maxTime")

                                # Convert to datetime objects
                                if isinstance(min_time, str):
                                    min_time = datetime.fromisoformat(
                                        min_time.replace("Z", "+00:00")
                                    )
                                if isinstance(max_time, str):
                                    max_time = datetime.fromisoformat(
                                        max_time.replace("Z", "+00:00")
                                    )

                                available_ranges[f"{bucket}.{measurement}"] = (
                                    min_time,
                                    max_time,
                                )
                                self.logger.info(
                                    f"  {bucket}.{measurement}: {min_time} to {max_time}"
                                )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to query range for {bucket}.{measurement}: {e}"
                        )

        return available_ranges

    async def get_available_winter_months(self) -> List[Tuple[int, int]]:
        """
        Query InfluxDB to find available winter months (Nov-Mar) with thermal data.

        Returns:
            List of (year, month) tuples for available winter months
        """
        self.logger.info("Querying available winter months from InfluxDB")

        winter_months = []

        # Query temperature data to find available months
        query = """
        from(bucket: "loxone")
          |> range(start: -3y)
          |> filter(fn: (r) => r["_measurement"] == "temperature")
          |> aggregateWindow(every: 1mo, fn: count, createEmpty: false)
          |> filter(fn: (r) => r._value > 0)
          |> group()
          |> sort(columns: ["_time"])
        """

        with InfluxDBClient(
            url=self.settings.influxdb.url,
            token=self.settings.influxdb.token.get_secret_value(),
            org=self.settings.influxdb.org,
        ) as client:
            query_api = client.query_api()

            try:
                result = query_api.query(org=self.settings.influxdb.org, query=query)

                if result:
                    for table in result:
                        for record in table.records:
                            timestamp = record.get_time()
                            if timestamp:
                                # Check if it's a winter month (Nov-Mar)
                                month = timestamp.month
                                year = timestamp.year

                                if month in [11, 12, 1, 2, 3]:
                                    winter_months.append((year, month))

                # Remove duplicates and sort
                winter_months = sorted(list(set(winter_months)))

                # Group consecutive winter months
                if winter_months:
                    self.logger.info("Available winter months:")
                    current_winter_start = None
                    prev_year, prev_month = winter_months[0]

                    for year, month in winter_months:
                        # Check if this is part of the same winter
                        if current_winter_start is None:
                            current_winter_start = (year, month)
                        elif not (
                            (month == prev_month + 1 and year == prev_year)
                            or (
                                month == 1
                                and prev_month == 12
                                and year == prev_year + 1
                            )
                        ):
                            # End of a winter period
                            self.logger.info(
                                f"  Winter {current_winter_start[0]}/{current_winter_start[0]+1}: "
                                f"{current_winter_start[1]}/{current_winter_start[0]} to "
                                f"{prev_month}/{prev_year}"
                            )
                            current_winter_start = (year, month)

                        prev_year, prev_month = year, month

                    # Log the last winter period
                    if current_winter_start:
                        self.logger.info(
                            f"  Winter {current_winter_start[0]}/{current_winter_start[0]+1}: "
                            f"{current_winter_start[1]}/{current_winter_start[0]} to "
                            f"{prev_month}/{prev_year}"
                        )

            except Exception as e:
                self.logger.error(f"Failed to query winter months: {e}")

        return winter_months

    def get_validation_report(self, dataset: EnergyDataset) -> Dict[str, Any]:
        """Generate comprehensive validation report for extracted dataset."""
        required_components = [
            "pv_production",
            "room_temperatures",
            "heating_relay_states",
        ]
        recommended_components = [
            "battery_storage",
            "weather_forecast",
            "energy_prices",
        ]

        validation_report = {
            "extraction_time": dataset.extraction_timestamp,
            "date_range": dataset.date_range,
            "is_ml_ready": True,
            "missing_required": [],
            "missing_recommended": [],
            "quality_issues": [],
            "recommendations": [],
        }

        # Check required components
        for component in required_components:
            field_value = getattr(dataset, component)
            is_empty = (
                isinstance(field_value, pd.DataFrame) and field_value.empty
            ) or (isinstance(field_value, dict) and len(field_value) == 0)

            if is_empty:
                validation_report["missing_required"].append(component)
                validation_report["is_ml_ready"] = False

        # Check recommended components
        for component in recommended_components:
            field_value = getattr(dataset, component)
            is_empty = (
                isinstance(field_value, pd.DataFrame) and field_value.empty
            ) or (isinstance(field_value, dict) and len(field_value) == 0)

            if is_empty:
                validation_report["missing_recommended"].append(component)

        # Analyze quality scores
        for component, scores in dataset.quality_scores.items():
            overall_quality = scores.get("overall", 0)

            if overall_quality < 0.7:
                validation_report["quality_issues"].append(
                    {
                        "component": component,
                        "overall_quality": overall_quality,
                        "issues": {
                            "completeness": scores.get("completeness", 1),
                            "consistency": scores.get("consistency", 1),
                            "temporal_consistency": scores.get(
                                "temporal_consistency", 1
                            ),
                            "reasonableness": scores.get("reasonableness", 1),
                        },
                    }
                )

        # Generate recommendations
        if validation_report["missing_required"]:
            validation_report["recommendations"].append(
                f"Critical: Missing required data components: {', '.join(validation_report['missing_required'])}"
            )

        if validation_report["missing_recommended"]:
            validation_report["recommendations"].append(
                f"Recommended: Add missing components for better ML performance: {', '.join(validation_report['missing_recommended'])}"
            )

        if validation_report["quality_issues"]:
            for issue in validation_report["quality_issues"]:
                component = issue["component"]
                problems = [k for k, v in issue["issues"].items() if v < 0.8]
                validation_report["recommendations"].append(
                    f"Quality: {component} has issues with {', '.join(problems)}"
                )

        # Calculate overall ML readiness score
        ml_readiness_factors = [
            len(validation_report["missing_required"]) == 0,  # Required data present
            len(validation_report["quality_issues"]) < 3,  # Quality issues manageable
            len(dataset.quality_scores) > 5,  # Sufficient data variety
        ]

        validation_report["ml_readiness_score"] = sum(ml_readiness_factors) / len(
            ml_readiness_factors
        )

        return validation_report
