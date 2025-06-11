"""
Loxone Field Adapter for PEMS v2.

This module provides field mapping and standardization for Loxone home automation
system data to make it compatible with the PEMS analysis pipeline.

The Loxone system uses specific naming conventions:
- Temperature fields: temperature_obyvak, temperature_kuchyne, etc.
- Humidity fields: humidity_obyvak, humidity_kuchyne, etc.
- Relay fields: obyvak, kuchyne, etc. (with tag1='heating')
- Target temperature: target_temp
- Solar data: sun_elevation, sun_direction, absolute_solar_irradiance

This adapter standardizes these fields to work with the existing analysis modules.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class LoxoneFieldAdapter:
    """
    Adapter for Loxone data field naming conventions.

    Maps Loxone-specific field names to standardized names used throughout
    the PEMS analysis pipeline.
    """

    # Standard field names used in PEMS analysis
    STANDARD_FIELDS = {
        "temperature": "temperature",
        "humidity": "humidity",
        "target_temp": "target_temp",
        "relay_state": "relay_state",
        "power_kw": "power_kw",
        "power_w": "power_w",
        "heating_on": "heating_on",
    }

    # Loxone field patterns
    LOXONE_PATTERNS = {
        "temperature": [r"temperature_(.+)", r"temp_(.+)"],
        "humidity": [r"humidity_(.+)", r"humid_(.+)"],
        "target_temp": [r"target_temp.*", r"target_temperature.*"],
        "relay": [r"^([a-zA-Z_]+)$"],  # Room names directly as relay fields
        "solar_elevation": [r"sun_elevation", r"solar_elevation"],
        "solar_direction": [r"sun_direction", r"solar_direction", r"sun_azimuth"],
        "solar_irradiance": [r"absolute_solar_irradiance", r"solar_irradiance", r"ghi"],
    }

    # Room name mappings - maintain Czech names as they are in the database
    # No translation needed - we preserve original Czech room names
    ROOM_NAME_MAPPING = {}

    def __init__(self):
        """Initialize the Loxone field adapter."""
        self.logger = logging.getLogger(f"{__name__}.LoxoneFieldAdapter")

    @classmethod
    def standardize_room_data(
        cls, room_df: pd.DataFrame, room_name: str
    ) -> pd.DataFrame:
        """
        Convert Loxone room data fields to standard names.

        Args:
            room_df: DataFrame with Loxone field names
            room_name: Name of the room (used for field identification)

        Returns:
            DataFrame with standardized field names
        """
        if room_df.empty:
            return pd.DataFrame()

        adapter = cls()
        adapter.logger.debug(f"Standardizing room data for {room_name}")
        adapter.logger.debug(f"Input columns: {list(room_df.columns)}")

        standardized = pd.DataFrame(index=room_df.index)

        # Map temperature field
        temp_field = adapter._find_field(room_df.columns, "temperature", room_name)
        if temp_field:
            standardized["temperature"] = room_df[temp_field]
            adapter.logger.debug(f"Mapped temperature: {temp_field} -> temperature")

        # Map humidity field
        humidity_field = adapter._find_field(room_df.columns, "humidity", room_name)
        if humidity_field:
            standardized["humidity"] = room_df[humidity_field]
            adapter.logger.debug(f"Mapped humidity: {humidity_field} -> humidity")

        # Map target temperature field
        target_temp_field = adapter._find_field(room_df.columns, "target_temp")
        if target_temp_field:
            standardized["target_temp"] = room_df[target_temp_field]
            adapter.logger.debug(
                f"Mapped target_temp: {target_temp_field} -> target_temp"
            )

        # Map relay state field (room name directly or with heating suffix)
        relay_field = adapter._find_relay_field(room_df.columns, room_name)
        if relay_field:
            standardized["relay_state"] = room_df[relay_field]
            # Convert to binary (0/1) if needed
            if standardized["relay_state"].dtype != bool:
                standardized["relay_state"] = (standardized["relay_state"] > 0).astype(
                    int
                )
            adapter.logger.debug(f"Mapped relay: {relay_field} -> relay_state")

        # Map any additional fields found in the data
        for col in room_df.columns:
            if col not in [temp_field, humidity_field, target_temp_field, relay_field]:
                # Keep additional fields with their original names
                standardized[col] = room_df[col]

        adapter.logger.debug(f"Output columns: {list(standardized.columns)}")
        adapter.logger.debug(
            f"Standardized {len(standardized)} records for {room_name}"
        )

        return standardized

    @classmethod
    def standardize_weather_data(cls, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Loxone weather data fields.

        Args:
            weather_df: DataFrame with Loxone weather fields

        Returns:
            DataFrame with standardized weather field names
        """
        if weather_df.empty:
            return pd.DataFrame()

        adapter = cls()
        adapter.logger.debug("Standardizing weather data")
        adapter.logger.debug(f"Input columns: {list(weather_df.columns)}")

        standardized = weather_df.copy()

        # Map solar fields
        solar_elevation_field = adapter._find_field(
            weather_df.columns, "solar_elevation"
        )
        if solar_elevation_field:
            standardized["sun_elevation"] = weather_df[solar_elevation_field]
            if solar_elevation_field != "sun_elevation":
                standardized = standardized.drop(columns=[solar_elevation_field])

        solar_direction_field = adapter._find_field(
            weather_df.columns, "solar_direction"
        )
        if solar_direction_field:
            standardized["sun_direction"] = weather_df[solar_direction_field]
            if solar_direction_field != "sun_direction":
                standardized = standardized.drop(columns=[solar_direction_field])

        solar_irradiance_field = adapter._find_field(
            weather_df.columns, "solar_irradiance"
        )
        if solar_irradiance_field:
            standardized["solar_irradiance"] = weather_df[solar_irradiance_field]
            if solar_irradiance_field != "absolute_solar_irradiance":
                standardized = standardized.drop(columns=[solar_irradiance_field])

        adapter.logger.debug(f"Output columns: {list(standardized.columns)}")

        return standardized

    @classmethod
    def standardize_relay_data(
        cls, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Standardize relay data for all rooms.

        Args:
            relay_data: Dictionary with room names as keys and relay DataFrames as values

        Returns:
            Dictionary with standardized relay data
        """
        adapter = cls()
        adapter.logger.info(f"Standardizing relay data for {len(relay_data)} rooms")

        standardized_data = {}

        for room_name, relay_df in relay_data.items():
            if relay_df.empty:
                adapter.logger.warning(f"Empty relay data for room: {room_name}")
                standardized_data[room_name] = pd.DataFrame()
                continue

            # Standardize room name
            standard_room_name = cls.standardize_room_name(room_name)

            # Create standardized DataFrame
            standardized_df = pd.DataFrame(index=relay_df.index)

            # Find relay state column
            relay_field = adapter._find_relay_field(relay_df.columns, room_name)
            if relay_field:
                standardized_df["relay_state"] = relay_df[relay_field]
                # Ensure binary state
                if standardized_df["relay_state"].dtype != bool:
                    standardized_df["relay_state"] = (
                        standardized_df["relay_state"] > 0
                    ).astype(int)

            # Add power calculation if we have power ratings
            if "relay_state" in standardized_df.columns:
                # TODO: Get power rating from config/energy_settings.py
                power_rating_kw = cls._get_room_power_rating(standard_room_name)
                standardized_df["power_kw"] = (
                    standardized_df["relay_state"] * power_rating_kw
                )
                standardized_df["power_w"] = standardized_df["power_kw"] * 1000

            # Copy any additional columns
            for col in relay_df.columns:
                if col != relay_field:
                    standardized_df[col] = relay_df[col]

            standardized_data[standard_room_name] = standardized_df
            adapter.logger.debug(
                f"Standardized relay data for {room_name} -> {standard_room_name}"
            )

        return standardized_data

    @classmethod
    def standardize_room_name(cls, loxone_room_name: str) -> str:
        """
        Standardize Loxone room name by removing prefixes/suffixes but preserving Czech names.

        Args:
            loxone_room_name: Loxone room name (e.g., 'obyvak', 'kuchyne', 'relay_obyvak')

        Returns:
            Standardized Czech room name (e.g., 'obyvak', 'kuchyne')
        """
        # Remove common prefixes/suffixes but preserve the Czech room name
        clean_name = loxone_room_name.lower().strip()
        clean_name = re.sub(r"^(relay_|temp_|temperature_|humidity_)", "", clean_name)
        clean_name = re.sub(r"(_heating|_temp|_relay)$", "", clean_name)

        # Return the cleaned Czech name (no translation to English)
        return clean_name

    @classmethod
    def get_room_list_from_data(cls, data: Dict[str, Any]) -> List[str]:
        """
        Extract list of rooms from various data sources.

        Args:
            data: Dictionary containing rooms, relay_states, or other room data

        Returns:
            List of standardized room names
        """
        rooms = set()

        # From rooms data
        if "rooms" in data and isinstance(data["rooms"], dict):
            for room_name in data["rooms"].keys():
                rooms.add(cls.standardize_room_name(room_name))

        # From relay states
        if "relay_states" in data and isinstance(data["relay_states"], dict):
            for room_name in data["relay_states"].keys():
                rooms.add(cls.standardize_room_name(room_name))

        # From temperature data by looking for temperature columns
        for key, value in data.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                for col in value.columns:
                    # Look for temperature_roomname pattern
                    match = re.match(r"temperature_(.+)", col.lower())
                    if match:
                        room_name = match.group(1)
                        rooms.add(cls.standardize_room_name(room_name))

        return sorted(list(rooms))

    def _find_field(
        self, columns: List[str], field_type: str, room_name: str = None
    ) -> Optional[str]:
        """
        Find a field of given type in the column list.

        Args:
            columns: List of column names to search
            field_type: Type of field to find ('temperature', 'humidity', etc.)
            room_name: Optional room name for room-specific fields

        Returns:
            Column name if found, None otherwise
        """
        if field_type not in self.LOXONE_PATTERNS:
            self.logger.warning(f"Unknown field type: {field_type}")
            return None

        patterns = self.LOXONE_PATTERNS[field_type]

        for col in columns:
            col_lower = col.lower()

            # Check each pattern for this field type
            for pattern in patterns:
                if re.match(pattern, col_lower):
                    # For room-specific fields, verify room name matches
                    if room_name and field_type in ["temperature", "humidity"]:
                        match = re.match(pattern, col_lower)
                        if match and len(match.groups()) > 0:
                            field_room = match.group(1)
                            # Check if room names match (accounting for standardization)
                            if (
                                field_room == room_name.lower()
                                or self.standardize_room_name(field_room)
                                == self.standardize_room_name(room_name)
                            ):
                                return col
                    else:
                        return col

        return None

    def _find_relay_field(self, columns: List[str], room_name: str) -> Optional[str]:
        """
        Find relay field for a specific room.

        Args:
            columns: List of column names to search
            room_name: Name of the room

        Returns:
            Column name if found, None otherwise
        """
        room_lower = room_name.lower()
        standard_room = self.standardize_room_name(room_name)

        # Check for exact room name match
        for col in columns:
            col_lower = col.lower()

            # Standard relay column names (most common)
            if col_lower in ["relay_state", "state", "heating_state", "relay"]:
                return col

            # Direct room name match
            if col_lower == room_lower:
                return col

            # Room name with common prefixes/suffixes
            if col_lower in [
                f"relay_{room_lower}",
                f"{room_lower}_relay",
                f"{room_lower}_heating",
                f"heating_{room_lower}",
            ]:
                return col

            # Check against standardized room name
            if col_lower == standard_room:
                return col

        return None

    @classmethod
    def _get_room_power_rating(cls, room_name: str) -> float:
        """
        Get power rating for a room using energy_settings.py configuration.

        Args:
            room_name: Czech room name (e.g., 'obyvak', 'kuchyne')

        Returns:
            Power rating in kW (default 1.0 kW if not found)
        """
        from config.settings import PEMSSettings

        try:
            settings = PEMSSettings()
            return settings.get_room_power(room_name)
        except Exception:
            # Fallback to default if settings can't be loaded
            return 1.0

    @classmethod
    def validate_data_structure(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and report on Loxone data structure.

        Args:
            data: Dictionary containing various data sources

        Returns:
            Validation report with field mappings and recommendations
        """
        adapter = cls()
        report = {
            "rooms_found": [],
            "field_mappings": {},
            "missing_fields": {},
            "recommendations": [],
        }

        # Check room data
        if "rooms" in data and isinstance(data["rooms"], dict):
            for room_name, room_df in data["rooms"].items():
                if isinstance(room_df, pd.DataFrame) and not room_df.empty:
                    standard_name = cls.standardize_room_name(room_name)
                    report["rooms_found"].append(standard_name)

                    room_mapping = {}

                    # Check for temperature field
                    temp_field = adapter._find_field(
                        room_df.columns, "temperature", room_name
                    )
                    if temp_field:
                        room_mapping["temperature"] = temp_field
                    else:
                        if "missing_fields" not in report:
                            report["missing_fields"] = {}
                        if standard_name not in report["missing_fields"]:
                            report["missing_fields"][standard_name] = []
                        report["missing_fields"][standard_name].append("temperature")

                    # Check for humidity field
                    humidity_field = adapter._find_field(
                        room_df.columns, "humidity", room_name
                    )
                    if humidity_field:
                        room_mapping["humidity"] = humidity_field

                    # Check for relay field
                    relay_field = adapter._find_relay_field(room_df.columns, room_name)
                    if relay_field:
                        room_mapping["relay_state"] = relay_field
                    else:
                        if standard_name not in report["missing_fields"]:
                            report["missing_fields"][standard_name] = []
                        report["missing_fields"][standard_name].append("relay_state")

                    report["field_mappings"][standard_name] = room_mapping

        # Generate recommendations
        if report["missing_fields"]:
            for room, missing in report["missing_fields"].items():
                if "temperature" in missing:
                    report["recommendations"].append(
                        f"Room '{room}': Add temperature field (expected pattern: temperature_{room})"
                    )
                if "relay_state" in missing:
                    report["recommendations"].append(
                        f"Room '{room}': Add relay state field (expected: {room} or relay_{room})"
                    )

        # Check data completeness
        total_rooms = len(report["rooms_found"])
        complete_rooms = len(
            [
                r
                for r in report["field_mappings"].values()
                if "temperature" in r and "relay_state" in r
            ]
        )

        report["completeness"] = {
            "total_rooms": total_rooms,
            "complete_rooms": complete_rooms,
            "completeness_ratio": complete_rooms / total_rooms
            if total_rooms > 0
            else 0,
        }

        adapter.logger.info(
            f"Data validation complete: {complete_rooms}/{total_rooms} rooms have complete data"
        )

        return report


class LoxoneDataIntegrator:
    """
    Higher-level class for integrating Loxone data across different analysis modules.

    This class provides methods to prepare Loxone data for use with existing
    PEMS analysis modules like thermal_analysis.py, pattern_analysis.py, etc.
    """

    def __init__(self):
        """Initialize the Loxone data integrator."""
        self.logger = logging.getLogger(f"{__name__}.LoxoneDataIntegrator")
        self.adapter = LoxoneFieldAdapter()

    def prepare_thermal_analysis_data(
        self,
        rooms_data: Dict[str, pd.DataFrame],
        relay_data: Dict[str, pd.DataFrame],
        weather_data: pd.DataFrame,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Prepare data for thermal analysis module.

        Args:
            rooms_data: Raw room data from Loxone
            relay_data: Raw relay data from Loxone
            weather_data: Weather data

        Returns:
            Tuple of (standardized_room_data, standardized_weather_data)
        """
        self.logger.info("Preparing data for thermal analysis")

        # Standardize room data
        standardized_rooms = {}
        for room_name, room_df in rooms_data.items():
            if room_df.empty:
                continue

            # Standardize the room data
            std_room_df = LoxoneFieldAdapter.standardize_room_data(room_df, room_name)

            # Ensure the room DataFrame index is sorted before any reindexing operations
            if not std_room_df.index.is_monotonic_increasing:
                self.logger.debug(f"Sorting non-monotonic index for room {room_name}")
                std_room_df = std_room_df.sort_index()

            # Add relay state if available
            standard_room_name = LoxoneFieldAdapter.standardize_room_name(room_name)
            if standard_room_name in relay_data or room_name in relay_data:
                relay_key = (
                    standard_room_name
                    if standard_room_name in relay_data
                    else room_name
                )
                relay_df = relay_data[relay_key]

                if not relay_df.empty:
                    # Ensure the relay DataFrame index is also sorted before reindexing
                    if not relay_df.index.is_monotonic_increasing:
                        self.logger.debug(
                            f"Sorting non-monotonic index for relay {relay_key}"
                        )
                        relay_df = relay_df.sort_index()

                    # Align relay data with room data
                    relay_field = self.adapter._find_relay_field(
                        relay_df.columns, room_name
                    )
                    if relay_field:
                        # Resample relay data to match room data frequency
                        # Use last() to get final state in each window (not max() which artificially extends heating periods)
                        relay_resampled = (
                            relay_df[[relay_field]].resample("5min").last()
                        )
                        relay_series = (
                            relay_resampled[relay_field]
                            .reindex(std_room_df.index)
                            .ffill()  # Forward fill is appropriate for relay states
                            .fillna(0)
                        )
                        std_room_df["heating_on"] = (relay_series > 0).astype(int)

            standardized_rooms[standard_room_name] = std_room_df

        # Standardize weather data
        standardized_weather = LoxoneFieldAdapter.standardize_weather_data(weather_data)

        self.logger.info(
            f"Prepared thermal analysis data for {len(standardized_rooms)} rooms"
        )
        return standardized_rooms, standardized_weather

    def prepare_relay_analysis_data(
        self, relay_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for relay pattern analysis.

        Args:
            relay_data: Raw relay data from Loxone

        Returns:
            Standardized relay data with power calculations
        """
        self.logger.info("Preparing data for relay analysis")

        standardized_relay = LoxoneFieldAdapter.standardize_relay_data(relay_data)

        self.logger.info(
            f"Prepared relay analysis data for {len(standardized_relay)} rooms"
        )
        return standardized_relay

    def prepare_pv_analysis_data(
        self, pv_data: pd.DataFrame, weather_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for PV analysis.

        Args:
            pv_data: PV production data
            weather_data: Weather data including Loxone solar fields

        Returns:
            Tuple of (pv_data, enhanced_weather_data)
        """
        self.logger.info("Preparing data for PV analysis")

        # PV data typically doesn't need field standardization
        standardized_pv = pv_data.copy()

        # Enhance weather data with Loxone solar fields
        standardized_weather = LoxoneFieldAdapter.standardize_weather_data(weather_data)

        self.logger.info("Prepared PV analysis data")
        return standardized_pv, standardized_weather
