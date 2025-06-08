"""
Energy system configuration and physical parameters for PEMS v2.

This module contains the physical configuration of the energy system including
room specifications, thermal parameters, and system constraints. These values
are derived from actual measurements and system specifications and should reflect
the real-world characteristics of the smart home installation.

Key Components:
- ROOM_CONFIG: Physical room parameters including heating power and thermal mass
- CONSUMPTION_CATEGORIES: Energy usage categorization for analysis
- DATA_QUALITY_THRESHOLDS: Data validation parameters for quality assurance

Data Sources:
- Room power ratings derived from heating pattern analysis
- Volumes calculated from architectural drawings
- System specifications from equipment datasheets
- Quality thresholds based on measurement accuracy requirements

Usage:
    from config.energy_settings import ROOM_CONFIG, get_total_heating_power
    
    total_power = get_total_heating_power()  # 18.12 kW
    living_rooms = get_rooms_by_zone("living")
"""

from typing import Any, Dict

# Room-specific configuration derived from heating pattern analysis
# Power ratings represent actual measured heating capacity per room
ROOM_CONFIG = {
    "rooms": {
        # Living spaces - highest power and comfort priority
        "hosti": {"power_kw": 2.02, "volume_m3": 75.67852925, "zone": "living"},
        "obyvak": {"power_kw": 3.0, "volume_m3": 102.6375, "zone": "living"},
        "kuchyne": {"power_kw": 1.8, "volume_m3": 62.0066, "zone": "living"},
        
        # Sleeping areas - comfort important during night hours
        "loznice": {"power_kw": 1.2, "volume_m3": 34.84, "zone": "sleeping"},
        "pokoj_1": {"power_kw": 1.2, "volume_m3": 36.5769, "zone": "sleeping"},
        "pokoj_2": {"power_kw": 1.2, "volume_m3": 36.5769, "zone": "sleeping"},
        
        # Circulation areas - lower priority, can handle temperature swings
        "chodba_dole": {"power_kw": 1.8, "volume_m3": 50.055, "zone": "circulation"},
        "chodba_nahore": {
            "power_kw": 1.2,
            "volume_m3": 53.55708977,
            "zone": "circulation",
        },
        "zadveri": {"power_kw": 0.82, "volume_m3": 23.3325, "zone": "circulation"},
        
        # Wet areas - require consistent heating to prevent moisture issues
        "koupelna_dole": {"power_kw": 0.47, "volume_m3": 15.87374, "zone": "wet"},
        "koupelna_nahore": {"power_kw": 0.62, "volume_m3": 21.86725157, "zone": "wet"},
        "zachod": {"power_kw": 0.22, "volume_m3": 7.630824, "zone": "wet"},
        
        # Working areas - comfort important during occupancy
        "pracovna": {"power_kw": 0.82, "volume_m3": 29.10398, "zone": "working"},
        
        # Storage and utility - lowest priority, can be cooler
        "satna_dole": {"power_kw": 0.82, "volume_m3": 23.79915, "zone": "storage"},
        "satna_nahore": {"power_kw": 0.56, "volume_m3": 31.5675, "zone": "storage"},
        "spajz": {"power_kw": 0.46, "volume_m3": 15.2337, "zone": "storage"},
        "technicka_mistnost": {
            "power_kw": 0.82,
            "volume_m3": 27.34875,
            "zone": "utility",
        },
        # Note: Some rooms may be excluded from relay control data per system configuration
    },
    "system": {
        # Battery energy storage system specifications
        "battery_capacity_kwh": 10.0,  # Total usable battery capacity
        "max_charge_power_kw": 5.0,    # Maximum charging power limit
        "max_discharge_power_kw": 5.0, # Maximum discharging power limit
        
        # Inverter and grid connection specifications
        "inverter_capacity_kw": 10.0,   # AC inverter power rating
        "grid_connection_kw": 20.0,     # Maximum grid connection capacity
        
        # Solar PV system specifications
        "pv_peak_power_kw": 15.0,       # Peak PV generation capacity under STC
    },
    "thermal": {
        # Temperature setpoints for optimal comfort and efficiency
        "default_setpoint_day": 21.0,      # Daytime comfort temperature (°C)
        "default_setpoint_night": 19.0,    # Nighttime energy-saving temperature (°C)
        "comfort_band": 0.5,               # Acceptable temperature deviation (±°C)
        
        # Heating system operational constraints
        "minimum_runtime_minutes": 15,     # Minimum heating cycle to prevent short-cycling
        "thermal_mass_factor": 0.8,        # Building thermal inertia (0.0-1.0)
    },
}

# Energy consumption categories for data analysis and optimization
# Defines how different energy uses are identified and measured in the system
# Note: Only heating is tracked via relay states in current database schema
CONSUMPTION_CATEGORIES = {
    "heating": {
        "measurement": "relay",              # InfluxDB measurement name
        "tag_filter": "tag1 == 'heating'",   # Filter for heating-related relays
        "description": "Space heating consumption by room",
    },
    # Future categories could include:
    # "hot_water": {...},
    # "appliances": {...},
    # "lighting": {...},
}

# Data quality thresholds for validation and outlier detection
# These thresholds ensure data integrity and identify measurement issues
DATA_QUALITY_THRESHOLDS = {
    # Completeness thresholds
    "max_missing_percentage": 10.0,        # Maximum acceptable missing data (10%)
    "max_gap_hours": 2.0,                  # Maximum time gap between measurements
    "min_data_points_per_day": 48,         # Minimum data points per day (15min intervals)
    
    # Physical value ranges for outlier detection
    "temperature_range": (-20.0, 50.0),    # Valid temperature range in °C
    "power_range": (0.0, 50000.0),         # Valid power range in W (0-50kW)
    "soc_range": (0.0, 100.0),             # Valid state of charge range in %
}


def get_room_power(room_name: str) -> float:
    """
    Get heating power rating for a specific room.
    
    Args:
        room_name: Name of the room (must match ROOM_CONFIG keys)
        
    Returns:
        float: Heating power rating in kW, defaults to 1.0 kW if room not found
        
    Example:
        power = get_room_power("obyvak")  # Returns 3.0 kW
    """
    return ROOM_CONFIG["rooms"].get(room_name, {}).get("power_kw", 1.0)


def get_total_heating_power() -> float:
    """
    Calculate total heating power capacity across all rooms.
    
    Returns:
        float: Total heating power in kW (approximately 18.12 kW)
        
    Note:
        This represents the maximum simultaneous heating load if all
        rooms are heating at once, which is used for system sizing
        and optimization constraints.
    """
    return sum(room["power_kw"] for room in ROOM_CONFIG["rooms"].values())


def get_rooms_by_zone(zone: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all rooms belonging to a specific functional zone.
    
    Args:
        zone: Zone name ("living", "sleeping", "circulation", "wet", "working", "storage", "utility")
        
    Returns:
        dict: Dictionary of room configurations for the specified zone
        
    Example:
        living_rooms = get_rooms_by_zone("living")
        # Returns: {"hosti": {...}, "obyvak": {...}, "kuchyne": {...}}
        
    Zone Classifications:
        - living: Main living areas with highest comfort priority
        - sleeping: Bedrooms requiring comfort during night hours
        - circulation: Hallways and passages with lower priority
        - wet: Bathrooms requiring consistent heating
        - working: Home office areas
        - storage: Closets and storage areas with lowest priority
        - utility: Technical rooms with minimal heating needs
    """
    return {
        name: config
        for name, config in ROOM_CONFIG["rooms"].items()
        if config.get("zone") == zone
    }
