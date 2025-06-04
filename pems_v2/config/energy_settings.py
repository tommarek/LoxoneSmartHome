"""Energy system configuration for PEMS v2."""

from typing import Any, Dict

# Room power ratings in kW (from your heating patterns notebook)
ROOM_CONFIG = {
    "rooms": {
        "hosti": {"power_kw": 2.02, "area_m2": 20, "zone": "living"},
        "chodba_dole": {"power_kw": 1.8, "area_m2": 15, "zone": "circulation"},
        "chodba_nahore": {"power_kw": 1.2, "area_m2": 10, "zone": "circulation"},
        "koupelna_dole": {"power_kw": 1.5, "area_m2": 8, "zone": "wet"},
        "koupelna_nahore": {"power_kw": 1.2, "area_m2": 6, "zone": "wet"},
        "kuchyn": {"power_kw": 2.5, "area_m2": 25, "zone": "living"},
        "loznice": {"power_kw": 1.8, "area_m2": 18, "zone": "sleeping"},
        "obyvak": {"power_kw": 3.0, "area_m2": 35, "zone": "living"},
        "pracovna": {"power_kw": 1.5, "area_m2": 12, "zone": "working"},
        "wc_dole": {"power_kw": 0.8, "area_m2": 4, "zone": "wet"},
        "wc_nahore": {"power_kw": 0.8, "area_m2": 4, "zone": "wet"},
        # Add more rooms as needed
    },
    "system": {
        "battery_capacity_kwh": 10.0,  # Your actual battery capacity
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
        "inverter_capacity_kw": 10.0,
        "grid_connection_kw": 20.0,
        "pv_peak_power_kw": 15.0,  # Your PV system peak power
    },
    "thermal": {
        "default_setpoint_day": 21.0,
        "default_setpoint_night": 19.0,
        "comfort_band": 0.5,  # ±°C
        "minimum_runtime_minutes": 15,  # Minimum heating on/off time
        "thermal_mass_factor": 0.8,  # Thermal inertia factor
    },
}

# Energy consumption categories for analysis
CONSUMPTION_CATEGORIES = {
    "heating": {
        "measurement": "relay",
        "tag_filter": "tag1 == 'heating'",
        "description": "Space heating consumption",
    },
    "hot_water": {
        "measurement": "relay",
        "tag_filter": "tag1 == 'hot_water'",
        "description": "Hot water heating consumption",
    },
    "ventilation": {
        "measurement": "relay",
        "tag_filter": "tag1 == 'ventilation'",
        "description": "Ventilation system consumption",
    },
    "appliances": {
        "measurement": "power",
        "tag_filter": "tag1 == 'appliances'",
        "description": "General appliance consumption",
    },
    "lighting": {
        "measurement": "power",
        "tag_filter": "tag1 == 'lighting'",
        "description": "Lighting consumption",
    },
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "max_missing_percentage": 10.0,  # Max % of missing data allowed
    "max_gap_hours": 2.0,  # Max time gap in hours
    "min_data_points_per_day": 48,  # Minimum data points per day (15min intervals)
    "temperature_range": (-20.0, 50.0),  # Valid temperature range °C
    "power_range": (0.0, 50000.0),  # Valid power range W
    "soc_range": (0.0, 100.0),  # Valid SOC range %
}


def get_room_power(room_name: str) -> float:
    """Get power rating for a room in kW."""
    return ROOM_CONFIG["rooms"].get(room_name, {}).get("power_kw", 1.0)


def get_total_heating_power() -> float:
    """Get total heating power capacity in kW."""
    return sum(room["power_kw"] for room in ROOM_CONFIG["rooms"].values())


def get_rooms_by_zone(zone: str) -> Dict[str, Dict[str, Any]]:
    """Get all rooms in a specific zone."""
    return {
        name: config for name, config in ROOM_CONFIG["rooms"].items() if config.get("zone") == zone
    }
