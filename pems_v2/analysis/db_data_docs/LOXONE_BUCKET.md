# Loxone InfluxDB Bucket Schema

## Overview
This document provides a comprehensive overview of the data structure in the **loxone** bucket, showing all available measurements, fields, and tags.

---

## Data Structure Summary

### Measurements Available
- `brightness`
- `current_weather`
- `humidity`
- `rain`
- `relay`
- `storm_warning`
- `sunshine`
- `target_temp`
- `temperature`
- `wind_speed`

### Tags
- **heating** - Used with relay measurement for heating controls
- **shading** - Used with relay measurement for shading/blinds controls
- **_** - Default/no specific tag

---

## Detailed Field Mapping

### Weather Data

#### Brightness
| Measurement | Field | Tag |
|-------------|-------|-----|
| brightness | brightness | _ |

#### Current Weather
| Measurement | Field | Tag |
|-------------|-------|-----|
| current_weather | absolute_solar_irradiance | _ |
| current_weather | current_temperature | _ |
| current_weather | minutes_past_midnight | _ |
| current_weather | precipitation | _ |
| current_weather | pressure | _ |
| current_weather | relative_humidity | _ |
| current_weather | sun_direction | _ |
| current_weather | sun_elevation | _ |
| current_weather | wind_direction | _ |

#### Other Weather Measurements
| Measurement | Field | Tag |
|-------------|-------|-----|
| rain | rain | _ |
| storm_warning | storm_warning | _ |
| sunshine | sunshine | _ |
| wind_speed | wind_speed | _ |

### Temperature Sensors

#### Temperature Measurements
| Measurement | Field | Tag | Location |
|-------------|-------|-----|----------|
| temperature | temperature_chodba_dole | _ | Hallway downstairs |
| temperature | temperature_chodba_nahore | _ | Hallway upstairs |
| temperature | temperature_hosti | _ | Guest room |
| temperature | temperature_koupelna_dole | _ | Bathroom downstairs |
| temperature | temperature_koupelna_nahore | _ | Bathroom upstairs |
| temperature | temperature_kuchyne | _ | Kitchen |
| temperature | temperature_loznice | _ | Bedroom |
| temperature | temperature_obyvak | _ | Living room |
| temperature | temperature_outside | _ | Outside |
| temperature | temperature_pokoj_1 | _ | Room 1 |
| temperature | temperature_pokoj_2 | _ | Room 2 |
| temperature | temperature_pracovna | _ | Office |
| temperature | temperature_satna_dole | _ | Wardrobe downstairs |
| temperature | temperature_satna_nahore | _ | Wardrobe upstairs |
| temperature | temperature_spajz | _ | Pantry |
| temperature | temperature_technicka_mistnost | _ | Technical room |
| temperature | temperature_zachod | _ | Toilet |
| temperature | temperature_zadveri | _ | Entryway |

#### Target Temperature
| Measurement | Field | Tag |
|-------------|-------|-----|
| target_temp | target_temp | _ |
| target_temp | target_temp_(2) | _ |

### Humidity Sensors

| Measurement | Field | Tag | Location |
|-------------|-------|-----|----------|
| humidity | humidity_chodba_dole | _ | Hallway downstairs |
| humidity | humidity_hosti | _ | Guest room |
| humidity | humidity_koupelna_dole | _ | Bathroom downstairs |
| humidity | humidity_koupelna_nahore | _ | Bathroom upstairs |
| humidity | humidity_kuchyne | _ | Kitchen |
| humidity | humidity_loznice | _ | Bedroom |
| humidity | humidity_obyvak | _ | Living room |
| humidity | humidity_pokoj_1 | _ | Room 1 |
| humidity | humidity_pokoj_2 | _ | Room 2 |
| humidity | humidity_pracovna | _ | Office |

### Relay Controls

#### Heating Relays
| Measurement | Field | Tag | Location |
|-------------|-------|-----|----------|
| relay | chodba_dole | heating | Hallway downstairs |
| relay | chodba_nahore | heating | Hallway upstairs |
| relay | hosti | heating | Guest room |
| relay | koupelna_dole | heating | Bathroom downstairs |
| relay | koupelna_nahore | heating | Bathroom upstairs |
| relay | kuchyne | heating | Kitchen |
| relay | loznice | heating | Bedroom |
| relay | obyvak | heating | Living room |
| relay | pokoj_1 | heating | Room 1 |
| relay | pokoj_2 | heating | Room 2 |
| relay | pracovna | heating | Office |
| relay | satna_dole | heating | Wardrobe downstairs |
| relay | satna_nahore | heating | Wardrobe upstairs |
| relay | spajz | heating | Pantry |
| relay | technicka_mistnost | heating | Technical room |
| relay | zachod | heating | Toilet |
| relay | zadveri | heating | Entryway |

#### Shading/Blinds Relays
| Measurement | Field | Tag | Location |
|-------------|-------|-----|----------|
| relay | hosti_vlevo_1 | shading | Guest room left 1 |
| relay | hosti_vlevo_2 | shading | Guest room left 2 |
| relay | hosti_vpravo_1 | shading | Guest room right 1 |
| relay | hosti_vpravo_2 | shading | Guest room right 2 |
| relay | koupelna_1 | shading | Bathroom 1 |
| relay | koupelna_2 | shading | Bathroom 2 |
| relay | kuchyne_vlevo_1 | shading | Kitchen left 1 |
| relay | kuchyne_vlevo_2 | shading | Kitchen left 2 |
| relay | kuchyne_vpravo_1 | shading | Kitchen right 1 |
| relay | kuchyne_vpravo_2 | shading | Kitchen right 2 |
| relay | loznice_1 | shading | Bedroom 1 |
| relay | loznice_2 | shading | Bedroom 2 |
| relay | pokoj_1_1 | shading | Room 1 blind 1 |
| relay | pokoj_1_2 | shading | Room 1 blind 2 |
| relay | pokoj_2_1 | shading | Room 2 blind 1 |
| relay | pokoj_2_2 | shading | Room 2 blind 2 |
| relay | pracovna_1 | shading | Office 1 |
| relay | pracovna_2 | shading | Office 2 |
| relay | satna_1 | shading | Wardrobe 1 |
| relay | satna_2 | shading | Wardrobe 2 |
| relay | zadveri_1 | shading | Entryway 1 |
| relay | zadveri_2 | shading | Entryway 2 |

---

## Query Examples

### Get all temperature data for a specific room:
```flux
from(bucket: "loxone")
|> range(start: -24h)
|> filter(fn: (r) => r._measurement == "temperature" and r._field == "temperature_obyvak")
```

### Get all heating relay states:
```flux
from(bucket: "loxone")
|> range(start: -24h)
|> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating")
```

### Get weather data:
```flux
from(bucket: "loxone")
|> range(start: -24h)
|> filter(fn: (r) => r._measurement == "current_weather")
```

---

## Summary Statistics
- **Total Fields**: 70+
- **Temperature Sensors**: 17 locations
- **Humidity Sensors**: 10 locations  
- **Heating Relays**: 17 zones
- **Shading Relays**: 22 controls
- **Weather Parameters**: 9 measurements