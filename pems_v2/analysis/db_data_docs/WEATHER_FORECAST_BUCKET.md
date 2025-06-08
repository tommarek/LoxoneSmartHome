# Weather Forecast InfluxDB Bucket Schema

## Overview
This document provides a comprehensive overview of the data structure in the **weather_forecast** bucket, containing predicted weather data from an online weather forecast service for your location. **Note: This data represents forecasts/predictions, not real-time measurements.**

---

## Data Structure Summary

### Measurements Available
- `weather_forecast` - All forecast parameters

### Data Categories
- **Temperature & Comfort**: 4 fields
- **Precipitation**: 6 fields  
- **Solar Radiation**: 8 fields
- **Wind**: 3 fields
- **Cloud Cover**: 4 fields
- **Air Quality**: 4 fields
- **Atmospheric**: 4 fields
- **Astronomical**: 2 fields
- **Visibility & Pressure**: 2 fields

### Data Type
- **Forecast Data** - Predicted values from weather service API
- **Time-series** - Future weather conditions with timestamps

---

## Detailed Field Mapping

### Temperature & Comfort Indicators

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | temperature_2m | Air temperature at 2m height | °C |
| weather_forecast | temperature_80m | Air temperature at 80m height | °C |
| weather_forecast | apparent_temperature | "Feels like" temperature | °C |
| weather_forecast | dewpoint_2m | Dew point temperature at 2m | °C |

### Humidity & Atmospheric Conditions

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | relativehumidity_2m | Relative humidity at 2m height | % |
| weather_forecast | surface_pressure | Atmospheric pressure at surface | hPa |
| weather_forecast | visibility | Visibility distance | km |
| weather_forecast | ozone | Ozone concentration | μg/m³ |

### Precipitation Forecasts

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | precipitation | General precipitation | mm |
| weather_forecast | precipitation_hours | Hours with precipitation | hours |
| weather_forecast | precipitation_sum | Total precipitation sum | mm |
| weather_forecast | rain | Rain amount | mm |
| weather_forecast | rain_sum | Total rain sum | mm |
| weather_forecast | showers | Shower precipitation | mm |
| weather_forecast | snowfall | Snowfall amount | cm |

### Solar Radiation (Important for Solar System!)

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | shortwave_radiation | Shortwave radiation | W/m² |
| weather_forecast | shortwave_radiation_instant | Instantaneous shortwave radiation | W/m² |
| weather_forecast | shortwave_radiation_sum | Total shortwave radiation | J/m² |
| weather_forecast | direct_radiation | Direct solar radiation | W/m² |
| weather_forecast | direct_radiation_instant | Instantaneous direct radiation | W/m² |
| weather_forecast | direct_normal_irradiance | Direct normal irradiance | W/m² |
| weather_forecast | direct_normal_irradiance_instant | Instantaneous direct normal irradiance | W/m² |
| weather_forecast | diffuse_radiation | Diffuse solar radiation | W/m² |
| weather_forecast | diffuse_radiation_instant | Instantaneous diffuse radiation | W/m² |
| weather_forecast | terrestrial_radiation | Terrestrial (longwave) radiation | W/m² |
| weather_forecast | terrestrial_radiation_instant | Instantaneous terrestrial radiation | W/m² |

### Cloud Cover Analysis

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | cloudcover | Total cloud coverage | % |
| weather_forecast | cloudcover_low | Low-level cloud coverage | % |
| weather_forecast | cloudcover_mid | Mid-level cloud coverage | % |
| weather_forecast | cloudcover_high | High-level cloud coverage | % |

### Wind Conditions

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | windspeed_10m | Wind speed at 10m height | km/h |
| weather_forecast | winddirection_10m | Wind direction at 10m height | ° |
| weather_forecast | windgusts_10m | Wind gusts at 10m height | km/h |

### Air Quality Indicators

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | pm2_5 | Particulate matter 2.5μm | μg/m³ |
| weather_forecast | pm10 | Particulate matter 10μm | μg/m³ |
| weather_forecast | aerosol_optical_depth | Atmospheric aerosol density | - |
| weather_forecast | uv_index | UV radiation index | index |

### Astronomical Data

| Measurement | Field | Description | Unit |
|-------------|-------|-------------|------|
| weather_forecast | sunrise | Sunrise time | timestamp |
| weather_forecast | sunset | Sunset time | timestamp |

---

## Query Examples

### Solar Production Forecasting:
```flux
from(bucket: "weather_forecast")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._measurement == "weather_forecast")
|> filter(fn: (r) => r._field =~ /^(shortwave_radiation|direct_radiation|cloudcover)$/)
|> sort(columns: ["_time"])
```

### Weather Comfort Analysis:
```flux
from(bucket: "weather_forecast")
|> range(start: now(), stop: +7d)
|> filter(fn: (r) => r._measurement == "weather_forecast")
|> filter(fn: (r) => r._field =~ /^(temperature_2m|apparent_temperature|relativehumidity_2m)$/)
|> aggregateWindow(every: 1d, fn: mean, createEmpty: false)
```

### Precipitation Forecast:
```flux
from(bucket: "weather_forecast")
|> range(start: now(), stop: +48h)
|> filter(fn: (r) => r._measurement == "weather_forecast")
|> filter(fn: (r) => r._field =~ /^(precipitation|rain|snowfall)$/)
|> sort(columns: ["_time"])
```

### Wind Conditions for Solar Panel Safety:
```flux
from(bucket: "weather_forecast")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._measurement == "weather_forecast")
|> filter(fn: (r) => r._field =~ /^(windspeed_10m|windgusts_10m)$/)
|> filter(fn: (r) => r._value > 50)  // Alert for high winds
```

---

## Smart Home Integration Opportunities

### Solar System Optimization:
```flux
// Predict solar production based on forecast radiation
solarForecast = from(bucket: "weather_forecast")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._field == "shortwave_radiation")

// Compare with actual solar production
actualSolar = from(bucket: "solar")
|> range(start: -24h)
|> filter(fn: (r) => r._field =~ /^PV.*InputPower$/)
```

### Heating System Preparation:
```flux
// Predict heating needs based on temperature forecast
tempForecast = from(bucket: "weather_forecast")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._field == "temperature_2m")
|> filter(fn: (r) => r._value < 15)  // Cold weather alert

// Integrate with Loxone heating system
heatingZones = from(bucket: "loxone")
|> range(start: -1h)
|> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating")
```

### Energy Cost Optimization:
```flux
// Combine weather forecast with electricity pricing
weatherForecast = from(bucket: "weather_forecast")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._field == "shortwave_radiation")

electricityPrices = from(bucket: "ote_prices")
|> range(start: now(), stop: +24h)
|> filter(fn: (r) => r._field == "price_czk_kwh")

// Plan energy usage based on solar forecast vs electricity prices
```

### Weather-Based Automation:
```flux
// Automated shading control based on solar radiation forecast
radiationForecast = from(bucket: "weather_forecast")
|> range(start: now(), stop: +4h)
|> filter(fn: (r) => r._field == "direct_radiation")
|> filter(fn: (r) => r._value > 800)  // High solar radiation

// Control Loxone shading relays
shadingControls = from(bucket: "loxone")
|> range(start: -1h)
|> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "shading")
```

---

## Forecast Accuracy Analysis

### Compare Forecast vs Actual:
```flux
// Compare forecasted vs actual temperature
forecastTemp = from(bucket: "weather_forecast")
|> range(start: -24h)
|> filter(fn: (r) => r._field == "temperature_2m")

actualTemp = from(bucket: "loxone")
|> range(start: -24h)
|> filter(fn: (r) => r._field == "temperature_outside")

// Calculate forecast accuracy
```

### Solar Prediction Validation:
```flux
// Compare radiation forecast with actual solar production
radiationForecast = from(bucket: "weather_forecast")
|> range(start: -24h)
|> filter(fn: (r) => r._field == "shortwave_radiation")

solarActual = from(bucket: "solar")
|> range(start: -24h)
|> filter(fn: (r) => r._field == "TodayGenerateEnergy")
```

---

## Automation Use Cases

1. **Smart Solar Management**: Use radiation forecasts to optimize battery charging schedules
2. **Predictive Heating**: Pre-heat home before cold weather based on temperature forecasts
3. **Shading Automation**: Automatically control blinds based on solar radiation predictions
4. **Energy Planning**: Schedule high-energy activities during predicted high solar production
5. **Weather Alerts**: Alert for extreme conditions (high winds, storms) affecting solar panels
6. **HVAC Optimization**: Adjust heating/cooling based on apparent temperature forecasts
7. **Air Quality Management**: Control ventilation based on PM2.5/PM10 forecasts

---

## Summary Statistics
- **Total Fields**: 39 forecast parameters
- **Forecast Categories**: 9 major categories
- **Solar-Relevant Fields**: 11 radiation and cloud cover metrics
- **Data Source**: Online weather forecast service
- **Data Type**: Predictive/Forecast (not real-time measurements)
- **Integration Potential**: High synergy with solar, heating, and pricing systems
- **Update Frequency**: Typically updated several times daily