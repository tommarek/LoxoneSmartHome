# OTE Prices InfluxDB Bucket Schema

## Overview
This document provides a comprehensive overview of the data structure in the **ote_prices** bucket, containing electricity pricing data from the Czech electricity market operator (OTE).

---

## Data Structure Summary

### Measurements Available
- `electricity_prices`

### Fields Available
- `price_czk_kwh`

### Tags
- No specific tags identified (likely using default `_` tag or no tags)

---

## Detailed Field Mapping

### Electricity Pricing Data

| Measurement | Field | Description |
|-------------|-------|-------------|
| electricity_prices | price_czk_kwh | Electricity price in Czech Koruna per kilowatt-hour |

---

## Data Description

### electricity_prices Measurement
This measurement contains spot electricity prices from the Czech electricity market, typically updated at regular intervals (likely hourly) to reflect market conditions.

**Field Details:**
- **price_czk_kwh**: The electricity price expressed in Czech Koruna (CZK) per kilowatt-hour (kWh)
  - Currency: Czech Koruna (CZK)
  - Unit: per kWh
  - Frequency: Likely hourly updates
  - Source: OTE (Czech electricity market operator)

---

## Query Examples

### Get latest electricity price:
```flux
from(bucket: "ote_prices")
|> range(start: -24h)
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> filter(fn: (r) => r._field == "price_czk_kwh")
|> last()
```

### Get hourly prices for today:
```flux
from(bucket: "ote_prices")
|> range(start: today())
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> filter(fn: (r) => r._field == "price_czk_kwh")
|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
```

### Get price statistics for the last week:
```flux
from(bucket: "ote_prices")
|> range(start: -7d)
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> filter(fn: (r) => r._field == "price_czk_kwh")
|> aggregateWindow(every: 1d, fn: mean, createEmpty: false)
```

### Find minimum and maximum prices in the last month:
```flux
from(bucket: "ote_prices")
|> range(start: -30d)
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> filter(fn: (r) => r._field == "price_czk_kwh")
|> group()
|> aggregateWindow(every: inf, fn: max, createEmpty: false)

// For minimum, change fn: max to fn: min
```

### Get average price by hour of day (last 30 days):
```flux
from(bucket: "ote_prices")
|> range(start: -30d)
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> filter(fn: (r) => r._field == "price_czk_kwh")
|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
|> map(fn: (r) => ({r with hour: date.hour(t: r._time)}))
|> group(columns: ["hour"])
|> mean(column: "_value")
```

---

## Use Cases

This data can be used for:

1. **Energy Cost Optimization**: Monitor real-time electricity prices to optimize energy consumption timing
2. **Smart Home Integration**: Integrate with your Loxone system to automatically control high-energy devices based on price levels
3. **Cost Analysis**: Track electricity costs over time and identify patterns
4. **Budget Planning**: Forecast energy costs based on historical price data
5. **Peak/Off-Peak Analysis**: Identify the most cost-effective times to run energy-intensive appliances

---

## Integration with Loxone System

Since you have both the `loxone` and `ote_prices` buckets, you could create powerful automation and analysis:

### Cost-Aware Heating Control:
```flux
// Get current electricity price
currentPrice = from(bucket: "ote_prices")
|> range(start: -1h)
|> filter(fn: (r) => r._measurement == "electricity_prices")
|> last()

// Get heating relay states
heatingStatus = from(bucket: "loxone")
|> range(start: -1h)
|> filter(fn: (r) => r._measurement == "relay" and r.tag1 == "heating")
|> last()

// Combine for cost analysis
```

### Daily Energy Cost Calculation:
```flux
// Combine electricity usage patterns with pricing data
```

---

## Summary Statistics
- **Total Measurements**: 1
- **Total Fields**: 1
- **Data Type**: Time-series electricity pricing
- **Currency**: Czech Koruna (CZK)
- **Unit**: per kWh
- **Update Frequency**: hourly
- **Source**: OTE (Czech Electricity Market Operator)