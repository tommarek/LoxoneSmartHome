# Winter Thermal Analysis Guide

This guide shows how to run thermal analysis on winter months (December + January) to get the most reliable thermal parameters.

## Why Winter Months?

Winter months provide the best data for thermal analysis because:
- **Maximum heating usage** → More heating cycles to analyze
- **Largest temperature differences** → Better signal-to-noise ratio
- **Stable outdoor conditions** → More predictable heat losses
- **Minimal solar gains** → Cleaner thermal signatures

## Quick Commands

### Run Winter Analysis (Most Recent Winter)
```bash
# Automatic: Last December + January
python run_analysis.py --winter

# Or manually specify months
python run_analysis.py --month "dec,jan" --thermal-only
```

### Specific Winter Season
```bash
# December 2024 + January 2025 (most recent complete winter)
python run_analysis.py --month "december,january" --year 2024 --thermal-only

# December 2023 + January 2024 
python run_analysis.py --month "12,1" --year 2023 --thermal-only
```

### Custom Month Combinations
```bash
# Just December 2024
python run_analysis.py --month "december" --year 2024 --thermal-only

# Extended winter season (Nov-Feb)
python run_analysis.py --month "11,12,1,2" --year 2024 --thermal-only

# Compare multiple winters
python run_analysis.py --start 2023-12-01 --end 2025-02-28 --thermal-only
```

## Month Format Options

The `--month` parameter accepts various formats:

```bash
# Full names
--month "december,january"

# Abbreviations  
--month "dec,jan"

# Numbers
--month "12,1"

# Mixed formats
--month "dec,1"
```

## Analysis Focus

Winter analysis automatically:
- ✅ **Enables thermal analysis** with sustained heating cycle detection
- ✅ **Enables relay pattern analysis** for heating behavior
- ✅ **Enables weather correlation** for outdoor temperature coupling
- ✅ **Enables base load analysis** for winter consumption patterns
- ❌ **Disables PV analysis** (less relevant in winter)

## Expected Results

Winter thermal analysis provides:

### Thermal Parameters
- **Time constants**: 15-25 hours (typical for well-insulated rooms)
- **Thermal capacitance**: 5-15 MJ/K (realistic for residential rooms)
- **Heat loss coefficients**: More accurate due to large ΔT
- **Room coupling**: Better detection of inter-room heat transfer

### Heating Cycles
- **Sustained heating cycles**: 2+ hour heating periods followed by 3+ hour decay
- **Peak temperature timing**: Thermal inertia and heat redistribution analysis
- **Energy efficiency**: Heat storage vs. heat loss ratios
- **Heat loss estimation**: To outside and adjacent rooms

### Quality Indicators
- **Lower variability** in thermal parameters (winter conditions are more stable)
- **More heating cycles** for statistical significance
- **Better decay curves** due to larger temperature differences

## Example Output

```
PEMS v2 Comprehensive Analysis
==================================================
Month-specific Analysis:
  December 2024: 2024-12-01 to 2024-12-31
  January 2025: 2025-01-01 to 2025-01-31
Combined Period: 2024-12-01 to 2025-01-31
Total Days: 62 days

Enabled Analysis Types:
  ✗ Pv
  ✓ Thermal
  ✓ Base Load  
  ✓ Relay Patterns
  ✓ Weather Correlation
==================================================
```

## Tips for Best Results

1. **Use complete months**: Avoid partial months for consistent statistics
2. **Include both Dec + Jan**: Captures the full winter heating pattern
3. **Check data availability**: Ensure your InfluxDB has data for the specified period
4. **Compare multiple winters**: Look for consistency across different years
5. **Verify relay data**: Make sure heating relay states are properly recorded

## Troubleshooting

If you get poor thermal parameter estimates:

1. **Check data quality**: Verify temperature and relay data exists
2. **Extend date range**: Use more months if heating cycles are sparse
3. **Check heating patterns**: Ensure sustained heating periods exist
4. **Verify room power ratings**: Update `room_power_ratings_kw` in settings

## Next Steps

After running winter analysis:

1. **Review thermal parameters** in the generated reports
2. **Check heating efficiency** from the cycle analysis
3. **Compare rooms** to identify poorly performing areas  
4. **Use parameters** for heating optimization and energy modeling