Of course. Here is a more detailed, code-centric TODO list with specific snippets to guide you through fixing the issues in your `pems_v2` project.

The following plan provides code modifications for each identified problem.

---

### ✅ **Detailed TODO with Code Implementation**

#### **Phase 1: Critical Bug Fixes & Refactoring**

These fixes address the immediate crashes and are essential for the system to run.

**1.1. Fix `TypeError` in `BaseLoadAnalyzer` Call**

* **File**: `pems_v2/analysis/pipelines/comprehensive_analysis.py`
* **Problem**: The `BaseLoadAnalyzer.analyze_base_load` method is called without the `pv_data` and `room_data` it requires.
* **Solution**: Pass the missing DataFrame arguments to the method within the `ComprehensiveAnalyzer` pipeline.

**💻 Code Implementation:**

```python
# In pems_v2/analysis/pipelines/comprehensive_analysis.py, method run_analysis

# --- BEFORE ---
# Around line 379
base_load_results = self.base_load_analyzer.analyze_base_load(
    grid_data=preprocessed_data["grid"]
)

# --- AFTER ---
# Pass the required dataframes to the analyzer.
base_load_results = self.base_load_analyzer.analyze_base_load(
    grid_data=preprocessed_data["grid"],
    pv_data=preprocessed_data["pv"],
    room_data=preprocessed_data["room_sensors"],
    relay_data=preprocessed_data["relays"],
    settings=self.settings
)
```

**1.2. Correct Data Handling in `LoxoneFieldAdapter`**

* **File**: `pems_v2/analysis/utils/loxone_adapter.py`
* **Problem**: `standardize_relay_data` fails with `AttributeError: 'Series' object has no attribute 'columns'` when only one relay is processed.
* **Solution**: Check if the input is a pandas Series and, if so, convert it to a DataFrame.

**💻 Code Implementation:**

```python
# In pems_v2/analysis/utils/loxone_adapter.py, method standardize_relay_data

import pandas as pd

# --- BEFORE ---
# Around line 210
def standardize_relay_data(self, relay_df: pd.DataFrame) -> pd.DataFrame:
    # This fails if relay_df is a Series
    if not all(col in relay_df.columns for col in self.expected_relay_cols):
        # ...
    
# --- AFTER ---
def standardize_relay_data(self, relay_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Standardizes relay data, ensuring it is always a DataFrame."""
    if isinstance(relay_df, pd.Series):
        relay_df = relay_df.to_frame() # Convert Series to DataFrame

    if not all(col in relay_df.columns for col in self.expected_relay_cols):
        # ...
```

**1.3. Update Configuration Imports and Access**

* **Files**: `pems_v2/analysis/analyzers/pattern_analysis.py`, `pems_v2/analysis/utils/loxone_adapter.py`
* **Problem**: The code uses an outdated import `from config.energy_settings import get_room_power`. Configuration should be accessed from the `PEMSSettings` object.
* **Solution**:
    1.  Pass the `settings` object to the `RelayPatternAnalyzer`.
    2.  Pass the settings down to the utility functions in `loxone_adapter`.
    3.  Replace the `get_room_power` call with a direct lookup from `settings.room_power_ratings_kw`.

**💻 Code Implementation:**

**Step 1: Update `RelayPatternAnalyzer` to accept settings**

```python
# In pems_v2/analysis/analyzers/pattern_analysis.py

class RelayPatternAnalyzer:
    # --- BEFORE ---
    def __init__(self, loxone_adapter: LoxoneFieldAdapter):
        self.loxone_adapter = loxone_adapter
        # ...

    # --- AFTER ---
    def __init__(self, loxone_adapter: LoxoneFieldAdapter, settings: PEMSSettings):
        self.loxone_adapter = loxone_adapter
        self.settings = settings # Store settings
        # ...

    def _analyze_relay_energy(self, relay_patterns: pd.DataFrame) -> pd.DataFrame:
        # Pass settings to the adapter method
        return self.loxone_adapter.calculate_energy_consumption(relay_patterns, self.settings)

```

**Step 2: Update `LoxoneFieldAdapter` to use settings**

```python
# In pems_v2/analysis/utils/loxone_adapter.py

# --- BEFORE ---
# Around line 385
from config.energy_settings import get_room_power

def calculate_energy_consumption(self, relay_df: pd.DataFrame) -> pd.DataFrame:
    for room, power_kw in get_room_power().items():
        # ...

# --- AFTER ---
# REMOVE the old import at the top of the file

def calculate_energy_consumption(self, relay_df: pd.DataFrame, settings: PEMSSettings) -> pd.DataFrame:
    """Calculates energy consumption for relays using power ratings from settings."""
    # Access power ratings directly from the settings object
    room_power_ratings = settings.room_power_ratings_kw.dict()

    for room, power_kw in room_power_ratings.items():
        if room in relay_df.columns:
            # ...
```

**1.4. Fix `KeyError: 'price'` in Economic Analysis**

* **File**: `pems_v2/analysis/analyzers/pattern_analysis.py`
* **Problem**: The `_analyze_economic_patterns` method fails because the price column is not found. The likely correct column name is `price_czk_kwh`.
* **Solution**: Use the correct column name in the `pd.merge_asof` call and add defensive checks for empty price data.

**💻 Code Implementation:**

```python
# In pems_v2/analysis/analyzers/pattern_analysis.py, method _analyze_economic_patterns

# --- BEFORE ---
# Around line 1166
merged_df = pd.merge_asof(
    sorted_energy, price_data, left_index=True, right_index=True
)
merged_df["cost"] = merged_df["energy_kwh"] * merged_df["price"]

# --- AFTER ---
def _analyze_economic_patterns(self, energy_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the economic impact of relay energy consumption."""
    if price_data is None or price_data.empty:
        logger.warning("Price data is empty, skipping economic pattern analysis.")
        return pd.DataFrame() # Return empty df

    # Ensure the price column exists, assuming 'price_czk_kwh' is the correct name
    if "price_czk_kwh" not in price_data.columns:
        logger.error("Price column 'price_czk_kwh' not found in price_data.")
        return pd.DataFrame()

    sorted_energy = energy_df.sort_index()
    
    # Use the correct column name for merging and calculation
    merged_df = pd.merge_asof(
        sorted_energy, price_data[['price_czk_kwh']], left_index=True, right_index=True
    )
    
    # Forward-fill any gaps in the merged price data
    merged_df['price_czk_kwh'] = merged_df['price_czk_kwh'].ffill()
    
    merged_df["cost_czk"] = merged_df["energy_kwh"] * merged_df["price_czk_kwh"]
    return merged_df
```

---

#### **Phase 2: Logic Improvements & Robustness**

These changes will make your calculations more accurate and the code more reliable.

**2.1. Refine Base Load Calculation Logic**

* **File**: `pems_v2/analysis/analyzers/base_load_analysis.py`
* **Problem**: The base load calculation relies on a potentially flawed energy conservation model.
* **Solution**: Set the more robust `statistical_minimum` method as the primary calculation. Improve its parameters for better accuracy.

**💻 Code Implementation:**

```python
# In pems_v2/analysis/analyzers/base_load_analysis.py, method _calculate_base_load

# --- BEFORE ---
# Logic arbitrarily chooses between two methods.
def _calculate_base_load(self, house_load: pd.Series, controllable_load: pd.Series) -> pd.DataFrame:
    # ... complex logic to switch between methods ...

# --- AFTER ---
def _calculate_base_load(self, house_load: pd.Series, controllable_load: pd.Series) -> pd.DataFrame:
    """
    Calculates the base load using a robust statistical method as the primary approach.
    """
    logger.info("Calculating base load using statistical minimum method.")

    # Primary method: Use a rolling quantile to find the baseline consumption
    # A 24-hour window and a low quantile (e.g., 5%) are robust settings
    statistical_base_load = house_load.rolling(window='24h', min_periods=1).quantile(0.05)
    statistical_base_load.name = "statistical_base_load_kw"

    # Secondary method (for comparison or specific use cases)
    conservation_base_load = (house_load - controllable_load).clip(lower=0)
    conservation_base_load.name = "conservation_base_load_kw"
    
    # Combine results into a single DataFrame
    base_load_df = pd.concat([statistical_base_load, conservation_base_load], axis=1)
    
    # The primary reported value should be the statistical one
    base_load_df["base_load_kw"] = statistical_base_load
    
    avg_base_load = base_load_df["base_load_kw"].mean()
    logger.info(f"Average statistical base load calculated: {avg_base_load:.3f} kW")

    return base_load_df

```

**2.2. Add Input Validation with Logging**

* **Files**: All analyzer entry-point methods.
* **Problem**: It's hard to debug when data passed between components is incorrect.
* **Solution**: Add logging at the beginning of major functions to inspect the received data.

**💻 Code Implementation (Example for `BaseLoadAnalyzer`)**

```python
# In pems_v2/analysis/analyzers/base_load_analysis.py

import logging
logger = logging.getLogger(__name__)

# --- BEFORE ---
def analyze_base_load(self, grid_data: pd.DataFrame, **kwargs) -> dict:
    # Starts calculation immediately

# --- AFTER ---
def analyze_base_load(
    self, 
    grid_data: pd.DataFrame, 
    pv_data: pd.DataFrame, 
    room_data: pd.DataFrame,
    **kwargs
) -> dict:
    """Analyzes the building's base load."""
    logger.info("Starting base load analysis...")
    
    # --- Input Validation Logging ---
    if grid_data.empty:
        logger.error("Grid data is empty. Aborting base load analysis.")
        return {"error": "Empty grid data"}
    if pv_data.empty:
        logger.warning("PV data is empty. Base load calculation may be less accurate.")
    if room_data.empty:
        logger.warning("Room data is empty. Controllable load cannot be determined.")

    logger.debug(f"Grid data shape: {grid_data.shape}, columns: {grid_data.columns.tolist()}")
    logger.debug(f"PV data shape: {pv_data.shape}, columns: {pv_data.columns.tolist()}")
    
    # ... rest of the calculation
```