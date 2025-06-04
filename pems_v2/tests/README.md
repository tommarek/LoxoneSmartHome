# PEMS v2 Tests

This directory contains clean, organized tests for the PEMS v2 analysis system.

## Test Files

### `test_basic_structure.py`
Tests basic project structure, imports, and dependencies without requiring external connections.

**Usage:**
```bash
make test-basic
# or
python3 tests/test_basic_structure.py
```

### `test_data_extraction.py` 
Tests data extraction from InfluxDB with real connection to your system at 192.168.0.201.

**Usage:**
```bash
make test-extraction
# or
python3 tests/test_data_extraction.py
```

### `test_relay_analysis.py`
Tests the relay-based heating analysis - the core functionality for your Loxone system.

**Usage:**
```bash
make test-relay
# or
python3 tests/test_relay_analysis.py
```

## Running All Tests

To run the complete test suite:
```bash
make test
```

## Test Order

1. **Basic Structure** - Verify project setup
2. **Data Extraction** - Test InfluxDB connectivity  
3. **Relay Analysis** - Test core relay functionality

Each test is independent and can be run separately.