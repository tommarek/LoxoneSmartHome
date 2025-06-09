#!/usr/bin/env python3
"""
Test data extraction functionality with mock configuration.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from pems_v2.analysis.core.data_extraction import DataExtractor


@pytest.mark.asyncio
async def test_data_extraction_init(pems_test_settings):
    """Test DataExtractor initialization with test settings."""

    # Create data extractor with test settings
    extractor = DataExtractor(pems_test_settings)

    # Verify basic initialization
    assert extractor.settings == pems_test_settings
    assert extractor.logger is not None
    assert hasattr(extractor, "quality_thresholds")

    # Check that InfluxDB client would be configured correctly
    assert pems_test_settings.influxdb.url == "http://test-influxdb:8086"
    assert pems_test_settings.influxdb.org == "test_org"


@pytest.mark.asyncio
async def test_extract_energy_prices_with_mock(pems_test_settings):
    """Test energy price extraction with mocked InfluxDB."""

    with patch(
        "pems_v2.analysis.core.data_extraction.InfluxDBClient"
    ) as mock_client_class:
        # Setup mock InfluxDB client
        mock_client = MagicMock()
        mock_query_api = MagicMock()
        mock_client.query_api.return_value = mock_query_api
        mock_client_class.return_value = mock_client

        # Setup mock query response
        mock_record = MagicMock()
        mock_record.get_time.return_value = datetime.now()
        mock_record.get_field.return_value = "price_czk_kwh"
        mock_record.get_value.return_value = 2.5

        mock_table = MagicMock()
        mock_table.records = [mock_record]
        mock_query_api.query.return_value = [mock_table]

        # Create extractor and test
        extractor = DataExtractor(pems_test_settings)

        start_date = datetime.now() - timedelta(hours=24)
        end_date = datetime.now()

        result = await extractor.extract_energy_prices(start_date, end_date)

        # Verify the bucket_prices setting was used in the query
        query_call = mock_query_api.query.call_args[0][0]
        assert (
            'from(bucket: "test_ote_prices")' in query_call
            or f'from(bucket: "{pems_test_settings.influxdb.bucket_prices}")'
            in query_call
        )


def test_get_room_power(pems_test_settings):
    """Test room power lookup from settings."""

    # Test existing room
    power = pems_test_settings.get_room_power("test_room")
    assert power == 1.0

    # Test non-existing room (should return 0.0)
    power = pems_test_settings.get_room_power("nonexistent_room")
    assert power == 0.0


@pytest.mark.asyncio
async def test_quality_thresholds(pems_test_settings):
    """Test that quality thresholds are properly configured."""

    extractor = DataExtractor(pems_test_settings)

    # Verify quality thresholds are set
    assert extractor.quality_thresholds is not None
    assert "max_missing_percentage" in extractor.quality_thresholds
    assert "max_gap_hours" in extractor.quality_thresholds
    assert extractor.max_missing_percentage == 10.0
    assert extractor.max_gap_hours == 2.0


def test_system_config_validation(mock_system_config):
    """Test that the mock system config is valid."""

    import json

    # Read the config file
    with open(mock_system_config, "r") as f:
        config_data = json.load(f)

    # Validate required sections exist
    assert "system" in config_data
    assert "models" in config_data
    assert "thermal_settings" in config_data
    assert "battery" in config_data
    assert "ev" in config_data

    # Validate test-safe values
    assert config_data["system"]["simulation_mode"] is True
    assert config_data["battery"]["capacity_kwh"] == 5.0  # Small test battery


if __name__ == "__main__":
    # Run with pytest for proper fixture support
    pytest.main([__file__])
