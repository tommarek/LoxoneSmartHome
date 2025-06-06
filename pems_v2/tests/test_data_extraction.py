#!/usr/bin/env python3
"""
Test data extraction functionality with real InfluxDB connection.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.core.data_extraction import DataExtractor
from config.settings import PEMSSettings


async def test_data_extraction():
    """Test basic data extraction functionality."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        # Load settings with explicit configuration for your system
        logger.info("Loading settings...")
        settings = PEMSSettings(
            influxdb={
                "url": "http://192.168.0.201:8086",
                "token": (
                    "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1"
                    "QezDh132utLbXi-IL8h9A=="
                ),
                "org": "loxone",
                "bucket_solar": "solar",
                "bucket_loxone": "loxone",
                "bucket_weather": "weather_forecast",
            }
        )
        logger.info(f"InfluxDB URL: {settings.influxdb.url}")

        # Create data extractor
        logger.info("Creating data extractor...")
        extractor = DataExtractor(settings)

        # Test date range - last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        logger.info(f"Testing data extraction from {start_date} to {end_date}")

        # Test relay data extraction (the important one for your system)
        logger.info("\n" + "=" * 50)
        logger.info("TESTING RELAY DATA EXTRACTION")
        logger.info("=" * 50)

        try:
            # Test the relay-specific extraction method
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "relay_analyzer",
                Path(__file__).parent.parent / "test_relay_analysis.py",
            )
            relay_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(relay_module)

            analyzer = relay_module.RelayAnalyzer()
            relay_data = await analyzer.extract_relay_data(start_date, end_date)

            logger.info(
                f"✓ Relay data extraction successful: {len(relay_data)} records"
            )
            if not relay_data.empty:
                logger.info(f"  Columns: {list(relay_data.columns)}")
                logger.info(f"  Rooms found: {relay_data['room'].unique().tolist()}")
                logger.info(
                    f"  Date range: {relay_data.index.min()} to {relay_data.index.max()}"
                )
        except Exception as e:
            logger.error(f"✗ Relay data extraction failed: {e}")

        # Test temperature data extraction
        logger.info("\n" + "=" * 50)
        logger.info("TESTING TEMPERATURE DATA EXTRACTION")
        logger.info("=" * 50)

        try:
            room_data = await extractor.extract_room_temperatures(start_date, end_date)
            logger.info(
                f"✓ Temperature data extraction successful: {len(room_data)} rooms"
            )
            for room_name, room_df in room_data.items():
                logger.info(f"  Room '{room_name}': {len(room_df)} records")
        except Exception as e:
            logger.error(f"✗ Temperature data extraction failed: {e}")

        # Test weather data extraction
        logger.info("\n" + "=" * 50)
        logger.info("TESTING WEATHER DATA EXTRACTION")
        logger.info("=" * 50)

        try:
            weather_data = await extractor.extract_weather_data(start_date, end_date)
            logger.info(
                f"✓ Weather data extraction successful: {len(weather_data)} records"
            )
            if not weather_data.empty:
                logger.info(f"  Columns: {list(weather_data.columns)}")
        except Exception as e:
            logger.error(f"✗ Weather data extraction failed: {e}")

        logger.info("\n" + "=" * 50)
        logger.info("DATA EXTRACTION TEST COMPLETED")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_data_extraction())
    sys.exit(0 if success else 1)
