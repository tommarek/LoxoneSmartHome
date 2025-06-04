#!/usr/bin/env python3
"""
Simple test script to verify data extraction from InfluxDB is working.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from analysis.data_extraction import DataExtractor
from config.settings import PEMSSettings as Settings


async def test_data_extraction():
    """Test basic data extraction functionality."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load settings with explicit configuration
        logger.info("Loading settings...")
        settings = Settings(
            influxdb={"url": "http://192.168.0.201:8086",
                     "token": "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A==",
                     "org": "loxone",
                     "bucket_historical": "loxone"}
        )
        logger.info(f"InfluxDB URL: {settings.influxdb.url}")
        logger.info(f"InfluxDB Org: {settings.influxdb.org}")
        logger.info(f"InfluxDB Bucket: {settings.influxdb.bucket_historical}")
        
        # Create data extractor
        logger.info("Creating data extractor...")
        extractor = DataExtractor(settings)
        
        # Test date range - last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Testing data extraction from {start_date} to {end_date}")
        
        # Test InfluxDB connection by trying to extract some data
        logger.info("\n" + "="*50)
        logger.info("TESTING PV DATA EXTRACTION")
        logger.info("="*50)
        
        try:
            pv_data = await extractor.extract_pv_data(start_date, end_date)
            logger.info(f"✓ PV data extraction successful: {len(pv_data)} records")
            if not pv_data.empty:
                logger.info(f"  Columns: {list(pv_data.columns)}")
                logger.info(f"  Date range: {pv_data.index.min()} to {pv_data.index.max()}")
        except Exception as e:
            logger.error(f"✗ PV data extraction failed: {e}")
        
        # Test temperature data extraction
        logger.info("\n" + "="*50)
        logger.info("TESTING TEMPERATURE DATA EXTRACTION") 
        logger.info("="*50)
        
        try:
            room_data = await extractor.extract_room_temperatures(start_date, end_date)
            logger.info(f"✓ Temperature data extraction successful: {len(room_data)} rooms")
            for room_name, room_df in room_data.items():
                logger.info(f"  Room '{room_name}': {len(room_df)} records")
                if not room_df.empty:
                    logger.info(f"    Columns: {list(room_df.columns)}")
        except Exception as e:
            logger.error(f"✗ Temperature data extraction failed: {e}")
        
        # Test weather data extraction
        logger.info("\n" + "="*50)
        logger.info("TESTING WEATHER DATA EXTRACTION")
        logger.info("="*50)
        
        try:
            weather_data = await extractor.extract_weather_data(start_date, end_date)
            logger.info(f"✓ Weather data extraction successful: {len(weather_data)} records")
            if not weather_data.empty:
                logger.info(f"  Columns: {list(weather_data.columns)}")
                logger.info(f"  Date range: {weather_data.index.min()} to {weather_data.index.max()}")
        except Exception as e:
            logger.error(f"✗ Weather data extraction failed: {e}")
        
        # Test energy consumption data extraction
        logger.info("\n" + "="*50)
        logger.info("TESTING ENERGY CONSUMPTION DATA EXTRACTION")
        logger.info("="*50)
        
        try:
            consumption_data = await extractor.extract_energy_consumption(start_date, end_date)
            logger.info(f"✓ Energy consumption data extraction successful: {len(consumption_data)} records")
            if not consumption_data.empty:
                logger.info(f"  Columns: {list(consumption_data.columns)}")
                logger.info(f"  Date range: {consumption_data.index.min()} to {consumption_data.index.max()}")
        except Exception as e:
            logger.error(f"✗ Energy consumption data extraction failed: {e}")
        
        # Test energy prices data extraction
        logger.info("\n" + "="*50)
        logger.info("TESTING ENERGY PRICES DATA EXTRACTION")
        logger.info("="*50)
        
        try:
            price_data = await extractor.extract_energy_prices(start_date, end_date)
            if price_data is not None and not price_data.empty:
                logger.info(f"✓ Energy prices data extraction successful: {len(price_data)} records")
                logger.info(f"  Columns: {list(price_data.columns)}")
                logger.info(f"  Date range: {price_data.index.min()} to {price_data.index.max()}")
            else:
                logger.info("✓ Energy prices data extraction completed (no data found)")
        except Exception as e:
            logger.error(f"✗ Energy prices data extraction failed: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("DATA EXTRACTION TEST COMPLETED")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_data_extraction())
    sys.exit(0 if success else 1)