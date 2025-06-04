#!/usr/bin/env python3
"""
Simple test to verify InfluxDB connection and data availability.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient


async def test_influxdb_connection():
    """Test direct InfluxDB connection."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # InfluxDB configuration
    url = "http://192.168.0.201:8086"
    token = "7HrEuj8kzOS1f-0mjU_GT4hS_9gHfdjUT6j5QAM22oDg0z44DsxmiveTGMqTa0Zl1QezDh132utLbXi-IL8h9A=="
    org = "loxone"
    bucket = "loxone"
    
    try:
        logger.info(f"Connecting to InfluxDB at {url}")
        
        # Create client
        client = InfluxDBClient(url=url, token=token, org=org, timeout=30000)
        query_api = client.query_api()
        
        # Test date range - last 24 hours
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)
        
        logger.info(f"Testing query for last 24 hours: {start_date} to {end_date}")
        
        # Simple query to test connection and see what measurements exist
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> limit(n: 10)
        '''
        
        logger.info("Executing test query...")
        tables = query_api.query(query)
        
        if not tables:
            logger.warning("No data returned from query")
            return False
        
        logger.info(f"✓ Query successful! Found {len(tables)} table(s)")
        
        # Show sample data
        record_count = 0
        measurements = set()
        fields = set()
        
        for table in tables:
            for record in table.records:
                record_count += 1
                measurements.add(record.get_measurement())
                fields.add(record.get_field())
                
                if record_count <= 5:  # Show first 5 records
                    logger.info(f"  Sample record: {record.get_time()} | {record.get_measurement()}.{record.get_field()} = {record.get_value()}")
        
        logger.info(f"Total records found: {record_count}")
        logger.info(f"Measurements found: {sorted(measurements)}")
        logger.info(f"Fields found: {sorted(fields)}")
        
        # Test query for specific data types we're interested in
        logger.info("\n" + "="*50)
        logger.info("TESTING SPECIFIC DATA QUERIES")
        logger.info("="*50)
        
        # Query for energy/power data
        energy_query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "energy" or r["_measurement"] == "power" or r["_measurement"] == "solar")
          |> limit(n: 5)
        '''
        
        logger.info("Looking for energy/power/solar data...")
        energy_tables = query_api.query(energy_query)
        if energy_tables:
            for table in energy_tables:
                for record in table.records:
                    logger.info(f"  Energy data: {record.get_time()} | {record.get_measurement()}.{record.get_field()} = {record.get_value()}")
        else:
            logger.info("  No energy/power/solar data found")
        
        # Query for temperature data
        temp_query = f'''
        from(bucket: "{bucket}")
          |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
          |> filter(fn: (r) => r["_measurement"] == "temperature" or r["_measurement"] == "heating")
          |> limit(n: 5)
        '''
        
        logger.info("Looking for temperature/heating data...")
        temp_tables = query_api.query(temp_query)
        if temp_tables:
            for table in temp_tables:
                for record in table.records:
                    logger.info(f"  Temperature data: {record.get_time()} | {record.get_measurement()}.{record.get_field()} = {record.get_value()}")
        else:
            logger.info("  No temperature/heating data found")
        
        client.close()
        
        logger.info("\n" + "="*50)
        logger.info("CONNECTION TEST SUCCESSFUL!")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_influxdb_connection())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")