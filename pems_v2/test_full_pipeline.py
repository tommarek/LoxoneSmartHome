#!/usr/bin/env python3
"""
Test the complete PEMS v2 analysis pipeline with actual Loxone data.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from analysis.run_analysis import AnalysisPipeline
from config.settings import PEMSSettings as Settings


async def test_full_pipeline():
    """Test the complete analysis pipeline with real data."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_pipeline.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load settings from environment variables (using .env file)
        logger.info("Loading settings...")
        settings = Settings()
        
        logger.info(f"InfluxDB URL: {settings.influxdb.url}")
        logger.info(f"InfluxDB Org: {settings.influxdb.org}")
        logger.info(f"InfluxDB Bucket: {settings.influxdb.bucket_historical}")
        
        # Define analysis period - last 7 days for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create and run analysis pipeline
        logger.info("Initializing analysis pipeline...")
        pipeline = AnalysisPipeline(settings)
        
        # Run the complete analysis
        logger.info("\n" + "="*60)
        logger.info("STARTING COMPLETE PEMS V2 ANALYSIS PIPELINE")
        logger.info("="*60)
        
        results = await pipeline.run_full_analysis(start_date, end_date)
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print summary of results
        logger.info("\nRESULTS SUMMARY:")
        logger.info("-" * 30)
        
        if 'data_quality' in results:
            logger.info("Data Quality:")
            for data_type, quality in results['data_quality'].items():
                if isinstance(quality, dict):
                    logger.info(f"  {data_type}: {quality.get('total_records', 0)} records, "
                              f"{quality.get('clean_missing_percentage', 0):.1f}% missing")
        
        if 'thermal_analysis' in results:
            logger.info(f"Thermal Analysis: {len(results['thermal_analysis'])} rooms analyzed")
        
        if 'pv_analysis' in results:
            logger.info(f"PV Analysis: {'completed' if results['pv_analysis'] else 'no data'}")
        
        if 'base_load_analysis' in results:
            logger.info(f"Base Load Analysis: {'completed' if results['base_load_analysis'] else 'no data'}")
        
        # Check if output files were created
        output_files = [
            "analysis/results/analysis_results.json",
            "analysis/reports/analysis_summary.txt",
            "analysis/analysis.log"
        ]
        
        logger.info("\nGenerated Files:")
        for file_path in output_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                logger.info(f"  ✓ {file_path} ({size} bytes)")
            else:
                logger.info(f"  ✗ {file_path} (not found)")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    print(f"\nFull Pipeline Test {'PASSED' if success else 'FAILED'}")
    
    # Show the analysis summary if it was generated
    summary_file = Path("analysis/reports/analysis_summary.txt")
    if summary_file.exists():
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY REPORT")
        print("="*60)
        with open(summary_file, 'r') as f:
            print(f.read())