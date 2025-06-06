"""
Main script to run the complete PEMS v2 analysis pipeline.

This script orchestrates the complete data analysis process using
the new comprehensive analysis framework:
1. Extract historical data from InfluxDB
2. Preprocess and clean the data
3. Run comprehensive analysis (PV, thermal, base load, patterns)
4. Generate detailed reports and visualizations
5. Save results for ML model training
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer
from config.settings import PEMSSettings as Settings

# Add the parent directory to the path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))


async def run_analysis(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    analysis_types: Optional[Dict[str, bool]] = None,
):
    """
    Run the comprehensive analysis pipeline.

    Args:
        start_date: Analysis start date (defaults to 2 years ago)
        end_date: Analysis end date (defaults to now)
        analysis_types: Optional dict to enable/disable specific analyses

    Returns:
        Analysis results dictionary
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("analysis/analysis.log"),
        ],
    )

    logger = logging.getLogger(__name__)

    try:
        # Load settings
        logger.info("Loading settings...")
        settings = Settings()

        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years

        # Set default analysis types if not provided
        if analysis_types is None:
            analysis_types = {
                "pv": True,
                "thermal": True,
                "base_load": True,
                "relay_patterns": True,
                "weather_correlation": True,
            }

        # Create and run comprehensive analysis pipeline
        logger.info("Initializing comprehensive analysis pipeline...")
        analyzer = ComprehensiveAnalyzer(settings)

        # Run the complete comprehensive analysis
        logger.info(
            f"Running analysis from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        results = await analyzer.run_comprehensive_analysis(
            start_date, end_date, analysis_types
        )

        logger.info("Comprehensive analysis pipeline completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Comprehensive analysis pipeline failed: {e}", exc_info=True)
        raise


async def main():
    """Main entry point for the comprehensive analysis pipeline."""
    return await run_analysis()


def run_quick_analysis(days_back: int = 30):
    """
    Run a quick analysis for recent data.

    Args:
        days_back: Number of days to analyze (default: 30)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    return asyncio.run(run_analysis(start_date, end_date))


def run_pv_only_analysis(days_back: int = 365):
    """
    Run analysis focused only on PV production.

    Args:
        days_back: Number of days to analyze (default: 365)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    analysis_types = {
        "pv": True,
        "thermal": False,
        "base_load": False,
        "relay_patterns": False,
        "weather_correlation": True,  # Keep for PV correlation
    }

    return asyncio.run(run_analysis(start_date, end_date, analysis_types))


def run_thermal_only_analysis(days_back: int = 180):
    """
    Run analysis focused only on thermal dynamics.

    Args:
        days_back: Number of days to analyze (default: 180)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    analysis_types = {
        "pv": False,
        "thermal": True,
        "base_load": False,
        "relay_patterns": True,  # Keep for heating pattern analysis
        "weather_correlation": True,  # Keep for temperature correlation
    }

    return asyncio.run(run_analysis(start_date, end_date, analysis_types))


if __name__ == "__main__":
    # Run the comprehensive analysis pipeline
    asyncio.run(main())
