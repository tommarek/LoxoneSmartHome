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

# Add the parent directory to the path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))

from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer
from config.settings import PEMSSettings as Settings


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


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date: {date_str}. Try format: YYYY-MM-DD")


def main_with_args():
    """Main entry point with command line argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PEMS v2 comprehensive analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Last 2 months
  python run_analysis.py --days 60
  
  # Last 6 months  
  python run_analysis.py --months 6
  
  # Specific date range
  python run_analysis.py --start 2024-01-01 --end 2024-03-01
  
  # Last year
  python run_analysis.py --years 1
  
  # Custom analysis types
  python run_analysis.py --days 30 --pv-only
  python run_analysis.py --months 3 --thermal-only
        """,
    )

    # Date range arguments
    date_group = parser.add_argument_group("Date Range Options")
    date_group.add_argument("--start", help="Start date (YYYY-MM-DD)")
    date_group.add_argument("--end", help="End date (YYYY-MM-DD)")
    date_group.add_argument("--days", type=int, help="Number of days back from now")
    date_group.add_argument("--weeks", type=int, help="Number of weeks back from now")
    date_group.add_argument(
        "--months", type=int, help="Number of months back from now (30 days each)"
    )
    date_group.add_argument("--years", type=int, help="Number of years back from now")

    # Analysis type arguments
    analysis_group = parser.add_argument_group("Analysis Types")
    analysis_group.add_argument(
        "--pv-only", action="store_true", help="Run only PV analysis"
    )
    analysis_group.add_argument(
        "--thermal-only", action="store_true", help="Run only thermal analysis"
    )
    analysis_group.add_argument(
        "--base-load-only", action="store_true", help="Run only base load analysis"
    )
    analysis_group.add_argument("--no-pv", action="store_true", help="Skip PV analysis")
    analysis_group.add_argument(
        "--no-thermal", action="store_true", help="Skip thermal analysis"
    )
    analysis_group.add_argument(
        "--no-base-load", action="store_true", help="Skip base load analysis"
    )
    analysis_group.add_argument(
        "--no-relay", action="store_true", help="Skip relay pattern analysis"
    )
    analysis_group.add_argument(
        "--no-weather", action="store_true", help="Skip weather correlation analysis"
    )

    # Preset options
    preset_group = parser.add_argument_group("Preset Options")
    preset_group.add_argument(
        "--quick", action="store_true", help="Quick analysis (last 30 days)"
    )
    preset_group.add_argument(
        "--seasonal", action="store_true", help="Seasonal analysis (last 6 months)"
    )
    preset_group.add_argument(
        "--full", action="store_true", help="Full analysis (last 2 years)"
    )

    args = parser.parse_args()

    # Determine date range
    end_date = datetime.now()
    start_date = None

    if args.start and args.end:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
    elif args.start:
        start_date = parse_date(args.start)
        # Keep end_date as now
    elif args.end:
        end_date = parse_date(args.end)
        start_date = end_date - timedelta(days=730)  # Default 2 years
    elif args.days:
        start_date = end_date - timedelta(days=args.days)
    elif args.weeks:
        start_date = end_date - timedelta(weeks=args.weeks)
    elif args.months:
        start_date = end_date - timedelta(days=args.months * 30)
    elif args.years:
        start_date = end_date - timedelta(days=args.years * 365)
    elif args.quick:
        start_date = end_date - timedelta(days=30)
    elif args.seasonal:
        start_date = end_date - timedelta(days=180)
    elif args.full:
        start_date = end_date - timedelta(days=730)
    else:
        # Default: last 2 months
        start_date = end_date - timedelta(days=60)

    # Determine analysis types
    analysis_types = {
        "pv": True,
        "thermal": True,
        "base_load": True,
        "relay_patterns": True,
        "weather_correlation": True,
    }

    # Handle exclusive analysis types
    if args.pv_only:
        analysis_types = {
            "pv": True,
            "thermal": False,
            "base_load": False,
            "relay_patterns": False,
            "weather_correlation": True,  # Keep for PV correlation
        }
    elif args.thermal_only:
        analysis_types = {
            "pv": False,
            "thermal": True,
            "base_load": False,
            "relay_patterns": True,  # Keep for heating pattern analysis
            "weather_correlation": True,  # Keep for temperature correlation
        }
    elif args.base_load_only:
        analysis_types = {
            "pv": False,
            "thermal": False,
            "base_load": True,
            "relay_patterns": False,
            "weather_correlation": False,
        }
    else:
        # Handle individual disable flags
        if args.no_pv:
            analysis_types["pv"] = False
        if args.no_thermal:
            analysis_types["thermal"] = False
        if args.no_base_load:
            analysis_types["base_load"] = False
        if args.no_relay:
            analysis_types["relay_patterns"] = False
        if args.no_weather:
            analysis_types["weather_correlation"] = False

    # Display analysis plan
    print("PEMS v2 Comprehensive Analysis")
    print("=" * 50)
    print(
        f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"Duration: {(end_date - start_date).days} days")
    print("\nEnabled Analysis Types:")
    for analysis_type, enabled in analysis_types.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {analysis_type.replace('_', ' ').title()}")
    print("=" * 50)

    # Run the analysis
    try:
        results = asyncio.run(run_analysis(start_date, end_date, analysis_types))

        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Results saved to: analysis/results/")
        print("Reports saved to: analysis/reports/")

        return results

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return None
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run with command line arguments if provided, otherwise use default
    if len(sys.argv) > 1:
        main_with_args()
    else:
        # Run the comprehensive analysis pipeline with defaults
        asyncio.run(main())
