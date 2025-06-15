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

# Add the parent directories to the path so we can import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add pems_v2 directory

try:
    # Try relative imports first (when run as module)
    from ..config.settings import PEMSSettings as Settings
    from .pipelines.comprehensive_analysis import ComprehensiveAnalyzer
except ImportError:
    # Fall back to absolute imports (when run directly)
    from pems_v2.analysis.pipelines.comprehensive_analysis import \
        ComprehensiveAnalyzer
    from pems_v2.config.settings import PEMSSettings as Settings


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


def run_winter_thermal_analysis(year: int = None, analyze_all: bool = False):
    """
    Run thermal analysis for winter months (November-March) for comprehensive heating data.

    Args:
        year: Base year for the winter season. Defaults to the most recent complete winter.
        analyze_all: If True, analyze ALL available winters since data collection began.
    """
    # Query available winter months from InfluxDB
    import sys
    from pathlib import Path

    # Add parent directory to path for imports when run directly
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from pems_v2.analysis.core.unified_data_extractor import \
        UnifiedDataExtractor
    from pems_v2.config.settings import PEMSSettings

    settings = PEMSSettings()
    extractor = UnifiedDataExtractor(settings)

    # Run async method to get available winter months
    winter_months = asyncio.run(extractor.get_available_winter_months())

    if not winter_months:
        print("No winter data found in InfluxDB")
        return None

    # Find the most recent complete winter period
    # Group months by winter season (Nov-Mar)
    winter_seasons = {}
    for year_month in winter_months:
        month_year, month = year_month
        # Determine which winter season this month belongs to
        if month >= 11:  # Nov, Dec
            winter_key = month_year
        else:  # Jan, Feb, Mar
            winter_key = month_year - 1

        if winter_key not in winter_seasons:
            winter_seasons[winter_key] = []
        winter_seasons[winter_key].append((month_year, month))

    # If analyzing all winters, combine all available data
    if analyze_all:
        print("PEMS v2 Comprehensive All-Winters Thermal Analysis")
        print("=" * 60)
        print(f"Found {len(winter_seasons)} winter seasons in database")

        # Find overall date range
        all_winter_months = []
        for winter_year in sorted(winter_seasons.keys()):
            months = winter_seasons[winter_year]
            print(
                f"Winter {winter_year}/{winter_year+1}: {len(months)} months - "
                + f"{', '.join([f'{m[1]}/{m[0]}' for m in sorted(months)])}"
            )
            all_winter_months.extend(months)

        if not all_winter_months:
            print("No winter data found")
            return None

        # Sort all months to find range
        all_winter_months = sorted(all_winter_months)
        first_year, first_month = all_winter_months[0]
        last_year, last_month = all_winter_months[-1]

        # Calculate the last day of the last month
        if last_month == 2:  # February
            last_day = (
                29
                if last_year % 4 == 0 and (last_year % 100 != 0 or last_year % 400 == 0)
                else 28
            )
        elif last_month in [4, 6, 9, 11]:
            last_day = 30
        else:
            last_day = 31

        start_date = datetime(first_year, first_month, 1)
        end_date = datetime(last_year, last_month, last_day, 23, 59, 59)

        print(
            f"\nCombined Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"Total Winter Months: {len(all_winter_months)}")
        print(f"Total Duration: {(end_date - start_date).days + 1} days")
        print("Focus: Multi-year winter thermal dynamics and heating patterns")
        print("Note: Analyzing ALL winter data since September 2022")
        print("=" * 60)

    else:
        # Original single-winter logic
        if year is not None:
            if year not in winter_seasons:
                print(f"No data found for winter {year}/{year+1}")
                return None
            best_winter = year
        else:
            # Find the most recent winter with at least 3 months of data
            best_winter = None
            for winter_year in sorted(winter_seasons.keys(), reverse=True):
                months = winter_seasons[winter_year]
                if len(months) >= 3:  # At least 3 months of winter data
                    best_winter = winter_year
                    break

            if best_winter is None:
                print("No winter season with sufficient data found")
                return None

        # Build date ranges for the selected winter
        all_months = sorted(winter_seasons[best_winter])

        # Calculate start and end dates
        first_year, first_month = all_months[0]
        last_year, last_month = all_months[-1]

        # Calculate the last day of the last month
        if last_month == 2:  # February
            last_day = (
                29
                if last_year % 4 == 0 and (last_year % 100 != 0 or last_year % 400 == 0)
                else 28
            )
        elif last_month in [4, 6, 9, 11]:
            last_day = 30
        else:
            last_day = 31

        start_date = datetime(first_year, first_month, 1)
        end_date = datetime(last_year, last_month, last_day, 23, 59, 59)

        print("PEMS v2 Complete Winter Thermal Analysis")
        print("=" * 60)
        print(f"Winter Season {best_winter}/{best_winter+1}:")
        print(f"Available months: {', '.join([f'{m[1]}/{m[0]}' for m in all_months])}")
        print(
            f"Total Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"Duration: {(end_date - start_date).days + 1} days")
        print("Focus: Complete winter thermal dynamics and heating patterns")
        print("Note: Analyzing all data since September 2022")
        print("=" * 60)

    # Focus on thermal analysis
    analysis_types = {
        "pv": False,  # Less relevant in winter
        "thermal": True,
        "base_load": True,  # Still useful for winter consumption patterns
        "relay_patterns": True,  # Essential for heating pattern analysis
        "weather_correlation": True,  # Critical for thermal analysis
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


def parse_months(month_str: str) -> list:
    """Parse month string into list of month numbers (1-12)."""

    # Month name mappings
    month_names = {
        "january": 1,
        "jan": 1,
        "1": 1,
        "february": 2,
        "feb": 2,
        "2": 2,
        "march": 3,
        "mar": 3,
        "3": 3,
        "april": 4,
        "apr": 4,
        "4": 4,
        "may": 5,
        "5": 5,
        "june": 6,
        "jun": 6,
        "6": 6,
        "july": 7,
        "jul": 7,
        "7": 7,
        "august": 8,
        "aug": 8,
        "8": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "9": 9,
        "october": 10,
        "oct": 10,
        "10": 10,
        "november": 11,
        "nov": 11,
        "11": 11,
        "december": 12,
        "dec": 12,
        "12": 12,
    }

    months = []
    for month_part in month_str.lower().split(","):
        month_part = month_part.strip()
        if month_part in month_names:
            months.append(month_names[month_part])
        else:
            try:
                month_num = int(month_part)
                if 1 <= month_num <= 12:
                    months.append(month_num)
                else:
                    raise ValueError(f"Month number must be 1-12, got {month_num}")
            except ValueError:
                raise ValueError(f"Unable to parse month: {month_part}")

    return months


def get_month_date_ranges(months: list, year: int = None) -> list:
    """Get date ranges for specified months."""

    if year is None:
        current_year = datetime.now().year
        current_month = datetime.now().month

        # Use current year for months that have already passed this year,
        # and previous year for future months (winter heating season)
        year = current_year

    date_ranges = []

    for month in months:
        # Determine the year for this month
        if year is None:
            current_month = datetime.now().month
            current_year = datetime.now().year

            # For winter months (Dec, Jan, Feb), use the most recent occurrence
            if month in [12, 1, 2]:  # Winter months
                if (
                    current_month >= 6
                ):  # After June, use last winter (Dec of prev year, Jan/Feb of current)
                    if month == 12:
                        month_year = current_year - 1  # Last December
                    else:  # Jan or Feb
                        month_year = current_year  # This year's Jan/Feb
                else:  # Before June, use current winter season
                    if month == 12:
                        month_year = current_year - 1  # Last December
                    else:  # Jan or Feb
                        month_year = current_year  # This year's Jan/Feb
            else:  # Non-winter months
                if month <= current_month:
                    month_year = current_year  # This year if already passed
                else:
                    month_year = current_year - 1  # Last year if not yet occurred
        else:
            # When year is specified, use it as the base year
            # For December, use the specified year
            # For January/February, use specified year + 1 if we're looking at a winter season
            if month == 12:
                month_year = year
            elif month in [1, 2] and year is not None:
                # Check if this should be the January/February following the December
                # This is a bit tricky - we'll assume if December is also in the list,
                # then January should be the following year
                if 12 in months and month < 12:
                    month_year = year + 1
                else:
                    month_year = year
            else:
                month_year = year

        # Get start and end dates for the month
        start_date = datetime(month_year, month, 1)

        # Get last day of month
        if month == 12:
            end_date = datetime(month_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(month_year, month + 1, 1) - timedelta(days=1)

        # Set end time to end of day
        end_date = end_date.replace(hour=23, minute=59, second=59)

        date_ranges.append((start_date, end_date))

    return date_ranges


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
  
  # Winter months (November through March) - for complete thermal analysis
  python run_analysis.py --month "november,december,january,february,march"
  python run_analysis.py --month "nov,dec,jan,feb,mar"
  python run_analysis.py --month "11,12,1,2,3"
  
  # Specific month in specific year
  python run_analysis.py --month "december" --year 2024
  
  # Last year
  python run_analysis.py --years 1
  
  # Custom analysis types
  python run_analysis.py --days 30 --pv-only
  python run_analysis.py --month "dec,jan" --thermal-only
  
  # Analyze single winter season
  python run_analysis.py --winter
  python run_analysis.py --winter --year 2023
  
  # Analyze ALL winters for comprehensive thermal data
  python run_analysis.py --all-winters
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
    date_group.add_argument(
        "--month",
        type=str,
        help="Specific month(s) to analyze (e.g., 'december', 'jan', '12', 'dec,jan', '12,1')",
    )
    date_group.add_argument(
        "--year",
        type=int,
        help="Year for month analysis (defaults to current year, or previous year if month hasn't occurred yet)",
    )

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
    preset_group.add_argument(
        "--winter",
        action="store_true",
        help="Complete winter thermal analysis (Nov through Mar)",
    )
    preset_group.add_argument(
        "--all-winters",
        action="store_true",
        help="Analyze ALL available winters since data collection began (Sept 2022)",
    )

    args = parser.parse_args()

    # Determine date range
    end_date = datetime.now()
    start_date = None
    month_ranges = None

    if args.month:
        # Handle month-specific analysis
        try:
            months = parse_months(args.month)
            month_ranges = get_month_date_ranges(months, args.year)

            # For month analysis, combine all ranges into one period
            all_starts = [start for start, _ in month_ranges]
            all_ends = [end for _, end in month_ranges]
            start_date = min(all_starts)
            end_date = max(all_ends)

        except ValueError as e:
            print(f"Error parsing months: {e}")
            return None

    elif args.start and args.end:
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
    elif args.winter or args.all_winters:
        # Use the dedicated winter analysis function
        return run_winter_thermal_analysis(
            year=args.year if args.winter else None, analyze_all=args.all_winters
        )
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

    if month_ranges:
        print(f"Month-specific Analysis:")
        for i, (start, end) in enumerate(month_ranges):
            month_name = start.strftime("%B %Y")
            print(
                f"  {month_name}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            )
        print(
            f"Combined Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        total_days = sum((end - start).days + 1 for start, end in month_ranges)
        print(f"Total Days: {total_days} days")
    else:
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
