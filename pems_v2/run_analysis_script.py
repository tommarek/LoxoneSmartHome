#!/usr/bin/env python3
"""
Run complete PEMS v2 analysis pipeline.

This script runs the full analysis including:
- Data extraction from InfluxDB
- Pattern analysis (PV, thermal, relay)
- Feature engineering
- Visualization generation
- Report creation

Usage:
    python run_analysis_script.py --days 30    # Last 30 days
    python run_analysis_script.py --start 2023-01-01 --end 2024-01-01  # Date range
    python run_analysis_script.py --full       # Last 2 years (full dataset)
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from analysis.run_analysis import AnalysisPipeline
from analysis.visualization import AnalysisVisualizer
from config.settings import PEMSSettings


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PEMS v2 complete analysis pipeline"
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--days",
        type=int,
        help="Number of days back from now to analyze"
    )
    date_group.add_argument(
        "--full",
        action="store_true",
        help="Run full 2-year analysis"
    )
    date_group.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Custom date range (YYYY-MM-DD format)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="analysis/results",
        help="Output directory for results (default: analysis/results)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (optional)"
    )
    
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation (faster for large datasets)"
    )
    
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save extracted data to parquet files for later use"
    )
    
    return parser.parse_args()


def calculate_date_range(args):
    """Calculate start and end dates based on arguments."""
    end_date = datetime.now()
    
    if args.days:
        start_date = end_date - timedelta(days=args.days)
    elif args.full:
        start_date = end_date - timedelta(days=730)  # 2 years
    else:  # date_range
        start_date = datetime.strptime(args.date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(args.date_range[1], "%Y-%m-%d")
    
    return start_date, end_date


async def run_full_analysis(
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    skip_visualizations: bool = False,
    save_data: bool = False
):
    """Run the complete analysis pipeline."""
    
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting PEMS v2 analysis")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Initialize settings and pipeline
    try:
        settings = PEMSSettings()
        pipeline = AnalysisPipeline(settings)
        
        logger.info("Initialized analysis pipeline")
        
        # Run the analysis
        logger.info("Running analysis pipeline...")
        results = await pipeline.run_full_analysis(start_date, end_date)
        
        if not results:
            logger.error("Analysis pipeline returned no results")
            return False
        
        logger.info(f"Analysis completed successfully!")
        logger.info(f"Generated {len(results)} analysis components")
        
        # Save results summary
        summary_file = output_path / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"PEMS v2 Analysis Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Analysis period: {start_date} to {end_date}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Components analyzed:\n")
            for component, data in results.items():
                if isinstance(data, dict):
                    f.write(f"- {component}: {len(data)} items\n")
                else:
                    f.write(f"- {component}: {type(data).__name__}\n")
        
        logger.info(f"Saved analysis summary to {summary_file}")
        
        # Generate visualizations if requested
        if not skip_visualizations:
            logger.info("Generating visualizations...")
            
            try:
                visualizer = AnalysisVisualizer(output_dir=str(output_path / "figures"))
                
                # Generate HTML report
                report_path = output_path / "analysis_report.html"
                visualizer.create_analysis_summary_report(
                    results, 
                    save_path=str(report_path)
                )
                
                logger.info(f"Generated analysis report: {report_path}")
                
                # Generate individual visualizations if we have the right data
                if "data" in results:
                    data = results["data"]
                    
                    # Save static plots
                    static_plots = visualizer.save_static_plots(
                        data, 
                        results, 
                        prefix="pems_analysis"
                    )
                    
                    if static_plots:
                        logger.info(f"Generated {len(static_plots)} static plots")
                        for plot in static_plots:
                            logger.info(f"  - {plot}")
                
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                logger.info("Analysis completed but visualizations failed")
        
        # Save extracted data if requested
        if save_data and "data" in results:
            logger.info("Saving extracted data...")
            
            data = results["data"]
            data_path = output_path / "extracted_data"
            data_path.mkdir(exist_ok=True)
            
            for data_type, df in data.items():
                if hasattr(df, 'to_parquet'):  # pandas DataFrame
                    file_path = data_path / f"{data_type}.parquet"
                    df.to_parquet(file_path)
                    logger.info(f"Saved {data_type} data to {file_path}")
                elif isinstance(df, dict):  # Room data, etc.
                    for key, sub_df in df.items():
                        if hasattr(sub_df, 'to_parquet'):
                            file_path = data_path / f"{data_type}_{key}.parquet"
                            sub_df.to_parquet(file_path)
                            logger.info(f"Saved {data_type}_{key} data to {file_path}")
        
        logger.info("Analysis pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Calculate date range
    start_date, end_date = calculate_date_range(args)
    
    # Validate date range
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1
    
    duration = end_date - start_date
    if duration.days > 1000:
        logger.warning(f"Large date range ({duration.days} days) - this may take a while")
    
    # Run the analysis
    success = await run_full_analysis(
        start_date=start_date,
        end_date=end_date,
        output_dir=args.output_dir,
        skip_visualizations=args.skip_visualizations,
        save_data=args.save_data
    )
    
    if success:
        logger.info("🎉 Analysis completed successfully!")
        return 0
    else:
        logger.error("❌ Analysis failed!")
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)