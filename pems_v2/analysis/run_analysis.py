"""
Main script to run the complete PEMS v2 analysis pipeline.

This script orchestrates the complete data analysis process:
1. Extract historical data from InfluxDB
2. Preprocess and clean the data
3. Run pattern analysis (PV, thermal, base load)
4. Generate visualizations and reports
5. Save results for ML model training
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add the parent directory to the path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))

from analysis.base_load_analysis import BaseLoadAnalyzer
from analysis.data_extraction import DataExtractor
from analysis.pattern_analysis import PVAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer
from config.settings import PEMSSettings as Settings


class DataPreprocessor:
    """Preprocess and clean extracted data."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(f"{__name__}.DataPreprocessor")
        self.quality_report = {}

    def process_dataset(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process and clean a dataset."""
        if df.empty:
            self.logger.warning(f"Empty dataset for {data_type}")
            return df

        self.logger.info(f"Processing {data_type} dataset with {len(df)} records")

        # Remove duplicates
        df_clean = df.loc[~df.index.duplicated(keep="first")]

        # Handle missing values
        if data_type == "pv":
            # For PV data, fill nighttime missing values with 0
            df_clean = self._clean_pv_data(df_clean)
        elif data_type == "temperature":
            # For temperature data, interpolate missing values
            df_clean = self._clean_temperature_data(df_clean)
        elif data_type == "weather":
            # For weather data, forward fill then interpolate
            df_clean = self._clean_weather_data(df_clean)
        else:
            # General cleaning
            df_clean = self._general_cleaning(df_clean)

        # Generate quality report
        self.quality_report[data_type] = self._generate_quality_report(df, df_clean, data_type)

        self.logger.info(
            f"Processed {data_type}: {len(df_clean)} clean records from {len(df)} original"
        )
        return df_clean

    def _clean_pv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean PV production data."""
        df_clean = df.copy()

        # Set negative values to 0 (no negative production)
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].clip(lower=0)

        # Fill missing values during nighttime with 0
        night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        night_mask = df_clean.index.hour.isin(night_hours)

        for col in numeric_cols:
            df_clean.loc[night_mask, col] = df_clean.loc[night_mask, col].fillna(0)
            # Interpolate remaining missing values during daylight
            df_clean[col] = df_clean[col].interpolate(method="linear", limit=6)  # Max 1.5 hours gap

        return df_clean

    def _clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean temperature data."""
        df_clean = df.copy()

        # Remove obviously wrong temperature values
        temp_cols = [col for col in df_clean.columns if "temp" in col.lower()]
        for col in temp_cols:
            if col in df_clean.columns:
                # Remove temperatures outside reasonable range (-10 to 50°C)
                df_clean[col] = df_clean[col].mask((df_clean[col] < -10) | (df_clean[col] > 50))
                # Interpolate missing values (max 30 minutes gap)
                df_clean[col] = df_clean[col].interpolate(method="linear", limit=6)

        return df_clean

    def _clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather data."""
        df_clean = df.copy()

        # Forward fill then interpolate
        df_clean = df_clean.fillna(method="ffill", limit=12)  # Forward fill up to 3 hours
        df_clean = df_clean.interpolate(method="linear", limit=24)  # Interpolate up to 6 hours

        return df_clean

    def _general_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """General data cleaning."""
        df_clean = df.copy()

        # Remove outliers using IQR method for numeric columns
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More lenient than 1.5*IQR
            upper_bound = Q3 + 3 * IQR

            df_clean[col] = df_clean[col].mask(
                (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            )

        # Interpolate missing values
        df_clean = df_clean.interpolate(method="linear", limit=12)

        return df_clean

    def _generate_quality_report(
        self, original_df: pd.DataFrame, clean_df: pd.DataFrame, data_type: str
    ) -> Dict[str, Any]:
        """Generate data quality report."""
        if original_df.empty:
            return {
                "total_records": 0,
                "date_range": (None, None),
                "missing_percentage": 100,
                "time_gaps": [],
                "cleaning_summary": "No data available",
            }

        # Calculate missing data percentage before and after cleaning
        original_missing = original_df.isnull().sum().sum()
        original_total = original_df.size
        original_missing_pct = (
            (original_missing / original_total * 100) if original_total > 0 else 100
        )

        clean_missing = clean_df.isnull().sum().sum()
        clean_total = clean_df.size
        clean_missing_pct = (clean_missing / clean_total * 100) if clean_total > 0 else 100

        # Find significant time gaps (>2 hours)
        if not clean_df.index.empty:
            time_diffs = clean_df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
            time_gaps = [(gap_time, gap_duration) for gap_time, gap_duration in large_gaps.items()]
        else:
            time_gaps = []

        return {
            "total_records": len(original_df),
            "clean_records": len(clean_df),
            "date_range": (
                (original_df.index.min(), original_df.index.max())
                if not original_df.index.empty
                else (None, None)
            ),
            "original_missing_percentage": round(original_missing_pct, 2),
            "clean_missing_percentage": round(clean_missing_pct, 2),
            "time_gaps": time_gaps[:10],  # First 10 gaps
            "cleaning_summary": f"Cleaned {len(original_df)} -> {len(clean_df)} records, missing data: {original_missing_pct:.1f}% -> {clean_missing_pct:.1f}%",
        }


class AnalysisVisualizer:
    """Create visualizations for analysis results."""

    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(f"{__name__}.AnalysisVisualizer")

    def create_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a text summary report."""
        report_lines = [
            "=" * 80,
            "PEMS v2 DATA ANALYSIS SUMMARY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # PV Analysis Summary
        if "pv_analysis" in analysis_results:
            pv_results = analysis_results["pv_analysis"]
            report_lines.extend(
                [
                    "PV PRODUCTION ANALYSIS",
                    "-" * 30,
                ]
            )

            if "basic_stats" in pv_results:
                stats = pv_results["basic_stats"]
                report_lines.extend(
                    [
                        f"• Total Energy Generated: {stats.get('total_energy_kwh', 0):.1f} kWh",
                        f"• Maximum Power: {stats.get('max_power', 0):.1f} W",
                        f"• Capacity Factor: {stats.get('capacity_factor', 0)*100:.1f}%",
                        f"• Peak Production Months: {', '.join(map(str, stats.get('peak_months', [])))}",
                    ]
                )

            if (
                "weather_correlations" in pv_results
                and "strongest_positive" in pv_results["weather_correlations"]
            ):
                strongest = pv_results["weather_correlations"]["strongest_positive"]
                if strongest:
                    report_lines.append(
                        f"• Strongest Weather Correlation: {strongest[0]} ({strongest[1]['correlation']:.3f})"
                    )

            report_lines.append("")

        # Thermal Analysis Summary
        if "thermal_analysis" in analysis_results:
            thermal_results = analysis_results["thermal_analysis"]
            report_lines.extend(
                [
                    "THERMAL DYNAMICS ANALYSIS",
                    "-" * 30,
                ]
            )

            for room_name, room_data in thermal_results.items():
                if room_name == "room_coupling":
                    continue

                if isinstance(room_data, dict) and "basic_stats" in room_data:
                    stats = room_data["basic_stats"]
                    report_lines.extend(
                        [
                            f"Room: {room_name}",
                            f"  • Mean Temperature: {stats.get('mean_temperature', 0):.1f}°C",
                            f"  • Temperature Range: {stats.get('temperature_range', 0):.1f}°C",
                            f"  • Heating Usage: {stats.get('heating_percentage', 0):.1f}%",
                        ]
                    )

                    if "time_constant" in room_data and isinstance(
                        room_data["time_constant"], dict
                    ):
                        tc = room_data["time_constant"]
                        if "time_constant_hours" in tc:
                            report_lines.append(
                                f"  • Time Constant: {tc['time_constant_hours']:.1f} hours"
                            )

            report_lines.append("")

        # Base Load Analysis Summary
        if "base_load_analysis" in analysis_results:
            base_load_results = analysis_results["base_load_analysis"]
            report_lines.extend(
                [
                    "BASE LOAD ANALYSIS",
                    "-" * 30,
                ]
            )

            if "basic_stats" in base_load_results:
                stats = base_load_results["basic_stats"]
                report_lines.extend(
                    [
                        f"• Mean Base Load: {stats.get('mean_base_load', 0):.1f} W",
                        f"• Total Base Load Energy: {stats.get('total_energy_kwh', 0):.1f} kWh",
                        f"• Base Load Percentage: {stats.get('base_load_percentage', 0):.1f}% of total consumption",
                        f"• Peak Hour: {stats.get('peak_hour', 'N/A')}:00",
                        f"• Load Factor: {stats.get('base_load_factor', 0):.3f}",
                    ]
                )

            if (
                "time_patterns" in base_load_results
                and "weekday_vs_weekend" in base_load_results["time_patterns"]
            ):
                pattern = base_load_results["time_patterns"]["weekday_vs_weekend"]
                report_lines.append(
                    f"• Weekend vs Weekday: {pattern.get('weekend_increase', 0):.1f}% higher on weekends"
                )

            report_lines.append("")

        # Data Quality Summary
        if "data_quality" in analysis_results:
            quality = analysis_results["data_quality"]
            report_lines.extend(
                [
                    "DATA QUALITY SUMMARY",
                    "-" * 30,
                ]
            )

            for data_type, report in quality.items():
                if isinstance(report, dict):
                    report_lines.extend(
                        [
                            f"• {data_type.title()}: {report.get('total_records', 0)} records",
                            f"  Missing data: {report.get('clean_missing_percentage', 0):.1f}%",
                            f"  Time gaps: {len(report.get('time_gaps', []))}",
                        ]
                    )

            report_lines.append("")

        # Recommendations
        report_lines.extend(
            [
                "RECOMMENDATIONS",
                "-" * 30,
                "• Monitor PV system performance during identified low-efficiency periods",
                "• Optimize heating schedules based on thermal time constants",
                "• Investigate high base load consumption periods for energy savings",
                "• Consider demand response strategies during peak consumption hours",
                "",
            ]
        )

        report_lines.extend(["=" * 80, "End of Report"])

        return "\n".join(report_lines)


class AnalysisPipeline:
    """Main analysis pipeline coordinator."""

    def __init__(self, settings: Settings):
        """Initialize the analysis pipeline."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.AnalysisPipeline")

        # Initialize components
        self.extractor = DataExtractor(settings)
        self.preprocessor = DataPreprocessor()
        self.pv_analyzer = PVAnalyzer()
        self.thermal_analyzer = ThermalAnalyzer()
        self.base_load_analyzer = BaseLoadAnalyzer()
        self.visualizer = AnalysisVisualizer()

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            "data/raw",
            "data/processed",
            "data/features",
            "analysis/results",
            "analysis/figures",
            "analysis/reports",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def run_full_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        self.logger.info("Starting PEMS v2 Data Analysis Pipeline")
        self.logger.info(
            f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            # Step 1: Extract data
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 1: EXTRACTING DATA FROM INFLUXDB")
            self.logger.info("=" * 50)
            data = await self._extract_all_data(start_date, end_date)

            # Step 2: Preprocess data
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 2: PREPROCESSING DATA")
            self.logger.info("=" * 50)
            processed_data = self._preprocess_all_data(data)

            # Step 3: Run analysis
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 3: RUNNING PATTERN ANALYSIS")
            self.logger.info("=" * 50)
            analysis_results = await self._run_all_analysis(processed_data)

            # Step 4: Save results
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 4: SAVING RESULTS")
            self.logger.info("=" * 50)
            self._save_all_results(processed_data, analysis_results)

            # Step 5: Generate reports
            self.logger.info("\n" + "=" * 50)
            self.logger.info("STEP 5: GENERATING REPORTS")
            self.logger.info("=" * 50)
            self._generate_reports(analysis_results)

            self.logger.info("\n" + "=" * 50)
            self.logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 50)

            return analysis_results

        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {e}", exc_info=True)
            raise

    async def _extract_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Extract all required data from InfluxDB."""
        data = {}

        try:
            # Extract PV data
            self.logger.info("Extracting PV/solar data...")
            data["pv"] = await self.extractor.extract_pv_data(start_date, end_date)
            if not data["pv"].empty:
                self.extractor.save_to_parquet(data["pv"], "pv_data")

            # Extract room temperature data
            self.logger.info("Extracting room temperature data...")
            data["rooms"] = await self.extractor.extract_room_temperatures(start_date, end_date)
            for room_name, room_df in data["rooms"].items():
                if not room_df.empty:
                    self.extractor.save_to_parquet(room_df, f"room_{room_name}")

            # Extract weather data
            self.logger.info("Extracting weather data...")
            data["weather"] = await self.extractor.extract_weather_data(start_date, end_date)
            if not data["weather"].empty:
                self.extractor.save_to_parquet(data["weather"], "weather_data")

            # Extract energy consumption data
            self.logger.info("Extracting energy consumption data...")
            data["consumption"] = await self.extractor.extract_energy_consumption(
                start_date, end_date
            )
            if not data["consumption"].empty:
                self.extractor.save_to_parquet(data["consumption"], "consumption_data")

            # Extract energy prices (optional)
            self.logger.info("Extracting energy price data...")
            try:
                data["prices"] = await self.extractor.extract_energy_prices(start_date, end_date)
                if data["prices"] is not None and not data["prices"].empty:
                    self.extractor.save_to_parquet(data["prices"], "energy_prices")
            except Exception as e:
                self.logger.warning(f"Could not extract energy prices: {e}")
                data["prices"] = None

        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise

        return data

    def _preprocess_all_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess all datasets."""
        processed = {}

        try:
            # Process PV data
            if not data["pv"].empty:
                self.logger.info("Processing PV data...")
                processed["pv"] = self.preprocessor.process_dataset(data["pv"], "pv")
            else:
                processed["pv"] = pd.DataFrame()

            # Process room data
            processed["rooms"] = {}
            for room_name, room_df in data["rooms"].items():
                if not room_df.empty:
                    self.logger.info(f"Processing room data for {room_name}...")
                    processed["rooms"][room_name] = self.preprocessor.process_dataset(
                        room_df, "temperature"
                    )
                else:
                    processed["rooms"][room_name] = pd.DataFrame()

            # Process weather data
            if not data["weather"].empty:
                self.logger.info("Processing weather data...")
                processed["weather"] = self.preprocessor.process_dataset(data["weather"], "weather")
            else:
                processed["weather"] = pd.DataFrame()

            # Process consumption data
            if not data["consumption"].empty:
                self.logger.info("Processing consumption data...")
                processed["consumption"] = self.preprocessor.process_dataset(
                    data["consumption"], "consumption"
                )
            else:
                processed["consumption"] = pd.DataFrame()

            # Process price data if available
            if data["prices"] is not None and not data["prices"].empty:
                self.logger.info("Processing energy price data...")
                processed["prices"] = self.preprocessor.process_dataset(data["prices"], "prices")
            else:
                processed["prices"] = pd.DataFrame()

            # Print data quality report
            self.logger.info("\nDATA QUALITY REPORT:")
            self.logger.info("-" * 40)
            for data_type, report in self.preprocessor.quality_report.items():
                self.logger.info(f"{data_type.upper()}:")
                self.logger.info(
                    f"  Records: {report['total_records']} -> {report['clean_records']}"
                )
                if report["date_range"][0] and report["date_range"][1]:
                    self.logger.info(
                        f"  Date range: {report['date_range'][0].strftime('%Y-%m-%d')} to {report['date_range'][1].strftime('%Y-%m-%d')}"
                    )
                self.logger.info(
                    f"  Missing data: {report['original_missing_percentage']}% -> {report['clean_missing_percentage']}%"
                )
                self.logger.info(f"  Time gaps: {len(report['time_gaps'])}")
                self.logger.info("")

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise

        return processed

    async def _run_all_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all analysis modules."""
        results = {"data_quality": self.preprocessor.quality_report}

        try:
            # PV analysis
            if not data["pv"].empty:
                self.logger.info("Running PV production analysis...")
                results["pv_analysis"] = self.pv_analyzer.analyze_pv_production(
                    data["pv"], data["weather"]
                )
            else:
                self.logger.warning("Skipping PV analysis - no data available")
                results["pv_analysis"] = {}

            # Thermal analysis
            if data["rooms"]:
                self.logger.info("Running thermal dynamics analysis...")
                results["thermal_analysis"] = self.thermal_analyzer.analyze_room_dynamics(
                    data["rooms"], data["weather"]
                )
            else:
                self.logger.warning("Skipping thermal analysis - no room data available")
                results["thermal_analysis"] = {}

            # Base load analysis
            if not data["consumption"].empty:
                self.logger.info("Running base load analysis...")
                results["base_load_analysis"] = self.base_load_analyzer.analyze_base_load(
                    data["consumption"], data["pv"], data["rooms"]
                )
            else:
                self.logger.warning("Skipping base load analysis - no consumption data available")
                results["base_load_analysis"] = {}

        except Exception as e:
            self.logger.error(f"Analysis execution failed: {e}")
            raise

        return results

    def _save_all_results(self, data: Dict[str, Any], results: Dict[str, Any]):
        """Save all results to files."""
        try:
            # Save processed data
            for data_type, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    filepath = Path(f"data/processed/{data_type}_processed.parquet")
                    df.to_parquet(filepath)
                    self.logger.info(f"Saved processed {data_type} data to {filepath}")
                elif isinstance(df, dict):  # Room data
                    for room_name, room_df in df.items():
                        if not room_df.empty:
                            filepath = Path(
                                f"data/processed/{data_type}_{room_name}_processed.parquet"
                            )
                            room_df.to_parquet(filepath)
                            self.logger.info(
                                f"Saved processed {data_type} {room_name} data to {filepath}"
                            )

            # Save analysis results
            import json

            # Convert numpy types to Python types for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, "item"):  # numpy scalars
                    return obj.item()
                elif hasattr(obj, "tolist"):  # numpy arrays
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            results_json = convert_for_json(results)

            results_file = Path("analysis/results/analysis_results.json")
            with open(results_file, "w") as f:
                json.dump(results_json, f, indent=2, default=str)
            self.logger.info(f"Saved analysis results to {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def _generate_reports(self, results: Dict[str, Any]):
        """Generate summary reports."""
        try:
            # Generate text summary report
            summary_report = self.visualizer.create_summary_report(results)

            report_file = Path("analysis/reports/analysis_summary.txt")
            with open(report_file, "w") as f:
                f.write(summary_report)

            self.logger.info(f"Generated summary report: {report_file}")

            # Print summary to console
            print("\n" + summary_report)

        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            raise


async def main():
    """Main entry point for the analysis pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("analysis/analysis.log")],
    )

    logger = logging.getLogger(__name__)

    try:
        # Load settings
        logger.info("Loading settings...")
        settings = Settings()

        # Define analysis period (default: last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        # Create and run analysis pipeline
        logger.info("Initializing analysis pipeline...")
        pipeline = AnalysisPipeline(settings)

        # Run the complete analysis
        results = await pipeline.run_full_analysis(start_date, end_date)

        logger.info("Analysis pipeline completed successfully!")
        return results

    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the analysis pipeline
    asyncio.run(main())
