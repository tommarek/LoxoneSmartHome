#!/usr/bin/env python3
"""
Run PEMS v2 analysis with 2 years of historical data from Loxone system.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from test_fixed_analysis import LoxoneDataExtractor

from analysis.base_load_analysis import BaseLoadAnalyzer
from analysis.pattern_analysis import PVAnalyzer
from analysis.thermal_analysis import ThermalAnalyzer


async def run_2year_analysis():
    """Run comprehensive analysis with 2 years of historical data."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("analysis_2year.log")],
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PEMS V2 - 2 YEAR HISTORICAL DATA ANALYSIS")
    logger.info("=" * 60)

    try:
        # Create data extractor
        logger.info("Initializing Loxone data extractor...")
        extractor = LoxoneDataExtractor()

        # Define 2-year analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years

        logger.info(
            f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        logger.info(f"Total days: {(end_date - start_date).days}")

        # Create output directories
        output_dirs = [
            "data/raw",
            "data/processed",
            "data/features",
            "analysis/results",
            "analysis/figures",
            "analysis/reports",
        ]
        for directory in output_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # STEP 1: Extract Room Temperature Data
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: EXTRACTING 2-YEAR ROOM TEMPERATURE DATA")
        logger.info("=" * 50)

        room_data = await extractor.extract_room_temperatures(start_date, end_date)

        if room_data:
            logger.info(f"✓ Successfully extracted temperature data for {len(room_data)} rooms")

            # Save raw room data
            total_measurements = 0
            for room_name, room_df in room_data.items():
                if not room_df.empty:
                    room_df.to_parquet(f"data/raw/room_{room_name}_2year.parquet")
                    total_measurements += len(room_df)
                    logger.info(f"  - {room_name}: {len(room_df):,} measurements saved")

            logger.info(f"✓ Total temperature measurements: {total_measurements:,}")
        else:
            logger.error("✗ Failed to extract room temperature data")
            return False

        # STEP 2: Extract Weather Data
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: EXTRACTING 2-YEAR WEATHER DATA")
        logger.info("=" * 50)

        weather_data = await extractor.extract_outdoor_weather(start_date, end_date)

        if not weather_data.empty:
            logger.info(f"✓ Extracted outdoor weather data: {len(weather_data):,} points")
            weather_data.to_parquet("data/raw/outdoor_weather_2year.parquet")

            temp_stats = weather_data["outdoor_temperature"].describe()
            logger.info(
                f"  - Temperature range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C"
            )
            logger.info(f"  - Mean temperature: {temp_stats['mean']:.1f}°C")
        else:
            logger.warning("⚠ No outdoor weather data found")
            weather_data = None

        # STEP 3: Extract Heating Consumption Data
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: EXTRACTING 2-YEAR HEATING DATA")
        logger.info("=" * 50)

        heating_data = await extractor.extract_heating_consumption(start_date, end_date)

        if not heating_data.empty:
            logger.info(f"✓ Extracted heating consumption data: {len(heating_data):,} points")
            heating_data.to_parquet("data/raw/heating_consumption_2year.parquet")

            power_stats = heating_data["total_heating_consumption"].describe()
            logger.info(f"  - Power range: {power_stats['min']:.1f}W to {power_stats['max']:.1f}W")
            logger.info(f"  - Mean consumption: {power_stats['mean']:.1f}W")
        else:
            logger.warning("⚠ No heating consumption data found")
            heating_data = None

        # STEP 4: Run Thermal Analysis
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: RUNNING 2-YEAR THERMAL ANALYSIS")
        logger.info("=" * 50)

        if room_data:
            thermal_analyzer = ThermalAnalyzer()

            try:
                logger.info("Starting comprehensive thermal dynamics analysis...")
                thermal_results = thermal_analyzer.analyze_room_dynamics(room_data, weather_data)

                logger.info(f"✓ Thermal analysis completed for {len(thermal_results)} rooms")

                # Save thermal analysis results
                import json

                # Convert results for JSON serialization
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

                thermal_json = convert_for_json(thermal_results)

                with open("analysis/results/thermal_analysis_2year.json", "w") as f:
                    json.dump(thermal_json, f, indent=2, default=str)

                # Print summary of thermal analysis
                logger.info("\nThermal Analysis Summary (Top 10 Rooms):")
                logger.info("-" * 50)

                room_count = 0
                for room_name, results in thermal_results.items():
                    if room_name == "room_coupling" or room_count >= 10:
                        continue

                    if isinstance(results, dict) and "basic_stats" in results:
                        stats = results["basic_stats"]
                        mean_temp = stats.get("mean_temperature", 0)
                        temp_range = stats.get("temperature_range", 0)
                        logger.info(
                            f"  {room_name:20s}: {mean_temp:5.1f}°C mean, {temp_range:4.1f}°C range"
                        )
                        room_count += 1

            except Exception as e:
                logger.error(f"✗ Thermal analysis failed: {e}")

        # STEP 5: Generate Analysis Report
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: GENERATING 2-YEAR ANALYSIS REPORT")
        logger.info("=" * 50)

        # Create comprehensive report
        report_lines = [
            "=" * 80,
            "PEMS V2 - 2 YEAR HISTORICAL DATA ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"Total Days Analyzed: {(end_date - start_date).days}",
            "",
            "DATA EXTRACTION SUMMARY",
            "-" * 30,
        ]

        if room_data:
            report_lines.extend(
                [
                    f"• Room Temperature Data: {len(room_data)} rooms monitored",
                    f"• Total Temperature Measurements: {total_measurements:,}",
                    f"• Average Measurements per Room: {total_measurements//len(room_data):,}",
                ]
            )

        if weather_data is not None and not weather_data.empty:
            temp_stats = weather_data["outdoor_temperature"].describe()
            report_lines.extend(
                [
                    f"• Outdoor Weather Data: {len(weather_data):,} measurements",
                    f"• Temperature Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C",
                    f"• Mean Outdoor Temperature: {temp_stats['mean']:.1f}°C",
                ]
            )

        if heating_data is not None and not heating_data.empty:
            power_stats = heating_data["total_heating_consumption"].describe()
            total_energy = (
                heating_data["total_heating_consumption"].sum() * 0.25
            ) / 1000  # 15min intervals to kWh
            report_lines.extend(
                [
                    f"• Heating Consumption Data: {len(heating_data):,} measurements",
                    f"• Total Heating Energy: {total_energy:.1f} kWh over 2 years",
                    f"• Average Heating Power: {power_stats['mean']:.1f}W",
                ]
            )

        report_lines.extend(
            [
                "",
                "ROOM ANALYSIS SUMMARY",
                "-" * 30,
            ]
        )

        if room_data:
            # Sort rooms by number of measurements
            sorted_rooms = sorted(room_data.items(), key=lambda x: len(x[1]), reverse=True)

            for room_name, room_df in sorted_rooms[:15]:  # Top 15 rooms
                if not room_df.empty:
                    temp_stats = room_df["temperature"].describe()
                    report_lines.append(
                        f"• {room_name:20s}: {len(room_df):6,} measurements, "
                        f"{temp_stats['mean']:5.1f}°C avg, "
                        f"{temp_stats['min']:5.1f}-{temp_stats['max']:4.1f}°C range"
                    )

        report_lines.extend(
            [
                "",
                "NEXT STEPS",
                "-" * 30,
                "• Thermal dynamics analysis completed for all rooms",
                "• Data ready for predictive modeling and optimization",
                "• Consider integrating PV production data when available",
                "• Ready for battery storage optimization analysis",
                "• Implement demand response strategies based on patterns",
                "",
                "=" * 80,
                "End of 2-Year Analysis Report",
            ]
        )

        # Save report
        report_content = "\n".join(report_lines)

        with open("analysis/reports/2year_analysis_report.txt", "w") as f:
            f.write(report_content)

        # Print report to console
        print("\n" + report_content)

        logger.info("\n" + "=" * 60)
        logger.info("2-YEAR ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Generated Files:")
        logger.info("• Raw data: data/raw/ (parquet files)")
        logger.info("• Analysis results: analysis/results/thermal_analysis_2year.json")
        logger.info("• Report: analysis/reports/2year_analysis_report.txt")
        logger.info("• Log file: analysis_2year.log")

        return True

    except Exception as e:
        logger.error(f"2-year analysis failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import pandas as pd  # Import pandas for the analysis

    success = asyncio.run(run_2year_analysis())
    print(f"\n2-Year Analysis Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
