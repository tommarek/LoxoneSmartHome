"""
Comprehensive Analysis Pipeline for PEMS v2.

This module provides the ComprehensiveAnalyzer class that orchestrates
all analysis components in a structured, modular way.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from analysis.analyzers.base_load_analysis import BaseLoadAnalyzer
from analysis.analyzers.pattern_analysis import (PVAnalyzer,
                                                 RelayPatternAnalyzer)
from analysis.analyzers.thermal_analysis import ThermalAnalyzer
from analysis.core.data_extraction import \
    DataExtractor  # TODO: Refactor to use UnifiedDataExtractor
from analysis.core.data_preprocessing import DataPreprocessor
from analysis.reports.report_generator import ReportGenerator
from config.settings import PEMSSettings as Settings


class ComprehensiveAnalyzer:
    """
    Main comprehensive analysis coordinator.

    This class orchestrates the complete PEMS v2 analysis pipeline:
    1. Data extraction from multiple sources
    2. Data preprocessing and quality assessment
    3. Multi-modal analysis (PV, thermal, base load, patterns)
    4. Result aggregation and report generation
    5. Output management and persistence
    """

    def __init__(self, settings: Settings):
        """Initialize the comprehensive analyzer."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.ComprehensiveAnalyzer")

        # Initialize core components
        self.extractor = DataExtractor(
            settings
        )  # TODO: Refactor to use UnifiedDataExtractor
        self.preprocessor = DataPreprocessor()
        self.report_generator = ReportGenerator()

        # Initialize analyzers
        self.pv_analyzer = PVAnalyzer()

        # Create a data loader adapter for ThermalAnalyzer
        class DataLoaderAdapter:
            def __init__(self, comprehensive_analyzer):
                self.analyzer = comprehensive_analyzer

            def get_dataset(self, name):
                """Get dataset from processed data."""
                if name == "outdoor_temp":
                    # Try to get outdoor temperature from processed data
                    if "outdoor_temp" in self.analyzer.processed_data:
                        return self.analyzer.processed_data["outdoor_temp"]
                    elif "weather" in self.analyzer.processed_data:
                        weather = self.analyzer.processed_data["weather"]
                        if (
                            isinstance(weather, pd.DataFrame)
                            and "temperature_2m" in weather.columns
                        ):
                            return weather[["temperature_2m"]].rename(
                                columns={"temperature_2m": "value"}
                            )
                elif name == "weather":
                    return self.analyzer.processed_data.get("weather")
                elif name.startswith("room_"):
                    # Get room data from processed rooms
                    rooms = self.analyzer.processed_data.get("rooms", {})
                    room_name = name.replace("room_", "")
                    return rooms.get(room_name)
                return None

        self.data_loader_adapter = DataLoaderAdapter(self)
        # Convert Pydantic settings to dict for ThermalAnalyzer
        settings_dict = (
            settings.model_dump()
            if hasattr(settings, "model_dump")
            else settings.dict()
        )
        self.thermal_analyzer = ThermalAnalyzer(
            self.data_loader_adapter, settings_dict, self.report_generator
        )
        self.base_load_analyzer = BaseLoadAnalyzer()
        self.relay_analyzer = RelayPatternAnalyzer()

        # Analysis state
        self.raw_data: Dict[str, Any] = {}
        self.processed_data: Dict[str, Any] = {}
        self.analysis_results: Dict[str, Any] = {}

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
            "analysis/full_results",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def run_comprehensive_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        analysis_types: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete comprehensive analysis pipeline.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            analysis_types: Optional dict to enable/disable specific analyses
                          e.g., {"pv": True, "thermal": True, "base_load": False}

        Returns:
            Dict containing all analysis results
        """
        self.logger.info("Starting PEMS v2 Comprehensive Analysis Pipeline")
        self.logger.info(
            f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        # Set default analysis types if not provided
        if analysis_types is None:
            analysis_types = {
                "pv": True,
                "thermal": True,
                "base_load": True,
                "relay_patterns": True,
                "weather_correlation": True,
            }

        try:
            # Step 1: Data Extraction
            await self._extract_all_data(start_date, end_date)

            # Step 2: Data Preprocessing
            self._preprocess_all_data()

            # Step 3: Run Analyses
            await self._run_comprehensive_analyses(analysis_types)

            # Step 4: Generate Reports
            self._generate_comprehensive_reports()

            # Step 5: Save Results
            self._save_comprehensive_results()

            self.logger.info("Comprehensive analysis pipeline completed successfully!")
            return self.analysis_results

        except Exception as e:
            self.logger.error(
                f"Comprehensive analysis pipeline failed: {e}", exc_info=True
            )
            raise

    async def _extract_all_data(self, start_date: datetime, end_date: datetime):
        """Extract all required data from InfluxDB."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 1: EXTRACTING DATA FROM INFLUXDB")
        self.logger.info("=" * 50)

        self.raw_data = {}

        try:
            # Extract PV data
            self.logger.info("Extracting PV/solar data...")
            self.raw_data["pv"] = await self.extractor.extract_pv_data(
                start_date, end_date
            )
            if not self.raw_data["pv"].empty:
                self.extractor.save_to_parquet(self.raw_data["pv"], "pv_data")

            # Extract room temperature data
            self.logger.info("Extracting room temperature data...")
            self.raw_data["rooms"] = await self.extractor.extract_room_temperatures(
                start_date, end_date
            )
            for room_name, room_df in self.raw_data["rooms"].items():
                if not room_df.empty:
                    self.extractor.save_to_parquet(room_df, f"room_{room_name}")

            # Extract weather data
            self.logger.info("Extracting weather data...")
            self.raw_data["weather"] = await self.extractor.extract_weather_data(
                start_date, end_date
            )
            if not self.raw_data["weather"].empty:
                self.extractor.save_to_parquet(self.raw_data["weather"], "weather_data")

            # Extract outdoor temperature data from teplomer sensor
            self.logger.info("Extracting outdoor temperature data...")
            try:
                self.raw_data[
                    "outdoor_temp"
                ] = await self.extractor.extract_outdoor_temperature_data(
                    start_date, end_date
                )
                if not self.raw_data["outdoor_temp"].empty:
                    self.extractor.save_to_parquet(
                        self.raw_data["outdoor_temp"], "outdoor_temperature"
                    )
            except Exception as e:
                self.logger.warning(f"Could not extract outdoor temperature data: {e}")
                self.raw_data["outdoor_temp"] = pd.DataFrame()

            # Extract energy consumption data
            self.logger.info("Extracting energy consumption data...")
            self.raw_data[
                "consumption"
            ] = await self.extractor.extract_energy_consumption(start_date, end_date)
            if not self.raw_data["consumption"].empty:
                self.extractor.save_to_parquet(
                    self.raw_data["consumption"], "consumption_data"
                )

            # Extract energy prices (optional)
            self.logger.info("Extracting energy price data...")
            try:
                self.raw_data["prices"] = await self.extractor.extract_energy_prices(
                    start_date, end_date
                )
                if (
                    self.raw_data["prices"] is not None
                    and not self.raw_data["prices"].empty
                ):
                    self.extractor.save_to_parquet(
                        self.raw_data["prices"], "energy_prices"
                    )
            except Exception as e:
                self.logger.warning(f"Could not extract energy prices: {e}")
                self.raw_data["prices"] = None

            # Extract relay states (for PEMS analysis)
            self.logger.info("Extracting relay state data...")
            try:
                self.raw_data[
                    "relay_states"
                ] = await self.extractor.extract_relay_states(start_date, end_date)
                for room_name, room_df in self.raw_data["relay_states"].items():
                    if not room_df.empty:
                        self.extractor.save_to_parquet(room_df, f"relay_{room_name}")
            except Exception as e:
                self.logger.warning(f"Could not extract relay states: {e}")
                self.raw_data["relay_states"] = {}

            # Extract battery data
            self.logger.info("Extracting battery data...")
            try:
                self.raw_data["battery"] = await self.extractor.extract_battery_data(
                    start_date, end_date
                )
                if (
                    self.raw_data["battery"] is not None
                    and not self.raw_data["battery"].empty
                ):
                    self.extractor.save_to_parquet(
                        self.raw_data["battery"], "battery_data"
                    )
            except Exception as e:
                self.logger.warning(f"Could not extract battery data: {e}")
                self.raw_data["battery"] = None

            # Extract EV charging data
            self.logger.info("Extracting EV charging data...")
            try:
                self.raw_data["ev"] = await self.extractor.extract_ev_data(
                    start_date, end_date
                )
                if self.raw_data["ev"] is not None and not self.raw_data["ev"].empty:
                    self.extractor.save_to_parquet(self.raw_data["ev"], "ev_data")
            except Exception as e:
                self.logger.warning(f"Could not extract EV data: {e}")
                self.raw_data["ev"] = None

            self.logger.info(
                f"Data extraction completed. Datasets: {list(self.raw_data.keys())}"
            )

        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}", exc_info=True)
            raise

    def _preprocess_all_data(self):
        """Preprocess all extracted data."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 2: PREPROCESSING DATA")
        self.logger.info("=" * 50)

        self.processed_data = {}

        # Process each dataset
        for data_type, data in self.raw_data.items():
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.info(f"Skipping empty {data_type} dataset")
                continue

            if data_type == "rooms":
                # Special handling for room data (dict of DataFrames)
                self.processed_data[data_type] = {}
                for room_name, room_df in data.items():
                    if not room_df.empty:
                        self.processed_data[data_type][
                            room_name
                        ] = self.preprocessor.process_dataset(
                            room_df, f"room_{room_name}"
                        )
            elif data_type == "relay_states":
                # Special handling for relay states (dict of DataFrames)
                self.processed_data[data_type] = {}
                for room_name, relay_df in data.items():
                    if not relay_df.empty:
                        self.processed_data[data_type][
                            room_name
                        ] = self.preprocessor.process_dataset(
                            relay_df, f"relay_{room_name}"
                        )
            else:
                # Regular DataFrame processing
                if isinstance(data, pd.DataFrame):
                    self.processed_data[data_type] = self.preprocessor.process_dataset(
                        data, data_type
                    )
                else:
                    self.processed_data[data_type] = data

        # Log preprocessing summary
        quality_report = getattr(self.preprocessor, "quality_report", {})
        self.logger.info(
            f"Data preprocessing completed. Quality report available for: {list(quality_report.keys())}"
        )

    async def _run_comprehensive_analyses(self, analysis_types: Dict[str, bool]):
        """Run all enabled analyses."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 3: RUNNING COMPREHENSIVE ANALYSIS")
        self.logger.info("=" * 50)

        self.analysis_results = {}

        # PV Analysis
        if analysis_types.get("pv", True) and "pv" in self.processed_data:
            self.logger.info("Running PV production analysis...")
            try:
                weather_data = self.processed_data.get("weather")
                self.analysis_results[
                    "pv_analysis"
                ] = self.pv_analyzer.analyze_pv_production(
                    self.processed_data["pv"], weather_data
                )
                self.logger.info("PV analysis completed successfully")
            except Exception as e:
                self.logger.error(f"PV analysis failed: {e}", exc_info=True)
                self.analysis_results["pv_analysis"] = {"error": str(e)}

        # Thermal Analysis
        if analysis_types.get("thermal", True) and "rooms" in self.processed_data:
            self.logger.info("Running thermal analysis...")
            try:
                # Use outdoor temperature data from teplomer sensor if available, otherwise fallback to weather data
                weather_data = self.processed_data.get("outdoor_temp")
                if weather_data is None or weather_data.empty:
                    self.logger.info("Using weather forecast data for thermal analysis")
                    weather_data = self.processed_data.get("weather")
                else:
                    self.logger.info(
                        "Using outdoor temperature data from teplomer sensor for thermal analysis"
                    )
                    self.logger.info(
                        f"Outdoor temperature data shape: {weather_data.shape}"
                    )
                    self.logger.info(
                        f"Outdoor temperature columns: {list(weather_data.columns)}"
                    )
                    self.logger.info(
                        f"Outdoor temperature data preview:\n{weather_data.head()}"
                    )

                relay_data = self.processed_data.get("relay_states", {})

                self.analysis_results[
                    "thermal_analysis"
                ] = self.thermal_analyzer.analyze_room_dynamics(
                    self.processed_data["rooms"], weather_data, relay_data
                )

                # Analyze unknown events and generate recommendations
                unknown_event_recommendations = self._analyze_unknown_events(
                    self.analysis_results["thermal_analysis"]
                )
                if unknown_event_recommendations:
                    self.analysis_results["thermal_analysis"][
                        "unknown_event_recommendations"
                    ] = unknown_event_recommendations
                    self.logger.info(
                        f"Generated {len(unknown_event_recommendations)} recommendations for unknown thermal events"
                    )

                self.logger.info(
                    f"Thermal analysis completed for {len(self.processed_data['rooms'])} rooms"
                )
            except Exception as e:
                self.logger.error(f"Thermal analysis failed: {e}", exc_info=True)
                self.analysis_results["thermal_analysis"] = {"error": str(e)}

        # Base Load Analysis
        if (
            analysis_types.get("base_load", True)
            and "consumption" in self.processed_data
        ):
            self.logger.info("Running base load analysis...")
            try:
                # Get required data for base load analysis
                grid_data = self.processed_data["consumption"]
                pv_data = self.processed_data.get("pv", pd.DataFrame())
                room_data = self.processed_data.get("rooms", {})
                relay_data = self.processed_data.get("relay_states", {})
                ev_data = self.processed_data.get("ev", None)
                battery_data = self.processed_data.get("battery", None)

                self.analysis_results[
                    "base_load_analysis"
                ] = self.base_load_analyzer.analyze_base_load(
                    grid_data, pv_data, room_data, relay_data, ev_data, battery_data
                )
                self.logger.info("Base load analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Base load analysis failed: {e}", exc_info=True)
                self.analysis_results["base_load_analysis"] = {"error": str(e)}

        # Relay Pattern Analysis
        if (
            analysis_types.get("relay_patterns", True)
            and "relay_states" in self.processed_data
        ):
            self.logger.info("Running relay pattern analysis...")
            try:
                # Pass all relay data at once instead of per-room
                relay_data = self.processed_data["relay_states"]
                weather_data = self.processed_data.get("weather")
                price_data = self.processed_data.get("prices")

                self.analysis_results[
                    "relay_analysis"
                ] = self.relay_analyzer.analyze_relay_patterns(
                    relay_data, weather_data, price_data
                )
                self.logger.info("Relay pattern analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Relay pattern analysis failed: {e}", exc_info=True)
                self.analysis_results["relay_analysis"] = {"error": str(e)}

        # Weather Correlation Analysis
        if (
            analysis_types.get("weather_correlation", True)
            and "weather" in self.processed_data
        ):
            self.logger.info("Running weather correlation analysis...")
            try:
                # Analyze correlations between weather and various systems
                weather_correlations = {}

                if "pv" in self.processed_data:
                    weather_correlations[
                        "pv"
                    ] = await self._analyze_weather_pv_correlation()

                if "consumption" in self.processed_data:
                    weather_correlations[
                        "consumption"
                    ] = await self._analyze_weather_consumption_correlation()

                self.analysis_results["weather_correlations"] = weather_correlations
                self.logger.info("Weather correlation analysis completed")
            except Exception as e:
                self.logger.error(
                    f"Weather correlation analysis failed: {e}", exc_info=True
                )
                self.analysis_results["weather_correlations"] = {"error": str(e)}

        self.logger.info(
            f"Analysis completed. Results available for: {list(self.analysis_results.keys())}"
        )

    async def _analyze_weather_pv_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between weather and PV production."""
        # This can be expanded with more sophisticated analysis
        return {"status": "placeholder - to be implemented"}

    async def _analyze_weather_consumption_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between weather and energy consumption."""
        # This can be expanded with more sophisticated analysis
        return {"status": "placeholder - to be implemented"}

    def _analyze_unknown_events(self, thermal_results: Dict[str, Any]) -> List[str]:
        """Analyze the frequency of unknown events to generate actionable recommendations."""
        recommendations = []

        for room_name, room_data in thermal_results.items():
            if (
                not isinstance(room_data, dict)
                or "adaptive_thermal_analysis" not in room_data
            ):
                continue

            adaptive_data = room_data["adaptive_thermal_analysis"]
            if "thermal_events" in adaptive_data:
                events = adaptive_data["thermal_events"]
                unknown_cooling = len(events.get("unknown_cooling_events", []))
                unknown_heating = len(events.get("unknown_heating_events", []))
                total_unknown = unknown_cooling + unknown_heating

                # Define a threshold for what constitutes a high number of unknown events
                # e.g., more than 10% of all detected events are unknown
                total_events = events.get("total_events", 0)
                if total_events > 0 and (total_unknown / total_events) > 0.1:
                    # Analyze patterns in unknown events for specific recommendations
                    cooling_events = events.get("unknown_cooling_events", [])
                    heating_events = events.get("unknown_heating_events", [])

                    # Check for patterns in timing
                    night_events = sum(
                        1
                        for event in cooling_events + heating_events
                        if 22 <= event.get("hour", 12) or event.get("hour", 12) <= 6
                    )
                    day_events = total_unknown - night_events

                    recommendation = f"Room '{room_name}' has {total_unknown} unexplained temperature changes ({(total_unknown / total_events * 100):.1f}% of events)."

                    # Add specific recommendations based on patterns
                    if night_events > day_events:
                        recommendation += " Most occur at night - check for thermal bridging, drafts, or HVAC cycling."
                    elif unknown_cooling > unknown_heating * 2:
                        recommendation += " Mostly cooling events - check for air leaks or uncontrolled ventilation."
                    elif unknown_heating > unknown_cooling * 2:
                        recommendation += " Mostly heating events - check for heat sources like appliances or direct sunlight."
                    else:
                        recommendation += " Consider checking for drafts, appliance heat sources, or thermostat placement."

                    recommendations.append(recommendation)

        return recommendations

    def _generate_comprehensive_reports(self):
        """Generate comprehensive reports from all analysis results."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 4: GENERATING COMPREHENSIVE REPORTS")
        self.logger.info("=" * 50)

        try:
            # Generate summary report
            summary_report = self.report_generator.create_comprehensive_summary(
                self.analysis_results
            )

            # Save summary report
            report_path = Path("analysis/reports/comprehensive_analysis_summary.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(summary_report)

            self.logger.info(f"Summary report saved to: {report_path}")

            # Generate detailed HTML report if possible
            try:
                html_report = self.report_generator.create_html_report(
                    self.analysis_results, self.processed_data
                )
                html_path = Path("analysis/reports/comprehensive_analysis_report.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_report)

                self.logger.info(f"HTML report saved to: {html_path}")
            except Exception as e:
                self.logger.warning(f"Could not generate HTML report: {e}")

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}", exc_info=True)

    def _save_comprehensive_results(self):
        """Save all analysis results to files."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("STEP 5: SAVING COMPREHENSIVE RESULTS")
        self.logger.info("=" * 50)

        try:
            # Save processed data
            for data_type, data in self.processed_data.items():
                if data_type in ["rooms", "relay_states"]:
                    # Handle dict of DataFrames
                    for sub_name, sub_data in data.items():
                        if isinstance(sub_data, pd.DataFrame) and not sub_data.empty:
                            output_path = f"data/processed/{data_type}_{sub_name}_processed.parquet"
                            sub_data.to_parquet(output_path)
                            self.logger.info(
                                f"Saved {data_type}_{sub_name} to {output_path}"
                            )
                elif isinstance(data, pd.DataFrame) and not data.empty:
                    output_path = f"data/processed/{data_type}_processed.parquet"
                    data.to_parquet(output_path)
                    self.logger.info(f"Saved {data_type} to {output_path}")

            # Save analysis results as JSON
            import json

            results_path = Path("analysis/results/comprehensive_analysis_results.json")

            # Convert results to JSON-serializable format
            json_results = self._prepare_results_for_json(self.analysis_results)

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, default=str)

            self.logger.info(f"Analysis results saved to: {results_path}")

            # Save quality report if available
            if hasattr(self.preprocessor, "quality_report"):
                quality_path = Path("analysis/reports/data_quality_report.json")
                with open(quality_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self.preprocessor.quality_report, f, indent=2, default=str
                    )

                self.logger.info(f"Data quality report saved to: {quality_path}")

        except Exception as e:
            self.logger.error(f"Result saving failed: {e}", exc_info=True)

    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare analysis results for JSON serialization."""
        json_results = {}

        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = self._prepare_results_for_json(value)
            elif isinstance(value, pd.DataFrame):
                # Convert DataFrame to summary info
                json_results[key] = {
                    "type": "DataFrame",
                    "shape": value.shape,
                    "columns": list(value.columns),
                    "summary": "DataFrame data not serialized to JSON",
                }
            elif isinstance(value, (pd.Series, pd.Index)):
                json_results[key] = {
                    "type": type(value).__name__,
                    "length": len(value),
                    "summary": "Series/Index data not serialized to JSON",
                }
            else:
                json_results[key] = value

        return json_results
