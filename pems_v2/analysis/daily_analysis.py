"""
Daily Analysis Workflow for PEMS v2.

This module provides automated daily analysis capabilities,
including incremental data processing, trend monitoring,
and alert generation for the PEMS system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.pipelines.comprehensive_analysis import ComprehensiveAnalyzer
from config.settings import PEMSSettings as Settings


class DailyAnalysisWorkflow:
    """
    Daily analysis workflow for PEMS v2.

    This class provides automated daily analysis capabilities:
    - Incremental data processing for recent data
    - Daily performance monitoring
    - Trend detection and alerting
    - Automated report generation
    - Data quality monitoring
    """

    def __init__(self, settings: Settings):
        """Initialize the daily analysis workflow."""
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.DailyAnalysisWorkflow")
        self.analyzer = ComprehensiveAnalyzer(settings)

        # Analysis configuration
        self.analysis_window_days = 7  # Days to analyze for daily workflow
        self.trend_detection_days = 30  # Days to look back for trend detection

    async def run_daily_analysis(
        self, target_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run daily analysis for a specific date.

        Args:
            target_date: Date to analyze (defaults to yesterday)

        Returns:
            Dictionary containing daily analysis results
        """
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)  # Yesterday

        self.logger.info(
            f"Starting daily analysis for {target_date.strftime('%Y-%m-%d')}"
        )

        try:
            # Run incremental analysis for the past week
            end_date = target_date + timedelta(days=1)  # Include full day
            start_date = target_date - timedelta(days=self.analysis_window_days - 1)

            # Configure analysis for daily workflow (faster, focused analysis)
            analysis_types = {
                "pv": True,
                "thermal": True,
                "base_load": True,
                "relay_patterns": True,
                "weather_correlation": False,  # Skip for daily analysis
            }

            # Run the analysis
            results = await self.analyzer.run_comprehensive_analysis(
                start_date, end_date, analysis_types
            )

            # Perform daily-specific analysis
            daily_insights = await self._analyze_daily_performance(results, target_date)

            # Detect trends and anomalies
            trend_analysis = await self._detect_trends(target_date)

            # Generate daily summary
            daily_summary = self._generate_daily_summary(
                daily_insights, trend_analysis, target_date
            )

            # Save daily results
            self._save_daily_results(daily_summary, target_date)

            self.logger.info(
                f"Daily analysis completed for {target_date.strftime('%Y-%m-%d')}"
            )

            return {
                "target_date": target_date,
                "daily_insights": daily_insights,
                "trend_analysis": trend_analysis,
                "daily_summary": daily_summary,
                "full_analysis_results": results,
            }

        except Exception as e:
            self.logger.error(
                f"Daily analysis failed for {target_date}: {e}", exc_info=True
            )
            raise

    async def _analyze_daily_performance(
        self, results: Dict[str, Any], target_date: datetime
    ) -> Dict[str, Any]:
        """Analyze daily performance metrics."""
        daily_insights = {}

        # PV daily performance
        if "pv_analysis" in results:
            daily_insights["pv"] = self._extract_daily_pv_metrics(
                results["pv_analysis"], target_date
            )

        # Thermal daily performance
        if "thermal_analysis" in results:
            daily_insights["thermal"] = self._extract_daily_thermal_metrics(
                results["thermal_analysis"], target_date
            )

        # Base load daily performance
        if "base_load_analysis" in results:
            daily_insights["base_load"] = self._extract_daily_base_load_metrics(
                results["base_load_analysis"], target_date
            )

        # Relay daily performance
        if "relay_analysis" in results:
            daily_insights["relay"] = self._extract_daily_relay_metrics(
                results["relay_analysis"], target_date
            )

        return daily_insights

    def _extract_daily_pv_metrics(
        self, pv_results: Dict[str, Any], target_date: datetime
    ) -> Dict[str, Any]:
        """Extract daily PV performance metrics."""
        # target_date reserved for future date-specific analysis
        basic_stats = pv_results.get("basic_stats", {})

        return {
            "daily_energy_kwh": basic_stats.get("total_energy_kwh", 0.0),
            "peak_power_w": basic_stats.get("max_power", 0.0),
            "capacity_factor": basic_stats.get("capacity_factor", 0.0),
            "weather_efficiency": self._calculate_weather_efficiency(pv_results),
            "daylight_capacity_factor": basic_stats.get(
                "daylight_capacity_factor", 0.0
            ),
            "total_records": basic_stats.get("total_records", 0),
        }

    def _extract_daily_thermal_metrics(
        self, thermal_results: Dict[str, Any], target_date: datetime
    ) -> Dict[str, Any]:
        """Extract daily thermal performance metrics."""
        # target_date reserved for future date-specific analysis
        room_metrics = {}
        total_heating_percentage = 0
        room_count = 0

        for room_name, room_data in thermal_results.items():
            if isinstance(room_data, dict) and "basic_stats" in room_data:
                basic_stats = room_data["basic_stats"]
                comfort_data = room_data.get("comfort_analysis", {})

                heating_percentage = basic_stats.get("heating_percentage", 0)
                heating_hours = (
                    heating_percentage * 24 / 100
                )  # Convert percentage to hours

                room_metrics[room_name] = {
                    "avg_temperature": basic_stats.get("mean_temperature", 0),
                    "temperature_range": basic_stats.get("temperature_range", 0),
                    "heating_time_hours": heating_hours,
                    "heating_percentage": heating_percentage,
                    "temperature_stability": 1
                    / (1 + basic_stats.get("temperature_std", 1)),  # Stability score
                    "comfort_score": comfort_data.get("overall_comfort_score", 0),
                    "outdoor_correlation": basic_stats.get("outdoor_correlation", 0),
                }

                total_heating_percentage += heating_percentage
                room_count += 1

        avg_efficiency = total_heating_percentage / room_count if room_count > 0 else 0

        return {
            "rooms": room_metrics,
            "rooms_analyzed": room_count,
            "total_heating_hours": sum(
                r.get("heating_time_hours", 0) for r in room_metrics.values()
            ),
            "avg_heating_percentage": total_heating_percentage / room_count
            if room_count > 0
            else 0,
            "avg_efficiency": (100 - avg_efficiency)
            / 100,  # Convert to efficiency score
        }

    def _extract_daily_base_load_metrics(
        self, base_load_results: Dict[str, Any], target_date: datetime
    ) -> Dict[str, Any]:
        """Extract daily base load metrics."""
        # target_date reserved for future date-specific analysis
        basic_stats = base_load_results.get("basic_stats", {})

        daily_energy = basic_stats.get("total_energy_kwh", 0)
        peak_demand = basic_stats.get("mean_base_load", 0)
        load_factor = basic_stats.get("base_load_percentage", 0) / 100

        # Calculate efficiency rating
        if load_factor > 0.7:
            efficiency_rating = "excellent"
        elif load_factor > 0.5:
            efficiency_rating = "good"
        elif load_factor > 0.3:
            efficiency_rating = "fair"
        else:
            efficiency_rating = "poor"

        return {
            "daily_base_load_kwh": daily_energy,
            "peak_demand_w": peak_demand,
            "load_factor": load_factor,
            "efficiency_rating": efficiency_rating,
            "base_load_percentage": basic_stats.get("base_load_percentage", 0),
        }

    def _extract_daily_relay_metrics(
        self, relay_results: Dict[str, Any], target_date: datetime
    ) -> Dict[str, Any]:
        """Extract daily relay performance metrics."""
        # target_date reserved for future date-specific analysis
        room_metrics = {}
        total_cycles = 0
        total_runtime = 0

        for room_name, room_data in relay_results.items():
            if isinstance(room_data, dict):
                # Extract switching patterns
                switching_data = room_data.get("switching_patterns", {})
                cycles = switching_data.get("switches_per_day", 0)

                # Calculate runtime from heating percentage (if available)
                heating_percentage = room_data.get("heating_percentage", 0)
                runtime_hours = heating_percentage * 24 / 100

                room_metrics[room_name] = {
                    "daily_cycles": cycles,
                    "total_runtime_hours": runtime_hours,
                    "efficiency_score": self._calculate_relay_efficiency(room_data),
                    "switches_per_day": cycles,
                }

                total_cycles += cycles
                total_runtime += runtime_hours

        return {
            "rooms": room_metrics,
            "rooms_analyzed": len(room_metrics),
            "total_cycles": total_cycles,
            "total_runtime_hours": total_runtime,
            "avg_cycles_per_room": total_cycles / len(room_metrics)
            if room_metrics
            else 0,
        }

    async def _detect_trends(self, target_date: datetime) -> Dict[str, Any]:
        """Detect trends and anomalies over the past period."""
        # Run trend analysis over the past 30 days
        end_date = target_date
        start_date = target_date - timedelta(days=self.trend_detection_days)

        try:
            # Get historical data for trend analysis
            trend_results = await self.analyzer.run_comprehensive_analysis(
                start_date,
                end_date,
                {
                    "pv": True,
                    "thermal": True,
                    "base_load": True,
                    "relay_patterns": True,
                    "weather_correlation": False,
                },
            )

            # Analyze trends in the results
            pv_trends = self._analyze_pv_trends(trend_results.get("pv_analysis", {}))
            thermal_trends = self._analyze_thermal_trends(
                trend_results.get("thermal_analysis", {})
            )
            base_load_trends = self._analyze_base_load_trends(
                trend_results.get("base_load_analysis", {})
            )

            # Detect anomalies
            anomalies = self._detect_anomalies(trend_results)
            alerts = self._generate_alerts(anomalies)

            return {
                "pv_trends": pv_trends,
                "thermal_trends": thermal_trends,
                "base_load_trends": base_load_trends,
                "anomalies_detected": anomalies,
                "alerts": alerts,
                "analysis_period_days": self.trend_detection_days,
            }

        except Exception as e:
            self.logger.warning(f"Trend detection failed: {e}")
            return {
                "pv_trends": {"energy_trend": "unknown", "efficiency_trend": "unknown"},
                "thermal_trends": {
                    "heating_usage_trend": "unknown",
                    "efficiency_trend": "unknown",
                },
                "base_load_trends": {
                    "consumption_trend": "unknown",
                    "efficiency_trend": "unknown",
                },
                "anomalies_detected": [],
                "alerts": ["Trend analysis failed - check data availability"],
                "status": "error",
            }

    def _generate_daily_summary(
        self,
        daily_insights: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        target_date: datetime,
    ) -> str:
        """Generate a daily summary report."""
        summary_lines = [
            f"PEMS v2 Daily Analysis Summary - {target_date.strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
        ]

        # PV Summary
        if "pv" in daily_insights:
            pv_data = daily_insights["pv"]
            summary_lines.extend(
                [
                    "PV Production:",
                    f"  • Daily Energy: {pv_data.get('daily_energy_kwh', 0):.1f} kWh",
                    f"  • Peak Power: {pv_data.get('peak_power_w', 0):.0f} W",
                    f"  • Capacity Factor: {pv_data.get('capacity_factor', 0)*100:.1f}%",
                    "",
                ]
            )

        # Thermal Summary
        if "thermal" in daily_insights:
            thermal_data = daily_insights["thermal"]
            summary_lines.extend(
                [
                    "Thermal Performance:",
                    f"  • Total Heating Hours: {thermal_data.get('total_heating_hours', 0):.1f}",
                    f"  • Rooms Analyzed: {len(thermal_data.get('rooms', {}))}",
                    f"  • Average Efficiency: {thermal_data.get('avg_efficiency', 0)*100:.1f}%",
                    "",
                ]
            )

        # Base Load Summary
        if "base_load" in daily_insights:
            base_load_data = daily_insights["base_load"]
            summary_lines.extend(
                [
                    "Base Load:",
                    f"  • Daily Consumption: {base_load_data.get('daily_base_load_kwh', 0):.1f} kWh",
                    f"  • Peak Demand: {base_load_data.get('peak_demand_w', 0):.0f} W",
                    f"  • Load Factor: {base_load_data.get('load_factor', 0):.3f}",
                    "",
                ]
            )

        # Trends Summary
        summary_lines.extend(
            [
                "Trend Analysis:",
                f"  • PV Trend: {trend_analysis.get('pv_trends', {}).get('energy_trend', 'unknown')}",
                f"  • Thermal Trend: {trend_analysis.get('thermal_trends', {}).get('efficiency_trend', 'unknown')}",
                f"  • Base Load Trend: {trend_analysis.get('base_load_trends', {}).get('consumption_trend', 'unknown')}",
                "",
            ]
        )

        # Alerts
        alerts = trend_analysis.get("alerts", [])
        if alerts:
            summary_lines.extend(
                [
                    "Alerts:",
                    *[f"  • {alert}" for alert in alerts],
                    "",
                ]
            )
        else:
            summary_lines.extend(["No alerts detected.", ""])

        summary_lines.extend(
            [
                "=" * 60,
                f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
        )

        return "\n".join(summary_lines)

    def _save_daily_results(self, daily_summary: str, target_date: datetime):
        """Save daily analysis results."""
        try:
            # Create daily reports directory
            daily_dir = Path("analysis/reports/daily")
            daily_dir.mkdir(parents=True, exist_ok=True)

            # Save daily summary
            summary_path = (
                daily_dir / f"daily_summary_{target_date.strftime('%Y%m%d')}.txt"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(daily_summary)

            self.logger.info(f"Daily summary saved to: {summary_path}")

        except Exception as e:
            self.logger.error(f"Failed to save daily results: {e}", exc_info=True)

    async def run_weekly_analysis(
        self, target_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Run weekly analysis workflow."""
        if target_date is None:
            target_date = datetime.now()

        # Run analysis for the past week
        end_date = target_date
        start_date = target_date - timedelta(days=7)

        self.logger.info(
            f"Running weekly analysis for week ending {target_date.strftime('%Y-%m-%d')}"
        )

        results = await self.analyzer.run_comprehensive_analysis(start_date, end_date)

        # Generate weekly summary
        weekly_summary = self._generate_weekly_summary(results, target_date)

        return {
            "target_date": target_date,
            "weekly_summary": weekly_summary,
            "full_results": results,
        }

    def _generate_weekly_summary(
        self, results: Dict[str, Any], target_date: datetime
    ) -> str:
        """Generate weekly summary report."""
        # TODO: Implement full weekly summary with results analysis
        return f"""
        PEMS v2 Weekly Analysis Summary - Week ending {target_date.strftime('%Y-%m-%d')}
        ===============================================================================
        
        This weekly summary is a placeholder and will be implemented with:
        - Weekly energy production and consumption totals
        - Efficiency metrics and trends
        - Performance comparisons vs. previous weeks
        - Optimization recommendations
        
        Data analyzed: {len(results)} analysis categories
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

    def _calculate_weather_efficiency(self, pv_results: Dict[str, Any]) -> float:
        """Calculate weather efficiency for PV system."""
        weather_corr = pv_results.get("weather_correlations", {})
        if weather_corr and "strongest_positive" in weather_corr:
            strongest = weather_corr["strongest_positive"]
            if strongest and len(strongest) >= 2:
                return abs(strongest[1].get("correlation", 0))
        return 0.0

    def _calculate_relay_efficiency(self, room_data: Dict[str, Any]) -> float:
        """Calculate relay efficiency score."""
        # Simple efficiency based on switching frequency and heating percentage
        switching = room_data.get("switching_patterns", {})
        switches_per_day = switching.get("switches_per_day", 0)

        # Lower switching frequency = higher efficiency (less wear)
        if switches_per_day < 10:
            return 0.9
        elif switches_per_day < 20:
            return 0.7
        else:
            return 0.5

    def _analyze_pv_trends(self, pv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PV trends from analysis results."""
        basic_stats = pv_analysis.get("basic_stats", {})

        # Simplified trend analysis - in real implementation would compare historical data
        capacity_factor = basic_stats.get("capacity_factor", 0)

        if capacity_factor > 0.2:
            energy_trend = "good"
        elif capacity_factor > 0.15:
            energy_trend = "stable"
        else:
            energy_trend = "declining"

        return {
            "energy_trend": energy_trend,
            "efficiency_trend": "stable",
            "capacity_factor": capacity_factor,
        }

    def _analyze_thermal_trends(
        self, thermal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze thermal trends from analysis results."""
        # Count successful analyses
        successful_rooms = sum(
            1
            for room_data in thermal_analysis.values()
            if isinstance(room_data, dict) and "basic_stats" in room_data
        )

        total_heating_pct = 0
        for room_data in thermal_analysis.values():
            if isinstance(room_data, dict) and "basic_stats" in room_data:
                total_heating_pct += room_data["basic_stats"].get(
                    "heating_percentage", 0
                )

        avg_heating = (
            total_heating_pct / successful_rooms if successful_rooms > 0 else 0
        )

        return {
            "heating_usage_trend": "moderate" if avg_heating < 50 else "high",
            "efficiency_trend": "stable",
            "rooms_analyzed": successful_rooms,
            "avg_heating_percentage": avg_heating,
        }

    def _analyze_base_load_trends(
        self, base_load_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze base load trends from analysis results."""
        basic_stats = base_load_analysis.get("basic_stats", {})

        base_load_pct = basic_stats.get("base_load_percentage", 0)

        if base_load_pct > 70:
            trend = "high"
        elif base_load_pct > 40:
            trend = "moderate"
        else:
            trend = "low"

        return {
            "consumption_trend": trend,
            "efficiency_trend": "stable",
            "base_load_percentage": base_load_pct,
        }

    def _detect_anomalies(self, trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in trend analysis results."""
        anomalies = []

        # Check PV anomalies
        pv_analysis = trend_results.get("pv_analysis", {})
        if "anomalies" in pv_analysis:
            pv_anomalies = pv_analysis["anomalies"]
            zero_production = pv_anomalies.get("zero_production_events", 0)
            if zero_production > 5:
                anomalies.append(
                    {
                        "type": "pv_zero_production",
                        "severity": "medium",
                        "description": f"{zero_production} periods of zero PV production detected",
                    }
                )

        # Check thermal anomalies
        thermal_analysis = trend_results.get("thermal_analysis", {})
        failed_analyses = sum(
            1
            for room_data in thermal_analysis.values()
            if isinstance(room_data, dict) and "error" in room_data
        )
        if failed_analyses > 0:
            anomalies.append(
                {
                    "type": "thermal_analysis_failures",
                    "severity": "low",
                    "description": f"{failed_analyses} rooms failed thermal analysis",
                }
            )

        return anomalies

    def _generate_alerts(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate alerts from detected anomalies."""
        alerts = []

        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            description = anomaly.get("description", "Unknown anomaly")

            if severity == "high":
                alerts.append(f"🔴 HIGH: {description}")
            elif severity == "medium":
                alerts.append(f"🟡 MEDIUM: {description}")
            else:
                alerts.append(f"🟢 INFO: {description}")

        return alerts


async def run_daily_workflow(target_date: Optional[datetime] = None):
    """Convenience function to run daily analysis workflow."""
    settings = Settings()
    workflow = DailyAnalysisWorkflow(settings)
    return await workflow.run_daily_analysis(target_date)


async def run_weekly_workflow(target_date: Optional[datetime] = None):
    """Convenience function to run weekly analysis workflow."""
    settings = Settings()
    workflow = DailyAnalysisWorkflow(settings)
    return await workflow.run_weekly_analysis(target_date)


if __name__ == "__main__":
    # Run daily analysis for yesterday
    asyncio.run(run_daily_workflow())
