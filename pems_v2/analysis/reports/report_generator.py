"""
Report Generation Module for PEMS v2.

This module provides comprehensive report generation capabilities
for analysis results, including text summaries, HTML reports,
and data quality assessments.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class ReportGenerator:
    """
    Comprehensive report generator for PEMS v2 analysis results.

    This class creates various types of reports from analysis results:
    - Text summary reports
    - HTML detailed reports
    - Data quality reports
    - Comparative analysis reports
    - Daily analysis summaries
    - Performance trend reports
    """

    def __init__(self, output_dir: str = "analysis/reports"):
        """Initialize the report generator."""
        self.logger = logging.getLogger(f"{__name__}.ReportGenerator")
        from pathlib import Path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_comprehensive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Create a comprehensive text summary report.

        Args:
            analysis_results: Dictionary containing all analysis results

        Returns:
            String containing the formatted report
        """
        report_lines = [
            "=" * 80,
            "PEMS v2 COMPREHENSIVE ANALYSIS SUMMARY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # PV Analysis Summary
        if "pv_analysis" in analysis_results:
            report_lines.extend(
                self._generate_pv_summary(analysis_results["pv_analysis"])
            )

        # Thermal Analysis Summary
        if "thermal_analysis" in analysis_results:
            report_lines.extend(
                self._generate_thermal_summary(analysis_results["thermal_analysis"])
            )

        # Base Load Analysis Summary
        if "base_load_analysis" in analysis_results:
            report_lines.extend(
                self._generate_base_load_summary(analysis_results["base_load_analysis"])
            )

        # Relay Analysis Summary
        if "relay_analysis" in analysis_results:
            report_lines.extend(
                self._generate_relay_summary(analysis_results["relay_analysis"])
            )

        # Weather Correlation Summary
        if "weather_correlations" in analysis_results:
            report_lines.extend(
                self._generate_weather_correlation_summary(
                    analysis_results["weather_correlations"]
                )
            )

        # Data Quality Summary
        if "data_quality" in analysis_results:
            report_lines.extend(
                self._generate_data_quality_summary(analysis_results["data_quality"])
            )

        # Recommendations
        report_lines.extend(self._generate_recommendations(analysis_results))

        report_lines.extend(["=" * 80, "End of Report"])

        return "\n".join(report_lines)

    def _generate_pv_summary(self, pv_results: Dict[str, Any]) -> list:
        """Generate PV analysis summary section."""
        lines = [
            "PV PRODUCTION ANALYSIS",
            "-" * 30,
        ]

        if "basic_stats" in pv_results:
            stats = pv_results["basic_stats"]
            lines.extend(
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
                correlation_val = strongest[1]["correlation"]
                lines.append(
                    f"• Strongest Weather Correlation: {strongest[0]} ({correlation_val:.3f})"
                )

        if "seasonal_patterns" in pv_results:
            seasonal = pv_results["seasonal_patterns"]
            best_month = seasonal.get("best_month", "N/A")
            worst_month = seasonal.get("worst_month", "N/A")
            lines.extend(
                [
                    f"• Best Production Month: {best_month}",
                    f"• Worst Production Month: {worst_month}",
                ]
            )

        lines.append("")
        return lines

    def _generate_thermal_summary(self, thermal_results: Dict[str, Any]) -> list:
        """Generate thermal analysis summary section."""
        lines = [
            "THERMAL DYNAMICS ANALYSIS",
            "-" * 30,
        ]

        room_count = 0
        total_heating_usage = 0

        for room_name, room_data in thermal_results.items():
            if room_name == "room_coupling" or "error" in str(room_data):
                continue

            if isinstance(room_data, dict) and "basic_stats" in room_data:
                room_count += 1
                stats = room_data["basic_stats"]
                heating_pct = stats.get("heating_percentage", 0)
                total_heating_usage += heating_pct

                lines.extend(
                    [
                        f"Room: {room_name}",
                        f"  • Mean Temperature: {stats.get('mean_temperature', 0):.1f}°C",
                        f"  • Temperature Range: {stats.get('temperature_range', 0):.1f}°C",
                        f"  • Heating Usage: {heating_pct:.1f}%",
                    ]
                )

                if "time_constant" in room_data and isinstance(
                    room_data["time_constant"], dict
                ):
                    tc = room_data["time_constant"]
                    if "time_constant_hours" in tc:
                        lines.append(
                            f"  • Time Constant: {tc['time_constant_hours']:.1f} hours"
                        )

        if room_count > 0:
            avg_heating = total_heating_usage / room_count
            lines.extend(
                [
                    "",
                    f"Summary: {room_count} rooms analyzed",
                    f"Average heating usage: {avg_heating:.1f}%",
                ]
            )

        lines.append("")
        return lines

    def _generate_base_load_summary(self, base_load_results: Dict[str, Any]) -> list:
        """Generate base load analysis summary section."""
        lines = [
            "BASE LOAD ANALYSIS",
            "-" * 30,
        ]

        if "basic_stats" in base_load_results:
            stats = base_load_results["basic_stats"]
            lines.extend(
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
            weekend_inc = pattern.get("weekend_increase", 0)
            lines.append(f"• Weekend vs Weekday: {weekend_inc:.1f}% higher on weekends")

        lines.append("")
        return lines

    def _generate_relay_summary(self, relay_results: Dict[str, Any]) -> list:
        """Generate relay analysis summary section."""
        lines = [
            "RELAY PATTERN ANALYSIS",
            "-" * 30,
        ]

        total_rooms = 0
        total_cycles = 0

        for room_name, room_data in relay_results.items():
            if "error" in str(room_data):
                continue

            if isinstance(room_data, dict):
                total_rooms += 1
                cycles = room_data.get("daily_cycles", {}).get("mean_cycles_per_day", 0)
                total_cycles += cycles

                lines.extend(
                    [
                        f"Room: {room_name}",
                        f"  • Daily Cycles: {cycles:.1f}",
                    ]
                )

                if "efficiency" in room_data:
                    eff = room_data["efficiency"]
                    lines.append(
                        f"  • Heating Efficiency: {eff.get('heating_efficiency', 0):.1f}%"
                    )

        if total_rooms > 0:
            avg_cycles = total_cycles / total_rooms
            lines.extend(
                [
                    "",
                    f"Summary: {total_rooms} rooms analyzed",
                    f"Average daily cycles: {avg_cycles:.1f}",
                ]
            )

        lines.append("")
        return lines

    def _generate_weather_correlation_summary(
        self, weather_results: Dict[str, Any]
    ) -> list:
        """Generate weather correlation summary section."""
        lines = [
            "WEATHER CORRELATION ANALYSIS",
            "-" * 30,
        ]

        if "pv" in weather_results:
            lines.append("• PV-Weather Correlations: Available")

        if "consumption" in weather_results:
            lines.append("• Consumption-Weather Correlations: Available")

        lines.append("")
        return lines

    def _generate_data_quality_summary(self, quality_results: Dict[str, Any]) -> list:
        """Generate data quality summary section."""
        lines = [
            "DATA QUALITY SUMMARY",
            "-" * 30,
        ]

        for data_type, report in quality_results.items():
            if isinstance(report, dict):
                lines.extend(
                    [
                        f"• {data_type.title()}: {report.get('total_records', 0)} records",
                        f"  Missing data: {report.get('clean_missing_percentage', 0):.1f}%",
                        f"  Time gaps: {len(report.get('time_gaps', []))}",
                    ]
                )

        lines.append("")
        return lines

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis results."""
        lines = [
            "RECOMMENDATIONS",
            "-" * 30,
        ]

        # Base recommendations
        recommendations = [
            "• Monitor PV system performance during identified low-efficiency periods",
            "• Optimize heating schedules based on thermal time constants",
            "• Investigate high base load consumption periods for energy savings",
            "• Consider demand response strategies during peak consumption hours",
        ]

        # Add specific recommendations based on results
        if "pv_analysis" in analysis_results:
            pv_data = analysis_results["pv_analysis"]
            if "basic_stats" in pv_data:
                capacity_factor = pv_data["basic_stats"].get("capacity_factor", 0)
                if capacity_factor < 0.15:  # Low capacity factor
                    recommendations.append(
                        "• PV capacity factor is low - check for shading or maintenance issues"
                    )

        if "thermal_analysis" in analysis_results:
            # Check for rooms with high heating usage
            high_usage_rooms = []
            for room_name, room_data in analysis_results["thermal_analysis"].items():
                if isinstance(room_data, dict) and "basic_stats" in room_data:
                    heating_pct = room_data["basic_stats"].get("heating_percentage", 0)
                    if heating_pct > 50:  # High heating usage
                        high_usage_rooms.append(room_name)

            if high_usage_rooms:
                recommendations.append(
                    f"• High heating usage detected in: {', '.join(high_usage_rooms)} - "
                    "consider insulation improvements"
                )

        lines.extend(recommendations)
        lines.append("")
        return lines

    def create_html_report(
        self,
        analysis_results: Dict[str, Any],
        processed_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create an enhanced HTML report with modern styling and responsive design.

        Args:
            analysis_results: Dictionary containing analysis results
            processed_data: Optional processed data for additional charts

        Returns:
            HTML string containing the report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PEMS v2 Comprehensive Analysis Report</title>
            <style>
                {self._get_enhanced_css_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="report-header">
                    <h1>PEMS v2 Comprehensive Analysis Report</h1>
                    <div class="report-meta">
                        <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                        <span class="version">Framework Version: Phase 1C</span>
                    </div>
                </header>
                
                <nav class="table-of-contents">
                    <h2>Table of Contents</h2>
                    <ul>
                        <li><a href="#executive-summary">Executive Summary</a></li>
                        <li><a href="#pv-analysis">PV Production Analysis</a></li>
                        <li><a href="#thermal-analysis">Thermal Dynamics</a></li>
                        <li><a href="#base-load-analysis">Base Load Analysis</a></li>
                        <li><a href="#relay-analysis">Relay Pattern Analysis</a></li>
                        <li><a href="#data-quality">Data Quality Assessment</a></li>
                        <li><a href="#recommendations">Recommendations</a></li>
                    </ul>
                </nav>
                
                {self._generate_executive_summary_html(analysis_results)}
                {self._generate_detailed_analysis_html(analysis_results)}
                {self._generate_data_quality_html(analysis_results)}
                {self._generate_recommendations_html(analysis_results)}
                
                <footer class="report-footer">
                    <p>Report generated by PEMS v2 Analysis Framework</p>
                    <p>For technical support, contact: system administrator</p>
                </footer>
            </div>
        </body>
        </html>
        """

        return html_content

    def create_daily_summary_report(self, daily_results: Dict[str, Any]) -> str:
        """
        Create a daily summary report for regular monitoring.

        Args:
            daily_results: Dictionary containing daily analysis results

        Returns:
            String containing formatted daily summary
        """
        target_date = daily_results.get("target_date", datetime.now())
        daily_insights = daily_results.get("daily_insights", {})
        trend_analysis = daily_results.get("trend_analysis", {})

        lines = [
            "=" * 70,
            f"PEMS v2 Daily Analysis Summary - {target_date.strftime('%Y-%m-%d')}",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # PV Summary
        if "pv" in daily_insights:
            pv_data = daily_insights["pv"]
            lines.extend(
                [
                    "☀️ PV Production:",
                    f"  • Daily Energy: {pv_data.get('daily_energy_kwh', 0):.1f} kWh",
                    f"  • Peak Power: {pv_data.get('peak_power_w', 0):,.0f} W",
                    f"  • Capacity Factor: {pv_data.get('capacity_factor', 0)*100:.1f}%",
                    f"  • Weather Efficiency: {pv_data.get('weather_efficiency', 0)*100:.1f}%",
                    "",
                ]
            )

        # Thermal Summary
        if "thermal" in daily_insights:
            thermal_data = daily_insights["thermal"]
            lines.extend(
                [
                    "🏠 Thermal Performance:",
                    f"  • Total Heating Hours: {thermal_data.get('total_heating_hours', 0):.1f}",
                    f"  • Rooms Analyzed: {thermal_data.get('rooms_analyzed', 0)}",
                    f"  • Average Efficiency: {thermal_data.get('avg_efficiency', 0)*100:.1f}%",
                    f"  • Average Heating Usage: {thermal_data.get('avg_heating_percentage', 0):.1f}%",
                    "",
                ]
            )

        # Base Load Summary
        if "base_load" in daily_insights:
            base_load_data = daily_insights["base_load"]
            lines.extend(
                [
                    "⚡ Base Load:",
                    f"  • Daily Consumption: {base_load_data.get('daily_base_load_kwh', 0):.1f} kWh",
                    f"  • Peak Demand: {base_load_data.get('peak_demand_w', 0):,.0f} W",
                    f"  • Load Factor: {base_load_data.get('load_factor', 0):.3f}",
                    f"  • Efficiency Rating: {base_load_data.get('efficiency_rating', 'unknown')}",
                    "",
                ]
            )

        # Relay Summary
        if "relay" in daily_insights:
            relay_data = daily_insights["relay"]
            lines.extend(
                [
                    "🔌 Relay Operations:",
                    f"  • Total Daily Cycles: {relay_data.get('total_cycles', 0):.0f}",
                    f"  • Total Runtime: {relay_data.get('total_runtime_hours', 0):.1f} hours",
                    f"  • Rooms Analyzed: {relay_data.get('rooms_analyzed', 0)}",
                    f"  • Avg Cycles per Room: {relay_data.get('avg_cycles_per_room', 0):.1f}",
                    "",
                ]
            )

        # Trend Analysis
        lines.extend(
            [
                "📈 Trend Analysis:",
                f"  • PV Trend: {trend_analysis.get('pv_trends', {}).get('energy_trend', 'unknown')}",
                f"  • Thermal Trend: {trend_analysis.get('thermal_trends', {}).get('efficiency_trend', 'unknown')}",
                f"  • Base Load Trend: {trend_analysis.get('base_load_trends', {}).get('consumption_trend', 'unknown')}",
                "",
            ]
        )

        # Alerts
        alerts = trend_analysis.get("alerts", [])
        if alerts:
            lines.extend(["🚨 Alerts:"])
            for alert in alerts:
                lines.append(f"  {alert}")
            lines.append("")
        else:
            lines.extend(["✅ No alerts detected.", ""])

        lines.extend(
            [
                "=" * 70,
                "Report completed successfully",
            ]
        )

        return "\n".join(lines)

    def create_data_quality_report(self, quality_data: Dict[str, Any]) -> str:
        """
        Create a detailed data quality report.

        Args:
            quality_data: Data quality information from preprocessing

        Returns:
            Formatted data quality report string
        """
        lines = [
            "=" * 60,
            "DATA QUALITY ASSESSMENT REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for data_type, report in quality_data.items():
            if isinstance(report, dict):
                lines.extend(
                    [
                        f"Dataset: {data_type.upper()}",
                        "-" * 30,
                        f"Total Records: {report.get('total_records', 0):,}",
                        f"Clean Records: {report.get('clean_records', 0):,}",
                        f"Original Missing Data: {report.get('original_missing_percentage', 0):.2f}%",
                        f"Final Missing Data: {report.get('clean_missing_percentage', 0):.2f}%",
                    ]
                )

                date_range = report.get("date_range", (None, None))
                if date_range[0] and date_range[1]:
                    lines.append(f"Date Range: {date_range[0]} to {date_range[1]}")

                time_gaps = report.get("time_gaps", [])
                if time_gaps:
                    lines.append(f"Significant Time Gaps: {len(time_gaps)}")
                    for i, (gap_time, duration) in enumerate(
                        time_gaps[:3]
                    ):  # Show first 3
                        lines.append(f"  Gap {i+1}: {gap_time} (Duration: {duration})")

                lines.extend(
                    [
                        f"Cleaning Summary: {report.get('cleaning_summary', 'No summary available')}",
                        "",
                    ]
                )

        return "\n".join(lines)

    def create_daily_summary_report(self, daily_results: Dict[str, Any]) -> str:
        """
        Create a daily summary report for regular monitoring.

        Args:
            daily_results: Dictionary containing daily analysis results

        Returns:
            String containing formatted daily summary
        """
        target_date = daily_results.get("target_date", datetime.now())
        daily_insights = daily_results.get("daily_insights", {})
        trend_analysis = daily_results.get("trend_analysis", {})

        lines = [
            "=" * 70,
            f"PEMS v2 Daily Analysis Summary - {target_date.strftime('%Y-%m-%d')}",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # PV Summary
        if "pv" in daily_insights:
            pv_data = daily_insights["pv"]
            lines.extend(
                [
                    "☀️ PV Production:",
                    f"  • Daily Energy: {pv_data.get('daily_energy_kwh', 0):.1f} kWh",
                    f"  • Peak Power: {pv_data.get('peak_power_w', 0):,.0f} W",
                    f"  • Capacity Factor: {pv_data.get('capacity_factor', 0)*100:.1f}%",
                    f"  • Weather Efficiency: {pv_data.get('weather_efficiency', 0)*100:.1f}%",
                    "",
                ]
            )

        # Thermal Summary
        if "thermal" in daily_insights:
            thermal_data = daily_insights["thermal"]
            lines.extend(
                [
                    "🏠 Thermal Performance:",
                    f"  • Total Heating Hours: {thermal_data.get('total_heating_hours', 0):.1f}",
                    f"  • Rooms Analyzed: {thermal_data.get('rooms_analyzed', 0)}",
                    f"  • Average Efficiency: {thermal_data.get('avg_efficiency', 0)*100:.1f}%",
                    f"  • Average Heating Usage: {thermal_data.get('avg_heating_percentage', 0):.1f}%",
                    "",
                ]
            )

        # Base Load Summary
        if "base_load" in daily_insights:
            base_load_data = daily_insights["base_load"]
            lines.extend(
                [
                    "⚡ Base Load:",
                    f"  • Daily Consumption: {base_load_data.get('daily_base_load_kwh', 0):.1f} kWh",
                    f"  • Peak Demand: {base_load_data.get('peak_demand_w', 0):,.0f} W",
                    f"  • Load Factor: {base_load_data.get('load_factor', 0):.3f}",
                    f"  • Efficiency Rating: {base_load_data.get('efficiency_rating', 'unknown')}",
                    "",
                ]
            )

        # Relay Summary
        if "relay" in daily_insights:
            relay_data = daily_insights["relay"]
            lines.extend(
                [
                    "🔌 Relay Operations:",
                    f"  • Total Daily Cycles: {relay_data.get('total_cycles', 0):.0f}",
                    f"  • Total Runtime: {relay_data.get('total_runtime_hours', 0):.1f} hours",
                    f"  • Rooms Analyzed: {relay_data.get('rooms_analyzed', 0)}",
                    f"  • Avg Cycles per Room: {relay_data.get('avg_cycles_per_room', 0):.1f}",
                    "",
                ]
            )

        # Trend Analysis
        lines.extend(
            [
                "📈 Trend Analysis:",
                f"  • PV Trend: {trend_analysis.get('pv_trends', {}).get('energy_trend', 'unknown')}",
                f"  • Thermal Trend: {trend_analysis.get('thermal_trends', {}).get('efficiency_trend', 'unknown')}",
                f"  • Base Load Trend: {trend_analysis.get('base_load_trends', {}).get('consumption_trend', 'unknown')}",
                "",
            ]
        )

        # Alerts
        alerts = trend_analysis.get("alerts", [])
        if alerts:
            lines.extend(["🚨 Alerts:"])
            for alert in alerts:
                lines.append(f"  {alert}")
            lines.append("")
        else:
            lines.extend(["✅ No alerts detected.", ""])

        lines.extend(
            [
                "=" * 70,
                "Report completed successfully",
            ]
        )

        return "\n".join(lines)

    def _get_enhanced_css_styles(self) -> str:
        """Get enhanced CSS styles for modern HTML reports."""
        return """
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #2c3e50;
            --border-color: #dee2e6;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: var(--card-background);
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 3px solid var(--secondary-color);
        }
        
        .report-header h1 {
            color: var(--primary-color);
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        
        .report-meta {
            display: flex;
            justify-content: center;
            gap: 30px;
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .table-of-contents {
            background: var(--background-color);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 40px;
        }
        
        .table-of-contents h2 {
            color: var(--primary-color);
            margin-bottom: 15px;
        }
        
        .table-of-contents ul {
            list-style: none;
        }
        
        .table-of-contents li {
            margin: 8px 0;
        }
        
        .table-of-contents a {
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: 500;
        }
        
        .table-of-contents a:hover {
            text-decoration: underline;
        }
        
        .analysis-section {
            margin-bottom: 50px;
            padding: 30px;
            background: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .analysis-section h2 {
            color: var(--primary-color);
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--secondary-color);
            font-size: 1.8rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .metric-card {
            background: var(--background-color);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .metric-card h4 {
            color: var(--primary-color);
            margin-bottom: 10px;
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--secondary-color);
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .data-table th,
        .data-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .data-table th {
            background: var(--primary-color);
            color: white;
            font-weight: 600;
        }
        
        .data-table tr:hover {
            background: var(--background-color);
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .status-good {
            background: var(--success-color);
            color: white;
        }
        
        .status-warning {
            background: var(--warning-color);
            color: white;
        }
        
        .status-error {
            background: var(--accent-color);
            color: white;
        }
        
        .recommendations-section {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 30px;
            border-radius: 8px;
            margin: 40px 0;
        }
        
        .recommendations-section h2 {
            color: #155724;
            margin-bottom: 20px;
        }
        
        .recommendations-list {
            list-style: none;
        }
        
        .recommendations-list li {
            margin: 12px 0;
            padding: 10px;
            background: rgba(255,255,255,0.7);
            border-radius: 4px;
            border-left: 4px solid #28a745;
        }
        
        .report-footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 1px solid var(--border-color);
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .report-header h1 {
                font-size: 2rem;
            }
            
            .report-meta {
                flex-direction: column;
                gap: 10px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _generate_executive_summary_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary HTML section."""
        return f"""
        <section id="executive-summary" class="analysis-section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                {self._generate_summary_metrics_html(analysis_results)}
            </div>
        </section>
        """

    def _generate_summary_metrics_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate summary metrics HTML."""
        metrics = []

        # PV metrics
        if "pv_analysis" in analysis_results:
            pv_data = analysis_results["pv_analysis"]
            basic_stats = pv_data.get("basic_stats", {})
            metrics.append(
                f"""
                <div class="metric-card">
                    <h4>☀️ PV Production</h4>
                    <div class="metric-value">{basic_stats.get('total_energy_kwh', 0):.1f}</div>
                    <div>kWh Total</div>
                </div>
            """
            )

        # Thermal metrics
        if "thermal_analysis" in analysis_results:
            thermal_data = analysis_results["thermal_analysis"]
            room_count = sum(
                1
                for room_data in thermal_data.values()
                if isinstance(room_data, dict) and "basic_stats" in room_data
            )
            metrics.append(
                f"""
                <div class="metric-card">
                    <h4>🏠 Rooms Analyzed</h4>
                    <div class="metric-value">{room_count}</div>
                    <div>Thermal Zones</div>
                </div>
            """
            )

        # Base load metrics
        if "base_load_analysis" in analysis_results:
            base_data = analysis_results["base_load_analysis"]
            basic_stats = base_data.get("basic_stats", {})
            metrics.append(
                f"""
                <div class="metric-card">
                    <h4>⚡ Base Load</h4>
                    <div class="metric-value">{basic_stats.get('mean_base_load', 0):.0f}</div>
                    <div>W Average</div>
                </div>
            """
            )

        return "".join(metrics)

    def _generate_detailed_analysis_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed analysis sections."""
        sections = []

        # PV Analysis
        if "pv_analysis" in analysis_results:
            sections.append(
                self._generate_pv_analysis_section_html(analysis_results["pv_analysis"])
            )

        # Thermal Analysis
        if "thermal_analysis" in analysis_results:
            sections.append(
                self._generate_thermal_analysis_section_html(
                    analysis_results["thermal_analysis"]
                )
            )

        # Base Load Analysis
        if "base_load_analysis" in analysis_results:
            sections.append(
                self._generate_base_load_section_html(
                    analysis_results["base_load_analysis"]
                )
            )

        # Relay Analysis
        if "relay_analysis" in analysis_results:
            sections.append(
                self._generate_relay_analysis_section_html(
                    analysis_results["relay_analysis"]
                )
            )

        return "".join(sections)

    def _generate_pv_analysis_section_html(self, pv_analysis: Dict[str, Any]) -> str:
        """Generate PV analysis HTML section."""
        basic_stats = pv_analysis.get("basic_stats", {})

        return f"""
        <section id="pv-analysis" class="analysis-section">
            <h2>☀️ PV Production Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Energy</h4>
                    <div class="metric-value">{basic_stats.get('total_energy_kwh', 0):.1f}</div>
                    <div>kWh</div>
                </div>
                <div class="metric-card">
                    <h4>Peak Power</h4>
                    <div class="metric-value">{basic_stats.get('max_power', 0):,.0f}</div>
                    <div>W</div>
                </div>
                <div class="metric-card">
                    <h4>Capacity Factor</h4>
                    <div class="metric-value">{basic_stats.get('capacity_factor', 0)*100:.1f}</div>
                    <div>%</div>
                </div>
                <div class="metric-card">
                    <h4>Daylight Factor</h4>
                    <div class="metric-value">{basic_stats.get('daylight_capacity_factor', 0)*100:.1f}</div>
                    <div>%</div>
                </div>
            </div>
        </section>
        """

    def _generate_thermal_analysis_section_html(
        self, thermal_analysis: Dict[str, Any]
    ) -> str:
        """Generate thermal analysis HTML section."""
        table_rows = []

        for room_name, room_data in thermal_analysis.items():
            if isinstance(room_data, dict) and "basic_stats" in room_data:
                stats = room_data["basic_stats"]
                heating_pct = stats.get("heating_percentage", 0)

                # Determine status based on heating percentage
                if heating_pct > 60:
                    status_class = "status-warning"
                    status_text = "High Usage"
                elif heating_pct > 30:
                    status_class = "status-good"
                    status_text = "Normal"
                else:
                    status_class = "status-good"
                    status_text = "Low Usage"

                table_rows.append(
                    f"""
                    <tr>
                        <td>{room_name}</td>
                        <td>{stats.get('mean_temperature', 0):.1f}°C</td>
                        <td>{stats.get('temperature_range', 0):.1f}°C</td>
                        <td>{heating_pct:.1f}%</td>
                        <td><span class="status-badge {status_class}">{status_text}</span></td>
                    </tr>
                """
                )

        return f"""
        <section id="thermal-analysis" class="analysis-section">
            <h2>🏠 Thermal Dynamics Analysis</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Room</th>
                        <th>Mean Temperature</th>
                        <th>Temperature Range</th>
                        <th>Heating Usage</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </section>
        """

    def _generate_base_load_section_html(
        self, base_load_analysis: Dict[str, Any]
    ) -> str:
        """Generate base load analysis HTML section."""
        basic_stats = base_load_analysis.get("basic_stats", {})

        return f"""
        <section id="base-load-analysis" class="analysis-section">
            <h2>⚡ Base Load Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Mean Base Load</h4>
                    <div class="metric-value">{basic_stats.get('mean_base_load', 0):,.0f}</div>
                    <div>W</div>
                </div>
                <div class="metric-card">
                    <h4>Total Energy</h4>
                    <div class="metric-value">{basic_stats.get('total_energy_kwh', 0):.1f}</div>
                    <div>kWh</div>
                </div>
                <div class="metric-card">
                    <h4>Load Factor</h4>
                    <div class="metric-value">{basic_stats.get('base_load_factor', 0):.3f}</div>
                    <div>Efficiency</div>
                </div>
                <div class="metric-card">
                    <h4>Base Load %</h4>
                    <div class="metric-value">{basic_stats.get('base_load_percentage', 0):.1f}</div>
                    <div>% of Total</div>
                </div>
            </div>
        </section>
        """

    def _generate_relay_analysis_section_html(
        self, relay_analysis: Dict[str, Any]
    ) -> str:
        """Generate relay analysis HTML section."""
        table_rows = []

        for room_name, room_data in relay_analysis.items():
            if isinstance(room_data, dict):
                switching_patterns = room_data.get("switching_patterns", {})
                switches_per_day = switching_patterns.get("switches_per_day", 0)

                # Determine efficiency status
                if switches_per_day > 20:
                    status_class = "status-warning"
                    status_text = "High Switching"
                elif switches_per_day > 10:
                    status_class = "status-good"
                    status_text = "Normal"
                else:
                    status_class = "status-good"
                    status_text = "Low Switching"

                table_rows.append(
                    f"""
                    <tr>
                        <td>{room_name}</td>
                        <td>{switches_per_day:.1f}</td>
                        <td><span class="status-badge {status_class}">{status_text}</span></td>
                    </tr>
                """
                )

        return f"""
        <section id="relay-analysis" class="analysis-section">
            <h2>🔌 Relay Pattern Analysis</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Room</th>
                        <th>Switches per Day</th>
                        <th>Efficiency Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </section>
        """

    def _generate_data_quality_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate data quality HTML section."""
        quality_data = analysis_results.get("data_quality", {})
        if not quality_data:
            return ""

        table_rows = []
        for source_name, source_data in quality_data.items():
            if isinstance(source_data, dict):
                if "records" in source_data:
                    records = source_data["records"]
                    missing_pct = source_data.get("missing_pct", 0)

                    if missing_pct < 5:
                        status_class = "status-good"
                        status_text = "Good"
                    elif missing_pct < 15:
                        status_class = "status-warning"
                        status_text = "Fair"
                    else:
                        status_class = "status-error"
                        status_text = "Poor"

                    table_rows.append(
                        f"""
                        <tr>
                            <td>{source_name}</td>
                            <td>{records:,}</td>
                            <td>{missing_pct:.1f}%</td>
                            <td><span class="status-badge {status_class}">{status_text}</span></td>
                        </tr>
                    """
                    )

        return f"""
        <section id="data-quality" class="analysis-section">
            <h2>📊 Data Quality Assessment</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Data Source</th>
                        <th>Records</th>
                        <th>Missing Data</th>
                        <th>Quality Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </section>
        """

    def _generate_recommendations_html(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations HTML section."""
        recommendations = self._generate_recommendations(analysis_results)

        if not recommendations:
            return ""

        rec_items = []
        for rec in recommendations:
            if rec.startswith("•"):
                rec_items.append(f"<li>{rec[2:].strip()}</li>")

        return f"""
        <section id="recommendations" class="recommendations-section">
            <h2>💡 Recommendations</h2>
            <ul class="recommendations-list">
                {''.join(rec_items)}
            </ul>
        </section>
        """
