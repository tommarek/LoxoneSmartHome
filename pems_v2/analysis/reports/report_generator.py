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
    """

    def __init__(self):
        """Initialize the report generator."""
        self.logger = logging.getLogger(f"{__name__}.ReportGenerator")

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
        Create an HTML report with visualizations.

        Args:
            analysis_results: Dictionary containing analysis results
            processed_data: Optional processed data for additional charts

        Returns:
            HTML string containing the report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PEMS v2 Comprehensive Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
                .summary-box {{ 
                    background-color: #ecf0f1; 
                    padding: 15px; 
                    border-left: 4px solid #3498db; 
                    margin: 15px 0; 
                }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px 15px; 
                    padding: 10px; 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                }}
                .recommendations {{ 
                    background-color: #d4edda; 
                    border: 1px solid #c3e6cb; 
                    padding: 15px; 
                    border-radius: 5px; 
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 15px 0; 
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>PEMS v2 Comprehensive Analysis Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            {self._generate_html_summary_section(analysis_results)}
            {self._generate_html_detailed_sections(analysis_results)}
            {self._generate_html_recommendations_section(analysis_results)}
            
        </body>
        </html>
        """

        return html_content

    def _generate_html_summary_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML summary section."""
        sections = []

        if "pv_analysis" in analysis_results:
            pv_data = analysis_results["pv_analysis"]
            if "basic_stats" in pv_data:
                stats = pv_data["basic_stats"]
                sections.append(
                    f"""
                <div class="summary-box">
                    <h3>PV Production Summary</h3>
                    <div class="metric">Total Energy: {stats.get('total_energy_kwh', 0):.1f} kWh</div>
                    <div class="metric">Max Power: {stats.get('max_power', 0):.1f} W</div>
                    <div class="metric">Capacity Factor: {stats.get('capacity_factor', 0)*100:.1f}%</div>
                </div>
                """
                )

        if "base_load_analysis" in analysis_results:
            base_data = analysis_results["base_load_analysis"]
            if "basic_stats" in base_data:
                stats = base_data["basic_stats"]
                sections.append(
                    f"""
                <div class="summary-box">
                    <h3>Base Load Summary</h3>
                    <div class="metric">Mean Base Load: {stats.get('mean_base_load', 0):.1f} W</div>
                    <div class="metric">Total Energy: {stats.get('total_energy_kwh', 0):.1f} kWh</div>
                    <div class="metric">Load Factor: {stats.get('base_load_factor', 0):.3f}</div>
                </div>
                """
                )

        return "".join(sections)

    def _generate_html_detailed_sections(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed HTML sections."""
        sections = ["<h2>Detailed Analysis Results</h2>"]

        # Thermal analysis table
        if "thermal_analysis" in analysis_results:
            thermal_data = analysis_results["thermal_analysis"]
            table_rows = []

            for room_name, room_data in thermal_data.items():
                if isinstance(room_data, dict) and "basic_stats" in room_data:
                    stats = room_data["basic_stats"]
                    table_rows.append(
                        f"""
                    <tr>
                        <td>{room_name}</td>
                        <td>{stats.get('mean_temperature', 0):.1f}°C</td>
                        <td>{stats.get('temperature_range', 0):.1f}°C</td>
                        <td>{stats.get('heating_percentage', 0):.1f}%</td>
                    </tr>
                    """
                    )

            if table_rows:
                sections.append(
                    f"""
                <h3>Room Thermal Analysis</h3>
                <table>
                    <tr>
                        <th>Room</th>
                        <th>Mean Temperature</th>
                        <th>Temperature Range</th>
                        <th>Heating Usage</th>
                    </tr>
                    {"".join(table_rows)}
                </table>
                """
                )

        return "".join(sections)

    def _generate_html_recommendations_section(
        self, analysis_results: Dict[str, Any]
    ) -> str:
        """Generate HTML recommendations section."""
        recommendations = self._generate_recommendations(analysis_results)

        # Convert to HTML list
        rec_html = []
        for rec in recommendations:
            if rec.startswith("•"):
                rec_html.append(f"<li>{rec[2:].strip()}</li>")  # Remove bullet and trim

        if rec_html:
            return f"""
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {"".join(rec_html)}
                </ul>
            </div>
            """

        return ""

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
