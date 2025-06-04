#!/usr/bin/env python3
"""
Generate corrected analysis report with proper relay understanding.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add the current directory to the path
import sys
sys.path.append(str(Path(__file__).parent))

from test_relay_analysis import RelayAnalyzer


async def generate_corrected_report():
    """Generate corrected analysis report understanding relay operations."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Generating Corrected PEMS v2 Analysis Report")
    
    # Create analyzer
    analyzer = RelayAnalyzer()
    
    # Analyze 2-year relay data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    logger.info(f"Extracting 2-year relay data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    relay_data = await analyzer.extract_relay_data(start_date, end_date)
    
    if not relay_data.empty:
        analysis_results = analyzer.analyze_relay_patterns(relay_data)
        
        # Create corrected report
        report_lines = [
            "="*80,
            "CORRECTED PEMS V2 - 2 YEAR RELAY-BASED HEATING ANALYSIS",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "",
            "RELAY SYSTEM UNDERSTANDING",
            "-" * 30,
            "• Heating uses ON/OFF relays (binary states: 0=OFF, 1=ON)",
            "• Each room has fixed power rating when relay is ON",
            "• Energy = Relay_State × Power_Rating × Time_Duration",
            "• Duty cycle = Percentage of time relay is ON",
            "• Total system capacity: 19.0 kW (sum of all room ratings)",
            "",
            "CORRECTED DATA ANALYSIS",
            "-" * 30,
        ]
        
        summary = analysis_results['summary']
        report_lines.extend([
            f"• Total Heating Energy (2 years): {summary['total_energy_kwh']:.1f} kWh",
            f"• Annual heating estimate: {summary['total_energy_kwh']/2:.1f} kWh/year",
            f"• System utilization: {summary['system_utilization_percent']:.1f}% of theoretical maximum",
            f"• Total relay records analyzed: {len(relay_data):,}",
            "",
            "ROOM POWER RATINGS & PERFORMANCE",
            "-" * 40,
        ])
        
        # Room details
        rooms = analysis_results['rooms']
        for room, stats in sorted(rooms.items(), key=lambda x: x[1]['total_energy_kwh'], reverse=True):
            duty = stats['duty_cycle_percent']
            energy = stats['total_energy_kwh']
            rating = stats['power_rating_kw']
            switches = stats['total_switches']
            
            report_lines.append(f"• {room:20s}: {rating:4.1f}kW rated, {energy:6.1f}kWh consumed, {duty:5.1f}% duty, {switches:4.0f} switches")
        
        report_lines.extend([
            "",
            "KEY RELAY INSIGHTS",
            "-" * 30,
            f"• Most energy: {max(rooms.items(), key=lambda x: x[1]['total_energy_kwh'])[0]} ({max(x['total_energy_kwh'] for x in rooms.values()):.1f} kWh)",
            f"• Highest duty cycle: {max(rooms.items(), key=lambda x: x[1]['duty_cycle_percent'])[0]} ({max(x['duty_cycle_percent'] for x in rooms.values()):.1f}%)",
            f"• Most switching: {max(rooms.items(), key=lambda x: x[1]['total_switches'])[0]} ({max(x['total_switches'] for x in rooms.values()):.0f} switches)",
            "",
            "SYSTEM EFFICIENCY OBSERVATIONS",
            "-" * 30,
            "• Low system utilization indicates efficient temperature control",
            "• High switching frequency may indicate need for control tuning",
            "• Duty cycles reflect actual heating demand vs comfort requirements",
            "• Room-specific patterns show usage and thermal characteristics",
            "",
            "OPTIMIZATION OPPORTUNITIES",
            "-" * 30,
            "• Relay coordination can reduce peak demand",
            "• Time-of-use scheduling can optimize energy costs",
            "• Predictive control can reduce switching frequency",
            "• Zone grouping can improve overall efficiency",
            "",
            "="*80,
            "End of Corrected Relay Analysis Report",
            "="*80
        ])
        
        # Save corrected report
        report_content = "\n".join(report_lines)
        
        with open("analysis/reports/corrected_relay_analysis_2year.txt", 'w') as f:
            f.write(report_content)
        
        # Print to console
        print("\n" + report_content)
        
        logger.info("✅ Corrected analysis report generated successfully!")
        
    else:
        logger.warning("No relay data found for 2-year analysis")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(generate_corrected_report())
    print(f"\nCorrected Report Generation: {'✅ SUCCESS' if success else '❌ FAILED'}")