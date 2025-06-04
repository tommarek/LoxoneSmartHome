#!/usr/bin/env python3
"""
Test relay-based heating analysis - the core functionality for your Loxone system.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the main relay analyzer from root directory
sys.path.append(str(Path(__file__).parent.parent))

# Import the RelayAnalyzer class from the main relay analysis script
import importlib.util
spec = importlib.util.spec_from_file_location("relay_analyzer", Path(__file__).parent.parent / "test_relay_analysis.py")
relay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay_module)
RelayAnalyzer = relay_module.RelayAnalyzer


async def test_relay_analysis():
    """Test the relay analysis functionality."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🔥 TESTING RELAY-BASED HEATING ANALYSIS")
    print("="*60)
    
    try:
        # Create analyzer
        analyzer = RelayAnalyzer()
        
        # Test with last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"Analyzing relay data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Extract relay data
        logger.info("Extracting relay data...")
        relay_data = await analyzer.extract_relay_data(start_date, end_date)
        
        if relay_data.empty:
            logger.warning("No relay data found in the specified period")
            return False
        
        logger.info(f"Found {len(relay_data)} relay records")
        logger.info(f"Rooms: {relay_data['room'].unique().tolist()}")
        
        # Analyze patterns
        logger.info("Analyzing relay patterns...")
        results = analyzer.analyze_relay_patterns(relay_data)
        
        # Display results
        print("\n📈 ANALYSIS RESULTS")
        print("="*30)
        
        summary = results['summary']
        print(f"Total Energy Consumed: {summary['total_energy_kwh']:.1f} kWh")
        print(f"System Utilization: {summary['system_utilization_percent']:.1f}%")
        print(f"Analysis Period: {summary['analysis_period_hours']:.1f} hours")
        
        print("\n🏠 ROOM BREAKDOWN")
        print("-" * 50)
        
        rooms = results['rooms']
        for room_name, room_stats in sorted(rooms.items(), key=lambda x: x[1]['total_energy_kwh'], reverse=True):
            duty = room_stats['duty_cycle_percent']
            energy = room_stats['total_energy_kwh']
            switches = room_stats['total_switches']
            power = room_stats['power_rating_kw']
            
            print(f"{room_name:20s}: {energy:6.1f} kWh, {duty:5.1f}% duty, {power:4.1f}kW, {switches:4.0f} switches")
        
        # Save results
        analyzer.save_results(results, "test_relay_analysis_results.json")
        
        logger.info("✅ Relay analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Relay analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_relay_analysis())
    if success:
        print("\n🎉 Relay analysis test passed!")
    else:
        print("\n❌ Relay analysis test failed.")
    sys.exit(0 if success else 1)
