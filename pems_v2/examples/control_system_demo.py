#!/usr/bin/env python3
"""
PEMS v2 Control System Demonstration.

This script demonstrates the usage of the PEMS v2 control system interfaces
including heating, battery, inverter control, and unified coordination.

Usage:
    python examples/control_system_demo.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.energy_settings import ROOM_CONFIG
# Import PEMS v2 control modules
from modules.control import (ChargingMode, InverterMode, StrategyContext,
                             StrategyType, create_battery_controller,
                             create_control_strategies,
                             create_heating_controller,
                             create_inverter_controller,
                             create_unified_controller)


def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


async def demo_heating_controller():
    """Demonstrate heating controller functionality."""
    print("\n🏠 === HEATING CONTROLLER DEMO ===")

    # Create heating controller configuration
    mqtt_config = {
        "broker": "localhost",
        "port": 1883,
        "heating_topic_prefix": "pems/heating",
    }

    heating_config = {
        "rooms": ROOM_CONFIG["rooms"],
        "mqtt": mqtt_config,
        "max_switching_per_hour": 12,
        "safety_timeout_minutes": 60,
    }

    # Create and initialize controller
    controller = create_heating_controller(ROOM_CONFIG["rooms"], mqtt_config)
    await controller.initialize()

    # Demonstrate room heating control
    print("Setting living room heating ON...")
    success = await controller.set_room_heating("obyvak", True, duration_minutes=30)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Demonstrate temperature control
    print("Setting bedroom temperature to 21°C...")
    success = await controller.set_room_temperature(
        "loznice", 21.0, duration_minutes=60
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Demonstrate zone control
    print("Setting living zone temperature to 22°C...")
    results = await controller.set_zone_temperature("living", 22.0)
    success_count = sum(results.values())
    print(f"Result: {success_count}/{len(results)} rooms successful")

    # Get status
    status = await controller.get_all_status()
    print(f"Active heating rooms: {len(status)}")

    return controller


async def demo_battery_controller():
    """Demonstrate battery controller functionality."""
    print("\n🔋 === BATTERY CONTROLLER DEMO ===")

    # Create battery controller configuration
    battery_config = {
        "capacity_kwh": 10.0,
        "max_charge_power_kw": 5.0,
        "max_discharge_power_kw": 5.0,
    }

    mqtt_config = {
        "battery_control_topic": "pems/battery/set",
        "battery_status_topic": "growatt/battery/status",
    }

    # Create and initialize controller
    controller = create_battery_controller(battery_config, mqtt_config)
    await controller.initialize()

    # Demonstrate charging control
    print("Enabling grid charging at 3kW...")
    success = await controller.enable_grid_charging(3.0)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Demonstrate mode switching
    print("Setting battery to PV-only mode...")
    success = await controller.set_charging_mode(ChargingMode.PV_ONLY)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Check charging availability
    available = await controller.is_charging_available()
    print(f"Charging available: {'✅ Yes' if available else '❌ No'}")

    # Get status
    status = await controller.get_status()
    if status:
        print(f"Battery SOC: {status.soc_percent}%")
        print(f"Battery power: {status.power_kw}kW")

    return controller


async def demo_inverter_controller():
    """Demonstrate inverter controller functionality."""
    print("\n⚡ === INVERTER CONTROLLER DEMO ===")

    # Create inverter controller configuration
    inverter_config = {
        "capacity_kw": 10.0,
        "max_export_kw": 10.0,
        "efficiency_percent": 95.0,
    }

    mqtt_config = {
        "inverter_mode_topic": "pems/inverter/mode/set",
        "inverter_export_topic": "pems/inverter/export/set",
        "inverter_status_topic": "growatt/inverter/status",
    }

    grid_config = {"connection_kw": 20.0, "export_limit_kw": 10.0}

    # Create and initialize controller
    controller = create_inverter_controller(inverter_config, mqtt_config, grid_config)
    await controller.initialize()

    # Demonstrate mode control
    print("Setting inverter to battery-first mode...")
    success = await controller.set_battery_first_mode()
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Demonstrate export control
    print("Enabling export with 5kW limit...")
    success = await controller.enable_export(5.0)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")

    # Price-based optimization
    print("Optimizing for prices (buy: 4.0, sell: 3.5 CZK/kWh)...")
    recommended_mode = await controller.optimize_for_price(4.0, 3.5)
    print(f"Recommended mode: {recommended_mode.value}")

    # Get power flows
    power_flows = await controller.get_power_flows()
    print(f"Power flows: {power_flows}")

    return controller


async def demo_unified_controller():
    """Demonstrate unified controller functionality."""
    print("\n🎛️  === UNIFIED CONTROLLER DEMO ===")

    # Create comprehensive configuration
    heating_config = {
        "rooms": ROOM_CONFIG["rooms"],
        "mqtt": {"broker": "localhost", "port": 1883},
        "max_switching_per_hour": 12,
    }

    battery_config = {
        "battery": {"capacity_kwh": 10.0, "max_charge_power_kw": 5.0},
        "mqtt": {"battery_control_topic": "pems/battery/set"},
        "safety": {"max_soc_percent": 95.0},
    }

    inverter_config = {
        "inverter": {"capacity_kw": 10.0, "max_export_kw": 10.0},
        "mqtt": {"inverter_mode_topic": "pems/inverter/mode/set"},
        "grid": {"connection_kw": 20.0},
    }

    system_config = {"max_total_power_kw": 25.0, "safety_timeout_minutes": 60}

    # Create and initialize unified controller
    controller = create_unified_controller(
        heating_config, battery_config, inverter_config, system_config
    )
    await controller.initialize()

    # Demonstrate system status
    print("Getting comprehensive system status...")
    status = await controller.get_system_status()
    if status:
        print(f"System mode: {status.mode.value}")
        print(f"Total power: {status.total_power_kw:.1f}kW")
        print(f"System healthy: {'✅ Yes' if status.is_healthy else '❌ No'}")
        print(f"Active heating rooms: {len(status.heating_status)}")

    # Get system limits
    limits = controller.get_system_limits()
    print(f"System limits: {limits}")

    return controller


async def demo_control_strategies():
    """Demonstrate control strategies functionality."""
    print("\n🧠 === CONTROL STRATEGIES DEMO ===")

    # Create control strategies
    strategies = create_control_strategies()

    # Create example context
    context = StrategyContext(
        current_time=datetime.now(),
        electricity_price_czk_kwh=4.5,
        export_price_czk_kwh=3.2,
        outdoor_temp_c=5.0,
        occupancy_active=True,
        battery_soc_percent=60.0,
        pv_forecast_kw=[0, 0, 2, 5, 8, 10, 8, 5] + [0] * 16,  # 24 hour forecast
        system_status={
            "heating_safe": True,
            "battery_safe": True,
            "inverter_safe": True,
        },
    )

    # Test different strategies
    strategy_types = [
        StrategyType.ECONOMIC,
        StrategyType.COMFORT,
        StrategyType.BALANCED,
    ]

    for strategy_type in strategy_types:
        print(f"\nTesting {strategy_type.value} strategy...")
        result = await strategies.execute_strategy(context, strategy_type)

        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Expected cost: {result.expected_cost_czk:.1f} CZK")
        print(f"  Expected comfort: {result.expected_comfort_score:.2f}")
        print(f"  Heating rooms: {len(result.schedule.heating_schedule)}")
        print(f"  Battery mode: {result.schedule.battery_mode}")
        print(f"  Inverter mode: {result.schedule.inverter_mode}")

    # Get performance statistics
    performance = strategies.get_strategy_performance()
    print(f"\nStrategy performance: {performance}")

    return strategies


async def main():
    """Main demonstration function."""
    print("🚀 PEMS v2 Control System Demonstration")
    print("=" * 50)

    try:
        # Run individual controller demos
        heating_controller = await demo_heating_controller()
        battery_controller = await demo_battery_controller()
        inverter_controller = await demo_inverter_controller()
        unified_controller = await demo_unified_controller()
        control_strategies = await demo_control_strategies()

        print("\n✅ All demonstrations completed successfully!")
        print("\n📊 Summary:")
        print(
            f"  - Heating controller: {len(heating_controller.rooms)} rooms configured"
        )
        print(f"  - Battery controller: {battery_controller.capacity_kwh}kWh capacity")
        print(f"  - Inverter controller: {inverter_controller.capacity_kw}kW capacity")
        print(f"  - Unified controller: Coordinating all systems")
        print(f"  - Control strategies: {len(StrategyType)} strategies available")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
