"""
Growatt Controller Interface for PEMS v2.

This module provides the interface between PEMS v2 optimization engine
and the existing Loxone Smart Home Growatt controller. It translates
optimization decisions into the specific MQTT commands that the Growatt
system understands.

Architecture Overview:
- Receives Growatt mode schedules from optimization engine
- Converts schedules into time-based MQTT commands
- Interfaces with existing growatt_controller.py via MQTT
- Provides status monitoring and command confirmation
- Handles mode transitions and safety interlocks

Key Features:
1. **Mode Control**: Battery-first, AC charging, and export control
2. **Time Scheduling**: Convert optimization schedules to timed commands
3. **Safety Systems**: Prevent conflicting modes and unsafe operations
4. **Status Monitoring**: Track current modes and power flows
5. **MQTT Integration**: Interface with existing Growatt controller

Growatt Modes:
- Battery First: Prioritize battery discharge over grid import
- AC Charge: Force battery charging from grid during low-price periods
- Export Enable: Allow battery discharge for grid export during high prices
- Load First: Standard operation (default safe mode)

Integration Points:
- Input: Optimization schedules from optimizer.py
- Output: MQTT commands to growatt_controller topics
- Monitoring: Status feedback from Growatt system
- Safety: Coordination with unified_controller.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class GrowattCommand:
    """
    Represents a single Growatt control command.

    This dataclass encapsulates all information needed to execute a
    Growatt control action, including the mode, timing, and parameters.

    Attributes:
        mode: Control mode ("battery_first", "ac_charge", "export")
        enabled: Whether the mode should be enabled (True) or disabled (False)
        start_time: Start time for timed modes (HH:MM format)
        stop_time: Stop time for timed modes (HH:MM format)
        priority: Command priority for scheduling
        timestamp: When command was created
    """

    mode: str  # "battery_first", "ac_charge", "export"
    enabled: bool
    start_time: Optional[str] = None  # HH:MM format
    stop_time: Optional[str] = None  # HH:MM format
    priority: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        """Auto-populate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class GrowattStatus:
    """
    Represents the current operational status of the Growatt system.

    Attributes:
        battery_first_enabled: Whether battery-first mode is active
        ac_charge_enabled: Whether AC charging is active
        export_enabled: Whether grid export is enabled
        battery_soc: Current battery state of charge (0-1)
        battery_power_w: Current battery power (positive = charging)
        grid_power_w: Current grid power (positive = import)
        pv_power_w: Current PV generation
        last_updated: Timestamp of last status update
    """

    battery_first_enabled: bool
    ac_charge_enabled: bool
    export_enabled: bool
    battery_soc: float  # 0-1
    battery_power_w: float  # Positive = charging
    grid_power_w: float  # Positive = import
    pv_power_w: float
    last_updated: datetime


class GrowattController:
    """
    PEMS v2 interface to Growatt inverter/battery system.

    This class serves as the bridge between PEMS v2's optimization engine
    and the existing Loxone Smart Home Growatt controller. It handles the
    translation of optimization schedules into appropriate MQTT commands
    while maintaining safety and coordination with other systems.

    Core Responsibilities:
    1. **Schedule Execution**: Convert optimization schedules into timed MQTT commands
    2. **Mode Management**: Coordinate battery-first, AC charge, and export modes
    3. **Safety Enforcement**: Prevent conflicting modes and unsafe operations
    4. **Status Monitoring**: Track system state and provide feedback
    5. **Command Queuing**: Manage command timing and execution

    MQTT Interface:
    Uses the existing Growatt controller MQTT topics:
    - Battery First: growatt/battery_first -> {"start": "HH:MM", "stop": "HH:MM", "enabled": bool, "slot": 1}
    - AC Charge: growatt/ac_charge -> {"value": bool}
    - Export: growatt/export_enable -> {"value": bool}

    Safety Features:
    - Prevent AC charge + export simultaneously
    - Respect SOC limits and battery protection
    - Graceful mode transitions with proper timing
    - Emergency stop capability
    """

    def __init__(self, mqtt_client, settings: Optional[Dict[str, Any]] = None):
        """
        Initialize Growatt controller with MQTT client and settings.

        Args:
            mqtt_client: MQTT client for communication
            settings: Configuration settings (optional)
        """
        self.mqtt_client = mqtt_client
        self.settings = settings or {}
        self.logger = logging.getLogger(__name__)

        # MQTT topics (matching existing growatt_controller.py)
        self.battery_first_topic = "growatt/battery_first"
        self.ac_charge_topic = "growatt/ac_charge"
        self.export_enable_topic = "growatt/export_enable"
        self.export_disable_topic = "growatt/export_disable"

        # Control state
        self.current_commands: List[GrowattCommand] = []
        self.last_status: Optional[GrowattStatus] = None
        self.command_queue: List[GrowattCommand] = []

        # Safety settings
        self.simulation_mode = self.settings.get("simulation_mode", True)
        self.safety_checks = self.settings.get("safety_checks", True)

        self.logger.info("Growatt controller initialized")

    async def initialize(self):
        """Initialize MQTT connections and status monitoring."""
        try:
            # Subscribe to status topics if available
            # In production, this would subscribe to Growatt status feeds
            self.logger.info("Growatt controller MQTT initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Growatt controller: {e}")
            raise

    async def execute_optimization_schedule(
        self,
        battery_first_schedule: pd.Series,
        ac_charge_schedule: pd.Series,
        export_schedule: pd.Series,
    ) -> Dict[str, bool]:
        """
        Execute optimization schedules for all Growatt modes.

        Args:
            battery_first_schedule: Binary schedule for battery-first mode
            ac_charge_schedule: Binary schedule for AC charging mode
            export_schedule: Binary schedule for export enable mode

        Returns:
            Dict with success status for each mode
        """
        results = {}

        try:
            self.logger.info("Executing Growatt optimization schedule")

            # Execute current timestep commands (first value in each schedule)
            current_battery_first = (
                bool(battery_first_schedule.iloc[0])
                if len(battery_first_schedule) > 0
                else False
            )
            current_ac_charge = (
                bool(ac_charge_schedule.iloc[0])
                if len(ac_charge_schedule) > 0
                else False
            )
            current_export = (
                bool(export_schedule.iloc[0]) if len(export_schedule) > 0 else False
            )

            # Safety check: prevent AC charge + export simultaneously
            if current_ac_charge and current_export:
                self.logger.warning(
                    "Safety violation: AC charge and export both enabled, disabling export"
                )
                current_export = False

            # Execute commands in parallel
            tasks = [
                self._set_battery_first_mode(current_battery_first),
                self._set_ac_charge_mode(current_ac_charge),
                self._set_export_mode(current_export),
            ]

            command_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            mode_names = ["battery_first", "ac_charge", "export"]
            for i, (mode, result) in enumerate(zip(mode_names, command_results)):
                results[mode] = not isinstance(result, Exception)
                if isinstance(result, Exception):
                    self.logger.error(f"Command failed for {mode}: {result}")

            success_count = sum(results.values())
            self.logger.info(
                f"Growatt schedule execution: {success_count}/{len(results)} modes successful"
            )

        except Exception as e:
            self.logger.error(f"Growatt schedule execution failed: {e}")
            results = {"battery_first": False, "ac_charge": False, "export": False}

        return results

    async def _set_battery_first_mode(self, enabled: bool) -> bool:
        """Set battery-first mode on/off."""
        try:
            if enabled:
                # Enable battery-first for next 2 hours (optimization will update)
                current_time = datetime.now()
                start_time = current_time.strftime("%H:%M")
                stop_time = (current_time + timedelta(hours=2)).strftime("%H:%M")

                payload = {
                    "start": start_time,
                    "stop": stop_time,
                    "enabled": True,
                    "slot": 1,
                }
            else:
                # Disable battery-first
                payload = {
                    "start": "00:00",
                    "stop": "00:00",
                    "enabled": False,
                    "slot": 1,
                }

            if self.simulation_mode:
                self.logger.info(
                    f"🔋 [SIMULATE] Battery-first {'ENABLED' if enabled else 'DISABLED'}"
                )
                return True

            await self.mqtt_client.publish(
                self.battery_first_topic, json.dumps(payload)
            )
            self.logger.info(f"🔋 Battery-first {'ENABLED' if enabled else 'DISABLED'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set battery-first mode: {e}")
            return False

    async def _set_ac_charge_mode(self, enabled: bool) -> bool:
        """Set AC charging mode on/off."""
        try:
            payload = {"value": enabled}

            if self.simulation_mode:
                self.logger.info(
                    f"⚡ [SIMULATE] AC charge {'ENABLED' if enabled else 'DISABLED'}"
                )
                return True

            await self.mqtt_client.publish(self.ac_charge_topic, json.dumps(payload))
            self.logger.info(f"⚡ AC charge {'ENABLED' if enabled else 'DISABLED'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set AC charge mode: {e}")
            return False

    async def _set_export_mode(self, enabled: bool) -> bool:
        """Set export mode on/off."""
        try:
            payload = {"value": enabled}
            topic = self.export_enable_topic if enabled else self.export_disable_topic

            if self.simulation_mode:
                self.logger.info(
                    f"⬆️ [SIMULATE] Export {'ENABLED' if enabled else 'DISABLED'}"
                )
                return True

            await self.mqtt_client.publish(topic, json.dumps(payload))
            self.logger.info(f"⬆️ Export {'ENABLED' if enabled else 'DISABLED'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to set export mode: {e}")
            return False

    async def get_status(self) -> Optional[GrowattStatus]:
        """Get current Growatt system status."""
        try:
            # In production, this would query actual status via MQTT/API
            # For now, return mock status based on last commands

            return GrowattStatus(
                battery_first_enabled=False,  # Would be read from system
                ac_charge_enabled=False,
                export_enabled=False,
                battery_soc=0.65,  # 65%
                battery_power_w=0.0,
                grid_power_w=2500.0,  # 2.5kW import
                pv_power_w=0.0,  # Night
                last_updated=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Failed to get Growatt status: {e}")
            return None

    async def emergency_stop(self) -> bool:
        """Emergency stop - disable all active modes."""
        try:
            self.logger.warning("Growatt emergency stop activated")

            tasks = [
                self._set_battery_first_mode(False),
                self._set_ac_charge_mode(False),
                self._set_export_mode(False),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)

            self.logger.info(
                f"Growatt emergency stop: {success_count}/3 modes disabled"
            )
            return success_count == 3

        except Exception as e:
            self.logger.error(f"Growatt emergency stop failed: {e}")
            return False

    def get_system_limits(self) -> Dict[str, Any]:
        """Get Growatt system limits and capabilities."""
        return {
            "max_charge_power_kw": 5.0,
            "max_discharge_power_kw": 5.0,
            "max_export_power_kw": 10.0,
            "battery_capacity_kwh": 10.0,
            "min_soc": 0.1,
            "max_soc": 0.9,
            "modes": ["load_first", "battery_first", "ac_charge", "export"],
        }


def create_growatt_controller(
    mqtt_client, settings: Dict[str, Any] = None
) -> GrowattController:
    """
    Create a configured Growatt controller.

    Args:
        mqtt_client: MQTT client for communication
        settings: Configuration settings

    Returns:
        Configured GrowattController instance
    """

    if settings is None:
        settings = {"simulation_mode": True, "safety_checks": True}

    return GrowattController(mqtt_client, settings)
