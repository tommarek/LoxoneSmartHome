"""
Unified Control Interface for PEMS v2.

This module provides a unified interface for controlling all energy systems
in the smart home including heating, battery, and inverter systems. It serves
as the primary control layer for the optimization engine and provides
coordinated control across all subsystems.

Key Features:
1. **Unified Interface**: Single point of control for all energy systems
2. **Coordinated Control**: Synchronize actions across heating, battery, inverter
3. **Safety Management**: System-wide safety checks and emergency procedures
4. **Status Aggregation**: Consolidated monitoring across all subsystems
5. **Configuration Management**: Centralized system configuration and limits

Control Capabilities:
- Heating: Room-by-room control and zone-based temperature management
- Battery: Charging control, mode switching, power management
- Inverter: Operating mode control, export management, energy flow optimization
- Safety: Emergency stops, system protection, constraint enforcement

Usage in PEMS v2 Workflow:
1. Optimization engine generates comprehensive control schedules
2. UnifiedController receives and validates all control commands
3. Commands distributed to appropriate subsystem controllers
4. Status monitoring and feedback collection from all systems
5. Coordinated safety management and emergency procedures
6. Performance metrics aggregation for optimization feedback

Integration Points:
- Input: Control schedules from PEMS v2 optimization engine
- Output: Commands to heating, battery, and inverter controllers
- Monitoring: Consolidated status from all energy subsystems
- Configuration: System specifications from energy_settings.py
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .growatt_controller import (GrowattCommand, GrowattController,
                                 GrowattStatus)
from .heating_controller import (HeatingCommand, HeatingController,
                                 HeatingStatus)


class SystemMode(Enum):
    """Overall system operating modes."""

    NORMAL = "normal"  # Normal optimization operation
    ECONOMY = "economy"  # Maximum cost savings mode
    COMFORT = "comfort"  # Maximum comfort mode
    EXPORT = "export"  # Maximum export revenue mode
    EMERGENCY = "emergency"  # Emergency safe mode
    MAINTENANCE = "maintenance"  # Maintenance mode


@dataclass
class SystemStatus:
    """
    Represents the overall system status.

    Attributes:
        mode: Current system operating mode
        heating_status: Status from all heating zones
        growatt_status: Growatt inverter/battery system status
        total_power_kw: Total system power consumption
        safety_status: System safety indicators
        last_updated: Timestamp of last status update
        is_healthy: Overall system health indicator
    """

    mode: SystemMode
    heating_status: Dict[str, HeatingStatus]
    growatt_status: Optional[GrowattStatus]
    total_power_kw: float
    safety_status: Dict[str, bool]
    last_updated: datetime
    is_healthy: bool = True


@dataclass
class ControlSchedule:
    """
    Represents a comprehensive control schedule for all systems.

    Attributes:
        heating_schedule: Room heating states and temperatures
        battery_first_enabled: Enable battery-first mode
        ac_charge_enabled: Enable AC charging from grid
        export_enabled: Enable grid export
        duration_minutes: Duration for this schedule
        priority: Schedule priority level
    """

    heating_schedule: Dict[str, Tuple[bool, Optional[float]]]  # room -> (state, temp)
    battery_first_enabled: Optional[bool] = None
    ac_charge_enabled: Optional[bool] = None
    export_enabled: Optional[bool] = None
    duration_minutes: Optional[int] = None
    priority: int = 0


class UnifiedController:
    """
    Unified interface for comprehensive energy system control.

    This class provides coordinated control across all energy subsystems
    including heating, battery, and inverter management. It ensures
    safe operation, optimal coordination, and comprehensive monitoring.

    Core Responsibilities:
    1. **Unified Control**: Single interface for all energy system commands
    2. **System Coordination**: Ensure compatible settings across subsystems
    3. **Safety Management**: System-wide safety checks and emergency procedures
    4. **Status Aggregation**: Consolidated monitoring and health assessment
    5. **Configuration Management**: Centralized system limits and constraints

    Control Hierarchy:
    - System-level modes and strategies
    - Subsystem-specific control commands
    - Room and device-level fine control
    - Emergency and safety overrides

    Safety Features:
    - Power limit validation across all systems
    - Mode compatibility checking
    - Emergency stop coordination
    - System health monitoring
    - Constraint enforcement
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize unified controller with comprehensive system configuration.

        Args:
            config: Configuration dictionary containing:
                - heating: Heating system configuration
                - battery: Battery system configuration
                - inverter: Inverter system configuration
                - system: Overall system limits and settings
                - mqtt: MQTT communication settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # System configuration
        self.system_config = config.get("system", {})
        self.max_total_power = self.system_config.get("max_total_power_kw", 25.0)
        self.safety_timeout = self.system_config.get("safety_timeout_minutes", 60)

        # Initialize subsystem controllers
        self.heating_controller = None
        self.growatt_controller = None

        # Control state
        self.current_mode = SystemMode.NORMAL
        self.current_schedule: Optional[ControlSchedule] = None
        self.last_status: Optional[SystemStatus] = None
        self.emergency_active = False

        # Command history for debugging
        self.command_history: List[Tuple[datetime, str, Any]] = []

        self.logger.info("Unified controller initialized")

    async def initialize(self):
        """Initialize all subsystem controllers and MQTT connections."""
        try:
            # Initialize heating controller
            heating_config = self.config.get("heating", {})
            if heating_config:
                self.heating_controller = HeatingController(heating_config)
                await self.heating_controller.initialize()
                self.logger.info("Heating controller initialized")

            # Initialize Growatt controller
            growatt_config = self.config.get("growatt", {})
            mqtt_client = self.config.get("mqtt_client")
            if growatt_config and mqtt_client:
                self.growatt_controller = GrowattController(mqtt_client, growatt_config)
                await self.growatt_controller.initialize()
                self.logger.info("Growatt controller initialized")

            self.logger.info("Unified controller fully initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize unified controller: {e}")
            raise

    async def execute_schedule(self, schedule: ControlSchedule) -> Dict[str, bool]:
        """
        Execute a comprehensive control schedule across all systems.

        Args:
            schedule: Complete control schedule for all systems

        Returns:
            Dict with success status for each subsystem
        """
        results = {}

        try:
            self.logger.info("Executing unified control schedule")

            # Safety checks
            if not self._validate_schedule(schedule):
                self.logger.error("Schedule validation failed")
                return {"validation": False}

            # Execute heating schedule
            if schedule.heating_schedule and self.heating_controller:
                heating_results = await self._execute_heating_schedule(schedule)
                results["heating"] = all(heating_results.values())
                self.logger.debug(f"Heating execution: {heating_results}")

            # Execute Growatt commands
            if self.growatt_controller and (
                schedule.battery_first_enabled is not None
                or schedule.ac_charge_enabled is not None
                or schedule.export_enabled is not None
            ):
                growatt_success = await self._execute_growatt_schedule(schedule)
                results["growatt"] = growatt_success
                self.logger.debug(f"Growatt execution: {growatt_success}")

            # Store successful schedule
            if all(results.values()):
                self.current_schedule = schedule
                self._log_command("execute_schedule", schedule)

            success_count = sum(results.values())
            total_count = len(results)
            self.logger.info(
                f"Schedule execution: {success_count}/{total_count} subsystems successful"
            )

        except Exception as e:
            self.logger.error(f"Schedule execution failed: {e}")
            results["error"] = False

        return results

    async def set_system_mode(self, mode: SystemMode) -> bool:
        """
        Set overall system operating mode.

        Args:
            mode: Target system mode

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Setting system mode to {mode.value}")

            # Create mode-specific schedule
            schedule = self._create_mode_schedule(mode)

            if schedule:
                results = await self.execute_schedule(schedule)
                success = all(results.values())

                if success:
                    self.current_mode = mode
                    self.logger.info(f"System mode set to {mode.value}")

                return success

            return False

        except Exception as e:
            self.logger.error(f"Failed to set system mode: {e}")
            return False

    async def get_system_status(self) -> Optional[SystemStatus]:
        """
        Get comprehensive system status from all subsystems.

        Returns:
            SystemStatus object with all subsystem information
        """
        try:
            # Collect status from all subsystems
            heating_status = {}
            if self.heating_controller:
                heating_status = await self.heating_controller.get_all_status()

            growatt_status = None
            if self.growatt_controller:
                growatt_status = await self.growatt_controller.get_status()

            # Calculate total power
            total_power = self._calculate_total_power(heating_status, growatt_status)

            # Assess safety status
            safety_status = self._assess_safety_status(heating_status, growatt_status)

            # Create comprehensive status
            status = SystemStatus(
                mode=self.current_mode,
                heating_status=heating_status,
                growatt_status=growatt_status,
                total_power_kw=total_power,
                safety_status=safety_status,
                last_updated=datetime.now(),
                is_healthy=all(safety_status.values()),
            )

            self.last_status = status
            return status

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return None

    async def emergency_stop(self) -> bool:
        """
        Execute emergency stop across all systems.

        Returns:
            True if all systems successfully stopped
        """
        try:
            self.logger.warning("EMERGENCY STOP ACTIVATED")
            self.emergency_active = True

            results = []

            # Stop all heating
            if self.heating_controller:
                heating_result = await self.heating_controller.emergency_stop()
                results.append(heating_result)
                self.logger.info(
                    f"Heating emergency stop: {'SUCCESS' if heating_result else 'FAILED'}"
                )

            # Stop Growatt operations
            if self.growatt_controller:
                growatt_result = await self.growatt_controller.emergency_stop()
                results.append(growatt_result)
                self.logger.info(
                    f"Growatt emergency stop: {'SUCCESS' if growatt_result else 'FAILED'}"
                )

            # Update system mode
            self.current_mode = SystemMode.EMERGENCY

            success = all(results)
            if success:
                self.logger.warning("EMERGENCY STOP COMPLETED SUCCESSFULLY")
            else:
                self.logger.error("EMERGENCY STOP PARTIALLY FAILED")

            self._log_command("emergency_stop", {"success": success})

            return success

        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False

    async def clear_emergency(self) -> bool:
        """
        Clear emergency mode and return to normal operation.

        Returns:
            True if successful
        """
        if not self.emergency_active:
            return True

        try:
            self.logger.info("Clearing emergency mode")

            # Return to normal mode
            success = await self.set_system_mode(SystemMode.NORMAL)

            if success:
                self.emergency_active = False
                self.logger.info("Emergency mode cleared")

            return success

        except Exception as e:
            self.logger.error(f"Failed to clear emergency mode: {e}")
            return False

    def get_system_limits(self) -> Dict[str, Any]:
        """
        Get comprehensive system limits and capabilities.

        Returns:
            Dictionary with all system limits
        """
        limits = {
            "max_total_power_kw": self.max_total_power,
            "safety_timeout_minutes": self.safety_timeout,
        }

        if self.heating_controller:
            limits["heating"] = {
                "total_power_kw": sum(
                    room.get("power_kw", 0)
                    for room in self.heating_controller.rooms.values()
                ),
                "room_count": len(self.heating_controller.rooms),
            }

        if self.growatt_controller:
            limits["growatt"] = self.growatt_controller.get_system_limits()

        return limits

    async def _execute_heating_schedule(
        self, schedule: ControlSchedule
    ) -> Dict[str, bool]:
        """Execute heating portion of schedule."""
        results = {}

        for room, (state, temp) in schedule.heating_schedule.items():
            try:
                if temp is not None:
                    # Temperature control
                    success = await self.heating_controller.set_room_temperature(
                        room, temp, schedule.duration_minutes
                    )
                else:
                    # On/off control
                    success = await self.heating_controller.set_room_heating(
                        room, state, schedule.duration_minutes
                    )

                results[room] = success

            except Exception as e:
                self.logger.error(f"Failed to execute heating command for {room}: {e}")
                results[room] = False

        return results

    async def _execute_growatt_schedule(self, schedule: ControlSchedule) -> bool:
        """Execute Growatt portion of schedule."""
        try:
            # Create mock schedules with single values for immediate execution
            import pandas as pd

            battery_first_schedule = pd.Series(
                [schedule.battery_first_enabled or False]
            )
            ac_charge_schedule = pd.Series([schedule.ac_charge_enabled or False])
            export_schedule = pd.Series([schedule.export_enabled or False])

            # Execute via Growatt controller
            results = await self.growatt_controller.execute_optimization_schedule(
                battery_first_schedule, ac_charge_schedule, export_schedule
            )

            # Return True if all modes executed successfully
            return all(results.values())

        except Exception as e:
            self.logger.error(f"Failed to execute Growatt schedule: {e}")
            return False

    def _validate_schedule(self, schedule: ControlSchedule) -> bool:
        """Validate schedule against system constraints."""
        try:
            # Check total power limits
            estimated_power = 0.0

            # Heating power
            for room, (state, _) in schedule.heating_schedule.items():
                if (
                    state
                    and self.heating_controller
                    and room in self.heating_controller.rooms
                ):
                    room_power = self.heating_controller.rooms[room].get("power_kw", 0)
                    estimated_power += room_power

            # AC charging power (when enabled, assume 5kW charging)
            if schedule.ac_charge_enabled:
                estimated_power += 5.0  # Growatt AC charging power

            # Check against limits
            if estimated_power > self.max_total_power:
                self.logger.error(
                    f"Schedule power {estimated_power}kW exceeds limit {self.max_total_power}kW"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Schedule validation failed: {e}")
            return False

    def _create_mode_schedule(self, mode: SystemMode) -> Optional[ControlSchedule]:
        """Create schedule based on system mode."""
        try:
            if mode == SystemMode.EMERGENCY:
                # Emergency: All off
                heating_schedule = {}
                if self.heating_controller:
                    heating_schedule = {
                        room: (False, None) for room in self.heating_controller.rooms
                    }

                return ControlSchedule(
                    heating_schedule=heating_schedule,
                    battery_first_enabled=False,
                    ac_charge_enabled=False,
                    export_enabled=False,
                )

            elif mode == SystemMode.ECONOMY:
                # Economy: Minimal heating, battery priority
                return ControlSchedule(
                    heating_schedule={},  # No heating changes
                    battery_first_enabled=True,
                    ac_charge_enabled=True,
                    export_enabled=False,
                )

            elif mode == SystemMode.EXPORT:
                # Export: Maximum export revenue
                return ControlSchedule(
                    heating_schedule={},  # No heating changes
                    battery_first_enabled=False,
                    ac_charge_enabled=False,
                    export_enabled=True,
                )

            else:  # NORMAL, COMFORT, MAINTENANCE
                # Normal operation - no changes
                return None

        except Exception as e:
            self.logger.error(f"Failed to create mode schedule: {e}")
            return None

    def _calculate_total_power(
        self,
        heating_status: Dict[str, HeatingStatus],
        growatt_status: Optional[GrowattStatus],
    ) -> float:
        """Calculate total system power consumption."""
        total_power = 0.0

        # Heating power
        for status in heating_status.values():
            total_power += status.power_w / 1000.0  # Convert to kW

        # Growatt power (from grid import)
        if growatt_status and growatt_status.grid_power_w > 0:
            total_power += growatt_status.grid_power_w / 1000.0  # Convert to kW

        return total_power

    def _assess_safety_status(
        self,
        heating_status: Dict[str, HeatingStatus],
        growatt_status: Optional[GrowattStatus],
    ) -> Dict[str, bool]:
        """Assess overall system safety status."""
        safety = {
            "heating_safe": True,
            "growatt_safe": True,
            "power_safe": True,
            "emergency_clear": not self.emergency_active,
        }

        # Check heating safety
        for status in heating_status.values():
            if status.power_w > 10000:  # 10kW per room seems excessive
                safety["heating_safe"] = False

        # Check Growatt safety
        if growatt_status:
            if not (0.1 <= growatt_status.battery_soc <= 0.9):
                safety["growatt_safe"] = False
            # Check for unsafe mode combinations
            if growatt_status.ac_charge_enabled and growatt_status.export_enabled:
                safety["growatt_safe"] = False

        # Check total power
        total_power = self._calculate_total_power(heating_status, growatt_status)
        if total_power > self.max_total_power:
            safety["power_safe"] = False

        return safety

    def _log_command(self, command_type: str, data: Any):
        """Log command to history."""
        self.command_history.append((datetime.now(), command_type, data))

        # Keep only last 100 commands
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]


def create_unified_controller(
    heating_config: Dict[str, Any],
    growatt_config: Dict[str, Any],
    mqtt_client,
    system_config: Dict[str, Any] = None,
) -> UnifiedController:
    """
    Create a configured unified controller.

    Args:
        heating_config: Heating system configuration
        growatt_config: Growatt system configuration
        mqtt_client: MQTT client for communication
        system_config: Overall system configuration

    Returns:
        Configured UnifiedController instance
    """

    if system_config is None:
        system_config = {"max_total_power_kw": 25.0, "safety_timeout_minutes": 60}

    config = {
        "heating": heating_config,
        "growatt": growatt_config,
        "mqtt_client": mqtt_client,
        "system": system_config,
    }

    return UnifiedController(config)
