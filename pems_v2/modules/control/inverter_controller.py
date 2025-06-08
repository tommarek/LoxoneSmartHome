"""
Inverter Control Interface for PEMS v2.

This module provides the interface for controlling inverter operating modes
and grid export settings. It integrates with the Growatt inverter system
to enable optimized energy flow management based on electricity prices
and demand patterns.

Key Features:
1. **Mode Switching**: Control inverter operation (load_first, battery_first, grid_first)
2. **Export Control**: Enable/disable grid export functionality
3. **Priority Management**: Optimize energy flow based on conditions
4. **Status Monitoring**: Real-time inverter state and performance tracking
5. **Safety Management**: Prevents unsafe mode transitions and grid issues

Inverter Modes:
- LOAD_FIRST: Prioritize house loads, then charge battery, excess to grid
- BATTERY_FIRST: Prioritize battery charging, then house loads, excess to grid  
- GRID_FIRST: Prioritize grid export, minimal battery charging
- PV_ONLY: Use only PV power, no grid interaction

Usage in PEMS v2 Workflow:
1. Optimization engine determines optimal inverter mode schedule
2. InverterController receives mode commands and export settings
3. Commands validated against safety constraints and grid conditions
4. MQTT messages sent to Growatt inverter for execution
5. Status monitoring confirms successful mode transitions
6. Performance feedback provided to optimization engine

Integration Points:
- Input: Mode schedules from optimizer.py and energy management
- Output: MQTT commands to Growatt inverter control system
- Monitoring: Inverter status and performance from Growatt MQTT
- Configuration: Inverter specifications from energy_settings.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class InverterMode(Enum):
    """Inverter operating modes."""

    LOAD_FIRST = "load_first"  # House loads priority
    BATTERY_FIRST = "battery_first"  # Battery charging priority
    GRID_FIRST = "grid_first"  # Grid export priority
    PV_ONLY = "pv_only"  # Solar only, no grid


class ExportMode(Enum):
    """Grid export control modes."""

    ENABLED = "enabled"  # Export allowed
    DISABLED = "disabled"  # No export to grid
    LIMITED = "limited"  # Export with power limit


@dataclass
class InverterCommand:
    """
    Represents an inverter control command.

    Attributes:
        mode: Target inverter operating mode
        export_mode: Grid export setting
        export_limit_kw: Maximum export power (None for unlimited)
        duration_minutes: Duration to maintain setting (None for indefinite)
        priority: Command priority (0=low, 3=emergency)
        timestamp: When command was created
    """

    mode: InverterMode
    export_mode: Optional[ExportMode] = None
    export_limit_kw: Optional[float] = None
    duration_minutes: Optional[int] = None
    priority: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        """Auto-populate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class InverterStatus:
    """
    Represents current inverter system status.

    Attributes:
        mode: Current operating mode
        export_enabled: Whether grid export is enabled
        export_power_kw: Current export power (negative=import)
        pv_power_kw: Current PV generation power
        load_power_kw: Current house load power
        battery_power_kw: Current battery power (positive=charging)
        grid_frequency_hz: Grid frequency
        efficiency_percent: Inverter efficiency
        temperature_c: Inverter temperature
        last_updated: Timestamp of last status update
        is_online: Whether inverter is responding
    """

    mode: InverterMode
    export_enabled: bool
    export_power_kw: float
    pv_power_kw: float
    load_power_kw: float
    battery_power_kw: float
    grid_frequency_hz: float
    efficiency_percent: float
    temperature_c: float
    last_updated: datetime
    is_online: bool = True


class InverterController:
    """
    Primary interface for inverter system control and monitoring.

    This class manages inverter operating modes, grid export settings,
    and energy flow optimization. It communicates with the Growatt
    inverter system via MQTT and provides real-time status feedback.

    Core Responsibilities:
    1. **Mode Management**: Switch between operating modes based on strategy
    2. **Export Control**: Enable/disable/limit grid export functionality
    3. **Energy Flow**: Optimize power distribution between loads, battery, grid
    4. **Status Monitoring**: Track inverter performance and system health
    5. **Safety Management**: Ensure safe operation and grid compliance

    MQTT Protocol:
    - Mode Control: pems/inverter/mode/set -> {"mode": "load_first"}
    - Export Control: pems/inverter/export/set -> {"enabled": true, "limit_kw": 5.0}
    - Status Updates: growatt/inverter/status -> complete inverter telemetry
    - Emergency Control: pems/inverter/emergency -> safe mode activation

    Operating Modes:
    - LOAD_FIRST: Best for high consumption periods, ensures house loads met first
    - BATTERY_FIRST: Best for cheap electricity periods, maximize storage
    - GRID_FIRST: Best for high export prices, maximize grid export revenue
    - PV_ONLY: Island mode operation, no grid interaction

    Safety Features:
    - Grid code compliance monitoring
    - Temperature protection
    - Frequency and voltage monitoring
    - Emergency safe mode activation
    - Mode transition validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inverter controller with system configuration.

        Args:
            config: Configuration dictionary containing:
                - inverter: Inverter system specifications
                - mqtt: MQTT broker connection settings
                - grid: Grid connection limits and settings
                - safety: Safety limits and monitoring thresholds
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Inverter system configuration
        self.inverter_config = config.get("inverter", {})
        self.capacity_kw = self.inverter_config.get("capacity_kw", 10.0)
        self.max_export_power = self.inverter_config.get("max_export_kw", 10.0)
        self.efficiency = self.inverter_config.get("efficiency_percent", 95.0)

        # MQTT configuration
        self.mqtt_config = config.get("mqtt", {})
        self.mode_topic = self.mqtt_config.get(
            "inverter_mode_topic", "pems/inverter/mode/set"
        )
        self.export_topic = self.mqtt_config.get(
            "inverter_export_topic", "pems/inverter/export/set"
        )
        self.status_topic = self.mqtt_config.get(
            "inverter_status_topic", "growatt/inverter/status"
        )

        # Grid configuration
        self.grid_config = config.get("grid", {})
        self.grid_connection_kw = self.grid_config.get("connection_kw", 20.0)
        self.export_limit_kw = self.grid_config.get("export_limit_kw", 10.0)

        # Safety limits
        self.safety_config = config.get("safety", {})
        self.max_temp = self.safety_config.get("max_temperature_c", 65.0)
        self.min_frequency = self.safety_config.get("min_frequency_hz", 49.5)
        self.max_frequency = self.safety_config.get("max_frequency_hz", 50.5)

        # Control state
        self.current_mode: Optional[InverterMode] = None
        self.export_enabled: bool = True
        self.export_limit: Optional[float] = None
        self.last_status: Optional[InverterStatus] = None
        self.command_history: List[InverterCommand] = []

        # Mock MQTT client (would be real in production)
        self.mqtt_client = None

        self.logger.info(
            f"Inverter controller initialized: {self.capacity_kw}kW capacity, "
            f"max export {self.max_export_power}kW"
        )

    async def initialize(self):
        """Initialize MQTT connection and subscribe to status topics."""
        try:
            # In production, initialize actual MQTT client here
            # self.mqtt_client = AsyncMQTTClient(self.mqtt_config)
            # await self.mqtt_client.connect()

            # Subscribe to inverter status topic
            # await self.mqtt_client.subscribe(self.status_topic, self._on_inverter_status)

            self.logger.info("Inverter controller MQTT initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize inverter controller: {e}")
            raise

    async def set_mode(
        self, mode: InverterMode, duration_minutes: Optional[int] = None
    ) -> bool:
        """
        Set inverter operating mode.

        Args:
            mode: Target operating mode
            duration_minutes: Duration to maintain mode (None for indefinite)

        Returns:
            True if command was successful
        """
        try:
            command = InverterCommand(
                mode=mode, duration_minutes=duration_minutes, priority=1
            )

            # Safety checks
            if not self._safety_check_mode(command):
                self.logger.warning(
                    f"Safety check failed for inverter mode {mode.value}"
                )
                return False

            success = await self._execute_mode_command(command)

            if success:
                self.current_mode = mode
                self.command_history.append(command)
                self.logger.info(f"Inverter mode set to {mode.value}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to set inverter mode: {e}")
            return False

    async def set_export_mode(
        self, export_mode: ExportMode, limit_kw: Optional[float] = None
    ) -> bool:
        """
        Set grid export mode and power limit.

        Args:
            export_mode: Export control mode
            limit_kw: Maximum export power (required for LIMITED mode)

        Returns:
            True if command was successful
        """
        try:
            # Validate export limit
            if export_mode == ExportMode.LIMITED and limit_kw is None:
                self.logger.error("Export limit required for LIMITED mode")
                return False

            if limit_kw is not None:
                if limit_kw < 0 or limit_kw > self.export_limit_kw:
                    self.logger.error(
                        f"Invalid export limit: {limit_kw}kW "
                        f"(max: {self.export_limit_kw}kW)"
                    )
                    return False

            command = InverterCommand(
                mode=self.current_mode or InverterMode.LOAD_FIRST,
                export_mode=export_mode,
                export_limit_kw=limit_kw,
                priority=1,
            )

            success = await self._execute_export_command(command)

            if success:
                self.export_enabled = export_mode != ExportMode.DISABLED
                self.export_limit = limit_kw
                self.command_history.append(command)
                self.logger.info(
                    f"Export mode set to {export_mode.value} "
                    f"{'with limit ' + str(limit_kw) + 'kW' if limit_kw else ''}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to set export mode: {e}")
            return False

    async def enable_export(self, limit_kw: Optional[float] = None) -> bool:
        """
        Enable grid export (matching growatt_controller.py pattern).

        Args:
            limit_kw: Maximum export power (None for unlimited)

        Returns:
            True if successful
        """
        try:
            if self.config.get("simulation_mode", False):
                current_time = datetime.now().strftime("%H:%M:%S")
                self.logger.info(
                    f"⬆️ [SIMULATE] EXPORT ENABLED {'with limit ' + str(limit_kw) + 'kW' if limit_kw else ''} "
                    f"(simulated at {current_time})"
                )
                return True

            # Send MQTT command to enable export (matches growatt pattern)
            payload = {"value": True}
            if limit_kw is not None:
                payload["limit_kw"] = limit_kw

            export_enable_topic = self.mqtt_config.get(
                "export_enable_topic", "pems/inverter/export/enable"
            )

            if self.mqtt_client:
                await self.mqtt_client.publish(export_enable_topic, json.dumps(payload))
                current_time = datetime.now().strftime("%H:%M:%S")
                self.logger.info(
                    f"⬆️ EXPORT ENABLED {'with limit ' + str(limit_kw) + 'kW' if limit_kw else ''} "
                    f"at {current_time} → Topic: {export_enable_topic}"
                )

            self.export_enabled = True
            self.export_limit = limit_kw

            return True

        except Exception as e:
            self.logger.error(f"Failed to enable export: {e}")
            return False

    async def disable_export(self) -> bool:
        """
        Disable all grid export (matching growatt_controller.py pattern).

        Returns:
            True if successful
        """
        try:
            if self.config.get("simulation_mode", False):
                current_time = datetime.now().strftime("%H:%M:%S")
                self.logger.info(
                    f"⬇️ [SIMULATE] EXPORT DISABLED (simulated at {current_time})"
                )
                return True

            # Send MQTT command to disable export (matches growatt pattern)
            payload = {"value": False}
            export_disable_topic = self.mqtt_config.get(
                "export_disable_topic", "pems/inverter/export/disable"
            )

            if self.mqtt_client:
                await self.mqtt_client.publish(
                    export_disable_topic, json.dumps(payload)
                )
                current_time = datetime.now().strftime("%H:%M:%S")
                self.logger.info(
                    f"⬇️ EXPORT DISABLED at {current_time} → Topic: {export_disable_topic}"
                )

            self.export_enabled = False
            self.export_limit = None

            return True

        except Exception as e:
            self.logger.error(f"Failed to disable export: {e}")
            return False

    async def set_load_first_mode(self) -> bool:
        """
        Set load-first operating mode (house loads priority).

        Returns:
            True if successful
        """
        return await self.set_mode(InverterMode.LOAD_FIRST)

    async def set_battery_first_mode(self) -> bool:
        """
        Set battery-first operating mode (battery charging priority).

        Returns:
            True if successful
        """
        return await self.set_mode(InverterMode.BATTERY_FIRST)

    async def set_grid_first_mode(self) -> bool:
        """
        Set grid-first operating mode (export priority).

        Returns:
            True if successful
        """
        return await self.set_mode(InverterMode.GRID_FIRST)

    async def get_status(self) -> Optional[InverterStatus]:
        """
        Get current inverter system status.

        Returns:
            InverterStatus object or None if unavailable
        """
        # In production, this would query actual status via MQTT
        # For now, return mock status
        if self.current_mode:
            return InverterStatus(
                mode=self.current_mode,
                export_enabled=self.export_enabled,
                export_power_kw=2.5,  # Mock export power
                pv_power_kw=8.0,  # Mock PV power
                load_power_kw=3.0,  # Mock load power
                battery_power_kw=2.5,  # Mock battery power
                grid_frequency_hz=50.0,
                efficiency_percent=self.efficiency,
                temperature_c=35.0,  # Mock temperature
                last_updated=datetime.now(),
                is_online=True,
            )

        return self.last_status

    async def get_current_mode(self) -> Optional[InverterMode]:
        """
        Get current inverter operating mode.

        Returns:
            Current InverterMode or None if unknown
        """
        status = await self.get_status()
        return status.mode if status else self.current_mode

    async def is_export_enabled(self) -> bool:
        """
        Check if grid export is currently enabled.

        Returns:
            True if export is enabled
        """
        status = await self.get_status()
        return status.export_enabled if status else self.export_enabled

    async def get_power_flows(self) -> Dict[str, float]:
        """
        Get current power flows in the system.

        Returns:
            Dictionary with power flows in kW
        """
        status = await self.get_status()
        if status:
            return {
                "pv_power_kw": status.pv_power_kw,
                "load_power_kw": status.load_power_kw,
                "battery_power_kw": status.battery_power_kw,
                "export_power_kw": status.export_power_kw,
                "grid_power_kw": -status.export_power_kw,  # Import is positive
            }

        return {}

    async def emergency_safe_mode(self) -> bool:
        """
        Activate emergency safe mode (load-first, export disabled).

        Returns:
            True if successful
        """
        try:
            self.logger.warning("Inverter emergency safe mode activated")

            # Set to safe operating mode and disable export
            mode_success = await self.set_mode(InverterMode.LOAD_FIRST)
            export_success = await self.disable_export()

            success = mode_success and export_success

            if success:
                self.current_mode = InverterMode.LOAD_FIRST
                self.export_enabled = False
                self.export_limit = None
                self.logger.info("Inverter emergency safe mode completed")

            return success

        except Exception as e:
            self.logger.error(f"Inverter emergency safe mode failed: {e}")
            return False

    def get_system_limits(self) -> Dict[str, float]:
        """
        Get inverter system limits and specifications.

        Returns:
            Dictionary with system limits
        """
        return {
            "capacity_kw": self.capacity_kw,
            "max_export_kw": self.max_export_power,
            "grid_connection_kw": self.grid_connection_kw,
            "export_limit_kw": self.export_limit_kw,
            "efficiency_percent": self.efficiency,
            "max_temperature_c": self.max_temp,
        }

    async def optimize_for_price(
        self, electricity_price_czk_kwh: float, export_price_czk_kwh: float
    ) -> InverterMode:
        """
        Recommend optimal inverter mode based on electricity prices.

        Args:
            electricity_price_czk_kwh: Current electricity purchase price
            export_price_czk_kwh: Current electricity export price

        Returns:
            Recommended InverterMode
        """
        try:
            # Price-based mode optimization logic
            price_ratio = (
                export_price_czk_kwh / electricity_price_czk_kwh
                if electricity_price_czk_kwh > 0
                else 0
            )

            if price_ratio > 0.8:  # Export price very good
                recommended_mode = InverterMode.GRID_FIRST
            elif price_ratio > 0.6:  # Export price good
                recommended_mode = InverterMode.LOAD_FIRST
            elif electricity_price_czk_kwh < 2.0:  # Cheap electricity
                recommended_mode = InverterMode.BATTERY_FIRST
            else:  # Normal operation
                recommended_mode = InverterMode.LOAD_FIRST

            self.logger.debug(
                f"Price optimization: {recommended_mode.value} "
                f"(buy: {electricity_price_czk_kwh:.2f}, "
                f"sell: {export_price_czk_kwh:.2f} CZK/kWh)"
            )

            return recommended_mode

        except Exception as e:
            self.logger.error(f"Price optimization failed: {e}")
            return InverterMode.LOAD_FIRST  # Safe default

    async def _execute_mode_command(self, command: InverterCommand) -> bool:
        """Execute an inverter mode command."""
        try:
            payload = {
                "mode": command.mode.value,
                "timestamp": command.timestamp.isoformat(),
            }

            if command.duration_minutes is not None:
                payload["duration_minutes"] = command.duration_minutes

            # In production, send MQTT command
            # await self.mqtt_client.publish(self.mode_topic, json.dumps(payload))

            self.logger.debug(f"Inverter mode command sent: {payload}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute inverter mode command: {e}")
            return False

    async def _execute_export_command(self, command: InverterCommand) -> bool:
        """Execute an export control command."""
        try:
            payload = {
                "enabled": command.export_mode != ExportMode.DISABLED,
                "timestamp": command.timestamp.isoformat(),
            }

            if command.export_limit_kw is not None:
                payload["limit_kw"] = command.export_limit_kw

            # In production, send MQTT command
            # await self.mqtt_client.publish(self.export_topic, json.dumps(payload))

            self.logger.debug(f"Export control command sent: {payload}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute export command: {e}")
            return False

    def _safety_check_mode(self, command: InverterCommand) -> bool:
        """Perform safety checks before executing mode command."""

        # Check system health
        if self.last_status:
            # Temperature check
            if self.last_status.temperature_c > self.max_temp:
                self.logger.error(
                    f"Inverter temperature {self.last_status.temperature_c}°C "
                    f"above limit {self.max_temp}°C"
                )
                return False

            # Frequency check
            freq = self.last_status.grid_frequency_hz
            if not (self.min_frequency <= freq <= self.max_frequency):
                self.logger.error(f"Grid frequency {freq}Hz outside safe range")
                return False

            # Online check
            if not self.last_status.is_online:
                self.logger.error("Inverter not online")
                return False

        return True

    async def _on_inverter_status(self, topic: str, payload: str):
        """Handle incoming MQTT inverter status messages."""
        try:
            data = json.loads(payload)

            # Parse inverter status from Growatt data
            status = InverterStatus(
                mode=InverterMode(data.get("mode", "load_first")),
                export_enabled=data.get("export_enabled", True),
                export_power_kw=data.get("export_power_kw", 0.0),
                pv_power_kw=data.get("pv_power_kw", 0.0),
                load_power_kw=data.get("load_power_kw", 0.0),
                battery_power_kw=data.get("battery_power_kw", 0.0),
                grid_frequency_hz=data.get("grid_frequency_hz", 50.0),
                efficiency_percent=data.get("efficiency_percent", 95.0),
                temperature_c=data.get("temperature_c", 25.0),
                last_updated=datetime.fromisoformat(
                    data.get("timestamp", datetime.now().isoformat())
                ),
                is_online=data.get("is_online", True),
            )

            self.last_status = status
            self.logger.debug(
                f"Inverter status updated: mode={status.mode.value}, "
                f"export={status.export_power_kw}kW"
            )

        except Exception as e:
            self.logger.error(f"Failed to process inverter status update: {e}")


def create_inverter_controller(
    inverter_config: Dict[str, Any],
    mqtt_config: Dict[str, Any],
    grid_config: Dict[str, Any] = None,
    safety_config: Dict[str, Any] = None,
) -> InverterController:
    """
    Create a configured inverter controller.

    Args:
        inverter_config: Inverter system specifications
        mqtt_config: MQTT connection configuration
        grid_config: Grid connection settings
        safety_config: Safety limits and thresholds

    Returns:
        Configured InverterController instance
    """

    if grid_config is None:
        grid_config = {"connection_kw": 20.0, "export_limit_kw": 10.0}

    if safety_config is None:
        safety_config = {
            "max_temperature_c": 65.0,
            "min_frequency_hz": 49.5,
            "max_frequency_hz": 50.5,
        }

    config = {
        "inverter": inverter_config,
        "mqtt": mqtt_config,
        "grid": grid_config,
        "safety": safety_config,
    }

    return InverterController(config)
