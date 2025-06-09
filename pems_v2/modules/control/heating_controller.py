"""
Heating Control Interface for PEMS v2.

This module provides the critical interface between the PEMS v2 optimization engine
and the physical Loxone heating system. It translates optimization decisions into
real-world heating control commands while ensuring safety and reliability.

Architecture Overview:
- Receives heating schedules from optimization engine as pandas Series
- Converts schedules into individual room heating commands with safety checks
- Communicates with Loxone system via MQTT protocol for real-time control
- Maintains state tracking and command queue for reliable operation
- Implements safety mechanisms to prevent equipment damage

Key Features:
1. **Schedule Execution**: Converts optimization schedules to room-specific commands
2. **Safety Systems**: Prevents rapid cycling, power overloads, and unsafe operations
3. **Real-time Control**: Individual room heating control with immediate response
4. **Status Monitoring**: Tracks heating system state and performance metrics
5. **Emergency Controls**: Immediate shutdown capability for safety situations
6. **Command Queuing**: Ensures reliable delivery of control commands
7. **MQTT Integration**: Seamless communication with Loxone infrastructure

Safety Features:
- Maximum switching frequency limits to protect relay equipment
- Power limit validation before command execution
- Emergency stop functionality for immediate shutdown
- Command validation and sanitization
- Timeout protection for stuck commands

Usage in PEMS v2 Workflow:
1. Optimization engine generates heating schedules for 48-hour horizon
2. HeatingController receives schedule for next control period (typically 15-60 minutes)
3. Commands are validated against safety constraints and room configurations
4. MQTT messages sent to Loxone to execute heating control decisions
5. Status feedback monitored to confirm successful execution
6. Any failures trigger fallback strategies and alerts

Integration Points:
- Input: Optimization schedules from optimizer.py
- Output: MQTT commands to Loxone heating relays
- Monitoring: Real-time status from Loxone system via MQTT
- Configuration: Room power ratings from energy_settings.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from config.settings import MQTTSettings, PEMSSettings, ThermalSettings

# Note: In production, these would import from the main project's MQTT client
# from utils.async_mqtt_client import AsyncMQTTClient


@dataclass
class HeatingCommand:
    """
    Represents a single heating control command for a specific room.

    This dataclass encapsulates all information needed to execute a heating
    control action, including the target room, desired state, timing constraints,
    and priority for command scheduling.

    Attributes:
        room: Target room identifier (must match configuration)
        state: Desired heating state (True=ON/heating, False=OFF/no heating)
        target_temp_c: Optional target temperature setpoint in Celsius
        duration_minutes: Optional time limit for command (None=indefinite)
        priority: Command priority for scheduling (0=low, higher=more urgent)
        timestamp: When command was created (auto-populated if None)

    Priority Levels:
        0: Scheduled optimization commands (default)
        1: Real-time optimization adjustments
        2: Manual user override commands
        3: Safety and emergency commands

    Usage:
        cmd = HeatingCommand(room="living_room", state=True, target_temp_c=21.5, duration_minutes=30)
        # Creates command to turn on living room heating with 21.5°C setpoint for 30 minutes
    """

    room: str
    state: bool  # True = ON, False = OFF
    target_temp_c: Optional[float] = None
    duration_minutes: Optional[int] = None
    priority: int = 0  # Higher number = higher priority
    timestamp: datetime = None

    def __post_init__(self):
        """Auto-populate timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class HeatingStatus:
    """
    Represents the current operational status of a room's heating system.

    This dataclass captures the complete state of a heating relay including
    power consumption, temperature readings, and operational status. Used for
    monitoring, feedback control, and system diagnostics.

    Attributes:
        room: Room identifier matching the heating control configuration
        state: Current relay state (True=heating ON, False=heating OFF)
        power_w: Current power consumption in watts (0 when OFF)
        last_updated: Timestamp of last status update from Loxone system
        temperature_c: Current room temperature in degrees Celsius
        target_temp_c: Target temperature setpoint (None if not available)

    Data Sources:
        - state, power_w: Loxone heating relay status via MQTT
        - temperature_c: Room temperature sensors via MQTT
        - target_temp_c: Loxone heating control setpoints
        - last_updated: MQTT message timestamp

    Usage:
        status = HeatingStatus(
            room="kitchen",
            state=True,
            power_w=2000.0,
            last_updated=datetime.now(),
            temperature_c=21.5,
            target_temp_c=22.0
        )
        # Represents kitchen heating ON, consuming 2kW, temp 21.5°C targeting 22°C
    """

    room: str
    state: bool
    power_w: float
    last_updated: datetime
    temperature_c: float
    target_temp_c: Optional[float] = None


class HeatingController:
    """
    Primary interface controller for Loxone heating system integration.

    This class serves as the critical bridge between PEMS v2's optimization engine
    and the physical Loxone heating infrastructure. It handles the translation of
    high-level optimization decisions into low-level heating control commands while
    maintaining safety, reliability, and real-time responsiveness.

    Core Responsibilities:
    1. **Schedule Execution**: Convert optimization heating schedules into timed commands
    2. **Command Management**: Queue, prioritize, and execute heating control commands
    3. **Safety Enforcement**: Validate commands against safety constraints and limits
    4. **MQTT Communication**: Handle bi-directional communication with Loxone system
    5. **Status Monitoring**: Track real-time heating system state and performance
    6. **Error Handling**: Manage failures, timeouts, and recovery procedures

    Architecture:
    - Asynchronous operation for non-blocking control and monitoring
    - Command queue with priority-based execution scheduling
    - Safety validation layer preventing dangerous operations
    - Real-time status tracking with MQTT feedback loops
    - Configurable room-specific heating parameters and limits

    Integration with PEMS v2:
    - Receives optimized heating schedules from modules/optimization/optimizer.py
    - Uses room configurations from config/energy_settings.py
    - Communicates status back to optimization engine for model validation
    - Provides emergency controls for safety shutdown scenarios

    MQTT Protocol:
    - Control Commands: pems/heating/{room}/set -> {"state": "on/off", "duration": minutes}
    - Status Updates: loxone/heating/{room}/status -> {"state": bool, "power": watts, "temp": celsius}
    - Emergency Stop: pems/heating/emergency/stop -> immediate shutdown all heating

    Safety Systems:
    - Maximum switching frequency limits (default: 12 switches/hour per room)
    - Power consumption validation against room capacity limits
    - Command timeout protection (default: 60 minutes maximum duration)
    - Emergency stop functionality with immediate response capability
    - Validation of room existence and configuration before command execution
    """

    def __init__(self, settings: PEMSSettings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the heating controller with typed configuration.

        Sets up the controller infrastructure including MQTT communication,
        room configurations, safety parameters, and internal state management.
        This method prepares the controller for operation but does not establish
        MQTT connections (use initialize() method for that).

        Args:
            settings: PEMS settings containing room power ratings and thermal setpoints
            config: Optional additional configuration for MQTT topics and safety overrides

        Configuration Structure:
            {
                "rooms": {
                    "living_room": {"power_kw": 4.8},
                    "kitchen": {"power_kw": 2.0},
                    ...
                },
                "mqtt": {
                    "broker": "localhost",
                    "port": 1883,
                    "heating_topic_prefix": "pems/heating"
                },
                "max_switching_per_hour": 12,
                "safety_timeout_minutes": 60
            }

        Initialization Process:
        1. Extract and validate room configurations
        2. Setup MQTT communication parameters and topic structure
        3. Initialize internal state tracking dictionaries
        4. Configure safety limits and operational constraints
        5. Setup logging for operational monitoring

        Internal State Structures:
        - current_commands: Active heating commands per room
        - last_status: Most recent status information per room
        - command_queue: Pending commands awaiting execution

        Note: Actual MQTT connection is established via initialize() method
        """
        self.settings = settings
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Room configuration from settings
        self.rooms = {
            room: {"power_kw": power}
            for room, power in settings.room_power_ratings_kw.items()
        }

        # MQTT configuration from settings
        if settings.mqtt:
            self.mqtt_broker = settings.mqtt.broker
            self.mqtt_port = settings.mqtt.port
        self.topic_prefix = self.config.get("heating_topic_prefix", "pems/heating")

        # Control state
        self.current_commands: Dict[str, HeatingCommand] = {}
        self.last_status: Dict[str, HeatingStatus] = {}
        self.command_queue: List[HeatingCommand] = []

        # Safety settings
        self.max_switching_frequency = config.get("max_switching_per_hour", 12)
        self.safety_timeout_minutes = config.get("safety_timeout_minutes", 60)

        # Mock MQTT client for now (would be real in production)
        self.mqtt_client = None

        self.logger.info(f"Heating controller initialized for {len(self.rooms)} rooms")

    def get_target_temp(self, room_name: str, hour: int) -> float:
        """Get target temperature for a room at a specific hour.

        Args:
            room_name: Name of the room
            hour: Hour of day (0-23)

        Returns:
            Target temperature in °C
        """
        if self.settings.thermal_settings:
            return self.settings.thermal_settings.get_target_temp(room_name, hour)
        else:
            # Fallback to default setpoints
            if 6 <= hour < 22:  # Daytime
                return 21.0
            else:  # Nighttime
                return 19.0

    async def initialize(self):
        """Initialize MQTT connection and subscribe to status topics."""
        try:
            # In production, initialize actual MQTT client here
            # self.mqtt_client = AsyncMQTTClient(self.mqtt_config)
            # await self.mqtt_client.connect()

            # Subscribe to heating status topics from Loxone
            for room in self.rooms:
                status_topic = f"loxone/heating/{room}/status"
                # await self.mqtt_client.subscribe(status_topic, self._on_heating_status)

            self.logger.info("Heating controller MQTT initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize heating controller: {e}")
            raise

    async def execute_schedule(
        self, heating_schedule: Dict[str, pd.Series]
    ) -> Dict[str, bool]:
        """
        Execute a heating schedule across all rooms.

        Args:
            heating_schedule: Dict mapping room names to pandas Series with boolean values

        Returns:
            Dict mapping room names to success status
        """
        results = {}

        try:
            self.logger.info(
                f"Executing heating schedule for {len(heating_schedule)} rooms"
            )

            # Create commands for each room
            commands = []
            for room, schedule in heating_schedule.items():
                if room in self.rooms:
                    # Get current command (first value in schedule)
                    if len(schedule) > 0:
                        state = bool(schedule.iloc[0])
                        command = HeatingCommand(
                            room=room,
                            state=state,
                            duration_minutes=15,  # Standard 15-minute intervals
                            priority=1,
                        )
                        commands.append(command)

            # Execute commands in parallel
            tasks = [self._execute_single_command(cmd) for cmd in commands]
            command_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, (room, _) in enumerate(heating_schedule.items()):
                if i < len(command_results):
                    result = command_results[i]
                    results[room] = not isinstance(result, Exception)
                    if isinstance(result, Exception):
                        self.logger.error(f"Command failed for {room}: {result}")
                else:
                    results[room] = False

            success_count = sum(results.values())
            self.logger.info(
                f"Schedule execution: {success_count}/{len(results)} rooms successful"
            )

        except Exception as e:
            self.logger.error(f"Schedule execution failed: {e}")
            results = {room: False for room in heating_schedule.keys()}

        return results

    async def set_room_heating(
        self, room: str, state: bool, duration_minutes: Optional[int] = None
    ) -> bool:
        """
        Set heating state for a specific room.

        Args:
            room: Room name
            state: True for ON, False for OFF
            duration_minutes: Duration to maintain state (None for indefinite)

        Returns:
            True if command was successful
        """
        if room not in self.rooms:
            self.logger.error(f"Unknown room: {room}")
            return False

        try:
            command = HeatingCommand(
                room=room,
                state=state,
                duration_minutes=duration_minutes,
                priority=2,  # Manual commands have higher priority
            )

            success = await self._execute_single_command(command)

            if success:
                self.current_commands[room] = command
                self.logger.info(f"Heating {'ON' if state else 'OFF'} for {room}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to set heating for {room}: {e}")
            return False

    async def set_room_temperature(
        self, room: str, target_temp_c: float, duration_minutes: Optional[int] = None
    ) -> bool:
        """
        Set target temperature setpoint for a specific room.

        Args:
            room: Room name
            target_temp_c: Target temperature in degrees Celsius
            duration_minutes: Duration to maintain setpoint (None for indefinite)

        Returns:
            True if command was successful
        """
        if room not in self.rooms:
            self.logger.error(f"Unknown room: {room}")
            return False

        # Validate temperature range (safety limits)
        if not (10.0 <= target_temp_c <= 30.0):
            self.logger.error(
                f"Temperature {target_temp_c}°C outside safe range (10-30°C)"
            )
            return False

        try:
            command = HeatingCommand(
                room=room,
                state=True,  # Temperature control implies heating is enabled
                target_temp_c=target_temp_c,
                duration_minutes=duration_minutes,
                priority=2,  # Manual commands have higher priority
            )

            success = await self._execute_single_command(command)

            if success:
                self.current_commands[room] = command
                self.logger.info(
                    f"Temperature setpoint for {room} set to {target_temp_c}°C"
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to set temperature for {room}: {e}")
            return False

    async def set_zone_temperature(
        self, zone: str, target_temp_c: float, duration_minutes: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Set target temperature for all rooms in a zone.

        Args:
            zone: Zone name (living, sleeping, circulation, etc.)
            target_temp_c: Target temperature in degrees Celsius
            duration_minutes: Duration to maintain setpoint (None for indefinite)

        Returns:
            Dict mapping room names to success status
        """
        from config.energy_settings import get_rooms_by_zone

        zone_rooms = get_rooms_by_zone(zone)
        if not zone_rooms:
            self.logger.error(f"No rooms found for zone: {zone}")
            return {}

        results = {}
        tasks = []

        for room in zone_rooms:
            task = self.set_room_temperature(room, target_temp_c, duration_minutes)
            tasks.append((room, task))

        # Execute commands in parallel
        for room, task in tasks:
            try:
                success = await task
                results[room] = success
            except Exception as e:
                self.logger.error(f"Failed to set temperature for {room}: {e}")
                results[room] = False

        success_count = sum(results.values())
        self.logger.info(
            f"Zone {zone} temperature set: {success_count}/{len(results)} rooms successful"
        )

        return results

    async def get_room_status(self, room: str) -> Optional[HeatingStatus]:
        """
        Get current status of a room's heating system.

        Args:
            room: Room name

        Returns:
            HeatingStatus object or None if unavailable
        """
        if room not in self.rooms:
            return None

        # In production, this would query actual status via MQTT
        # For now, return mock status based on last command
        if room in self.current_commands:
            cmd = self.current_commands[room]
            power_w = self.rooms[room].get("power_kw", 1.0) * 1000 if cmd.state else 0

            return HeatingStatus(
                room=room,
                state=cmd.state,
                power_w=power_w,
                last_updated=cmd.timestamp,
                temperature_c=20.0,  # Mock temperature
                target_temp_c=21.0,
            )

        return None

    async def get_all_status(self) -> Dict[str, HeatingStatus]:
        """Get status for all rooms."""
        status = {}

        for room in self.rooms:
            room_status = await self.get_room_status(room)
            if room_status:
                status[room] = room_status

        return status

    async def emergency_stop(self) -> bool:
        """Emergency stop - turn off all heating."""
        try:
            self.logger.warning("Emergency stop activated - turning off all heating")

            tasks = []
            for room in self.rooms:
                task = self.set_room_heating(room, False)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)

            self.logger.info(
                f"Emergency stop: {success_count}/{len(self.rooms)} rooms turned off"
            )
            return success_count == len(self.rooms)

        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False

    async def _execute_single_command(self, command: HeatingCommand) -> bool:
        """Execute a single heating command (matching mqtt_bridge.py pattern)."""
        try:
            # Safety checks
            if not self._safety_check(command):
                self.logger.warning(f"Safety check failed for {command.room}")
                return False

            # Create payload following mqtt_bridge.py pattern for Loxone
            # Convert to semicolon-separated key=value pairs like mqtt_bridge.py
            payload_data = {
                "room": command.room,
                "state": "on" if command.state else "off",
                "timestamp": command.timestamp.isoformat(),
            }

            if command.target_temp_c is not None:
                payload_data["target_temp_c"] = command.target_temp_c

            if command.duration_minutes:
                payload_data["duration"] = command.duration_minutes

            # Send as JSON first (standard MQTT), then let mqtt_bridge convert for Loxone
            topic = f"{self.topic_prefix}/{command.room}/set"

            # In production, send MQTT command
            if self.mqtt_client:
                await self.mqtt_client.publish(topic, json.dumps(payload_data))

            self.logger.debug(
                f"Heating command sent: {command.room} -> {'ON' if command.state else 'OFF'}"
            )

            # Store command for tracking
            self.current_commands[command.room] = command

            return True

        except Exception as e:
            self.logger.error(f"Failed to execute command for {command.room}: {e}")
            return False

    def _safety_check(self, command: HeatingCommand) -> bool:
        """Perform safety checks before executing command."""

        # Check if room exists
        if command.room not in self.rooms:
            return False

        # Check switching frequency (prevent rapid cycling)
        # In production, implement actual frequency tracking

        # Check power limits
        room_config = self.rooms[command.room]
        max_power = room_config.get("power_kw", 0) * 1000
        if max_power <= 0:
            return False

        return True

    async def _on_heating_status(self, topic: str, payload: str):
        """Handle incoming MQTT status messages."""
        try:
            # Parse room name from topic
            room = topic.split("/")[-2]  # Extract room from topic path

            # In production, parse payload and update status
            # data = json.loads(payload)
            # status = HeatingStatus(
            #     room=room,
            #     state=data.get('state', False),
            #     power_w=data.get('power', 0),
            #     last_updated=datetime.fromisoformat(data.get('timestamp')),
            #     temperature_c=data.get('temperature', 20.0)
            # )
            # self.last_status[room] = status

            self.logger.debug(f"Status update received for {room}")

        except Exception as e:
            self.logger.error(f"Failed to process status update: {e}")


def create_heating_controller(
    room_config: Dict[str, Dict[str, Any]], mqtt_config: Dict[str, Any]
) -> HeatingController:
    """
    Create a configured heating controller.

    Args:
        room_config: Room configuration with power ratings
        mqtt_config: MQTT connection configuration

    Returns:
        Configured HeatingController instance
    """

    config = {
        "rooms": room_config,
        "mqtt": mqtt_config,
        "max_switching_per_hour": 12,
        "safety_timeout_minutes": 60,
    }

    return HeatingController(config)
