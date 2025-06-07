"""
Heating Control Interface for PEMS v2.

Interfaces with Loxone heating system via MQTT for automated control.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass

# Note: In production, these would import from the main project's MQTT client
# from utils.async_mqtt_client import AsyncMQTTClient


@dataclass
class HeatingCommand:
    """Command to control a heating relay."""
    
    room: str
    state: bool  # True = ON, False = OFF
    duration_minutes: Optional[int] = None
    priority: int = 0  # Higher number = higher priority
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class HeatingStatus:
    """Current status of a heating relay."""
    
    room: str
    state: bool
    power_w: float
    last_updated: datetime
    temperature_c: float
    target_temp_c: Optional[float] = None


class HeatingController:
    """
    Controller for Loxone heating system via MQTT.
    
    Provides interface between optimization engine and physical heating relays.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize heating controller.
        
        Args:
            config: Configuration dictionary with MQTT and room settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Room configuration
        self.rooms = config.get('rooms', {})
        
        # MQTT configuration
        self.mqtt_config = config.get('mqtt', {})
        self.topic_prefix = self.mqtt_config.get('heating_topic_prefix', 'loxone/heating')
        
        # Control state
        self.current_commands: Dict[str, HeatingCommand] = {}
        self.last_status: Dict[str, HeatingStatus] = {}
        self.command_queue: List[HeatingCommand] = []
        
        # Safety settings
        self.max_switching_frequency = config.get('max_switching_per_hour', 12)
        self.safety_timeout_minutes = config.get('safety_timeout_minutes', 60)
        
        # Mock MQTT client for now (would be real in production)
        self.mqtt_client = None
        
        self.logger.info(f"Heating controller initialized for {len(self.rooms)} rooms")
    
    async def initialize(self):
        """Initialize MQTT connection and subscribe to status topics."""
        try:
            # In production, initialize actual MQTT client here
            # self.mqtt_client = AsyncMQTTClient(self.mqtt_config)
            # await self.mqtt_client.connect()
            
            # Subscribe to heating status topics
            for room in self.rooms:
                status_topic = f"{self.topic_prefix}/{room}/status"
                # await self.mqtt_client.subscribe(status_topic, self._on_heating_status)
                
            self.logger.info("Heating controller MQTT initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize heating controller: {e}")
            raise
    
    async def execute_schedule(self, heating_schedule: Dict[str, pd.Series]) -> Dict[str, bool]:
        """
        Execute a heating schedule across all rooms.
        
        Args:
            heating_schedule: Dict mapping room names to pandas Series with boolean values
            
        Returns:
            Dict mapping room names to success status
        """
        results = {}
        
        try:
            self.logger.info(f"Executing heating schedule for {len(heating_schedule)} rooms")
            
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
                            priority=1
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
            self.logger.info(f"Schedule execution: {success_count}/{len(results)} rooms successful")
            
        except Exception as e:
            self.logger.error(f"Schedule execution failed: {e}")
            results = {room: False for room in heating_schedule.keys()}
        
        return results
    
    async def set_room_heating(self, room: str, state: bool, duration_minutes: Optional[int] = None) -> bool:
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
                priority=2  # Manual commands have higher priority
            )
            
            success = await self._execute_single_command(command)
            
            if success:
                self.current_commands[room] = command
                self.logger.info(f"Heating {'ON' if state else 'OFF'} for {room}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to set heating for {room}: {e}")
            return False
    
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
            power_w = self.rooms[room].get('power_kw', 1.0) * 1000 if cmd.state else 0
            
            return HeatingStatus(
                room=room,
                state=cmd.state,
                power_w=power_w,
                last_updated=cmd.timestamp,
                temperature_c=20.0,  # Mock temperature
                target_temp_c=21.0
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
            
            self.logger.info(f"Emergency stop: {success_count}/{len(self.rooms)} rooms turned off")
            return success_count == len(self.rooms)
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False
    
    async def _execute_single_command(self, command: HeatingCommand) -> bool:
        """Execute a single heating command."""
        try:
            # Safety checks
            if not self._safety_check(command):
                self.logger.warning(f"Safety check failed for {command.room}")
                return False
            
            # In production, send MQTT command here
            topic = f"{self.topic_prefix}/{command.room}/set"
            payload = {
                "state": "on" if command.state else "off",
                "timestamp": command.timestamp.isoformat()
            }
            
            if command.duration_minutes:
                payload["duration"] = command.duration_minutes
            
            # Mock MQTT publish
            # await self.mqtt_client.publish(topic, json.dumps(payload))
            
            self.logger.debug(f"Heating command sent: {command.room} -> {'ON' if command.state else 'OFF'}")
            
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
        max_power = room_config.get('power_kw', 0) * 1000
        if max_power <= 0:
            return False
        
        return True
    
    async def _on_heating_status(self, topic: str, payload: str):
        """Handle incoming MQTT status messages."""
        try:
            # Parse room name from topic
            room = topic.split('/')[-2]  # Extract room from topic path
            
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


def create_heating_controller(room_config: Dict[str, Dict[str, Any]], 
                            mqtt_config: Dict[str, Any]) -> HeatingController:
    """
    Create a configured heating controller.
    
    Args:
        room_config: Room configuration with power ratings
        mqtt_config: MQTT connection configuration
        
    Returns:
        Configured HeatingController instance
    """
    
    config = {
        'rooms': room_config,
        'mqtt': mqtt_config,
        'max_switching_per_hour': 12,
        'safety_timeout_minutes': 60
    }
    
    return HeatingController(config)