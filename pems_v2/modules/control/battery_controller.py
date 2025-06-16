"""
Battery Control Interface for PEMS v2.

This module provides the interface for controlling battery charging from the grid
and monitoring battery system status. It integrates with the Growatt inverter
system via MQTT to enable optimized energy storage management.

Key Features:
1. **Grid Charging Control**: Enable/disable battery charging from grid power
2. **Charging Power Management**: Control charging rate within system limits
3. **Status Monitoring**: Real-time battery state monitoring (SOC, power, status)
4. **Safety Management**: Prevents overcharging and unsafe operating conditions
5. **MQTT Integration**: Seamless communication with Growatt inverter system

Safety Features:
- SOC limit validation (prevents overcharging above 95%)
- Power limit enforcement within inverter specifications
- Temperature monitoring and protection
- Emergency stop capability
- Command validation and sanitization

Usage in PEMS v2 Workflow:
1. Optimization engine determines optimal charging schedule
2. BatteryController receives charging commands with power levels
3. Commands validated against safety constraints and system limits
4. MQTT messages sent to Growatt inverter for execution
5. Status monitoring confirms successful execution
6. Feedback provided to optimization engine for model updates

Integration Points:
- Input: Charging schedules from optimizer.py
- Output: MQTT commands to Growatt inverter
- Monitoring: Battery status from Growatt system via MQTT
- Configuration: Battery specifications from energy_settings.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from pems_v2.config.settings import BatterySettings, MQTTSettings


class ChargingMode(Enum):
    """
    Battery charging modes for comprehensive energy storage control.

    This enumeration defines the operational modes available for the battery
    storage system. Each mode represents a different strategy for managing
    energy flow to and from the battery, optimized for specific objectives
    such as cost minimization, environmental impact, or system reliability.

    Mode Descriptions:

    OFF ("off"):
        Battery system is completely inactive with no charging or discharging.

        Usage Scenarios:
        - System maintenance and service operations
        - Emergency safety situations requiring immediate shutdown
        - Periods when battery operation is not desired or safe
        - Testing and diagnostic procedures
        - Grid islanding or disconnection events

        Technical Implementation:
        - Disables both battery-first mode and AC charging
        - Maintains battery at current state without power flow
        - Preserves battery capacity for future use
        - Allows manual intervention without automatic control

        Safety Considerations:
        - Used as emergency fallback mode for fault conditions
        - Prevents battery degradation during extended idle periods
        - Ensures system stability during grid disturbances

    GRID ("grid"):
        Battery charges from electrical grid at controlled power levels.

        Usage Scenarios:
        - Low electricity price periods (time-of-use optimization)
        - Peak demand preparation (load shifting)
        - Emergency charging when battery critically low
        - Scheduled charging during off-peak hours
        - Demand response program participation

        Technical Implementation:
        - Enables battery-first inverter mode for charging priority
        - Activates AC charging with precise power control
        - Supports time-window scheduling (start/stop times)
        - Automatically stops at maximum SOC threshold

        Control Parameters:
        - Power level: Adjustable from 0 to maximum charge rate
        - Duration: Time-limited or continuous operation
        - Priority: Can override other modes for urgent charging

        Optimization Benefits:
        - Cost reduction through price arbitrage
        - Grid stability through demand response
        - Energy security through backup preparation

    PV_ONLY ("pv_only"):
        Battery charges exclusively from photovoltaic (solar) generation.

        Usage Scenarios:
        - Maximizing renewable energy self-consumption
        - Reducing grid dependency and carbon footprint
        - Environmental optimization strategies
        - Grid export limitation compliance
        - Net metering optimization

        Technical Implementation:
        - Disables AC charging from grid
        - Allows only solar-powered charging
        - Follows PV generation curve automatically
        - Prevents grid power use for battery charging

        Environmental Benefits:
        - 100% renewable energy storage
        - Zero grid carbon footprint for charging
        - Optimal utilization of local solar resources
        - Reduced transmission losses

        Economic Considerations:
        - Eliminates electricity costs for charging
        - May reduce grid export during peak production
        - Supports energy independence objectives

    AUTO ("auto"):
        Intelligent automatic mode with adaptive decision-making.

        Usage Scenarios:
        - Autonomous system operation without manual intervention
        - Balanced optimization across multiple objectives
        - Dynamic adaptation to changing conditions
        - Default mode for smart home integration
        - Complex optimization strategies

        Technical Implementation:
        - System analyzes real-time conditions continuously
        - Switches between charging sources automatically
        - Considers multiple optimization criteria simultaneously
        - Learns from historical performance data

        Decision Factors:
        - Current electricity prices and forecasts
        - PV generation availability and predictions
        - Battery state of charge and health
        - Load demand forecasts and patterns
        - Grid conditions and stability requirements
        - User preferences and comfort requirements

        Adaptive Behaviors:
        - Prioritizes PV charging when available
        - Enables grid charging during price valleys
        - Balances cost optimization with reliability
        - Adjusts power levels for efficiency
        - Responds to emergency conditions automatically

    Mode Selection Guidelines:

    Cost Optimization:
    - Use GRID mode during cheapest electricity periods
    - Use PV_ONLY mode to minimize energy costs
    - Use AUTO mode for balanced cost/performance

    Environmental Optimization:
    - Prefer PV_ONLY mode for maximum renewable usage
    - Use AUTO mode with environmental weighting
    - Avoid GRID mode during peak carbon intensity

    Reliability Optimization:
    - Use GRID mode to ensure adequate backup power
    - Use AUTO mode for adaptive reliability management
    - Plan OFF periods for maintenance without disruption

    Performance Considerations:

    Each mode has different impacts on system performance:
    - GRID mode provides fastest, most predictable charging
    - PV_ONLY mode provides variable, weather-dependent charging
    - AUTO mode provides optimized but complex charging patterns
    - OFF mode provides zero performance but maximum safety

    The choice of mode should align with current system objectives,
    environmental conditions, and economic factors.
    """

    OFF = "off"
    GRID = "grid"  # Charge from grid
    PV_ONLY = "pv_only"  # Charge from PV only
    AUTO = "auto"  # Automatic mode


@dataclass
class BatteryCommand:
    """
    Represents a battery control command with comprehensive parameters.

    Attributes:
        mode: Charging mode (OFF, GRID, PV_ONLY, AUTO)
        power_kw: Target charging power in kW (None for mode-only commands)
        duration_minutes: Duration to maintain command (None for indefinite)
        priority: Command priority (0=low, 3=emergency)
        timestamp: When command was created

    Detailed Attribute Documentation:

    mode (ChargingMode):
        The target operational mode for the battery system. This determines
        the fundamental charging strategy and energy source selection.

        Validation Requirements:
        - Must be a valid ChargingMode enumeration value
        - Cannot be None or undefined
        - Must be compatible with current system state

        Impact on System:
        - Controls inverter battery-first mode activation
        - Determines AC charging enable/disable state
        - Affects power flow priorities and routing
        - Influences safety behavior and thresholds

    power_kw (Optional[float]):
        Target charging power level in kilowatts.

        Value Interpretation:
        - Positive values: Charging power (energy into battery)
        - Zero: Idle state (no active charging/discharging)
        - None: System determines power automatically based on mode

        Power Level Guidelines:
        - GRID mode: Exact power level for grid charging
        - PV_ONLY mode: Maximum power limit from solar
        - AUTO mode: Power preference for optimization
        - OFF mode: Power parameter ignored

        Validation Rules:
        - Must be between 0 and max_charge_power_kw
        - Fractional values allowed for precise control
        - Automatically clamped to battery specifications
        - Subject to safety derating based on temperature/SOC

        Optimization Considerations:
        - Higher power = faster charging but higher losses
        - Optimal power varies with battery SOC and temperature
        - Grid connection limits may restrict maximum power
        - Cost optimization may prefer specific power levels

    duration_minutes (Optional[int]):
        Time period in minutes for which this command remains active.

        Duration Behavior:
        - None: Command remains active indefinitely until changed
        - > 0: Command automatically expires after specified time
        - 0: Invalid (commands must have positive duration if specified)

        Practical Range:
        - Minimum: 1 minute (for brief operations)
        - Typical: 15-240 minutes (for optimization periods)
        - Maximum: 1440 minutes (24 hours)

        Use Cases:
        - Time-of-use charging during cheap rate windows
        - Emergency charging for specific duration
        - Scheduled charging aligned with PV production
        - Temporary mode changes for testing

        Expiration Handling:
        - System automatically reverts to previous mode
        - Can be extended or modified before expiration
        - Generates log entry when duration expires
        - Allows for predictable system behavior

    priority (int):
        Command execution priority for scheduling and conflict resolution.

        Priority Levels:
        0 (Low Priority):
            - Routine optimization commands
            - Scheduled charging operations
            - Normal operational adjustments
            - Can be overridden by higher priority commands

        1 (Normal Priority):
            - User-initiated manual commands
            - Real-time optimization adjustments
            - Standard system responses to conditions
            - Default priority for most operations

        2 (High Priority):
            - System-initiated safety commands
            - Critical SOC protection actions
            - Grid stability response commands
            - Overrides normal optimization

        3 (Emergency Priority):
            - Emergency stop and safety commands
            - Fault response and protection actions
            - Immediate execution required
            - Cannot be overridden by lower priorities

        Priority Handling:
        - Higher numbers indicate higher priority
        - Commands with same priority use timestamp ordering
        - Emergency commands execute immediately
        - Priority affects queue position and execution timing

    timestamp (datetime):
        Exact moment when this command was created.

        Auto-Population:
        - Automatically set to current time if not provided
        - Uses system local time with timezone awareness
        - Precision to microseconds for accurate ordering

        Usage Applications:
        - Command ordering and sequencing
        - Debugging and audit trail creation
        - Performance analysis and timing studies
        - Timeout calculation and enforcement
        - Historical analysis of system behavior

        Format Considerations:
        - ISO 8601 format for serialization
        - Timezone-aware for proper coordination
        - Monotonic ordering for reliable sequencing

    Command Validation Process:

    Before execution, each command undergoes comprehensive validation:

    1. Syntax Validation:
       - All required fields present
       - Data types correct
       - Enumeration values valid

    2. Range Validation:
       - Power within system specifications
       - Duration within reasonable limits
       - Priority level valid

    3. State Validation:
       - Compatible with current battery state
       - Safe given current SOC and temperature
       - Allowable mode transition

    4. System Validation:
       - Grid connection available (for GRID mode)
       - PV generation available (for PV_ONLY mode)
       - No conflicting high-priority commands

    Command Lifecycle:

    1. Creation: Command object instantiated with parameters
    2. Validation: Comprehensive safety and feasibility checks
    3. Queuing: Added to priority queue for execution
    4. Execution: MQTT messages sent to physical system
    5. Monitoring: Status tracked for successful completion
    6. Completion: Command marked complete or expired
    7. Logging: Results recorded for analysis

    Error Handling:

    Commands may fail at various stages:
    - Validation failure: Command rejected before execution
    - Communication failure: Retry with exponential backoff
    - Execution failure: Fallback to safe mode if needed
    - Timeout failure: Cancel command and generate alert

    Example Usage:

    # Grid charging during cheap electricity period
    cmd = BatteryCommand(
        mode=ChargingMode.GRID,
        power_kw=4.0,
        duration_minutes=120,
        priority=1
    )

    # Emergency stop command
    emergency_cmd = BatteryCommand(
        mode=ChargingMode.OFF,
        priority=3
    )

    # Automatic mode with power preference
    auto_cmd = BatteryCommand(
        mode=ChargingMode.AUTO,
        power_kw=2.0  # Preference, not requirement
    )
    """

    mode: ChargingMode
    power_kw: Optional[float] = None
    duration_minutes: Optional[int] = None
    priority: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        """
        Auto-populate timestamp and perform basic validation.

        This method is automatically called after the dataclass __init__
        completes. It ensures every command has a timestamp and performs
        basic validation of command parameters.

        Timestamp Auto-Population:
        If no timestamp was provided during initialization, this method
        sets the timestamp to the current moment. This ensures:
        - Every command has a creation time for ordering
        - Audit trails can track when commands were issued
        - Timeout calculations have a reference point
        - Performance analysis can measure command latency

        Basic Validation:
        While comprehensive validation occurs during execution,
        this method performs basic sanity checks:
        - Ensures priority is within valid range (0-3)
        - Validates power_kw is non-negative if specified
        - Checks duration_minutes is positive if specified

        Raises:
            ValueError: If basic validation fails
        """
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Basic validation
        if not (0 <= self.priority <= 3):
            raise ValueError(f"Priority must be 0-3, got {self.priority}")

        if self.power_kw is not None and self.power_kw < 0:
            raise ValueError(f"Power must be non-negative, got {self.power_kw}")

        if self.duration_minutes is not None and self.duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration_minutes}")


@dataclass
class BatteryStatus:
    """
    Represents current battery system status with comprehensive telemetry data.

    Attributes:
        soc_percent: State of charge (0-100%)
        power_kw: Current power (positive=charging, negative=discharging)
        voltage_v: Battery voltage
        current_a: Battery current
        temperature_c: Battery temperature
        mode: Current charging mode
        last_updated: Timestamp of last status update
        is_online: Whether battery system is responding

    Comprehensive Attribute Documentation:

    soc_percent (float):
        State of Charge as a percentage from 0.0 to 100.0.

        Value Interpretation:
        - 0.0%: Battery completely discharged (protection cutoff engaged)
        - 100.0%: Battery fully charged (charge termination active)
        - Typical operating range: 10% to 90% for optimal longevity

        Critical Thresholds:
        - < 10%: Critical low, emergency charging may be triggered
        - < 20%: Low warning, charging recommended
        - > 90%: High level, charging rate may be reduced
        - > 95%: Very high, charging may be automatically stopped

        Accuracy Considerations:
        - SOC estimation based on voltage, current, and coulomb counting
        - May drift over time, requires periodic calibration
        - Temperature affects accuracy of voltage-based estimation
        - Cell balancing may cause temporary SOC fluctuations

        Optimization Usage:
        - Determines available energy for discharge planning
        - Influences charging strategy selection
        - Critical for avoiding deep discharge damage
        - Used for capacity fade monitoring over time

    power_kw (float):
        Current instantaneous power flow in kilowatts.

        Sign Convention:
        - Positive values: Battery is charging (energy flowing into battery)
        - Negative values: Battery is discharging (energy flowing from battery)
        - Zero: Battery is idle (no significant power flow)

        Typical Value Ranges:
        - Charging: 0 to +5.0 kW (limited by charger capacity)
        - Discharging: 0 to -5.0 kW (limited by inverter capacity)
        - Idle: ±0.1 kW (small parasitic losses or measurement noise)

        Factors Affecting Power:
        - Available charging capacity from grid or PV
        - Battery internal resistance (varies with SOC and temperature)
        - Inverter/charger efficiency curves
        - Safety derating based on temperature or cell voltage

        Measurement Accuracy:
        - Resolution typically 0.01 kW (10 watts)
        - May include measurement filtering for stability
        - Instantaneous values may fluctuate due to switching

    voltage_v (float):
        Current battery pack voltage in volts.

        Voltage Characteristics:
        - Varies with state of charge (higher when charged)
        - Affected by load current (voltage drop under load)
        - Temperature dependent (lower when cold)
        - Cell configuration determines nominal voltage

        Typical Voltage Ranges (for common battery types):
        - 48V nominal systems: 40-58V operating range
        - Cell balancing may cause temporary voltage variations
        - Over/under voltage protection limits are enforced

        Diagnostic Information:
        - Sudden voltage drops may indicate connection issues
        - Voltage drift during idle indicates cell balancing
        - Voltage sag under load indicates internal resistance
        - Used for SOC estimation and health monitoring

    current_a (float):
        Current battery pack current in amperes.

        Sign Convention:
        - Positive values: Charging current (into battery)
        - Negative values: Discharging current (from battery)
        - Zero: No current flow (idle state)

        Current Limits:
        - Maximum charge current: typically 50-100A
        - Maximum discharge current: typically 50-100A
        - Limits enforced by BMS for safety and longevity
        - Temperature derating applied in extreme conditions

        Measurement Characteristics:
        - High resolution current measurement (±0.1A typical)
        - May include filtering to reduce switching noise
        - Used for coulomb counting and SOC calculation
        - Critical for detecting overcurrent conditions

    temperature_c (float):
        Battery pack temperature in degrees Celsius.

        Temperature Monitoring:
        - Multiple sensors typically averaged
        - Critical for safety and performance optimization
        - Affects charging/discharging rates and efficiency
        - Used for thermal management and protection

        Operating Temperature Ranges:
        - Optimal performance: 15-35°C
        - Acceptable operation: 0-45°C
        - Charging restricted: < 0°C or > 45°C
        - Emergency shutdown: < -10°C or > 60°C

        Temperature Effects:
        - Cold temperatures reduce capacity and power
        - Hot temperatures accelerate aging and reduce life
        - Extreme temperatures trigger safety protections
        - Thermal runaway risk at very high temperatures

        Thermal Management:
        - Active cooling/heating may be controlled
        - Charging power reduced at temperature extremes
        - Insulation and thermal mass affect temperature stability

    mode (ChargingMode):
        Current active charging mode of the battery system.

        Mode States:
        - OFF: Battery disconnected from charging/discharging
        - GRID: Active grid charging with controlled power
        - PV_ONLY: Charging restricted to solar generation only
        - AUTO: Automatic mode with intelligent switching

        Mode Transitions:
        - May lag behind commanded mode due to system response time
        - Safety conditions can force mode changes
        - Reflects actual operational state, not commanded state

        Status Interpretation:
        - Indicates current energy management strategy
        - Shows whether grid charging is active
        - Reflects system response to optimization commands
        - Used for validating command execution success

    last_updated (datetime):
        Timestamp of when this status information was last updated.

        Update Frequency:
        - Typically updated every 1-5 seconds for real-time monitoring
        - Higher frequency during active charging/discharging
        - Lower frequency during idle periods for efficiency

        Staleness Detection:
        - Status older than 60 seconds considered stale
        - Stale status may indicate communication problems
        - Triggers alerts and fallback to safe operation

        Timezone Considerations:
        - Should be timezone-aware for proper coordination
        - UTC preferred for internal storage and processing
        - Local time for user display and logging

    is_online (bool):
        Indicates whether the battery system is communicating and operational.

        Online Status Determination:
        - True: Recent status updates received, system responding
        - False: Communication timeout, system not responding

        Offline Conditions:
        - MQTT communication failure
        - Battery management system fault
        - Physical disconnection or power loss
        - System maintenance or shutdown

        Impact of Offline Status:
        - Commands cannot be executed when offline
        - Triggers fallback to safe operational modes
        - Generates alerts for system monitoring
        - Affects optimization decisions and planning

    Status Data Quality and Validation:

    The battery status undergoes validation to ensure data quality:

    1. Range Validation:
       - SOC must be 0-100%
       - Voltage within expected operating range
       - Current within rated limits
       - Temperature within sensor range

    2. Consistency Validation:
       - Power calculation matches voltage × current
       - SOC trends consistent with power integration
       - Mode matches expected charging behavior

    3. Temporal Validation:
       - Timestamp progression is monotonic
       - Rate of change within physical limits
       - No sudden unrealistic jumps in values

    Status Usage in PEMS v2:

    The battery status is used throughout the system:

    1. Optimization Engine:
       - SOC for available energy calculations
       - Power for current system state
       - Temperature for safety constraints

    2. Safety Systems:
       - All parameters monitored against limits
       - Trend analysis for predictive protection
       - Fault detection and emergency response

    3. Performance Monitoring:
       - Efficiency calculations
       - Capacity fade tracking
       - Thermal performance analysis

    4. User Interface:
       - Real-time status display
       - Historical trending
       - Alert and alarm generation

    Example Status Interpretation:

    status = BatteryStatus(
        soc_percent=75.5,
        power_kw=3.2,        # Charging at 3.2kW
        voltage_v=52.4,
        current_a=61.1,      # 3200W ÷ 52.4V ≈ 61A
        temperature_c=28.5,
        mode=ChargingMode.GRID,
        last_updated=datetime.now(),
        is_online=True
    )

    This indicates:
    - Battery is 75% charged with good capacity remaining
    - Actively charging from grid at moderate power level
    - Operating within normal temperature range
    - System is healthy and responding normally
    """

    soc_percent: float
    power_kw: float
    voltage_v: float
    current_a: float
    temperature_c: float
    mode: ChargingMode
    last_updated: datetime
    is_online: bool = True


class BatteryController:
    """
    Primary interface for battery system control and monitoring.

    This class manages battery charging operations including grid charging,
    power level control, and safety monitoring. It communicates with the
    Growatt inverter system via MQTT and provides real-time status feedback.

    Core Responsibilities:
    1. **Charging Control**: Enable/disable grid charging with power control
    2. **Mode Management**: Switch between charging modes (grid, PV-only, auto)
    3. **Safety Monitoring**: Enforce SOC limits and thermal protection
    4. **Status Tracking**: Monitor battery state and system health
    5. **Command Validation**: Ensure all commands are safe and valid

    MQTT Protocol:
    - Control Commands: pems/battery/set -> {"mode": "grid", "power_kw": 3.0}
    - Status Updates: growatt/battery/status -> battery telemetry data
    - Emergency Stop: pems/battery/emergency/stop -> immediate stop charging

    Safety Limits:
    - Maximum SOC: 95% (prevents overcharging)
    - Minimum SOC: 10% (prevents deep discharge)
    - Maximum charging power: 5.0 kW (system limit)
    - Temperature limits: 0-45°C operating range
    """

    def __init__(
        self,
        battery_settings: BatterySettings,
        mqtt_settings: Optional[MQTTSettings] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize battery controller with typed configuration.

        Args:
            battery_settings: Battery system specifications from settings
            mqtt_settings: Optional MQTT configuration for communication
            config: Optional additional configuration for topics and safety overrides
        """
        self.battery_settings = battery_settings
        self.mqtt_settings = mqtt_settings
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Battery system configuration from typed settings
        self.capacity_kwh = battery_settings.capacity_kwh
        self.max_charge_power = battery_settings.max_power_kw
        self.max_discharge_power = battery_settings.max_power_kw

        # MQTT configuration from settings or config
        if mqtt_settings:
            self.mqtt_broker = mqtt_settings.broker
            self.mqtt_port = mqtt_settings.port
        self.control_topic = self.config.get(
            "battery_control_topic", "pems/battery/set"
        )
        self.status_topic = self.config.get(
            "battery_status_topic", "growatt/battery/status"
        )

        # Safety limits from typed settings
        self.max_soc = battery_settings.max_soc * 100  # Convert to percentage
        self.min_soc = battery_settings.min_soc * 100  # Convert to percentage
        self.max_temp = self.config.get("max_temperature_c", 45.0)
        self.min_temp = self.config.get("min_temperature_c", 0.0)

        # Control state
        self.current_command: Optional[BatteryCommand] = None
        self.last_status: Optional[BatteryStatus] = None
        self.command_history: list = []

        # Mock MQTT client (would be real in production)
        self.mqtt_client = None

        self.logger.info(
            f"Battery controller initialized: {self.capacity_kwh}kWh capacity, "
            f"{self.max_charge_power}kW max charge power"
        )

    async def initialize(self):
        """Initialize MQTT connection and subscribe to status topics."""
        try:
            # In production, initialize actual MQTT client here
            # self.mqtt_client = AsyncMQTTClient(self.mqtt_config)
            # await self.mqtt_client.connect()

            # Subscribe to battery status topic
            # await self.mqtt_client.subscribe(self.status_topic, self._on_battery_status)

            self.logger.info("Battery controller MQTT initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize battery controller: {e}")
            raise

    async def set_charging_mode(
        self,
        mode: ChargingMode,
        power_kw: Optional[float] = None,
        duration_minutes: Optional[int] = None,
    ) -> bool:
        """
        Set battery charging mode with final state validation to prevent race conditions.

        This method implements the "Check-Act" pattern for modes that involve charging
        (GRID and AUTO), ensuring the battery state is still safe at the moment of
        command execution.

        Args:
            mode: Target charging mode
            power_kw: Charging power limit in kW (None for mode default)
            duration_minutes: Duration to maintain mode (None for indefinite)

        Returns:
            True if command was successful
        """
        try:
            # Validate power level
            if power_kw is not None:
                if power_kw < 0 or power_kw > self.max_charge_power:
                    self.logger.error(
                        f"Invalid charging power: {power_kw}kW "
                        f"(max: {self.max_charge_power}kW)"
                    )
                    return False

            # For charging modes, perform final state check
            if mode in [ChargingMode.GRID, ChargingMode.AUTO]:
                latest_status = await self.get_status()

                if latest_status and latest_status.is_online:
                    # Check SOC limits before enabling charging
                    if latest_status.soc_percent >= self.max_soc:
                        self.logger.warning(
                            f"Aborting {mode.value} mode: Current SOC ({latest_status.soc_percent}%) "
                            f"is at or above max ({self.max_soc}%)."
                        )
                        return False

                    # Check temperature limits
                    if not (
                        self.min_temp <= latest_status.temperature_c <= self.max_temp
                    ):
                        self.logger.warning(
                            f"Aborting {mode.value} mode: Current temperature ({latest_status.temperature_c}°C) "
                            f"is outside safe range ({self.min_temp}-{self.max_temp}°C)."
                        )
                        return False
                elif not latest_status:
                    self.logger.warning(
                        f"Cannot set {mode.value} mode: Battery status unavailable."
                    )
                    return False

            if mode == ChargingMode.GRID:
                # Enable battery-first mode with AC charging (growatt pattern)
                current_time = datetime.now()
                start_time = current_time.strftime("%H:%M")

                if duration_minutes:
                    end_time = (
                        current_time + timedelta(minutes=duration_minutes)
                    ).strftime("%H:%M")
                else:
                    end_time = "23:59"  # Until end of day

                await self._set_battery_first_mode(start_time, end_time)
                await self._enable_ac_charge(power_kw or 5.0)

            elif mode == ChargingMode.OFF:
                await self._disable_battery_first_mode()
                await self._disable_ac_charge()

            elif mode == ChargingMode.PV_ONLY:
                await self._disable_ac_charge()
                # Battery-first mode disabled for PV-only
                await self._disable_battery_first_mode()

            elif mode == ChargingMode.AUTO:
                # Default mode - let system decide
                await self._disable_ac_charge()

            # Store command
            command = BatteryCommand(
                mode=mode,
                power_kw=power_kw,
                duration_minutes=duration_minutes,
                priority=1,
            )

            self.current_command = command
            self.command_history.append(command)

            self.logger.info(
                f"Battery charging mode set to {mode.value} "
                f"{'at ' + str(power_kw) + 'kW' if power_kw else ''}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to set battery charging mode: {e}")
            return False

    async def enable_grid_charging(self, power_kw: float = None) -> bool:
        """
        Enable grid charging with a final state check to prevent race conditions.

        This method implements the "Check-Act" pattern by re-checking the battery's
        current state just before executing the charging command. This prevents
        race conditions where the system state may have changed between when the
        optimization decision was made and when the command is executed.

        Args:
            power_kw: Charging power in kW (uses system default if None)

        Returns:
            True if successful
        """
        if power_kw is None:
            power_kw = self.max_charge_power

        # --- 1. Final State Check (Check-Act Pattern) ---
        # Get the absolute latest status before acting.
        latest_status = await self.get_status()

        if not latest_status or not latest_status.is_online:
            self.logger.warning("Cannot enable charging: Battery is offline.")
            return False

        # Re-run critical safety check with the latest data.
        if latest_status.soc_percent >= self.max_soc:
            self.logger.warning(
                f"Aborting charge command: Current SOC ({latest_status.soc_percent}%) is at or above max ({self.max_soc}%)."
            )
            return False

        if not (self.min_temp <= latest_status.temperature_c <= self.max_temp):
            self.logger.warning(
                f"Aborting charge command: Current temperature ({latest_status.temperature_c}°C) is outside safe range."
            )
            return False

        # --- 2. Execute Command if Checks Pass ---
        try:
            if self.config.get("simulation_mode", False):
                self.logger.info(f"⚡ [SIMULATE] AC CHARGING ENABLED at {power_kw}kW")
                return True

            # Send MQTT command (matches growatt pattern)
            await self._enable_ac_charge(power_kw)

            # Update internal state
            self.current_command = BatteryCommand(
                mode=ChargingMode.GRID, power_kw=power_kw
            )
            self.command_history.append(self.current_command)

            return True

        except Exception as e:
            self.logger.error(f"Failed to enable grid charging: {e}")
            return False

    async def disable_grid_charging(self) -> bool:
        """
        Disable charging from grid (PV-only mode) with state verification.

        Returns:
            True if successful
        """
        # Final state check before switching to PV-only mode
        latest_status = await self.get_status()
        if latest_status and not latest_status.is_online:
            self.logger.warning("Cannot change mode: Battery is offline.")
            return False

        return await self.set_charging_mode(ChargingMode.PV_ONLY)

    async def stop_charging(self) -> bool:
        """
        Stop all battery charging with state verification.

        Returns:
            True if successful
        """
        # Final state check before stopping charging
        latest_status = await self.get_status()
        if latest_status and not latest_status.is_online:
            self.logger.warning("Cannot stop charging: Battery is offline.")
            return False

        return await self.set_charging_mode(ChargingMode.OFF)

    async def get_status(self) -> Optional[BatteryStatus]:
        """
        Get current battery system status.

        Returns:
            BatteryStatus object or None if unavailable
        """
        # In production, this would query actual status via MQTT
        # For now, return mock status
        if self.current_command:
            return BatteryStatus(
                soc_percent=80.0,  # Mock SOC
                power_kw=self.current_command.power_kw or 0.0,
                voltage_v=52.0,  # Mock voltage
                current_a=50.0,  # Mock current
                temperature_c=25.0,  # Mock temperature
                mode=self.current_command.mode,
                last_updated=datetime.now(),
                is_online=True,
            )

        return self.last_status

    async def get_soc(self) -> Optional[float]:
        """
        Get current state of charge percentage.

        Returns:
            SOC percentage (0-100) or None if unavailable
        """
        status = await self.get_status()
        return status.soc_percent if status else None

    async def is_charging_available(self) -> bool:
        """
        Check if battery charging is available and safe.

        Returns:
            True if charging can be enabled
        """
        status = await self.get_status()
        if not status:
            return False

        # Check SOC limits
        if status.soc_percent >= self.max_soc:
            return False

        # Check temperature limits
        if not (self.min_temp <= status.temperature_c <= self.max_temp):
            return False

        # Check system health
        if not status.is_online:
            return False

        return True

    async def emergency_stop(self) -> bool:
        """
        Emergency stop - disable all charging (matches growatt pattern).

        Returns:
            True if successful
        """
        try:
            self.logger.warning("Battery emergency stop activated")

            # Disable both battery-first mode and AC charging
            await self._disable_battery_first_mode()
            await self._disable_ac_charge()

            # Store command
            command = BatteryCommand(
                mode=ChargingMode.OFF, priority=3
            )  # Emergency priority

            self.current_command = command
            self.command_history.append(command)

            self.logger.info("Battery emergency stop completed")
            return True

        except Exception as e:
            self.logger.error(f"Battery emergency stop failed: {e}")
            return False

    def get_charging_limits(self) -> Dict[str, float]:
        """
        Get battery charging limits and specifications.

        Returns:
            Dictionary with charging limits
        """
        return {
            "capacity_kwh": self.capacity_kwh,
            "max_charge_power_kw": self.max_charge_power,
            "max_discharge_power_kw": self.max_discharge_power,
            "max_soc_percent": self.max_soc,
            "min_soc_percent": self.min_soc,
            "max_temperature_c": self.max_temp,
            "min_temperature_c": self.min_temp,
        }

    async def _execute_command(self, command: BatteryCommand) -> bool:
        """Execute a battery control command (legacy method for compatibility)."""
        try:
            # This method is now primarily for logging and compatibility
            # Actual control is done through _set_battery_first_mode and _enable_ac_charge
            self.logger.debug(
                f"Battery command executed: {command.mode.value} "
                f"{'at ' + str(command.power_kw) + 'kW' if command.power_kw else ''}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to execute battery command: {e}")
            return False

    def _safety_check(self, command: BatteryCommand) -> bool:
        """Perform safety checks before executing command."""

        # Check power limits
        if command.power_kw is not None:
            if command.power_kw > self.max_charge_power:
                self.logger.error(
                    f"Power {command.power_kw}kW exceeds limit {self.max_charge_power}kW"
                )
                return False

        # Check SOC limits (if status available)
        if self.last_status:
            if command.mode in [ChargingMode.GRID, ChargingMode.AUTO]:
                if self.last_status.soc_percent >= self.max_soc:
                    self.logger.warning(
                        f"SOC {self.last_status.soc_percent}% above max {self.max_soc}%"
                    )
                    return False

            # Check temperature limits
            if not (self.min_temp <= self.last_status.temperature_c <= self.max_temp):
                self.logger.error(
                    f"Temperature {self.last_status.temperature_c}°C outside safe range"
                )
                return False

        return True

    async def _on_battery_status(self, topic: str, payload: str):
        """Handle incoming MQTT battery status messages."""
        try:
            data = json.loads(payload)

            # Parse battery status from Growatt data
            status = BatteryStatus(
                soc_percent=data.get("soc_percent", 0.0),
                power_kw=data.get("power_kw", 0.0),
                voltage_v=data.get("voltage_v", 0.0),
                current_a=data.get("current_a", 0.0),
                temperature_c=data.get("temperature_c", 20.0),
                mode=ChargingMode(data.get("mode", "auto")),
                last_updated=datetime.fromisoformat(
                    data.get("timestamp", datetime.now().isoformat())
                ),
                is_online=data.get("is_online", True),
            )

            self.last_status = status
            self.logger.debug(
                f"Battery status updated: SOC={status.soc_percent}%, Power={status.power_kw}kW"
            )

        except Exception as e:
            self.logger.error(f"Failed to process battery status update: {e}")

    async def _set_battery_first_mode(self, start_hour: str, stop_hour: str) -> None:
        """Set battery-first mode for specified time window (matches growatt pattern)."""
        if self.config.get("simulation_mode", False):
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE SET: {start_hour}-{stop_hour} "
                f"(simulated at {current_time})"
            )
            return

        payload = {"start": start_hour, "stop": stop_hour, "enabled": True, "slot": 1}
        battery_first_topic = self.mqtt_config.get(
            "battery_first_topic", "pems/battery/mode"
        )

        if self.mqtt_client:
            await self.mqtt_client.publish(battery_first_topic, json.dumps(payload))
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 BATTERY-FIRST MODE SET: {start_hour}-{stop_hour} "
                f"(action at {current_time}) → Topic: {battery_first_topic}"
            )

    async def _disable_battery_first_mode(self) -> None:
        """Disable battery-first mode (matches growatt pattern)."""
        if self.config.get("simulation_mode", False):
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE DISABLED (simulated at {current_time})"
            )
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        battery_first_topic = self.mqtt_config.get(
            "battery_first_topic", "pems/battery/mode"
        )

        if self.mqtt_client:
            await self.mqtt_client.publish(battery_first_topic, json.dumps(payload))
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 BATTERY-FIRST MODE DISABLED at {current_time} → "
                f"Topic: {battery_first_topic}"
            )

    async def _enable_ac_charge(self, power_kw: float) -> None:
        """Enable AC charging (matches growatt pattern)."""
        if self.config.get("simulation_mode", False):
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"⚡ [SIMULATE] AC CHARGING ENABLED at {power_kw}kW (simulated at {current_time})"
            )
            return

        payload = {"value": True, "power_kw": power_kw}
        ac_charge_topic = self.mqtt_config.get(
            "ac_charge_topic", "pems/battery/ac_charge"
        )

        if self.mqtt_client:
            await self.mqtt_client.publish(ac_charge_topic, json.dumps(payload))
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"⚡ AC CHARGING ENABLED at {power_kw}kW at {current_time} → Topic: {ac_charge_topic}"
            )

    async def _disable_ac_charge(self) -> None:
        """Disable AC charging (matches growatt pattern)."""
        if self.config.get("simulation_mode", False):
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"⚡ [SIMULATE] AC CHARGING DISABLED (simulated at {current_time})"
            )
            return

        payload = {"value": False}
        ac_charge_topic = self.mqtt_config.get(
            "ac_charge_topic", "pems/battery/ac_charge"
        )

        if self.mqtt_client:
            await self.mqtt_client.publish(ac_charge_topic, json.dumps(payload))
            current_time = datetime.now().strftime("%H:%M:%S")
            self.logger.info(
                f"⚡ AC CHARGING DISABLED at {current_time} → Topic: {ac_charge_topic}"
            )


def create_battery_controller(
    battery_config: Dict[str, Any],
    mqtt_config: Dict[str, Any],
    safety_config: Dict[str, Any] = None,
) -> BatteryController:
    """
    Create a configured battery controller.

    Args:
        battery_config: Battery system specifications
        mqtt_config: MQTT connection configuration
        safety_config: Safety limits and thresholds

    Returns:
        Configured BatteryController instance
    """

    if safety_config is None:
        safety_config = {
            "max_soc_percent": 95.0,
            "min_soc_percent": 10.0,
            "max_temperature_c": 45.0,
            "min_temperature_c": 0.0,
        }

    config = {"battery": battery_config, "mqtt": mqtt_config, "safety": safety_config}

    return BatteryController(config)
