"""
Control Strategy Execution for PEMS v2.

This module implements various control strategies for energy optimization
including price-based optimization, comfort optimization, and emergency
protocols. It serves as the high-level strategy layer that coordinates
complex control scenarios across all energy systems.

Key Features:
1. **Price-Based Strategies**: Optimize based on electricity and export prices
2. **Comfort Strategies**: Maintain optimal comfort while minimizing energy use
3. **Emergency Protocols**: Coordinated emergency response procedures
4. **Seasonal Adaptation**: Strategies adapted for different seasons/weather
5. **Load Balancing**: Distribute energy consumption optimally

Strategy Types:
- Economic: Minimize energy costs through smart scheduling
- Comfort: Maintain comfort levels with efficient energy use
- Environmental: Maximize use of renewable energy sources
- Emergency: Safe operation during system faults or emergencies
- Maintenance: Controlled operation during system maintenance

Usage in PEMS v2 Workflow:
1. Strategy selection based on current conditions and priorities
2. Strategy execution generates control schedules for all systems
3. Real-time adaptation based on changing conditions
4. Performance monitoring and strategy effectiveness assessment
5. Automatic fallback to safe strategies during issues

Integration Points:
- Input: Current prices, weather, occupancy, system status
- Output: Control schedules for unified controller
- Monitoring: System performance and strategy effectiveness
- Configuration: Strategy parameters and preferences
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .battery_controller import ChargingMode
from .inverter_controller import ExportMode, InverterMode
from .unified_controller import ControlSchedule, SystemMode, UnifiedController


class StrategyType(Enum):
    """Available control strategy types."""

    ECONOMIC = "economic"  # Cost minimization priority
    COMFORT = "comfort"  # Comfort optimization priority
    ENVIRONMENTAL = "environmental"  # Environmental impact priority
    EMERGENCY = "emergency"  # Emergency safety priority
    MAINTENANCE = "maintenance"  # Maintenance mode
    BALANCED = "balanced"  # Balanced optimization


class TimeOfDay(Enum):
    """Time of day periods for strategy adaptation."""

    NIGHT = "night"  # 22:00 - 06:00
    MORNING = "morning"  # 06:00 - 10:00
    DAY = "day"  # 10:00 - 18:00
    EVENING = "evening"  # 18:00 - 22:00


@dataclass
class StrategyContext:
    """
    Context information for strategy execution.

    Attributes:
        current_time: Current timestamp
        electricity_price_czk_kwh: Current electricity purchase price
        export_price_czk_kwh: Current electricity export price
        outdoor_temp_c: Current outdoor temperature
        occupancy_active: Whether house is currently occupied
        battery_soc_percent: Current battery state of charge
        pv_forecast_kw: PV generation forecast for next hours
        system_status: Current system health status
        emergency_active: Whether emergency mode is active
    """

    current_time: datetime
    electricity_price_czk_kwh: float
    export_price_czk_kwh: float
    outdoor_temp_c: float
    occupancy_active: bool
    battery_soc_percent: float
    pv_forecast_kw: List[float]  # Next 24 hours
    system_status: Dict[str, bool]
    emergency_active: bool = False


@dataclass
class StrategyResult:
    """
    Result of strategy execution.

    Attributes:
        schedule: Generated control schedule
        strategy_type: Strategy that was executed
        confidence: Confidence level in strategy (0.0-1.0)
        expected_cost_czk: Expected cost for this schedule
        expected_comfort_score: Expected comfort level (0.0-1.0)
        valid_until: When this strategy should be re-evaluated
        fallback_strategy: Fallback strategy if this one fails
    """

    schedule: ControlSchedule
    strategy_type: StrategyType
    confidence: float
    expected_cost_czk: float
    expected_comfort_score: float
    valid_until: datetime
    fallback_strategy: Optional[StrategyType] = None


class ControlStrategies:
    """
    High-level control strategy execution and coordination.

    This class implements various energy management strategies that coordinate
    control across heating, battery, and inverter systems. It provides
    intelligent decision-making based on current conditions and optimization
    objectives.

    Core Responsibilities:
    1. **Strategy Selection**: Choose optimal strategy based on conditions
    2. **Schedule Generation**: Create detailed control schedules
    3. **Real-time Adaptation**: Adapt strategies to changing conditions
    4. **Performance Monitoring**: Track strategy effectiveness
    5. **Fallback Management**: Handle strategy failures gracefully

    Available Strategies:
    - Economic: Minimize energy costs through price-based optimization
    - Comfort: Maintain optimal comfort with efficient energy use
    - Environmental: Maximize renewable energy utilization
    - Emergency: Safe operation during faults or emergencies
    - Balanced: Optimize multiple objectives simultaneously

    Strategy Inputs:
    - Real-time electricity and export prices
    - Weather conditions and forecasts
    - Occupancy patterns and preferences
    - System status and performance data
    - Battery state and PV generation forecasts
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize control strategies with configuration.

        Args:
            config: Configuration dictionary containing:
                - preferences: User comfort and cost preferences
                - thresholds: Price and condition thresholds
                - schedules: Default schedules and timings
                - rooms: Room configuration for heating strategies
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Strategy preferences
        self.preferences = config.get("preferences", {})
        self.cost_weight = self.preferences.get("cost_weight", 0.7)
        self.comfort_weight = self.preferences.get("comfort_weight", 0.3)
        self.max_cost_czk_kwh = self.preferences.get("max_cost_czk_kwh", 8.0)

        # Price thresholds for strategy triggers
        self.thresholds = config.get("thresholds", {})
        self.cheap_price_threshold = self.thresholds.get("cheap_price_czk_kwh", 2.0)
        self.expensive_price_threshold = self.thresholds.get(
            "expensive_price_czk_kwh", 6.0
        )
        self.good_export_threshold = self.thresholds.get("good_export_czk_kwh", 3.0)

        # Temperature preferences
        self.temp_preferences = config.get("temperature", {})
        self.comfort_temp_day = self.temp_preferences.get("comfort_day_c", 21.0)
        self.comfort_temp_night = self.temp_preferences.get("comfort_night_c", 19.0)
        self.economy_temp_day = self.temp_preferences.get("economy_day_c", 20.0)
        self.economy_temp_night = self.temp_preferences.get("economy_night_c", 18.0)

        # Room configuration
        self.room_config = config.get("rooms", {})

        # Strategy state
        self.current_strategy: Optional[StrategyType] = None
        self.last_execution: Optional[datetime] = None
        self.strategy_history: List[Tuple[datetime, StrategyType, float]] = []

        self.logger.info("Control strategies initialized")

    async def execute_strategy(
        self,
        context: StrategyContext,
        preferred_strategy: Optional[StrategyType] = None,
    ) -> StrategyResult:
        """
        Execute optimal control strategy based on current context.

        Args:
            context: Current system and environmental context
            preferred_strategy: Optional strategy preference override

        Returns:
            StrategyResult with generated control schedule
        """
        try:
            # Select optimal strategy
            if preferred_strategy:
                strategy = preferred_strategy
            else:
                strategy = self._select_strategy(context)

            self.logger.info(f"Executing {strategy.value} strategy")

            # Execute selected strategy
            if strategy == StrategyType.ECONOMIC:
                result = await self._execute_economic_strategy(context)
            elif strategy == StrategyType.COMFORT:
                result = await self._execute_comfort_strategy(context)
            elif strategy == StrategyType.ENVIRONMENTAL:
                result = await self._execute_environmental_strategy(context)
            elif strategy == StrategyType.EMERGENCY:
                result = await self._execute_emergency_strategy(context)
            elif strategy == StrategyType.BALANCED:
                result = await self._execute_balanced_strategy(context)
            else:
                # Default to balanced strategy
                result = await self._execute_balanced_strategy(context)

            # Update strategy state
            self.current_strategy = strategy
            self.last_execution = context.current_time
            self._log_strategy_execution(strategy, result.confidence)

            return result

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}")
            # Return emergency fallback
            return await self._execute_emergency_strategy(context)

    async def _execute_economic_strategy(
        self, context: StrategyContext
    ) -> StrategyResult:
        """Execute cost-minimization strategy."""
        try:
            self.logger.debug("Executing economic strategy")

            # Economic heating schedule
            heating_schedule = self._create_economic_heating_schedule(context)

            # Battery strategy: charge during cheap periods
            battery_mode = None
            battery_power = None
            if context.electricity_price_czk_kwh <= self.cheap_price_threshold:
                battery_mode = ChargingMode.GRID
                battery_power = 5.0  # Maximum charging
            elif context.battery_soc_percent < 20:
                battery_mode = ChargingMode.PV_ONLY
            else:
                battery_mode = ChargingMode.OFF

            # Inverter strategy: export when profitable
            inverter_mode = InverterMode.LOAD_FIRST
            export_enabled = True
            export_limit = None

            if context.export_price_czk_kwh >= self.good_export_threshold:
                inverter_mode = InverterMode.GRID_FIRST
                export_limit = 10.0  # Maximum export
            elif context.electricity_price_czk_kwh <= self.cheap_price_threshold:
                inverter_mode = InverterMode.BATTERY_FIRST
                export_enabled = False

            # Create schedule
            schedule = ControlSchedule(
                heating_schedule=heating_schedule,
                battery_mode=battery_mode,
                battery_power_kw=battery_power,
                inverter_mode=inverter_mode,
                export_enabled=export_enabled,
                export_limit_kw=export_limit,
                duration_minutes=60,  # Re-evaluate hourly
                priority=1,
            )

            # Calculate expected cost and comfort
            expected_cost = self._estimate_cost(schedule, context)
            expected_comfort = self._estimate_comfort(schedule, context)

            return StrategyResult(
                schedule=schedule,
                strategy_type=StrategyType.ECONOMIC,
                confidence=0.85,
                expected_cost_czk=expected_cost,
                expected_comfort_score=expected_comfort,
                valid_until=context.current_time + timedelta(hours=1),
                fallback_strategy=StrategyType.BALANCED,
            )

        except Exception as e:
            self.logger.error(f"Economic strategy failed: {e}")
            return await self._execute_balanced_strategy(context)

    async def _execute_comfort_strategy(
        self, context: StrategyContext
    ) -> StrategyResult:
        """Execute comfort-maximization strategy."""
        try:
            self.logger.debug("Executing comfort strategy")

            # Comfort heating schedule - maintain optimal temperatures
            heating_schedule = self._create_comfort_heating_schedule(context)

            # Battery strategy: ensure power availability
            battery_mode = ChargingMode.AUTO
            battery_power = 3.0  # Moderate charging to ensure availability

            if context.battery_soc_percent < 30:
                battery_mode = ChargingMode.GRID
                battery_power = 5.0

            # Inverter strategy: prioritize loads
            inverter_mode = InverterMode.LOAD_FIRST
            export_enabled = True
            export_limit = 5.0  # Limited export to prioritize house loads

            # Create schedule
            schedule = ControlSchedule(
                heating_schedule=heating_schedule,
                battery_mode=battery_mode,
                battery_power_kw=battery_power,
                inverter_mode=inverter_mode,
                export_enabled=export_enabled,
                export_limit_kw=export_limit,
                duration_minutes=30,  # More frequent updates for comfort
                priority=1,
            )

            # Calculate expected cost and comfort
            expected_cost = self._estimate_cost(schedule, context)
            expected_comfort = self._estimate_comfort(schedule, context)

            return StrategyResult(
                schedule=schedule,
                strategy_type=StrategyType.COMFORT,
                confidence=0.90,
                expected_cost_czk=expected_cost,
                expected_comfort_score=expected_comfort,
                valid_until=context.current_time + timedelta(minutes=30),
                fallback_strategy=StrategyType.BALANCED,
            )

        except Exception as e:
            self.logger.error(f"Comfort strategy failed: {e}")
            return await self._execute_balanced_strategy(context)

    async def _execute_environmental_strategy(
        self, context: StrategyContext
    ) -> StrategyResult:
        """Execute environmental impact minimization strategy."""
        try:
            self.logger.debug("Executing environmental strategy")

            # Environmental heating: efficient use during PV generation
            heating_schedule = self._create_environmental_heating_schedule(context)

            # Battery strategy: maximize PV storage
            battery_mode = ChargingMode.PV_ONLY
            battery_power = None  # Use PV availability

            # Only charge from grid if battery very low and no PV
            if (
                context.battery_soc_percent < 15
                and sum(context.pv_forecast_kw[:4]) < 2.0
            ):
                battery_mode = ChargingMode.GRID
                battery_power = 2.0  # Minimal grid charging

            # Inverter strategy: minimize grid interaction
            inverter_mode = InverterMode.PV_ONLY
            export_enabled = True
            export_limit = None  # Export excess PV

            # If PV forecast is good, allow battery priority
            if sum(context.pv_forecast_kw[:6]) > 20.0:
                inverter_mode = InverterMode.BATTERY_FIRST

            # Create schedule
            schedule = ControlSchedule(
                heating_schedule=heating_schedule,
                battery_mode=battery_mode,
                battery_power_kw=battery_power,
                inverter_mode=inverter_mode,
                export_enabled=export_enabled,
                export_limit_kw=export_limit,
                duration_minutes=45,
                priority=1,
            )

            # Calculate expected cost and comfort
            expected_cost = self._estimate_cost(schedule, context)
            expected_comfort = self._estimate_comfort(schedule, context)

            return StrategyResult(
                schedule=schedule,
                strategy_type=StrategyType.ENVIRONMENTAL,
                confidence=0.80,
                expected_cost_czk=expected_cost,
                expected_comfort_score=expected_comfort,
                valid_until=context.current_time + timedelta(minutes=45),
                fallback_strategy=StrategyType.BALANCED,
            )

        except Exception as e:
            self.logger.error(f"Environmental strategy failed: {e}")
            return await self._execute_balanced_strategy(context)

    async def _execute_emergency_strategy(
        self, context: StrategyContext
    ) -> StrategyResult:
        """Execute emergency safety strategy."""
        try:
            self.logger.warning("Executing emergency strategy")

            # Emergency heating: minimal essential heating only
            heating_schedule = {}
            essential_rooms = ["obyvak", "kuchyne", "loznice"]  # Key living areas

            for room in essential_rooms:
                if room in self.room_config:
                    # Minimal heating for essential comfort
                    heating_schedule[room] = (True, 18.0)

            # Battery strategy: stop charging, preserve power
            battery_mode = ChargingMode.OFF

            # Inverter strategy: safe load-first mode
            inverter_mode = InverterMode.LOAD_FIRST
            export_enabled = False  # No export during emergency

            # Create minimal schedule
            schedule = ControlSchedule(
                heating_schedule=heating_schedule,
                battery_mode=battery_mode,
                battery_power_kw=None,
                inverter_mode=inverter_mode,
                export_enabled=export_enabled,
                export_limit_kw=None,
                duration_minutes=15,  # Frequent re-evaluation
                priority=3,  # Emergency priority
            )

            return StrategyResult(
                schedule=schedule,
                strategy_type=StrategyType.EMERGENCY,
                confidence=1.0,  # Always confident in safety
                expected_cost_czk=50.0,  # Minimal cost
                expected_comfort_score=0.4,  # Reduced comfort
                valid_until=context.current_time + timedelta(minutes=15),
                fallback_strategy=None,  # No fallback from emergency
            )

        except Exception as e:
            self.logger.error(f"Emergency strategy failed: {e}")
            # Return absolute minimal schedule
            return StrategyResult(
                schedule=ControlSchedule(heating_schedule={}, priority=3),
                strategy_type=StrategyType.EMERGENCY,
                confidence=0.5,
                expected_cost_czk=0.0,
                expected_comfort_score=0.2,
                valid_until=context.current_time + timedelta(minutes=10),
            )

    async def _execute_balanced_strategy(
        self, context: StrategyContext
    ) -> StrategyResult:
        """Execute balanced optimization strategy."""
        try:
            self.logger.debug("Executing balanced strategy")

            # Balanced heating schedule
            heating_schedule = self._create_balanced_heating_schedule(context)

            # Battery strategy: smart charging based on prices and SOC
            battery_mode = ChargingMode.AUTO
            battery_power = 3.0

            if context.electricity_price_czk_kwh <= self.cheap_price_threshold:
                battery_mode = ChargingMode.GRID
                battery_power = 5.0
            elif context.battery_soc_percent < 25:
                battery_mode = ChargingMode.GRID
                battery_power = 2.0
            elif context.electricity_price_czk_kwh >= self.expensive_price_threshold:
                battery_mode = ChargingMode.PV_ONLY

            # Inverter strategy: adaptive based on conditions
            inverter_mode = InverterMode.LOAD_FIRST
            export_enabled = True
            export_limit = 7.0

            if context.export_price_czk_kwh >= self.good_export_threshold:
                inverter_mode = InverterMode.GRID_FIRST
                export_limit = 10.0
            elif context.electricity_price_czk_kwh <= self.cheap_price_threshold:
                inverter_mode = InverterMode.BATTERY_FIRST
                export_limit = 3.0

            # Create schedule
            schedule = ControlSchedule(
                heating_schedule=heating_schedule,
                battery_mode=battery_mode,
                battery_power_kw=battery_power,
                inverter_mode=inverter_mode,
                export_enabled=export_enabled,
                export_limit_kw=export_limit,
                duration_minutes=45,
                priority=1,
            )

            # Calculate expected cost and comfort
            expected_cost = self._estimate_cost(schedule, context)
            expected_comfort = self._estimate_comfort(schedule, context)

            return StrategyResult(
                schedule=schedule,
                strategy_type=StrategyType.BALANCED,
                confidence=0.88,
                expected_cost_czk=expected_cost,
                expected_comfort_score=expected_comfort,
                valid_until=context.current_time + timedelta(minutes=45),
                fallback_strategy=StrategyType.EMERGENCY,
            )

        except Exception as e:
            self.logger.error(f"Balanced strategy failed: {e}")
            return await self._execute_emergency_strategy(context)

    def _select_strategy(self, context: StrategyContext) -> StrategyType:
        """Select optimal strategy based on current context."""

        # Emergency conditions
        if context.emergency_active or not all(context.system_status.values()):
            return StrategyType.EMERGENCY

        # Price-based selection
        price_ratio = context.export_price_czk_kwh / context.electricity_price_czk_kwh

        # Very cheap electricity - economic focus
        if context.electricity_price_czk_kwh <= self.cheap_price_threshold:
            return StrategyType.ECONOMIC

        # Very expensive electricity - environmental focus
        if context.electricity_price_czk_kwh >= self.expensive_price_threshold:
            return StrategyType.ENVIRONMENTAL

        # Good export prices - economic focus
        if context.export_price_czk_kwh >= self.good_export_threshold:
            return StrategyType.ECONOMIC

        # Occupancy-based selection
        if context.occupancy_active:
            return StrategyType.COMFORT

        # Default to balanced
        return StrategyType.BALANCED

    def _create_economic_heating_schedule(
        self, context: StrategyContext
    ) -> Dict[str, Tuple[bool, Optional[float]]]:
        """Create heating schedule optimized for cost."""
        schedule = {}

        # Use economy temperatures
        target_temp = (
            self.economy_temp_day
            if self._is_day_time(context.current_time)
            else self.economy_temp_night
        )

        # Heat only if price is reasonable or temperature is critical
        if (
            context.electricity_price_czk_kwh <= self.expensive_price_threshold
            or context.outdoor_temp_c < 0
        ):
            # Priority rooms get heating
            priority_rooms = ["obyvak", "kuchyne", "loznice"]
            for room in priority_rooms:
                if room in self.room_config:
                    schedule[room] = (True, target_temp)

        return schedule

    def _create_comfort_heating_schedule(
        self, context: StrategyContext
    ) -> Dict[str, Tuple[bool, Optional[float]]]:
        """Create heating schedule optimized for comfort."""
        schedule = {}

        # Use comfort temperatures
        target_temp = (
            self.comfort_temp_day
            if self._is_day_time(context.current_time)
            else self.comfort_temp_night
        )

        # Heat all occupied areas for comfort
        if context.occupancy_active:
            comfort_rooms = ["obyvak", "kuchyne", "loznice", "pracovna", "hosti"]
            for room in comfort_rooms:
                if room in self.room_config:
                    schedule[room] = (True, target_temp)
        else:
            # Minimal heating when not occupied
            essential_rooms = ["obyvak", "loznice"]
            for room in essential_rooms:
                if room in self.room_config:
                    schedule[room] = (True, target_temp - 1.0)

        return schedule

    def _create_environmental_heating_schedule(
        self, context: StrategyContext
    ) -> Dict[str, Tuple[bool, Optional[float]]]:
        """Create heating schedule optimized for environmental impact."""
        schedule = {}

        # Heat during PV generation periods
        pv_available = sum(context.pv_forecast_kw[:2]) > 3.0  # Next 2 hours

        if pv_available or context.outdoor_temp_c < 5:
            target_temp = (
                self.economy_temp_day
                if self._is_day_time(context.current_time)
                else self.economy_temp_night
            )

            # Heat key rooms when PV is available
            key_rooms = ["obyvak", "kuchyne", "loznice"]
            for room in key_rooms:
                if room in self.room_config:
                    schedule[room] = (True, target_temp)

        return schedule

    def _create_balanced_heating_schedule(
        self, context: StrategyContext
    ) -> Dict[str, Tuple[bool, Optional[float]]]:
        """Create balanced heating schedule."""
        schedule = {}

        # Balanced temperatures based on time and occupancy
        if context.occupancy_active:
            base_temp = (
                self.comfort_temp_day
                if self._is_day_time(context.current_time)
                else self.comfort_temp_night
            )
        else:
            base_temp = (
                self.economy_temp_day
                if self._is_day_time(context.current_time)
                else self.economy_temp_night
            )

        # Adjust for price
        if context.electricity_price_czk_kwh >= self.expensive_price_threshold:
            base_temp -= 1.0  # Reduce for expensive periods
        elif context.electricity_price_czk_kwh <= self.cheap_price_threshold:
            base_temp += 0.5  # Increase for cheap periods

        # Apply to appropriate rooms based on occupancy
        if context.occupancy_active:
            active_rooms = ["obyvak", "kuchyne", "loznice", "pracovna"]
        else:
            active_rooms = ["obyvak", "loznice"]

        for room in active_rooms:
            if room in self.room_config:
                schedule[room] = (True, base_temp)

        return schedule

    def _is_day_time(self, current_time: datetime) -> bool:
        """Check if current time is day time (6:00-22:00)."""
        hour = current_time.hour
        return 6 <= hour < 22

    def _get_time_of_day(self, current_time: datetime) -> TimeOfDay:
        """Get time of day period."""
        hour = current_time.hour

        if 22 <= hour or hour < 6:
            return TimeOfDay.NIGHT
        elif 6 <= hour < 10:
            return TimeOfDay.MORNING
        elif 10 <= hour < 18:
            return TimeOfDay.DAY
        else:  # 18 <= hour < 22
            return TimeOfDay.EVENING

    def _estimate_cost(
        self, schedule: ControlSchedule, context: StrategyContext
    ) -> float:
        """Estimate cost for a schedule."""
        try:
            total_cost = 0.0

            # Heating cost
            heating_power = 0.0
            for room, (state, temp) in schedule.heating_schedule.items():
                if state and room in self.room_config:
                    room_power = self.room_config[room].get("power_kw", 1.0)
                    heating_power += room_power

            heating_cost = heating_power * context.electricity_price_czk_kwh
            total_cost += heating_cost

            # Battery charging cost
            if schedule.battery_mode == ChargingMode.GRID and schedule.battery_power_kw:
                battery_cost = (
                    schedule.battery_power_kw * context.electricity_price_czk_kwh
                )
                total_cost += battery_cost

            # Duration factor
            duration_hours = (schedule.duration_minutes or 60) / 60.0
            total_cost *= duration_hours

            return total_cost

        except Exception:
            return 100.0  # Default high cost if calculation fails

    def _estimate_comfort(
        self, schedule: ControlSchedule, context: StrategyContext
    ) -> float:
        """Estimate comfort score for a schedule (0.0-1.0)."""
        try:
            if not context.occupancy_active:
                return 0.8  # Comfort less important when not occupied

            comfort_score = 0.0
            total_rooms = len(self.room_config)

            # Heating comfort
            heated_rooms = sum(
                1 for state, temp in schedule.heating_schedule.values() if state
            )
            heating_comfort = heated_rooms / max(total_rooms, 1)

            # Temperature comfort
            temp_comfort = 1.0
            for room, (state, temp) in schedule.heating_schedule.items():
                if state and temp:
                    ideal_temp = (
                        self.comfort_temp_day
                        if self._is_day_time(context.current_time)
                        else self.comfort_temp_night
                    )
                    temp_diff = abs(temp - ideal_temp)
                    room_comfort = max(0.0, 1.0 - temp_diff / 3.0)  # 3°C tolerance
                    temp_comfort = min(temp_comfort, room_comfort)

            # Battery availability comfort
            battery_comfort = 1.0
            if context.battery_soc_percent < 20:
                battery_comfort = 0.6
            elif context.battery_soc_percent < 40:
                battery_comfort = 0.8

            # Weighted average
            comfort_score = (
                heating_comfort * 0.4 + temp_comfort * 0.4 + battery_comfort * 0.2
            )

            return min(1.0, max(0.0, comfort_score))

        except Exception:
            return 0.5  # Default moderate comfort if calculation fails

    def _log_strategy_execution(self, strategy: StrategyType, confidence: float):
        """Log strategy execution to history."""
        self.strategy_history.append((datetime.now(), strategy, confidence))

        # Keep only last 100 executions
        if len(self.strategy_history) > 100:
            self.strategy_history = self.strategy_history[-100:]

    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        if not self.strategy_history:
            return {}

        # Count strategy usage
        strategy_counts = {}
        total_confidence = 0.0

        for timestamp, strategy, confidence in self.strategy_history:
            strategy_counts[strategy.value] = strategy_counts.get(strategy.value, 0) + 1
            total_confidence += confidence

        avg_confidence = total_confidence / len(self.strategy_history)

        return {
            "total_executions": len(self.strategy_history),
            "strategy_usage": strategy_counts,
            "average_confidence": avg_confidence,
            "current_strategy": self.current_strategy.value
            if self.current_strategy
            else None,
            "last_execution": self.last_execution.isoformat()
            if self.last_execution
            else None,
        }


def create_control_strategies(
    preferences: Dict[str, Any] = None,
    thresholds: Dict[str, Any] = None,
    temperature: Dict[str, Any] = None,
    rooms: Dict[str, Any] = None,
) -> ControlStrategies:
    """
    Create a configured control strategies instance.

    Args:
        preferences: User preferences for cost/comfort weighting
        thresholds: Price thresholds for strategy triggers
        temperature: Temperature preferences and settings
        rooms: Room configuration for heating strategies

    Returns:
        Configured ControlStrategies instance
    """

    if preferences is None:
        preferences = {
            "cost_weight": 0.7,
            "comfort_weight": 0.3,
            "max_cost_czk_kwh": 8.0,
        }

    if thresholds is None:
        thresholds = {
            "cheap_price_czk_kwh": 2.0,
            "expensive_price_czk_kwh": 6.0,
            "good_export_czk_kwh": 3.0,
        }

    if temperature is None:
        temperature = {
            "comfort_day_c": 21.0,
            "comfort_night_c": 19.0,
            "economy_day_c": 20.0,
            "economy_night_c": 18.0,
        }

    if rooms is None:
        from config.energy_settings import ROOM_CONFIG

        rooms = ROOM_CONFIG.get("rooms", {})

    config = {
        "preferences": preferences,
        "thresholds": thresholds,
        "temperature": temperature,
        "rooms": rooms,
    }

    return ControlStrategies(config)
