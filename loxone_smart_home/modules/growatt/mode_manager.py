"""Mode management for Growatt controller - handles battery and grid modes."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    from ..growatt_controller import GrowattController


class ModeManager:
    """Manages Growatt inverter modes including battery-first, grid-first, and load-first."""

    def __init__(self, controller: GrowattController) -> None:
        """Initialize ModeManager with reference to main controller.

        Args:
            controller: Reference to the GrowattController instance
        """
        self.controller = controller
        self.logger = controller.logger
        self.mqtt_client = controller.mqtt_client
        self.config = controller.config
        self._optional_config: Dict[str, Any] = controller._optional_config
        self._local_tz = controller._local_tz
        self._last_applied = controller._last_applied

        # Mode tracking
        self._battery_first_slots: Dict[int, Dict[str, Any]] = {}
        self._ac_enabled = False
        self._export_enabled = False

    def _get_local_now(self) -> datetime:
        """Get current time in local timezone."""
        return datetime.now(self._local_tz)

    def _to_device_hhmm(self, s: str) -> str:
        """Convert time string to device format (HH:MM)."""
        result = self.controller._to_device_hhmm(s)
        assert isinstance(result, str)
        return result

    def _ensure_future_start(
        self,
        start_hour: str,
        stop_hour: str,
        preserve_duration: bool = True,
        immediate_activation: bool = False
    ) -> tuple[str, str]:
        """Ensure start time is in future for inverter scheduling."""
        result = self.controller._ensure_future_start(
            start_hour, stop_hour, preserve_duration, immediate_activation=immediate_activation
        )
        assert isinstance(result, tuple) and len(result) == 2
        return result

    async def _wait_for_command_result(self, command_path: str) -> Optional[Dict[str, Any]]:
        """Wait for command result from inverter."""
        result = await self.controller._wait_for_command_result(command_path)
        return result

    async def _query_inverter_state(self) -> Dict[str, Any]:
        """Query inverter state."""
        result = await self.controller._query_inverter_state()
        assert isinstance(result, dict)
        return result

    async def _execute_command_with_retry(
        self,
        topic: str,
        payload: Dict[str, Any],
        command_type: str,
        command_description: str
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Execute MQTT command with retry logic on failure.

        Args:
            topic: MQTT topic to publish to
            payload: Command payload to send
            command_type: Type of command for result matching (e.g., "batteryfirst/set/timeslot")
            command_description: Human-readable description for logging

        Returns:
            Tuple of (success, result_data)
        """
        if self._optional_config.get("simulation_mode", False):
            self.logger.info(f"[SIMULATE] {command_description}")
            return (True, {"success": True, "message": "Simulated"})

        assert self.mqtt_client is not None
        retry_count = self.config.command_retry_count
        retry_delay = self.config.command_retry_delay

        for attempt in range(1, retry_count + 1):
            # Log the initial attempt only
            if attempt == 1:
                self.logger.debug(
                    f"Sending {command_description}: topic={topic}"
                )

            # Send the command
            await self.mqtt_client.publish(topic, json.dumps(payload))

            # Wait for result with configured timeout
            result = await self.controller._wait_for_command_result(
                command_type, timeout=self.config.command_timeout
            )

            if result and result.get("success", False):
                # Success!
                if attempt > 1:
                    self.logger.info(
                        f"✅ {command_description} succeeded ({attempt - 1} retries)"
                    )
                else:
                    self.logger.debug(f"✅ {command_description} succeeded")
                return (True, result)

            # Command failed - store error for potential final report
            error_msg = result.get("message", "Unknown error") if result else "Timeout"

            if attempt < retry_count:
                # Will retry - only log to debug
                self.logger.debug(
                    f"Command attempt {attempt} failed: {error_msg}, retrying..."
                )

                # Wait before retry with exponential backoff (capped at 30s)
                wait_time = min(30.0, retry_delay * (1.5 ** (attempt - 1)))
                await asyncio.sleep(wait_time)
            else:
                # Final failure - single concise error message
                self.logger.error(
                    f"❌ {command_description} failed after {retry_count} attempts"
                )
                self.logger.debug(f"Final error: {error_msg}")
                if result:
                    self.logger.debug(f"Final response: {json.dumps(result, indent=2)}")
                return (False, result)

        # Should never reach here, but just in case
        return (False, None)

    async def ensure_exclusive(self, primary: str, previous_mode: Optional[str] = None) -> None:
        """Ensure modes are mutually exclusive at the device level.

        Args:
            primary: The mode being activated ("battery_first" or "grid_first")
            previous_mode: The previously active mode (if known). If None, disables both for safety.

        Behavior:
        - If previous_mode is None (startup/unknown): Disable both modes for full reset
        - If previous_mode is known and != primary: Only disable that specific mode
        - If previous_mode == primary: No disable needed (already exclusive)
        """
        if previous_mode is None:
            # Startup or unknown state - full reset for safety
            self.logger.info(
                f"🔄 Ensuring exclusive mode for {primary}, "
                f"full reset (unknown previous state)..."
            )
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
            self.logger.debug(f"Full reset completed before applying {primary}")
        elif previous_mode == primary:
            # Already in the target mode, no disable needed
            self.logger.debug(f"Already in {primary} mode, no reset needed")
        elif previous_mode == "battery_first":
            # Only disable battery_first
            self.logger.info(f"🔄 Disabling {previous_mode} before applying {primary}...")
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "grid_first":
            # Only disable grid_first
            self.logger.info(f"🔄 Disabling {previous_mode} before applying {primary}...")
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "load_first":
            # Coming from load_first, no active slots to disable
            self.logger.debug(f"Transitioning from load_first to {primary}, no slots to disable")
        else:
            # Unknown mode - full reset for safety
            self.logger.warning(
                f"Unknown previous mode '{previous_mode}', performing full reset..."
            )
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)

        state = await self._query_inverter_state()

        if state.get("battery_first"):
            bf = state["battery_first"]
            if bf.get("success"):
                slots = bf.get("timeSlots", [])
                for slot in slots:
                    if slot.get("enabled"):
                        self.logger.warning(
                            f"⚠️ Battery-first slot {slot.get('slot')} still ENABLED after reset! "
                            f"({slot.get('start')}-{slot.get('stop')})"
                        )

        if state.get("grid_first"):
            gf = state["grid_first"]
            if gf.get("success"):
                slots = gf.get("timeSlots", [])
                for slot in slots:
                    if slot.get("enabled"):
                        self.logger.warning(
                            f"⚠️ Grid-first slot {slot.get('slot')} still ENABLED after reset! "
                            f"({slot.get('start')}-{slot.get('stop')})"
                        )

    async def set_battery_first(
        self, start_hour: str, stop_hour: str, stop_soc: Optional[int] = None,
        power_rate: int = 100, *, preserve_duration: bool = True, pre_scheduled: bool = False,
        immediate_activation: bool = False, previous_mode: Optional[str] = None
    ) -> None:
        """Set battery-first mode for specified time window.

        Battery-first mode prioritizes charging battery from grid/solar.
        stop_soc: Battery level to stop charging at (default from config.max_soc)
        power_rate: Charge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        immediate_activation: If True, set start time in past for immediate activation
        previous_mode: The previously active mode (if known). If None, disables both for safety.
        """
        if stop_soc is None:
            stop_soc = int(self.config.max_soc)
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour,
                preserve_duration=preserve_duration,
                immediate_activation=immediate_activation
            )

        if adjusted_start == adjusted_stop:
            self.logger.debug(
                f"Battery-first window collapsed after bump ({adjusted_start}=={adjusted_stop}); "
                "skipping."
            )
            return

        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        sig = ("battery_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        if self._last_applied.get("battery_first") == sig:
            self.logger.debug(
                f"Battery-first {adjusted_start}-{adjusted_stop} already applied, skipping"
            )
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%) at {current_time}"
            )
            self._last_applied["battery_first"] = sig
            return

        assert self.mqtt_client is not None

        await self.ensure_exclusive("battery_first", previous_mode=previous_mode)

        if stop_soc != 90:
            stopsoc_topic = "energy/solar/command/batteryfirst/set/stopsoc"
            stopsoc_payload = {"value": stop_soc}
            self.logger.debug(f"Setting battery-first stopSOC to {stop_soc}%")
            success, _ = await self._execute_command_with_retry(
                stopsoc_topic,
                stopsoc_payload,
                "batteryfirst/set/stopsoc",
                f"battery-first stopSOC set to {stop_soc}%"
            )
            if not success:
                self.logger.error("Failed to set battery-first stopSOC, aborting mode change")
                return
            await asyncio.sleep(self.config.command_delay)

        if power_rate != 100:
            powerrate_topic = "energy/solar/command/batteryfirst/set/powerrate"
            powerrate_payload = {"value": power_rate}
            self.logger.debug(f"Setting battery-first powerRate to {power_rate}%")
            success, _ = await self._execute_command_with_retry(
                powerrate_topic,
                powerrate_payload,
                "batteryfirst/set/powerrate",
                f"battery-first powerRate set to {power_rate}%"
            )
            if not success:
                self.logger.error("Failed to set battery-first powerRate, aborting mode change")
                return
            await asyncio.sleep(self.config.command_delay)

        # Use slot 1
        start_dev = self._to_device_hhmm(adjusted_start)
        stop_dev = self._to_device_hhmm(adjusted_stop)
        payload = {"start": start_dev, "stop": stop_dev, "enabled": True, "slot": 1}

        # DEBUG
        self.logger.warning(
            f"🔍 DEBUG battery-first payload: start={start_dev!r}, stop={stop_dev!r}, "
            f"enabled=True, slot=1 (adjusted from {adjusted_start} to {adjusted_stop})"
        )

        self.logger.debug(f"Enabling battery-first mode for {adjusted_start}-{adjusted_stop}")
        success, _ = await self._execute_command_with_retry(
            self.config.battery_first_topic,
            payload,
            "batteryfirst/set/timeslot",
            f"battery-first mode set for {adjusted_start}-{adjusted_stop}"
        )
        if not success:
            self.logger.error(
                "⚠️ Battery-first mode NOT activated. Inverter may still be in previous mode. "
                "Check inverter display or query current state."
            )
            return

        self._battery_first_slots[1] = {
            "enabled": True,
            "start": start_dev,
            "stop": stop_dev,
            "stop_soc": stop_soc,
            "power_rate": power_rate
        }

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
            f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
            f"powerRate={power_rate}%) at {current_time} → "
            f"Topic: {self.config.battery_first_topic}"
        )
        self._last_applied["battery_first"] = sig

        await asyncio.sleep(self.config.command_delay)
        state = await self._query_inverter_state()
        self.logger.info("📋 Inverter state after battery-first command:")
        if state.get("battery_first"):
            self.logger.info(f"   Battery-first: {state['battery_first']}")
        if state.get("grid_first"):
            self.logger.info(f"   Grid-first: {state['grid_first']}")

    async def set_ac_charge(self, enabled: bool) -> None:
        """Set AC charging state (unified setter)."""
        if self._ac_enabled == enabled:
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING {state} (simulated at {current_time})")
            self._ac_enabled = enabled
            return

        payload = {"value": 1 if enabled else 0}
        assert self.mqtt_client is not None
        state_text = "ENABLED" if enabled else "DISABLED"
        success, _ = await self._execute_command_with_retry(
            self.config.ac_charge_topic,
            payload,
            "batteryfirst/set/acchargeenabled",
            f"AC charge {state_text}"
        )
        if not success:
            self.logger.error(f"Failed to set AC charge to {state_text}")
            # Continue anyway, not critical

        current_time = self._get_local_now().strftime("%H:%M:%S")
        state = "ENABLED" if enabled else "DISABLED"
        self.logger.info(
            f"⚡ AC CHARGING {state} at {current_time} → Topic: {self.config.ac_charge_topic}"
        )
        self._ac_enabled = enabled

    async def enable_ac_charge(self) -> None:
        """Enable AC charging during battery-first mode."""
        await self.set_ac_charge(True)

    async def disable_ac_charge(self) -> None:
        """Disable AC charging."""
        await self.set_ac_charge(False)

    async def disable_battery_first(self) -> None:
        """Disable battery-first mode."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE DISABLED (simulated at {current_time})"
            )
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        # DEBUG
        self.logger.warning(
            "🔍 DEBUG disable battery-first payload: start='00:00', stop='00:00', "
            "enabled=False, slot=1"
        )
        success, _ = await self._execute_command_with_retry(
            self.config.battery_first_topic,
            payload,
            "batteryfirst/set/timeslot",
            "disable battery-first mode"
        )
        if not success:
            self.logger.error("Failed to disable battery-first mode")
            # Continue anyway to update state

        self._battery_first_slots[1] = {
            "enabled": False,
            "start": "00:00",
            "stop": "00:00",
            "stop_soc": 90,
            "power_rate": 100
        }

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.battery_first_topic}"
        )

    async def set_export(self, enabled: bool) -> None:
        """Set export state. Uses edge-triggered topics for external systems."""
        if self._export_enabled == enabled:
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            emoji = '⬆️' if enabled else '⬇️'
            self.logger.info(
                f"{emoji} [SIMULATE] EXPORT {state} (simulated at {current_time})"
            )
            self._export_enabled = enabled
            return

        # Edge-triggered topics
        payload = {"value": True}
        assert self.mqtt_client is not None

        if enabled:
            topic = self.config.export_enable_topic
            success, _ = await self._execute_command_with_retry(
                topic,
                payload,
                "export/enable",
                "export enable"
            )
            if not success:
                self.logger.error("Failed to enable export")
                # Continue anyway to update state

            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬆️ EXPORT ENABLED at {current_time} → Topic: {topic}")
        else:
            topic = self.config.export_disable_topic
            success, _ = await self._execute_command_with_retry(
                topic,
                payload,
                "export/disable",
                "export disable"
            )
            if not success:
                self.logger.error("Failed to disable export")
                # Continue anyway to update state

            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬇️ EXPORT DISABLED at {current_time} → Topic: {topic}")

        self._export_enabled = enabled

    async def enable_export(self) -> None:
        """Enable electricity export to grid."""
        await self.set_export(True)

    async def disable_export(self) -> None:
        """Disable electricity export to grid."""
        await self.set_export(False)

    async def set_grid_first(
        self, start_hour: str, stop_hour: str, stop_soc: int = 20, power_rate: int = 100,
        *, preserve_duration: bool = True, pre_scheduled: bool = False,
        immediate_activation: bool = False, previous_mode: Optional[str] = None
    ) -> None:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc: Battery level to stop discharging at (default 20%)
        power_rate: Discharge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        immediate_activation: If True, set start time in past for immediate activation
        previous_mode: The previously active mode (if known). If None, disables both for safety.
        """
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour,
                preserve_duration=preserve_duration,
                immediate_activation=immediate_activation
            )

        if adjusted_start == adjusted_stop:
            self.logger.debug(
                f"Grid-first window collapsed after bump ({adjusted_start}=={adjusted_stop}); "
                "skipping."
            )
            return

        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        sig = ("grid_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        if self._last_applied.get("grid_first") == sig:
            self.logger.debug(
                f"Grid-first {adjusted_start}-{adjusted_stop} "
                f"(stopSOC={stop_soc}, rate={power_rate}) already applied, skipping"
            )
            return

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔌 [SIMULATE] GRID-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%, simulated at {current_time})"
            )
            self._last_applied["grid_first"] = sig
            return

        assert self.mqtt_client is not None

        await self.ensure_exclusive("grid_first", previous_mode=previous_mode)

        # First set the parameters before enabling the mode
        # Set stop SOC (battery level to stop discharging)
        stopsoc_payload = {"value": stop_soc}
        self.logger.debug(f"Setting grid-first stopSOC to {stop_soc}%")
        # DEBUG
        self.logger.warning(f"🔍 DEBUG grid-first stopSOC payload: value={stop_soc}")
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_stopsoc_topic,
            stopsoc_payload,
            "gridfirst/set/stopsoc",
            f"grid-first stopSOC set to {stop_soc}%"
        )
        if not success:
            self.logger.error("Failed to set grid-first stopSOC, aborting mode change")
            return

        # Delay between commands
        await asyncio.sleep(self.config.command_delay)

        powerrate_payload = {"value": power_rate}
        self.logger.debug(f"Setting grid-first powerRate to {power_rate}%")
        # DEBUG
        self.logger.warning(f"🔍 DEBUG grid-first powerRate payload: value={power_rate}")
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_powerrate_topic,
            powerrate_payload,
            "gridfirst/set/powerrate",
            f"grid-first powerRate set to {power_rate}%"
        )
        if not success:
            self.logger.error("Failed to set grid-first powerRate, aborting mode change")
            return

        # Delay before enabling the mode
        await asyncio.sleep(self.config.command_delay)

        # Finally set the time slot to enable the mode
        # IMPORTANT: Both battery-first and grid-first MUST use slot 1!
        # The inverter prioritizes slot 1, so using slot 2 for grid-first prevents
        # proper export functionality when switching between modes.
        # Convert to HH:MM format required by device
        start_dev = self._to_device_hhmm(adjusted_start)
        stop_dev = self._to_device_hhmm(adjusted_stop)
        timeslot_payload = {
            "start": start_dev, "stop": stop_dev, "enabled": True, "slot": 1
        }

        # TEMPORARY DEBUG: Show exact payload and parameters
        self.logger.warning(
            f"🔍 DEBUG grid-first timeslot payload: start={start_dev!r}, stop={stop_dev!r}, "
            f"enabled=True, slot=1 (adjusted from {adjusted_start} to {adjusted_stop}), "
            f"stopSOC={stop_soc}, powerRate={power_rate}"
        )

        self.logger.debug(f"Enabling grid-first mode for {adjusted_start}-{adjusted_stop}")
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_topic,
            timeslot_payload,
            "gridfirst/set/timeslot",
            f"grid-first mode set for {adjusted_start}-{adjusted_stop}"
        )
        if not success:
            self.logger.error(
                "⚠️ Grid-first mode NOT activated. Inverter may still be in previous mode. "
                "Check inverter display or query current state."
            )
            return

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
            f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
            f"powerRate={power_rate}%) at {current_time} → "
            f"Topics: {self.config.grid_first_topic}, {self.config.grid_first_stopsoc_topic}, "
            f"{self.config.grid_first_powerrate_topic}"
        )
        self._last_applied["grid_first"] = sig

        # Give the inverter more time to process grid-first mode changes
        # as some firmware versions need extra time to update their state
        await asyncio.sleep(max(self.config.command_delay * 2, 2.0))

        # Query state but don't fail if the query doesn't work immediately
        # The command has already succeeded, the query is just for logging
        try:
            state = await self._query_inverter_state()
            self.logger.info("📋 Inverter state after grid-first command:")
            if state.get("battery_first"):
                self.logger.info(f"   Battery-first: {state['battery_first']}")
            if state.get("grid_first"):
                self.logger.info(f"   Grid-first: {state['grid_first']}")
        except Exception as e:
            self.logger.debug(f"Could not query state after grid-first command: {e}")
            self.logger.info(
                "Note: Grid-first command succeeded, but state query failed. "
                "This is normal for some firmware versions."
            )

    async def disable_grid_first(self) -> None:
        """Disable grid-first mode."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"🔌 [SIMULATE] GRID-FIRST MODE DISABLED (simulated at {current_time})")
            return

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        # DEBUG
        self.logger.warning(
            "🔍 DEBUG disable grid-first payload: start='00:00', stop='00:00', "
            "enabled=False, slot=1"
        )
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_topic,
            payload,
            "gridfirst/set/timeslot",
            "disable grid-first mode"
        )
        if not success:
            self.logger.error("Failed to disable grid-first mode")
            # Continue anyway to update state

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.grid_first_topic}"
        )

    async def set_load_first(
        self, stop_soc: Optional[int] = None, previous_mode: Optional[str] = None
    ) -> None:
        """Set load-first mode (default/neutral mode).

        This is the inverter's default state where it prioritizes powering loads.

        Args:
            stop_soc: Minimum battery level to maintain in load-first mode
                     (default from config.min_soc)
            previous_mode: The previously active mode (if known).
                          If None, disables both for safety.

        Behavior:
        - If previous_mode is None (startup): Disable both modes for full reset
        - If previous_mode is "battery_first": Only disable battery_first
        - If previous_mode is "grid_first": Only disable grid_first
        - If previous_mode is "load_first": No disable needed (already in that mode)
        """
        if stop_soc is None:
            stop_soc = int(self.config.min_soc)

        stop_soc = max(5, min(100, stop_soc))

        sig = ("load_first", stop_soc)

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🏠 [SIMULATE] LOAD-FIRST MODE SET with stopSOC={stop_soc}% "
                f"(simulated at {current_time})"
            )
            self._last_applied["load_first"] = sig
            return

        # Disable the previous mode to achieve load-first
        if previous_mode is None:
            # Startup or unknown state - full reset for safety
            self.logger.info("🏠 Setting LOAD-FIRST mode (full reset - unknown previous state)")
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "load_first":
            # Already in load_first, only update SOC if needed
            self.logger.debug(f"Already in load_first mode, updating SOC to {stop_soc}%")
        elif previous_mode == "battery_first":
            # Only disable battery_first
            self.logger.info(f"🏠 Setting LOAD-FIRST mode (disabling {previous_mode})")
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "grid_first":
            # Only disable grid_first
            self.logger.info(f"🏠 Setting LOAD-FIRST mode (disabling {previous_mode})")
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
        else:
            # Unknown mode - full reset for safety
            self.logger.warning(
                f"Unknown previous mode '{previous_mode}', performing full reset..."
            )
            await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)

        # Set the stop SOC for load-first mode
        if hasattr(self.config, 'load_first_stopsoc_topic'):
            stopsoc_payload = {"value": stop_soc}
            self.logger.debug(f"Setting load-first stopSOC to {stop_soc}%")
            assert self.mqtt_client is not None
            success, _ = await self._execute_command_with_retry(
                self.config.load_first_stopsoc_topic,
                stopsoc_payload,
                "loadfirst/set/stopsoc",
                f"load-first stopSOC set to {stop_soc}%"
            )
            if not success:
                self.logger.error("Failed to set load-first stopSOC")
                # Continue anyway, not critical

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🏠 LOAD-FIRST MODE SET with stopSOC={stop_soc}% at {current_time}"
        )

        # Mark as successfully applied
        self._last_applied["load_first"] = sig

    def get_ac_charge_enabled(self) -> bool:
        """Get current AC charge enabled state."""
        return self._ac_enabled

    def get_export_enabled(self) -> bool:
        """Get current export enabled state."""
        return self._export_enabled

    def get_battery_first_slots(self) -> Dict[int, Dict[str, Any]]:
        """Get battery-first slot configuration."""
        return self._battery_first_slots
