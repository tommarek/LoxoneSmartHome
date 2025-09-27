"""Mode management for Growatt controller - handles battery and grid modes."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    from modules.growatt_controller import GrowattController


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
        self._optional_config = controller._optional_config
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
        return self.controller._to_device_hhmm(s)

    def _ensure_future_start(
        self, start_hour: str, stop_hour: str, preserve_duration: bool = True
    ) -> tuple[str, str]:
        """Ensure start time is in future for inverter scheduling."""
        return self.controller._ensure_future_start(start_hour, stop_hour, preserve_duration)

    async def _wait_for_command_result(self, command_path: str) -> Optional[Dict[str, Any]]:
        """Wait for command result from inverter."""
        return await self.controller._wait_for_command_result(command_path)

    async def _query_inverter_state(self) -> Dict[str, Any]:
        """Query inverter state."""
        return await self.controller._query_inverter_state()

    async def ensure_exclusive(self, primary: str) -> None:
        """Ensure modes are mutually exclusive at the device level.

        First resets to load-first mode to ensure clean state, then applies the requested mode.
        """
        self.logger.info(f"🔄 Ensuring exclusive mode for {primary}, resetting other modes...")

        await self.disable_battery_first()
        await asyncio.sleep(0.3)
        await self.disable_grid_first()
        await asyncio.sleep(0.3)

        self.logger.debug(f"Reset to load-first mode before applying {primary}")

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
        power_rate: int = 100, *, preserve_duration: bool = True, pre_scheduled: bool = False
    ) -> None:
        """Set battery-first mode for specified time window.

        Battery-first mode prioritizes charging battery from grid/solar.
        stop_soc: Battery level to stop charging at (default from config.max_soc)
        power_rate: Charge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        """
        if stop_soc is None:
            stop_soc = int(self.config.max_soc)
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour, preserve_duration=preserve_duration
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

        await self.ensure_exclusive("battery_first")

        if stop_soc != 90:
            stopsoc_topic = "energy/solar/command/batteryfirst/set/stopsoc"
            stopsoc_payload = {"value": stop_soc}
            self.logger.debug(f"Setting battery-first stopSOC to {stop_soc}%")
            await self.mqtt_client.publish(stopsoc_topic, json.dumps(stopsoc_payload))
            await asyncio.sleep(0.5)

        if power_rate != 100:
            powerrate_topic = "energy/solar/command/batteryfirst/set/powerrate"
            powerrate_payload = {"value": power_rate}
            self.logger.debug(f"Setting battery-first powerRate to {power_rate}%")
            await self.mqtt_client.publish(powerrate_topic, json.dumps(powerrate_payload))
            await asyncio.sleep(0.5)

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
        self.logger.info(
            f"📤 Sending battery-first SET command: topic={self.config.battery_first_topic}, "
            f"payload={json.dumps(payload)}"
        )
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))

        result = await self._wait_for_command_result("batteryfirst/set/timeslot")
        if result and not result.get("success", False):
            self.logger.error(
                f"⚠️ Battery-first command FAILED! "
                f"Message: {result.get('message', 'Unknown error')}"
            )
            self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")
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

        await asyncio.sleep(0.5)
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
        self.logger.info(
            f"📤 Sending AC charge command: topic={self.config.ac_charge_topic}, "
            f"payload={json.dumps(payload)}"
        )
        await self.mqtt_client.publish(self.config.ac_charge_topic, json.dumps(payload))

        result = await self._wait_for_command_result("batteryfirst/set/acchargeenabled")
        if result and not result.get("success", False):
            self.logger.error(
                f"⚠️ AC charge command failed! Message: {result.get('message', 'Unknown error')}"
            )
            self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

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
        self.logger.info(
            f"📤 Disabling battery-first: topic={self.config.battery_first_topic}, "
            f"payload={json.dumps(payload)}"
        )
        await self.mqtt_client.publish(self.config.battery_first_topic, json.dumps(payload))

        result = await self._wait_for_command_result("batteryfirst/set/timeslot")
        if result and not result.get("success", False):
            self.logger.error(
                f"⚠️ Failed to disable battery-first! "
                f"Message: {result.get('message', 'Unknown error')}"
            )
            self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

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
            self.logger.info(
                f"📤 Sending export enable: topic={topic}, payload={json.dumps(payload)}"
            )
            await self.mqtt_client.publish(topic, json.dumps(payload))

            # Wait for result
            result = await self._wait_for_command_result("export/enable")
            if result and not result.get("success", False):
                self.logger.error(
                    f"⚠️ Export enable failed! Message: {result.get('message', 'Unknown error')}"
                )
                self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬆️ EXPORT ENABLED at {current_time} → Topic: {topic}")
        else:
            topic = self.config.export_disable_topic
            self.logger.info(
                f"📤 Sending export disable: topic={topic}, payload={json.dumps(payload)}"
            )
            await self.mqtt_client.publish(topic, json.dumps(payload))

            # Wait for result
            result = await self._wait_for_command_result("export/disable")
            if result and not result.get("success", False):
                self.logger.error(
                    f"⚠️ Export disable failed! Message: {result.get('message', 'Unknown error')}"
                )
                self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

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
        *, preserve_duration: bool = True, pre_scheduled: bool = False
    ) -> None:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc: Battery level to stop discharging at (default 20%)
        power_rate: Discharge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        """
        if pre_scheduled:
            adjusted_start, adjusted_stop = start_hour, stop_hour
        else:
            adjusted_start, adjusted_stop = self._ensure_future_start(
                start_hour, stop_hour, preserve_duration=preserve_duration
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

        await self.ensure_exclusive("grid_first")

        # First set the parameters before enabling the mode
        # Set stop SOC (battery level to stop discharging)
        stopsoc_payload = {"value": stop_soc}
        self.logger.debug(f"Setting grid-first stopSOC to {stop_soc}%")
        # DEBUG
        self.logger.warning(f"🔍 DEBUG grid-first stopSOC payload: value={stop_soc}")
        self.logger.info(
            f"📤 Sending grid-first stopSOC: topic={self.config.grid_first_stopsoc_topic}, "
            f"payload={json.dumps(stopsoc_payload)}"
        )
        await self.mqtt_client.publish(
            self.config.grid_first_stopsoc_topic, json.dumps(stopsoc_payload)
        )

        # Small delay between commands
        await asyncio.sleep(0.5)

        powerrate_payload = {"value": power_rate}
        self.logger.debug(f"Setting grid-first powerRate to {power_rate}%")
        # DEBUG
        self.logger.warning(f"🔍 DEBUG grid-first powerRate payload: value={power_rate}")
        self.logger.info(
            f"📤 Sending grid-first powerRate: topic={self.config.grid_first_powerrate_topic}, "
            f"payload={json.dumps(powerrate_payload)}"
        )
        await self.mqtt_client.publish(
            self.config.grid_first_powerrate_topic, json.dumps(powerrate_payload)
        )

        # Small delay before enabling the mode
        await asyncio.sleep(0.5)

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
        self.logger.info(
            f"📤 Sending grid-first SET command: topic={self.config.grid_first_topic}, "
            f"payload={json.dumps(timeslot_payload)}"
        )
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(timeslot_payload))

        result = await self._wait_for_command_result("gridfirst/set/timeslot")
        if result and not result.get("success", False):
            self.logger.error(
                f"❌ Grid-first command FAILED! "
                f"Message: {result.get('message', 'Unknown error')}"
            )
            self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")
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

        await asyncio.sleep(0.5)
        state = await self._query_inverter_state()
        self.logger.info("📋 Inverter state after grid-first command:")
        if state.get("battery_first"):
            self.logger.info(f"   Battery-first: {state['battery_first']}")
        if state.get("grid_first"):
            self.logger.info(f"   Grid-first: {state['grid_first']}")

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
        self.logger.info(
            f"📤 Disabling grid-first: topic={self.config.grid_first_topic}, "
            f"payload={json.dumps(payload)}"
        )
        await self.mqtt_client.publish(self.config.grid_first_topic, json.dumps(payload))

        result = await self._wait_for_command_result("gridfirst/set/timeslot")
        if result and not result.get("success", False):
            self.logger.error(
                f"⚠️ Failed to disable grid-first! "
                f"Message: {result.get('message', 'Unknown error')}"
            )
            self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.grid_first_topic}"
        )

    async def set_load_first(self, stop_soc: Optional[int] = None) -> None:
        """Set load-first mode (default/neutral mode).

        This is the inverter's default state where it prioritizes powering loads.
        stop_soc: Minimum battery level to maintain in load-first mode (default from config.min_soc)
        """
        if stop_soc is None:
            stop_soc = int(self.config.min_soc)

        stop_soc = max(5, min(100, stop_soc))

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🏠 [SIMULATE] LOAD-FIRST MODE SET with stopSOC={stop_soc}% "
                f"(simulated at {current_time})"
            )
            return

        # Load-first is achieved by disabling both battery-first and grid-first
        self.logger.info("🏠 Setting LOAD-FIRST mode (disabling all time-based modes)")

        # Disable battery-first and grid-first to achieve load-first
        await self.disable_battery_first()
        await asyncio.sleep(0.3)
        await self.disable_grid_first()
        await asyncio.sleep(0.3)

        # Set the stop SOC for load-first mode
        if hasattr(self.config, 'load_first_stopsoc_topic'):
            stopsoc_payload = {"value": stop_soc}
            self.logger.debug(f"Setting load-first stopSOC to {stop_soc}%")
            self.logger.info(
                f"📤 Sending load-first stopSOC: topic={self.config.load_first_stopsoc_topic}, "
                f"payload={json.dumps(stopsoc_payload)}"
            )
            assert self.mqtt_client is not None
            await self.mqtt_client.publish(
                self.config.load_first_stopsoc_topic, json.dumps(stopsoc_payload)
            )

            result = await self._wait_for_command_result("loadfirst/set/stopsoc")
            if result and not result.get("success", False):
                self.logger.error(
                    f"⚠️ Load-first stopSOC command failed! "
                    f"Message: {result.get('message', 'Unknown error')}"
                )
                self.logger.error(f"📋 Full response: {json.dumps(result, indent=2)}")

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🏠 LOAD-FIRST MODE SET with stopSOC={stop_soc}% at {current_time}"
        )

    def get_ac_charge_enabled(self) -> bool:
        """Get current AC charge enabled state."""
        return self._ac_enabled

    def get_export_enabled(self) -> bool:
        """Get current export enabled state."""
        return self._export_enabled

    def get_battery_first_slots(self) -> Dict[int, Dict[str, Any]]:
        """Get battery-first slot configuration."""
        return self._battery_first_slots
