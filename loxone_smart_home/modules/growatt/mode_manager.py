"""Mode management for Growatt controller - handles battery and grid modes."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .types import InverterAdapter


class ModeManager:
    """Manages Growatt inverter modes including battery-first, grid-first, and load-first."""

    def __init__(
        self,
        logger: logging.Logger,
        mqtt_client: Any,
        config: Any,
        optional_config: Dict[str, Any],
        local_tz: Any,
        last_applied: Dict[str, Tuple[Any, ...]],
        adapter: InverterAdapter,
    ) -> None:
        """Initialize ModeManager with explicit dependencies.

        Args:
            logger: Logger instance
            mqtt_client: Async MQTT client for sending commands
            config: GrowattConfig with topic and parameter settings
            optional_config: Runtime override dict (simulation_mode, etc.)
            local_tz: Local timezone (e.g., Europe/Prague)
            last_applied: Shared dict tracking last-applied command signatures
            adapter: InverterAdapter for time helpers and inverter queries
        """
        self.logger = logger
        # The shared AsyncMQTTClient instance. It reconnects internally (the
        # object is created once in BaseModule.__init__ and never rebound), so
        # holding the reference is safe — there is no live client to track.
        self.mqtt_client = mqtt_client
        self.config = config
        self._optional_config = optional_config
        self._local_tz = local_tz
        self._last_applied = last_applied
        self._adapter = adapter

        # Mode tracking
        self._battery_first_slots: Dict[int, Dict[str, Any]] = {}
        self._ac_enabled = False
        self._export_enabled = False
        # Inverter on/off state — None means "unknown, force-send first command"
        # to recover from a previous run that may have left the hardware in an
        # unexpected state.
        self._inverter_on: Optional[bool] = None

    def _get_local_now(self) -> datetime:
        """Get current time in local timezone."""
        return datetime.now(self._local_tz)

    def _to_device_hhmm(self, s: str) -> str:
        """Convert time string to device format (HH:MM)."""
        return self._adapter._to_device_hhmm(s)

    def _ensure_future_start(
        self,
        start_hour: str,
        stop_hour: str,
        preserve_duration: bool = True,
        immediate_activation: bool = False
    ) -> tuple[str, str]:
        """Ensure start time is in future for inverter scheduling."""
        # Forward preserve_duration BY KEYWORD: the adapter signature is
        # (start, stop, min_future_minutes=1, preserve_duration=False, ...), so
        # passing it positionally bound the bool to min_future_minutes and left
        # the adapter's preserve_duration at False — silently dropping the slot-
        # duration-preservation intent when a start time is bumped to the future.
        return self._adapter._ensure_future_start(
            start_hour, stop_hour,
            preserve_duration=preserve_duration,
            immediate_activation=immediate_activation,
        )

    async def _wait_for_command_result(self, command_path: str) -> Optional[Dict[str, Any]]:
        """Wait for command result from inverter."""
        return await self._adapter._wait_for_command_result(command_path)

    async def _query_inverter_state(self) -> Dict[str, Any]:
        """Query inverter state."""
        return await self._adapter._query_inverter_state()

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

            # Send the command, then register + await its response. This is
            # race-safe because mqtt_client.publish() only ENQUEUES the message
            # (the actual wire-send happens later in the publisher loop), so the
            # pending command is registered before the broker can reply.
            await self.mqtt_client.publish(topic, json.dumps(payload))
            result = await self._adapter._wait_for_command_result(
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

            # Command failed
            error_msg = result.get("message", "Unknown error") if result else "Timeout"

            if attempt < retry_count:
                # Will retry - log concisely
                self.logger.warning(
                    f"⚠️ Attempt {attempt}/{retry_count} failed for {command_description}: "
                    f"{error_msg}"
                )

                # Wait before retry with exponential backoff (capped at 30s)
                wait_time = min(30.0, retry_delay * (1.5 ** (attempt - 1)))
                self.logger.info(f"⏳ Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                # Final failure - show full response for debugging
                self.logger.error(
                    f"❌ {command_description} FAILED after {retry_count} attempts"
                )
                self.logger.error(f"Final error: {error_msg}")
                if result:
                    self.logger.error(
                        f"📋 Final response: {json.dumps(result, indent=2)}"
                    )
                else:
                    self.logger.error("📋 Final attempt: No response (timeout)")
                return (False, result)

        # Should never reach here, but just in case
        return (False, None)

    async def ensure_exclusive(self, primary: str, previous_mode: Optional[str] = None) -> bool:
        """Ensure modes are mutually exclusive at the device level.

        Args:
            primary: The mode being activated ("battery_first" or "grid_first")
            previous_mode: The previously active mode (if known). If None, disables both for safety.

        Behavior:
        - If previous_mode is None (startup/unknown): Disable both modes for full reset
        - If previous_mode is known and != primary: Only disable that specific mode
        - If previous_mode == primary: No disable needed (already exclusive)

        Returns:
            True if every required disable succeeded (or none was needed), False
            if any disable command failed — so the caller can abort and not
            advance tracked state past a mode whose old slot is still enabled.
        """
        ok = True
        if previous_mode is None:
            # Startup or unknown state - full reset for safety
            self.logger.info(
                f"🔄 Ensuring exclusive mode for {primary}, "
                f"full reset (unknown previous state)..."
            )
            ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            ok = await self.disable_grid_first() and ok
            await asyncio.sleep(self.config.command_delay)
            self.logger.debug(f"Full reset completed before applying {primary}")
        elif previous_mode == primary:
            # Already in the target mode, no disable needed
            self.logger.debug(f"Already in {primary} mode, no reset needed")
        elif previous_mode == "battery_first":
            # Only disable battery_first
            self.logger.info(f"🔄 Disabling {previous_mode} before applying {primary}...")
            ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "grid_first":
            # Only disable grid_first
            self.logger.info(f"🔄 Disabling {previous_mode} before applying {primary}...")
            ok = await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "load_first":
            # Coming from load_first, no active slots to disable
            self.logger.debug(f"Transitioning from load_first to {primary}, no slots to disable")
        else:
            # Unknown mode - full reset for safety
            self.logger.warning(
                f"Unknown previous mode '{previous_mode}', performing full reset..."
            )
            ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            ok = await self.disable_grid_first() and ok
            await asyncio.sleep(self.config.command_delay)
        if not ok:
            self.logger.error(
                f"ensure_exclusive({primary}): a disable command failed — the old "
                "mode's slot may still be enabled; caller must not advance state."
            )

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

        return ok

    async def set_battery_first(
        self, start_hour: str, stop_hour: str, stop_soc: Optional[int] = None,
        power_rate: int = 100, *, preserve_duration: bool = True, pre_scheduled: bool = False,
        immediate_activation: bool = False, previous_mode: Optional[str] = None,
        force_power_rate: bool = False
    ) -> bool:
        """Set battery-first mode for specified time window.

        Battery-first mode prioritizes charging battery from grid/solar.
        stop_soc: Battery level to stop charging at (default from config.max_soc)
        power_rate: Charge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        immediate_activation: If True, set start time in past for immediate activation
        previous_mode: The previously active mode (if known). If None, disables both for safety.

        Returns True if the mode is in the desired state after this call (a
        benign skip, simulation, or every sub-command succeeded), False if any
        sub-command failed. Callers must not advance tracked state unless True.
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
            # Defensive: the caller (_ensure_future_start) now prevents collapse
            # by snapping the start back a minute, so this should be unreachable.
            # Treat it as a benign no-op (return True) rather than a hard failure
            # — raising would roll back the WHOLE state apply (export, inverter-on)
            # for a transient near-boundary timing edge.
            self.logger.warning(
                f"Battery-first window collapsed ({adjusted_start}=={adjusted_stop}); "
                "skipping actuation (benign no-op)."
            )
            return True

        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        sig = ("battery_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        # force_power_rate must bypass this signature short-circuit: the same sig
        # (e.g. 100%) can be stored from a previous battery_first while the
        # inverter's actual powerRate was since clobbered by a grid_first
        # discharge (separate _last_applied key). Skipping here would leave that
        # stale rate in place — the exact stale-powerRate bug force was added for.
        if self._last_applied.get("battery_first") == sig and not force_power_rate:
            self.logger.debug(
                f"Battery-first {adjusted_start}-{adjusted_stop} already applied, skipping"
            )
            return True

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%) at {current_time}"
            )
            self._last_applied["battery_first"] = sig
            return True

        assert self.mqtt_client is not None

        if not await self.ensure_exclusive("battery_first", previous_mode=previous_mode):
            self.logger.error(
                "Aborting set_battery_first: failed to disable the previous mode "
                "(would leave both slots enabled / desynced)."
            )
            return False

        # Always send stopSOC (matching set_grid_first). The previous `!= 90`
        # skip assumed the inverter's resting default is 90% with no tracking of
        # the actual hardware value — if a prior charge cycle left stopSOC at
        # 100, a later stop_soc=90 would be silently skipped and the battery
        # would charge past the intended target.
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
            return False
        await asyncio.sleep(self.config.command_delay)

        # Send powerRate when it's non-default (100) OR when the caller forces it
        # (adaptive charging needs the exact rate written even at 100%, because
        # the inverter otherwise keeps a stale rate from a previous grid-first
        # discharge — the latent bug that pinned all "100%" charging at 25%).
        if power_rate != 100 or force_power_rate:
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
                return False
            await asyncio.sleep(self.config.command_delay)

        # Use slot 1
        start_dev = self._to_device_hhmm(adjusted_start)
        stop_dev = self._to_device_hhmm(adjusted_stop)
        payload = {"start": start_dev, "stop": stop_dev, "enabled": True, "slot": 1}

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
            return False

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
        return True

    async def set_ac_charge(self, enabled: bool) -> bool:
        """Set AC charging state (unified setter).

        Returns True if AC charge is in the desired state after this call
        (already there, simulated, or the command succeeded), False if the
        command failed. Callers must not advance tracked state unless True.
        """
        if self._ac_enabled == enabled:
            return True

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            self.logger.info(f"⚡ [SIMULATE] AC CHARGING {state} (simulated at {current_time})")
            self._ac_enabled = enabled
            return True

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
            return False  # Don't update state if command failed

        current_time = self._get_local_now().strftime("%H:%M:%S")
        state = "ENABLED" if enabled else "DISABLED"
        self.logger.info(
            f"⚡ AC CHARGING {state} at {current_time} → Topic: {self.config.ac_charge_topic}"
        )
        self._ac_enabled = enabled
        return True

    async def enable_ac_charge(self) -> bool:
        """Enable AC charging during battery-first mode."""
        return await self.set_ac_charge(True)

    async def disable_ac_charge(self) -> bool:
        """Disable AC charging."""
        return await self.set_ac_charge(False)

    async def disable_battery_first(self) -> bool:
        """Disable battery-first mode. Returns True on success/sim, False on a
        failed command (so ensure_exclusive can propagate the failure)."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔋 [SIMULATE] BATTERY-FIRST MODE DISABLED (simulated at {current_time})"
            )
            return True

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        success, _ = await self._execute_command_with_retry(
            self.config.battery_first_topic,
            payload,
            "batteryfirst/set/timeslot",
            "disable battery-first mode"
        )
        if not success:
            self.logger.error("Failed to disable battery-first mode")
            return False  # Don't update state if command failed

        self._battery_first_slots[1] = {
            "enabled": False,
            "start": "00:00",
            "stop": "00:00",
            "stop_soc": 90,
            "power_rate": 100
        }
        # Clear the cached signature: the slot is now disabled, so a later
        # set_battery_first with the SAME (start,stop,soc,rate) must not be
        # short-circuited against a sig that no longer reflects an enabled slot.
        self._last_applied.pop("battery_first", None)

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔋 BATTERY-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.battery_first_topic}"
        )
        return True

    async def set_export(self, enabled: bool) -> bool:
        """Set export state. Uses edge-triggered topics for external systems.

        Returns True if export is in the desired state after this call (already
        there, simulated, or the command succeeded), False if the command
        failed. Callers must not advance tracked state unless this returns True.
        """
        if self._export_enabled == enabled:
            return True

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            state = "ENABLED" if enabled else "DISABLED"
            emoji = '⬆️' if enabled else '⬇️'
            self.logger.info(
                f"{emoji} [SIMULATE] EXPORT {state} (simulated at {current_time})"
            )
            self._export_enabled = enabled
            return True

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
                return False  # Don't update state if command failed

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
                return False  # Don't update state if command failed

            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"⬇️ EXPORT DISABLED at {current_time} → Topic: {topic}")

        self._export_enabled = enabled
        return True

    async def enable_export(self) -> bool:
        """Enable electricity export to grid. Returns True on success."""
        return await self.set_export(True)

    async def disable_export(self) -> bool:
        """Disable electricity export to grid. Returns True on success."""
        return await self.set_export(False)

    async def set_inverter_power(self, on: bool) -> bool:
        """Power the inverter on or off via Modbus holding register 0.

        Uses the OpenInverterGateway `modbus/set` command. Idempotent —
        if the desired state matches the last successfully applied one,
        skips the MQTT command.

        Returns True if the inverter is in the desired state after this call
        (either already there, or the Modbus write succeeded), False if the
        write failed. Callers must not assume the hardware changed unless this
        returns True.
        """
        if self._inverter_on == on:
            return True  # No change needed

        topic = self.config.inverter_onoff_topic
        payload = {
            "id": 0,
            "type": "16b",
            "registerType": "H",
            "value": 1 if on else 0,
        }
        action = "ON" if on else "OFF"
        success, _ = await self._execute_command_with_retry(
            topic,
            payload,
            "modbus/set",
            f"inverter {action.lower()}"
        )
        if not success:
            self.logger.error(f"Failed to power inverter {action}")
            return False  # Don't update state if command failed

        self._inverter_on = on
        current_time = self._get_local_now().strftime("%H:%M:%S")
        emoji = "⚡" if on else "🛑"
        self.logger.info(
            f"{emoji} INVERTER {action} at {current_time} "
            f"via holding register 0 → Topic: {topic}"
        )
        return True

    async def set_grid_first(
        self, start_hour: str, stop_hour: str, stop_soc: int = 20, power_rate: int = 100,
        *, preserve_duration: bool = True, pre_scheduled: bool = False,
        immediate_activation: bool = False, previous_mode: Optional[str] = None
    ) -> bool:
        """Set grid-first mode for specified time window.

        Grid-first mode prioritizes selling to grid over charging battery.
        stop_soc: Battery level to stop discharging at (default 20%)
        power_rate: Discharge rate in % (default 100%)
        pre_scheduled: If True, command is being sent in advance (no time adjustment needed)
        immediate_activation: If True, set start time in past for immediate activation
        previous_mode: The previously active mode (if known). If None, disables both for safety.

        Returns True if the mode is in the desired state after this call (a
        benign skip, simulation, or every sub-command succeeded), False if any
        sub-command failed. Callers must not advance tracked state unless True.
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
            # Defensive: the caller now prevents collapse by snapping the start
            # back a minute, so this should be unreachable. Benign no-op (return
            # True) rather than a hard failure that would roll back the whole
            # state apply for a transient near-boundary timing edge.
            self.logger.warning(
                f"Grid-first window collapsed ({adjusted_start}=={adjusted_stop}); "
                "skipping actuation (benign no-op)."
            )
            return True

        stop_soc = max(5, min(100, stop_soc))
        power_rate = max(1, min(100, power_rate))

        sig = ("grid_first", adjusted_start, adjusted_stop, stop_soc, power_rate)
        if self._last_applied.get("grid_first") == sig:
            self.logger.debug(
                f"Grid-first {adjusted_start}-{adjusted_stop} "
                f"(stopSOC={stop_soc}, rate={power_rate}) already applied, skipping"
            )
            return True

        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(
                f"🔌 [SIMULATE] GRID-FIRST MODE SET: {adjusted_start}-{adjusted_stop} "
                f"(original: {start_hour}-{stop_hour}, stopSOC={stop_soc}%, "
                f"powerRate={power_rate}%, simulated at {current_time})"
            )
            self._last_applied["grid_first"] = sig
            return True

        assert self.mqtt_client is not None

        if not await self.ensure_exclusive("grid_first", previous_mode=previous_mode):
            self.logger.error(
                "Aborting set_grid_first: failed to disable the previous mode "
                "(would leave both slots enabled / desynced)."
            )
            return False

        # First set the parameters before enabling the mode
        # Set stop SOC (battery level to stop discharging)
        stopsoc_payload = {"value": stop_soc}
        self.logger.debug(f"Setting grid-first stopSOC to {stop_soc}%")
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_stopsoc_topic,
            stopsoc_payload,
            "gridfirst/set/stopsoc",
            f"grid-first stopSOC set to {stop_soc}%"
        )
        if not success:
            self.logger.error("Failed to set grid-first stopSOC, aborting mode change")
            return False

        # Delay between commands
        await asyncio.sleep(self.config.command_delay)

        powerrate_payload = {"value": power_rate}
        self.logger.debug(f"Setting grid-first powerRate to {power_rate}%")
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_powerrate_topic,
            powerrate_payload,
            "gridfirst/set/powerrate",
            f"grid-first powerRate set to {power_rate}%"
        )
        if not success:
            self.logger.error("Failed to set grid-first powerRate, aborting mode change")
            return False

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
            return False

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
        return True

    async def disable_grid_first(self) -> bool:
        """Disable grid-first mode. Returns True on success/sim, False on a
        failed command (so ensure_exclusive can propagate the failure)."""
        if self._optional_config.get("simulation_mode", False):
            current_time = self._get_local_now().strftime("%H:%M:%S")
            self.logger.info(f"🔌 [SIMULATE] GRID-FIRST MODE DISABLED (simulated at {current_time})")
            return True

        payload = {"start": "00:00", "stop": "00:00", "enabled": False, "slot": 1}
        assert self.mqtt_client is not None
        success, _ = await self._execute_command_with_retry(
            self.config.grid_first_topic,
            payload,
            "gridfirst/set/timeslot",
            "disable grid-first mode"
        )
        if not success:
            self.logger.error("Failed to disable grid-first mode")
            return False  # Don't update state if command failed

        # Clear the cached signature so a same-minute re-enable with an identical
        # (start,stop,soc,rate) isn't short-circuited against a now-disabled slot
        # (which would leave tracked=grid_first while the hardware slot is off).
        self._last_applied.pop("grid_first", None)

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🔌 GRID-FIRST MODE DISABLED at {current_time} → "
            f"Topic: {self.config.grid_first_topic}"
        )
        return True

    async def set_load_first(
        self, stop_soc: Optional[int] = None, previous_mode: Optional[str] = None
    ) -> bool:
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

        Returns True if the mode is in the desired state after this call
        (simulation or every sub-command succeeded), False if the stopSOC
        command failed. Callers must not advance tracked state unless True.
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
            return True

        # Disable the previous mode to achieve load-first. Capture the disable
        # results and abort on failure — otherwise the old battery_first/grid_first
        # slot stays ENABLED while we report success, desyncing tracked vs hardware
        # (mirrors set_battery_first/set_grid_first's ensure_exclusive hardening).
        disable_ok = True
        if previous_mode is None:
            # Startup or unknown state - full reset for safety
            self.logger.info("🏠 Setting LOAD-FIRST mode (full reset - unknown previous state)")
            disable_ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            disable_ok = await self.disable_grid_first() and disable_ok
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "load_first":
            # Already in load_first, only update SOC if needed
            self.logger.debug(f"Already in load_first mode, updating SOC to {stop_soc}%")
        elif previous_mode == "battery_first":
            # Only disable battery_first
            self.logger.info(f"🏠 Setting LOAD-FIRST mode (disabling {previous_mode})")
            disable_ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
        elif previous_mode == "grid_first":
            # Only disable grid_first
            self.logger.info(f"🏠 Setting LOAD-FIRST mode (disabling {previous_mode})")
            disable_ok = await self.disable_grid_first()
            await asyncio.sleep(self.config.command_delay)
        else:
            # Unknown mode - full reset for safety
            self.logger.warning(
                f"Unknown previous mode '{previous_mode}', performing full reset..."
            )
            disable_ok = await self.disable_battery_first()
            await asyncio.sleep(self.config.command_delay)
            disable_ok = await self.disable_grid_first() and disable_ok
            await asyncio.sleep(self.config.command_delay)

        if not disable_ok:
            self.logger.error(
                "Aborting set_load_first: failed to disable the previous mode "
                "(its slot may still be enabled; tracked state must not advance)."
            )
            return False

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
                return False  # Don't update state if command failed

        current_time = self._get_local_now().strftime("%H:%M:%S")
        self.logger.info(
            f"🏠 LOAD-FIRST MODE SET with stopSOC={stop_soc}% at {current_time}"
        )

        # Mark as successfully applied
        self._last_applied["load_first"] = sig
        return True

    def get_ac_charge_enabled(self) -> bool:
        """Get current AC charge enabled state."""
        return self._ac_enabled

    def get_export_enabled(self) -> bool:
        """Get current export enabled state."""
        return self._export_enabled

    def get_battery_first_slots(self) -> Dict[int, Dict[str, Any]]:
        """Get battery-first slot configuration."""
        return self._battery_first_slots
