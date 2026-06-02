"""Tests for Growatt ModeManager retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from modules.growatt.mode_manager import ModeManager
from config.settings import GrowattConfig


@pytest.fixture
def mock_controller():
    """Create a mock GrowattController."""
    controller = MagicMock()
    controller.logger = MagicMock()
    controller.mqtt_client = AsyncMock()
    controller.config = GrowattConfig(
        # Set retry parameters for testing
        command_delay=0.1,  # Short delay for tests
        command_retry_count=3,
        command_retry_delay=0.5,  # Minimum allowed retry delay
        command_timeout=1.0,
        # Other required config
        battery_first_topic="energy/solar/command/batteryfirst/set/timeslot",
        ac_charge_topic="energy/solar/command/batteryfirst/set/acchargeenabled",
        export_enable_topic="energy/solar/command/export/enable",
        export_disable_topic="energy/solar/command/export/disable",
        grid_first_topic="energy/solar/command/gridfirst/set/timeslot",
        grid_first_stopsoc_topic="energy/solar/command/gridfirst/set/stopsoc",
        grid_first_powerrate_topic="energy/solar/command/gridfirst/set/powerrate",
    )
    controller._optional_config = {"simulation_mode": False}
    controller._local_tz = None
    controller._last_applied = {}
    controller._wait_for_command_result = AsyncMock()
    return controller


@pytest.fixture
def mode_manager(mock_controller):
    """Create a ModeManager instance with explicit dependencies."""
    return ModeManager(
        logger=mock_controller.logger,
        mqtt_client=mock_controller.mqtt_client,
        config=mock_controller.config,
        optional_config=mock_controller._optional_config,
        local_tz=mock_controller._local_tz,
        last_applied=mock_controller._last_applied,
        adapter=mock_controller,
    )


@pytest.mark.asyncio
async def test_command_retry_on_failure(mode_manager, mock_controller):
    """Test that commands are retried on failure."""
    # Mock the command result to fail twice, then succeed
    mock_controller._wait_for_command_result.side_effect = [
        {"success": False, "message": "First failure"},
        {"success": False, "message": "Second failure"},
        {"success": True, "message": "Success"}
    ]

    # Execute a command with retry
    success, result = await mode_manager._execute_command_with_retry(
        "test/topic",
        {"value": 123},
        "test/command",
        "test command"
    )

    # Verify success after retries
    assert success is True
    assert result["success"] is True

    # Verify MQTT was called 3 times (initial + 2 retries)
    assert mock_controller.mqtt_client.publish.call_count == 3

    # Verify wait_for_command_result was called 3 times
    assert mock_controller._wait_for_command_result.call_count == 3

    # Verify logger was called for successful retries (new quiet logging)
    # Note: succeeds on 3rd attempt = 2 retries
    assert any(
        "succeeded (2 retries)" in str(call)
        for call in mode_manager.logger.info.call_args_list
    )


@pytest.mark.asyncio
async def test_command_retry_max_attempts(mode_manager, mock_controller):
    """Test that command fails after max retry attempts."""
    # Mock the command result to always fail
    mock_controller._wait_for_command_result.return_value = {
        "success": False,
        "message": "Persistent failure"
    }

    # Execute a command with retry
    success, result = await mode_manager._execute_command_with_retry(
        "test/topic",
        {"value": 123},
        "test/command",
        "test command"
    )

    # Verify failure after max retries
    assert success is False
    assert result["success"] is False

    # Verify MQTT was called 3 times (max retry count)
    assert mock_controller.mqtt_client.publish.call_count == 3

    # Verify error was logged with correct retry count
    assert any(
        "FAILED after 3 attempts" in str(call)
        for call in mode_manager.logger.error.call_args_list
    )


@pytest.mark.asyncio
async def test_command_retry_with_timeout(mode_manager, mock_controller):
    """Test that command retries on timeout."""
    # Mock the command result to timeout (return None), then succeed
    mock_controller._wait_for_command_result.side_effect = [
        None,  # Timeout
        {"success": True, "message": "Success"}
    ]

    # Execute a command with retry
    success, result = await mode_manager._execute_command_with_retry(
        "test/topic",
        {"value": 123},
        "test/command",
        "test command"
    )

    # Verify success after timeout retry
    assert success is True

    # Verify timeout was logged to warning
    assert any(
        "Timeout" in str(call)
        for call in mode_manager.logger.warning.call_args_list
    )


@pytest.mark.asyncio
async def test_command_exponential_backoff(mode_manager, mock_controller):
    """Test that retry uses exponential backoff."""
    # Mock the command result to fail twice, then succeed
    mock_controller._wait_for_command_result.side_effect = [
        {"success": False, "message": "First failure"},
        {"success": False, "message": "Second failure"},
        {"success": True, "message": "Success"}
    ]

    # Track sleep calls
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        success, _ = await mode_manager._execute_command_with_retry(
            "test/topic",
            {"value": 123},
            "test/command",
            "test command"
        )

        assert success is True

        # Verify exponential backoff delays (1.5x multiplier)
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        # First retry: 0.5 * 1.5^0 = 0.5
        # Second retry: 0.5 * 1.5^1 = 0.75
        assert len(sleep_calls) >= 2
        assert abs(sleep_calls[-2] - 0.5) < 0.01  # First retry delay
        assert abs(sleep_calls[-1] - 0.75) < 0.01  # Second retry delay (1.5x exponential)


@pytest.mark.asyncio
async def test_battery_first_with_retry(mode_manager, mock_controller):
    """Test that battery_first mode uses retry logic."""
    # Mock necessary methods
    mock_controller._wait_for_command_result.side_effect = [
        {"success": False, "message": "First failure"},  # stopSOC fails
        {"success": True, "message": "Success"},  # stopSOC retry succeeds
        {"success": True, "message": "Success"},  # powerRate succeeds
        {"success": True, "message": "Success"},  # timeslot succeeds
    ]

    mode_manager._to_device_hhmm = MagicMock(side_effect=lambda x: x)
    mode_manager._ensure_future_start = MagicMock(return_value=("12:00", "14:00"))
    mode_manager.ensure_exclusive = AsyncMock()
    mode_manager._query_inverter_state = AsyncMock(return_value={})

    # Call set_battery_first
    await mode_manager.set_battery_first("12:00", "14:00", stop_soc=95, power_rate=50)

    # Verify that commands were sent and retried
    assert mock_controller.mqtt_client.publish.call_count >= 3  # At least 3 commands

    # Verify battery-first slots were updated
    assert 1 in mode_manager._battery_first_slots
    assert mode_manager._battery_first_slots[1]["enabled"] is True


@pytest.mark.asyncio
async def test_simulation_mode(mode_manager, mock_controller):
    """Test that simulation mode bypasses actual commands."""
    # Enable simulation mode
    mock_controller._optional_config["simulation_mode"] = True

    # Execute a command
    success, result = await mode_manager._execute_command_with_retry(
        "test/topic",
        {"value": 123},
        "test/command",
        "test command"
    )

    # Verify simulation success
    assert success is True
    assert result["message"] == "Simulated"

    # Verify no actual MQTT calls were made
    assert mock_controller.mqtt_client.publish.call_count == 0
    assert mock_controller._wait_for_command_result.call_count == 0


@pytest.mark.asyncio
async def test_disable_battery_first_fails_gracefully(mode_manager, mock_controller):
    """Test that disable_battery_first returns early on failure without updating state."""
    # Set initial state
    mode_manager._battery_first_slots[1] = {"enabled": True}

    # Mock command to always fail
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }

    # Call disable
    await mode_manager.disable_battery_first()

    # Verify state was NOT updated (still enabled)
    assert mode_manager._battery_first_slots[1]["enabled"] is True

    # Verify error was logged
    assert any("Failed to disable battery-first mode" in str(call)
               for call in mode_manager.logger.error.call_args_list)

    # Verify success message was NOT logged
    assert not any("BATTERY-FIRST MODE DISABLED" in str(call)
                   for call in mode_manager.logger.info.call_args_list)


@pytest.mark.asyncio
async def test_disable_grid_first_fails_gracefully(mode_manager, mock_controller):
    """Test that disable_grid_first returns early on failure without updating state."""
    # Mock command to always fail
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }

    # Call disable
    await mode_manager.disable_grid_first()

    # Verify error was logged
    assert any("Failed to disable grid-first mode" in str(call)
               for call in mode_manager.logger.error.call_args_list)

    # Verify success message was NOT logged
    assert not any("GRID-FIRST MODE DISABLED" in str(call)
                   for call in mode_manager.logger.info.call_args_list)


@pytest.mark.asyncio
async def test_set_export_fails_gracefully(mode_manager, mock_controller):
    """Test that set_export returns early on failure without updating state."""
    mode_manager._export_enabled = False

    # Mock command to always fail
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }

    # Try to enable export
    await mode_manager.set_export(True)

    # Verify state was NOT updated (still disabled)
    assert mode_manager._export_enabled is False

    # Verify error was logged
    assert any("Failed to enable export" in str(call)
               for call in mode_manager.logger.error.call_args_list)

    # Verify success message was NOT logged
    assert not any("EXPORT ENABLED" in str(call)
                   for call in mode_manager.logger.info.call_args_list)


@pytest.mark.asyncio
async def test_set_ac_charge_fails_gracefully(mode_manager, mock_controller):
    """Test that set_ac_charge returns early on failure without updating state."""
    mode_manager._ac_enabled = False

    # Mock command to always fail
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }

    # Try to enable AC charge
    await mode_manager.set_ac_charge(True)

    # Verify state was NOT updated (still disabled)
    assert mode_manager._ac_enabled is False

    # Verify error was logged
    assert any("Failed to set AC charge" in str(call)
               for call in mode_manager.logger.error.call_args_list)


@pytest.mark.asyncio
async def test_set_load_first_aborts_on_stopsoc_failure(mode_manager, mock_controller):
    """Test that set_load_first returns early when stopSOC fails."""
    mock_controller.config.load_first_stopsoc_topic = "test/stopsoc"

    # Mock disable methods to succeed, but stopSOC to fail
    mode_manager.disable_battery_first = AsyncMock()
    mode_manager.disable_grid_first = AsyncMock()
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }

    # Try to set load_first
    ok = await mode_manager.set_load_first(stop_soc=100, previous_mode=None)

    # A failed command must report False so the controller does not advance
    # tracked state past the hardware.
    assert ok is False

    # Verify error was logged
    assert any("Failed to set load-first stopSOC" in str(call)
               for call in mode_manager.logger.error.call_args_list)

    # Verify final success message was NOT logged
    assert not any("LOAD-FIRST MODE SET" in str(call)
                   for call in mode_manager.logger.info.call_args_list)


@pytest.mark.asyncio
async def test_mode_setters_return_bool_for_desync_protection(
    mode_manager, mock_controller
):
    """The mode setters must return True only when the hardware command
    actually succeeded, so the controller can raise (and trigger rollback)
    rather than committing tracked state past a never-applied mode change."""
    mode_manager._to_device_hhmm = MagicMock(side_effect=lambda x: x)
    mode_manager._ensure_future_start = MagicMock(return_value=("12:00", "14:00"))
    mode_manager.ensure_exclusive = AsyncMock()
    mode_manager._query_inverter_state = AsyncMock(return_value={})
    mock_controller.config.load_first_stopsoc_topic = "test/stopsoc"
    mode_manager.disable_battery_first = AsyncMock()
    mode_manager.disable_grid_first = AsyncMock()

    # All commands succeed → True
    mock_controller._wait_for_command_result.return_value = {
        "success": True, "message": "ok"
    }
    assert await mode_manager.set_battery_first(
        "12:00", "14:00", stop_soc=95, power_rate=50
    ) is True
    assert await mode_manager.set_grid_first(
        "12:00", "14:00", stop_soc=20, power_rate=100
    ) is True
    assert await mode_manager.set_load_first(
        stop_soc=20, previous_mode="grid_first"
    ) is True

    # Command fails → False (clear last-applied so the call is not skipped)
    mode_manager._last_applied.clear()
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "Timeout"
    }
    assert await mode_manager.set_battery_first(
        "12:00", "14:00", stop_soc=95, power_rate=50
    ) is False
    assert await mode_manager.set_grid_first(
        "12:00", "14:00", stop_soc=20, power_rate=100
    ) is False


@pytest.mark.asyncio
async def test_set_inverter_power_off_publishes_register_0_value_0(
    mode_manager, mock_controller
):
    """set_inverter_power(False) publishes the OIG modbus/set payload."""
    mock_controller.config.inverter_onoff_topic = "energy/solar/command/modbus/set"
    mock_controller._wait_for_command_result.return_value = {
        "success": True, "message": "ok"
    }

    await mode_manager.set_inverter_power(False)

    # First (and only) publish call carries the register-0=0 payload
    publish_call = mock_controller.mqtt_client.publish.await_args_list[0]
    assert publish_call.args[0] == "energy/solar/command/modbus/set"
    import json
    body = json.loads(publish_call.args[1])
    assert body == {"id": 0, "type": "16b", "registerType": "H", "value": 0}
    assert mode_manager._inverter_on is False


@pytest.mark.asyncio
async def test_set_inverter_power_on_publishes_register_0_value_1(
    mode_manager, mock_controller
):
    """set_inverter_power(True) publishes value=1 only when state needs change."""
    mock_controller.config.inverter_onoff_topic = "energy/solar/command/modbus/set"
    mock_controller._wait_for_command_result.return_value = {
        "success": True, "message": "ok"
    }

    # Force initial state = False so True is a change
    mode_manager._inverter_on = False
    await mode_manager.set_inverter_power(True)

    import json
    body = json.loads(mock_controller.mqtt_client.publish.await_args.args[1])
    assert body["value"] == 1
    assert mode_manager._inverter_on is True


@pytest.mark.asyncio
async def test_set_inverter_power_idempotent(mode_manager, mock_controller):
    """No MQTT publish when desired state already matches current."""
    mode_manager._inverter_on = True
    await mode_manager.set_inverter_power(True)
    mock_controller.mqtt_client.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_inverter_power_sends_first_call_even_when_matching(
    mode_manager, mock_controller
):
    """Initial _inverter_on=None must NOT short-circuit — first call always sends.
    This recovers from container restart where the controller has no idea what
    the hardware's actual state is."""
    mock_controller.config.inverter_onoff_topic = "energy/solar/command/modbus/set"
    mock_controller._wait_for_command_result.return_value = {"success": True}
    assert mode_manager._inverter_on is None  # default
    await mode_manager.set_inverter_power(True)
    mock_controller.mqtt_client.publish.assert_awaited()


@pytest.mark.asyncio
async def test_set_inverter_power_does_not_update_state_on_failure(
    mode_manager, mock_controller
):
    """If the MQTT command fails after retries, _inverter_on must NOT change."""
    mock_controller.config.inverter_onoff_topic = "energy/solar/command/modbus/set"
    mock_controller._wait_for_command_result.return_value = {
        "success": False, "message": "device unreachable"
    }

    # State starts True; off-attempt fails; state must remain True
    mode_manager._inverter_on = True
    await mode_manager.set_inverter_power(False)
    assert mode_manager._inverter_on is True
