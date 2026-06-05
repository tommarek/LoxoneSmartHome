"""Unit tests for high-load detection: EV (InfluxDB) + heating (MQTT) merge.

The discharge-blocking GATE lives in the decision engine and is covered by
test_decision_engine.py (test_high_load_protection_triggers). These tests cover
the controller plumbing added to FEED that gate:
- _query_ev_charging_from_influx (parse the `ev` measurement)
- _recompute_high_load_state (merge heating+EV, honour the enable flag, fire a
  re-evaluation only on a transition)
- request_restart (SIGTERM-based restart)

Methods are bound onto a light stub so we don't need the controller's full
live-dependency __init__.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.growatt_controller import GrowattController


def _record(field, value):
    r = MagicMock()
    r.get_field.return_value = field
    r.get_value.return_value = value
    return r


def _influx_result(pairs):
    """Build a fake InfluxDB result: one table whose records are (field,value)."""
    table = SimpleNamespace(records=[_record(f, v) for f, v in pairs])
    return [table]


def _stub(**overrides):
    cfg = SimpleNamespace(
        high_load_protection_enabled=True,
        ev_charging_power_threshold_w=100,
        ev_high_load_poll_seconds=60,
    )
    s = SimpleNamespace(
        config=cfg,
        logger=MagicMock(),
        influxdb_client=AsyncMock(),
        settings=SimpleNamespace(influxdb=SimpleNamespace(bucket_loxone="loxone")),
        _ev_high_load=False,
        _ev_high_load_power=0.0,
        _high_loads_active=False,
        _high_load_details={},
        _home_status={},
        _handle_high_load_start=AsyncMock(),
        _evaluate_conditions=AsyncMock(),
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _bind(stub, name):
    return getattr(GrowattController, name).__get__(stub, GrowattController)


# --- _query_ev_charging_from_influx ---------------------------------------

async def test_ev_query_detects_charging_flag():
    stub = _stub()
    stub.influxdb_client.query = AsyncMock(
        return_value=_influx_result([("ev_charging", 1), ("ev_charging_power", 7200)])
    )
    active, power = await _bind(stub, "_query_ev_charging_from_influx")()
    assert active is True and power == 7200


async def test_ev_query_power_over_threshold_without_flag():
    stub = _stub()
    stub.influxdb_client.query = AsyncMock(
        return_value=_influx_result([("ev_charging", 0), ("ev_charging_power", 3500)])
    )
    active, power = await _bind(stub, "_query_ev_charging_from_influx")()
    assert active is True and power == 3500


async def test_ev_query_idle_returns_inactive():
    stub = _stub()
    stub.influxdb_client.query = AsyncMock(
        return_value=_influx_result([("ev_charging", 0), ("ev_charging_power", 0)])
    )
    active, power = await _bind(stub, "_query_ev_charging_from_influx")()
    assert active is False and power == 0.0


async def test_ev_query_no_rows_returns_inactive():
    stub = _stub()
    stub.influxdb_client.query = AsyncMock(return_value=[])
    active, power = await _bind(stub, "_query_ev_charging_from_influx")()
    assert active is False and power == 0.0


async def test_ev_query_error_never_wedges_protection():
    stub = _stub()
    stub.influxdb_client.query = AsyncMock(side_effect=RuntimeError("influx down"))
    active, power = await _bind(stub, "_query_ev_charging_from_influx")()
    assert active is False and power == 0.0  # fail-safe: not charging


# --- _recompute_high_load_state -------------------------------------------

async def test_recompute_ev_only_activates_and_fires_once():
    stub = _stub(_ev_high_load=True, _ev_high_load_power=7200)
    recompute = _bind(stub, "_recompute_high_load_state")
    await recompute({"heating_active": False, "heating_relays": []})
    assert stub._high_loads_active is True
    assert stub._high_load_details["ev_charging"] is True
    stub._handle_high_load_start.assert_awaited_once()
    # Re-running with the same state must NOT fire again (no transition).
    stub._handle_high_load_start.reset_mock()
    await recompute({"heating_active": False, "heating_relays": []})
    stub._handle_high_load_start.assert_not_awaited()


async def test_recompute_heating_only_activates():
    stub = _stub()
    await _bind(stub, "_recompute_high_load_state")(
        {"heating_active": True, "heating_relays": ["koupelna"]}
    )
    assert stub._high_loads_active is True
    assert stub._high_load_details["heating_active"] is True
    assert stub._high_load_details["heating_relays"] == ["koupelna"]


async def test_recompute_clear_fires_cleared_eval():
    stub = _stub(_high_loads_active=True, _ev_high_load=False)
    await _bind(stub, "_recompute_high_load_state")(
        {"heating_active": False, "heating_relays": []}
    )
    assert stub._high_loads_active is False
    stub._evaluate_conditions.assert_awaited_once_with("high_load_cleared")


async def test_recompute_disabled_forces_inactive():
    stub = _stub(_ev_high_load=True, _ev_high_load_power=9000)
    stub.config.high_load_protection_enabled = False
    await _bind(stub, "_recompute_high_load_state")(
        {"heating_active": True, "heating_relays": ["x"]}
    )
    # Protection off -> never gates, even with EV + heating both "on".
    assert stub._high_loads_active is False
    stub._handle_high_load_start.assert_not_awaited()


# --- request_restart -------------------------------------------------------

async def test_request_restart_schedules_sigterm():
    import signal as _signal
    stub = _stub()
    captured = {}

    fake_loop = MagicMock()
    fake_loop.call_later = lambda delay, cb: captured.update(delay=delay, cb=cb)

    with patch("modules.growatt_controller.asyncio.get_event_loop", return_value=fake_loop), \
         patch("modules.growatt_controller.os.kill") as kill, \
         patch("modules.growatt_controller.os.getpid", return_value=4242):
        result = await _bind(stub, "request_restart")("test")
        assert result["success"] is True
        assert captured["delay"] == 1.0
        captured["cb"]()  # fire the scheduled callback
        kill.assert_called_once_with(4242, _signal.SIGTERM)
