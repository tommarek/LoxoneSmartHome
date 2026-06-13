"""Tests for the inverter-state InfluxDB persistence helper.

Verifies the write_point payload shape produced by
GrowattController._write_inverter_state_point — used both event-driven on
state change and on a 5-min heartbeat. Pure unit test: it doesn't
instantiate the full controller (the existing test file's env-dependent
fixtures break import in some environments). Instead it calls the
unbound method on a SimpleNamespace stand-in with just the attributes
the helper actually reads.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from modules.growatt.inverter_state import InverterState
from modules.growatt_controller import GrowattController


def _make_state(*, export_enabled: bool, inverter_mode: str = "load_first",
                stop_soc: int = 20, power_rate: int = 25,
                ac_charge: bool = False, inverter_on: bool = True) -> InverterState:
    return InverterState(
        inverter_mode=inverter_mode,
        stop_soc=stop_soc,
        power_rate=power_rate,
        time_start="00:00",
        time_stop="23:59",
        ac_charge_enabled=ac_charge,
        export_enabled=export_enabled,
        timestamp=datetime(2026, 4, 27, 12, 0),
        source="evaluation",
        inverter_on=inverter_on,
    )


def _make_fake_controller(state: InverterState, mode: str = "regular"):
    """Build a minimal stand-in with the attributes the helper reads."""
    influxdb_client = AsyncMock()
    influxdb_client.write_point = AsyncMock()
    settings = SimpleNamespace(
        influxdb=SimpleNamespace(bucket_solar="solar"),
    )
    fixed_now = datetime(2026, 4, 27, 12, 0)
    return SimpleNamespace(
        influxdb_client=influxdb_client,
        settings=settings,
        _current_inverter_state=state,
        _current_mode=mode,
        _get_local_now=lambda: fixed_now,
        logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_write_inverter_state_point_export_enabled() -> None:
    """Happy-path: writes one point with export_enabled=1 and the matching
    tags/fields/bucket/measurement."""
    fake = _make_fake_controller(
        state=_make_state(export_enabled=True, inverter_mode="grid_first"),
        mode="sell_production",
    )

    await GrowattController._write_inverter_state_point(fake, source="mode_change")  # type: ignore[arg-type]

    fake.influxdb_client.write_point.assert_awaited_once()
    kwargs = fake.influxdb_client.write_point.await_args.kwargs
    assert kwargs["bucket"] == "solar"
    assert kwargs["measurement"] == "inverter_state"
    assert kwargs["tags"] == {
        "inverter_mode": "grid_first",
        "operational_mode": "sell_production",
        "source": "mode_change",
    }
    assert kwargs["fields"] == {
        "export_enabled": 1,
        "stop_soc": 20,
        "power_rate": 25,
        "ac_charge_enabled": 0,
        "inverter_on": 1,  # default True
    }


@pytest.mark.asyncio
async def test_write_inverter_state_point_export_disabled() -> None:
    """export_enabled=0 plus a non-default operational_mode tag."""
    fake = _make_fake_controller(
        state=_make_state(export_enabled=False, inverter_mode="load_first",
                          stop_soc=100, power_rate=100, ac_charge=True),
        mode="high_load_protected",
    )

    await GrowattController._write_inverter_state_point(fake, source="heartbeat")  # type: ignore[arg-type]

    kwargs = fake.influxdb_client.write_point.await_args.kwargs
    assert kwargs["tags"]["operational_mode"] == "high_load_protected"
    assert kwargs["tags"]["source"] == "heartbeat"
    assert kwargs["fields"]["export_enabled"] == 0
    assert kwargs["fields"]["stop_soc"] == 100
    assert kwargs["fields"]["ac_charge_enabled"] == 1


@pytest.mark.asyncio
async def test_write_inverter_state_point_inverter_off() -> None:
    """inverter_on=False is persisted as int 0 with the gate's source tag."""
    fake = _make_fake_controller(
        state=_make_state(export_enabled=False, inverter_on=False),
        mode="regular",
    )

    await GrowattController._write_inverter_state_point(  # type: ignore[arg-type]
        fake, source="price_threshold_gate"
    )

    kwargs = fake.influxdb_client.write_point.await_args.kwargs
    assert kwargs["fields"]["inverter_on"] == 0
    assert kwargs["tags"]["source"] == "price_threshold_gate"


@pytest.mark.asyncio
async def test_write_inverter_state_point_skips_when_no_client() -> None:
    """No InfluxDB client → silently skip; never raise."""
    fake = _make_fake_controller(state=_make_state(export_enabled=True))
    fake.influxdb_client = None

    # Should not raise even with no client
    await GrowattController._write_inverter_state_point(fake)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_write_inverter_state_point_skips_when_no_state() -> None:
    """No current inverter state yet (e.g., before first evaluation) → skip."""
    fake = _make_fake_controller(state=_make_state(export_enabled=True))
    fake._current_inverter_state = None

    await GrowattController._write_inverter_state_point(fake)  # type: ignore[arg-type]
    fake.influxdb_client.write_point.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_inverter_state_point_swallows_write_errors() -> None:
    """A failing InfluxDB write must not propagate — telemetry storage
    must never block inverter control."""
    fake = _make_fake_controller(state=_make_state(export_enabled=True))
    fake.influxdb_client.write_point.side_effect = RuntimeError("influx down")

    # Should not raise
    await GrowattController._write_inverter_state_point(fake)  # type: ignore[arg-type]
    fake.logger.debug.assert_called()
