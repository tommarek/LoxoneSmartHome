"""Unit tests for the pure schedule formatter in web/api/energy.py.

_format_day_schedule_from_controller receives prices ALREADY in CZK/kWh (the
controller converts at storage time) and must pass them through unchanged — the
dead eur_czk_rate param was removed; this pins the no-conversion contract.
"""
from datetime import datetime


def test_format_day_schedule_prices_are_czk_passthrough():
    from web.api.energy import _format_day_schedule_from_controller

    now = datetime(2026, 6, 4, 10, 5)
    out = _format_day_schedule_from_controller(
        prices={("10:00", "10:15"): 3.0, ("11:00", "11:15"): 5.0},
        date=now.date(),
        label="TODAY",
        charge_blocks={("10:00", "10:15")},
        pre_discharge_blocks=set(),
        discharge_blocks={("11:00", "11:15")},
        now=now,
    )
    assert out["label"] == "TODAY"
    blocks = [b for h in out["hours"] for b in h["blocks"]]
    by_time = {b["time"]: b for b in blocks}
    # Prices preserved verbatim (NO EUR→CZK conversion).
    assert by_time["10:00-10:15"]["price_czk_kwh"] == 3.0
    assert by_time["11:00-11:15"]["price_czk_kwh"] == 5.0
    # Mode classification from the per-action block-sets.
    assert by_time["10:00-10:15"]["mode"] == "charge"
    assert by_time["11:00-11:15"]["mode"] == "discharge"
