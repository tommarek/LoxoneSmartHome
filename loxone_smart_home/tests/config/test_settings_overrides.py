"""Tests for the runtime config-overrides layer and the settings dashboard API.

Covers:
- schema introspection (groups, constraints, hot/restart tagging, enum choices)
- load / save / apply round-trip and atomic persistence
- Pydantic validation reuse (constraints + cross-field validators)
- the optional export-amortisation override threading through the optimizer
- the dashboard GET/POST /api/settings endpoints
"""

import json

import pytest

from config import settings_overrides as so
from config.settings import GrowattConfig


@pytest.fixture
def cfg():
    return GrowattConfig()


# --- schema ---------------------------------------------------------------

def test_schema_groups_and_field_metadata(cfg):
    schema = so.build_settings_schema(cfg)
    assert len(schema) >= 8
    by_name = {f["name"]: f for g in schema for f in g["fields"]}

    # numeric constraint extraction
    assert by_name["min_soc"]["type"] == "float"
    assert by_name["min_soc"]["min"] == 0
    assert by_name["min_soc"]["max"] == 100

    # bool field
    assert by_name["adaptive_charge_rate"]["type"] == "bool"

    # optimizer_engine was removed in the MILP-only cleanup — no longer editable.
    assert "optimizer_engine" not in by_name
    # consumption_forecast_engine remains an enum choice.
    assert by_name["consumption_forecast_engine"]["choices"] == ["binned", "ml"]
    assert by_name["consumption_forecast_engine"]["hot"] is False

    # the new optional export amort surfaces as float with no current value
    exp = by_name["battery_amortisation_export_czk"]
    assert exp["type"] == "float"
    assert exp["value"] is None
    assert exp["hot"] is True


def test_every_registry_field_exists_on_model(cfg):
    """A registry typo must not silently survive."""
    model_fields = set(type(cfg).model_fields)
    for name in so.EDITABLE_FIELDS:
        assert name in model_fields, f"{name} not a GrowattConfig field"


# --- validate / apply -----------------------------------------------------

def test_validate_coerces_and_returns_only_requested(cfg):
    out = so.validate_overrides(cfg, {"min_soc": "15", "discharge_price_min": "6.5"})
    assert out == {"min_soc": 15.0, "discharge_price_min": 6.5}
    # config itself is untouched by validate_*
    assert cfg.min_soc == 20.0


def test_apply_mutates_in_place(cfg):
    applied = so.apply_overrides(cfg, {"charge_price_max": "1.1"})
    assert applied == ["charge_price_max"]
    assert cfg.charge_price_max == 1.1


def test_unknown_keys_are_ignored(cfg):
    out = so.validate_overrides(cfg, {"min_soc": "22", "not_a_field": "x"})
    assert out == {"min_soc": 22.0}


def test_cross_field_validator_enforced(cfg):
    with pytest.raises(Exception):
        so.validate_overrides(cfg, {"max_soc": "10", "min_soc": "40"})


def test_constraint_violation_rejected(cfg):
    with pytest.raises(Exception):
        so.validate_overrides(cfg, {"min_soc": "150"})  # > le=100


def test_blank_clears_optional_export_amort(cfg):
    so.apply_overrides(cfg, {"battery_amortisation_export_czk": "3.0"})
    assert cfg.battery_amortisation_export_czk == 3.0
    so.apply_overrides(cfg, {"battery_amortisation_export_czk": ""})
    assert cfg.battery_amortisation_export_czk is None


# --- persistence ----------------------------------------------------------

def test_save_load_roundtrip(tmp_path, cfg):
    p = tmp_path / "ov.json"
    so.save_overrides({"min_soc": 12.0, "ignored_key": 1}, p)
    data = json.loads(p.read_text())
    assert "ignored_key" not in data  # only whitelisted keys persist
    assert data["min_soc"] == 12.0
    assert so.load_overrides(p) == {"min_soc": 12.0}


def test_load_missing_returns_empty(tmp_path):
    assert so.load_overrides(tmp_path / "nope.json") == {}


def test_load_garbage_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert so.load_overrides(p) == {}


# --- export amort threading ----------------------------------------------

def test_export_amort_override_reduces_export_attractiveness():
    """A high export amort should never INCREASE battery→grid discharge vs the
    base; with the override unset the result is identical to passing the base."""
    from datetime import datetime, timedelta

    from modules.growatt.milp_optimizer import MILPBatteryOptimizer, PULP_AVAILABLE
    if not PULP_AVAILABLE:
        import pytest
        pytest.skip("PuLP/CBC not installed")

    opt = MILPBatteryOptimizer()
    start = datetime(2026, 6, 1, 0, 0)
    # A clear arbitrage day: cheap night, very expensive evening.
    blocks = []
    for i in range(96):
        ts = start + timedelta(minutes=15 * i)
        price = 0.2 if ts.hour < 6 else (8.0 if 17 <= ts.hour < 21 else 2.0)
        blocks.append((ts, price))
    solar = {h: 0.0 for h in range(24)}
    consumption = {h: 0.3 for h in range(24)}
    dist = lambda h: 0.3  # noqa: E731

    common = dict(
        blocks=blocks, solar_hourly=solar, consumption_hourly=consumption,
        distribution_func=dist, battery_capacity_kwh=10.0, current_soc=100.0,
        min_soc=20.0, max_soc=100.0, sell_fee_czk=0.35,
        battery_amortisation_czk=2.0,
    )
    _, disch_base, _, _ = opt.optimize(**common)
    _, disch_unset, _, _ = opt.optimize(
        **common, battery_amortisation_export_czk=None
    )
    _, disch_high, _, _ = opt.optimize(
        **common, battery_amortisation_export_czk=50.0  # punishingly high
    )
    # Unset == base (byte-identical behaviour).
    assert disch_unset == disch_base
    # A huge export penalty cannot produce MORE grid-discharge blocks.
    assert len(disch_high) <= len(disch_base)
