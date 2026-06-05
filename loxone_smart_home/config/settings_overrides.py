"""Runtime-editable configuration overrides for the Growatt controller.

The controller's config (``GrowattConfig``) is loaded once from the encrypted
``.env`` at startup. This module adds a thin, persisted override layer on top
so a curated subset of knobs can be edited live from the monitoring dashboard
(port 5555) without a redeploy:

    .env defaults  ->  overrides JSON  ->  live GrowattConfig (mutated in place)

Design rules:
- **Whitelist only.** ``EDITABLE_FIELDS`` is the single source of truth for what
  the UI may change. Anything not listed is untouchable from the web — infra
  (MQTT/InfluxDB), topics and API URLs stay ``.env``-only.
- **Hot vs restart.** Each field is tagged ``hot=True`` when the controller reads
  ``self.config.<field>`` fresh on every evaluation tick (so mutating it takes
  effect on the next re-evaluation), or ``hot=False`` when the value is consumed
  once at startup (forecaster/optimizer construction, log level) and therefore
  only takes effect after a container restart. The UI surfaces this distinction.
- **Validation reuses Pydantic.** Saving re-validates the full model
  (``model_validate``) so field constraints AND cross-field validators
  (``max_soc > min_soc``, tariff-hour parsing, …) all still apply. A bad value is
  rejected wholesale; nothing is half-applied.
- **Type/constraint metadata is derived**, not duplicated: min/max/description/
  default come from the model field. The registry only adds presentation (group,
  label, unit) and the hot/restart tag.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, get_args, get_origin

logger = logging.getLogger("GROWATT.settings")

# Location of the persisted overrides file. Lives under a dedicated state dir
# that is backed by a named docker volume so edits survive container rebuilds
# (see docker-compose.yml). Mounting a *directory* (not a single file) avoids
# the "bind-mount of a missing file becomes a directory" trap. Overridable via
# the CONFIG_OVERRIDES_PATH env var for local runs / tests.
DEFAULT_OVERRIDES_PATH = Path(
    os.environ.get("CONFIG_OVERRIDES_PATH", "/app/config_state/config_overrides.json")
)


@dataclass(frozen=True)
class FieldSpec:
    """Presentation + hot/restart metadata for one editable config field."""

    name: str
    label: str
    unit: str = ""
    hot: bool = True
    # Optional explicit choices for enum-like string fields (rendered as a
    # <select>). When omitted, a pattern on the model field is parsed instead.
    choices: Optional[List[str]] = None
    # Optional override of the field's help text (else the model description).
    help: Optional[str] = None


@dataclass(frozen=True)
class FieldGroup:
    """A titled section of related fields in the settings UI."""

    title: str
    description: str
    fields: List[FieldSpec] = dc_field(default_factory=list)


# --- The editable surface -------------------------------------------------
# Ordered for display. `hot=True` => mutating the live config applies on the
# next evaluation tick; `hot=False` => needs a container restart.
EDITABLE_GROUPS: List[FieldGroup] = [
    FieldGroup(
        "Battery & SOC limits",
        "Capacity and state-of-charge window the optimizer plans within.",
        [
            FieldSpec("battery_capacity", "Battery capacity", "kWh"),
            FieldSpec("min_soc", "Minimum SOC", "%"),
            FieldSpec("max_soc", "Maximum SOC", "%"),
            FieldSpec("discharge_min_soc", "Stop-discharge SOC", "%"),
            FieldSpec("battery_efficiency", "Round-trip efficiency", "0-1"),
        ],
    ),
    FieldGroup(
        "Battery power / C-rate",
        "How fast the battery may charge and discharge. Adaptive flags let the "
        "optimizer pick the inverter powerRate per window.",
        [
            FieldSpec("battery_charge_rate_kw", "Gentle charge power", "kW"),
            FieldSpec("battery_discharge_rate_kw", "Gentle discharge power", "kW"),
            FieldSpec("discharge_power_rate", "Discharge powerRate", "%"),
            FieldSpec("adaptive_charge_rate", "Adaptive charge rate", ""),
            FieldSpec("battery_charge_max_kw", "Hardware max charge", "kW"),
            FieldSpec("max_charge_power_kw", "Charge C-rate ceiling", "kW"),
            FieldSpec("min_charge_power_rate", "Min charge powerRate", "%"),
            FieldSpec("adaptive_discharge_rate", "Adaptive discharge rate", ""),
            FieldSpec("battery_discharge_max_kw", "Hardware max discharge", "kW"),
            FieldSpec("max_discharge_power_kw", "Discharge C-rate ceiling", "kW"),
        ],
    ),
    FieldGroup(
        "Economics",
        "Prices and wear costs the optimizer trades off. Amortisation is the "
        "battery wear cost; the optional export penalty raises the hurdle for "
        "selling to grid (arbitrage) without changing self-consumption.",
        [
            FieldSpec("charge_price_max", "Charge below", "CZK/kWh"),
            FieldSpec("export_price_min", "Export floor", "CZK/kWh"),
            FieldSpec("discharge_price_min", "Discharge above", "CZK/kWh"),
            FieldSpec("discharge_profit_margin", "Arbitrage margin", "×"),
            FieldSpec("sell_fee_czk", "Sell fee", "CZK/kWh"),
            FieldSpec("battery_amortisation_czk", "Battery wear (shared)", "CZK/kWh"),
            FieldSpec(
                "battery_amortisation_export_czk",
                "Extra wear on grid export (optional)", "CZK/kWh",
                help="Optional. Leave blank to use the shared wear cost for "
                     "export too. Set higher than the shared cost to make the "
                     "optimizer more reluctant to cycle the battery purely to "
                     "sell to grid, while still happily serving your own house.",
            ),
            FieldSpec("eur_czk_rate", "EUR→CZK rate", "CZK"),
        ],
    ),
    FieldGroup(
        "Distribution tariff & season",
        "Per-kWh import surcharge (D57d VT/NT) and summer-detection thresholds.",
        [
            FieldSpec("distribution_tariff_high", "High tariff (VT)", "CZK/kWh"),
            FieldSpec("distribution_tariff_low", "Low tariff (NT)", "CZK/kWh"),
            FieldSpec("low_tariff_hours", "Low-tariff hours", "ranges"),
            FieldSpec("summer_charge_price_max", "Summer charge max", "CZK/kWh"),
            FieldSpec("summer_temp_threshold", "Summer temp threshold", "°C"),
            FieldSpec("temperature_avg_days", "Temp avg window", "days"),
        ],
    ),
    FieldGroup(
        "Inverter on/off & export gates",
        "When to power the inverter off on deeply-negative prices, and the SOC "
        "margin within which surplus solar is exported.",
        [
            FieldSpec("inverter_off_price_threshold_czk", "Inverter-off price", "CZK/kWh"),
            FieldSpec("inverter_off_price_hysteresis_czk", "Hysteresis", "CZK/kWh"),
            FieldSpec("sell_production_min_soc_margin", "Sell-production SOC margin", "%"),
        ],
    ),
    FieldGroup(
        "Charge scheduling (rule-based)",
        "Block counts for the rule-based scheduler and the defer-to-tomorrow "
        "threshold. (The MILP/greedy optimizer ignores fixed block counts.)",
        [
            FieldSpec("battery_charge_blocks", "Charge blocks", "×15min"),
            FieldSpec("dynamic_charge_blocks", "Dynamic block count", ""),
            FieldSpec("min_charge_blocks", "Min charge blocks", "×15min"),
            FieldSpec("max_charge_blocks", "Max charge blocks", "×15min"),
            FieldSpec("pre_discharge_charge_blocks", "Pre-discharge blocks", "×15min"),
            FieldSpec("pre_discharge_window_hours", "Pre-discharge look-back", "h"),
            FieldSpec("defer_to_tomorrow_threshold", "Defer-to-tomorrow", "%"),
        ],
    ),
    FieldGroup(
        "High-load protection",
        "Prevent the battery from discharging while a big load (EV charging or "
        "heating) runs. Heating is detected from Loxone relays; EV charging from "
        "InfluxDB (teslamate).",
        [
            FieldSpec("high_load_protection_enabled", "Protection enabled", ""),
            FieldSpec("ev_charging_power_threshold_w", "EV charging threshold", "W"),
            FieldSpec("ev_high_load_poll_seconds", "EV poll interval", "s", hot=False),
        ],
    ),
    FieldGroup(
        "Optimizer engine",
        "Engine selection takes effect on restart (the optimizer is built once "
        "at startup). The MILP switch penalty applies live.",
        [
            FieldSpec("optimizer_enabled", "Optimizer enabled", "", hot=False),
            FieldSpec("optimizer_engine", "Engine", "", hot=False,
                      choices=["greedy", "milp"]),
            FieldSpec("milp_switch_penalty_czk", "MILP switch penalty", "CZK"),
        ],
    ),
    FieldGroup(
        "Forecasting (restart to apply)",
        "Forecaster construction happens at startup, so toggles and engine "
        "choices need a restart. Quantiles/confidence are read by the built "
        "forecaster and also generally need a restart.",
        [
            FieldSpec("solar_forecast_enabled", "Solar forecast", "", hot=False),
            FieldSpec("solar_forecast_confidence", "Solar confidence", "0-1", hot=False),
            FieldSpec("solar_model_quantile", "Solar model quantile", "", hot=False),
            FieldSpec("consumption_forecast_enabled", "Consumption forecast", "", hot=False),
            FieldSpec("consumption_forecast_engine", "Consumption engine", "", hot=False,
                      choices=["binned", "ml"]),
            FieldSpec("ml_consumption_quantile", "ML consumption quantile", "", hot=False),
            FieldSpec("solcast_quantile", "Solcast quantile", "", hot=False,
                      choices=["p10", "p50", "p90"]),
        ],
    ),
    FieldGroup(
        "Operational",
        "General controller behaviour. Log level and simulation mode take "
        "effect on restart.",
        [
            FieldSpec("log_level", "Log level", "", hot=False,
                      choices=["SUMMARY", "DETAIL", "VERBOSE", "DEBUG"]),
            FieldSpec("simulation_mode", "Simulation mode", "", hot=False),
        ],
    ),
]

# Flat lookups derived once.
EDITABLE_FIELDS: Dict[str, FieldSpec] = {
    spec.name: spec for group in EDITABLE_GROUPS for spec in group.fields
}


def _python_type_name(annotation: Any) -> str:
    """Map a field annotation to a UI input type: bool / int / float / str."""
    # Unwrap Optional[X] / Union[X, None].
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            annotation = args[0]
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def _constraint(metadata: List[Any], attr: str) -> Optional[float]:
    """Pull a numeric constraint (ge/gt/le/lt) out of FieldInfo.metadata."""
    for item in metadata:
        val = getattr(item, attr, None)
        if val is not None:
            return val
    return None


def build_settings_schema(config: Any) -> List[Dict[str, Any]]:
    """Build the grouped settings schema (metadata + current values) for the UI.

    Returns a JSON-serialisable list of groups; each field carries its current
    value, type, constraints, unit, hot/restart flag, help text and choices.
    """
    model_fields = type(config).model_fields
    groups: List[Dict[str, Any]] = []
    for group in EDITABLE_GROUPS:
        fields_out: List[Dict[str, Any]] = []
        for spec in group.fields:
            info = model_fields.get(spec.name)
            if info is None:
                # Registry references a field the model doesn't have — skip
                # rather than break the whole page.
                logger.warning("Settings field %r not on GrowattConfig", spec.name)
                continue
            meta = list(getattr(info, "metadata", []) or [])
            value = getattr(config, spec.name, None)
            fields_out.append({
                "name": spec.name,
                "label": spec.label,
                "unit": spec.unit,
                "hot": spec.hot,
                "type": _python_type_name(info.annotation),
                "value": value,
                "default": info.default if info.default is not None else None,
                "min": _constraint(meta, "ge") if _constraint(meta, "ge") is not None
                       else _constraint(meta, "gt"),
                "max": _constraint(meta, "le") if _constraint(meta, "le") is not None
                       else _constraint(meta, "lt"),
                "choices": spec.choices,
                "help": spec.help or (info.description or ""),
            })
        if fields_out:
            groups.append({
                "title": group.title,
                "description": group.description,
                "fields": fields_out,
            })
    return groups


def load_overrides(path: Path = DEFAULT_OVERRIDES_PATH) -> Dict[str, Any]:
    """Load the persisted overrides JSON. Missing/invalid file -> empty dict."""
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            logger.warning("Overrides file %s is not a JSON object — ignoring", path)
            return {}
        # Keep only whitelisted keys; drop anything stale/unknown silently.
        return {k: v for k, v in data.items() if k in EDITABLE_FIELDS}
    except (OSError, ValueError) as e:
        logger.warning("Failed to read overrides %s: %s — ignoring", path, e)
        return {}


def save_overrides(overrides: Dict[str, Any], path: Path = DEFAULT_OVERRIDES_PATH) -> None:
    """Persist the overrides dict atomically (write temp + replace)."""
    clean = {k: v for k, v in overrides.items() if k in EDITABLE_FIELDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean, indent=2, sort_keys=True))
    tmp.replace(path)


def validate_overrides(config: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validate `updates` against the model, returning coerced editable values.

    Raises ``pydantic.ValidationError`` if any value (or cross-field rule) is
    violated. Does NOT mutate `config`. Unknown / non-editable keys are dropped.
    Empty string for an Optional field is treated as "clear to None".
    """
    clean: Dict[str, Any] = {}
    for key, raw in updates.items():
        if key not in EDITABLE_FIELDS:
            continue
        # Blank optional => None (lets the UI clear battery_amortisation_export).
        info = type(config).model_fields.get(key)
        if raw == "" and info is not None and _is_optional(info.annotation):
            clean[key] = None
        else:
            clean[key] = raw
    merged = {**config.model_dump(), **clean}
    validated = type(config).model_validate(merged)
    # Return only the keys that were actually requested, with coerced types.
    return {k: getattr(validated, k) for k in clean}


def apply_overrides(config: Any, overrides: Dict[str, Any]) -> List[str]:
    """Validate + mutate `config` in place. Returns the list of applied fields.

    Mutating in place (rather than replacing the object) means every holder of
    the same ``GrowattConfig`` reference — the controller and any module it
    handed ``self.config`` to — observes the new values.
    """
    coerced = validate_overrides(config, overrides)
    for key, value in coerced.items():
        setattr(config, key, value)
    return list(coerced.keys())


def _is_optional(annotation: Any) -> bool:
    """True if annotation is Optional[...] / Union[..., None]."""
    if get_origin(annotation) is None:
        return False
    return type(None) in get_args(annotation)
