"""Validation of user-supplied soil measurements against the model's feature schema."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    values: dict = field(default_factory=dict)      # cleaned float values (only if ok)
    errors: list = field(default_factory=list)      # block prediction
    warnings: list = field(default_factory=list)    # allow prediction, but flag

    @property
    def ok(self) -> bool:
        return not self.errors


def _to_float(raw):
    """Return (value, error_message). Missing values are never replaced by defaults."""
    if raw is None:
        return None, "is missing"
    if isinstance(raw, bool):
        return None, "must be a number, not True/False"
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None, "is missing"
        try:
            raw = float(text)
        except ValueError:
            return None, f"'{text}' is not a valid number"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "must be a number"
    if math.isnan(value):
        return None, "is missing (NaN)"
    if math.isinf(value):
        return None, "must be a finite number"
    return value, None


def validate_input(raw: dict, schema: dict) -> ValidationResult:
    """Check every required feature; returns cleaned values, errors and warnings."""
    result = ValidationResult()
    if not isinstance(raw, dict):
        result.errors.append("Input must be a set of named soil parameters.")
        return result
    info = schema["feature_info"]
    unexpected = [k for k in raw if k not in info]
    if unexpected:
        result.errors.append(f"Unexpected parameter(s): {', '.join(map(str, unexpected))}.")
    for feat in schema["features"]:
        spec = info[feat]
        name = f"{spec['label']} [{spec['unit']}]"
        value, err = _to_float(raw.get(feat))
        if err:
            result.errors.append(f"{name} {err}.")
            continue
        if value < 0:
            result.errors.append(f"{name} cannot be negative (got {value:g}).")
            continue
        if value < spec["hard_min"] or value > spec["hard_max"]:
            result.errors.append(
                f"{name} = {value:g} is outside the accepted range "
                f"{spec['hard_min']:g} to {spec['hard_max']:g}. Please check the value and its unit.")
            continue
        if value < spec["observed_min"] or value > spec["observed_max"]:
            result.warnings.append(
                f"{name} = {value:g} lies outside the range seen in the training data "
                f"({spec['observed_min']:g} to {spec['observed_max']:g}); the prediction is an extrapolation "
                f"and should be treated with extra caution.")
        result.values[feat] = value
    if result.errors:
        result.values = {}
    return result
