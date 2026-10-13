"""Reusable inference interface for the experimental heatwave ANN model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "ml" / "models" / "heatwave" / "feedforward_ann_mlp.joblib"
SCALER_PATH = ROOT / "ml" / "models" / "heatwave" / "heatwave_feature_scaler.joblib"

CLASSIFICATION_THRESHOLD = 0.5

APPROVED_PREDICTORS = (
    "latitude",
    "longitude",
    "population_2011",
    "avg_temperature_c",
    "max_temperature_c",
    "min_temperature_c",
    "avg_humidity_pct",
    "total_rainfall_mm",
    "rainy_days",
    "heatwave_days_T",
    "severe_heatwave_days_T",
    "has_heatwave_T",
    "temp_range_c",
    "rain_intensity",
)

_model = None
_scaler = None


def _load_artifacts():
    """Load and cache the selected ANN and its matching scaler."""
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return _model, _scaler

    missing = [
        str(path)
        for path in (MODEL_PATH, SCALER_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required heatwave inference artifact(s) not found: "
            + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not hasattr(model, "predict_proba") or not hasattr(model, "predict"):
        raise TypeError("The saved ANN does not support prediction as expected.")

    if getattr(model, "n_features_in_", None) != len(APPROVED_PREDICTORS):
        raise ValueError("The ANN input feature count does not match the approved schema.")

    if getattr(scaler, "n_features_in_", None) != len(APPROVED_PREDICTORS):
        raise ValueError("The scaler input feature count does not match the approved schema.")

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and tuple(scaler_features) != APPROVED_PREDICTORS:
        raise ValueError(
            "The scaler feature order differs from the approved inference schema."
        )

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes or 0 not in classes:
        raise ValueError(f"Expected binary ANN classes 0 and 1; found {classes}.")

    _model, _scaler = model, scaler
    return _model, _scaler


def validate_features(features: Mapping[str, Any]) -> dict[str, float]:
    """Validate one complete feature record and return values in approved order."""
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping of feature names to numeric values.")

    missing = [name for name in APPROVED_PREDICTORS if name not in features]
    extra = [str(name) for name in features if name not in APPROVED_PREDICTORS]

    if missing or extra:
        details = []
        if missing:
            details.append(f"missing features: {missing}")
        if extra:
            details.append(f"unexpected features: {extra}")
        raise ValueError("; ".join(details))

    values: dict[str, float] = {}

    for name in APPROVED_PREDICTORS:
        value = features[name]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a numeric value, not {type(value).__name__}.")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite, not NaN or infinity.")

        values[name] = numeric_value

    if not -90 <= values["latitude"] <= 90:
        raise ValueError("latitude must be between -90 and 90.")

    if not -180 <= values["longitude"] <= 180:
        raise ValueError("longitude must be between -180 and 180.")

    if values["population_2011"] < 0:
        raise ValueError("population_2011 cannot be negative.")

    if not 0 <= values["avg_humidity_pct"] <= 100:
        raise ValueError("avg_humidity_pct must be between 0 and 100.")

    if values["total_rainfall_mm"] < 0:
        raise ValueError("total_rainfall_mm cannot be negative.")

    for name in ("rainy_days", "heatwave_days_T", "severe_heatwave_days_T"):
        value = values[name]
        if not value.is_integer() or not 0 <= value <= 366:
            raise ValueError(f"{name} must be a whole number between 0 and 366.")

    if values["has_heatwave_T"] not in (0.0, 1.0):
        raise ValueError("has_heatwave_T must be 0 or 1.")

    if values["severe_heatwave_days_T"] > values["heatwave_days_T"]:
        raise ValueError("Severe heatwave days cannot exceed total heatwave days.")

    expected_indicator = float(values["heatwave_days_T"] > 0)
    if values["has_heatwave_T"] != expected_indicator:
        raise ValueError("has_heatwave_T must agree with heatwave_days_T.")

    if values["max_temperature_c"] < values["min_temperature_c"]:
        raise ValueError("max_temperature_c cannot be below min_temperature_c.")

    if not (
        values["min_temperature_c"]
        <= values["avg_temperature_c"]
        <= values["max_temperature_c"]
    ):
        raise ValueError("avg_temperature_c must lie between the minimum and maximum.")

    expected_range = (
        values["max_temperature_c"] - values["min_temperature_c"]
    )
    if not math.isclose(
        values["temp_range_c"], expected_range, rel_tol=1e-5, abs_tol=1e-4
    ):
        raise ValueError("temp_range_c is inconsistent with maximum and minimum temperature.")

    expected_rain_intensity = (
        values["total_rainfall_mm"] / (values["rainy_days"] + 1)
    )
    if not math.isclose(
        values["rain_intensity"],
        expected_rain_intensity,
        rel_tol=1e-5,
        abs_tol=1e-4,
    ):
        raise ValueError("rain_intensity is inconsistent with rainfall and rainy_days.")

    return values


def predict_heatwave(features: Mapping[str, Any]) -> dict[str, Any]:
    """
    Predict next-calendar-year district heatwave occurrence.

    The returned raw positive-class score is not a calibrated real-world
    probability. This experimental model does not provide month-specific
    or individual-level health-risk predictions.
    """
    values = validate_features(features)
    model, scaler = _load_artifacts()

    # Explicit ordering prevents accidental feature-column mismatches.
    input_frame = pd.DataFrame(
        [[values[name] for name in APPROVED_PREDICTORS]],
        columns=APPROVED_PREDICTORS,
    )

    scaled_input = scaler.transform(input_frame)
    predicted_class = int(model.predict(scaled_input)[0])

    classes = list(model.classes_)
    positive_index = classes.index(1)
    raw_positive_score = float(
        model.predict_proba(scaled_input)[0, positive_index]
    )

    return {
        "model": "Feedforward ANN/MLP",
        "task": "next_calendar_year_heatwave_occurrence",
        "predicted_class": predicted_class,
        "predicted_label": (
            "heatwave_occurrence" if predicted_class == 1
            else "no_heatwave_occurrence"
        ),
        "raw_positive_score": round(raw_positive_score, 6),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "score_interpretation": (
            "Uncalibrated model score. Do not interpret it as a reliable "
            "real-world probability."
        ),
        "target_definition": (
            "At least one day with maximum temperature at or above 45 C "
            "during the next calendar year, according to the training target."
        ),
        "limitations": [
            "District-level annual prediction, not a month-specific forecast.",
            "Not a personal health-risk assessment.",
            "Raw model scores have not been shown to be reliably calibrated.",
        ],
    }