"""Single ML interface for the backend: hazard ANN inference.

    from ml.hazard_inference import predict_hazard, predict_for_district, list_hazards

    list_hazards()                                   # which hazards are supported and why not
    predict_for_district("flood", 536)               # builds features from processed data
    predict_hazard("malaria", {...14 features...})   # caller supplies the feature record

Every function returns a JSON-serialisable dict with a `status`:
  ok | unsupported | unavailable | invalid_input
It never invents a score, never fills a missing input and never returns
absolute local paths. The ANN bundle (network + scaler + calibration +
threshold + feature order) is loaded from ml/models/hazards/<hazard>/.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.hazard_datasets import BUILDERS, DEFAULT_DATA_DIR, ECE_ACCEPTABLE, HAZARDS, SPECS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "models" / "hazards"

# A calibrated probability is shown only if the model card says so. The card's decision comes from a
# forward calibration check on VALIDATION years only (ECE <= ECE_ACCEPTABLE); test years take no part
# in it. See ml/ANN_DESIGN_DECISION.md section 8.

UNSUPPORTED = {
    "dengue": ("State-year labels exist only for 2021-2025 (no training year before 2020), "
               "so no chronological train/validation/test split is possible, and the labels "
               "are state-level.",
               "Obtain a longer state-year (ideally monthly) dengue series; keep the West "
               "Bengal 2024 'verification_pending' record out of scoring until confirmed."),
    "chikungunya": ("No chikungunya data exists in the repository.",
                    "Obtain NCVBDC chikungunya tables, then rerun `python -m ml.hazard_datasets --readiness`."),
}

# Plausible ranges (inclusive) used to reject nonsense input.
RANGES: dict[str, tuple[float, float]] = {
    "latitude": (6.0, 38.0), "longitude": (68.0, 98.5), "log_population": (0.0, 25.0),
    "avg_temperature_c": (-40.0, 50.0), "max_temperature_c": (-20.0, 55.0),
    "min_temperature_c": (-60.0, 45.0), "temp_range_c": (0.0, 100.0),
    "avg_humidity_pct": (0.0, 100.0), "total_rainfall_mm": (0.0, 15000.0), "rainy_days": (0.0, 366.0),
    "heatwave_days_T": (0.0, 366.0), "severe_heatwave_days_T": (0.0, 366.0), "heatwave_days_Tm1": (0.0, 366.0),
    "recorded_flood_events_T": (0.0, 5000.0), "recorded_flood_events_Tm1": (0.0, 5000.0),
    "flood_years_last5": (0.0, 5.0), "flood_years_last10": (0.0, 10.0),
    "log1p_incidence_T": (0.0, 20.0), "log1p_incidence_Tm1": (0.0, 20.0),
    "incidence_rank_pct_T": (0.0, 1.0), "high_burden_T": (0.0, 1.0),
}
# Wording mirrors the target definition: heatwave = threshold indicator, flood = inventory record.
CLASS_LABELS = {
    "heatwave": {0: "heatwave_threshold_day_not_predicted", 1: "heatwave_threshold_day_predicted"},
    "flood": {0: "recorded_flood_not_predicted", 1: "recorded_flood_predicted"},
    "malaria": {0: "high_reported_incidence_not_predicted", 1: "high_reported_incidence_predicted"},
}
WHOLE_NUMBER = {"rainy_days", "heatwave_days_T", "severe_heatwave_days_T", "heatwave_days_Tm1",
                "recorded_flood_events_T", "recorded_flood_events_Tm1", "flood_years_last5",
                "flood_years_last10", "high_burden_T"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _load(hazard: str, model_dir: str) -> tuple[dict, dict]:
    d = Path(model_dir) / hazard
    bundle_path, card_path = d / "ann_bundle.joblib", d / "model_card.json"
    if not bundle_path.is_file() or not card_path.is_file():
        raise FileNotFoundError(hazard)
    bundle = joblib.load(bundle_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if list(bundle["features"]) != list(SPECS[hazard].features) and list(bundle["features"]) != card["features"]:
        raise ValueError("feature order mismatch")
    return bundle, card


def clear_caches() -> None:
    _load.cache_clear()
    _feature_frame.cache_clear()


@functools.lru_cache(maxsize=8)
def _feature_frame(hazard: str, data_dir: str) -> pd.DataFrame:
    return BUILDERS[hazard](Path(data_dir), inference=True)[0]


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def list_hazards(model_dir: Path | str = MODEL_DIR) -> dict[str, Any]:
    """Support status of every hazard named in the project plan."""
    out: dict[str, Any] = {}
    for h in HAZARDS:
        try:
            _, card = _load(h, str(model_dir))
            out[h] = {"status": "supported" if h != "flood" else "supported_with_caveats",
                      "model_version": card["model_version"], "target": card["target_definition"],
                      "geography": card["geography"], "horizon": card["horizon"],
                      "calibrated_probability_available": _calibration_ok(card)}
        except Exception:
            out[h] = {"status": "unavailable", "reason": "trained model artifacts not found or unreadable",
                      "recovery": "python -m ml.hazard_datasets --build && python -m ml.hazard_training --hazard " + h}
    for h, (why, path) in UNSUPPORTED.items():
        out[h] = {"status": "unsupported", "reason": why, "recovery_path": path}
    return out


def _calibration_ok(card: Mapping) -> bool:
    cal = card.get("calibration", {})
    return bool(cal.get("fitted")) and cal.get("display_probability") is True


def _calibration_note(card: Mapping, ok: bool) -> str:
    chk = card["calibration"].get("validation_forward_check", {})
    detail = (f"forward check on validation years: Platt fitted on target years {chk.get('fit_target_years')}, "
              f"ECE {chk.get('ece')} on {chk.get('eval_target_year')}")
    if ok:
        return f"Platt-calibrated on validation years; {detail} (limit {ECE_ACCEPTABLE})."
    return (f"Withheld: {detail} exceeds the limit {ECE_ACCEPTABLE}, so the score is not reliably calibrated "
            "across years. Treat raw_model_score as an uncalibrated ranking score, not a probability.")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_features(hazard: str, features: Any) -> tuple[dict[str, float], list[str]]:
    """Return (ordered numeric values, error list). No value is ever filled in."""
    names = SPECS[hazard].features
    if not isinstance(features, Mapping):
        return {}, ["features must be a mapping of feature name to number"]
    errors = []
    missing = [n for n in names if n not in features]
    extra = [str(k) for k in features if k not in names]
    if missing:
        errors.append(f"missing features: {missing}")
    if extra:
        errors.append(f"unexpected features: {extra}")
    values: dict[str, float] = {}
    for n in names:
        if n not in features:
            continue
        v = features[n]
        if isinstance(v, bool) or not isinstance(v, Real):
            errors.append(f"{n} must be a number, not {type(v).__name__}")
            continue
        v = float(v)
        if not math.isfinite(v):
            errors.append(f"{n} must be finite (NaN/inf is not accepted; missing data is not imputed)")
            continue
        lo, hi = RANGES.get(n, (-math.inf, math.inf))
        if not lo <= v <= hi:
            errors.append(f"{n}={v} outside plausible range [{lo}, {hi}]")
        if n in WHOLE_NUMBER and v != int(v):
            errors.append(f"{n} must be a whole number")
        values[n] = v
    if not errors:
        has = values.keys()
        if {"max_temperature_c", "min_temperature_c"} <= has and values["max_temperature_c"] < values["min_temperature_c"]:
            errors.append("max_temperature_c cannot be below min_temperature_c")
        if {"temp_range_c", "max_temperature_c", "min_temperature_c"} <= has and not math.isclose(
                values["temp_range_c"], values["max_temperature_c"] - values["min_temperature_c"], abs_tol=0.02):
            errors.append("temp_range_c must equal max_temperature_c - min_temperature_c")
        if {"severe_heatwave_days_T", "heatwave_days_T"} <= has and values["severe_heatwave_days_T"] > values["heatwave_days_T"]:
            errors.append("severe_heatwave_days_T cannot exceed heatwave_days_T")
        if {"flood_years_last5", "flood_years_last10"} <= has and values["flood_years_last5"] > values["flood_years_last10"]:
            errors.append("flood_years_last5 cannot exceed flood_years_last10")
    return values, errors


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _score(bundle: dict, values: Mapping[str, float]) -> tuple[float, float]:
    from ml.hazard_training import apply_platt  # tiny pure function; no training at inference

    row = pd.DataFrame([[values[f] for f in bundle["features"]]], columns=bundle["features"])
    x = bundle["scaler"].transform(row.to_numpy())
    net = bundle["net"]
    raw = float(net.predict_proba(x)[0, list(net.classes_).index(1)])
    cal = float(apply_platt(np.array([raw]), bundle["platt"]["a"], bundle["platt"]["b"])[0])
    return raw, cal


def _explain(bundle: dict, card: dict, values: Mapping[str, float], use_cal: bool, base: float, top: int = 5) -> dict:
    """Occlusion: replace one input by its training median and see how the score moves."""
    med = bundle["training_medians"]
    effects = []
    for f in bundle["features"]:
        alt = dict(values)
        alt[f] = med[f]
        raw, cal = _score(bundle, alt)
        effects.append({"feature": f, "value": values[f], "training_median": med[f],
                        "effect_on_score": round(base - (cal if use_cal else raw), 4)})
    effects.sort(key=lambda e: -abs(e["effect_on_score"]))
    return {
        "method": "occlusion: change in the model output when one input is replaced by its training median",
        "approximate": True,
        "note": ("Local, one-feature-at-a-time and non-causal; interactions are ignored. A positive effect "
                 "means this input pushes the output up relative to a typical district-year."),
        "top_factors": effects[:top],
        "global_importance": {"method": card["global_feature_importance"]["method"],
                              "top": card["global_feature_importance"]["values"][:5]},
    }


def _base(hazard: str) -> dict[str, Any]:
    s = SPECS[hazard]
    return {"hazard": hazard, "target": s.target_definition, "geography": s.geography, "horizon": s.horizon}


def predict_hazard(hazard: str, features: Any, explain: bool = True,
                   model_dir: Path | str = MODEL_DIR, _years: tuple[int, int] | None = None) -> dict[str, Any]:
    """Predict one hazard from an explicit feature record (see ml/ML_MODEL_CONTRACT.md)."""
    if not isinstance(hazard, str):
        return {"hazard": None, "status": "invalid_input", "prediction": None,
                "errors": [f"hazard must be a string; use one of {list(HAZARDS) + list(UNSUPPORTED)}"]}
    if hazard in UNSUPPORTED:
        why, path = UNSUPPORTED[hazard]
        return {"hazard": hazard, "status": "unsupported", "reason": why, "recovery_path": path,
                "prediction": None}
    if hazard not in HAZARDS:
        return {"hazard": hazard, "status": "invalid_input", "prediction": None,
                "errors": [f"unknown hazard; use one of {list(HAZARDS) + list(UNSUPPORTED)}"]}
    values, errors = validate_features(hazard, features)
    if errors:
        return {**_base(hazard), "status": "invalid_input", "errors": errors, "prediction": None}
    try:
        bundle, card = _load(hazard, str(model_dir))
    except Exception:
        return {**_base(hazard), "status": "unavailable", "prediction": None,
                "reason": "trained model artifacts not found or unreadable",
                "recovery": f"python -m ml.hazard_datasets --build && python -m ml.hazard_training --hazard {hazard}"}

    raw, cal = _score(bundle, values)
    ok = _calibration_ok(card)
    use = cal if ok else raw
    # The threshold lives on the calibrated-probability scale, so a class is produced only when
    # calibration is accepted; otherwise no class is claimed (a raw score is not compared with it).
    pred = int(cal >= bundle["threshold"]) if ok else None
    labels = CLASS_LABELS[hazard]
    result: dict[str, Any] = {
        **_base(hazard), "status": "ok", "model_version": card["model_version"],
        "prediction": {
            "predicted_class": pred,
            "predicted_label": None if pred is None else labels[pred],
            "predicted_class_note": ("class derived from the calibrated probability and the validation-chosen threshold"
                                     if ok else "withheld: the validation-year calibration check was not acceptable, so no "
                                                "class is claimed; use raw_model_score only as a ranking score"),
            "raw_model_score": round(raw, 4),
            "calibrated_probability": round(cal, 4) if ok else None,
            "calibration_note": _calibration_note(card, ok),
            "decision_threshold": bundle["threshold"] if ok else None,
            "decision_threshold_note": ("chosen on validation years" if ok else
                                        "withheld together with the probability because the validation-year "
                                        "calibration check failed"),
            "risk_band": None,
            "risk_band_status": card["risk_band_status"],
        },
        "confidence": {"data_coverage": "complete", "missing_inputs": [],
                       "note": ("Risk score, data coverage and model reliability are different things; "
                                "see model_reliability for held-out performance.")},
        "model_reliability": {k: card["test_metrics"][k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "ece", "prevalence")},
        "training_window": card["training_window"],
        "label_reliability": card["label_reliability"],
        "limitations": card["limitations"] + ["Annual district-level prediction; not a travel-date or month forecast."],
    }
    if _years:
        result["feature_year"], result["target_year"] = _years
    if explain:
        result["explanation"] = _explain(bundle, card, values, ok, use)
    return result


def predict_for_district(hazard: str, district_code: int, feature_year: int | None = None, explain: bool = True,
                         data_dir: Path | str = DEFAULT_DATA_DIR, model_dir: Path | str = MODEL_DIR) -> dict[str, Any]:
    """Build the feature record from processed data, then predict.

    `feature_year` is the last fully observed year T (the prediction is for T+1).
    If omitted, the latest year with a complete record is used.
    """
    if not isinstance(hazard, str) or hazard in UNSUPPORTED or hazard not in HAZARDS:
        return predict_hazard(hazard, {})
    if isinstance(district_code, bool) or not isinstance(district_code, (int, np.integer)):
        return {**_base(hazard), "status": "invalid_input", "errors": ["district_code must be an integer"], "prediction": None}
    if feature_year is not None and (isinstance(feature_year, bool) or not isinstance(feature_year, (int, np.integer))):
        return {**_base(hazard), "status": "invalid_input", "errors": ["feature_year must be an integer"], "prediction": None}
    try:
        frame = _feature_frame(hazard, str(Path(data_dir)))
    except Exception:
        return {**_base(hazard), "status": "unavailable", "prediction": None,
                "reason": "processed input data could not be read"}
    rows = frame[frame["census_district_code"] == int(district_code)]
    if rows.empty:
        return {**_base(hazard), "status": "unavailable", "prediction": None,
                "confidence": {"data_coverage": "none", "missing_inputs": ["all"]},
                "reason": (f"District {int(district_code)} has no complete feature record for this hazard "
                           "(not covered by the hazard's label source, or inputs missing). Nothing is imputed.")}
    if feature_year is None:
        row = rows.sort_values("feature_year").iloc[-1]
    else:
        pick = rows[rows["feature_year"] == int(feature_year)]
        if pick.empty:
            have = sorted(int(y) for y in rows["feature_year"].unique())
            return {**_base(hazard), "status": "unavailable", "prediction": None,
                    "confidence": {"data_coverage": "none", "missing_inputs": [f"feature year {int(feature_year)}"]},
                    "reason": f"No complete feature record for feature year {int(feature_year)}; available years: {have[0]}-{have[-1]}."}
        row = pick.iloc[0]
    feats = {f: float(row[f]) for f in SPECS[hazard].features}
    out = predict_hazard(hazard, feats, explain=explain, model_dir=model_dir,
                         _years=(int(row["feature_year"]), int(row["target_year"])))
    out["district"] = {"census_district_code": int(row["census_district_code"]), "state_name": str(row["state_name"])}
    out["input_features"] = feats
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Hazard ANN inference (prints JSON)")
    p.add_argument("--list", action="store_true", help="show support status of all hazards")
    p.add_argument("--hazard")
    p.add_argument("--district-code", type=int)
    p.add_argument("--feature-year", type=int)
    p.add_argument("--features-json", help="JSON object with the feature record instead of --district-code")
    p.add_argument("--no-explain", action="store_true")
    a = p.parse_args(argv)
    if a.list:
        out = list_hazards()
    elif a.features_json:
        out = predict_hazard(a.hazard, json.loads(a.features_json), explain=not a.no_explain)
    elif a.hazard and a.district_code is not None:
        out = predict_for_district(a.hazard, a.district_code, a.feature_year, explain=not a.no_explain)
    else:
        p.error("use --list, or --hazard with --district-code or --features-json")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out.get("status", "ok") in ("ok",) or a.list else 1


if __name__ == "__main__":
    raise SystemExit(main())
