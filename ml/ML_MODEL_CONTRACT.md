# ML Model Contract (for the backend: Rajdeep)

One interface, one file: `ml/hazard_inference.py`. Do **not** re-implement feature
building, scaling, calibration or thresholds in FastAPI; call these functions.

```python
from ml.hazard_inference import list_hazards, predict_for_district, predict_hazard

list_hazards()                                   # support status of all 5 hazards
predict_for_district("flood", 536)               # latest complete year (T) -> predicts T+1
predict_for_district("malaria", 146, feature_year=2022)
predict_hazard("heatwave", {...13 features...})  # caller-supplied feature record
```

Install: `python -m pip install -r ml/requirements-inference.txt` (Python 3.13, scikit-learn 1.7.2).
Run from the repository root (paths are resolved from the module location, not the CWD).
Every function returns a JSON-serialisable `dict`; none raises for bad input (including non-string hazards,
missing files and unsupported hazards) and none returns local paths.

## 0. What this module does, does not do, and how it differs from destination analysis

**Does:** next-calendar-year, per-district predictions from three saved ANNs (heatwave, flood, malaria), with
input validation, calibration rule, explanation and honest status values.

**Does not:** forecast for a travel date, week or month (no sub-annual data exists); produce a combined or overall
risk score; produce Low/Medium/High bands; predict dengue or chikungunya; diagnose illness or declare a
destination safe. It is not a FastAPI service, a database or a frontend; those belong to the backend and frontend owners.

| | `ml.destination_analysis.build_destination_summary` | `ml.hazard_inference.predict_for_district` / `predict_hazard` |
|---|---|---|
| Purpose | what is **recorded** for a destination (history, coverage, provenance) | what an **ANN estimates** for the next calendar year |
| Input | district code or name+state, optional dates | hazard name, district code (+ optional feature year) or an explicit feature record |
| Output | weather, heatwave-threshold, malaria, dengue, flood records + precautions | one hazard's score/probability, reliability, explanation |
| Uses models | no (optional `include_experimental_ann` asks the interface above) | yes |
| Dates | selects the calendar years they overlap; values stay annual | none; `feature_year` T means "predict T+1" |
| Unsupported hazards | n/a (dengue is shown as state-level records) | `status: "unsupported"`, no prediction |

Show both **separately**: historical records are observations; ANN output is an experimental estimate.

## 1. Hazards

| Hazard | `status` | Model version | Target (year T+1) | Geography |
|---|---|---|---|---|
| `heatwave` | supported | `heatwave-ann-mlp-v2` | >= 1 day with modelled Tmax >= 45 degC (not IMD-declared) | district (640) |
| `flood` | supported_with_caveats | `flood-ann-mlp-v2` | inventory records >= 1 flood event (presence-only; negative = none recorded) | district (502 mapped) |
| `malaria` | supported | `malaria-ann-mlp-v2` | reported incidence in the top quartile of reporting districts | district (517) |
| `dengue` | **unsupported** | - | - | state-level labels, 2021-2025 only |
| `chikungunya` | **unsupported** | - | - | no data |

All predictions are **annual** (next calendar year). There is no month- or travel-date model;
a trip in year Y is served with `feature_year = Y - 1` (if available). Outputs for hazards are
independent; there is no combined score.

## 2. `status` values (every result has one)

| `status` | Meaning | `prediction` |
|---|---|---|
| `ok` | prediction made | object |
| `unsupported` | no valid training labels (`reason`, `recovery_path` given) | `null` |
| `unavailable` | model artifacts missing, or no complete input record for that district/year (`reason`; nothing imputed) | `null` |
| `invalid_input` | bad arguments or feature record (`errors` list) | `null` |

Suggested HTTP mapping: ok 200, invalid_input 422, unsupported 200 with the status shown (or 501),
unavailable 503/404 depending on `reason`.

## 3. `predict_for_district(hazard, district_code, feature_year=None, explain=True, data_dir=..., model_dir=...)`

`district_code` is the integer Census-2011 code (`data/reference/district_master.csv`).
`feature_year` is the last fully observed year T; the result also contains
`feature_year`, `target_year = T+1` and the `input_features` used.
It reads the processed tables with the same builders used for training (a test proves the
features equal the training rows).

## 4. `predict_hazard(hazard, features, explain=True, model_dir=...)` input contract

`features` is a mapping with **exactly** the names below (no missing, no extra, finite numbers,
no booleans, plausible ranges, whole numbers where stated). Missing values are rejected, never filled.

| Hazard | Features (order is irrelevant; units) |
|---|---|
| all | `latitude` (deg), `longitude` (deg), `log_population` (= ln(1 + Census 2011 population)) |
| heatwave | `avg_temperature_c`, `max_temperature_c`, `min_temperature_c`, `temp_range_c` (= max - min), `avg_humidity_pct` (0-100), `total_rainfall_mm`, `rainy_days` (whole), `heatwave_days_T`, `severe_heatwave_days_T`, `heatwave_days_Tm1` (whole days; severe <= total) |
| flood | `avg_humidity_pct`, `total_rainfall_mm`, `rainy_days`, `avg_temperature_c`, `max_temperature_c`, `recorded_flood_events_T`, `recorded_flood_events_Tm1`, `flood_years_last5` (0-5), `flood_years_last10` (0-10, >= last5) |
| malaria | all 7 weather fields as heatwave, plus `log1p_incidence_T`, `log1p_incidence_Tm1` (= ln(1 + cases per 100,000, 2011 population)), `incidence_rank_pct_T` (0-1), `high_burden_T` (0 or 1) |

Authoritative lists: `ml/data/hazards/<hazard>/feature_dictionary.csv` and each `model_card.json`.

## 5. Output (status `ok`)

```text
hazard, target, geography, horizon, status, model_version
prediction:
  predicted_class            0/1, or null when calibration is not accepted (see probability rule)
  predicted_label            heatwave_threshold_day_(not_)predicted | recorded_flood_(not_)predicted |
                             high_reported_incidence_(not_)predicted, or null (wording mirrors the target)
  predicted_class_note       why the class is shown or withheld
  raw_model_score            0-1, the ANN sigmoid output (uncalibrated ranking score)
  calibrated_probability     number or null
  calibration_note           why it is shown or withheld
  decision_threshold         number or null (chosen on validation years)
  risk_band                  always null
  risk_band_status           "unavailable: no validated thresholds ..."
confidence: {data_coverage, missing_inputs, note}   # coverage is separate from risk and reliability
model_reliability: {pr_auc, roc_auc, f1, recall, precision, brier, ece, prevalence}   # held-out test years
training_window: {train/validation/test target years}
label_reliability, limitations[]
explanation (if explain=True): {method, approximate: true, top_factors[5], global_importance}
feature_year, target_year, district, input_features        # predict_for_district only
```

**Probability rule.** `calibrated_probability`, `decision_threshold`, `predicted_class` and `predicted_label` are
returned only if the model card's validation-year calibration check passed (Platt fitted on earlier validation years,
ECE <= 0.10 on the last validation year; test years take no part). Today: malaria shown (ECE 0.056); **heatwave
(0.173) and flood (0.149) withheld**, so for them all four fields are `null` and only `raw_model_score` is returned.
The threshold lives on the calibrated-probability scale, so no class is claimed without it. Never show a
withheld raw score as a percentage, a probability or a yes/no prediction (`calibration_note` explains why).
`model_reliability.ece` is the held-out test-year ECE, for information only; it does not drive this rule.

**Risk bands:** do not display Low/Medium/High for any ML output.

## 6. Explanation fields

`top_factors[i] = {feature, value, training_median, effect_on_score}` - how much the output changes when
that input is replaced by its training median (occlusion). `global_importance` is permutation importance
on validation years. Both are approximate and non-causal and only name real model inputs.

## 7. Examples (real output, trimmed)

`predict_for_district("malaria", 146, explain=False)`:
```json
{"hazard": "malaria", "status": "ok", "model_version": "malaria-ann-mlp-v2",
 "prediction": {"predicted_class": 0, "predicted_label": "high_reported_incidence_not_predicted",
   "raw_model_score": 0.01, "calibrated_probability": 0.0217, "decision_threshold": 0.17,
   "risk_band": null},
 "feature_year": 2023, "target_year": 2024}
```
Unsupported: `predict_hazard("dengue", {})` ->
`{"hazard": "dengue", "status": "unsupported", "reason": "...", "recovery_path": "...", "prediction": null}`.
Invalid: `predict_hazard("flood", {"latitude": 17.4})` -> `{"status": "invalid_input", "errors": ["missing features: [...]"], "prediction": null}`.
Uncovered district: `predict_for_district("malaria", 327)` (Darjiling has no malaria rows) -> `status: "unavailable"`.

## 7b. Copy-paste backend example (handles every status)

This block is executed by `ml/test_contract_examples.py`, so it is known to run against the saved models.

<!-- handoff-example -->
```python
from ml.hazard_inference import list_hazards, predict_for_district, predict_hazard


def http_status(result: dict) -> int:
    """Suggested HTTP status for a result of predict_for_district / predict_hazard."""
    status = result["status"]
    if status == "ok":
        return 200
    if status == "invalid_input":
        return 422          # fix the request; see result["errors"]
    if status == "unsupported":
        return 501          # no valid model for this hazard; see result["reason"], result["recovery_path"]
    if status == "unavailable":
        # no complete input record for this district/year -> 404; model files or data unreadable -> 503
        return 404 if result.get("confidence", {}).get("data_coverage") == "none" else 503
    return 500


def hazard_card(hazard: str, district_code: int, feature_year: int | None = None) -> tuple[int, dict]:
    """What an endpoint can return to the frontend for one hazard."""
    result = predict_for_district(hazard, district_code, feature_year, explain=True)
    code = http_status(result)
    if code != 200:
        return code, {"hazard": hazard, "status": result["status"],
                      "message": result.get("reason") or result.get("errors"), "prediction": None}
    p = result["prediction"]
    return 200, {
        "hazard": hazard, "model_version": result["model_version"],
        "based_on_year": result["feature_year"], "predicts_year": result["target_year"],
        "target": result["target"],
        # Show a probability/label ONLY when they are not null. Otherwise show nothing but the ranking score.
        "probability": p["calibrated_probability"], "label": p["predicted_label"],
        "raw_score_for_ranking_only": p["raw_model_score"], "note": p["calibration_note"],
        "risk_band": None,                      # never invent Low/Medium/High
        "reliability": result["model_reliability"], "limitations": result["limitations"],
        "top_factors": result["explanation"]["top_factors"][:3],
    }


# Usage (repository root as working directory):
status_code, body = hazard_card("malaria", 146)       # 200: probability present
status_code, body = hazard_card("flood", 536)         # 200: probability is None (withheld), score only
status_code, body = hazard_card("dengue", 146)        # 501: unsupported, no prediction
status_code, body = hazard_card("malaria", 327)       # 404: Darjiling has no malaria records
status_code, body = hazard_card("tsunami", 146)       # 422: unknown hazard
print(list_hazards()["heatwave"]["calibrated_probability_available"])   # False
```

Run the same thing from a shell: `python -m ml.hazard_inference --hazard flood --district-code 536`
and `python -m ml.hazard_inference --list` (exit code 0 only for `ok`).

## 8. Artifacts and versioning

`ml/models/hazards/<hazard>/ann_bundle.joblib` (network, scaler, calibration, threshold, feature
order, training medians) and `model_card.json` (architecture, windows, metrics, importance, source file
hashes, library versions). Pickles are tied to the recorded scikit-learn version. If the artifacts
are missing the status is `unavailable`; retrain with
`python -m ml.hazard_datasets --build && python -m ml.hazard_training --hazard all`.

## 9. Limits to show the user

Annual and district-level only; no month/date model; heatwave is a temperature-threshold indicator,
flood is presence-only inventory data, malaria is a relative reported-burden ranking; held-out metrics
in `model_reliability` show the ANN is **not** uniformly better than simpler models
(`ml/EVALUATION_REPORT.md`); never present a result as medical advice, a guarantee of safety or an
official warning.
