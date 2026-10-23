# Heatwave ANN Inference Handoff

## 1. Status and version

This handoff covers the first heatwave ANN inference integration.

> **LEGACY (v1) - superseded.** This is the first heatwave-only ANN experiment (heatwave-ann-mlp-v1).
> The authoritative ML interface is now `ml/hazard_inference.py` (see `ml/ML_MODEL_CONTRACT.md`,
> `ml/ANN_DESIGN_DECISION.md`). v1 used a different split, no calibration and a test set that was
> inspected during audit. It is kept as evidence and regression-tested, and the destination summary no
> longer calls it. It is optional and experimental, not the product's centre.

- Model identifier for integration: `heatwave-ann-mlp-v1`
- Selected model: feed-forward ANN / `MLPClassifier`
- Candidate selected using validation F1: `ANN_64_32`
- Input feature count: 14
- Layer sizes: 14 -> 64 -> 32 -> 1
- Hidden activation: ReLU
- Output activation: logistic sigmoid
- Optimizer: Adam
- L2 regularization parameter: 0.01
- Initial learning rate: 0.001
- Random seed used during training: 42
- Saved model artifacts were committed in `64a1d1d`.
- The reusable inference module was committed in `2a53550`.

The version identifier above is a project integration label. It is not a
version string embedded in the serialized estimator.

## 2. Files

Inference entry point:

    from ml.heatwave_inference import predict_heatwave

Implementation:

    ml/heatwave_inference.py

Saved artifacts:

    ml/models/heatwave/feedforward_ann_mlp.joblib
    ml/models/heatwave/heatwave_feature_scaler.joblib

The model and scaler are loaded from paths relative to the repository.
The model is not retrained when a prediction is requested.

## 3. Input contract

`predict_heatwave(features)` accepts a mapping containing exactly these
14 numeric features, all describing the same district and feature year T.

| Feature | Meaning |
|---|---|
| `latitude` | Representative district latitude in degrees |
| `longitude` | Representative district longitude in degrees |
| `population_2011` | District population from Census 2011 |
| `avg_temperature_c` | Annual mean temperature for year T, in C |
| `max_temperature_c` | Maximum daily maximum temperature recorded in year T, in C |
| `min_temperature_c` | Minimum daily minimum temperature recorded in year T, in C |
| `avg_humidity_pct` | Annual mean relative humidity, in percent |
| `total_rainfall_mm` | Annual accumulated rainfall, in mm |
| `rainy_days` | Number of days with at least 1 mm precipitation |
| `heatwave_days_T` | Heatwave days in year T, using the training threshold |
| `severe_heatwave_days_T` | Severe heatwave days in year T, using the training threshold |
| `has_heatwave_T` | 1 if `heatwave_days_T` > 0; otherwise 0 |
| `temp_range_c` | `max_temperature_c` - `min_temperature_c` |
| `rain_intensity` | `total_rainfall_mm` / (`rainy_days` + 1) |

Do not pass identifiers, year metadata, target labels, or split columns as
predictors. The function rejects missing or extra features, non-numeric or
non-finite values, and several inconsistent feature combinations.

The backend must not fill missing values with invented zeros. It should
return an appropriate data-unavailable response if a valid feature record
cannot be obtained.

## 4. Example input

This mapping was taken from a genuine row in the prepared dataset for
feature year 2011. It is a schema smoke-test example, not a current forecast
or a representative demo destination.

    {
      "latitude": 34.5775790190758,
      "longitude": 74.2890328628661,
      "population_2011": 870354.0,
      "avg_temperature_c": 8.75,
      "max_temperature_c": 31.28,
      "min_temperature_c": -22.73,
      "avg_humidity_pct": 58.42,
      "total_rainfall_mm": 746.51,
      "rainy_days": 112.0,
      "heatwave_days_T": 0.0,
      "severe_heatwave_days_T": 0.0,
      "has_heatwave_T": 0.0,
      "temp_range_c": 54.01,
      "rain_intensity": 6.6063
    }

Example call, from the repository root:

    from ml.heatwave_inference import predict_heatwave

    result = predict_heatwave(features)

The tested example returned class 0, labelled `no_heatwave_occurrence`.
Its raw positive-class score printed as 0.0 after rounding to six decimals.

## 5. Output contract

The function returns a dictionary containing:

- `model`: model name
- `task`: `next_calendar_year_heatwave_occurrence`
- `predicted_class`: 0 or 1
- `predicted_label`: `no_heatwave_occurrence` or `heatwave_occurrence`
- `raw_positive_score`: raw positive-class model score, rounded to six decimals
- `classification_threshold`: 0.5
- `score_interpretation`: warning that the score is uncalibrated
- `target_definition`: annual heatwave target definition
- `limitations`: list of scope and calibration limitations

The backend should retain the selected feature year T and derive target year
T+1 from the underlying data record. The current inference function does not
receive or return these year fields itself.

Suggested API response envelope, to be implemented by the backend:

    {
      "model_version": "heatwave-ann-mlp-v1",
      "feature_year": 2011,
      "target_year": 2012,
      "prediction": {
        "...": "fields returned by predict_heatwave()"
      }
    }

This envelope is a recommendation for integration, not an existing API
endpoint or the direct return value of the inference function.

## 6. Prediction scope

The target is whether at least one day with maximum temperature at or above
45 C occurs during the next calendar year T+1, according to the prepared
training target.

This is district-level annual classification. It is not:

- A month-specific travel forecast
- A daily weather forecast
- An individual health-risk assessment
- An emergency warning system

The ANN's raw outputs have shown calibration problems. Do not describe its
raw score as a reliable real-world probability, convert it to a percentage,
or map it to high/medium/low risk categories without a separate, justified
calibration and thresholding procedure.

The current prepared dataset uses feature years 2011-2023. A destination
and travel date alone are not sufficient inputs. The backend needs a valid
feature record built with definitions consistent with the training schema.

## 7. Backend responsibilities

The backend should:

1. Validate the request and resolve the destination to a canonical district.
2. Determine the available feature year and target year.
3. Retrieve a compatible feature record for that district and year.
4. Construct the exact 14-feature mapping.
5. Call `predict_heatwave(features)`.
6. Include the feature year, target year, model identifier, data availability
   and limitations in its response.
7. Handle invalid input, missing artifacts, and unavailable feature data
   with controlled API errors. Do not expose internal stack traces.

This ML module does not currently implement destination search, district
resolution, feature-data retrieval, FastAPI routes, database access,
calibration, or explainability.

## 8. Dependencies and tests

The locally tested environment used Python 3.13.14. See
`ml/requirements-inference.txt` for the recorded package versions.

Run from the repository root:

    python3.13 -m unittest ml.test_heatwave_inference -v
    python3.13 ml/test_heatwave_dataset.py
    python3.13 ml/test_heatwave_models.py

The inference test suite contains 10 tests; the dataset and model suites
contain 10 and 8 tests respectively.

## 9. Dependency reconciliation warning

The current `requirements-data.txt` pins NumPy 2.2.6 and pandas 2.3.3.
The environment that passed the inference tests used NumPy 2.3.2 and
pandas 2.3.1.

These version declarations are not yet unified. Do not install both
dependency files into one environment blindly. Reconcile the project's
shared dependency set before integrated deployment.