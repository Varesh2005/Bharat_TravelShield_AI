# Data-Analysis Layer Handoff

Audience: backend/frontend integrators. For the **ANN models** use `ml/ML_MODEL_CONTRACT.md` (the other contract
for this module); this file covers the **data summary** only. Field-level contract:
`data/schemas/destination_summary_contract.md`. Real example:
`docs/samples/destination_summary_agra_2024-10.json`. Data quality:
`ml/DATA_QUALITY_REPORT.md`. Work status: `ml/IMPLEMENTATION_STATUS.md`.

## 1. Purpose

Bharat TravelShield is an **information-aggregation** product. This layer
puts the scattered weather, disease and disaster tables behind one function
that returns, for a destination district, what is actually recorded, at the
resolution the source supports, with provenance, coverage and limitations.

It does **not** forecast, score or rank risk. There is no overall risk
number, no weighting and no "safe/unsafe" label. The hazard ANNs are an
optional experiment (section 6) and are off by default.

## 2. Architecture and data flow

```
data/reference/district_master.csv ----+
data/processed/weather/...year...csv --+
data/processed/disaster/heatwave/...csv +--> ml/destination_analysis.py --> JSON summary
data/processed/disease/malaria/...csv --+      DataStore (load once, validate, cache)
data/processed/disease/dengue/...csv ---+      resolve_destination -> parse_period
data/processed/disaster/flood/...csv ---+      one builder per hazard -> precautions
                                               (optional) ml/hazard_inference.py

ml/data_audit.py  --> ml/DATA_QUALITY_REPORT.md   (independent, repeatable audit)
```

* Pure local files, pandas only, no network, no database, no secrets.
* Each source table is read once per process (`get_store`) and validated
  (file present, readable, required columns, no duplicate keys). A bad source
  only turns **its own** hazard into `source_unavailable`.
* No local paths are placed in any output.

## 3. Functions (`from ml.destination_analysis import ...`)

| Function | Use |
|---|---|
| `build_destination_summary(district_code=None, district_name=None, state_name=None, start_date=None, end_date=None, include_experimental_ann=False, ann_feature_year=None, data_dir=None)` | Returns the summary dict; raises `DestinationAnalysisError` subclasses for bad input. |
| `safe_destination_summary(**same)` | Same, but returns `{"error": {...}}` instead of raising. |
| `resolve_destination(store, ...)`, `parse_period(start, end)` | Reusable building blocks. |
| `get_store(data_dir=None)`, `clear_store_cache()` | Cached table access (e.g. call `clear_store_cache()` after data files change). |
| Exceptions | `InvalidRequestError`, `DestinationNotFoundError`, `AmbiguousDestinationError` (has `.extra["candidates"]`), `DataUnavailableError`; all have `.code`, `.message`, `.to_dict()`. |

### Backend example (FastAPI-style; the backend is not part of this repo yet)

```python
from ml.destination_analysis import (
    build_destination_summary, DestinationAnalysisError,
)

def destination_endpoint(district_code: int | None = None, district_name: str | None = None,
                         state: str | None = None, start: str | None = None, end: str | None = None):
    try:
        return build_destination_summary(
            district_code=district_code, district_name=district_name, state_name=state,
            start_date=start, end_date=end,
        )
    except DestinationAnalysisError as exc:
        status = {"destination_not_found": 404, "ambiguous_destination": 409,
                  "data_unavailable": 503}.get(exc.code, 422)
        return status, exc.to_dict()       # {"error": {"code", "message", ...}}
```

Recommended request shape: `district_code` (preferred, unambiguous) **or**
`district_name` + `state_name`; optional ISO `start_date`/`end_date` (both or
neither). Because 6 district names exist in two states (e.g. Bilaspur,
Aurangabad), a bare name can return `ambiguous_destination` with `candidates`
for the UI to offer.

Run once at startup with the repository root as working directory or rely on
the module's own `ROOT`-relative paths (it does not depend on the CWD).

## 4. What the output means (reading rules for the frontend)

1. **Annual only.** Every source is annual (or coarser). Dates select the
   calendar years they overlap; show `query_period.note` whenever
   `spans_partial_year` is true. Never present annual values as the weather
   or disease burden of the travel dates.
2. **Missing is not zero.** `records` hold only years the source has;
   `null` fields are unknown; `status` + `status_reason` say why a block is
   empty. Show "no data", not "0" or "low".
3. **Zero is real only when present** in `records` (e.g. `heatwave_days: 0`).
4. **Dengue is state-level context.** Display it as "State-level (X)",
   never as the district's own count. Use `dengue.state_used` and caveats.
   Show each dengue record's `data_quality`: `verification_pending`
   (currently West Bengal 2024, value 441, label `West Bengal*`) must be
   displayed with its source value and a "not yet verified" note, and
   `not_reported` (blank/dash) as "not reported", never 0. Leh and Kargil
   never show Jammu & Kashmir figures. **Anything that scores, ranks or
   feeds a model must use only records with
   `data_quality.usable_as_verified_input == true`.**
5. **Counts are not rates.** Do not rank or compare districts by case counts.
6. **Heatwave = threshold indicator** (NASA POWER Tmax >= 45 degC), not IMD declarations.
7. **Freshness.** Use `last_observed_year` and `coverage.source_year_range`
   per hazard (malaria ends 2024 and that year is through April; dengue 2026
   is provisional; flood ends 2023).
8. **Precautions** are generic, optional to display, and must keep their
   `kind` label ("not official advice").

## 5. Error and edge-case behaviour

| Situation | Behaviour |
|---|---|
| Unknown / ambiguous destination, bad code type | `destination_not_found` (with `suggestions`) / `ambiguous_destination` (with `candidates`) / `invalid_request` |
| Bad date text, impossible date, reversed range, only one date | `invalid_request` |
| Dates entirely outside every source (e.g. 2035) | Normal summary; each hazard `no_records_in_period`, no precautions |
| A hazard file missing/corrupt/duplicated keys | That hazard `source_unavailable`; others unaffected |
| District reference file missing | `data_unavailable` |
| Destination absent from a source | `no_records_for_destination` (unknown, not zero) |

## 6. Optional experimental hazard ANNs

Off by default (`include_experimental_ann=True` / `--with-ann` to request). The data summary is
unchanged by the models: every ANN problem (missing artifacts, missing scikit-learn, no complete
input record, unsupported hazard) only affects that hazard's entry under
`experimental_ann_predictions.hazards`, with a short, path-free `reason`.

The block calls the single ML interface `ml/hazard_inference.py` for heatwave, flood, malaria, dengue
and chikungunya (dengue and chikungunya return `unsupported` with the evidence and a recovery path).
Targets, features, split, baselines, calibration, explanation and limits are documented in
`ml/ANN_DESIGN_DECISION.md`, `ml/ML_MODEL_CONTRACT.md` and `ml/EVALUATION_REPORT.md`; do not restate
them here. In short: annual district-level, next-calendar-year targets; heatwave is a NASA POWER
threshold indicator (not an IMD declaration); raw scores are uncalibrated unless the contract says a
calibrated probability is present (currently only malaria; heatwave and flood are withheld); no risk bands; not a travel-date forecast.
`ann_feature_year` selects the last observed year T (default: the latest complete year per hazard).

The earlier heatwave v1 module (`ml/heatwave_inference.py`, `ml/INFERENCE_HANDOFF.md`) is legacy
evidence, no longer called by the summary.

## 7. Dependencies

The destination summary needs only `pandas` (declared in both requirement files); the data audit also reads
parquet files (`pyarrow`). The ANN block additionally needs the packages in `ml/requirements-inference.txt`
(scikit-learn, joblib, scipy). Do not install `requirements-data.txt` and
`ml/requirements-inference.txt` into one environment blindly: their numpy and
pandas pins differ (see `ml/INFERENCE_HANDOFF.md` section 9). All tests in
this repository were run in the environment described by
`ml/requirements-inference.txt` (Python 3.13).

## 8. Known data issues found during the audit

* The existing master feature table (`travelshield_district_year_features_2011_2024.csv`)
  joined dengue by exact normalised state name, so four states/UTs
  (Andaman & Nicobar, Arunanchal Pradesh, Dadara & Nagar Havelli, Daman &
  Diu) have no dengue values although the source has them, and the ten
  Census-2011 Telangana districts carry Andhra Pradesh values. The
  destination layer reads the source tables directly and handles both; the
  master table was **not** rebuilt (it feeds the ANN dataset, which does not
  use dengue). Fix `normalize_state` in
  `scripts/features/build_master_feature_dataset.py` before anyone relies on
  its dengue columns.
* Dengue decisions (confirmed, implemented): West Bengal 2024 is
  `verification_pending` via `data/reference/dengue_data_quality_flags.csv`
  (annotation only; the NCVBDC source file is untouched; the audit fails if the
  flagged value stops matching the source). Telangana districts use Telangana's
  own series. Jammu & Kashmir stays state-level and is never assigned to Leh or
  Kargil; blank/dash Ladakh values are `not_reported`. Still open: the meaning
  of the `*` and whether West Bengal 2024 is complete. When the data owner
  answers, update the flag's status/reason in that CSV (no code change).
* Malaria covers 517 of 640 districts, flood 502 of 640.
* The daily NASA POWER file is absent; month-specific weather is therefore not
  available. Recovery: copy `weather_daily_2011_2025.parquet` into
  `data/processed/weather/` from the teammate who has it, or regenerate (long
  API download) with `scripts/ingestion/download_nasa_power.py` followed by
  `scripts/cleaning/combine_weather_data.py`. A monthly summary table would
  then be a new, separate build step; it must not be derived from the annual
  table.

## 9. Reproduction (run from the repository root)

Use a Python 3.13 environment matching `ml/requirements-inference.txt`.

```bash
python -m pip install -r ml/requirements-inference.txt
python -m ml.data_audit                                  # writes ml/DATA_QUALITY_REPORT.md
python -m ml.data_audit --check                          # audit only; exit 1 on structural errors
python -m ml.destination_analysis --district-name Agra --state-name "Uttar Pradesh" --start-date 2024-10-01 --end-date 2024-10-31 --with-ann --output docs/samples/destination_summary_agra_2024-10.json
python -m unittest ml.test_destination_analysis ml.test_data_audit -v     # this layer's tests
powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1                   # everything, incl. the ANN suites
```

Rebuilding processed features (inputs already in `data/processed/`):

```bash
python scripts/features/build_disease_features.py
python scripts/features/build_master_feature_dataset.py
python scripts/validation/validate_master_feature_dataset.py
```

Only run the rebuild commands deliberately: they overwrite tracked data files.
Rebuilding the daily weather needs the long NASA POWER download described above.
