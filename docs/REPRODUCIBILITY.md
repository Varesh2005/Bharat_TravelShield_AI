# Reproducibility Guide

What can be run from this repository alone, what needs inputs that are **not**
in Git, and which committed files are inputs, outputs or evidence. Statements
marked *inferred* come from reading script input/output paths; those scripts
were not re-run for this guide.

## 1. Environments (one per workflow)

| Workflow | Requirements | Python | Notes |
|---|---|---|---|
| Destination summary, data audit, heatwave ANN/model tests | `ml/requirements-inference.txt` | 3.13 (tested with 3.13.14 recorded; 3.13 used for the verification run) | pandas 2.3.1, numpy 2.3.2, scikit-learn 1.7.2 |
| Data pipeline (ingestion, cleaning, features) | `requirements-data.txt` | not recorded in the repo | pandas 2.3.3, numpy 2.2.6 |

The two files pin different numpy/pandas versions; use **separate virtual
environments**. Do not merge them without testing. The pipeline scripts also
import `pdfplumber` and `geopandas`, which are **not declared in either
requirements file** (undeclared dependency gap; versions were not pinned here
because they could not be tested in this environment).

Setup (repository root):

```bash
python -m venv .venv-ml && .venv-ml/Scripts/activate      # Windows; use bin/activate elsewhere
python -m pip install -r ml/requirements-inference.txt
```

`.env.example` is an empty placeholder; no secrets or API keys are required by
any script (NASA POWER needs none).

## 2. Verification commands (all run from the repository root)

```bash
python -m unittest ml.test_destination_analysis ml.test_data_audit ml.test_heatwave_inference -v
python ml/test_heatwave_dataset.py
python ml/test_heatwave_models.py
python -m ml.data_audit --check
python -m ml.destination_analysis --district-name Agra --state-name "Uttar Pradesh" --start-date 2024-10-01 --end-date 2024-10-31
```

* `ml/test_heatwave_dataset.py` also reads the Git-ignored
  `ml/data/heatwave/heatwave_t_to_t1_dataset.csv` (a duplicate of the committed
  parquet). On a fresh clone create it first with
  `python ml/prepare_heatwave_dataset.py` (this rewrites the dataset files;
  review `git diff` afterwards).
* `ml/test_heatwave_models.py` needs all four model files in
  `ml/models/heatwave/`; they are committed.
* The destination summary and audit run with no model files at all.

## 3. Raw inputs that are not in the repository

`data/raw/` is Git-ignored and **absent from this checkout**. The processed
datasets below were built from these inputs and cannot be re-derived without them:

| Input expected at | Used by | Source (per scripts/reports) |
|---|---|---|
| `data/raw/2011_Dist.shp` (+ sidecar files) | `scripts/geography/*` | Census 2011 district boundaries |
| `data/raw/census_population_2011.xlsx` | `build_district_population.py` | Census of India 2011 |
| `data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf` | `scripts/ingestion/extract_malaria_*.py` | NCVBDC district-wise malaria |
| `data/raw/disease/dengue/ncvbdc_dengue_state_2021_2026_raw.html` | `build_dengue_state_dataset.py` | NCVBDC (https://ncvbdc.mohfw.gov.in/) |
| `data/raw/disaster/flood/India_Flood_Inventory_v3.csv` | `build_flood_*.py`, `check_flood_lgd_mapping.py` | India Flood Inventory v3 |
| `data/raw/disaster/heatwave/imd_heatwave_guidance.html` | `find_imd_heatwave.py` | IMD heat-wave criteria |
| `data/raw/weather/*.csv` (640 files) and `data/processed/weather/weather_daily_2011_2025.{csv,parquet}` | weather scripts | NASA POWER daily API (see `data/provenance/weather_nasa_power.md`) |

Redistribution terms of these sources were not reviewed; none are committed.
The raw files, the shapefile and the **daily weather file** exist (or existed)
on a teammate's machine and must be obtained from them or re-downloaded.

**Reproducibility blockers**

1. `weather_district_year_2011_2025.csv` and
   `heatwave_district_year_2011_2025.csv` are produced from the absent daily
   weather file (`scripts/features/build_weather_features.py`,
   `build_heatwave_dataset.py`). They are committed as canonical outputs but
   cannot be regenerated from this checkout.
2. All other processed datasets likewise need the raw files above.
3. Without daily weather, month- or travel-window weather cannot be produced;
   the destination summary reports annual values only and says so.

## 4. Pipeline order (*inferred* from script inputs/outputs)

1. **Geography:** `build_district_population.py`; `repair_district_geometry.py`
   (writes a repaired shapefile under `data/interim/`); `build_district_master.py`
   -> `data/reference/district_master.csv`; `validate_district_shapefile.py`,
   `check_*`/`inspect_*` are diagnostics.
2. **Weather:** `scripts/ingestion/download_nasa_power.py` (long API download) ->
   `scripts/validation/validate_nasa_power_weather.py` ->
   `scripts/cleaning/combine_weather_data.py` -> `scripts/features/build_weather_features.py`
   -> `scripts/features/build_heatwave_dataset.py`.
3. **Malaria:** `scripts/ingestion/extract_malaria_final.py` (and
   `extract_malaria_coordinate.py`) -> `scripts/cleaning/build_malaria_final.py` ->
   `scripts/validation/validate_final_malaria.py`, `validate_malaria_*`.
   `scripts/cleaning/build_final_malaria_dataset.py` writes a separate
   `*_canonical.csv` that is not part of the committed data.
4. **Dengue:** `scripts/ingestion/build_dengue_state_dataset.py`.
5. **Flood:** `scripts/validation/check_flood_lgd_mapping.py` ->
   `scripts/cleaning/build_flood_final.py` (final; `build_flood_district_year.py`
   is the earlier variant).
6. **Features:** `scripts/features/build_disease_features.py` ->
   `build_master_feature_dataset.py` -> `scripts/validation/validate_master_feature_dataset.py`;
   `create_ml_handoff.py` writes ML handoff files.
7. **ML (optional):** `ml/prepare_heatwave_dataset.py` -> `ml/train_heatwave_models.py`
   -> tests above. Retraining rewrites model artifacts and experiment reports.

The `scripts/ingestion/inspect_*`, `test_*` and `scripts/geography/inspect_*`
files are exploratory diagnostics written while developing the extraction; they
are kept as research evidence and are not required to rebuild the data.

## 5. File roles

* **Authoritative inputs to the application:** `data/reference/`,
  `data/processed/` (annual weather, heatwave, malaria, dengue, flood, master
  features), `data/reference/dengue_data_quality_flags.csv`.
* **Intermediate:** `data/processed/disease/*_features_*.csv` (consumed by
  `build_master_feature_dataset.py`).
* **Evidence / reports:** `data/reports/`, `data/provenance/`, `ml/experiments/`.
* **Generated, Git-ignored duplicate:** `ml/data/heatwave/heatwave_t_to_t1_dataset.csv`.
* Three already-committed generated files embed local Windows paths
  (`data/reports/disaster/heatwave/heatwave_build_report.txt`,
  `data/reports/disaster/flood_district_year_build_report.txt`,
  `ml/experiments/heatwave_experiment_results.json`). They were left unedited
  as build evidence; they are not read by any code.
