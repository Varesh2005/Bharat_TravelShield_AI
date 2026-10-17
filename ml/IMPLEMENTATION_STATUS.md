# Implementation Status

Branch `feature/ml`. Nothing is committed or pushed; all changes are in the working tree.
Test environment: Python 3.13, pandas 2.3.1, numpy 2.3.2, scikit-learn 1.7.2
(`ml/requirements-inference.txt`). Runner: `unittest` (no pytest in the project).

## Completed

| Item | Evidence |
|---|---|
| Repository audit; clean tree at start | `git status` was empty on `feature/ml` before edits |
| Data inventory and quality audit | `ml/data_audit.py`, generated `ml/DATA_QUALITY_REPORT.md`; real run: 0 errors, 17 warnings |
| Canonical data contract | `data/schemas/destination_summary_contract.md` |
| Destination analysis layer | `ml/destination_analysis.py`: resolution, period handling, 5 hazard blocks, provenance, coverage, precautions, optional ANN, CLI |
| Missing vs zero, annual vs monthly, state-level dengue | Implemented and covered by tests (see below) |
| Tests | 44 (destination) + 12 (audit) new; 10 existing inference tests |
| Backend handoff | `ml/DATA_ANALYSIS_HANDOFF.md` (imports, example, error mapping, reading rules) |
| Real-data sample | `docs/samples/destination_summary_agra_2024-10.json` (real data, ANN included) |
| ANN kept optional and documented | handoff section 6; summary works without model files (tested with failing/absent model) |
| README | `README.md` (was empty) |
| `.gitignore` | removed a duplicated block of two rules |

## Dengue data-quality decisions (implemented)

* West Bengal 2024 (`West Bengal*`, 441): `verification_pending`, value and label unchanged, `usable_as_verified_input: false`,
  excluded from dengue precaution support. Annotation file `data/reference/dengue_data_quality_flags.csv`;
  dictionary `data/schemas/dengue_data_dictionary.csv`. Raw source dataset not modified.
* Telangana districts use Telangana's own state-level series, labelled `state_level_context_not_district_specific`.
* J&K stays state-level; never assigned to Leh/Kargil. Blank/dash cells (Ladakh or any state) are `not_reported`, never zero.
* Still unconfirmed: meaning of the `*`; completeness of West Bengal 2024.

## Test results (observed)

| Command | Result |
|---|---|
| `python -m unittest ml.test_destination_analysis ml.test_data_audit ml.test_heatwave_inference` | 80 tests, OK |
| `python ml/test_heatwave_dataset.py` | 10 checks passed |
| `python ml/test_heatwave_models.py` | 8 checks passed |
| `python -m ml.data_audit --check` on the real repository | exit 0 (PASSED_WITH_WARNINGS, 0 errors, 19 warnings) |

The existing ANN suites passed unchanged before any edit (baseline).

## Partial / limitations

* Weather and heatwave are annual: no travel-window weather. No monthly table exists.
* Malaria: 517/640 districts, to April 2024. Flood: 502/640 districts, to 2023, absence unknown.
  Dengue: state-level, 2021 to 2026-02-28 (provisional).
* ANN default feature year is 2025 (target 2026), beyond the data; override with `ann_feature_year`.
  The model is uncalibrated; the test set was inspected during audit.
* Telangana mapping (district codes 532-541), the Leh/Kargil rule and the West Bengal 2024 flag are
  confirmed decisions implemented in `destination_analysis.py` / the flags CSV; they are not source facts.
  The `*` meaning and West Bengal 2024 completeness remain unconfirmed.

## Blocked

* All raw inputs (`data/raw/`: Census shapefile/population, NCVBDC malaria PDF and dengue HTML, flood inventory)
  are absent from the repository, so the processed datasets cannot be rebuilt from a fresh clone.
  Details and run order: `docs/REPRODUCIBILITY.md`. `pdfplumber` and `geopandas` (used by pipeline scripts)
  are not declared in `requirements-data.txt`.

* `data/processed/weather/weather_daily_2011_2025.{csv,parquet}` is not in the repository
  (Git-ignored; teammate's machine). Needed for any month- or date-specific weather.
  Recovery: copy the parquet into `data/processed/weather/`, or regenerate (long API
  download) with `scripts/ingestion/download_nasa_power.py` then
  `scripts/cleaning/combine_weather_data.py`. Not run here.

## Findings not fixed (owner decision needed)

* Master feature table dengue join defects (see `DATA_ANALYSIS_HANDOFF.md` section 8):
  88 rows lack dengue for four states/UTs; 40 Telangana rows carry Andhra Pradesh values; 8 Leh/Kargil rows.
  Fix `normalize_state` in `scripts/features/build_master_feature_dataset.py`, then rebuild
  and regenerate the ANN dataset only if you also want it consistent (the ANN does not use dengue).
* Dependency pins conflict between `requirements-data.txt` and `ml/requirements-inference.txt`.
* Possibly unnecessary tracked files left in place because they are referenced or are provenance
  for the teammate-owned pipeline: exploratory `scripts/ingestion/inspect_*.py` and
  `test_malaria_*.py`, large diagnostic CSVs in `data/reports/malaria_*validation*.csv`.
  `ml/data/heatwave/heatwave_t_to_t1_dataset.csv` (Git-ignored duplicate of the parquet) is read by
  `ml/test_heatwave_dataset.py`, so it must stay.

## Next action

Backend owner: wire `build_destination_summary` into the API using the example in
`DATA_ANALYSIS_HANDOFF.md`, then obtain the daily weather file to unlock travel-window weather.
