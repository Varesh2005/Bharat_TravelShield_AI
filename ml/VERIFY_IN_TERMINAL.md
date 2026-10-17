# Verify the ML module yourself (VS Code terminal, Windows PowerShell)

One command runs everything below and prints real input and output. It does not commit,
push or change tracked files.

```powershell
powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1
```

If `python` is not the Python 3.13 environment with `ml/requirements-inference.txt` installed:

```powershell
powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1 -Python "C:\path\to\python.exe"
```

It takes about two minutes (the training-protocol tests train tiny networks).

## What each step proves

| Step | Command (inside the script) | Expected |
|---|---|---|
| 1, 10 | `git branch --show-current`, `git status --short` | branch `feature/ml`; no commit is made |
| 2 | prints Python/scikit-learn/numpy/pandas versions | 3.13.x, scikit-learn 1.7.2 |
| 3 | `python -m unittest ml.test_hazard_datasets ml.test_hazard_training ml.test_hazard_inference` | `Ran 50 tests ... OK` |
| 3b | data-layer, audit, legacy heatwave, demo and backend-contract-example suites | `Ran 88 tests ... OK`, then `ALL 10 ...` and `ALL MODEL EVALUATION TESTS PASSED` |
| 4 | `python -m ml.hazard_datasets --check` | `OK: datasets match a fresh build` |
| 5 | `--readiness`, `ml.hazard_inference --list` | heatwave supported, flood supported_with_caveats, malaria supported, dengue and chikungunya unsupported with the reason |
| 6 | flood, Hyderabad (code 536) | JSON with hazard, model version, target, horizon, raw score, limitations, ANN explanation; `calibrated_probability` is `null` (validation-year calibration check failed) with the reason in `calibration_note` |
| 6b | malaria for Agra (146); heatwave for Agra | malaria has a calibrated probability and a class; heatwave has `calibrated_probability`, `predicted_class` and `predicted_label` all `null` with a withheld note |
| 6c | `python -m ml.demo_local destination` | readable Agra view: records (state-level dengue labelled, unknown never zero), then ANN outputs |
| 7 | bad inputs | `invalid_input` with reasons; NaN rejected; dengue and chikungunya `unsupported`; Darjiling malaria and year 1999 `unavailable`; unknown destination -> `destination_not_found`; Bilaspur -> `ambiguous_destination`; none returns a score |
| 8 | two separate interpreter runs | `IDENTICAL output from two fresh processes` |
| 9 | `python -m ml.data_audit --check` | `PASSED_WITH_WARNINGS: 0 error(s), 22 warning(s)` |

In step 3b you will see lines such as `FAILED: 1 error(s)` and `ERROR: malaria_district_year ... is missing`. They are not failures:
they are printed by audit tests that deliberately run the audit CLI on a broken *synthetic* folder. The line to trust is
`OK` after `Ran 88 tests`.

Passing tests proves the software behaves as designed. It does not prove the source labels are
valid or the models are scientifically adequate; read `ml/EVALUATION_REPORT.md` for that.

## Beginner demo (readable output, real data)

`ml/demo_local.py` only calls the existing functions and prints a readable view. See the top of that file
for details; the quickest start is:

```powershell
python -m ml.demo_local destination                 # Agra, Oct 2024: records + ANN outputs
python -m ml.demo_local list                        # which hazards are supported
python -m ml.demo_local features --hazard flood --district-code 146 --out $env:TEMP\agra_flood.json
python -m ml.demo_local predict  --hazard flood --features-file $env:TEMP\agra_flood.json
```

Its own tests (5, real data): `python -m unittest ml.test_demo_local`.
## Running one thing by hand

```powershell
python -m ml.hazard_inference --list
python -m ml.hazard_inference --hazard flood --district-code 536
python -m ml.hazard_inference --hazard malaria --district-code 146 --feature-year 2022 --no-explain
python -m ml.destination_analysis --district-name Agra --state-name "Uttar Pradesh" --start-date 2024-10-01 --end-date 2024-10-31 --with-ann
```

Retrain from the committed processed data (about two minutes, deterministic seed 42; rewrites the
model bundles, `research/experiments/*` and `ml/EVALUATION_REPORT.md`):

```powershell
python -m ml.hazard_datasets --build
python -m ml.hazard_training --hazard all
```

## The 22 audit warnings, explained

None is an error (`0 error(s)`). They are known limitations that the audit reports every time.

| Warning | Why it remains | Action |
|---|---|---|
| district_master: Telangana under Andhra Pradesh, Ladakh under J&K | Census-2011 boundaries | handled in code (Telangana series; no J&K values for Leh/Kargil) |
| weather_year: annual only | no monthly weather table in the repository | blocks travel-window weather; needs the daily file |
| heatwave_year: threshold indicator | derived from NASA POWER Tmax | wording rule: never call it an IMD declaration |
| malaria: only matched districts; 2024 partial | extraction/mapping coverage (517 of 640); source stops in April 2024 | 2024 is never used for modelling |
| dengue: 'Telangana' has no master equivalent | master predates the split | mapped through the documented district list |
| dengue: `West Bengal*` marker | meaning of the asterisk is unconfirmed | label kept; 2024 value flagged |
| dengue: state-level only | source resolution | never shown as district data |
| dengue: no Ladakh series | source has none | J&K values are not assigned to Leh/Kargil |
| dengue: 2026 provisional | partial year | marked in outputs |
| flood: unmapped districts excluded | only deterministic LGD mappings retained | 502 of 640 districts |
| master_features: dengue copied to districts | feature-table design | the ML layer does not use dengue |
| heatwave_ann_dataset: legacy v1 table | superseded by the v2 datasets | kept as evidence |
| model_dataset_heatwave / flood / malaria (3) | notes on target definitions and label reliability | see `ml/ANN_DESIGN_DECISION.md` |
| weather_daily missing (2) | the Git-ignored daily file is not in the repository | recovery commands are printed in the audit |
| master_features: 88 rows lack dengue for four states/territories | name aliases not applied by the original builder | documented, builder not changed |
| master_features: 40 Telangana rows carry Andhra Pradesh values | Census-2011 grouping | the data layer maps Telangana separately |
| master_features: 8 Leh/Kargil rows carry J&K values | same | the data layer withholds dengue for them |
| dengue West Bengal 2024 `verification_pending` | awaiting the data owner | value kept unchanged |
