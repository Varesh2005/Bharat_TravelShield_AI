# Implementation Status (ML module)

Branch `feature/ml`; state at the final ML/ANN handoff to the backend team.
Environment used for every result: Python 3.13.14, scikit-learn 1.7.2, numpy 2.3.2, pandas 2.3.1
(`ml/requirements-inference.txt`), also verified in a brand-new virtual environment created from that file alone. Runner: `unittest`. Reproduce everything with
`powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1` (see `ml/VERIFY_IN_TERMINAL.md`).

## Hazard status (computed by `python -m ml.hazard_datasets --readiness`)

| Hazard | Status | Why |
|---|---|---|
| heatwave | supported (`heatwave-ann-mlp-v2`) | 640 districts, complete labels 2011-2025; validation-year calibration check fails, so probability and class withheld |
| flood | supported with caveats (`flood-ann-mlp-v2`) | presence-only inventory labels, 502 of 640 districts; probability withheld (validation check ECE 0.149) |
| malaria | supported (`malaria-ann-mlp-v2`) | 517 districts, relative top-quartile target; the ANN mostly uses prior-year incidence |
| dengue | **unsupported** | state-level labels, 2021-2025 only: 0 training and 0 validation transitions |
| chikungunya | **unsupported** | no data in the repository |

## Completed
| Item | Evidence |
|---|---|
| ANN design decision and claims | `ml/ANN_DESIGN_DECISION.md` |
| Leakage-safe datasets per hazard + readiness evidence | `ml/hazard_datasets.py`, `ml/data/hazards/` |
| ANN training/evaluation vs persistence, climatology, LR, RF on one split; calibration; ablation; seeds; held-out states; error analysis | `ml/hazard_training.py`, `ml/EVALUATION_REPORT.md`, `research/experiments/` |
| Single backend interface, input validation, unsupported/unavailable handling, ANN explanation | `ml/hazard_inference.py`, `ml/ML_MODEL_CONTRACT.md` |
| Destination summary now uses the single interface (schema 1.1) | `ml/destination_analysis.py`, `data/schemas/destination_summary_contract.md` |
| Data audit covers the new model datasets | `ml/data_audit.py`, `ml/DATA_QUALITY_REPORT.md` |
| Terminal verification | `ml/verify_ml.ps1`, `ml/VERIFY_IN_TERMINAL.md` |
| Audit note and file review | `ml/ML_AUDIT_NOTE.md` |

## Test results (observed in the clean environment; `powershell -ExecutionPolicy Bypass -File ml/verify_ml.ps1`)
| Suite | Tests |
|---|---|
| New hazard suites: `ml.test_hazard_datasets` 16, `ml.test_hazard_training` 13, `ml.test_hazard_inference` 21 | 50, OK |
| `ml.test_destination_analysis` 54, `ml.test_data_audit` 15, `ml.test_heatwave_inference` 10 (legacy), `ml.test_demo_local` 5, `ml.test_contract_examples` 4 | 88, OK |
| Legacy `ml/test_heatwave_dataset.py` / `ml/test_heatwave_models.py` | 10 / 8 checks passed |
| `python -m ml.hazard_datasets --check`, `python -m ml.data_audit --check` | datasets match a fresh build; 0 errors, 22 warnings |

Count history: at commit 1793b59 the three older suites had 55 + 15 + 10 = 80 tests; the destination-summary ANN block tests
were then rewritten (8 old tests tied to the v1 feature builder replaced by 7 for the unified interface), giving 54 + 15 + 10 = 79.
## Second review (defects found and fixed)
Unhashable hazard raised TypeError; heatwave showed a class while its probability was withheld; class labels overstated
presence-only/threshold targets; flood cohort rule made explicit (no data change); probability-display rule moved from test to
validation data. Final handoff review (file relevance, documentation, clean-environment run): `ml/ML_AUDIT_NOTE.md`.
Details: `ml/ANN_DESIGN_DECISION.md` sections 8 and 11.

## What the results mean (honest reading)
* The ANN is **competitive, not superior**: on held-out years logistic regression has equal or higher PR-AUC for
  all three hazards (random forest also for flood); the ANN's PR-AUC beats persistence and climatology on all
  three, but for heatwave climatology has a much higher F1 at the validation-chosen threshold.
* Heatwave: the validation window contains the anomalous 2021, so the validation-chosen threshold is too strict
  (recall 0.35) and Platt calibration under-predicts on test (ECE 0.221, information only). The validation-only forward check
  (fit 2020, evaluate 2021: ECE 0.173) withholds the probability and class; flood is withheld too (0.149, base rate 47% -> 33%);
  only malaria (0.056) shows a probability. Raw scores are shown only as a ranking.
* Test evidence is **exploratory-informed, not a pristine final test**: the display rule originally used test ECE; it now uses
  validation years only, and the models/metrics are bit-identical (`ml/ANN_DESIGN_DECISION.md` section 8).
* Flood: weak signal (ROC-AUC about 0.70); history features carry most of it.
* Malaria: prior-year incidence dominates (removing history drops PR-AUC to about 0.65; removing weather does not hurt).

## Partial / limitations
* Annual district-level only. No month/travel-date prediction (no sub-annual data). No risk bands.
* Validation is only two years and serves several purposes (early stopping, grid, calibration, threshold).
* LightGBM omitted (not installed). No multi-task ANN (evidence against, see the design decision).
* The data-layer limits in `docs/REPRODUCIBILITY.md` still apply (raw inputs and daily weather absent).

## Blocked
* **Dengue and chikungunya models:** need longer or additional labelled series (recovery paths in the readiness output).
* **Dengue West Bengal 2024:** still `verification_pending`.
* **Rebuilding the upstream datasets:** raw inputs and the daily NASA POWER file are not in the repository.

## Next action
Backend owner: call `ml.hazard_inference.predict_for_district` per `ml/ML_MODEL_CONTRACT.md` and show
`status`, `model_reliability`, `limitations` and the withheld-probability rule; data owner: provide longer dengue history.
