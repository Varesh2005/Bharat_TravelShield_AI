# ML Module Audit Note (Phase A, written before the ANN rebuild)

Start state: branch `feature/ml`, HEAD `1793b59`, clean working tree, in sync with `origin/feature/ml`.
Project PDFs read (text extracted with PyPDF2): `ANN work distribution.pdf` (24 pages) and
`TRAVELRISK_AI_Implementation_Plan.pdf` (10 pages).

## What the plans require of this module
ANN/MLP as the primary, understandable model; independent hazard outputs (dengue, malaria,
flood, heatwave); no arbitrary combined score; leakage-safe features; baselines on the same split
(seasonal-naive, LR, RF, LightGBM); PR-AUC/ROC-AUC/F1/Brier/ECE; calibration; an ANN explanation the team can
explain; a loadable model with input/output schema for the backend; fixed seeds and recorded versions.
The plan itself says: do not manufacture district labels from state-level data, and show
degraded confidence instead of a normal-looking score when inputs are missing.

## Baseline (before any change)
| Command | Result |
|---|---|
| `python -m unittest ml.test_destination_analysis ml.test_data_audit ml.test_heatwave_inference` | 80 tests, OK |
| `python ml/test_heatwave_dataset.py` | all 10 checks passed |
| `python ml/test_heatwave_models.py` | all 8 checks passed |
| `python -m ml.data_audit --check` | exit 0, 0 errors, 19 warnings |

No pre-existing failures.

## What existed and its gaps against the plans
* **Only a heatwave ANN** (v1). Dengue, malaria and flood had no model, readiness analysis or decision.
* v1 target is annual and district-level (fine), but: the test set was inspected during audit; raw scores
  uncalibrated (ECE 0.11) with no validated threshold; the split and feature set were heatwave-specific;
  the code would not extend to other hazards without copying it.
* Baselines compared in v1 (persistence, LR, RF) but no climatology baseline, no ablation, no seed
  stability, no held-out-geography test, no error analysis by year/state, no ANN explanation.
* Backend contract existed for the heatwave only (`INFERENCE_HANDOFF.md`) and the data-summary ANN block
  rebuilt the 14 v1 features itself (a second prediction path).
* All sources are annual: the plan's travel-date/month and seasonal-feature design cannot be met.

## Data evidence collected (drives the design)
* Heatwave: complete 640 districts x 2011-2025; positive rate by year 4.7% (2021) to 54.8%.
* Flood: 502 districts, presence-only events 1967-2023; yearly positive rate among them 21-50%.
* Malaria: 517 districts; national reported cases fell 1.07M (2011) -> 0.12M (2021); 2024 is partial.
* Dengue: 36 state series, full years 2021-2025 only (no year before 2020; state-level labels).
* Chikungunya: no data. Daily weather: absent. Raw inputs (`data/raw`): absent.
* Known master-feature dengue join defects (88 rows missing for four states/territories, 40 Telangana rows with Andhra Pradesh values,
  8 Leh/Kargil rows with J&K values) are not used by the new models (dengue is not a feature).

## Second review (after the rebuild)
Independent recomputation of labels/features, leakage and calibration review, contract-vs-output check and
file-reference scan. Defects found and fixed: unhashable hazard raised TypeError; heatwave class shown while its
probability was withheld; class labels overstated presence-only/threshold targets; flood cohort rule made explicit
(no data change). Details in `ml/ANN_DESIGN_DECISION.md` section 11. No tracked file was found obsolete.

## Third review (calibration methodology)
The probability-display rule first used test-year ECE. It now uses a forward check on validation years only (fit 2020, evaluate
2021). Models, thresholds and metrics are bit-identical; flood moved from shown to withheld. Test evidence is labelled
exploratory-informed, not pristine. See `ml/ANN_DESIGN_DECISION.md` section 8.

## Final handoff review (file relevance, ANN audit, documentation)

Repository state at the start: branch `feature/ml`, HEAD `1793b59`, 0 commits ahead/behind `origin/feature/ml`, no merge/rebase in
progress, no stash; every uncommitted change was this work's own (no teammate edits found; remote `feature/data` untouched).

**File relevance audit (164 files, every tracked and untracked file, with reference counts from code, tests and docs):**

| Role | Files | Decision |
|---|---|---|
| 1 Runtime / inference | `ml/hazard_inference.py`, `ml/destination_analysis.py`, `ml/models/hazards/*/{ann_bundle.joblib,model_card.json}` | keep |
| 2 Training / evaluation | `ml/hazard_datasets.py`, `ml/hazard_training.py`, `ml/data/hazards/*` (generated, reproducible, 1 MB), `research/experiments/*` (generated results and figures) | keep |
| 3 Tests | 9 test modules in `ml/` (new hazard, demo, contract-example suites and the earlier ones) | keep |
| 4 Data preparation / upstream | `scripts/**` (ingestion, cleaning, geography, features, validation), `data/processed`, `data/reference`, `data/reports`, `data/provenance` | keep, Adhith-owned, not edited |
| 5 Backend handoff | `ml/ML_MODEL_CONTRACT.md`, `ml/DATA_ANALYSIS_HANDOFF.md`, `data/schemas/destination_summary_contract.md`, `docs/samples/*` | keep |
| 6 Useful current docs | README, `ml/ANN_DESIGN_DECISION.md`, `ml/EVALUATION_REPORT.md` (generated), `ml/IMPLEMENTATION_STATUS.md`, `ml/DATA_QUALITY_REPORT.md` (generated), `ml/VERIFY_IN_TERMINAL.md`, `docs/REPRODUCIBILITY.md`, `research/README.md`, this note | keep |
| 7 Legacy but tested / evidence | v1 heatwave pipeline, its four models, dataset, experiment report (`ml/*heatwave*`, `ml/models/heatwave`, `ml/data/heatwave`, `ml/experiments`, `ml/INFERENCE_HANDOFF.md`) | keep, labelled LEGACY |
| 8 Redundant / obsolete / accidental | none found | **nothing deleted** |

Deletion candidates examined (none removed): 21 unreferenced scripts under `scripts/ingestion`, `scripts/geography`,
`scripts/validation`, `scripts/cleaning`. They are Adhith's exploratory or version-2 extraction and validation scripts; the
apparent duplicate `validate_malaria_against_district_master.py` exists in two folders but the files differ (16.2 KB v1 vs 3.9 KB v2) and
write different committed reports (`malaria_district_mapping_report*.csv`), so neither is redundant. Deleting teammate-owned research code would also
conflict with `feature/data`. `scripts/test_environment.py` is an unreferenced setup check and stays. `ml/data/heatwave/heatwave_t_to_t1_dataset.csv`
(Git-ignored duplicate) is read by a legacy test and stays local only. Only disposable `__pycache__` folders were removed (never tracked).

**ANN audit (machine-checked):** feature lists and order identical across spec, dataset, feature dictionary, scaler/network input size,
model card, bundle and results; card architecture equals the loaded network; card and bundle agree on threshold and calibration; split
rows and boundaries equal the documented rule; every model card's source-file hashes equal the current data files (no stale model); labels
and features recomputed independently from the raw CSVs (0 mismatches in 1,500-row samples per hazard); leakage and test-label-isolation
tests pass; dengue and chikungunya return `unsupported` on every entry point.

**Documentation audit:** every path and `python -m` module named in the docs exists; commands run from the repository root; sample JSON
is byte-identical to a fresh run and conforms to the contract; no stale keys, labels or counts remain. Fixed this round: a stray form-feed
character in `ml/ANN_DESIGN_DECISION.md`, stale test lists, a stale "(not for heatwave)" (flood is withheld too), a missing blank line, the
design-rule outcome that was never stated (the ANN does not exceed the best non-neural baseline on any hazard), and the backend contract
now has an explicit scope section, a destination-analysis vs hazard-inference comparison and an executed copy-paste example for every status.

**Observations for other owners (not changed here):** `requirements-data.txt` is UTF-16 encoded (pip reads it); `pdfplumber` and `geopandas` are
used by pipeline scripts but not declared; three committed generated files embed local Windows paths; `README.md`, `.gitignore` and `docs/` are
shared/Pranav-owned and were edited minimally (see the commit).
## Decisions taken
See `ml/ANN_DESIGN_DECISION.md`. In short: three separate hazard ANNs (heatwave, flood, malaria) on one
shared protocol; dengue and chikungunya explicitly unsupported with evidence and recovery path; heatwave
probability and class withheld where the validation-year calibration check fails (heatwave, flood; only malaria shows a probability); no risk bands.

## File review (Phase G) - nothing deleted except disposable caches
| File(s) | Decision | Reference check / reason |
|---|---|---|
| `ml/prepare_heatwave_dataset.py`, `train_heatwave_models.py`, `heatwave_inference.py`, `test_heatwave_*.py`, `ml/data/heatwave/*`, `ml/models/heatwave/*` (4 joblib), `ml/experiments/*`, `ml/INFERENCE_HANDOFF.md` | **Keep as legacy v1** | Superseded but referenced by tests that pass, and the only record of the first experiment (including its reported calibration problem). The summary no longer calls it. Deleting would erase tested work and evidence. Marked LEGACY in docs. |
| `ml/data/heatwave/heatwave_t_to_t1_dataset.csv` (Git-ignored duplicate of the parquet) | **Keep** | `ml/test_heatwave_dataset.py` reads it. |
| `ml/IMPLEMENTATION_STATUS.md`, `ml/DATA_ANALYSIS_HANDOFF.md` | **Refactor (updated)** | Contained stale statements about the ANN path; rewritten, not removed. |
| `data/reports/*`, `data/provenance/*`, schemas, `scripts/*` | **Keep** | Adhith's pipeline; not touched. |
| `ml/**/__pycache__/` | **Remove (disposable, untracked)** | Regenerated automatically; Git-ignored. |
| `ml/models/heatwave/random_forest.joblib`, `logistic_regression.joblib` | **Keep** | Needed by `test_heatwave_models.py`. |

No tracked file was removed, and none was merged away. A reference scan of every candidate (code, tests, docs) found no obsolete
tracked file; the only unreferenced one, `scripts/test_environment.py`, is a data-environment setup check and stays. v2 adds one authoritative artifact path per
model: `ml/models/hazards/<hazard>/ann_bundle.joblib` (+ `model_card.json`); no baseline artifacts are duplicated.
