# Bharat TravelShield AI

India-focused travel information and risk-awareness platform. Its core is a
**data-aggregation layer**: weather, disease and disaster information that is
scattered across sources is brought together for a destination district, with
provenance, coverage and limitations. It is decision support only, not a
diagnosis, a forecast, an emergency alert or a substitute for official advisories.

An experimental heatwave ANN also exists. It is optional and is not the
product's centre.

## Where things are

| Path | Contents |
|---|---|
| `data/` | Reference, processed data, schemas, reports, provenance |
| `scripts/` | Ingestion, cleaning, geography, feature and validation pipeline |
| `ml/destination_analysis.py` | Destination summary (the integration entry point) |
| `ml/data_audit.py` | Repeatable data inventory and quality audit |
| `ml/heatwave_inference.py` | Optional experimental heatwave ANN inference |
| `ml/DATA_ANALYSIS_HANDOFF.md` | **Start here for backend integration** |
| `data/schemas/destination_summary_contract.md` | Output contract |
| `ml/DATA_QUALITY_REPORT.md` | Latest audit output |
| `ml/IMPLEMENTATION_STATUS.md` | What is done, partial and blocked |
| `docs/REPRODUCIBILITY.md` | Environments, raw inputs not in Git, pipeline order, blockers |
| `docs/samples/` | Real-data example destination summary |

## Quick start (repository root, Python 3.13)

```bash
python -m pip install -r ml/requirements-inference.txt
python -m ml.data_audit
python -m ml.destination_analysis --district-name Agra --state-name "Uttar Pradesh" --start-date 2024-10-01 --end-date 2024-10-31
python -m unittest ml.test_destination_analysis ml.test_data_audit ml.test_heatwave_inference -v
```

```python
from ml.destination_analysis import build_destination_summary
summary = build_destination_summary(district_name="Agra", state_name="Uttar Pradesh")
```

`requirements-data.txt` (data pipeline) and `ml/requirements-inference.txt`
pin different numpy/pandas versions; use one environment per file until they
are reconciled.

**Known limits:** the raw source files and the daily NASA POWER weather file are
not in this repository, so the processed datasets are committed as-is and the
pipeline cannot be fully re-run from a fresh clone; only annual weather is
available (no travel-window weather). Dengue is state-level context; West Bengal
2024 is `verification_pending`. See `docs/REPRODUCIBILITY.md` and
`ml/IMPLEMENTATION_STATUS.md`.
