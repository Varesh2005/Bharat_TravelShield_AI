# research/

Experiment evidence for the hazard ANNs. Everything here is **generated** by
`python -m ml.hazard_training --hazard all` (deterministic, seed 42); do not edit by hand.

```
research/experiments/<hazard>/        hazard = heatwave | flood | malaria
  results.json                        every model's validation/test metrics, ANN grid-search log,
                                      5-seed stability, feature-group ablation, held-out-states test,
                                      error analysis, learning curve
  ann_learning_curve.png              train vs validation loss of the selected ANN
  ann_reliability_and_pr.png          calibration (reliability) curve and precision-recall curve on test years
```

The test years were scored in an exploratory run before the probability-display rule was fixed, so these are held-out
but **exploratory-informed**, not a pristine final test (see ml/ANN_DESIGN_DECISION.md section 8).

Human-readable summary: `ml/EVALUATION_REPORT.md`. Design and claims: `ml/ANN_DESIGN_DECISION.md`.
Earlier heatwave v1 experiment results are in `ml/experiments/` (legacy).
(`research/results/` is Git-ignored by the project's `.gitignore`, which is why these live in `experiments/`.)
