"""Train and evaluate the hazard ANNs (primary) against baselines on one split.

    python -m ml.hazard_training --hazard all          # heatwave, flood, malaria
    python -m ml.hazard_training --hazard flood --quick

What happens for each hazard (all steps use the SAME chronological split):
  1. Standardise inputs (scaler fit on train only).
  2. ANN = dense MLP (ReLU hidden layers, one sigmoid output, log-loss, Adam).
     Trained epoch by epoch; stops when VALIDATION loss stops improving.
     Hyper-parameters are chosen by validation PR-AUC from a small fixed grid.
  3. Platt calibration and the decision threshold are fitted on VALIDATION only.
  4. The untouched TEST years are scored once per model.
  5. Baselines: persistence, district climatology, logistic regression, random forest.
  6. Extras: 5-seed stability, feature-group ablation, held-out-states test,
     error analysis, permutation importance.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ml.hazard_datasets import DEFAULT_OUT_DIR, ECE_ACCEPTABLE, HAZARDS, SPECS, HazardSpec

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "models" / "hazards"
RESULT_DIR = ROOT / "research" / "experiments"
SEED = 42
MODEL_VERSION = "{hazard}-ann-mlp-v2"


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_platt(score: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt scaling: calibrated = sigmoid(a * logit(raw) + b)."""
    return 1.0 / (1.0 + np.exp(-(a * _logit(np.asarray(score, dtype=float)) + b)))


def fit_platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Fit on validation data only. Returns (a, b, fitted)."""
    if len(np.unique(y)) < 2:
        return 1.0, 0.0, False
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0]), True


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(float(y[m].mean()) - float(p[m].mean()))
        total += gap * m.sum() / len(y)
        rows.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                     "mean_predicted": round(float(p[m].mean()), 4),
                     "observed_rate": round(float(y[m].mean()), 4)})
    return total, rows


def forward_calibration_check(years: np.ndarray, y: np.ndarray, score: np.ndarray,
                              max_ece: float = ECE_ACCEPTABLE) -> dict:
    """Out-of-sample calibration check that uses VALIDATION data only.

    Platt scaling is fitted on the earlier validation target years and its ECE is measured on the
    last validation target year (forward in time). Decides whether a probability may be displayed.
    """
    years, y, score = np.asarray(years), np.asarray(y), np.asarray(score)
    ys = sorted(int(v) for v in np.unique(years))
    out = {"uses_test_years": False, "max_allowed_ece": max_ece, "display_probability": False,
           "fit_target_years": ys[:-1], "eval_target_year": ys[-1] if ys else None}
    if len(ys) < 2:
        return {**out, "ece": None, "reason": "fewer than two validation years: no out-of-sample check possible"}
    fit = years < ys[-1]
    a, b, fitted = fit_platt(score[fit], y[fit])
    if not fitted:
        return {**out, "ece": None, "reason": "calibration could not be fitted (single class)"}
    p = apply_platt(score[~fit], a, b)
    e = float(ece(y[~fit], p)[0])
    return {**out, "ece": round(e, 4), "n_fit": int(fit.sum()), "n_eval": int((~fit).sum()),
            "eval_prevalence": round(float(y[~fit].mean()), 4),
            "fit_prevalence": round(float(y[fit].mean()), 4),
            "display_probability": bool(e <= max_ece)}


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on VALIDATION data."""
    best, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y, p >= t, zero_division=0)
        if f > best_f1 + 1e-12:
            best, best_f1 = float(round(t, 2)), f
    return best


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    """Full metric set. `score` ranks; `prob` (if given) is a calibrated probability."""
    out: dict[str, Any] = {"n": int(len(y)), "prevalence": round(float(y.mean()), 4)}
    two = len(np.unique(y)) == 2
    out["pr_auc"] = round(float(average_precision_score(y, score)), 4) if two else None
    out["roc_auc"] = round(float(roc_auc_score(y, score)), 4) if two else None
    out["precision"] = round(float(precision_score(y, pred, zero_division=0)), 4)
    out["recall"] = round(float(recall_score(y, pred, zero_division=0)), 4)
    out["f1"] = round(float(f1_score(y, pred, zero_division=0)), 4)
    out["macro_f1"] = round(float(f1_score(y, pred, average="macro", zero_division=0)), 4)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["per_class"] = {
        "negative": {"precision": round(float(precision_score(y, pred, pos_label=0, zero_division=0)), 4),
                     "recall": round(float(recall_score(y, pred, pos_label=0, zero_division=0)), 4)},
        "positive": {"precision": out["precision"], "recall": out["recall"]}}
    if prob is not None:
        out["brier"] = round(float(np.mean((prob - y) ** 2)), 4)
        out["log_loss"] = round(float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])), 4)
        e, table = ece(y, prob)
        out["ece"] = round(float(e), 4)
        out["reliability_table"] = table
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ANNConfig:
    hidden: tuple[int, ...] = (64, 32)
    alpha: float = 1e-3            # L2 weight decay
    balance: bool = False          # random oversampling of the minority class (train only)
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 15
    seed: int = SEED

    def label(self) -> str:
        return f"ANN{'-'.join(map(str, self.hidden))}_a{self.alpha:g}{'_oversample' if self.balance else ''}"


FULL_GRID = [ANNConfig(h, a, b) for h in ((32, 16), (64, 32), (128, 64))
             for a in (1e-3, 1e-2) for b in (False, True)]
QUICK_GRID = [ANNConfig((16, 8), 1e-3, False), ANNConfig((16, 8), 1e-2, True)]


def _oversample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    if len(small) == 0:
        return X, y
    extra = rng.choice(small, size=len(big) - len(small), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


def train_ann(Xtr, ytr, Xva, yva, cfg: ANNConfig) -> tuple[MLPClassifier, dict]:
    """Epoch-by-epoch training with early stopping on validation log-loss."""
    X, y = _oversample(Xtr, ytr, cfg.seed) if cfg.balance else (Xtr, ytr)
    net = MLPClassifier(hidden_layer_sizes=cfg.hidden, activation="relu", solver="adam",
                        alpha=cfg.alpha, batch_size=cfg.batch_size,
                        learning_rate_init=cfg.learning_rate, random_state=cfg.seed)
    best, best_loss, stale, history = None, np.inf, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        net.partial_fit(X, y, classes=np.array([0, 1]))
        val_loss = float(log_loss(yva, net.predict_proba(Xva)[:, 1], labels=[0, 1]))
        history.append({"epoch": epoch, "train_loss": round(float(net.loss_), 5), "val_loss": round(val_loss, 5)})
        if val_loss < best_loss - 1e-4:
            best, best_loss, stale = copy.deepcopy(net), val_loss, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    info = {"epochs_run": len(history), "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1),
            "best_val_loss": round(best_loss, 5), "history": history}
    return best, info


def _positive(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def _prep(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].astype(float).to_numpy()


def climatology_scores(train: pd.DataFrame, df: pd.DataFrame, target: str) -> np.ndarray:
    """District event rate in the training years (global rate for unseen districts)."""
    rate = train.groupby("census_district_code")[target].mean()
    return df["census_district_code"].map(rate).fillna(float(train[target].mean())).to_numpy(dtype=float)


# --------------------------------------------------------------------------
# One full experiment on a given split
# --------------------------------------------------------------------------

def _score_model(name: str, val_score, test_score, yva, yte, extra: dict | None = None) -> dict:
    """Calibrate on validation, choose threshold on validation, evaluate test once."""
    a, b, fitted = fit_platt(val_score, yva)
    pv, pt = apply_platt(val_score, a, b), apply_platt(test_score, a, b)
    thr = best_threshold(yva, pv)
    res = {"model": name, "threshold_from_validation": thr, "platt": {"a": a, "b": b, "fitted": fitted},
           "validation": metrics(yva, val_score, pv >= thr, pv),
           "test": metrics(yte, test_score, pt >= thr, pt),
           "test_raw_score_ece": round(float(ece(yte, np.clip(test_score, 0, 1))[0]), 4)}
    if extra:
        res.update(extra)
    return res, pt


def run_experiment(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, spec: HazardSpec,
                   features: list[str] | None = None, grid: list[ANNConfig] | None = None,
                   quick: bool = False, keep_models: bool = False) -> dict:
    feats = features or spec.features
    grid = grid or (QUICK_GRID if quick else FULL_GRID)
    t = spec.target
    ytr, yva, yte = (d[t].to_numpy(dtype=int) for d in (train, val, test))
    scaler = StandardScaler().fit(_prep(train, feats))
    Xtr, Xva, Xte = (scaler.transform(_prep(d, feats)) for d in (train, val, test))
    out: dict[str, Any] = {"features": feats, "rows": {"train": len(train), "val": len(val), "test": len(test)}}
    fitted: dict[str, Any] = {"scaler": scaler}

    # --- ANN: grid search on validation PR-AUC --------------------------------
    log, best = [], None
    for cfg in grid:
        net, info = train_ann(Xtr, ytr, Xva, yva, cfg)
        pr = float(average_precision_score(yva, _positive(net, Xva)))
        log.append({"config": cfg.label(), "val_pr_auc": round(pr, 4), **{k: info[k] for k in ("epochs_run", "best_epoch", "best_val_loss")}})
        if best is None or pr > best[0] + 1e-9:
            best = (pr, cfg, net, info)
    _, cfg, net, info = best
    out["ann_grid_search"] = log
    out["ann_selected_config"] = {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                                  "label": cfg.label(),
                                  **{k: v for k, v in info.items() if k != "history"}}
    sv, st = _positive(net, Xva), _positive(net, Xte)
    ann_res, ann_prob = _score_model("ANN (MLP)", sv, st, yva, yte)
    out["ann"] = ann_res
    out["ann_validation_calibration_check"] = forward_calibration_check(
        val["target_year"].to_numpy(), yva, sv)
    fitted.update(net=net, ann_cfg=cfg, ann_val_score=sv, ann_test_prob=ann_prob, history=info["history"],
                  platt=ann_res["platt"], threshold=ann_res["threshold_from_validation"])

    # --- Baselines ---------------------------------------------------------------
    base: dict[str, Any] = {}
    # persistence: "same as this year"
    pv_, pt_ = val[spec.persistence_feature].to_numpy(dtype=int), test[spec.persistence_feature].to_numpy(dtype=int)
    base["Persistence (last year repeats)"] = {
        "model": "Persistence", "note": "hard 0/1 prediction; PR-AUC/ROC-AUC use the 0/1 value as the score",
        "validation": metrics(yva, pv_.astype(float), pv_, pv_.astype(float)),
        "test": metrics(yte, pt_.astype(float), pt_, pt_.astype(float))}
    # climatology
    cv_, ct_ = climatology_scores(train, val, t), climatology_scores(train, test, t)
    base["Climatology (district training rate)"], _ = _score_model("Climatology", cv_, ct_, yva, yte)
    # logistic regression
    lr_best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        for cw in (None, "balanced"):
            m = LogisticRegression(C=C, class_weight=cw, max_iter=2000).fit(Xtr, ytr)
            pr = average_precision_score(yva, _positive(m, Xva))
            if lr_best is None or pr > lr_best[0] + 1e-9:
                lr_best = (pr, m, f"C={C}, class_weight={cw}")
    base["Logistic Regression"], _ = _score_model("Logistic Regression", _positive(lr_best[1], Xva),
                                                  _positive(lr_best[1], Xte), yva, yte, {"selected": lr_best[2]})
    # random forest
    rf_grid = [(d, leaf, cw) for d in (6, 12) for leaf in (1, 5) for cw in (None, "balanced_subsample")]
    if quick:
        rf_grid = rf_grid[:2]
    rf_best = None
    for d, leaf, cw in rf_grid:
        m = RandomForestClassifier(n_estimators=100 if quick else 300, max_depth=d, min_samples_leaf=leaf,
                                   class_weight=cw, random_state=SEED, n_jobs=1).fit(Xtr, ytr)
        pr = average_precision_score(yva, _positive(m, Xva))
        if rf_best is None or pr > rf_best[0] + 1e-9:
            rf_best = (pr, m, f"max_depth={d}, min_samples_leaf={leaf}, class_weight={cw}")
    base["Random Forest"], _ = _score_model("Random Forest", _positive(rf_best[1], Xva),
                                            _positive(rf_best[1], Xte), yva, yte, {"selected": rf_best[2]})
    out["baselines"] = base
    out["_fitted"] = fitted if keep_models else None
    return out


# --------------------------------------------------------------------------
# Extra analyses
# --------------------------------------------------------------------------

def seed_stability(train, val, test, spec, cfg: ANNConfig, seeds=(0, 1, 2, 3, 4)) -> dict:
    scaler = StandardScaler().fit(_prep(train, spec.features))
    Xtr, Xva, Xte = (scaler.transform(_prep(d, spec.features)) for d in (train, val, test))
    ytr, yva, yte = (d[spec.target].to_numpy(dtype=int) for d in (train, val, test))
    rows = []
    for s in seeds:
        c = ANNConfig(cfg.hidden, cfg.alpha, cfg.balance, cfg.learning_rate, cfg.batch_size, cfg.max_epochs, cfg.patience, s)
        net, _ = train_ann(Xtr, ytr, Xva, yva, c)
        res, _ = _score_model("ANN", _positive(net, Xva), _positive(net, Xte), yva, yte)
        rows.append({"seed": s, "test_pr_auc": res["test"]["pr_auc"], "test_f1": res["test"]["f1"],
                     "test_brier": res["test"]["brier"]})
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in ("test_pr_auc", "test_f1", "test_brier")}
    return {"per_seed": rows, **{f"{k}_mean": round(float(v.mean()), 4) for k, v in arr.items()},
            **{f"{k}_std": round(float(v.std()), 4) for k, v in arr.items()}}


def ablation(train, val, test, spec, cfg: ANNConfig) -> dict:
    """Retrain the selected ANN without each feature group (same split, same config)."""
    ytr, yva, yte = (d[spec.target].to_numpy(dtype=int) for d in (train, val, test))
    results = {}
    runs = {"all_features": spec.features}
    for g in spec.groups:
        runs[f"without_{g}"] = [f for f in spec.features if f not in spec.groups[g]]
    for name, feats in runs.items():
        if not feats:
            continue
        sc = StandardScaler().fit(_prep(train, feats))
        Xtr, Xva, Xte = (sc.transform(_prep(d, feats)) for d in (train, val, test))
        net, _ = train_ann(Xtr, ytr, Xva, yva, cfg)
        res, _ = _score_model("ANN", _positive(net, Xva), _positive(net, Xte), yva, yte)
        results[name] = {"n_features": len(feats), "test_pr_auc": res["test"]["pr_auc"],
                         "test_roc_auc": res["test"]["roc_auc"], "test_f1": res["test"]["f1"]}
    return results


def heldout_states(states: list[str], fraction: float = 0.2, seed: int = SEED) -> list[str]:
    ordered = sorted(set(states))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(len(ordered) * fraction)))
    return sorted(rng.choice(ordered, size=k, replace=False).tolist())


def spatiotemporal(train, val, test, spec, quick: bool) -> dict:
    """Train on seen states, test on LATER years in states never seen in training."""
    held = heldout_states(train["state_name"].tolist())
    res = run_experiment(train[~train["state_name"].isin(held)], val[~val["state_name"].isin(held)],
                         test[test["state_name"].isin(held)], spec, quick=quick)
    return {"held_out_states": held, "rows": res["rows"], "ann_selected_config": res["ann_selected_config"]["label"],
            "ann": {k: res["ann"]["test"][k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "ece")},
            "baselines": {n: {k: b["test"][k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision")}
                          for n, b in res["baselines"].items()}}


def error_analysis(test: pd.DataFrame, spec: HazardSpec, prob: np.ndarray, threshold: float) -> dict:
    d = test.copy()
    d["prob"], d["pred"] = prob, (prob >= threshold).astype(int)
    y = d[spec.target]
    d["fp"], d["fn"] = (d["pred"] == 1) & (y == 0), (d["pred"] == 0) & (y == 1)
    by_year = []
    for yr, g in d.groupby("target_year"):
        two = g[spec.target].nunique() == 2
        by_year.append({"target_year": int(yr), "n": int(len(g)), "prevalence": round(float(g[spec.target].mean()), 4),
                        "pr_auc": round(float(average_precision_score(g[spec.target], g["prob"])), 4) if two else None,
                        "recall": round(float(recall_score(g[spec.target], g["pred"], zero_division=0)), 4),
                        "precision": round(float(precision_score(g[spec.target], g["pred"], zero_division=0)), 4),
                        "false_positives": int(g["fp"].sum()), "false_negatives": int(g["fn"].sum())})
    by_state = []
    for st, g in d.groupby("state_name"):
        if len(g) < 40:
            continue
        by_state.append({"state": st, "n": int(len(g)), "prevalence": round(float(g[spec.target].mean()), 4),
                         "recall": round(float(recall_score(g[spec.target], g["pred"], zero_division=0)), 4),
                         "precision": round(float(precision_score(g[spec.target], g["pred"], zero_division=0)), 4),
                         "f1": round(float(f1_score(g[spec.target], g["pred"], zero_division=0)), 4),
                         "false_positives": int(g["fp"].sum()), "false_negatives": int(g["fn"].sum())})
    by_state.sort(key=lambda r: (r["f1"], r["state"]))
    pers = d[spec.persistence_feature] == 1
    missed_new = int((d["fn"] & ~pers).sum())
    return {"threshold": threshold,
            "false_positives": int(d["fp"].sum()), "false_negatives": int(d["fn"].sum()),
            "false_negatives_where_last_year_was_negative": missed_new,
            "false_negatives_where_last_year_was_positive": int((d["fn"] & pers).sum()),
            "by_target_year": by_year, "weakest_states_by_f1": by_state[:5], "strongest_states_by_f1": by_state[-3:],
            "states_reported": len(by_state)}


def permutation_importance(net, scaler, val, spec, repeats: int = 10, seed: int = SEED) -> list[dict]:
    """Drop in validation PR-AUC when one input is shuffled (model-agnostic, approximate)."""
    X = scaler.transform(_prep(val, spec.features))
    y = val[spec.target].to_numpy(dtype=int)
    base = average_precision_score(y, _positive(net, X))
    rng = np.random.default_rng(seed)
    out = []
    for j, f in enumerate(spec.features):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - average_precision_score(y, _positive(net, Xp)))
        out.append({"feature": f, "pr_auc_drop_mean": round(float(np.mean(drops)), 4),
                    "pr_auc_drop_std": round(float(np.std(drops)), 4)})
    return sorted(out, key=lambda r: -r["pr_auc_drop_mean"])


# --------------------------------------------------------------------------
# Hazard-level driver
# --------------------------------------------------------------------------

def load_dataset(hazard: str, base_dir: Path = DEFAULT_OUT_DIR) -> tuple[pd.DataFrame, dict]:
    d = Path(base_dir) / hazard
    return (pd.read_parquet(d / "dataset.parquet"),
            json.loads((d / "dataset_metadata.json").read_text(encoding="utf-8")))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def train_hazard(hazard: str, spec: HazardSpec | None = None, frame: pd.DataFrame | None = None,
                 meta: dict | None = None, model_dir: Path = MODEL_DIR, result_dir: Path = RESULT_DIR,
                 quick: bool = False, plots: bool = True) -> dict:
    spec = spec or SPECS[hazard]
    if frame is None:
        frame, meta = load_dataset(hazard)
    meta = meta or {}
    train, val, test = (frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    t0 = time.time()
    main = run_experiment(train, val, test, spec, quick=quick, keep_models=True)
    fitted = main.pop("_fitted")
    cfg: ANNConfig = fitted["ann_cfg"]
    stability = seed_stability(train, val, test, spec, cfg, seeds=(0, 1) if quick else (0, 1, 2, 3, 4))
    abl = ablation(train, val, test, spec, cfg)
    spatio = spatiotemporal(train, val, test, spec, quick)
    errors = error_analysis(test, spec, fitted["ann_test_prob"], fitted["threshold"])
    importance = permutation_importance(fitted["net"], fitted["scaler"], val, spec, repeats=3 if quick else 10)

    version = MODEL_VERSION.format(hazard=hazard)
    medians = {f: float(train[f].median()) for f in spec.features}
    card = {
        "model_version": version, "hazard": hazard, "title": spec.title,
        "task": "binary classification, next calendar year, per district",
        "target": spec.target, "target_definition": spec.target_definition,
        "geography": spec.geography, "horizon": spec.horizon,
        "label_source": spec.label_source, "label_reliability": spec.label_reliability,
        "limitations": list(spec.limitations),
        "features": spec.features, "feature_groups": spec.groups,
        "architecture": {"input": len(spec.features), "hidden": list(cfg.hidden), "hidden_activation": "relu",
                         "output": "1 unit, logistic sigmoid", "loss": "binary cross-entropy (log-loss)",
                         "optimizer": "Adam", "learning_rate": cfg.learning_rate, "batch_size": cfg.batch_size,
                         "l2_alpha": cfg.alpha, "max_epochs": cfg.max_epochs, "early_stopping_patience": cfg.patience,
                         "class_imbalance": "random oversampling of the minority class in train" if cfg.balance else "none",
                         "seed": cfg.seed, "epochs_run": main["ann_selected_config"]["epochs_run"],
                         "best_epoch": main["ann_selected_config"]["best_epoch"]},
        "training_window": {"train_target_years": meta.get("splits", {}).get("train", {}).get("target_years"),
                            "validation_target_years": meta.get("splits", {}).get("val", {}).get("target_years"),
                            "test_target_years": meta.get("splits", {}).get("test", {}).get("target_years"),
                            "trained_on": "train years only (validation used for early stopping, grid, calibration, threshold)"},
        "dataset_source_sha256": {k: v["sha256"] for k, v in meta.get("source_files", {}).items()},
        "calibration": {"method": "Platt scaling fitted on validation", **fitted["platt"],
                        "validation_forward_check": main["ann_validation_calibration_check"],
                        "display_probability": bool(fitted["platt"]["fitted"]
                                                    and main["ann_validation_calibration_check"]["display_probability"]),
                        "display_rule": (f"probability is shown only if the forward check on validation years has ECE <= "
                                         f"{ECE_ACCEPTABLE}; test years take no part in this decision"),
                        "test_ece_calibrated": main["ann"]["test"]["ece"],
                        "test_ece_raw_score": main["ann"]["test_raw_score_ece"],
                        "test_ece_note": "reported for information only; not used for any decision"},
        "decision_threshold": fitted["threshold"],
        "decision_threshold_note": "chosen on validation to maximise F1 of the calibrated probability",
        "test_metrics": {k: main["ann"]["test"][k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "macro_f1", "brier", "log_loss", "ece", "prevalence", "n")},
        "global_feature_importance": {"method": "permutation on validation set (drop in PR-AUC); approximate, not causal",
                                      "values": importance},
        "training_medians": medians,
        "risk_bands": None,
        "risk_band_status": "unavailable: no validated thresholds (see ANN_DESIGN_DECISION.md section 8)",
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    bundle = {"net": fitted["net"], "scaler": fitted["scaler"], "features": spec.features,
              "platt": fitted["platt"], "threshold": fitted["threshold"],
              "training_medians": medians, "model_version": version, "hazard": hazard}
    mdir = Path(model_dir) / hazard
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, mdir / "ann_bundle.joblib")
    (mdir / "model_card.json").write_text(json.dumps(card, indent=2, default=_json) + "\n",
                                          encoding="utf-8", newline="\n")

    results = {"hazard": hazard, "model_version": version, "dataset": {k: meta.get(k) for k in
               ("rows", "splits", "exclusions", "feature_year_range", "split_rule")},
               "primary_split": main, "seed_stability": stability, "ablation": abl,
               "held_out_states": spatio, "error_analysis": errors,
               "learning_curve": fitted["history"],
               "quick_run": quick}
    rdir = Path(result_dir) / hazard
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "results.json").write_text(json.dumps(results, indent=2, default=_json) + "\n",
                                       encoding="utf-8", newline="\n")
    if plots:
        make_plots(rdir, hazard, main, fitted, test, spec)
    results["_seconds"] = round(time.time() - t0, 1)
    return results


def make_plots(rdir: Path, hazard: str, main: dict, fitted: dict, test: pd.DataFrame, spec: HazardSpec) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except Exception:
        return
    y = test[spec.target].to_numpy(dtype=int)
    p = fitted["ann_test_prob"]
    hist = fitted["history"]
    fig0, ax0 = plt.subplots(figsize=(5, 3.5))
    ax0.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train loss")
    ax0.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="validation loss")
    ax0.set(xlabel="epoch", ylabel="log-loss", title=f"{hazard}: ANN learning curve")
    ax0.legend()
    fig0.tight_layout()
    fig0.savefig(rdir / "ann_learning_curve.png", dpi=110)
    plt.close(fig0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    table = main["ann"]["test"]["reliability_table"]
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].plot([r["mean_predicted"] for r in table], [r["observed_rate"] for r in table], "o-")
    ax[0].set(xlabel="calibrated probability", ylabel="observed rate", title=f"{hazard}: ANN reliability (test)")
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr)
    ax[1].axhline(y.mean(), color="gray", ls=":", label="prevalence")
    ax[1].set(xlabel="recall", ylabel="precision", title=f"{hazard}: ANN precision-recall (test)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(rdir / "ann_reliability_and_pr.png", dpi=110)
    plt.close(fig)


def write_report(all_results: dict[str, dict], path: Path) -> None:
    """Human-readable evaluation report generated from the results (no hand-typed numbers)."""
    L = ["# ANN Evaluation Report (generated)", "",
         "Generated by `python -m ml.hazard_training --hazard all`. All numbers come from "
         "`research/experiments/<hazard>/results.json`. Thresholds, calibration, hyper-parameters and the "
         "probability-display decision use training/validation data only.", "",
         "**Status of the test-year evidence.** The test years (target >= 2022) were first scored in an "
         "exploratory run, before the probability-display rule was fixed. Since then no training, selection, "
         "calibration, threshold or display decision uses them (the models, thresholds and metrics are "
         "bit-identical; only the display rule moved from test to validation data). The test results are therefore "
         "held-out but exploratory-informed, **not a pristine final test**; no unseen period remains in the data. "
         "Test calibration error is shown for information only.", ""]
    for h, r in all_results.items():
        m = r["primary_split"]
        L += [f"## {h}", "", f"Model `{r['model_version']}`; selected config `{m['ann_selected_config']['label']}`; "
              f"split rows {m['rows']}; features {len(m['features'])}.", "",
              "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Macro-F1 | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|"]
        rows = [("**ANN (MLP)**", m["ann"]["test"])] + [(n, b["test"]) for n, b in m["baselines"].items()]
        for n, t in rows:
            L.append(f"| {n} | {t['pr_auc']} | {t['roc_auc']} | {t['precision']} | {t['recall']} | {t['f1']} | "
                     f"{t['macro_f1']} | {t.get('brier')} | {t.get('ece')} |")
        c = m["ann"]["test"]["confusion"]
        s = r["seed_stability"]
        k = m["ann_validation_calibration_check"]
        L += ["", "Probability-display decision (validation years only): Platt fitted on target years "
              f"{k['fit_target_years']} (prevalence {k.get('fit_prevalence')}), ECE measured on {k['eval_target_year']} "
              f"(prevalence {k.get('eval_prevalence')}) = {k['ece']}; limit {k['max_allowed_ece']} -> "
              f"**{'probability shown' if k['display_probability'] else 'probability withheld'}**. "
              f"Test ECE for information only: {m['ann']['test']['ece']} (raw score {m['ann']['test_raw_score_ece']}).",
              "", f"ANN test confusion matrix: TN {c['tn']}, FP {c['fp']}, FN {c['fn']}, TP {c['tp']}; prevalence "
              f"{m['ann']['test']['prevalence']}.",
              f"Seed stability (n={len(s['per_seed'])}): PR-AUC {s['test_pr_auc_mean']} +/- {s['test_pr_auc_std']}, "
              f"F1 {s['test_f1_mean']} +/- {s['test_f1_std']}.", "",
              "Ablation (test PR-AUC): " + "; ".join(f"{k}: {v['test_pr_auc']}" for k, v in r["ablation"].items()) + ".",
              "", "Held-out states (" + ", ".join(r["held_out_states"]["held_out_states"]) + "): ANN PR-AUC "
              f"{r['held_out_states']['ann']['pr_auc']}, F1 {r['held_out_states']['ann']['f1']}; "
              + "; ".join(f"{n} PR-AUC {v['pr_auc']}" for n, v in r["held_out_states"]["baselines"].items()) + ".", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hazard", default="all", choices=[*HAZARDS, "all"])
    p.add_argument("--quick", action="store_true", help="small grid for smoke runs (not for reporting)")
    p.add_argument("--report-only", action="store_true",
                   help="rewrite ml/EVALUATION_REPORT.md from the saved results.json files (no training)")
    a = p.parse_args(argv)
    if a.report_only:
        saved = {h: json.loads((RESULT_DIR / h / "results.json").read_text(encoding="utf-8")) for h in HAZARDS}
        write_report(saved, ROOT / "ml" / "EVALUATION_REPORT.md")
        print("wrote ml/EVALUATION_REPORT.md from saved results")
        return 0
    names = HAZARDS if a.hazard == "all" else (a.hazard,)
    results = {}
    for h in names:
        r = train_hazard(h, quick=a.quick)
        t = r["primary_split"]["ann"]["test"]
        print(f"{h:9} ANN test PR-AUC {t['pr_auc']}  F1 {t['f1']}  Brier {t['brier']}  ECE {t['ece']}  ({r['_seconds']}s)")
        results[h] = r
    if not a.quick and a.hazard == "all":
        write_report(results, ROOT / "ml" / "EVALUATION_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
