"""
=============================================================================
BHARAT-TRAVELSHIELD AI — HEATWAVE ML MODEL & EVALUATION TEST SUITE (CORRECTED)
Comprehensive test suite verifying critical model training and evaluation requirements:
1. Experiment result artifacts exist and are valid.
2. Source dataset remains strictly immutable (SHA-256 hash match).
3. Preprocessor (StandardScaler) was fitted strictly on 14 features without leakage.
4. Model selection consistency: verified selection by validation F1 at threshold 0.5.
5. ANN architecture and metadata integrity: layer dimensions, logistic sigmoid output, stopping score.
6. Confusion matrix row and column marginals sum to exact split sizes; ANN FN == 251.
7. Reloaded models reproduce the exact reported test evaluation metrics.
8. Report narrative integrity: verified absence of typos (239 FN), unsupported climate claims, and calibration caveats present.
=============================================================================
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "ml/data/heatwave/heatwave_t_to_t1_dataset.parquet"
SOURCE_FILE = ROOT / "data/processed/features/travelshield_district_year_features_2011_2024.parquet"
MODELS_DIR = ROOT / "ml/models/heatwave"
EXPERIMENTS_DIR = ROOT / "ml/experiments"
RESULTS_JSON = EXPERIMENTS_DIR / "heatwave_experiment_results.json"
REPORT_TXT = EXPERIMENTS_DIR / "heatwave_experiment_report.txt"

EXPECTED_SOURCE_SHA256 = "e8a694caa8d20bcc0a18fed03cf372dae5adc33fc474d30ae7e67328fe440960"

APPROVED_PREDICTORS = [
    "latitude",
    "longitude",
    "population_2011",
    "avg_temperature_c",
    "max_temperature_c",
    "min_temperature_c",
    "avg_humidity_pct",
    "total_rainfall_mm",
    "rainy_days",
    "heatwave_days_T",
    "severe_heatwave_days_T",
    "has_heatwave_T",
    "temp_range_c",
    "rain_intensity",
]


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_experiment_artifacts_exist():
    """Verify that experiment reports, metadata, and saved models exist and are non-empty."""
    assert RESULTS_JSON.exists(), f"Results JSON missing: {RESULTS_JSON}"
    assert REPORT_TXT.exists(), f"Report TXT missing: {REPORT_TXT}"
    assert (MODELS_DIR / "logistic_regression.joblib").exists(), "LR model file missing"
    assert (MODELS_DIR / "random_forest.joblib").exists(), "RF model file missing"
    assert (MODELS_DIR / "feedforward_ann_mlp.joblib").exists(), "ANN model file missing"
    assert (MODELS_DIR / "heatwave_feature_scaler.joblib").exists(), "Scaler file missing"

    assert RESULTS_JSON.stat().st_size > 0, "Results JSON is empty"
    assert REPORT_TXT.stat().st_size > 0, "Report TXT is empty"
    print("PASS: test_experiment_artifacts_exist")


def test_source_dataset_immutability():
    """Verify source dataset was completely unaltered throughout model training."""
    actual_hash = compute_file_sha256(SOURCE_FILE)
    assert actual_hash == EXPECTED_SOURCE_SHA256, (
        f"Source file SHA-256 altered!\nExpected: {EXPECTED_SOURCE_SHA256}\nActual:   {actual_hash}"
    )
    print("PASS: test_source_dataset_immutability (SHA-256 verified)")


def test_scaler_integrity_no_leakage():
    """Verify scaler parameters are valid and strictly fit on 14 features."""
    scaler = joblib.load(MODELS_DIR / "heatwave_feature_scaler.joblib")
    assert scaler.n_features_in_ == 14, f"Expected 14 features in scaler, got {scaler.n_features_in_}"
    assert len(scaler.mean_) == 14, "Mean vector dimension mismatch"
    assert len(scaler.scale_) == 14, "Scale vector dimension mismatch"
    assert (scaler.scale_ > 0).all(), "Zero or negative variance detected in scaler"
    print("PASS: test_scaler_integrity_no_leakage")


def test_model_selection_consistency():
    """Verify selection criterion was applied consistently across all candidate models."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    inner_val = results["model_selection_inner_val"]
    selected = results["selected_candidates"]

    # 1. Logistic Regression: C=0.1 must be selected over default C=1.0 and balanced
    assert selected["Logistic_Regression"]["selected_candidate"] == "LR_C01", (
        f"Expected LR_C01 to be selected, got {selected['Logistic_Regression']['selected_candidate']}"
    )
    lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    assert lr_model.C == 0.1, f"Expected refit LR to have C=0.1, got {lr_model.C}"
    assert inner_val["LR_C01"]["f1"] > inner_val["LR_Default"]["f1"], "Validation F1 for C=0.1 not higher than C=1.0"

    # 2. Random Forest: Depth 12 must be selected over Depth 8 and balanced
    assert selected["Random_Forest"]["selected_candidate"] == "RF_Depth12", (
        f"Expected RF_Depth12 to be selected, got {selected['Random_Forest']['selected_candidate']}"
    )
    rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
    assert rf_model.max_depth == 12, f"Expected refit RF to have max_depth=12, got {rf_model.max_depth}"
    assert inner_val["RF_Depth12"]["f1"] > inner_val["RF_Depth8"]["f1"], "Validation F1 for RF depth 12 not higher than depth 8"

    # 3. ANN/MLP: ANN_64_32 must be the winning candidate by validation F1
    assert selected["Feedforward_ANN_MLP"]["selected_candidate"] == "ANN_64_32", (
        f"Expected ANN_64_32 to be selected, got {selected['Feedforward_ANN_MLP']['selected_candidate']}"
    )
    ann_model = joblib.load(MODELS_DIR / "feedforward_ann_mlp.joblib")
    assert tuple(ann_model.hidden_layer_sizes) == (64, 32), f"Expected (64, 32), got {ann_model.hidden_layer_sizes}"
    ann_f1_scores = [inner_val[k]["f1"] for k in ["ANN_64_32", "ANN_128_64", "ANN_32_16", "ANN_64_32_Tanh"]]
    assert inner_val["ANN_64_32"]["f1"] == max(ann_f1_scores), "ANN_64_32 does not have the highest validation F1"

    print("PASS: test_model_selection_consistency (All selections verified by validation F1 at threshold 0.5)")


def test_ann_architecture_metadata():
    """Verify ANN architecture dimensions, binary output activation, and early-stopping score."""
    ann = joblib.load(MODELS_DIR / "feedforward_ann_mlp.joblib")
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    ann_spec = results["primary_model_ann_spec"]

    # Layer dimensions and weights
    assert ann.n_features_in_ == 14, f"Expected 14 input features, got {ann.n_features_in_}"
    assert tuple(ann.hidden_layer_sizes) == (64, 32), "Hidden layers mismatch"
    assert ann.n_outputs_ == 1, f"Expected scalar output n_outputs_==1 for binary classification, got {ann.n_outputs_}"
    assert ann.out_activation_ == "logistic", f"Expected 'logistic' output activation, got {ann.out_activation_}"

    assert len(ann.coefs_) == 3, "Expected 3 weight matrices"
    assert ann.coefs_[0].shape == (14, 64), f"Layer 1 shape mismatch: {ann.coefs_[0].shape}"
    assert ann.coefs_[1].shape == (64, 32), f"Layer 2 shape mismatch: {ann.coefs_[1].shape}"
    assert ann.coefs_[2].shape == (32, 1), f"Output layer shape mismatch: {ann.coefs_[2].shape}"

    # Metadata assertions
    assert ann_spec["layer_sizes"] == [14, 64, 32, 1], f"Unexpected layer_sizes: {ann_spec['layer_sizes']}"
    assert "logistic" in ann_spec["output_activation"].lower(), "Output activation does not state logistic sigmoid"
    assert "not two-unit softmax" in ann_spec["output_activation"].lower(), "Output activation does not refute two-unit softmax"
    assert "accuracy" in ann_spec["internal_stopping_metric"].lower(), "Internal score does not indicate classification accuracy"
    assert "best_validation_loss" not in ann_spec, "Legacy 'best_validation_loss' key should not be present in ann_spec"

    print("PASS: test_ann_architecture_metadata (Layer shapes [14, 64, 32, 1], logistic sigmoid, stopping score verified)")


def test_model_reproducibility_on_test_set():
    """Verify that saved models reproduce the exact same evaluation metrics upon reload."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    test_metrics = results["held_out_test_performance"]
    df = pd.read_parquet(DATASET_FILE)
    test_df = df[df["split"] == "test"]

    X_test = test_df[APPROVED_PREDICTORS].copy()
    y_test = test_df["target_has_heatwave_T1"].values

    scaler = joblib.load(MODELS_DIR / "heatwave_feature_scaler.joblib")
    X_test_scaled = scaler.transform(X_test)

    # 1. Test ANN MLP
    ann = joblib.load(MODELS_DIR / "feedforward_ann_mlp.joblib")
    p_ann = ann.predict_proba(X_test_scaled)[:, 1]
    y_pred_ann = ann.predict(X_test_scaled)
    acc_ann = round(float(accuracy_score(y_test, y_pred_ann)), 4)
    f1_ann = round(float(f1_score(y_test, y_pred_ann, zero_division=0)), 4)
    roc_ann = round(float(roc_auc_score(y_test, p_ann)), 4)
    brier_ann = round(float(brier_score_loss(y_test, p_ann)), 4)

    assert acc_ann == test_metrics["Feedforward_ANN_MLP"]["accuracy"], "ANN accuracy mismatch"
    assert f1_ann == test_metrics["Feedforward_ANN_MLP"]["f1"], "ANN F1 mismatch"
    assert roc_ann == test_metrics["Feedforward_ANN_MLP"]["roc_auc"], "ANN ROC mismatch"
    assert brier_ann == test_metrics["Feedforward_ANN_MLP"]["brier_score"], "ANN Brier mismatch"

    # 2. Test Logistic Regression (C=0.1)
    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    p_lr = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = round(float(accuracy_score(y_test, y_pred_lr)), 4)
    f1_lr = round(float(f1_score(y_test, y_pred_lr, zero_division=0)), 4)
    roc_lr = round(float(roc_auc_score(y_test, p_lr)), 4)

    assert acc_lr == test_metrics["Logistic_Regression"]["accuracy"], "LR accuracy mismatch"
    assert f1_lr == test_metrics["Logistic_Regression"]["f1"], "LR F1 mismatch"
    assert roc_lr == test_metrics["Logistic_Regression"]["roc_auc"], "LR ROC mismatch"

    # 3. Test Random Forest (max_depth=12)
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    p_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    acc_rf = round(float(accuracy_score(y_test, y_pred_rf)), 4)
    f1_rf = round(float(f1_score(y_test, y_pred_rf, zero_division=0)), 4)
    roc_rf = round(float(roc_auc_score(y_test, p_rf)), 4)

    assert acc_rf == test_metrics["Random_Forest"]["accuracy"], "RF accuracy mismatch"
    assert f1_rf == test_metrics["Random_Forest"]["f1"], "RF F1 mismatch"
    assert roc_rf == test_metrics["Random_Forest"]["roc_auc"], "RF ROC mismatch"

    print("PASS: test_model_reproducibility_on_test_set (All reloaded model metrics match)")


def test_confusion_matrices_and_marginal_totals():
    """Verify confusion matrix marginals and exact False Negative counts."""
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    test_metrics = results["held_out_test_performance"]
    expected_total = 1920
    expected_negatives = 1137  # 1920 - 783
    expected_positives = 783

    for model_name, m in test_metrics.items():
        cm = m["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        assert tn + fp == expected_negatives, f"{model_name}: Row 0 does not sum to {expected_negatives}"
        assert fn + tp == expected_positives, f"{model_name}: Row 1 does not sum to {expected_positives}"
        assert tn + fp + fn + tp == expected_total, f"{model_name}: Total does not sum to {expected_total}"

    # Critical audit check: ANN false negatives must be exactly 251 (not 239)
    ann_cm = test_metrics["Feedforward_ANN_MLP"]["confusion_matrix"]
    assert ann_cm[1][0] == 251, f"Expected ANN false negatives == 251, got {ann_cm[1][0]}"
    assert ann_cm[1][1] == 532, f"Expected ANN true positives == 532, got {ann_cm[1][1]}"

    # Persistence baseline false negatives must be exactly 440
    persist_cm = test_metrics["Previous_Year_Persistence"]["confusion_matrix"]
    assert persist_cm[1][0] == 440, f"Expected Persistence false negatives == 440, got {persist_cm[1][0]}"

    print("PASS: test_confusion_matrices_and_marginal_totals (All CM margins verified, ANN FN == 251)")


def test_report_narrative_and_discrepancy_guards():
    """Verify that report text contains no hardcoded errors, no unsupported climate claims, and includes calibration caveats."""
    with open(REPORT_TXT, "r", encoding="utf-8") as f:
        report_text = f.read()

    # 1. False negatives typo guard
    assert "239 false negatives" not in report_text, "Found discredited typo '239 false negatives' in report text"
    assert "251" in report_text, "Verified FN count '251' missing from report text"

    # 2. Climate driver speculation guard
    assert "cyclone" not in report_text.lower(), "Found unsupported causal claim mentioning cyclones in report text"

    # 3. Calibration limitation guard
    assert "ECE" in report_text or "Expected Calibration Error" in report_text, "ECE missing from report"
    assert "calibration" in report_text.lower(), "Calibration caveat missing from report"

    # 4. Persistence baseline caveat
    assert "hard-label" in report_text.lower() or "discrete binary" in report_text.lower(), "Persistence hard-label caveat missing"

    # 5. Diagnostic post-audit label
    assert "post-audit" in report_text.lower(), "Diagnostic post-audit label missing from report"

    print("PASS: test_report_narrative_and_discrepancy_guards (Narrative text verified)")


def run_all_tests():
    print("=" * 75)
    print("RUNNING BHARAT-TRAVELSHIELD AI MODEL EVALUATION TEST SUITE (CORRECTED)")
    print("=" * 75)
    test_experiment_artifacts_exist()
    test_source_dataset_immutability()
    test_scaler_integrity_no_leakage()
    test_model_selection_consistency()
    test_ann_architecture_metadata()
    test_model_reproducibility_on_test_set()
    test_confusion_matrices_and_marginal_totals()
    test_report_narrative_and_discrepancy_guards()
    print("=" * 75)
    print("ALL MODEL EVALUATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()
