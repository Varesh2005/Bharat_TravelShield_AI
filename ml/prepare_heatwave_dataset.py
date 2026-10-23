"""
=============================================================================
BHARAT-TRAVELSHIELD AI — ML DATASET PREPARATION PIPELINE
Task: Next-Year District Heatwave Occurrence Classification (T -> T+1)
Target: at least one threshold-based heatwave day in year T+1 (target > 0)
=============================================================================

This script creates a reproducible, leakage-safe, district-partitioned dataset
for predicting next-year heatwaves.

Key Features & Guardrails:
1. Read-only access to master features dataset (SHA-256 hash checked before and after).
2. Deterministic sorting by census_district_code and year.
3. Target construction via an explicit, validated 1-to-1 join on (census_district_code, year + 1).
   This strictly requires a source observation in exact calendar year T+1 for the same district.
   A missing next year produces NaN and is cleanly filtered out; it is never silently skipped.
4. Final year (2024) has no 2025 target and is filtered out BEFORE binary conversion.
   It is never converted to negative (0).
5. Approved 14 candidate predictors strictly from year T or static geography/demography.
6. Fixed chronological splits:
   - Full Training: feature years 2011–2020 (target years 2012–2021) -> 6,400 rows
   - Held-Out Test: feature years 2021–2023 (target years 2022–2024) -> 1,920 rows
   - Inner Train (fit): feature years 2011–2018 (target years 2012–2019) -> 5,120 rows
   - Inner Validation: feature years 2019–2020 (target years 2020–2021) -> 1,280 rows
=============================================================================
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Input master dataset (READ-ONLY)
SOURCE_FILE = (
    ROOT / "data/processed/features/travelshield_district_year_features_2011_2024.parquet"
)

# Output directory inside ml/
OUTPUT_DIR = ROOT / "ml/data/heatwave"
PARQUET_OUT = OUTPUT_DIR / "heatwave_t_to_t1_dataset.parquet"
CSV_OUT = OUTPUT_DIR / "heatwave_t_to_t1_dataset.csv"
METADATA_OUT = OUTPUT_DIR / "heatwave_split_metadata.json"
DICTIONARY_OUT = OUTPUT_DIR / "heatwave_feature_dictionary.csv"
REPORT_OUT = OUTPUT_DIR / "heatwave_dataset_audit_report.txt"


# 14 Approved Candidate Predictors
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

# Metadata columns (kept for tracking, grouping, and split verification)
METADATA_COLS = [
    "census_district_code",
    "year",
    "target_year",
    "split",
    "model_selection_split",
]

# Target columns
TARGET_COLS = [
    "target_heatwave_days_T1",
    "target_has_heatwave_T1",
]


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset() -> pd.DataFrame:
    """Load source master data, construct T+1 targets safely via validated calendar join, and build dataset."""
    print("=" * 70)
    print("BHARAT-TRAVELSHIELD AI: PREPARE HEATWAVE T -> T+1 DATASET")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found at: {SOURCE_FILE}")

    # Compute source file hash before reading
    source_sha256_before = compute_file_sha256(SOURCE_FILE)
    print(f"Source file: {SOURCE_FILE.name}")
    print(f"Source SHA-256 (before): {source_sha256_before}")

    df_raw = pd.read_parquet(SOURCE_FILE)
    print(f"Source raw rows: {len(df_raw):,}, columns: {len(df_raw.columns)}")

    # 1. Source Data Integrity Checks
    # Check for duplicate (census_district_code, year)
    duplicate_count = df_raw.duplicated(subset=["census_district_code", "year"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Source file contains {duplicate_count} duplicate (census_district_code, year) records."
        )

    # Check contiguous calendar years per district
    for code, grp in df_raw.groupby("census_district_code"):
        dist_years = sorted(grp["year"].tolist())
        expected_range = list(range(dist_years[0], dist_years[-1] + 1))
        if dist_years != expected_range:
            missing = set(expected_range) - set(dist_years)
            print(f"WARNING: District {code} has non-contiguous calendar years; missing: {missing}")

    # 2. Deterministic sort by district and year
    df = df_raw.sort_values(["census_district_code", "year"]).reset_index(drop=True)

    # 3. Construct T+1 targets via explicit, validated calendar year join
    # A groupby shift alone is insufficient if years can be missing.
    # We create a target lookup from source data for exact (census_district_code, target_year)
    target_lookup = df_raw[["census_district_code", "year", "heatwave_days"]].copy()
    target_lookup = target_lookup.rename(
        columns={
            "year": "target_year",
            "heatwave_days": "target_heatwave_days_T1",
        }
    )

    # For each row (district d, feature year T), the exact target year must be T + 1
    df["target_year"] = df["year"] + 1

    # Validated 1-to-1 merge ensures each district-year matches at most one exact target observation
    df_merged = df.merge(
        target_lookup,
        on=["census_district_code", "target_year"],
        how="left",
        validate="one_to_one",
    )

    # 4. Filter missing shifted targets BEFORE converting to binary label
    # Final year (2024) has no 2025 target observation and must never be labeled as negative (0).
    # Any missing intermediate year T+1 also produces NaN and is cleanly filtered out.
    unlabeled_mask = df_merged["target_heatwave_days_T1"].isna()
    unlabeled_count = int(unlabeled_mask.sum())
    print(f"Rows with unavailable future targets in exact year T+1: {unlabeled_count:,}")

    df_labeled = df_merged[~unlabeled_mask].copy()

    # Explicitly enforce consecutive calendar year check
    assert (
        df_labeled["target_year"] == df_labeled["year"] + 1
    ).all(), "Consecutive calendar year violation detected!"

    # Cast target counts to integer and derive binary label (1 if heatwave days in T+1 > 0 else 0)
    df_labeled["target_heatwave_days_T1"] = df_labeled["target_heatwave_days_T1"].astype(int)
    df_labeled["target_has_heatwave_T1"] = (df_labeled["target_heatwave_days_T1"] > 0).astype(int)

    # 5. Construct candidate predictors (strictly Year T information)
    df_labeled["heatwave_days_T"] = df_labeled["heatwave_days"].astype(int)
    df_labeled["severe_heatwave_days_T"] = df_labeled["severe_heatwave_days"].astype(int)
    df_labeled["has_heatwave_T"] = (df_labeled["heatwave_days_T"] > 0).astype(int)

    # Derived interaction features
    df_labeled["temp_range_c"] = (
        df_labeled["max_temperature_c"] - df_labeled["min_temperature_c"]
    ).round(2)
    df_labeled["rain_intensity"] = (
        df_labeled["total_rainfall_mm"] / (df_labeled["rainy_days"] + 1.0)
    ).round(4)

    # 6. Fixed chronological split assignment
    # Full split: train (feature years 2011-2020) vs test (feature years 2021-2023)
    df_labeled["split"] = np.where(df_labeled["year"] <= 2020, "train", "test")

    # Inner model selection split:
    # inner_train: 2011-2018 (fit during model selection)
    # inner_val: 2019-2020 (evaluate during model selection)
    # test: 2021-2023 (final test evaluation only)
    conditions = [
        df_labeled["year"] <= 2018,
        df_labeled["year"].isin([2019, 2020]),
        df_labeled["year"] >= 2021,
    ]
    choices = ["inner_train", "inner_val", "test"]
    df_labeled["model_selection_split"] = np.select(conditions, choices, default="unknown")

    # 7. Assemble final DataFrame: Metadata + Predictors + Targets
    final_cols = METADATA_COLS + APPROVED_PREDICTORS + TARGET_COLS
    df_final = df_labeled[final_cols].copy()

    # Sort deterministically
    df_final = df_final.sort_values(["census_district_code", "year"]).reset_index(drop=True)

    # 8. Verify source file hash is unchanged after preparation
    source_sha256_after = compute_file_sha256(SOURCE_FILE)
    print(f"Source SHA-256 (after) : {source_sha256_after}")
    assert (
        source_sha256_before == source_sha256_after
    ), "CRITICAL: SOURCE FILE WAS MODIFIED DURING PREPARATION!"

    # 9. Run pipeline assertion suite
    run_pipeline_assertions(df_raw, df_final)

    return df_final


def run_pipeline_assertions(df_raw: pd.DataFrame, df_final: pd.DataFrame):
    """Run strict verification assertions to ensure leakage safety and schema compliance."""
    print("\nRunning pipeline verification assertions...")

    # Assertion 1: No target labels for unavailable future years
    assert len(df_final) == 8320, f"Expected 8,320 rows, got {len(df_final)}"
    assert df_final["year"].max() == 2023, "Max feature year must be 2023"
    assert 2024 not in df_final["year"].values, "Feature year 2024 must not have a target"
    assert df_final["target_heatwave_days_T1"].isna().sum() == 0, "No null targets allowed"
    assert df_final["target_has_heatwave_T1"].isna().sum() == 0, "No null binary targets allowed"

    # Assertion 2: Each target belongs to the same district as its feature row and exact target_year
    # Cross-verify with source dataset directly
    source_subset = df_raw[["census_district_code", "year", "heatwave_days"]].rename(
        columns={"year": "target_year", "heatwave_days": "expected_hw_days"}
    )
    verification_merge = pd.merge(
        df_final[["census_district_code", "year", "target_year", "target_heatwave_days_T1"]],
        source_subset,
        on=["census_district_code", "target_year"],
        how="inner",
    )
    assert len(verification_merge) == 8320, "Verification merge row count mismatch"
    assert (
        verification_merge["target_heatwave_days_T1"].values
        == verification_merge["expected_hw_days"].values
    ).all(), "Target mismatch! Prepared targets do not equal source heatwave_days for exact target_year."

    # Assertion 3: All feature years precede their corresponding target years
    assert (df_final["target_year"] == df_final["year"] + 1).all(), "Target year must be year + 1"
    assert (df_final["year"] < df_final["target_year"]).all(), "Feature year must precede target year"

    # Assertion 4: No test target year overlaps training target years
    train_target_years = set(df_final[df_final["split"] == "train"]["target_year"].unique())
    test_target_years = set(df_final[df_final["split"] == "test"]["target_year"].unique())
    assert train_target_years == set(range(2012, 2022)), f"Unexpected train targets: {train_target_years}"
    assert test_target_years == {2022, 2023, 2024}, f"Unexpected test targets: {test_target_years}"
    assert len(train_target_years.intersection(test_target_years)) == 0, "Train and test target years overlap!"
    assert max(train_target_years) < min(test_target_years), "Temporal ordering violation between train and test"

    # Assertion 5: Predictor columns exactly match the approved list
    actual_predictors = [c for c in df_final.columns if c in APPROVED_PREDICTORS]
    assert actual_predictors == APPROVED_PREDICTORS, "Predictor columns mismatch approved list"
    assert len(actual_predictors) == 14, f"Expected 14 predictors, got {len(actual_predictors)}"

    # Check that forbidden columns are strictly absent from dataset
    forbidden = [
        "district_name",
        "state_name",
        "heatwave_max_temperature_c",
        "malaria_cases",
        "malaria_present",
        "dengue_cases",
        "dengue_deaths",
        "flood_event_count",
    ]
    for col in forbidden:
        assert col not in df_final.columns, f"Forbidden column found in dataset: {col}"

    # Check zero missing values across all predictors
    missing_dict = df_final[APPROVED_PREDICTORS].isna().sum().to_dict()
    for col, count in missing_dict.items():
        assert count == 0, f"Missing values found in predictor {col}: {count}"

    # Assertion 6: Target consistency
    assert (
        df_final["target_has_heatwave_T1"] == (df_final["target_heatwave_days_T1"] > 0).astype(int)
    ).all(), "Binary target does not match target_heatwave_days_T1 > 0"
    assert (df_final["target_heatwave_days_T1"] >= 0).all(), "Negative target heatwave days found"

    print("All pipeline assertions PASSED successfully!")


def save_metadata_and_dictionary():
    """Save machine-readable split metadata and feature data dictionary."""
    source_sha256 = compute_file_sha256(SOURCE_FILE)

    metadata = {
        "dataset_name": "Bharat-TravelShield Heatwave T-to-T+1 Classification Dataset",
        "task": "Binary Classification (heatwave_days_T+1 > 0)",
        "source_file": "data/processed/features/travelshield_district_year_features_2011_2024.parquet",
        "source_file_sha256": source_sha256,
        "total_rows": 8320,
        "districts_count": 640,
        "feature_years": [2011, 2023],
        "target_years": [2012, 2024],
        "positive_rate_overall_pct": 39.36,
        "candidate_predictors_count": 14,
        "candidate_predictors": APPROVED_PREDICTORS,
        "metadata_columns": METADATA_COLS,
        "target_columns": TARGET_COLS,
        "target_construction_method": "Validated 1-to-1 calendar join on (census_district_code, year + 1)",
        "splits": {
            "full_evaluation": {
                "train": {
                    "feature_years": "2011-2020",
                    "target_years": "2012-2021",
                    "rows": 6400,
                    "positive_count": 2492,
                    "positive_rate_pct": 38.94,
                },
                "test": {
                    "feature_years": "2021-2023",
                    "target_years": "2022-2024",
                    "rows": 1920,
                    "positive_count": 783,
                    "positive_rate_pct": 40.78,
                },
            },
            "model_selection_inner": {
                "inner_train": {
                    "feature_years": "2011-2018",
                    "target_years": "2012-2019",
                    "rows": 5120,
                    "positive_count": 2244,
                    "positive_rate_pct": 43.83,
                    "usage": "Fit models during hyperparameter search & architecture tuning",
                },
                "inner_val": {
                    "feature_years": "2019-2020",
                    "target_years": "2020-2021",
                    "rows": 1280,
                    "positive_count": 248,
                    "positive_rate_pct": 19.38,
                    "usage": "Validate model choices & early stopping (contains 2021 low-heatwave year)",
                },
                "test": {
                    "feature_years": "2021-2023",
                    "target_years": "2022-2024",
                    "rows": 1920,
                    "positive_count": 783,
                    "positive_rate_pct": 40.78,
                    "usage": "Held-out test set evaluated strictly once after final model selection",
                },
            },
        },
    }

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Split metadata saved to: {METADATA_OUT}")

    dict_rows = [
        {"column": "census_district_code", "role": "metadata", "data_type": "int64", "description": "Canonical Census 2011 district code. Excluded from predictor matrix."},
        {"column": "year", "role": "metadata", "data_type": "int64", "description": "Feature year T (information available through end of calendar year T)."},
        {"column": "target_year", "role": "metadata", "data_type": "int64", "description": "Target year T+1 (calendar year of heatwave outcome)."},
        {"column": "split", "role": "metadata", "data_type": "string", "description": "Chronological evaluation split: 'train' (feature years 2011-2020) or 'test' (feature years 2021-2023)."},
        {"column": "model_selection_split", "role": "metadata", "data_type": "string", "description": "Inner chronological split: 'inner_train' (2011-2018), 'inner_val' (2019-2020), or 'test' (2021-2023)."},
        {"column": "latitude", "role": "predictor", "data_type": "float64", "description": "Representative point latitude (degrees) of district geometry."},
        {"column": "longitude", "role": "predictor", "data_type": "float64", "description": "Representative point longitude (degrees) of district geometry."},
        {"column": "population_2011", "role": "predictor", "data_type": "int64", "description": "District population from Census 2011."},
        {"column": "avg_temperature_c", "role": "predictor", "data_type": "float64", "description": "Annual mean daily 2m air temperature in year T (°C)."},
        {"column": "max_temperature_c", "role": "predictor", "data_type": "float64", "description": "Maximum daily T2M_MAX observed during year T (°C)."},
        {"column": "min_temperature_c", "role": "predictor", "data_type": "float64", "description": "Minimum daily T2M_MIN observed during year T (°C)."},
        {"column": "avg_humidity_pct", "role": "predictor", "data_type": "float64", "description": "Annual mean relative humidity at 2m in year T (%)."},
        {"column": "total_rainfall_mm", "role": "predictor", "data_type": "float64", "description": "Annual accumulated precipitation in year T (mm)."},
        {"column": "rainy_days", "role": "predictor", "data_type": "int64", "description": "Number of days with precipitation >= 1mm in year T."},
        {"column": "heatwave_days_T", "role": "predictor", "data_type": "int64", "description": "Days where daily T2M_MAX >= 45°C during year T."},
        {"column": "severe_heatwave_days_T", "role": "predictor", "data_type": "int64", "description": "Days where daily T2M_MAX >= 47°C during year T."},
        {"column": "has_heatwave_T", "role": "predictor", "data_type": "int64", "description": "Binary persistence: 1 if heatwave_days_T > 0, else 0."},
        {"column": "temp_range_c", "role": "predictor", "data_type": "float64", "description": "Annual diurnal/extreme thermal range (max_temperature_c - min_temperature_c) (°C)."},
        {"column": "rain_intensity", "role": "predictor", "data_type": "float64", "description": "Average rainfall per rainy day: total_rainfall_mm / (rainy_days + 1)."},
        {"column": "target_heatwave_days_T1", "role": "target_continuous", "data_type": "int64", "description": "Continuous count of days with T2M_MAX >= 45°C in exact target year T+1."},
        {"column": "target_has_heatwave_T1", "role": "target_binary", "data_type": "int64", "description": "Primary binary classification target: 1 if target_heatwave_days_T1 > 0 else 0."},
    ]
    pd.DataFrame(dict_rows).to_csv(DICTIONARY_OUT, index=False)
    print(f"Data dictionary saved to: {DICTIONARY_OUT}")


def generate_audit_report(df: pd.DataFrame):
    """Generate comprehensive audit report for review."""
    total_rows = len(df)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    itrain_df = df[df["model_selection_split"] == "inner_train"]
    ival_df = df[df["model_selection_split"] == "inner_val"]
    source_sha256 = compute_file_sha256(SOURCE_FILE)

    report_lines = [
        "=" * 70,
        "BHARAT-TRAVELSHIELD AI — HEATWAVE ML DATASET AUDIT REPORT",
        "=" * 70,
        "",
        "1. DATASET OVERVIEW & REPRODUCIBILITY",
        "-" * 70,
        f"Input source file        : {SOURCE_FILE.name}",
        f"Source file SHA-256      : {source_sha256}",
        f"Output parquet file      : {PARQUET_OUT.name}",
        f"Output CSV file          : {CSV_OUT.name}",
        f"Total observation rows   : {total_rows:,}",
        f"Unique districts         : {df['census_district_code'].nunique()}",
        f"Observations per district: {int(total_rows / df['census_district_code'].nunique())}",
        f"Feature years (T)        : {df['year'].min()} to {df['year'].max()} (13 years)",
        f"Target years (T+1)       : {df['target_year'].min()} to {df['target_year'].max()} (13 years)",
        f"Target construction      : Validated 1-to-1 calendar join on (census_district_code, year + 1)",
        f"Boundary handling        : Year 2024 filtered (no 2025 target in master dataset)",
        "",
        "2. TARGET DISTRIBUTION",
        "-" * 70,
        f"Overall positive rows    : {df['target_has_heatwave_T1'].sum():,} / {total_rows:,} ({df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        f"Overall negative rows    : {(df['target_has_heatwave_T1'] == 0).sum():,} / {total_rows:,} ({(1 - df['target_has_heatwave_T1'].mean())*100:.2f}%)",
        "",
        "Year-by-Year Target Breakdown (Feature Year T -> Target Year T+1):",
    ]

    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        pos = sub["target_has_heatwave_T1"].sum()
        pct = sub["target_has_heatwave_T1"].mean() * 100
        tot_days = sub["target_heatwave_days_T1"].sum()
        report_lines.append(
            f"  Feature T={yr} -> Target T+1={yr+1:4d} | Pos: {pos:3d}/640 ({pct:5.1f}%) | Total HW Days: {tot_days:5d}"
        )

    report_lines.extend([
        "",
        "3. CHRONOLOGICAL SPLIT BOUNDARIES",
        "-" * 70,
        "A. Full Evaluation Split (for final model refitting & testing):",
        f"  - Train Set : Feature Years 2011–2020 -> Target Years 2012–2021",
        f"                Rows: {len(train_df):,} ({len(train_df)/total_rows*100:.1f}%) | Positives: {train_df['target_has_heatwave_T1'].sum():,} ({train_df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        f"  - Test Set  : Feature Years 2021–2023 -> Target Years 2022–2024",
        f"                Rows: {len(test_df):,} ({len(test_df)/total_rows*100:.1f}%) | Positives: {test_df['target_has_heatwave_T1'].sum():,} ({test_df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        f"  - Overlap   : ZERO. Max train target year (2021) < Min test target year (2022).",
        "",
        "B. Model Selection Inner Split (for architecture tuning & early stopping):",
        f"  - Inner Train : Feature Years 2011–2018 -> Target Years 2012–2019",
        f"                  Rows: {len(itrain_df):,} ({len(itrain_df)/total_rows*100:.1f}%) | Positives: {itrain_df['target_has_heatwave_T1'].sum():,} ({itrain_df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        f"  - Inner Val   : Feature Years 2019–2020 -> Target Years 2020–2021",
        f"                  Rows: {len(ival_df):,} ({len(ival_df)/total_rows*100:.1f}%) | Positives: {ival_df['target_has_heatwave_T1'].sum():,} ({ival_df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        f"  - Test (Held) : Feature Years 2021–2023 -> Target Years 2022–2024",
        f"                  Rows: {len(test_df):,} ({len(test_df)/total_rows*100:.1f}%) | Positives: {test_df['target_has_heatwave_T1'].sum():,} ({test_df['target_has_heatwave_T1'].mean()*100:.2f}%)",
        "",
        "4. FEATURE SCHEMA & MISSING VALUES",
        "-" * 70,
        f"Approved candidate predictors: {len(APPROVED_PREDICTORS)}",
    ])

    for i, col in enumerate(APPROVED_PREDICTORS, 1):
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        vmin = df[col].min()
        vmax = df[col].max()
        report_lines.append(
            f"  {i:2d}. {col:<24} | Type: {str(dtype):<7} | Nulls: {nulls} | Range: [{vmin}, {vmax}]"
        )

    report_lines.extend([
        "",
        "5. EXCLUSION AUDIT & LEAKAGE PREVENTION",
        "-" * 70,
        "  [PASS] Target construction        : Explicit join on (census_district_code, year + 1)",
        "  [PASS] Missing year guard         : Non-consecutive years produce NaN and are dropped",
        "  [PASS] heatwave_max_temperature_c : EXCLUDED (100% duplicate of max_temperature_c)",
        "  [PASS] district_name, state_name  : EXCLUDED (high-cardinality text)",
        "  [PASS] census_district_code       : KEPT IN METADATA ONLY (not in candidate predictors)",
        "  [PASS] year, target_year          : KEPT IN METADATA ONLY (not in candidate predictors)",
        "  [PASS] Disease columns            : EXCLUDED (malaria_cases, malaria_present, dengue_cases, dengue_deaths)",
        "  [PASS] Disaster columns           : EXCLUDED (flood_event_count)",
        "  [PASS] Year T+1 future columns    : EXCLUDED from predictors (used only as target labels)",
        "  [PASS] Source immutability        : VERIFIED (SHA-256 before == SHA-256 after)",
        "",
        "6. AUDIT STATUS",
        "-" * 70,
        "STATUS: PASSED (All validation assertions satisfied)",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Audit report saved to: {REPORT_OUT}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_prepared = prepare_dataset()

    print(f"\nSaving parquet dataset to: {PARQUET_OUT}")
    df_prepared.to_parquet(PARQUET_OUT, index=False)

    print(f"Saving CSV dataset to: {CSV_OUT}")
    df_prepared.to_csv(CSV_OUT, index=False)

    save_metadata_and_dictionary()
    generate_audit_report(df_prepared)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
