"""
=============================================================================
BHARAT-TRAVELSHIELD AI — HEATWAVE ML DATASET TEST SUITE
Test Suite verifying critical dataset preparation requirements:
1. Source dataset immutability (SHA-256 cryptographic hash verification).
2. Source dataset integrity (zero duplicates, calendar year continuity).
3. Target equivalence: every prepared target strictly equals the source
   heatwave_days value for the same district and exact target_year (T+1).
4. Missing intermediate year simulation: verifies that missing consecutive
   years produce NaN and are dropped, never silently shifted from T+2.
5. No target labels for unavailable future years (year 2024 filtered, never 0).
6. Temporal precedence: feature year T strictly precedes target year T+1.
7. Fixed chronological splits: verified row counts, year ranges, and zero overlap.
8. Predictor columns: exactly 14 approved predictors; forbidden columns excluded.
9. Target consistency: binary target matches target_heatwave_days_T1 > 0.
=============================================================================
"""

import hashlib
import json
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASET_PARQUET = ROOT / "ml/data/heatwave/heatwave_t_to_t1_dataset.parquet"
DATASET_CSV = ROOT / "ml/data/heatwave/heatwave_t_to_t1_dataset.csv"
METADATA_FILE = ROOT / "ml/data/heatwave/heatwave_split_metadata.json"
DICTIONARY_FILE = ROOT / "ml/data/heatwave/heatwave_feature_dictionary.csv"
REPORT_FILE = ROOT / "ml/data/heatwave/heatwave_dataset_audit_report.txt"
SOURCE_FILE = ROOT / "data/processed/features/travelshield_district_year_features_2011_2024.parquet"

# Known baseline SHA-256 hash of the master source dataset
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
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_artifacts_exist():
    """Verify that all generated files exist and are non-empty."""
    assert DATASET_PARQUET.exists(), f"Parquet dataset missing: {DATASET_PARQUET}"
    assert DATASET_CSV.exists(), f"CSV dataset missing: {DATASET_CSV}"
    assert METADATA_FILE.exists(), f"Metadata file missing: {METADATA_FILE}"
    assert DICTIONARY_FILE.exists(), f"Dictionary file missing: {DICTIONARY_FILE}"
    assert REPORT_FILE.exists(), f"Audit report missing: {REPORT_FILE}"

    assert DATASET_PARQUET.stat().st_size > 0, "Parquet dataset is empty"
    assert DATASET_CSV.stat().st_size > 0, "CSV dataset is empty"
    assert METADATA_FILE.stat().st_size > 0, "Metadata file is empty"
    print("PASS: test_artifacts_exist")


def test_source_immutability_and_hash():
    """Verify source dataset is completely unchanged using cryptographic hash."""
    actual_hash = compute_file_sha256(SOURCE_FILE)
    assert actual_hash == EXPECTED_SOURCE_SHA256, (
        f"Source file SHA-256 mismatch!\nExpected: {EXPECTED_SOURCE_SHA256}\nActual:   {actual_hash}"
    )
    print("PASS: test_source_immutability_and_hash (SHA-256 verified)")


def test_source_integrity_duplicates_and_continuity():
    """Verify source dataset contains zero duplicates and contiguous calendar years."""
    df_source = pd.read_parquet(SOURCE_FILE)

    # 1. Zero duplicate (census_district_code, year)
    dups = df_source.duplicated(subset=["census_district_code", "year"]).sum()
    assert dups == 0, f"Source dataset contains {dups} duplicate district-year records"

    # 2. Exactly 640 districts and 14 years (2011 to 2024)
    assert df_source["census_district_code"].nunique() == 640, "District count != 640"
    assert df_source["year"].min() == 2011, "Source min year != 2011"
    assert df_source["year"].max() == 2024, "Source max year != 2024"

    # 3. Check every district has contiguous years from 2011 to 2024
    expected_years = list(range(2011, 2025))
    for code, grp in df_source.groupby("census_district_code"):
        dist_years = sorted(grp["year"].tolist())
        assert dist_years == expected_years, f"District {code} has non-contiguous years: {dist_years}"

    print("PASS: test_source_integrity_duplicates_and_continuity")


def test_target_equivalence_against_source():
    """Verify every prepared target equals source heatwave_days for same district and exact target_year."""
    df_prepared = pd.read_parquet(DATASET_PARQUET)
    df_source = pd.read_parquet(SOURCE_FILE)

    # Create source lookup table keyed on (census_district_code, year)
    source_lookup = df_source.set_index(["census_district_code", "year"])["heatwave_days"]

    # Verify every single row in df_prepared
    mismatches = 0
    for idx, row in df_prepared.iterrows():
        code = int(row["census_district_code"])
        feat_yr = int(row["year"])
        tgt_yr = int(row["target_year"])
        prepared_tgt = int(row["target_heatwave_days_T1"])

        # Target year must strictly be feature year + 1
        assert tgt_yr == feat_yr + 1, f"Row {idx}: target_year {tgt_yr} != year + 1 ({feat_yr + 1})"

        # Lookup in source
        key = (code, tgt_yr)
        assert key in source_lookup.index, f"Target key {key} not found in source dataset!"
        source_hw = int(source_lookup.loc[key])

        if prepared_tgt != source_hw:
            mismatches += 1

    assert mismatches == 0, f"Found {mismatches} target value mismatches against source dataset!"
    print(f"PASS: test_target_equivalence_against_source (All {len(df_prepared):,} rows verified)")


def test_missing_intermediate_year_handling_simulation():
    """Verify that a missing intermediate calendar year produces NaN and is dropped, never shifted from T+2."""
    # Synthetic test panel: District 999 has years [2011, 2012, 2014] (missing 2013)
    synth_data = pd.DataFrame([
        {"census_district_code": 999, "year": 2011, "heatwave_days": 2},
        {"census_district_code": 999, "year": 2012, "heatwave_days": 5},
        {"census_district_code": 999, "year": 2014, "heatwave_days": 9},  # Note: 2013 is missing!
    ])

    # 1. Naive shift(-1) behavior (FAILURE MODE):
    synth_shift = synth_data.copy()
    synth_shift["naive_shifted_target"] = synth_shift.groupby("census_district_code")["heatwave_days"].shift(-1)
    # Under naive shift, year 2012 falsely gets year 2014's value (9)!
    val_2012_naive = synth_shift.loc[synth_shift["year"] == 2012, "naive_shifted_target"].values[0]
    assert val_2012_naive == 9, "Naive shift should have produced the erroneous 2014 value"

    # 2. Validated calendar join behavior (CORRECT PIPELINE IMPLEMENTATION):
    lookup = synth_data[["census_district_code", "year", "heatwave_days"]].rename(
        columns={"year": "target_year", "heatwave_days": "target_heatwave_days_T1"}
    )
    synth_prep = synth_data.copy()
    synth_prep["target_year"] = synth_prep["year"] + 1
    synth_merged = synth_prep.merge(lookup, on=["census_district_code", "target_year"], how="left", validate="one_to_one")

    # In validated join, year 2012 looks for target_year 2013. Since 2013 is missing, it gets NaN!
    val_2012_join = synth_merged.loc[synth_merged["year"] == 2012, "target_heatwave_days_T1"].values[0]
    assert pd.isna(val_2012_join), f"Expected NaN for missing year 2013, got {val_2012_join}"

    # After filtering missing targets, year 2012 is dropped safely; it never borrows 2014
    synth_labeled = synth_merged[synth_merged["target_heatwave_days_T1"].notna()]
    assert 2012 not in synth_labeled["year"].values, "Year 2012 with missing 2013 target must be dropped"
    assert len(synth_labeled) == 1, "Only year 2011 (which has 2012) should survive"
    assert synth_labeled["year"].iloc[0] == 2011
    assert synth_labeled["target_heatwave_days_T1"].iloc[0] == 5

    print("PASS: test_missing_intermediate_year_handling_simulation (Missing next year safely dropped)")


def test_no_target_labels_for_unavailable_future_years():
    """Verify no target labels for unavailable future years (year 2024 has no target and is not 0)."""
    df = pd.read_parquet(DATASET_PARQUET)

    assert 2024 not in df["year"].values, "Feature year 2024 must not be in dataset (no target for 2025)"
    assert df["year"].max() == 2023, f"Expected max feature year 2023, got {df['year'].max()}"
    assert df["target_year"].max() == 2024, f"Expected max target year 2024, got {df['target_year'].max()}"
    assert len(df) == 8320, f"Expected exactly 8,320 rows (640 districts * 13 years), got {len(df)}"
    assert df["target_heatwave_days_T1"].isna().sum() == 0, "Null values found in target"
    assert df["target_has_heatwave_T1"].isna().sum() == 0, "Null values found in binary target"
    print("PASS: test_no_target_labels_for_unavailable_future_years")


def test_feature_years_precede_target_years():
    """Verify all feature years strictly precede their corresponding target years."""
    df = pd.read_parquet(DATASET_PARQUET)

    assert (df["target_year"] == df["year"] + 1).all(), "Target year must be strictly year + 1"
    assert (df["year"] < df["target_year"]).all(), "Feature year must precede target year"
    print("PASS: test_feature_years_precede_target_years")


def test_chronological_splits_and_no_target_overlap():
    """Verify chronological split boundaries, row counts, and zero target year overlap directly from data."""
    df = pd.read_parquet(DATASET_PARQUET)

    # 1. Full Evaluation Split
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    assert train_mask.sum() == 6400, f"Expected 6,400 train rows, got {train_mask.sum()}"
    assert test_mask.sum() == 1920, f"Expected 1,920 test rows, got {test_mask.sum()}"

    train_feat_years = set(df.loc[train_mask, "year"].unique())
    train_tgt_years = set(df.loc[train_mask, "target_year"].unique())
    test_feat_years = set(df.loc[test_mask, "year"].unique())
    test_tgt_years = set(df.loc[test_mask, "target_year"].unique())

    assert train_feat_years == set(range(2011, 2021)), f"Train feature years mismatch: {train_feat_years}"
    assert train_tgt_years == set(range(2012, 2022)), f"Train target years mismatch: {train_tgt_years}"
    assert test_feat_years == {2021, 2022, 2023}, f"Test feature years mismatch: {test_feat_years}"
    assert test_tgt_years == {2022, 2023, 2024}, f"Test target years mismatch: {test_tgt_years}"

    # Overlap assertions
    target_overlap = train_tgt_years.intersection(test_tgt_years)
    assert len(target_overlap) == 0, f"Target year overlap between train and test: {target_overlap}"
    assert max(train_tgt_years) < min(test_tgt_years), "Temporal ordering violation between train and test targets"

    # 2. Inner Model Selection Split
    itrain_mask = df["model_selection_split"] == "inner_train"
    ival_mask = df["model_selection_split"] == "inner_val"

    assert itrain_mask.sum() == 5120, f"Expected 5,120 inner_train rows, got {itrain_mask.sum()}"
    assert ival_mask.sum() == 1280, f"Expected 1,280 inner_val rows, got {ival_mask.sum()}"
    assert itrain_mask.sum() + ival_mask.sum() == train_mask.sum(), "Inner train + inner val != full train"

    itrain_feat_years = set(df.loc[itrain_mask, "year"].unique())
    itrain_tgt_years = set(df.loc[itrain_mask, "target_year"].unique())
    ival_feat_years = set(df.loc[ival_mask, "year"].unique())
    ival_tgt_years = set(df.loc[ival_mask, "target_year"].unique())

    assert itrain_feat_years == set(range(2011, 2019)), f"Inner train feature years mismatch: {itrain_feat_years}"
    assert itrain_tgt_years == set(range(2012, 2020)), f"Inner train target years mismatch: {itrain_tgt_years}"
    assert ival_feat_years == {2019, 2020}, f"Inner val feature years mismatch: {ival_feat_years}"
    assert ival_tgt_years == {2020, 2021}, f"Inner val target years mismatch: {ival_tgt_years}"

    print("PASS: test_chronological_splits_and_no_target_overlap (All split boundaries verified)")


def test_predictor_columns_exact_match_and_exclusions():
    """Verify predictor columns match approved list and forbidden columns are strictly absent."""
    df = pd.read_parquet(DATASET_PARQUET)

    non_predictor_cols = {
        "census_district_code",
        "year",
        "target_year",
        "split",
        "model_selection_split",
        "target_heatwave_days_T1",
        "target_has_heatwave_T1",
    }

    actual_predictors = [c for c in df.columns if c not in non_predictor_cols]
    assert actual_predictors == APPROVED_PREDICTORS, (
        f"Predictor columns mismatch!\nActual:   {actual_predictors}\nExpected: {APPROVED_PREDICTORS}"
    )
    assert len(actual_predictors) == 14, f"Expected 14 predictors, got {len(actual_predictors)}"

    # Strictly check that forbidden columns are NOT present anywhere in the dataframe
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
        assert col not in df.columns, f"Forbidden column {col} present in dataset!"

    # Ensure 0 nulls across all predictors
    assert df[APPROVED_PREDICTORS].isna().sum().sum() == 0, "Missing values in predictors"
    print("PASS: test_predictor_columns_exact_match_and_exclusions")


def test_target_consistency_and_validity():
    """Verify target values are non-negative, binary matches heatwave_days_T1 > 0, and no nulls."""
    df = pd.read_parquet(DATASET_PARQUET)

    # 1. Non-negative counts
    assert (df["target_heatwave_days_T1"] >= 0).all(), "Negative target heatwave days found!"

    # 2. Binary target strictly matches threshold rule
    expected_binary = (df["target_heatwave_days_T1"] > 0).astype(int)
    assert (df["target_has_heatwave_T1"] == expected_binary).all(), (
        "Binary target mismatch with (target_heatwave_days_T1 > 0)!"
    )

    # 3. Binary target values only {0, 1}
    assert set(df["target_has_heatwave_T1"].unique()) == {0, 1}, "Binary target has values outside {0, 1}"

    # 4. Zero nulls anywhere in the prepared dataset
    assert df.isna().sum().sum() == 0, "Null values detected in prepared dataset!"

    # 5. Overall positive rate check
    pos_rate = df["target_has_heatwave_T1"].mean() * 100
    assert 39.0 <= pos_rate <= 40.0, f"Unexpected overall positive rate: {pos_rate:.2f}%"

    print("PASS: test_target_consistency_and_validity")


def run_all_tests():
    print("=" * 70)
    print("RUNNING BHARAT-TRAVELSHIELD AI HEATWAVE ML DATASET TEST SUITE")
    print("=" * 70)
    test_artifacts_exist()
    test_source_immutability_and_hash()
    test_source_integrity_duplicates_and_continuity()
    test_target_equivalence_against_source()
    test_missing_intermediate_year_handling_simulation()
    test_no_target_labels_for_unavailable_future_years()
    test_feature_years_precede_target_years()
    test_chronological_splits_and_no_target_overlap()
    test_predictor_columns_exact_match_and_exclusions()
    test_target_consistency_and_validity()
    print("=" * 70)
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
