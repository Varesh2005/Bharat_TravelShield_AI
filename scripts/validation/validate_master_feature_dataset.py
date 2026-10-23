from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/features/"
    "travelshield_district_year_features_2011_2024.csv"
)

REPORT = (
    ROOT / "data/reports/"
    "master_feature_validation_report.txt"
)


def main():

    print("=" * 70)
    print("VALIDATE MASTER TRAVELSHIELD FEATURE DATASET")
    print("=" * 70)

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(
        f"Rows loaded : {len(df):,}"
    )

    errors = []
    warnings = []

    # =========================================================
    # REQUIRED SCHEMA
    # =========================================================

    required_columns = [
        "census_district_code",
        "district_name",
        "state_name",
        "population_2011",
        "latitude",
        "longitude",
        "year",

        "avg_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "avg_humidity_pct",
        "total_rainfall_mm",
        "rainy_days",

        "malaria_cases",
        "malaria_present",

        "dengue_cases",
        "dengue_deaths",

        "flood_event_count",

        "heatwave_days",
        "severe_heatwave_days",
        "heatwave_max_temperature_c",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )

    # =========================================================
    # BASIC SHAPE
    # =========================================================

    expected_rows = 640 * 14

    if len(df) != expected_rows:
        errors.append(
            f"Expected {expected_rows:,} rows, "
            f"found {len(df):,}"
        )

    if df["census_district_code"].nunique() != 640:
        errors.append(
            "Expected exactly 640 unique districts."
        )

    if df["year"].min() != 2011:
        errors.append(
            f"Minimum year is {df['year'].min()}, expected 2011."
        )

    if df["year"].max() != 2024:
        errors.append(
            f"Maximum year is {df['year'].max()}, expected 2024."
        )

    # =========================================================
    # DUPLICATE KEYS
    # =========================================================

    duplicate_count = df.duplicated(
        subset=[
            "census_district_code",
            "year"
        ]
    ).sum()

    if duplicate_count > 0:
        errors.append(
            f"Duplicate district-year rows: {duplicate_count}"
        )

    # =========================================================
    # MISSING CANONICAL GEOGRAPHY
    # =========================================================

    geography_columns = [
        "census_district_code",
        "district_name",
        "state_name",
        "population_2011",
        "latitude",
        "longitude",
    ]

    for col in geography_columns:

        count = df[col].isna().sum()

        if count > 0:
            errors.append(
                f"Missing canonical geography values "
                f"in {col}: {count}"
            )

    # =========================================================
    # POPULATION
    # =========================================================

    if (
        df["population_2011"] <= 0
    ).any():
        errors.append(
            "Non-positive population detected."
        )

    # =========================================================
    # COORDINATES
    # =========================================================

    if (
        (df["latitude"] < -90) |
        (df["latitude"] > 90)
    ).any():
        errors.append(
            "Invalid latitude detected."
        )

    if (
        (df["longitude"] < -180) |
        (df["longitude"] > 180)
    ).any():
        errors.append(
            "Invalid longitude detected."
        )

    # =========================================================
    # WEATHER
    # =========================================================

    if (
        df["avg_humidity_pct"].dropna().lt(0).any()
        or
        df["avg_humidity_pct"].dropna().gt(100).any()
    ):
        errors.append(
            "Humidity outside 0-100% range."
        )

    if (
        df["total_rainfall_mm"].dropna() < 0
    ).any():
        errors.append(
            "Negative rainfall detected."
        )

    if (
        df["rainy_days"].dropna() < 0
    ).any():
        errors.append(
            "Negative rainy-day count detected."
        )

    if (
        df["rainy_days"].dropna() > 366
    ).any():
        errors.append(
            "Rainy days greater than 366 detected."
        )

    # =========================================================
    # DISEASE
    # =========================================================

    for col in [
        "malaria_cases",
        "dengue_cases",
        "dengue_deaths",
    ]:

        if (
            df[col].dropna() < 0
        ).any():
            errors.append(
                f"Negative values detected in {col}."
            )

    # =========================================================
    # DISASTER
    # =========================================================

    for col in [
        "flood_event_count",
        "heatwave_days",
        "severe_heatwave_days",
    ]:

        if (
            df[col].dropna() < 0
        ).any():
            errors.append(
                f"Negative values detected in {col}."
            )

    if (
        df["heatwave_days"].dropna() > 366
    ).any():
        errors.append(
            "Heatwave days greater than 366 detected."
        )

    if (
        df["severe_heatwave_days"].dropna() > 366
    ).any():
        errors.append(
            "Severe heatwave days greater than 366 detected."
        )

    # =========================================================
    # HEATWAVE LOGIC
    # =========================================================

    invalid_severe = (
        df["severe_heatwave_days"].notna()
        &
        df["heatwave_days"].notna()
        &
        (
            df["severe_heatwave_days"]
            >
            df["heatwave_days"]
        )
    )

    if invalid_severe.any():
        errors.append(
            "Severe heatwave days exceed total heatwave days."
        )

    # =========================================================
    # TEMPERATURE LOGIC
    # =========================================================

    invalid_temp = (
        df["max_temperature_c"].notna()
        &
        df["min_temperature_c"].notna()
        &
        (
            df["max_temperature_c"]
            <
            df["min_temperature_c"]
        )
    )

    if invalid_temp.any():
        errors.append(
            "Maximum temperature below minimum temperature."
        )

    # =========================================================
    # MISSINGNESS REPORT
    # =========================================================

    feature_columns = [
        "avg_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "avg_humidity_pct",
        "total_rainfall_mm",
        "rainy_days",
        "malaria_cases",
        "dengue_cases",
        "dengue_deaths",
        "flood_event_count",
        "heatwave_days",
        "severe_heatwave_days",
    ]

    missingness = {}

    for col in feature_columns:

        count = int(
            df[col].isna().sum()
        )

        percentage = (
            count / len(df)
        ) * 100

        missingness[col] = (
            count,
            percentage
        )

    # =========================================================
    # EXPECTED MISSINGNESS
    # =========================================================

    # Weather and heatwave should be complete.
    for col in [
        "avg_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "avg_humidity_pct",
        "total_rainfall_mm",
        "rainy_days",
        "heatwave_days",
        "severe_heatwave_days",
    ]:

        if df[col].isna().any():
            warnings.append(
                f"{col} contains missing values."
            )

    # Disease/disaster missingness is expected because
    # source coverage is incomplete.
    warnings.append(
        "Missing malaria/dengue/flood values are retained "
        "as missing and must not automatically be converted "
        "to zero."
    )

    # =========================================================
    # REPORT
    # =========================================================

    report = []

    report.append(
        "TRAVELSHIELD MASTER FEATURE DATASET VALIDATION"
    )
    report.append("=" * 60)
    report.append("")

    report.append(
        f"Rows: {len(df):,}"
    )

    report.append(
        f"Unique districts: "
        f"{df['census_district_code'].nunique():,}"
    )

    report.append(
        f"Year range: "
        f"{df['year'].min()} - {df['year'].max()}"
    )

    report.append("")

    report.append("MISSINGNESS")
    report.append("-" * 60)

    for col, (count, pct) in missingness.items():

        report.append(
            f"{col}: {count:,} "
            f"({pct:.2f}%)"
        )

    report.append("")
    report.append("WARNINGS")
    report.append("-" * 60)

    if warnings:

        for warning in warnings:
            report.append(
                f"- {warning}"
            )

    else:
        report.append(
            "None"
        )

    report.append("")
    report.append("ERRORS")
    report.append("-" * 60)

    if errors:

        for error in errors:
            report.append(
                f"- {error}"
            )

    else:
        report.append(
            "None"
        )

    report.append("")

    if errors:

        report.append(
            "STATUS: FAILED"
        )

    else:

        report.append(
            "STATUS: PASSED"
        )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # =========================================================
    # TERMINAL OUTPUT
    # =========================================================

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Rows              : {len(df):,}"
    )

    print(
        f"Districts         : "
        f"{df['census_district_code'].nunique():,}"
    )

    print(
        f"Year range        : "
        f"{df['year'].min()} - "
        f"{df['year'].max()}"
    )

    print()
    print("Missingness:")
    print("-" * 70)

    for col, (count, pct) in missingness.items():

        print(
            f"{col:<30} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    print()

    if errors:

        print("ERRORS:")
        for error in errors:
            print(f"  ❌ {error}")

        print()
        print("STATUS: FAILED")

        raise SystemExit(1)

    else:

        print("Errors: 0")
        print(
            f"Warnings: {len(warnings)}"
        )
        print()
        print(
            f"REPORT : {REPORT}"
        )
        print()
        print("STATUS: PASSED")


if __name__ == "__main__":
    main()