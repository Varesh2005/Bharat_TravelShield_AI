from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[2]

MASTER_FILE = ROOT / "data/reference/district_master.csv"

WEATHER_FILE = (
    ROOT / "data/processed/weather/"
    "weather_district_year_2011_2025.csv"
)

MALARIA_FILE = (
    ROOT / "data/processed/disease/"
    "malaria_district_year_features_2000_2024.csv"
)

DENGUE_FILE = (
    ROOT / "data/processed/disease/"
    "dengue_state_year_features_2021_2026.csv"
)

FLOOD_FILE = (
    ROOT / "data/processed/disaster/flood/"
    "flood_district_year_1967_2023.csv"
)

HEATWAVE_FILE = (
    ROOT / "data/processed/disaster/heatwave/"
    "heatwave_district_year_2011_2025.csv"
)

OUT_DIR = ROOT / "data/processed/features"

OUT_FILE = (
    OUT_DIR / "travelshield_district_year_features_2011_2024.csv"
)

REPORT_FILE = (
    ROOT / "data/reports/"
    "travelshield_master_feature_build_report.txt"
)


def normalize_state(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    aliases = {
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "pondicherry": "puducherry",
        "nct of delhi": "delhi",
        "delhi nct": "delhi",
        "jammu and kashmir": "jammu kashmir",
        "jammu & kashmir": "jammu kashmir",
    }

    return aliases.get(x, x)


def load_check(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        f"{name:<12}: {len(df):,} rows"
    )

    return df


def main():

    print("=" * 70)
    print("BUILD MASTER TRAVELSHIELD FEATURE DATASET")
    print("=" * 70)

    # =========================================================
    # 1. MASTER GEOGRAPHY
    # =========================================================

    master = load_check(
        MASTER_FILE,
        "MASTER"
    )

    required_master = [
        "census_district_code",
        "district_name",
        "state_name",
        "population_2011",
        "latitude",
        "longitude",
    ]

    missing = [
        c for c in required_master
        if c not in master.columns
    ]

    if missing:
        raise ValueError(
            f"Missing master columns: {missing}"
        )

    master = master[
        required_master
    ].copy()

    master["census_district_code"] = pd.to_numeric(
        master["census_district_code"],
        errors="coerce"
    )

    master["population_2011"] = pd.to_numeric(
        master["population_2011"],
        errors="coerce"
    )

    master["latitude"] = pd.to_numeric(
        master["latitude"],
        errors="coerce"
    )

    master["longitude"] = pd.to_numeric(
        master["longitude"],
        errors="coerce"
    )

    if master["census_district_code"].duplicated().any():
        raise ValueError(
            "Duplicate Census district codes in master."
        )

    # =========================================================
    # 2. CREATE 2011-2024 BACKBONE
    # =========================================================

    years = pd.DataFrame(
        {
            "year": range(2011, 2025)
        }
    )

    master["_key"] = 1
    years["_key"] = 1

    final = master.merge(
        years,
        on="_key",
        how="outer"
    ).drop(
        columns="_key"
    )

    print()
    print(
        f"Backbone rows : {len(final):,}"
    )

    # =========================================================
    # 3. WEATHER
    # =========================================================

    weather = load_check(
        WEATHER_FILE,
        "WEATHER"
    )

    weather["census_district_code"] = pd.to_numeric(
        weather["census_district_code"],
        errors="coerce"
    )

    weather["year"] = pd.to_numeric(
        weather["year"],
        errors="coerce"
    )

    weather = weather[
        (weather["year"] >= 2011) &
        (weather["year"] <= 2024)
    ].copy()

    weather_cols = [
        "census_district_code",
        "year",
        "avg_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "avg_humidity_pct",
        "total_rainfall_mm",
        "rainy_days",
    ]

    weather = weather[
        weather_cols
    ]

    if weather.duplicated(
        ["census_district_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate weather district-year rows."
        )

    final = final.merge(
        weather,
        on=[
            "census_district_code",
            "year"
        ],
        how="left",
        validate="one_to_one"
    )

    # =========================================================
    # 4. MALARIA
    # =========================================================

    malaria = load_check(
        MALARIA_FILE,
        "MALARIA"
    )

    malaria["census_district_code"] = pd.to_numeric(
        malaria["census_district_code"],
        errors="coerce"
    )

    malaria["year"] = pd.to_numeric(
        malaria["year"],
        errors="coerce"
    )

    malaria = malaria[
        (malaria["year"] >= 2011) &
        (malaria["year"] <= 2024)
    ].copy()

    malaria_cols = [
        "census_district_code",
        "year",
        "malaria_cases",
        "malaria_present",
    ]

    malaria = malaria[
        malaria_cols
    ]

    if malaria.duplicated(
        ["census_district_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate malaria district-year rows."
        )

    final = final.merge(
        malaria,
        on=[
            "census_district_code",
            "year"
        ],
        how="left",
        validate="one_to_one"
    )

    # =========================================================
    # 5. DENGUE — STATE LEVEL
    # =========================================================

    dengue = load_check(
        DENGUE_FILE,
        "DENGUE"
    )

    dengue["year"] = pd.to_numeric(
        dengue["year"],
        errors="coerce"
    )

    dengue = dengue[
        (dengue["year"] >= 2011) &
        (dengue["year"] <= 2024)
    ].copy()

    dengue["state_key"] = dengue[
        "state_name"
    ].map(normalize_state)

    dengue_cols = [
        "state_key",
        "year",
        "dengue_cases",
        "dengue_deaths",
    ]

    dengue = dengue[
        dengue_cols
    ]

    if dengue.duplicated(
        ["state_key", "year"]
    ).any():
        raise ValueError(
            "Duplicate dengue state-year rows."
        )

    final["state_key"] = final[
        "state_name"
    ].map(normalize_state)

    final = final.merge(
        dengue,
        on=[
            "state_key",
            "year"
        ],
        how="left",
        validate="many_to_one"
    )

    # =========================================================
    # 6. FLOOD
    # =========================================================

    flood = load_check(
        FLOOD_FILE,
        "FLOOD"
    )

    flood["census_district_code"] = pd.to_numeric(
        flood["census_district_code"],
        errors="coerce"
    )

    flood["year"] = pd.to_numeric(
        flood["year"],
        errors="coerce"
    )

    flood = flood[
        (flood["year"] >= 2011) &
        (flood["year"] <= 2023)
    ].copy()

    flood_cols = [
        "census_district_code",
        "year",
        "flood_event_count",
    ]

    flood = flood[
        flood_cols
    ]

    if flood.duplicated(
        ["census_district_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate flood district-year rows."
        )

    final = final.merge(
        flood,
        on=[
            "census_district_code",
            "year"
        ],
        how="left",
        validate="one_to_one"
    )

    # =========================================================
    # 7. HEATWAVE
    # =========================================================

    heatwave = load_check(
        HEATWAVE_FILE,
        "HEATWAVE"
    )

    heatwave["census_district_code"] = pd.to_numeric(
        heatwave["census_district_code"],
        errors="coerce"
    )

    heatwave["year"] = pd.to_numeric(
        heatwave["year"],
        errors="coerce"
    )

    heatwave = heatwave[
        (heatwave["year"] >= 2011) &
        (heatwave["year"] <= 2024)
    ].copy()

    heatwave_cols = [
        "census_district_code",
        "year",
        "heatwave_days",
        "severe_heatwave_days",
        "max_temperature_c",
    ]

    heatwave = heatwave[
        heatwave_cols
    ].rename(
        columns={
            "max_temperature_c":
                "heatwave_max_temperature_c"
        }
    )

    if heatwave.duplicated(
        ["census_district_code", "year"]
    ).any():
        raise ValueError(
            "Duplicate heatwave district-year rows."
        )

    final = final.merge(
        heatwave,
        on=[
            "census_district_code",
            "year"
        ],
        how="left",
        validate="one_to_one"
    )

    # =========================================================
    # 8. REMOVE TEMPORARY KEY
    # =========================================================

    final = final.drop(
        columns=["state_key"]
    )

    # =========================================================
    # 9. SORT
    # =========================================================

    final = final.sort_values(
        [
            "census_district_code",
            "year"
        ]
    ).reset_index(
        drop=True
    )

    # =========================================================
    # 10. VALIDATION
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    expected_rows = 640 * 14

    if len(final) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"got {len(final):,}"
        )

    if final["census_district_code"].nunique() != 640:
        raise ValueError(
            "Final dataset does not contain 640 districts."
        )

    if final.duplicated(
        [
            "census_district_code",
            "year"
        ]
    ).any():
        raise ValueError(
            "Duplicate final district-year keys."
        )

    if final["population_2011"].isna().any():
        raise ValueError(
            "Missing population values."
        )

    if (
        final["total_rainfall_mm"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative rainfall detected."
        )

    if (
        final["malaria_cases"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative malaria cases detected."
        )

    if (
        final["dengue_cases"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative dengue cases detected."
        )

    if (
        final["flood_event_count"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative flood counts detected."
        )

    if (
        final["heatwave_days"].dropna() < 0
    ).any():
        raise ValueError(
            "Negative heatwave days detected."
        )

    # =========================================================
    # 11. COVERAGE REPORT
    # =========================================================

    coverage = {}

    feature_columns = [
        "avg_temperature_c",
        "total_rainfall_mm",
        "malaria_cases",
        "dengue_cases",
        "flood_event_count",
        "heatwave_days",
    ]

    for col in feature_columns:
        coverage[col] = int(
            final[col].notna().sum()
        )

    # =========================================================
    # 12. SAVE
    # =========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUT_FILE,
        index=False
    )

    report_lines = [
        "TRAVELSHIELD MASTER FEATURE DATASET",
        "=" * 50,
        "",
        f"Rows: {len(final):,}",
        f"Districts: {final['census_district_code'].nunique():,}",
        f"Years: {final['year'].min()} - {final['year'].max()}",
        "",
        "FEATURE COVERAGE",
        "-" * 50,
    ]

    for col, count in coverage.items():
        report_lines.append(
            f"{col}: {count:,}/{len(final):,}"
        )

    report_lines.extend(
        [
            "",
            "IMPORTANT DATA RULES",
            "-" * 50,
            "Missing disease/disaster observations were NOT "
            "converted to zero.",
            "Dengue is a state-level feature replicated to "
            "districts through state_name.",
            "Flood contains only deterministic LGD-to-Census "
            "mappings.",
            "Heatwave is a NASA POWER T2M_MAX threshold-based "
            "indicator using IMD temperature thresholds.",
            "",
            "STATUS: PASSED",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # =========================================================
    # 13. OUTPUT
    # =========================================================

    print()
    print("OUTPUT")
    print("-" * 70)
    print(
        f"Final rows       : {len(final):,}"
    )
    print(
        f"Unique districts : "
        f"{final['census_district_code'].nunique():,}"
    )
    print(
        f"Year range       : "
        f"{final['year'].min()} - "
        f"{final['year'].max()}"
    )

    print()
    print("Feature coverage:")
    for col, count in coverage.items():
        print(
            f"{col:<30} "
            f"{count:,}/{len(final):,}"
        )

    print()
    print(f"OUTPUT : {OUT_FILE}")
    print(f"REPORT : {REPORT_FILE}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()