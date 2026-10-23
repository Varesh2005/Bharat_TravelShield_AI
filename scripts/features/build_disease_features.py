from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MALARIA_FILE = (
    ROOT / "data/processed/disease/malaria/"
    "malaria_district_year_2000_2024.csv"
)

DENGUE_FILE = (
    ROOT / "data/processed/disease/dengue/"
    "dengue_state_year_2021_2026.csv"
)

MASTER_FILE = (
    ROOT / "data/reference/district_master.csv"
)

OUT_DIR = ROOT / "data/processed/disease"

MALARIA_OUT = (
    OUT_DIR / "malaria_district_year_features_2000_2024.csv"
)

DENGUE_OUT = (
    OUT_DIR / "dengue_state_year_features_2021_2026.csv"
)


def main():

    print("=" * 70)
    print("BUILD DISEASE FEATURES")
    print("=" * 70)

    # =========================================================
    # MALARIA
    # =========================================================

    malaria = pd.read_csv(
        MALARIA_FILE,
        low_memory=False
    )

    print()
    print("MALARIA")
    print("-" * 70)

    print(
        f"Rows loaded : {len(malaria):,}"
    )

    required_malaria = [
        "census_district_code",
        "year",
        "total_cases"
    ]

    missing = [
        c for c in required_malaria
        if c not in malaria.columns
    ]

    if missing:
        raise ValueError(
            f"Missing malaria columns: {missing}"
        )

    malaria["census_district_code"] = pd.to_numeric(
        malaria["census_district_code"],
        errors="coerce"
    )

    malaria["year"] = pd.to_numeric(
        malaria["year"],
        errors="coerce"
    )

    malaria["total_cases"] = pd.to_numeric(
        malaria["total_cases"],
        errors="coerce"
    )

    malaria = malaria.dropna(
        subset=[
            "census_district_code",
            "year",
            "total_cases"
        ]
    )

    # Never convert missing disease observations to zero.
    # Only create a presence flag for observed records.
    malaria["malaria_present"] = (
        malaria["total_cases"] > 0
    ).astype("int8")

    # Ensure one district-year row
    if malaria.duplicated(
        subset=[
            "census_district_code",
            "year"
        ]
    ).any():
        raise ValueError(
            "Duplicate malaria district-year rows detected."
        )

    malaria_features = malaria[
        [
            "census_district_code",
            "year",
            "total_cases",
            "malaria_present"
        ]
    ].copy()

    malaria_features = malaria_features.rename(
        columns={
            "total_cases": "malaria_cases"
        }
    )

    malaria_features = malaria_features.sort_values(
        [
            "census_district_code",
            "year"
        ]
    )

    # =========================================================
    # DENGUE
    # =========================================================

    dengue = pd.read_csv(
        DENGUE_FILE,
        low_memory=False
    )

    print()
    print("DENGUE")
    print("-" * 70)

    print(
        f"Rows loaded : {len(dengue):,}"
    )

    required_dengue = [
        "state_name",
        "year",
        "dengue_cases",
        "dengue_deaths"
    ]

    missing = [
        c for c in required_dengue
        if c not in dengue.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dengue columns: {missing}"
        )

    dengue["year"] = pd.to_numeric(
        dengue["year"],
        errors="coerce"
    )

    dengue["dengue_cases"] = pd.to_numeric(
        dengue["dengue_cases"],
        errors="coerce"
    )

    dengue["dengue_deaths"] = pd.to_numeric(
        dengue["dengue_deaths"],
        errors="coerce"
    )

    dengue = dengue.dropna(
        subset=[
            "state_name",
            "year",
            "dengue_cases",
            "dengue_deaths"
        ]
    )

    # State-year must be unique.
    if dengue.duplicated(
        subset=[
            "state_name",
            "year"
        ]
    ).any():
        raise ValueError(
            "Duplicate dengue state-year rows detected."
        )

    dengue_features = dengue[
        [
            "state_name",
            "year",
            "dengue_cases",
            "dengue_deaths"
        ]
    ].copy()

    dengue_features = dengue_features.sort_values(
        [
            "state_name",
            "year"
        ]
    )

    # =========================================================
    # SAVE
    # =========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    malaria_features.to_csv(
        MALARIA_OUT,
        index=False
    )

    dengue_features.to_csv(
        DENGUE_OUT,
        index=False
    )

    # =========================================================
    # FINAL VALIDATION
    # =========================================================

    if (
        malaria_features["malaria_cases"] < 0
    ).any():
        raise ValueError(
            "Negative malaria cases detected."
        )

    if (
        dengue_features["dengue_cases"] < 0
    ).any():
        raise ValueError(
            "Negative dengue cases detected."
        )

    if (
        dengue_features["dengue_deaths"] < 0
    ).any():
        raise ValueError(
            "Negative dengue deaths detected."
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print()
    print("MALARIA")
    print(
        f"Rows             : "
        f"{len(malaria_features):,}"
    )
    print(
        f"Unique districts : "
        f"{malaria_features['census_district_code'].nunique():,}"
    )
    print(
        f"Year range       : "
        f"{malaria_features['year'].min()} - "
        f"{malaria_features['year'].max()}"
    )

    print()
    print("DENGUE")
    print(
        f"Rows             : "
        f"{len(dengue_features):,}"
    )
    print(
        f"Unique states    : "
        f"{dengue_features['state_name'].nunique():,}"
    )
    print(
        f"Year range       : "
        f"{dengue_features['year'].min()} - "
        f"{dengue_features['year'].max()}"
    )

    print()
    print(f"MALARIA OUTPUT : {MALARIA_OUT}")
    print(f"DENGUE OUTPUT  : {DENGUE_OUT}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()