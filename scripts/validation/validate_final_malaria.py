from pathlib import Path
import pandas as pd

DATA = Path(
    "data/processed/disease/malaria/"
    "malaria_district_year_2000_2024.csv"
)

MASTER = Path(
    "data/reference/district_master.csv"
)


print("=" * 70)
print("FINAL MALARIA DATASET VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA)
master = pd.read_csv(MASTER)

errors = []

# ------------------------------------------------------------
# 1. Basic structure
# ------------------------------------------------------------

required_columns = [
    "year",
    "census_state_code",
    "census_district_code",
    "state_name",
    "district_name",
    "latitude",
    "longitude",
    "population_2011",
    "households_2011",
    "total_cases",
    "total_deaths",
    "data_coverage",
    "source_page",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    errors.append(
        f"Missing columns: {missing_columns}"
    )


# ------------------------------------------------------------
# 2. Year validation
# ------------------------------------------------------------

if not df["year"].between(2000, 2024).all():
    errors.append("Year outside 2000-2024 found")


# ------------------------------------------------------------
# 3. Duplicate validation
# ------------------------------------------------------------

duplicate_count = df.duplicated(
    subset=[
        "year",
        "census_district_code"
    ]
).sum()

if duplicate_count:
    errors.append(
        f"Duplicate year/district records: {duplicate_count}"
    )


# ------------------------------------------------------------
# 4. District master validation
# ------------------------------------------------------------

master_codes = set(
    master["census_district_code"]
    .dropna()
    .astype(int)
)

data_codes = set(
    df["census_district_code"]
    .dropna()
    .astype(int)
)

invalid_codes = data_codes - master_codes

if invalid_codes:
    errors.append(
        f"District codes absent from master: "
        f"{len(invalid_codes)}"
    )


# ------------------------------------------------------------
# 5. Missing canonical geography
# ------------------------------------------------------------

for column in [
    "census_district_code",
    "state_name",
    "district_name",
    "latitude",
    "longitude",
]:
    missing = df[column].isna().sum()

    if missing:
        errors.append(
            f"{column}: {missing} missing values"
        )


# ------------------------------------------------------------
# 6. Malaria values
# ------------------------------------------------------------

if (df["total_cases"] < 0).any():
    errors.append("Negative malaria cases found")

if (df["total_deaths"] < 0).any():
    errors.append("Negative malaria deaths found")

if (df["total_deaths"] > df["total_cases"]).any():
    errors.append(
        "Deaths greater than cases found"
    )


# ------------------------------------------------------------
# 7. Coverage validation
# ------------------------------------------------------------

bad_2024 = df.loc[
    (df["year"] == 2024)
    & (df["data_coverage"] != "THROUGH_APRIL_2024")
]

if len(bad_2024):
    errors.append(
        f"2024 rows without THROUGH_APRIL_2024 flag: "
        f"{len(bad_2024)}"
    )

bad_old = df.loc[
    (df["year"] < 2024)
    & (df["data_coverage"] != "FULL_YEAR")
]

if len(bad_old):
    errors.append(
        f"Pre-2024 rows incorrectly marked: "
        f"{len(bad_old)}"
    )


# ------------------------------------------------------------
# 8. Geographic sanity
# ------------------------------------------------------------

if not df["latitude"].between(-90, 90).all():
    errors.append("Invalid latitude")

if not df["longitude"].between(-180, 180).all():
    errors.append("Invalid longitude")


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print()
print(f"Rows                 : {len(df):,}")
print(
    f"Unique districts     : "
    f"{df['census_district_code'].nunique():,}"
)
print(
    f"Year range            : "
    f"{df['year'].min()} - {df['year'].max()}"
)
print(
    f"Duplicate records     : {duplicate_count}"
)
print(
    f"Invalid district IDs  : {len(invalid_codes)}"
)

print()
print("Rows by coverage:")
print(df["data_coverage"].value_counts())

print()
print("Rows by year:")
print(df["year"].value_counts().sort_index())

print()

if errors:
    print("=" * 70)
    print("VALIDATION FAILED")
    print("=" * 70)

    for error in errors:
        print("ERROR:", error)

    raise SystemExit(1)

else:
    print("=" * 70)
    print("STATUS: MALARIA DATASET VALIDATED")
    print("=" * 70)