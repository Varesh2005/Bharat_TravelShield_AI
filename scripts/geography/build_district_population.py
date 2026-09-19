from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "census_population_2011.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reference"
OUTPUT_FILE = OUTPUT_DIR / "district_population_2011.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading Census 2011 dataset...")

df = pd.read_excel(
    RAW_FILE,
    sheet_name="Data"
)

print(f"Raw rows loaded: {len(df):,}")


# ---------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------

required_columns = [
    "State",
    "District",
    "Name",
    "TRU",
    "TOT_P",
    "No_HH",
    "Level"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# EXTRACT DISTRICT TOTALS
# ---------------------------------------------------------

districts = df[
    (df["Level"] == "DISTRICT") &
    (df["TRU"] == "Total")
].copy()

print(f"District total records: {len(districts):,}")


# ---------------------------------------------------------
# UNIQUENESS CHECK
# ---------------------------------------------------------

duplicate_mask = districts.duplicated(
    subset=["State", "District"],
    keep=False
)

if duplicate_mask.any():
    duplicates = districts.loc[
        duplicate_mask,
        ["State", "District", "Name", "TRU"]
    ]

    raise ValueError(
        "Duplicate district records detected:\n"
        f"{duplicates.to_string(index=False)}"
    )


# ---------------------------------------------------------
# SELECT REQUIRED COLUMNS
# ---------------------------------------------------------

districts = districts[
    [
        "State",
        "District",
        "Name",
        "TOT_P",
        "No_HH"
    ]
].copy()


# ---------------------------------------------------------
# RENAME COLUMNS
# ---------------------------------------------------------

districts.rename(
    columns={
        "State": "census_state_code",
        "District": "census_district_code",
        "Name": "district_name",
        "TOT_P": "population_2011",
        "No_HH": "households_2011"
    },
    inplace=True
)


# ---------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------

districts["census_state_code"] = (
    pd.to_numeric(
        districts["census_state_code"],
        errors="raise"
    )
    .astype("int64")
)

districts["census_district_code"] = (
    pd.to_numeric(
        districts["census_district_code"],
        errors="raise"
    )
    .astype("int64")
)

districts["population_2011"] = (
    pd.to_numeric(
        districts["population_2011"],
        errors="raise"
    )
    .astype("int64")
)

districts["households_2011"] = (
    pd.to_numeric(
        districts["households_2011"],
        errors="raise"
    )
    .astype("int64")
)


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

districts["district_name"] = (
    districts["district_name"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)


# ---------------------------------------------------------
# VALIDATE VALUES
# ---------------------------------------------------------

if (districts["population_2011"] < 0).any():
    raise ValueError(
        "Negative population values detected."
    )

if (districts["households_2011"] < 0).any():
    raise ValueError(
        "Negative household values detected."
    )

if districts["district_name"].eq("").any():
    raise ValueError(
        "Empty district names detected."
    )


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

districts.sort_values(
    by=[
        "census_state_code",
        "census_district_code"
    ],
    inplace=True
)

districts.reset_index(drop=True, inplace=True)


# ---------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------

expected_columns = [
    "census_state_code",
    "census_district_code",
    "district_name",
    "population_2011",
    "households_2011"
]

assert list(districts.columns) == expected_columns

assert len(districts) == 640

assert not districts.duplicated(
    subset=[
        "census_state_code",
        "census_district_code"
    ]
).any()


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

districts.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 60)
print("CENSUS DISTRICT POPULATION PROCESSING COMPLETE")
print("=" * 60)

print(f"Districts: {len(districts):,}")
print(
    f"Population total: "
    f"{districts['population_2011'].sum():,}"
)
print(
    f"Households total: "
    f"{districts['households_2011'].sum():,}"
)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("First 10 records:")
print(
    districts.head(10).to_string(index=False)
)