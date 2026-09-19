from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_WEATHER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather"
)

PROCESSED_WEATHER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather"
)

DISTRICT_MASTER = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_master.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_DISTRICTS = 640

EXPECTED_ROWS_PER_DISTRICT = 5479

EXPECTED_TOTAL_ROWS = (
    EXPECTED_DISTRICTS
    * EXPECTED_ROWS_PER_DISTRICT
)

WEATHER_COLUMNS = [
    "census_district_code",
    "district_name",
    "state_name",
    "latitude",
    "longitude",
    "date",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
]

NUMERIC_COLUMNS = [
    "census_district_code",
    "latitude",
    "longitude",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
]


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

PROCESSED_WEATHER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("COMBINING NASA POWER WEATHER DATA")
print("=" * 70)


# ============================================================
# LOAD DISTRICT MASTER
# ============================================================

print("\nLoading district master...")

district_master = pd.read_csv(
    DISTRICT_MASTER
)

expected_codes = set(
    district_master[
        "census_district_code"
    ].astype(int)
)

print(
    f"Districts in master: "
    f"{len(expected_codes)}"
)


if len(expected_codes) != EXPECTED_DISTRICTS:

    raise ValueError(
        "District master does not contain "
        "exactly 640 districts."
    )


# ============================================================
# FIND RAW WEATHER FILES
# ============================================================

weather_files = sorted(
    RAW_WEATHER_DIR.glob("weather_*.csv")
)

print(
    f"Raw weather files found: "
    f"{len(weather_files)}"
)


if len(weather_files) != EXPECTED_DISTRICTS:

    raise ValueError(
        f"Expected {EXPECTED_DISTRICTS} "
        f"weather files, found {len(weather_files)}"
    )


# ============================================================
# READ AND COMBINE
# ============================================================

dataframes = []

print("\nReading weather files...")

for index, file_path in enumerate(
    weather_files,
    start=1,
):

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    if df.columns.tolist() != WEATHER_COLUMNS:

        raise ValueError(
            f"Unexpected columns in "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(df) != EXPECTED_ROWS_PER_DISTRICT:

        raise ValueError(
            f"{file_path.name}: expected "
            f"{EXPECTED_ROWS_PER_DISTRICT} rows, "
            f"found {len(df)}"
        )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():

        raise ValueError(
            f"Invalid date found in "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    if df[WEATHER_COLUMNS].isna().any().any():

        raise ValueError(
            f"Missing values found in "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # DISTRICT ID CHECK
    # --------------------------------------------------------

    district_codes = (
        df["census_district_code"]
        .astype(int)
        .unique()
    )

    if len(district_codes) != 1:

        raise ValueError(
            f"Multiple district codes found in "
            f"{file_path.name}"
        )

    district_code = district_codes[0]

    if district_code not in expected_codes:

        raise ValueError(
            f"Unknown district code "
            f"{district_code} in "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # DATE DUPLICATE CHECK
    # --------------------------------------------------------

    if df.duplicated(
        subset=[
            "census_district_code",
            "date",
        ]
    ).any():

        raise ValueError(
            f"Duplicate district-date rows "
            f"in {file_path.name}"
        )

    dataframes.append(df)

    print(
        f"[{index:03d}/{EXPECTED_DISTRICTS}] "
        f"{file_path.name} loaded"
    )


# ============================================================
# CONCATENATE
# ============================================================

print("\nCombining all districts...")

weather = pd.concat(
    dataframes,
    ignore_index=True,
)


# ============================================================
# SORT
# ============================================================

weather = weather.sort_values(
    by=[
        "census_district_code",
        "date",
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL ROW COUNT
# ============================================================

print(
    f"\nCombined rows: "
    f"{len(weather):,}"
)

print(
    f"Expected rows: "
    f"{EXPECTED_TOTAL_ROWS:,}"
)

if len(weather) != EXPECTED_TOTAL_ROWS:

    raise ValueError(
        "Final row count does not match "
        "expected value."
    )


# ============================================================
# FINAL DISTRICT COUNT
# ============================================================

final_district_count = (
    weather[
        "census_district_code"
    ].nunique()
)

print(
    f"Final district count: "
    f"{final_district_count}"
)

if final_district_count != EXPECTED_DISTRICTS:

    raise ValueError(
        "Final dataset does not contain "
        "640 unique districts."
    )


# ============================================================
# FINAL DUPLICATE CHECK
# ============================================================

duplicate_count = weather.duplicated(
    subset=[
        "census_district_code",
        "date",
    ]
).sum()

print(
    f"Duplicate district-date rows: "
    f"{duplicate_count}"
)

if duplicate_count != 0:

    raise ValueError(
        "Duplicate district-date records found."
    )


# ============================================================
# DATE RANGE CHECK
# ============================================================

print(
    f"Date range: "
    f"{weather['date'].min()} "
    f"to "
    f"{weather['date'].max()}"
)

if weather["date"].min() != pd.Timestamp(
    "2011-01-01"
):

    raise ValueError(
        "Incorrect minimum date."
    )

if weather["date"].max() != pd.Timestamp(
    "2025-12-31"
):

    raise ValueError(
        "Incorrect maximum date."
    )


# ============================================================
# WEATHER RANGE CHECKS
# ============================================================

print("\nChecking weather ranges...")

if not weather["RH2M"].between(
    0,
    100,
).all():

    raise ValueError(
        "RH2M contains values outside "
        "0–100%."
    )


if (
    weather["PRECTOTCORR"] < 0
).any():

    raise ValueError(
        "Negative precipitation detected."
    )


if (
    weather["T2M_MAX"]
    < weather["T2M_MIN"]
).any():

    raise ValueError(
        "T2M_MAX is lower than T2M_MIN."
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_output = (
    PROCESSED_WEATHER_DIR
    / "weather_daily_2011_2025.csv"
)

print(
    "\nSaving CSV..."
)

weather.to_csv(
    csv_output,
    index=False,
)


# ============================================================
# SAVE PARQUET
# ============================================================

parquet_output = (
    PROCESSED_WEATHER_DIR
    / "weather_daily_2011_2025.parquet"
)

print(
    "Saving Parquet..."
)

weather.to_parquet(
    parquet_output,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEATHER COMBINATION COMPLETE")
print("=" * 70)

print(
    f"Districts       : {final_district_count}"
)

print(
    f"Total rows      : {len(weather):,}"
)

print(
    f"Start date      : {weather['date'].min()}"
)

print(
    f"End date        : {weather['date'].max()}"
)

print(
    f"Duplicate rows  : {duplicate_count}"
)

print(
    "\nCSV:"
)

print(csv_output)

print(
    "\nParquet:"
)

print(parquet_output)

print(
    "\nALL WEATHER DATA PASSED FINAL CHECKS."
)

print("=" * 70)