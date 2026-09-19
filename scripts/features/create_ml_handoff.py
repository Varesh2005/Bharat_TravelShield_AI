from pathlib import Path
import pandas as pd


# =============================================================
# PATHS
# =============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data/processed/features/"
    "travelshield_district_year_features_2011_2024.csv"
)

OUTPUT_DIR = ROOT / "data/processed/features"

PARQUET_OUT = (
    OUTPUT_DIR
    / "travelshield_district_year_features_2011_2024.parquet"
)

DICTIONARY_OUT = (
    ROOT
    / "data/schemas/"
    "travelshield_feature_data_dictionary.csv"
)


# =============================================================
# MAIN
# =============================================================

def main():

    print("=" * 70)
    print("CREATE ML HANDOFF")
    print("=" * 70)

    # =========================================================
    # LOAD MASTER DATASET
    # =========================================================

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT}"
        )

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    print(
        f"Rows loaded : {len(df):,}"
    )

    # =========================================================
    # EXPECTED COLUMNS
    # =========================================================

    columns = [
        "census_district_code",
        "district_name",
        "state_name",
        "latitude",
        "longitude",
        "population_2011",
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
        col
        for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {c}"
                for c in missing_columns
            )
        )

    # Keep exact stable order
    df = df[columns].copy()

    # =========================================================
    # NUMERIC CONVERSION
    # =========================================================

    # ---------------------------------------------------------
    # Guaranteed complete integer columns
    # ---------------------------------------------------------

    required_integer_columns = [
        "census_district_code",
        "population_2011",
        "year",
        "rainy_days",
        "heatwave_days",
        "severe_heatwave_days",
    ]

    for col in required_integer_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

        if df[col].isna().any():
            raise ValueError(
                f"Unexpected missing value in "
                f"required integer column: {col}"
            )

        df[col] = df[col].astype("int64")

    # ---------------------------------------------------------
    # Nullable integer columns
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Missing disease/disaster observations are NOT zero.
    #
    # Pandas Int64 (capital I) supports NA.
    # ---------------------------------------------------------

    nullable_integer_columns = [
        "malaria_present",
        "flood_event_count",
    ]

    for col in nullable_integer_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("Int64")

    # ---------------------------------------------------------
    # Floating-point columns
    # ---------------------------------------------------------

    float_columns = [
        "latitude",
        "longitude",
        "avg_temperature_c",
        "max_temperature_c",
        "min_temperature_c",
        "avg_humidity_pct",
        "total_rainfall_mm",
        "malaria_cases",
        "dengue_cases",
        "dengue_deaths",
        "heatwave_max_temperature_c",
    ]

    for col in float_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("float64")

    # =========================================================
    # STRING COLUMNS
    # =========================================================

    df["district_name"] = (
        df["district_name"]
        .astype("string")
    )

    df["state_name"] = (
        df["state_name"]
        .astype("string")
    )

    # =========================================================
    # BASIC DATA VALIDATION
    # =========================================================

    expected_rows = 640 * 14

    if len(df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"found {len(df):,}"
        )

    if df["census_district_code"].nunique() != 640:

        raise ValueError(
            "Expected exactly 640 unique districts."
        )

    if df["year"].min() != 2011:

        raise ValueError(
            "Minimum year is not 2011."
        )

    if df["year"].max() != 2024:

        raise ValueError(
            "Maximum year is not 2024."
        )

    # =========================================================
    # DUPLICATE CHECK
    # =========================================================

    duplicate_count = df.duplicated(
        subset=[
            "census_district_code",
            "year"
        ]
    ).sum()

    if duplicate_count > 0:

        raise ValueError(
            f"Duplicate district-year keys: "
            f"{duplicate_count}"
        )

    # =========================================================
    # VALUE VALIDATION
    # =========================================================

    # Population
    if (
        df["population_2011"] <= 0
    ).any():

        raise ValueError(
            "Non-positive population detected."
        )

    # Latitude
    if (
        (df["latitude"] < -90)
        |
        (df["latitude"] > 90)
    ).any():

        raise ValueError(
            "Invalid latitude detected."
        )

    # Longitude
    if (
        (df["longitude"] < -180)
        |
        (df["longitude"] > 180)
    ).any():

        raise ValueError(
            "Invalid longitude detected."
        )

    # Humidity
    humidity = df[
        "avg_humidity_pct"
    ].dropna()

    if (
        (humidity < 0)
        |
        (humidity > 100)
    ).any():

        raise ValueError(
            "Humidity outside 0-100% detected."
        )

    # Rainfall
    rainfall = df[
        "total_rainfall_mm"
    ].dropna()

    if (
        rainfall < 0
    ).any():

        raise ValueError(
            "Negative rainfall detected."
        )

    # Rainy days
    rainy_days = df[
        "rainy_days"
    ].dropna()

    if (
        (rainy_days < 0)
        |
        (rainy_days > 366)
    ).any():

        raise ValueError(
            "Invalid rainy-day count detected."
        )

    # Disease
    for col in [
        "malaria_cases",
        "dengue_cases",
        "dengue_deaths",
    ]:

        values = df[col].dropna()

        if (
            values < 0
        ).any():

            raise ValueError(
                f"Negative values detected in {col}."
            )

    # Flood
    flood = df[
        "flood_event_count"
    ].dropna()

    if (
        flood < 0
    ).any():

        raise ValueError(
            "Negative flood event count detected."
        )

    # Heatwave
    for col in [
        "heatwave_days",
        "severe_heatwave_days",
    ]:

        values = df[col].dropna()

        if (
            (values < 0)
            |
            (values > 366)
        ).any():

            raise ValueError(
                f"Invalid value detected in {col}."
            )

    # Severe heatwave cannot exceed heatwave days
    invalid_severe = (
        df["severe_heatwave_days"]
        >
        df["heatwave_days"]
    )

    if invalid_severe.any():

        raise ValueError(
            "Severe heatwave days exceed "
            "total heatwave days."
        )

    # Temperature consistency
    temp_check = (
        df["max_temperature_c"]
        <
        df["min_temperature_c"]
    )

    if temp_check.any():

        raise ValueError(
            "Maximum temperature is below "
            "minimum temperature."
        )

    # =========================================================
    # SORT
    # =========================================================

    df = df.sort_values(
        [
            "census_district_code",
            "year"
        ]
    ).reset_index(
        drop=True
    )

    # =========================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DICTIONARY_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================
    # WRITE PARQUET
    # =========================================================

    df.to_parquet(
        PARQUET_OUT,
        engine="pyarrow",
        index=False
    )

    # =========================================================
    # DATA DICTIONARY
    # =========================================================

    dictionary_rows = [

        {
            "column": "census_district_code",
            "data_type": "int64",
            "description": "Canonical Census 2011 district identifier.",
            "source": "Census of India 2011 / district master",
            "unit": "code",
        },

        {
            "column": "district_name",
            "data_type": "string",
            "description": "Canonical Census 2011 district name.",
            "source": "District master",
            "unit": "",
        },

        {
            "column": "state_name",
            "data_type": "string",
            "description": "Canonical state name.",
            "source": "District master",
            "unit": "",
        },

        {
            "column": "latitude",
            "data_type": "float64",
            "description": "Representative point latitude of district geometry.",
            "source": "2011 district boundary",
            "unit": "degrees",
        },

        {
            "column": "longitude",
            "data_type": "float64",
            "description": "Representative point longitude of district geometry.",
            "source": "2011 district boundary",
            "unit": "degrees",
        },

        {
            "column": "population_2011",
            "data_type": "int64",
            "description": "District population from Census 2011.",
            "source": "Census of India 2011",
            "unit": "persons",
        },

        {
            "column": "year",
            "data_type": "int64",
            "description": "Calendar year.",
            "source": "Derived",
            "unit": "year",
        },

        {
            "column": "avg_temperature_c",
            "data_type": "float64",
            "description": "Annual mean daily 2-metre air temperature.",
            "source": "NASA POWER",
            "unit": "°C",
        },

        {
            "column": "max_temperature_c",
            "data_type": "float64",
            "description": "Maximum daily T2M_MAX observed during the year.",
            "source": "NASA POWER",
            "unit": "°C",
        },

        {
            "column": "min_temperature_c",
            "data_type": "float64",
            "description": "Minimum daily T2M_MIN observed during the year.",
            "source": "NASA POWER",
            "unit": "°C",
        },

        {
            "column": "avg_humidity_pct",
            "data_type": "float64",
            "description": "Annual mean relative humidity at 2 metres.",
            "source": "NASA POWER",
            "unit": "%",
        },

        {
            "column": "total_rainfall_mm",
            "data_type": "float64",
            "description": "Annual accumulated corrected precipitation.",
            "source": "NASA POWER",
            "unit": "mm",
        },

        {
            "column": "rainy_days",
            "data_type": "int64",
            "description": "Number of days with precipitation >= 1 mm.",
            "source": "NASA POWER; derived",
            "unit": "days",
        },

        {
            "column": "malaria_cases",
            "data_type": "float64_nullable",
            "description": "Reported district-level malaria cases. Missing means no usable retained source observation.",
            "source": "NCVBDC",
            "unit": "cases",
        },

        {
            "column": "malaria_present",
            "data_type": "Int64_nullable",
            "description": "1 when retained malaria cases are greater than zero; 0 when retained observation reports zero; missing when no malaria observation exists.",
            "source": "NCVBDC; derived",
            "unit": "binary",
        },

        {
            "column": "dengue_cases",
            "data_type": "float64_nullable",
            "description": "State-level dengue cases associated with the district's state. Missing where source coverage is unavailable.",
            "source": "NCVBDC",
            "unit": "cases",
        },

        {
            "column": "dengue_deaths",
            "data_type": "float64_nullable",
            "description": "State-level dengue deaths associated with the district's state.",
            "source": "NCVBDC",
            "unit": "deaths",
        },

        {
            "column": "flood_event_count",
            "data_type": "Int64_nullable",
            "description": "Deterministically mapped flood event count for district-year. Missing means no usable deterministic LGD-to-Census mapping was retained.",
            "source": "India Flood Inventory v3; derived",
            "unit": "events",
        },

        {
            "column": "heatwave_days",
            "data_type": "int64",
            "description": "Days where NASA POWER T2M_MAX met the threshold-based heatwave criterion.",
            "source": "NASA POWER; IMD threshold methodology; derived",
            "unit": "days",
        },

        {
            "column": "severe_heatwave_days",
            "data_type": "int64",
            "description": "Days where NASA POWER T2M_MAX met the severe heatwave threshold.",
            "source": "NASA POWER; IMD threshold methodology; derived",
            "unit": "days",
        },

        {
            "column": "heatwave_max_temperature_c",
            "data_type": "float64",
            "description": "Maximum T2M_MAX used in the heatwave feature calculation.",
            "source": "NASA POWER; derived",
            "unit": "°C",
        },
    ]

    dictionary = pd.DataFrame(
        dictionary_rows
    )

    dictionary.to_csv(
        DICTIONARY_OUT,
        index=False
    )

    # =========================================================
    # VERIFY PARQUET
    # =========================================================

    check = pd.read_parquet(
        PARQUET_OUT
    )

    if check.shape != df.shape:

        raise ValueError(
            "Parquet verification failed: "
            "shape mismatch."
        )

    if list(check.columns) != list(df.columns):

        raise ValueError(
            "Parquet verification failed: "
            "column mismatch."
        )

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"Districts  : "
        f"{df['census_district_code'].nunique():,}"
    )

    print(
        f"Years      : "
        f"{df['year'].min()} - "
        f"{df['year'].max()}"
    )

    print()

    print(
        f"PARQUET    : {PARQUET_OUT}"
    )

    print(
        f"DICTIONARY : {DICTIONARY_OUT}"
    )

    print()

    print("STATUS: PASSED")


if __name__ == "__main__":
    main()