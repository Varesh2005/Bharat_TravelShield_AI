from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISTRICT_MASTER = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_master.csv"
)

WEATHER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "weather"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_DISTRICTS = 640

EXPECTED_START = pd.Timestamp("2011-01-01")
EXPECTED_END = pd.Timestamp("2025-12-31")

EXPECTED_DAYS = len(
    pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="D",
    )
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


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NASA POWER WEATHER VALIDATION")
print("=" * 70)


# ============================================================
# LOAD DISTRICT MASTER
# ============================================================

district_master = pd.read_csv(DISTRICT_MASTER)

expected_codes = set(
    district_master["census_district_code"]
    .astype(int)
)

print(
    f"\nDistrict master records: "
    f"{len(district_master)}"
)


if len(expected_codes) != EXPECTED_DISTRICTS:
    raise ValueError(
        "District master does not contain 640 unique districts."
    )


# ============================================================
# FIND WEATHER FILES
# ============================================================

weather_files = sorted(
    WEATHER_DIR.glob("weather_*.csv")
)

print(
    f"Weather files found: "
    f"{len(weather_files)}"
)


# ============================================================
# FILE COUNT CHECK
# ============================================================

if len(weather_files) != EXPECTED_DISTRICTS:

    raise ValueError(
        f"Expected {EXPECTED_DISTRICTS} weather files, "
        f"found {len(weather_files)}"
    )

print("File count check: PASSED")


# ============================================================
# VALIDATION STORAGE
# ============================================================

results = []

invalid_files = []

total_rows = 0


# ============================================================
# VALIDATE EACH DISTRICT FILE
# ============================================================

for index, file_path in enumerate(weather_files, start=1):

    try:

        df = pd.read_csv(file_path)

        # ----------------------------------------------------
        # COLUMN CHECK
        # ----------------------------------------------------

        if df.columns.tolist() != WEATHER_COLUMNS:

            raise ValueError(
                "Unexpected column structure."
            )

        # ----------------------------------------------------
        # DISTRICT CODE
        # ----------------------------------------------------

        district_codes = (
            df["census_district_code"]
            .dropna()
            .astype(int)
            .unique()
        )

        if len(district_codes) != 1:

            raise ValueError(
                "File contains multiple district codes "
                "or missing district code."
            )

        district_code = district_codes[0]

        if district_code not in expected_codes:

            raise ValueError(
                f"District code {district_code} "
                f"not found in district master."
            )

        # ----------------------------------------------------
        # ROW COUNT
        # ----------------------------------------------------

        if len(df) != EXPECTED_DAYS:

            raise ValueError(
                f"Expected {EXPECTED_DAYS} rows, "
                f"found {len(df)}."
            )

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        if df["date"].isna().any():

            raise ValueError(
                "Invalid date values found."
            )

        if df["date"].min() != EXPECTED_START:

            raise ValueError(
                f"Incorrect start date: "
                f"{df['date'].min()}"
            )

        if df["date"].max() != EXPECTED_END:

            raise ValueError(
                f"Incorrect end date: "
                f"{df['date'].max()}"
            )

        # ----------------------------------------------------
        # DUPLICATE DATE CHECK
        # ----------------------------------------------------

        if df["date"].duplicated().any():

            raise ValueError(
                "Duplicate dates found."
            )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        missing = df[
            WEATHER_COLUMNS
        ].isna().sum()

        if missing.any():

            raise ValueError(
                f"Missing values found: "
                f"{missing[missing > 0].to_dict()}"
            )

        # ----------------------------------------------------
        # WEATHER RANGE CHECKS
        # ----------------------------------------------------

        if not df["RH2M"].between(0, 100).all():

            raise ValueError(
                "RH2M outside 0–100% range."
            )

        if (df["PRECTOTCORR"] < 0).any():

            raise ValueError(
                "Negative precipitation detected."
            )

        if (df["T2M_MAX"] < df["T2M_MIN"]).any():

            raise ValueError(
                "T2M_MAX is lower than T2M_MIN."
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        results.append(
            {
                "census_district_code": district_code,
                "file": file_path.name,
                "rows": len(df),
                "start_date": df["date"].min(),
                "end_date": df["date"].max(),
                "missing_values": int(
                    df[WEATHER_COLUMNS].isna().sum().sum()
                ),
                "status": "PASS",
            }
        )

        total_rows += len(df)

        print(
            f"[{index:03d}/{EXPECTED_DISTRICTS}] "
            f"District {district_code}: PASS"
        )

    except Exception as error:

        invalid_files.append(
            {
                "file": file_path.name,
                "error": str(error),
            }
        )

        print(
            f"[{index:03d}/{EXPECTED_DISTRICTS}] "
            f"{file_path.name}: FAIL"
        )

        print(
            f"    Reason: {error}"
        )


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

validation_report = pd.DataFrame(results)

validation_file = (
    REPORT_DIR
    / "nasa_power_weather_validation.csv"
)

validation_report.to_csv(
    validation_file,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

passed = len(results)
failed = len(invalid_files)

print("\n" + "=" * 70)
print("WEATHER VALIDATION COMPLETE")
print("=" * 70)

print(
    f"Expected districts : {EXPECTED_DISTRICTS}"
)

print(
    f"Passed districts   : {passed}"
)

print(
    f"Failed districts   : {failed}"
)

print(
    f"Expected rows/file : {EXPECTED_DAYS}"
)

print(
    f"Total valid rows   : {total_rows:,}"
)

print(
    f"\nValidation report:"
)

print(validation_file)


# ============================================================
# FAILED FILE REPORT
# ============================================================

if invalid_files:

    failed_file = (
        REPORT_DIR
        / "nasa_power_weather_failed_files.csv"
    )

    pd.DataFrame(invalid_files).to_csv(
        failed_file,
        index=False,
    )

    print(
        f"\nFailed file report:"
    )

    print(failed_file)

    raise RuntimeError(
        f"Weather validation failed for "
        f"{failed} district files."
    )


print("\nALL 640 WEATHER FILES PASSED VALIDATION.")
print("=" * 70)