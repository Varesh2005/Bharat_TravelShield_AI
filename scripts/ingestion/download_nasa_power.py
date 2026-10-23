import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "20110101"
END_DATE = "20251231"

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR"

COMMUNITY = "SB"

REQUEST_TIMEOUT = 60

SLEEP_SECONDS = 1

MAX_RETRIES = 3


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISTRICT_MASTER = (
    PROJECT_ROOT / "data" / "reference" / "district_master.csv"
)

RAW_WEATHER_DIR = (
    PROJECT_ROOT / "data" / "raw" / "weather"
)

REPORT_WEATHER_DIR = (
    PROJECT_ROOT / "data" / "reports" / "weather"
)


# ============================================================
# PREPARE DIRECTORIES
# ============================================================

RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
REPORT_WEATHER_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DISTRICT MASTER
# ============================================================

print("=" * 70)
print("NASA POWER WEATHER INGESTION")
print("=" * 70)

print("\nLoading district master...")

districts = pd.read_csv(DISTRICT_MASTER)

print(f"District rows loaded: {len(districts)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "census_district_code",
    "district_name",
    "state_name",
    "latitude",
    "longitude",
]

missing_columns = [
    column
    for column in required_columns
    if column not in districts.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


if len(districts) != 640:
    raise ValueError(
        f"Expected 640 districts, found {len(districts)}"
    )


if districts["census_district_code"].duplicated().any():
    raise ValueError(
        "Duplicate census_district_code values found."
    )


if districts["latitude"].isna().any():
    raise ValueError("Missing latitude values found.")


if districts["longitude"].isna().any():
    raise ValueError("Missing longitude values found.")


if not districts["latitude"].between(-90, 90).all():
    raise ValueError("Invalid latitude value found.")


if not districts["longitude"].between(-180, 180).all():
    raise ValueError("Invalid longitude value found.")


print("District master validation: PASSED")


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_district_weather(row):

    district_code = int(row["census_district_code"])
    district_name = row["district_name"]
    state_name = row["state_name"]

    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    output_file = (
        RAW_WEATHER_DIR
        / f"weather_{district_code:03d}.csv"
    )

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"[{district_code}/640] "
            f"{district_name} -> ALREADY EXISTS, SKIPPING"
        )

        return "skipped"


    print(
        f"\n[{district_code}/640] "
        f"{district_name}, {state_name}"
    )

    params = {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }


    # --------------------------------------------------------
    # RETRIES
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            print(
                f"  Attempt {attempt}: "
                f"HTTP {response.status_code}"
            )

            response.raise_for_status()

            data = response.json()

            parameters = data["properties"]["parameter"]

            weather = pd.DataFrame(parameters)

            weather.index.name = "date"

            weather = weather.reset_index()

            weather["date"] = pd.to_datetime(
                weather["date"].astype(str),
                format="%Y%m%d",
            )

            weather["census_district_code"] = district_code
            weather["district_name"] = district_name
            weather["state_name"] = state_name
            weather["latitude"] = latitude
            weather["longitude"] = longitude

            column_order = [
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

            weather = weather[column_order]


            # ------------------------------------------------
            # DISTRICT-LEVEL VALIDATION
            # ------------------------------------------------

            expected_days = len(
                pd.date_range(
                    start="2011-01-01",
                    end="2025-12-31",
                    freq="D",
                )
            )

            if len(weather) != expected_days:

                raise ValueError(
                    f"Expected {expected_days} rows, "
                    f"received {len(weather)}"
                )


            if weather["date"].duplicated().any():

                raise ValueError(
                    "Duplicate dates detected."
                )


            weather_variables = [
                "T2M",
                "T2M_MAX",
                "T2M_MIN",
                "RH2M",
                "PRECTOTCORR",
            ]

            missing_values = weather[
                weather_variables
            ].isna().sum()

            if missing_values.any():

                raise ValueError(
                    f"Missing weather values: "
                    f"{missing_values.to_dict()}"
                )


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            weather.to_csv(
                output_file,
                index=False,
            )

            print(
                f"  SUCCESS: {len(weather)} rows saved"
            )

            return "success"


        except Exception as error:

            print(
                f"  ERROR on attempt {attempt}: {error}"
            )

            if attempt < MAX_RETRIES:

                print("  Waiting before retry...")
                time.sleep(5)


    return "failed"


# ============================================================
# MAIN DOWNLOAD LOOP
# ============================================================

successful = []
skipped = []
failed = []


print("\n" + "=" * 70)
print("STARTING 640-DISTRICT DOWNLOAD")
print("=" * 70)

print(
    f"\nPeriod: {START_DATE} to {END_DATE}"
)

print(
    f"Variables: {PARAMETERS}"
)

print(
    f"Total districts: {len(districts)}"
)


for _, row in districts.iterrows():

    district_code = int(
        row["census_district_code"]
    )

    result = download_district_weather(row)

    if result == "success":

        successful.append(district_code)

    elif result == "skipped":

        skipped.append(district_code)

    else:

        failed.append(district_code)

    time.sleep(SLEEP_SECONDS)


# ============================================================
# SAVE DOWNLOAD REPORT
# ============================================================

report = pd.DataFrame(
    {
        "census_district_code": (
            successful
            + skipped
            + failed
        ),
        "status": (
            ["success"] * len(successful)
            + ["skipped"] * len(skipped)
            + ["failed"] * len(failed)
        ),
    }
)

report_file = (
    REPORT_WEATHER_DIR
    / "nasa_power_download_report.csv"
)

report.to_csv(
    report_file,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NASA POWER WEATHER INGESTION COMPLETE")
print("=" * 70)

print(f"Successful : {len(successful)}")
print(f"Skipped    : {len(skipped)}")
print(f"Failed     : {len(failed)}")

print(
    f"\nReport saved to:\n{report_file}"
)

if failed:

    print("\nFailed district codes:")

    for code in failed:

        print(code)

else:

    print(
        "\nALL 640 DISTRICTS DOWNLOADED SUCCESSFULLY."
    )