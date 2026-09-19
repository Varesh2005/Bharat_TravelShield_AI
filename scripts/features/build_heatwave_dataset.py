from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

WEATHER_FILE = (
    ROOT / "data/processed/weather/weather_daily_2011_2025.parquet"
)

OUT_DIR = ROOT / "data/processed/disaster/heatwave"
OUT_FILE = OUT_DIR / "heatwave_district_year_2011_2025.csv"

REPORT_DIR = ROOT / "data/reports/disaster/heatwave"
REPORT_FILE = REPORT_DIR / "heatwave_build_report.txt"


def main():

    print("=" * 70)
    print("BUILD HEATWAVE DISTRICT-YEAR DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load only required weather columns
    # ---------------------------------------------------------

    df = pd.read_parquet(
        WEATHER_FILE,
        columns=[
            "census_district_code",
            "date",
            "T2M_MAX"
        ]
    )

    print(f"Weather rows loaded : {len(df):,}")

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["T2M_MAX"] = pd.to_numeric(
        df["T2M_MAX"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "census_district_code",
            "date",
            "T2M_MAX"
        ]
    )

    # ---------------------------------------------------------
    # Actual-temperature heat threshold
    #
    # IMD:
    # Plains:
    # Heat Wave >= 45 C
    # Severe Heat Wave >= 47 C
    #
    # We are using this as a threshold-based
    # heat-extreme indicator, NOT claiming official
    # IMD-declared heatwave events.
    # ---------------------------------------------------------

    df["heatwave_day"] = (
        df["T2M_MAX"] >= 45.0
    ).astype("int8")

    df["severe_heatwave_day"] = (
        df["T2M_MAX"] >= 47.0
    ).astype("int8")

    # ---------------------------------------------------------
    # Year
    # ---------------------------------------------------------

    df["year"] = df["date"].dt.year

    # ---------------------------------------------------------
    # District-year aggregation
    # ---------------------------------------------------------

    final = (
        df.groupby(
            [
                "census_district_code",
                "year"
            ],
            as_index=False
        )
        .agg(
            heatwave_days=(
                "heatwave_day",
                "sum"
            ),
            severe_heatwave_days=(
                "severe_heatwave_day",
                "sum"
            ),
            max_temperature_c=(
                "T2M_MAX",
                "max"
            )
        )
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if final.empty:
        raise ValueError(
            "Final heatwave dataset is empty."
        )

    if final.duplicated(
        subset=[
            "census_district_code",
            "year"
        ]
    ).any():
        raise ValueError(
            "Duplicate district-year rows detected."
        )

    if (
        final["heatwave_days"] < 0
    ).any():
        raise ValueError(
            "Negative heatwave day count detected."
        )

    if (
        final["severe_heatwave_days"] < 0
    ).any():
        raise ValueError(
            "Negative severe heatwave day count detected."
        )

    # ---------------------------------------------------------
    # Add source/method metadata
    # ---------------------------------------------------------

    final["heatwave_definition"] = (
        "IMD_ACTUAL_TMAX_THRESHOLD"
    )

    final["source"] = (
        "NASA POWER daily T2M_MAX; "
        "thresholds based on IMD heat-wave criteria"
    )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    final = final.sort_values(
        [
            "census_district_code",
            "year"
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report = f"""
HEATWAVE DATASET BUILD REPORT
==============================

Input:
{WEATHER_FILE}

Weather rows loaded:
{len(df):,}

Final district-year rows:
{len(final):,}

Unique districts:
{final['census_district_code'].nunique():,}

Year range:
{final['year'].min()} - {final['year'].max()}

Total heatwave-threshold days:
{int(final['heatwave_days'].sum()):,}

Total severe-heatwave-threshold days:
{int(final['severe_heatwave_days'].sum()):,}

Maximum observed T2M_MAX:
{final['max_temperature_c'].max():.2f} C

Definition:
IMD actual-temperature threshold

Heatwave threshold:
T2M_MAX >= 45 C

Severe threshold:
T2M_MAX >= 47 C

IMPORTANT LIMITATION:
This dataset is a threshold-based heat-extreme indicator
derived from NASA POWER daily maximum temperature.
It is NOT an official IMD-declared heatwave-event dataset.

IMD official declaration also uses departure-from-normal
criteria and spatial/temporal declaration rules.

STATUS:
PASSED
"""

    REPORT_FILE.write_text(
        report.strip(),
        encoding="utf-8"
    )

    print()
    print("OUTPUT")
    print("-" * 70)
    print(
        f"District-year rows : {len(final):,}"
    )
    print(
        f"Unique districts   : "
        f"{final['census_district_code'].nunique():,}"
    )
    print(
        f"Year range         : "
        f"{final['year'].min()} - "
        f"{final['year'].max()}"
    )
    print(
        f"Heatwave days      : "
        f"{int(final['heatwave_days'].sum()):,}"
    )
    print(
        f"Severe days        : "
        f"{int(final['severe_heatwave_days'].sum()):,}"
    )

    print()
    print(f"OUTPUT : {OUT_FILE}")
    print(f"REPORT : {REPORT_FILE}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()