from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "disaster"
    / "flood"
    / "India_Flood_Inventory_v3.csv"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_master.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "disaster"
    / "flood"
)

REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "disaster"

OUTPUT_FILE = OUTPUT_DIR / "flood_district_year_1967_2023.csv"
REPORT_FILE = REPORT_DIR / "flood_district_year_build_report.txt"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_lgd_codes(value):
    """
    Convert comma-separated District_LGD_Codes into integer codes.

    Invalid values such as None are ignored and reported separately.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    codes = []

    for part in text.split(","):
        part = part.strip()

        if not part or part.lower() == "none":
            continue

        try:
            code = int(float(part))
            codes.append(code)
        except ValueError:
            continue

    return codes


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD FLOOD DISTRICT-YEAR DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load raw flood data
    # --------------------------------------------------------

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw flood file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, low_memory=False)

    print(f"Raw flood events : {len(df):,}")
    print(f"Raw columns      : {len(df.columns)}")

    required_columns = [
        "UEI",
        "Start Date",
        "End Date",
        "Main Cause",
        "Districts",
        "State",
        "District_LGD_Codes",
        "State_Codes",
        "Event Source",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. Parse dates
    # --------------------------------------------------------

    df["start_date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce",
        dayfirst=True
    )

    df["end_date"] = pd.to_datetime(
        df["End Date"],
        errors="coerce",
        dayfirst=True
    )

    df["year"] = df["start_date"].dt.year

    invalid_year = df["year"].isna().sum()

    print(f"Missing/invalid start dates : {invalid_year:,}")

    # --------------------------------------------------------
    # 3. Parse district LGD codes
    # --------------------------------------------------------

    df["lgd_codes"] = df["District_LGD_Codes"].apply(
        parse_lgd_codes
    )

    missing_lgd_events = (
        df["lgd_codes"].apply(len).eq(0)
    ).sum()

    print(
        f"Events without usable LGD codes : "
        f"{missing_lgd_events:,}"
    )

    # --------------------------------------------------------
    # 4. Explode event → district
    # --------------------------------------------------------

    exploded = df[
        [
            "UEI",
            "start_date",
            "end_date",
            "year",
            "Main Cause",
            "Districts",
            "State",
            "State_Codes",
            "Event Source",
            "lgd_codes",
        ]
    ].copy()

    exploded = exploded.explode(
        "lgd_codes",
        ignore_index=True
    )

    exploded = exploded.rename(
        columns={
            "lgd_codes": "district_lgd_code"
        }
    )

    exploded = exploded[
        exploded["district_lgd_code"].notna()
    ].copy()

    exploded["district_lgd_code"] = (
        pd.to_numeric(
            exploded["district_lgd_code"],
            errors="coerce"
        )
    )

    exploded = exploded[
        exploded["district_lgd_code"].notna()
    ].copy()

    exploded["district_lgd_code"] = (
        exploded["district_lgd_code"]
        .astype(int)
    )

    print(
        f"Event-district rows after explosion : "
        f"{len(exploded):,}"
    )

    # --------------------------------------------------------
    # 5. Load district master
    # --------------------------------------------------------

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"District master not found: {MASTER_FILE}"
        )

    master = pd.read_csv(MASTER_FILE)

    required_master_columns = [
        "census_district_code",
        "district_name",
        "state_name",
    ]

    missing_master = [
        col for col in required_master_columns
        if col not in master.columns
    ]

    if missing_master:
        raise ValueError(
            f"Missing district master columns: {missing_master}"
        )

    print(f"District master rows : {len(master):,}")

    # --------------------------------------------------------
    # IMPORTANT:
    # Flood dataset uses LGD district codes.
    #
    # Our district_master uses Census 2011 district codes.
    #
    # Therefore DO NOT join these two codes directly.
    #
    # We first validate coverage of LGD codes separately.
    # --------------------------------------------------------

    unique_lgd_codes = set(
        exploded["district_lgd_code"].unique()
    )

    print(
        f"Unique flood LGD district codes : "
        f"{len(unique_lgd_codes):,}"
    )

    # --------------------------------------------------------
    # 6. Create district-year event counts
    # --------------------------------------------------------

    valid_year = exploded[
        exploded["year"].notna()
    ].copy()

    valid_year["year"] = (
        valid_year["year"]
        .astype(int)
    )

    valid_year = valid_year[
        (valid_year["year"] >= 1967)
        & (valid_year["year"] <= 2023)
    ].copy()

    # Remove duplicate event-district combinations.
    # One flood event mentioning the same district multiple times
    # should count as ONE event for that district.
    event_district = (
        valid_year[
            [
                "UEI",
                "year",
                "district_lgd_code"
            ]
        ]
        .drop_duplicates()
    )

    flood_district_year = (
        event_district
        .groupby(
            [
                "district_lgd_code",
                "year"
            ],
            as_index=False
        )
        .agg(
            flood_event_count=(
                "UEI",
                "nunique"
            )
        )
    )

    flood_district_year = flood_district_year.sort_values(
        [
            "district_lgd_code",
            "year"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 7. Basic validation
    # --------------------------------------------------------

    duplicate_keys = flood_district_year.duplicated(
        subset=[
            "district_lgd_code",
            "year"
        ]
    ).sum()

    if duplicate_keys != 0:
        raise ValueError(
            f"Duplicate district-year keys: {duplicate_keys}"
        )

    if (
        flood_district_year["flood_event_count"] <= 0
    ).any():
        raise ValueError(
            "Invalid non-positive flood event counts found."
        )

    # --------------------------------------------------------
    # 8. Save output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    flood_district_year.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 9. Build report
    # --------------------------------------------------------

    years = sorted(
        flood_district_year["year"].unique()
    )

    report_lines = [
        "FLOOD DISTRICT-YEAR BUILD REPORT",
        "=" * 60,
        "",
        f"Raw flood events: {len(df):,}",
        f"Event-district rows: {len(exploded):,}",
        f"District-year rows: {len(flood_district_year):,}",
        f"Unique LGD district codes: {len(unique_lgd_codes):,}",
        f"Year range: {min(years)} - {max(years)}",
        f"Events without LGD codes: {missing_lgd_events:,}",
        "",
        "OUTPUT",
        "-" * 60,
        str(OUTPUT_FILE),
        "",
        "IMPORTANT LIMITATION",
        "-" * 60,
        (
            "Flood inventory uses LGD district codes. "
            "These must not be directly equated with "
            "Census 2011 district codes."
        ),
        (
            "District-level canonical joining to the "
            "project district master requires a verified "
            "LGD-to-Census district crosswalk."
        ),
        (
            "Events without usable district LGD codes "
            "were excluded from district-level aggregation "
            "rather than being assigned by inference."
        ),
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 10. Final output
    # --------------------------------------------------------

    print()
    print("OUTPUT")
    print("-" * 70)
    print(f"District-year rows : {len(flood_district_year):,}")
    print(
        f"Unique LGD districts : "
        f"{flood_district_year['district_lgd_code'].nunique():,}"
    )
    print(
        f"Year range : "
        f"{flood_district_year['year'].min()} - "
        f"{flood_district_year['year'].max()}"
    )

    print()
    print("YEAR COVERAGE")
    print(
        flood_district_year
        .groupby("year")
        .size()
        .to_string()
    )

    print()
    print(f"OUTPUT : {OUTPUT_FILE}")
    print(f"REPORT : {REPORT_FILE}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()