from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[2]

FLOOD_FILE = ROOT / "data/raw/disaster/flood/India_Flood_Inventory_v3.csv"
MASTER_FILE = ROOT / "data/reference/district_master.csv"

MAPPING_FILE = (
    ROOT / "data/reports/disaster/flood_lgd_census_mapping_check.csv"
)

OUT_DIR = ROOT / "data/processed/disaster/flood"
OUT_FILE = OUT_DIR / "flood_district_year_1967_2023.csv"
REPORT_FILE = (
    ROOT / "data/reports/disaster/flood/flood_final_build_report.txt"
)


def normalize(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    replacements = {
        "prayagraj": "allahabad",
        "ahmednagar": "ahmednagar",
        "chhatrapati sambhajinagar": "aurangabad",
        "gurugram": "gurgaon",
        "sri muktsar sahib": "muktsar",
        "nuh": "mewat",
        "s a s nagar": "sahibzada ajit singh nagar",
        "pondicherry": "puducherry",
        "viluppuram": "viluppuram",
        "dharashiv": "osmanabad",
        "east nimar": "khandwa",
        "narmadapuram": "hoshangabad",
        "palghar": "palghar",
    }

    x = replacements.get(x, x)

    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()


def parse_codes(x):
    if pd.isna(x):
        return []

    result = []

    for value in str(x).split(","):
        value = value.strip()

        if not value or value.lower() == "none":
            continue

        try:
            result.append(int(float(value)))
        except ValueError:
            pass

    return result


def main():

    print("=" * 70)
    print("BUILD FINAL FLOOD DISTRICT-YEAR DATASET")
    print("=" * 70)

    flood = pd.read_csv(
        FLOOD_FILE,
        low_memory=False
    )

    master = pd.read_csv(
        MASTER_FILE,
        low_memory=False
    )

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False
    )

    # --------------------------------------------------------
    # Only deterministic mappings
    # --------------------------------------------------------

    mapping = mapping[
        mapping["status"] == "EXACT"
    ].copy()

    mapping["lgd_district_code"] = pd.to_numeric(
        mapping["lgd_district_code"],
        errors="coerce"
    )

    mapping["census_district_code"] = pd.to_numeric(
        mapping["census_district_code"],
        errors="coerce"
    )

    mapping = mapping.dropna(
        subset=[
            "lgd_district_code",
            "census_district_code"
        ]
    )

    mapping["lgd_district_code"] = (
        mapping["lgd_district_code"].astype(int)
    )

    mapping["census_district_code"] = (
        mapping["census_district_code"].astype(int)
    )

    # One LGD code must map to exactly one Census code.
    mapping = mapping.drop_duplicates(
        subset=["lgd_district_code"]
    )

    lgd_to_census = dict(
        zip(
            mapping["lgd_district_code"],
            mapping["census_district_code"]
        )
    )

    print(
        f"Deterministic LGD mappings : "
        f"{len(lgd_to_census):,}"
    )

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    flood["start_date"] = pd.to_datetime(
        flood["Start Date"],
        errors="coerce",
        dayfirst=True
    )

    flood["year"] = flood["start_date"].dt.year

    # --------------------------------------------------------
    # Explode LGD codes
    # --------------------------------------------------------

    rows = []

    for _, row in flood.iterrows():

        year = row["year"]

        if pd.isna(year):
            continue

        year = int(year)

        if year < 1967 or year > 2023:
            continue

        for lgd in parse_codes(
            row["District_LGD_Codes"]
        ):

            census = lgd_to_census.get(lgd)

            if census is None:
                continue

            rows.append(
                {
                    "UEI": row["UEI"],
                    "year": year,
                    "lgd_district_code": lgd,
                    "census_district_code": census
                }
            )

    event_district = pd.DataFrame(rows)

    print(
        f"Mapped event-district rows : "
        f"{len(event_district):,}"
    )

    # --------------------------------------------------------
    # Deduplicate event + district
    # --------------------------------------------------------

    event_district = (
        event_district
        .drop_duplicates(
            subset=[
                "UEI",
                "year",
                "census_district_code"
            ]
        )
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    final = (
        event_district
        .groupby(
            [
                "census_district_code",
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

    # --------------------------------------------------------
    # Join canonical geography
    # --------------------------------------------------------

    master_small = master[
        [
            "census_district_code",
            "district_name",
            "state_name"
        ]
    ].copy()

    final = final.merge(
        master_small,
        on="census_district_code",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if final["district_name"].isna().any():
        raise ValueError(
            "Some mapped Census codes are absent from district_master."
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
        final["flood_event_count"] <= 0
    ).any():
        raise ValueError(
            "Invalid flood event counts detected."
        )

    final = final[
        [
            "census_district_code",
            "district_name",
            "state_name",
            "year",
            "flood_event_count"
        ]
    ].sort_values(
        [
            "census_district_code",
            "year"
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    report = f"""
FLOOD FINAL BUILD REPORT
========================

Raw flood events: {len(flood):,}

Deterministic LGD → Census mappings:
{len(lgd_to_census):,}

Final event-district rows:
{len(event_district):,}

Final district-year rows:
{len(final):,}

Unique Census districts:
{final['census_district_code'].nunique():,}

Year range:
{final['year'].min()} - {final['year'].max()}

District-year duplicate keys:
{final.duplicated(['census_district_code', 'year']).sum()}

STATUS:
PASSED

LIMITATION:
Only deterministic LGD → Census mappings were retained.
Unmapped and ambiguous LGD districts were excluded.
No district was assigned by fuzzy matching or inference.
"""

    REPORT_FILE.write_text(
        report.strip(),
        encoding="utf-8"
    )

    print()
    print("OUTPUT")
    print("-" * 70)
    print(f"District-year rows : {len(final):,}")
    print(
        f"Unique districts  : "
        f"{final['census_district_code'].nunique():,}"
    )
    print(
        f"Year range        : "
        f"{final['year'].min()} - "
        f"{final['year'].max()}"
    )

    print()
    print(f"OUTPUT : {OUT_FILE}")
    print(f"REPORT : {REPORT_FILE}")
    print()
    print("STATUS: PASSED")


if __name__ == "__main__":
    main()