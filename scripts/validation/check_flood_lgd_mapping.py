from pathlib import Path
import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[2]

FLOOD_FILE = (
    ROOT / "data" / "raw" / "disaster" / "flood"
    / "India_Flood_Inventory_v3.csv"
)

MASTER_FILE = (
    ROOT / "data" / "reference"
    / "district_master.csv"
)

REPORT_DIR = ROOT / "data" / "reports" / "disaster"
REPORT_FILE = REPORT_DIR / "flood_lgd_mapping_report.txt"


def normalize(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("&", "and")
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_codes(value):
    if pd.isna(value):
        return []

    result = []

    for item in str(value).split(","):
        item = item.strip()

        if not item or item.lower() == "none":
            result.append(None)
            continue

        try:
            result.append(int(float(item)))
        except ValueError:
            result.append(None)

    return result


def parse_names(value):
    if pd.isna(value):
        return []

    return [x.strip() for x in str(value).split(",")]


def main():

    print("=" * 70)
    print("FLOOD LGD → CENSUS MAPPING — GLOBAL DISTRICT CHECK")
    print("=" * 70)

    flood = pd.read_csv(
        FLOOD_FILE,
        low_memory=False
    )

    master = pd.read_csv(
        MASTER_FILE,
        low_memory=False
    )

    print(f"Flood rows  : {len(flood):,}")
    print(f"Master rows : {len(master):,}")

    # --------------------------------------------------------
    # Master district-name lookup
    # --------------------------------------------------------

    master["district_norm"] = master[
        "district_name"
    ].apply(normalize)

    district_lookup = {}

    for _, row in master.iterrows():

        name = row["district_norm"]

        if not name:
            continue

        district_lookup.setdefault(
            name,
            set()
        ).add(
            int(row["census_district_code"])
        )

    # --------------------------------------------------------
    # Observe LGD code → district names
    # --------------------------------------------------------

    lgd_to_names = {}

    for _, row in flood.iterrows():

        districts = parse_names(
            row["Districts"]
        )

        codes = parse_codes(
            row["District_LGD_Codes"]
        )

        if not districts or not codes:
            continue

        # Only use rows where the number of district
        # names and codes actually correspond.
        if len(districts) != len(codes):
            continue

        for district_name, lgd_code in zip(
            districts,
            codes
        ):

            if lgd_code is None:
                continue

            district_norm = normalize(
                district_name
            )

            if not district_norm:
                continue

            lgd_to_names.setdefault(
                int(lgd_code),
                set()
            ).add(
                district_norm
            )

    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    exact = []
    unmatched = []
    ambiguous = []

    for lgd_code, names in sorted(
        lgd_to_names.items()
    ):

        census_codes = set()

        matched_names = []

        for name in names:

            candidates = district_lookup.get(
                name,
                set()
            )

            if candidates:
                matched_names.append(name)
                census_codes.update(candidates)

        if len(census_codes) == 1:

            census_code = next(
                iter(census_codes)
            )

            exact.append(
                {
                    "lgd_district_code": lgd_code,
                    "census_district_code": census_code,
                    "status": "EXACT",
                    "observed_district_names":
                        " | ".join(sorted(names))
                }
            )

        elif len(census_codes) == 0:

            unmatched.append(
                {
                    "lgd_district_code": lgd_code,
                    "status": "UNMATCHED",
                    "observed_district_names":
                        " | ".join(sorted(names))
                }
            )

        else:

            ambiguous.append(
                {
                    "lgd_district_code": lgd_code,
                    "status": "AMBIGUOUS",
                    "candidate_census_codes":
                        " | ".join(
                            map(
                                str,
                                sorted(census_codes)
                            )
                        ),
                    "observed_district_names":
                        " | ".join(sorted(names))
                }
            )

    # --------------------------------------------------------
    # Save detailed mapping report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    exact_df = pd.DataFrame(exact)
    unmatched_df = pd.DataFrame(unmatched)
    ambiguous_df = pd.DataFrame(ambiguous)

    mapping_file = (
        REPORT_DIR
        / "flood_lgd_census_mapping_check.csv"
    )

    combined = pd.concat(
        [
            exact_df,
            unmatched_df,
            ambiguous_df
        ],
        ignore_index=True,
        sort=False
    )

    combined.to_csv(
        mapping_file,
        index=False
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("RESULT")
    print("-" * 70)

    print(
        f"LGD codes with usable name alignment : "
        f"{len(lgd_to_names):,}"
    )

    print(
        f"Exact mappings : "
        f"{len(exact_df):,}"
    )

    print(
        f"Unmatched      : "
        f"{len(unmatched_df):,}"
    )

    print(
        f"Ambiguous      : "
        f"{len(ambiguous_df):,}"
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        f"Mapping check : {mapping_file}"
    )

    print()
    print("STATUS")

    if len(ambiguous_df) == 0:
        print(
            "NO AMBIGUOUS MAPPINGS"
        )
    else:
        print(
            "AMBIGUOUS MAPPINGS REMAIN — "
            "DO NOT FINALIZE YET"
        )


if __name__ == "__main__":
    main()