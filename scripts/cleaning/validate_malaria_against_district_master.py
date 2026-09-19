import pandas as pd
import re
from pathlib import Path
from rapidfuzz import process, fuzz


# ============================================================
# CONFIG
# ============================================================

MALARIA_CSV = Path(
    "data/processed/disease/malaria/"
    "malaria_district_year_2000_2024.csv"
)

MASTER_CSV = Path(
    "data/reference/district_master.csv"
)

OUTPUT_DIR = Path(
    "data/reports/disease"
)

MAPPING_REPORT = OUTPUT_DIR / (
    "malaria_district_mapping_report_v2.csv"
)

SUMMARY_REPORT = OUTPUT_DIR / (
    "malaria_district_mapping_summary_v2.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(value):

    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    # Remove special marker.
    value = value.replace("#", "")

    # Standardize ampersand.
    value = value.replace("&", " AND ")

    # Remove punctuation.
    value = value.replace(".", " ")
    value = value.replace(",", " ")
    value = value.replace("-", " ")

    # Remove brackets.
    value = re.sub(
        r"[\(\)\[\]]",
        " ",
        value
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# DOCUMENTED NAME VARIANTS
# ============================================================

NAME_VARIANTS = {

    "AHMADNAGAR":
        "AHMEDNAGAR",

    "BANGALORE":
        "BENGALURU",

    "BANGALORE RURAL":
        "BENGALURU RURAL",

    "BANGALORE URBAN":
        "BENGALURU URBAN",

    "CALCUTTA":
        "KOLKATA",

    "GURGAON":
        "GURUGRAM",

    "ALLAHABAD":
        "PRAYAGRAJ",

    "FAIZABAD":
        "AYODHYA",

    "TRIVANDRUM":
        "THIRUVANANTHAPURAM",

    "COOCH BEHAR":
        "COOCHBEHAR",

    "COOCHBEHAR":
        "COOCH BEHAR",

    "BARABANKI":
        "BARA BANKI",

    "BARA BANKI":
        "BARABANKI",

    "NORTH 24 PARGANAS":
        "NORTH TWENTY FOUR PARGANAS",

    "SOUTH 24 PARGANAS":
        "SOUTH TWENTY FOUR PARGANAS",

    "24 PARAGANAS NORTH":
        "NORTH TWENTY FOUR PARGANAS",

    "24 PARAGANAS SOUTH":
        "SOUTH TWENTY FOUR PARGANAS",
}


def canonical_name(value):

    value = normalize_name(value)

    return NAME_VARIANTS.get(
        value,
        value
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not MALARIA_CSV.exists():

        raise FileNotFoundError(
            f"Malaria CSV not found:\n{MALARIA_CSV}"
        )

    if not MASTER_CSV.exists():

        raise FileNotFoundError(
            f"District master not found:\n{MASTER_CSV}"
        )

    malaria = pd.read_csv(
        MALARIA_CSV
    )

    master = pd.read_csv(
        MASTER_CSV
    )

    return malaria, master


# ============================================================
# PREPARE MASTER
# ============================================================

def prepare_master(master):

    required = [
        "census_state_code",
        "census_district_code",
        "district_name",
        "state_name",
    ]

    missing = [
        c
        for c in required
        if c not in master.columns
    ]

    if missing:

        raise ValueError(
            "Missing district-master columns: "
            + ", ".join(missing)
        )

    master = master.copy()

    master["canonical_district"] = (
        master["district_name"]
        .apply(canonical_name)
    )

    # Validate canonical names.
    duplicate_names = (
        master[
            "canonical_district"
        ]
        .value_counts()
    )

    duplicate_names = duplicate_names[
        duplicate_names > 1
    ]

    print(
        f"Master districts       : {len(master):,}"
    )

    print(
        f"Unique district names  : "
        f"{master['canonical_district'].nunique():,}"
    )

    print(
        f"Ambiguous names        : "
        f"{len(duplicate_names):,}"
    )

    return master


# ============================================================
# EXACT GLOBAL MATCH
# ============================================================

def exact_global_match(
    malaria_name,
    master
):

    candidates = master[
        master["canonical_district"]
        == malaria_name
    ]

    if len(candidates) == 1:

        return candidates.iloc[0], (
            "EXACT_GLOBAL",
            100.0
        )

    if len(candidates) > 1:

        return None, (
            "AMBIGUOUS_DISTRICT_NAME",
            100.0
        )

    return None, (
        "NO_EXACT_MATCH",
        None
    )


# ============================================================
# FUZZY GLOBAL MATCH
# ============================================================

def fuzzy_global_match(
    malaria_name,
    master
):

    choices = (
        master[
            "canonical_district"
        ]
        .drop_duplicates()
        .tolist()
    )

    if not malaria_name:
        return None, (
            "EMPTY_DISTRICT",
            None
        )

    matches = process.extract(
        malaria_name,
        choices,
        scorer=fuzz.ratio,
        limit=5
    )

    if not matches:
        return None, (
            "NO_FUZZY_MATCH",
            None
        )

    best_name, best_score, _ = matches[0]

    # Conservative threshold.
    if best_score < 90:

        return None, (
            "LOW_CONFIDENCE",
            round(best_score, 2)
        )

    candidates = master[
        master["canonical_district"]
        == best_name
    ]

    if len(candidates) != 1:

        return None, (
            "AMBIGUOUS_FUZZY_MATCH",
            round(best_score, 2)
        )

    return candidates.iloc[0], (
        "FUZZY_GLOBAL",
        round(best_score, 2)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MALARIA GLOBAL DISTRICT MAPPING")
    print("=" * 70)

    malaria, master = load_data()

    print(
        f"\nMalaria records : "
        f"{len(malaria):,}"
    )

    master = prepare_master(
        master
    )

    # --------------------------------------------------------
    # Required malaria columns
    # --------------------------------------------------------

    required_malaria = [
        "year",
        "state",
        "district",
        "total_cases",
        "total_deaths",
        "source_page"
    ]

    missing = [
        c
        for c in required_malaria
        if c not in malaria.columns
    ]

    if missing:

        raise ValueError(
            "Missing malaria columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Map UNIQUE district names only.
    # --------------------------------------------------------

    unique_districts = (
        malaria[
            [
                "state",
                "district"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"\nUnique malaria districts: "
        f"{len(unique_districts):,}"
    )

    results = []

    exact_count = 0
    fuzzy_count = 0
    special_count = 0
    ambiguous_count = 0
    unmatched_count = 0

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for _, row in unique_districts.iterrows():

        source_state = str(
            row["state"]
        ).strip()

        source_district = str(
            row["district"]
        ).strip()

        # ----------------------------------------------------
        # SPECIAL ENTRIES
        # ----------------------------------------------------

        if source_district.endswith("#"):

            results.append({

                "source_state":
                    source_state,

                "source_district":
                    source_district,

                "normalized_district":
                    canonical_name(
                        source_district
                    ),

                "match_type":
                    "SPECIAL_ENTRY",

                "similarity":
                    None,

                "canonical_state":
                    None,

                "canonical_district":
                    None,

                "census_state_code":
                    None,

                "census_district_code":
                    None,

            })

            special_count += 1

            continue

        normalized = canonical_name(
            source_district
        )

        # ----------------------------------------------------
        # EXACT GLOBAL MATCH
        # ----------------------------------------------------

        match, info = exact_global_match(
            normalized,
            master
        )

        match_type, similarity = info

        if match is not None:

            results.append({

                "source_state":
                    source_state,

                "source_district":
                    source_district,

                "normalized_district":
                    normalized,

                "match_type":
                    match_type,

                "similarity":
                    similarity,

                "canonical_state":
                    match["state_name"],

                "canonical_district":
                    match["district_name"],

                "census_state_code":
                    match["census_state_code"],

                "census_district_code":
                    match[
                        "census_district_code"
                    ],

            })

            exact_count += 1

            continue

        if match_type == (
            "AMBIGUOUS_DISTRICT_NAME"
        ):

            results.append({

                "source_state":
                    source_state,

                "source_district":
                    source_district,

                "normalized_district":
                    normalized,

                "match_type":
                    match_type,

                "similarity":
                    similarity,

                "canonical_state":
                    None,

                "canonical_district":
                    None,

                "census_state_code":
                    None,

                "census_district_code":
                    None,

            })

            ambiguous_count += 1

            continue

        # ----------------------------------------------------
        # FUZZY GLOBAL MATCH
        # ----------------------------------------------------

        match, info = fuzzy_global_match(
            normalized,
            master
        )

        match_type, similarity = info

        if match is not None:

            results.append({

                "source_state":
                    source_state,

                "source_district":
                    source_district,

                "normalized_district":
                    normalized,

                "match_type":
                    match_type,

                "similarity":
                    similarity,

                "canonical_state":
                    match["state_name"],

                "canonical_district":
                    match["district_name"],

                "census_state_code":
                    match["census_state_code"],

                "census_district_code":
                    match[
                        "census_district_code"
                    ],

            })

            fuzzy_count += 1

            continue

        # ----------------------------------------------------
        # UNMATCHED
        # ----------------------------------------------------

        results.append({

            "source_state":
                source_state,

            "source_district":
                source_district,

            "normalized_district":
                normalized,

            "match_type":
                match_type,

            "similarity":
                similarity,

            "canonical_state":
                None,

            "canonical_district":
                None,

            "census_state_code":
                None,

            "census_district_code":
                None,

        })

        unmatched_count += 1

    # ========================================================
    # MAPPING DATAFRAME
    # ========================================================

    mapping = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mapping.to_csv(
        MAPPING_REPORT,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # STATE CONFLICT ANALYSIS
    # ========================================================

    matched = mapping[
        mapping["canonical_state"]
        .notna()
    ].copy()

    matched["source_state_norm"] = (
        matched["source_state"]
        .apply(normalize_name)
    )

    matched["canonical_state_norm"] = (
        matched["canonical_state"]
        .apply(normalize_name)
    )

    matched[
        "source_state_conflict"
    ] = (
        matched["source_state_norm"]
        !=
        matched["canonical_state_norm"]
    )

    conflicts = matched[
        matched["source_state_conflict"]
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([

        {
            "metric":
                "malaria_records",

            "value":
                len(malaria)
        },

        {
            "metric":
                "unique_malaria_districts",

            "value":
                len(unique_districts)
        },

        {
            "metric":
                "exact_global_matches",

            "value":
                exact_count
        },

        {
            "metric":
                "fuzzy_global_matches",

            "value":
                fuzzy_count
        },

        {
            "metric":
                "special_entries",

            "value":
                special_count
        },

        {
            "metric":
                "ambiguous_names",

            "value":
                ambiguous_count
        },

        {
            "metric":
                "unmatched",

            "value":
                unmatched_count
        },

        {
            "metric":
                "source_state_conflicts",

            "value":
                len(conflicts)
        },
    ])

    summary.to_csv(
        SUMMARY_REPORT,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Exact global matches : "
        f"{exact_count:,}"
    )

    print(
        f"Fuzzy global matches : "
        f"{fuzzy_count:,}"
    )

    print(
        f"Special entries      : "
        f"{special_count:,}"
    )

    print(
        f"Ambiguous names      : "
        f"{ambiguous_count:,}"
    )

    print(
        f"Unmatched            : "
        f"{unmatched_count:,}"
    )

    print(
        f"\nSource-state conflicts "
        f"(diagnostic only): "
        f"{len(conflicts):,}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The source PDF state column is NOT used "
        "to determine canonical state."
    )

    print(
        "Canonical state comes from the validated "
        "district_master."
    )

    print(
        "\nOutput:"
    )

    print(MAPPING_REPORT)

    print(
        "\nSummary:"
    )

    print(SUMMARY_REPORT)


if __name__ == "__main__":
    main()