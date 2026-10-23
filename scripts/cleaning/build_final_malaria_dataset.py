from pathlib import Path
import pandas as pd
import re
import unicodedata


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MALARIA_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "disease"
    / "malaria"
    / "malaria_district_year_2000_2024.csv"
)

DISTRICT_MASTER = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_master.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "disease"
    / "malaria"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "disease"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "malaria_district_year_2000_2024_canonical.csv"
)

MAPPING_REPORT = (
    REPORT_DIR
    / "malaria_mapping_final_report.csv"
)

EXCLUSION_REPORT = (
    REPORT_DIR
    / "malaria_mapping_exclusions.csv"
)


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(value):
    """
    Normalize a district name for deterministic matching.

    This function does NOT perform fuzzy matching.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    # Unicode normalization
    value = unicodedata.normalize("NFKD", value)

    # Remove accents / diacritics
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    # Standardize ampersand
    value = value.replace("&", " AND ")

    # Remove special-entry marker for normalization.
    # Special entries are detected separately.
    value = value.replace("#", "")

    # Replace punctuation with spaces
    value = re.sub(r"[^A-Z0-9]+", " ", value)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# VERIFIED NAME VARIANTS
# ============================================================
#
# ONLY genuine spelling / historical-name variants belong here.
#
# Generic fuzzy matching is intentionally NOT used.
# ============================================================

NAME_VARIANTS = {

    # --------------------------------------------------------
    # Maharashtra
    # --------------------------------------------------------

    "GONDIA": "GONDIYA",

    # --------------------------------------------------------
    # Odisha
    # --------------------------------------------------------

    "NABARANGPUR": "NABARANGAPUR",

    # --------------------------------------------------------
    # Rajasthan
    # --------------------------------------------------------

    "JALORE": "JALOR",
    "JHUNJHUNU": "JHUNJHUNUN",

    # --------------------------------------------------------
    # Uttar Pradesh
    # --------------------------------------------------------

    "BARABANKI": "BARA BANKI",
    "KUSHI NAGAR": "KUSHINAGAR",
    "MAHARAJGANJ": "MAHRAJGANJ",

    # --------------------------------------------------------
    # Uttarakhand
    # --------------------------------------------------------

    "HARIDWAR": "HARDWAR",
    "RUDRA PRAYAG": "RUDRAPRAYAG",
    "SIDDHARTH NAGAR": "SIDDHARTHNAGAR",
    "UDAM SINGH NAGAR": "UDHAM SINGH NAGAR",
    "UTTAR KASHI": "UTTARKASHI",

    # --------------------------------------------------------
    # Historical / renamed districts
    # --------------------------------------------------------

    "PRAYAGRAJ": "ALLAHABAD",

    # --------------------------------------------------------
    # Karnataka
    # --------------------------------------------------------

    "DAVANGERE": "DAVANAGERE",
    "CHAMARAJANAGARA": "CHAMARAJANAGAR",

    # --------------------------------------------------------
    # Tamil Nadu
    # --------------------------------------------------------

    "VILLUPURAM": "VILUPPURAM",

    # --------------------------------------------------------
    # Historical naming
    # --------------------------------------------------------

    "AYODHYA": "FAIZABAD",

    # --------------------------------------------------------
    # Haryana
    # --------------------------------------------------------

    "GURUGRAM": "GURGAON",

    # --------------------------------------------------------
    # West Bengal
    # --------------------------------------------------------

    "24 PARAGANAS NORTH": "NORTH TWENTY FOUR PARGANAS",
    "24 PARAGANAS SOUTH": "SOUTH TWENTY FOUR PARGANAS",
}


# ============================================================
# SPECIAL ENTRY DETECTION
# ============================================================

def is_special_entry(name):
    """
    Detect NCVBDC special/non-district entries.

    Examples include:
    - Nagpur MC#
    - Pune MC#
    - A.M.U. entries#
    - Health-unit entries#
    """

    if pd.isna(name):
        return False

    return "#" in str(name)


# ============================================================
# MASTER LOOKUP
# ============================================================

def build_master_lookup(master_df):
    """
    Create:

        normalized district name
                ↓
        list of matching master rows

    A list is used because some district names are ambiguous
    across states.
    """

    lookup = {}

    for _, row in master_df.iterrows():

        normalized = normalize_name(
            row["district_name"]
        )

        if not normalized:
            continue

        lookup.setdefault(
            normalized,
            []
        ).append(row)

    return lookup


# ============================================================
# APPLY VERIFIED VARIANT
# ============================================================

def resolve_variant(normalized_name):

    return NAME_VARIANTS.get(
        normalized_name,
        normalized_name
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL MALARIA CANONICAL DATASET BUILD")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MALARIA_INPUT.exists():
        raise FileNotFoundError(
            f"\nMalaria input file not found:\n"
            f"{MALARIA_INPUT}"
        )

    if not DISTRICT_MASTER.exists():
        raise FileNotFoundError(
            f"\nDistrict master file not found:\n"
            f"{DISTRICT_MASTER}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MALARIA
    # --------------------------------------------------------

    print("\nLoading malaria dataset...")

    malaria = pd.read_csv(
        MALARIA_INPUT
    )

    print(
        f"Malaria records loaded : "
        f"{len(malaria):,}"
    )

    # --------------------------------------------------------
    # LOAD DISTRICT MASTER
    # --------------------------------------------------------

    print("\nLoading district master...")

    master = pd.read_csv(
        DISTRICT_MASTER
    )

    print(
        f"Master districts       : "
        f"{len(master):,}"
    )

    # --------------------------------------------------------
    # VALIDATE MALARIA COLUMNS
    # --------------------------------------------------------

    required_malaria_columns = {
        "year",
        "state",
        "district",
        "total_cases",
        "total_deaths",
    }

    missing_malaria = (
        required_malaria_columns
        - set(malaria.columns)
    )

    if missing_malaria:

        raise ValueError(
            "Missing malaria columns: "
            f"{sorted(missing_malaria)}"
        )

    # --------------------------------------------------------
    # VALIDATE MASTER COLUMNS
    # --------------------------------------------------------

    required_master_columns = {
        "census_state_code",
        "census_district_code",
        "district_name",
        "state_name",
    }

    missing_master = (
        required_master_columns
        - set(master.columns)
    )

    if missing_master:

        raise ValueError(
            "Missing district master columns: "
            f"{sorted(missing_master)}"
        )

    # --------------------------------------------------------
    # MASTER SIZE CHECK
    # --------------------------------------------------------

    if len(master) != 640:

        raise ValueError(
            "District master must contain exactly "
            f"640 districts. Found {len(master)}."
        )

    # --------------------------------------------------------
    # YEAR VALIDATION
    # --------------------------------------------------------

    if malaria["year"].isna().any():

        raise ValueError(
            "Malaria dataset contains missing years."
        )

    malaria["year"] = pd.to_numeric(
        malaria["year"],
        errors="raise"
    ).astype(int)

    # --------------------------------------------------------
    # NUMERIC VALUE VALIDATION
    # --------------------------------------------------------

    malaria["total_cases"] = pd.to_numeric(
        malaria["total_cases"],
        errors="raise"
    )

    malaria["total_deaths"] = pd.to_numeric(
        malaria["total_deaths"],
        errors="raise"
    )

    if malaria["total_cases"].lt(0).any():

        raise ValueError(
            "Negative malaria case count detected."
        )

    if malaria["total_deaths"].lt(0).any():

        raise ValueError(
            "Negative malaria death count detected."
        )

    # --------------------------------------------------------
    # NORMALIZE MASTER NAMES
    # --------------------------------------------------------

    master["normalized_district"] = (
        master["district_name"]
        .apply(normalize_name)
    )

    master_lookup = build_master_lookup(
        master
    )

    # --------------------------------------------------------
    # NORMALIZE MALARIA NAMES
    # --------------------------------------------------------

    malaria["normalized_district"] = (
        malaria["district"]
        .apply(normalize_name)
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    canonical_rows = []

    mapping_rows = []

    exclusion_rows = []

    counters = {
        "EXACT": 0,
        "VARIANT": 0,
        "SPECIAL": 0,
        "AMBIGUOUS": 0,
        "UNMAPPED": 0,
    }

    # ========================================================
    # PROCESS EVERY MALARIA RECORD
    # ========================================================

    for idx, row in malaria.iterrows():

        source_state = row["state"]
        source_district = row["district"]

        normalized = row[
            "normalized_district"
        ]

        # ----------------------------------------------------
        # SPECIAL ENTRY
        # ----------------------------------------------------

        if is_special_entry(
            source_district
        ):

            counters["SPECIAL"] += 1

            mapping_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "match_type": "SPECIAL_ENTRY",
                "canonical_state": "",
                "canonical_district": "",
                "census_state_code": "",
                "census_district_code": "",
                "source_state_conflict": "",
            })

            exclusion_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "exclusion_reason": "SPECIAL_ENTRY",
                "candidate_districts": "",
            })

            continue

        # ----------------------------------------------------
        # EMPTY DISTRICT NAME
        # ----------------------------------------------------

        if not normalized:

            counters["UNMAPPED"] += 1

            mapping_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "match_type": "UNMAPPED",
                "canonical_state": "",
                "canonical_district": "",
                "census_state_code": "",
                "census_district_code": "",
                "source_state_conflict": "",
            })

            exclusion_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "exclusion_reason": "EMPTY_DISTRICT_NAME",
                "candidate_districts": "",
            })

            continue

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        candidates = master_lookup.get(
            normalized,
            []
        )

        match_type = "EXACT"

        # ----------------------------------------------------
        # VERIFIED NAME VARIANT
        # ----------------------------------------------------

        if not candidates:

            variant_name = resolve_variant(
                normalized
            )

            if variant_name != normalized:

                candidates = master_lookup.get(
                    variant_name,
                    []
                )

                if candidates:

                    match_type = "VARIANT"

        # ----------------------------------------------------
        # NO VERIFIED MATCH
        # ----------------------------------------------------

        if not candidates:

            counters["UNMAPPED"] += 1

            mapping_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "match_type": "UNMAPPED",
                "canonical_state": "",
                "canonical_district": "",
                "census_state_code": "",
                "census_district_code": "",
                "source_state_conflict": "",
            })

            exclusion_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "exclusion_reason": (
                    "NO_VERIFIED_2011_DISTRICT_MATCH"
                ),
                "candidate_districts": "",
            })

            continue

        # ----------------------------------------------------
        # AMBIGUOUS MATCH
        # ----------------------------------------------------

        if len(candidates) > 1:

            counters["AMBIGUOUS"] += 1

            candidate_names = "; ".join(
                sorted(
                    set(
                        str(candidate["state_name"])
                        + " -> "
                        + str(candidate["district_name"])
                        for candidate in candidates
                    )
                )
            )

            mapping_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "match_type": "AMBIGUOUS",
                "canonical_state": "",
                "canonical_district": "",
                "census_state_code": "",
                "census_district_code": "",
                "source_state_conflict": "",
            })

            exclusion_rows.append({
                "source_row": idx,
                "year": row["year"],
                "source_state": source_state,
                "source_district": source_district,
                "normalized_district": normalized,
                "exclusion_reason": (
                    "AMBIGUOUS_2011_DISTRICT_NAME"
                ),
                "candidate_districts": candidate_names,
            })

            continue

        # ----------------------------------------------------
        # VALID CANONICAL MATCH
        # ----------------------------------------------------

        canonical = candidates[0]

        counters[match_type] += 1

        canonical_state = canonical[
            "state_name"
        ]

        canonical_district = canonical[
            "district_name"
        ]

        census_state_code = canonical[
            "census_state_code"
        ]

        census_district_code = canonical[
            "census_district_code"
        ]

        # ----------------------------------------------------
        # SOURCE-STATE CONFLICT
        # ----------------------------------------------------

        source_state_normalized = (
            normalize_name(source_state)
        )

        canonical_state_normalized = (
            normalize_name(canonical_state)
        )

        state_conflict = (
            source_state_normalized
            != canonical_state_normalized
        )

        # ----------------------------------------------------
        # CANONICAL RECORD
        # ----------------------------------------------------

        canonical_rows.append({

            "year": int(row["year"]),

            "census_state_code": int(
                census_state_code
            ),

            "census_district_code": int(
                census_district_code
            ),

            "canonical_state": (
                canonical_state
            ),

            "canonical_district": (
                canonical_district
            ),

            "total_cases": int(
                row["total_cases"]
            ),

            "total_deaths": int(
                row["total_deaths"]
            ),

            # Keep original source values
            # for traceability.
            "source_state": source_state,

            "source_district": source_district,

            "mapping_method": match_type,

            "source_state_conflict": bool(
                state_conflict
            ),

            # NCVBDC source says 2024 data
            # is only available till April.
            "is_partial_year": (
                int(row["year"]) == 2024
            ),
        })

        # ----------------------------------------------------
        # MAPPING AUDIT RECORD
        # ----------------------------------------------------

        mapping_rows.append({

            "source_row": idx,

            "year": row["year"],

            "source_state": source_state,

            "source_district": source_district,

            "normalized_district": normalized,

            "match_type": match_type,

            "canonical_state": canonical_state,

            "canonical_district": canonical_district,

            "census_state_code": census_state_code,

            "census_district_code": (
                census_district_code
            ),

            "source_state_conflict": (
                state_conflict
            ),
        })

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    canonical_df = pd.DataFrame(
        canonical_rows
    )

    mapping_df = pd.DataFrame(
        mapping_rows
    )

    exclusion_df = pd.DataFrame(
        exclusion_rows
    )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    print(
        "\nChecking canonical duplicates..."
    )

    duplicate_columns = [
        "year",
        "census_state_code",
        "census_district_code",
    ]

    duplicate_mask = (
        canonical_df
        .duplicated(
            subset=duplicate_columns,
            keep=False
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print(
        f"Duplicate canonical rows : "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        duplicate_file = (
            REPORT_DIR
            / "malaria_duplicate_canonical_records.csv"
        )

        (
            canonical_df.loc[
                duplicate_mask
            ]
            .sort_values(
                duplicate_columns
            )
            .to_csv(
                duplicate_file,
                index=False
            )
        )

        print(
            "\nDuplicate report written:"
        )

        print(
            duplicate_file
        )

        raise ValueError(
            "\nDuplicate year + Census district "
            "records detected.\n"
            "The script stopped intentionally.\n"
            "Do NOT aggregate these records automatically."
        )

    # ========================================================
    # CANONICAL VALUE CHECKS
    # ========================================================

    print(
        "\nValidating canonical values..."
    )

    if not canonical_df.empty:

        if canonical_df[
            "total_cases"
        ].lt(0).any():

            raise ValueError(
                "Negative total_cases detected."
            )

        if canonical_df[
            "total_deaths"
        ].lt(0).any():

            raise ValueError(
                "Negative total_deaths detected."
            )

        if canonical_df[
            "census_district_code"
        ].isna().any():

            raise ValueError(
                "Missing Census district code detected."
            )

        # Every canonical ID must exist
        # in the 640-district master.
        valid_codes = set(
            master[
                "census_district_code"
            ].astype(int)
        )

        output_codes = set(
            canonical_df[
                "census_district_code"
            ].astype(int)
        )

        invalid_codes = (
            output_codes
            - valid_codes
        )

        if invalid_codes:

            raise ValueError(
                "Invalid Census district codes "
                f"found: {sorted(invalid_codes)}"
            )

    # ========================================================
    # SORT CANONICAL DATA
    # ========================================================

    if not canonical_df.empty:

        canonical_df = (
            canonical_df
            .sort_values(
                [
                    "census_state_code",
                    "census_district_code",
                    "year",
                ]
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    canonical_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    mapping_df.to_csv(
        MAPPING_REPORT,
        index=False
    )

    exclusion_df.to_csv(
        EXCLUSION_REPORT,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MALARIA MAPPING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Input records              : "
        f"{len(malaria):,}"
    )

    print(
        f"Canonical records           : "
        f"{len(canonical_df):,}"
    )

    print(
        f"Exact matches               : "
        f"{counters['EXACT']:,}"
    )

    print(
        f"Verified variant matches    : "
        f"{counters['VARIANT']:,}"
    )

    print(
        f"Special entries excluded    : "
        f"{counters['SPECIAL']:,}"
    )

    print(
        f"Ambiguous excluded          : "
        f"{counters['AMBIGUOUS']:,}"
    )

    print(
        f"Unmapped excluded           : "
        f"{counters['UNMAPPED']:,}"
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if not canonical_df.empty:

        unique_districts = (
            canonical_df[
                "census_district_code"
            ].nunique()
        )

        min_year = (
            canonical_df["year"].min()
        )

        max_year = (
            canonical_df["year"].max()
        )

        print(
            f"\nCanonical districts covered : "
            f"{unique_districts:,}"
        )

        print(
            f"Years covered               : "
            f"{min_year} - {max_year}"
        )

        conflict_count = int(
            canonical_df[
                "source_state_conflict"
            ].sum()
        )

        print(
            f"Source-state conflicts      : "
            f"{conflict_count:,}"
        )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 70
    )

    print(
        "\nCanonical dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMapping report:"
    )

    print(
        MAPPING_REPORT
    )

    print(
        "\nExclusion report:"
    )

    print(
        EXCLUSION_REPORT
    )

    # ========================================================
    # FINAL RULES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MAPPING RULES USED"
    )

    print(
        "=" * 70
    )

    print(
        "1. Exact district-name matching"
    )

    print(
        "2. Explicit verified name variants"
    )

    print(
        "3. No generic fuzzy matching"
    )

    print(
        "4. Canonical geography comes from district_master"
    )

    print(
        "5. Special # entries are excluded"
    )

    print(
        "6. Ambiguous districts are excluded"
    )

    print(
        "7. Unverified districts are excluded"
    )

    print(
        "8. 2024 records are marked as partial-year"
    )

    print(
        "\nFINAL MALARIA DATASET BUILD COMPLETE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()