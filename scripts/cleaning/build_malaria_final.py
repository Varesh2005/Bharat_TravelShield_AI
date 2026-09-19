from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SOURCE = Path(
    "data/raw/disease/malaria/malaria_extracted_coordinate.csv"
)

MASTER = Path(
    "data/reference/district_master.csv"
)

OUTPUT = Path(
    "data/processed/disease/malaria/"
    "malaria_district_year_2000_2024.csv"
)

REPORT = Path(
    "data/reports/disease/"
    "malaria_final_build_report.txt"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BUILD FINAL MALARIA DATASET")
print("=" * 70)

mal = pd.read_csv(SOURCE)
master = pd.read_csv(MASTER)

print(f"Extracted rows : {len(mal):,}")
print(f"Master rows    : {len(master):,}")


# ============================================================
# BASIC CLEANING
# ============================================================

mal["year"] = pd.to_numeric(
    mal["year"],
    errors="coerce"
).astype("Int64")

mal["total_cases"] = pd.to_numeric(
    mal["total_cases"],
    errors="coerce"
)

mal["total_deaths"] = pd.to_numeric(
    mal["total_deaths"],
    errors="coerce"
)

mal["district_norm"] = (
    mal["district"]
    .astype(str)
    .str.upper()
    .str.strip()
)

master["district_norm"] = (
    master["district_name"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# NORMALIZE COMMON NAME VARIANTS
# ============================================================

def normalize_name(x):

    x = str(x).upper().strip()

    replacements = {
        "BARDHAMAN": "BARDDHAMAN",
        "BARDHAMAN DISTRICT": "BARDDHAMAN",
        "BANGALORE": "BANGALORE",
        "BELGAUM": "BELGAUM",
        "BELLARY": "BELLARY",
        "NICOBAR": "NICOBARS",
    }

    return replacements.get(x, x)


mal["district_norm"] = mal["district_norm"].map(
    normalize_name
)

master["district_norm"] = master["district_norm"].map(
    normalize_name
)


# ============================================================
# GLOBAL EXACT DISTRICT MATCH
#
# IMPORTANT:
# We intentionally DO NOT use extracted state because
# validation proved that state extraction is corrupted.
# ============================================================

lookup = (
    master[
        [
            "district_norm",
            "census_district_code"
        ]
    ]
    .drop_duplicates("district_norm")
)


mal = mal.merge(
    lookup,
    on="district_norm",
    how="left"
)


# ============================================================
# IDENTIFY MATCH STATUS
# ============================================================

mal["match_status"] = "UNMATCHED"

mal.loc[
    mal["census_district_code"].notna(),
    "match_status"
] = "EXACT_GLOBAL"


# ============================================================
# REMOVE AMBIGUOUS DISTRICT NAMES
#
# A district name appearing for multiple Census districts
# cannot be safely mapped by name alone.
# ============================================================

name_counts = (
    master.groupby("district_norm")[
        "census_district_code"
    ]
    .nunique()
)

ambiguous_names = set(
    name_counts[
        name_counts > 1
    ].index
)

mal.loc[
    mal["district_norm"].isin(ambiguous_names),
    "match_status"
] = "AMBIGUOUS"
    
mal.loc[
    mal["district_norm"].isin(ambiguous_names),
    "census_district_code"
] = pd.NA


# ============================================================
# KEEP ONLY VALID YEARS
# ============================================================

mal.loc[
    ~mal["year"].between(2000, 2024),
    "match_status"
] = "INVALID_YEAR"


# ============================================================
# REMOVE UNMATCHED / AMBIGUOUS
# ============================================================

matched = mal[
    mal["match_status"] == "EXACT_GLOBAL"
].copy()

print()
print(
    f"Exact deterministic matches : {len(matched):,}"
)


# ============================================================
# DUPLICATE YEAR + DISTRICT CHECK
# ============================================================

duplicate_mask = matched.duplicated(
    subset=[
        "year",
        "census_district_code"
    ],
    keep=False
)

duplicate_rows = matched[
    duplicate_mask
].copy()

print(
    f"Duplicate canonical rows     : "
    f"{len(duplicate_rows):,}"
)


# ============================================================
# CONSERVATIVE RULE:
# REMOVE DUPLICATE CANONICAL YEAR/DISTRICT ROWS
#
# We do NOT sum or average them because the source extraction
# does not establish which duplicate row is authoritative.
# ============================================================

matched = matched[
    ~duplicate_mask
].copy()


print(
    f"Rows after duplicate removal : "
    f"{len(matched):,}"
)


# ============================================================
# JOIN CANONICAL DISTRICT MASTER
# ============================================================

master_columns = [
    "census_district_code",
    "census_state_code",
    "district_name",
    "state_name",
    "population_2011",
    "households_2011",
    "latitude",
    "longitude",
]

canonical = master[
    master_columns
].drop_duplicates(
    "census_district_code"
)

final = matched.merge(
    canonical,
    on="census_district_code",
    how="left",
    validate="many_to_one"
)


# ============================================================
# 2024 COVERAGE FLAG
# ============================================================

final["data_coverage"] = "FULL_YEAR"

final.loc[
    final["year"] == 2024,
    "data_coverage"
] = "THROUGH_APRIL_2024"


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

final = final[
    [
        "year",
        "census_state_code",
        "census_district_code",
        "state_name",
        "district_name",
        "latitude",
        "longitude",
        "population_2011",
        "households_2011",
        "total_cases",
        "total_deaths",
        "data_coverage",
        "source_page",
    ]
].sort_values(
    [
        "year",
        "census_state_code",
        "census_district_code"
    ]
).reset_index(drop=True)


# ============================================================
# FINAL VALIDATION
# ============================================================

errors = []

if final["census_district_code"].isna().any():
    errors.append(
        "Missing census district codes"
    )

if final["state_name"].isna().any():
    errors.append(
        "Missing canonical states"
    )

if final["district_name"].isna().any():
    errors.append(
        "Missing canonical districts"
    )

if final["total_cases"].isna().any():
    errors.append(
        "Missing malaria case values"
    )

if (final["total_cases"] < 0).any():
    errors.append(
        "Negative malaria case values"
    )

if (final["total_deaths"] < 0).any():
    errors.append(
        "Negative malaria death values"
    )

if final.duplicated(
    subset=[
        "year",
        "census_district_code"
    ]
).any():

    errors.append(
        "Duplicate year/district records remain"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)

if errors:

    print()
    print("=" * 70)
    print("FINAL VALIDATION FAILED")
    print("=" * 70)

    for error in errors:
        print("ERROR:", error)

    raise RuntimeError(
        "Final malaria dataset failed validation."
    )


final.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("FINAL MALARIA DATASET REPORT")
report.append("=" * 70)
report.append("")
report.append(
    f"Source extracted rows        : {len(mal):,}"
)
report.append(
    f"Exact deterministic rows     : "
    f"{(mal['match_status'] == 'EXACT_GLOBAL').sum():,}"
)
report.append(
    f"Duplicate rows removed       : "
    f"{len(duplicate_rows):,}"
)
report.append(
    f"Final rows                   : {len(final):,}"
)
report.append(
    f"Unique districts             : "
    f"{final['census_district_code'].nunique():,}"
)
report.append(
    f"Years                        : "
    f"{final['year'].min()}-{final['year'].max()}"
)
report.append("")
report.append("Rows by year:")
report.append(
    final["year"]
    .value_counts()
    .sort_index()
    .to_string()
)
report.append("")
report.append("Rows by coverage:")
report.append(
    final["data_coverage"]
    .value_counts()
    .to_string()
)
report.append("")
report.append("Columns:")
report.extend(final.columns.tolist())

REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL MALARIA DATASET CREATED")
print("=" * 70)

print(f"Final rows       : {len(final):,}")
print(
    f"Unique districts : "
    f"{final['census_district_code'].nunique():,}"
)
print(
    f"Year range       : "
    f"{final['year'].min()} - {final['year'].max()}"
)

print()
print("Rows by year:")
print(
    final["year"]
    .value_counts()
    .sort_index()
)

print()
print(f"OUTPUT : {OUTPUT}")
print(f"REPORT : {REPORT}")

print()
print("=" * 70)
print("STATUS: PASSED")
print("=" * 70)