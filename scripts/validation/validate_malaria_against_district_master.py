from pathlib import Path
import pandas as pd
import re
from rapidfuzz import fuzz, process

MALARIA = Path(
    "data/raw/disease/malaria/malaria_extracted_coordinate.csv"
)

MASTER = Path(
    "data/reference/district_master.csv"
)

OUT = Path(
    "data/reports/malaria_validation.csv"
)


def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).upper().strip()
    x = x.replace("&", "AND")
    x = re.sub(r"[^A-Z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


print("=" * 70)
print("MALARIA vs DISTRICT MASTER VALIDATION")
print("=" * 70)

mal = pd.read_csv(MALARIA)
master = pd.read_csv(MASTER)

print(f"Malaria rows       : {len(mal):,}")
print(f"Master rows        : {len(master):,}")

# ------------------------------------------------------------
# Normalize names
# ------------------------------------------------------------

mal["state_norm"] = mal["state"].map(norm)
mal["district_norm"] = mal["district"].map(norm)

master["state_norm"] = master["state_name"].map(norm)
master["district_norm"] = master["district_name"].map(norm)

# Exact state + district matching
master_key = (
    master["state_norm"] + "|" +
    master["district_norm"]
)

mal_key = (
    mal["state_norm"] + "|" +
    mal["district_norm"]
)

lookup = dict(
    zip(master_key, master["census_district_code"])
)

mal["census_district_code"] = mal_key.map(lookup)

mal["match_type"] = mal["census_district_code"].notna().map(
    {True: "EXACT", False: "UNMATCHED"}
)

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

exact = (mal["match_type"] == "EXACT").sum()
unmatched = (mal["match_type"] == "UNMATCHED").sum()

print()
print("EXACT MATCHES :", exact)
print("UNMATCHED     :", unmatched)

print()
print("Unique exact districts:",
      mal.loc[
          mal["match_type"] == "EXACT",
          "census_district_code"
      ].nunique()
)

# ------------------------------------------------------------
# Unmatched names
# ------------------------------------------------------------

unmatched_df = mal[
    mal["match_type"] == "UNMATCHED"
].copy()

# Unique unmatched state/district combinations
unmatched_summary = (
    unmatched_df[
        ["state", "district"]
    ]
    .drop_duplicates()
    .sort_values(["state", "district"])
)

print()
print(
    f"Unique unmatched state/district combinations: "
    f"{len(unmatched_summary):,}"
)

print()
print("First 100 unmatched:")
print(
    unmatched_summary.head(100).to_string(index=False)
)

# ------------------------------------------------------------
# Duplicate canonical keys
# ------------------------------------------------------------

matched = mal[
    mal["census_district_code"].notna()
].copy()

duplicates = matched[
    matched.duplicated(
        subset=["year", "census_district_code"],
        keep=False
    )
].sort_values(
    ["year", "census_district_code"]
)

print()
print(
    f"Matched rows with duplicate "
    f"(year, census_district_code): {len(duplicates):,}"
)

print()
print("First duplicate examples:")

if len(duplicates):
    print(
        duplicates[
            [
                "year",
                "state",
                "district",
                "total_cases",
                "total_deaths",
                "census_district_code"
            ]
        ].head(100).to_string(index=False)
    )

# ------------------------------------------------------------
# Save detailed validation report
# ------------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

mal.to_csv(
    OUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 70)
print("VALIDATION REPORT CREATED")
print("=" * 70)
print(f"Output: {OUT}")

print()
print("IMPORTANT:")
print("Do NOT aggregate duplicate district-year records yet.")
print("Do NOT build the final malaria dataset yet.")