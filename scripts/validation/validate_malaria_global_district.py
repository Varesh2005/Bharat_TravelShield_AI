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
    "data/reports/malaria_global_district_validation.csv"
)


def norm(x):
    if pd.isna(x):
        return ""

    x = str(x).upper().strip()

    x = x.replace("&", "AND")

    x = re.sub(r"[^A-Z0-9]+", " ", x)

    return re.sub(r"\s+", " ", x).strip()


mal = pd.read_csv(MALARIA)
master = pd.read_csv(MASTER)

master["district_norm"] = master["district_name"].map(norm)
mal["district_norm"] = mal["district"].map(norm)

# ------------------------------------------------------------
# Exact global district-name matches
# ------------------------------------------------------------

exact_lookup = {}

for _, r in master.iterrows():

    name = r["district_norm"]

    if not name:
        continue

    # Keep only unique names.
    exact_lookup.setdefault(name, []).append(
        int(r["census_district_code"])
    )


results = []

for _, r in mal.iterrows():

    name = r["district_norm"]

    candidates = exact_lookup.get(name, [])

    if len(candidates) == 1:

        code = candidates[0]

        results.append({
            "year": r["year"],
            "source_state": r["state"],
            "source_district": r["district"],
            "total_cases": r["total_cases"],
            "total_deaths": r["total_deaths"],
            "census_district_code": code,
            "match_type": "EXACT_GLOBAL"
        })

    elif len(candidates) > 1:

        results.append({
            "year": r["year"],
            "source_state": r["state"],
            "source_district": r["district"],
            "total_cases": r["total_cases"],
            "total_deaths": r["total_deaths"],
            "census_district_code": None,
            "match_type": "AMBIGUOUS_GLOBAL"
        })

    else:

        results.append({
            "year": r["year"],
            "source_state": r["state"],
            "source_district": r["district"],
            "total_cases": r["total_cases"],
            "total_deaths": r["total_deaths"],
            "census_district_code": None,
            "match_type": "UNMATCHED"
        })


out = pd.DataFrame(results)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 70)
print("GLOBAL DISTRICT-NAME VALIDATION")
print("=" * 70)

print(f"Total rows       : {len(out):,}")

print()
print(
    out["match_type"]
    .value_counts()
    .to_string()
)

exact = out[
    out["match_type"] == "EXACT_GLOBAL"
]

print()
print(
    f"Exact global rows: {len(exact):,}"
)

print(
    f"Unique canonical districts: "
    f"{exact['census_district_code'].nunique():,}"
)

# ------------------------------------------------------------
# Check duplicate canonical year/district combinations
# ------------------------------------------------------------

duplicates = exact[
    exact.duplicated(
        subset=["year", "census_district_code"],
        keep=False
    )
].sort_values(
    ["year", "census_district_code"]
)

print()
print(
    f"Duplicate canonical "
    f"(year,district) rows: {len(duplicates):,}"
)

if len(duplicates):

    print()
    print("First 100 duplicates:")
    print(
        duplicates[
            [
                "year",
                "source_state",
                "source_district",
                "total_cases",
                "total_deaths",
                "census_district_code"
            ]
        ]
        .head(100)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# Fuzzy candidates ONLY for unmatched names.
# We do NOT automatically accept them.
# ------------------------------------------------------------

choices = master[
    ["district_norm", "district_name", "state_name",
     "census_district_code"]
].drop_duplicates()

choice_names = choices["district_norm"].tolist()

fuzzy_rows = []

for name in (
    out.loc[
        out["match_type"] == "UNMATCHED",
        "source_district"
    ]
    .dropna()
    .unique()
):

    n = norm(name)

    matches = process.extract(
        n,
        choice_names,
        scorer=fuzz.ratio,
        limit=3
    )

    for candidate, score, _ in matches:

        m = choices[
            choices["district_norm"] == candidate
        ].iloc[0]

        fuzzy_rows.append({
            "source_district": name,
            "candidate_district": m["district_name"],
            "candidate_state": m["state_name"],
            "census_district_code":
                m["census_district_code"],
            "score": score
        })


fuzzy = pd.DataFrame(fuzzy_rows)

print()
print(
    f"Unmatched unique districts: "
    f"{out.loc[out['match_type']=='UNMATCHED', 'source_district'].nunique():,}"
)

print()
print("Top fuzzy candidates for inspection:")

if len(fuzzy):
    print(
        fuzzy.sort_values(
            ["source_district", "score"],
            ascending=[True, False]
        )
        .head(100)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUT,
    index=False,
    encoding="utf-8-sig"
)

if len(fuzzy):
    fuzzy.to_csv(
        OUT.with_name(
            "malaria_fuzzy_candidates.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(f"Validation: {OUT}")

if len(fuzzy):
    print(
        "Fuzzy candidates: "
        f"{OUT.with_name('malaria_fuzzy_candidates.csv')}"
    )