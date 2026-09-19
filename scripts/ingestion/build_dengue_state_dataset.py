from pathlib import Path
import pandas as pd
import requests
from io import StringIO


# ============================================================
# CONFIG
# ============================================================

URL = (
    "https://ncvbdc.mohfw.gov.in/"
    "index4.php?Background=Dark&font=Normal&lang=1"
    "&level=0&lid=+3715&linkid=431&theme=Blue"
)

RAW = Path(
    "data/raw/disease/dengue/"
    "ncvbdc_dengue_state_2021_2026_raw.html"
)

OUTPUT = Path(
    "data/processed/disease/dengue/"
    "dengue_state_year_2021_2026.csv"
)

REPORT = Path(
    "data/reports/disease/"
    "dengue_state_build_report.txt"
)


# ============================================================
# STATE NORMALIZATION
# ============================================================

def normalize_state(name):

    name = str(name).strip()

    replacements = {
        "Chattisgarh": "Chhattisgarh",
        "J & K": "Jammu and Kashmir",
        "J&K": "Jammu and Kashmir",
        "Uttrakhand": "Uttarakhand",
        "Puduchery": "Puducherry",
        "A& N Island": "Andaman and Nicobar Islands",
        "A&N Island": "Andaman and Nicobar Islands",
        "D&N Haveli": "Dadra and Nagar Haveli",
        "Daman & Diu": "Daman and Diu",
    }

    return replacements.get(name, name)


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 70)
print("BUILD DENGUE STATE-YEAR DATASET")
print("=" * 70)

response = requests.get(
    URL,
    timeout=60,
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

response.raise_for_status()

html = response.text

RAW.parent.mkdir(
    parents=True,
    exist_ok=True
)

RAW.write_text(
    html,
    encoding="utf-8"
)

print("Official NCVBDC page downloaded.")
print(f"Raw HTML saved: {RAW}")


# ============================================================
# READ TABLE
# ============================================================

tables = pd.read_html(
    StringIO(html)
)

print(f"Tables detected: {len(tables)}")

if len(tables) != 1:
    raise RuntimeError(
        f"Expected 1 table, found {len(tables)}"
    )

table = tables[0]

print(
    f"Dengue table shape: {table.shape}"
)

print()
print(table.head(10).to_string())


# ============================================================
# CONFIRMED TABLE STRUCTURE
#
# Column:
# 0  = Sl. No.
# 1  = State
#
# 2021 -> columns 2,3
# 2022 -> columns 4,5
# 2023 -> columns 6,7
# 2024 -> columns 8,9
# 2025 -> columns 10,11
# 2026 -> columns 12,13
#
# ============================================================

if table.shape[1] < 14:
    raise RuntimeError(
        "Unexpected dengue table structure."
    )


# ============================================================
# BUILD LONG DATASET
# ============================================================

records = []

year_columns = {
    2021: (2, 3),
    2022: (4, 5),
    2023: (6, 7),
    2024: (8, 9),
    2025: (10, 11),
    2026: (12, 13),
}


for _, row in table.iterrows():

    state = normalize_state(row.iloc[1])

    # Skip header rows / totals / empty rows
    if (
        not state
        or state.lower() in {
            "nan",
            "affected states/uts",
            "state",
            "total",
            "india",
        }
    ):
        continue

    # Skip rows without an actual state name
    if state.replace(".", "").isdigit():
        continue

    for year, (case_col, death_col) in year_columns.items():

        cases = pd.to_numeric(
            row.iloc[case_col],
            errors="coerce"
        )

        deaths = pd.to_numeric(
            row.iloc[death_col],
            errors="coerce"
        )

        # Keep rows where at least one value exists
        if pd.isna(cases) and pd.isna(deaths):
            continue

        records.append({
            "year": year,
            "state_name": state,
            "dengue_cases": cases,
            "dengue_deaths": deaths,
        })


df = pd.DataFrame(records)


# ============================================================
# NUMERIC TYPES
# ============================================================

df["year"] = df["year"].astype(int)

df["dengue_cases"] = pd.to_numeric(
    df["dengue_cases"],
    errors="coerce"
)

df["dengue_deaths"] = pd.to_numeric(
    df["dengue_deaths"],
    errors="coerce"
)


# ============================================================
# COVERAGE
# ============================================================

df["data_coverage"] = "FULL_YEAR"

df.loc[
    df["year"] == 2026,
    "data_coverage"
] = "PROVISIONAL_THROUGH_2026_02_28"


# ============================================================
# SOURCE
# ============================================================

df["source"] = (
    "NCVBDC, Ministry of Health & Family Welfare, "
    "Government of India"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated(
    subset=[
        "year",
        "state_name"
    ],
    keep=False
)

if duplicates.any():

    print()
    print(
        "WARNING: duplicate state-year rows detected:"
    )

    print(
        df[duplicates].to_string(index=False)
    )

    raise RuntimeError(
        "Duplicate state-year records found."
    )


# ============================================================
# VALIDATION
# ============================================================

errors = []

if df.empty:
    errors.append(
        "No dengue records extracted."
    )

if not df["year"].between(2021, 2026).all():
    errors.append(
        "Invalid year found."
    )

if df["dengue_cases"].isna().all():
    errors.append(
        "All dengue case values are missing."
    )

if (
    df["dengue_cases"].dropna() < 0
).any():
    errors.append(
        "Negative dengue cases found."
    )

if (
    df["dengue_deaths"].dropna() < 0
).any():
    errors.append(
        "Negative dengue deaths found."
    )

if (
    df["dengue_deaths"].fillna(0)
    > df["dengue_cases"].fillna(0)
).any():
    errors.append(
        "Deaths greater than cases found."
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "year",
        "state_name"
    ]
).reset_index(drop=True)


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
    print("VALIDATION FAILED")
    print("=" * 70)

    for error in errors:
        print("ERROR:", error)

    raise RuntimeError(
        "Dengue dataset failed validation."
    )


df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("DENGUE STATE-YEAR DATASET REPORT")
report.append("=" * 70)
report.append("")
report.append(
    f"Rows              : {len(df):,}"
)
report.append(
    f"Unique states     : "
    f"{df['state_name'].nunique():,}"
)
report.append(
    f"Year range        : "
    f"{df['year'].min()}-{df['year'].max()}"
)
report.append("")
report.append("Rows by year:")
report.append(
    df["year"]
    .value_counts()
    .sort_index()
    .to_string()
)
report.append("")
report.append("Coverage:")
report.append(
    df["data_coverage"]
    .value_counts()
    .to_string()
)

REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("DENGUE DATASET CREATED")
print("=" * 70)

print(f"Rows          : {len(df):,}")

print(
    f"Unique states : "
    f"{df['state_name'].nunique():,}"
)

print()
print("Rows by year:")
print(
    df["year"]
    .value_counts()
    .sort_index()
)

print()
print("Coverage:")
print(
    df["data_coverage"]
    .value_counts()
)

print()
print(f"OUTPUT : {OUTPUT}")
print(f"REPORT : {REPORT}")

print()
print("=" * 70)
print("STATUS: PASSED")
print("=" * 70)