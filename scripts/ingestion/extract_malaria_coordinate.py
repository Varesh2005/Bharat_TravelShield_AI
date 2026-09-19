import pdfplumber
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)

OUTPUT_CSV = Path(
    "data/raw/disease/malaria/malaria_extracted_coordinate.csv"
)

REPORT_TXT = Path(
    "data/raw/disease/malaria/malaria_coordinate_extraction_report.txt"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def norm_text(x):
    if x is None:
        return ""
    x = str(x).strip().upper()
    x = x.replace("&", "AND")
    x = re.sub(r"[^A-Z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def is_number(x):
    if not x:
        return False
    x = str(x).strip().replace(",", "")
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", x))


def number_value(x):
    x = str(x).strip().replace(",", "")
    try:
        return int(float(x))
    except Exception:
        return None


# Official/known Indian state + UT names.
# Used only to recognize state labels in the PDF.
KNOWN_STATES = {
    norm_text(x)
    for x in [
        "Andaman and Nicobar Islands",
        "Andhra Pradesh",
        "Arunachal Pradesh",
        "Assam",
        "Bihar",
        "Chandigarh",
        "Chhattisgarh",
        "Dadra and Nagar Haveli",
        "Dadra and Nagar Haveli and Daman and Diu",
        "Daman and Diu",
        "Delhi",
        "Goa",
        "Gujarat",
        "Haryana",
        "Himachal Pradesh",
        "Jammu and Kashmir",
        "Jammu & Kashmir",
        "Jharkhand",
        "Karnataka",
        "Kerala",
        "Ladakh",
        "Lakshadweep",
        "Madhya Pradesh",
        "Maharashtra",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Odisha",
        "Orissa",
        "Puducherry",
        "Punjab",
        "Rajasthan",
        "Sikkim",
        "Tamil Nadu",
        "Telangana",
        "Tripura",
        "Uttar Pradesh",
        "Uttarakhand",
        "Uttaranchal",
        "West Bengal",
    ]
}


def canonical_state_name(state):
    n = norm_text(state)

    aliases = {
        "ORISSA": "Odisha",
        "UTTARANCHAL": "Uttarakhand",
        "JAMMU AND KASHMIR": "Jammu and Kashmir",
        "JAMMU KASHMIR": "Jammu and Kashmir",
        "DADRA AND NAGAR HAVELI": "Dadra and Nagar Haveli",
        "DAMAN AND DIU": "Daman and Diu",
        "DELHI": "Delhi",
    }

    return aliases.get(n, state.strip())


def group_words_by_row(words, tolerance=3.0):
    """
    Group words using their vertical center.
    """
    rows = []

    for w in sorted(words, key=lambda z: (z["top"], z["x0"])):
        cy = (float(w["top"]) + float(w["bottom"])) / 2

        placed = False

        for row in rows:
            if abs(cy - row["cy"]) <= tolerance:
                row["words"].append(w)
                row["cy"] = sum(
                    (float(a["top"]) + float(a["bottom"])) / 2
                    for a in row["words"]
                ) / len(row["words"])
                placed = True
                break

        if not placed:
            rows.append({
                "cy": cy,
                "words": [w]
            })

    for row in rows:
        row["words"].sort(key=lambda z: z["x0"])

    return rows


def row_text(words):
    return " ".join(str(w["text"]).strip() for w in words)


# ------------------------------------------------------------
# Extraction
# ------------------------------------------------------------

print("=" * 70)
print("FINAL MALARIA COORDINATE EXTRACTION")
print("=" * 70)

all_rows = []

# Global marker lists.
year_markers = []
state_markers = []

page_offsets = []
global_offset = 0.0

with pdfplumber.open(PDF_PATH) as pdf:

    total_pages = len(pdf.pages)

    print(f"PDF pages: {total_pages}")

    # --------------------------------------------------------
    # PASS 1
    # Extract all words with GLOBAL coordinates.
    # --------------------------------------------------------

    for page_no, page in enumerate(pdf.pages, start=1):

        page_width = float(page.width)
        page_height = float(page.height)

        page_offsets.append(global_offset)

        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=2,
            keep_blank_chars=False,
            use_text_flow=False
        )

        converted = []

        for w in words:

            try:
                x0 = float(w["x0"])
                x1 = float(w["x1"])
                top = float(w["top"])
                bottom = float(w["bottom"])
            except Exception:
                continue

            converted.append({
                "text": str(w["text"]).strip(),
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "page": page_no,
                "global_top": global_offset + top,
                "global_bottom": global_offset + bottom,
                "global_cy": global_offset + (top + bottom) / 2,
            })

        # ----------------------------------------------------
        # YEAR MARKERS
        #
        # Year column is approximately x=50-80.
        # Year is a merged cell, so it appears only once.
        # We keep its global vertical center.
        # ----------------------------------------------------

        for w in converted:

            if w["x0"] > 85:
                continue

            txt = w["text"].strip()

            # Normal years 2000-2023
            m = re.fullmatch(r"(20\d{2})", txt)

            if m:
                year = int(m.group(1))

                if 2000 <= year <= 2024:
                    year_markers.append({
                        "year": year,
                        "y": w["global_cy"],
                        "page": page_no,
                        "text": txt
                    })
                continue

            # Handle PDF's 2024 formatting variants.
            if txt in {"024", "2024", "24"}:
                year_markers.append({
                    "year": 2024,
                    "y": w["global_cy"],
                    "page": page_no,
                    "text": txt
                })

        # ----------------------------------------------------
        # STATE MARKERS
        #
        # State column approximately x=90-180.
        # We identify state labels and retain their global
        # vertical center.
        # ----------------------------------------------------

        state_zone = [
            w for w in converted
            if 80 <= w["x0"] <= 190
        ]

        state_rows = group_words_by_row(state_zone, tolerance=4)

        for r in state_rows:

            text = row_text(r["words"]).strip()

            if not text:
                continue

            normalized = norm_text(text)

            if normalized in KNOWN_STATES:
                state_markers.append({
                    "state": canonical_state_name(text),
                    "y": r["cy"],
                    "page": page_no,
                    "text": text
                })

        # ----------------------------------------------------
        # DISTRICT DATA ROWS
        #
        # District x≈210-350
        # Cases    x≈390-435
        # Deaths   x≈500-535
        # ----------------------------------------------------

        district_zone = [
            w for w in converted
            if 205 <= w["x0"] <= 350
        ]

        district_rows = group_words_by_row(
            district_zone,
            tolerance=3.5
        )

        # Numeric columns indexed by vertical center.
        case_words = [
            w for w in converted
            if 385 <= w["x0"] <= 440 and is_number(w["text"])
        ]

        death_words = [
            w for w in converted
            if 495 <= w["x0"] <= 535 and is_number(w["text"])
        ]

        for r in district_rows:

            district = row_text(r["words"]).strip()

            if not district:
                continue

            # Skip obvious headers.
            dnorm = norm_text(district)

            if dnorm in {
                "DISTRICT",
                "DISTRICT NAME",
                "DISTRICTNAME",
            }:
                continue

            cy = global_offset + r["cy"]

            # Find nearest case value on same row.
            nearby_cases = [
                w for w in case_words
                if abs(
                    ((w["top"] + w["bottom"]) / 2 + global_offset)
                    - cy
                ) <= 5
            ]

            nearby_deaths = [
                w for w in death_words
                if abs(
                    ((w["top"] + w["bottom"]) / 2 + global_offset)
                    - cy
                ) <= 5
            ]

            if not nearby_cases:
                continue

            cases = number_value(
                min(
                    nearby_cases,
                    key=lambda w: abs(
                        ((w["top"] + w["bottom"]) / 2 + global_offset)
                        - cy
                    )
                )["text"]
            )

            deaths = None

            if nearby_deaths:
                deaths = number_value(
                    min(
                        nearby_deaths,
                        key=lambda w: abs(
                            ((w["top"] + w["bottom"]) / 2 + global_offset)
                            - cy
                        )
                    )["text"]
                )

            # Store global row.
            all_rows.append({
                "page": page_no,
                "global_y": cy,
                "district": district,
                "cases": cases,
                "deaths": deaths
            })

        global_offset += page_height

        if page_no % 25 == 0:
            print(f"Processed {page_no}/{total_pages}")


# ------------------------------------------------------------
# CLEAN MARKERS
# ------------------------------------------------------------

# Remove duplicated year markers that can occur due PDF extraction.
def deduplicate_markers(markers, key_fields, y_tolerance=5):

    output = []

    markers = sorted(
        markers,
        key=lambda x: (x["y"], x["page"])
    )

    for marker in markers:

        duplicate = False

        for existing in output[-3:]:
            same_key = all(
                marker[k] == existing[k]
                for k in key_fields
            )

            if same_key and abs(marker["y"] - existing["y"]) <= y_tolerance:
                duplicate = True
                break

        if not duplicate:
            output.append(marker)

    return output


year_markers = deduplicate_markers(
    year_markers,
    ["year"]
)

state_markers = deduplicate_markers(
    state_markers,
    ["state"]
)

print()
print("Markers detected:")
print(f"Year markers : {len(year_markers)}")
print(f"State markers: {len(state_markers)}")


# ------------------------------------------------------------
# ASSIGN YEAR
#
# Since year is a vertically merged cell, nearest year marker
# in GLOBAL document coordinates is the correct block anchor.
# ------------------------------------------------------------

def nearest_marker(y, markers):
    if not markers:
        return None

    return min(
        markers,
        key=lambda m: abs(m["y"] - y)
    )


records = []

for row in all_rows:

    ym = nearest_marker(
        row["global_y"],
        year_markers
    )

    sm = nearest_marker(
        row["global_y"],
        state_markers
    )

    year = ym["year"] if ym else None
    state = sm["state"] if sm else None

    records.append({
        "year": year,
        "state": state,
        "district": row["district"],
        "total_cases": row["cases"],
        "total_deaths": row["deaths"],
        "source_page": row["page"]
    })


df = pd.DataFrame(records)


# ------------------------------------------------------------
# BASIC CLEANING
# ------------------------------------------------------------

df["district"] = (
    df["district"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["state"] = (
    df["state"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df.loc[
    df["district"].str.lower().isin(
        ["district", "district name", "nan"]
    ),
    "district"
] = pd.NA


# Remove rows where no disease numbers were extracted.
df = df[
    df["total_cases"].notna()
].copy()


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

missing_years = int(df["year"].isna().sum())
missing_states = int(df["state"].isna().sum())

years = sorted(
    df["year"].dropna().astype(int).unique().tolist()
)

states = sorted(
    df["state"].dropna().unique().tolist()
)

district_count = df["district"].nunique()


# Duplicate source rows are allowed to exist for now.
# We DO NOT aggregate them because previous diagnostics proved
# that some apparent duplicates are caused by wrong state
# assignment in the source extraction.
duplicates = int(
    df.duplicated(
        subset=["year", "state", "district"]
    ).sum()
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

year_counts = (
    df["year"]
    .value_counts(dropna=False)
    .sort_index()
)

report = []

report.append("=" * 70)
report.append("MALARIA COORDINATE EXTRACTION REPORT")
report.append("=" * 70)
report.append("")
report.append(f"Records                 : {len(df):,}")
report.append(f"Missing years           : {missing_years:,}")
report.append(f"Missing states          : {missing_states:,}")
report.append(f"Unique states           : {len(states)}")
report.append(f"Unique districts        : {district_count}")
report.append(f"Duplicate year/state/district: {duplicates:,}")
report.append("")
report.append(f"Year markers detected   : {len(year_markers)}")
report.append(f"State markers detected  : {len(state_markers)}")
report.append("")
report.append("Years:")
report.append(str(years))
report.append("")
report.append("Records by year:")
report.append(str(year_counts))
report.append("")
report.append("States:")
report.extend(states)
report.append("")
report.append("First 30 records:")
report.append(str(df.head(30).to_string(index=False)))

REPORT_TXT.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXTRACTION SUMMARY")
print("=" * 70)

print(f"Records          : {len(df):,}")
print(f"Missing years    : {missing_years:,}")
print(f"Missing states   : {missing_states:,}")
print(f"Unique states    : {len(states)}")
print(f"Unique districts : {district_count}")
print(f"Duplicates       : {duplicates:,}")

print()
print("Years:")
print(years)

print()
print("Records by year:")
print(year_counts)

print()
print(f"CSV    : {OUTPUT_CSV}")
print(f"Report : {REPORT_TXT}")

print()
print("=" * 70)

if missing_years > len(df) * 0.05:
    print("STATUS: FAILED - YEAR ASSIGNMENT STILL INVALID")
elif missing_states > len(df) * 0.05:
    print("STATUS: FAILED - STATE ASSIGNMENT STILL INVALID")
elif len(years) < 24:
    print("STATUS: FAILED - INCOMPLETE YEAR COVERAGE")
else:
    print("STATUS: EXTRACTION PASSED BASIC STRUCTURAL CHECKS")

print("=" * 70)