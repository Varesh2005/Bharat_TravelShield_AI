import pdfplumber
import pandas as pd
import re
from pathlib import Path


PDF_PATH = Path(
    "data/raw/disease/malaria/"
    "ncvbdc_district_wise_malaria_2000_2024.pdf"
)

OUTPUT_PATH = Path(
    "data/processed/disease/malaria/"
    "malaria_district_year_2000_2024.csv"
)

REPORT_PATH = Path(
    "data/reports/disease/"
    "malaria_extraction_report.csv"
)


YEAR_RE = re.compile(r"^(20\d{2})$")
NUMBER_RE = re.compile(r"^\d+$")


def clean_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).replace("\n", " ")
    ).strip()


def is_number(value):
    value = clean_text(value)
    return bool(NUMBER_RE.fullmatch(value))


def extract_tables():
    records = []

    current_year = None
    current_state = None

    with pdfplumber.open(PDF_PATH) as pdf:

        print("=" * 70)
        print("MALARIA PDF EXTRACTION - CLEAN ROW MODE")
        print("=" * 70)
        print(f"PDF pages : {len(pdf.pages)}")

        for page_no, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()

            if not tables:
                continue

            for table in tables:

                if not table:
                    continue

                for row in table:

                    if not row:
                        continue

                    cells = [clean_text(x) for x in row]

                    # Remove completely empty rows
                    if not any(cells):
                        continue

                    # --------------------------------------------------
                    # YEAR DETECTION
                    # --------------------------------------------------
                    for cell in cells:
                        match = YEAR_RE.fullmatch(cell)

                        if match:
                            year = int(match.group(1))

                            if 2000 <= year <= 2024:
                                current_year = year

                    # --------------------------------------------------
                    # HEADER
                    # --------------------------------------------------
                    joined = " ".join(cells).lower()

                    if (
                        "district" in joined
                        and "total cases" in joined
                    ):
                        continue

                    # --------------------------------------------------
                    # Find numeric columns
                    # Expected row:
                    #
                    # [state, district, cases, deaths]
                    # OR
                    # [year, state, district, cases, deaths]
                    # --------------------------------------------------

                    numeric_positions = [
                        i for i, value in enumerate(cells)
                        if is_number(value)
                    ]

                    if len(numeric_positions) < 2:
                        continue

                    # Last two numeric values are cases/deaths
                    cases_pos = numeric_positions[-2]
                    deaths_pos = numeric_positions[-1]

                    try:
                        total_cases = int(cells[cases_pos])
                        total_deaths = int(cells[deaths_pos])
                    except ValueError:
                        continue

                    # --------------------------------------------------
                    # District is normally immediately before cases.
                    # --------------------------------------------------

                    district_pos = cases_pos - 1

                    if district_pos < 0:
                        continue

                    district = clean_text(cells[district_pos])

                    if not district:
                        continue

                    # --------------------------------------------------
                    # State is the text cell before district.
                    # IMPORTANT:
                    # We do NOT use page-to-page state markers.
                    # --------------------------------------------------

                    state = ""

                    for i in range(district_pos - 1, -1, -1):

                        candidate = clean_text(cells[i])

                        if (
                            candidate
                            and not is_number(candidate)
                            and candidate.lower()
                            not in {
                                "year",
                                "state",
                                "district",
                                "total cases",
                                "total deaths",
                            }
                        ):
                            state = candidate
                            break

                    # --------------------------------------------------
                    # SPECIAL ENTRIES
                    # --------------------------------------------------

                    is_special = "#" in district

                    # Ignore obvious table garbage
                    if district.lower() in {
                        "district",
                        "state",
                        "total cases",
                        "total deaths",
                    }:
                        continue

                    # --------------------------------------------------
                    # Store
                    # --------------------------------------------------

                    records.append(
                        {
                            "year": current_year,
                            "state": state,
                            "district": district,
                            "total_cases": total_cases,
                            "total_deaths": total_deaths,
                            "source_page": page_no,
                            "is_special_entry": is_special,
                        }
                    )

            if page_no % 25 == 0:
                print(f"Processed {page_no}/{len(pdf.pages)} pages")

    return pd.DataFrame(records)


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = extract_tables()

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print("Records extracted :", len(df))

    if df.empty:
        raise ValueError(
            "No malaria records extracted."
        )

    # --------------------------------------------------
    # Remove rows where year could not be determined
    # --------------------------------------------------

    missing_year = df["year"].isna().sum()

    print("Missing year rows  :", missing_year)

    df = df.dropna(
        subset=["year"]
    ).copy()

    df["year"] = df["year"].astype(int)

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    print(
        "Years              :",
        sorted(df["year"].unique())
    )

    print(
        "Unique states      :",
        df["state"].nunique()
    )

    print(
        "Unique districts   :",
        df["district"].nunique()
    )

    print()
    print("Records by year:")
    print(
        df.groupby("year")
        .size()
        .to_string()
    )

    # --------------------------------------------------
    # Exact duplicate source rows
    # --------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "year",
            "state",
            "district",
            "total_cases",
            "total_deaths",
            "source_page",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print()
    print(
        "Exact duplicate source rows :",
        duplicate_count
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    report = pd.DataFrame(
        [
            {
                "metric": "records_extracted",
                "value": len(df),
            },
            {
                "metric": "missing_year_rows",
                "value": missing_year,
            },
            {
                "metric": "unique_states",
                "value": df["state"].nunique(),
            },
            {
                "metric": "unique_districts",
                "value": df["district"].nunique(),
            },
            {
                "metric": "exact_duplicate_source_rows",
                "value": duplicate_count,
            },
        ]
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Output:",
        OUTPUT_PATH
    )

    print(
        "Report:",
        REPORT_PATH
    )


if __name__ == "__main__":
    main()