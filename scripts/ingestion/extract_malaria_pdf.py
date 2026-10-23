from pathlib import Path
import csv
import re

import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)

OUTPUT_PATH = Path(
    "data/interim/disease/malaria/malaria_district_raw_extracted.csv"
)


EXPECTED_COLUMNS = [
    "year",
    "state_name",
    "district_name",
    "total_cases",
    "total_deaths",
]


def clean_text(value):
    """Clean extracted PDF cell text."""
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_integer(value):
    """Convert a numeric PDF cell to integer."""
    value = clean_text(value)

    if not value:
        return None

    # Remove commas such as 12,345
    value = value.replace(",", "")

    # Only accept integer values
    if not re.fullmatch(r"-?\d+", value):
        return None

    return int(value)


def is_year(value):
    """Check whether a value represents a four-digit year."""
    value = clean_text(value)
    return bool(re.fullmatch(r"20\d{2}", value))


def main():

    print("=" * 70)
    print("MALARIA PDF EXTRACTION")
    print("=" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Malaria PDF not found: {PDF_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Input PDF : {PDF_PATH}")
    print(f"Output CSV: {OUTPUT_PATH}")
    print(f"PDF size  : {PDF_PATH.stat().st_size:,} bytes")

    extracted_rows = []

    current_year = None
    current_state = None

    total_pages = 0
    total_tables = 0

    with pdfplumber.open(PDF_PATH) as pdf:

        total_pages = len(pdf.pages)

        print(f"Total pages: {total_pages}")

        for page_number, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()

            total_tables += len(tables)

            for table in tables:

                if not table:
                    continue

                for row in table:

                    if not row:
                        continue

                    # Normalize row length
                    row = list(row)

                    while len(row) < 5:
                        row.append(None)

                    row = row[:5]

                    year_cell = clean_text(row[0])
                    state_cell = clean_text(row[1])
                    district_cell = clean_text(row[2])
                    cases_cell = clean_text(row[3])
                    deaths_cell = clean_text(row[4])

                    # Skip table header
                    if (
                        year_cell.lower() == "year"
                        and state_cell.lower() == "state"
                    ):
                        continue

                    # Detect year
                    if is_year(year_cell):
                        current_year = int(year_cell)

                    # Detect state
                    if state_cell:
                        current_state = state_cell

                    # District must exist
                    if not district_cell:
                        continue

                    # Ignore obvious repeated headers
                    if district_cell.lower() == "district":
                        continue

                    # Parse numerical fields
                    total_cases = parse_integer(cases_cell)
                    total_deaths = parse_integer(deaths_cell)

                    # Store extracted record
                    extracted_rows.append(
                        {
                            "year": current_year,
                            "state_name": current_state,
                            "district_name": district_cell,
                            "total_cases": total_cases,
                            "total_deaths": total_deaths,
                            "source_page": page_number,
                        }
                    )

            # Progress every 25 pages
            if page_number % 25 == 0:
                print(
                    f"Processed pages: {page_number}/{total_pages} | "
                    f"Rows extracted: {len(extracted_rows):,}"
                )

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"Pages processed       : {total_pages}")
    print(f"Tables found          : {total_tables}")
    print(f"Rows extracted        : {len(extracted_rows):,}")

    # Basic diagnostics
    missing_year = sum(
        row["year"] is None
        for row in extracted_rows
    )

    missing_state = sum(
        not row["state_name"]
        for row in extracted_rows
    )

    missing_district = sum(
        not row["district_name"]
        for row in extracted_rows
    )

    missing_cases = sum(
        row["total_cases"] is None
        for row in extracted_rows
    )

    missing_deaths = sum(
        row["total_deaths"] is None
        for row in extracted_rows
    )

    print()
    print("Missing values:")
    print(f"  Year          : {missing_year:,}")
    print(f"  State         : {missing_state:,}")
    print(f"  District      : {missing_district:,}")
    print(f"  Total Cases   : {missing_cases:,}")
    print(f"  Total Deaths  : {missing_deaths:,}")

    # Year distribution
    years = sorted(
        {
            row["year"]
            for row in extracted_rows
            if row["year"] is not None
        }
    )

    print()
    print(
        f"Years detected: "
        f"{years[0] if years else 'N/A'} "
        f"to "
        f"{years[-1] if years else 'N/A'}"
    )

    # State count
    states = sorted(
        {
            row["state_name"]
            for row in extracted_rows
            if row["state_name"]
        }
    )

    print(f"Unique states/UTs detected: {len(states)}")

    # Save CSV
    fieldnames = EXPECTED_COLUMNS + ["source_page"]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(extracted_rows)

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Output file:")
    print(OUTPUT_PATH)

    # Print sample rows
    print()
    print("First 10 extracted records:")

    for row in extracted_rows[:10]:
        print(row)


if __name__ == "__main__":
    main()