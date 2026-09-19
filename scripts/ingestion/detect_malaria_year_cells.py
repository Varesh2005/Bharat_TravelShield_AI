from pathlib import Path
import pdfplumber

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)

VALID_YEARS = {
    str(year)
    for year in range(2000, 2025)
}


def find_years_in_year_column(page):
    """
    Find 4-digit year values located in the leftmost Year column.

    The PDF was generated from Excel and the Year column contains
    vertically merged cells. Therefore, text extraction may place
    the year at an unexpected position in reading order.

    We use the physical X coordinate to identify the Year column.
    """

    chars = page.chars

    # Keep characters that are:
    # - digits
    # - physically located in the Year column
    #
    # The Year column is approximately x=50 to x=130.
    year_column_chars = [
        char
        for char in chars
        if char.get("text", "").isdigit()
        and 45 <= char.get("x0", 999) <= 130
    ]

    # Group characters by approximate vertical position.
    rows = {}

    for char in year_column_chars:

        top = round(char.get("top", 0), 1)

        rows.setdefault(top, []).append(char)

    detected = []

    for top, row_chars in rows.items():

        row_chars = sorted(
            row_chars,
            key=lambda c: c.get("x0", 0)
        )

        text = "".join(
            char.get("text", "")
            for char in row_chars
        )

        if text in VALID_YEARS:

            detected.append(
                {
                    "year": int(text),
                    "top": top,
                    "x0": min(c.get("x0", 0) for c in row_chars),
                    "x1": max(c.get("x1", 0) for c in row_chars),
                }
            )

    return detected


def main():

    print("=" * 70)
    print("MALARIA YEAR-CELL DETECTION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        print("\nYEAR CELLS DETECTED:")

        total_detections = 0

        for page_number, page in enumerate(pdf.pages, start=1):

            detections = find_years_in_year_column(page)

            if detections:

                for detection in detections:

                    print(
                        f"Page {page_number:3d} | "
                        f"Year {detection['year']} | "
                        f"top={detection['top']:.2f} | "
                        f"x0={detection['x0']:.2f} | "
                        f"x1={detection['x1']:.2f}"
                    )

                    total_detections += 1

        print("\n" + "-" * 70)

        print(
            f"Total year-cell detections: "
            f"{total_detections}"
        )

    print("\n" + "=" * 70)
    print("DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()