from pathlib import Path
import pdfplumber
from collections import Counter

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)

YEARS = {str(year) for year in range(2000, 2025)}


def inspect_page(page, page_number):
    print("\n" + "=" * 70)
    print(f"PAGE {page_number} CHARACTER INSPECTION")
    print("=" * 70)

    chars = page.chars

    print(f"Total characters: {len(chars)}")

    # Show all characters that could belong to a year.
    year_chars = []

    for char in chars:
        text = char.get("text", "")

        if text.isdigit():
            year_chars.append(char)

    print(f"Numeric characters: {len(year_chars)}")

    print("\nFIRST 100 NUMERIC CHARACTERS:")

    for char in year_chars[:100]:
        print(
            repr(char.get("text")),
            f"x0={char.get('x0'):.2f}",
            f"top={char.get('top'):.2f}",
            f"x1={char.get('x1'):.2f}",
            f"bottom={char.get('bottom'):.2f}"
        )

    # Reconstruct nearby numeric sequences by position.
    print("\nCHARACTERS NEAR TOP OF PAGE:")

    top_chars = [
        char
        for char in chars
        if char.get("top", 9999) < 100
    ]

    for char in top_chars:
        print(
            repr(char.get("text")),
            f"x0={char.get('x0'):.2f}",
            f"top={char.get('top'):.2f}"
        )

    print("\n" + "-" * 70)


def main():
    print("=" * 70)
    print("MALARIA PDF CHARACTER-LEVEL INSPECTION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        # Inspect representative pages.
        pages_to_check = [
            1,
            2,
            18,
            20,
            31,
            61,
            100,
            200,
            276,
            291,
            306,
            322,
            323,
            347,
        ]

        for page_number in pages_to_check:
            page = pdf.pages[page_number - 1]
            inspect_page(page, page_number)

        print("\n" + "=" * 70)
        print("SEARCHING CHARACTER OBJECTS FOR EXACT 4-DIGIT YEARS")
        print("=" * 70)

        year_locations = Counter()

        for page_number, page in enumerate(pdf.pages, start=1):

            chars = page.chars

            # Sort characters by reading position.
            chars_sorted = sorted(
                chars,
                key=lambda c: (
                    round(c.get("top", 0), 1),
                    c.get("x0", 0)
                )
            )

            # Build text from character objects.
            text = "".join(
                char.get("text", "")
                for char in chars_sorted
            )

            for year in YEARS:

                if year in text:

                    # Ignore the static title years.
                    if year in {"2000", "2024"}:
                        continue

                    year_locations[year] += 1

        print("\nYEAR OCCURRENCES IN CHARACTER TEXT:")

        if not year_locations:
            print("No non-title years detected.")

        else:
            for year in sorted(year_locations):
                print(
                    f"{year}: "
                    f"{year_locations[year]} pages"
                )

    print("\n" + "=" * 70)
    print("CHARACTER INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()