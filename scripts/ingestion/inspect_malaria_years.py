from pathlib import Path
import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():

    print("=" * 70)
    print("MALARIA PDF YEAR STRUCTURE INSPECTION")
    print("=" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        # Inspect selected pages from different parts
        pages_to_inspect = [
            1,
            2,
            10,
            20,
            30,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            320,
            321,
            322,
            323,
            324,
            325,
            330,
            340,
            347,
        ]

        for page_number in pages_to_inspect:

            if page_number > len(pdf.pages):
                continue

            page = pdf.pages[page_number - 1]

            print()
            print("=" * 70)
            print(f"PAGE {page_number}")
            print("=" * 70)

            # Extract raw page text
            text = page.extract_text()

            if text:
                lines = text.splitlines()

                print("--- PAGE TEXT (FIRST 20 LINES) ---")

                for line in lines[:20]:
                    print(repr(line))

                print("--- END PAGE TEXT ---")

            else:
                print("NO TEXT EXTRACTED")

            # Also inspect first table
            tables = page.extract_tables()

            print()
            print(f"Tables found: {len(tables)}")

            if tables:

                table = tables[0]

                print("First 5 table rows:")

                for row in table[:5]:
                    print(row)

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()