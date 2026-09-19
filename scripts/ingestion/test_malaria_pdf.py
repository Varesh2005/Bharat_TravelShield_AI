from pathlib import Path
import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():
    print("=" * 60)
    print("MALARIA PDF EXTRACTION TEST")
    print("=" * 60)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"PDF: {PDF_PATH}")
    print(f"Size: {PDF_PATH.stat().st_size:,} bytes")

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        # Test first 3 pages only
        for page_number in range(1, 4):
            page = pdf.pages[page_number - 1]

            print("\n" + "-" * 60)
            print(f"PAGE {page_number}")
            print("-" * 60)

            tables = page.extract_tables()

            print(f"Tables found: {len(tables)}")

            for table_number, table in enumerate(tables, start=1):
                print(f"\nTable {table_number}:")
                print(f"Rows extracted: {len(table)}")

                # Print first 10 rows only
                for row in table[:10]:
                    print(row)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()