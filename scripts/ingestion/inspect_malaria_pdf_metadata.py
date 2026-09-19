from pathlib import Path
import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():
    print("=" * 70)
    print("MALARIA PDF METADATA / PAGE STRUCTURE INSPECTION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        print("\nPDF METADATA:")
        print(pdf.metadata)

        print("\nPAGE SIZES:")
        for page_number in [1, 2, 3, 100, 200, 300, 347]:
            page = pdf.pages[page_number - 1]

            print(
                f"Page {page_number}: "
                f"width={page.width}, "
                f"height={page.height}, "
                f"rotation={page.rotation}"
            )

        print("\nPAGE TEXT LENGTHS:")
        for page_number in range(1, len(pdf.pages) + 1):
            page = pdf.pages[page_number - 1]
            text = page.extract_text() or ""

            if page_number <= 10 or page_number % 25 == 0:
                print(
                    f"Page {page_number:3d}: "
                    f"{len(text):5d} characters"
                )

        print("\nSEARCHING ALL PAGE TEXT FOR YEAR-LIKE VALUES...")

        found = {}

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            for year in range(2000, 2025):

                if str(year) in text:
                    found.setdefault(year, []).append(page_number)

        for year in sorted(found):
            pages = found[year]

            print(
                f"{year}: "
                f"{len(pages)} pages -> "
                f"{pages[:20]}"
                + (" ..." if len(pages) > 20 else "")
            )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()