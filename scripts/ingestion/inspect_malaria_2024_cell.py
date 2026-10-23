from pathlib import Path
import pdfplumber

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():

    print("=" * 70)
    print("2024 MALARIA YEAR CELL INSPECTION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:

        for page_number in [337, 338, 339]:

            page = pdf.pages[page_number - 1]

            print("\n" + "=" * 70)
            print(f"PAGE {page_number}")
            print("=" * 70)

            print("\nCHARACTERS IN YEAR COLUMN (x < 130):")

            chars = [
                char
                for char in page.chars
                if 45 <= char.get("x0", 999) <= 130
            ]

            chars = sorted(
                chars,
                key=lambda c: (
                    c.get("top", 0),
                    c.get("x0", 0)
                )
            )

            for char in chars:

                print(
                    repr(char.get("text")),
                    f"x0={char.get('x0'):.2f}",
                    f"top={char.get('top'):.2f}",
                    f"x1={char.get('x1'):.2f}",
                    f"bottom={char.get('bottom'):.2f}"
                )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()