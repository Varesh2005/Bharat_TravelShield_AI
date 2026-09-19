from pathlib import Path
import pdfplumber

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():

    print("=" * 70)
    print("MALARIA STATE-CELL POSITION INSPECTION")
    print("=" * 70)

    pages_to_check = [
        1,
        7,
        18,
        19,
        20,
        31,
        322,
        323,
        324,
        338,
        339,
    ]

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        for page_number in pages_to_check:

            page = pdf.pages[page_number - 1]

            print("\n" + "=" * 70)
            print(f"PAGE {page_number}")
            print("=" * 70)

            # Extract all words with coordinates.
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False
            )

            # Print words located in the approximate State column.
            #
            # The Year column is around x=48-78.
            # State is expected to be to its right.
            #
            # We inspect x=75 to x=300 so we can determine
            # the actual State-column boundaries.
            state_area = [
                word
                for word in words
                if 75 <= word["x0"] <= 300
            ]

            state_area = sorted(
                state_area,
                key=lambda w: (
                    round(w["top"], 1),
                    w["x0"]
                )
            )

            print("\nWORDS IN STATE-AREA:")

            for word in state_area:

                print(
                    f"{word['text']!r:35} "
                    f"x0={word['x0']:7.2f} "
                    f"x1={word['x1']:7.2f} "
                    f"top={word['top']:7.2f} "
                    f"bottom={word['bottom']:7.2f}"
                )

    print("\n" + "=" * 70)
    print("STATE-CELL INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()