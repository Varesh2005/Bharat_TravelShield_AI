from pathlib import Path
import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():
    print("=" * 70)
    print("MALARIA PDF PAGE LAYOUT INSPECTION")
    print("=" * 70)

    with pdfplumber.open(PDF_PATH) as pdf:

        pages_to_check = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
                          220, 240, 260, 280, 300, 320, 330, 340, 347]

        for page_number in pages_to_check:

            page = pdf.pages[page_number - 1]

            print()
            print("-" * 70)
            print(f"PAGE {page_number}")
            print("-" * 70)

            # Extract every word with its coordinates.
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=False
            )

            # Look specifically for anything that looks like a year.
            year_words = []

            for word in words:
                text = word["text"].strip()

                if text.isdigit() and len(text) == 4:
                    year = int(text)

                    if 1990 <= year <= 2030:
                        year_words.append(
                            (
                                text,
                                round(word["x0"], 2),
                                round(word["top"], 2),
                                round(word["x1"], 2),
                                round(word["bottom"], 2),
                            )
                        )

            if year_words:
                print("YEAR-LIKE WORDS FOUND:")
                for item in year_words:
                    print(item)
            else:
                print("NO YEAR-LIKE WORDS FOUND")

            # Print the first 30 words with coordinates.
            print()
            print("FIRST 30 WORDS WITH COORDINATES:")

            for word in words[:30]:
                print(
                    f"text={word['text']!r:30} "
                    f"x0={word['x0']:.1f} "
                    f"top={word['top']:.1f} "
                    f"x1={word['x1']:.1f} "
                    f"bottom={word['bottom']:.1f}"
                )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()