from pathlib import Path
import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


def main():
    print("=" * 80)
    print("MALARIA STATE-CELL BOUNDARY INSPECTION")
    print("=" * 80)

    pages_to_check = [
        7,
        20,
        31,
        322,
        323,
        338,
        339,
    ]

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        for page_number in pages_to_check:

            page = pdf.pages[page_number - 1]

            print("\n" + "=" * 80)
            print(f"PAGE {page_number}")
            print("=" * 80)

            print(f"Page width  : {page.width}")
            print(f"Page height : {page.height}")

            # ---------------------------------------------------------
            # HORIZONTAL LINES
            # ---------------------------------------------------------

            print("\nHORIZONTAL LINES:")

            horizontal_lines = []

            for line in page.lines:

                x0 = line.get("x0", 0)
                x1 = line.get("x1", 0)
                top = line.get("top", 0)
                bottom = line.get("bottom", 0)

                # Horizontal line
                if abs(x1 - x0) > 20 and abs(bottom - top) < 2:

                    horizontal_lines.append(
                        {
                            "x0": x0,
                            "x1": x1,
                            "top": top,
                            "bottom": bottom,
                        }
                    )

            horizontal_lines.sort(
                key=lambda x: (
                    round(x["top"], 2),
                    round(x["x0"], 2)
                )
            )

            for line in horizontal_lines:

                print(
                    f"x0={line['x0']:8.2f} "
                    f"x1={line['x1']:8.2f} "
                    f"top={line['top']:8.2f}"
                )

            # ---------------------------------------------------------
            # VERTICAL LINES
            # ---------------------------------------------------------

            print("\nVERTICAL LINES:")

            vertical_lines = []

            for line in page.lines:

                x0 = line.get("x0", 0)
                x1 = line.get("x1", 0)
                top = line.get("top", 0)
                bottom = line.get("bottom", 0)

                # Vertical line
                if abs(bottom - top) > 20 and abs(x1 - x0) < 2:

                    vertical_lines.append(
                        {
                            "x0": x0,
                            "x1": x1,
                            "top": top,
                            "bottom": bottom,
                        }
                    )

            vertical_lines.sort(
                key=lambda x: (
                    round(x["x0"], 2),
                    round(x["top"], 2)
                )
            )

            for line in vertical_lines:

                print(
                    f"x={line['x0']:8.2f} "
                    f"top={line['top']:8.2f} "
                    f"bottom={line['bottom']:8.2f}"
                )

    print("\n" + "=" * 80)
    print("STATE-CELL BOUNDARY INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()