import pdfplumber
import re
from pathlib import Path


PDF_PATH = Path(
    "data/raw/disease/malaria/"
    "ncvbdc_district_wise_malaria_2000_2024.pdf"
)

PAGE_NUMBER = 2


def clean(text):
    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).replace("\n", " ")
    ).strip()


with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[PAGE_NUMBER - 1]

    print("=" * 80)
    print("MALARIA PDF COORDINATE INSPECTION")
    print("=" * 80)

    print(f"Page          : {PAGE_NUMBER}")
    print(f"Page width    : {page.width}")
    print(f"Page height   : {page.height}")
    print()

    words = page.extract_words(
        keep_blank_chars=False,
        use_text_flow=False
    )

    print(f"Total words   : {len(words)}")
    print()

    # ---------------------------------------------------------
    # Print every word in approximate reading order
    # ---------------------------------------------------------

    for i, word in enumerate(words):

        text = clean(word.get("text"))

        if not text:
            continue

        x0 = word.get("x0", 0)
        x1 = word.get("x1", 0)
        top = word.get("top", 0)
        bottom = word.get("bottom", 0)

        print(
            f"{i:4d} | "
            f"x0={x0:8.2f} | "
            f"x1={x1:8.2f} | "
            f"top={top:8.2f} | "
            f"bottom={bottom:8.2f} | "
            f"{text}"
        )