from pathlib import Path
import re

import pdfplumber


PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)


# These are the states/UTs represented in the 2011 district reference.
# We use them only to recognize state-cell text, not to invent disease data.
KNOWN_STATES = {
    "Andaman And Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra And Nagar Haveli",
    "Daman And Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu And Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
}


def normalize_text(text):
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_state_candidates(page):
    """
    Find text in the State column area.

    The state column is approximately x=90 to x=200.
    State names can occupy multiple words, so words on the same
    vertical position are grouped together.
    """

    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
    )

    candidates = []

    # Ignore title/header area.
    words = [
        w
        for w in words
        if 85 <= w["x0"] <= 205
        and w["top"] > 90
    ]

    # Group words by approximately identical vertical position.
    rows = []

    for word in sorted(words, key=lambda w: w["top"]):

        placed = False

        for row in rows:

            if abs(row["top"] - word["top"]) <= 1.0:
                row["words"].append(word)
                placed = True
                break

        if not placed:

            rows.append(
                {
                    "top": word["top"],
                    "words": [word],
                }
            )

    for row in rows:

        row["words"].sort(key=lambda w: w["x0"])

        text = " ".join(
            word["text"]
            for word in row["words"]
        )

        text = normalize_text(text)

        # Try exact state matching.
        matched_state = None

        for state in KNOWN_STATES:

            if text.lower() == state.lower():
                matched_state = state
                break

        if matched_state is not None:

            candidates.append(
                {
                    "state": matched_state,
                    "top": row["top"],
                }
            )

    return candidates


def get_district_rows(page):
    """
    Extract district-column words.

    District names are located approximately at x >= 215.
    """

    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=3,
        keep_blank_chars=False,
    )

    district_words = [
        w
        for w in words
        if 215 <= w["x0"] <= 340
        and w["top"] > 90
    ]

    # Group words by vertical position.
    rows = []

    for word in sorted(district_words, key=lambda w: w["top"]):

        placed = False

        for row in rows:

            if abs(row["top"] - word["top"]) <= 1.0:
                row["words"].append(word)
                placed = True
                break

        if not placed:

            rows.append(
                {
                    "top": word["top"],
                    "words": [word],
                }
            )

    output = []

    for row in rows:

        row["words"].sort(key=lambda w: w["x0"])

        text = " ".join(
            word["text"]
            for word in row["words"]
        )

        text = normalize_text(text)

        # Ignore header.
        if text.lower() == "district":
            continue

        output.append(
            {
                "district": text,
                "top": row["top"],
            }
        )

    return output


def main():

    print("=" * 80)
    print("MALARIA STATE ASSIGNMENT DIAGNOSTIC")
    print("=" * 80)

    pages_to_check = [
        1,
        7,
        18,
        19,
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

            states = get_state_candidates(page)
            districts = get_district_rows(page)

            print("\n" + "=" * 80)
            print(f"PAGE {page_number}")
            print("=" * 80)

            print("\nSTATE MARKERS:")

            if states:

                for state in states:

                    print(
                        f"State = {state['state']!r:35} "
                        f"top = {state['top']:8.2f}"
                    )

            else:

                print("NONE")

            print("\nDISTRICT ROWS + NEAREST STATE MARKER:")

            if not districts:

                print("NONE")
                continue

            for district in districts:

                if states:

                    nearest = min(
                        states,
                        key=lambda s: abs(
                            s["top"] - district["top"]
                        )
                    )

                    distance = abs(
                        nearest["top"] - district["top"]
                    )

                    print(
                        f"{district['district']!r:35} "
                        f"district_top={district['top']:8.2f} "
                        f"nearest_state={nearest['state']!r:30} "
                        f"distance={distance:7.2f}"
                    )

                else:

                    print(
                        f"{district['district']!r:35} "
                        f"district_top={district['top']:8.2f} "
                        f"nearest_state=NONE"
                    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()