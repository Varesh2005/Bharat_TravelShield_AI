import pdfplumber
import re

PDF_PATH = "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"

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


def normalize(text):
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def group_words_by_y(words, tolerance=1.0):
    """
    Groups words that belong to the same visual row.
    """
    rows = []

    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):

        placed = False

        for row in rows:
            if abs(row["top"] - word["top"]) <= tolerance:
                row["words"].append(word)
                placed = True
                break

        if not placed:
            rows.append({
                "top": word["top"],
                "words": [word]
            })

    return rows


def get_state_markers(page):
    words = page.extract_words()

    # State names are located in the State column.
    state_words = [
        w for w in words
        if 75 <= w["x0"] <= 210
        and w["top"] > 90
    ]

    rows = group_words_by_y(state_words)

    markers = []

    for row in rows:

        text = normalize(
            " ".join(
                w["text"]
                for w in sorted(row["words"], key=lambda x: x["x0"])
            )
        )

        if text in KNOWN_STATES:
            markers.append({
                "state": text,
                "top": row["top"]
            })

    return markers


def get_district_rows(page):
    words = page.extract_words()

    district_words = [
        w for w in words
        if 210 <= w["x0"] <= 390
        and w["top"] > 90
    ]

    rows = group_words_by_y(district_words)

    districts = []

    for row in rows:

        text = normalize(
            " ".join(
                w["text"]
                for w in sorted(row["words"], key=lambda x: x["x0"])
            )
        )

        # Ignore obvious header text
        if text.lower() in {
            "district",
            "total cases",
            "total deaths",
        }:
            continue

        if text:
            districts.append({
                "district": text,
                "top": row["top"]
            })

    return districts


def assign_by_boundaries(districts, markers):
    """
    State markers represent the CENTER of merged state cells.

    Boundaries between consecutive state markers are therefore
    placed halfway between their Y positions.

    District rows are assigned to the state interval in which
    they fall.
    """

    if not markers:
        return []

    markers = sorted(markers, key=lambda x: x["top"])

    boundaries = []

    for i in range(len(markers) - 1):

        midpoint = (
            markers[i]["top"] +
            markers[i + 1]["top"]
        ) / 2

        boundaries.append(midpoint)

    results = []

    for district in districts:

        y = district["top"]

        state_index = 0

        for boundary in boundaries:

            if y < boundary:
                break

            state_index += 1

        state = markers[state_index]["state"]

        results.append({
            "district": district["district"],
            "district_top": y,
            "assigned_state": state
        })

    return results


TEST_PAGES = [
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


print("=" * 80)
print("MALARIA STATE ASSIGNMENT — INTERVAL BOUNDARY TEST")
print("=" * 80)

with pdfplumber.open(PDF_PATH) as pdf:

    for page_number in TEST_PAGES:

        page = pdf.pages[page_number - 1]

        markers = get_state_markers(page)
        districts = get_district_rows(page)

        results = assign_by_boundaries(
            districts,
            markers
        )

        print()
        print("=" * 80)
        print(f"PAGE {page_number}")
        print("=" * 80)

        print("\nSTATE MARKERS:")

        for marker in markers:
            print(
                f"State = {marker['state']!r:40s}"
                f" top = {marker['top']:8.2f}"
            )

        print("\nDISTRICT ASSIGNMENTS:")

        for result in results:
            print(
                f"{result['district']!r:35s}"
                f" district_top={result['district_top']:8.2f}"
                f" assigned_state={result['assigned_state']!r}"
            )

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)