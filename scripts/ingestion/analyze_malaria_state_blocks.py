import pdfplumber
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = Path(
    "data/raw/disease/malaria/"
    "ncvbdc_district_wise_malaria_2000_2024.pdf"
)

# State/UT names expected in the PDF.
KNOWN_STATES = [
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
    "Andaman & Nicobar Islands",
    "Jammu & Kashmir",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """Normalize whitespace and common PDF text artifacts."""
    if text is None:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_state(text):
    """Normalize state name for comparison."""
    text = normalize_text(text).lower()

    replacements = {
        "&": "and",
        ".": "",
        "#": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


NORMALIZED_STATES = {
    normalize_state(state): state
    for state in KNOWN_STATES
}


def is_integer(text):
    """Return True if text is an integer."""
    if text is None:
        return False

    text = text.strip().replace(",", "")

    return bool(re.fullmatch(r"-?\d+", text))


# ============================================================
# EXTRACT STATE MARKERS
# ============================================================

def extract_state_markers(page):
    """
    Detect state names from the State column.

    Important:
    State names in this PDF are merged cells. Therefore the
    detected state's Y-coordinate represents approximately the
    CENTER of the state's block, NOT its starting row.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=1,
        keep_blank_chars=False
    )

    # State column is approximately x=75 to x=215.
    state_words = [
        w for w in words
        if 70 <= w["x0"] <= 215
        and w["top"] > 90
    ]

    # Group words by vertical position.
    groups = []

    for word in sorted(state_words, key=lambda x: (x["top"], x["x0"])):

        placed = False

        for group in groups:

            if abs(group["top"] - word["top"]) <= 1.5:
                group["words"].append(word)
                placed = True
                break

        if not placed:
            groups.append({
                "top": word["top"],
                "words": [word]
            })

    # Convert each group to text.
    lines = []

    for group in groups:

        text = " ".join(
            w["text"]
            for w in sorted(group["words"], key=lambda x: x["x0"])
        )

        lines.append({
            "top": group["top"],
            "text": normalize_text(text)
        })

    markers = []

    # First check single-line state names.
    for line in lines:

        normalized = normalize_state(line["text"])

        if normalized in NORMALIZED_STATES:

            markers.append({
                "state": NORMALIZED_STATES[normalized],
                "top": line["top"],
                "text": line["text"],
            })

    # Then check combinations of nearby lines.
    # This handles names such as:
    # Andaman And Nicobar
    # Islands
    #
    # or other PDF wrapping artifacts.
    for i in range(len(lines)):

        combined = ""

        for j in range(i, min(i + 4, len(lines))):

            if j > i:

                vertical_gap = lines[j]["top"] - lines[j - 1]["top"]

                # Do not combine distant table rows.
                if vertical_gap > 22:
                    break

            if combined:
                combined += " "

            combined += lines[j]["text"]

            normalized = normalize_state(combined)

            if normalized in NORMALIZED_STATES:

                # Avoid duplicate detection.
                already_exists = any(
                    m["state"] == NORMALIZED_STATES[normalized]
                    and abs(m["top"] - lines[i]["top"]) < 25
                    for m in markers
                )

                if not already_exists:

                    markers.append({
                        "state": NORMALIZED_STATES[normalized],
                        "top": lines[i]["top"],
                        "text": combined,
                    })

    return sorted(markers, key=lambda x: x["top"])


# ============================================================
# EXTRACT TRUE ROW ANCHORS
# ============================================================

def extract_row_anchors(page):
    """
    Detect actual table rows using the numeric Cases/Deaths
    columns instead of relying only on district-name text.

    This is important because some district names wrap over
    multiple visual text lines.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=1,
        keep_blank_chars=False
    )

    numeric_words = [
        w for w in words
        if 340 <= w["x0"] <= 560
        and w["top"] > 90
        and is_integer(w["text"])
    ]

    # Group numeric values by Y coordinate.
    groups = []

    for word in sorted(numeric_words, key=lambda x: (x["top"], x["x0"])):

        placed = False

        for group in groups:

            if abs(group["top"] - word["top"]) <= 1.5:
                group["words"].append(word)
                placed = True
                break

        if not placed:
            groups.append({
                "top": word["top"],
                "words": [word]
            })

    anchors = []

    for group in groups:

        values = [
            w["text"]
            for w in sorted(group["words"], key=lambda x: x["x0"])
        ]

        # A valid row should normally contain Cases and Deaths.
        if len(values) >= 2:

            anchors.append({
                "top": group["top"],
                "values": values,
            })

    return anchors


# ============================================================
# EXTRACT NEAREST DISTRICT NAME
# ============================================================

def get_district_near_row(page, row_top):
    """
    Extract the district-name text closest to a detected
    numeric row anchor.
    """

    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=1,
        keep_blank_chars=False
    )

    district_words = [
        w for w in words
        if 205 <= w["x0"] <= 350
        and w["top"] > 90
        and abs(w["top"] - row_top) <= 8
    ]

    district_words = sorted(
        district_words,
        key=lambda x: x["x0"]
    )

    text = " ".join(
        w["text"]
        for w in district_words
    )

    return normalize_text(text)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("MALARIA STATE BLOCK ANALYSIS")
    print("=" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found:\n{PDF_PATH}"
        )

    print(f"\nPDF: {PDF_PATH}")

    all_rows = []
    all_markers = []

    with pdfplumber.open(PDF_PATH) as pdf:

        total_pages = len(pdf.pages)

        print(f"Total pages: {total_pages}")
        print("\nScanning all pages...\n")

        for page_number, page in enumerate(pdf.pages, start=1):

            page_height = float(page.height)

            row_anchors = extract_row_anchors(page)
            state_markers = extract_state_markers(page)

            # Store rows globally.
            for local_index, row in enumerate(
                row_anchors,
                start=1
            ):

                global_y = (
                    (page_number - 1) * page_height
                    + row["top"]
                )

                district = get_district_near_row(
                    page,
                    row["top"]
                )

                all_rows.append({
                    "page": page_number,
                    "local_index": local_index,
                    "top": row["top"],
                    "global_y": global_y,
                    "district": district,
                    "values": row["values"],
                })

            # Store state markers globally.
            for marker in state_markers:

                global_y = (
                    (page_number - 1) * page_height
                    + marker["top"]
                )

                all_markers.append({
                    "page": page_number,
                    "top": marker["top"],
                    "global_y": global_y,
                    "state": marker["state"],
                    "text": marker["text"],
                })

            if page_number % 25 == 0:

                print(
                    f"Processed pages: "
                    f"{page_number}/{total_pages}"
                )

    # ========================================================
    # SORT
    # ========================================================

    all_rows.sort(key=lambda x: x["global_y"])
    all_markers.sort(key=lambda x: x["global_y"])

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    print(f"Total district row anchors : {len(all_rows)}")
    print(f"Total state markers        : {len(all_markers)}")

    state_counts = Counter(
        marker["state"]
        for marker in all_markers
    )

    print(
        f"Unique states detected     : "
        f"{len(state_counts)}"
    )

    duplicates = {
        state: count
        for state, count in state_counts.items()
        if count > 1
    }

    print(
        f"States detected >1 time    : "
        f"{len(duplicates)}"
    )

    if duplicates:

        print("\nDuplicate state markers:")

        for state, count in duplicates.items():
            print(f"  {state}: {count}")

    # ========================================================
    # STATE MARKER → ROW POSITION
    # ========================================================

    print("\n" + "=" * 70)
    print("STATE MARKER POSITIONS")
    print("=" * 70)

    for marker_number, marker in enumerate(
        all_markers,
        start=1
    ):

        # Find nearest row.
        nearest_index = min(
            range(len(all_rows)),
            key=lambda i: abs(
                all_rows[i]["global_y"]
                - marker["global_y"]
            )
        )

        nearest_row = all_rows[nearest_index]

        # Previous and next row.
        previous_row = (
            all_rows[nearest_index - 1]
            if nearest_index > 0
            else None
        )

        next_row = (
            all_rows[nearest_index + 1]
            if nearest_index + 1 < len(all_rows)
            else None
        )

        distance = abs(
            nearest_row["global_y"]
            - marker["global_y"]
        )

        print(
            f"\n[{marker_number:02d}] "
            f"{marker['state']}"
        )

        print(
            f"    Marker page/top : "
            f"{marker['page']} / {marker['top']:.2f}"
        )

        print(
            f"    Nearest row     : "
            f"global #{nearest_index + 1}"
        )

        print(
            f"    Row page/index  : "
            f"{nearest_row['page']} / "
            f"{nearest_row['local_index']}"
        )

        print(
            f"    Row top         : "
            f"{nearest_row['top']:.2f}"
        )

        print(
            f"    Distance        : "
            f"{distance:.2f} pt"
        )

        print(
            f"    District near marker: "
            f"{nearest_row['district']}"
        )

        if previous_row:

            print(
                f"    Previous row    : "
                f"{previous_row['district']}"
            )

        if next_row:

            print(
                f"    Next row        : "
                f"{next_row['district']}"
            )

    # ========================================================
    # STATE ORDER
    # ========================================================

    print("\n" + "=" * 70)
    print("STATE ORDER DETECTED IN PDF")
    print("=" * 70)

    for i, marker in enumerate(all_markers, start=1):

        print(
            f"{i:02d}. "
            f"{marker['state']:<35} "
            f"(page {marker['page']})"
        )

    # ========================================================
    # PAGE ROW COUNTS
    # ========================================================

    page_row_counts = Counter(
        row["page"]
        for row in all_rows
    )

    print("\n" + "=" * 70)
    print("ROW COUNT SUMMARY")
    print("=" * 70)

    print(
        f"Minimum rows/page : "
        f"{min(page_row_counts.values())}"
    )

    print(
        f"Maximum rows/page : "
        f"{max(page_row_counts.values())}"
    )

    print(
        f"Average rows/page : "
        f"{sum(page_row_counts.values()) / len(page_row_counts):.2f}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nThe state marker is expected to represent the CENTER"
        "\nof a merged state cell, not the beginning of the state."
    )

    print(
        "\nDo NOT use this output as the final malaria dataset."
    )

    print(
        "This analysis is used to derive the correct state-block"
        "\nboundaries before production extraction."
    )


if __name__ == "__main__":
    main()