from pathlib import Path
import pdfplumber

PDF_PATH = Path(
    "data/raw/disease/malaria/ncvbdc_district_wise_malaria_2000_2024.pdf"
)

OUTPUT_DIR = Path("data/interim/malaria_page_images")


def main():
    print("=" * 70)
    print("MALARIA PDF PAGE RENDER TEST")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages_to_render = [1, 2, 18, 20, 31, 61, 100, 200, 276, 291, 306, 322, 323, 347]

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"Total pages: {len(pdf.pages)}")

        for page_number in pages_to_render:

            page = pdf.pages[page_number - 1]

            image = page.to_image(resolution=150)

            output_path = OUTPUT_DIR / f"page_{page_number:03d}.png"

            image.save(output_path)

            print(
                f"Rendered page {page_number:3d} -> "
                f"{output_path}"
            )

    print("\n" + "=" * 70)
    print("RENDER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()