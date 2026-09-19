import requests
from bs4 import BeautifulSoup
from pathlib import Path

URL = "https://mausam.imd.gov.in/responsive/heatwave_guidance.php"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/disaster/heatwave/imd_heatwave_guidance.html"

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(
    URL,
    headers=headers,
    timeout=30
)

r.raise_for_status()

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(r.text, encoding="utf-8")

print("=" * 70)
print("IMD HEATWAVE SOURCE CHECK")
print("=" * 70)
print("HTTP status :", r.status_code)
print("HTML size   :", len(r.text))

soup = BeautifulSoup(r.text, "html.parser")

links = []

for a in soup.find_all("a", href=True):

    text = a.get_text(" ", strip=True)
    href = a["href"]

    if any(
        word in (text + " " + href).lower()
        for word in [
            "annual",
            "heat wave",
            "heatwave",
            "historical",
            "report"
        ]
    ):
        links.append((text, href))

print()
print("Relevant links found:")
print("-" * 70)

for text, href in links:
    print(f"{text} -> {href}")

print()
print("Saved:", OUT)