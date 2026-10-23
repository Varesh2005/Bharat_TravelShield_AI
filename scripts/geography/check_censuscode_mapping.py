from pathlib import Path
import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPEFILE = PROJECT_ROOT / "data" / "raw" / "2011_Dist.shp"
POP_FILE = PROJECT_ROOT / "data" / "reference" / "district_population_2011.csv"

print("=" * 70)
print("CENSUS CODE MAPPING CHECK")
print("=" * 70)

# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

gdf = gpd.read_file(SHAPEFILE)
pop = pd.read_csv(POP_FILE)

print(f"\nShapefile records : {len(gdf)}")
print(f"Population records: {len(pop)}")

# ------------------------------------------------------------
# Show shapefile code structure
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. SHAPEFILE CODE SAMPLE")
print("-" * 70)

print(
    gdf[
        [
            "DISTRICT",
            "ST_NM",
            "ST_CEN_CD",
            "DT_CEN_CD",
            "censuscode"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Check censuscode uniqueness
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. CENSUSCODE UNIQUENESS")
print("-" * 70)

print(f"Total shapefile records : {len(gdf)}")
print(f"Unique censuscode values: {gdf['censuscode'].nunique()}")

duplicates = gdf[
    gdf["censuscode"].duplicated(keep=False)
].sort_values("censuscode")

if duplicates.empty:
    print("Duplicate censuscode values: NONE")
else:
    print("\nDuplicate censuscode records:")
    print(
        duplicates[
            [
                "DISTRICT",
                "ST_NM",
                "ST_CEN_CD",
                "DT_CEN_CD",
                "censuscode"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# Check special records
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. SPECIAL / POSSIBLY EXTRA RECORDS")
print("-" * 70)

special = gdf[
    (gdf["ST_CEN_CD"] == 99) |
    (gdf["DT_CEN_CD"] == 99) |
    (gdf["censuscode"] == 999)
]

if special.empty:
    print("No obvious special records found.")
else:
    print(
        special[
            [
                "DISTRICT",
                "ST_NM",
                "ST_CEN_CD",
                "DT_CEN_CD",
                "censuscode"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# Population code range
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("4. POPULATION DATA CODE RANGE")
print("-" * 70)

print(
    f"State codes: "
    f"{pop['census_state_code'].min()} "
    f"to "
    f"{pop['census_state_code'].max()}"
)

print(
    f"District codes: "
    f"{pop['census_district_code'].min()} "
    f"to "
    f"{pop['census_district_code'].max()}"
)

print("\nFirst 20 population records:")

print(
    pop[
        [
            "census_state_code",
            "census_district_code",
            "district_name"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Check whether shapefile censuscode corresponds to
# population district code
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("5. CENSUSCODE vs POPULATION DISTRICT CODE")
print("-" * 70)

shape_codes = set(
    pd.to_numeric(
        gdf["censuscode"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
)

pop_district_codes = set(
    pd.to_numeric(
        pop["census_district_code"],
        errors="coerce"
    )
    .dropna()
    .astype(int)
)

intersection = shape_codes & pop_district_codes

print(f"Unique shapefile censuscodes       : {len(shape_codes)}")
print(f"Unique population district codes  : {len(pop_district_codes)}")
print(f"Common numeric values              : {len(intersection)}")

# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MAPPING CHECK COMPLETE")
print("=" * 70)
