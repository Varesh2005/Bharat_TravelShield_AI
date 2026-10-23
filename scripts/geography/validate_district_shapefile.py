from pathlib import Path
import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHAPEFILE = PROJECT_ROOT / "data" / "raw" / "2011_Dist.shp"

print("=" * 70)
print("CENSUS 2011 DISTRICT SHAPEFILE VALIDATION")
print("=" * 70)

gdf = gpd.read_file(SHAPEFILE)

print(f"\nTotal features: {len(gdf)}")

# ------------------------------------------------------------
# 1. Check duplicate census codes
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. DUPLICATE CENSUS CODES")
print("-" * 70)

duplicate_census = gdf[
    gdf["censuscode"].duplicated(keep=False)
].sort_values("censuscode")

if duplicate_census.empty:
    print("No duplicate census codes found.")
else:
    print(f"Duplicate records found: {len(duplicate_census)}")
    print(
        duplicate_census[
            ["DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD", "censuscode"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# 2. Check duplicate State + District codes
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. DUPLICATE STATE + DISTRICT CODES")
print("-" * 70)

duplicate_codes = gdf[
    gdf.duplicated(
        subset=["ST_CEN_CD", "DT_CEN_CD"],
        keep=False
    )
].sort_values(["ST_CEN_CD", "DT_CEN_CD"])

if duplicate_codes.empty:
    print("No duplicate State + District code combinations found.")
else:
    print(f"Duplicate combinations found: {len(duplicate_codes)}")
    print(
        duplicate_codes[
            ["DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD", "censuscode"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# 3. Check duplicate district names within same state
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. DUPLICATE DISTRICT NAMES WITHIN STATE")
print("-" * 70)

duplicate_names = gdf[
    gdf.duplicated(
        subset=["ST_NM", "DISTRICT"],
        keep=False
    )
].sort_values(["ST_NM", "DISTRICT"])

if duplicate_names.empty:
    print("No duplicate district names within states.")
else:
    print(f"Duplicate name combinations found: {len(duplicate_names)}")
    print(
        duplicate_names[
            ["DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD", "censuscode"]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# 4. Check unique census codes
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("4. UNIQUE COUNTS")
print("-" * 70)

print(f"Total features              : {len(gdf)}")
print(f"Unique census codes        : {gdf['censuscode'].nunique()}")
print(
    f"Unique state+district codes: "
    f"{gdf[['ST_CEN_CD', 'DT_CEN_CD']].drop_duplicates().shape[0]}"
)

# ------------------------------------------------------------
# 5. Check census population reference
# ------------------------------------------------------------

POP_FILE = PROJECT_ROOT / "data" / "reference" / "district_population_2011.csv"

if POP_FILE.exists():

    pop = pd.read_csv(POP_FILE)

    print("\n" + "-" * 70)
    print("5. COMPARISON WITH CENSUS POPULATION DATA")
    print("-" * 70)

    print(f"Population reference districts: {len(pop)}")

    print(
        f"Shapefile districts           : {len(gdf)}"
    )

    # Compare state + district codes
    shape_keys = set(
        zip(
            gdf["ST_CEN_CD"].astype(str),
            gdf["DT_CEN_CD"].astype(str)
        )
    )

    pop_keys = set(
        zip(
            pop["census_state_code"].astype(str),
            pop["census_district_code"].astype(str)
        )
    )

    shape_not_population = shape_keys - pop_keys
    population_not_shape = pop_keys - shape_keys

    print(
        f"\nShapefile codes NOT in population reference: "
        f"{len(shape_not_population)}"
    )

    print(
        f"Population codes NOT in shapefile: "
        f"{len(population_not_shape)}"
    )

    if shape_not_population:
        print("\nShapefile-only codes:")
        for key in sorted(shape_not_population):
            print(f"  State={key[0]}, District={key[1]}")

    if population_not_shape:
        print("\nPopulation-only codes:")
        for key in sorted(population_not_shape):
            print(f"  State={key[0]}, District={key[1]}")

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)