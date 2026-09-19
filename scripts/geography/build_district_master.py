from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use the validated/repaired working shapefile.
# The original raw shapefile remains untouched.
SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "geography"
    / "2011_Dist_repaired.shp"
)

# Validated Census 2011 district population reference.
POPULATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_population_2011.csv"
)

# Final geographic master.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "district_master.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BUILDING CENSUS 2011 DISTRICT MASTER")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n1. Checking input files...")

if not SHAPEFILE.exists():
    raise FileNotFoundError(
        f"Repaired shapefile not found:\n{SHAPEFILE}"
    )

if not POPULATION_FILE.exists():
    raise FileNotFoundError(
        f"Population reference not found:\n{POPULATION_FILE}"
    )

print("Repaired shapefile found.")
print("Population reference found.")


# ============================================================
# 2. LOAD REPAIRED SHAPEFILE
# ============================================================

print("\n2. Loading repaired district shapefile...")

gdf = gpd.read_file(SHAPEFILE)

print(f"Records loaded: {len(gdf)}")
print(f"CRS: {gdf.crs}")


# ============================================================
# 3. VALIDATE CRS
# ============================================================

print("\n3. Validating coordinate reference system...")

if gdf.crs is None:
    raise ValueError(
        "Shapefile has no CRS information."
    )

if gdf.crs.to_epsg() != 4326:
    raise ValueError(
        f"Expected EPSG:4326, found {gdf.crs}."
    )

print("CRS confirmed: EPSG:4326")


# ============================================================
# 4. VALIDATE DISTRICT COUNT
# ============================================================

print("\n4. Validating district count...")

if len(gdf) != 640:
    raise ValueError(
        f"Expected 640 districts, but found {len(gdf)}."
    )

print("640 districts confirmed.")


# ============================================================
# 5. VALIDATE CENSUS CODES
# ============================================================

print("\n5. Validating Census district codes...")

gdf["censuscode"] = pd.to_numeric(
    gdf["censuscode"],
    errors="raise"
).astype(int)

if (gdf["censuscode"] == 0).any():
    raise ValueError(
        "Placeholder censuscode=0 still exists."
    )

if gdf["censuscode"].nunique() != 640:
    raise ValueError(
        "censuscode is not unique across 640 districts."
    )

print("All 640 census codes are unique.")
print("No placeholder census code found.")


# ============================================================
# 6. VALIDATE GEOMETRIES
# ============================================================

print("\n6. Validating district geometries...")

missing_geometry = gdf.geometry.isna().sum()
invalid_geometry = (~gdf.geometry.is_valid).sum()

print(f"Missing geometries: {missing_geometry}")
print(f"Invalid geometries: {invalid_geometry}")

if missing_geometry != 0:
    raise ValueError(
        "Missing district geometries detected."
    )

if invalid_geometry != 0:
    raise ValueError(
        "Invalid district geometries detected."
    )

print("All 640 district geometries are valid.")


# ============================================================
# 7. CREATE REPRESENTATIVE POINTS
# ============================================================

print("\n7. Creating district representative points...")

# representative_point() is intentionally used instead of
# centroid because it guarantees the generated point lies
# within the district geometry.

representative_points = gdf.geometry.representative_point()

gdf["longitude"] = representative_points.x
gdf["latitude"] = representative_points.y

print("Representative points generated.")


# ============================================================
# 8. VALIDATE COORDINATES
# ============================================================

print("\n8. Validating latitude and longitude...")

if gdf["latitude"].isna().any():
    raise ValueError(
        "Missing latitude values detected."
    )

if gdf["longitude"].isna().any():
    raise ValueError(
        "Missing longitude values detected."
    )

invalid_latitude = ~gdf["latitude"].between(-90, 90)
invalid_longitude = ~gdf["longitude"].between(-180, 180)

if invalid_latitude.any():
    raise ValueError(
        "Invalid latitude values detected."
    )

if invalid_longitude.any():
    raise ValueError(
        "Invalid longitude values detected."
    )

print("All latitude values are valid.")
print("All longitude values are valid.")


# ============================================================
# 9. LOAD POPULATION DATA
# ============================================================

print("\n9. Loading Census 2011 population reference...")

population = pd.read_csv(POPULATION_FILE)

print(f"Population records: {len(population)}")

if len(population) != 640:
    raise ValueError(
        f"Expected 640 population records, "
        f"but found {len(population)}."
    )


# ============================================================
# 10. VALIDATE POPULATION COLUMNS
# ============================================================

print("\n10. Validating population schema...")

required_population_columns = [
    "census_state_code",
    "census_district_code",
    "district_name",
    "population_2011",
    "households_2011",
]

missing_population_columns = [
    column
    for column in required_population_columns
    if column not in population.columns
]

if missing_population_columns:
    raise ValueError(
        "Missing population columns: "
        + ", ".join(missing_population_columns)
    )

print("Population schema confirmed.")


# ============================================================
# 11. PREPARE DISTRICT IDENTIFIERS
# ============================================================

print("\n11. Preparing district identifiers...")

population["census_district_code"] = pd.to_numeric(
    population["census_district_code"],
    errors="raise"
).astype(int)

population["census_state_code"] = pd.to_numeric(
    population["census_state_code"],
    errors="raise"
).astype(int)

print("District identifiers prepared.")


# ============================================================
# 12. VALIDATE POPULATION DISTRICT CODES
# ============================================================

print("\n12. Validating population district codes...")

if population["census_district_code"].nunique() != 640:
    raise ValueError(
        "Population reference does not contain "
        "640 unique district codes."
    )

print("Population reference contains 640 unique district codes.")


# ============================================================
# 13. VALIDATE CODE COVERAGE
# ============================================================

print("\n13. Validating geographic-population code coverage...")

shape_codes = set(
    gdf["censuscode"]
)

population_codes = set(
    population["census_district_code"]
)

common_codes = shape_codes & population_codes

shape_only = shape_codes - population_codes
population_only = population_codes - shape_codes

print(f"Shapefile unique codes : {len(shape_codes)}")
print(f"Population unique codes: {len(population_codes)}")
print(f"Common codes           : {len(common_codes)}")

if shape_only:
    raise ValueError(
        "District codes present in shapefile but absent "
        f"from population reference: {sorted(shape_only)}"
    )

if population_only:
    raise ValueError(
        "District codes present in population reference but "
        f"absent from shapefile: {sorted(population_only)}"
    )

if len(common_codes) != 640:
    raise ValueError(
        "Expected 640 common district codes."
    )

print("All 640 district codes match.")


# ============================================================
# 14. SELECT GEOGRAPHIC ATTRIBUTES
# ============================================================

print("\n14. Preparing geographic attributes...")

geography = gdf[
    [
        "censuscode",
        "DISTRICT",
        "ST_NM",
        "latitude",
        "longitude",
    ]
].copy()

geography = geography.rename(
    columns={
        "censuscode": "censuscode_2011",
        "DISTRICT": "district_name_shapefile",
        "ST_NM": "state_name",
    }
)

print("Geographic attributes prepared.")


# ============================================================
# 15. SELECT POPULATION ATTRIBUTES
# ============================================================

print("\n15. Preparing population attributes...")

population_subset = population[
    [
        "census_state_code",
        "census_district_code",
        "district_name",
        "population_2011",
        "households_2011",
    ]
].copy()

print("Population attributes prepared.")


# ============================================================
# 16. MERGE GEOGRAPHY + POPULATION
# ============================================================

print("\n16. Combining geography and population...")

master = geography.merge(
    population_subset,
    left_on="censuscode_2011",
    right_on="census_district_code",
    how="left",
    validate="one_to_one",
)

print(f"Records after merge: {len(master)}")


# ============================================================
# 17. VALIDATE MERGE
# ============================================================

print("\n17. Validating final merge...")

if len(master) != 640:
    raise ValueError(
        f"Expected 640 final records, "
        f"but found {len(master)}."
    )

missing_population = master[
    "population_2011"
].isna().sum()

missing_households = master[
    "households_2011"
].isna().sum()

missing_state_codes = master[
    "census_state_code"
].isna().sum()

missing_district_codes = master[
    "census_district_code"
].isna().sum()

print(
    f"Missing population values : {missing_population}"
)

print(
    f"Missing household values  : {missing_households}"
)

print(
    f"Missing state codes       : {missing_state_codes}"
)

print(
    f"Missing district codes    : {missing_district_codes}"
)

if missing_population != 0:
    raise ValueError(
        "Population join contains missing values."
    )

if missing_households != 0:
    raise ValueError(
        "Household join contains missing values."
    )

if missing_state_codes != 0:
    raise ValueError(
        "State code join contains missing values."
    )

if missing_district_codes != 0:
    raise ValueError(
        "District code join contains missing values."
    )

print("Population merge validated successfully.")


# ============================================================
# 18. VALIDATE FINAL DISTRICT IDENTIFIERS
# ============================================================

print("\n18. Validating final district identifiers...")

if master["censuscode_2011"].nunique() != 640:
    raise ValueError(
        "Final censuscode_2011 is not unique."
    )

if master["census_district_code"].nunique() != 640:
    raise ValueError(
        "Final census_district_code is not unique."
    )

print("640 unique district identifiers confirmed.")


# ============================================================
# 19. VALIDATE POPULATION VALUES
# ============================================================

print("\n19. Validating population values...")

if master["population_2011"].le(0).any():
    raise ValueError(
        "Non-positive population values detected."
    )

if master["households_2011"].le(0).any():
    raise ValueError(
        "Non-positive household values detected."
    )

print("Population values are valid.")
print("Household values are valid.")


# ============================================================
# 20. ADD COORDINATE PROVENANCE
# ============================================================

print("\n20. Adding coordinate provenance...")

master["coordinate_method"] = (
    "district_representative_point"
)

master["coordinate_source"] = (
    "Census 2011 district shapefile"
)

print("Coordinate provenance added.")


# ============================================================
# 21. FINAL COLUMN ORDER
# ============================================================

print("\n21. Applying final schema...")

final_columns = [
    "census_state_code",
    "census_district_code",
    "censuscode_2011",
    "district_name",
    "district_name_shapefile",
    "state_name",
    "population_2011",
    "households_2011",
    "latitude",
    "longitude",
    "coordinate_method",
    "coordinate_source",
]

master = master[final_columns]

print(f"Final columns: {len(master.columns)}")


# ============================================================
# 22. FINAL VALIDATION
# ============================================================

print("\n22. Running final validation...")

# Row count
if len(master) != 640:
    raise ValueError(
        "Final dataset must contain exactly 640 rows."
    )

# Required columns
for column in final_columns:
    if column not in master.columns:
        raise ValueError(
            f"Missing final column: {column}"
        )

# Missing values in critical fields
critical_columns = [
    "census_state_code",
    "census_district_code",
    "censuscode_2011",
    "district_name",
    "state_name",
    "population_2011",
    "households_2011",
    "latitude",
    "longitude",
]

missing_critical = master[
    critical_columns
].isna().sum()

if missing_critical.sum() != 0:
    raise ValueError(
        "Missing critical values detected:\n"
        + missing_critical[
            missing_critical > 0
        ].to_string()
    )

# Coordinate ranges
if not master["latitude"].between(-90, 90).all():
    raise ValueError(
        "Invalid latitude detected in final dataset."
    )

if not master["longitude"].between(-180, 180).all():
    raise ValueError(
        "Invalid longitude detected in final dataset."
    )

print("Final validation passed.")


# ============================================================
# 23. SAVE FINAL CSV
# ============================================================

print("\n23. Saving district master CSV...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

master.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nOutput saved to:")
print(OUTPUT_FILE)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DISTRICT MASTER BUILD COMPLETE")
print("=" * 70)

print(f"\nFinal districts : {len(master)}")
print(f"Final columns   : {len(master.columns)}")

print("\nFinal schema:")

for column in master.columns:
    print(f"  {column}")

print("\nFirst 10 records:")

print(
    master.head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)