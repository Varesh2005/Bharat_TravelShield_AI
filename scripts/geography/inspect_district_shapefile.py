from pathlib import Path
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPEFILE = PROJECT_ROOT / "data" / "raw" / "2011_Dist.shp"


print("=" * 70)
print("CENSUS 2011 DISTRICT SHAPEFILE INSPECTION")
print("=" * 70)

print(f"\nInput file:")
print(SHAPEFILE)

if not SHAPEFILE.exists():
    raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE}")

# Load shapefile
gdf = gpd.read_file(SHAPEFILE)

print("\n✅ Shapefile loaded successfully.")

# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("1. BASIC INFORMATION")
print("-" * 70)

print(f"Number of rows/features : {len(gdf)}")
print(f"Number of columns       : {len(gdf.columns)}")
print(f"CRS                     : {gdf.crs}")

# ---------------------------------------------------------
# COLUMNS
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("2. COLUMNS")
print("-" * 70)

for column in gdf.columns:
    print(f"  {column}")

# ---------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("3. GEOMETRY")
print("-" * 70)

print(gdf.geometry.geom_type.value_counts().to_string())

print(f"\nMissing geometries: {gdf.geometry.isna().sum()}")

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("4. MISSING VALUES")
print("-" * 70)

missing = gdf.isna().sum()

for column, count in missing.items():
    if count > 0:
        print(f"  {column}: {count}")

if missing.sum() == 0:
    print("  None")

# ---------------------------------------------------------
# SAMPLE ATTRIBUTES
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("5. FIRST 10 RECORDS — ATTRIBUTES ONLY")
print("-" * 70)

# Don't print geometry because it is huge
attribute_columns = [
    column for column in gdf.columns
    if column != gdf.geometry.name
]

print(
    gdf[attribute_columns]
    .head(10)
    .to_string(index=False)
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)