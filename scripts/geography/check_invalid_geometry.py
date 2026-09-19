from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2011_Dist.shp"
)


print("=" * 70)
print("INVALID DISTRICT GEOMETRY DIAGNOSTIC")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load shapefile
# ------------------------------------------------------------

print("\nLoading shapefile...")

gdf = gpd.read_file(SHAPEFILE)

print(f"Raw records: {len(gdf)}")


# ------------------------------------------------------------
# 2. Remove placeholder record
# ------------------------------------------------------------

gdf = gdf[gdf["censuscode"] != 0].copy()

print(f"District records: {len(gdf)}")


# ------------------------------------------------------------
# 3. Find invalid geometries
# ------------------------------------------------------------

invalid = gdf[~gdf.geometry.is_valid].copy()

print("\n" + "-" * 70)
print("INVALID GEOMETRIES")
print("-" * 70)

print(f"Invalid geometry count: {len(invalid)}")


# ------------------------------------------------------------
# 4. Print details
# ------------------------------------------------------------

for index, row in invalid.iterrows():

    print("\n" + "-" * 70)
    print("INVALID DISTRICT")
    print("-" * 70)

    print(f"Row index    : {index}")
    print(f"District     : {row['DISTRICT']}")
    print(f"State        : {row['ST_NM']}")
    print(f"State code   : {row['ST_CEN_CD']}")
    print(f"District code: {row['DT_CEN_CD']}")
    print(f"Census code  : {row['censuscode']}")

    print("\nGeometry type:")
    print(row.geometry.geom_type)

    print("\nValidity reason:")

    try:
        from shapely.validation import explain_validity

        reason = explain_validity(row.geometry)
        print(reason)

    except Exception as e:
        print(f"Could not determine reason: {e}")


# ------------------------------------------------------------
# 5. Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)