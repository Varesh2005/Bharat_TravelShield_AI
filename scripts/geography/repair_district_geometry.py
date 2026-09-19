from pathlib import Path

import geopandas as gpd
from shapely.validation import explain_validity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2011_Dist.shp"
)


print("=" * 70)
print("DISTRICT GEOMETRY REPAIR")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load raw shapefile
# ------------------------------------------------------------

print("\n1. Loading raw shapefile...")

gdf = gpd.read_file(SHAPEFILE)

print(f"Raw records: {len(gdf)}")


# ------------------------------------------------------------
# 2. Remove placeholder record
# ------------------------------------------------------------

gdf = gdf[gdf["censuscode"] != 0].copy()

print(f"District records: {len(gdf)}")


# ------------------------------------------------------------
# 3. Identify Cuddalore
# ------------------------------------------------------------

target_mask = gdf["censuscode"] == 617

if target_mask.sum() != 1:
    raise ValueError(
        "Expected exactly one district with censuscode 617."
    )

target_index = gdf.index[target_mask][0]

target = gdf.loc[target_index]

print("\nTarget district:")
print(f"  District    : {target['DISTRICT']}")
print(f"  State       : {target['ST_NM']}")
print(f"  Census code : {target['censuscode']}")
print(f"  Geometry    : {target.geometry.geom_type}")

print("\nOriginal validity:")
print(f"  Valid: {target.geometry.is_valid}")
print(
    f"  Reason: {explain_validity(target.geometry)}"
)


# ------------------------------------------------------------
# 4. Repair ONLY Cuddalore
# ------------------------------------------------------------

print("\n2. Repairing Cuddalore geometry...")

original_geometry = target.geometry

# make_valid() repairs invalid polygon topology while
# preserving the geometry as much as possible.

try:
    repaired_geometry = original_geometry.make_valid()
except AttributeError:
    from shapely.validation import make_valid
    repaired_geometry = make_valid(original_geometry)


print("\nRepaired geometry:")
print(f"  Type : {repaired_geometry.geom_type}")
print(f"  Valid: {repaired_geometry.is_valid}")

if not repaired_geometry.is_valid:
    raise ValueError(
        "Cuddalore geometry is still invalid after repair."
    )


# ------------------------------------------------------------
# 5. Check geometry type
# ------------------------------------------------------------

if repaired_geometry.geom_type not in [
    "Polygon",
    "MultiPolygon"
]:
    raise ValueError(
        "Repair produced an unexpected geometry type: "
        f"{repaired_geometry.geom_type}"
    )


# ------------------------------------------------------------
# 6. Replace only target geometry
# ------------------------------------------------------------

gdf.loc[target_index, "geometry"] = repaired_geometry


# ------------------------------------------------------------
# 7. Validate ALL geometries
# ------------------------------------------------------------

print("\n3. Validating all 640 district geometries...")

invalid_count = (~gdf.geometry.is_valid).sum()

print(f"Invalid geometries after repair: {invalid_count}")

if invalid_count != 0:

    invalid_rows = gdf[
        ~gdf.geometry.is_valid
    ]

    print("\nRemaining invalid districts:")

    print(
        invalid_rows[
            [
                "DISTRICT",
                "ST_NM",
                "ST_CEN_CD",
                "DT_CEN_CD",
                "censuscode"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "Some district geometries remain invalid."
    )


# ------------------------------------------------------------
# 8. Confirm district count
# ------------------------------------------------------------

print("\n4. Final district count...")

if len(gdf) != 640:
    raise ValueError(
        f"Expected 640 districts, found {len(gdf)}."
    )

print("640 districts confirmed.")


# ------------------------------------------------------------
# 9. Save repaired shapefile
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "geography"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "2011_Dist_repaired.shp"

print("\n5. Saving repaired shapefile...")

gdf.to_file(
    OUTPUT_FILE,
    driver="ESRI Shapefile",
    encoding="UTF-8"
)

print(f"Saved to:")
print(OUTPUT_FILE)


# ------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GEOMETRY REPAIR COMPLETE")
print("=" * 70)

print("\nRaw source:")
print(SHAPEFILE)

print("\nRepaired working copy:")
print(OUTPUT_FILE)

print("\nDistricts:")
print(len(gdf))

print("\nInvalid geometries:")
print((~gdf.geometry.is_valid).sum())

print("\nCuddalore validity:")
print(
    gdf.loc[
        gdf["censuscode"] == 617,
        "geometry"
    ].iloc[0].is_valid
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)