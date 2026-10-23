"""
Bharat-TravelShield AI - Destination & Location Routes
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Provides autocomplete search, location details, and comprehensive
historical destination data summaries for Rosemariya's Next.js frontend.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status

from database.repositories.location_repository import LocationRepository
from ml.destination_analysis import safe_destination_summary
from backend.app.schemas.api_schemas import (
    LocationItem,
    LocationSearchResponse,
)

router = APIRouter(prefix="/api/v1/destinations", tags=["Destinations"])

# Common historical and contemporary city/alias mappings in India
CITY_ALIASES = {
    "bengaluru": ["bangalore"],
    "bangalore": ["bengaluru"],
    "bombay": ["mumbai"],
    "calcutta": ["kolkata"],
    "madras": ["chennai"],
    "cochin": ["kochi", "ernakulam"],
    "ernakulam": ["kochi"],
    "benaras": ["varanasi"],
    "kashi": ["varanasi"],
    "prayagraj": ["allahabad"],
    "allahabad": ["prayagraj"],
    "pondicherry": ["puducherry"],
    "puducherry": ["pondicherry"],
    "gurgaon": ["gurugram"],
    "gurugram": ["gurgaon"],
    "gauhati": ["guwahati", "kamrup"],
    "guwahati": ["kamrup"],
    "mysuru": ["mysore"],
    "mysore": ["mysuru"],
    "trivandrum": ["thiruvananthapuram"],
    "thiruvananthapuram": ["trivandrum"],
    "calangute": ["goa", "north goa"],
    "panaji": ["goa", "north goa"],
    "margao": ["goa", "south goa"],
}


@router.get(
    "/search",
    response_model=LocationSearchResponse,
    summary="Search Indian destinations",
    description="Case-insensitive search across district names, state names, and tourist spots with alias and fuzzy fallback.",
)
def search_destinations(
    q: str = Query(..., min_length=1, description="Search query string (e.g. 'Jaipur', 'Goa', 'Bengaluru')"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results to return"),
):
    repo = LocationRepository()
    results = repo.search(query=q, limit=limit)

    # If few results, search common synonyms/aliases (e.g. Bengaluru -> Bangalore)
    if len(results) < limit:
        q_lower = q.strip().lower()
        aliases = CITY_ALIASES.get(q_lower, [])
        seen_ids = {r.get("id") for r in results}
        for alias in aliases:
            alias_results = repo.search(query=alias, limit=limit - len(results))
            for ar in alias_results:
                if ar.get("id") not in seen_ids:
                    results.append(ar)
                    seen_ids.add(ar.get("id"))
                if len(results) >= limit:
                    break

    # Format into response schema
    items = []
    for r in results:
        items.append(
            LocationItem(
                id=r.get("id", ""),
                census_district_code=r.get("census_district_code"),
                district_name=r.get("district_name", ""),
                state_name=r.get("state_name", ""),
                latitude=float(r.get("latitude", 20.0)),
                longitude=float(r.get("longitude", 78.0)),
                elevation_m=float(r["elevation_m"]) if r.get("elevation_m") is not None else None,
                climate_zone=r.get("climate_zone"),
                population_density=float(r["population_density"]) if r.get("population_density") is not None else None,
                popular_tourist_spots=r.get("popular_tourist_spots", []),
            )
        )

    return LocationSearchResponse(query=q, count=len(items), results=items)


@router.get(
    "",
    response_model=List[LocationItem],
    summary="List registered districts",
    description="Lists canonical Indian districts stored in the database.",
)
def list_destinations(
    limit: int = Query(50, ge=1, le=640, description="Maximum number of districts to list"),
):
    repo = LocationRepository()
    records = repo.list_all(limit=limit)
    items = []
    for r in records:
        items.append(
            LocationItem(
                id=r.get("id", ""),
                census_district_code=r.get("census_district_code"),
                district_name=r.get("district_name", ""),
                state_name=r.get("state_name", ""),
                latitude=float(r.get("latitude", 20.0)),
                longitude=float(r.get("longitude", 78.0)),
                elevation_m=float(r["elevation_m"]) if r.get("elevation_m") is not None else None,
                climate_zone=r.get("climate_zone"),
                population_density=float(r["population_density"]) if r.get("population_density") is not None else None,
                popular_tourist_spots=r.get("popular_tourist_spots", []),
            )
        )
    return items


@router.get(
    "/{location_id}",
    response_model=LocationItem,
    summary="Get location details by ID or code",
    description="Retrieves a single location by its document ID (e.g. 'delhi_central') or Census district code (e.g. '536').",
)
def get_destination_by_id(location_id: str):
    repo = LocationRepository()
    record = repo.get_by_id(location_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Destination '{location_id}' not found in canonical directory.",
        )

    return LocationItem(
        id=record.get("id", location_id),
        census_district_code=record.get("census_district_code"),
        district_name=record.get("district_name", ""),
        state_name=record.get("state_name", ""),
        latitude=float(record.get("latitude", 20.0)),
        longitude=float(record.get("longitude", 78.0)),
        elevation_m=float(record["elevation_m"]) if record.get("elevation_m") is not None else None,
        climate_zone=record.get("climate_zone"),
        population_density=float(record["population_density"]) if record.get("population_density") is not None else None,
        popular_tourist_spots=record.get("popular_tourist_spots", []),
    )


@router.get(
    "/district/{district_code}/summary",
    summary="Get destination historical data summary",
    description="Combines historical weather, heatwave thresholds, malaria surveillance, dengue context, and flood records for a destination district.",
)
def get_destination_summary(
    district_code: int,
    start_date: Optional[str] = Query(None, description="Optional start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Optional end date (YYYY-MM-DD)"),
    include_ann: bool = Query(False, description="Whether to include experimental hazard ANN outputs"),
    ann_feature_year: Optional[int] = Query(None, description="Optional feature year for ANN inference"),
):
    summary = safe_destination_summary(
        district_code=district_code,
        start_date=start_date,
        end_date=end_date,
        include_experimental_ann=include_ann,
        ann_feature_year=ann_feature_year,
    )

    if "error" in summary:
        err = summary["error"]
        err_code = err.get("code")
        if err_code == "destination_not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err)
        elif err_code in ("invalid_request", "ambiguous_destination"):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)
        else:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=err)

    return summary
