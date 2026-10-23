"""
Bharat-TravelShield AI - Locations API Routes
Author: Pranav (Database + Integration)
Project: Bharat-TravelShield AI
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from database.repositories.location_repository import LocationRepository
from backend.app.schemas.api_schemas import LocationSearchResponse, LocationDetail

router = APIRouter(prefix="/api/v1/locations", tags=["Locations"])


@router.get("", response_model=LocationSearchResponse)
def list_or_search_locations(
    q: Optional[str] = Query(None, description="Search query for district, state, or tourist spot"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Retrieves or searches Indian districts stored in Firebase Firestore.
    """
    repo = LocationRepository()
    if q and q.strip():
        items = repo.search(q.strip(), limit=limit)
    else:
        items = repo.list_all(limit=limit)

    return LocationSearchResponse(
        total=len(items),
        locations=[LocationDetail(**loc) for loc in items]
    )


@router.get("/{location_id}", response_model=LocationDetail)
def get_location_by_id(location_id: str):
    """
    Retrieves detailed metadata for a specific district by its unique ID.
    """
    repo = LocationRepository()
    loc = repo.get_by_id(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found in registry.")
    return LocationDetail(**loc)
