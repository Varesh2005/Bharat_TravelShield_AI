"""
Bharat-TravelShield AI - Historical Surveillance & Climate API Routes
Author: Pranav (Database + Integration)
Project: Bharat-TravelShield AI
"""

from fastapi import APIRouter, HTTPException, Query
from database.repositories.history_repository import HistoryRepository
from database.repositories.location_repository import LocationRepository

router = APIRouter(prefix="/api/v1/locations", tags=["Historical Data"])


@router.get("/{location_id}/history")
def get_location_history(
    location_id: str,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Fetches historical weather observations and epidemiological surveillance records
    for a given district. Demonstrates clear separation between ground-truth history and ANN predictions.
    """
    loc_repo = LocationRepository()
    hist_repo = HistoryRepository()

    loc = loc_repo.get_by_id(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location '{location_id}' not found.")

    weather_records = hist_repo.get_weather_history(location_id, limit=limit)
    disease_records = hist_repo.get_disease_history(location_id, limit=limit)

    return {
        "location_id": location_id,
        "district_name": loc.get("district_name"),
        "state_name": loc.get("state_name"),
        "weather_observations": weather_records,
        "disease_surveillance": disease_records,
        "data_provenance": {
            "weather_source": "IMD (India Meteorological Department)",
            "disease_source": "NVBDCP / NCDC (National Vector Borne Disease Control Programme)",
        }
    }
