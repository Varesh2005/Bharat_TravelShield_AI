"""
Bharat-TravelShield AI - History & Audit Trail Routes
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Provides historical weather observations, disease surveillance records,
and persistent prediction audit lookups from Pranav's Firestore database.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status

from database.repositories.history_repository import HistoryRepository
from database.repositories.prediction_repository import PredictionRepository

router = APIRouter(prefix="/api/v1", tags=["History & Audit"])


@router.get(
    "/history/weather/{location_id}",
    summary="Get historical weather records",
    description="Retrieves ground-truth meteorological observations for a destination district.",
)
def get_weather_history(
    location_id: str,
    limit: int = Query(30, ge=1, le=100, description="Maximum number of historical observations"),
):
    repo = HistoryRepository()
    records = repo.get_weather_history(location_id=location_id, limit=limit)
    return {
        "location_id": location_id,
        "count": len(records),
        "records": records,
    }


@router.get(
    "/history/disease/{location_id}",
    summary="Get disease surveillance history",
    description="Retrieves epidemiological surveillance reports (Dengue/Malaria) for a destination district.",
)
def get_disease_history(
    location_id: str,
    limit: int = Query(30, ge=1, le=100, description="Maximum number of reports"),
):
    repo = HistoryRepository()
    records = repo.get_disease_history(location_id=location_id, limit=limit)
    return {
        "location_id": location_id,
        "count": len(records),
        "records": records,
    }


@router.get(
    "/predictions/recent",
    summary="Get recent predictions audit log",
    description="Fetches recent multi-hazard prediction inference logs across all destinations.",
)
def get_recent_predictions(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of recent records"),
):
    repo = PredictionRepository()
    records = repo.get_recent(limit=limit)
    return {
        "count": len(records),
        "predictions": records,
    }


@router.get(
    "/predictions/{prediction_id}",
    summary="Get prediction by ID",
    description="Retrieves a specific prediction audit record by its unique ID (e.g. 'pred_a1b2c3d4e5f6').",
)
def get_prediction_by_id(prediction_id: str):
    repo = PredictionRepository()
    record = repo.get_by_id(prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record '{prediction_id}' not found.",
        )
    return record


@router.get(
    "/predictions/location/{location_id}",
    summary="Get predictions for a specific location",
    description="Retrieves recent prediction audit logs for a specific district.",
)
def get_predictions_by_location(
    location_id: str,
    limit: int = Query(10, ge=1, le=50),
):
    repo = PredictionRepository()
    records = repo.get_by_location(location_id=location_id, limit=limit)
    return {
        "location_id": location_id,
        "count": len(records),
        "predictions": records,
    }
