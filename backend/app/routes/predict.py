"""
Bharat-TravelShield AI - Real-Time Multi-Hazard Prediction API Routes
Author: Pranav (Database + Integration) & Varesh (ML)
Project: Bharat-TravelShield AI
"""

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from database.repositories.location_repository import LocationRepository
from database.repositories.prediction_repository import PredictionRepository
from database.schema import create_prediction_doc
from ml.inference.predict import get_predictor, SpatiotemporalInput
from backend.app.schemas.api_schemas import (
    PredictionRequest,
    PredictionResponse,
    MultiHazardRisks,
    HazardDetail,
)

router = APIRouter(prefix="/api/v1", tags=["Prediction & Risk Intelligence"])


@router.post("/predict", response_model=PredictionResponse)
def predict_hazard_risks(request: PredictionRequest):
    """
    Executes real-time multi-hazard risk assessment using Varesh's trained ANN model.
    Persists the inference log to Firebase Firestore for auditability.
    """
    loc_repo = LocationRepository()
    pred_repo = PredictionRepository()
    predictor = get_predictor()

    # 1. Resolve Location
    location_data = None
    if request.location_id:
        location_data = loc_repo.get_by_id(request.location_id)
    elif request.destination_name:
        search_results = loc_repo.search(request.destination_name, limit=1)
        if search_results:
            location_data = search_results[0]

    # Fallback to default coordinates (Delhi) if location unrecognized
    if not location_data:
        if request.destination_name:
            location_data = {
                "id": "custom_search",
                "district_name": request.destination_name,
                "state_name": "India",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "elevation_m": 216.0,
                "population_density": 1000.0,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'location_id' or 'destination_name' must be provided."
            )

    # 2. Extract Weather Overrides (if What-If scenario requested)
    max_t = None
    min_t = None
    r7 = None
    r30 = None
    rh = None
    if request.weather_override:
        max_t = request.weather_override.max_temp_c
        min_t = request.weather_override.min_temp_c
        r7 = request.weather_override.rainfall_7d_mm
        r30 = request.weather_override.rainfall_30d_mm
        rh = request.weather_override.relative_humidity_pct

    # 3. Build Spatiotemporal Feature Input
    spatio_input = SpatiotemporalInput(
        target_date=request.target_date,
        latitude=float(location_data.get("latitude", 20.0)),
        longitude=float(location_data.get("longitude", 78.0)),
        elevation_m=float(location_data.get("elevation_m", 200.0)),
        max_temp_c=max_t,
        min_temp_c=min_t,
        rainfall_7d_mm=r7,
        rainfall_30d_mm=r30,
        relative_humidity_pct=rh,
        population_density=float(location_data.get("population_density", 800.0)),
    )

    # 4. Invoke Multi-Task Neural Network Inference
    ann_result = predictor.predict(spatio_input)

    # 5. Persist Prediction Log to Firebase Firestore
    pred_id = f"pred_{uuid.uuid4().hex[:12]}"
    created_timestamp = datetime.now(timezone.utc).isoformat()

    doc_data = create_prediction_doc(
        prediction_id=pred_id,
        location_id=location_data.get("id", "custom"),
        district_name=location_data.get("district_name", "Unknown"),
        state_name=location_data.get("state_name", "India"),
        target_travel_date=request.target_date,
        model_version=ann_result["model_version"],
        hazard_risks=ann_result["hazard_risks"],
        features_summary=ann_result["features_summary"],
        user_id=request.user_id,
    )
    doc_data["created_at"] = created_timestamp

    persisted = True
    try:
        pred_repo.save(doc_data)
    except Exception as e:
        print(f"[PredictionRoute] Warning: Could not save prediction to Firestore: {e}")
        persisted = False

    # 6. Build Standardized Pydantic Response
    risks_payload = MultiHazardRisks(
        dengue=HazardDetail(**ann_result["hazard_risks"]["dengue"]),
        malaria=HazardDetail(**ann_result["hazard_risks"]["malaria"]),
        flood=HazardDetail(**ann_result["hazard_risks"]["flood"]),
        heatwave=HazardDetail(**ann_result["hazard_risks"]["heatwave"]),
    )

    return PredictionResponse(
        prediction_id=pred_id,
        location_id=location_data.get("id", "custom"),
        district_name=location_data.get("district_name", "Unknown"),
        state_name=location_data.get("state_name", "India"),
        target_travel_date=request.target_date,
        model_version=ann_result["model_version"],
        model_type=ann_result["model_type"],
        coordinates=ann_result["coordinates"],
        hazard_risks=risks_payload,
        features_summary=ann_result["features_summary"],
        persisted_to_firebase=persisted,
        created_at=created_timestamp,
    )


@router.get("/predictions/recent")
def get_recent_predictions(limit: int = Query(10, ge=1, le=50)):
    """
    Retrieves recent prediction queries persisted in Firebase Firestore.
    """
    pred_repo = PredictionRepository()
    return {
        "total": limit,
        "predictions": pred_repo.get_recent(limit=limit),
    }
