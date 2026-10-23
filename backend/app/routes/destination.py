"""
Bharat-TravelShield AI - Official Destination & Hazard API Routes
Author: Pranav (Integration) & Varesh (ML)
Project: Bharat-TravelShield AI

Exposes Varesh's authoritative Destination Analysis layer and Hazard Inference engine
(heatwave, flood, malaria, dengue) via standardized FastAPI endpoints.
"""

from typing import Optional, Any, Dict
from fastapi import APIRouter, HTTPException, Query

from ml.destination_analysis import build_destination_summary, DestinationAnalysisError
from ml import hazard_inference as hi

router = APIRouter(prefix="/api/v1", tags=["Destination Analysis & Hazard Models"])


@router.get("/destination/summary")
def get_destination_summary(
    district_code: Optional[int] = Query(None, description="Census 2011 district code (e.g. 146 for Agra)"),
    district_name: Optional[str] = Query(None, description="District name (e.g. 'Agra')"),
    state_name: Optional[str] = Query(None, description="State name (e.g. 'Uttar Pradesh')"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    with_ann: bool = Query(True, description="Include experimental hazard ANN inferences"),
    ann_feature_year: Optional[int] = Query(None, description="Feature year for ANN (defaults to latest complete year)")
):
    """
    Returns an aggregated spatiotemporal destination summary for a district,
    combining historical weather, flood, heatwave, malaria, and dengue records,
    plus experimental hazard ANN predictions.
    """
    try:
        summary = build_destination_summary(
            district_code=district_code,
            district_name=district_name,
            state_name=state_name,
            start_date=start_date,
            end_date=end_date,
            include_experimental_ann=with_ann,
            ann_feature_year=ann_feature_year,
        )
        return summary
    except DestinationAnalysisError as exc:
        status_map = {
            "destination_not_found": 404,
            "ambiguous_destination": 409,
            "data_unavailable": 503,
            "invalid_request": 400,
        }
        status_code = status_map.get(exc.code, 422)
        raise HTTPException(status_code=status_code, detail=exc.to_dict()["error"])


@router.get("/hazards/list")
def list_supported_hazards():
    """
    Returns readiness and support status for all hazard models.
    """
    return hi.list_hazards()


@router.get("/hazards/{hazard}/district/{district_code}")
def predict_hazard_for_district(
    hazard: str,
    district_code: int,
    feature_year: Optional[int] = Query(None, description="Feature year T (prediction will be for T+1)"),
    explain: bool = Query(True, description="Compute occlusion feature attribution factors")
):
    """
    Evaluates hazard ANN prediction for a specific district code according to ml/ML_MODEL_CONTRACT.md.
    """
    result = hi.predict_for_district(
        hazard=hazard,
        district_code=district_code,
        feature_year=feature_year,
        explain=explain
    )

    status = result.get("status")
    if status == "ok":
        return result
    elif status == "invalid_input":
        raise HTTPException(status_code=422, detail=result)
    elif status == "unsupported":
        raise HTTPException(status_code=501, detail=result)
    elif status == "unavailable":
        raise HTTPException(status_code=404 if result.get("confidence", {}).get("data_coverage") == "none" else 503, detail=result)
    else:
        raise HTTPException(status_code=500, detail=result)
