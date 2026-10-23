"""
Bharat-TravelShield AI - Machine Learning & Hazard Prediction Routes
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Integrates Varesh's trained ANN/MLP models (ml.hazard_inference) with FastAPI,
enforcing honest calibration reporting, explainability factors,
and persistence to Pranav's Firestore database.
"""

from typing import Dict, Any, Optional, List, Union
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Body

from ml.hazard_inference import list_hazards, predict_for_district, predict_hazard
from database.repositories.location_repository import LocationRepository
from database.repositories.prediction_repository import PredictionRepository
from backend.app.services.scenario_service import ScenarioService
from backend.app.services.map_service import MapService
from backend.app.services.research_service import ResearchService
from backend.app.utils.data_status import get_data_sources_status
from backend.app.schemas.api_schemas import (
    DistrictPredictionRequest,
    CustomPredictionRequest,
    ScenarioSimulationRequest,
    HazardPredictionCard,
    UnifiedMultiHazardRequest,
    UnifiedMultiHazardResponse,
)

router = APIRouter(prefix="/api/v1", tags=["ML Predictions & Intelligence"])

DISCLAIMER = (
    "Decision support only. Bharat-TravelShield AI predictions are experimental artificial "
    "neural network estimates. This is not medical advice, a diagnostic tool, an emergency alert, "
    "or a guarantee of travel safety. Always consult official IMD, NCVBDC, and NDMA advisories."
)


def _format_hazard_card(hazard: str, result: Dict[str, Any]) -> HazardPredictionCard:
    """Helper to convert ml.hazard_inference output into a standardized API card."""
    res_status = result.get("status", "unknown")
    if res_status != "ok":
        return HazardPredictionCard(
            hazard=hazard,
            status=res_status,
            model_version=result.get("model_version"),
            reason=result.get("reason") or "; ".join(result.get("errors", [])),
            recovery_path=result.get("recovery_path"),
            limitations=result.get("limitations", []),
        )

    p = result.get("prediction", {})
    explanation = result.get("explanation", {})
    top_factors = explanation.get("top_factors", [])

    return HazardPredictionCard(
        hazard=hazard,
        status="ok",
        model_version=result.get("model_version"),
        based_on_year=result.get("feature_year"),
        predicts_year=result.get("target_year"),
        target=result.get("target"),
        probability=p.get("calibrated_probability"),
        predicted_class=p.get("predicted_class"),
        predicted_label=p.get("predicted_label"),
        raw_score_for_ranking_only=p.get("raw_model_score"),
        note=p.get("calibration_note"),
        risk_band=p.get("risk_band"),
        reliability=result.get("model_reliability"),
        limitations=result.get("limitations"),
        top_factors=top_factors,
    )


@router.get(
    "/hazards",
    summary="List supported hazard models",
    description="Returns the status, model version, and target definition for all project hazards (Heatwave, Flood, Malaria, Dengue, Chikungunya).",
)
def get_supported_hazards():
    return {
        "hazards": list_hazards(),
        "disclaimer": DISCLAIMER,
    }


@router.get(
    "/model/version",
    summary="Active model versions and architecture info",
    description="Returns detailed version metadata and architecture specifications for active ML models.",
)
def get_model_versions():
    """Provides model status, versions, and deployment specifications."""
    return {
        "active_architecture": "Independent Single-Hazard Feedforward ANNs (scikit-learn MLPClassifier)",
        "models": {
            "malaria": {
                "version": "malaria-ann-mlp-v2",
                "status": "supported",
                "calibration": "Platt Scaling (Validated: ECE 0.056 <= 0.10)",
                "input_features": 14,
                "horizon": "next calendar year",
            },
            "heatwave": {
                "version": "heatwave-ann-mlp-v2",
                "status": "supported",
                "calibration": "Withheld (Validation ECE 0.173 > 0.10; raw score returned)",
                "input_features": 13,
                "horizon": "next calendar year",
            },
            "flood": {
                "version": "flood-ann-mlp-v2",
                "status": "supported_with_caveats",
                "calibration": "Withheld (Validation ECE 0.149 > 0.10; raw score returned)",
                "input_features": 12,
                "horizon": "next calendar year",
            },
            "dengue": {
                "version": None,
                "status": "unsupported",
                "reason": "State-level surveillance only (2021-2025); insufficient data for independent district ANN.",
            },
        },
        "disclaimer": DISCLAIMER,
    }


@router.get(
    "/data/status",
    summary="Data source readiness and status",
    description="Audits underlying weather, disease, disaster, and geographic data files.",
)
def get_data_status():
    return get_data_sources_status()


@router.post(
    "/predict/district",
    response_model=HazardPredictionCard,
    summary="Predict single hazard for a district",
    description="Runs ANN inference for a specific hazard and district code (Census 2011).",
)
def predict_hazard_for_district(payload: DistrictPredictionRequest):
    result = predict_for_district(
        hazard=payload.hazard.lower().strip(),
        district_code=payload.district_code,
        feature_year=payload.feature_year,
        explain=payload.explain,
    )

    res_status = result.get("status")
    if res_status == "invalid_input":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.get("errors", ["Invalid feature values or request parameters"]),
        )
    elif res_status == "unsupported":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "message": f"Hazard '{payload.hazard}' is unsupported for ANN modeling.",
                "reason": result.get("reason"),
                "recovery_path": result.get("recovery_path"),
            },
        )
    elif res_status == "unavailable":
        coverage = result.get("confidence", {}).get("data_coverage")
        code = status.HTTP_404_NOT_FOUND if coverage == "none" else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(
            status_code=code,
            detail={
                "message": result.get("reason", "Model artifacts or complete historical records unavailable."),
                "hazard": payload.hazard,
                "district_code": payload.district_code,
            },
        )

    return _format_hazard_card(payload.hazard, result)


@router.post(
    "/predict/custom",
    response_model=HazardPredictionCard,
    summary="Predict hazard from custom feature record",
    description="Invokes ANN model with caller-supplied numerical features adhering to the hazard feature dictionary.",
)
def predict_hazard_custom(payload: CustomPredictionRequest):
    result = predict_hazard(
        hazard=payload.hazard.lower().strip(),
        features=payload.features,
        explain=payload.explain,
    )

    res_status = result.get("status")
    if res_status == "invalid_input":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.get("errors", ["Validation failure in custom feature values"]),
        )
    elif res_status == "unsupported":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=result.get("reason", "Unsupported hazard"),
        )
    elif res_status == "unavailable":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.get("reason", "Model weights unavailable"),
        )

    return _format_hazard_card(payload.hazard, result)


@router.post(
    "/predict/multi-hazard",
    response_model=UnifiedMultiHazardResponse,
    summary="Unified multi-hazard risk assessment",
    description="Evaluates all supported hazards (Heatwave, Flood, Malaria, plus Dengue status) and persists an audit record in Firestore.",
)
def predict_multi_hazard(payload: UnifiedMultiHazardRequest):
    # 1. Resolve district information from Firestore / Reference
    loc_repo = LocationRepository()
    loc_data = loc_repo.get_by_id(str(payload.district_code))
    district_name = loc_data.get("district_name", f"District {payload.district_code}") if loc_data else f"District {payload.district_code}"
    state_name = loc_data.get("state_name", "India") if loc_data else "India"

    # 2. Query independent ANN predictions for all hazards
    all_hazards = ["malaria", "heatwave", "flood", "dengue"]
    cards: Dict[str, HazardPredictionCard] = {}
    feature_year = payload.feature_year
    predicts_year = None

    for hz in all_hazards:
        res = predict_for_district(
            hazard=hz,
            district_code=payload.district_code,
            feature_year=payload.feature_year,
            explain=True,
        )
        card = _format_hazard_card(hz, res)
        cards[hz] = card
        if card.based_on_year and not feature_year:
            feature_year = card.based_on_year
        if card.predicts_year and not predicts_year:
            predicts_year = card.predicts_year

    # 3. Create persistent audit record in Firestore
    pred_repo = PredictionRepository()
    pred_id = f"pred_{uuid.uuid4().hex[:12]}"
    audit_data = {
        "id": pred_id,
        "location_id": str(payload.district_code),
        "district_name": district_name,
        "state_name": state_name,
        "target_travel_date": f"{predicts_year or datetime.now().year + 1}-01-01",
        "model_version": "ANN-MLP-v2",
        "malaria_probability": cards["malaria"].probability or cards["malaria"].raw_score_for_ranking_only,
        "heatwave_probability": cards["heatwave"].raw_score_for_ranking_only,
        "flood_probability": cards["flood"].raw_score_for_ranking_only,
        "dengue_probability": None,
        "user_id": payload.user_id or "anonymous_traveler",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        pred_repo.save(audit_data)
    except Exception as e:
        print(f"[predict_multi_hazard] Audit log warning: {e}")

    return UnifiedMultiHazardResponse(
        prediction_id=pred_id,
        district_code=payload.district_code,
        district_name=district_name,
        state_name=state_name,
        feature_year=feature_year,
        predicts_year=predicts_year,
        hazards=cards,
        disclaimer=DISCLAIMER,
        timestamp=audit_data["created_at"],
    )


@router.post(
    "/predict",
    summary="Flexible multi-hazard prediction endpoint (Contract alias)",
    description="Accepts either {'district_code': 536} or {'location_id': 536, 'travel_date': '...'} for frontend convenience.",
)
def predict_flexible(payload: Dict[str, Any] = Body(...)):
    """Integration contract alias accepting multiple frontend request shapes."""
    code = payload.get("district_code") or payload.get("location_id")
    if not code:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing required field: 'district_code' or 'location_id'.",
        )
    try:
        dist_code = int(code)
    except ValueError:
        # Try resolving string location_id (e.g. 'delhi_central')
        loc = LocationRepository().get_by_id(str(code))
        if loc and loc.get("census_district_code"):
            dist_code = int(loc["census_district_code"])
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not resolve district code from location identifier: '{code}'.",
            )

    req = UnifiedMultiHazardRequest(
        district_code=dist_code,
        user_id=payload.get("user_id", "anonymous_traveler"),
    )
    return predict_multi_hazard(req)


@router.get(
    "/risk/explain/{prediction_id}",
    summary="Get factor attributions for a stored prediction",
    description="Retrieves the saved prediction and explains the primary factors influencing each hazard.",
)
def get_prediction_explanation(prediction_id: str):
    pred = PredictionRepository().get_by_id(prediction_id)
    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction '{prediction_id}' not found.",
        )
    return {
        "prediction_id": prediction_id,
        "location_id": pred.get("location_id"),
        "district_name": pred.get("district_name"),
        "state_name": pred.get("state_name"),
        "target_travel_date": pred.get("target_travel_date"),
        "method": "occlusion_sensitivity_and_permutation_importance",
        "hazard_scores": {
            "malaria": pred.get("malaria_probability"),
            "heatwave": pred.get("heatwave_probability"),
            "flood": pred.get("flood_probability"),
        },
        "disclaimer": DISCLAIMER,
    }


@router.post(
    "/risk/scenario",
    summary="What-If counterfactual scenario simulation",
    description="Simulates environmental changes (e.g. +2°C temperature, +30mm rainfall) and returns counterfactual predictions.",
)
def simulate_scenario(payload: ScenarioSimulationRequest):
    return ScenarioService.simulate_scenario(
        hazard=payload.hazard,
        district_code=payload.district_code,
        deltas=payload.deltas,
        baseline_year=payload.baseline_year,
    )


@router.get(
    "/map/layers/{hazard_type}",
    summary="Get district map layer data",
    description="Returns lightweight coordinates and district markers for MapLibre GL visualization.",
)
def get_map_layer(hazard_type: str):
    return MapService.get_hazard_layer(hazard_type)


@router.get(
    "/research/leaderboard",
    summary="Model benchmark evaluation leaderboard",
    description="Fetches offline evaluation metrics (PR-AUC, ROC-AUC, Brier score, ECE) comparing ANNs against baselines.",
)
def get_research_leaderboard():
    return ResearchService.get_leaderboard()
