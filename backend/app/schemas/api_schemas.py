"""
Bharat-TravelShield AI - Pydantic Request & Response Schemas
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Provides strict validation and schema definitions for all API endpoints,
ensuring seamless integration with Rosemariya's Next.js frontend,
Pranav's Firebase layer, and Varesh's ML hazard models.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# ------------------------------------------------------------------------------
# 1. System & Health Schemas
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    api_version: str = Field(..., example="1.0.0")
    database_connected: bool = Field(..., example=True)
    database_mode: str = Field(..., example="cloud_firestore")
    ml_models_ready: bool = Field(..., example=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SystemInfoResponse(BaseModel):
    project: str
    system: str
    version: str
    team: Dict[str, str]
    database: str
    supported_hazards: List[str]
    disclaimer: str


# ------------------------------------------------------------------------------
# 2. Location & Destination Schemas
# ------------------------------------------------------------------------------

class LocationItem(BaseModel):
    id: str = Field(..., example="delhi_central")
    census_district_code: Optional[int] = Field(None, example=90)
    district_name: str = Field(..., example="Central Delhi")
    state_name: str = Field(..., example="Delhi")
    latitude: float = Field(..., example=28.6139)
    longitude: float = Field(..., example=77.2090)
    elevation_m: Optional[float] = Field(None, example=216.0)
    climate_zone: Optional[str] = Field(None, example="Semi-Arid Subtropical")
    population_density: Optional[float] = Field(None, example=11300.0)
    popular_tourist_spots: List[str] = Field(default_factory=list, example=["India Gate", "Red Fort"])


class LocationSearchResponse(BaseModel):
    query: str
    count: int
    results: List[LocationItem]


class DestinationSummaryRequest(BaseModel):
    district_code: Optional[int] = Field(None, description="Census 2011 district code (e.g., 536)")
    district_name: Optional[str] = Field(None, description="District name (e.g., 'Bengaluru Urban')")
    state_name: Optional[str] = Field(None, description="State name (e.g., 'Karnataka')")
    start_date: Optional[str] = Field(None, description="Travel start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Travel end date (YYYY-MM-DD)")
    include_ann: bool = Field(False, description="Whether to include experimental hazard ANN predictions")
    ann_feature_year: Optional[int] = Field(None, description="Override feature year for ANN")


# ------------------------------------------------------------------------------
# 3. Machine Learning & Hazard Prediction Schemas
# ------------------------------------------------------------------------------

class DistrictPredictionRequest(BaseModel):
    hazard: str = Field(..., description="Hazard type: 'heatwave', 'flood', or 'malaria'", example="malaria")
    district_code: int = Field(..., description="Census 2011 district code", example=146)
    feature_year: Optional[int] = Field(None, description="Feature year T (predicts T+1). None = latest complete year", example=2022)
    explain: bool = Field(True, description="Whether to include factor explanation weights")


class CustomPredictionRequest(BaseModel):
    hazard: str = Field(..., description="Hazard type: 'heatwave', 'flood', or 'malaria'", example="heatwave")
    features: Dict[str, float] = Field(..., description="13 or 14 strict numerical features matching the model contract")
    explain: bool = Field(True, description="Whether to include factor explanation")


class TopFactor(BaseModel):
    feature: str
    value: Any
    training_median: Optional[float] = None
    effect_on_score: Optional[float] = None


class HazardPredictionCard(BaseModel):
    hazard: str
    status: str
    model_version: Optional[str] = None
    based_on_year: Optional[int] = None
    predicts_year: Optional[int] = None
    target: Optional[str] = None
    probability: Optional[float] = None
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    raw_score_for_ranking_only: Optional[float] = None
    note: Optional[str] = None
    risk_band: Optional[str] = None
    reliability: Optional[Dict[str, Any]] = None
    limitations: Optional[List[str]] = None
    top_factors: Optional[List[Dict[str, Any]]] = None
    reason: Optional[str] = None
    recovery_path: Optional[str] = None


class UnifiedMultiHazardRequest(BaseModel):
    district_code: int = Field(..., description="Census 2011 district code", example=536)
    feature_year: Optional[int] = Field(None, description="Feature year T (predicts T+1)")
    user_id: Optional[str] = Field("anonymous_traveler", description="Client user ID or 'anonymous_traveler'")


class UnifiedMultiHazardResponse(BaseModel):
    prediction_id: str
    district_code: int
    district_name: str
    state_name: str
    feature_year: Optional[int] = None
    predicts_year: Optional[int] = None
    hazards: Dict[str, HazardPredictionCard]
    disclaimer: str
    timestamp: str


# ------------------------------------------------------------------------------
# 4. History & Audit Trail Schemas
# ------------------------------------------------------------------------------

class WeatherHistoryRecord(BaseModel):
    location_id: str
    observation_date: str
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    source: Optional[str] = "IMD"


class DiseaseHistoryRecord(BaseModel):
    location_id: str
    report_date: str
    disease_type: str
    cases_reported: Optional[int] = None
    source: Optional[str] = "NVBDCP"


class PredictionAuditRecord(BaseModel):
    id: str
    location_id: str
    district_name: Optional[str] = None
    state_name: Optional[str] = None
    target_travel_date: Optional[str] = None
    model_version: Optional[str] = None
    dengue_probability: Optional[float] = None
    malaria_probability: Optional[float] = None
    flood_probability: Optional[float] = None
    heatwave_probability: Optional[float] = None
    created_at: str
