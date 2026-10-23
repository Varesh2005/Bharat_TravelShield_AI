"""
Bharat-TravelShield AI - Pydantic Request & Response Schemas
Author: Pranav (Database + Integration) & Rajdeep (Backend)
Project: Bharat-TravelShield AI
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class WeatherOverride(BaseModel):
    max_temp_c: Optional[float] = Field(None, description="Maximum temperature in Celsius")
    min_temp_c: Optional[float] = Field(None, description="Minimum temperature in Celsius")
    rainfall_7d_mm: Optional[float] = Field(None, description="7-day cumulative rainfall in mm")
    rainfall_30d_mm: Optional[float] = Field(None, description="30-day cumulative rainfall in mm")
    relative_humidity_pct: Optional[float] = Field(None, description="Relative humidity percentage (0-100)")


class PredictionRequest(BaseModel):
    location_id: Optional[str] = Field(None, description="Location ID, e.g. 'delhi_central' or 'karnataka_bengaluru'")
    destination_name: Optional[str] = Field(None, description="Fuzzy district or city name, e.g. 'Delhi' or 'Manali'")
    target_date: str = Field(..., description="Target travel date in YYYY-MM-DD format")
    weather_override: Optional[WeatherOverride] = Field(None, description="Optional custom weather conditions for What-If scenario analysis")
    user_id: Optional[str] = Field(None, description="Optional user ID for tracking saved itineraries")


class HazardDetail(BaseModel):
    probability: float = Field(..., description="Independent risk probability [0.0 to 1.0]")
    risk_tier: str = Field(..., description="Categorical risk tier: Low, Moderate, Elevated, High, Severe")
    contributing_factors: List[str] = Field(..., description="Key environmental and epidemiological drivers")


class MultiHazardRisks(BaseModel):
    dengue: HazardDetail
    malaria: HazardDetail
    flood: HazardDetail
    heatwave: HazardDetail


class PredictionResponse(BaseModel):
    prediction_id: str
    location_id: str
    district_name: str
    state_name: str
    target_travel_date: str
    model_version: str
    model_type: str
    coordinates: Dict[str, float]
    hazard_risks: MultiHazardRisks
    features_summary: Dict[str, Any]
    persisted_to_firebase: bool
    created_at: str


class LocationDetail(BaseModel):
    id: str
    census_district_code: Optional[int] = None
    district_name: str
    state_name: str
    normalized_key: str
    latitude: float
    longitude: float
    elevation_m: float
    climate_zone: str
    population_density: float
    popular_tourist_spots: List[str] = Field(default_factory=list)


class LocationSearchResponse(BaseModel):
    total: int
    locations: List[LocationDetail]
