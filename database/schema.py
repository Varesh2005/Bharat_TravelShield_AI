"""
Bharat-TravelShield AI - Firestore Schema Definitions & Validation
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Defines document structure, field specifications, and collection names
for the Firebase Firestore database layer.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


# Collection Names
COLLECTION_LOCATIONS = "locations"
COLLECTION_PREDICTIONS = "predictions"
COLLECTION_WEATHER = "weather_observations"
COLLECTION_DISEASE = "disease_surveillance"
COLLECTION_DISASTER = "disaster_events"
COLLECTION_MODELS = "ml_models"
COLLECTION_USERS = "users"


def create_location_doc(
    location_id: str,
    district_name: str,
    state_name: str,
    latitude: float,
    longitude: float,
    elevation_m: float = 200.0,
    climate_zone: str = "Subtropical",
    population_density: float = 800.0,
    popular_tourist_spots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generates a validated Location document for Firestore."""
    return {
        "id": location_id,
        "district_name": district_name,
        "state_name": state_name,
        "normalized_key": f"{state_name}_{district_name}".lower().replace(" ", "_"),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation_m": float(elevation_m),
        "climate_zone": climate_zone,
        "population_density": float(population_density),
        "popular_tourist_spots": popular_tourist_spots or [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def create_prediction_doc(
    prediction_id: str,
    location_id: str,
    district_name: str,
    state_name: str,
    target_travel_date: str,
    model_version: str,
    hazard_risks: Dict[str, Any],
    features_summary: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates an auditable Prediction document for Firestore."""
    return {
        "id": prediction_id,
        "location_id": location_id,
        "district_name": district_name,
        "state_name": state_name,
        "target_travel_date": target_travel_date,
        "model_version": model_version,
        "dengue_probability": hazard_risks["dengue"]["probability"],
        "dengue_tier": hazard_risks["dengue"]["risk_tier"],
        "dengue_factors": hazard_risks["dengue"]["contributing_factors"],
        "malaria_probability": hazard_risks["malaria"]["probability"],
        "malaria_tier": hazard_risks["malaria"]["risk_tier"],
        "malaria_factors": hazard_risks["malaria"]["contributing_factors"],
        "flood_probability": hazard_risks["flood"]["probability"],
        "flood_tier": hazard_risks["flood"]["risk_tier"],
        "flood_factors": hazard_risks["flood"]["contributing_factors"],
        "heatwave_probability": hazard_risks["heatwave"]["probability"],
        "heatwave_tier": hazard_risks["heatwave"]["risk_tier"],
        "heatwave_factors": hazard_risks["heatwave"]["contributing_factors"],
        "features_summary": features_summary or {},
        "user_id": user_id or "anonymous_traveler",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def create_weather_doc(
    location_id: str,
    observation_date: str,
    max_temp_c: float,
    min_temp_c: float,
    rainfall_mm: float,
    humidity_pct: float,
    source: str = "IMD",
) -> Dict[str, Any]:
    """Generates a historical Weather observation document for Firestore."""
    return {
        "location_id": location_id,
        "observation_date": observation_date,
        "max_temp_c": float(max_temp_c),
        "min_temp_c": float(min_temp_c),
        "rainfall_mm": float(rainfall_mm),
        "humidity_pct": float(humidity_pct),
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def create_disease_doc(
    location_id: str,
    report_date: str,
    disease_type: str,  # 'Dengue' or 'Malaria'
    cases_reported: int,
    source: str = "NVBDCP",
) -> Dict[str, Any]:
    """Generates a historical Disease surveillance document for Firestore."""
    return {
        "location_id": location_id,
        "report_date": report_date,
        "disease_type": disease_type,
        "cases_reported": int(cases_reported),
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
