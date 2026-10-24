"""API Schemas for Bharat-TravelShield AI."""
from backend.app.schemas.api_schemas import (
    HealthResponse,
    SystemInfoResponse,
    LocationItem,
    LocationSearchResponse,
    DestinationSummaryRequest,
    DistrictPredictionRequest,
    CustomPredictionRequest,
    HazardPredictionCard,
    UnifiedMultiHazardRequest,
    UnifiedMultiHazardResponse,
    WeatherHistoryRecord,
    DiseaseHistoryRecord,
    PredictionAuditRecord,
)

__all__ = [
    "HealthResponse",
    "SystemInfoResponse",
    "LocationItem",
    "LocationSearchResponse",
    "DestinationSummaryRequest",
    "DistrictPredictionRequest",
    "CustomPredictionRequest",
    "HazardPredictionCard",
    "UnifiedMultiHazardRequest",
    "UnifiedMultiHazardResponse",
    "WeatherHistoryRecord",
    "DiseaseHistoryRecord",
    "PredictionAuditRecord",
]
