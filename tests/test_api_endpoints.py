"""
Bharat-TravelShield AI - Comprehensive Backend & API Integration Tests
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Tests all FastAPI REST endpoints, Pydantic validations, ML inference integration,
and Firestore repository persistence.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. System & Health Tests
# ------------------------------------------------------------------------------

def test_root_overview():
    """Verify system info endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Bharat-TravelShield AI"
    assert data["version"] == "1.0.0"
    assert "Rajdeep" in data["team"]["backend_api"]
    assert "heatwave" in data["supported_hazards"]


def test_health_check_endpoints():
    """Verify both /health and /api/v1/health endpoints."""
    for path in ["/health", "/api/v1/health"]:
        res = client.get(path)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["database_connected"] is True
        assert data["ml_models_ready"] is True


# ------------------------------------------------------------------------------
# 2. Destination & Search Tests
# ------------------------------------------------------------------------------

def test_destination_search_known_cities():
    """Verify search returns matched districts for popular Indian destinations and aliases."""
    queries = ["Jaipur", "Mumbai", "Delhi", "Goa", "Bengaluru", "Calcutta"]
    for q in queries:
        res = client.get(f"/api/v1/destinations/search?q={q}")
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == q
        assert data["count"] > 0
        assert len(data["results"]) > 0
        # Verify first result has expected structure
        first = data["results"][0]
        assert "district_name" in first
        assert "state_name" in first
        assert "latitude" in first
        assert "longitude" in first


def test_list_all_destinations():
    """Verify listing canonical districts."""
    res = client.get("/api/v1/destinations?limit=15")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 15


def test_get_destination_by_id():
    """Verify retrieving a single location by its document ID or code."""
    res = client.get("/api/v1/destinations/delhi_central")
    assert res.status_code == 200
    data = res.json()
    assert "Central" in data["district_name"]
    assert "Delhi" in data["state_name"]


def test_get_destination_by_id_not_found():
    """Verify 404 response for non-existent destination."""
    res = client.get("/api/v1/destinations/atlantis_city")
    assert res.status_code == 404


def test_destination_summary_census_code():
    """Verify data aggregation summary for a valid Census district code (e.g. 536 - Hyderabad)."""
    res = client.get("/api/v1/destinations/district/536/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["summary_type"] == "destination_data_summary"
    assert "hazards" in data
    assert "weather" in data["hazards"]
    assert "precautions" in data
    assert "disclaimer" in data


# ------------------------------------------------------------------------------
# 3. Machine Learning & Hazard Prediction Tests
# ------------------------------------------------------------------------------

def test_get_supported_hazards():
    """Verify list of supported hazards."""
    res = client.get("/api/v1/hazards")
    assert res.status_code == 200
    data = res.json()
    assert "hazards" in data
    hazards = data["hazards"]
    assert "heatwave" in hazards
    assert "flood" in hazards
    assert "malaria" in hazards
    assert "dengue" in hazards
    assert hazards["malaria"]["status"] == "supported"
    assert hazards["dengue"]["status"] == "unsupported"


def test_predict_malaria_district():
    """Verify Malaria ANN inference with validated calibrated probability."""
    payload = {
        "hazard": "malaria",
        "district_code": 146,
        "feature_year": 2022,
        "explain": True,
    }
    res = client.post("/api/v1/predict/district", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["hazard"] == "malaria"
    assert data["status"] == "ok"
    assert data["probability"] is not None
    assert 0.0 <= data["probability"] <= 1.0
    assert len(data["top_factors"]) > 0


def test_predict_flood_district_withheld_probability():
    """Verify Flood ANN inference returns raw score while withholding calibrated probability."""
    payload = {
        "hazard": "flood",
        "district_code": 536,
        "explain": True,
    }
    res = client.post("/api/v1/predict/district", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["hazard"] == "flood"
    assert data["status"] == "ok"
    # Flood calibration did not pass validation check, probability must be null
    assert data["probability"] is None
    assert data["raw_score_for_ranking_only"] is not None
    assert 0.0 <= data["raw_score_for_ranking_only"] <= 1.0


def test_predict_unsupported_hazard_dengue():
    """Verify 501 Not Implemented response for unsupported Dengue hazard model."""
    payload = {
        "hazard": "dengue",
        "district_code": 146,
    }
    res = client.post("/api/v1/predict/district", json=payload)
    assert res.status_code == 501
    data = res.json()
    assert "unsupported" in str(data).lower()


def test_predict_invalid_hazard():
    """Verify 422 Unprocessable Entity for unknown hazard."""
    payload = {
        "hazard": "tornado",
        "district_code": 146,
    }
    res = client.post("/api/v1/predict/district", json=payload)
    assert res.status_code == 422


def test_predict_multi_hazard_unified():
    """Verify unified multi-hazard prediction endpoint for Next.js frontend."""
    payload = {
        "district_code": 146,
        "user_id": "test_traveler_1",
    }
    res = client.post("/api/v1/predict/multi-hazard", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction_id" in data
    assert data["prediction_id"].startswith("pred_")
    assert data["district_code"] == 146
    assert "hazards" in data
    assert "malaria" in data["hazards"]
    assert "flood" in data["hazards"]
    assert "heatwave" in data["hazards"]
    assert "dengue" in data["hazards"]
    assert data["hazards"]["dengue"]["status"] == "unsupported"


# ------------------------------------------------------------------------------
# 4. History & Audit Trail Tests
# ------------------------------------------------------------------------------

def test_get_weather_history():
    """Verify historical weather query."""
    res = client.get("/api/v1/history/weather/delhi_central?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["location_id"] == "delhi_central"
    assert "records" in data


def test_get_disease_history():
    """Verify historical disease surveillance query."""
    res = client.get("/api/v1/history/disease/delhi_central?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["location_id"] == "delhi_central"
    assert "records" in data


def test_get_recent_predictions_audit():
    """Verify retrieving recent prediction logs."""
    res = client.get("/api/v1/predictions/recent?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


# ------------------------------------------------------------------------------
# 5. Advanced Intelligence, Scenario, Map & Research Tests
# ------------------------------------------------------------------------------

def test_security_headers_present():
    """Verify security headers are attached by middleware."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert "X-XSS-Protection" in res.headers


def test_model_version_endpoint():
    """Verify active model versions endpoint."""
    res = client.get("/api/v1/model/version")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "malaria" in data["models"]
    assert data["models"]["malaria"]["version"] == "malaria-ann-mlp-v2"


def test_data_status_endpoint():
    """Verify data sources audit status."""
    res = client.get("/api/v1/data/status")
    assert res.status_code == 200
    data = res.json()
    assert "overall_status" in data
    assert data["sources_count"] > 0
    assert len(data["sources"]) > 0


def test_locations_alias_query():
    """Verify GET /api/v1/locations?query=... alias contract."""
    res = client.get("/api/v1/locations?query=Jaipur")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Jaipur" in data[0]["district_name"]


def test_predict_flexible_contract_alias():
    """Verify POST /api/v1/predict contract alias."""
    payload = {"district_code": 146}
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction_id" in data
    assert data["district_code"] == 146
    assert "hazards" in data


def test_whatif_scenario_simulation():
    """Verify What-If scenario simulation returns delta comparisons."""
    payload = {
        "hazard": "heatwave",
        "district_code": 146,
        "deltas": {"max_temperature_c": 3.0},
    }
    res = client.post("/api/v1/risk/scenario", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "baseline" in data
    assert "scenario" in data
    assert "applied_deltas" in data
    assert data["applied_deltas"]["max_temperature_c"] == 3.0


def test_risk_explain_endpoint():
    """Verify risk explanation retrieval for generated prediction."""
    # First generate a prediction
    pred_res = client.post("/api/v1/predict/multi-hazard", json={"district_code": 146})
    pred_id = pred_res.json()["prediction_id"]

    # Retrieve explanation
    exp_res = client.get(f"/api/v1/risk/explain/{pred_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["prediction_id"] == pred_id
    assert "hazard_scores" in exp_data


def test_map_layer_endpoint():
    """Verify lightweight spatial map layer for MapLibre GL."""
    res = client.get("/api/v1/map/layers/malaria")
    assert res.status_code == 200
    data = res.json()
    assert data["hazard"] == "malaria"
    assert data["count"] > 0
    assert len(data["districts"]) > 0


def test_research_leaderboard_endpoint():
    """Verify research benchmark leaderboard for Research Lab."""
    res = client.get("/api/v1/research/leaderboard")
    assert res.status_code == 200
    data = res.json()
    assert "hazards" in data
    assert "heatwave" in data["hazards"]
    assert "malaria" in data["hazards"]

