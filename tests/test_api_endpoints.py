"""
Bharat-TravelShield AI - Comprehensive Integration Test Suite
Validates:
  1. Root & Health API endpoints
  2. Firebase Firestore querying (Locations, History)
  3. Varesh's MultiHazardRiskANN inference execution
  4. Real-time prediction persistence in Firebase Firestore
  5. Explainability factors & Risk Tier classification
"""

import pytest
from starlette.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Bharat-TravelShield AI"
    assert "MultiHazardRiskANN" in data["model"]


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True


def test_locations_search(client):
    response = client.get("/api/v1/locations?q=delhi")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any("delhi" in loc["id"].lower() for loc in data["locations"])


def test_location_by_id(client):
    response = client.get("/api/v1/locations/uttar_pradesh_agra")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "uttar_pradesh_agra"
    assert data["district_name"] == "Agra"


def test_prediction_inference_and_persistence(client):
    payload = {
        "location_id": "uttar_pradesh_agra",
        "target_date": "2026-07-20",
        "weather_override": {
            "rainfall_7d_mm": 115.0,
            "max_temp_c": 33.5,
            "min_temp_c": 25.0,
            "relative_humidity_pct": 89.0
        },
        "user_id": "traveler_pranav_001"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check structure
    assert "prediction_id" in data
    assert data["district_name"] == "Agra"
    assert data["target_travel_date"] == "2026-07-20"
    assert data["persisted_to_firebase"] is True

    # Check 4 independent hazards
    hazards = data["hazard_risks"]
    for hazard in ["dengue", "malaria", "flood", "heatwave"]:
        assert hazard in hazards
        h_data = hazards[hazard]
        assert 0.0 <= h_data["probability"] <= 1.0
        assert h_data["risk_tier"] in ["Low", "Moderate", "Elevated", "High", "Severe"]
        assert len(h_data["contributing_factors"]) > 0


def test_recent_predictions(client):
    # Make a prediction first
    client.post("/api/v1/predict", json={
        "location_id": "goa_north_goa",
        "target_date": "2026-08-10"
    })

    response = client.get("/api/v1/predictions/recent?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) >= 1


def test_model_info(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "MultiHazardRiskANN"
    assert len(data["hazards"]) == 4
    assert "dengue" in data["evaluation_metrics"]


def test_destination_summary_endpoint(client):
    response = client.get("/api/v1/destination/summary?district_name=Agra&state_name=Uttar Pradesh&start_date=2024-10-01&end_date=2024-10-31&with_ann=true")
    assert response.status_code == 200
    data = response.json()
    assert data["destination"]["district_name"] == "Agra"
    assert "weather" in data["hazards"]
    assert "experimental_ann_predictions" in data


def test_hazards_list_endpoint(client):
    response = client.get("/api/v1/hazards/list")
    assert response.status_code == 200
    data = response.json()
    assert "heatwave" in data
    assert "malaria" in data
    assert "dengue" in data


def test_hazard_predict_district(client):
    response = client.get("/api/v1/hazards/malaria/district/146")
    assert response.status_code == 200
    data = response.json()
    assert data["hazard"] == "malaria"
    assert data["status"] == "ok"
    assert "prediction" in data
