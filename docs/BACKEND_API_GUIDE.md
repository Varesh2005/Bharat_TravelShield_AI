# Bharat-TravelShield AI — Backend API Integration Guide

**Owner / Author:** Rajdeep (Role: Backend / API)  
**Target Audience:** Rosemariya (Frontend / UI), Pranav (Integration & DB), Varesh (ML / ANN)  
**API Version:** `1.0.0`  
**OpenAPI Specification:** [docs/BACKEND_API_SPEC.json](file:///docs/BACKEND_API_SPEC.json)  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc`  

---

## 1. Quick Start

### Start the FastAPI Microservice
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload --port 8000
```

### Run Automated API Test Suite
```powershell
pytest tests/test_api_endpoints.py -v
```

---

## 2. API Endpoints Overview

| Method | Endpoint | Description | Frontend Use Case |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` / `/api/v1/health` | System health & DB connection status | Ping / Status badge in UI |
| `GET` | `/` | System overview, team metadata, disclaimer | Footer / About modal |
| `GET` | `/api/v1/destinations/search?q={query}` | Autocomplete search across all 640 districts | Hero destination search bar |
| `GET` | `/api/v1/destinations/{location_id}` | Detailed coordinates, elevation, tourist spots | Destination info panel |
| `GET` | `/api/v1/destinations/district/{code}/summary` | Comprehensive historical observations & precautions | Destination Report Page |
| `GET` | `/api/v1/hazards` | Status and calibration availability for all hazards | Hazard selection filter |
| `POST`| `/api/v1/predict/district` | Single hazard ANN prediction for a district | Granular hazard risk card |
| `POST`| `/api/v1/predict/custom` | ANN inference with caller-supplied feature record | What-If Simulator |
| `POST`| `/api/v1/predict/multi-hazard` | **Unified deck of hazard cards** + Firestore audit log | **Main Travel Risk Dashboard** |
| `GET` | `/api/v1/history/weather/{loc_id}` | Ground-truth historical weather observations | Climate trend charts |
| `GET` | `/api/v1/history/disease/{loc_id}` | Historical disease surveillance reports | Epidemiological charts |
| `GET` | `/api/v1/predictions/recent` | Audit log of recent travel predictions | Recent Searches widget |

---

## 3. Key Endpoint Contracts & Payloads

### 3.1 Destination Autocomplete Search
- **Route:** `GET /api/v1/destinations/search?q=Bengaluru&limit=5`
- **Response `(200 OK)`:**
```json
{
  "query": "Bengaluru",
  "count": 1,
  "results": [
    {
      "id": "karnataka_bangalore",
      "census_district_code": 572,
      "district_name": "Bangalore",
      "state_name": "Karnataka",
      "latitude": 12.9716,
      "longitude": 77.5946,
      "elevation_m": 920.0,
      "climate_zone": "Tropical Savanna",
      "population_density": 4.38,
      "popular_tourist_spots": ["Lalbagh", "Cubbon Park", "Bangalore Palace"]
    }
  ]
}
```
*Note: Supports common Indian city aliases (e.g. Bengaluru -> Bangalore, Bombay -> Mumbai, Calcutta -> Kolkata, Kashi -> Varanasi).*

---

### 3.2 Unified Multi-Hazard Travel Intelligence (For Rosemariya's UI)
- **Route:** `POST /api/v1/predict/multi-hazard`
- **Request:**
```json
{
  "district_code": 146,
  "user_id": "traveler_99"
}
```
- **Response `(200 OK)`:**
```json
{
  "prediction_id": "pred_a1b2c3d4e5f6",
  "district_code": 146,
  "district_name": "Agra",
  "state_name": "Uttar Pradesh",
  "feature_year": 2023,
  "predicts_year": 2024,
  "hazards": {
    "malaria": {
      "hazard": "malaria",
      "status": "ok",
      "model_version": "malaria-ann-mlp-v2",
      "target": "high_reported_incidence_next_calendar_year",
      "probability": 0.0217,
      "predicted_class": 0,
      "predicted_label": "high_reported_incidence_not_predicted",
      "raw_score_for_ranking_only": 0.01,
      "top_factors": [
        {"feature": "log1p_incidence_T", "value": 0.12, "effect_on_score": -0.05}
      ]
    },
    "heatwave": {
      "hazard": "heatwave",
      "status": "ok",
      "model_version": "heatwave-ann-mlp-v2",
      "probability": null,
      "raw_score_for_ranking_only": 0.385,
      "note": "Calibration validation check failed (ECE 0.173 > 0.10); calibrated probability withheld per scientific standards.",
      "top_factors": [
        {"feature": "max_temperature_c", "value": 44.5, "effect_on_score": 0.12}
      ]
    },
    "flood": {
      "hazard": "flood",
      "status": "ok",
      "model_version": "flood-ann-mlp-v2",
      "probability": null,
      "raw_score_for_ranking_only": 0.124,
      "note": "Calibration validation check failed (ECE 0.149 > 0.10); calibrated probability withheld.",
      "top_factors": []
    },
    "dengue": {
      "hazard": "dengue",
      "status": "unsupported",
      "reason": "Dengue data is available only at state level (2021-2025). Independent district ANN training is scientifically unsupported.",
      "recovery_path": "Use historical destination summary (/api/v1/destinations/district/{code}/summary) for state-level surveillance records."
    }
  },
  "disclaimer": "Decision support only. Predictions are experimental artificial neural network estimates.",
  "timestamp": "2026-10-23T10:40:43Z"
}
```

---

## 4. Frontend Integration Best Practices (For Rosemariya)

1. **Probability Display Rule:**
   - Always check if `probability !== null`.
   - If `probability` is a number (e.g. Malaria), display it as a percentage: `(probability * 100).toFixed(1) + "%"`.
   - If `probability === null` (Heatwave, Flood), display `raw_score_for_ranking_only` as a relative score and show the `note` indicating calibration is withheld.
2. **Never Show Unsupported Hazards as Zero Risk:**
   - For `dengue`, check `status === "unsupported"`. Display the state-level surveillance note rather than claiming 0% risk.
3. **Audit Tracking:**
   - Every prediction generates a `prediction_id` that is persisted to Firebase Firestore. Use this ID to display shareable link URLs (e.g. `/report/[prediction_id]`).
