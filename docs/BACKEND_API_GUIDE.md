# Bharat-TravelShield AI — Complete Backend API Integration Guide

**Owner / Author:** Rajdeep (Role: Backend / API)  
**Target Audience:** Rosemariya (Frontend / UI), Pranav (Integration & DB), Varesh (ML / ANN)  
**API Version:** `1.0.0`  
**OpenAPI Specification:** [docs/BACKEND_API_SPEC.json](file:///docs/BACKEND_API_SPEC.json)  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**Interactive ReDoc UI:** `http://localhost:8000/redoc`  

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

## 2. API Endpoints Directory (All 17 Routes)

### A. System & Status
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | System overview, team roster, supported hazards |
| `GET` | `/health` / `/api/v1/health` | Live readiness & database health status |
| `GET` | `/api/v1/data/status` | Underlying weather, disaster & health data file audit |
| `GET` | `/api/v1/model/version` | Active neural network versions and calibration status |

### B. Destinations & Search
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/destinations/search?q={query}` | Autocomplete search across 640 districts with Indian city aliases |
| `GET` | `/api/v1/destinations` | List registered districts (paginated) |
| `GET` | `/api/v1/destinations/{id}` | Destination coordinates, elevation, tourist spots |
| `GET` | `/api/v1/destinations/district/{code}/summary` | Full historical climate, disease surveillance & precautions |
| `GET` | `/api/v1/locations?query={q}` | Integration contract alias for destination search |
| `GET` | `/api/v1/locations/{id}/history?year={y}` | Integration contract alias for annual historical summary |

### C. Predictions & Intelligence
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/predict/multi-hazard` | **Unified deck of hazard cards** + Firestore audit log |
| `POST` | `/api/v1/predict` | Flexible contract alias accepting `{district_code}` or `{location_id, travel_date}` |
| `POST` | `/api/v1/predict/district` | Single hazard ANN inference (Heatwave, Flood, Malaria) |
| `POST` | `/api/v1/predict/custom` | Model inference on custom caller-supplied feature records |
| `GET` | `/api/v1/hazards` | List hazard support status (validated vs unsupported) |

### D. Counterfactual Simulator, Maps & Research
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/risk/scenario` | **What-If Simulator:** counterfactual predictions with environmental deltas |
| `GET` | `/api/v1/risk/explain/{prediction_id}` | Factor attributions for a saved prediction |
| `GET` | `/api/v1/map/layers/{hazard_type}` | Lightweight district coordinates and markers for MapLibre GL |
| `GET` | `/api/v1/research/leaderboard` | Research Lab: ANN vs Baseline metrics (PR-AUC, ROC-AUC, Brier, ECE) |

### E. Historical Ground Truth & Auditing
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/history/weather/{loc_id}` | Ground-truth historical weather observations (NASA POWER / IMD) |
| `GET` | `/api/v1/history/disease/{loc_id}` | Historical disease surveillance reports (NCVBDC) |
| `GET` | `/api/v1/predictions/recent` | Audit trail of recent traveler predictions |
| `GET` | `/api/v1/predictions/{id}` | Retrieve specific prediction audit record by ID |

---

## 3. Key Request & Response Examples

### 3.1 Unified Multi-Hazard Travel Intelligence (Primary Dashboard)
- **POST** `/api/v1/predict/multi-hazard`
```json
{
  "district_code": 146,
  "user_id": "traveler_99"
}
```
- **Response (200 OK):**
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
      "note": "Calibration validation check failed (ECE 0.173 > 0.10); calibrated probability withheld.",
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
      "note": "Calibration validation check failed (ECE 0.149 > 0.10); calibrated probability withheld."
    },
    "dengue": {
      "hazard": "dengue",
      "status": "unsupported",
      "reason": "Dengue surveillance is available only at state level (2021-2025). Independent district ANN training is scientifically unsupported."
    }
  },
  "disclaimer": "Decision support only. Predictions are experimental artificial neural network estimates.",
  "timestamp": "2026-10-23T10:40:43Z"
}
```

---

### 3.2 What-If Scenario Simulator
- **POST** `/api/v1/risk/scenario`
```json
{
  "hazard": "heatwave",
  "district_code": 146,
  "deltas": {
    "max_temperature_c": 3.0
  }
}
```
- **Response (200 OK):**
```json
{
  "status": "ok",
  "hazard": "heatwave",
  "district_code": 146,
  "applied_deltas": {
    "max_temperature_c": 3.0
  },
  "baseline": {
    "raw_score": 0.385
  },
  "scenario": {
    "raw_score": 0.542
  },
  "score_delta": 0.157,
  "risk_impact": "increased",
  "disclaimer": "Simulated counterfactual outcome. Model response is non-linear and indicative."
}
```

---

## 4. Frontend Integration Guidelines for Rosemariya

1. **Probability Handling:**
   - Display `probability` as percentage when not null (`(prob * 100).toFixed(1) + "%"`).
   - When `probability` is `null` (Heatwave/Flood), display `raw_score_for_ranking_only` and render the tooltip containing `note`.
2. **Honest Science & Ethics:**
   - For `dengue`, check `status === "unsupported"`. Display the state-level surveillance advisory rather than claiming 0% risk.
3. **Security Headers:**
   - Standard security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`) are automatically enforced on every response.
