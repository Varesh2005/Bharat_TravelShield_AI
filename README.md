# BHARAT-TRAVELSHIELD AI 🇮🇳
### India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Firebase](https://img.shields.io/badge/Database-Firebase%20Firestore-FFA611.svg)](https://firebase.google.com/)
[![Architecture](https://img.shields.io/badge/Model-MultiHazardRiskANN-blueviolet.svg)](docs/ML_ANN_SPECIFICATION.md)
[![License](https://img.shields.io/badge/License-Academic%20Evaluation-brightgreen.svg)]()

---

## 1. Project Overview & Problem Statement

**Bharat-TravelShield AI** is a state-of-the-art spatiotemporal risk intelligence platform designed specifically for the Indian subcontinent. Travelers in India face dynamic, localized, and multi-dimensional hazards ranging from vector-borne epidemics (Dengue, Malaria) during monsoon cycles to extreme natural disasters (Floods, Heatwaves) in summer and monsoon periods.

Traditional travel advisories provide coarse, delayed, or generic regional warnings. Bharat-TravelShield AI provides **hyper-local, district-level, forward-looking risk intelligence** by analyzing spatiotemporal features (cyclical seasonality, temperature lapse rates, cumulative precipitation, and historical surveillance) through a **Deep Multi-Task Artificial Neural Network (MultiHazardRiskANN)** backed by **Google Cloud Firebase Firestore**.

### Key Hazard Intelligence
Rather than calculating an arbitrary blended score, Bharat-TravelShield AI predicts **four independent risk probabilities** alongside standardized categorical risk tiers:
1. **Dengue Risk** ($\in [0.0, 1.0]$) — Vector-borne epidemiological surge
2. **Malaria Risk** ($\in [0.0, 1.0]$) — Vector-borne transmission in tropical basins
3. **Flood Risk** ($\in [0.0, 1.0]$) — Hydrological inundation and extreme rainfall surge
4. **Heatwave Risk** ($\in [0.0, 1.0]$) — Meteorological extreme thermal emergency

---

## 2. Team Work Distribution & Ownership

In accordance with the official team work distribution (`Bharat-TravelShield-AI-Team-Work-Distribution.pdf`), responsibilities and repository ownership are distributed across 5 team members:

| Member | Assigned Role | Repository Scope | Primary Deliverables | Feature Branch |
| :--- | :--- | :--- | :--- | :--- |
| **Pranav** | **Database + Integration + Docs** | `database/`, `docs/`, `docker-compose.yml`, `.env.example`, `README.md` | Firebase Firestore layer, repositories, schemas, integration contracts, documentation | `feature/database` |
| **Varesh** | **ML / ANN & Research** | `ml/`, `research/` | MultiHazardRiskANN architecture, feature pipeline, training loop, evaluation metrics, model weights | `feature/ml` |
| **Rajdeep** | **Backend / API** | `backend/` | FastAPI REST microservice, request validation, prediction orchestration, routing | `feature/backend` |
| **Rosemariya** | **Frontend / UI** | `frontend/` | Next.js interactive web dashboard, destination search, risk visualization, map | `feature/frontend` |
| **Adhith** | **Data Engineering & Dataset** | `data/` | Data collection (IMD, NVBDCP, NDMA), cleaning, canonical datasets | `feature/data` |

---

## 3. System Architecture & End-to-End Flow

```
                      ┌────────────────────────────────────────┐
                      │          ROSEMARIYA: FRONTEND          │
                      │     Next.js Web UI / Search & Map      │
                      └───────────────────▲────────────────────┘
                                          │ REST API / JSON
                                          ▼
                      ┌────────────────────────────────────────┐
                      │           RAJDEEP: BACKEND             │
                      │        FastAPI Microservice            │
                      └───────────────▲────────▲───────────────┘
                                      │        │
               1. Query Location/Hist │        │ 3. Persist Inference Log
                                      │        │
                                      ▼        ▼
┌──────────────────────────────┐    ┌───────────────────────────────────┐
│     VARESH: ML / ANN         │    │       PRANAV: DATABASE            │
│  MultiHazardRiskANN (PyTorch)│◄───┤    Google Firebase Firestore      │
│  - 18 Spatiotemporal Features│    │  - locations (District Registry)  │
│  - Shared Feature Trunk      │    │  - predictions (Audit Logs)       │
│  - 4 Hazard Prediction Heads │    │  - weather_observations (IMD)     │
│  - Explainability Engine     │    │  - disease_surveillance (NVBDCP)  │
└──────────────────────────────┘    └───────────────────────────────────┘
```

---

## 4. Technology Stack

- **Machine Learning & ANN:** PyTorch 2.2+, NumPy, Scikit-Learn (ROC-AUC, PR-AUC, Brier score)
- **Database & Cloud Persistence:** Google Cloud Firestore (via Firebase Admin SDK) + Built-in In-Memory Mock Fallback
- **Backend API:** FastAPI, Uvicorn, Pydantic v2, Starlette
- **Containerization & Deployment:** Docker, Docker Compose
- **Quality Assurance & Testing:** Pytest, HTTPX

---

## 5. Repository Directory Layout

```
c:\backend ANN\
├── backend/                         # Backend microservice (Rajdeep + Pranav)
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint & lifespan
│   │   ├── routes/
│   │   │   ├── predict.py           # /api/v1/predict & /api/v1/predictions/recent
│   │   │   ├── locations.py         # /api/v1/locations (Search & detail)
│   │   │   ├── history.py           # /api/v1/locations/{id}/history
│   │   │   └── model_info.py        # /api/v1/model/info
│   │   └── schemas/
│   │       └── api_schemas.py       # Pydantic request/response validation schemas
│   └── requirements.txt             # Python dependencies
├── database/                        # Database & Persistence Layer (Pranav)
│   ├── firebase_client.py           # Firebase Admin SDK & In-Memory fallback client
│   ├── schema.py                    # Firestore document definitions & helpers
│   ├── seed_firebase.py             # Seed script for 32 Indian district hubs
│   └── repositories/
│       ├── location_repository.py   # District & state queries
│       ├── prediction_repository.py # Inference logs & audit trails
│       └── history_repository.py    # Historical weather & disease records
├── ml/                              # Machine Learning & Neural Network (Varesh)
│   ├── models/
│   │   ├── ann_model.py             # MultiHazardRiskANN (Multi-Task PyTorch MLP)
│   │   └── weights/
│   │       ├── travelshield_ann.pt  # Trained PyTorch neural network weights
│   │       ├── travelshield_weights.json # Portable NumPy weights
│   │       └── metrics.json         # Benchmark metrics (ROC-AUC, PR-AUC)
│   ├── preprocessing/
│   │   └── feature_pipeline.py      # 18-dim spatiotemporal feature engineering
│   ├── training/
│   │   └── train_ann.py             # Training loop, multi-task loss, validation
│   └── inference/
│       └── predict.py               # Real-time inference engine & explainability
├── docs/                            # Comprehensive Technical Documentation (Pranav)
│   ├── FIREBASE_ARCHITECTURE.md     # Firebase collections, security rules & schema
│   ├── ML_ANN_SPECIFICATION.md      # Mathematical ANN formulations & viva notes
│   ├── DATABASE_ARCHITECTURE.md     # Relational database specifications
│   ├── INTEGRATION_CONTRACTS.md     # Inter-module API contracts
│   └── REPOSITORY_AUDIT_REPORT.md   # Initial repository baseline audit
├── tests/                           # Integration Testing Suite
│   └── test_api_endpoints.py        # Pytest test suite for end-to-end flows
├── .env.example                     # Environment configuration template
├── firebase_credentials.example.json# Firebase service account template
├── docker-compose.yml               # Container orchestration
└── Dockerfile                       # Production Docker container build
```

---

## 6. Quickstart Guide (Running Locally)

### Step 1: Clone and Navigate to Repository
```bash
git clone https://github.com/Varesh2005/Bharat_TravelShield_AI.git
cd Bharat_TravelShield_AI
```

### Step 2: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 3: Run the Multi-Hazard ANN Training (Optional — Pre-trained Weights Included)
```bash
python -m ml.training.train_ann
```

### Step 4: Seed the Database
```bash
python -m database.seed_firebase
```
*(Runs seamlessly in zero-config local memory mode, or connects to your cloud project if `firebase_credentials.json` is configured).*

### Step 5: Start the FastAPI Backend Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Docs: `http://localhost:8000/docs`
- Alternative ReDoc: `http://localhost:8000/redoc`

### Step 6: Execute Integration Tests
```bash
python -m pytest tests/test_api_endpoints.py -v
```

---

## 7. API Reference & Examples

### Predict Hazard Risks
`POST /api/v1/predict`
```json
{
  "location_id": "delhi_central",
  "target_date": "2026-07-20",
  "weather_override": {
    "rainfall_7d_mm": 115.0,
    "max_temp_c": 33.5,
    "min_temp_c": 25.0,
    "relative_humidity_pct": 89.0
  }
}
```
**Response ($200\text{ OK}$):**
```json
{
  "prediction_id": "pred_4d98a2b1f0c3",
  "location_id": "delhi_central",
  "district_name": "Central Delhi",
  "state_name": "Delhi",
  "target_travel_date": "2026-07-20",
  "model_version": "MultiHazardRiskANN-v1.0",
  "model_type": "Artificial Neural Network (Multi-Task MLP)",
  "hazard_risks": {
    "dengue": {
      "probability": 0.7443,
      "risk_tier": "High",
      "contributing_factors": [
        "Active monsoon rainfall creates stagnant water pools ideal for Aedes mosquito breeding",
        "High relative humidity (89%) accelerates mosquito survival and viral replication"
      ]
    },
    "malaria": {
      "probability": 0.4910,
      "risk_tier": "Elevated",
      "contributing_factors": [
        "Post-monsoon vegetation and water accumulation facilitate Anopheles vector proliferation"
      ]
    },
    "flood": {
      "probability": 0.6418,
      "risk_tier": "Elevated",
      "contributing_factors": [
        "High 7-day cumulative rainfall (115.0 mm) elevates hydrological surge risk"
      ]
    },
    "heatwave": {
      "probability": 0.0294,
      "risk_tier": "Low",
      "contributing_factors": [
        "Ambient temperatures well below physiological heat emergency thresholds"
      ]
    }
  },
  "persisted_to_firebase": true
}
```

---

## 8. ANN Model Performance Summary

| Hazard Output | Validation ROC-AUC | Validation PR-AUC | Calibration Brier Score | Risk Tier Thresholds |
| :--- | :---: | :---: | :---: | :--- |
| **Dengue** | **0.9130** | **0.8051** | **0.0974** | Low (<0.20), Mod (0.20-0.45), Elev (0.45-0.65), High (0.65-0.85), Severe (>0.85) |
| **Malaria** | **0.8852** | **0.6539** | **0.0945** | Low (<0.20), Mod (0.20-0.45), Elev (0.45-0.65), High (0.65-0.85), Severe (>0.85) |
| **Flood** | **0.9368** | **0.8908** | **0.0656** | Low (<0.20), Mod (0.20-0.45), Elev (0.45-0.65), High (0.65-0.85), Severe (>0.85) |
| **Heatwave** | **0.9736** | **0.9276** | **0.0322** | Low (<0.20), Mod (0.20-0.45), Elev (0.45-0.65), High (0.65-0.85), Severe (>0.85) |

---

## 9. Viva Voce Defense Key Points

1. **Why an ANN and not only tree models?**  
   The project is an ANN Mini Project. The multi-task MLP network utilizes a shared spatiotemporal latent embedding (64 $\rightarrow$ 32 dims) to model interconnected geophysical factors (monsoon, humidity, lapse rate) while keeping 4 dedicated prediction heads for hazard independence.
2. **Why not a combined risk score?**  
   Combining different hazards (e.g., flood and malaria) produces an uninterpretable score. Independent probabilities provide actionable precautions for travelers.
3. **How does Firebase integrate?**  
   Firebase Firestore acts as the persistent cloud registry for locations, user prediction audit trails, and historical surveillance. A mock Firestore fallback allows the code to run out-of-the-box in offline or test environments.
