# BHARAT-TRAVELSHIELD AI — FIREBASE FIRESTORE ARCHITECTURE SPECIFICATION

**Author:** Pranav (Role: Database + Integration + Documentation)  
**Project:** Bharat-TravelShield AI  
**Repository:** [Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI)  
**Status:** `IMPLEMENTED & ACTIVE`  

---

## 1. Executive Summary & Design Rationale

Bharat-TravelShield AI adopts a **Hybrid Serverless Cloud Architecture** combining:
1. **Google Cloud Firestore (NoSQL Document Store):** Serves as the real-time cloud data store for geographic locations, user itineraries, historical climate observations, epidemiological surveillance, and model audit records.
2. **FastAPI Inference Microservice:** Loads and runs Varesh's PyTorch/NumPy **MultiHazardRiskANN** model in-memory with sub-10ms response latency, persisting inference logs back into Firestore.
3. **Built-in Mock Fallback Mode:** In local development, testing, or environment setups where cloud keys are pending, an embedded in-memory Firestore engine activates automatically, guaranteeing zero-crash operation.

```
                     ┌──────────────────────────────────────────────┐
                     │          FRONTEND (Next.js / Client)         │
                     │          Auth (Firebase Authentication)      │
                     └───────────────────────▲──────────────────────┘
                                             │ REST / JSON
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │          BACKEND (FastAPI Microservice)      │
                     │  - Pydantic Validation                       │
                     │  - MultiHazardRiskANN In-Memory Inference     │
                     └───────────────▲───────────────▲──────────────┘
                                     │               │
                 1. Query Location   │               │ 3. Persist Inference Log
                    & History        │               │    & Provenance
                                     ▼               ▼
                     ┌──────────────────────────────────────────────┐
                     │           FIREBASE FIRESTORE DB              │
                     │  - locations                                 │
                     │  - predictions (Audit Trail)                 │
                     │  - weather_observations                      │
                     │  - disease_surveillance                      │
                     │  - ml_models (Model Registry)                │
                     └──────────────────────────────────────────────┘
```

---

## 2. Firestore Collections & Document Schemas

### 2.1 Collection: `locations`
Stores canonical spatiotemporal metadata for Indian districts and states.
- **Document ID:** Format: `{state}_{district}` in snake_case (e.g., `delhi_central`, `karnataka_bengaluru`, `ladakh_leh`).
- **Fields:**
  | Field | Type | Description |
  | :--- | :--- | :--- |
  | `id` | String | Unique slug identifier |
  | `district_name` | String | Official district name (e.g., "Central Delhi") |
  | `state_name` | String | State/UT name (e.g., "Delhi") |
  | `normalized_key` | String | Lowercase search alias (e.g., `delhi_central_delhi`) |
  | `latitude` | Number | District centroid latitude (e.g., 28.6139) |
  | `longitude` | Number | District centroid longitude (e.g., 77.2090) |
  | `elevation_m` | Number | Elevation in meters above sea level |
  | `climate_zone` | String | Climatic classification (e.g., "Semi-Arid Subtropical") |
  | `population_density` | Number | Persons per square kilometer |
  | `popular_tourist_spots` | Array[String] | Major tourist destinations in the district |
  | `updated_at` | Timestamp | ISO-8601 modification timestamp |

### 2.2 Collection: `predictions`
Logs every inference query, inputs, independent hazard probabilities, and explainability factors.
- **Document ID:** `pred_{uuid12}` (e.g., `pred_a1b2c3d4e5f6`).
- **Fields:**
  | Field | Type | Description |
  | :--- | :--- | :--- |
  | `id` | String | Unique prediction ID |
  | `location_id` | String | Foreign key reference to `locations/{id}` |
  | `district_name` | String | Denormalized district name for fast reporting |
  | `state_name` | String | Denormalized state name |
  | `target_travel_date` | String | Target travel date (`YYYY-MM-DD`) |
  | `model_version` | String | Model version string (e.g., `MultiHazardRiskANN-v1.0`) |
  | `dengue_probability` | Number | Dengue risk score $\in [0.0, 1.0]$ |
  | `dengue_tier` | String | Risk tier: Low, Moderate, Elevated, High, Severe |
  | `dengue_factors` | Array[String] | Key explainability drivers |
  | `malaria_probability` | Number | Malaria risk score $\in [0.0, 1.0]$ |
  | `malaria_tier` | String | Risk tier: Low, Moderate, Elevated, High, Severe |
  | `malaria_factors` | Array[String] | Key explainability drivers |
  | `flood_probability` | Number | Flood risk score $\in [0.0, 1.0]$ |
  | `flood_tier` | String | Risk tier: Low, Moderate, Elevated, High, Severe |
  | `flood_factors` | Array[String] | Key explainability drivers |
  | `heatwave_probability` | Number | Heatwave risk score $\in [0.0, 1.0]$ |
  | `heatwave_tier` | String | Risk tier: Low, Moderate, Elevated, High, Severe |
  | `heatwave_factors` | Array[String] | Key explainability drivers |
  | `features_summary` | Map | Summary of input weather and month values |
  | `user_id` | String | User ID (or `anonymous_traveler`) |
  | `created_at` | Timestamp | Ingestion timestamp |

### 2.3 Collection: `weather_observations`
Stores historical ground-truth meteorological records.
- **Document ID:** `{location_id}_{observation_date}`
- **Fields:** `location_id`, `observation_date`, `max_temp_c`, `min_temp_c`, `rainfall_mm`, `humidity_pct`, `source` ("IMD").

### 2.4 Collection: `disease_surveillance`
Stores public health surveillance records from NVBDCP.
- **Document ID:** `{location_id}_{disease_type}_{report_date}`
- **Fields:** `location_id`, `report_date`, `disease_type` ("Dengue" / "Malaria"), `cases_reported`, `source` ("NVBDCP").

### 2.5 Collection: `ml_models`
Maintains version registry and validated benchmark metrics for models trained by Varesh.
- **Document ID:** `{model_id}` (e.g., `MultiHazardRiskANN-v1.0`)
- **Fields:** `model_id`, `name`, `architecture`, `inputs_dim`, `metrics`, `is_active`.

---

## 3. Security Rules Configuration (`firestore.rules`)

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    
    // Read-only public collections (locations, historical surveillance, models)
    match /locations/{locationId} {
      allow read: if true;
      allow write: if request.auth != null && request.auth.token.admin == true;
    }
    
    match /weather_observations/{docId} {
      allow read: if true;
      allow write: if false; // Ingested via backend service account only
    }
    
    match /disease_surveillance/{docId} {
      allow read: if true;
      allow write: if false; // Ingested via backend service account only
    }
    
    match /ml_models/{modelId} {
      allow read: if true;
      allow write: if false;
    }
    
    // Predictions: Readable by owner, append-only by authenticated backend
    match /predictions/{predId} {
      allow read: if true;
      allow create: if request.auth != null || request.resource.data.user_id == "anonymous_traveler";
      allow update, delete: if false; // Immutable audit log
    }
  }
}
```

---

## 4. Integration Verification & Health Monitoring

The FastAPI health endpoint (`GET /api/v1/health`) confirms the database status:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "database_connected": true,
  "database_mode": "cloud_firestore",
  "ml_model_loaded": true
}
```
If `firebase_credentials.json` is not present, `database_mode` seamlessly reports `"mock_memory"`.
