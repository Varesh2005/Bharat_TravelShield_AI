# BHARAT-TRAVELSHIELD AI — PRELIMINARY INTEGRATION CONTRACTS

**Author:** Pranav (Role: Database + Integration + Documentation)  
**Project:** Bharat-TravelShield AI  
**Repository:** [Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI)  
**Document Status:** `PRELIMINARY INTERFACE BASELINE`  
**Phase:** Phase 1 — Database Architecture & Integration Contracts  
**Last Updated:** September 13, 2026  

---

## 1. INTERFACE: ADHITH (DATA) ──► DATABASE (PRANAV)

### Current Known Information
- Adhith is responsible for collecting, cleaning, and formatting datasets from official sources: IMD (Weather), NVBDCP/NCDC (Dengue, Malaria), NDMA/CWC (Floods, Heatwaves), and Geographic boundary/coordinate references.
- Output from `data/` must be delivered as clean, model-ready, canonical tabular files.
- Pranav will provide seed loaders (`database/seeds/`) to ingest these canonical datasets into PostgreSQL.

### Information Required from Adhith
1. `[NEEDS TEAM CONFIRMATION]` **Canonical District & State Identifiers:** Official naming standard or code (LGD code, Census 2011 code, or cleaned English strings).
2. `[NEEDS TEAM CONFIRMATION]` **Temporal Frequency:** Granularity of observations (daily, weekly, or monthly aggregated).
3. `[NEEDS TEAM CONFIRMATION]` **Exact Column Names & Metric Units:** Confirmed units for precipitation (mm), temperature (°C), and disease statistics (case counts vs. normalized incidence rates per 100,000).
4. `[NEEDS TEAM CONFIRMATION]` **Delivery File Format & Location:** Exact relative path (e.g., `data/processed/canonical_weather.csv` or `.parquet`).

### Proposed Data Format `[PROPOSED]`
```csv
# Proposed: data/processed/canonical_weather.csv
location_id,observation_date,temp_mean_c,temp_max_c,temp_min_c,rainfall_mm,humidity_pct,source_code
1,2023-07-15,28.5,33.2,24.1,12.4,82.0,IMD_WEATHER
```
```csv
# Proposed: data/processed/canonical_disease.csv
location_id,report_date,disease_type,cases_reported,incidence_rate_per_100k,source_code
1,2023-07-01,DENGUE,45,3.21,NVBDCP_DISEASE
```

### Unresolved Questions
- Are disease counts available at the district level for all 750+ districts, or only state-level aggregated reports?
- How are missing weather station observations interpolated?

**Owner Responsible for Confirmation:** Adhith (Data Engineering & Dataset)

---

## 2. INTERFACE: DATABASE (PRANAV) ──► VARESH (ML / ANN)

### Current Known Information
- Varesh is responsible for the neural network (ANN/MLP) risk prediction models and baseline comparisons (LightGBM, Random Forest).
- The models require historical features (weather lags, seasonal cycles, previous disease outbreaks, disaster occurrences) and spatial indicators.
- Varesh may train models directly on canonical datasets or extract historical feature vectors from PostgreSQL.

### Information Required from Varesh
1. `[NEEDS TEAM CONFIRMATION]` **Training Data Extraction Method:** Will Varesh train offline directly from Adhith's CSVs in `data/`, or does he require SQL export views from PostgreSQL?
2. `[NEEDS TEAM CONFIRMATION]` **Feature Aggregation Window:** What historical rolling windows are needed (e.g. 7-day rainfall sum, 14-day mean temperature, 30-day disease incidence lag)?

### Proposed Database Views / Queries `[PROPOSED]`
```sql
-- Proposed query helper for ML feature extraction
SELECT 
    l.id AS location_id,
    l.state_name,
    l.district_name,
    l.latitude,
    l.longitude,
    w.observation_date,
    w.temp_mean_c,
    w.rainfall_mm,
    w.humidity_pct
FROM locations l
JOIN weather_observations w ON l.id = w.location_id
WHERE l.id = :location_id 
  AND w.observation_date BETWEEN :start_date AND :end_date
ORDER BY w.observation_date ASC;
```

### Unresolved Questions
- Does Varesh require precomputed seasonal rolling averages stored inside the database, or will feature engineering occur in-memory within Python?

**Owner Responsible for Confirmation:** Varesh (ML / ANN & Research)

---

## 3. INTERFACE: DATABASE (PRANAV) ──► RAJDEEP (BACKEND / API)

### Current Known Information
- Rajdeep is building the FastAPI backend application in `backend/`.
- Backend must query PostgreSQL for location lookups, fetch recent observational context, invoke the ML model, and persist prediction records.
- Pranav will provide `database/connection.py`, `database/models.py`, and `database/repository.py`.

### Information Required from Rajdeep
1. `[NEEDS TEAM CONFIRMATION]` **SQLAlchemy Paradigm:** Synchronous session (`psycopg` / `psycopg2`) or asynchronous session (`asyncpg` with `AsyncSession`)?
2. `[NEEDS TEAM CONFIRMATION]` **Endpoint Latency Budget:** Target maximum execution time for database queries.
3. `[NEEDS TEAM CONFIRMATION]` **Connection Injection:** Agreement on standard FastAPI `Depends(get_db)`.

### Proposed Repository Interface `[PROPOSED]`
```python
# Proposed: database/repository.py interface for Rajdeep
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date
from database.models import Location, Prediction

class LocationRepository:
    @staticmethod
    def search(db: Session, query_str: str, limit: int = 10) -> List[Location]:
        """Autocomplete search for districts and states."""
        pass

    @staticmethod
    def get_by_id(db: Session, location_id: int) -> Optional[Location]:
        """Fetch location by internal surrogate ID."""
        pass

class PredictionRepository:
    @staticmethod
    def save(
        db: Session,
        location_id: int,
        target_date: date,
        model_id: int,
        dengue_prob: float,
        malaria_prob: float,
        flood_prob: float,
        heatwave_prob: float,
        dengue_tier: str,
        malaria_tier: str,
        flood_tier: str,
        heatwave_tier: str,
        feature_snapshot: Optional[dict] = None
    ) -> Prediction:
        """Persist generated prediction record to PostgreSQL."""
        pass
```

### Unresolved Questions
- Should the backend cache prediction queries in Redis/memory or rely on database index lookups on `(location_id, target_travel_date)`?

**Owner Responsible for Confirmation:** Rajdeep (Backend / API)

---

## 4. INTERFACE: VARESH (ML) ──► RAJDEEP (BACKEND / API)

### Current Known Information
- Varesh will deliver trained model weights in `ml/models/` alongside inference code.
- Rajdeep's FastAPI service will load the model in-memory at application startup (`@app.on_event("startup")` or lifespan context manager).
- When a user requests a prediction, backend builds the feature vector and calls the predictor function.

### Information Required from Varesh
1. `[NEEDS TEAM CONFIRMATION]` **Exact Feature Vector Schema:** The ordered list of features, expected data types, and normalizer/scaler object (`scaler.joblib`).
2. `[NEEDS TEAM CONFIRMATION]` **Inference Function Signature:** Exact Python function name and parameter types.
3. `[NEEDS TEAM CONFIRMATION]` **Model Version Tag:** Naming convention for the active model artifact.
4. `[NEEDS TEAM CONFIRMATION]` **Risk-Tier Thresholds:** Cutoff values to map numeric probabilities to categories (`Low`, `Moderate`, `Elevated`, `High`, `Severe`).

### Proposed ML Inference Contract `[PROPOSED]`

#### A. Input Feature Vector (Python Dictionary / Pydantic Schema) `[PROPOSED]`
```json
{
  "location_id": 142,
  "target_month": 8,
  "month_sin": -0.8660,
  "month_cos": -0.5000,
  "temp_max_historical_mean": 32.4,
  "rainfall_recent_14d_sum": 145.2,
  "humidity_historical_mean": 78.5,
  "dengue_historical_case_rate": 4.12,
  "malaria_historical_case_rate": 1.05,
  "elevation_m": 450.0
}
```

#### B. Output Prediction Schema `[PROPOSED EXAMPLE ONLY]`
> **CRITICAL RULE:** This is a preliminary example only. It is explicitly marked `PROPOSED` until Varesh approves or updates the final output contract.

```json
{
  "model_version": "ann_v1_202609",
  "dengue_probability": 0.0,
  "malaria_probability": 0.0,
  "flood_probability": 0.0,
  "heatwave_probability": 0.0,
  "risk_tiers": {
    "dengue": "Low",
    "malaria": "Low",
    "flood": "Low",
    "heatwave": "Low"
  }
}
```

### Unresolved Questions
- Does the ANN predict all 4 hazards through a single multi-output neural network head (`output_dim=4`), or through 4 separate single-hazard models?
- Are explainability factor weights (e.g. SHAP or permutation importance) returned dynamically with the prediction?

**Owner Responsible for Confirmation:** Varesh (ML / ANN & Research)

---

## 5. INTERFACE: RAJDEEP (BACKEND) ──► ROSEMARIYA (FRONTEND)

### Current Known Information
- Rosemariya is developing the Next.js web application in `frontend/`.
- The user selects a target district (destination) and a travel date.
- The UI must display independent risk levels for Dengue, Malaria, Flood, and Heatwave, along with historical reports.

### Information Required from Rajdeep & Rosemariya
1. `[NEEDS TEAM CONFIRMATION]` **REST API Endpoints:** Approved URL path names and query parameter formats.
2. `[NEEDS TEAM CONFIRMATION]` **Error Response Structure:** Standard error payload for invalid input, missing data, or unknown district.

### Proposed REST API Contracts `[PROPOSED]`

#### 1. Destination Search Endpoint
- **Request:** `GET /api/v1/locations?query=bengaluru`
- **Response:** `200 OK`
```json
[
  {
    "id": 142,
    "state": "Karnataka",
    "district": "Bengaluru Urban",
    "latitude": 12.9716,
    "longitude": 77.5946
  }
]
```

#### 2. Risk Prediction Endpoint
- **Request:** `POST /api/v1/predict`
```json
{
  "location_id": 142,
  "travel_date": "2026-10-15"
}
```
- **Response:** `200 OK`
```json
{
  "prediction_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "location": {
    "id": 142,
    "state": "Karnataka",
    "district": "Bengaluru Urban"
  },
  "travel_date": "2026-10-15",
  "model_version": "ann_v1_202609",
  "hazards": {
    "dengue": {
      "probability": 0.42,
      "risk_tier": "Elevated",
      "primary_factors": ["Post-monsoon season", "High humidity (78%)"]
    },
    "malaria": {
      "probability": 0.15,
      "risk_tier": "Low",
      "primary_factors": ["Low historical transmission rate"]
    },
    "flood": {
      "probability": 0.08,
      "risk_tier": "Low",
      "primary_factors": ["Normal rainfall forecast"]
    },
    "heatwave": {
      "probability": 0.02,
      "risk_tier": "Low",
      "primary_factors": ["Moderate temperatures (<30°C)"]
    }
  },
  "generated_at": "2026-09-13T16:50:00Z"
}
```

#### 3. Historical Surveillance Endpoint
- **Request:** `GET /api/v1/locations/142/history?year=2023`
- **Response:** `200 OK`
```json
{
  "location_id": 142,
  "district": "Bengaluru Urban",
  "annual_summary": {
    "mean_temperature_c": 24.5,
    "total_rainfall_mm": 980.2,
    "dengue_total_cases": 1240,
    "malaria_total_cases": 82,
    "flood_events_count": 1,
    "heatwave_days_count": 0
  }
}
```

### Unresolved Questions
- Does Rosemariya require GeoJSON boundaries directly from the backend, or will the frontend bundle a lightweight India district GeoJSON map file?

**Owners Responsible for Confirmation:** Rajdeep (Backend / API) & Rosemariya (Frontend / UI)
