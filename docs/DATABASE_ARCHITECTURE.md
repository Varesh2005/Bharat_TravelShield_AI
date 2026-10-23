# BHARAT-TRAVELSHIELD AI — DATABASE ARCHITECTURE SPECIFICATION

**Author:** Pranav (Role: Database + Integration + Documentation)  
**Project:** Bharat-TravelShield AI  
**Repository:** [Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI)  
**Document Status:** `PROPOSED & ARCHITECTURAL BASELINE`  
**Phase:** Phase 1 — Database Architecture & Integration Contracts  
**Last Updated:** September 13, 2026  

---

## A. PURPOSE OF THE DATABASE

Bharat-TravelShield AI is an India-first spatiotemporal travel health and natural disaster risk intelligence platform. The system evaluates and predicts independent risk probabilities across Indian districts for four distinct hazards:
1. **Dengue** (Vector-borne health hazard)
2. **Malaria** (Vector-borne health hazard)
3. **Flood** (Hydrological disaster hazard)
4. **Heatwave** (Meteorological disaster hazard)

The purpose of the database layer is to provide a robust, consistent, relational, and auditable data foundation that:
- Acts as the central spatiotemporal data store for official Indian geographical units (districts and states).
- Ingests and maintains historical climate/weather observations, public health disease surveillance records, and historical natural disaster occurrences.
- Permanently logs forward-looking inference queries, model outputs, and generated multi-hazard probabilities for accountability and reproducibility.
- Decouples raw feature engineering and offline training pipelines from live backend API serving.

---

## B. DATABASE RESPONSIBILITIES

### Core Responsibilities (`database/` Scope)
1. **Canonical Location Registry:** Maintain a single, immutable source of truth for Indian states and districts, preventing spelling conflicts and ID collisions across team modules.
2. **Historical Data Storage:** Securely store cleaned historical observations (weather, disease, disaster) delivered by Adhith (Data Engineering).
3. **Inference & Prediction Persistence:** Record user prediction queries, target travel dates, model versions, and individual hazard risk scores computed by the ANN/ML models.
4. **Model Registry Tracking:** Maintain version metadata, evaluation metrics, and artifact identifiers for all active and historical ML/ANN models deployed by Varesh.
5. **Data Provenance & Auditability:** Track upstream source organizations (e.g., IMD, NVBDCP/NCDC, NDMA/CWC) and data ingestion timestamps.
6. **Query Optimization:** Provide structured indexes for low-latency queries by Rajdeep's FastAPI backend (target: lookup response < 50ms).

### Explicit Non-Responsibilities (Out of Scope for Database)
- **Zero In-Database ML Computation:** The database will not compute machine learning predictions, matrix multiplications, neural network inferences, or complex feature scalers. These belong strictly in `ml/` (Varesh) and `backend/` (Rajdeep).
- **Zero Frontend Access:** The database will never be exposed to the internet or directly queried by Rosemariya's Next.js frontend (`frontend/`). All access is mediated through FastAPI.
- **Zero Data Scraping/Cleaning:** Raw web scraping and unstructured cleaning belong in `data/` (Adhith).

---

## C. SYSTEM ARCHITECTURE & INTEGRATION FLOW

### 1. High-Level Data & Model Execution Flow
```
   [DATA ENGINEERING: Adhith]
                ↓ (Canonical Processed Datasets)
    [ML / ANN MODELING: Varesh]
                ↓ (Packaged Model Artifacts + Inference Contract)
     [BACKEND / API: Rajdeep]
                ↓ (REST Endpoints / JSON Payloads)
       [FRONTEND: Rosemariya]
                ↓
             [USER]
```

### 2. Bidirectional Database ↔ Backend Interaction
```
                  ┌────────────────────────────────────────┐
                  │           FASTAPI BACKEND              │
                  │   Validation (Pydantic) | Endpoints    │
                  └──────────▲──────────────────▲──────────┘
                             │                  │
           1. Query Location │                  │ 4. Log Prediction
              & History      │                  │    & Metrics
                             ▼                  ▼
              ┌────────────────────────────────────────┐
              │          POSTGRESQL DATABASE           │
              │  - locations       - weather_obs       │
              │  - ml_models       - disease_surv      │
              │  - predictions     - disaster_events   │
              │  - data_sources                        │
              └────────────────────────────────────────┘
```

**Step-by-Step Flow:**
1. **User Request:** Frontend sends a query for a target destination (e.g., district ID / name) and target travel date.
2. **Location & Context Lookup:** Backend queries PostgreSQL (`locations`, `weather_observations`, `disease_surveillance`) to fetch location metadata and recent baseline trends.
3. **ML Inference:** Backend formats the feature vector and invokes Varesh's loaded ANN model in-memory.
4. **Prediction Persistence:** Backend records the query, inputs, independent probabilities, and model version in `predictions`.
5. **Client Response:** Backend returns the validated response to Frontend.

---

## D. PROPOSED ENTITIES

The schema is structured into seven distinct relational entities:

1. **`locations` (Master Spatial Reference):**  
   Represents Indian administrative units (States and Districts). All historical observations and predictions link to this table.
2. **`weather_observations` (Spatiotemporal Climate Data):**  
   Stores historical climate observations: temperature (min, max, mean), rainfall (precipitation), and relative humidity.
3. **`disease_surveillance` (Public Health Surveillance):**  
   Stores epidemiology surveillance records for Dengue and Malaria (case counts, deaths, incidence rates per 100k).
4. **`disaster_events` (Historical Extreme Natural Events):**  
   Stores recorded Flood and Heatwave disaster occurrences, severity levels, and duration.
5. **`ml_models` (Model Registry & Provenance):**  
   Maintains metadata for trained models (ANN/MLP vs. LightGBM baselines, version strings, training timestamps, validation metrics).
6. **`predictions` (Inference & Audit Trail):**  
   Logs user prediction requests, target travel dates, model IDs, and four independent risk probabilities ($[0.0, 1.0]$) alongside categorical risk tiers.
7. **`data_sources` (Agency & Provenance Registry):**  
   Records source agencies (e.g., IMD, NVBDCP, NDMA, CWC), fetch URLs, and sync metadata.

---

## E. RELATIONSHIPS BETWEEN ENTITIES

```
data_sources (1) ────< (N) weather_observations (N) >──── (1) locations
data_sources (1) ────< (N) disease_surveillance (N) >──── (1) locations
data_sources (1) ────< (N) disaster_events     (N) >──── (1) locations
                                                               │
                                                               │ 1
                                                               │
                                                               v N
ml_models    (1) ───────────────────────────────────< (N) predictions
```

- **`locations` 1 : N `weather_observations`:** Each district has multiple temporal weather observations.
- **`locations` 1 : N `disease_surveillance`:** Each district has multiple surveillance records for Dengue and Malaria.
- **`locations` 1 : N `disaster_events`:** Each district has multiple historical disaster events.
- **`locations` 1 : N `predictions`:** Each district has multiple logged prediction queries.
- **`ml_models` 1 : N `predictions`:** Each prediction log records the exact ML model version used.
- **`data_sources` 1 : N `weather_observations` / `disease_surveillance` / `disaster_events`:** Each observational record traces back to its originating agency.

---

## F. HISTORICAL DATA VS. PREDICTION DATA SEPARATION

A core architectural principle of Bharat-TravelShield AI is the strict physical separation between **ground-truth historical records** and **model-generated predictions**:

| Criterion | Historical Surveillance Data | Prediction Inference Data |
| :--- | :--- | :--- |
| **Tables** | `weather_observations`, `disease_surveillance`, `disaster_events` | `predictions` |
| **Nature of Data** | Empirical ground truth recorded by official agencies (IMD, NVBDCP). | Probabilistic mathematical estimation produced by an ANN model. |
| **Mutability** | Immutable once ingested and validated. | Append-only per user query (or read-through cache). |
| **Integrity Constraint** | Foreign keys link to `data_sources` and `locations`. | Foreign keys link to `ml_models` and `locations`. |
| **UI Presentation** | Displayed in "Historical Surveillance Reports & Trends" tab. | Displayed in "Active Travel Risk Intelligence" dashboard cards. |

> **Architectural Guardrail:** Historical case counts must never be commingled with predicted risk probabilities. A user viewing the UI must always be informed whether a data point represents a confirmed past event or a forecasted future probability.

---

## G. LOCATION IDENTITY STRATEGY

### Evaluation of Options
A major integration risk is district naming variation (e.g., "Bangalore" vs. "Bengaluru Urban", "Delhi Central" vs. "Central Delhi").

1. **Option 1: LGD (Local Government Directory) Code `[PROPOSED]`**  
   - *Pros:* Official Ministry of Panchayati Raj standard; unique integer assigned to every Indian state and district; stable across census boundary shifts.  
   - *Cons:* Requires Adhith to map raw dataset names to official LGD codes during preprocessing.
2. **Option 2: Census 2011 Code `[PROPOSED ALTERNATIVE]`**  
   - *Pros:* Widely used in academic datasets and historical health records.  
   - *Cons:* Outdated; does not include newly created districts post-2014 (e.g., in Telangana, Ladakh).
3. **Option 3: Internal Surrogate Primary Key (`id SERIAL`) `[CONFIRMED COMPONENT]`**  
   - *Pros:* Clean, auto-incrementing integer key for fast database foreign key joins.  
   - *Cons:* Internal only; meaningless to external datasets without a mapping table.

### Recommended Strategy `[PROPOSED]`
**Composite Identity Strategy:**  
- Primary Key: Internal `id SERIAL PRIMARY KEY` (integer) for internal FK efficiency.
- Official External Key: `lgd_code VARCHAR(20) UNIQUE` or `census_code VARCHAR(20) UNIQUE` (nullable if unassigned).
- Normalized Name: `district_name VARCHAR(100) NOT NULL` and `state_name VARCHAR(100) NOT NULL`.
- Unique Constraint: `UNIQUE(state_name, district_name)`.
- Search Alias Column: `normalized_key VARCHAR(100) NOT NULL` (e.g., `karnataka_bengaluru_urban`) for fuzzy/case-insensitive backend lookups.

> `[NEEDS TEAM CONFIRMATION]` **Decision Required from Adhith:** Adhith must confirm which standard identifier is present in the raw/canonical datasets.

---

## H. TEMPORAL DATA STRATEGY

### Evaluation of Granularities
1. **Daily Resolution (`observation_date DATE`):**  
   - Standard for meteorological data (IMD daily rainfall, max temp).
   - High storage requirement (~750 districts $\times$ 365 days $\times$ 5 years $\approx$ 1.3 million rows). PostgreSQL handles this with trivial overhead.
2. **Weekly / Epidemiological Week (`epi_week VARCHAR(10)` or `week_start_date DATE`):**  
   - Standard for NVBDCP malaria/dengue reports (often published by epi-week).
3. **Monthly Aggregation (`year_month DATE` or `month_start_date DATE`):**  
   - Common for compiled state-level health statistics and long-term climate normals.

### Recommended Strategy `[PROPOSED]`
- **Date Representation:** Use SQL `DATE` format (`YYYY-MM-DD`) for all time fields. Never store dates as freeform strings (`VARCHAR`) or integer epoch timestamps.
- **Polymorphic Time Window:** For observations with lower frequency (e.g., monthly health data), set `observation_date` to the first day of that period (`2024-07-01`) and specify duration in an optional `period_type` enum (`'DAILY'`, `'WEEKLY'`, `'MONTHLY'`).

> `[NEEDS TEAM CONFIRMATION]` **Decision Required from Adhith:** Adhith must confirm the temporal resolution of the raw and cleaned datasets.

---

## I. GEOGRAPHIC STRATEGY: POSTGIS VS. POINT COORDINATES

### Evaluation
1. **Point Coordinates (`latitude DECIMAL(9,6)`, `longitude DECIMAL(9,6)`) `[RECOMMENDED & PROPOSED]`:**  
   - *Suitability:* Sufficient if frontend map queries only need district centroid markers, pin placements, and radius lookups.
   - *Overhead:* Zero extra extensions; standard PostgreSQL numeric indexing (B-Tree).
2. **PostGIS Extension (`GEOMETRY(Point, 4326)` or `GEOMETRY(MultiPolygon, 4326)`) `[OPTIONAL RESEARCH EXTENSION]`:**  
   - *Suitability:* Necessary only if the user can click an arbitrary coordinate and the database must compute a spatial point-in-polygon containment query to identify the district boundary.
   - *Overhead:* Requires PostGIS extension, GeoJSON boundary files for all 750+ districts, and higher memory overhead.

### Architectural Recommendation
Start with standard **`DECIMAL(9,6)` latitude and longitude centroids** in the core schema. Design the table so that a PostGIS `geom GEOMETRY(Geometry, 4326)` column can be added via an `ALTER TABLE` migration in a future phase without breaking existing queries.

> `[NEEDS TEAM CONFIRMATION]` **Decision Required from Rosemariya & Rajdeep:** Does the frontend require vector polygon boundary clipping, or are centroid coordinates and district name matching sufficient?

---

## J. PREDICTION PERSISTENCE STRATEGY

Every inference generated for a user must be recorded in the `predictions` table:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          predictions TABLE                             │
├────────────────────────────────────────────────────────────────────────┤
│ id                      : UUID / BIGSERIAL                             │
│ location_id             : FK -> locations.id                           │
│ target_travel_date      : DATE                                         │
│ model_id                : FK -> ml_models.id                           │
│ dengue_probability      : NUMERIC(5,4)  -- [0.0000 to 1.0000]          │
│ malaria_probability     : NUMERIC(5,4)  -- [0.0000 to 1.0000]          │
│ flood_probability       : NUMERIC(5,4)  -- [0.0000 to 1.0000]          │
│ heatwave_probability    : NUMERIC(5,4)  -- [0.0000 to 1.0000]          │
│ dengue_tier             : VARCHAR(20)   -- Low/Moderate/High/etc.      │
│ malaria_tier            : VARCHAR(20)   -- Low/Moderate/High/etc.      │
│ flood_tier              : VARCHAR(20)   -- Low/Moderate/High/etc.      │
│ heatwave_tier           : VARCHAR(20)   -- Low/Moderate/High/etc.      │
│ created_at              : TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Rules
1. **Independent Hazards:** The four hazard probabilities are stored in distinct columns. No artificial blended score is computed or stored.
2. **Deterministic Model Linkage:** Every prediction points to a registered `model_id`, guaranteeing full auditability of which ANN weights generated the risk score.
3. **Caching & Query Deduplication:** If another user queries the same `(location_id, target_travel_date)` within a 24-hour cache TTL, the backend can optionally return the cached prediction row rather than re-evaluating the neural network.

---

## K. DATA PROVENANCE & AUDITABILITY

Every historical record traces back to the `data_sources` registry table:
- IMD (India Meteorological Department) → Weather
- NVBDCP / NCDC (National Vector Borne Disease Control Programme) → Dengue / Malaria
- NDMA / CWC (National Disaster Management Authority / Central Water Commission) → Floods / Heatwaves

Tables `weather_observations`, `disease_surveillance`, and `disaster_events` will include a foreign key `source_id REFERENCES data_sources(id)` and a timestamp `ingested_at TIMESTAMPTZ`.

---

## L. PROPOSED INDEXING STRATEGY

The following indexes are proposed to optimize query performance:

| Table | Proposed Index Columns | Index Type | Purpose | Status |
| :--- | :--- | :--- | :--- | :--- |
| `locations` | `(state_name, district_name)` | B-Tree / Unique | Fast lookup and duplicate prevention. | `PROPOSED` |
| `locations` | `(normalized_key)` | B-Tree | Fast API search/autocomplete lookups. | `PROPOSED` |
| `weather_observations` | `(location_id, observation_date)` | B-Tree / Composite | Optimizes historical window queries for ML feature extraction. | `PROPOSED` |
| `disease_surveillance` | `(location_id, report_date, disease_type)` | B-Tree / Composite | Fast filtering of disease time-series by district. | `PROPOSED` |
| `disaster_events` | `(location_id, event_date)` | B-Tree / Composite | Fast retrieval of disaster occurrence history. | `PROPOSED` |
| `predictions` | `(location_id, target_travel_date)` | B-Tree / Composite | Fast lookup for caching and analytics. | `PROPOSED` |
| `predictions` | `(created_at DESC)` | B-Tree | Fast retrieval of recent user activity logs. | `PROPOSED` |

---

## M. DATA INTEGRITY RULES

1. **Range Validation Constraints (CHECK):**
   - Probabilities: `CHECK (dengue_probability BETWEEN 0.0 AND 1.0)`, etc.
   - Coordinates: `CHECK (latitude BETWEEN 6.0 AND 38.0)` (India bounds), `CHECK (longitude BETWEEN 68.0 AND 98.0)`.
   - Humidity: `CHECK (humidity_pct BETWEEN 0.0 AND 100.0)`.
   - Rainfall: `CHECK (rainfall_mm >= 0.0)`.
2. **Referential Integrity:** All relational children use `ON DELETE RESTRICT` or `ON DELETE CASCADE` appropriately. Core reference entities like `locations` cannot be deleted if associated historical data or predictions exist.
3. **Non-Nullability:** Primary identifiers, location foreign keys, and observation dates are marked `NOT NULL`.

---

## N. SECURITY CONSIDERATIONS

1. **Zero Secret Storage in Git:** No database passwords, hostnames, or usernames will be committed to GitHub.
2. **Environment Variable Loading:** The database connection URI (`DATABASE_URL`) will be loaded exclusively via environment variables using `.env`.
3. **Role-Based PostgreSQL Access:**
   - App user (`bharat_app`): `SELECT`, `INSERT`, `UPDATE` permissions only.
   - Migration/Admin user (`bharat_admin`): DDL alteration permissions.
4. **Parameterized Queries:** All queries executed via SQLAlchemy ORM or repository layers will use parameter binding to prevent SQL injection vulnerabilities.

---

## O. SCALABILITY CONSIDERATIONS

1. **Estimated Storage Sizing:**
   - Districts: ~750 rows.
   - Weather: 750 districts $\times$ 365 days $\times$ 5 years $\approx$ 1.37M rows (~150 MB).
   - Disease: 750 districts $\times$ 52 weeks $\times$ 5 years $\times$ 2 diseases $\approx$ 390k rows (~40 MB).
   - Predictions: ~100k queries/year (~15 MB).
   - *Total Database Footprint:* < 500 MB. Fits easily within standard PostgreSQL configurations and micro-cloud tiers (Supabase free tier, Neon, or local Docker).
2. **Connection Pooling:** SQLAlchemy connection pool with `pool_size=10` and `max_overflow=20` to prevent connection exhaustion under concurrent API traffic.

---

## P. INTEGRATION WITH FASTAPI (RAJDEEP)

- **Dependency Injection:** Provide a standard `get_db()` generator yielding a scoped SQLAlchemy session:
  ```python
  def get_db():
      db = SessionLocal()
      try:
          yield db
      finally:
          db.close()
  ```
- **Repository Abstraction:** Rajdeep's API endpoints will call repository functions (e.g., `LocationRepository.get_by_id(db, id)`) rather than writing raw SQL strings in route handlers.

---

## Q. INTEGRATION WITH ML (VARESH)

- **Feature Retrieval:** For dynamic inference, backend queries PostgreSQL for the target district's recent weather metrics and historical seasonal averages.
- **Model Registration:** When Varesh outputs a newly trained ANN model, its metadata is registered in `ml_models` with its assigned `model_version` tag (e.g., `ann_v1_202609`).

---

## R. INTEGRATION WITH FRONTEND (ROSEMARIYA)

- **Endpoint Contracts:** The database feeds FastAPI endpoints that Rosemariya consumes:
  - `GET /api/v1/locations?q=delhi` (Autocompletes districts from `locations`).
  - `GET /api/v1/locations/{id}/history` (Fetches historical trends from `weather_observations` and `disease_surveillance`).
  - `POST /api/v1/predict` (Stores and returns the prediction result from `predictions`).
- **No Direct DB Access:** Frontend interacts solely with JSON REST endpoints.
