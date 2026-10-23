# BHARAT-TRAVELSHIELD AI — COMPREHENSIVE OFFICIAL REPOSITORY AUDIT REPORT

**Project:** Bharat-TravelShield AI (India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform)  
**Assigned Role:** Pranav — Database + Integration + Documentation  
**Assigned Scope:** `database/`, `docs/`, `README.md`, `.env.example`, `docker-compose.yml`, shared integration configuration  
**Audit Target:** Official Team Repository ([Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI))  
**Local Workspace Path:** `C:\backend ANN`  
**Git Commit Audited:** `85514f0b0a3f29a4b31b7ab4935d764c809cf41e`  
**Git Branch:** `main` (synchronized with `origin/main`, `origin/feature/data`, `origin/feature/ml`)  
**Audit Timestamp:** September 13, 2026, 16:50 IST  
**System Environment:** Windows 10/11, Python 3.14.7, Node.js v24.19.0, npm 11.17.0, Git 2.47.1.windows.1  

---

## 1. CURRENT PROJECT STRUCTURE

### 1.1 Physical Directory & File Manifest
Every file and directory in the repository was inspected recursively from the filesystem root and git index. The complete manifest is detailed below:

```
C:\backend ANN\
├── .git/                                # Git metadata directory (HEAD -> refs/heads/main)
├── .env.example                         # 2 Bytes  | 1 Line   | Empty placeholder
├── .gitignore                           # 1,760 B  | 108 L    | Multi-environment ignore file
├── README.md                            # 2 Bytes  | 1 Line   | Empty placeholder
├── backend/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
├── data/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
├── database/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
├── docs/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
├── frontend/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
├── ml/
│   └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
└── research/
    └── .gitkeep                         # 2 Bytes  | 1 Line   | Empty placeholder
```

### 1.2 Detailed Inspection of Every Existing File

#### A. Root Configuration Files
1. **`.gitignore`** (1,760 Bytes, 108 Lines)
   - **Content Analysis:** Comprehensive Git ignore rules authored by Varesh on Sep 9, 2026.
   - **Key Rules Established:**
     - *Python:* Ignores `__pycache__/`, `*.py[cod]`, `.venv/`, `venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`.
     - *Jupyter:* Ignores `.ipynb_checkpoints/`.
     - *Machine Learning Artifacts:* Ignores `*.pkl`, `*.pickle`, `*.joblib`, `*.h5`, `*.keras`, `*.onnx`, `ml/models/*` (retains `!ml/models/.gitkeep`), and `research/results/*` (retains `!research/results/.gitkeep`).
     - *Node.js / Next.js:* Ignores `node_modules/`, `.next/`, `out/`, `.turbo/`, `*.log`.
     - *Raw Datasets:* Explicitly prevents large raw dataset leakage via `data/raw/*` (retains `!data/raw/.gitkeep`) and `data/interim/*` (retains `!data/interim/.gitkeep`).
     - *Secrets:* Ignores `.env`, `.env.*` while explicitly keeping `!.env.example`.
2. **`.env.example`** (2 Bytes, 1 Line)
   - **Content Analysis:** Contains exactly a single newline (`\n`). No environment variables, port configurations, secret templates, or database URIs are defined.
3. **`README.md`** (2 Bytes, 1 Line)
   - **Content Analysis:** Contains exactly a single newline (`\n`). There is no project description, team roster, setup guide, architectural blueprint, or API documentation.

#### B. Subdirectory Gitkeep Placeholders
Each of the following files has a size of 2 Bytes and contains only a newline:
- `backend/.gitkeep`
- `data/.gitkeep`
- `database/.gitkeep`
- `docs/.gitkeep`
- `frontend/.gitkeep`
- `ml/.gitkeep`
- `research/.gitkeep`

#### C. Git History and Branches
- **Total Commits on `main`:** 10 commits (all created on Wed Sep 9 2026 by `Varesh Hudar`).
- **Remote Branches:**
  - `origin/main` (`85514f0`)
  - `origin/feature/data` (`85514f0`)
  - `origin/feature/ml` (`85514f0`)
- **Remote Divergence:** None. All three remote branches point to the identical initial setup commit. No pull requests or forks exist on GitHub.

---

## 2. IMPLEMENTATION STATUS OF EVERY MODULE

| Major Module | Assigned Member | Classification | Detailed Audit Findings |
| :--- | :--- | :--- | :--- |
| **Data (`data/`)** | Adhith | **Skeleton** | Only `.gitkeep` exists. No collection scripts (IMD, NVBDCP, NDMA), no data cleaning pipelines, no raw CSVs, and no canonical feature datasets exist in the repository. |
| **ML (`ml/`)** | Varesh | **Skeleton** | Only `.gitkeep` exists. No ANN/MLP model definitions, no training loops, no hyperparameter configs, no saved model artifacts (`.pt`, `.joblib`), and no inference functions exist. |
| **Research (`research/`)** | Varesh | **Skeleton** | Only `.gitkeep` exists. No baseline models (Logistic Regression, Random Forest, LightGBM), no HSTM-Net code, and no experimental notebooks exist. |
| **Backend (`backend/`)** | Rajdeep | **Skeleton** | Only `.gitkeep` exists. No FastAPI instance, no `main.py`, no route handlers, no Pydantic request/response schemas, and no backend dependencies exist. |
| **Frontend (`frontend/`)** | Rosemariya | **Skeleton** | Only `.gitkeep` exists. No Next.js application, no `package.json`, no Tailwind/CSS styling, and no UI components exist. |
| **Database (`database/`)** | Pranav | **Skeleton** | Only `.gitkeep` exists. No SQL files, no schema DDL, no Alembic migrations, no SQLAlchemy ORM models, and no seed scripts exist. |
| **Documentation (`docs/`, `README.md`)** | Pranav | **Skeleton** | Both `README.md` and `docs/` are empty placeholders. |
| **Docker / Containers** | Pranav | **Not Started** | No `Dockerfile`, `docker-compose.yml`, or container configuration exists anywhere in the repository. |
| **Testing Suite** | Team / Pranav | **Not Started** | No `tests/` directory, no `pytest.ini`, and zero unit/integration tests exist. |

---

## 3. CURRENT DATABASE STATUS

1. **Does a database already exist?**  
   **No.** There is no active local database, embedded database file (e.g., SQLite), or cloud-hosted database connection.
2. **Which database technology is currently configured?**  
   **None in code.** However, the team specification mandates **PostgreSQL** (with PostGIS evaluated for spatial requirements).
3. **Does `database/` already exist?**  
   **Yes**, as an empty directory holding only `.gitkeep`.
4. **Are SQL files present?**  
   **No.** There are zero `.sql` DDL, DML, or migration files.
5. **Are migrations present?**  
   **No.** No migration tooling (such as Alembic or Flyway) is initialized.
6. **Are ORM models present?**  
   **No.** No SQLAlchemy, SQLModel, Tortoise-ORM, or Django models exist.
7. **Are database schemas already defined?**  
   **No.** No schema definition exists in code or documentation.
8. **Are seed scripts present?**  
   **No.** No scripts exist to populate Indian districts, states, coordinates, or baseline historical records.
9. **Does backend already connect to a database?**  
   **No.** Backend has no application code and no database drivers installed or configured.
10. **What tables/models currently exist?**  
    **Zero tables.**
11. **What is missing?**  
    Everything in the database layer:
    - PostgreSQL schema DDL script (`database/schema.sql`).
    - Python ORM models (`database/models.py`).
    - Engine initialization and session factory (`database/connection.py`).
    - Data repository/DAO layer (`database/repository.py`).
    - Seeding scripts for India districts and baseline data (`database/seeds/`).
    - Database configuration and environment variables in `.env.example`.

---

## 4. DATABASE REQUIREMENTS

The Bharat-TravelShield AI platform requires an intelligent spatiotemporal data store to support multi-hazard risk predictions across Indian districts for **Dengue**, **Malaria**, **Flood**, and **Heatwave**.

### 4.1 Required Database Entities

```
                          ┌───────────────────────┐
                          │       locations       │
                          │ (Districts & States)  │
                          └───────────┬───────────┘
                                      │ 1
                 ┌────────────────────┼────────────────────┐
                 │ 1:N                │ 1:N                │ 1:N
                 ▼                    ▼                    ▼
    ┌─────────────────────────┐  ┌────────────────┐  ┌────────────────────┐
    │  weather_observations   │  │    disease_    │  │  disaster_events   │
    │ (Temp, Rain, Humidity)  │  │  surveillance  │  │ (Flood, Heatwave)  │
    └─────────────────────────┘  └────────────────┘  └────────────────────┘
                                                           ▲
                                                           │
                                      ┌────────────────────┼────────────────────┐
                                      │ 1:N                                     │
                                      ▼                                         │
                          ┌───────────────────────┐                             │
                          │      predictions      │                             │
                          │ (Risk Probabilities)  │                             │
                          └───────────▲───────────┘                             │
                                      │ N:1                                     │
                          ┌───────────┴───────────┐                             │
                          │       ml_models       │                             │
                          │  (Metadata & Version) │                             │
                          └───────────────────────┘                             │
                                      │ N:1                                     │
                          ┌───────────┴───────────┐                             │
                          │     data_sources      │─────────────────────────────┘
                          │ (IMD, NVBDCP, NDMA)   │
                          └───────────────────────┘
```

1. **`locations` (Master Spatial Entity):**  
   Stores geographic boundaries and centers for Indian states and districts (`id`, `state_name`, `district_name`, `latitude`, `longitude`, `elevation_m`, `climate_zone`, `created_at`).
2. **`weather_observations` (Spatiotemporal Climate Entity):**  
   Stores historical climate observations (`id`, `location_id`, `observation_date`, `temp_max_c`, `temp_min_c`, `temp_mean_c`, `rainfall_mm`, `humidity_pct`, `pressure_hpa`, `source_id`).
3. **`disease_surveillance` (Health Intelligence Entity):**  
   Stores health records for vector-borne diseases (`id`, `location_id`, `report_date`, `disease_type` [Dengue/Malaria], `cases_reported`, `deaths_reported`, `incidence_rate_per_100k`, `source_id`).
4. **`disaster_events` (Disaster History Entity):**  
   Stores historical extreme natural events (`id`, `location_id`, `event_date`, `disaster_type` [Flood/Heatwave], `severity_tier`, `affected_population`, `source_id`).
5. **`ml_models` (Model Registry Entity):**  
   Tracks ML/ANN model versions (`id`, `model_name`, `model_version`, `architecture_type` [ANN/LightGBM], `trained_on_date`, `metrics_json`, `artifact_path`, `is_active`).
6. **`predictions` (Inference Log Entity):**  
   Stores user prediction queries and generated probabilities (`id`, `location_id`, `target_travel_date`, `dengue_probability`, `malaria_probability`, `flood_probability`, `heatwave_probability`, `dengue_tier`, `malaria_tier`, `flood_tier`, `heatwave_tier`, `model_id`, `created_at`).
7. **`data_sources` (Provenance Entity):**  
   Tracks upstream agencies (`id`, `source_name` [IMD/NVBDCP/NDMA], `source_url`, `last_synced_at`).

### 4.2 Confirmed Requirements vs. Items Needing Team Confirmation

| Area | Confirmed Requirements | Needs Team Confirmation |
| :--- | :--- | :--- |
| **Geographic Resolution** | Must support all Indian states and districts with latitude and longitude centroids. | Whether PostGIS polygon boundary geometry (`GEOMETRY(Polygon, 4326)`) is required or if point coordinates (`DECIMAL(9,6)`) are sufficient. |
| **Temporal Frequency** | System must support travel date queries across seasons. | Whether weather and disease records are stored at daily, weekly, or monthly granularity. |
| **Risk Hazards** | Exactly 4 hazards: Dengue, Malaria, Flood, Heatwave. | Numerical cutoffs for categorical tiers: Low (<0.20), Moderate (0.20-0.40), Elevated (0.40-0.60), High (0.60-0.80), Severe (>=0.80). |
| **Data Separation** | Historical ground-truth surveillance must be stored strictly separate from model predictions. | Retention policy for user prediction queries (e.g., permanent vs. 90-day caching). |
| **Database Engine** | PostgreSQL is the chosen technology. | Hosting environment (local PostgreSQL container vs. cloud instance like Supabase/Neon). |

---

## 5. INTEGRATION AUDIT: DATA → ML → BACKEND → FRONTEND

```
[DATA: Adhith]       ──(Broken)──>  [ML / ANN: Varesh]
                                           │
                                        (Broken)
                                           ▼
[DATABASE: Pranav]  <──(Broken)──>  [BACKEND: Rajdeep]  ──(Broken)──>  [FRONTEND: Rosemariya]
```

### 5.1 Layer-by-Layer Flow Analysis
1. **Data Layer (`data/`) → ML Layer (`ml/`):**  
   - **Current State:** **COMPLETELY BROKEN (Not Implemented).**  
   - **Gap:** `data/` has no datasets, and `ml/` has no training scripts. There is no agreed data format (CSV, Parquet) or column schema for training the ANN.
2. **ML Layer (`ml/`) → Backend Layer (`backend/`):**  
   - **Current State:** **COMPLETELY BROKEN (Not Implemented).**  
   - **Gap:** No serialized model artifact (`.pt`, `.joblib`, `.onnx`) exists. The backend has no predictor module to import and no defined feature preprocessing function.
3. **Backend Layer (`backend/`) → Frontend Layer (`frontend/`):**  
   - **Current State:** **COMPLETELY BROKEN (Not Implemented).**  
   - **Gap:** Backend has no running FastAPI routes (e.g., `POST /api/v1/predict`, `GET /api/v1/locations`). Frontend has no API service or client to fetch live data.

---

## 6. INTEGRATION AUDIT: DATABASE ↔ BACKEND

### 6.1 Current Database ↔ Backend Connection
- **Current State:** **COMPLETELY BROKEN (Not Implemented).**
- **Findings:**
  - `backend/` contains no code; no FastAPI `app` exists.
  - No database connection pool or session manager exists (`database/connection.py`).
  - No database dependencies (`SQLAlchemy`, `asyncpg`, `psycopg2`) are specified in any backend requirements file.
  - No environment configuration loader exists in `backend/` to read `DATABASE_URL`.
  - Zero SQL queries or repository functions are available for backend routes to consume.

---

## 7. TEAM INTERFACE & DEPENDENCY ANALYSIS

Pranav sits at the center of the project architecture. The table below specifies exactly what Pranav requires from each teammate and what Pranav delivers to them:

```
                      ┌──────────────────────┐
                      │    Adhith (Data)     │
                      └──────────┬───────────┘
                                 │ Clean Datasets & Schemas
                                 ▼
┌─────────────────────┐  Schema  ┌──────────────────────┐  Feature Specs ┌──────────────────────┐
│  Rosemariya (UI)    │◄─────────┤   Pranav (DB/Int)    ├───────────────►│    Varesh (ML)       │
└─────────────────────┘          └──────────┬───────────┘                └──────────────────────┘
                                 │ DB Session & Queries
                                 ▼
                      ┌──────────────────────┐
                      │   Rajdeep (Backend)  │
                      └──────────────────────┘
```

### 7.1 Interface with Adhith (Data Engineering & Dataset)
- **What Pranav Needs from Adhith:**
  - `[NEEDS CONFIRMATION]` **Standardized Location Master:** Exact list of Indian districts and states with consistent naming (to prevent mismatches like "Bengaluru" vs. "Bangalore").
  - `[NEEDS CONFIRMATION]` **Dataset Schema & Types:** Column names, data types, and units for weather (Celsius, mm, %) and disease data (case counts vs. incidence rates).
  - `[NEEDS CONFIRMATION]` **Date Format:** Exact temporal format (ISO-8601 `YYYY-MM-DD` preferred).
  - `[NEEDS CONFIRMATION]` **Data Delivery Format:** Clean canonical CSV or Parquet files delivered to `data/processed/` for database seeding.
- **What Pranav Delivers to Adhith:**
  - Seed loader scripts (`database/seeds/seed_historical_data.py`) to ingest canonical datasets into PostgreSQL.
  - Database schema documentation (`docs/DATA_DICTIONARY.md`).

### 7.2 Interface with Varesh (ML / ANN & Research)
- **What Pranav Needs from Varesh:**
  - `[NEEDS CONFIRMATION]` **Model Feature Vector:** The exact ordered list of input features expected by the ANN (e.g., `temp_mean`, `rainfall_7d_sum`, `humidity_mean`, `month_sin`, `month_cos`, `disease_lag_1m`).
  - `[NEEDS CONFIRMATION]` **Prediction Output Format:** The exact data structure returned by the model:
    ```json
    {
      "dengue_probability": 0.35,
      "malaria_probability": 0.12,
      "flood_probability": 0.05,
      "heatwave_probability": 0.82
    }
    ```
  - `[NEEDS CONFIRMATION]` **Risk Category Cutoffs:** The threshold mapping values from probabilities to risk tiers (Low, Moderate, Elevated, High, Severe).
  - `[NEEDS CONFIRMATION]` **Model Packaging:** Serialized model file (`.pt` or `.joblib`) and a standalone Python inference function: `predict(features: dict) -> dict`.
- **What Pranav Delivers to Varesh:**
  - `predictions` and `ml_models` database table structures.
  - Historical weather and surveillance query scripts so Varesh can extract training feature matrices directly from PostgreSQL if needed.

### 7.3 Interface with Rajdeep (Backend / API)
- **What Pranav Needs from Rajdeep:**
  - `[NEEDS CONFIRMATION]` **ORM & Driver Preference:** Agreement on SQLAlchemy 2.0 with `psycopg` (sync) or `asyncpg` (async).
  - `[NEEDS CONFIRMATION]` **Required Database Queries:** Specific endpoints requiring DB integration:
    - Autocomplete destination lookup: `search_locations(query: str)`
    - Location detail and coordinates: `get_location(location_id: int)`
    - Historical trend retrieval: `get_historical_metrics(location_id: int, start_date, end_date)`
    - Inference logging: `save_prediction(prediction_data: dict)`
- **What Pranav Delivers to Rajdeep:**
  - Modular database connection factory and FastAPI dependency: `get_db()`.
  - Compiled SQLAlchemy ORM models (`database/models.py`).
  - High-level repository functions (`database/repository.py`).
  - Database error handling guidance and connection pooling configuration.

### 7.4 Interface with Rosemariya (Frontend / UI)
- **What Pranav Needs from Rosemariya:**
  - `[NEEDS CONFIRMATION]` **Location Display Preferences:** Free-text search vs. two-stage State/District dropdowns.
  - `[NEEDS CONFIRMATION]` **Dashboard Data Requirements:** Confirmation that risk cards display individual risk probabilities alongside categorical tiers.
- **What Pranav Delivers to Rosemariya:**
  - Clear architectural documentation showing how user search triggers backend queries and returns database-backed historical and predicted data.
  - Verification that the UI clearly separates historical surveillance data from predicted future risks.

---

## 8. INTEGRATION BLOCKERS

| Severity | Blocker Description | System Impact | Mitigation / Resolution |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | **No Canonical Location Identifier Standard** | If Adhith uses one district spelling, Varesh trains on another, and Rajdeep's API receives a third, all database joins and predictions will fail. | Pranav will establish `database/seeds/locations.csv` with official LGD/Census codes as the project's single source of truth. |
| **CRITICAL** | **Undefined ML Inference Contract** | Backend cannot implement `POST /predict` without knowing the exact model input features and output structure. | Varesh and Pranav must define a locked Pydantic contract: `PredictionInputSchema` and `PredictionOutputSchema`. |
| **HIGH** | **Temporal Resolution Mismatch** | If weather is daily, disease surveillance is monthly, and user queries a specific calendar day, feature generation could fail. | Define standard aggregation rules (e.g., rolling 30-day weather statistics mapped to query dates). |
| **HIGH** | **Local System Lacks PostgreSQL & Docker** | Pranav's local machine does not have `psql` or `docker` installed in the system PATH. | Utilize a remote PostgreSQL instance (Supabase/Neon) via `DATABASE_URL` in `.env`, or install local PostgreSQL tools before running DDL. |
| **MEDIUM** | **Risk of Mock Data Leaking into Production UI** | Hardcoded JSON probabilities used during early frontend/backend prototyping could accidentally remain in the final viva demo. | Enforce database persistence for all predictions, and attach a `model_version` metadata tag to every API response. |
| **LOW** | **Shared File Merge Conflicts** | Multiple developers concurrently editing `README.md` or `.env.example` will produce Git conflicts across branches. | Pranav strictly owns project-level configuration files; teammates submit configuration requests to Pranav. |

---

## 9. PRANAV'S COMPLETE IMPLEMENTATION PLAN

### Phase 1 — Database Architecture & Data Dictionary
- **Objective:** Finalize the entity-relationship design, naming standards, and data dictionary.
- **Prerequisites:** Completion of this audit; receipt of location naming and feature lists.
- **Files Involved:** `docs/DATABASE_ARCHITECTURE.md`, `docs/DATA_DICTIONARY.md`.
- **Dependencies on Teammates:** Adhith (data attributes), Varesh (target labels).
- **Expected Output:** Comprehensive architectural specification document.
- **Testing Required:** Review document against team work distribution requirements.

### Phase 2 — Relational Schema & DDL Design
- **Objective:** Create PostgreSQL DDL scripts defining all tables, primary keys, foreign keys, constraints, and indexes.
- **Prerequisites:** Phase 1 sign-off.
- **Files Involved:** `database/schema.sql`, `docs/ER_DIAGRAM.md`.
- **Dependencies on Teammates:** Rajdeep (query pattern alignment).
- **Expected Output:** Valid SQL script creating `locations`, `weather_observations`, `disease_surveillance`, `disaster_events`, `ml_models`, and `predictions`.
- **Testing Required:** Syntax verification against PostgreSQL 15+.

### Phase 3 — Database Connection & Environment Management
- **Objective:** Build the database engine, session factory, and environment variable loaders.
- **Prerequisites:** PostgreSQL database instance available (local or cloud).
- **Files Involved:** `.env.example`, `database/config.py`, `database/connection.py`.
- **Dependencies on Teammates:** None.
- **Expected Output:** Reusable engine and session manager loading credentials from `.env`.
- **Testing Required:** Execution of a connection test script asserting successful database ping and connection pool health.

### Phase 4 — ORM Implementation & Migrations
- **Objective:** Implement Python ORM models and set up migration tracking.
- **Prerequisites:** Phase 2 and Phase 3 complete.
- **Files Involved:** `database/models.py`, `database/session.py`, `alembic.ini`, `database/migrations/*`.
- **Dependencies on Teammates:** Rajdeep (backend dependency alignment).
- **Expected Output:** Fully typed SQLAlchemy models mapping directly to `schema.sql`.
- **Testing Required:** Automated migration test: `alembic upgrade head` and schema validation.

### Phase 5 — Data Seeding & Ingestion Pipelines
- **Objective:** Ingest canonical Indian district master records and historical baseline data into PostgreSQL.
- **Prerequisites:** Adhith provides canonical CSV datasets in `data/`.
- **Files Involved:** `database/seeds/seed_locations.py`, `database/seeds/seed_historical_data.py`.
- **Dependencies on Teammates:** Adhith (data format compliance).
- **Expected Output:** Populated database with 700+ Indian districts and initial historical records.
- **Testing Required:** Row count assertions, null checks on required fields, foreign key integrity checks.

### Phase 6 — Backend ↔ Database Integration
- **Objective:** Provide Rajdeep with ready-to-use database session dependencies and query functions.
- **Prerequisites:** Phase 4 and Phase 5 complete.
- **Files Involved:** `database/repository.py`, `backend/dependencies.py`.
- **Dependencies on Teammates:** Rajdeep (FastAPI routing).
- **Expected Output:** Clean data access functions: `search_locations()`, `get_location()`, `get_historical_risk()`, `save_prediction()`.
- **Testing Required:** Unit tests verifying repository query return formats and execution times (<50ms).

### Phase 7 — ML ↔ Backend ↔ Database Integration
- **Objective:** Wire the end-to-end prediction pipeline: Backend receives location/date → fetches historical context from DB → passes feature vector to Varesh's ANN model → receives 4 risk probabilities → logs prediction to DB → returns response.
- **Prerequisites:** Varesh provides packaged model in `ml/models/` and inference wrapper.
- **Files Involved:** `backend/services/prediction_service.py`.
- **Dependencies on Teammates:** Varesh (model artifact), Rajdeep (FastAPI endpoint).
- **Expected Output:** Functional prediction pipeline recording every user query and model output in PostgreSQL.
- **Testing Required:** Integration test asserting realistic risk probabilities across Dengue, Malaria, Flood, and Heatwave.

### Phase 8 — Multi-Container Docker Environment
- **Objective:** Build a unified multi-container configuration orchestrating PostgreSQL, FastAPI Backend, and Next.js Frontend.
- **Prerequisites:** Individual modules have initial code and Dockerfiles.
- **Files Involved:** `docker-compose.yml`, `.dockerignore`.
- **Dependencies on Teammates:** Rajdeep (backend Dockerfile), Rosemariya (frontend Dockerfile).
- **Expected Output:** Single-command system startup via `docker compose up --build`.
- **Testing Required:** Container healthchecks and service-to-service networking verification.

### Phase 9 — End-to-End Integration Testing
- **Objective:** Validate complete user travel scenarios from UI input to database verification.
- **Prerequisites:** All 5 modules integrated on a shared staging/integration branch.
- **Files Involved:** `tests/test_integration_e2e.py`.
- **Dependencies on Teammates:** All team members.
- **Expected Output:** Automated integration test suite passing 100% of defined travel risk scenarios.
- **Testing Required:** E2E test verifying: UI Search → Backend API → Database Lookup → ANN Inference → Result Display.

### Phase 10 — Project Documentation & Viva Preparation
- **Objective:** Author comprehensive project documentation, viva presentation materials, and architecture diagrams.
- **Prerequisites:** Successful Phase 9 completion.
- **Files Involved:** `README.md`, `docs/ARCHITECTURE.md`, `docs/INTEGRATION_GUIDE.md`.
- **Dependencies on Teammates:** Contribution summaries.
- **Expected Output:** Professional, exhaustive documentation detailing system architecture, setup instructions, and evaluation results.
- **Testing Required:** Clean clone test verifying that a new developer can set up and run the project following `README.md`.

---

## 10. FILES TO CREATE

| File Path | Component | Purpose & Contents |
| :--- | :--- | :--- |
| `database/schema.sql` | Database | Core PostgreSQL DDL: table definitions, primary keys, foreign keys, unique constraints, and B-Tree indexes. |
| `database/connection.py` | Database | SQLAlchemy engine initialization, sessionmaker factory, connection pooling settings, and healthcheck function. |
| `database/models.py` | Database | Declarative SQLAlchemy ORM models corresponding to `schema.sql`. |
| `database/repository.py` | Database | High-level data access functions for location searching, history retrieval, and prediction logging. |
| `database/seeds/seed_locations.py` | Database | Script to seed official Indian states, districts, and coordinates. |
| `database/seeds/seed_historical_data.py` | Database | Script to ingest Adhith's processed canonical datasets. |
| `docs/DATABASE_ARCHITECTURE.md` | Docs | Comprehensive documentation of database design, entity relationships, and indexing strategy. |
| `docs/DATA_DICTIONARY.md` | Docs | Exhaustive dictionary of all table fields, units, nullability, and descriptions. |
| `docs/ER_DIAGRAM.md` | Docs | Mermaid entity-relationship diagram of all tables. |
| `docs/INTEGRATION_CONTRACTS.md` | Docs | Locked JSON schemas and interface contracts between Data, ML, Backend, Frontend, and DB. |
| `docker-compose.yml` | Orchestration | Multi-container setup for PostgreSQL, FastAPI, and Next.js (created in Phase 8). |
| `tests/test_database.py` | Testing | Automated tests for DB connection, schema integrity, and repository queries. |

---

## 11. FILES TO MODIFY

| File Path | Current Content | Required Modifications |
| :--- | :--- | :--- |
| `.env.example` | 2 Bytes (Empty newline) | Populate with full configuration template: `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`, `API_PORT`, `MODEL_PATH`, `ENVIRONMENT`. |
| `README.md` | 2 Bytes (Empty newline) | Write full project README: problem statement, system architecture diagram, team member roles and ownership, setup instructions, API documentation, and running instructions. |

---

## 12. FILES TO LEAVE UNTOUCHED

| Directory / File | Assigned Owner | Isolation Rationale |
| :--- | :--- | :--- |
| `data/*` | **Adhith** | Adhith is solely responsible for scraping, cleaning, and preprocessing data. Pranav must not edit data collection scripts or raw files. |
| `ml/*` | **Varesh** | Varesh owns the ANN/MLP neural network architecture, feature engineering, and model training. Pranav must not modify model internals. |
| `research/*` | **Varesh** | Varesh owns baseline comparisons (Random Forest, LightGBM) and research experiments. |
| `backend/*` | **Rajdeep** | Rajdeep owns the FastAPI application logic and route handlers. Pranav provides database dependencies (`get_db`) but does not rewrite backend code. |
| `frontend/*` | **Rosemariya** | Rosemariya owns the Next.js UI, styles, and page layouts. Pranav must not modify frontend code. |
| `.gitignore` | Shared / Varesh | Already well-configured; preserve existing rules unless a new tool specifically requires an update. |

---

## 13. ARCHITECTURAL RISKS

1. **Inconsistent Location Identifiers (Critical Risk):**  
   If Adhith's datasets use one spelling (`"Bengaluru Urban"`), Varesh's ML training uses another (`"Bangalore"`), and Rosemariya's UI sends `"bengaluru-urban"`, queries and joins will fail silently. Establishing a single canonical `locations` table with unique integer IDs and standardized aliases is essential.
2. **Inconsistent Date/Time Formats (High Risk):**  
   Frontend date pickers produce ISO-8601 strings (`2026-10-15`). If historical datasets store dates as integers (`20261015`) or strings (`15/10/2026`), database index scans will fail. The database must enforce strict SQL `DATE` types.
3. **ML Feature Drift and Dimension Mismatch (Critical Risk):**  
   If Varesh updates the model to expect 18 features but backend feature-building logic passes 17 features, inference will crash with a runtime shape mismatch exception (`ValueError`). A strict schema validation layer must be implemented.
4. **Historical vs. Predicted Data Confusion (High Risk):**  
   Users and college evaluators will reject the system if past disease case numbers are displayed as future predictions. The database and backend API must strictly isolate historical ground-truth tables from model prediction tables.
5. **Hardcoded Mock Data Leakage (Medium Risk):**  
   Developers often insert mock JSON responses (`{"dengue": 0.5}`) to test UI components. If not systematically replaced by database queries and ANN inference, the final project presentation will fail scrutiny.
6. **Secret Leakage & Missing Environment Variables (Medium Risk):**  
   Hardcoded database passwords or API keys committed to Git can compromise security and cause failures across different team members' machines. All credentials must be loaded exclusively from `.env`.

---

## 14. RECOMMENDED ARCHITECTURE

```
                  ┌────────────────────────────────────────┐
                  │          NEXT.JS FRONTEND              │
                  │   Destination Search | Risk Cards      │
                  │   Historical Reports | Explanations    │
                  └───────────────────▲────────────────────┘
                                      │ HTTP / JSON
                                      ▼
                  ┌────────────────────────────────────────┐
                  │           FASTAPI BACKEND              │
                  │   Validation (Pydantic) | Endpoints    │
                  └──────────▲──────────────────▲──────────┘
                             │                  │
               SQL Queries / │                  │ In-Memory Feature Vector
               ORM Sessions  ▼                  ▼
  ┌─────────────────────────────┐   ┌─────────────────────────────┐
  │     POSTGRESQL DATABASE     │   │      ANN / ML SERVICE       │
  │  - locations                │   │  - PyTorch / LightGBM       │
  │  - weather_observations     │   │  - Leakage-safe Scaler      │
  │  - disease_surveillance     │   │  - Predictor Interface      │
  │  - disaster_events          │   │  - Feature Extraction       │
  │  - predictions              │   └──────────────▲──────────────┘
  │  - ml_models                │                  │
  └──────────────▲──────────────┘                  │ Model Artifacts
                 │ ETL Batch Ingestion             │
  ┌──────────────┴──────────────┐                  │
  │     DATA PIPELINE (DATA/)   │──────────────────┘
  │  - IMD Weather Data         │ Clean Model-Ready
  │  - NVBDCP Disease Data      │ Feature Sets
  │  - Disaster History         │
  └─────────────────────────────┘
```

### Architectural Rules:
1. **PostgreSQL as the Single Source of Truth:** Locations, historical surveillance, and prediction logs reside permanently in PostgreSQL.
2. **FastAPI as the Stateless Orchestrator:** Backend handles request validation, queries PostgreSQL for location and historical features, invokes the loaded ANN model, saves the prediction result back to PostgreSQL, and returns the response to Next.js.
3. **Pydantic Shared Contracts:** All data crossing service boundaries (DB → Backend → ML → Frontend) must conform to strict Pydantic schemas.
4. **Clean Role Separation:** No ML math in the backend; no direct database calls from the frontend; no database operations inside the ML training pipeline.

---

## 15. FINAL VERDICT

### A. What is Already Good
- The repository structure is well-organized with clean, purposeful top-level directories (`backend/`, `data/`, `database/`, `docs/`, `frontend/`, `ml/`, `research/`).
- The `.gitignore` is comprehensive, professional, and pre-configured for Python, Next.js, ML models, and raw datasets.
- The team work distribution document clearly delineates responsibilities across all 5 members.

### B. What is Weak
- All 7 subdirectories are empty skeletons containing only `.gitkeep`.
- Root configuration files (`.env.example` and `README.md`) are empty 2-byte placeholders.
- No dependency management files (`requirements.txt`, `package.json`) exist.

### C. What is Missing
- The entire database layer: DDL schema, ORM models, migration tools, database connection logic, and seed datasets.
- Backend API endpoints, ML model binaries, data scraping code, and frontend UI components.
- Containerization files (`Dockerfile`, `docker-compose.yml`) and automated testing suites.

### D. Biggest Integration Risk
**Inconsistent Location Identifiers:** If Adhith's datasets, Varesh's training matrices, and Rajdeep's API routes use different district names or codes, the application cannot connect data to predictions.

### E. First Thing Pranav Should Implement
Draft and commit **`docs/DATABASE_ARCHITECTURE.md`** and **`database/schema.sql`**, establishing the standardized `locations` table and multi-hazard data schema.

### F. Information That Must Be Obtained from Teammates Before Database Implementation
1. **From Adhith:** Canonical location naming convention (LGD code, state, district) and temporal resolution (daily vs. monthly).
2. **From Varesh:** Confirmed prediction output format (individual probabilities for Dengue, Malaria, Flood, Heatwave) and categorical risk threshold values.
3. **From Rajdeep:** Preferred ORM framework (SQLAlchemy 2.0 with sync vs. asyncpg) and primary API query access patterns.

---

## 16. EXACT FIRST IMPLEMENTATION TASK

Upon receiving your authorization to begin implementation, Pranav's exact first task will be:

> **TASK 1: Core Database Architecture & PostgreSQL Schema Creation**
> 1. Create `docs/DATABASE_ARCHITECTURE.md` and `docs/DATA_DICTIONARY.md`, specifying the complete schema design and data contracts.
> 2. Author `database/schema.sql` containing production-ready PostgreSQL DDL for all 6 core tables (`locations`, `weather_observations`, `disease_surveillance`, `disaster_events`, `ml_models`, `predictions`) with indexes, constraints, and foreign keys.
> 3. Populate `.env.example` with the complete set of required database and environment configuration variables.
> 4. Author `database/connection.py` with the SQLAlchemy 2.0 engine, connection pool, and session factory.
