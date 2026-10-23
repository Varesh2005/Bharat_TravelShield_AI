# BHARAT-TRAVELSHIELD AI — PRELIMINARY DATA DICTIONARY

**Author:** Pranav (Role: Database + Integration + Documentation)  
**Project:** Bharat-TravelShield AI  
**Repository:** [Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI)  
**Document Status:** `PRELIMINARY ARCHITECTURAL BASELINE`  
**Phase:** Phase 1 — Database Architecture & Integration Contracts  
**Last Updated:** September 13, 2026  

---

### Status Legend
- **`CONFIRMED`**: Explicitly specified in the official project charter / team work distribution requirements.
- **`PROPOSED`**: Architecturally sound standard proposed by Pranav; ready for implementation upon team alignment.
- **`NEEDS TEAM CONFIRMATION`**: Requires explicit schema confirmation from the owning teammate (Adhith, Varesh, or Rajdeep) before finalizing DDL.

---

## 1. TABLE: `locations`
**Description:** Master reference table containing all Indian administrative spatial entities (States, Union Territories, and Districts).  
**Ownership / Source:** Pranav (DB) & Adhith (Data Engineering)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `SERIAL` | No | Yes | No | Internal surrogate primary key | - | System Generated | `CONFIRMED` |
| `state_name` | `VARCHAR(100)` | No | No | No | Name of Indian State or Union Territory | - | Official Indian Registry | `CONFIRMED` |
| `district_name` | `VARCHAR(100)` | No | No | No | Name of the District | - | Official Indian Registry | `CONFIRMED` |
| `normalized_key` | `VARCHAR(120)` | No | No | No | Lowercase sanitized search key (e.g., `karnataka_bengaluru_urban`) | - | System Generated | `PROPOSED` |
| `lgd_code` | `VARCHAR(20)` | Yes | No | No | Local Government Directory standard district code | - | Ministry of Panchayati Raj / LGD | `NEEDS TEAM CONFIRMATION` |
| `census_code` | `VARCHAR(20)` | Yes | No | No | 2011 Indian Census district code | - | Census of India | `NEEDS TEAM CONFIRMATION` |
| `latitude` | `DECIMAL(9,6)` | No | No | No | Centroid latitude coordinate | Decimal Degrees (°N) | Adhith / GIS Reference | `CONFIRMED` |
| `longitude` | `DECIMAL(9,6)` | No | No | No | Centroid longitude coordinate | Decimal Degrees (°E) | Adhith / GIS Reference | `CONFIRMED` |
| `elevation_m` | `DECIMAL(7,2)` | Yes | No | No | Average district elevation above sea level | Meters (m) | Topographic Data | `PROPOSED` |
| `climate_zone` | `VARCHAR(50)` | Yes | No | No | Koppen climate classification or IMD zone | - | IMD Climate Classification | `PROPOSED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Record creation timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 2. TABLE: `weather_observations`
**Description:** Historical and seasonal climate observations per district across time.  
**Ownership / Source:** Adhith (Data Engineering / IMD / ERA5)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | No | Yes | No | Unique record identifier | - | System Generated | `CONFIRMED` |
| `location_id` | `INTEGER` | No | No | Yes | Foreign key reference to `locations.id` | - | `locations` Table | `CONFIRMED` |
| `observation_date` | `DATE` | No | No | No | Calendar date of observation | Date (`YYYY-MM-DD`) | IMD / Weather Station | `CONFIRMED` |
| `period_type` | `VARCHAR(20)` | No | No | No | Granularity: `DAILY`, `WEEKLY`, `MONTHLY` | - | Adhith Pipeline | `NEEDS TEAM CONFIRMATION` |
| `temp_mean_c` | `DECIMAL(5,2)` | Yes | No | No | Mean ambient air temperature | Degrees Celsius (°C) | IMD | `CONFIRMED` |
| `temp_max_c` | `DECIMAL(5,2)` | Yes | No | No | Daily maximum air temperature | Degrees Celsius (°C) | IMD | `CONFIRMED` |
| `temp_min_c` | `DECIMAL(5,2)` | Yes | No | No | Daily minimum air temperature | Degrees Celsius (°C) | IMD | `CONFIRMED` |
| `rainfall_mm` | `DECIMAL(7,2)` | Yes | No | No | Accumulated precipitation | Millimeters (mm) | IMD / Rain Gauge | `CONFIRMED` |
| `humidity_pct` | `DECIMAL(5,2)` | Yes | No | No | Relative humidity | Percentage (%) | IMD | `CONFIRMED` |
| `pressure_hpa` | `DECIMAL(6,2)` | Yes | No | No | Atmospheric surface pressure | Hectopascals (hPa) | IMD / Weather Station | `PROPOSED` |
| `source_id` | `INTEGER` | No | No | Yes | Foreign key reference to `data_sources.id` | - | `data_sources` Table | `CONFIRMED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Record ingestion timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 3. TABLE: `disease_surveillance`
**Description:** Historical public health surveillance records for vector-borne diseases (Dengue and Malaria).  
**Ownership / Source:** Adhith (Data Engineering / NVBDCP / NCDC)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | No | Yes | No | Unique record identifier | - | System Generated | `CONFIRMED` |
| `location_id` | `INTEGER` | No | No | Yes | Foreign key reference to `locations.id` | - | `locations` Table | `CONFIRMED` |
| `report_date` | `DATE` | No | No | No | Reporting period date (e.g. week/month start) | Date (`YYYY-MM-DD`) | NVBDCP / State Health Portal | `CONFIRMED` |
| `disease_type` | `VARCHAR(30)` | No | No | No | Hazard: `'DENGUE'` or `'MALARIA'` | - | NVBDCP Classification | `CONFIRMED` |
| `cases_reported` | `INTEGER` | Yes | No | No | Number of confirmed diagnosed cases | Count | Health Department | `CONFIRMED` |
| `deaths_reported` | `INTEGER` | Yes | No | No | Number of attributed fatalities | Count | Health Department | `PROPOSED` |
| `incidence_rate_per_100k` | `DECIMAL(7,3)` | Yes | No | No | Normalized incidence per 100,000 population | Rate | Adhith / Calculation | `NEEDS TEAM CONFIRMATION` |
| `reporting_granularity` | `VARCHAR(20)` | No | No | No | `'MONTHLY'`, `'EPIDEMIOLOGICAL_WEEK'` | - | NVBDCP Data Format | `NEEDS TEAM CONFIRMATION` |
| `source_id` | `INTEGER` | No | No | Yes | Foreign key reference to `data_sources.id` | - | `data_sources` Table | `CONFIRMED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Ingestion timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 4. TABLE: `disaster_events`
**Description:** Historical records of natural disaster events (Floods and Heatwaves) across Indian districts.  
**Ownership / Source:** Adhith (Data Engineering / NDMA / CWC / Disaster Database)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `BIGSERIAL` | No | Yes | No | Unique disaster event record ID | - | System Generated | `CONFIRMED` |
| `location_id` | `INTEGER` | No | No | Yes | Foreign key reference to `locations.id` | - | `locations` Table | `CONFIRMED` |
| `event_date` | `DATE` | No | No | No | Start date of recorded event | Date (`YYYY-MM-DD`) | NDMA / IMD Alert Archive | `CONFIRMED` |
| `disaster_type` | `VARCHAR(30)` | No | No | No | Disaster type: `'FLOOD'` or `'HEATWAVE'` | - | Official Disaster Classification | `CONFIRMED` |
| `severity_tier` | `VARCHAR(20)` | Yes | No | No | Recorded severity: `'MODERATE'`, `'HIGH'`, `'SEVERE'` | - | NDMA / IMD Warning Level | `PROPOSED` |
| `duration_days` | `INTEGER` | Yes | No | No | Duration of extreme event | Days | Disaster Report | `PROPOSED` |
| `affected_population` | `INTEGER` | Yes | No | No | Estimated population affected | Count | Relief & Rehabilitation Dept | `NEEDS TEAM CONFIRMATION` |
| `source_id` | `INTEGER` | No | No | Yes | Foreign key reference to `data_sources.id` | - | `data_sources` Table | `CONFIRMED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Ingestion timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 5. TABLE: `ml_models`
**Description:** Registry tracking trained model versions, neural network architectures, and performance metrics.  
**Ownership / Source:** Varesh (ML / ANN & Research)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `SERIAL` | No | Yes | No | Unique model registry ID | - | System Generated | `CONFIRMED` |
| `model_name` | `VARCHAR(100)` | No | No | No | Name identifier (e.g. `'bharat_travelshield_ann'`) | - | Varesh ML Registry | `CONFIRMED` |
| `model_version` | `VARCHAR(50)` | No | No | No | Version string (e.g. `'v1.0.0'`, `'ann_mlp_202609'`) | - | Varesh ML Registry | `CONFIRMED` |
| `architecture_type` | `VARCHAR(50)` | No | No | No | Architecture: `'ANN_MLP'`, `'LIGHTGBM_BASELINE'` | - | Varesh Model Spec | `CONFIRMED` |
| `trained_at` | `TIMESTAMPTZ` | No | No | No | Timestamp when training completed | - | Training Pipeline | `CONFIRMED` |
| `metrics_json` | `JSONB` | Yes | No | No | Validation metrics (PR-AUC, ROC-AUC, F1, Brier) | JSON Key-Value | Evaluation Script | `PROPOSED` |
| `artifact_path` | `VARCHAR(255)` | No | No | No | Relative path to model weights (e.g. `ml/models/ann_v1.pt`) | - | File System Storage | `PROPOSED` |
| `is_active` | `BOOLEAN` | No | No | No | Flag indicating if this model serves production API | - | System Configuration (`TRUE`/`FALSE`) | `PROPOSED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Registry creation timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 6. TABLE: `predictions`
**Description:** Persistent log of user travel risk queries and generated four-hazard independent probabilities.  
**Ownership / Source:** Rajdeep (Backend / API), Varesh (Model Inference), Pranav (DB Storage)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `UUID` | No | Yes | No | Unique prediction transaction UUID | - | System Generated (`gen_random_uuid()`) | `PROPOSED` |
| `location_id` | `INTEGER` | No | No | Yes | Foreign key reference to `locations.id` | - | User Query / Request | `CONFIRMED` |
| `target_travel_date` | `DATE` | No | No | No | Projected travel date requested by traveler | Date (`YYYY-MM-DD`) | User Query / Request | `CONFIRMED` |
| `model_id` | `INTEGER` | No | No | Yes | Foreign key reference to `ml_models.id` | - | Serving Backend Config | `CONFIRMED` |
| `dengue_probability` | `NUMERIC(5,4)` | No | No | No | Independent risk probability for Dengue | $[0.0000, 1.0000]$ | Varesh ANN Inference | `CONFIRMED` |
| `malaria_probability`| `NUMERIC(5,4)` | No | No | No | Independent risk probability for Malaria | $[0.0000, 1.0000]$ | Varesh ANN Inference | `CONFIRMED` |
| `flood_probability`  | `NUMERIC(5,4)` | No | No | No | Independent risk probability for Flood | $[0.0000, 1.0000]$ | Varesh ANN Inference | `CONFIRMED` |
| `heatwave_probability`| `NUMERIC(5,4)`| No | No | No | Independent risk probability for Heatwave | $[0.0000, 1.0000]$ | Varesh ANN Inference | `CONFIRMED` |
| `dengue_tier` | `VARCHAR(20)` | No | No | No | Categorical tier (`Low`, `Moderate`, `Elevated`, `High`, `Severe`) | - | Threshold Mapping | `PROPOSED` |
| `malaria_tier` | `VARCHAR(20)` | No | No | No | Categorical tier (`Low`, `Moderate`, `Elevated`, `High`, `Severe`) | - | Threshold Mapping | `PROPOSED` |
| `flood_tier` | `VARCHAR(20)` | No | No | No | Categorical tier (`Low`, `Moderate`, `Elevated`, `High`, `Severe`) | - | Threshold Mapping | `PROPOSED` |
| `heatwave_tier` | `VARCHAR(20)` | No | No | No | Categorical tier (`Low`, `Moderate`, `Elevated`, `High`, `Severe`) | - | Threshold Mapping | `PROPOSED` |
| `feature_vector_snapshot` | `JSONB` | Yes | No | No | Snapshot of features used during inference | JSON Key-Value | Backend Request Builder | `PROPOSED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Inference timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |

---

## 7. TABLE: `data_sources`
**Description:** Provenance registry of governmental and scientific agencies supplying data to the platform.  
**Ownership / Source:** Pranav (DB) & Adhith (Data Engineering)

| Column Name | Proposed Type | Nullable? | PK? | FK? | Description | Unit | Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `SERIAL` | No | Yes | No | Unique source identifier | - | System Generated | `CONFIRMED` |
| `source_code` | `VARCHAR(30)` | No | No | No | Code tag (e.g. `'IMD_WEATHER'`, `'NVBDCP_DISEASE'`, `'NDMA_DISASTER'`) | - | System Definition | `CONFIRMED` |
| `source_name` | `VARCHAR(150)` | No | No | No | Official agency full name | - | Official Agency | `CONFIRMED` |
| `agency_type` | `VARCHAR(50)` | No | No | No | Category: `'CENTRAL_GOVERNMENT'`, `'STATE_GOVERNMENT'`, `'RESEARCH_DATASET'` | - | System Definition | `PROPOSED` |
| `source_url` | `VARCHAR(255)` | Yes | No | No | Portal URL / documentation link | - | Agency Documentation | `PROPOSED` |
| `last_synced_at` | `TIMESTAMPTZ` | Yes | No | No | Timestamp of most recent batch ingestion | - | Ingestion Pipeline | `CONFIRMED` |
| `created_at` | `TIMESTAMPTZ` | No | No | No | Record creation timestamp | - | System Generated (`NOW()`) | `CONFIRMED` |
