# BHARAT-TRAVELSHIELD AI — MACHINE LEARNING & ANN ARCHITECTURE SPECIFICATION

**Author:** Varesh (ML / ANN & Research) + Pranav (Integration)  
**Project:** Bharat-TravelShield AI  
**Repository:** [Varesh2005/Bharat_TravelShield_AI](https://github.com/Varesh2005/Bharat_TravelShield_AI)  
**Status:** `TRAINED, EVALUATED & PRODUCTION READY`  

---

## 1. Executive Summary & Problem Formulation

Bharat-TravelShield AI addresses spatiotemporal travel health and natural disaster risk prediction across India. Rather than predicting a simplistic, arbitrary single composite risk index, the project requires **independent probability estimation** for four distinct hazards:
1. **Dengue Risk** ($\hat{y}_{\text{dengue}} \in [0.0, 1.0]$)
2. **Malaria Risk** ($\hat{y}_{\text{malaria}} \in [0.0, 1.0]$)
3. **Flood Risk** ($\hat{y}_{\text{flood}} \in [0.0, 1.0]$)
4. **Heatwave Risk** ($\hat{y}_{\text{heatwave}} \in [0.0, 1.0]$)

---

## 2. Artificial Neural Network Architecture: `MultiHazardRiskANN`

### 2.1 Why Multi-Task Deep Learning?
In epidemiology and meteorology, hazards share underlying geophysical and spatiotemporal factors:
- **Monsoon rains & high humidity** foster mosquito breeding (Dengue & Malaria) and drive riverine runoff (Floods).
- **Extreme summer heat** drives physiological Heatwaves while simultaneously suppressing vector lifecycles above 38°C.

A **Multi-Task Neural Network** captures shared spatiotemporal latent representations in a **shared representation trunk** while preserving task specialization through **independent output heads**.

### 2.2 Network Diagram
```
                     Input Features (D = 18)
                                │
                                ▼
            ┌───────────────────────────────────────┐
            │       SHARED TRUNK: LAYER 1           │
            │  Linear(18 → 64)                      │
            │  BatchNorm1d(64)                      │
            │  LeakyReLU(α = 0.1)                   │
            │  Dropout(p = 0.2)                     │
            └───────────────────┬───────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────┐
            │       SHARED TRUNK: LAYER 2           │
            │  Linear(64 → 32)                      │
            │  BatchNorm1d(32)                      │
            │  LeakyReLU(α = 0.1)                   │
            │  Dropout(p = 0.2)                     │
            └───────────────────┬───────────────────┘
                                │
                 Shared Latent Embedding (dim = 32)
                                │
         ┌──────────────┬───────┴───────┬──────────────┐
         ▼              ▼               ▼              ▼
    [DENGUE HEAD] [MALARIA HEAD]   [FLOOD HEAD]  [HEATWAVE HEAD]
   Linear(32 → 16) Linear(32 → 16) Linear(32 → 16) Linear(32 → 16)
   LeakyReLU(0.1)  LeakyReLU(0.1)  LeakyReLU(0.1)  LeakyReLU(0.1)
   Linear(16 → 1)  Linear(16 → 1)  Linear(16 → 1)  Linear(16 → 1)
      Sigmoid         Sigmoid         Sigmoid         Sigmoid
         │              │               │              │
         ▼              ▼               ▼              ▼
   P(Dengue)       P(Malaria)       P(Flood)       P(Heatwave)
```

---

## 3. Spatiotemporal Input Vector Specification (18 Dimensions)

| Index | Feature Symbol | Mathematical Formulation | Physical Rationale |
| :---: | :--- | :--- | :--- |
| 0 | `month_sin` | $\sin(2\pi \cdot (m - 1) / 12)$ | Continuous cyclical seasonal encoding |
| 1 | `month_cos` | $\cos(2\pi \cdot (m - 1) / 12)$ | Ensures Dec 31 and Jan 1 are contiguous |
| 2 | `latitude_norm` | $(\text{lat} - 20.0) / 10.0$ | North-South tropical gradient across India |
| 3 | `longitude_norm` | $(\text{lon} - 80.0) / 10.0$ | East-West coastal vs continental gradient |
| 4 | `elevation_norm` | $\text{elevation}_{\text{meters}} / 1000.0$ | Lapse rate cooling & mountain barrier effect |
| 5 | `max_temp_norm` | $(T_{\max} - 30.0) / 10.0$ | Heatwave threshold & vector activity bounds |
| 6 | `min_temp_norm` | $(T_{\min} - 20.0) / 10.0$ | Night-time cooling & thermal comfort |
| 7 | `temp_diurnal_norm`| $(T_{\max} - T_{\min}) / 10.0$ | Continental dry heat vs coastal humidity |
| 8 | `rainfall_7d_norm` | $R_{7\text{d}} / 100.0$ | Short-term flash flood surge risk |
| 9 | `rainfall_30d_norm`| $R_{30\text{d}} / 300.0$ | Cumulative waterlogging & breeding habitat |
| 10 | `humidity_norm` | $\text{RH}_{\%} / 100.0$ | Vector survival & physiological heat index |
| 11 | `monsoon_flag` | $\mathbb{I}(m \in \{6, 7, 8, 9\})$ | South-West monsoon season activation |
| 12 | `summer_flag` | $\mathbb{I}(m \in \{4, 5, 6\} \land T_{\max} \ge 38^\circ\text{C})$ | Peak pre-monsoon heat crisis indicator |
| 13 | `dengue_lag_norm` | $\ln(1 + \text{cases}_{\text{lag}}) / 5.0$ | Local endemic epidemiological reservoir |
| 14 | `malaria_lag_norm`| $\ln(1 + \text{cases}_{\text{lag}}) / 5.0$ | Parasite reservoir in community |
| 15 | `flood_history_freq`| Historical flood events in 5 yrs / 5.0 | Topographical drainage vulnerability |
| 16 | `heatwave_freq` | Historical heatwave days in 5 yrs / 5.0 | Microclimatic heat island vulnerability |
| 17 | `pop_density_norm`| $\ln(1 + \text{pop\_density}) / 10.0$ | Urban transmission velocity for Aedes |

---

## 4. Multi-Task Loss Formulation

The optimization objective is the joint binary cross-entropy across all four hazard outputs:
$$\mathcal{L}_{\text{total}} = \sum_{k \in \{\text{dengue}, \text{malaria}, \text{flood}, \text{heatwave}\}} \mathcal{L}_{\text{BCE}}(y_k, \hat{y}_k)$$
Where each individual hazard loss is:
$$\mathcal{L}_{\text{BCE}}(y_k, \hat{y}_k) = - \frac{1}{N} \sum_{i=1}^N \left[ y_k^{(i)} \log(\hat{y}_k^{(i)}) + (1 - y_k^{(i)}) \log(1 - \hat{y}_k^{(i)}) \right]$$

---

## 5. Experimental Evaluation Results

Evaluated on 2,400 independent out-of-sample spatiotemporal validation records:

| Hazard Target | ROC-AUC | PR-AUC (Precision-Recall) | Brier Score (Calibration) | Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Dengue** | **0.9130** | **0.8051** | **0.0974** | 22.4% |
| **Malaria** | **0.8852** | **0.6539** | **0.0945** | 16.8% |
| **Flood** | **0.9368** | **0.8908** | **0.0656** | 18.2% |
| **Heatwave** | **0.9736** | **0.9276** | **0.0322** | 11.5% |

### Categorical Risk Tier Mapping
The continuous probabilities $\hat{y}_k$ are mapped to standardized project risk tiers:
- **Low:** $[0.00, 0.20)$
- **Moderate:** $[0.20, 0.45)$
- **Elevated:** $[0.45, 0.65)$
- **High:** $[0.65, 0.85)$
- **Severe:** $[0.85, 1.00]$

---

## 6. Viva Voce Defense Guide (Anticipated Questions & Model Answers)

### Q1: Why use an Artificial Neural Network (ANN) rather than simple Decision Trees?
> **Answer:** Environmental hazard forecasting across India involves non-linear spatiotemporal interactions between elevation, cyclical season angles, temperature lapse rates, and cumulative precipitation. A multi-task ANN enables shared representation learning where common geophysical factors are encoded in the shared trunk, allowing simultaneous prediction of multiple related hazards with a single forward pass (< 5ms inference latency).

### Q2: Why did you not compute a single blended "Overall Hazard Score"?
> **Answer:** Combining a 70% flood risk with a 10% malaria risk into an arbitrary blended score (e.g. "40% Risk") is dangerous and misleading for travelers. A traveler needs actionable, hazard-specific intelligence: precautions against flooding (transport disruption) are completely different from precautions against malaria (prophylaxis and mosquito repellents). Independent probabilities preserve complete diagnostic integrity.

### Q3: Why is month encoded using sine and cosine?
> **Answer:** If month is represented as a scalar integer ($1, 2, \dots, 12$), month 12 (December) and month 1 (January) have a numerical distance of 11, despite being chronologically adjacent winter months. Cyclical encoding:
> $$\text{month\_sin} = \sin(2\pi(m-1)/12), \quad \text{month\_cos} = \cos(2\pi(m-1)/12)$$
> maps the temporal continuum onto a 2D unit circle where December and January are neighbors in latent space.

### Q4: How is data leakage prevented in feature engineering?
> **Answer:** All features are strictly temporal-lagged: 7-day and 30-day cumulative weather metrics, prior epidemiological surveillance reports, and historical hazard frequencies. The network never accesses future information from after the target prediction date.
