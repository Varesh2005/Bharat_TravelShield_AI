"""
Bharat-TravelShield AI - Real-Time Multi-Hazard ANN Inference Engine
Author: Varesh (ML / ANN & Research) + Pranav
Project: Bharat-TravelShield AI

Provides high-performance risk inference, risk tier categorization,
and feature contribution explainability for the FastAPI backend and Firebase layer.
"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ml.models.ann_model import MultiHazardRiskANN, PureNumpyANNInference, FEATURE_NAMES, NUM_FEATURES
from ml.preprocessing.feature_pipeline import FeaturePipeline, SpatiotemporalInput


def get_risk_tier(prob: float) -> str:
    """Categorizes independent probability into standardized project risk tiers."""
    if prob < 0.20:
        return "Low"
    elif prob < 0.45:
        return "Moderate"
    elif prob < 0.65:
        return "Elevated"
    elif prob < 0.85:
        return "High"
    else:
        return "Severe"


class HazardPredictor:
    """
    Inference service managing trained ANN weights and explainability heuristics.
    """

    def __init__(self, weights_dir: str = "ml/models/weights"):
        self.weights_dir = weights_dir
        self.pipeline = FeaturePipeline()
        self.torch_model: Optional[Any] = None
        self.numpy_model: Optional[PureNumpyANNInference] = None
        self._load_weights()

    def _load_weights(self):
        pt_path = os.path.join(self.weights_dir, "travelshield_ann.pt")
        json_path = os.path.join(self.weights_dir, "travelshield_weights.json")

        if TORCH_AVAILABLE and os.path.exists(pt_path):
            try:
                model = MultiHazardRiskANN(input_dim=NUM_FEATURES)
                model.load_state_dict(torch.load(pt_path, map_location="cpu"))
                model.eval()
                self.torch_model = model
                print(f"[HazardPredictor] Loaded PyTorch ANN from {pt_path}")
                return
            except Exception as e:
                print(f"[HazardPredictor] PyTorch load failed: {e}. Falling back to JSON weights...")

        if os.path.exists(json_path):
            try:
                self.numpy_model = PureNumpyANNInference.load_from_json(json_path)
                print(f"[HazardPredictor] Loaded PureNumpy ANN from {json_path}")
                return
            except Exception as e:
                print(f"[HazardPredictor] JSON load failed: {e}.")

        print("[HazardPredictor] No pre-trained weights found. Training new weights is recommended.")

    def _explain_factors(self, inp: SpatiotemporalInput, probs: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Derives human-readable key contributing factors per hazard
        for frontend and explainability compliance.
        """
        explanations = {
            "dengue": [],
            "malaria": [],
            "flood": [],
            "heatwave": [],
        }

        month = FeaturePipeline._parse_month(inp.target_date)
        is_monsoon = month in [6, 7, 8, 9]
        is_summer = month in [4, 5, 6]

        # Dengue Factors
        if is_monsoon:
            explanations["dengue"].append("Active monsoon rainfall creates stagnant water pools ideal for Aedes mosquito breeding")
        if inp.relative_humidity_pct and inp.relative_humidity_pct > 75:
            explanations["dengue"].append(f"High relative humidity ({inp.relative_humidity_pct:.0f}%) accelerates mosquito survival and viral replication")
        if inp.population_density and inp.population_density > 1000:
            explanations["dengue"].append(f"High urban population density ({inp.population_density:.0f}/km²) increases transmission velocity")
        if not explanations["dengue"]:
            explanations["dengue"].append("Baseline historical seasonal trend for district")

        # Malaria Factors
        if is_monsoon or month in [10, 11]:
            explanations["malaria"].append("Post-monsoon vegetation and water accumulation facilitate Anopheles vector proliferation")
        if inp.elevation_m < 800:
            explanations["malaria"].append("Low elevation tropical floodplain favors sustained parasite sporogony")
        if not explanations["malaria"]:
            explanations["malaria"].append("Low seasonal vector transmission in current temporal window")

        # Flood Factors
        if inp.rainfall_7d_mm and inp.rainfall_7d_mm > 70:
            explanations["flood"].append(f"High 7-day cumulative rainfall ({inp.rainfall_7d_mm:.1f} mm) elevates hydrological surge risk")
        if inp.flood_history_freq and inp.flood_history_freq > 0.4:
            explanations["flood"].append("District has documented high historical inundation frequency")
        if not explanations["flood"]:
            explanations["flood"].append("Precipitation levels within seasonal soil absorption thresholds")

        # Heatwave Factors
        if inp.max_temp_c and inp.max_temp_c >= 40:
            explanations["heatwave"].append(f"Severe maximum temperature ({inp.max_temp_c:.1f}°C) exceeds IMD heatwave alert criteria")
        elif is_summer:
            explanations["heatwave"].append("Peak pre-monsoon solar radiation and hot continental advection")
        if inp.relative_humidity_pct and inp.relative_humidity_pct < 40 and is_summer:
            explanations["heatwave"].append("Dry desiccating air amplifies thermal stress and dehydration hazards")
        if not explanations["heatwave"]:
            explanations["heatwave"].append("Ambient temperatures well below physiological heat emergency thresholds")

        return explanations

    def predict(self, inp: SpatiotemporalInput) -> Dict[str, Any]:
        """
        Executes prediction and packages multi-hazard intelligence.
        """
        vector = self.pipeline.transform(inp)

        probs: Dict[str, float] = {}

        if self.torch_model is not None:
            with torch.no_grad():
                tensor_x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                out = self.torch_model(tensor_x)
                probs = {
                    "dengue": round(float(out["dengue"].item()), 4),
                    "malaria": round(float(out["malaria"].item()), 4),
                    "flood": round(float(out["flood"].item()), 4),
                    "heatwave": round(float(out["heatwave"].item()), 4),
                }
        elif self.numpy_model is not None:
            probs = self.numpy_model.predict(vector)
        else:
            # Fallback heuristic if model not trained yet
            month = FeaturePipeline._parse_month(inp.target_date)
            is_monsoon = month in [6, 7, 8, 9]
            is_summer = month in [4, 5, 6]
            probs = {
                "dengue": 0.65 if is_monsoon else 0.15,
                "malaria": 0.45 if is_monsoon else 0.10,
                "flood": 0.70 if (is_monsoon and (inp.rainfall_7d_mm or 0) > 50) else 0.08,
                "heatwave": 0.80 if (is_summer and (inp.max_temp_c or 0) > 40) else 0.05,
            }

        tiers = {hazard: get_risk_tier(p) for hazard, p in probs.items()}
        explanations = self._explain_factors(inp, probs)

        return {
            "model_version": "MultiHazardRiskANN-v1.0",
            "model_type": "Artificial Neural Network (Multi-Task MLP)",
            "target_date": str(inp.target_date)[:10],
            "coordinates": {
                "latitude": inp.latitude,
                "longitude": inp.longitude,
            },
            "hazard_risks": {
                "dengue": {
                    "probability": probs["dengue"],
                    "risk_tier": tiers["dengue"],
                    "contributing_factors": explanations["dengue"],
                },
                "malaria": {
                    "probability": probs["malaria"],
                    "risk_tier": tiers["malaria"],
                    "contributing_factors": explanations["malaria"],
                },
                "flood": {
                    "probability": probs["flood"],
                    "risk_tier": tiers["flood"],
                    "contributing_factors": explanations["flood"],
                },
                "heatwave": {
                    "probability": probs["heatwave"],
                    "risk_tier": tiers["heatwave"],
                    "contributing_factors": explanations["heatwave"],
                },
            },
            "features_summary": {
                "month": FeaturePipeline._parse_month(inp.target_date),
                "elevation_m": inp.elevation_m,
                "max_temp_c": inp.max_temp_c,
                "min_temp_c": inp.min_temp_c,
                "rainfall_7d_mm": inp.rainfall_7d_mm,
                "humidity_pct": inp.relative_humidity_pct,
            },
        }


# Singleton instance
_PREDICTOR_INSTANCE: Optional[HazardPredictor] = None


def get_predictor(weights_dir: str = "ml/models/weights") -> HazardPredictor:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = HazardPredictor(weights_dir=weights_dir)
    return _PREDICTOR_INSTANCE
