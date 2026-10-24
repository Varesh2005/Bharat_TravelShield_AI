"""
Bharat-TravelShield AI - What-If Scenario Simulation Service
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Allows users to simulate climate/environmental changes (e.g. +3°C temperature,
+50mm rainfall) and compare counterfactual ANN hazard predictions against baseline.
"""

from typing import Dict, Any, Optional
from ml.hazard_inference import predict_for_district, predict_hazard


class ScenarioService:
    """Business logic for What-If scenario simulations."""

    @staticmethod
    def simulate_scenario(
        hazard: str,
        district_code: int,
        deltas: Dict[str, float],
        baseline_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs baseline prediction, applies environmental feature deltas,
        and runs counterfactual ANN inference.
        """
        hz = hazard.lower().strip()
        # 1. Fetch baseline prediction and input features
        base_res = predict_for_district(
            hazard=hz,
            district_code=district_code,
            feature_year=baseline_year,
            explain=True,
        )

        if base_res.get("status") != "ok":
            return {
                "status": base_res.get("status"),
                "hazard": hz,
                "district_code": district_code,
                "error": base_res.get("reason") or "Baseline feature record unavailable.",
            }

        base_features = dict(base_res.get("input_features", {}))
        modified_features = dict(base_features)

        # 2. Apply deltas to valid numeric features
        applied_deltas = {}
        for feat_name, delta_val in deltas.items():
            if feat_name in modified_features and isinstance(delta_val, (int, float)):
                modified_features[feat_name] = round(modified_features[feat_name] + float(delta_val), 4)
                applied_deltas[feat_name] = delta_val

        # Auto-reconcile mathematical constraints for physical consistency:
        # e.g., temp_range_c = max_temperature_c - min_temperature_c
        if "max_temperature_c" in modified_features and "min_temperature_c" in modified_features and "temp_range_c" in modified_features:
            modified_features["temp_range_c"] = round(modified_features["max_temperature_c"] - modified_features["min_temperature_c"], 4)

        if "heatwave_days_T" in modified_features and "severe_heatwave_days_T" in modified_features:
            if modified_features["severe_heatwave_days_T"] > modified_features["heatwave_days_T"]:
                modified_features["severe_heatwave_days_T"] = modified_features["heatwave_days_T"]

        # 3. Predict counterfactual scenario
        scenario_res = predict_hazard(
            hazard=hz,
            features=modified_features,
            explain=True,
        )

        if scenario_res.get("status") != "ok":
            return {
                "status": scenario_res.get("status"),
                "hazard": hz,
                "district_code": district_code,
                "error": scenario_res.get("errors") or scenario_res.get("reason"),
            }

        # 4. Compute risk score differential
        base_pred = base_res["prediction"]
        scen_pred = scenario_res["prediction"]

        base_score = base_pred.get("calibrated_probability") or base_pred.get("raw_model_score")
        scen_score = scen_pred.get("calibrated_probability") or scen_pred.get("raw_model_score")
        score_diff = round(scen_score - base_score, 4) if (scen_score is not None and base_score is not None) else None

        return {
            "status": "ok",
            "hazard": hz,
            "district_code": district_code,
            "feature_year": base_res.get("feature_year"),
            "predicts_year": base_res.get("target_year"),
            "applied_deltas": applied_deltas,
            "baseline": {
                "probability": base_pred.get("calibrated_probability"),
                "raw_score": base_pred.get("raw_model_score"),
                "predicted_label": base_pred.get("predicted_label"),
            },
            "scenario": {
                "probability": scen_pred.get("calibrated_probability"),
                "raw_score": scen_pred.get("raw_model_score"),
                "predicted_label": scen_pred.get("predicted_label"),
            },
            "score_delta": score_diff,
            "risk_impact": "increased" if (score_diff and score_diff > 0.05) else ("decreased" if (score_diff and score_diff < -0.05) else "stable"),
            "disclaimer": "Simulated counterfactual outcome. Model response is non-linear and indicative.",
        }
