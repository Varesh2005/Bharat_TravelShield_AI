"""
Bharat-TravelShield AI - Research Leaderboard Service
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Provides model comparison benchmarks (ANN vs Baselines) for Rosemariya's Research Lab.
"""

import os
import json
from typing import Dict, Any, List


class ResearchService:
    """Service to load and format offline research experiments for the UI."""

    @staticmethod
    def get_leaderboard() -> Dict[str, Any]:
        """Loads evaluation results for all 3 supported hazards."""
        hazards = ["heatwave", "flood", "malaria"]
        leaderboard_data: Dict[str, Any] = {}

        for hz in hazards:
            res_path = os.path.join("research", "experiments", hz, "results.json")
            if os.path.exists(res_path):
                try:
                    with open(res_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Extract models comparison table
                    models_summary = []
                    models = data.get("models", {})
                    for m_name, m_info in models.items():
                        metrics = m_info.get("test_metrics", {})
                        models_summary.append({
                            "model_name": m_name,
                            "model_type": "ANN / MLP" if "ann" in m_name.lower() or "mlp" in m_name.lower() else "Baseline",
                            "pr_auc": metrics.get("pr_auc"),
                            "roc_auc": metrics.get("roc_auc"),
                            "f1": metrics.get("f1"),
                            "brier_score": metrics.get("brier"),
                            "ece": metrics.get("ece"),
                        })

                    leaderboard_data[hz] = {
                        "hazard": hz,
                        "model_version": data.get("model_version"),
                        "training_window": data.get("dataset", {}).get("split_rule", {}),
                        "models_compared": models_summary,
                    }
                except Exception as e:
                    leaderboard_data[hz] = {"error": f"Failed to parse results: {e}"}
            else:
                leaderboard_data[hz] = {"status": "results_pending"}

        return {
            "title": "Bharat-TravelShield AI — Empirical Model Benchmark Leaderboard",
            "hazards": leaderboard_data,
            "evaluation_protocol": "Chronological holdout split: Train <= 2019, Validation 2020-2021, Test >= 2022",
        }
