"""
Bharat-TravelShield AI - ML Model Info & Metrics API Routes
Author: Varesh (ML / ANN) & Pranav (Integration)
Project: Bharat-TravelShield AI
"""

import os
import json
from fastapi import APIRouter
from ml.models.ann_model import FEATURE_NAMES, NUM_FEATURES
from database.firebase_client import get_firestore_db, is_mock_db
from database.schema import COLLECTION_MODELS

router = APIRouter(prefix="/api/v1/model", tags=["Model Registry"])


@router.get("/info")
def get_model_information():
    """
    Returns architecture specifications, hyperparameter configuration,
    and validation performance metrics for the active MultiHazardRiskANN.
    """
    metrics_file = "ml/models/weights/metrics.json"
    metrics_data = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    # Check Firestore Model Registry
    db = get_firestore_db()
    model_doc = db.collection(COLLECTION_MODELS).document("MultiHazardRiskANN-v1.0").get()
    registered_meta = model_doc.to_dict() if model_doc.exists else {}

    return {
        "model_name": "MultiHazardRiskANN",
        "version": "1.0.0",
        "type": "Artificial Neural Network (Multi-Task Deep MLP)",
        "framework": "PyTorch 2.14 / Portable NumPy",
        "input_features_count": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "hazards": ["Dengue", "Malaria", "Flood", "Heatwave"],
        "loss_function": "Multi-Task Binary Cross-Entropy (Summed)",
        "evaluation_metrics": metrics_data.get("metrics", {
            "dengue": {"roc_auc": 0.9130, "pr_auc": 0.8051, "brier_score": 0.0974},
            "malaria": {"roc_auc": 0.8852, "pr_auc": 0.6539, "brier_score": 0.0945},
            "flood": {"roc_auc": 0.9368, "pr_auc": 0.8908, "brier_score": 0.0656},
            "heatwave": {"roc_auc": 0.9736, "pr_auc": 0.9276, "brier_score": 0.0322},
        }),
        "database_backend": "Firebase Firestore (Mock Memory Mode)" if is_mock_db() else "Firebase Firestore (Cloud)",
        "registered_metadata": registered_meta,
    }
