"""
Bharat-TravelShield AI - Prediction Repository (Firestore)
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Manages persistence of user inference queries and ANN predictions for auditability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from database.firebase_client import get_firestore_db
from database.schema import COLLECTION_PREDICTIONS


class PredictionRepository:
    """Manages Firestore access for the 'predictions' collection."""

    def __init__(self, db=None):
        self.db = db or get_firestore_db()
        self.collection = self.db.collection(COLLECTION_PREDICTIONS)

    def save(self, prediction_data: Dict[str, Any]) -> str:
        """Saves a prediction result to Firestore."""
        doc_id = prediction_data.get("id") or f"pred_{uuid.uuid4().hex[:12]}"
        prediction_data["id"] = doc_id
        if "created_at" not in prediction_data:
            prediction_data["created_at"] = datetime.now(timezone.utc).isoformat()

        self.collection.document(doc_id).set(prediction_data)
        return doc_id

    def get_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single prediction by ID."""
        doc = self.collection.document(prediction_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves the most recent predictions across all destinations."""
        docs = self.collection.stream()
        results = [doc.to_dict() for doc in docs if doc.to_dict()]
        # Sort descending by created_at
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[:limit]

    def get_by_location(self, location_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent predictions for a specific district."""
        docs = self.collection.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if data and data.get("location_id") == location_id:
                results.append(data)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[:limit]

    def find_cached(self, location_id: str, target_travel_date: str) -> Optional[Dict[str, Any]]:
        """
        Finds a cached prediction matching location_id and target date.
        """
        docs = self.collection.stream()
        for doc in docs:
            data = doc.to_dict()
            if data and data.get("location_id") == location_id and data.get("target_travel_date") == target_travel_date:
                return data
        return None
