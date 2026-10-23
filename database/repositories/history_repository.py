"""
Bharat-TravelShield AI - History Repository (Firestore)
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Manages historical weather observations and epidemiological disease surveillance records.
"""

from typing import List, Dict, Any, Optional
from database.firebase_client import get_firestore_db
from database.schema import COLLECTION_WEATHER, COLLECTION_DISEASE


class HistoryRepository:
    """Manages historical climate and epidemiology data in Firestore."""

    def __init__(self, db=None):
        self.db = db or get_firestore_db()
        self.weather_coll = self.db.collection(COLLECTION_WEATHER)
        self.disease_coll = self.db.collection(COLLECTION_DISEASE)

    def get_weather_history(self, location_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetches historical weather observations for a district."""
        docs = self.weather_coll.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if data and data.get("location_id") == location_id:
                results.append(data)
        results.sort(key=lambda x: x.get("observation_date", ""), reverse=True)
        return results[:limit]

    def get_disease_history(self, location_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetches historical disease surveillance reports for a district."""
        docs = self.disease_coll.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if data and data.get("location_id") == location_id:
                results.append(data)
        results.sort(key=lambda x: x.get("report_date", ""), reverse=True)
        return results[:limit]

    def save_weather_record(self, data: Dict[str, Any]) -> str:
        doc_id = f"{data.get('location_id')}_{data.get('observation_date')}"
        self.weather_coll.document(doc_id).set(data, merge=True)
        return doc_id

    def save_disease_record(self, data: Dict[str, Any]) -> str:
        doc_id = f"{data.get('location_id')}_{data.get('disease_type')}_{data.get('report_date')}"
        self.disease_coll.document(doc_id).set(data, merge=True)
        return doc_id
