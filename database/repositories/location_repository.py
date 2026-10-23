"""
Bharat-TravelShield AI - Location Repository (Firestore)
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Provides high-speed queries, search, and retrieval for Indian districts and states.
"""

from typing import List, Dict, Any, Optional
from database.firebase_client import get_firestore_db
from database.schema import COLLECTION_LOCATIONS


class LocationRepository:
    """Manages Firestore access for the 'locations' collection."""

    def __init__(self, db=None):
        self.db = db or get_firestore_db()
        self.collection = self.db.collection(COLLECTION_LOCATIONS)

    def get_by_id(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single location by its document ID, alias, or census_district_code."""
        loc_str = str(location_id).strip()
        doc = self.collection.document(loc_str).get()
        if doc.exists:
            return doc.to_dict()

        # Check census_district_code or normalized_key match
        for d in self.collection.stream():
            data = d.to_dict()
            if not data:
                continue
            if loc_str.isdigit() and data.get("census_district_code") == int(loc_str):
                return data
            norm = data.get("normalized_key", "")
            if loc_str.lower() == norm or loc_str.lower() in norm or norm in loc_str.lower():
                return data
        return None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches locations by district name or state name (case-insensitive fuzzy match).
        """
        q_clean = query.strip().lower()
        results = []

        # Iterate through collection documents
        docs = self.collection.stream()
        for doc in docs:
            data = doc.to_dict()
            if not data:
                continue

            dist = data.get("district_name", "").lower()
            state = data.get("state_name", "").lower()
            norm = data.get("normalized_key", "")
            spots = [s.lower() for s in data.get("popular_tourist_spots", [])]

            if (q_clean in dist or
                q_clean in state or
                q_clean in norm or
                any(q_clean in spot for spot in spots)):
                results.append(data)
                if len(results) >= limit:
                    break

        return results

    def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists all registered districts in the database."""
        docs = self.collection.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if data:
                results.append(data)
                if len(results) >= limit:
                    break
        return results

    def upsert(self, location_data: Dict[str, Any]) -> str:
        """Creates or updates a location document in Firestore."""
        doc_id = location_data.get("id")
        if not doc_id:
            doc_id = f"{location_data.get('state_name', '')}_{location_data.get('district_name', '')}".lower().replace(" ", "_")
            location_data["id"] = doc_id

        self.collection.document(doc_id).set(location_data, merge=True)
        return doc_id
