"""
Bharat-TravelShield AI - Firebase Firestore Connection Manager
Author: Pranav (Database + Integration + Documentation)
Project: Bharat-TravelShield AI

Provides singleton access to Firebase Admin SDK and Cloud Firestore.
Includes automatic fallback to an in-memory document store if no credentials
are provided, enabling zero-crash local development, CI/CD, and viva demos.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    FIREBASE_ADMIN_AVAILABLE = False


class InMemoryDocumentSnapshot:
    """Mock Firestore Document Snapshot for local execution without credentials."""
    def __init__(self, doc_id: str, data: Optional[Dict[str, Any]]):
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return dict(self._data) if self._data else None


class InMemoryDocumentReference:
    """Mock Firestore Document Reference."""
    def __init__(self, doc_id: str, collection: "InMemoryCollectionReference"):
        self.id = doc_id
        self.collection = collection

    def set(self, data: Dict[str, Any], merge: bool = False):
        if merge and self.id in self.collection._docs:
            self.collection._docs[self.id].update(data)
        else:
            self.collection._docs[self.id] = dict(data)

    def get(self) -> InMemoryDocumentSnapshot:
        data = self.collection._docs.get(self.id)
        return InMemoryDocumentSnapshot(self.id, data)

    def update(self, data: Dict[str, Any]):
        if self.id in self.collection._docs:
            self.collection._docs[self.id].update(data)
        else:
            self.collection._docs[self.id] = dict(data)

    def delete(self):
        self.collection._docs.pop(self.id, None)


class InMemoryCollectionReference:
    """Mock Firestore Collection Reference."""
    def __init__(self, name: str, db: "InMemoryFirestore"):
        self.name = name
        self.db = db
        self._docs: Dict[str, Dict[str, Any]] = {}

    def document(self, doc_id: Optional[str] = None) -> InMemoryDocumentReference:
        if not doc_id:
            doc_id = f"doc_{len(self._docs) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
        return InMemoryDocumentReference(doc_id, self)

    def add(self, data: Dict[str, Any]) -> tuple:
        doc_ref = self.document()
        doc_ref.set(data)
        return None, doc_ref

    def stream(self) -> List[InMemoryDocumentSnapshot]:
        return [InMemoryDocumentSnapshot(k, v) for k, v in self._docs.items()]

    def where(self, field: str, op: str, value: Any) -> "InMemoryQuery":
        return InMemoryQuery(self).where(field, op, value)

    def order_by(self, field: str, direction: str = "ASCENDING") -> "InMemoryQuery":
        return InMemoryQuery(self).order_by(field, direction)

    def limit(self, count: int) -> "InMemoryQuery":
        return InMemoryQuery(self).limit(count)


class InMemoryQuery:
    """Mock Firestore Query chaining."""
    def __init__(self, collection: InMemoryCollectionReference):
        self.collection = collection
        self.filters: List[tuple] = []
        self._limit_val: Optional[int] = None
        self._order_by_field: Optional[str] = None
        self._order_desc: bool = False

    def where(self, field: str, op: str, value: Any) -> "InMemoryQuery":
        self.filters.append((field, op, value))
        return self

    def order_by(self, field: str, direction: str = "ASCENDING") -> "InMemoryQuery":
        self._order_by_field = field
        self._order_desc = (direction.upper() == "DESCENDING")
        return self

    def limit(self, count: int) -> "InMemoryQuery":
        self._limit_val = count
        return self

    def stream(self) -> List[InMemoryDocumentSnapshot]:
        results = []
        for doc_id, doc_data in self.collection._docs.items():
            match = True
            for field, op, val in self.filters:
                doc_val = doc_data.get(field)
                if op == "==" and doc_val != val:
                    match = False
                    break
                elif op == ">=" and (doc_val is None or doc_val < val):
                    match = False
                    break
                elif op == "<=" and (doc_val is None or doc_val > val):
                    match = False
                    break
                elif op == "in" and (doc_val not in val):
                    match = False
                    break
            if match:
                results.append(InMemoryDocumentSnapshot(doc_id, doc_data))

        if self._order_by_field:
            results.sort(
                key=lambda x: (x.to_dict() or {}).get(self._order_by_field, ""),
                reverse=self._order_desc
            )

        if self._limit_val:
            results = results[:self._limit_val]

        return results


class InMemoryFirestore:
    """In-memory Firestore client matching Google Cloud Firestore API."""
    def __init__(self):
        self._collections: Dict[str, InMemoryCollectionReference] = {}

    def collection(self, name: str) -> InMemoryCollectionReference:
        if name not in self._collections:
            self._collections[name] = InMemoryCollectionReference(name, self)
        return self._collections[name]


# Module Globals
_FIREBASE_APP = None
_FIRESTORE_DB = None
_IS_USING_MOCK_DB = False


def init_firebase() -> Any:
    """
    Initializes Firebase Admin SDK.
    Priority:
      1. Explicit FIREBASE_CREDENTIALS_PATH environment variable
      2. Local 'firebase_credentials.json' in repository root
      3. Fallback to InMemoryFirestore (enables zero-configuration execution)
    """
    global _FIREBASE_APP, _FIRESTORE_DB, _IS_USING_MOCK_DB

    if _FIRESTORE_DB is not None:
        return _FIRESTORE_DB

    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")

    project_id = os.getenv("FIREBASE_PROJECT_ID", "bharat-travelshield-ai")

    if FIREBASE_ADMIN_AVAILABLE:
        # Priority 1: Service Account JSON certificate
        if os.path.exists(cred_path):
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(cred_path)
                    _FIREBASE_APP = firebase_admin.initialize_app(cred, {"projectId": project_id})
                else:
                    _FIREBASE_APP = firebase_admin.get_app()

                _FIRESTORE_DB = firestore.client()
                _IS_USING_MOCK_DB = False
                print(f"[Firebase] Successfully connected to Google Cloud Firestore via {cred_path}")
                return _FIRESTORE_DB
            except Exception as e:
                print(f"[Firebase] Certificate initialization notice: {e}")

        # Priority 2: Application Default Credentials / Project ID
        try:
            if not firebase_admin._apps:
                _FIREBASE_APP = firebase_admin.initialize_app(options={"projectId": project_id})
            else:
                _FIREBASE_APP = firebase_admin.get_app()

            _FIRESTORE_DB = firestore.client()
            # Test ping
            _ = list(_FIRESTORE_DB.collection("locations").limit(1).stream())
            _IS_USING_MOCK_DB = False
            print(f"[Firebase] Connected to Google Cloud Firestore project '{project_id}' via ADC!")
            return _FIRESTORE_DB
        except Exception:
            pass

    # Fallback to in-memory Firestore
    _FIRESTORE_DB = InMemoryFirestore()
    _IS_USING_MOCK_DB = True
    print(f"[Firebase] Running in Local Memory Firestore mode for project '{project_id}'.")

    # Auto-seed initial locations into mock Firestore
    try:
        from database.seed_firebase import seed_database
        seed_database()
    except Exception as e:
        print(f"[Firebase] Note: Auto-seed skipped ({e})")

    return _FIRESTORE_DB


def get_firestore_db():
    """Returns singleton instance of Firestore DB client."""
    global _FIRESTORE_DB
    if _FIRESTORE_DB is None:
        init_firebase()
    return _FIRESTORE_DB


def get_firebase_app():
    """Returns initialized Firebase app instance."""
    global _FIREBASE_APP
    if _FIREBASE_APP is None:
        init_firebase()
    return _FIREBASE_APP


def is_mock_db() -> bool:
    """Returns True if the backend is currently using the mock in-memory Firestore."""
    return _IS_USING_MOCK_DB
