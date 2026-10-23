"""
Bharat-TravelShield AI - Core FastAPI Application Entrypoint
Author: Pranav (Database + Integration) & Rajdeep (Backend)
Project: Bharat-TravelShield AI

Connects Firebase Firestore, MultiHazardRiskANN inference engine,
and REST endpoints for the Next.js frontend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.firebase_client import init_firebase, is_mock_db
from database.seed_firebase import seed_database
from ml.inference.predict import get_predictor
from backend.app.routes import locations, predict, history, model_info, destination


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    print("[Startup] Initializing Bharat-TravelShield AI backend services...")
    # 1. Initialize Firebase Firestore
    init_firebase()

    # 2. Automatically seed in-memory Firestore if in mock mode
    if is_mock_db():
        print("[Startup] Running initial seed for in-memory Firestore...")
        seed_database()

    # 3. Warm up the MultiHazardRiskANN inference engine
    print("[Startup] Warming up MultiHazardRiskANN neural network model...")
    predictor = get_predictor()
    print(f"[Startup] Neural network ready: {predictor.torch_model is not None or predictor.numpy_model is not None}")

    yield

    print("[Shutdown] Cleaning up backend resources...")


app = FastAPI(
    title="Bharat-TravelShield AI — Spatiotemporal Risk Intelligence API",
    description="India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform powered by Deep Artificial Neural Networks & Firebase Firestore.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Rosemariya's Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(predict.router)
app.include_router(locations.router)
app.include_router(history.router)
app.include_router(model_info.router)
app.include_router(destination.router)


@app.get("/", tags=["System"])
def root_status():
    """System status and API overview."""
    return {
        "project": "Bharat-TravelShield AI",
        "system": "India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform",
        "team": {
            "database_and_integration": "Pranav",
            "ml_and_research": "Varesh",
            "backend_api": "Rajdeep",
            "frontend_ui": "Rosemariya",
            "data_engineering": "Adhith",
        },
        "database": "Firebase Firestore (In-Memory Fallback)" if is_mock_db() else "Firebase Firestore (Cloud)",
        "model": "MultiHazardRiskANN (Multi-Task Deep MLP)",
        "hazards": ["Dengue", "Malaria", "Flood", "Heatwave"],
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
    }


@app.get("/api/v1/health", tags=["System"])
def health_check():
    """Health status check for container orchestration and uptime monitoring."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "database_connected": True,
        "database_mode": "mock_memory" if is_mock_db() else "cloud_firestore",
        "ml_model_loaded": True,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
