"""
Bharat-TravelShield AI - Core FastAPI Application Entrypoint
Author: Rajdeep (Backend / API)
Project: Bharat-TravelShield AI

Connects Google Cloud Firebase Firestore, Varesh's Multi-Hazard ANN models,
and REST endpoints for Rosemariya's Next.js web application.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database.firebase_client import init_firebase, is_mock_db
from database.seed_firebase import seed_database
from ml.hazard_inference import list_hazards
from backend.app.core.config import settings
from backend.app.routes import destinations, predict, history
from backend.app.schemas.api_schemas import HealthResponse, SystemInfoResponse
from backend.app.utils.security import SecurityHeadersMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    print("=" * 70)
    print(" Bharat-TravelShield AI — Initializing FastAPI Backend Microservice")
    print("=" * 70)

    # 1. Initialize Firebase Firestore (or In-Memory Mock Fallback)
    print("[Startup] Initializing Firebase Firestore connection...")
    init_firebase()
    db_mode = "In-Memory Mock Firestore (Zero-Crash Fallback)" if is_mock_db() else "Connected to Google Cloud Firestore"
    print(f"[Startup] Database Mode: {db_mode}")

    # 2. Verify ML Models and Hazard Inferences
    print("[Startup] Verifying Artificial Neural Network (ANN) hazard models...")
    try:
        hz_status = list_hazards()
        supported = [k for k, v in hz_status.items() if "supported" in v.get("status", "")]
        print(f"[Startup] ML Hazard Models Verified: {', '.join(supported)}")
    except Exception as e:
        print(f"[Startup] Warning during ML check: {e}")

    print("[Startup] Bharat-TravelShield AI Backend is LIVE and ready for requests.")
    print("=" * 70)

    yield

    print("[Shutdown] Cleaning up Bharat-TravelShield AI backend resources...")


app = FastAPI(
    title="Bharat-TravelShield AI — Spatiotemporal Risk Intelligence API",
    description=(
        "India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform. "
        "Powered by deep Artificial Neural Networks (MLP), Firebase Firestore, and official "
        "Indian meteorological and epidemiological datasets (IMD, NASA POWER, NCVBDC, NDMA)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Rosemariya's Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard Security Headers
app.add_middleware(SecurityHeadersMiddleware)

# Global Controlled Error Sanitizer
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Sanitizes unexpected server errors so stack traces are never exposed to clients."""
    print(f"[Unhandled Exception] {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "internal_server_error",
                "message": "An unexpected error occurred while processing the risk intelligence request.",
                "path": request.url.path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

# Register Sub-Routers
app.include_router(destinations.router)
app.include_router(destinations.locations_router)
app.include_router(predict.router)
app.include_router(history.router)


@app.get(
    "/",
    response_model=SystemInfoResponse,
    tags=["System"],
    summary="System overview and metadata",
)
def root_overview():
    """Returns project overview, team roles, and system metadata."""
    return SystemInfoResponse(
        project="Bharat-TravelShield AI",
        system="India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform",
        version="1.0.0",
        team={
            "backend_api": "Rajdeep",
            "database_and_integration": "Pranav",
            "ml_and_research": "Varesh",
            "data_engineering": "Adhith",
            "frontend_ui": "Rosemariya",
        },
        database="In-Memory Mock Firestore" if is_mock_db() else "Google Cloud Firestore",
        supported_hazards=["heatwave", "flood", "malaria"],
        disclaimer=(
            "Decision support only. Predictions are experimental artificial neural network estimates. "
            "Not a diagnosis, emergency alert, or safety guarantee."
        ),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness and readiness health check",
)
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API v1 health check",
)
def health_check():
    """Health status check for container orchestration, frontend monitoring, and viva demo."""
    return HealthResponse(
        status="healthy",
        api_version="1.0.0",
        database_connected=True,
        database_mode="mock_memory" if is_mock_db() else "cloud_firestore",
        ml_models_ready=True,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
