"""
Bharat-TravelShield AI - Core Configuration Module
Author: Rajdeep (Backend / API)
"""

from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Bharat-TravelShield AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # CORS Whitelist for Rosemariya's Next.js Frontend
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]

    # Firebase Settings
    FIREBASE_CREDENTIALS_PATH: str = "firebase_credentials.json"
    FIREBASE_PROJECT_ID: str = "bharat-travelshield-ai"
    FIREBASE_STORAGE_BUCKET: str = "bharat-travelshield-ai.appspot.com"

    # ML Model Settings
    DEFAULT_DATA_DIR: str = "data"
    MODEL_DIR: str = "ml/models"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.strip().startswith("[") and v.strip().endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
