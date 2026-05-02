"""
backend/core/config.py

Centralized configuration using Pydantic Settings.
All environment variables are loaded from the .env file automatically.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    MediLens application settings.
    All values are loaded from .env file or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # --- LLM Providers ---
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # --- Groq Model Selection ---
    groq_model: str = "llama-3.3-70b-versatile"   # fast + high quality
    ollama_vision_model: str = "llava"             # for OCR fallback + skin description

    # --- Database ---
    database_url: str = "postgresql://postgres:password@localhost:5432/medilens"

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    # --- Google Maps ---
    google_maps_api_key: str = ""

    # --- Vector Store ---
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"    # local HuggingFace model

    # --- OCR Settings ---
    ocr_confidence_threshold: float = 0.70        # below this → use LLaVA fallback

    # --- App ---
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    secret_key: str = "change-me-in-production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def has_google_maps(self) -> bool:
        return bool(self.google_maps_api_key)

    @property
    def has_twilio(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token)


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this everywhere to avoid reloading .env on every call.

    Usage:
        from backend.core.config import get_settings
        settings = get_settings()
        print(settings.groq_api_key)
    """
    return Settings()
