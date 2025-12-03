"""
AVA Backend Configuration
=========================

Centralized configuration using pydantic-settings.
All values are environment-variable configurable.

Author: AVA Trading Platform
Updated: 2025-11-28
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional, List, Dict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    Defaults are development-friendly.
    """

    # ==========================================================================
    # App Config
    # ==========================================================================
    APP_NAME: str = "AVA Trading Platform API"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ==========================================================================
    # CORS Configuration
    # ==========================================================================
    CORS_ORIGINS: str = Field(
        default=(
            "http://localhost:5173,http://localhost:5174,"
            "http://localhost:5175,http://localhost:5179,http://localhost:3000"
        ),
        description="Comma-separated list of allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list"""
        origins = self.CORS_ORIGINS.split(",")
        return [origin.strip() for origin in origins if origin.strip()]

    # ==========================================================================
    # API Keys
    # ==========================================================================
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    TRADIER_API_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None

    # ==========================================================================
    # Reddit Config
    # ==========================================================================
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "AVA/1.0"

    # ==========================================================================
    # Redis Config
    # ==========================================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False

    # ==========================================================================
    # LLM Config
    # ==========================================================================
    LLM_PROVIDER: str = "ollama"  # or "openai", "anthropic"
    LLM_MODEL: str = "llama3.2"   # or "gpt-4", "claude-3-5-sonnet"
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # ==========================================================================
    # Robinhood Config
    # ==========================================================================
    ROBINHOOD_USERNAME: Optional[str] = None
    ROBINHOOD_PASSWORD: Optional[str] = None
    ROBINHOOD_TOTP: Optional[str] = None

    # ==========================================================================
    # Database Config
    # ==========================================================================
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ava"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_POOL_MIN: int = 5
    DB_POOL_MAX: int = 50
    DB_QUERY_TIMEOUT: int = 60

    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ==========================================================================
    # Cache TTLs (seconds)
    # ==========================================================================
    CACHE_TTL_POSITIONS: int = 30
    CACHE_TTL_METADATA: int = 300
    CACHE_TTL_SUMMARY: int = 60
    CACHE_TTL_MARKET_DATA: int = 5
    CACHE_TTL_OPTIONS_CHAIN: int = 60
    CACHE_TTL_PREDICTIONS: int = 300

    # ==========================================================================
    # AI Prediction Settings
    # ==========================================================================
    PREDICTION_CONFIDENCE_HIGH: int = 85
    PREDICTION_CONFIDENCE_MEDIUM: int = 70
    PREDICTION_CONFIDENCE_LOW: int = 55

    @property
    def prediction_confidence_map(self) -> Dict[str, int]:
        """Centralized confidence score mapping"""
        return {
            'high': self.PREDICTION_CONFIDENCE_HIGH,
            'medium': self.PREDICTION_CONFIDENCE_MEDIUM,
            'low': self.PREDICTION_CONFIDENCE_LOW
        }

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    RATE_LIMIT_PER_SECOND: float = 10.0
    RATE_LIMIT_BURST: int = 20

    # ==========================================================================
    # API Client Settings
    # ==========================================================================
    API_MAX_RETRIES: int = 3
    API_RETRY_BACKOFF: float = 0.5
    API_CONNECT_TIMEOUT: float = 5.0
    API_READ_TIMEOUT: float = 30.0
    API_CIRCUIT_FAILURES: int = 5
    API_CIRCUIT_RECOVERY: float = 60.0

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience exports
settings = get_settings()
