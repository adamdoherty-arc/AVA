from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Magnus API"
    DEBUG: bool = True
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Reddit Config
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "Magnus/1.0"
    
    # Redis Config
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # LLM Config
    LLM_PROVIDER: str = "ollama"  # or "openai"
    LLM_MODEL: str = "llama3.2"   # or "gpt-4"

    # Robinhood Config
    ROBINHOOD_USERNAME: Optional[str] = None
    ROBINHOOD_PASSWORD: Optional[str] = None
    ROBINHOOD_TOTP: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()
